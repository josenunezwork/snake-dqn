"""Stock wait (scarce GPUs): a round of create tries about once per minute.

gpu_run_v2 ``launch`` and pod_bench ``bench`` poll for capacity with ``--stock-wait-hours`` /
``--stock-poll-s``. No RunPod call is ever made: the RunPod client, ledger, TLS preflight,
watchdog, pod agent and clock are fakes (RpClient also refuses real rp.py calls under pytest).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.perf_podbench_20261013 import pod_bench as pb
from research.redesign_m3c_20261008 import gpu_run_v2 as gr
from research.runpod_fanout.rp_client import RunPodError
from tests.test_m3c_gpu_run_v2 import _plan
from tests.test_perf_podbench import COMMANDS, FakeAgent, _args

NO_CAPACITY = {
    "http_status": 500,
    "error": {"error": "create pod: There are no instances currently available", "status": 500},
}
BAD_REQUEST = {"http_status": 400, "error": {"error": "gpuTypeIds: unknown GPU type"}}
SERVER_ERROR = {"http_status": 500, "error": {"error": "internal server error"}}
THROTTLED = {"http_status": 429, "error": {"error": "Too Many Requests"}}


def _status(code, msg="request refused"):
    return {"http_status": code, "error": {"error": msg}}


class Clock:
    """Fake time for gr.now and time.sleep; ``interrupt_at`` raises in the n-th poll sleep."""

    def __init__(self, monkeypatch, poll_s=None, interrupt_at=None, t0=3_000_000.0):
        self.t, self.t0, self.sleeps = t0, t0, []
        self.poll_s, self.interrupt_at, self.poll_sleeps = poll_s, interrupt_at, 0
        monkeypatch.setattr(gr, "now", lambda: self.t)
        monkeypatch.setattr(gr.time, "sleep", self.sleep)

    def sleep(self, s):
        self.sleeps.append(s)
        if self.poll_s is not None and math.isclose(s, self.poll_s):
            self.poll_sleeps += 1
            if self.poll_sleeps == self.interrupt_at:
                raise KeyboardInterrupt("signal 15")
        self.t += s


class CapacityRp:
    """RunPod: the first ``refusals`` creates are refused (``payload``), then the pod is made.

    ``stock`` = data centers listing the RTX 4090 (the stock query); ``ghost_after`` = a pod of
    ours named like the first create that shows up in GET /pods that many seconds after it.
    ``sequence`` = one refusal payload per create (then the pod is made); ``undeletable`` =
    pod ids whose DELETE fails; ``others`` = pods of another prefix (never ours to touch)."""

    def __init__(
        self,
        clock,
        refusals=0,
        payload=NO_CAPACITY,
        stock=(),
        ghost_after=None,
        sequence=None,
        undeletable=(),
        others=(),
    ):
        if sequence is not None:
            refusals = len(sequence)
        self.clock, self.refusals, self.payload = clock, refusals, payload
        self.sequence = list(sequence) if sequence is not None else None
        self.stock, self.ghost_after = list(stock), ghost_after
        self.pods, self.deleted, self.creates, self.ghost_gone = [], [], [], False
        self.undeletable, self.others, self.deleted_at = set(undeletable), list(others), {}

    def _call(self, *args):
        if not self.stock:
            raise RuntimeError("no stock query in this fake")
        return {
            "dataCenters": [
                {"id": dc, "gpuAvailability": [{"gpuTypeId": gr.GPU, "stockStatus": "Low"}]}
                for dc in self.stock
            ]
        }

    def _ghost(self):
        if self.ghost_after is None or self.ghost_gone or not self.creates:
            return []
        first = self.creates[0]
        if self.clock.t < first["t"] + self.ghost_after:
            return []
        return [{"id": "ghost", "name": first["name"]}]

    def list_pods(self):
        return list(self.pods) + self._ghost() + list(self.others)

    def balance(self):
        return 50.0

    def get_pod(self, pid):
        return next((p for p in self.list_pods() if p["id"] == pid), None)

    def delete_pod(self, pid, confirm):
        assert confirm
        assert pid not in {p["id"] for p in self.others}, f"deleted another prefix's pod {pid}"
        if pid in self.undeletable:
            raise RunPodError("rp.py ('rest', 'DELETE') exit 1", SERVER_ERROR)
        self.deleted.append(pid)
        self.deleted_at.setdefault(pid, self.clock.t)
        self.pods = [p for p in self.pods if p["id"] != pid]
        self.ghost_gone = self.ghost_gone or pid == "ghost"

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm
        body = json.loads(Path(body_path).read_text())
        self.creates.append(
            {"t": self.clock.t, "name": body["name"], "dcs": body.get("dataCenterIds")}
        )
        if len(self.creates) <= self.refusals:
            payload = self.payload
            if self.sequence is not None:
                payload = self.sequence[len(self.creates) - 1]
            if payload is None:  # the answer was lost: an unknown outcome
                raise RunPodError("rp.py ('rest', 'POST', '/pods') timed out")
            raise RunPodError("rp.py ('rest', 'POST', '/pods') exit 1", payload)
        self.pods.append({"id": "pod1", "name": body["name"]})
        return {"id": "pod1", "costPerHr": 0.69, "machine": {"gpuTypeId": gr.GPU}}

    @property
    def made(self):
        return len(self.creates) - min(self.refusals, len(self.creates))


class Ledger:
    """The shared ledger: one live run at a time, every reservation released."""

    runs: dict = {}

    def __init__(self, policy):
        pass

    def register_run(self, run_id, job_id, run_dir, budget, pid):
        assert run_id not in Ledger.runs
        live = [r for r, v in Ledger.runs.items() if v["status"] == "live"]
        assert not live, f"a refused round left a live ledger run: {live}"
        Ledger.runs[run_id] = {"status": "live", "pods": {}}

    def reserve_pod(self, run_id, key, rate, until, balance):
        Ledger.runs[run_id]["pods"][key] = {"pod_id": None, "cost": None}

    def bind_pod(self, run_id, key, pod_id, rate):
        Ledger.runs[run_id]["pods"][key]["pod_id"] = pod_id

    def release_pod(self, run_id, key, cost):
        Ledger.runs[run_id]["pods"][key]["cost"] = cost

    def finish_run(self, run_id, success, complete):
        Ledger.runs[run_id].update(status="finished", success=success)

    @classmethod
    def summary(cls):
        rows = list(cls.runs.values())
        failed = [r for r in rows if r["status"] == "finished" and not r["success"]]
        ok = [r for r in rows if r["status"] == "finished" and r["success"]]
        live = [r for r in rows if r["status"] != "finished"]
        unreleased = [p for r in rows for p in r["pods"].values() if p["cost"] is None]
        return failed, ok, live, unreleased


class Watchdogs:
    """spawn / stop of the detached watchdog; ``spawn`` checks what the real one needs: no
    stale ``watchdog.stop`` (it would exit at once) and no stale "armed" line (it would pass
    the spawn check without a live watchdog)."""

    def __init__(self, monkeypatch):
        self.spawned, self.stopped = [], []
        self.names, self.stopped_at, self.max_alive = {}, {}, 0
        monkeypatch.setattr(gr, "spawn_watchdog", self.spawn)
        monkeypatch.setattr(gr, "stop_watchdog", self.stop)

    def spawn(self, run_dir, until, name):
        assert not (run_dir / "watchdog.stop").exists(), "stale watchdog.stop"
        log = run_dir / "watchdog.log"
        assert not (log.exists() and '"armed"' in log.read_text()), "stale armed line"
        log.write_text(json.dumps({"watchdog": "armed", "name": name}) + "\n")
        self.spawned.append(1000 + len(self.spawned))
        self.names[self.spawned[-1]] = name
        self.max_alive = max(self.max_alive, len(self.running))
        return self.spawned[-1]

    def stop(self, pid, run_dir=None):
        if run_dir is not None:
            (run_dir / "watchdog.stop").write_text("stopped")
        self.stopped.append(pid)
        self.stopped_at.setdefault(pid, gr.now())

    @property
    def running(self):
        return [p for p in self.spawned if p not in self.stopped]


def _assert_backstop(rp, wd):
    """Every refused round's watchdog stayed armed until CREATE_SETTLE_S after its round's
    last create try (the late-listing window), never stopped right after one clean listing."""
    last_try = {c["name"]: c["t"] for c in rp.creates}
    refused = {c["name"] for c in rp.creates[: rp.refusals]}
    checked = 0
    for pid, name in wd.names.items():
        if name in refused and pid in wd.stopped_at:
            assert wd.stopped_at[pid] - last_try[name] >= gr.CREATE_SETTLE_S, (name, pid)
            checked += 1
    assert checked or not refused


def _common(monkeypatch, rp):
    import research.runpod_fanout.ledger as ledger_mod
    import research.runpod_fanout.rp_client as rp_mod
    import research.runpod_fanout.tls as tls_mod

    Ledger.runs = {}
    monkeypatch.setattr(ledger_mod, "SharedLedger", Ledger)
    monkeypatch.setattr(rp_mod, "RpClient", lambda: rp)
    monkeypatch.setattr(tls_mod, "preflight", lambda: [])
    monkeypatch.setattr(gr.signal, "signal", lambda *a: None)
    return Watchdogs(monkeypatch)


def _wd():
    """The Watchdogs fake of a launch that raised (it is bound to the patched spawn)."""
    return gr.spawn_watchdog.__self__


def _rows(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


# ----------------------------------------------------------------------------- gpu_run_v2
def _gr_launch(tmp_path, monkeypatch, rp, wait_h, drive="complete", **kw):
    wd = _common(monkeypatch, rp)
    monkeypatch.setattr(gr.subprocess, "Popen", lambda *a, **k: None)  # caffeinate
    monkeypatch.setattr(
        gr,
        "build_inputs",
        lambda work, plan: {
            "commit": "c",
            "files": [],
            "demo_sha256": "d",
            "bytes": 0,
            "plan_sha256": plan["_sha256"],
        },
    )
    monkeypatch.setattr(gr, "drive", drive if callable(drive) else (lambda *a: drive))
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()))
    args = SimpleNamespace(
        run_dir=str(tmp_path / "run"),
        plan=str(plan_path),
        lifetime_hours=8.0,
        spend_cap=10.0,
        rate=0.89,
        stock_wait_hours=wait_h,
        **kw,
    )
    return gr.launch(args), tmp_path / "run", wd


def test_gr_refused_rounds_then_success_once_a_minute(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=5)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)  # default poll
    assert code == 0 and gr.STOCK_POLL_S == 60
    # exactly one pod: created on the 6th round, then deleted
    assert len(rp.creates) == 6 and rp.made == 1
    assert rp.pods == [] and rp.deleted == ["pod1"]
    # a round every 60 s (start to start), no 5-min settle between the refused rounds
    starts = [c["t"] for c in rp.creates]
    assert [b - a for a, b in zip(starts, starts[1:])] == [60.0] * 5
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 5 and len(ok) == 1 and not live and not unreleased
    assert all(p["cost"] == 0.0 and p["pod_id"] is None for r in failed for p in r["pods"].values())
    assert [p["pod_id"] for p in ok[0]["pods"].values()] == ["pod1"]
    assert len(wd.spawned) == 6 and wd.running == []
    _assert_backstop(rp, wd)  # each refused round's watchdog outlived its late-listing window
    # bounded attempt dirs: the latest refused round + the session
    assert sorted(p.name for p in root.glob("attempt*")) == ["attempt04", "attempt05"]
    assert (root / "CURRENT_ATTEMPT").read_text() == "attempt05"
    st = json.loads((root / "attempt05/state.json").read_text())
    assert st["pod_id"] == "pod1" and st["outcome"] == "complete" and st["pods_deleted_verified"]
    last = json.loads((root / "attempt04/state.json").read_text())
    assert last["settle_deferred"] and last["pods_deleted_verified"] and "pod_id" not in last
    rows = _rows(root / "stock_wait.jsonl")
    assert [r["round"] for r in rows] == [0, 1, 2, 3, 4]
    assert all(r["cost_usd"] == 0.0 and r["pods_deleted_verified"] for r in rows)
    assert all(r["error"].startswith("no capacity") for r in rows)
    assert all(r["tries"] == [{"gpu": gr.GPU, "dcs": None}] for r in rows)


def test_gr_gives_up_at_the_bound_with_nothing_live_and_bounded_dirs(tmp_path, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0, stock_poll_s=60)
    assert code == 1
    rounds = len(rp.creates)
    assert 179 <= rounds <= 182  # ~ once per minute for 3 h
    starts = [c["t"] for c in rp.creates]
    assert all(b - a == 60.0 for a, b in zip(starts, starts[1:]))
    assert rp.made == 0 and rp.pods == [] and rp.deleted == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == rounds and not ok and not live and not unreleased
    assert all(p["cost"] == 0.0 for r in failed for p in r["pods"].values())
    assert len(wd.spawned) == rounds and wd.running == []
    _assert_backstop(rp, wd)
    # reaped between rounds once their window passed clean: ~5 armed at a time, not 180
    assert wd.max_alive <= 6
    attempts = sorted(p.name for p in root.glob("attempt*"))
    assert len(attempts) <= 2  # not 180 dirs
    assert len(_rows(root / "stock_wait.jsonl")) == rounds
    # the late-listing settle still runs once before giving up (after the last refused create)
    st = json.loads((root / (root / "CURRENT_ATTEMPT").read_text() / "state.json").read_text())
    assert st["late_listing_settle"]["pods_deleted_verified"] and st["pods_deleted_verified"]
    assert clock.t - starts[-1] >= gr.CREATE_SETTLE_S
    assert clock.t - clock.t0 <= 3 * 3600 + gr.CREATE_SETTLE_S + 120
    out = capsys.readouterr()
    assert "giving up at $0" in out.err
    # one compact line per round, no per-try JSON error spam
    assert out.out.count("stock wait round ") == rounds
    assert "create try" not in out.out and "http_status" not in out.out


def test_gr_one_create_per_dc_per_round(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=6, stock=("EU-RO-1", "EUR-IS-1"))
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)
    assert code == 0 and rp.made == 1 and len(rp.creates) == 7
    tries = [c["dcs"] for c in rp.creates]
    assert tries[:6] == [None, ["EU-RO-1"], ["EUR-IS-1"]] * 2  # any DC + each stocked DC once
    # a round of 3 tries 20 s apart (40 s) is followed 60 s after its start
    assert rp.creates[3]["t"] - rp.creates[0]["t"] == 60.0
    assert rp.creates[6]["t"] - rp.creates[3]["t"] == 60.0
    assert wd.running == [] and not Ledger.summary()[2]


@pytest.mark.parametrize(
    "payload, why",
    [(BAD_REQUEST, "4xx refusal"), (SERVER_ERROR, "5xx, not capacity"), (None, "unknown outcome")],
)
def test_gr_non_capacity_create_error_does_not_loop(tmp_path, monkeypatch, capsys, payload, why):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, payload=payload)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    assert code == 1, why
    assert len(rp.creates) == 1 and rp.pods == [], why
    assert [p.name for p in root.glob("attempt*")] == ["attempt00"]
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 1 and not live and not unreleased and wd.running == []
    st = json.loads((root / "attempt00/state.json").read_text())
    assert "settle_deferred" not in st and st["pods_deleted_verified"]
    assert clock.t - rp.creates[0]["t"] >= gr.CREATE_SETTLE_S  # settled in full, as before
    assert "giving up" not in capsys.readouterr().err


def test_gr_without_stock_wait_is_a_single_round_as_before(tmp_path, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=0.0)
    assert code == 1 and len(rp.creates) == 1
    assert [p.name for p in root.glob("attempt*")] == ["attempt00"]
    assert not (root / "stock_wait.jsonl").exists()
    st = json.loads((root / "attempt00/state.json").read_text())
    assert "settle_deferred" not in st and st["pods_deleted_verified"]
    assert clock.t - rp.creates[0]["t"] >= gr.CREATE_SETTLE_S  # the full settle, as before
    out = capsys.readouterr()
    assert "create try 0" in out.out and "giving up at $0" in out.err


def test_gr_interrupt_between_rounds_settles_and_leaves_nothing_live(tmp_path, monkeypatch):
    clock = Clock(monkeypatch, poll_s=60.0, interrupt_at=3)
    rp = CapacityRp(clock, refusals=10**9)
    with pytest.raises(KeyboardInterrupt):
        _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    assert len(rp.creates) == 3 and rp.pods == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 3 and not live and not unreleased
    root = tmp_path / "run"
    st = json.loads((root / "attempt02/state.json").read_text())
    assert st["late_listing_settle"]["pods_deleted_verified"]
    assert clock.t - rp.creates[-1]["t"] >= gr.CREATE_SETTLE_S
    assert _wd().running == []  # every round's watchdog stopped, after the clean settle
    _assert_backstop(rp, _wd())


def test_gr_late_listed_pod_after_a_refusal_is_deleted(tmp_path, monkeypatch):
    """Belt and braces: a pod RunPod made despite refusing (listed 30 s later) is found by the
    next round's pre-create check, which stops the wait; the settle deletes it."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, ghost_after=30)
    with pytest.raises(SystemExit, match="already exists"):
        _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    assert len(rp.creates) == 1 and rp.deleted == ["ghost"] and rp.list_pods() == []
    assert not Ledger.summary()[2]
    # round 0's watchdog stayed armed while its pod was listed: stopped only after the settle
    # deleted that pod and verified a clean listing
    wd = _wd()
    assert wd.running == [] and wd.stopped_at[wd.spawned[0]] > rp.deleted_at["ghost"]


def test_gr_refused_round_watchdogs_stay_armed_through_their_window(tmp_path, monkeypatch):
    """A refused round's watchdog is the backstop for a pod made despite the refusal: it stays
    armed until CREATE_SETTLE_S after that round's create AND a clean listing (reaped between
    rounds), or until a later round's full settle covers it; never just one clean listing."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=8)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)
    assert code == 0 and rp.made == 1 and len(rp.creates) == 9 and wd.running == []
    _assert_backstop(rp, wd)
    # rounds 0-3: stopped at the start of round 5-8, exactly when their window had passed
    assert [wd.stopped_at[p] - clock.t0 for p in wd.spawned[:4]] == [300.0, 360.0, 420.0, 480.0]
    # rounds 4-7 were still in their window when round 8 got the pod: armed through the
    # session, stopped once its full settle (a clean listing >= 300 s after a later create)
    session = rp.creates[8]["t"]
    assert all(wd.stopped_at[p] >= session + gr.CREATE_SETTLE_S for p in wd.spawned[4:8])
    assert wd.max_alive == 5  # bounded: the window / the poll + the current round
    last = json.loads((root / "attempt07/state.json").read_text())
    assert last["watchdog_stopped"] and "full late-listing settle" in last["watchdog_stop"]["why"]


def test_gr_reap_keeps_a_listed_pod_s_watchdog_armed(tmp_path, monkeypatch):
    """Round 0's pod is listed late (250 s, after round 4's check): at round 5 its window has
    passed but the listing shows it, so its watchdog stays armed; round 5's pre-create check
    stops the wait and the settle deletes the pod before that watchdog is stopped."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, ghost_after=250)
    with pytest.raises(SystemExit, match="already exists"):
        _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    assert len(rp.creates) == 5 and rp.deleted == ["ghost"] and rp.list_pods() == []
    wd = _wd()
    assert wd.running == [] and wd.stopped_at[wd.spawned[0]] > rp.deleted_at["ghost"]
    _assert_backstop(rp, wd)


def test_gr_failed_settle_leaves_the_watchdogs_armed(tmp_path, monkeypatch, capsys):
    """A pod of ours that cannot be deleted: the settle does not verify, so the refused round's
    watchdog (the only thing that will still delete that pod) stays armed."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, ghost_after=30, undeletable={"ghost"})
    with pytest.raises(SystemExit, match="already exists"):
        _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    wd = _wd()
    assert len(wd.spawned) == 1 and wd.running == wd.spawned  # left armed
    st = json.loads((tmp_path / "run/attempt00/state.json").read_text())
    assert not st["late_listing_settle"]["pods_deleted_verified"] and st["watchdog_left_armed"]
    assert "left armed" in capsys.readouterr().err


def test_gr_throttled_rounds_back_off_and_keep_waiting(tmp_path, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, sequence=[THROTTLED] * 4 + [NO_CAPACITY] * 2)
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)
    assert code == 0 and rp.made == 1 and len(rp.creates) == 7
    # 429: 2x, 4x, then capped at STOCK_THROTTLE_MAX_S; back to the poll after a capacity refusal
    starts = [c["t"] for c in rp.creates]
    assert [b - a for a, b in zip(starts, starts[1:])] == [120.0, 240.0, 300.0, 300.0, 60.0, 60.0]
    rows = _rows(root / "stock_wait.jsonl")
    assert [r["throttled"] for r in rows] == [True] * 4 + [False] * 2
    assert rows[0]["error"].startswith("throttled") and rows[-1]["error"].startswith("no capacity")
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 6 and len(ok) == 1 and not live and not unreleased
    assert wd.running == []
    _assert_backstop(rp, wd)
    out = capsys.readouterr()
    assert out.out.count("(throttled: backing off)") == 4 and "not polling again" not in out.err


def test_gr_throttle_ends_its_round_and_the_backoff_stays_within_the_wait(
    tmp_path, monkeypatch, capsys
):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, payload=THROTTLED, stock=("EU-RO-1", "EUR-IS-1"))
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=0.5)
    assert code == 1 and rp.made == 0 and rp.pods == []
    # one POST per round (a throttled try ends it: no DC hopping), rounds backed off within 0.5 h
    assert [c["dcs"] for c in rp.creates] == [None] * 7
    starts = [c["t"] - clock.t0 for c in rp.creates]
    assert starts == [0.0, 120.0, 360.0, 660.0, 960.0, 1260.0, 1560.0]
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 7 and not live and not unreleased and wd.running == []
    assert "giving up at $0" in capsys.readouterr().err


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_gr_genuine_4xx_ends_the_wait_fast(tmp_path, monkeypatch, capsys, status):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, payload=_status(status))
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=3.0)
    assert code == 1 and len(rp.creates) == 1 and rp.pods == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 1 and not live and not unreleased and wd.running == []
    assert clock.t - rp.creates[0]["t"] >= gr.CREATE_SETTLE_S  # settled in full
    assert _rows(root / "stock_wait.jsonl")[0]["error"].startswith("refused")
    assert "not polling again" in capsys.readouterr().err


def test_gr_throttled_read_before_create_backs_off(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=0)
    real, calls = rp.balance, {"n": 0}

    def throttled_twice():
        calls["n"] += 1
        if calls["n"] <= 2:
            raise RunPodError("rp.py ('gql',) exit 1", THROTTLED)
        return real()

    rp.balance = throttled_twice
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)
    assert code == 0 and rp.made == 1 and rp.creates[0]["t"] - clock.t0 == 360.0  # 120 + 240
    rows = _rows(root / "stock_wait.jsonl")
    assert [r["throttled"] for r in rows] == [True, True]
    assert all(r["error"].startswith("transient before create") for r in rows)


def test_create_refusal_kinds():
    def e(status, body):
        return RunPodError("rp.py exit 1", {"http_status": status, "error": body})

    assert gr.refusal_kind(e(429, {"error": "slow down"})) == "throttled"
    assert gr.refusal_kind(e(400, {"error": "Rate limit exceeded"})) == "throttled"
    assert gr.refusal_kind(e(500, NO_CAPACITY["error"])) == "no_capacity"
    for status in (400, 401, 403, 404, 422):
        assert gr.refusal_kind(e(status, {"error": "nope"})) == "refused"
    # no definite answer: the pod may exist (adopt by name, never re-POST)
    assert gr.refusal_kind(e(500, {"error": "internal server error"})) == "unknown"
    assert gr.refusal_kind(e(502, "<html>bad gateway</html>")) == "unknown"
    assert gr.refusal_kind(RunPodError("rp.py timed out")) == "unknown"
    # a 5xx create stays unknown even when it says "too many requests" (a pod may exist) ...
    assert gr.refusal_kind(e(503, "Too Many Requests")) == "unknown"
    # ... but a read (no side effect) that says so is a throttle: the stock wait backs off
    assert gr.PreCreateTransient.of(e(503, "Too Many Requests")).throttled
    assert gr.PreCreateTransient.of(e(429, "x")).throttled
    assert not gr.PreCreateTransient.of(RunPodError("rp.py timed out")).throttled


def test_gr_throttled_listing_between_tries_ends_the_round_and_backs_off(tmp_path, monkeypatch):
    """The adopt-check listing between two create tries is throttled: no more POSTs in that
    round (the earlier tries were definite refusals: nothing to adopt); the wait backs off."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=2, stock=("EU-RO-1",))
    real, calls = rp.list_pods, {"n": 0}

    def throttled_second():
        calls["n"] += 1
        if calls["n"] == 2:  # round 0: pre-create check OK, then the listing before try 1
            raise RunPodError("rp.py ('rest', 'GET', '/pods') exit 1", THROTTLED)
        return real()

    rp.list_pods = throttled_second
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0)
    assert code == 0 and rp.made == 1
    assert [(c["t"] - clock.t0, c["dcs"]) for c in rp.creates] == [
        (0.0, None),
        (120.0, None),
        (140.0, ["EU-RO-1"]),
    ]
    rows = _rows(root / "stock_wait.jsonl")
    assert len(rows) == 1 and rows[0]["throttled"]
    assert rows[0]["tries"] == [{"gpu": gr.GPU, "dcs": None}]  # the POSTs only
    assert wd.running == [] and not Ledger.summary()[2]


def _guarding_drive(minutes):
    """A session that only runs its guard once a minute (as drive / drive_bench do)."""

    def drive(agent, rp, run_dir, state, inputs, args, plan):
        for _ in range(minutes):
            assert gr.guard(rp, state, args) is None
            gr.time.sleep(60)
        return "complete"

    return drive


def test_gr_session_deletes_a_stray_pod_of_an_earlier_round(tmp_path, monkeypatch):
    """Round 0's refused create made a pod after all, listed only once round 1's session is
    running (after its pre-create check): the session's sweep deletes it within ~5 min and
    records it; a pod of another prefix is never touched; round 0's watchdog stays armed."""
    clock = Clock(monkeypatch)
    other = {"id": "other1", "name": "perfbench-0204000000"}
    rp = CapacityRp(clock, refusals=1, ghost_after=400, others=[other])
    code, root, wd = _gr_launch(tmp_path, monkeypatch, rp, wait_h=1.0, drive=_guarding_drive(20))
    assert code == 0 and rp.made == 1
    listed_at = rp.creates[0]["t"] + 400
    assert rp.deleted == ["ghost", "pod1"]  # the stray during the session, ours at its end
    assert listed_at <= rp.deleted_at["ghost"] <= listed_at + gr.STRAY_SWEEP_S + 60
    st = json.loads((root / "attempt01/state.json").read_text())
    assert [(r["id"], r["name"], r["deleted"]) for r in st["strays"]] == [
        ("ghost", rp.creates[0]["name"], True)
    ]
    assert rp.list_pods() == [other]
    assert wd.running == [] and wd.stopped_at[wd.spawned[0]] > rp.deleted_at["ghost"]
    _assert_backstop(rp, wd)


def test_sweep_strays_deletes_only_this_runner_s_other_round_pods(monkeypatch):
    clock = Clock(monkeypatch)

    class Rp:
        def __init__(self, pods):
            self.pods, self.lists, self.deleted, self.fail = list(pods), 0, [], set()

        def list_pods(self):
            self.lists += 1
            return list(self.pods)

        def balance(self):
            return 50.0

        def delete_pod(self, pid, confirm):
            assert confirm
            if pid in self.fail:
                raise RunPodError("rp.py DELETE exit 1", SERVER_ERROR)
            self.deleted.append(pid)
            self.pods = [p for p in self.pods if p["id"] != pid]

    own, stray = "m3c-gpu-0204101000", "m3c-gpu-0204100900"
    untouched = [
        {"id": "own", "name": own},
        {"id": "n1", "name": "m3c-gpu-b-0204100900"},  # a prefix that extends ours
        {"id": "o1", "name": "perfbench-0204100900"},  # another prefix
        {"id": "x1", "name": "m3c-gpu-dryrun"},  # not a round's name
    ]
    rp = Rp(untouched + [{"id": "s1", "name": stray}])
    state = {"name": own, "pod_id": "own", "deadline": clock.t + 3600, "balance0": 50.0}
    args = SimpleNamespace(spend_cap=10.0)
    assert gr.guard(rp, state, args) is None
    assert rp.deleted == ["s1"] and rp.lists == 1
    assert state["strays"] == [{"id": "s1", "name": stray, "t": clock.t, "deleted": True}]
    clock.t += gr.STRAY_SWEEP_S - 1  # not again before STRAY_SWEEP_S
    assert gr.guard(rp, state, args) is None and rp.lists == 1
    # a failed delete is recorded and retried at the next sweep
    rp.pods.append({"id": "s2", "name": "m3c-gpu-0204100800"})
    rp.fail = {"s2"}
    clock.t += 1
    gr.guard(rp, state, args)
    assert rp.lists == 2 and not state["strays"][-1]["deleted"] and state["strays"][-1]["error"]
    rp.fail = set()
    clock.t += gr.STRAY_SWEEP_S
    gr.guard(rp, state, args)
    assert rp.deleted == ["s1", "s2"] and state["strays"][-1]["deleted"]
    assert rp.pods == untouched
    # a session without its own name never sweeps (it could not tell its pod apart)
    assert gr.sweep_strays(rp, {"pod_id": "own"}) == [] and rp.lists == 3
    # pod_bench's prefix: the same rule under perfbench-
    monkeypatch.setattr(gr, "POD_NAME", gr.POD_NAME)  # restored after the test
    pb.bind_prefix("perfbench-")
    rp = Rp(
        [
            {"id": "own", "name": "perfbench-0204101000"},
            {"id": "s3", "name": "perfbench-0204100900"},
            {"id": "n2", "name": "perfbench-a-0204100900"},
            {"id": "m1", "name": stray},
        ]
    )
    state = {"name": "perfbench-0204101000", "pod_id": "own"}
    assert [r["id"] for r in gr.sweep_strays(rp, state)] == ["s3"]
    assert {p["id"] for p in rp.pods} == {"own", "n2", "m1"}


def test_gr_stock_poll_validation(tmp_path, monkeypatch, capsys):
    import research.runpod_fanout.rp_client as rp_mod

    def no_rp(*a, **k):
        raise AssertionError("RunPod touched")

    monkeypatch.setattr(rp_mod, "RpClient", no_rp)
    for bad in (29.9, 0, -60, 3601, float("nan"), "x", None):
        assert gr.stock_poll_problem(bad), bad
    for ok in (30, 60, 600.5, 3600):
        assert gr.stock_poll_problem(ok) is None
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(json.dumps(_plan()))
    for argv in (
        ["cleanup", "--stock-poll-s", "10"],
        ["launch", "--plan", str(plan_path), "--confirm", "--stock-poll-s", "4000"],
        ["launch", "--plan", str(plan_path), "--confirm", "--stock-poll-s", "nan"],
    ):
        assert gr.main(argv) == 2
        assert "--stock-poll-s must be in [30, 3600]" in capsys.readouterr().err
    # a valid interval passes the check (and launch still needs --confirm)
    assert gr.main(["launch", "--plan", str(plan_path), "--stock-poll-s", "30"]) == 2
    assert "launch spends money" in capsys.readouterr().err
    # launch itself refuses before anything is created
    args = SimpleNamespace(run_dir=str(tmp_path / "run"), plan=str(plan_path), stock_poll_s=5)
    with pytest.raises(SystemExit, match="stock-poll-s"):
        gr.launch(args)
    assert not (tmp_path / "run").exists()


# ----------------------------------------------------------------------------- pod_bench
def _pb_launch(tmp_path, monkeypatch, rp, agent=None, name="session", **kw):
    wd = _common(monkeypatch, rp)
    monkeypatch.setattr(gr, "POD_NAME", gr.POD_NAME)  # restored after the test
    monkeypatch.setattr(pb, "keep_awake", lambda: None)
    monkeypatch.setattr(pb, "make_agent", lambda pod_id, token: agent or FakeAgent())
    monkeypatch.setattr(
        pb,
        "build_bench_inputs",
        lambda work, commit: {
            "commit": commit,
            "files": [{"dst": "repo.tar.gz", "src": "x", "sha256": "s"}],
            "bytes": 1,
        },
    )
    out_dir = tmp_path / name
    code = pb.launch(_args(out_dir, confirm=True, **kw), "c0ffee", COMMANDS, [])
    return code, out_dir, wd


def test_pb_refused_rounds_then_success_keeps_the_session_layout(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    single = CapacityRp(clock, refusals=0)
    code1, one, _ = _pb_launch(tmp_path, monkeypatch, single, name="single")
    assert code1 == 0
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=4)
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert code == 0
    assert len(rp.creates) == 5 and rp.made == 1 and rp.pods == [] and rp.deleted == ["pod1"]
    starts = [c["t"] for c in rp.creates]
    assert [b - a for a, b in zip(starts, starts[1:])] == [60.0] * 4
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 4 and len(ok) == 1 and not live and not unreleased
    assert all(p["cost"] == 0.0 for r in failed for p in r["pods"].values())
    assert len(wd.spawned) == 5 and wd.running == []
    # the session dir is a single try's plus the stock-wait record (latest refused round only)
    top = {p.name for p in out_dir.iterdir()}
    assert top == {p.name for p in one.iterdir()} | {pb.STOCK_LOG, pb.STOCK_ASIDE}
    assert sorted(p.name for p in (out_dir / "out").iterdir()) == sorted(
        p.name for p in (one / "out").iterdir()
    )
    st = json.loads((out_dir / "state.json").read_text())
    summary = json.loads((out_dir / "bench_session.json").read_text())
    assert st["pod_id"] == "pod1" and summary["outcome"] == "complete"
    assert [c["returncode"] for c in summary["commands"]] == [0, 0, 0]
    assert not list(out_dir.glob("pod_body_try1.json"))
    aside = json.loads((out_dir / pb.STOCK_ASIDE / "state.json").read_text())
    assert aside["settle_deferred"] and aside["outcome"].startswith("create refused")
    rows = _rows(out_dir / pb.STOCK_LOG)
    assert [r["round"] for r in rows] == [0, 1, 2, 3]
    assert all(r["cost_usd"] == 0.0 and r["error"].startswith("no capacity") for r in rows)


def test_pb_gives_up_at_the_bound_with_nothing_live(tmp_path, monkeypatch, capsys):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9)
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=0.1, stock_poll_s=60)
    assert code == 1
    assert len(rp.creates) == 7  # rounds at 0, 60, ..., 360 s of a 360 s wait
    assert rp.made == 0 and rp.pods == [] and rp.deleted == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 7 and not ok and not live and not unreleased
    assert all(p["cost"] == 0.0 for r in failed for p in r["pods"].values())
    assert wd.running == []
    st = json.loads((out_dir / "state.json").read_text())
    assert st["late_listing_settle"]["pods_deleted_verified"] and st["pods_deleted_verified"]
    summary = json.loads((out_dir / "bench_session.json").read_text())
    assert summary["outcome"].startswith("create refused") and summary["pods_deleted_verified"]
    assert summary["cost_estimate_usd"] == 0.0
    assert len(_rows(out_dir / pb.STOCK_LOG)) == 7
    out = capsys.readouterr()
    assert out.out.count("stock wait round ") == 7 and "create try" not in out.out
    assert json.loads(out.out.strip().splitlines()[-1])["outcome"].startswith("create refused")


@pytest.mark.parametrize("payload", [BAD_REQUEST, SERVER_ERROR])
def test_pb_non_capacity_create_error_does_not_loop(tmp_path, monkeypatch, payload):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, payload=payload)
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=2.0)
    assert code == 1 and len(rp.creates) == 1 and rp.pods == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 1 and not live and not unreleased and wd.running == []
    st = json.loads((out_dir / "state.json").read_text())
    assert "settle_deferred" not in st and st["pods_deleted_verified"]
    assert not (out_dir / pb.STOCK_ASIDE).exists()


def test_pb_transient_read_before_create_is_retried(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=0)
    real, calls = rp.balance, {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] == 1:
            raise RunPodError("rp.py gql timed out")
        return real()

    rp.balance = flaky
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert code == 0 and rp.made == 1 and len(Ledger.runs) == 1
    rows = _rows(out_dir / pb.STOCK_LOG)
    assert len(rows) == 1 and rows[0]["error"].startswith("transient before create")


def test_pb_stock_wait_options_and_validation(tmp_path):
    a = pb.build_parser().parse_args(["bench", "--dry-run"])
    assert a.stock_wait_hours == 0.0 and a.stock_poll_s == 60.0  # single try by default
    assert pb.check_args(_args(tmp_path, stock_wait_hours=3.0, stock_poll_s=30)) is None
    for kw in (
        {"stock_poll_s": 29},
        {"stock_poll_s": 3601},
        {"stock_poll_s": float("nan")},
        {"stock_wait_hours": -1.0},
        {"stock_wait_hours": 3.5},
        {"stock_wait_hours": float("nan")},
    ):
        assert pb.check_args(_args(tmp_path, **kw)), kw
    assert pb.bench_cmd(_args(tmp_path / "o", confirm=True, stock_poll_s=5)) == 2
    assert not (tmp_path / "o").exists()


def test_pb_ctrl_c_between_rounds_settles_in_the_set_aside_round(tmp_path, monkeypatch):
    """Ctrl-C in the sleep after round 2: that round's files were already set aside, so the
    final settle (deferred_dir retargeted) writes its record there, not into the session dir;
    every round's watchdog is stopped only after the clean settle."""
    clock = Clock(monkeypatch, poll_s=60.0, interrupt_at=3)
    rp = CapacityRp(clock, refusals=10**9)
    with pytest.raises(KeyboardInterrupt):
        _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert len(rp.creates) == 3 and rp.pods == [] and rp.deleted == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 3 and not live and not unreleased
    wd = _wd()
    assert len(wd.spawned) == 3 and wd.running == []
    _assert_backstop(rp, wd)
    assert clock.t - rp.creates[-1]["t"] >= gr.CREATE_SETTLE_S
    out_dir = tmp_path / "session"
    aside = json.loads((out_dir / pb.STOCK_ASIDE / "state.json").read_text())
    assert aside["name"] == rp.creates[-1]["name"]
    assert aside["late_listing_settle"]["pods_deleted_verified"] and aside["watchdog_stopped"]
    assert not (out_dir / "state.json").exists()  # nothing written into the session dir
    assert not (out_dir / "watchdog.stop").exists()  # shared dir: stopped by process group
    assert not (out_dir / pb.STOCK_ARMED).exists()  # every set-aside round was stopped
    assert len(_rows(out_dir / pb.STOCK_LOG)) == 3


def test_pb_late_listed_pod_found_at_the_next_pre_create_check(tmp_path, monkeypatch):
    """A pod made despite round 0's refusal, listed 30 s later: round 1's pre-create check
    finds it and stops the wait; the settle deletes it, then stops round 0's watchdog."""
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=10**9, ghost_after=30)
    with pytest.raises(SystemExit, match="already exists"):
        _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert len(rp.creates) == 1 and rp.deleted == ["ghost"] and rp.list_pods() == []
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 1 and not live and not unreleased
    wd = _wd()
    assert len(wd.spawned) == 1 and wd.running == []
    assert wd.stopped_at[wd.spawned[0]] > rp.deleted_at["ghost"]
    _assert_backstop(rp, wd)
    aside = json.loads((tmp_path / "session" / pb.STOCK_ASIDE / "state.json").read_text())
    assert aside["late_listing_settle"]["pods_deleted_verified"]


def test_pb_set_aside_retargets_the_deferred_settle_and_armed_watchdogs(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    wd = Watchdogs(monkeypatch)
    monkeypatch.setattr(gr.signal, "signal", lambda *a: None)
    run_dir = tmp_path / "session"
    (run_dir / "inputs").mkdir(parents=True)
    keep = {"inputs"}
    wait = gr.StockWait(3600.0, 60.0, run_dir / pb.STOCK_LOG, stop_files=False)

    def refused_round(name):
        pid = wd.spawn(run_dir, clock.t + 3600, name)
        st = {
            "name": name,
            "create_attempted_at": clock.t,
            "settle_deferred": True,
            "pods_deleted_verified": True,
            "watchdog_pid": pid,
            "watchdog_stopped": False,
        }
        gr.save_state(run_dir, st)
        wait.track(run_dir, st)
        return pid

    a = refused_round("perfbench-0204000000")
    aside = pb.set_round_aside(run_dir, keep, wait)
    assert wait.deferred_dir == aside and [e["dir"] for e in wait.pending] == [aside]
    assert not (run_dir / "state.json").exists() and (aside / "watchdog.log").exists()
    clock.t += 60
    b = refused_round("perfbench-0204000100")
    pb.set_round_aside(run_dir, keep, wait)
    armed = run_dir / pb.STOCK_ARMED / "perfbench-0204000000"
    assert [e["dir"] for e in wait.pending] == [armed, aside] and wait.deferred_dir == aside
    assert json.loads((armed / "state.json").read_text())["name"] == "perfbench-0204000000"
    pb.prune_armed(run_dir, wait)
    assert armed.exists()  # its watchdog is still armed: its files stay
    clock.t += gr.CREATE_SETTLE_S - 60  # round a's window has passed, b's has not

    class Rp:
        def list_pods(self):
            return []

    assert [e["name"] for e in wait.reap(Rp())] == ["perfbench-0204000000"]
    assert wd.running == [b] and wd.stopped == [a]
    pb.prune_armed(run_dir, wait)
    assert not (run_dir / pb.STOCK_ARMED).exists()
    assert wait.settle(Rp()) is True and wd.running == []
    st = json.loads((aside / "state.json").read_text())
    assert st["late_listing_settle"]["pods_deleted_verified"] and st["watchdog_stopped"]
    assert not list(run_dir.rglob("watchdog.stop"))  # by process group only (a shared dir)


def test_pb_our_pods_failure_is_a_pre_create_transient(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=0)

    def down():
        raise RunPodError("rp.py ('rest', 'GET', '/pods') exit 1", THROTTLED)

    rp.list_pods = down
    with pytest.raises(gr.PreCreateTransient) as exc:  # single try: raised, nothing made
        _pb_launch(tmp_path, monkeypatch, rp, name="single")
    assert exc.value.throttled
    assert _wd().spawned == [] and Ledger.runs == {} and rp.creates == []
    # in a stock wait the round is retried (backed off: it was a throttle)
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, refusals=0)
    real, calls = rp.list_pods, {"n": 0}

    def once():
        calls["n"] += 1
        if calls["n"] == 1:
            down()
        return real()

    rp.list_pods = once
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert code == 0 and rp.made == 1 and rp.creates[0]["t"] - clock.t0 == 120.0
    rows = _rows(out_dir / pb.STOCK_LOG)
    assert len(rows) == 1 and rows[0]["throttled"]
    assert rows[0]["error"].startswith("transient before create")


def test_pb_throttled_round_backs_off_and_keeps_waiting(tmp_path, monkeypatch):
    clock = Clock(monkeypatch)
    rp = CapacityRp(clock, sequence=[THROTTLED, NO_CAPACITY])
    code, out_dir, wd = _pb_launch(tmp_path, monkeypatch, rp, stock_wait_hours=1.0)
    assert code == 0 and rp.made == 1
    assert [c["t"] - clock.t0 for c in rp.creates] == [0.0, 120.0, 180.0]
    assert [r["throttled"] for r in _rows(out_dir / pb.STOCK_LOG)] == [True, False]
    failed, ok, live, unreleased = Ledger.summary()
    assert len(failed) == 2 and len(ok) == 1 and not live and not unreleased
    assert wd.running == []
    _assert_backstop(rp, wd)
    assert not (out_dir / pb.STOCK_ARMED).exists()
