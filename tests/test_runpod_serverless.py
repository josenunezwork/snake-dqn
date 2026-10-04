"""Tests for the serverless backend of research/runpod_fanout (no network, no RunPod).

The "endpoint" is a fake that runs ``sls_handler.handle`` in-process against a fake volume
directory; the seeder pod is a local ``seed_agent.py`` process on 127.0.0.1.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from research.runpod_fanout import jobspec, ledger, runner, serverless, sls_handler, watchdog
from research.runpod_fanout.rp_client import RpClient, RunPodError
from tests.test_runpod_fanout import GREEDY, RANDOM, STUB, Clock, ep, free_port, job, make_repo

MODEL = "AMD EPYC 9655 96-Core Processor"


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    def fatal_call(self, *args):
        raise AssertionError(f"real rp.py call attempted: {args[:3]}")

    monkeypatch.setattr(RpClient, "_call", fatal_call)
    real_connect = socket.socket.connect

    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network call to {address} attempted")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


def fpol(tmp, **over):
    p = dict(jobspec.load_policy(), artifacts_root=str(Path(tmp) / "artifacts"))
    p.update(over)
    return p


def spol(**over):
    return dict(serverless.load_sls_policy(), **over)


# ---------------------------------------------------------------- fake endpoint


class FakeSls:
    """Fake RunPod for serverless: jobs run ``sls_handler.handle`` when first polled."""

    def __init__(self, models=(MODEL,), never_start=(), bad_output_key=None):
        self.endpoints, self.jobs = {}, {}
        self.created, self.deleted, self.scaled, self.cancelled, self.purged = [], [], [], [], []
        self.models = list(models)
        self.never_start = set(never_start)
        self.bad_output_key = bad_output_key
        self.balance_value = 40.0
        self.lock = threading.Lock()
        self.n = 0
        self.executed = 0
        self.other = {"id": "otherep", "name": "someone-else", "workersMax": 3}

    def balance(self):
        return self.balance_value

    def list_pods(self):
        return []

    def list_endpoints(self):
        return [self.other] + [dict(e) for e in self.endpoints.values()]

    def get_endpoint(self, eid, workers=False):
        e = self.endpoints.get(eid)
        return None if e is None else dict(e, workers=[])

    def create_endpoint(self, body_path, confirm):
        assert confirm is True
        body = json.loads(Path(body_path).read_text())
        self.n += 1
        eid = f"ep{self.n}"
        self.endpoints[eid] = dict(body, id=eid)
        self.created.append(body)
        return {"id": eid, "name": body["name"]}

    def update_endpoint(self, eid, body, confirm):
        assert confirm is True
        if eid == "otherep":
            raise AssertionError("touched an endpoint the runner does not own")
        if eid not in self.endpoints:
            raise RunPodError("nf", {"http_status": 404})
        self.endpoints[eid].update(body)
        self.scaled.append((eid, dict(body)))
        return {}

    def delete_endpoint(self, eid, confirm):
        assert confirm is True
        if eid == "otherep":
            raise AssertionError("deleted an endpoint the runner does not own")
        if eid not in self.endpoints:
            raise RunPodError("nf", {"http_status": 404})
        assert self.endpoints[eid]["workersMax"] == 0, "delete before scale-to-0"
        self.endpoints.pop(eid)
        self.deleted.append(eid)
        return {}

    def endpoint_billing(self, eid, start, end):
        return 0.001

    def sls(self, eid, op, job_id=None, body=None, confirm=False):
        if op == "run":
            assert confirm is True and self.endpoints[eid]["workersMax"] >= 1
            assert set(body) == {"input", "policy"}
            self.n += 1
            jid = f"job{self.n}"
            self.jobs[jid] = {"endpoint": eid, "input": body["input"], "done": None}
            return {"id": jid, "status": "IN_QUEUE"}
        if op in ("cancel", "purge-queue"):
            assert confirm is True
            (self.cancelled if op == "cancel" else self.purged).append((eid, job_id))
            return {}
        assert op == "status"
        j = self.jobs.get(job_id)
        if j is None:
            raise RunPodError("nf", {"http_status": 404})
        if self.endpoints.get(eid, {}).get("cpuFlavorIds", [""])[0] in self.never_start:
            return {"id": job_id, "status": "IN_QUEUE"}
        if j["done"] is None:
            with self.lock:
                model = self.models[min(self.executed, len(self.models) - 1)]
                self.executed += 1
                orig = sls_handler.cpu_model
                sls_handler.cpu_model = lambda: model
                try:
                    out = sls_handler.handle(j["input"])
                finally:
                    sls_handler.cpu_model = orig
                if self.bad_output_key and out.get("results"):
                    out["results"][0]["key"] = self.bad_output_key
                j["done"] = {
                    "id": job_id,
                    "status": "COMPLETED",
                    "output": out,
                    "executionTime": 1500,
                    "workerId": f"w-{model[:4]}",
                }
        return j["done"]


def build_volume(tmp, repo, commit, ckpt_bytes, fp, sp):
    """A fake volume with the commit archive, the checkpoint and a READY runtime."""
    root = Path(tmp) / "vol" / "rpf"
    cdir = root / "commits" / commit
    cdir.mkdir(parents=True)
    repo_sha = jobspec.git_archive(repo, commit, cdir / "repo.tar.gz")
    (cdir / "manifest.json").write_text(json.dumps({"commit": commit, "repo_sha256": repo_sha}))
    (root / "ckpt").mkdir()
    sha = hashlib.sha256(ckpt_bytes).hexdigest()
    (root / "ckpt" / f"{sha}.pth").write_bytes(ckpt_bytes)
    rid = serverless.runtime_id(fp, sp)
    rdir = root / "runtime" / rid
    rdir.mkdir(parents=True)
    ready = {
        "runtime_id": rid,
        "handler_sha256": serverless.handler_sha256(),
        "torch": "2.9.1+cpu",
        "numpy": "1.26.4",
        "python": "3.12.x",
    }
    (rdir / "READY.json").write_text(json.dumps(ready))
    return root, rdir, repo_sha, rid, ready


@pytest.fixture
def world(tmp_path, monkeypatch):
    """Repo + checkpoint + fake volume + registry + handler wiring."""
    repo, commit = make_repo(tmp_path)
    ck = tmp_path / "ck"
    ck.mkdir()
    data = b"fake hero weights"
    hero = hashlib.sha256(data).hexdigest()
    (ck / "hero.pth").write_bytes(data)
    fp = fpol(tmp_path, checkpoint_root=str(ck))
    sp = spol(poll_seconds=5, startup_deadline_seconds=120, cold_start_seconds_est=60)
    root, rdir, repo_sha, rid, ready = build_volume(tmp_path, repo, commit, data, fp, sp)
    monkeypatch.setattr(sls_handler, "ROOT", root)
    monkeypatch.setattr(sls_handler, "RUNTIME_DIR", rdir)
    monkeypatch.setattr(sls_handler, "LOCAL", tmp_path / "worker-local")
    sls_handler._VERIFIED.clear()
    (tmp_path / "stub.py").write_text(STUB)
    state = tmp_path / "stubstate"
    state.mkdir()
    monkeypatch.setenv(
        "RPF_TEST_EPISODE_EXEC_JSON", json.dumps([sys.executable, str(tmp_path / "stub.py")])
    )
    monkeypatch.setenv("STUB_STATE", str(state))
    reg = serverless.Registry(fp)
    with reg.locked() as st:
        st["volume"] = {"id": "vol1", "dataCenterId": "EU-RO-1", "size": 5}
        st["commits"][commit] = {"repo_sha256": repo_sha}
        st["ckpts"][hero] = "t"
        st["runtimes"][rid] = {"ready": ready, "template_id": "tpl1"}
    allow = {hero: {"path": "hero.pth", "sha256": hero}}
    return {
        "repo": repo,
        "commit": commit,
        "hero": hero,
        "fp": fp,
        "sp": sp,
        "allow": allow,
        "root": root,
        "rid": rid,
        "repo_sha": repo_sha,
        "tmp": tmp_path,
        "reg": reg,
    }


def make_sls_runner(w, episodes, fake, budget=3.0, workers=2, vcpu=4, flavors=("cpu5c",)):
    j = job(episodes, repo_commit=w["commit"], checkpoints=[w["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [w["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    clock = Clock()
    return serverless.ServerlessRunner(
        j,
        w["fp"],
        w["allow"],
        w["tmp"] / "runs" / "sls-r1",
        sls_policy=w["sp"],
        sizing={"workers": workers, "vcpu_per_worker": vcpu},
        flavors=list(flavors),
        registry=w["reg"],
        rp=fake,
        budget=budget,
        confirm=True,
        repo=w["repo"],
        clock=clock,
        sleep=clock.sleep,
        spawn_watchdog=lambda r: None,
        preflight_fn=lambda: [],
        probe_workers=2,
    )


def eps6():
    return [ep(a, seed=s, index=i) for a in "AB" for i, s in enumerate((11, 12, 13))]


# ---------------------------------------------------------------- end to end (fake)


def test_serverless_end_to_end(world):
    fake = FakeSls()
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    recs = sorted((r.run_dir / "records").rglob("*.json"))
    assert len(recs) == 6
    one = json.loads(next((r.run_dir / "records").rglob("A-scripted-11.json")).read_text())
    assert one["fanout"]["job_id"] == "unit-job" and one["platform"]["platform_id"] == "stub"
    body = fake.created[0]
    assert body["workersMin"] == 0 and body["workersMax"] == 2 and body["vcpuCount"] == 4
    assert body["cpuFlavorIds"] == ["cpu5c"] and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["networkVolumeId"] == "vol1" and body["templateId"] == "tpl1"
    assert body["name"].startswith("rpf-sls-unit-job--sls-r1-")
    # torn down: scaled to 0 (asserted by the fake) then deleted; nobody else touched
    assert fake.deleted == ["ep1"] and fake.endpoints == {}
    assert ("ep1", {"workersMin": 0, "workersMax": 0}) in fake.scaled
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["backend"] == "serverless" and receipt["exit_code"] == 0
    assert receipt["final_endpoints_runner_owned"] == [] and receipt["completed"] == 6
    assert receipt["cpu_model_pin"] == MODEL
    # the episode jobs carried the pin, the provider and <= slots episodes each
    sub = [j["input"] for j in fake.jobs.values() if j["input"]["op"] == "episodes"]
    assert all(i["require_cpu_model"] == MODEL for i in sub)
    assert all(i["provider"] == "runpod-serverless:cpu5c" and len(i["episodes"]) <= 2 for i in sub)
    # ledger: one serverless reservation, released with a positive cost
    state = json.loads(
        (Path(world["fp"]["artifacts_root"]) / "runpod-fanout/ledger-v2.json").read_text()
    )
    rows = [p for run in state["runs"].values() for p in run["pods"].values()]
    assert len(rows) == 1 and rows[0]["kind"] == "serverless" and rows[0]["pod_id"] is None
    assert rows[0]["deleted"] is not None and rows[0]["cost"] > 0
    assert rows[0]["endpoint_id"] == "ep1"


def test_failed_episode_retried_once(world, monkeypatch):
    monkeypatch.setenv("STUB_FAIL_ONCE", "apex-veto-v8-screen-v1__h1000__live/A-scripted-12.json")
    fake = FakeSls()
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    assert r.attempts["apex-veto-v8-screen-v1__h1000__live/A-scripted-12.json"] == 1
    assert list((r.run_dir / "failures").glob("*.json"))


def test_cpu_model_mismatch_is_refused_and_requeued_free(world):
    fake = FakeSls(models=[MODEL, "Intel Xeon Other", MODEL])
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    assert r.refusals == 1 and all(v == 0 for v in r.attempts.values())
    platforms = {
        json.loads(p.read_text())["platform"]["platform_id"]
        for p in (r.run_dir / "records").rglob("*.json")
    }
    assert platforms == {"stub"}


def test_falls_back_to_cpu3c_when_no_cpu5c_worker_starts(world):
    fake = FakeSls(never_start={"cpu5c"})
    r = make_sls_runner(world, eps6()[:2], fake, flavors=("cpu5c", "cpu3c"))
    assert r.run() == 0
    assert [b["cpuFlavorIds"] for b in fake.created] == [["cpu5c"], ["cpu3c"]]
    assert len(fake.deleted) == 2 and fake.endpoints == {}


def test_unknown_record_aborts_and_tears_down(world):
    fake = FakeSls(bad_output_key="apex-veto-v8-screen-v1__h1000__live/Z-x-1.json")
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 3
    assert fake.endpoints == {} and fake.deleted


def test_budget_refuses_endpoint_and_nothing_is_created(world):
    fake = FakeSls()
    r = make_sls_runner(world, eps6(), fake, budget=0.01, workers=4, vcpu=32)
    assert r.run() == 3
    assert fake.created == []


def test_not_seeded_refuses_before_any_endpoint(world):
    with world["reg"].locked() as st:
        st["commits"] = {}
    fake = FakeSls()
    r = make_sls_runner(world, eps6(), fake)
    with pytest.raises(jobspec.JobError, match="not seeded"):
        r.run()
    assert fake.created == []


# ---------------------------------------------------------------- handler


def handler_input(w, **over):
    spec = dict(ep("A", seed=11), roster_member_sha256s=[w["hero"], RANDOM, GREEDY, RANDOM, GREEDY])
    inp = {
        "op": "episodes",
        "runtime_id": w["rid"],
        "commit": w["commit"],
        "repo_sha256": w["repo_sha"],
        "checkpoints": [w["hero"]],
        "job_id": "unit-job",
        "provider": "runpod-serverless:cpu5c",
        "slots": 2,
        "timeout_seconds": 60,
        "deadline_epoch": time.time() + 600,
        "require_cpu_model": "unknown",
        "episodes": [{"key": jobspec.episode_key(spec), "spec": spec}],
    }
    inp.update(over)
    return inp


def test_handler_runs_episode_and_strips_secrets(world, monkeypatch, tmp_path):
    stub = tmp_path / "envstub.py"
    stub.write_text(
        "import os, sys\nassert not any(k for k in os.environ if 'KEY' in k or "
        "'TOKEN' in k), sorted(os.environ)\n" + STUB
    )
    monkeypatch.setenv("RPF_TEST_EPISODE_EXEC_JSON", json.dumps([sys.executable, str(stub)]))
    monkeypatch.setenv("RUNPOD_API_KEY", "must-not-reach-the-episode")
    monkeypatch.setenv("RUNPOD_AI_API_KEY", "must-not-reach-the-episode")
    monkeypatch.setattr(sls_handler, "cpu_model", lambda: "unknown")
    out = sls_handler.handle(handler_input(world))
    assert out["ok"] is True, out
    (res,) = out["results"]
    assert res["ok"] and hashlib.sha256(res["record"].encode()).hexdigest() == res["sha256"]
    assert json.loads(res["record"])["fanout"]["job_id"] == "unit-job"


@pytest.mark.parametrize(
    "over, why",
    [
        ({"runtime_id": "0" * 16}, "runtime"),
        ({"repo_sha256": "1" * 64}, "sha256"),
        ({"require_cpu_model": "Other CPU"}, "cpu_model"),
        ({"require_cpu_model": None}, "cpu_model"),
        ({"job_id": "Bad Id"}, "job id"),
        ({"slots": 99}, "range"),
    ],
)
def test_handler_refusals(world, monkeypatch, over, why):
    monkeypatch.setattr(sls_handler, "cpu_model", lambda: "unknown")
    out = sls_handler.handle(handler_input(world, **over))
    assert why in str(out.get("refused")), out
    if why == "cpu_model":
        assert out["refresh_worker"] is True


def test_handler_refuses_key_spec_mismatch_and_traversal(world, monkeypatch):
    monkeypatch.setattr(sls_handler, "cpu_model", lambda: "unknown")
    inp = handler_input(world)
    inp["episodes"][0]["key"] = "apex-veto-v8-screen-v1__h1000__live/B-scripted-11.json"
    assert "does not match" in sls_handler.handle(inp)["refused"]
    inp["episodes"][0]["key"] = "../../etc/x.json"
    assert "bad episode key" in sls_handler.handle(inp)["refused"]


def test_handler_probe_reports_worker(world, monkeypatch):
    monkeypatch.setattr(sls_handler, "cpu_model", lambda: MODEL)
    out = sls_handler.handle(
        {
            "op": "probe",
            "runtime_id": world["rid"],
            "commit": world["commit"],
            "repo_sha256": world["repo_sha"],
            "checkpoints": [world["hero"]],
        }
    )
    assert out["ok"] and out["worker"]["cpu_model"] == MODEL
    assert out["worker"]["runtime_id"] == world["rid"]


# ---------------------------------------------------------------- bodies / sizing / policy


def test_template_endpoint_and_seeder_bodies_have_no_secrets(tmp_path):
    fp, sp = fpol(tmp_path), spol()
    rid = serverless.runtime_id(fp, sp)
    t = serverless.template_body(sp, rid)
    assert t["isServerless"] is True and t["isPublic"] is False and t["volumeInGb"] == 0
    assert "@sha256:" in t["imageName"]
    assert f"/runpod-volume/rpf/runtime/{rid}/sls_handler.py" in t["dockerStartCmd"][2]
    e = serverless.endpoint_body(
        sp, "rpf-sls-x--y-1", "tpl", {"id": "v", "dataCenterId": "EU-RO-1"}, "cpu5c", 3, 8, 600
    )
    assert (
        e["workersMin"] == 0
        and e["cpuFlavorIds"] == ["cpu5c"]
        and e["executionTimeoutMs"] == 600000
    )
    s = serverless.seeder_pod_body(
        fp, sp, "rpf-seed-x--y-1", {"id": "v", "dataCenterId": "EU-RO-1"}, "ab" * 32, 123, "seed-x"
    )
    assert s["networkVolumeId"] == "v" and s["volumeMountPath"] == "/runpod-volume"
    assert s["imageName"] == t["imageName"] and s["cpuFlavorIds"] == ["cpu5c"]
    assert s["vcpuCount"] == 2 and s["cloudType"] == "SECURE"
    for body in (t, e, s):
        text = json.dumps(body.get("env", {})).upper()
        assert "API_KEY" not in text and "RUNPOD_" not in text
    spec = json.loads(s["env"]["SEED_RUNTIME_JSON"])
    assert spec["evenv_pip"][0][3:] == fp["pip_pins"] and "torch==2.9.1" in spec["evenv_pip"][1]
    assert spec["expect"] == {"torch": "2.9.1+cpu", "numpy": "1.26.4"}


def test_runtime_id_tracks_handler_and_pins(tmp_path, monkeypatch):
    fp, sp = fpol(tmp_path), spol()
    a = serverless.runtime_id(fp, sp)
    assert a == serverless.runtime_id(fp, sp) and len(a) == 16
    assert serverless.runtime_id(dict(fp, torch_pin="torch==2.9.0"), sp) != a
    assert serverless.runtime_id(fp, dict(sp, image="python:3.12-slim@sha256:" + "0" * 64)) != a


def test_policy_guards(tmp_path):
    p = tmp_path / "p.json"
    base = serverless.load_sls_policy()
    p.write_text(json.dumps(dict(base, image="python:3.12-slim")))
    with pytest.raises(jobspec.JobError, match="digest"):
        serverless.load_sls_policy(p)
    p.write_text(json.dumps(dict(base, flavors_pref=["cpu5g"])))
    with pytest.raises(jobspec.JobError):
        serverless.load_sls_policy(p)


def test_sizing_prefers_cheapest_within_target_and_respects_limits(tmp_path):
    fp, sp = fpol(tmp_path), spol()
    eps = [dict(ep("A", seed=s), horizon=5000) for s in range(400)]
    plan = serverless.sizing_plan(
        fp, sp, eps, budget=20.0, max_wall_minutes=90, flavor="cpu5c", headroom=30.0
    )
    c = plan["choice"]
    assert plan["choice_ok"] and c["wall_minutes"] <= 30 and not c["refused"]
    assert c["worst_case_usd"] <= 20.0
    walls = [r["wall_minutes"] for r in plan["speed_cost_tradeoff"]]
    assert walls == sorted(walls)
    costs = [r["expected_usd"] for r in plan["speed_cost_tradeoff"]]
    assert costs == sorted(costs, reverse=True)
    tiny = serverless.sizing_plan(fp, sp, eps, budget=0.05, max_wall_minutes=90, flavor="cpu5c")
    assert not tiny["choice_ok"]
    forced = serverless.sizing_plan(fp, sp, eps, 20.0, 90, "cpu5c", workers=3, vcpu=8)
    assert (forced["choice"]["workers"], forced["choice"]["vcpu_per_worker"]) == (3, 8)
    capped = serverless.sizing_plan(fp, sp, eps, 1000.0, 90, "cpu5c", headroom=1.0)
    assert all(r["worst_case_usd"] <= 1.0 for r in capped["speed_cost_tradeoff"])


# ---------------------------------------------------------------- ledger compatibility


def old_ledger_module():
    """The ledger.py the LIVE pod runs use (main @ 8b05817), loaded from git."""
    src = subprocess.run(
        ["git", "-C", str(runner.REPO), "show", "8b05817:research/runpod_fanout/ledger.py"],
        capture_output=True,
        text=True,
    )
    if src.returncode != 0:
        pytest.skip("commit 8b05817 not available")
    spec = importlib.util.spec_from_loader("old_ledger", loader=None)
    mod = importlib.util.module_from_spec(spec)
    exec(compile(src.stdout, "old_ledger.py", "exec"), mod.__dict__)
    return mod


class ListPods:
    def __init__(self, rows=()):
        self.rows = list(rows)

    def list_pods(self):
        return self.rows

    def delete_pod(self, pod_id, confirm):
        raise AssertionError("old reconcile must not delete anything here")


def test_serverless_reservation_is_honoured_by_the_old_ledger_code(tmp_path):
    fp = fpol(tmp_path)
    t = [1000.0]
    new = ledger.SharedLedger(fp, clock=lambda: t[0], alive=lambda *a: True)
    new.register_run("run-sls", "job-sls", "d", 5.0, os.getpid())
    assert (
        new.reserve_pod(
            "run-sls", "sls:rpf-sls-job-sls--r-1", 4.0, 1000 + 3600, 40.0, kind="serverless"
        )
        is None
    )
    old_mod = old_ledger_module()
    old = old_mod.SharedLedger(fp, clock=lambda: t[0] + 600, alive=lambda *a: False)
    tot = old.totals(json.loads(new.path.read_text()))
    assert tot["reserved_usd"] > 3.0 and tot["live_pods"] == 1
    # an old runner reconciling (run looks dead to it) keeps the reservation before horizon
    t[0] += 600
    assert old.reconcile(ListPods(), delete_dead=True)["released"] == []
    assert new.peek()["reserved_usd"] > 3.0
    # an old runner's own reservation still fails when the serverless one eats the headroom
    old.register_run("run-old", "job-old", "d2", 5.0, os.getpid())
    why = old.reserve_pod("run-old", "rpf-job-old--r-1", 1.0, t[0] + 5400, 8.0)
    assert old.reserve_pod("run-old", "rpf-job-old--r-2", 1.0, t[0] + 5400, 40.0) is None
    assert why and "floor" in why
    # the new code releases with the settled cost; both readers agree it is spent, not reserved
    before = old.totals(json.loads(new.path.read_text()))
    new.release_pod("run-sls", "sls:rpf-sls-job-sls--r-1", 0.25)
    tot = old.totals(json.loads(new.path.read_text()))
    assert tot["reserved_usd"] < before["reserved_usd"] - 2.5 and tot["live_pods"] == 1


def test_new_reconcile_skips_live_serverless_rows_and_frees_dead_ones(tmp_path):
    fp = fpol(tmp_path)
    t = [1000.0]
    alive = {"v": True}
    lg = ledger.SharedLedger(fp, clock=lambda: t[0], alive=lambda *a: alive["v"])
    lg.register_run("r", "j", "d", 5.0, os.getpid())
    lg.reserve_pod("r", "sls:rpf-sls-j--x-1", 2.0, 1000 + 1800, 40.0, kind="serverless")
    t[0] += 3600
    assert lg.reconcile(ListPods())["released"] == []  # live run: untouched
    alive["v"] = False
    out = lg.reconcile(ListPods())
    assert out["released"] == ["sls:rpf-sls-j--x-1"]
    row = json.loads(lg.path.read_text())["runs"]["r"]["pods"]["sls:rpf-sls-j--x-1"]
    assert row["cost"] == pytest.approx(2.0 * 1800 / 3600)


# ---------------------------------------------------------------- watchdog


def test_watchdog_tears_down_only_this_runs_endpoints(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "endpoints").mkdir(parents=True)
    mine = "rpf-sls-unit-job--sls-r1-"
    (run_dir / "endpoints" / f"{mine}1.response.json").write_text(
        json.dumps({"id": "e1", "name": f"{mine}1", "rpf_until_epoch": time.time() + 999})
    )
    (run_dir / "receipt.json").write_text("{}")
    state = tmp_path / "fake.json"
    state.write_text(
        json.dumps(
            {
                "pods": [],
                "endpoints": [
                    {"id": "e1", "name": f"{mine}1", "workersMax": 4},
                    {
                        "id": "e2",
                        "name": f"{mine}2 -fb",
                        "workersMax": 4,
                    },  # surfaced late, listed only
                    {
                        "id": "e3",
                        "name": "rpf-sls-unit-job--sls-r10-1",
                        "workersMax": 4,
                    },  # other run
                    {"id": "e4", "name": "someone-else", "workersMax": 4},
                ],
            }
        )
    )
    reason = watchdog.watch(
        "unit-job",
        time.time() + 999,
        run_dir,
        os.getpid(),
        rp=watchdog.FileFakeRp(str(state)),
        sleep=lambda s: None,
        prefix="rpf-unit-job--sls-r1-",
        endpoint_prefix=mine,
        resweep_seconds=0,
    )
    assert reason == "runner finished"
    data = json.loads(state.read_text())
    assert sorted(data["deleted_endpoints"]) == ["e1", "e2"]
    assert sorted(e["id"] for e in data["endpoints"]) == ["e3", "e4"]
    assert set(data["scaled"]) == {"e1", "e2"}
    with pytest.raises(SystemExit):
        watchdog.watch(
            "unit-job",
            0,
            run_dir,
            os.getpid(),
            rp=watchdog.FileFakeRp(str(state)),
            sleep=lambda s: None,
            endpoint_prefix="rpf-sls-other--x-",
        )


def test_watchdog_argv_carries_the_endpoint_prefix(world):
    r = make_sls_runner(world, eps6(), FakeSls())
    r.deadline = time.time() + 60
    argv = runner.watchdog_argv(r, None)
    assert argv[argv.index("--endpoint-prefix") + 1] == "rpf-sls-unit-job--sls-r1-"


# ---------------------------------------------------------------- seed agent + seeder


def start_seed_agent(tmp, token, runtime_json, port):
    env = dict(
        os.environ,
        FANOUT_TOKEN_SHA256=hashlib.sha256(token.encode()).hexdigest(),
        SEED_ROOT=str(tmp / "vol" / "rpf"),
        SEED_STAGE=str(tmp / "stage"),
        SEED_LOG=str(tmp / "seed.log"),
        SEED_RUNTIME_JSON=json.dumps(runtime_json),
        FANOUT_PORT=str(port),
    )
    proc = subprocess.Popen(
        [sys.executable, str(serverless.SEED_AGENT_SOURCE)],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.05)
    return proc


TEST_RUNTIME = {
    "evenv_pip": [],
    "hvenv_pip": [],
    "expect": {"torch": "2.9.1+cpu"},
    "probe_code": "import json,sys;print(json.dumps({'torch':'2.9.1+cpu',"
    "'numpy':'1.26.4','python':sys.version.split()[0]}))",
}


def test_seed_agent_publishes_immutably_and_builds_runtime(tmp_path):
    import urllib.error

    repo, commit = make_repo(tmp_path)
    archive = tmp_path / "a.tar.gz"
    repo_sha = jobspec.git_archive(repo, commit, archive)
    port, token = free_port(), "tok-" + "x" * 30
    proc = start_seed_agent(tmp_path, token, TEST_RUNTIME, port)
    try:
        c = runner.AgentClient(f"http://127.0.0.1:{port}", token, timeout=60)
        bad = runner.AgentClient(f"http://127.0.0.1:{port}", "wrong", timeout=5)
        with pytest.raises(urllib.error.HTTPError):
            bad.get_json("/health")
        assert c.put_file(f"/in/commit/{commit}.tar.gz", archive)["sha256"] == repo_sha
        assert (
            c.post_json("/commit", {"commit": commit, "repo_sha256": repo_sha})["status"]
            == "published"
        )
        assert (
            c.post_json("/commit", {"commit": commit, "repo_sha256": repo_sha})["status"]
            == "exists"
        )
        other = tmp_path / "other.tar.gz"
        other.write_bytes(archive.read_bytes() + b"x")
        c.put_file(f"/in/commit/{commit}.tar.gz", other)
        with pytest.raises(urllib.error.HTTPError) as e409:
            c.post_json("/commit", {"commit": commit, "repo_sha256": jobspec.sha256_file(other)})
        assert e409.value.code == 409
        ck = tmp_path / "w.pth"
        ck.write_bytes(b"weights")
        with pytest.raises(urllib.error.HTTPError):
            c.put_file(f"/in/ckpt/{'0' * 64}.pth", ck)
        sha = jobspec.sha256_file(ck)
        assert c.put_file(f"/in/ckpt/{sha}.pth", ck)["sha256"] == sha
        rid = "0123456789abcdef"
        c.put_file(f"/in/handler/{rid}", serverless.HANDLER_SOURCE)
        out = c.post_json(
            "/runtime", {"runtime_id": rid, "handler_sha256": serverless.handler_sha256()}
        )
        assert out["status"] == "building"
        for _ in range(600):
            h = c.get_json("/health")
            if h["build"].get(rid) != "building":
                break
            time.sleep(0.1)
        assert h["build"][rid] == "ready", c.get_bytes("/log")[-800:]
        rep = c.post_json("/verify", {})
        assert rep["bad"] == [] and rep["checked"] == 3
        inv = rep["inventory"]
        assert inv["commits"] == {commit: repo_sha} and inv["ckpts"] == [sha]
        assert inv["runtimes"][rid]["handler_sha256"] == serverless.handler_sha256()
        with pytest.raises(urllib.error.HTTPError):  # READY runtimes are immutable
            c.put_file(f"/in/handler/{rid}", serverless.HANDLER_SOURCE)
        root = tmp_path / "vol" / "rpf"
        assert (root / "runtime" / rid / "evenv" / "bin" / "python").exists()
        assert not (root / "commits" / commit / "untracked_secret.txt").exists()
    finally:
        proc.kill()


class SeedFakeRp:
    """Fake rp for SeedRunner: create_pod starts a local seed agent on 127.0.0.1."""

    def __init__(self, tmp):
        self.tmp, self.procs, self.ports, self.names = Path(tmp), {}, {}, {}
        self.created, self.deleted, self.n = [], [], 0

    def balance(self):
        return 40.0

    def stock(self, flavor, vcpu, ram, dc):
        assert (flavor, vcpu, ram, dc) == ("cpu5c", 2, 4, "EU-RO-1")
        return {"stockStatus": "High", "securePrice": 0.07}

    def create_pod(self, body_path, max_hourly, confirm):
        assert confirm is True
        body = json.loads(Path(body_path).read_text())
        assert body["networkVolumeId"] == "vol1" and body["dataCenterIds"] == ["EU-RO-1"]
        self.n += 1
        pid, port = f"pod{self.n}", free_port()
        env = dict(os.environ, **body["env"])
        env.update(
            SEED_ROOT=str(self.tmp / "vol" / "rpf"),
            SEED_STAGE=str(self.tmp / "stage"),
            SEED_LOG=str(self.tmp / "seed.log"),
            SEED_RUNTIME_JSON=json.dumps(TEST_RUNTIME),
            FANOUT_PORT=str(port),
        )
        self.procs[pid] = subprocess.Popen(
            [sys.executable, str(serverless.SEED_AGENT_SOURCE)],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self.ports[pid], self.names[pid] = port, body["name"]
        self.created.append(body)
        for _ in range(100):
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
                break
            except OSError:
                time.sleep(0.05)
        return {"id": pid, "costPerHr": 0.07}

    def delete_pod(self, pod_id, confirm):
        assert confirm is True
        if pod_id in self.procs:
            self.procs.pop(pod_id).kill()
            self.names.pop(pod_id)
            self.deleted.append(pod_id)
        return {}

    def get_pod(self, pod_id):
        return {"id": pod_id} if pod_id in self.procs else None

    def list_pods(self):
        return [{"id": i, "name": n} for i, n in self.names.items()]

    def close(self):
        for p in self.procs.values():
            p.kill()


def test_seed_runner_seeds_registers_and_deletes_the_pod(tmp_path):
    repo, commit = make_repo(tmp_path)
    ck = tmp_path / "ck"
    ck.mkdir()
    (ck / "hero.pth").write_bytes(b"hero")
    hero = hashlib.sha256(b"hero").hexdigest()
    fp = fpol(tmp_path, checkpoint_root=str(ck))
    sp = spol()
    allow = {hero: {"path": "hero.pth", "sha256": hero}}
    fake = SeedFakeRp(tmp_path)
    clock = Clock()

    def make(name):
        return serverless.SeedRunner(
            fp,
            sp,
            allow,
            tmp_path / "seedruns" / name,
            commit=commit,
            volume={"id": "vol1", "dataCenterId": "EU-RO-1"},
            ckpts=[hero],
            rp=fake,
            budget=0.2,
            confirm=True,
            repo=repo,
            clock=clock,
            sleep=clock.sleep,
            agent_factory=lambda pod_id, token: runner.AgentClient(
                f"http://127.0.0.1:{fake.ports[pod_id]}", token, timeout=30
            ),
            spawn_watchdog=lambda r: None,
            preflight_fn=lambda: [],
        )

    try:
        r = make("s1")
        assert r.run() == 0, (r.run_dir / "events.jsonl").read_text()[-1500:]
        assert fake.deleted == ["pod1"] and fake.procs == {}
        body = fake.created[0]
        assert body["name"].startswith(f"rpf-seed-{commit[:8]}-") and body["vcpuCount"] == 2
        reg = serverless.Registry(fp).read()
        rid = serverless.runtime_id(fp, sp)
        assert reg["commits"][commit]["repo_sha256"] == jobspec.git_archive(
            repo, commit, tmp_path / "x.tgz"
        )
        assert hero in reg["ckpts"] and reg["runtimes"][rid]["ready"]["runtime_id"] == rid
        assert serverless.seed_problems(
            reg, commit, reg["commits"][commit]["repo_sha256"], [hero], rid
        ) == [f"no serverless template for runtime {rid} (serverless.py template-create)"]
        assert make("s2").run() == runner.EXIT_DUPLICATE  # same commit+runtime: idempotent
    finally:
        fake.close()


# ---------------------------------------------------------------- CLI


def test_cli_default_backend_is_serverless(tmp_path, monkeypatch, capsys):
    jpath = tmp_path / "job.json"
    jpath.write_text(json.dumps(job()))
    assert runner.main(["run", str(jpath), "--confirm"]) == 2
    assert "--backend serverless needs --budget" in capsys.readouterr().err
    assert runner.main(["run", str(jpath), "--backend", "runpod", "--confirm"]) == 2
    assert "--backend pods needs --budget" in capsys.readouterr().err


def test_plan_serverless_reports_seed_state_and_sizing(world):
    class PlanRp(FakeSls):
        pass

    j = job(eps6(), repo_commit=world["commit"], checkpoints=[world["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    out = serverless.plan_serverless(
        j, world["fp"], world["sp"], world["allow"], 2.0, rp=PlanRp(), repo=world["repo"]
    )
    assert out["seeding"]["ready"] is True and out["commit_problems"] == []
    assert out["sizing"]["cpu5c"]["choice_ok"] and out["flavors_in_order"] == ["cpu5c", "cpu3c"]
    body = out["endpoint_body_dry_run"]
    assert body["workersMin"] == 0 and body["templateId"] == "tpl1"
    assert out["storage"]["usd_per_month"] == pytest.approx(0.35)
    assert [e["name"] for e in out["existing_runner_endpoints"]] == []
