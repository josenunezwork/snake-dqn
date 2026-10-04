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

from research.runpod_fanout import (
    jobspec,
    ledger,
    platform_rule,
    runner,
    serverless,
    sls_handler,
    watchdog,
)
from research.runpod_fanout.rp_client import RpClient, RunPodError
from tests import test_runpod_fanout as tfo
from tests.test_runpod_fanout import GREEDY, RANDOM, STUB, Clock, ep, free_port, job, make_repo

MODEL = "AMD EPYC 9655 96-Core Processor"
OTHER = "Intel Xeon Gold 6338"
# STUB plus the numerics env the episode process saw (recorded inside "record").
ENV_STUB = STUB.replace(
    '"probes": {"x": {"apply_seconds_total": time.time()}}},',
    '"probes": {"x": {"apply_seconds_total": time.time()}}, "env": {'
    '"MKL_CBWR": os.environ.get("MKL_CBWR"), "OMP": os.environ.get("OMP_NUM_THREADS"), '
    '"MKL": os.environ.get("MKL_NUM_THREADS")}},',
)
assert ENV_STUB != STUB


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


ISA = {MODEL: ["avx2", "avx512f"], OTHER: ["avx2"]}


class FakeSls:
    """Fake RunPod for serverless: a job runs ``sls_handler.handle`` once polled enough."""

    def __init__(
        self,
        models=(MODEL,),
        never_start=False,
        bad_output_key=None,
        fail_unit_once=(),
        delete_failures=0,
        create_mode="ok",
        running_polls=0,
    ):
        self.endpoints, self.jobs = {}, {}
        self.created, self.deleted, self.scaled, self.cancelled, self.purged = [], [], [], [], []
        self.models = list(models)
        self.never_start = never_start
        self.bad_output_key = bad_output_key
        self.fail_unit_once = set(fail_unit_once)
        self.delete_failures = delete_failures
        self.create_mode = create_mode
        self.running_polls = running_polls
        self.balance_value = 40.0
        self.billed = 0.001
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
        self.created.append(body)
        if self.create_mode == "5xx-absent":
            raise RunPodError("gateway", {"http_status": 502})
        self.endpoints[eid] = dict(body, id=eid)
        if self.create_mode == "5xx-created":
            raise RunPodError("gateway", {"http_status": 502})
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
        if self.delete_failures > 0:
            self.delete_failures -= 1
            raise RunPodError("busy", {"http_status": 500})
        self.endpoints.pop(eid)
        self.deleted.append(eid)
        return {}

    def endpoint_billing(self, eid, start, end):
        return self.billed

    def sls(self, eid, op, job_id=None, body=None, confirm=False):
        if op == "run":
            assert confirm is True and self.endpoints[eid]["workersMax"] >= 1
            assert set(body) == {"input", "policy"}
            self.n += 1
            jid = f"job{self.n}"
            self.jobs[jid] = {"endpoint": eid, "input": body["input"], "done": None, "polls": 0}
            return {"id": jid, "status": "IN_QUEUE"}
        if op in ("cancel", "purge-queue"):
            assert confirm is True
            (self.cancelled if op == "cancel" else self.purged).append((eid, job_id))
            return {}
        assert op == "status"
        j = self.jobs.get(job_id)
        if j is None:
            raise RunPodError("nf", {"http_status": 404})
        if self.never_start:
            return {"id": job_id, "status": "IN_QUEUE"}
        j["polls"] += 1
        if j["done"] is None and j["polls"] <= self.running_polls:
            return {"id": job_id, "status": "IN_PROGRESS"}
        if j["done"] is None:
            with self.lock:
                model = self.models[self.executed % len(self.models)]
                self.executed += 1
                keys = [e["key"] for e in j["input"].get("episodes", [])]
                unit = platform_rule.unit_of_key(keys[0]) if keys else None
                if unit in self.fail_unit_once:
                    self.fail_unit_once.discard(unit)
                    j["done"] = {"id": job_id, "status": "FAILED", "error": "worker died"}
                    return j["done"]
                orig_cpu, orig_isa = sls_handler.cpu_model, sls_handler.isa_flags
                sls_handler.cpu_model = lambda: model
                sls_handler.isa_flags = lambda: list(ISA.get(model, []))
                try:
                    out = sls_handler.handle(j["input"])
                finally:
                    sls_handler.cpu_model, sls_handler.isa_flags = orig_cpu, orig_isa
                out.setdefault("worker", {})["worker_id"] = f"w{self.executed}-{model[:5]}"
                if self.bad_output_key and out.get("results"):
                    out["results"][0]["key"] = self.bad_output_key
                j["done"] = {
                    "id": job_id,
                    "status": "COMPLETED",
                    "output": out,
                    "executionTime": 1500,
                    "workerId": f"w{self.executed}",
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
    (tmp_path / "stub.py").write_text(ENV_STUB)
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


def make_sls_runner(
    w,
    episodes,
    fake,
    budget=3.0,
    workers=2,
    vcpu=4,
    flavors=("cpu5c", "cpu3c"),
    name="sls-r1",
    fp=None,
    **kw,
):
    j = job(episodes, repo_commit=w["commit"], checkpoints=[w["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [w["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    clock = Clock()
    return serverless.ServerlessRunner(
        j,
        fp or w["fp"],
        w["allow"],
        w["tmp"] / "runs" / name,
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
        **kw,
    )


def eps6():
    """Three worlds x arms A, B: three units of two episodes each."""
    return [ep(a, seed=s, index=i) for a in "AB" for i, s in enumerate((11, 12, 13))]


def records_of(run_dir):
    return runner.load_records(run_dir)


def ledger_rows(w):
    state = json.loads(
        (Path(w["fp"]["artifacts_root"]) / "runpod-fanout/ledger-v2.json").read_text()
    )
    return [p for run in state["runs"].values() for p in run["pods"].values()]


K = "apex-veto-v8-screen-v1__h1000__live/"


# ---------------------------------------------------------------- end to end (fake)


def test_serverless_end_to_end_one_world_per_job(world):
    fake = FakeSls(running_polls=1)
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    recs = records_of(r.run_dir)
    assert len(recs) == 6
    one = recs[K + "A-scripted-11.json"]
    assert one["fanout"]["job_id"] == "unit-job" and one["platform"]["platform_id"] == "stub"
    assert one["platform"]["isa_flags"] == ["avx2", "avx512f"]
    assert one["platform"]["backend"] == "serverless" and one["platform"]["worker_id"]
    # numerics reached the episode process: fixed threads + MKL_CBWR=COMPATIBLE
    assert one["record"]["env"] == {"MKL_CBWR": "COMPATIBLE", "OMP": "2", "MKL": "2"}
    body = fake.created[0]
    assert body["workersMin"] == 0 and body["workersMax"] == 2 and body["vcpuCount"] == 4
    assert body["cpuFlavorIds"] == ["cpu5c", "cpu3c"] and body["dataCenterIds"] == ["EU-RO-1"]
    assert body["networkVolumeId"] == "vol1" and body["templateId"] == "tpl1"
    assert body["name"].startswith("rpf-sls-unit-job--sls-r1-")
    # one job = one world: both arms of a world, never two worlds
    sub = [j["input"] for j in fake.jobs.values() if j["input"]["op"] == "episodes"]
    assert len(sub) == 3
    for i in sub:
        units = {platform_rule.unit_of_key(e["key"]) for e in i["episodes"]}
        assert len(units) == 1 and len(i["episodes"]) == 2
        assert i["require_cpu_model"] is None and i["numerics_env"]["MKL_CBWR"] == "COMPATIBLE"
    # torn down: scaled to 0 (asserted by the fake) then deleted; nobody else touched
    assert fake.deleted == ["ep1"] and fake.endpoints == {}
    assert ("ep1", {"workersMin": 0, "workersMax": 0}) in fake.scaled
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["backend"] == "serverless" and receipt["exit_code"] == 0
    assert receipt["final_endpoints_runner_owned"] == [] and receipt["completed"] == 6
    assert receipt["per_world_problems"] == [] and receipt["units_published"] == 3
    assert json.loads((r.run_dir / "endpoints" / "jobs.json").read_text()) == {}
    # ledger: one serverless reservation, released at the upper bound (>= estimate)
    (row,) = ledger_rows(world)
    assert row["kind"] == "serverless" and row["pod_id"] is None and row["settled"] == "runner"
    assert row["cost"] >= row["settle"]["estimate"] > 0 and row["endpoint_id"] == "ep1"
    assert row["cost"] == pytest.approx(row["settle"]["upper"])
    assert row["cost"] >= receipt["ledger_cost_usd_upper_bound"] - 1e-6


def test_worlds_may_differ_in_cpu_model_but_each_world_is_one_worker(world):
    fake = FakeSls(models=[MODEL, OTHER])  # probe, then alternating models per job
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    recs = records_of(r.run_dir)
    sigs = platform_rule.unit_signatures(recs)
    assert all(len(v) == 1 for v in sigs.values())
    assert len({s for v in sigs.values() for s in v}) == 2  # two models across worlds
    assert platform_rule.check_per_world(recs) == []
    for unit, keys in platform_rule.group_units(recs).items():
        assert len({recs[k]["platform"]["worker_id"] for k in keys}) == 1


def test_lost_worker_restarts_the_whole_world_on_one_worker(world):
    unit = "apex-veto-v8-screen-v1|scripted|12"
    fake = FakeSls(fail_unit_once={unit})
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    recs = records_of(r.run_dir)
    keys = platform_rule.group_units(recs)[unit]
    assert len(keys) == 2 and len({recs[k]["platform"]["worker_id"] for k in keys}) == 1
    assert all(r.attempts[k] == 1 for k in keys)


def test_failed_episode_restarts_its_world_and_duplicates_must_match(world, monkeypatch):
    monkeypatch.setenv("STUB_FAIL_ONCE", K + "A-scripted-12.json")
    fake = FakeSls()
    r = make_sls_runner(world, eps6(), fake)
    assert r.run() == 0
    recs = records_of(r.run_dir)
    a, b = recs[K + "A-scripted-12.json"], recs[K + "B-scripted-12.json"]
    assert a["platform"]["worker_id"] == b["platform"]["worker_id"]  # both from the re-run
    assert r.attempts[K + "A-scripted-12.json"] == 1
    assert list((r.run_dir / "failures").glob("*.json"))
    assert "stale_record" not in (r.run_dir / "events.jsonl").read_text()


def test_no_worker_starts_stops_early_and_tears_down(world):
    fake = FakeSls(never_start=True)
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 4
    assert "no serverless worker" in r.stop_reason
    assert fake.endpoints == {} and fake.deleted == ["ep1"]


def test_unknown_record_aborts_and_tears_down(world):
    fake = FakeSls(bad_output_key=K + "Z-x-1.json")
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


def test_delete_failures_are_retried_and_swept(world):
    fake = FakeSls(delete_failures=5)  # close_endpoint gives up after 4, the sweep finishes
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 0
    assert fake.endpoints == {} and fake.deleted == ["ep1"]
    (row,) = ledger_rows(world)
    assert row["deleted"] is not None and row["settled"] == "runner"


def test_create_with_unknown_outcome_is_adopted(world):
    fake = FakeSls(create_mode="5xx-created")
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 0
    assert fake.endpoints == {} and fake.deleted == ["ep1"]


def test_create_with_unknown_outcome_absent_releases_the_reservation(world):
    fake = FakeSls(create_mode="5xx-absent")
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 3
    (row,) = ledger_rows(world)
    assert row["deleted"] is not None and row["cost"] == 0.0


def test_billing_guard_aborts_when_billing_exceeds_the_reservation(world):
    fake = FakeSls(running_polls=10**6)  # jobs never finish
    fake.billed = 50.0
    r = make_sls_runner(world, eps6()[:2], fake)
    assert r.run() == 3
    assert "billing" in r.stop_reason and fake.endpoints == {}


def test_resume_runs_only_missing_worlds_and_merges_per_world(world, monkeypatch):
    monkeypatch.setenv("STUB_FAIL_ONCE", K + "B-scripted-13.json")
    fp1 = dict(world["fp"], max_attempts_per_episode=1)  # the first run gives up on world 13
    r1 = make_sls_runner(world, eps6(), FakeSls(), fp=fp1, name="sls-r1")
    assert r1.run() == 4
    assert len(records_of(r1.run_dir)) == 4
    fake2 = FakeSls(models=[OTHER])
    r2 = make_sls_runner(world, eps6(), fake2, name="sls-r2", resume_from=[r1.run_dir])
    assert r2.resume_info["units_already_complete"] == 2 and r2.order == [
        K + "A-scripted-13.json",
        K + "B-scripted-13.json",
    ]
    assert r2.run() == 0  # an incomplete earlier run does not block (idempotency rc 6)
    sub = [j["input"] for j in fake2.jobs.values() if j["input"]["op"] == "episodes"]
    assert len(sub) == 1
    out = runner.merge_runs([r1.run_dir, r2.run_dir], world["tmp"] / "merged")
    assert out["records"] == 6 and out["units"] == 3
    merged = runner.load_records(world["tmp"] / "merged")
    assert platform_rule.check_per_world(merged) == []
    receipt = json.loads((r2.run_dir / "receipt.json").read_text())
    assert receipt["resume"]["keys_skipped"] == 4
    # the job is now complete: another run is refused (rc 6) and creates nothing
    fake3 = FakeSls()
    r3 = make_sls_runner(world, eps6(), fake3, name="sls-r3", resume_from=[r1.run_dir, r2.run_dir])
    assert r3.order == [] and r3.run() == runner.EXIT_DUPLICATE and fake3.created == []


def test_resume_refuses_another_job(world):
    r1 = make_sls_runner(world, eps6()[:2], FakeSls(), name="sls-r1")
    assert r1.run() == 0
    with pytest.raises(jobspec.JobError, match="different job"):
        make_sls_runner(world, eps6(), FakeSls(), name="sls-r2", resume_from=[r1.run_dir])


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
        ({"numerics_env": {"LD_PRELOAD": "/x.so"}}, "not allowed"),
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
        sp,
        "rpf-sls-x--y-1",
        "tpl",
        {"id": "v", "dataCenterId": "EU-RO-1"},
        ["cpu5c", "cpu3c"],
        3,
        8,
        600,
    )
    assert e["workersMin"] == 0 and e["cpuFlavorIds"] == ["cpu5c", "cpu3c"]
    assert e["executionTimeoutMs"] == 600000
    s = serverless.seeder_pod_body(
        fp, sp, "rpf-seed-x--y-1", {"id": "v", "dataCenterId": "EU-RO-1"}, "ab" * 32, 123, "seed-x"
    )
    assert s["networkVolumeId"] == "v" and s["volumeMountPath"] == "/runpod-volume"
    assert s["imageName"] == t["imageName"] and s["cpuFlavorIds"] == ["cpu3c"]
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
    eps = [dict(ep(a, seed=s), horizon=5000) for s in range(200) for a in "AB"]
    plan = serverless.sizing_plan(
        fp, sp, eps, budget=20.0, max_wall_minutes=90, flavors=["cpu5c", "cpu3c"], headroom=30.0
    )
    assert plan["world_units"] == 200 and plan["usd_per_vcpu_hr_reserved"] == 0.042
    c = plan["choice"]
    assert plan["choice_ok"] and c["wall_minutes"] <= 30 and not c["refused"]
    assert c["worst_case_usd"] <= 20.0
    walls = [r["wall_minutes"] for r in plan["speed_cost_tradeoff"]]
    assert walls == sorted(walls)
    costs = [r["expected_usd"] for r in plan["speed_cost_tradeoff"]]
    assert costs == sorted(costs, reverse=True)
    tiny = serverless.sizing_plan(fp, sp, eps, budget=0.05, max_wall_minutes=90, flavors=["cpu5c"])
    assert not tiny["choice_ok"]
    forced = serverless.sizing_plan(fp, sp, eps, 20.0, 90, ["cpu5c"], workers=3, vcpu=8)
    assert (forced["choice"]["workers"], forced["choice"]["vcpu_per_worker"]) == (3, 8)
    capped = serverless.sizing_plan(fp, sp, eps, 1000.0, 90, ["cpu5c"], headroom=1.0)
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
        assert flavor in ("cpu3c", "cpu5c") and ram == 2 * vcpu and dc == "EU-RO-1"
        if (flavor, vcpu) == ("cpu5c", 2):
            return {"stockStatus": "High", "securePrice": 0.07}
        return {"stockStatus": None, "securePrice": 0.06}  # cpu3c out of stock here

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
        assert body["name"].startswith("rpf-seed-") and body["vcpuCount"] == 2
        assert body["cpuFlavorIds"] == ["cpu5c"]  # the stocked one
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
    assert out["sizing"]["choice_ok"] and out["flavors_in_order"] == ["cpu5c", "cpu3c"]
    body = out["endpoint_body_dry_run"]
    assert body["workersMin"] == 0 and body["templateId"] == "tpl1"
    assert out["storage"]["usd_per_month"] == pytest.approx(0.35)
    assert [e["name"] for e in out["existing_runner_endpoints"]] == []


# ---------------------------------------------------------------- pods backend: per-world


def test_pods_dispatch_whole_worlds_with_numerics_env(tmp_path):
    state = tmp_path / "stubstate"
    state.mkdir()
    fake = tfo.FakeRp(tmp_path, stub_env={"STUB_STATE": str(state)})
    try:
        eps = [ep(a, seed=s, index=i) for a in "AB" for i, s in enumerate((21, 22, 23))]
        r = tfo.make_runner(tmp_path, eps, fake)
        (tmp_path / "stub.py").write_text(ENV_STUB)
        (Path(fake.tmp) / "stub.py").write_text(ENV_STUB)
        assert r.run() == 0
    finally:
        fake.close()
    body = fake.created[0]
    assert json.loads(body["env"]["FANOUT_NUMERICS_JSON"]) == {
        "ATEN_CPU_CAPABILITY": "avx2",
        "MKL_CBWR": "COMPATIBLE",
        "MKL_NUM_THREADS": "2",
        "OMP_NUM_THREADS": "2",
    }
    recs = records_of(r.run_dir)
    assert len(recs) == 6
    for e in recs.values():
        assert e["record"]["env"] == {"MKL_CBWR": "COMPATIBLE", "OMP": "2", "MKL": "2"}
        assert e["platform"]["backend"] == "pods" and "isa_flags" in e["platform"]
        assert e["platform"]["numerics_env"]["MKL_CBWR"] == "COMPATIBLE"
    for unit, keys in platform_rule.group_units(recs).items():
        assert len({recs[k]["platform"]["worker_id"] for k in keys}) == 1
    events = [json.loads(x) for x in (r.run_dir / "events.jsonl").read_text().splitlines()]
    assigned = [u for e in events if e["event"] == "assigned" for u in e["units"]]
    assert sorted(assigned) == sorted(platform_rule.group_units(recs))
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["per_world_problems"] == [] and receipt["units_published"] == 3


def test_pods_lost_worker_restarts_whole_world_and_late_records_are_only_checked(tmp_path):
    state = tmp_path / "stubstate"
    state.mkdir()
    fake = tfo.FakeRp(tmp_path, stub_env={"STUB_STATE": str(state)})
    eps = [ep("A", seed=31), ep("B", seed=31)]
    r = tfo.make_runner(tmp_path, eps, fake)
    fake.close()
    r.run_dir.mkdir(parents=True)
    ka, kb = (jobspec.episode_key(e) for e in r.job["episodes"])
    unit = platform_rule.unit_of_key(ka)

    def rec(key, mass=1.0):
        e = r.episodes[key]
        entry = {
            k: e[k] for k in ("arm", "mix", "world_seed", "world_index", "roster_member_sha256s")
        }
        entry.update(
            {
                "record": {"mass_integral": mass},
                "platform": {"platform_id": "stub"},
                "fanout": {
                    "job_id": r.job["job_id"],
                    "episode_key": key,
                    "repo_commit": r.job["repo_commit"],
                    "spec_sha256": runner.spec_sha256(e),
                },
            }
        )
        return json.dumps(entry).encode()

    pod_x = runner.Pod(id="pX", name="n", vcpu=8, dc="X", rate=0.1, token="", created=0)
    r.pods["pX"] = pod_x
    r.workers_info["pX"] = {"isa_flags": ["avx2"], "worker_id": "pX", "backend": "pods"}
    r.workers_info["pY"] = {"isa_flags": ["avx2", "avx512f"], "worker_id": "pY", "backend": "pods"}
    assert r.take_unit(unit, "pX") == [ka, kb]
    pod_x.inflight.update([ka, kb])
    r.accept_record(ka, rec(ka), "pX")  # A done on pX, then pX is lost
    assert not (r.run_dir / "records").exists()
    r.requeue(pod_x, [ka, kb], "pod lost")
    assert r.pending == [ka, kb] and unit not in r.staged and r.attempts[ka] == 1
    r.take_unit(unit, "pY")
    r.accept_record(ka, rec(ka), "pX")  # a late copy from the lost worker: checked, ignored
    r.accept_record(ka, rec(ka), "pY")
    assert not (r.run_dir / "records").exists()  # the unit waits for B from pY
    r.accept_record(kb, rec(kb, 2.0), "pY")
    recs = records_of(r.run_dir)
    assert {e["platform"]["worker_id"] for e in recs.values()} == {"pY"}
    assert platform_rule.check_per_world(recs) == []
    with pytest.raises(runner.Abort, match="DIVERGENT"):  # any differing duplicate aborts
        r.accept_record(ka, rec(ka, 9.0), "pX")


def test_numerics_env_and_isa_cap_override(tmp_path):
    fp = fpol(tmp_path)
    assert runner.numerics_env(fp) == {
        "OMP_NUM_THREADS": "2",
        "MKL_NUM_THREADS": "2",
        "MKL_CBWR": "COMPATIBLE",
        "ATEN_CPU_CAPABILITY": "avx2",
    }
    capped = runner.numerics_env(dict(fp, isa_cap="AVX2"))
    assert capped["MKL_CBWR"] == "AVX2" and capped["ONEDNN_MAX_CPU_ISA"] == "AVX2"


def test_platform_rule_units_and_signatures():
    assert platform_rule.unit_of_key("w__h1000__live/K12S-mixed-7.json") == "w|mixed|7"
    recs = {
        "w__h1000__live/A-mixed-7.json": {"platform": {"platform_id": "p", "isa_flags": ["avx2"]}},
        "w__h1000__live/B-mixed-7.json": {
            "platform": {"platform_id": "p", "isa_flags": ["avx2", "avx512f"]}
        },
        "w__h1000__live/A-mixed-8.json": {"platform": {"platform_id": "q"}},
    }
    probs = platform_rule.check_per_world(recs)
    assert len(probs) == 1 and "w|mixed|7" in probs[0]
    flags = platform_rule.isa_flags_from_cpuinfo("flags\t: fpu sse4_2 avx avx2 avx512f amx_tile\n")
    assert flags == ["sse4_2", "avx", "avx2", "avx512f", "amx_tile"]


# ---------------------------------------------------------------- ledger settle


def test_old_zero_release_is_overridden_and_unsettled_rows_count_worst(tmp_path):
    fp = fpol(tmp_path)
    t = [1000.0]
    new = ledger.SharedLedger(fp, clock=lambda: t[0], alive=lambda *a: True)
    new.register_run("r", "j", "d", 5.0, os.getpid())
    new.reserve_pod("r", "sls:e", 3.6, 1000 + 3600, 40.0, kind="serverless")
    old = old_ledger_module().SharedLedger(fp, clock=lambda: t[0], alive=lambda *a: False)
    t[0] += 3600 + 400  # past horizon + slack: the old reconcile records it at $0
    assert old.reconcile(ListPods())["released"] == ["sls:e"]
    row = json.loads(new.path.read_text())["runs"]["r"]["pods"]["sls:e"]
    assert row["cost"] == 0.0 and "settled" not in row
    # new readers charge the worst case for such a row ...
    spent0 = float(fp.get("prior_spend_usd", 0.0))
    assert new.peek()["spent_usd"] == pytest.approx(spent0 + 3.6, rel=1e-6)
    # ... and the runner's later release still lands (max with the $0)
    new.release_pod("r", "sls:e", 1.25)
    row = json.loads(new.path.read_text())["runs"]["r"]["pods"]["sls:e"]
    assert row["cost"] == 1.25 and row["settled"] == "runner"


def test_settle_serverless_lowers_to_final_billing(tmp_path):
    fp = fpol(tmp_path)
    t = [1000.0]
    lg = ledger.SharedLedger(fp, clock=lambda: t[0], alive=lambda *a: True)
    lg.register_run("r", "j", "d", 5.0, os.getpid())
    lg.reserve_pod("r", "sls:e", 3.6, 1000 + 3600, 40.0, kind="serverless")
    lg.release_pod("r", "sls:e", 2.0)
    lg.annotate(
        "r",
        "sls:e",
        settle={"endpoint_id": "ep1", "start": 1000, "end": 1500, "estimate": 0.3, "upper": 2.0},
    )

    class Billing:
        amount = 0.0

        def endpoint_billing(self, eid, start, end):
            assert eid == "ep1"
            return self.amount

    b = Billing()
    assert lg.settle_serverless(b) == []  # too early: hour buckets not final
    t[0] += 3 * 3600
    assert lg.settle_serverless(b) == []  # zero billing keeps the upper bound
    b.amount = 0.42
    assert lg.settle_serverless(b) == ["sls:e"]
    row = json.loads(lg.path.read_text())["runs"]["r"]["pods"]["sls:e"]
    assert row["cost"] == 0.42 and row["settled"] == "billing"
    assert lg.settle_serverless(b) == []


# ---------------------------------------------------------------- cleanup / watchdog / seed


def test_endpoint_cleanup_all_runner_endpoints_matches(monkeypatch, tmp_path, capsys):
    assert serverless.owned_endpoint("rpf-sls-", "rpf-sls-myjob--sls-x-1")
    assert serverless.owned_endpoint("rpf-sls-myjob--", "rpf-sls-myjob--sls-x-1 -fb")
    assert not serverless.owned_endpoint("rpf-sls-myjob--sls-x-", "rpf-sls-myjob--sls-x-10a")
    assert not serverless.owned_endpoint("rpf-sls-", "someone-else")
    fake = FakeSls()
    fake.endpoints["e9"] = {"id": "e9", "name": "rpf-sls-myjob--sls-x-1", "workersMax": 2}
    monkeypatch.setattr(serverless, "RpClient", lambda: fake)
    fp, sp = fpol(tmp_path), spol()
    a = type(
        "A", (), {"job_id": None, "all_runner_endpoints": True, "confirm": True, "force": False}
    )()
    assert serverless.cmd_endpoint_cleanup(a, fp, sp) == 0
    assert fake.deleted == ["e9"] and "otherep" not in fake.deleted


def test_watchdog_runner_dead_cancels_persisted_jobs(tmp_path):
    run_dir = tmp_path / "run"
    (run_dir / "endpoints").mkdir(parents=True)
    mine = "rpf-sls-unit-job--sls-r1-"
    (run_dir / "endpoints" / f"{mine}1.response.json").write_text(
        json.dumps({"id": "e1", "name": f"{mine}1", "rpf_until_epoch": time.time() + 9999})
    )
    (run_dir / "endpoints" / "jobs.json").write_text(json.dumps({"e1": ["j1", "j2"]}))
    state = tmp_path / "fake.json"
    state.write_text(
        json.dumps({"pods": [], "endpoints": [{"id": "e1", "name": f"{mine}1", "workersMax": 4}]})
    )

    class Rp(watchdog.FileFakeRp):
        cancelled = []

        def sls(self, endpoint_id, op, job_id=None, body=None, confirm=False):
            assert confirm
            self.cancelled.append((op, job_id))
            return {}

    rp = Rp(str(state))
    t = [time.time()]

    def sleep(s):
        t[0] += s

    reason = watchdog.watch(
        "unit-job",
        t[0] + 99999,
        run_dir,
        999999,
        rp=rp,
        sleep=sleep,
        clock=lambda: t[0],
        alive=lambda pid: False,
        prefix="rpf-unit-job--sls-r1-",
        endpoint_prefix=mine,
        resweep_seconds=0,
    )
    assert reason == "runner dead"
    assert ("cancel", "j1") in rp.cancelled and ("cancel", "j2") in rp.cancelled
    assert json.loads(state.read_text())["deleted_endpoints"] == ["e1"]


def test_seed_job_id_depends_on_volume_and_checkpoints_and_one_seeder_at_a_time(tmp_path):
    import fcntl

    fp, sp = fpol(tmp_path), spol()
    c = "a" * 40

    def jid(vol, ck):
        return serverless.SeedRunner(
            fp,
            sp,
            {},
            tmp_path / "x",
            commit=c,
            volume={"id": vol, "dataCenterId": "EU-RO-1"},
            ckpts=ck,
            rp=FakeSls(),
        ).job["job_id"]

    assert jid("v1", ["1" * 64]) == jid("v1", ["1" * 64])
    assert jid("v1", ["1" * 64]) != jid("v2", ["1" * 64]) != jid("v1", ["1" * 64, "2" * 64])
    assert len(jid("v1", [])) <= 24
    reg = serverless.Registry(fp)
    reg.root.mkdir(parents=True)
    with (reg.root / "seed.lock").open("a") as h:
        fcntl.flock(h.fileno(), fcntl.LOCK_EX)
        r = serverless.SeedRunner(
            fp,
            sp,
            {},
            tmp_path / "x",
            commit=c,
            volume={"id": "v1", "dataCenterId": "EU-RO-1"},
            ckpts=[],
            rp=FakeSls(),
            confirm=True,
        )
        assert r.run() == 2


def test_isa_flag_lists_are_identical_everywhere():
    import ast

    def const(path, name):
        tree = ast.parse(Path(path).read_text())
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                getattr(t, "id", None) == name for t in node.targets
            ):
                return ast.literal_eval(node.value)
        raise AssertionError(f"{name} not in {path}")

    want = platform_rule.ISA_FLAGS
    assert const(runner.AGENT_SOURCE, "ISA_FLAGS") == want
    assert const(serverless.HANDLER_SOURCE, "ISA_FLAGS") == want


def test_pod_episode_error_retries_same_pod_once_then_restarts_unit_anywhere(tmp_path):
    fake = tfo.FakeRp(tmp_path, stub_env={})
    r = tfo.make_runner(tmp_path, [ep("A", seed=41), ep("B", seed=41)], fake)
    fake.close()
    ka, kb = list(r.episodes)
    unit = platform_rule.unit_of_key(ka)
    pod = runner.Pod(id="pX", name="n", vcpu=8, dc="X", rate=0.1, token="", created=0)
    r.pods["pX"] = pod

    class Agent:
        accept = True
        posts = 0

        def post_json(self, path, body):
            Agent.posts += 1
            return {"accepted": [e["key"] for e in body["episodes"]] if Agent.accept else []}

    r.agents["pX"] = Agent()
    r.take_unit(unit, "pX")
    pod.inflight.update([ka, kb])
    r.retry_on_pod(pod, ka, {"rc": 1})  # first failure: same pod
    assert r.attempts[ka] == 1 and Agent.posts == 1 and r.pending == []
    r.retry_on_pod(pod, ka, {"rc": 1})  # second failure: the whole world, any pod
    assert r.pending == [ka, kb] and r.unit_attempts[unit] == 1 and not r.failed
    assert r.attempts[ka] == 2 and r.attempts[kb] == 1  # no double count
    r.take_unit(unit, "pX")
    pod.inflight.update([ka, kb])
    r.retry_on_pod(pod, ka, {"rc": 1})  # attempts exhausted (max 2): the world fails
    assert set(r.failed) == {ka, kb} and r.pending == []
    r.run_dir.mkdir(parents=True)
    e = r.episodes[kb]
    late = {k: e[k] for k in ("arm", "mix", "world_seed", "world_index", "roster_member_sha256s")}
    late.update(
        {
            "record": {},
            "platform": {"platform_id": "stub"},
            "fanout": {
                "job_id": r.job["job_id"],
                "episode_key": kb,
                "repo_commit": r.job["repo_commit"],
                "spec_sha256": runner.spec_sha256(e),
            },
        }
    )
    r.accept_record(kb, json.dumps(late).encode(), "pX")  # a failed world is never published
    assert not (r.run_dir / "records").exists()
