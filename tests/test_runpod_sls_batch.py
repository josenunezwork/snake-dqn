"""Batched serverless jobs (K whole world units per job) and the account worker-quota guard.

No network, no RunPod: the endpoint is ``tests.test_runpod_serverless.FakeSls`` (it runs
``sls_handler.handle`` in-process against a fake volume); episodes are the stub process.
"""

from __future__ import annotations

import importlib.util
import json
import math
import subprocess
import sys
import time

import pytest

from research.runpod_fanout import jobspec, platform_rule, runner, serverless
from research.runpod_fanout.rp_client import RunPodError
from tests import test_runpod_serverless as tsl
from tests.test_runpod_fanout import GREEDY, RANDOM, Clock, ep, job

world = tsl.world  # fake volume + registry + handler wiring (fixture)
no_network = tsl.no_network  # autouse: real RunPod calls / non-loopback sockets are fatal
FakeSls, K, MODEL, OTHER = tsl.FakeSls, tsl.K, tsl.MODEL, tsl.OTHER
PRE_BATCHING = "4fd6e9e"  # main before batching: the serverless.py live runs use today


def eps12():
    """Six worlds x arms A, B: six units of two episodes each."""
    return [ep(a, seed=s, index=i) for a in "AB" for i, s in enumerate(range(11, 17))]


def unit(seed):
    return f"apex-veto-v8-screen-v1|scripted|{seed}"


def make(w, episodes, fake, units_per_job="auto", workers=2, vcpu=8, name="b1", **kw):
    """ServerlessRunner with an explicit ``units_per_job`` (tsl.make_sls_runner has none)."""
    clock = kw.pop("clock", None) or Clock()
    fp = kw.pop("fp", None) or w["fp"]
    sp = kw.pop("sp", None) or w["sp"]
    j = job(episodes, repo_commit=w["commit"], checkpoints=[w["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [w["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    sizing = {"workers": workers, "vcpu_per_worker": vcpu}
    if units_per_job is not None:
        sizing["units_per_job"] = units_per_job
    return serverless.ServerlessRunner(
        j,
        fp,
        w["allow"],
        w["tmp"] / "runs" / name,
        sls_policy=sp,
        sizing=sizing,
        flavors=["cpu5c", "cpu3c"],
        registry=w["reg"],
        rp=fake,
        budget=kw.pop("budget", 3.0),
        confirm=True,
        repo=w["repo"],
        clock=clock,
        sleep=clock.sleep,
        spawn_watchdog=lambda r: None,
        preflight_fn=lambda: [],
        probe_workers=2,
        **kw,
    )


class RecordingSls(FakeSls):
    """FakeSls that keeps every /run body (input AND policy), in submission order."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self.bodies = []

    def sls(self, eid, op, job_id=None, body=None, confirm=False):
        if op == "run":
            self.bodies.append(json.loads(json.dumps(body)))
        return super().sls(eid, op, job_id, body, confirm)


def episode_jobs(fake):
    return [b for b in fake.bodies if b["input"]["op"] == "episodes"]


def units_of(body):
    seen = []
    for e in body["input"]["episodes"]:
        u = platform_rule.unit_of_key(e["key"])
        if u not in seen:
            seen.append(u)
    return seen


def events(r):
    return [json.loads(x) for x in (r.run_dir / "events.jsonl").read_text().splitlines()]


# ---------------------------------------------------------------- batches end to end


def test_batch_runs_whole_units_concurrently_on_one_worker(world):
    fake = RecordingSls(models=[MODEL, OTHER], running_polls=1)
    r = make(world, eps12(), fake, workers=2, vcpu=8)  # 4 slots per worker
    assert r.run() == 0
    jobs = episode_jobs(fake)
    # auto: each job filled to one wave (4 slots = two 2-episode units): the fewest jobs
    assert [len(units_of(b)) for b in jobs] == [2, 2, 2]
    for b in jobs:
        assert b["input"]["slots"] == 4 and len(b["input"]["episodes"]) <= 4
        # every unit is whole inside its job: both arms
        for u in units_of(b):
            arms = [e["spec"]["arm"] for e in b["input"]["episodes"]]
            assert sorted(
                a
                for a, e in zip(arms, b["input"]["episodes"])
                if platform_rule.unit_of_key(e["key"]) == u
            ) == ["A", "B"]
        # executionTimeout = 4 x critical path (ceil(<=4/4) = 1 wave x 30.2 s) + 300
        longest = 12.0 + 0.0182 * 1000
        assert b["policy"]["executionTimeout"] == int((4 * longest + 300) * 1000)
        assert b["input"]["timeout_seconds"] == pytest.approx(4 * longest + 300 - 60)
    recs = tsl.records_of(r.run_dir)
    assert len(recs) == 12 and platform_rule.check_per_world(recs) == []
    by_unit = platform_rule.group_units(recs)
    workers_of = {u: {recs[k]["platform"]["worker_id"] for k in ks} for u, ks in by_unit.items()}
    assert all(len(v) == 1 for v in workers_of.values())  # per-world affinity
    # the two units of one job share that job's worker; jobs alternate CPU models
    for b in jobs:
        a, c = units_of(b)
        assert workers_of[a] == workers_of[c]
    assert len({s for v in platform_rule.unit_signatures(recs).values() for s in v}) == 2
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["batching"]["jobs_by_units_per_job"] == {"2": 3}
    assert receipt["batching"]["units_per_job"] == "auto" and receipt["units_published"] == 6
    # the endpoint's executionTimeout covers every job's; ledger = workers x max(vCPU x
    # price, the per-worker floor) x margin: 8 x 0.042 = 0.336 < 0.61
    body = fake.created[0]
    assert body["executionTimeoutMs"] >= max(b["policy"]["executionTimeout"] for b in jobs)
    (row,) = tsl.ledger_rows(world)
    assert row["rate"] == pytest.approx(2 * 0.61 * 1.25)
    guard = receipt["balance_guard"]
    assert guard["balance_drop"] == 0.0 and guard["exceeded"] is False
    assert guard["allowed_at_reserved_rates"] >= world["sp"]["balance_guard_slack_usd"]


def test_fixed_k_allows_several_waves_and_per_episode_caps_fit_the_job(world):
    fake = RecordingSls()
    r = make(world, eps12(), fake, units_per_job=3, workers=2, vcpu=4)  # 2 slots
    assert r.run() == 0
    jobs = episode_jobs(fake)
    assert [len(units_of(b)) for b in jobs] == [3, 2, 1]  # K=3 caps, the tail spreads
    longest = 12.0 + 0.0182 * 1000
    for b, want in zip(jobs, (3, 2, 1)):
        n = len(b["input"]["episodes"])
        waves = math.ceil(n / 2)
        assert waves == want
        timeout = b["policy"]["executionTimeout"] / 1000
        assert timeout == pytest.approx(4 * waves * longest + 300, abs=1e-3)
        # wave bound: waves of capped episodes + 60 s setup end inside executionTimeout
        assert waves * b["input"]["timeout_seconds"] + 60 <= timeout + 1e-6
    assert len(tsl.records_of(r.run_dir)) == 12


def test_partial_batch_publishes_complete_units_and_requeues_the_other_whole(world, monkeypatch):
    monkeypatch.setenv("STUB_FAIL_ONCE", K + "A-scripted-12.json")
    fake = RecordingSls()
    r = make(world, eps12(), fake, workers=2, vcpu=8)
    assert r.run() == 0
    jobs = episode_jobs(fake)
    first = next(b for b in jobs if unit(12) in units_of(b))
    partner = [u for u in units_of(first) if u != unit(12)][0]
    ev = events(r)
    published = {e["unit"]: e["source"] for e in ev if e["event"] == "unit_published"}
    collected = [e for e in ev if e["event"] == "collected" and e.get("units_requeued")]
    assert [c["units_requeued"] for c in collected] == [[unit(12)]]
    # the partner unit published from the batch that failed unit 12; unit 12 from a later job
    assert published[partner] == f"sls:{collected[0]['job']}"
    assert published[unit(12)] != published[partner]
    later = [b for b in jobs[jobs.index(first) + 1 :] if unit(12) in units_of(b)]
    assert len(later) == 1  # re-dispatched whole (both arms) in one job, ALONE (as for K=1)
    assert units_of(later[0]) == [unit(12)] and unit(12) in r.solo
    assert sorted(
        e["spec"]["arm"]
        for e in later[0]["input"]["episodes"]
        if platform_rule.unit_of_key(e["key"]) == unit(12)
    ) == ["A", "B"]
    recs = tsl.records_of(r.run_dir)
    assert len(recs) == 12 and platform_rule.check_per_world(recs) == []
    assert r.attempts[K + "A-scripted-12.json"] == 1 and r.attempts[K + "A-scripted-11.json"] == 0
    assert "stale_record" not in (r.run_dir / "events.jsonl").read_text()


def test_lost_batch_requeues_every_unit_whole_and_they_then_run_alone(world):
    fake = RecordingSls(fail_unit_once={unit(11)})  # the job whose first unit is 11 dies
    r = make(world, eps12(), fake, workers=2, vcpu=8)
    assert r.run() == 0
    jobs = episode_jobs(fake)
    lost = units_of(jobs[0])
    assert lost[0] == unit(11) and len(lost) == 2
    for u in lost:
        again = [b for b in jobs[1:] if u in units_of(b)]
        assert len(again) == 1 and units_of(again[0]) == [u]  # alone after the loss
        for k in r.units[u]:
            assert r.attempts[k] == 0  # a lost batch is not pinned on any one unit
    ev = events(r)
    assert [e["units"] for e in ev if e["event"] == "batch_lost"] == [lost]
    assert sorted(e["unit"] for e in ev if e["event"] == "unit_redispatched") == sorted(lost)
    recs = tsl.records_of(r.run_dir)
    assert len(recs) == 12 and platform_rule.check_per_world(recs) == []
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["batching"]["solo_units"] == 2


def test_a_lost_batch_never_costs_a_batch_mate_its_last_attempt(world, monkeypatch):
    """Review finding: the batch [11, 12] is lost, then unit 12 has one transient episode
    error running alone. As for K=1 (one attempt used), it reruns and completes; before the
    fix the lost batch had used its attempt and the run ended exit 4 with 12 failed."""
    monkeypatch.setenv("STUB_FAIL_ONCE", K + "A-scripted-12.json")
    fake = RecordingSls(fail_unit_once={unit(11)})
    r = make(world, eps12(), fake, workers=2, vcpu=8)
    assert r.run() == 0 and r.failed == {}
    jobs = episode_jobs(fake)
    assert set(units_of(jobs[0])) == {unit(11), unit(12)}
    alone = [b for b in jobs[1:] if unit(12) in units_of(b)]
    assert len(alone) == 2 and all(units_of(b) == [unit(12)] for b in alone)
    assert r.attempts[K + "A-scripted-12.json"] == 1  # only the solo episode error counted
    assert len(tsl.records_of(r.run_dir)) == 12


class StuckSls(RecordingSls):
    """The first job carrying ``stuck`` stays IN_PROGRESS forever (a hung worker)."""

    def __init__(self, stuck, **kw):
        super().__init__(**kw)
        self.stuck, self.stuck_job = stuck, None

    def sls(self, eid, op, job_id=None, body=None, confirm=False):
        if op == "status":
            j = self.jobs.get(job_id)
            keys = [e["key"] for e in (j or {}).get("input", {}).get("episodes", [])]
            if self.stuck_job is None and any(
                platform_rule.unit_of_key(k) == self.stuck for k in keys
            ):
                self.stuck_job = job_id
            if job_id == self.stuck_job:
                return {"id": job_id, "status": "IN_PROGRESS"}
        return super().sls(eid, op, job_id, body, confirm)


def test_overrunning_batch_is_cancelled_and_all_its_units_rerun(world):
    sp = dict(world["sp"], poll_seconds=60)
    fake = StuckSls(unit(13))
    r = make(world, eps12(), fake, workers=2, vcpu=8, sp=sp)
    assert r.run() == 0
    stuck = fake.jobs[fake.stuck_job]["input"]
    stuck_units = {platform_rule.unit_of_key(e["key"]) for e in stuck["episodes"]}
    assert (fake.endpoints == {}) and any(c[1] == fake.stuck_job for c in fake.cancelled)
    lost = [e for e in events(r) if e["event"] == "batch_lost"]
    assert len(lost) == 1 and set(lost[0]["units"]) == stuck_units
    assert "overran" in lost[0]["why"]
    recs = tsl.records_of(r.run_dir)
    assert len(recs) == 12 and platform_rule.check_per_world(recs) == []


def test_auto_fills_one_wave_and_output_budget_caps_the_batch(world):
    s = world["sp"]
    # 16 slots hold all 6 units of 2 in one wave: ONE job (splitting it over the 3 workers
    # would not finish sooner, a one-wave job lasts its slowest episode, but bills 3 workers)
    fake = RecordingSls()
    r = make(world, eps12(), fake, workers=3, vcpu=32, name="b-fill", budget=10.0)
    assert r.run() == 0
    assert fake.created[0]["workersMax"] == 3
    assert [len(units_of(b)) for b in episode_jobs(fake)] == [6]
    # big records shrink the batch: 1.5 MB records -> 6 MB / (1.5 MB x 1.3) = 3 episodes
    assert serverless.job_episode_cap(s, 16, None) == 16
    assert serverless.job_episode_cap(s, 16, None, record_bytes=1.5 * 2**20) == 3
    assert serverless.job_episode_cap(s, 16, 8) == int(
        (6 * 2**20 - serverless.OUTPUT_FIXED_BYTES) // (262144 * 1.3)
    )
    r2 = make(world, eps12(), FakeSls(), workers=1, vcpu=32, name="b-bytes")
    r2.record_bytes_seen = int(1.5 * 2**20)
    batch = r2.next_batch(serverless.Endpoint("x", "k", [], 1, 32, 1, 1, 0, 0))
    assert len(batch) == 1  # 2-episode units: only one fits a 3-episode budget


def test_pack_batch_rules():
    units = [("a", 3), ("b", 3), ("c", 1), ("d", 2), ("e", 1)]
    assert serverless.pack_batch(units, 9, 4) == ["a", "c"]  # first fit, never split
    assert serverless.pack_batch(units, 2, 16) == ["a", "b"]  # K cap
    assert serverless.pack_batch([("big", 20), ("c", 1)], 5, 16) == ["big"]  # alone, waves
    assert serverless.pack_batch(units, 9, 16, solo={"a"}) == ["a"]  # solo runs alone
    assert serverless.pack_batch(units, 9, 16, solo={"b"}) == ["a", "c", "d", "e"]
    assert serverless.units_cap(104, 10, None) == 104  # auto: the one-wave episode cap binds
    assert serverless.units_cap(104, 10, 1) == 1 and serverless.units_cap(104, 10, 5) == 5
    assert serverless.units_cap(6, 2, 8) == 3  # a fixed K spreads its waves over the workers
    with pytest.raises(jobspec.JobError):
        serverless.parse_units_per_job("0")
    assert serverless.parse_units_per_job("auto") is None
    assert serverless.parse_units_per_job("3") == 3


# ---------------------------------------------------------------- K=1 == pre-batching


def module_at(commit, rel, name):
    src = subprocess.run(
        ["git", "-C", str(runner.REPO), "show", f"{commit}:{rel}"], capture_output=True, text=True
    )
    if src.returncode != 0:
        pytest.skip(f"{commit} not in this clone")
    spec = importlib.util.spec_from_loader(name, loader=None)
    mod = importlib.util.module_from_spec(spec)
    mod.__file__ = str(runner.HERE / "serverless.py")  # same policy / handler paths
    sys.modules[name] = mod  # dataclasses look their module up
    exec(compile(src.stdout, f"{rel}@{commit}", "exec"), mod.__dict__)
    return mod


@pytest.mark.parametrize("units_per_job", [1, None])  # explicit K=1, and no key (default)
def test_k1_submits_exactly_what_the_pre_batching_runner_did(world, units_per_job):
    old = module_at(PRE_BATCHING, "research/runpod_fanout/serverless.py", "rpf_sls_old")
    t0 = float(int(time.time()) + 3600)
    out = {}
    for label, mod in (("old", old), ("new", serverless)):
        fake = RecordingSls(models=[MODEL, OTHER])
        clock = Clock()
        clock.t = t0
        fp = dict(world["fp"], artifacts_root=str(world["tmp"] / f"art-{label}"))
        j = job(eps12(), repo_commit=world["commit"], checkpoints=[world["hero"]])
        for e in j["episodes"]:
            e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
        sizing = {"workers": 2, "vcpu_per_worker": 4}
        if label == "new" and units_per_job is not None:
            sizing["units_per_job"] = units_per_job
        r = mod.ServerlessRunner(
            j,
            fp,
            world["allow"],
            world["tmp"] / "runs" / f"k1-{label}",
            sls_policy=world["sp"],
            sizing=sizing,
            flavors=["cpu5c", "cpu3c"],
            registry=world["reg"],
            rp=fake,
            budget=3.0,
            confirm=True,
            repo=world["repo"],
            clock=clock,
            sleep=clock.sleep,
            spawn_watchdog=lambda r: None,
            preflight_fn=lambda: [],
            probe_workers=1,  # FakeSls numbers workers in poll order: keep it deterministic
        )
        assert r.run() == 0
        recs = tsl.records_of(r.run_dir)
        out[label] = {
            "bodies": fake.bodies,
            "endpoint": [{k: v for k, v in b.items() if k != "name"} for b in fake.created],
            "records": {k: jobspec.deterministic_bytes(e) for k, e in recs.items()},
            "workers": {k: e["platform"]["worker_id"] for k, e in recs.items()},
            # the balance guard (added after batching) only logs balance_check
            "events": [e["event"] for e in events(r) if e["event"] != "balance_check"],
        }
    assert out["new"]["bodies"] == out["old"]["bodies"]  # input AND policy, job by job
    assert out["new"]["endpoint"] == out["old"]["endpoint"]
    assert out["new"]["records"] == out["old"]["records"] and len(out["new"]["records"]) == 12
    assert out["new"]["workers"] == out["old"]["workers"]
    assert out["new"]["events"] == out["old"]["events"]
    assert all(len(units_of(b)) == 1 for b in episode_jobs_from(out["new"]["bodies"]))


def episode_jobs_from(bodies):
    return [b for b in bodies if b["input"]["op"] == "episodes"]


def test_k1_sizing_matches_the_pre_batching_planner(tmp_path):
    old = module_at(PRE_BATCHING, "research/runpod_fanout/serverless.py", "rpf_sls_old_plan")
    fp = tsl.fpol(tmp_path)
    eps = [dict(ep(a, seed=s), horizon=5000) for s in range(60) for a in "ABC"]
    # quota guard, worker price floor, serverless time factor and wall fill off: the same
    # option space and estimates as the pre-batching planner
    sp = dict(
        tsl.spol(),
        account_worker_quota=0,
        usd_per_worker_hr_min=0,
        episode_time_factor=1.0,
        max_wall_fill=1.0,
    )
    a = old.sizing_plan(fp, sp, eps, 20.0, 90, ["cpu5c", "cpu3c"], 30.0)
    b = serverless.sizing_plan(fp, sp, eps, 20.0, 90, ["cpu5c", "cpu3c"], 30.0, units_per_job=1)
    keys = ("workers", "vcpu_per_worker", "slots_per_worker", "wall_minutes", "expected_usd")
    keys += ("worst_case_usd", "usd_per_hr", "refused", "parallel_units")
    pick = lambda c: {k: c[k] for k in keys}  # noqa: E731
    assert pick(a["choice"]) == pick(b["choice"])
    assert [pick(r) for r in a["speed_cost_tradeoff"]] == [
        pick(r) for r in b["speed_cost_tradeoff"]
    ]


def test_batched_sizing_is_quota_capped_and_shows_k(tmp_path):
    fp, sp = tsl.fpol(tmp_path), tsl.spol()
    # 104 units of 3 H5000 episodes (the kr-robust-d shape)
    eps = [dict(ep(a, seed=s), horizon=5000) for s in range(104) for a in "ABC"]
    plan = serverless.sizing_plan(fp, sp, eps, 20.0, 120, ["cpu5c", "cpu3c"], 30.0)
    assert plan["account_worker_quota"] == 10 and plan["units_per_job_mode"] == "auto"
    c = plan["choice"]
    assert plan["choice_ok"] and c["workers"] <= 10
    assert c["units_per_job_max"] * 3 <= c["slots_per_worker"] or c["units_per_job_max"] == 1
    forced = serverless.sizing_plan(fp, sp, eps, 20.0, 120, ["cpu5c"], 30.0, workers=10, vcpu=32)
    f = forced["choice"]
    assert (f["workers"], f["vcpu_per_worker"], f["slots_per_worker"]) == (10, 32, 16)
    assert f["units_per_job_max"] == 5 and f["parallel_episodes"] == 150
    assert "hourly cap" in f["refused"]  # 10 x 32 vCPU x 0.042 x 1.25 = $16.8/hr > $15
    ok = serverless.sizing_plan(fp, sp, eps, 20.0, 120, ["cpu5c"], 30.0, workers=10, vcpu=16)
    # 16 vCPU = 8 slots: two 3-episode units per wave
    assert ok["choice_ok"] and ok["choice"]["parallel_episodes"] == 60
    one = serverless.sizing_plan(
        fp, sp, eps, 20.0, 120, ["cpu5c"], 30.0, workers=10, vcpu=16, units_per_job=1
    )
    assert one["choice"]["units_per_job_max"] == 1
    assert one["choice"]["wall_minutes"] > 1.5 * ok["choice"]["wall_minutes"]
    eleven = serverless.sizing_plan(fp, sp, eps, 20.0, 120, ["cpu5c"], 30.0, workers=11, vcpu=4)
    assert "account worker quota" in eleven["choice"]["refused"]


# ---------------------------------------------------------------- worker quota guard


class QuotaSls(RecordingSls):
    def __init__(
        self, held, free_after_lists=None, list_fails=False, quota_refusals=0, status=400, **kw
    ):
        super().__init__(**kw)
        self.status = status
        self.other = {"id": "otherep", "name": "someone-else", "workersMax": held}
        self.lists = 0
        self.free_after_lists = free_after_lists
        self.list_fails = list_fails
        self.quota_refusals = quota_refusals

    def list_endpoints(self):
        self.lists += 1
        if self.list_fails:
            raise RunPodError("down", {"http_status": 503})
        if self.free_after_lists is not None and self.lists > self.free_after_lists:
            self.other = dict(self.other, workersMax=0)
        return super().list_endpoints()

    def create_endpoint(self, body_path, confirm):
        if self.quota_refusals > 0:
            self.quota_refusals -= 1
            body = json.loads(open(body_path).read())
            self.created.append(body)
            raise RunPodError(  # the live 2026-10-04 refusal came back as HTTP 500
                f"rest POST /endpoints -> {self.status}",
                {
                    "http_status": self.status,
                    "error": {
                        "error": "create endpoint: create endpoint: graphql: Max workers "
                        "across all endpoints must not exceed your workers quota (10). Reduce "
                        "the max workers for other endpoints or lower the max worker count "
                        "for this endpoint to at most 0.",
                        "status": self.status,
                    },
                },
            )
        return super().create_endpoint(body_path, confirm)


def test_quota_caps_this_runs_workers_and_its_ledger_reservation(world):
    fake = QuotaSls(held=7)
    r = make(world, eps12(), fake, workers=5, vcpu=8)
    assert r.run() == 0
    assert [b["workersMax"] for b in fake.created] == [3]
    (row,) = tsl.ledger_rows(world)
    assert row["rate"] == pytest.approx(3 * 0.61 * 1.25)
    capped = [e for e in events(r) if e["event"] == "workers_capped_by_quota"]
    assert (
        capped
        and capped[0]["got"] == 3
        and capped[0]["held_by_other_endpoints"] == {"someone-else (otherep)": 7}
    )
    assert [len(units_of(b)) for b in episode_jobs(fake)] == [2, 2, 2]


def test_quota_full_waits_then_refuses_cleanly_and_creates_nothing(world, capsys):
    sp = dict(world["sp"], quota_wait_seconds=300, quota_poll_seconds=60)
    fake = QuotaSls(held=10)
    r = make(world, eps12(), fake, sp=sp)
    assert r.run() == 3
    assert fake.created == [] and tsl.ledger_rows(world) == []
    assert "worker quota full" in r.stop_reason and "someone-else" in r.stop_reason
    assert "nothing was created" in r.stop_reason
    assert "waiting: serverless worker quota full" in capsys.readouterr().err
    assert fake.lists >= 5  # it polled while waiting
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["exit_code"] == 3 and receipt["final_endpoints_runner_owned"] == []


def test_quota_frees_while_waiting_then_the_run_proceeds(world):
    sp = dict(world["sp"], quota_wait_seconds=1800, quota_poll_seconds=60)
    fake = QuotaSls(held=10, free_after_lists=3)
    r = make(world, eps12(), fake, sp=sp, workers=2)
    assert r.run() == 0
    assert [b["workersMax"] for b in fake.created] == [2]
    assert any(e["event"] == "worker_quota_full" for e in events(r))


@pytest.mark.parametrize("status", [400, 500])
def test_create_refused_for_quota_rechecks_and_retries(world, status):
    fake = QuotaSls(held=0, quota_refusals=1, status=status)
    r = make(world, eps12(), fake, workers=2)
    assert r.run() == 0
    assert len(fake.created) == 2 and len(fake.deleted) == 1
    rows = tsl.ledger_rows(world)
    assert sorted(round(x["cost"], 6) for x in rows)[0] == 0.0  # the refused one released
    assert any(e["event"] == "endpoint_quota_retry" for e in events(r))


def test_quota_unreadable_refuses_before_any_reservation(world):
    fake = QuotaSls(held=0, list_fails=True)
    r = make(world, eps12(), fake)
    assert r.run() == 3
    assert fake.created == [] and tsl.ledger_rows(world) == []
    assert "worker quota" in r.stop_reason


def test_plan_reports_quota_and_batches(world):
    fake = QuotaSls(held=7)
    j = job(eps12(), repo_commit=world["commit"], checkpoints=[world["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    out = serverless.plan_serverless(
        j, world["fp"], world["sp"], world["allow"], 2.0, rp=fake, repo=world["repo"]
    )
    q = out["worker_quota"]
    assert q["account_worker_quota"] == 10 and q["workers_in_use"] == 7 and q["free_now"] == 3
    assert q["workers_max_held_by_endpoints"] == {"someone-else (otherep)": 7}
    assert q["a_run_started_now"].startswith("runs ") and "before a run starts" in q["rule"]
    assert out["sizing"]["worker_quota_free_now"] == 3 and out["sizing"]["choice"]["workers"] <= 3
    c = out["sizing"]["choice"]
    assert c["units_per_job"] == "auto" and c["units_per_job_max"] >= 1
    assert out["endpoint_body_dry_run"]["executionTimeoutMs"] >= 1000 * (
        4 * (12.0 + 0.0182 * 1000) + 300
    )
    assert fake.created == [] and fake.bodies == []  # read-only


# ---------------------------------------------------------------- guards kept


def test_strict_and_serving_jobs_refused_with_batching_flags(tmp_path):
    for over in ({"purpose": "strict gate run"}, {"job_id": "v8-serving-a"}):
        p = tmp_path / "job.json"
        p.write_text(json.dumps(job(**over)))
        for cmd in (["plan", str(p)], ["run", str(p), "--budget", "1", "--confirm"]):
            with pytest.raises(jobspec.JobError):
                runner.main(cmd + ["--units-per-job", "4"])


def test_bad_units_per_job_is_a_usage_error(tmp_path, capsys):
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job()))
    assert runner.main(["plan", str(p), "--units-per-job", "0"]) == 2
    assert "units-per-job" in capsys.readouterr().err


def test_batching_needs_no_new_runtime():
    """K>1 rides the handler that is already seeded (rpf-sls-handler/v1): a flat episode
    list (<= 64 episodes, <= 16 slots). Editing sls_handler.py changes the runtime id and
    needs `serverless.py seed` + `template-create` before any run (update these pins)."""
    fp, sp = jobspec.load_policy(), serverless.load_sls_policy()
    assert serverless.handler_sha256() == (
        "ecd376bd54b2dc88b5b19057da6fe7f4baaa3853c1811519bc9b80a472fc7d2e"
    )
    assert serverless.runtime_id(fp, sp) == "64083e3428a874e7"
    assert serverless.MAX_JOB_SLOTS * int(fp["threads_per_episode"]) >= max(sp["vcpu_sizes"])


# ---------------------------------------------------------------- shorter run wall


def test_shorter_run_wall_shrinks_deadline_watchdog_and_ledger_horizon(world):
    fake = FakeSls()
    r = make(world, eps12(), fake, max_wall_minutes=10)
    assert r.run() == 0
    assert r.deadline == pytest.approx(r.start + 600)
    (row,) = tsl.ledger_rows(world)
    grace = float(world["fp"]["watchdog_grace_seconds"])
    assert row["until"] == pytest.approx(r.start + 600 + grace + 300)
    assert fake.created[0]["executionTimeoutMs"] <= (row["until"] - r.start) * 1000
    argv = runner.watchdog_argv(r, None)
    assert float(argv[argv.index("--fire-epoch") + 1]) == pytest.approx(r.deadline + grace)
    plan = json.loads((r.run_dir / "plan.json").read_text())
    assert plan["max_wall_minutes"] == 10 and plan["units_per_job"] == "auto"
    assert json.loads((r.run_dir / "job.json").read_text())["max_wall_minutes"] == 30  # as is
    with pytest.raises(jobspec.JobError, match="max-wall-minutes"):
        make(world, eps12(), FakeSls(), name="too-long", max_wall_minutes=31)


def test_plan_with_a_shorter_wall_reserves_less(world, tmp_path, capsys):
    j = job(eps12(), repo_commit=world["commit"], checkpoints=[world["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    kw = dict(rp=FakeSls(), repo=world["repo"], workers=2, vcpu=8)
    full = serverless.plan_serverless(j, world["fp"], world["sp"], world["allow"], 3.0, **kw)
    short = serverless.plan_serverless(
        j, world["fp"], world["sp"], world["allow"], 3.0, max_wall_minutes=10, **kw
    )
    assert full["max_wall_minutes_run"] == 30 and short["max_wall_minutes_run"] == 10
    assert short["sizing"]["choice"]["worst_case_usd"] < full["sizing"]["choice"]["worst_case_usd"]
    p = tmp_path / "job.json"
    p.write_text(json.dumps(job()))
    assert runner.main(["plan", str(p), "--max-wall-minutes", "45"]) == 2
    assert "max-wall-minutes" in capsys.readouterr().err
    assert runner.main(["run", str(p), "--backend", "local", "--units-per-job", "2"]) == 2
    assert "serverless backend only" in capsys.readouterr().err


# ---------------------------------------------------------------- review fixes (2nd round)


def test_quota_counts_endpoints_that_share_a_name_or_have_none():
    rows = [
        {"id": "a", "name": "inference", "workersMax": 5},
        {"id": "b", "name": "inference", "workersMax": 5},
        {"id": "a", "name": "inference", "workersMax": 5},  # the same endpoint listed twice
        {"workersMax": 2},  # no name, no id
        {"id": "c", "name": "idle", "workersMax": 0},
    ]
    held, total = serverless.quota_held(rows)
    assert total == 12 and len(held) == 3
    assert held == {"inference (a)": 5, "inference (b)": 5, "<no name> (row 3)": 2}
    q = serverless.quota_status(rows, tsl.spol())
    assert q["workers_in_use"] == 12 and q["free_now"] == -2


class TwinSls(QuotaSls):
    """Two other endpoints with the SAME name fill the quota (5 + 5)."""

    def list_endpoints(self):
        self.lists += 1
        twins = [
            {"id": "t1", "name": "inference", "workersMax": 5},
            {"id": "t2", "name": "inference", "workersMax": 5},
        ]
        return twins + [dict(e) for e in self.endpoints.values()]


def test_endpoints_sharing_a_name_fill_the_quota(world):
    sp = dict(world["sp"], quota_wait_seconds=120, quota_poll_seconds=60)
    fake = TwinSls(held=0)
    r = make(world, eps12(), fake, sp=sp)
    assert r.run() == 3
    assert fake.created == [] and tsl.ledger_rows(world) == []
    assert "hold 10 of the account's 10" in r.stop_reason


def sls_args(**over):
    import argparse

    a = dict(
        flavor=None,
        budget=3.0,
        target_minutes=None,
        workers=None,
        vcpu_per_worker=None,
        units_per_job="auto",
        max_wall_minutes=None,
        resume_from=[],
    )
    a.update(over)
    return argparse.Namespace(**a)


def run_sls(world, fake, name, args, sp=None):
    """runner.py run's serverless path (run_serverless) against the fake account."""
    j = job(eps12(), repo_commit=world["commit"], checkpoints=[world["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    clock = Clock()
    made = []

    class Capture(serverless.ServerlessRunner):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            made.append(self)

    rc = serverless.run_serverless(
        args,
        j,
        world["fp"],
        world["allow"],
        world["tmp"] / "runs" / name,
        rp=fake,
        sp=sp or world["sp"],
        clock=clock,
        sleep=clock.sleep,
        runner_cls=Capture,
        registry=world["reg"],
        repo=world["repo"],
        spawn_watchdog=lambda r: None,
        preflight_fn=lambda: [],
        probe_workers=1,
    )
    return rc, made, clock


def test_run_waits_for_the_quota_before_it_starts_and_refuses_with_nothing_created(world, capsys):
    sp = dict(world["sp"], quota_wait_seconds=300, quota_poll_seconds=60)
    fake = QuotaSls(held=10)
    rc, made, clock = run_sls(world, fake, "q-full", sls_args(), sp)
    assert rc == 2 and made == []  # no runner: no run dir, ledger entry, watchdog
    assert not (world["tmp"] / "runs" / "q-full").exists()
    ledger = world["tmp"] / "artifacts" / "runpod-fanout" / "ledger-v2.json"
    assert not ledger.exists() or tsl.ledger_rows(world) == []
    assert fake.created == [] and fake.lists >= 5
    err = capsys.readouterr().err
    assert "waiting: serverless worker quota: 0 of the account's 10" in err
    assert "waited 300 s; nothing was created" in err and "someone-else (otherep)" in err


def test_the_quota_wait_does_not_consume_the_run_wall(world):
    sp = dict(world["sp"], quota_wait_seconds=1800, quota_poll_seconds=60)
    fake = QuotaSls(held=10, free_after_lists=10)  # frees after ~10 minutes of waiting
    rc, (r,), clock = run_sls(world, fake, "q-wait", sls_args(), sp)
    assert rc == 0 and len(tsl.records_of(r.run_dir)) == 12
    assert r.start >= 540  # the run started after the wait ...
    assert r.deadline - r.start == pytest.approx(30 * 60)  # ... with its whole wall
    assert not any(e["event"] == "worker_quota_full" for e in events(r))


def test_run_sizes_itself_to_the_free_quota(world, capsys):
    fake = QuotaSls(held=7)
    rc, (r,), _ = run_sls(world, fake, "q-some", sls_args(workers=9, vcpu_per_worker=8))
    assert rc == 0 and r.workers == 3 and [b["workersMax"] for b in fake.created] == [3]
    out = capsys.readouterr().out
    choice = json.loads(next(x for x in out.splitlines() if "serverless_choice" in x))
    assert choice["worker_quota_free"] == 3 and choice["serverless_choice"]["workers"] == 3


def test_quota_race_at_create_with_too_few_workers_for_the_wall_aborts(world, capsys):
    """Sized for 5 at start; at create only 1 is free and 1 worker would not finish inside
    the 5-min wall: it waits only while 5 could still finish, then aborts (nothing made)."""
    sp = dict(world["sp"], quota_wait_seconds=300, quota_poll_seconds=60)
    fake = QuotaSls(held=9)
    r = make(world, eps12(), fake, sp=sp, workers=5, max_wall_minutes=5)
    assert r.run() == 3
    assert fake.created == [] and tsl.ledger_rows(world) == []
    assert "quota short" in r.stop_reason and "fewer would not finish" in r.stop_reason
    assert "waited 120 s" in r.stop_reason and fake.lists == 3


def test_quota_full_with_no_wall_to_spare_refuses_at_once(world):
    sp = dict(world["sp"], quota_wait_seconds=1800, quota_poll_seconds=60)
    fake = QuotaSls(held=10)
    long = [dict(e, horizon=5000) for e in eps12()]  # 2 rounds x 103 s x 1.5 > 0.6 x 5 min
    r = make(world, long, fake, sp=sp, workers=2, max_wall_minutes=5)
    assert r.run() == 3 and fake.lists == 1 and fake.created == []
    assert "the run wall leaves no time to wait for the quota" in r.stop_reason


class DrainSls(FakeSls):
    """The account balance falls $1 per read once an endpoint exists (a size priced far
    above what serverless_policy.json reserves); billing still reads ~$0 (as live)."""

    def balance(self):
        if self.endpoints:
            self.balance_value -= 1.0
        return self.balance_value


def test_balance_guard_aborts_when_a_size_bills_above_its_reservation(world):
    fake = DrainSls(running_polls=10**6)
    r = make(world, eps12(), fake, workers=2, vcpu=8)
    assert r.run() == 3
    assert "account balance fell" in r.stop_reason and "under-prices" in r.stop_reason
    assert fake.endpoints == {} and fake.deleted == ["ep1"]
    receipt = json.loads((r.run_dir / "receipt.json").read_text())
    assert receipt["balance_guard"]["exceeded"] is True
    checks = [e for e in events(r) if e["event"] == "balance_check"]
    assert checks and checks[-1]["exceeded"] is True


def test_worker_price_floor_wall_factor_and_fill(tmp_path):
    fp, sp = tsl.fpol(tmp_path), tsl.spol()
    both = ["cpu5c", "cpu3c"]
    assert serverless.worker_rate(sp, both, 2) == pytest.approx(0.61)  # billed 2-vCPU rate
    assert serverless.worker_rate(sp, both, 32) == pytest.approx(32 * 0.042)
    assert serverless.worker_rate(dict(sp, usd_per_worker_hr_min=0), both, 2) == 0.084
    eps = [dict(ep(a, seed=s), horizon=5000) for s in range(40) for a in "ABC"]
    one = dict(sp, episode_time_factor=1.0, max_wall_fill=1.0)
    kw = dict(workers=4, vcpu=16)
    a = serverless.sizing_plan(fp, one, eps, 20.0, 120, both, 30.0, **kw)["choice"]
    b = serverless.sizing_plan(fp, sp, eps, 20.0, 120, both, 30.0, **kw)["choice"]
    cold = sp["cold_start_seconds_est"] / 60
    assert b["wall_minutes"] - cold == pytest.approx(1.5 * (a["wall_minutes"] - cold), abs=0.15)
    assert b["usd_per_hr"] == pytest.approx(4 * 16 * 0.042)
    # a wall between max_wall_fill x the run wall and the run wall is refused
    m = (b["wall_minutes"] / 0.6 + b["wall_minutes"]) / 2
    tight = serverless.sizing_plan(fp, sp, eps, 20.0, m, both, 30.0, **kw)["choice"]
    assert "max wall" in tight["refused"]
    assert not serverless.sizing_plan(fp, one, eps, 20.0, m, both, 30.0, **kw)["choice"]["refused"]


def test_plan_sizes_to_the_free_quota_and_shows_the_full_quota_choice(world):
    fake = QuotaSls(held=10)
    j = job(eps12(), repo_commit=world["commit"], checkpoints=[world["hero"]])
    for e in j["episodes"]:
        e["roster_member_sha256s"] = [world["hero"], RANDOM, GREEDY, RANDOM, GREEDY]
    out = serverless.plan_serverless(
        j, world["fp"], world["sp"], world["allow"], 3.0, rp=fake, repo=world["repo"]
    )
    q = out["worker_quota"]
    assert q["free_now"] == 0 and not out["sizing"]["choice_ok"]
    assert q["a_run_started_now"].startswith("waits for the quota")
    assert (
        q["choice_when_quota_free"]["workers"] >= 1 and not q["choice_when_quota_free"]["refused"]
    )
