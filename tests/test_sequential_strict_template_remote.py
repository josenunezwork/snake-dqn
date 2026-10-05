"""Template v3 of research/sequential_strict_template: the opt-in RunPod serverless backend.

Governance amendment strict on RunPod (2026-10-05). No test touches the network or RunPod: the
endpoint is ``FakeStrictRp``, which plays each episode in-process through
``remote_worker.execute`` (the code a worker runs) on the test's fake episode function and
stamps it as a Linux x86 worker; the shared RunPod ledger lives in tmp; TLS, seeding and the
watchdog are injected; and every path to a real rollout, probe or child process raises (the
guards of the v1 tests).
"""

from __future__ import annotations

import hashlib
import json
import socket
import sys
import time
from pathlib import Path

import pytest

from research.runpod_fanout import jobspec, serverless
from research.runpod_fanout.ledger import SharedLedger
from research.runpod_fanout.rp_client import RpClient, RunPodError
from research.runpod_fanout.wrappers import HERO_SHA256
from research.sequential_strict_template import remote_backend as RB
from research.sequential_strict_template import remote_worker as W
from research.sequential_strict_template import sequential_audit as A
from research.sequential_strict_template import sequential_runner as R
from research.sequential_strict_template import strict_sls_handler as H
from tests.test_sequential_strict_template import (  # noqa: F401 - guards is an autouse fixture
    MIXES,
    N_MAX,
    FakeWorld,
    fake_skew,
    guards,
    intent_kwargs,
    make_spec,
    output_of,
    receipts,
    why,
)

EPYC = "AMD EPYC 9655 96-Core Processor"
XEON = "Intel Xeon Gold 6338"
TIER1_RUNTIME_ID = "64083e3428a874e7"  # main before template v3 (the seeded Tier-1 runtime)
SEEDED = {
    "runtime_id": "strict-test",
    "repo_sha256": "0" * 64,
    "template_id": "tpl-strict",
    "ready": True,
    "needs": None,
    "problems": [],
    "volume": {"id": "vol1", "dataCenterId": "EU-RO-1"},
}
CONFIG = {
    "schema": RB.CONFIG_SCHEMA,
    "platform": "runpod-serverless",
    "budget_usd": 20.0,
    "identity_budget_usd": 3.0,
    "remote_wall_minutes": 60,
    "identity_wall_minutes": 40,
    "mac_episode_seconds": 100.0,
    "mac_episode_seconds_source": "test value",
    "engine": "live",
    "horizon": 5000,
    "horizon_path": "horizon",
    "cloud_episode_seconds": 10.0,
    "workers": 4,
    "vcpu_per_worker": 16,
}


def _fatal(*args, **kwargs):
    raise AssertionError("tests must never reach RunPod, the network or a real watchdog")


@pytest.fixture(autouse=True)
def remote_guards(monkeypatch):
    """No rp.py, no sockets but loopback, no real watchdog or TLS probe; the lid is open."""
    monkeypatch.setattr(RpClient, "_call", _fatal)
    monkeypatch.setattr(RB.rmod, "spawn_watchdog_process", _fatal)
    monkeypatch.setattr(RB.rmod, "preflight", _fatal)
    monkeypatch.setattr(R, "lid_open", lambda: True)
    real_connect = socket.socket.connect

    def guarded(sock, address):
        host = address[0] if isinstance(address, tuple) else address
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise AssertionError(f"network call to {address} attempted")
        return real_connect(sock, address)

    monkeypatch.setattr(socket.socket, "connect", guarded)


class RemoteWorld(FakeWorld):
    """FakeWorld records plus the pinned horizon and a wall-clock probe (dropped as timing)."""

    def __call__(self, episode, row, context):
        record = super().__call__(episode, row, context)
        record["horizon"] = 5000
        record["probes"] = {"veto": {"apply_seconds_total": time.monotonic()}}
        return record


def make_remote_spec(world, tmp_path, **over):
    base = make_spec(world, tmp_path)
    Path(base.protocol_path).write_text(
        "protocol.md: test pre-registration document\nExecution platform: runpod-serverless\n"
    )
    fields = {
        **base.__dict__,
        "remote_worker_setup": lambda intent, ckpt_dir: {"ckpt_dir": str(ckpt_dir)},
        "remote_checkpoints": lambda: [HERO_SHA256],
    }
    fields.update(over)
    return R.StudySpec(**fields)


def write_config(tmp_path, **over):
    path = tmp_path / "remote_config.json"
    path.write_text(json.dumps({**CONFIG, **over}))
    return path


def prepare_remote(tmp_path, spec, config=None, seeding=SEEDED, **overrides):
    cfg = write_config(tmp_path, **(config or {}))
    kwargs = intent_kwargs(tmp_path, **overrides)
    return R.prepare(R.build_intent(spec, **kwargs, remote_config=cfg, remote_seeding=seeding))


class FakeStrictRp:
    """Fake RunPod: endpoints, jobs and the account balance; a job plays its units with the
    worker's own code (``remote_worker.execute``) once it has been polled ``running_polls``
    times. Knobs inject the failures the backend must survive or stop on."""

    def __init__(
        self,
        spec,
        *,
        models=(EPYC, XEON),
        running_polls=0,
        fail_jobs_once=(),
        worker_failure_once=(),
        episode_error=(),
        perturb=None,
        horizon_off=(),
        binding_tamper=(),
        drain_per_status=0.0,
        never_start=False,
    ):
        self.spec = spec
        self.models = list(models)
        self.running_polls = running_polls
        self.fail_jobs_once = set(fail_jobs_once)
        self.worker_failure_once = set(worker_failure_once)
        self.episode_error = set(episode_error)
        self.perturb = perturb
        self.horizon_off = set(horizon_off)
        self.binding_tamper = set(binding_tamper)
        self.drain = drain_per_status
        self.never_start = never_start
        self.balance_value = 40.0
        self.endpoints, self.jobs = {}, {}
        self.created, self.deleted, self.cancelled, self.bodies = [], [], [], []
        self.plays = {}
        self.n = 0
        self.executed = 0

    def balance(self):
        return self.balance_value

    def list_endpoints(self):
        return [dict(e) for e in self.endpoints.values()]

    def get_endpoint(self, eid, workers=False):
        e = self.endpoints.get(eid)
        return None if e is None else dict(e, workers=[])

    def create_endpoint(self, body_path, confirm):
        assert confirm is True
        body = json.loads(Path(body_path).read_text())
        self.n += 1
        eid = f"ep{self.n}"
        self.created.append(body)
        self.endpoints[eid] = dict(body, id=eid)
        return {"id": eid, "name": body["name"]}

    def update_endpoint(self, eid, body, confirm):
        assert confirm is True
        if eid not in self.endpoints:
            raise RunPodError("nf", {"http_status": 404})
        self.endpoints[eid].update(body)
        return {}

    def delete_endpoint(self, eid, confirm):
        assert confirm is True and self.endpoints[eid]["workersMax"] == 0, "scale to 0 first"
        self.endpoints.pop(eid)
        self.deleted.append(eid)
        return {}

    def endpoint_billing(self, eid, start, end):
        return 0.0

    def sls(self, eid, op, job_id=None, body=None, confirm=False):
        if op == "run":
            assert confirm is True and eid in self.endpoints
            self.n += 1
            jid = f"job{self.n}"
            self.bodies.append(json.loads(json.dumps(body)))
            self.jobs[jid] = {"input": body["input"], "done": None, "polls": 0}
            return {"id": jid, "status": "IN_QUEUE"}
        if op in ("cancel", "purge-queue"):
            self.cancelled.append((eid, op, job_id))
            return {}
        assert op == "status"
        self.balance_value -= self.drain
        j = self.jobs[job_id]
        if self.never_start:
            return {"id": job_id, "status": "IN_QUEUE"}
        j["polls"] += 1
        if j["done"] is None and j["polls"] <= self.running_polls:
            return {"id": job_id, "status": "IN_PROGRESS"}
        if j["done"] is None:
            j["done"] = self.execute(job_id, j["input"])
        return j["done"]

    def execute(self, job_id, inp):
        model = self.models[self.executed % len(self.models)]
        self.executed += 1
        if inp["op"] == "probe":
            worker = {
                "runtime_id": inp["runtime_id"],
                "handler_sha256": hashlib.sha256(RB.STRICT_HANDLER.read_bytes()).hexdigest(),
                "cpu_model": model,
            }
            return {"id": job_id, "status": "COMPLETED", "output": {"ok": True, "worker": worker}}
        first = inp["units"][0]["unit"]
        if first in self.fail_jobs_once:
            self.fail_jobs_once.discard(first)
            return {"id": job_id, "status": "FAILED", "error": "worker died"}
        units = []
        for unit in inp["units"]:
            results = []
            for episode in unit["episodes"]:
                eid = episode["episode_id"]
                self.plays[eid] = nth = self.plays.get(eid, 0) + 1
                if eid in self.worker_failure_once and nth == 1:
                    results.append({"episode_id": eid, "kind": "worker_failure", "rc": -9})
                    continue
                if eid in self.episode_error:
                    results.append({"episode_id": eid, "kind": "episode_error", "error": "boom"})
                    continue
                payload = {
                    "unit": unit["unit"],
                    "episode": episode,
                    "row": unit["row"],
                    "intent_sha256": inp["intent_sha256"],
                    "job_id": inp["job_id"],
                }
                code, out = W.execute(
                    inp["intent_text"], inp["intent_sha256"], payload, Path("/ckpt"), spec=self.spec
                )
                assert code == 0, out
                out["platform"] = {
                    **out["platform"],
                    "platform_id": f"linux-x86_64|py3.12.15|torch-2.9.1+cpu|cpu-{model}",
                    "cpu_model": model,
                }
                if self.perturb is not None and self.perturb(eid, nth):
                    out["record"]["mass_integral"] += 1e-9
                if eid in self.horizon_off:
                    out["record"]["horizon"] = 4999
                if eid in self.binding_tamper:
                    out["checks"]["arm_identity_sha256"] = "0" * 64
                data = json.dumps(out, sort_keys=True)
                results.append(
                    {
                        "episode_id": eid,
                        "kind": "ok",
                        "rc": 0,
                        "output": data,
                        "sha256": hashlib.sha256(data.encode()).hexdigest(),
                    }
                )
            units.append({"unit": unit["unit"], "results": results})
        worker = {"worker_id": f"w{self.executed}", "cpu_model": model, "isa_flags": ["avx2"]}
        return {
            "id": job_id,
            "status": "COMPLETED",
            "workerId": f"w{self.executed}",
            "output": {"ok": True, "worker": worker, "units": units},
        }


def session_factory(tmp_path, fake, *, rpol_over=None, power=None, lid=None, seed=SEEDED):
    rpol, fp, sp = RB.policies()
    rpol = dict(rpol, **(rpol_over or {}))
    fp = dict(fp, artifacts_root=str(tmp_path / "rp-artifacts"))
    ledger = SharedLedger(fp)

    def factory(**kw):
        return RB.RemoteSession(
            **kw,
            rp=fake,
            policies_=(rpol, fp, sp),
            ledger=ledger,
            clock=time.time,
            sleep=lambda s: None,
            power_check=power or (lambda: True),
            lid_check=lid or (lambda: True),
            spawn_watchdog=lambda target: None,
            preflight_fn=lambda: [],
            seed_check=lambda: dict(seed),
            transport="injected:FakeStrictRp",
        )

    factory.ledger = ledger
    return factory


def identity(intent_path, spec, factory, mac=RB.in_process_mac_identity):
    return RB.run_identity_check(
        intent_path,
        spec=spec,
        mac_runner=mac,
        session_factory=factory,
        power_check=lambda: True,
        lid_check=lambda: True,
    )


def execute_remote(intent_path, spec, factory):
    return R.run(
        intent_path,
        spec=spec,
        executor=R.InProcessExecutor(),
        skew_runner=fake_skew(),
        audit_runner=R.in_process_audit_runner,
        remote_factory=lambda path, intent, s: RB.RemoteExecutor(
            path, intent, s, session_factory=factory
        ),
    )


def records(intent_path, phase="final"):
    folder = output_of(intent_path) / phase / "records"
    return {p.stem: json.loads(p.read_text()) for p in sorted(folder.glob("*.json"))}


def ledger_runs(factory):
    with factory.ledger.locked(write=False) as state:
        return json.loads(json.dumps(state["runs"]))


def passing_remote_run(tmp_path, world=None, **fake_kw):
    spec = make_remote_spec(world or RemoteWorld(effect=400.0), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    fake = FakeStrictRp(spec, **fake_kw)
    factory = session_factory(tmp_path, fake)
    assert identity(intent_path, spec, factory)["passes"] is True
    return spec, intent_path, fake, factory


# ---------------------------------------------------------------- unchanged when not opted in


def test_v1_v2_intents_keep_no_execution_block_and_the_tier1_runtime_is_unchanged(tmp_path):
    spec = make_spec(FakeWorld(), tmp_path)
    intent = R.build_intent(spec, **intent_kwargs(tmp_path))
    assert intent["template_version"] == R.TEMPLATE_VERSION and "execution" not in intent
    with pytest.raises(R.StrictRunError, match="remote_seeding without remote_config"):
        R.build_intent(spec, **intent_kwargs(tmp_path), remote_seeding=SEEDED)
    fp, sp = jobspec.load_policy(), serverless.load_sls_policy()
    assert serverless.runtime_id(fp, sp) == TIER1_RUNTIME_ID
    assert serverless.runtime_id(fp, sp, serverless.HANDLER_SOURCE) == TIER1_RUNTIME_ID
    strict = serverless.runtime_id(fp, sp, serverless.HANDLERS["strict"])
    assert strict != TIER1_RUNTIME_ID and serverless.HANDLERS["strict"] == RB.STRICT_HANDLER
    seeder = serverless.SeedRunner(
        fp,
        sp,
        {},
        tmp_path / "seed",
        commit="0" * 40,
        volume={"id": "v", "dataCenterId": "EU-RO-1"},
        ckpts=[],
        rp=object(),
        handler_source=RB.STRICT_HANDLER,
    )
    assert seeder.rid == strict and seeder.job["job_id"].startswith("seed-")
    with pytest.raises(R.StrictRunError, match="remote_factory needs a template v3"):
        R.run(
            R.prepare(intent),
            spec=spec,
            executor=R.InProcessExecutor(),
            remote_factory=lambda *a: None,
        )


# ---------------------------------------------------------------- prepare and plan


def test_prepare_v3_freezes_platform_plan_caps_sample_and_fallback(tmp_path):
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    intent = R.read_json(prepare_remote(tmp_path, spec))
    assert intent["template_version"] == R.TEMPLATE_VERSION_REMOTE
    assert intent["spec"]["template_version"] == R.TEMPLATE_VERSION  # band template v1
    block = intent["execution"]
    assert block["platform"] == block["backend"] == "runpod-serverless"
    assert block["band_template_version"] == R.TEMPLATE_VERSION
    assert block["remote_config"]["sha256"] == R.sha256_file(tmp_path / "remote_config.json")
    assert block["handler"]["runtime_id"] == serverless.runtime_id(
        jobspec.load_policy(), serverless.load_sls_policy(), RB.STRICT_HANDLER
    )
    assert block["checkpoints"] == [HERO_SHA256]
    assert block["sizing"]["workers"] == 4 and block["sizing"]["slots_per_worker"] == 8
    assert block["plan"]["meets_min_speedup"] is True and block["speedup"] >= 5.0
    assert block["forced_below_min"] is False
    assert block["cost_cap"] == {
        "gate_usd": 20.0,
        "identity_usd": 3.0,
        "rule": block["cost_cap"]["rule"],
    }
    assert "INVALID_STOP" in block["cost_cap"]["rule"]
    sample = block["identity_check"]["sample"]
    assert len(sample) == 3 * len(MIXES) and {s["mix"] for s in sample} == set(MIXES)
    assert all(s["world_index"] < intent["plan"]["look_sizes"][0] for s in sample)
    assert sample == RB.identity_sample(intent, 3)
    assert block["fallback"]["backend"] == RB.FALLBACK_BACKEND and block["fallback"]["slots"] == 2
    assert block["amendment"]["sha256"] and block["amendment"]["ratified"] is False
    R.validate_intent(intent, spec)


def test_prepare_refuses_without_hooks_platform_line_allowlist_or_5x(tmp_path):
    no_hooks = make_spec(RemoteWorld(), tmp_path)
    Path(no_hooks.protocol_path).write_text("Execution platform: runpod-serverless\n")
    with pytest.raises(R.StrictRunError, match="remote_worker_setup and remote_checkpoints"):
        prepare_remote(tmp_path, no_hooks)
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    Path(spec.protocol_path).write_text("no platform named here\n")
    with pytest.raises(R.StrictRunError, match="protocol must name the platform"):
        prepare_remote(tmp_path, spec)
    spec = make_remote_spec(RemoteWorld(), tmp_path, remote_checkpoints=lambda: ["a" * 64])
    with pytest.raises(R.StrictRunError, match="allow-list"):
        prepare_remote(tmp_path, spec)
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    with pytest.raises(R.StrictRunError, match="budget_usd"):
        prepare_remote(tmp_path, spec, config={"budget_usd": 400.0})
    with pytest.raises(R.StrictRunError, match="per-step 5x rule"):
        prepare_remote(tmp_path, spec, config={"mac_episode_seconds": 20.0})
    forced = R.read_json(
        prepare_remote(tmp_path, spec, config={"mac_episode_seconds": 20.0, "force_below_5x": True})
    )
    assert forced["execution"]["forced_below_min"] is True
    assert forced["execution"]["plan"]["meets_min_speedup"] is False


def test_plan_projects_a_v8_sized_gate_against_two_mac_slots():
    """The v8 strict gate's shape (N_max 249, 16 calibration worlds, 34.4 s Mac episodes)."""
    rpol, fp, sp = RB.policies()
    rpol_ok, fp_ok, sp_ok = dict(rpol), dict(fp), dict(sp)
    base = {
        **CONFIG,
        "cloud_episode_seconds": None,
        "workers": None,
        "vcpu_per_worker": None,
        "remote_wall_minutes": 120,
        "identity_wall_minutes": 45,
        "budget_usd": 30.0,
        "identity_budget_usd": 5.0,
        "mac_episode_seconds": 34.4,
    }
    plan = RB.plan_remote(
        mixes=MIXES,
        n_calibration=16,
        look_sizes=[63, 125, 187, 249],
        cfg={**base, "flavors": list(sp["flavors_pref"]), "identity_worlds_per_mix": 3},
        rpol=rpol_ok,
        fp=fp_ok,
        sp=sp_ok,
        seeding={"needs": None},
    )
    choice = plan["choice"]
    assert plan["episodes_total"] == 16 * 3 + 249 * 3 * 2
    assert plan["mac_minutes"] == pytest.approx(1542 * 34.4 / 2 / 60, abs=0.1)
    assert plan["meets_min_speedup"] is True and choice["speedup"] >= 5.0
    assert choice["reserved_usd_per_hr"] <= sp["max_endpoint_hourly_usd"]
    assert choice["worst_gate_usd"] <= 30.0 and choice["workers"] <= sp["account_worker_quota"]
    assert len(choice["speedup_by_stop_look"]) == 4
    assert choice["speedup_by_stop_look"][0] < choice["speedup_by_stop_look"][-1]
    # seeding a fresh runtime is counted, and a small slow-to-pay-off plan is flagged
    unseeded = RB.plan_remote(
        mixes=MIXES,
        n_calibration=16,
        look_sizes=[63, 125, 187, 249],
        cfg={**base, "flavors": list(sp["flavors_pref"]), "identity_worlds_per_mix": 3},
        rpol=rpol_ok,
        fp=fp_ok,
        sp=sp_ok,
        seeding=None,
    )
    assert unseeded["seeding"]["minutes"] == rpol["seed_minutes_runtime"]
    tiny = RB.plan_remote(
        mixes=MIXES,
        n_calibration=4,
        look_sizes=[3, 6, 9, 12],
        cfg={**base, "flavors": list(sp["flavors_pref"]), "identity_worlds_per_mix": 3},
        rpol=rpol_ok,
        fp=fp_ok,
        sp=sp_ok,
        seeding={"needs": None},
    )
    assert tiny["meets_min_speedup"] is False


def test_execution_documents_are_frozen(tmp_path):
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    intent = R.read_json(intent_path)
    (tmp_path / "remote_config.json").write_text(json.dumps({**CONFIG, "budget_usd": 25.0}))
    with pytest.raises(R.StrictRunError, match="remote config drift"):
        R.validate_intent(intent, spec)
    assert RB.execution_drift(intent) == [str((tmp_path / "remote_config.json").resolve())]


def test_production_remote_intent_needs_the_ratified_amendment(tmp_path):
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    intent = R.read_json(prepare_remote(tmp_path, spec))
    production = dict(intent, dry_run=False, allow_dirty=False)
    with pytest.raises(R.StrictRunError, match="not ratified"):
        RB.execution_block(spec, production, tmp_path / "remote_config.json", seeding=SEEDED)
    with pytest.raises(R.StrictRunError, match="ratified amendment"):
        RB.validate_execution(production, spec)
    doc = tmp_path / "amendment.md"
    doc.write_text("## Ratification\n\n- Decision: ratified (owner)\n- Date: 2026-10-06\n")
    assert RB.amendment_status(doc)["ratified"] is True
    doc.write_text("## Ratification\n\n- Decision: ______\n")
    assert RB.amendment_status(doc)["ratified"] is False
    assert RB.amendment_status(RB.REPO / RB.load_remote_policy()["amendment_path"])["sha256"]


# ---------------------------------------------------------------- identity check and backends


def test_identity_pass_runs_the_gate_on_runpod_and_the_audit_passes(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    result = R.read_json(intent_path.parent / RB.IDENTITY_FILE)
    assert result["passes"] is True and result["backend_decided"] == RB.REMOTE_BACKEND
    assert result["identical"] == result["expected"] == 18
    assert len(fake.created) == 1 and len(fake.deleted) == 1  # identity endpoint, torn down
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["provenance"]["executor"] == R.REMOTE_EXECUTOR
    assert closeout["execution"]["backend"] == RB.REMOTE_BACKEND
    assert closeout["execution"]["identity_check"]["state"] == "PASSED"
    assert closeout["execution"]["remote"]["leftover_endpoints"] == []
    assert len(fake.created) == 2 and len(fake.deleted) == 2 and not fake.endpoints
    started = R.read_json(output_of(intent_path) / "started.json")
    assert started["execution"]["liveness_lock"].endswith("orchestrator.lock")
    assert started["cpu_slot_locks_held"] == []  # the remote orchestrator holds no CPU slot
    final = records(intent_path)
    assert len(final) == 2 * 30
    by_unit = {}
    for entry in final.values():
        assert entry["platform"]["backend"] == RB.REMOTE_BACKEND
        assert entry["platform"]["platform_id"].startswith("linux-x86_64")
        key = (entry["mix"], entry["world_index"])
        by_unit.setdefault(key, set()).add(
            (entry["fanout"]["job_id"], entry["platform"]["worker_id"])
        )
    assert all(len(v) == 1 for v in by_unit.values())  # one job and one worker per world
    calibration = records(intent_path, "calibration")
    assert all(e["platform"]["backend"] == RB.REMOTE_BACKEND for e in calibration.values())
    seg = R.read_json(output_of(intent_path) / "final" / "segments" / "look-0" / "remote.json")
    assert seg["planned_units"] == sorted(seg["units"], key=lambda k: seg["units"][k]["index"])
    assert len(seg["units"]) == 30 and seg["cause"] is None
    [receipt] = receipts(intent_path)
    assert receipt["action"] == "stop" and receipt["per_worker"] == [15, 15]
    shard = R.read_json(
        output_of(intent_path) / "final" / "segments" / "look-0" / "shard-0" / "started.json"
    )
    assert shard["gate"]["identity_check"]["state"] == "PASSED"
    for run in ledger_runs(factory).values():
        assert run["status"] == "finished"
        assert all(pod["deleted"] is not None for pod in run["pods"].values())
    jobs = [b for b in fake.bodies if b["input"]["op"] == "units"]
    for body in jobs:  # whole units only, at most one wave per job
        eps = [e for u in body["input"]["units"] for e in u["episodes"]]
        assert len(eps) <= body["input"]["slots"]
        for unit in body["input"]["units"]:
            assert len({(e["mix"], e["world_index"]) for e in unit["episodes"]}) == 1
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]
    assert report["schema_version"] == "sequential-strict-audit/v3"
    rules = {row["rule"] for row in report["checks"]}
    assert {
        "execution.platform_named",
        "identity.result_and_backend",
        "platform.per_world_single",
        "remote.segments_exact_units",
        "remote.spend_stop_is_invalid",
    } <= rules


def test_identity_mismatch_falls_back_to_the_mac_and_shows_no_data(tmp_path):
    spec = make_remote_spec(RemoteWorld(effect=400.0), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    target = "final-candidate-scripted-w"
    fake = FakeStrictRp(spec, perturb=lambda eid, nth: eid.startswith(target))
    factory = session_factory(tmp_path, fake)
    result = identity(intent_path, spec, factory)
    assert result["passes"] is False and result["backend_decided"] == RB.FALLBACK_BACKEND
    assert len(result["mismatched"]) == 3 and result["mismatched_fields"]
    assert all(f == ["mass_integral"] for f in result["mismatched_fields"].values())
    saved = json.loads((intent_path.parent / RB.IDENTITY_FILE).read_text())
    keys = set()

    def walk(value):
        if isinstance(value, dict):
            keys.update(value)
            for key, item in value.items():
                if key != "digests":
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)

    walk(saved)
    assert not keys & {"record", "mass_integral", "survival_fraction", "horizon"}  # no data
    for side in ("mac", "remote"):
        for digests in saved[side]["digests"].values():  # salted digests only
            leaves = [digests["digest"], *digests["fields"].values()]
            assert all(len(x) == 64 and int(x, 16) >= 0 for x in leaves)
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["provenance"]["executor"] == "in-process"
    assert closeout["execution"]["backend"] == RB.FALLBACK_BACKEND
    assert len(fake.created) == 1  # the gate never created an endpoint
    assert all(e["platform"]["backend"] == "in-process" for e in records(intent_path).values())
    assert not list((output_of(intent_path) / "final" / "segments").glob("*/remote.json"))
    report = A.run_audit(intent_path.parent)
    assert report["status"] == "PASS", report["failures"]


class SimulatedCrash(BaseException):
    pass


def test_abandoned_identity_check_falls_back_and_is_never_rerun(tmp_path):
    spec = make_remote_spec(RemoteWorld(effect=400.0), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    fake = FakeStrictRp(spec)
    factory = session_factory(tmp_path, fake)

    def crashing_mac(*args, **kwargs):
        raise SimulatedCrash("killed mid-check")

    with pytest.raises(SimulatedCrash):
        identity(intent_path, spec, factory, mac=crashing_mac)
    assert not fake.endpoints  # the session was torn down on the way out
    assert RB.identity_state(intent_path.parent)["state"] == "ABANDONED"
    with pytest.raises(R.StrictRunError, match="never re-run"):
        identity(intent_path, spec, factory)
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert closeout["execution"]["backend"] == RB.FALLBACK_BACKEND
    assert closeout["execution"]["identity_check"]["state"] == "ABANDONED"
    assert A.run_audit(intent_path.parent)["status"] == "PASS"


def test_run_refuses_without_an_identity_check_and_a_refused_check_stays_not_run(tmp_path):
    spec = make_remote_spec(RemoteWorld(effect=400.0), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    fake = FakeStrictRp(spec)
    with pytest.raises(R.StrictRunError, match="identity check NOT_RUN"):
        execute_remote(intent_path, spec, session_factory(tmp_path, fake))
    unseeded = session_factory(tmp_path, fake, seed=dict(SEEDED, ready=False, problems=["x"]))
    with pytest.raises(RB.SessionRefused, match="not ready"):
        identity(intent_path, spec, unseeded)
    assert RB.identity_state(intent_path.parent)["state"] == "NOT_RUN"
    assert not fake.created and not output_of(intent_path).exists()
    assert identity(intent_path, spec, session_factory(tmp_path, fake))["passes"] is True


def test_preflight_refusal_starts_nothing(tmp_path):
    spec, intent_path, fake, _ = passing_remote_run(tmp_path)
    fake.never_start = True
    factory = session_factory(tmp_path, fake, rpol_over={"startup_deadline_seconds": 0})
    with pytest.raises(RB.SessionRefused, match="capacity"):
        execute_remote(intent_path, spec, factory)
    assert not output_of(intent_path).exists() and not (tmp_path / "ledger.jsonl").exists()
    assert not fake.endpoints and len(fake.deleted) == len(fake.created) == 2
    assert R.run_status(intent_path.parent)["state"] == "NOT_STARTED"


# ---------------------------------------------------------------- failures inside the gate


def test_lost_job_redispatches_the_whole_unit(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    fake.fail_jobs_once = {"final|frozen|0"}
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    seg = R.read_json(output_of(intent_path) / "final" / "segments" / "look-0" / "remote.json")
    assert seg["lost"] and "final|frozen|0" in seg["lost"][0]["units"]
    assert seg["units"]["final|frozen|0"]["attempt"] == 2
    final = records(intent_path)
    assert final["final-candidate-frozen-w00000"]["fanout"]["attempt"] == 2
    assert A.run_audit(intent_path.parent)["status"] == "PASS"


def test_partial_unit_failure_redispatches_the_unit_and_verifies_duplicates(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    fake.worker_failure_once = {"final-candidate-mixed-w00002"}
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "DRY_RUN_PASS", why(closeout)
    assert fake.plays["final-incumbent-mixed-w00002"] == 2  # replayed whole, on one worker
    receipt = R.read_json(Path(closeout["execution"]["remote"]["run_dir"]) / "receipt.json")
    assert receipt["dispatch"]["duplicates_verified"] >= 1
    final = records(intent_path)
    assert (
        final["final-incumbent-mixed-w00002"]["fanout"]
        == final["final-candidate-mixed-w00002"]["fanout"]
    )
    assert A.run_audit(intent_path.parent)["status"] == "PASS"


def test_divergent_duplicate_stops_invalid_and_deletes_the_endpoint(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    fake.worker_failure_once = {"final-candidate-mixed-w00002"}
    fake.perturb = lambda eid, nth: eid == "final-incumbent-mixed-w00002" and nth == 2
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "INVALID_STOP" and "DIVERGENT" in closeout["failure"]
    assert not fake.endpoints
    assert "final-incumbent-mixed-w00002" not in records(intent_path)


@pytest.mark.parametrize(
    "knob, match",
    [
        ({"episode_error": {"final-candidate-frozen-w00001"}}, "episode_error"),
        ({"horizon_off": {"final-incumbent-scripted-w00001"}}, "horizon"),
        ({"binding_tamper": {"final-candidate-frozen-w00001"}}, "worker bindings differ"),
    ],
)
def test_bad_remote_results_stop_invalid(tmp_path, knob, match):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    for name, value in knob.items():
        setattr(fake, name, value)
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "INVALID_STOP" and match in closeout["failure"], why(closeout)
    assert not fake.endpoints


def test_unit_lost_too_often_stops_invalid(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)

    class AlwaysFails(set):
        def discard(self, item):  # never "once": this unit's job fails every time
            return None

    fake.fail_jobs_once = AlwaysFails({"final|frozen|0"})
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "INVALID_STOP" and "lost 3 times" in closeout["failure"]


def test_spend_cap_hard_stop_tears_down_and_is_invalid_stop(tmp_path):
    spec, intent_path, fake, _ = passing_remote_run(tmp_path)
    fake.drain = 5.0  # $5 per status poll: the $20 cap is reached in look 0
    factory = session_factory(tmp_path, fake, rpol_over={"balance_guard_seconds": 0})
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "INVALID_STOP", why(closeout)
    assert "spend cap reached" in closeout["failure"]
    assert "spend cap reached" in closeout["execution"]["remote"]["stop_reason"]
    assert not fake.endpoints
    report = A.run_audit(intent_path.parent)
    check = next(c for c in report["checks"] if c["rule"] == "remote.spend_stop_is_invalid")
    assert check["passes"] is True


@pytest.mark.parametrize("guard", ["power", "lid"])
def test_battery_or_lid_during_a_remote_segment_stops_invalid(tmp_path, guard):
    spec, intent_path, fake, _ = passing_remote_run(tmp_path)
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        return calls["n"] <= 2

    kw = {guard: flaky}
    factory = session_factory(tmp_path, fake, rpol_over={"power_poll_seconds": 0}, **kw)
    closeout = execute_remote(intent_path, spec, factory)
    assert closeout["outcome"] == "INVALID_STOP", why(closeout)
    assert ("on_battery" if guard == "power" else "lid_closed") in closeout["failure"]
    assert not fake.endpoints


# ---------------------------------------------------------------- audit


def test_audit_detects_platform_split_extra_units_and_identity_tamper(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    execute_remote(intent_path, spec, factory)
    output = output_of(intent_path)
    path = output / "final" / "records" / "final-candidate-frozen-w00000.json"
    entry = json.loads(path.read_text())
    entry["platform"]["worker_id"] = "elsewhere"
    path.write_text(json.dumps(entry))
    seg = output / "final" / "segments" / "look-0" / "remote.json"
    info = json.loads(seg.read_text())
    info["planned_units"].append("final|frozen|39")
    seg.write_text(json.dumps(info))
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert {"platform.per_world_single", "remote.segments_exact_units"} <= failed
    result = json.loads((intent_path.parent / RB.IDENTITY_FILE).read_text())
    result["passes"] = False
    (intent_path.parent / RB.IDENTITY_FILE).write_text(json.dumps(result))
    failed = {row["rule"] for row in A.run_audit(intent_path.parent)["failures"]}
    assert "identity.result_and_backend" in failed


def test_audit_provenance_requires_the_remote_executor_on_a_remote_run():
    production = {
        "dry_run": False,
        "allow_dirty": False,
        "source_closure": {"dirty": []},
        "slot_lock_root": A.SLOT_LOCK_ROOT,
        "ledger_path": A.LEDGER_PATH,
    }
    remote = {
        "provenance": {
            "executor": "runpod-serverless",
            "skew_runner": "subprocess_skew_runner",
            "audit_runner": "subprocess_audit_runner",
            "allow_dirty": False,
            "dry_run": False,
        },
        "execution": {
            "backend": "runpod-serverless",
            "transport": "rp.py",
            "remote_factory": "production",
        },
    }
    seg = {"executor": "runpod-serverless", "children": [{"command": ["runpod-serverless"]}]}
    assert A.provenance_problems(production, remote, [seg], None, None) == []
    local = {"children": [{"command": ["python", "sequential_runner.py", "worker"]}]}
    assert A.provenance_problems(production, remote, [local], None, None)
    fallback = dict(remote, execution={"backend": "local-mac-fallback"})
    assert A.provenance_problems(production, fallback, [local], None, None)  # executor name
    injected = dict(remote, execution=dict(remote["execution"], transport="injected:fake"))
    assert A.provenance_problems(production, injected, [seg], None, None)


# ---------------------------------------------------------------- worker side


def test_remote_worker_binds_intent_closure_and_spec(tmp_path):
    spec, intent_path, fake, factory = passing_remote_run(tmp_path)
    text = intent_path.read_text()
    sha = R.sha256_file(intent_path)
    intent = json.loads(text)
    unit = RB.sample_units(intent, spec)[0]
    payload = {"unit": unit["unit"], "episode": unit["episodes"][0], "row": unit["row"]}
    payload["intent_sha256"] = sha
    code, out = W.execute(text, sha, payload, tmp_path, spec=spec)
    assert code == W.EXIT_OK and out["checks"]["intent_sha256"] == sha
    assert out["checks"]["closure_digest"] == intent["source_closure"]["digest"]
    code, out = W.execute(text + " ", sha, payload, tmp_path, spec=spec)
    assert code == W.EXIT_BINDING_ERROR and "sha256" in out["error"]
    code, out = W.execute(text, sha, payload, tmp_path, spec=spec, repo_root=tmp_path)
    assert code == W.EXIT_BINDING_ERROR and "source closure differs" in out["error"]
    bare = R.StudySpec(**{**spec.__dict__, "remote_worker_setup": None})
    code, out = W.execute(text, sha, payload, tmp_path, spec=bare)
    assert code == W.EXIT_BINDING_ERROR and "remote_worker_setup" in out["error"]

    def boom(episode, row, context):
        raise ValueError("study code failed")

    failing = R.StudySpec(**{**spec.__dict__, "episode_runner": boom})
    code, out = W.execute(text, sha, payload, tmp_path, spec=failing)
    assert code == W.EXIT_EPISODE_ERROR and "study code failed" in out["error"]
    files = {"intent": tmp_path / "i.json", "payload": tmp_path / "p.json"}
    files["intent"].write_text(text)
    files["payload"].write_text(json.dumps(payload))
    out_path = tmp_path / "out.json"
    code = W.main(
        [
            "--intent",
            str(files["intent"]),
            "--payload",
            str(files["payload"]),
            "--ckpt-dir",
            str(tmp_path),
            "--out",
            str(out_path),
        ]
    )  # the test spec_ref does not resolve: a binding error, reported in the output
    assert code == W.EXIT_BINDING_ERROR and "binding" in json.loads(out_path.read_text())["error"]


STUB_WORKER = r"""
import json, os, sys
args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
payload = json.load(open(args["--payload"]))
eid = payload["episode"]["episode_id"]
mode = json.loads(os.environ.get("STUB_MODES", "{}")).get(eid, "ok")
if mode == "crash":
    os._exit(9)
out = {"schema": "sequential-strict-remote-episode/v1", "episode_id": eid,
       "omp": os.environ.get("OMP_NUM_THREADS"), "intent_ok": os.path.isfile(args["--intent"])}
if mode != "ok":
    out["error"] = mode
open(args["--out"], "x").write(json.dumps(out))
sys.exit({"ok": 0, "episode": 3, "binding": 4}[mode])
"""


@pytest.fixture
def handler_volume(tmp_path, monkeypatch):
    from tests.test_runpod_fanout import make_repo

    repo, commit = make_repo(tmp_path)
    root = tmp_path / "vol" / "rpf"
    cdir = root / "commits" / commit
    cdir.mkdir(parents=True)
    repo_sha = jobspec.git_archive(repo, commit, cdir / "repo.tar.gz")
    (cdir / "manifest.json").write_text(json.dumps({"commit": commit, "repo_sha256": repo_sha}))
    (root / "ckpt").mkdir()
    data = b"weights"
    ck = hashlib.sha256(data).hexdigest()
    (root / "ckpt" / f"{ck}.pth").write_bytes(data)
    rdir = root / "runtime" / "rt1"
    rdir.mkdir(parents=True)
    handler_sha = hashlib.sha256(Path(H.__file__).read_bytes()).hexdigest()
    (rdir / "READY.json").write_text(
        json.dumps({"runtime_id": "rt1", "handler_sha256": handler_sha})
    )
    monkeypatch.setattr(H, "ROOT", root)
    monkeypatch.setattr(H, "RUNTIME_DIR", rdir)
    monkeypatch.setattr(H, "LOCAL", tmp_path / "local")
    H._VERIFIED.clear()
    (tmp_path / "stub.py").write_text(STUB_WORKER)
    monkeypatch.setenv(
        "RPF_TEST_STRICT_WORKER_EXEC_JSON", json.dumps([sys.executable, str(tmp_path / "stub.py")])
    )
    text = json.dumps({"intent": "text"})
    base = {
        "runtime_id": "rt1",
        "commit": commit,
        "repo_sha256": repo_sha,
        "checkpoints": [ck],
        "job_id": "sg-0123456789ab",
        "slots": 2,
        "timeout_seconds": 30,
        "deadline_epoch": time.time() + 600,
        "numerics_env": {"MKL_CBWR": "COMPATIBLE"},
        "intent_text": text,
        "intent_sha256": hashlib.sha256(text.encode()).hexdigest(),
    }
    return base


def handler_unit(mix, world, seed, arms=("incumbent", "candidate")):
    episodes = [
        {
            "episode_id": f"final-{arm}-{mix}-w{world:05d}",
            "phase": "final",
            "arm": arm,
            "mix": mix,
            "world_index": world,
            "world_seed": seed,
        }
        for arm in arms
    ]
    row = {"mix": mix, "world_index": world, "world_seed": seed}
    return {"unit": f"final|{mix}|{world}", "row": row, "episodes": episodes}


def test_strict_handler_runs_whole_units_and_classifies_results(handler_volume, monkeypatch):
    probe = H.handle(dict(handler_volume, op="probe"))
    assert probe["ok"] is True and probe["worker"]["runtime_id"] == "rt1"
    modes = {
        "final-candidate-frozen-w00001": "episode",
        "final-incumbent-mixed-w00002": "binding",
        "final-candidate-mixed-w00002": "crash",
    }
    monkeypatch.setenv("STUB_MODES", json.dumps(modes))
    units = [
        handler_unit("frozen", 0, 11),
        handler_unit("frozen", 1, 12),
        handler_unit("mixed", 2, 13),
    ]
    out = H.handle(dict(handler_volume, op="units", units=units))
    assert out["ok"] is True
    kinds = {r["episode_id"]: r["kind"] for u in out["units"] for r in u["results"]}
    assert kinds == {
        "final-incumbent-frozen-w00000": "ok",
        "final-candidate-frozen-w00000": "ok",
        "final-incumbent-frozen-w00001": "ok",
        "final-candidate-frozen-w00001": "episode_error",
        "final-incumbent-mixed-w00002": "binding_error",
        "final-candidate-mixed-w00002": "worker_failure",
    }
    ok = next(r for r in out["units"][0]["results"] if r["kind"] == "ok")
    assert hashlib.sha256(ok["output"].encode()).hexdigest() == ok["sha256"]
    assert json.loads(ok["output"])["omp"] == "2" and json.loads(ok["output"])["intent_ok"]
    assert [u["unit"] for u in out["units"]] == [u["unit"] for u in units]


def test_strict_handler_refuses_split_units_bad_ids_and_foreign_intents(handler_volume):
    split = handler_unit("frozen", 0, 11)
    split["episodes"][1] = dict(
        split["episodes"][1], world_index=1, episode_id="final-candidate-frozen-w00001"
    )
    out = H.handle(dict(handler_volume, op="units", units=[split]))
    assert "spans several worlds" in out["refused"]
    bad = handler_unit("frozen", 0, 11)
    bad["episodes"][0]["episode_id"] = "../../etc/passwd"
    assert (
        "bad or duplicate episode id"
        in H.handle(dict(handler_volume, op="units", units=[bad]))["refused"]
    )
    foreign = dict(handler_volume, op="units", units=[handler_unit("frozen", 0, 11)])
    foreign["intent_sha256"] = "f" * 64
    assert "intent text" in H.handle(foreign)["refused"]
    wrong_runtime = dict(handler_volume, op="probe", runtime_id="other")
    assert "not this worker's" in H.handle(wrong_runtime)["refused"]
    assert "unknown op" in H.handle(dict(handler_volume, op="shell"))["refused"]


# ---------------------------------------------------------------- small pieces


def test_pack_units_keeps_units_whole_and_runs_lost_batch_units_alone():
    pending = [("a", 2), ("b", 2), ("c", 2), ("d", 2), ("e", 1)]
    assert RB.pack_units(pending, 8) == ["a", "b", "c", "d"]
    assert RB.pack_units(pending, 5) == ["a", "b", "e"]
    assert RB.pack_units(pending, 1) == ["a"]  # a unit bigger than a wave still runs whole
    assert RB.pack_units(pending, 8, ["a"]) == ["a"]
    assert RB.pack_units(pending, 8, ["b"]) == ["a", "c", "d", "e"]


def test_clamshell_parser_fails_closed():
    assert R.parse_clamshell('| "AppleClamshellState" = No') is True
    assert R.parse_clamshell('| "AppleClamshellState" = Yes') is False
    assert R.parse_clamshell("") is None and R.parse_clamshell('"AppleClamshellState" = ?') is None


def test_run_status_of_a_remote_run_uses_the_orchestrator_lock(tmp_path):
    spec = make_remote_spec(RemoteWorld(), tmp_path)
    intent_path = prepare_remote(tmp_path, spec)
    lock = RB.acquire_orchestrator_lock(intent_path.parent)
    try:
        output = output_of(intent_path)
        output.mkdir()
        R.write_once(
            output / "started.json",
            {"pid": R.os.getpid(), "execution": {"liveness_lock": lock.name}},
        )
        assert RB._lock_held_elsewhere(Path(lock.name)) is True
        assert R.run_status(intent_path.parent)["state"] == "IN_PROGRESS"
        with pytest.raises(R.StrictRunError, match="orchestrator lock"):
            RB.acquire_orchestrator_lock(intent_path.parent)
    finally:
        RB.release_lock(lock)
    assert R.run_status(intent_path.parent)["state"] == "ABANDONED"


def test_remote_plan_report_and_example_spec_hooks(tmp_path):
    from research.sequential_strict_template import example_spec

    spec = make_remote_spec(RemoteWorld(), tmp_path)
    params = R.study_plan_parameters(spec, n_max=N_MAX, mde=30.0)
    report = RB.remote_plan_report(spec, params, write_config(tmp_path), 4, seeding=SEEDED)
    assert report["dry_run"] is True and report["admissible"] is True
    assert report["plan"]["choice"]["speedup"] >= 5.0 and report["owner_steps"] == []
    assert any("not ratified" in b for b in report["blockers"])
    unseeded = dict(SEEDED, ready=False, needs="runtime")
    report = RB.remote_plan_report(spec, params, write_config(tmp_path), 4, seeding=unseeded)
    assert report["owner_steps"][0].startswith("serverless.py seed --handler strict --commit ")
    assert report["plan"]["seeding"]["minutes"] == RB.load_remote_policy()["seed_minutes_runtime"]
    assert "account" not in report  # no RunPod call unless --account
    with pytest.raises(NotImplementedError):
        example_spec.SPEC.remote_checkpoints()
    with pytest.raises(NotImplementedError):
        example_spec.SPEC.remote_worker_setup({}, tmp_path)


def test_v1_run_and_v2_intent_carry_nothing_of_v3_and_v3_wraps_paired_bands(tmp_path):
    from tests.test_sequential_strict_template import execute
    from tests.test_sequential_strict_template_paired import (
        DEV_DELTA_NI,
        RATIFIED,
        make_paired_spec,
        prepare_paired,
    )

    first = tmp_path / "v1"
    first.mkdir()
    spec = make_spec(FakeWorld(effect=400.0), first)
    intent_path = R.prepare(R.build_intent(spec, **intent_kwargs(first)))
    closeout = execute(intent_path, spec)
    assert closeout["outcome"] == "DRY_RUN_PASS" and "execution" not in closeout
    started = R.read_json(output_of(intent_path) / "started.json")
    assert "execution" not in started and "execution" not in started["provenance"]
    assert all("platform" not in e and "fanout" not in e for e in records(intent_path).values())
    shard = output_of(intent_path) / "calibration" / "segments" / "look-0" / "shard-0"
    assert R.read_json(shard / "started.json")["gate"] == {}
    second = tmp_path / "v2"
    second.mkdir()
    paired = make_paired_spec(FakeWorld(effect=400.0), second)
    v2 = R.read_json(prepare_paired(second, paired))
    assert v2["template_version"] == R.TEMPLATE_VERSION_PAIRED and "execution" not in v2
    third = tmp_path / "v3"
    third.mkdir()
    paired = make_paired_spec(RemoteWorld(effect=400.0), third)
    Path(paired.protocol_path).write_text("Execution platform: runpod-serverless\n")
    paired = R.StudySpec(
        **{
            **paired.__dict__,
            "remote_worker_setup": lambda intent, ckpt_dir: None,
            "remote_checkpoints": lambda: [HERO_SHA256],
        }
    )
    v3 = R.build_intent(
        paired,
        **intent_kwargs(third, paired_band=RATIFIED, development_delta_ni=DEV_DELTA_NI),
        remote_config=write_config(third),
        remote_seeding=SEEDED,
    )
    assert v3["template_version"] == R.TEMPLATE_VERSION_REMOTE
    assert v3["execution"]["band_template_version"] == R.TEMPLATE_VERSION_PAIRED
    assert v3["paired_band_check"]["passes"] is True and v3["plan"]["band_policy"] == (
        "paired_ni_at_stop"
    )
    R.validate_intent(R.read_json(R.prepare(v3)), paired)
