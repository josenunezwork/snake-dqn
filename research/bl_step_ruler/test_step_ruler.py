"""Offline tests for the Step -1 world-step ruler (no checkpoints, no Colab)."""

from __future__ import annotations

import json
import os
import subprocess
import time
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.bl_labels.exports import publish_json
from research.bl_step_ruler import analyze, job, pack, ticks

COMMIT = "a" * 40


# ------------------------------------------------------------------ fakes


class FakeDriver:
    def step(self, forced=None, packet=None):
        return {"forced": forced}

    def snapshot(self):
        return b"xyz"

    @classmethod
    def restore(cls, payload):
        return cls()


def _fake_factory():
    module = SimpleNamespace()

    def restore_root(snapshots, tape, root, checkpoint, *, restore):
        driver = restore(b"s")
        for _ in range(3):
            driver.step(1)
        return driver

    def branch(snapshot, action, depth, *, restore, pulse=lambda: None):
        driver = restore(snapshot)
        for _ in range(depth):
            driver.step(action)
        driver.snapshot()
        return {"frames_executed": depth}

    module._restore_root = restore_root
    module._branch = branch
    return module


def test_ticks_attribute_phases_and_restore_originals():
    module = _fake_factory()
    originals = (
        FakeDriver.__dict__["step"],
        FakeDriver.__dict__["restore"],
        module._branch,
        module._restore_root,
    )
    counter = ticks.TickCounter()
    with ticks.instrument(counter, driver_cls=FakeDriver, factory_module=module):
        driver = FakeDriver()
        driver.step()
        driver.step()
        driver.snapshot()
        restore = FakeDriver.restore
        replayed = module._restore_root({}, {}, None, "c", restore=restore)
        replayed.snapshot()
        module._branch(b"p", 2, 5, restore=restore)
        FakeDriver.restore(b"death")
    value = counter.as_dict()
    assert value["ticks"] == {"play": 2, "replay": 3, "branch": 5}
    assert value["world_steps"] == 10
    assert value["restores"] == {"play": 1, "replay": 1, "branch": 1}
    assert value["snapshots"] == {"play": 2, "replay": 0, "branch": 1}
    assert value["calls"] == {"replay": 1, "branch": 1}
    assert counter.stack == ["play"]
    assert (
        FakeDriver.__dict__["step"],
        FakeDriver.__dict__["restore"],
        module._branch,
        module._restore_root,
    ) == originals


def test_ticks_restore_originals_after_exception():
    module = _fake_factory()
    step = FakeDriver.__dict__["step"]
    with pytest.raises(RuntimeError):
        with ticks.instrument(ticks.TickCounter(), driver_cls=FakeDriver, factory_module=module):
            raise RuntimeError("boom")
    assert FakeDriver.__dict__["step"] is step


def test_real_targets_are_the_collector_seams():
    from research.bl_labels import factory
    from research.bl_served_search.b2_engine import WatchDriver

    assert isinstance(WatchDriver.__dict__["restore"], classmethod)
    counter = ticks.TickCounter()
    step = WatchDriver.__dict__["step"]
    with ticks.instrument(counter):
        assert WatchDriver.__dict__["step"] is not step
        assert factory._branch.__wrapped__ is not None
    assert WatchDriver.__dict__["step"] is step
    assert not hasattr(factory._branch, "__wrapped__")


# ------------------------------------------------------------------ ledger


def _control(ns: str = "ns-dev") -> dict:
    return {"schema": job.CONTROL_SCHEMA, "namespace": ns, "allocation": pack.allocation(ns)}


def test_ledger_refuses_foreign_namespace_allocation_and_duplicates(tmp_path):
    control = _control()
    ledger = tmp_path / "ledger.json"
    publish_json(ledger, job.empty_ledger("ns-dev", control["allocation"]))
    assert job.load_ledger(ledger, "ns-dev", control["allocation"]) == set()
    with pytest.raises(ValueError):
        job.load_ledger(ledger, "other", control["allocation"])
    with pytest.raises(ValueError):
        job.load_ledger(ledger, "ns-dev", control["allocation"][:-1])
    bad = tmp_path / "dup.json"
    row = {"id": "watch6-000", "path": "units/watch6-000.json", "sha256": "0" * 64, "size": 1}
    publish_json(bad, {**job.empty_ledger("ns-dev", control["allocation"]), "units": [row, row]})
    with pytest.raises(ValueError):
        job.load_ledger(bad, "ns-dev", control["allocation"])


def test_build_ledger_unions_verified_units_and_refuses_foreign(tmp_path):
    control = _control()
    for attempt, unit in (("a1", "watch6-000"), ("a2", "gate-mixed-001"), ("a3", "watch6-000")):
        publish_json(tmp_path / attempt / "units" / f"{unit}.json", {"unit": unit})
    out = tmp_path / "ledger" / "verified-ledger.json"
    value = job.build_ledger(control, [tmp_path / "a1", tmp_path / "a2", tmp_path / "a3"], out)
    assert [row["id"] for row in value["units"]] == ["gate-mixed-001", "watch6-000"]
    assert job.load_ledger(out, "ns-dev", control["allocation"]) == {"gate-mixed-001", "watch6-000"}
    publish_json(tmp_path / "a4" / "units" / "intruder-000.json", {})
    with pytest.raises(ValueError):
        job.build_ledger(control, [tmp_path / "a4"], tmp_path / "l2" / "x.json")
    # Different bytes for the same unit are refused.
    publish_json(tmp_path / "a5" / "units" / "watch6-000.json", {"unit": "other"})
    with pytest.raises(ValueError):
        job.build_ledger(control, [tmp_path / "a1", tmp_path / "a5"], tmp_path / "l3" / "x.json")


def test_collect_exports_counters_only(tmp_path, monkeypatch):
    from research.bl_labels import factory

    def fake_collect(allocation, b8, frp3, namespace):
        allocation["publish_life"]({"rows": [{"died": [True]}]})
        return {
            "status": "COMPLETE",
            "rows": [{"a_star": 1}],
            "counters": {
                "deaths": 2,
                "roots": 9,
                "digest_checks": 9,
                "bare_rerolls": 1,
                "decisive_rows": 3,
                "rescue_bearing_lives": 1,
                "h1440": 1,
                "reversals": 0,
                "bare_audit_mismatches": 0,
            },
            "timing_seconds": {"play": 1.0},
            "provenance": {"elapsed_seconds": 2.5},
        }

    monkeypatch.setattr(factory, "collect_unit", fake_collect)
    saved = {key: os.environ.get(key) for key in job.THREAD_ENV}
    try:
        item = pack.allocation("ns-dev")[0]
        status = job._collect((item, "b8", "frp3", "ns-dev", str(tmp_path), None))
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    assert status["status"] == "COMPLETE"
    path = tmp_path / "units" / f"{item['allocation_id']}.json"
    job.verify_unit(path)
    body = json.loads(path.read_text())
    assert body["work_counters"] == {"deaths": 2, "roots": 9, "digest_checks": 9, "bare_rerolls": 1}
    assert "rows" not in body and not set(job.SEALED_COUNTERS) & set(body)
    assert body["sealed_counters_dropped"] == list(job.SEALED_COUNTERS)
    text = path.read_text()
    assert '"a_star"' not in text and '"died"' not in text
    assert list(tmp_path.glob("**/*life*")) == []


# -------------------------------------------------------------------- pack


def test_allocation_is_deterministic_dev_and_watch6_first():
    ns = pack.namespace(COMMIT)
    assert ns == f"bl-step-ruler-{COMMIT}-dev"
    rows = pack.allocation(ns)
    assert rows == pack.allocation(ns)
    assert len(rows) == 12 and [r["stream"] for r in rows[:4]] == ["watch6"] * 4
    assert {r["stream"] for r in rows} == set(analyze.STREAMS)
    assert len({r["seed"] for r in rows} | {pack.self_check_seed(ns)}) == 13
    assert all(r["horizon"] == (14400 if r["stream"] == "watch6" else 5000) for r in rows)
    with pytest.raises(ValueError):
        pack.namespace("ABC")


def test_envelopes_double_once_and_respect_guard():
    probe, main = pack.stage_envelope("transport_probe"), pack.stage_envelope("main")
    assert probe["effective_minutes"] == 20 and probe["effective_units"] == 0.04
    assert main["effective_minutes"] == 260 and main["effective_units"] == 0.36
    assert pack.series_effective_units() <= pack.SERIES_GUARD_UNITS
    manifest = {"requirements": {}, "budget": {}, "env": {}}
    pack.apply_stage(manifest, "transport_probe")
    assert manifest["timeout_minutes"] <= 30  # family canary bound
    assert manifest["env"]["F0_SESSION_SECONDS"] == "1200"


def test_probe_control_first_parent_per_stream_horizon_three(tmp_path):
    control = _control()
    publish_json(tmp_path / "control.json", control)
    publish_json(tmp_path / "ledger.json", job.empty_ledger("ns-dev", control["allocation"]))
    pack.probe_control(
        tmp_path / "control.json",
        tmp_path / "ledger.json",
        tmp_path / "p" / "c.json",
        tmp_path / "p" / "l.json",
    )
    probe = json.loads((tmp_path / "p" / "c.json").read_text())
    assert probe["namespace"] == "ns-dev-transport-probe"
    assert [r["allocation_id"] for r in probe["allocation"]] == [
        "watch6-000",
        "gate-frozen-000",
        "gate-mixed-000",
        "gate-scripted-000",
    ]
    assert {r["horizon"] for r in probe["allocation"]} == {3}
    job.load_ledger(tmp_path / "p" / "l.json", probe["namespace"], probe["allocation"])


def test_probe_manifest_is_short_canary_and_main_is_resumable():
    inputs = [{"path": "/x", "dest": "metadata.tar", "sha256": "0" * 64}]
    main = {
        "job_id": "m",
        "commit": COMMIT,
        "script": pack.SCRIPT,
        "env": {},
        "inputs": inputs,
        "requirements": {"gpu": "cpu"},
        "budget": {},
        "resume": dict(pack.RESUME),
        "canary": {"template": True},
        "step_ruler": {},
    }
    pack.apply_stage(main, "main")
    probe = pack.make_probe(main, Path("/tmp/out-probe"), pack.namespace(COMMIT))
    assert "resume" not in probe and probe["resume_exemption"]["kind"] == "short-canary"
    assert probe["env"]["SNAKE_COLAB_CANARY_PACKAGE"] == "research/bl_step_ruler"
    assert probe["canary"]["family"] == pack.SCRIPT
    assert probe["timeout_minutes"] == 20
    assert main["resume"]["max_outcome_blind_retries"] == 2


# ----------------------------------------------------------------- analyze


def _triplet(path: Path, value: dict) -> None:
    publish_json(path, value)


def test_fleet_terminal_and_live_attempts(tmp_path):
    state = tmp_path / "state"
    out = tmp_path / "out" / "lives"
    out.mkdir(parents=True)
    attest = {"platform": "darwin-arm64"}
    _triplet(
        state / "attempt-a.start.json",
        {
            "source_commit": analyze.OLD_COLLECTOR,
            "platform_attestation": attest,
            "workers": 6,
            "output": str(tmp_path / "old"),
            "started_epoch": 0,
        },
    )
    _triplet(
        state / "attempt-a.terminal.json",
        {
            "workers": 6,
            "admission_workers_final": 6,
            "operational_width_fallback": None,
            "collector_source_commit": analyze.OLD_COLLECTOR,
            "elapsed_seconds": 3600.0,
            "completed_parents": [
                "d-mac-watch6-001",
                "d-mac-gate-frozen-002",
                "d-mac-gate-frozen-003",
            ],
        },
    )
    started = time.time() - 100
    _triplet(
        state / "attempt-b.start.json",
        {
            "source_commit": analyze.OLD_COLLECTOR,
            "platform_attestation": attest,
            "workers": 6,
            "output": str(tmp_path / "out"),
            "started_epoch": started,
        },
    )
    for index, offset in enumerate((10, 50)):
        marker = out / f"d-mac-gate-mixed-00{index}--session.json.complete"
        marker.write_text("complete\n")
        os.utime(marker, (started + offset, started + offset))
    stale = out / "d-mac-gate-mixed-009--session.json.complete"
    stale.write_text("complete\n")
    os.utime(stale, (started - 500, started - 500))
    _triplet(
        state / "attempt-c.start.json",
        {"source_commit": "f" * 40, "platform_attestation": attest, "workers": 6},
    )
    attempts = analyze.fleet([state])
    by = {Path(a["start"]).name: a for a in attempts}
    assert by["attempt-a.start.json"]["completed_by_stream"]["gate-frozen"] == 2
    live = by["attempt-b.start.json"]
    assert live["kind"] == "live" and live["completed_by_stream"]["gate-mixed"] == 2
    assert abs(live["elapsed_seconds"] - 50) < 1e-6
    assert by["attempt-c.start.json"]["eligible"] is False
    k = {"watch6": 1000.0, "gate-frozen": 100.0, "gate-mixed": 10.0, "gate-scripted": 1.0}
    rate = analyze.fleet_rate(attempts, k, 6)
    assert rate["world_steps_per_second"] == pytest.approx((1000 + 200 + 20) / 3650)
    assert analyze.fleet_rate(attempts, k, 12)["world_steps_per_second"] is None


def test_fleet_refuses_width_change_and_non_old_collector(tmp_path):
    state = tmp_path / "s"
    attest = {"platform": "darwin-arm64"}
    for name, terminal in (
        (
            "w",
            {
                "workers": 12,
                "operational_width_fallback": "W6",
                "elapsed_seconds": 1.0,
                "completed_parents": ["x-watch6-1"],
            },
        ),
        (
            "c",
            {
                "workers": 12,
                "collector_source_commit": "e" * 40,
                "elapsed_seconds": 1.0,
                "completed_parents": ["x-watch6-1"],
            },
        ),
    ):
        _triplet(
            state / f"attempt-{name}.start.json",
            {"source_commit": analyze.OLD_COLLECTOR, "platform_attestation": attest, "workers": 12},
        )
        _triplet(state / f"attempt-{name}.terminal.json", terminal)
    assert [a["eligible"] for a in analyze.fleet([state])] == [False, False]


def _dev(k: dict[str, float], complete: int = 12, x86: float = 100.0) -> dict:
    return {
        "complete_units": complete,
        "per_stream": {s: {"mean_world_steps_per_parent": v} for s, v in k.items()},
        "x86_worker_world_steps_per_second": x86,
    }


def _attempts(width: int = 6, elapsed: float = 3600.0) -> list[dict]:
    return [
        {
            "eligible": True,
            "workers": width,
            "elapsed_seconds": elapsed,
            "completed_by_stream": {
                "watch6": 1,
                "gate-frozen": 0,
                "gate-mixed": 0,
                "gate-scripted": 0,
            },
        }
    ]


K = {"watch6": 3600.0 * 1000, "gate-frozen": 1.0, "gate-mixed": 1.0, "gate-scripted": 1.0}


def test_decide_close_and_admit_boundaries():
    # Fleet = 1000 steps/s. Best GPU cell 7000 -> optimistic 4.55, pessimistic 2.45.
    gpu = {
        "cells": [
            {"roster": "scripted", "E": 1024, "rate": 7000.0},
            {"roster": "frozen", "E": 1024, "rate": 6000.0},
        ]
    }
    admitted = analyze.decide(_dev(K), _attempts(), gpu)
    assert admitted["verdict"] == "S0_ADMITTED"
    assert admitted["ratios_primary"]["optimistic"] == pytest.approx(4.55)
    assert admitted["pessimistic"] == "PESSIMISTIC_BELOW_4"
    assert admitted["gpu_best_cell"]["rate"] == 7000.0
    closed = analyze.decide(
        _dev(K), _attempts(), {"cells": [{"roster": "frozen", "E": 1024, "rate": 6000.0}]}
    )
    assert closed["verdict"] == "CLOSE_NO_BOUND"  # 0.65*6000/1000 = 3.9
    proxy = admitted["sensitivity"]["mac_w6_plus_3_l4_proxy"]["optimistic"]
    assert proxy == pytest.approx(0.65 * 7000 / (1000 + 3 * 12 * 100))


def test_decide_incomplete_without_minimums():
    gpu = {"cells": [{"roster": "frozen", "E": 1024, "rate": 6000.0}]}
    short = analyze.decide(_dev(K, complete=5), _attempts(), gpu)
    assert short["verdict"] == "INCOMPLETE"
    missing = dict(K)
    del missing["gate-mixed"]
    assert analyze.decide(_dev(missing), _attempts(), gpu)["verdict"] == "INCOMPLETE"
    assert analyze.decide(_dev(K), _attempts(width=12), gpu)["verdict"] == "INCOMPLETE"
    assert analyze.decide(_dev(K), _attempts(), {"cells": []})["verdict"] == "INCOMPLETE"


def test_gpu_cells_r2_hash_bound_and_r6_rules(tmp_path):
    with pytest.raises(ValueError):
        bogus = tmp_path / "R2.json"
        bogus.write_text("{}")
        analyze.gpu_cells(bogus, tmp_path / "absent.json")
    if not analyze.R2_ANALYSIS.is_file():
        pytest.skip("R2 artifact not on this host")
    base = analyze.gpu_cells(analyze.R2_ANALYSIS, tmp_path / "absent.json")
    assert {c["E"] for c in base["cells"]} == {1024}
    assert base["sources"]["R6"]["used"] is False
    r6 = tmp_path / "R6.json"
    publish_json(
        r6,
        {
            "verdict": "DROP_E4096",
            "mode": "main",
            "cells": {
                "frozen": {
                    "2048": {"median_rate": 9000.0, "exact": True},
                    "4096": {"median_rate": 99999.0, "exact": False},
                    "8192": {"median_rate": 50000.0, "exact": True},
                }
            },
        },
    )
    value = analyze.gpu_cells(analyze.R2_ANALYSIS, r6)
    r6_cells = [c for c in value["cells"] if c["source"] == "R6"]
    assert r6_cells == [{"source": "R6", "roster": "frozen", "E": 2048, "rate": 9000.0}]
    invalid = tmp_path / "R6b.json"
    publish_json(invalid, {"verdict": "INVALID", "mode": "main", "cells": {}})
    assert analyze.gpu_cells(analyze.R2_ANALYSIS, invalid)["sources"]["R6"]["used"] is False


def test_stream_parse_and_unit_schema_constants():
    assert analyze.stream_of("d1024-mac-gate-scripted-00592") == "gate-scripted"
    assert analyze.stream_of("d256-mac-watch6-039") == "watch6"
    with pytest.raises(ValueError):
        analyze.stream_of("weird")
    assert set(job.SEALED_COUNTERS).isdisjoint(job.WORK_COUNTERS)


def test_prepare_attempt_rules_refuse_before_any_side_effect(tmp_path):
    args = (COMMIT, tmp_path / "pack", tmp_path / "out", tmp_path / "b8", tmp_path / "frp3")
    with pytest.raises(ValueError):
        pack.prepare(*args, attempt=2)
    with pytest.raises(ValueError):
        pack.prepare(*args, ledger=tmp_path / "ledger.json", attempt=1)
    with pytest.raises(ValueError):
        pack.prepare(*args, ledger=tmp_path / "ledger.json", attempt=4)
    assert not (tmp_path / "pack").exists()


def test_cpu_default_manifest_matches_original_bytes(monkeypatch):
    """Freeze the old pack implementation independently, including defaults."""
    old = types.ModuleType("research.bl_step_ruler._original_pack")
    old.__file__ = str(pack.ROOT / "research/bl_step_ruler/pack.py")
    old.__package__ = "research.bl_step_ruler"
    source = subprocess.check_output(
        ["git", "show", f"{pack.RULER_PIN}:research/bl_step_ruler/pack.py"], cwd=pack.ROOT
    )
    exec(compile(source, old.__file__, "exec"), old.__dict__)
    # Only the package digest necessarily changes when pack.py changes.
    monkeypatch.setattr(job, "source_sha256", lambda root=None: "f" * 64)
    args = (COMMIT, Path("/tmp/ruler-out"), [], "0" * 64, pack.namespace(COMMIT))
    before = old.make_main(*args)
    after = pack.make_main(*args)
    assert json.dumps(before, sort_keys=True) == json.dumps(after, sort_keys=True)
    before_probe = old.make_probe(before, Path("/tmp/ruler-probe"), args[-1])
    after_probe = pack.make_probe(after, Path("/tmp/ruler-probe"), args[-1])
    assert json.dumps(before_probe, sort_keys=True) == json.dumps(after_probe, sort_keys=True)
    assert pack.completion_options(1, "cpu", None, None) is False


def test_l4_completion_and_same_family_probe():
    main = pack.make_main(COMMIT, Path("/tmp/ruler-out"), [], "0" * 64, "ns-dev")
    pack.apply_completion(main, "l4", 260, 12)
    assert main["requirements"] == {
        "gpu": "L4",
        "allowed_gpus": ["L4"],
        "min_vcpu": 12,
        "est_minutes": 130,
    }
    assert main["timeout_minutes"] == 260
    assert main["env"]["F0_SESSION_SECONDS"] == "15600"
    assert main["env"]["SNAKE_DQN_DEVICE"] == "cpu"
    assert "STEP_RULER_WORKERS" not in main["env"]  # job defaults to nproc
    assert all(main["env"][key] == "1" for key in job.THREAD_ENV)
    assert main["budget"] == {
        "unit_cap": 12,
        "reserve_units": 2.4,
        "rate_units_per_hour": 2.0,
        "other_assignments": {},
    }
    assert main["step_ruler"]["completion_ruling_sha256"] == pack.COMPLETION_RULING_SHA256
    assert main["step_ruler"]["completion_envelope"]["scope"] == "OWN_ATTEMPT_NOT_SHARED"
    probe = pack.make_probe(main, Path("/tmp/ruler-probe"), "ns-dev")
    assert probe["timeout_minutes"] == 20
    assert probe["requirements"]["est_minutes"] == 10
    assert probe["requirements"]["gpu"] == "L4"
    assert probe["budget"]["unit_cap"] == pytest.approx(2 / 3)
    assert probe["inputs"] == main["inputs"]
    assert probe["resume_exemption"]["timeout_minutes"] == 20
    assert main["timeout_minutes"] == 260  # independent probe envelope


@pytest.mark.parametrize(
    "attempt,runtime,timeout,guard",
    [
        (1, "l4", None, None),
        (2, "cuda", 260, 12),
        (2, "l4", 261, 12),
        (2, "l4", float("nan"), 12),
        (2, "l4", 260, float("inf")),
        (2, "l4", 260, 0),
        (2, "l4", 260, 13),
        (2, "l4", 260, 8),
        (2, "l4", 10, 12),
    ],
)
def test_bad_completion_options_refuse_before_artifacts(tmp_path, attempt, runtime, timeout, guard):
    with pytest.raises(ValueError):
        pack.prepare(
            COMMIT,
            tmp_path / "pack",
            tmp_path / "out",
            Path("b8"),
            Path("frp3"),
            attempt=attempt,
            ledger=Path("ledger") if attempt > 1 else None,
            runtime_type=runtime,
            timeout_minutes=timeout,
            unit_guard=guard,
        )
    assert not (tmp_path / "pack").exists()


def test_completion_probe_uses_uploaded_empty_ledger(tmp_path):
    control = _control()
    control["completion_ruling_sha256"] = pack.COMPLETION_RULING_SHA256
    control["runtime_type"] = "l4"
    publish_json(tmp_path / "control.json", control)
    ledger = job.empty_ledger("ns-dev", control["allocation"])
    row = {"id": "watch6-000", "path": "units/watch6-000.json", "sha256": "0" * 64, "size": 1}
    publish_json(tmp_path / "verified-ledger.json", {**ledger, "units": [row]})
    publish_json(tmp_path / "probe-ledger.json", ledger)
    pack.probe_control(
        tmp_path / "control.json",
        tmp_path / "verified-ledger.json",
        tmp_path / "probe/control.json",
        tmp_path / "probe/ledger.json",
    )
    derived = json.loads((tmp_path / "probe/control.json").read_text())
    assert len(derived["allocation"]) == 4
    assert derived["runtime_type"] == "l4"
    assert all(row["horizon"] == 3 for row in derived["allocation"])
    assert (
        job.load_ledger(tmp_path / "probe/ledger.json", derived["namespace"], derived["allocation"])
        == set()
    )


def test_collector_closure_pin_and_drift_refusal(tmp_path, monkeypatch):
    record = pack.check_collector_closure()
    assert record["status"] == "PASS" and record["files"] >= 200
    from tools.ops import import_preflight

    path = "research/bl_step_ruler/ticks.py"
    target = tmp_path / path
    target.parent.mkdir(parents=True)
    target.write_text("# drift\n")
    original_root = pack.ROOT
    monkeypatch.setattr(pack, "ROOT", tmp_path)
    monkeypatch.setattr(import_preflight, "import_closure", lambda *args: (["ticks"], []))
    monkeypatch.setattr(import_preflight, "module_path", lambda *args: target)
    actual_check_output = subprocess.check_output
    monkeypatch.setattr(
        subprocess,
        "check_output",
        lambda argv, **kwargs: actual_check_output(argv, **{**kwargs, "cwd": original_root}),
    )
    with pytest.raises(ValueError, match="closure differs"):
        pack.check_collector_closure()


def test_platform_changed_rates_are_descriptive_counts_still_pool(tmp_path):
    ns = pack.namespace(pack.RULER_PIN)
    control = _control(ns)
    control.update(runtime_type="l4", runtime_commit=COMMIT)
    exports = tmp_path / "exports"
    for index, pin in enumerate((pack.RULER_PIN, COMMIT)):
        item = control["allocation"][index]
        publish_json(
            exports / "units" / f"{item['allocation_id']}.json",
            {
                "schema": job.UNIT_SCHEMA,
                "namespace": ns,
                **item,
                "status": "COMPLETE",
                "world_steps": 100 * (index + 1),
                "tick_counters": {
                    "ticks": {"play": 100 * (index + 1), "replay": 0, "branch": 0},
                    "restores": {"play": 0, "replay": 0, "branch": 0},
                },
                "elapsed_seconds": 10,
                "work_counters": {"roots": 0},
                "provenance": {"platform": "x86", "code_sha": pin},
            },
        )
    ledger = tmp_path / "verified-ledger.json"
    job.build_ledger(control, [exports], ledger)
    dev = analyze.dev_units(control, ledger)
    assert dev["per_stream"]["watch6"]["mean_world_steps_per_parent"] == 150
    assert dev["x86_worker_world_steps_per_second"] == 10
    changed = next(row for row in dev["rates_by_runtime_commit"] if row["code_sha"] == COMMIT)
    assert changed["worker_world_steps_per_second"] == 20
    assert changed["rate_use"] == "DESCRIPTIVE_ONLY_PLATFORM_CHANGED"
    assert changed["runtime_type"] == "l4"
