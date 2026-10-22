"""M0 replay safety, controls, fixed reachability boundaries and immutable resume tests.

Synthetic worlds only: this suite never reconstructs a production rescue.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.bl_served_search import lead_census_b as b
from research.bl_served_search.exports import export_json, verify_export


def test_probe_executes_bare_even_when_search_proposes_rescue(monkeypatch) -> None:
    from research.t3_rollout_labels_20261015 import bank_engine
    from src.simd_env import ego_raster, eval_engine

    class Sim:
        S = 1

        def __init__(self):
            self.length = np.array([[800]])

        def get_resolved_action_mask(self):
            return np.ones((1, 1, 6), dtype=bool)

        def get_advisory_action_mask(self):
            return np.array([[[True, True, True, False, False, False]]])

        def step_with_policy(self, choose):
            self.executed = choose(self).copy()

    class Runtime:
        def prepare(self, sim):
            self.prepared = getattr(self, "prepared", 0) + 1

        def observe_step(self, sim):
            self.observed = True

    class World:
        mix = "mixed"
        hero = None
        opponents = []

        def __init__(self):
            self.sim, self.runtime = Sim(), Runtime()

        def clone(self):
            return copy.deepcopy(self)

    calls = []

    def rollout(root, actions, depth, decision, **kwargs):
        calls.append(actions[0])
        root.sim.length[0, 0] = 1  # A branch mutation must stay off the live trajectory.
        return {
            actions[0]: {
                "survives": actions[0] == 1,
                "frames_executed": 300,
                "death_decision": None if actions[0] == 1 else decision + 299,
            }
        }

    monkeypatch.setattr(bank_engine, "prepared_q", lambda *a: np.arange(6, 0, -1))
    monkeypatch.setattr(bank_engine, "rollout_all", rollout)
    monkeypatch.setattr(ego_raster, "build_ego_raster", lambda *a, **k: {})
    monkeypatch.setattr(eval_engine, "_dispatch_actions", lambda s, m, a, h, o: a.fill(0))
    world = World()
    selected = b.probe_step(world, 10, True, lambda: None)
    assert calls == [0, 1]  # Stops at the first alternative survivor.
    assert selected["proposed"] == 1
    assert selected["executed"] == selected["bare"] == int(world.sim.executed[0, 0]) == 0
    assert int(world.sim.length[0, 0]) == 800
    assert world.runtime.prepared == 1
    assert world.runtime.observed


def fake_replay(monkeypatch):
    from research.bl_served_search import ceiling
    from research.t3_rollout_labels_20261015 import bank_engine

    sim = SimpleNamespace(
        alive=np.array([[True]]), length=np.array([[800]]), get_death_cause=lambda: np.array([[4]])
    )
    world = SimpleNamespace(sim=sim)
    attestation = {"opponent_member_sha256s": [], "opponent_veto_slots": []}
    monkeypatch.setattr(bank_engine, "build_world", lambda *a, **k: world)
    monkeypatch.setattr(ceiling, "opponent_attestation", lambda w: attestation)
    monkeypatch.setattr(ceiling, "portable_plan", lambda p, *a, **k: p)
    monkeypatch.setattr(ceiling, "worker_threads", lambda: {"torch": 1, "numba": 1})
    probes = []

    def batch(decision):
        return {
            "decision": decision,
            "bare": 0,
            "chosen": 1,
            "evaluation_order": [0, 1],
            "branches_evaluated": 2,
            "branches_skipped": 0,
            "outcomes": {
                0: {"survives": False, "death_decision": 12},
                1: {"survives": decision <= 7},
            },
        }

    def probe(w, decision, search, check):
        check()
        sim.alive[0, 0] = decision < 12
        if search:
            probes.append(decision)
        return {
            "bare": 0,
            "executed": 0,
            "proposed": 1,
            "length": 800,
            "batch": batch(decision) if search else None,
        }

    monkeypatch.setattr(b, "probe_step", probe)
    row = {
        "plan": {},
        **attestation,
        "baseline": {"death_frame": 12},
        "trajectories": {
            "baseline_alive": [True] * 12 + [False],
            "baseline_mass": [800] * 12 + [0],
        },
        "dense": {"batch_diagnostics": [json.loads(json.dumps(batch(2)))]},
    }
    record = {
        "world_id": "synthetic",
        "mix": "frozen",
        "f": 2,
        "D": 10,
        "hero_length_at_f": 800,
        "baseline_death_length": 800,
        "baseline_death_cause": 4,
    }
    return row, record, probes


def test_replay_attests_bare_prefix_and_continues_after_all_die(monkeypatch) -> None:
    row, record, probes = fake_replay(monkeypatch)
    result = b.replay_world(row, record, Path("unused"), Path("unused"), lambda: None)
    assert probes == [2, 7, 12]
    assert result["a"] == 7
    assert result["a_minus_f"] == 5
    assert result["executed_changes"] == 0
    assert result["baseline_frames_verified"] == 13
    assert result["bare_branch_controls_verified"] == 3


def test_baseline_divergence_invalidates_replay(monkeypatch) -> None:
    row, record, _ = fake_replay(monkeypatch)
    row["trajectories"]["baseline_mass"][0] = 799
    with pytest.raises(RuntimeError, match="trajectory"):
        b.replay_world(row, record, Path("unused"), Path("unused"), lambda: None)


def test_replay_does_not_stop_at_first_all_die_batch(monkeypatch) -> None:
    row, record, probes = fake_replay(monkeypatch)
    probe = b.probe_step

    def all_die_first(w, decision, search, check):
        result = probe(w, decision, search, check)
        if decision == record["f"]:
            result["batch"]["outcomes"][1]["survives"] = False
        return result

    row["dense"]["batch_diagnostics"][0]["outcomes"]["1"]["survives"] = False
    monkeypatch.setattr(b, "probe_step", all_die_first)
    result = b.replay_world(row, record, Path("unused"), Path("unused"), lambda: None)
    assert probes == [2, 7, 12]
    assert result["a"] == 7


def test_curves_charge_every_miss_and_include_equality() -> None:
    results = [
        {"mix": "frozen", "f": 10, "a": 84},
        {"mix": "mixed", "f": 10, "a": None},
        {"mix": "scripted", "f": 10, "a": 83},
    ]
    curves = b.curves(results)
    row = next(r for r in curves["frozen+mixed"] if r["L"] == 72)
    assert row == {"L": 72, "reachable": 1, "denominator": 2, "fraction": 0.5}
    assert next(r for r in curves["scripted"] if r["L"] == 72)["reachable"] == 0
    assert 75 in {r["L"] for r in curves["frozen+mixed"]}
    assert 432 in {r["L"] for r in curves["frozen+mixed"]}


def ledger_fixture(tmp_path):
    identity = {"namespace": b.NAMESPACE, "allocation_sha256": "fixed"}
    records = [{"world_id": "world-0"}, {"world_id": "world-1"}]
    path = b.unit_path(tmp_path / "prior", 0)
    metadata = export_json(path, {"sealed": "outcome contents are not read by ledger verification"})
    ledger = {
        "schema": "bl-c15-m0-ledger/v1",
        "binding": identity,
        "units": [{**metadata, "allocation_index": 0, "world_id": "world-0"}],
        "charged_seconds": 100,
        "completed_attempts": 1,
    }
    return identity, records, path, ledger


def test_resume_skips_exactly_verified_unit_and_rejects_mutation(tmp_path) -> None:
    identity, records, path, ledger = ledger_fixture(tmp_path)
    assert set(b.verify_ledger(ledger, identity, records)) == {0}
    path.write_text("changed")
    with pytest.raises(ValueError, match="digest"):
        b.verify_ledger(ledger, identity, records)


@pytest.mark.parametrize("mutation", ["namespace", "duplicate", "allocation", "partial", "cap"])
def test_resume_refuses_foreign_duplicate_partial_or_expired(tmp_path, mutation) -> None:
    identity, records, path, ledger = ledger_fixture(tmp_path)
    if mutation == "namespace":
        ledger["binding"] = {**identity, "namespace": "foreign"}
    elif mutation == "duplicate":
        ledger["units"] *= 2
    elif mutation == "allocation":
        ledger["units"][0]["world_id"] = "foreign"
    elif mutation == "partial":
        Path(str(path) + ".complete").unlink()
    else:
        ledger["charged_seconds"] = b.CAP_SECONDS
    with pytest.raises(ValueError):
        b.verify_ledger(ledger, identity, records)


def test_empty_first_ledger_offline() -> None:
    identity = {"namespace": b.NAMESPACE}
    empty = {
        "schema": "bl-c15-m0-ledger/v1",
        "binding": identity,
        "units": [],
        "charged_seconds": 0,
        "completed_attempts": 0,
    }
    assert b.verify_ledger(empty, identity, [{"world_id": "0"}]) == {}


def test_exports_are_create_only_and_completion_is_required(tmp_path) -> None:
    path = tmp_path / "unit.json"
    export_json(path, {"id": "fixed"})
    verify_export(path)
    with pytest.raises(FileExistsError):
        export_json(path, {"id": "changed"})
    Path(str(path) + ".complete").unlink()
    with pytest.raises(ValueError, match="missing"):
        verify_export(path)


def test_resume_after_payload_publish_before_completion_marker(tmp_path, monkeypatch) -> None:
    part_a_path = tmp_path / "part-a.json"
    export_json(part_a_path, {})
    records = [{"world_id": "world-0"}, {"world_id": "world-1"}]
    monkeypatch.setattr(b, "load_part_a", lambda p: ({"rows": records}, []))
    monkeypatch.setattr(b, "source_manifest", lambda: {})
    identity = b.binding(part_a_path, records, {})
    first_ledger = tmp_path / "empty-ledger.json"
    export_json(
        first_ledger,
        {
            "schema": "bl-c15-m0-ledger/v1",
            "binding": identity,
            "units": [],
            "charged_seconds": 0,
            "completed_attempts": 0,
        },
    )
    prior = tmp_path / "attempt-01"
    complete = b.unit_path(prior, 0)
    export_json(complete, {"sealed": True})
    partial = b.unit_path(prior, 1)
    partial.write_text('{"sealed": true}\n')
    export_json(
        prior / "attempt.json",
        {
            "binding": identity,
            "status": "INCOMPLETE_MECHANICAL",
            "attempt_number": 1,
            "verified_stop": True,
            "failure_kind": "worker_exit",
            "ledger_path": str(first_ledger),
            "ledger_sha256": verify_export(first_ledger),
            "series_charged_seconds": 100,
        },
    )
    ledger = b.make_ledger(part_a_path, tmp_path / "resume.json", prior)
    assert [u["allocation_index"] for u in ledger["units"]] == [0]
    assert partial.read_text() == '{"sealed": true}\n'
    assert set(b.verify_ledger(ledger, identity, records)) == {0}


def validated_result_fixture():
    record = {"f": 2, "baseline_death_frame": 7}
    batch = {
        "decision": 2,
        "bare": 0,
        "chosen": 1,
        "evaluation_order": [0, 1],
        "branches_evaluated": 2,
        "branches_skipped": 1,
        "outcomes": {
            "0": {"survives": False, "death_decision": 7, "frames_executed": 6},
            "1": {"survives": True, "frames_executed": 300, "death_decision": None},
            "2": {"status": "NOT_EVALUATED"},
        },
    }
    result = {
        "batches": [batch],
        "empty_advisory_decisions": [7],
        "search_mode": "exact_stop",
        "native_mode": "stock",
        "a": 2,
        "a_minus_f": 0,
        "proposed_changes": 1,
    }
    return result, record


def test_aggregator_recomputes_lad_from_witnesses() -> None:
    result, record = validated_result_fixture()
    b.validate_result(result, record)
    result["a"] = 250
    with pytest.raises(ValueError, match="LAD"):
        b.validate_result(result, record)


@pytest.mark.parametrize(
    "mutation", ["field_only", "cadence", "control", "skips", "native", "search", "short_survivor"]
)
def test_aggregator_rejects_unwitnessed_or_foreign_results(mutation) -> None:
    result, record = validated_result_fixture()
    if mutation == "field_only":
        result["batches"] = []
    elif mutation == "cadence":
        result["empty_advisory_decisions"] = []
    elif mutation == "control":
        result["batches"][0]["outcomes"]["0"]["death_decision"] = 8
    elif mutation == "skips":
        result["batches"][0]["outcomes"]["2"] = {"survives": True}
    elif mutation == "native":
        result["native_mode"] = "kernel"
    elif mutation == "short_survivor":
        result["batches"][0]["outcomes"]["1"]["frames_executed"] = 1
    else:
        result["search_mode"] = "all"
    with pytest.raises(ValueError):
        b.validate_result(result, record)


@pytest.mark.parametrize("mutation", [None, "threads", "kernel_rows", "opponents"])
def test_execution_attestation(mutation) -> None:
    from research.bl_k1.native import RowStats

    source = {"opponent_member_sha256s": ["fixture"], "opponent_veto_slots": [[]]}
    result = {
        "worker_threads": {
            "torch": 1,
            "numba": 1,
            "environment": {k: "1" for k in b.THREAD_VARS[:5]},
        },
        "native_v8": RowStats().as_dict(),
        "opponents": source.copy(),
    }
    if mutation == "threads":
        result["worker_threads"]["torch"] = 2
    elif mutation == "kernel_rows":
        result["native_v8"]["kernel_rows"] = 1
    elif mutation == "opponents":
        result["opponents"] = {}
    if mutation:
        with pytest.raises(ValueError):
            b.validate_execution_attestation(result, source)
    else:
        b.validate_execution_attestation(result, source)


def test_runtime_holds_canonical_process_live_fence(tmp_path, monkeypatch) -> None:
    from research.mac_lease import LeaseHeld, held

    path = tmp_path / "mac-lease.json"

    def inspect_lease(args, lease):
        assert json.loads(path.read_text())["id"] == lease.lease_id
        with pytest.raises(LeaseHeld):
            with held("other", "timing", seconds=30, path=path):
                pytest.fail("foreign timed job acquired M0's fence")
        return {"fixture": True}

    monkeypatch.setattr(b, "run_leased", inspect_lease)
    assert b.run(SimpleNamespace(lease_owner="fixture", lease=path)) == {"fixture": True}
    assert not path.exists()
