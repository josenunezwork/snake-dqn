"""Interrupted full entrypoint parity; all positive support is synthetic test data."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.bl_ppo import p1_runtime, run
from research.bl_ppo.exports import export_json, verify_export
from research.bl_ppo.learner import PPOLearner
from research.bl_ppo.p1_resume import build_ledger
from research.bl_ppo.recovery import read_snapshot
from research.bl_ppo.test_p1_entrypoints import _copy_export, _entrypoint
from research.bl_ppo.test_p1_replay_amendment02 import synthetic_natural_support, tiny_p1

# Register imported fixtures in this module; the natural-labelled rows are
# synthetic inside pytest only, never evidence of critic support or L4 Y3.
__all__ = ["synthetic_natural_support", "tiny_p1"]


def _cadence_clock(monkeypatch, *, first_actor_capture=1):
    class Clock:
        now = 0.0
        refit = 0
        actor = 0

        def tick(self):
            self.now += 0.0001
            return self.now

    clock = Clock()
    update = PPOLearner.update

    def controlled_update(self, *args, **kwargs):
        result = update(self, *args, **kwargs)
        phase = "actor" if kwargs.get("actor_enabled") else "refit"
        count = getattr(clock, phase) + 1
        setattr(clock, phase, count)
        if count == (first_actor_capture if phase == "actor" else 1):
            clock.now += 901
        return result

    from types import SimpleNamespace

    monkeypatch.setattr(run, "time", SimpleNamespace(monotonic=clock.tick))
    monkeypatch.setattr(PPOLearner, "update", controlled_update)
    return clock


def test_exact_actor_origin_waits_for_real_periodic_capture(tmp_path, monkeypatch, tiny_p1):
    """The production origin says actor but has no OPERATIONAL metadata.

    Three fast actor updates must not satisfy the periodic proof from this
    exact origin. The first actual cadence capture is on update four, after
    which the registered suffix still needs three whole updates.
    """
    clock = _cadence_clock(monkeypatch, first_actor_capture=4)
    _, source, argv = _source(tmp_path / "source", monkeypatch, tiny_p1)
    assert p1_runtime.main(argv) == 0
    proof = json.loads((source / "receipt.json").read_text())
    origin = read_snapshot(source / "origin" / proof["origin_snapshot"]["name"])
    assert origin["execution"]["phase"] == "actor"
    assert "operational" not in origin
    latest = max(proof["operational_snapshots"], key=lambda row: row["index"])
    anchor = read_snapshot(source / "source-continuation" / latest["path"])
    trace = json.loads((source / "source-continuation/operational-action-trace.json").read_text())
    assert trace["snapshot"] == latest["path"]
    assert anchor["operational"]["label"] == "OPERATIONAL"
    assert anchor["operational"]["execution"]["phase"] == "actor"
    assert anchor["counters"]["actor_updates"] == 4
    assert trace["final_actor_updates"] - anchor["counters"]["actor_updates"] >= 3
    assert trace["final_episode_boundaries"] > anchor["counters"]["actor_episode_boundaries"]
    assert trace["actions"] >= 1000
    assert clock.actor >= 7


def _source(tmp_path, monkeypatch, tiny_p1):
    return _entrypoint(
        tmp_path,
        monkeypatch,
        tiny_p1,
        "y3-source",
        operational_replay_required=True,
        checkpoint_every_seconds=900,
    )


def _flat_transport(source: Path, target: Path) -> None:
    target.mkdir()
    receipt = json.loads((source / "receipt.json").read_text())
    for row in receipt["transitive_snapshot_catalog"] + receipt["operational_snapshot_catalog"]:
        candidates = list(source.rglob(row["name"]))
        assert candidates
        _copy_export(candidates[0], target / row["name"])
    for name in ("action-trace.json", "operational-action-trace.json"):
        _copy_export(source / "source-continuation" / name, target / name)
    _copy_export(source / "receipt.json", target / "receipt.json")


@pytest.mark.parametrize("bad_suffix", [False, True])
def test_real_two_replays_use_source_stop_and_periodic_suffix(
    tmp_path, monkeypatch, tiny_p1, bad_suffix
):
    _cadence_clock(monkeypatch)
    _, source, argv = _source(tmp_path / "source", monkeypatch, tiny_p1)
    assert p1_runtime.main(argv) == 0
    source_proof = json.loads((source / "receipt.json").read_text())
    assert source_proof["operational_snapshots"]
    assert list((source / "origin").glob("operational-*.pt")), "refit must have a cadence snapshot"
    flat = tmp_path / "flat"
    _flat_transport(source, flat)
    source.rename(source.with_name("source-unavailable-to-replacement"))
    if bad_suffix:
        trace = flat / "operational-action-trace.json"
        proof = flat / "receipt.json"
        trace_value, proof_value = json.loads(trace.read_text()), json.loads(proof.read_text())
        trace_value["full_tape_sha256"] = "00" * 32
        # Deliberate alternative, internally hash-consistent test evidence;
        # production exports remain immutable and no real outcomes are used.
        for path in (trace, proof):
            for suffix in ("", ".sha256", ".complete"):
                Path(str(path) + suffix).unlink()
        export_json(trace, trace_value)
        proof_value["operational_trace"]["sha256"] = verify_export(trace)
        export_json(proof, proof_value)
    # Replacement clock has no cadence jump; source final absolute counter
    # is nevertheless the same, and both actual model/collector replays run.
    origin = flat / source_proof["origin_snapshot"]["name"]
    _, output, argv = _entrypoint(
        tmp_path / "replacement",
        monkeypatch,
        tiny_p1,
        "y3-replacement",
        resume=origin,
        proof=flat / "receipt.json",
        operational_replay_required=True,
    )
    assert p1_runtime.main(argv) == 0
    value = json.loads((output / "receipt.json").read_text())
    assert value["status"] == ("Y3_PASS_BOUNDARY_ONLY" if bad_suffix else "PASS")
    assert value["operational_replay"]["status"] == ("FAIL" if bad_suffix else "PASS")
    assert value["operational_replay"]["actions_compared"] >= 1000


@pytest.mark.parametrize("phase", ["refit", "actor"])
def test_interrupted_source_resumes_latest_without_recomputing(
    tmp_path, monkeypatch, tiny_p1, phase
):
    clock = _cadence_clock(monkeypatch)
    _, baseline, argv = _source(tmp_path / "baseline", monkeypatch, tiny_p1)
    assert p1_runtime.main(argv) == 0
    baseline_proof = json.loads((baseline / "receipt.json").read_text())
    baseline_final = baseline / baseline_proof["source_continuation_snapshot"]["relative_path"]
    clock.refit = clock.actor = 0
    clock.now = 0.0
    original_export = run.export_torch
    failed = False

    def fail_after_complete_snapshot(path, value, **kwargs):
        nonlocal failed
        result = original_export(path, value, **kwargs)
        metadata = value.get("operational", {})
        if not failed and metadata.get("execution", {}).get("phase") == phase:
            failed = True
            raise OSError("test-only mechanical transport loss after completed snapshot")
        return result

    monkeypatch.setattr(run, "export_torch", fail_after_complete_snapshot)
    _, interrupted, argv = _source(tmp_path / "interrupted", monkeypatch, tiny_p1)
    with pytest.raises(OSError, match="test-only mechanical"):
        p1_runtime.main(argv)
    monkeypatch.setattr(run, "export_torch", original_export)
    candidates = list(interrupted.rglob("operational-*.pt"))
    latest = max(candidates, key=lambda path: read_snapshot(path)["counters"]["hero_transitions"])
    saved = read_snapshot(latest)
    assert saved["operational"]["execution"]["phase"] == phase
    from research.bl_ppo.recovery import copy_snapshot_closure

    flat = tmp_path / "resumed-inputs"
    copy_snapshot_closure(latest, flat)
    inputs, resumed, argv = _source(tmp_path / "resumed", monkeypatch, tiny_p1)
    ledger = inputs / "verified-ledger.json"
    for suffix in ("", ".sha256", ".complete"):
        Path(str(ledger) + suffix).unlink()
    build_ledger(interrupted, ledger, namespace="test-y3", allocation="source")
    argv += ["--resume", str(flat / latest.name)]
    assert p1_runtime.main(argv) == 0
    result = json.loads((resumed / "receipt.json").read_text())
    final = resumed / result["source_continuation_snapshot"]["relative_path"]
    assert (
        result["source_trace"]["full_tape_sha256"]
        == baseline_proof["source_trace"]["full_tape_sha256"]
    )
    assert (
        result["source_trace"]["full_tape_actions"]
        == baseline_proof["source_trace"]["full_tape_actions"]
    )
    assert p1_runtime._snapshot_fingerprint(final) == p1_runtime._snapshot_fingerprint(
        baseline_final
    )
    assert (
        read_snapshot(final)["counters"]["critic_refit_transitions"]
        == read_snapshot(baseline_final)["counters"]["critic_refit_transitions"]
    )
    assert not any(
        path.name == latest.name
        for path in resumed.rglob("operational-*.pt")
        if path.parent.name == "origin"
    )
