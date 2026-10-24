"""Review-03 faults traverse the actual stale fixture, writer and reducer."""

from __future__ import annotations

import argparse
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any

import pytest
import torch

from research.gpu_sim_spike import t4a_analyze as analyze
from research.gpu_sim_spike import t4a_bench as bench
from research.gpu_sim_spike import t4a_events as events
from research.gpu_sim_spike.test_t4a_evidence import COMMIT, _tree

CHECKPOINT = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth")


@pytest.mark.integration
@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="real T4A checkpoint is not installed")
@pytest.mark.parametrize("fault", ["none", "seven_prepared_ticks", "clone_warm"])
def test_real_warm_fault_writer_and_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, fault: str
) -> None:
    """Common-mode tick corruption must disagree with the independent warm seam."""
    original_tick, original_warm = events._tick, events._warm_output_capture
    changed = []
    calls = 0

    def tick(case: tuple, *args: Any, **kwargs: Any) -> dict[str, torch.Tensor]:
        prepared = bool(case[0]._prepared and case[1].vector_prepared)
        fields = original_tick(case, *args, **kwargs)
        if fault == "seven_prepared_ticks" and prepared:
            fields["output.q.0"].view(torch.uint8).reshape(-1)[0] ^= 1
            changed.append(id(case[0]))
        return fields

    def warm(case: tuple) -> dict[str, torch.Tensor]:
        nonlocal calls
        calls += 1
        fields = original_warm(case)
        if fault == "clone_warm" and calls == 2:
            fields["output.q.0"].view(torch.uint8).reshape(-1)[0] ^= 1
        return fields

    monkeypatch.setattr(events, "_tick", tick)
    monkeypatch.setattr(events, "_warm_output_capture", warm)
    monkeypatch.setattr(events, "EVENTS", ("stale_cache",))
    result = events.run_events(CHECKPOINT, "scripted", "cpu", "smoke", float("inf"))
    row = result["evidence"][0]
    witness = row["witness"]
    assert witness["ticks"] == 2
    assert len(witness["prepared_tick_captures"]) == 7
    assert witness["mismatch"] is None
    assert events._TAPES == {}
    if fault == "seven_prepared_ticks":
        assert len(changed) == 7
        assert all(not check["equal"] for check in witness["prepared_tick_comparisons"].values())
    if fault == "none":
        assert row["status"] == "PASS"
        assert (
            analyze._event_predicate_error("stale_cache", witness, 4, witness["mapped_tape"])
            is None
        )
    else:
        assert row["status"] == "INVALID_EVIDENCE"
        assert witness["evidence_errors"]

    def inject(content: dict) -> None:
        # Synthetic surrounding admission records, real producer/writer row.
        content["event_records"] = [deepcopy(row)]

    root = _tree(tmp_path / "aggregate", mutate=inject, mutate_mix="scripted")
    reduced = analyze.aggregate(root, COMMIT, canary=True)
    if fault == "none":
        assert not reduced["invalid_evidence"]
    else:
        assert reduced["verdict"] == "INVALID_EVIDENCE"
        assert any("lifecycle/invalidation" in error for error in reduced["invalid_evidence"])
        assert not reduced["mismatches"]

        # Retain this actual fixture fault through the runtime failure writer,
        # not just a synthetic surrounding completed roster unit.
        failure_root = tmp_path / "failure-writer"

        def write_actual_fault(_args: argparse.Namespace) -> None:
            bench._require_valid_event_evidence(result["evidence"], failure_root)

        monkeypatch.setattr(bench, "_run", write_actual_fault)
        monkeypatch.setattr(bench, "source_commit", lambda: COMMIT)
        monkeypatch.setattr(bench, "source_hashes", lambda: {})
        failure = bench.run(argparse.Namespace(output_dir=failure_root, smoke=False, canary=True))
        assert failure["status"] == "INVALID_EVIDENCE"
        assert (
            json.loads((failure_root / "diagnostics/invalid-event-evidence.json").read_text())[
                "event_records"
            ]
            == result["evidence"]
        )
        failed_reduction = analyze.aggregate(failure_root, COMMIT, canary=True)
        assert failed_reduction["verdict"] == "INVALID_EVIDENCE"
        assert not failed_reduction["mismatches"]


@pytest.mark.parametrize("status", ["PASS", "INCOMPLETE", "timeout", "INVALID_EVIDENCE"])
def test_completed_contradiction_cannot_hide_behind_status(tmp_path: Path, status: str) -> None:
    def inject(content: dict) -> None:
        row = next(row for row in content["event_records"] if row["event"] == "stale_cache")
        row["status"] = status
        row["witness"]["warm_output_comparisons"]["source_clone"]["equal"] = False
        row["witness_sha256"] = hashlib.sha256(
            json.dumps(row["witness"], sort_keys=True, default=str).encode()
        ).hexdigest()

    result = analyze.aggregate(_tree(tmp_path / "aggregate", mutate=inject), COMMIT, canary=True)
    assert result["verdict"] == "INVALID_EVIDENCE"
    assert any("lifecycle/invalidation" in error for error in result["invalid_evidence"])


def test_internal_mismatch_survives_missing_e4_reference(monkeypatch: pytest.MonkeyPatch) -> None:
    witness = {"field": "sim.rng_counter", "byte_offset": 0, "expected": 0, "actual": 1}
    raw = [{"sim.frame": torch.tensor([0])}, {"sim.frame": torch.tensor([1])}]

    def run_one(
        event: str, checkpoint: Path, roster: str, worlds: int, device: str, lane: str
    ) -> tuple[bool, dict]:
        if worlds == 4:
            raise RuntimeError("missing E4")
        return False, {"mismatch": witness, "_raw_mapped_tape": raw}

    monkeypatch.setattr(events, "EVENTS", ("stale_cache",))
    monkeypatch.setattr(events, "_run_one", run_one)
    result = events.run_events(Path("checkpoint"), "scripted", "cpu", "main", float("inf"))
    assert [row["status"] for row in result["evidence"]] == [
        "INCOMPLETE",
        "INVALID_SELF_CONSISTENCY",
    ]
    assert result["evidence"][-1]["witness"]["first_witness"] == witness
