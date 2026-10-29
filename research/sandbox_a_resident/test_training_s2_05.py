"""Progress and failed-preparation regression twins for the S2-04 diagnosis."""

import hashlib
import json
import subprocess
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch

from research.sandbox_a_resident import test_training_production as production_tests
from research.sandbox_a_resident import train, training_compact
from research.sandbox_a_resident.training_prep import PreparationFailure
from research.sandbox_a_resident.training_resume import ResumeRefused, build_ledger
from research.sandbox_a_resident.training_units import validate_compact_source_packet
from research.sandbox_a_resident.units import publish

production = production_tests.production


@pytest.mark.parametrize("suffix", ["", "-c1", "-c2-i0016"])
def test_corrected_packet_family_is_bounded(suffix):
    validate_compact_source_packet("packet-train01-s2-05-r1" + suffix, 0)
    for name in ("packet-train01-s2-05-r2", "packet-train01-s2-05-r1-c3"):
        with pytest.raises(ValueError):
            validate_compact_source_packet(name, 0)


def test_failed_preparation_snapshot_never_enters_early_publication_or_proxy(
    production, tmp_path, monkeypatch
):
    control, calls = production

    def fail(*args, **kwargs):
        raise PreparationFailure(
            "critic preparation endpoint failed: quality={'passed': 0.0}",
            SimpleNamespace(state_dict=lambda: {"stage": "failed"}),
        )

    monkeypatch.setattr(train, "prepare", fail)
    monkeypatch.setattr(train, "evaluate", lambda *a, **k: pytest.fail("proxy must not run"))
    events = []
    with pytest.raises(PreparationFailure, match="critic preparation endpoint failed"):
        production_tests.execute(
            control,
            tmp_path,
            early_pre_actor_snapshot=True,
            snapshot_format="compact",
            on_progress=lambda stage, values: events.append((stage, dict(values))),
        )
    info = json.loads((tmp_path / "iteration-0000.json").read_text())
    assert info["metadata"]["phase"] == "preparation-failed"
    assert info["metadata"]["prepare_phase"] == "failed"
    assert not (tmp_path / "iteration-0000-actor.pth").exists()
    assert "preparation-failed" in [stage for stage, _ in events]
    assert events[-1][0] == "snapshot-compact-complete"
    assert events[-1][1]["snapshot_phase"] == "preparation-failed"
    with pytest.raises(ResumeRefused, match="no complete full-trajectory"):
        build_ledger(tmp_path, train.asdict(control))
    # Even supplying the absent actor companion cannot promote failed prep.
    publish(tmp_path / "iteration-0000-actor.pth", lambda path: path.write_bytes(b"fixture"))
    with pytest.raises(ResumeRefused, match="not a safe continuation"):
        build_ledger(tmp_path, train.asdict(control))


def test_publication_progress_preserves_bytes_rng_and_complete_last(tmp_path, monkeypatch):
    state = {"tensor": torch.arange(500000), "stage": "ready"}
    rng = torch.get_rng_state().clone()
    records = []
    # Simulate elapsed time only at completed byte-work callbacks; no sleep or
    # timer thread can manufacture progress during a blocked operation.
    clock = iter(range(0, 100000, 31))
    monkeypatch.setattr(training_compact.time, "monotonic", lambda: next(clock))

    def report(stage, values):
        assert not (tmp_path / "with-progress/iteration-0000.json.complete").exists()
        records.append((stage, dict(values)))

    # Frozen, pre-change implementation: the byte oracle must not call the
    # candidate's changed hashing/compression helpers.
    source = subprocess.check_output(
        [
            "git",
            "show",
            "c08934152e4e09cfa4bd82540a120715edd8e3a7:"
            "research/sandbox_a_resident/training_compact.py",
        ],
        cwd=Path(__file__).resolve().parents[2],
    )
    baseline = ModuleType("frozen_s2_04_compact")
    exec(compile(source, "frozen_s2_04_compact.py", "exec"), baseline.__dict__)
    baseline.publish_compact_snapshot(
        tmp_path / "plain",
        0,
        lambda path: torch.save(state, path),
        metadata={"phase": "pre-actor"},
    )
    training_compact.publish_compact_snapshot(
        tmp_path / "with-progress",
        0,
        lambda path: torch.save(state, path),
        metadata={"phase": "pre-actor"},
        on_progress=report,
    )
    for suffix in ("pt.xz", "pt.xz.manifest"):
        assert (tmp_path / f"plain/iteration-0000.{suffix}").read_bytes() == (
            tmp_path / f"with-progress/iteration-0000.{suffix}"
        ).read_bytes()
    assert torch.equal(torch.get_rng_state(), rng)
    stages = [stage for stage, _ in records]
    assert {
        "snapshot-serialized",
        "snapshot-file-hash",
        "snapshot-loaded",
        "snapshot-tensor-hash",
        "snapshot-compress",
        "snapshot-compressed",
        "snapshot-roundtrip",
    } <= set(stages)
    assert sum(stage == "snapshot-file-hash" for stage in stages) > 1
    assert sum(stage == "snapshot-tensor-hash" for stage in stages) > 1
    raw = memoryview(state["tensor"].view(torch.uint8).numpy())
    assert training_compact._tensor_sha256(state["tensor"]) == hashlib.sha256(raw).hexdigest()


def test_publication_does_not_report_compression_work_that_failed(tmp_path, monkeypatch):
    records = []

    def blocked(*args, **kwargs):
        raise RuntimeError("compressor stalled fixture")

    monkeypatch.setattr(training_compact, "_compress", blocked)
    with pytest.raises(RuntimeError, match="compressor stalled fixture"):
        training_compact.publish_compact_snapshot(
            tmp_path,
            0,
            lambda path: torch.save({"tensor": torch.zeros(2)}, path),
            metadata={},
            on_progress=lambda stage, values: records.append(stage),
        )
    assert "snapshot-compress" not in records
    assert "snapshot-compressed" not in records
    assert not (tmp_path / "iteration-0000.json.complete").exists()
