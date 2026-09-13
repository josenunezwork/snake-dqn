"""Regression tests for immutable tournament-evaluation inputs."""

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.core.config_loader import load_config
from src.core.runtime_contract import EffectiveWorldConfig
from src.core.world_runtime import WorldRuntimeSpec
from src.evaluation.artifacts import EvaluationArtifacts, SnapshotError
from src.evaluation.protocol import EvaluationProfile, promotion_v2_watch_rect


def _promotion_profile(*, width: int = 400) -> EvaluationProfile:
    world = EffectiveWorldConfig(
        width=width,
        height=300,
        segment_size=10,
        wall_thickness=10,
        arena_type="rectangular",
        mechanics_version=2,
        num_snakes=3,
        max_frames=5000,
        initial_food=2,
        max_food=2,
        min_boost_length=5,
        boost_length_cost_frames=3,
        frame_rate=1,
        max_length=100,
        starvation_max_frames=500,
        max_capacity=400,
        kill_scale=0.3,
        death_value=-3.0,
        normalization={"max_frames": 5000.0, "starvation_max": 500.0, "max_length": 100.0},
    )
    return promotion_v2_watch_rect(world)


def test_equal_checkpoint_aliases_share_one_content_identity(tmp_path):
    first = tmp_path / "first.pth"
    second = tmp_path / "second.pth"
    first.write_bytes(b"same checkpoint bytes")
    second.write_bytes(first.read_bytes())

    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    first_snapshot = artifacts.snapshot_checkpoint(first)
    second_snapshot = artifacts.snapshot_checkpoint(second)

    assert first_snapshot.sha256 == second_snapshot.sha256
    assert first_snapshot.snapshot_path == second_snapshot.snapshot_path
    assert Path(first_snapshot.snapshot_path).read_bytes() == first.read_bytes()
    assert len(artifacts.checkpoint_receipts()) == 2


def test_atomic_source_replacement_keeps_the_opened_checkpoint_bytes(tmp_path, monkeypatch):
    source = tmp_path / "rolling.pth"
    original = b"old checkpoint"
    source.write_bytes(original)
    replacement = tmp_path / "replacement.pth"
    replacement.write_bytes(b"new checkpoint")

    real_fstat = os.fstat
    calls = 0

    def replace_after_open(fd):
        nonlocal calls
        calls += 1
        result = real_fstat(fd)
        if calls == 1:
            os.replace(replacement, source)
        return result

    monkeypatch.setattr("src.evaluation.artifacts.os.fstat", replace_after_open)
    snapshot = EvaluationArtifacts(tmp_path / "artifacts").snapshot_checkpoint(source)

    assert Path(snapshot.snapshot_path).read_bytes() == original
    assert source.read_bytes() == b"new checkpoint"


def test_in_place_mutation_during_copy_fails_and_does_not_publish_snapshot(tmp_path, monkeypatch):
    source = tmp_path / "mutable.pth"
    source.write_bytes(b"checkpoint")
    real_fstat = os.fstat
    calls = 0

    def changed_second_stat(fd):
        nonlocal calls
        calls += 1
        result = real_fstat(fd)
        if calls == 2:
            return SimpleNamespace(
                st_dev=result.st_dev,
                st_ino=result.st_ino,
                st_size=result.st_size,
                st_mtime_ns=result.st_mtime_ns + 1,
            )
        return result

    root = tmp_path / "artifacts"
    monkeypatch.setattr("src.evaluation.artifacts.os.fstat", changed_second_stat)
    with pytest.raises(SnapshotError, match="changed while being snapshotted"):
        EvaluationArtifacts(root).snapshot_checkpoint(source)

    assert not list(root.rglob("*.pth"))


def test_receipt_records_hashes_effective_config_and_source_agents(tmp_path):
    checkpoint = tmp_path / "candidate.pth"
    checkpoint.write_bytes(b"candidate")
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("print('evaluator')\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    artifacts.snapshot_checkpoint(checkpoint)
    config_snapshot = artifacts.snapshot_config(config)

    receipt_path = artifacts.write_receipt(
        config_snapshot=config_snapshot,
        effective_config={"game": {"width": 400}},
        evaluator_path=evaluator,
        source_specs=[("checkpoint", str(checkpoint)), ("scripted", "greedy_food")],
    )

    receipt = json.loads(receipt_path.read_text())
    assert receipt["config"]["effective"]["game"]["width"] == 400
    assert receipt["checkpoint_snapshots"][0]["source_path"] == str(checkpoint)
    assert receipt["source_agents"][-1] == {"kind": "scripted", "reference": "greedy_food"}
    assert len(receipt["config"]["sha256"]) == 64
    assert len(receipt["evaluator"]["sha256"]) == 64


def test_receipt_preserves_supplied_yaml_fields_from_loaded_appconfig(tmp_path):
    config_path = tmp_path / "eval.yaml"
    config_path.write_text("pqn:\n  recipe: corrected-v3\n  lr: 0.0002\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config_path)
    config = load_config(snapshot.snapshot_path)
    evaluator = tmp_path / "evaluator.py"
    evaluator.write_text("# fixture\n")

    receipt_path = artifacts.write_receipt(
        config_snapshot=snapshot,
        effective_config=config,
        evaluator_path=evaluator,
        source_specs=[],
    )

    effective = json.loads(receipt_path.read_text())["config"]["effective"]
    assert effective["provided_fields"] == ["pqn.lr", "pqn.recipe"]
    assert effective["pqn"]["recipe"] == "corrected-v3"
    assert effective["pqn"]["lr"] == 0.0002
    assert effective["pqn"]["gamma"] is None
    assert config.provided_fields == frozenset({"pqn.lr", "pqn.recipe"})


def test_evaluator_source_manifest_changes_when_a_dependency_changes(tmp_path):
    evaluator = tmp_path / "src" / "scripts" / "tournament_eval.py"
    dependency = tmp_path / "src" / "scripts" / "eval_stats.py"
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text("EVALUATOR = 1\n")
    dependency.write_text("STATS = 1\n")

    from src.evaluation.artifacts import evaluator_provenance

    first = evaluator_provenance(evaluator, [evaluator, dependency])
    dependency.write_text("STATS = 2\n")
    second = evaluator_provenance(evaluator, [evaluator, dependency])

    assert first["source_manifest"] != second["source_manifest"]


def test_default_source_closure_binds_profile_metric_anchor_and_runtime_adapters():
    """E1 profile metrics cannot be evaluated against an E0-only source receipt."""
    from src.evaluation.artifacts import _default_evaluator_sources

    repository = Path(__file__).resolve().parents[1]
    closure = {str(path.relative_to(repository)) for path in _default_evaluator_sources(repository)}

    assert {
        "src/evaluation/protocol.py",
        "src/evaluation/metrics.py",
        "src/evaluation/anchors.py",
        "src/core/world_runtime.py",
        "src/simd_env/eval_engine.py",
        "src/simd_env/live_adapter.py",
        "src/simd_env/batch_sim.py",
        "src/simd_env/featurizer.py",
        "web/backend/raster_policy.py",
    } <= closure


def test_receipt_binds_explicit_evaluation_profile(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config)
    profile = _promotion_profile()
    receipt = json.loads(
        artifacts.write_receipt(
            config_snapshot=snapshot,
            effective_config={},
            evaluator_path=__file__,
            source_specs=[],
            evaluation_profile=profile,
        ).read_text()
    )

    assert receipt["artifact_version"] == "evaluation-input-snapshot-v1"
    assert set(receipt) == {
        "artifact_version",
        "checkpoint_digest_algorithm",
        "checkpoint_snapshots",
        "config",
        "evaluation_profile",
        "evaluator",
        "source_agents",
    }
    assert receipt["evaluation_profile"]["digest"] == profile.digest


def test_explicit_world_runtime_spec_selects_v2_and_binds_exact_identity(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config)
    profile = _promotion_profile()
    runtime_spec = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)

    receipt = json.loads(
        artifacts.write_receipt(
            config_snapshot=snapshot,
            effective_config={},
            evaluator_path=__file__,
            source_specs=[],
            evaluation_profile=profile,
            world_runtime_spec=runtime_spec,
        ).read_text()
    )

    assert receipt["artifact_version"] == "evaluation-input-snapshot-v2"
    assert receipt["world_runtime_spec"] == {
        "descriptor": runtime_spec.descriptor(),
        "digest": runtime_spec.digest,
    }
    assert (
        WorldRuntimeSpec.from_descriptor(
            receipt["world_runtime_spec"]["descriptor"],
            expected_digest=receipt["world_runtime_spec"]["digest"],
        )
        == runtime_spec
    )


def test_world_runtime_receipt_rejects_tampered_detached_digest(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config)
    profile = _promotion_profile()
    runtime_spec = WorldRuntimeSpec.fresh_reset_horizon_bound(profile)

    class TamperedRuntimeSpec(WorldRuntimeSpec):
        @property
        def digest(self) -> str:
            return "0" * 64

    tampered = TamperedRuntimeSpec(**runtime_spec.__dict__)

    with pytest.raises(ValueError, match="descriptor digest"):
        artifacts.write_receipt(
            config_snapshot=snapshot,
            effective_config={},
            evaluator_path=__file__,
            source_specs=[],
            evaluation_profile=profile,
            world_runtime_spec=tampered,
        )

    assert not (artifacts.root / "receipt.json").exists()


def test_world_runtime_receipt_rejects_profile_mismatch_before_publication(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config)
    runtime_spec = WorldRuntimeSpec.fresh_reset_horizon_bound(_promotion_profile())

    with pytest.raises(ValueError, match="profile digest"):
        artifacts.write_receipt(
            config_snapshot=snapshot,
            effective_config={},
            evaluator_path=__file__,
            source_specs=[],
            evaluation_profile=_promotion_profile(width=410),
            world_runtime_spec=runtime_spec,
        )

    assert not (artifacts.root / "receipt.json").exists()


def test_world_runtime_receipt_requires_an_explicit_profile(tmp_path):
    config = tmp_path / "eval.yaml"
    config.write_text("game: {}\n")
    artifacts = EvaluationArtifacts(tmp_path / "artifacts")
    snapshot = artifacts.snapshot_config(config)
    profile = _promotion_profile()

    with pytest.raises(ValueError, match="requires an explicit evaluation profile"):
        artifacts.write_receipt(
            config_snapshot=snapshot,
            effective_config={},
            evaluator_path=__file__,
            source_specs=[],
            world_runtime_spec=WorldRuntimeSpec.source_exact(profile),
        )

    assert not (artifacts.root / "receipt.json").exists()
