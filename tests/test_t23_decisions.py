"""Decision boundaries and the actual writer-to-analysis contract."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.redesign_scope_20261007 import grid_h5000_identity as gid
from research.t23_r0_screen_20261017 import analyze, bank, evaluate
from tests.test_t23_analysis import _manifest


@pytest.mark.parametrize("delta,expected", [(0, "KILL"), (19, "EXTEND"), (20, "GO")])
def test_efficacy_boundaries(delta, expected):
    assert analyze.analyze(_manifest(delta), 2)["verdict"] == expected


def test_equal_mix_weight_and_scripted_point_guard():
    value = _manifest(30)
    for row in value["rows"]:
        if row["arm"] == "candidate" and row["mix"] == "scripted":
            row["mass_integral"] = 100
    result = analyze.analyze(value, 2)
    assert result["steps"]["8000000"]["G"]["mean"] == 20
    assert result["verdict"] == "GO"
    for row in value["rows"]:
        if row["arm"] == "candidate" and row["mix"] == "scripted":
            row["mass_integral"] -= 0.1
    assert analyze.analyze(value, 2)["verdict"] == "EXTEND"


@pytest.mark.parametrize("fraction,expected", [(0.97, "GO"), (0.9698, "KILL")])
def test_pooled_alive_guard_boundary(fraction, expected):
    value = _manifest()
    for row in value["rows"]:
        if row["arm"] == "candidate":
            row["alive_frame_fraction"] = fraction
    assert analyze.analyze(value, 2)["verdict"] == expected


def test_four_positive_seed_requirement():
    value = _manifest(50)
    for row in value["rows"]:
        if row["arm"] == "candidate" and row["training_seed"] in (7101, 7102):
            row["mass_integral"] = 99
    result = analyze.analyze(value, 2)
    assert result["steps"]["8000000"]["G"]["lower90"] > 0
    assert result["verdict"] == "EXTEND"


def test_no_death_boundary_and_tripwire():
    value = _manifest()
    for index, row in enumerate(value["rows"]):
        if row["arm"] == "candidate":
            row["no_death_episode"] = index % 4 == 1
    assert analyze.analyze(value, 2)["verdict"] == "GO"
    for row in value["rows"]:
        if (
            row["arm"] == "candidate"
            and row["checkpoint_step"] == 8_000_000
            and row["no_death_episode"]
        ):
            row["no_death_episode"] = False
            break
    assert analyze.analyze(value, 2)["verdict"] == "KILL"
    value = _manifest()
    value["tripwires"]["7102"] = True
    assert analyze.analyze(value, 2)["verdict"] == "KILL"


def test_partial_descriptive_is_invalid_not_exception():
    value = _manifest()
    value["rows"] = [
        r for r in value["rows"] if r["checkpoint_step"] != 2_000_000 or r["training_seed"] == 7101
    ]
    assert analyze.analyze(value, 2)["verdict"] == "INVALID"
    value["rows"] = [r for r in value["rows"] if r["checkpoint_step"] == 8_000_000]
    assert analyze.analyze(value, 2)["verdict"] == "GO"


def _generate_receipts(
    tmp_path, monkeypatch, delta=25, phase="initial", steps=(8000,), triggered=False
):
    manifest = bank.load_world_bank()
    context = {
        "profile": SimpleNamespace(digest=manifest["profile"]["base_profile_digest"]),
        "lookup": {gid.FRP3_S12[1]: ("checkpoint", "carrier")},
    }

    def metadata(path, *args, **kwargs):
        return {"sha256": hashlib.sha256(str(path).encode()).hexdigest(), "path": str(path)}

    monkeypatch.setattr(evaluate, "checkpoint_training_metadata", metadata)
    monkeypatch.setattr(
        evaluate, "checkpoint_metadata", lambda *a, **kw: {"sha256": evaluate.B8_SHA256}
    )

    def world_result(world, checkpoint, *args, **kwargs):
        return {
            "schema": "t23-eval-row/v1",
            "world_id": world["world_semantic_sha256"],
            "world_semantic_sha256": world["world_semantic_sha256"],
            "world_seed": world["world_seed"],
            "mix": world["mix"],
            "mass_integral": 100 + (delta if str(checkpoint) != "b8" else 0),
            "alive_frame_fraction": 1.0,
            "no_death_episode": True,
            "no_death_episode_survival": True,
            "horizon": 5000,
            "frames_stepped": 5000,
        }

    monkeypatch.setattr(evaluate, "evaluate_world", world_result)
    provenance = {
        "clean": True,
        "tripwire_triggered": triggered,
        "problems": [],
        "summary_sha256": "1" * 64,
        "manifest_sha256": "2" * 64,
        "telemetry_sha256": "3" * 64,
        "run_log_sha256": "4" * 64,
    }
    paths = []
    for seed in evaluate.TRAINING_SEEDS:
        for step in steps:
            for shard in range(3):
                paths.append(
                    evaluate._run_checkpoint(
                        root=tmp_path,
                        bank=manifest,
                        context=context,
                        training_seed=seed,
                        checkpoint=Path(f"seed{seed}-{step}k"),
                        checkpoint_label=f"{step}k",
                        b8=Path("b8"),
                        phase=phase,
                        shard=shard,
                        shards=3,
                        provenance=provenance,
                    )
                )
    return paths


def test_actual_receipt_writer_to_analysis_and_resume(tmp_path, monkeypatch):
    paths = _generate_receipts(tmp_path, monkeypatch)
    result = analyze.analyze(analyze._load(tmp_path / "evaluation"), 2)
    assert result["verdict"] == "GO", result["validation_errors"]
    digest = evaluate.sha256_file(paths[0])
    _generate_receipts(tmp_path, monkeypatch)
    assert evaluate.sha256_file(paths[0]) == digest
    with pytest.raises(RuntimeError, match="mismatched identity"):
        _generate_receipts(tmp_path, monkeypatch, triggered=True)
    receipt = json.loads(paths[0].read_text())
    receipt["rows"][0]["mass_integral"] += 1
    paths[0].write_text(json.dumps(receipt))
    assert analyze.analyze(analyze._load(tmp_path / "evaluation"), 2)["verdict"] == "INVALID"


def test_actual_extension_combines_frozen_initial_receipts(tmp_path, monkeypatch):
    _generate_receipts(tmp_path, monkeypatch, delta=10, steps=(2000, 4000, 8000))
    root = tmp_path / "evaluation"
    prior = analyze.analyze(analyze._load(root), 2)
    assert prior["verdict"] == "EXTEND"
    authorization = {
        "prior_verdict": "EXTEND",
        "prior_result": prior,
        "prior_result_sha256": hashlib.sha256(evaluate.canonical(prior).encode()).hexdigest(),
        "frozen_checkpoint_sha256s": prior["frozen_checkpoint_sha256s"],
    }
    (root / "extension_authorization.json").write_text(json.dumps(authorization))
    _generate_receipts(tmp_path, monkeypatch, delta=10, phase="extension")
    result = analyze.analyze(analyze._load(root), 2)
    assert result["verdict"] == "INCONCLUSIVE", result["validation_errors"]
    authorization["prior_result"]["verdict"] = "INCONCLUSIVE"
    (root / "extension_authorization.json").write_text(json.dumps(authorization))
    assert analyze.analyze(analyze._load(root), 2)["verdict"] == "INVALID"
