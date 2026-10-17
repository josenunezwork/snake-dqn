"""Receipt-boundary tests for the T10 late-horizon shard driver."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("sklearn")

import research.t10_certificate_horizon_20261017.horizon_shards as horizon_shards  # noqa: E402


def _report(results: dict, *, horizons: list[int], status: str | None = None) -> dict:
    report = {
        "schema": "t10-horizon-v1",
        "authority": "diagnostic only",
        "data_sha256": "data",
        "fold_receipt_sha256": "folds",
        "bootstrap_world_draws": 2,
        "bootstrap": "world",
        "model": "fixed",
        "feature_sets": {"60": 66},
        "source_sha256_before": {},
        "results": results,
    }
    if status is not None:
        report.update({"horizons": horizons, "status": status, "source_sha256_after": {}})
    return report


def _prediction(path, y: np.ndarray) -> None:
    np.savez(
        path,
        y=y,
        state=np.array([0, 1]),
        action=np.array([0, 1]),
        world=np.array(["a", "b"]),
        fold=np.array([0, 1]),
        oof_60=np.array([0.1, 0.9]),
    )


def _launch(path) -> None:
    path.write_text(
        json.dumps(
            {
                "driver": str(Path(horizon_shards.__file__).resolve()),
                "driver_sha256": horizon_shards.sha256(Path(horizon_shards.__file__)),
                "implementation": str(Path(horizon_shards.horizon.__file__).resolve()),
                "implementation_sha256": horizon_shards.sha256(
                    Path(horizon_shards.horizon.__file__)
                ),
            }
        )
    )


def test_combine_marks_interrupted_prefix_and_copies_nonoverlapping_receipts(
    tmp_path, monkeypatch
) -> None:
    prefix, late, combined = tmp_path / "prefix", tmp_path / "late", tmp_path / "combined"
    prefix.mkdir()
    late.mkdir()
    rows = {
        f"D{value}": {"label_counts": {"legal_branches": 2}}
        for value in horizon_shards.PREFIX_HORIZONS
    }
    (prefix / "horizon_fits.json").write_text(json.dumps(_report(rows, horizons=[])))
    for value in horizon_shards.PREFIX_HORIZONS:
        _prediction(prefix / f"D{value}.predictions.npz", np.array([0, 1]))
    horizon_shards.mark_prefix_boundary(prefix)
    late_rows = {
        f"D{value}": {"label_counts": {"legal_branches": 2}}
        for value in horizon_shards.LATE_HORIZONS
    }
    (late / "horizon_fits.json").write_text(
        json.dumps(_report(late_rows, horizons=[200, 300], status="PASS"))
    )
    for value in horizon_shards.LATE_HORIZONS:
        _prediction(late / f"D{value}.predictions.npz", np.array([0, 1]))
    _launch(late / "late_launch.json")
    monkeypatch.setattr(horizon_shards, "_hashes_match_receipt", lambda report: True)
    monkeypatch.setattr(
        horizon_shards, "_artifact_hashes_match_receipt", lambda directory, report: True
    )
    result = horizon_shards.combine(prefix, late, combined)
    assert result["status"] == "PASS"
    assert (
        result["provenance"]["prefix_boundary"]["status"] == "INTENTIONALLY_INTERRUPTED_AFTER_D100"
    )
    assert set(result["results"]) == {"D30", "D60", "D100", "D200", "D300"}
    assert (combined / "D300.predictions.npz").is_file()


def test_combine_rejects_overlap_or_changed_branch_identity(tmp_path, monkeypatch) -> None:
    prefix, late, combined = tmp_path / "prefix", tmp_path / "late", tmp_path / "combined"
    prefix.mkdir()
    late.mkdir()
    prefix_rows = {f"D{value}": {} for value in horizon_shards.PREFIX_HORIZONS}
    late_rows = {f"D{value}": {} for value in horizon_shards.LATE_HORIZONS}
    (prefix / "horizon_fits.json").write_text(json.dumps(_report(prefix_rows, horizons=[])))
    (late / "horizon_fits.json").write_text(
        json.dumps(_report(late_rows, horizons=[200, 300], status="PASS"))
    )
    for value in horizon_shards.PREFIX_HORIZONS:
        _prediction(prefix / f"D{value}.predictions.npz", np.array([0, 1]))
    horizon_shards.mark_prefix_boundary(prefix)
    for value in horizon_shards.LATE_HORIZONS:
        _prediction(late / f"D{value}.predictions.npz", np.array([0, 1]))
    _launch(late / "late_launch.json")
    np.savez(
        late / "D300.predictions.npz",
        y=np.array([0, 1]),
        state=np.array([1, 0]),
        action=np.array([0, 1]),
        world=np.array(["a", "b"]),
        fold=np.array([0, 1]),
        oof_60=np.array([0.1, 0.9]),
    )
    monkeypatch.setattr(horizon_shards, "_hashes_match_receipt", lambda report: True)
    monkeypatch.setattr(
        horizon_shards, "_artifact_hashes_match_receipt", lambda directory, report: True
    )
    with pytest.raises(ValueError, match="branch identity"):
        horizon_shards.combine(prefix, late, combined)


def test_hash_guards_reject_tampered_source_or_saved_output(tmp_path) -> None:
    source, output = tmp_path / "source.py", tmp_path / "D200.predictions.npz"
    source.write_text("original\n")
    output.write_bytes(b"original")
    report = {
        "source_sha256_after": {str(source): horizon_shards.sha256(source)},
        "artifact_sha256": {output.name: horizon_shards.sha256(output)},
    }
    assert horizon_shards._hashes_match_receipt(report)
    assert horizon_shards._artifact_hashes_match_receipt(tmp_path, report)
    source.write_text("changed\n")
    output.write_bytes(b"changed")
    assert not horizon_shards._hashes_match_receipt(report)
    assert not horizon_shards._artifact_hashes_match_receipt(tmp_path, report)
