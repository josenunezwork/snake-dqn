"""Pure receipt-contract coverage for the dense T3 report renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.t3_rollout_labels_20261015 import dense_report


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _row(arm: str, env: int, dense: bool, *, bad_delta: bool = False) -> dict:
    survived = True
    baseline = {"survived": True, "survival_fraction": 1.0, "mass_integral": 10.0}
    return {
        "status": "PASS",
        "prefix_identity": True,
        "arm": arm,
        "env": env,
        "world_seed": 1000 + env,
        "checkpoint": {"sha256": f"{arm}-sha"},
        "source_identity": {"dense-source": "same"} if dense else {"sparse-source": "same"},
        "trigger_mode": "dense" if dense else "sparse",
        "episode_survival": survived,
        "alive_frame_fraction": 1.0,
        "mass_integral": 10.0,
        "baseline": baseline,
        "paired_delta": (
            {}
            if bad_delta
            else {
                "episode_survival": 0,
                "alive_frame_fraction": 0.0,
                "mass_integral": 0.0,
            }
        ),
        "death_t": None,
        "death_length": None,
        "decisions": 5000,
        "trigger_scope_decisions": 5000,
        "raw_trigger_count": 1000,
        "batch_count": 1000,
        "action_changes": 0,
    }


def _receipts(root: Path, *, bad_delta: bool = False, summary_go: bool = True) -> None:
    for dense, location in ((False, root), (True, root / "dense")):
        for arm in dense_report.ARMS:
            for env in dense_report.ENVS:
                _write(
                    location / "ceiling" / f"{arm}_{env}.json",
                    _row(arm, env, dense, bad_delta=bad_delta),
                )
        _write(
            location / "ceiling" / "summary.json",
            {"status": "PASS", "episodes": 64, "go": summary_go, "wall_seconds": 1.0},
        )


def test_dense_report_preserves_sparse_prefix_and_is_idempotent(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "artifacts"
    _receipts(root)
    report_path = tmp_path / "ceiling.md"
    sparse_bytes = b"# Sparse result\n\nExactly as run.\n"
    report_path.write_bytes(sparse_bytes)
    monkeypatch.setattr(dense_report, "REPORT_PATH", report_path)

    first = dense_report.render_dense(root)
    first_document = report_path.read_bytes()
    second = dense_report.render_dense(root)

    assert first["verdict"] == second["verdict"] == "GO"
    assert first_document.startswith(sparse_bytes)
    assert report_path.read_bytes() == first_document
    assert report_path.read_text().count(dense_report.MARKER) == 1


@pytest.mark.parametrize("bad_delta", (True,))
def test_dense_report_rejects_missing_paired_delta(
    tmp_path: Path, monkeypatch, bad_delta: bool
) -> None:
    root = tmp_path / "artifacts"
    _receipts(root, bad_delta=bad_delta)
    report_path = tmp_path / "ceiling.md"
    report_path.write_text("sparse\n")
    monkeypatch.setattr(dense_report, "REPORT_PATH", report_path)

    with pytest.raises(ValueError, match="delta missing"):
        dense_report.render_dense(root)


def test_dense_report_rejects_world_pairing_mismatch(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "artifacts"
    _receipts(root)
    path = root / "dense" / "ceiling" / "b8_s0_0.json"
    row = json.loads(path.read_text())
    row["world_seed"] += 1
    _write(path, row)
    report_path = tmp_path / "ceiling.md"
    report_path.write_text("sparse\n")
    monkeypatch.setattr(dense_report, "REPORT_PATH", report_path)

    with pytest.raises(ValueError, match="world seed mismatch"):
        dense_report.render_dense(root)


def test_dense_report_rejects_summary_go_mismatch(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "artifacts"
    _receipts(root, summary_go=False)
    report_path = tmp_path / "ceiling.md"
    report_path.write_text("sparse\n")
    monkeypatch.setattr(dense_report, "REPORT_PATH", report_path)

    with pytest.raises(ValueError, match="verdict mismatch"):
        dense_report.render_dense(root)
