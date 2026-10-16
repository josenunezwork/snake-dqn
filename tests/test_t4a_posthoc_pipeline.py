"""Readiness and aggregation boundaries for the post-hoc report."""

import json

import pytest

from research.t4a_posthoc_20261016 import pipeline


def test_readiness_requires_all_successful_terminal_finals(tmp_path) -> None:
    for seed in range(10, 15):
        folder = tmp_path / f"TWO_SIDED_s{seed}"
        folder.mkdir()
        (folder / "final.pth").write_bytes(b"fixture")
        (folder / "summary.json").write_text("{}")
    assert not pipeline.readiness(tmp_path)["ready"]
    (tmp_path / "lane0-status.txt").write_text(
        "TWO_SIDED_s10 exit=0\nTWO_SIDED_s12 exit=0\nTWO_SIDED_s14 exit=0\n"
    )
    (tmp_path / "lane1-status.txt").write_text("TWO_SIDED_s11 exit=0\nTWO_SIDED_s13 exit=1\n")
    assert not pipeline.readiness(tmp_path)["ready"]
    (tmp_path / "lane1-status.txt").write_text("TWO_SIDED_s11 exit=0\nTWO_SIDED_s13 exit=0\n")
    assert pipeline.readiness(tmp_path)["ready"]
    (tmp_path / "TWO_SIDED_s14" / "final.pth").unlink()
    assert not pipeline.readiness(tmp_path)["ready"]


def test_receipt_rejects_changed_input(tmp_path) -> None:
    source = tmp_path / "source"
    source.write_bytes(b"original")
    path = tmp_path / "receipt.json"
    path.write_text(
        json.dumps(
            {"schema": "fixture", "inputs_sha256": {str(source): pipeline.spillover.sha256(source)}}
        )
    )
    pipeline._receipt(path, "fixture")
    source.write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        pipeline._receipt(path, "fixture")


def test_growth_pools_survivors_not_seed_means() -> None:
    rows = []
    for count, value in ((1, 10), (3, 30), (0, None)):
        curve = {
            str(frame): {
                "shared_survivors": count,
                "candidate_mean_alive_length": value,
                "ctrl_mean_alive_length": value,
                "candidate_minus_ctrl_mean_alive_length": 0 if count else None,
            }
            for frame in pipeline.spillover.GROWTH_FRAMES
        }
        rows.append({"frames": curve, "fixed_h5000_shared_survivor_frames": curve})
    pooled = pipeline._pool_growth(rows)
    assert pooled["frames"]["1000"]["shared_survivors"] == 4
    assert pooled["frames"]["1000"]["candidate_mean_alive_length"] == 25
