"""Verify the saved-update interpolation and incomplete-run rejection."""

import json

import pytest

from research.t4e_return_20261016.summarize_calibration import summarize


def test_log_interpolation_and_fail_closed_smoke_evidence(tmp_path):
    for seed in (10, 12):
        for weight in (1.0, 0.2, 0.05, 0.02, 0.01):
            run = tmp_path / f"B8_s{seed-10}_W{weight}"
            run.mkdir()
            (run / "calibration_receipt.json").write_text(
                json.dumps({"trainer_exit": 0, "device": "mps"})
            )
            (run / "summary.json").write_text(
                json.dumps(
                    {
                        "args": {"seed": str(seed), "return_weight": str(weight)},
                        "updates": 300,
                        "halt": None,
                        "actor_exitcodes": [0, 0],
                        "transitions": 78000,
                        "wall_seconds": 1,
                        "flags": [],
                    }
                )
            )
            row = {
                "return_to_td_grad_ratio": 20 * weight,
                "max_abs_q": 3,
                "anchor_rows": 256,
                "return_rows": 128,
                "agent_td_rows": 768,
            }
            (run / "calibration_updates.jsonl").write_text((json.dumps(row) + "\n") * 300)
    result = summarize(tmp_path)
    assert result["chosen_weight"] == 0.025
    assert result["interpolated_weight"] == pytest.approx(0.025)
    assert len(result["runs"]) == 10
    path = run / "summary.json"
    summary = json.loads(path.read_text())
    summary["updates"] = 301
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="update count"):
        summarize(tmp_path)
    summary["updates"] = 300
    summary["actor_exitcodes"] = [0, None]
    path.write_text(json.dumps(summary))
    with pytest.raises(ValueError, match="actor cleanup"):
        summarize(tmp_path)
