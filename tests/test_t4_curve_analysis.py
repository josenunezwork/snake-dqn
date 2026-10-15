"""Contracts for the pure T3 round-64 checkpoint curve analysis."""

from __future__ import annotations

import copy

import pytest

from research.t4_curve_20261015 import analyze


def _row(name: str, effect: float, *, round_index: int = 64) -> dict:
    return {
        "schema": "t2-screen-probe/v1",
        "name": name,
        "round_index": round_index,
        "mix": "scripted",
        "veto": "none",
        "worlds": 32,
        "frames": 5000,
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory",
        "v61_backend": "numba",
        "anchors": "fast",
        "mass": {world: 100.0 + effect for world in analyze.EXPECTED_WORLD_SEEDS},
        "deaths": {world: 0 if effect > 0 else 1 for world in analyze.EXPECTED_WORLD_SEEDS},
        "alive_frames": {
            world: 5000 if effect > 0 else 2500 for world in analyze.EXPECTED_WORLD_SEEDS
        },
    }


def _complete_rows() -> list[dict]:
    rows = []
    candidate_effect = {"2M": 0.25, "4M": 0.8, "6M": 0.9, "8M": 1.0}
    for arm in analyze.ARMS:
        for seed in analyze.SEEDS:
            for checkpoint, effect in candidate_effect.items():
                rows.append(_row("%s_s%d__%s" % (arm, seed, checkpoint), effect))
    for seed in analyze.SEEDS:
        rows.append(_row("CTRL_s%d__4M" % seed, 0.0))
        rows.append(_row("CTRL_s%d__8M" % seed, 0.0))
    return rows


def test_complete_curve_uses_fixed_ctrl8_and_reports_matched_4m() -> None:
    result = analyze.analyze(_complete_rows())
    arm = result["arms"]["LABELS"]
    assert result["row_count"] == 70
    assert arm["fixed_ctrl_8m_effects"]["4M"]["episode_survival"]["mean"] == 1.0
    assert arm["matched_4m_effect"]["episode_survival"]["mean"] == 1.0
    assert arm["absolute_metrics"]["8M"]["mass"]["mean"] == 101.0
    assert result["absolute_metrics"]["controls"]["4M"]["mass"]["mean"] == 100.0
    assert result["ctrl_4m_minus_ctrl_8m"]["mass"]["mean"] == 0.0
    assert arm["fixed_reference_4m_over_8m_ratios"]["episode_survival"] == {
        "ratio": 1.0,
        "status": "AT_LEAST_80_PERCENT",
        "carries_at_least_80_percent": True,
    }
    assert arm["matched_reference_4m_over_8m_ratios"]["mass"]["ratio"] == pytest.approx(0.8)
    table = analyze.markdown(result)
    assert "matched 4M vs CTRL-4M" in table
    assert "CTRL-8M" in table


def test_nonpositive_8m_is_not_a_ratio_pass() -> None:
    rows = _complete_rows()
    for row in rows:
        if row["name"].startswith("LABELS_s") and row["name"].endswith("__8M"):
            row["deaths"] = {world: 1 for world in analyze.EXPECTED_WORLD_SEEDS}
            row["alive_frames"] = {world: 0 for world in analyze.EXPECTED_WORLD_SEEDS}
    result = analyze.analyze(rows)
    verdict = result["arms"]["LABELS"]["four_m_verdict"]
    assert verdict["status"] == "NON_DEMONSTRATED_NONPOSITIVE_8M"
    assert verdict["ratio"] is None
    assert verdict["carries_at_least_80_percent"] is False


@pytest.mark.parametrize(
    "mutate, match",
    [
        (lambda rows: rows.pop(), "exactly 70"),
        (lambda rows: rows.__setitem__(0, {**rows[0], "round_index": 62}), "round_index"),
        (
            lambda rows: rows.__setitem__(
                0, {**rows[0], "mass": {world: 1 for world in analyze.EXPECTED_WORLD_SEEDS[:-1]}}
            ),
            "inconsistent world metrics",
        ),
        (lambda rows: rows[0].__setitem__("veto", "v8"), "incompatible veto"),
        (lambda rows: rows[0].__setitem__("v61_backend", "python"), "incompatible v61_backend"),
    ],
)
def test_analysis_rejects_incomplete_or_wrong_probe_contract(mutate, match: str) -> None:
    rows = copy.deepcopy(_complete_rows())
    mutate(rows)
    with pytest.raises(ValueError, match=match):
        analyze.analyze(rows)


def test_analysis_requires_same_world_identities_across_every_row() -> None:
    rows = _complete_rows()
    rows[1]["mass"] = {world + 1000: value for world, value in rows[1]["mass"].items()}
    rows[1]["deaths"] = {world + 1000: value for world, value in rows[1]["deaths"].items()}
    rows[1]["alive_frames"] = {
        world + 1000: value for world, value in rows[1]["alive_frames"].items()
    }
    with pytest.raises(ValueError, match="registered round-64 world seeds"):
        analyze.analyze(rows)


def test_analysis_rejects_nonfinite_probe_values() -> None:
    rows = _complete_rows()
    rows[0]["mass"][analyze.EXPECTED_WORLD_SEEDS[0]] = float("nan")
    with pytest.raises(ValueError, match="non-finite mass"):
        analyze.analyze(rows)
