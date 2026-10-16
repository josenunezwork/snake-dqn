"""Pure analysis for the T3 round-64 checkpoint-length diagnostic.

This is deliberately descriptive.  It consumes the 70 completed CPU probe rows
and reports paired candidate-minus-control effects; it does not implement or
alter the registered T3 screen decision.
"""

from __future__ import annotations

import math
import re
from typing import Any, Dict, Mapping, Sequence, Tuple

ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
SEEDS = tuple(range(10, 15))
CANDIDATE_CHECKPOINTS = ("2M", "4M", "6M", "8M")
CONTROL_CHECKPOINTS = ("4M", "8M")
ROUND_INDEX = 64
FRAMES = 5000
WORLDS = 32
EXPECTED_WORLD_SEEDS = (
    1156914137,
    3408819188,
    2758109167,
    748826962,
    3001793538,
    299114136,
    1974794810,
    3528950842,
    2300749571,
    965781687,
    1320593681,
    1710811116,
    3480050714,
    786256828,
    634261790,
    3274222233,
    4235700269,
    245304655,
    139629245,
    184802272,
    3386420751,
    2056287200,
    2783434160,
    1897790469,
    1494893705,
    3525657138,
    1953063860,
    672996920,
    3164975637,
    3018453301,
    1251555711,
    1976234048,
)

_NAME = re.compile(r"^(?P<arm>[A-Z0-9_]+)_s(?P<seed>1[0-4])__(?P<checkpoint>[2468]M)$")


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _seed_se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return 0.0
    average = _mean(values)
    return math.sqrt(
        sum((value - average) ** 2 for value in values) / (len(values) - 1)
    ) / math.sqrt(len(values))


def _number_map(row: Mapping[str, Any], field: str) -> Dict[str, float]:
    raw = row.get(field)
    if not isinstance(raw, Mapping):
        raise ValueError("probe row %r must contain a %s mapping" % (row.get("name"), field))
    try:
        result = {str(world): float(value) for world, value in raw.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("probe row %r has non-numeric %s" % (row.get("name"), field)) from exc
    if any(not math.isfinite(value) for value in result.values()):
        raise ValueError("probe row %r has non-finite %s" % (row.get("name"), field))
    return result


def _metrics(row: Mapping[str, Any]) -> Dict[str, Dict[str, float]]:
    deaths = _number_map(row, "deaths")
    alive = _number_map(row, "alive_frames")
    mass = _number_map(row, "mass")
    worlds = set(mass)
    if not worlds or set(deaths) != worlds or set(alive) != worlds:
        raise ValueError("probe row %r has inconsistent world metrics" % row.get("name"))
    if len(worlds) != WORLDS:
        raise ValueError("probe row %r must contain exactly %d worlds" % (row.get("name"), WORLDS))
    if worlds != {str(seed) for seed in EXPECTED_WORLD_SEEDS}:
        raise ValueError(
            "probe row %r does not use the registered round-64 world seeds" % row.get("name")
        )
    if any(death not in (0.0, 1.0) for death in deaths.values()):
        raise ValueError("probe row %r deaths must be 0 or 1" % row.get("name"))
    if any(value < 0.0 or value > FRAMES for value in alive.values()):
        raise ValueError(
            "probe row %r alive_frames must be within the H5000 horizon" % row.get("name")
        )
    return {
        "episode_survival": {world: 1.0 - deaths[world] for world in worlds},
        "alive_frame_survival": {world: alive[world] / FRAMES for world in worlds},
        "mass": mass,
    }


def _parse_name(name: Any) -> Tuple[str, int, str]:
    match = _NAME.fullmatch(str(name))
    if match is None:
        raise ValueError("invalid checkpoint-curve probe name: %r" % name)
    return match.group("arm"), int(match.group("seed")), match.group("checkpoint")


def _expected_names() -> set[str]:
    candidates = {
        "%s_s%d__%s" % (arm, seed, checkpoint)
        for arm in ARMS
        for seed in SEEDS
        for checkpoint in CANDIDATE_CHECKPOINTS
    }
    controls = {
        "CTRL_s%d__%s" % (seed, checkpoint) for seed in SEEDS for checkpoint in CONTROL_CHECKPOINTS
    }
    return candidates | controls


def _validate_rows(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Mapping[str, Any]]:
    expected = _expected_names()
    if len(rows) != len(expected):
        raise ValueError("T4 curve analysis requires exactly 70 probe rows")
    grouped: Dict[str, Mapping[str, Any]] = {}
    shared_worlds: set[str] | None = None
    for row in rows:
        if not isinstance(row, Mapping):
            raise ValueError("every probe row must be a mapping")
        name = str(row.get("name"))
        arm, _seed, checkpoint = _parse_name(name)
        if arm not in ARMS + ("CTRL",):
            raise ValueError("unregistered curve arm in %r" % name)
        if checkpoint not in (CANDIDATE_CHECKPOINTS if arm != "CTRL" else CONTROL_CHECKPOINTS):
            raise ValueError("unexpected checkpoint identity in %r" % name)
        if name in grouped:
            raise ValueError("duplicate curve probe row: %s" % name)
        if int(row.get("round_index", -1)) != ROUND_INDEX:
            raise ValueError("curve probe %s must use round_index %d" % (name, ROUND_INDEX))
        if row.get("frames") != FRAMES:
            raise ValueError("curve probe %s must use %d frames" % (name, FRAMES))
        expected_fields = {
            "schema": "t2-screen-probe/v1",
            "veto": "none",
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
            "worlds": WORLDS,
            "v61_backend": "numba",
            "anchors": "fast",
        }
        for field, expected_value in expected_fields.items():
            if row.get(field) != expected_value:
                raise ValueError(
                    "curve probe %s has incompatible %s=%r" % (name, field, row.get(field))
                )
        if row.get("mix", "scripted") != "scripted":
            raise ValueError("curve probe %s must use the scripted mix" % name)
        metrics = _metrics(row)
        worlds = set(metrics["mass"])
        if shared_worlds is None:
            shared_worlds = worlds
        elif worlds != shared_worlds:
            raise ValueError("all curve rows must use identical world identities")
        grouped[name] = row
    if set(grouped) != expected:
        missing = sorted(expected - set(grouped))
        extra = sorted(set(grouped) - expected)
        raise ValueError(
            "incomplete or unexpected curve identities; missing=%s extra=%s" % (missing, extra)
        )
    return grouped


def _effect(candidate: Mapping[str, Any], control: Mapping[str, Any]) -> Dict[str, Any]:
    candidate_metrics, control_metrics = _metrics(candidate), _metrics(control)
    output: Dict[str, Any] = {}
    for metric in ("episode_survival", "alive_frame_survival", "mass"):
        candidate_values, control_values = candidate_metrics[metric], control_metrics[metric]
        if set(candidate_values) != set(control_values):
            raise ValueError("paired rows do not share identical world identities")
        output[metric] = _mean(
            [candidate_values[world] - control_values[world] for world in sorted(candidate_values)]
        )
    return output


def _summary(per_seed: Mapping[str, Mapping[str, float]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {
        "per_seed": dict(per_seed),
        "seed_n": len(per_seed),
        "worlds_per_seed": WORLDS,
    }
    for metric in ("episode_survival", "alive_frame_survival", "mass"):
        values = [per_seed[str(seed)][metric] for seed in SEEDS]
        summary[metric] = {"mean": _mean(values), "seed_se": _seed_se(values)}
    return summary


def _absolute(row: Mapping[str, Any]) -> Dict[str, float]:
    return {metric: _mean(list(values.values())) for metric, values in _metrics(row).items()}


def _ratio(effect_4m: float, effect_8m: float) -> Dict[str, Any]:
    if effect_8m <= 0.0:
        return {
            "ratio": None,
            "status": "NON_DEMONSTRATED_NONPOSITIVE_8M",
            "carries_at_least_80_percent": False,
        }
    ratio = effect_4m / effect_8m
    if effect_4m <= 0.0:
        status = "NOT_80_PERCENT_NONPOSITIVE_4M"
    elif ratio >= 0.8:
        status = "AT_LEAST_80_PERCENT"
    else:
        status = "BELOW_80_PERCENT"
    return {
        "ratio": ratio,
        "status": status,
        "carries_at_least_80_percent": status == "AT_LEAST_80_PERCENT",
    }


def analyze(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Analyze all required T3 curve rows with CTRL-8M as the fixed reference.

    Candidate checkpoints at 2M, 4M, 6M and 8M are paired by originating seed
    with CTRL at 8M.  The separately available CTRL-4M result is reported as a
    matched 4M comparison, but does not replace the fixed-reference primary.
    """
    grouped = _validate_rows(rows)
    controls = {
        checkpoint: _summary(
            {str(seed): _absolute(grouped["CTRL_s%d__%s" % (seed, checkpoint)]) for seed in SEEDS}
        )
        for checkpoint in CONTROL_CHECKPOINTS
    }
    control_4m_minus_8m = _summary(
        {
            str(seed): _effect(grouped["CTRL_s%d__4M" % seed], grouped["CTRL_s%d__8M" % seed])
            for seed in SEEDS
        }
    )
    arms: Dict[str, Any] = {}
    for arm in ARMS:
        fixed: Dict[str, Any] = {}
        for checkpoint in CANDIDATE_CHECKPOINTS:
            per_seed = {
                str(seed): _effect(
                    grouped["%s_s%d__%s" % (arm, seed, checkpoint)],
                    grouped["CTRL_s%d__8M" % seed],
                )
                for seed in SEEDS
            }
            fixed[checkpoint] = _summary(per_seed)
        absolute = {
            checkpoint: _summary(
                {
                    str(seed): _absolute(grouped["%s_s%d__%s" % (arm, seed, checkpoint)])
                    for seed in SEEDS
                }
            )
            for checkpoint in CANDIDATE_CHECKPOINTS
        }
        matched_4m = _summary(
            {
                str(seed): _effect(
                    grouped["%s_s%d__4M" % (arm, seed)],
                    grouped["CTRL_s%d__4M" % seed],
                )
                for seed in SEEDS
            }
        )
        ratios = {
            metric: _ratio(fixed["4M"][metric]["mean"], fixed["8M"][metric]["mean"])
            for metric in ("episode_survival", "alive_frame_survival", "mass")
        }
        matched_reference_ratios = {
            metric: _ratio(matched_4m[metric]["mean"], fixed["8M"][metric]["mean"])
            for metric in ("episode_survival", "alive_frame_survival", "mass")
        }
        arms[arm] = {
            "absolute_metrics": absolute,
            "fixed_ctrl_8m_effects": fixed,
            "matched_4m_effect": matched_4m,
            "fixed_reference_4m_over_8m_ratios": ratios,
            "matched_reference_4m_over_8m_ratios": matched_reference_ratios,
            "four_m_verdict": ratios["episode_survival"],
        }
    return {
        "study": "T3 checkpoint effect-versus-length curve",
        "evidence": "diagnostic only; not a T3 screen decision or promotion evidence",
        "round_index": ROUND_INDEX,
        "mix": "scripted",
        "frames": FRAMES,
        "worlds": WORLDS,
        "row_count": len(rows),
        "fixed_reference": "CTRL 8M, paired by seed",
        "matched_reference_note": "CTRL 4M is additionally reported only for the matched 4M comparison.",
        "absolute_metrics": {"controls": controls},
        "ctrl_4m_minus_ctrl_8m": control_4m_minus_8m,
        "arms": arms,
    }


def _format(effect: Mapping[str, float]) -> str:
    return "%+.4f (SE %.4f)" % (effect["mean"], effect["seed_se"])


def markdown(result: Mapping[str, Any]) -> str:
    """Render the compact curve table for the T4 research note."""
    lines = [
        "| arm | checkpoint | episode survival vs CTRL-8M | alive-frame survival vs CTRL-8M | mass vs CTRL-8M |",
        "|---|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        for checkpoint in CANDIDATE_CHECKPOINTS:
            effect = data["fixed_ctrl_8m_effects"][checkpoint]
            lines.append(
                "| %s | %s | %s | %s | %s |"
                % (
                    arm,
                    checkpoint,
                    _format(effect["episode_survival"]),
                    _format(effect["alive_frame_survival"]),
                    _format(effect["mass"]),
                )
            )
        matched = data["matched_4m_effect"]
        verdict = data["four_m_verdict"]
        ratio = verdict["ratio"]
        ratio_text = "n/a" if ratio is None else "%.3f" % ratio
        lines.append(
            "| %s | matched 4M vs CTRL-4M | %s | %s | %s |"
            % (
                arm,
                _format(matched["episode_survival"]),
                _format(matched["alive_frame_survival"]),
                _format(matched["mass"]),
            )
        )
        lines.append(
            "| %s | 4M/8M primary ratio | %s (%s) |  |  |" % (arm, ratio_text, verdict["status"])
        )
    return "\n".join(lines)
