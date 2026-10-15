#!/usr/bin/env python3
"""Analyze the pre-registered T3 label-bank development screen.

This module deliberately extends the T2 probe row contract.  The primary
uncertainty is the seed-paired standard error, never the pooled-world SE.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from research.t2_screen_20261015.analyze_t2 import _mean, _number_map, _se, _survival_maps

ARMS = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
MIXES = ("scripted", "frozen", "mixed")
SEEDS = tuple(range(10, 15))
INITIAL_ROUND = 62
EXTENSION_ROUND = 63
FRAMES = 5000


def _group(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    result: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row["name"]), int(row["round_index"]), str(row["mix"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every row needs name, round_index, and mix") from exc
        if key in result:
            raise ValueError("duplicate probe row: %r" % (key,))
        result[key] = row
    return result


def _metric(row: Mapping[str, Any], metric: str) -> Dict[str, float]:
    if metric == "episode_survival":
        return _survival_maps(row)[0]
    if metric == "alive_survival":
        return _survival_maps(row)[1]
    if metric == "death_length":
        raw = row.get("death_lengths", row.get("length_at_death", {}))
        return {str(key): float(value) for key, value in raw.items() if value is not None}
    return _number_map(row, metric)


def _joined(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], name: str, mix: str, metric: str
) -> Dict[str, float]:
    result: Dict[str, float] = {}
    for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
        row = grouped.get((name, round_index, mix))
        if row is None:
            continue
        values = _metric(row, metric)
        overlap = set(result) & set(values)
        if overlap:
            raise ValueError("worlds repeated across rounds for %s/%s" % (name, mix))
        result.update(values)
    return result


def _paired(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str, metric: str
) -> Dict[str, Any]:
    per_seed: List[float] = []
    worlds: List[float] = []
    seeds: List[int] = []
    for seed in SEEDS:
        candidate: Dict[str, float] = {}
        control: Dict[str, float] = {}
        for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
            candidate_row = grouped.get(("%s_s%d" % (arm, seed), round_index, mix))
            control_row = grouped.get(("CTRL_s%d" % seed, round_index, mix))
            if candidate_row is None or control_row is None:
                continue
            candidate_values = _metric(candidate_row, metric)
            control_values = _metric(control_row, metric)
            if set(candidate_values) != set(control_values):
                raise ValueError("candidate and CTRL worlds must match for paired contrasts")
            candidate.update(candidate_values)
            control.update(control_values)
        shared = sorted(set(candidate) & set(control))
        if shared:
            values = [candidate[key] - control[key] for key in shared]
            per_seed.append(_mean(values))
            worlds.extend(values)
            seeds.append(seed)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(worlds),
        "seeds": seeds,
        "seed_n": len(per_seed),
        "world_n": len(worlds),
        "complete": seeds == list(SEEDS),
    }


def _incumbent(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str, metric: str
) -> Dict[str, Any]:
    incumbent_row = grouped.get(("baseline", INITIAL_ROUND, mix))
    baseline = _metric(incumbent_row, metric) if incumbent_row else {}
    per_seed: List[float] = []
    worlds: List[float] = []
    for seed in SEEDS:
        row = grouped.get(("%s_s%d" % (arm, seed), INITIAL_ROUND, mix))
        candidate = _metric(row, metric) if row else {}
        if candidate and baseline and set(candidate) != set(baseline):
            raise ValueError("candidate and incumbent worlds must match")
        shared = sorted(set(candidate) & set(baseline))
        if shared:
            values = [candidate[key] - baseline[key] for key in shared]
            per_seed.append(_mean(values))
            worlds.extend(values)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(worlds),
        "seed_n": len(per_seed),
        "world_n": len(worlds),
    }


def _death_distribution(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], name: str, mix: str
) -> Dict[str, Any]:
    """Describe only deaths for one policy; death events cannot be paired rowwise."""
    values: List[float] = []
    for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
        row = grouped.get((name, round_index, mix))
        if row is None:
            continue
        deaths = _number_map(row, "deaths")
        lengths = _metric(row, "death_length")
        values.extend(lengths[key] for key, death in deaths.items() if death and key in lengths)
    values.sort()
    if not values:
        return {
            "death_count": 0,
            "mean": float("nan"),
            "median": float("nan"),
            "q25": float("nan"),
            "q75": float("nan"),
        }

    def quantile(fraction: float) -> float:
        return values[round((len(values) - 1) * fraction)]

    return {
        "death_count": len(values),
        "mean": _mean(values),
        "median": quantile(0.5),
        "q25": quantile(0.25),
        "q75": quantile(0.75),
    }


def _death_lengths(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str
) -> Dict[str, Any]:
    candidates = [_death_distribution(grouped, "%s_s%d" % (arm, seed), mix) for seed in SEEDS]
    controls = [_death_distribution(grouped, "CTRL_s%d" % seed, mix) for seed in SEEDS]
    return {
        "candidate": candidates,
        "ctrl": controls,
        "candidate_mean": _mean([x["mean"] for x in candidates if math.isfinite(x["mean"])]),
        "ctrl_mean": _mean([x["mean"] for x in controls if math.isfinite(x["mean"])]),
    }


def _telemetry_for(telemetry: Mapping[str, Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    runs = {"%s_s%d" % (arm, seed): telemetry.get("%s_s%d" % (arm, seed), {}) for seed in SEEDS}
    final = [
        float(value["greedy_boost_rate"])
        for value in runs.values()
        if value.get("greedy_boost_rate") is not None
    ]

    def bank_value(value: Any) -> float:
        return float(value["value"]) if isinstance(value, Mapping) else float(value)

    bank2 = [
        bank_value(value["bank_agree_2m"])
        for value in runs.values()
        if value.get("bank_agree_2m") is not None
    ]
    bank8 = [
        bank_value(value["bank_agree_8m"])
        for value in runs.values()
        if value.get("bank_agree_8m") is not None
    ]
    return {
        "greedy_boost_rate_mean": _mean(final),
        "greedy_boost_rate_n": len(final),
        "bank_agree_2m_mean": _mean(bank2),
        "bank_agree_2m_n": len(bank2),
        "bank_agree_8m_mean": _mean(bank8),
        "bank_agree_8m_n": len(bank8),
        "curves": {name: value.get("greedy_boost_rate_curve", []) for name, value in runs.items()},
    }


def _decision(data: Mapping[str, Any], statuses: Mapping[str, int], arm: str) -> str:
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    if any(statuses.get("%s_s%d" % (arm, seed)) != 0 for seed in SEEDS):
        return "KILL"
    if not math.isfinite(primary):
        return "INCOMPLETE"
    if primary <= 0:
        return "KILL"
    if not all(
        data["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    ):
        return "INCOMPLETE"
    masses = [data["vs_ctrl"][mix]["mass"]["mean"] for mix in MIXES]
    other = [data["vs_ctrl"][mix]["episode_survival"]["mean"] for mix in ("frozen", "mixed")]
    if (
        primary >= 0.06
        and all(value >= -30 for value in masses)
        and all(value >= -0.05 for value in other)
    ):
        return "GO"
    return "EXTEND"


def _validate(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    expected = {"%s_s%d" % (arm, seed) for arm in ARMS for seed in SEEDS}
    if set(statuses) != expected or any(not isinstance(value, int) for value in statuses.values()):
        raise ValueError("T3 analysis requires integer terminal statuses for all 15 runs")
    required = {("CTRL_s%d" % seed, INITIAL_ROUND, mix) for seed in SEEDS for mix in MIXES}
    required |= {("baseline", INITIAL_ROUND, mix) for mix in MIXES}
    for arm in ARMS:
        for seed in SEEDS:
            if statuses["%s_s%d" % (arm, seed)] == 0:
                required |= {("%s_s%d" % (arm, seed), INITIAL_ROUND, mix) for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("initial T3 analysis requires all successful initial rows")
    for row in grouped.values():
        values = [_number_map(row, key) for key in ("mass", "deaths", "alive_frames")]
        if not values[0] or any(set(value) != set(values[0]) for value in values):
            raise ValueError("probe world metrics are missing or inconsistent")
        if row.get("frames") != FRAMES:
            raise ValueError("T3 requires the complete H5000 horizon")


def analyze(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, int],
    telemetry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    rerolls: Optional[Mapping[str, Mapping[str, Any]]] = None,
) -> Dict[str, Any]:
    """Apply T3's binding decision rule to terminal statuses and T2-compatible rows."""
    failed = {name for name, code in statuses.items() if code != 0}
    grouped = _group(row for row in rows if str(row.get("name")) not in failed)
    _validate(grouped, statuses)
    telemetry = telemetry or {}
    rerolls = rerolls or {}
    arms: Dict[str, Any] = {}
    for arm in ARMS:
        data: Dict[str, Any] = {
            "vs_ctrl": {
                mix: {
                    metric: _paired(grouped, arm, mix, metric)
                    for metric in ("episode_survival", "alive_survival", "mass")
                }
                for mix in MIXES
            },
            "mass_vs_incumbent": {mix: _incumbent(grouped, arm, mix, "mass") for mix in MIXES},
            "death_lengths": {mix: _death_lengths(grouped, arm, mix) for mix in MIXES},
            "telemetry": _telemetry_for(telemetry, arm),
            "reroll": {
                key: _mean(
                    [
                        float(value[key])
                        for name, value in rerolls.items()
                        if name.startswith(arm + "_s") and value.get(key) is not None
                    ]
                )
                for key in (
                    "labelled_survivor_action_survival_fraction",
                    "final_raw_argmax_in_labelled_surviving_set_fraction",
                    "final_advisory_argmax_in_labelled_surviving_set_fraction",
                )
            },
            "failed_runs": [
                {"name": "%s_s%d" % (arm, seed), "exit": statuses["%s_s%d" % (arm, seed)]}
                for seed in SEEDS
                if statuses["%s_s%d" % (arm, seed)] != 0
            ],
            "round_indices": sorted(
                {rnd for name, rnd, _ in grouped if name.startswith(arm + "_s")}
            ),
        }
        data["extension_complete"] = (
            all(
                ("%s_s%d" % (arm, seed), EXTENSION_ROUND, "scripted") in grouped
                and ("CTRL_s%d" % seed, EXTENSION_ROUND, "scripted") in grouped
                for seed in SEEDS
            )
            and not data["failed_runs"]
        )
        data["extension_started"] = any(
            rnd == EXTENSION_ROUND and name.startswith(arm + "_s") for name, rnd, _ in grouped
        )
        data["decision"] = _decision(data, statuses, arm)
        if data["extension_started"] and not data["extension_complete"] and not data["failed_runs"]:
            data["decision"] = "EXTEND"
        data["extension_exhausted"] = data["extension_complete"] and data["decision"] == "EXTEND"
        arms[arm] = data
    result: Dict[str, Any] = {
        "study": "T3 label-bank screen",
        "evidence": "development screen, not gate evidence",
        "disclosure": (
            "labels came from simulator rollouts of B8; the served artifact stays a bare network"
        ),
        "initial_round_index": INITIAL_ROUND,
        "extension_round_index": EXTENSION_ROUND,
        "terminal_statuses": dict(statuses),
        "arms": arms,
        "telemetry_by_run": dict(telemetry),
        "rerolls_by_run": dict(rerolls),
        "attribution": "descriptive comparison pending complete endpoints",
    }
    result["attribution"] = attribution(arms)
    if any(data["extension_started"] for data in arms.values()):
        initial = analyze(
            [row for row in rows if int(row["round_index"]) == INITIAL_ROUND],
            statuses,
            telemetry,
            rerolls,
        )
        for arm, data in arms.items():
            data["initial_decision"] = initial["arms"][arm]["decision"]
    return result


def attribution(arms: Mapping[str, Any]) -> str:
    """Apply registered cases without inventing a numerical similarity threshold."""
    primary = {
        arm: data["vs_ctrl"]["scripted"]["episode_survival"]["mean"] for arm, data in arms.items()
    }
    decisions = {arm: data["decision"] for arm, data in arms.items()}
    if decisions["LABELS"] == decisions["LABELS_KEEPV8"] == "GO":
        note = "Both scheduled arms GO: under the registered attribution rule, labels carry the effect."
    elif primary["LABELS"] > 0 and primary["LABELS_KEEPV8"] <= 0:
        note = (
            "Only LABELS moves positively among the two scheduled arms: the registered "
            "interpretation favors the anchor regime, not an isolated label effect."
        )
    else:
        note = (
            "The registered decisive attribution cases do not establish an isolated label "
            "effect. Similarity had no registered numerical cutoff; compare the reported "
            "paired deltas descriptively."
        )
    if decisions["LABELS_KEEPV8_LO"] == "GO":
        note += (
            " LABELS_KEEPV8_LO also GO: label contents meet the screen under CTRL's low "
            "anchor weight without the initial high-weight schedule."
        )
    else:
        note += (
            " LABELS_KEEPV8_LO does not meet GO under CTRL's low anchor weight; its "
            "endpoint does not establish that changing label contents alone is sufficient."
        )
    note += " Scripted deltas: " + ", ".join(f"{arm}={primary[arm]:+.4f}" for arm in ARMS) + "."
    if any(data["failed_runs"] for data in arms.values()):
        note += " Halted arms have incomplete numerical comparisons; interpret these descriptively."
    return note


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    grouped = _group(rows)
    jobs: List[Dict[str, Any]] = []
    seen = set()
    for arm, data in result["arms"].items():
        if data["decision"] != "EXTEND" or data.get("extension_complete"):
            continue
        for seed in SEEDS:
            for name in ("%s_s%d" % (arm, seed), "CTRL_s%d" % seed):
                key = (name, EXTENSION_ROUND, "scripted")
                if key not in grouped and key not in seen:
                    jobs.append(
                        {
                            "name": name,
                            "round_index": EXTENSION_ROUND,
                            "mix": "scripted",
                            "worlds": 96,
                        }
                    )
                    seen.add(key)
    return jobs


def _nearest(
    lines: Sequence[Mapping[str, Any]],
    target: int,
    field: str,
    *,
    crossing: bool = False,
    bound: int = 100_000,
) -> Optional[Dict[str, float]]:
    candidates = [
        row for row in lines if row.get(field) is not None and row.get("transitions") is not None
    ]
    if crossing:
        candidates = [
            row for row in candidates if target <= int(row["transitions"]) < target + bound
        ]
    else:
        candidates = [row for row in candidates if abs(int(row["transitions"]) - target) <= bound]
    if not candidates:
        return None
    row = (
        min(candidates, key=lambda value: int(value["transitions"]))
        if crossing
        else min(candidates, key=lambda value: abs(int(value["transitions"]) - target))
    )
    return {
        "transitions": int(row["transitions"]),
        "value": float(row[field]),
        "target": target,
        "bound": bound,
    }


def read_telemetry(
    run_dirs: Union[Sequence[Path], Mapping[str, Path]],
) -> Dict[str, Dict[str, Any]]:
    """Read required bank_agree checkpoints and greedy boost curve from training logs."""
    output: Dict[str, Dict[str, Any]] = {}
    roots = (
        run_dirs.items() if isinstance(run_dirs, Mapping) else ((None, root) for root in run_dirs)
    )
    for supplied_name, root in roots:
        root = Path(root)
        paths = [root / "log.jsonl"] if supplied_name is not None else root.glob("*_s*/log.jsonl")
        for log_path in paths:
            if not log_path.exists():
                continue
            lines = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
            boost = [
                {"transitions": int(row["transitions"]), "value": float(row["greedy_boost_rate"])}
                for row in lines
                if row.get("greedy_boost_rate") is not None
            ]
            output[supplied_name or log_path.parent.name] = {
                "greedy_boost_rate": boost[-1]["value"] if boost else None,
                "greedy_boost_rate_curve": boost,
                "bank_agree_2m": _nearest(lines, 2_000_000, "bank_agree"),
                "bank_agree_8m": _nearest(lines, 8_000_000, "bank_agree", crossing=True),
            }
    return output


def _fmt(value: Mapping[str, Any]) -> str:
    return "%+.3f (seed SE %.3f; world SE %.3f)" % (
        value["mean"],
        value["seed_se"],
        value["pooled_world_se"],
    )


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# T3 development screen, not gate evidence (2026-10-15)",
        "",
        "**Development screen, not gate evidence.**",
        "",
        "Labels are simulator rollouts of B8 in the `t3-rollout-labels-dense/v1` development "
        "namespace, pooled across B8 seeds; the served artifact stays a bare network. LABELS "
        "has an early bank-only trust region, while LABELS_KEEPV8_LO keeps CTRL's low anchor "
        "weight and changes anchor contents.",
        "",
        "## Decisions",
        "",
        "| arm | initial / final | scripted episode delta vs CTRL | scripted mass delta vs CTRL | "
        "failed runs |",
        "|---|---|---:|---:|---|",
    ]
    for arm, data in result["arms"].items():
        script = data["vs_ctrl"]["scripted"]
        failed = (
            ", ".join("%s exit=%s" % (x["name"], x["exit"]) for x in data["failed_runs"]) or "none"
        )
        lines.append(
            "| %s | %s / %s | %s | %s | %s |"
            % (
                arm,
                data.get("initial_decision", data["decision"]),
                data["decision"],
                _fmt(script["episode_survival"]),
                _fmt(script["mass"]),
                failed,
            )
        )
    lines += ["", "### Greedy boost rate per 100k", "", "| run | transition:value |", "|---|---|"]
    for name, data in sorted(result.get("telemetry_by_run", {}).items()):
        curve = data.get("greedy_boost_rate_curve", [])
        rendered = ", ".join("%s:%.6f" % (item["transitions"], item["value"]) for item in curve)
        lines.append("| %s | %s |" % (name, rendered or "missing"))
    lines += [
        "",
        "GO requires primary >= +0.06, mass delta >= -30 on every mix, and no frozen/mixed "
        "episode-survival drop beyond -0.05. KILL applies to primary <= 0 or any halted run. "
        "Otherwise one automatic round-63 scripted extension (96 worlds for arm and CTRL) is "
        "allowed, then stops.",
        "",
        "## Secondary outcomes",
        "",
        "| arm | mix | episode delta CTRL | alive-frame delta CTRL | mass delta CTRL | mass "
        "delta incumbent | death mean candidate / CTRL |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        for mix in MIXES:
            ctrl = data["vs_ctrl"][mix]
            lines.append(
                "| %s | %s | %s | %s | %s | %s | %s |"
                % (
                    arm,
                    mix,
                    _fmt(ctrl["episode_survival"]),
                    _fmt(ctrl["alive_survival"]),
                    _fmt(ctrl["mass"]),
                    _fmt(data["mass_vs_incumbent"][mix]),
                    "%.1f / %.1f"
                    % (
                        data["death_lengths"][mix]["candidate_mean"],
                        data["death_lengths"][mix]["ctrl_mean"],
                    ),
                )
            )
    lines += [
        "",
        "### Death-length distributions",
        "",
        "Statistics below average the per-training-seed death distributions; they are not "
        "quantiles of a pooled death sample.",
        "",
        "| arm | mix | policy | deaths | mean | mean median | mean q25 | mean q75 |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        for mix in MIXES:
            lengths = data["death_lengths"][mix]
            for policy in ("candidate", "ctrl"):
                distributions = lengths[policy]
                deaths = sum(item["death_count"] for item in distributions)
                finite = {
                    key: _mean([item[key] for item in distributions if math.isfinite(item[key])])
                    for key in ("mean", "median", "q25", "q75")
                }
                lines.append(
                    "| %s | %s | %s | %d | %.1f | %.1f | %.1f | %.1f |"
                    % (
                        arm,
                        mix,
                        policy,
                        deaths,
                        finite["mean"],
                        finite["median"],
                        finite["q25"],
                        finite["q75"],
                    )
                )
    lines += [
        "",
        "## Training telemetry and re-roll",
        "",
        "| arm | final greedy boost rate | bank_agree 2M | bank_agree 8M | re-roll labelled "
        "survivors | re-roll argmax in survivor set |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        telemetry, reroll = data["telemetry"], data["reroll"]
        lines.append(
            "| %s | %s | %s | %s | %s | %s |"
            % (
                arm,
                *(
                    "missing" if not math.isfinite(float(x)) else "%.4f" % float(x)
                    for x in (
                        telemetry["greedy_boost_rate_mean"],
                        telemetry["bank_agree_2m_mean"],
                        telemetry["bank_agree_8m_mean"],
                    )
                ),
                reroll.get("labelled_survivor_action_survival_fraction", "missing"),
                reroll.get("final_raw_argmax_in_labelled_surviving_set_fraction", "missing"),
            )
        )
    lines += ["", "## Per-final re-roll", ""]
    lines += [
        "| final | survivor action survival | raw argmax in labelled surviving set | "
        "advisory argmax in labelled surviving set |",
        "|---|---:|---:|---:|",
    ]
    for name, reroll in sorted(result.get("rerolls_by_run", {}).items()):
        lines.append(
            "| %s | %s | %s | %s |"
            % (
                name,
                reroll.get("labelled_survivor_action_survival_fraction", "missing"),
                reroll.get("final_raw_argmax_in_labelled_surviving_set_fraction", "missing"),
                reroll.get("final_advisory_argmax_in_labelled_surviving_set_fraction", "missing"),
            )
        )
    lines += [
        "",
        "Attribution across all three arms: %s." % result["attribution"],
        "",
        "Seed-level paired SE is the decision uncertainty; pooled-world SE is descriptive. "
        "`bank_agree` samples are bound to a 100k transition window: 2M is nearest and 8M is "
        "the first post-target observation. The 2.0-2.3M transient is not interpreted. This "
        "screen neither promotes nor changes serving.",
        "",
    ]
    return "\n".join(lines)


def write_results(result: Mapping[str, Any], json_path: Path, markdown_path: Path) -> None:
    def clean(value: Any) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            return None
        if isinstance(value, dict):
            return {key: clean(item) for key, item in value.items()}
        if isinstance(value, list):
            return [clean(item) for item in value]
        return value

    Path(json_path).write_text(json.dumps(clean(result), indent=2, allow_nan=False) + "\n")
    Path(markdown_path).write_text(markdown(result))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True)
    parser.add_argument("--statuses", type=Path, required=True)
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--rerolls", type=Path)
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    text = args.rows.read_text().strip()
    rows = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )

    def load(path: Optional[Path]) -> Mapping[str, Any]:
        return json.loads(path.read_text()) if path else {}

    result = analyze(rows, load(args.statuses), load(args.telemetry), load(args.rerolls))
    result["extension_requirements"] = extension_requirements(result, rows)
    write_results(result, args.json_out, args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
