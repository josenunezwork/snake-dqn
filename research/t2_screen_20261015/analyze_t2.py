#!/usr/bin/env python3
"""Analyze the T2 learner-regime screen probes.

This is a development-screen report, not promotion/gate evidence.  The input is
deliberately a plain list of probe rows so the watcher can call :func:`analyze`
while probes arrive and the CLI can replay the same calculation from JSONL.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ARMS = ("HUBER", "ANCHOR0", "HUBER_ANCHOR0")
MIXES = ("scripted", "frozen", "mixed")
INITIAL_ROUND = 60
EXTENSION_ROUND = 61
FRAMES = 5000


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def _se(values: Sequence[float]) -> float:
    if len(values) < 2:
        return float("nan")
    avg = _mean(values)
    return math.sqrt(sum((value - avg) ** 2 for value in values) / (len(values) - 1)) / math.sqrt(
        len(values)
    )


def _number_map(row: Mapping[str, Any], key: str) -> Dict[str, float]:
    raw = row.get(key, {})
    if not isinstance(raw, Mapping):
        return {}
    return {str(world): float(value) for world, value in raw.items() if value is not None}


def _survival_maps(row: Mapping[str, Any]) -> Tuple[Dict[str, float], Dict[str, float]]:
    deaths = _number_map(row, "deaths")
    alive = _number_map(row, "alive_frames")
    worlds = set(deaths) | set(alive) | set(_number_map(row, "mass"))
    frames = float(row.get("frames", FRAMES))
    episode = {world: 1.0 if deaths.get(world, 0.0) == 0.0 else 0.0 for world in worlds}
    alive_fraction = {world: alive.get(world, 0.0) / frames for world in worlds}
    return episode, alive_fraction


def _arm_seed(name: str) -> Tuple[str, Optional[int]]:
    if name == "baseline":
        return name, None
    arm, sep, seed = name.rpartition("_s")
    if not sep or not seed.isdigit():
        raise ValueError("probe name must be baseline or ARM_s<seed>: %r" % name)
    return arm, int(seed)


def _group(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    grouped: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            name = str(row["name"])
            round_index = int(row["round_index"])
            mix = str(row["mix"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every row needs name, round_index, and mix") from exc
        key = (name, round_index, mix)
        if key in grouped:
            raise ValueError("duplicate probe row: %r" % (key,))
        grouped[key] = row
    return grouped


def _joined_maps(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], name: str, mix: str, metric: str
) -> Dict[str, float]:
    """Join unique world observations from rounds 60 and 61 for one metric."""
    output: Dict[str, float] = {}
    for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
        row = grouped.get((name, round_index, mix))
        if row is None:
            continue
        if metric == "episode_survival":
            values = _survival_maps(row)[0]
        elif metric == "alive_survival":
            values = _survival_maps(row)[1]
        else:
            values = _number_map(row, metric)
        overlap = set(output) & set(values)
        if overlap:
            raise ValueError("worlds repeated across rounds for %s/%s: %s" % (name, mix, overlap))
        output.update(values)
    return output


def _paired_summary(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]],
    arm: str,
    mix: str,
    metric: str,
) -> Dict[str, Any]:
    """Seed-paired and world-pooled deltas of ``arm`` versus CTRL."""
    seed_values: List[float] = []
    world_values: List[float] = []
    seeds: List[int] = []
    names = sorted({name for name, _, _ in grouped if name.startswith(arm + "_s")})
    for name in names:
        _, seed = _arm_seed(name)
        assert seed is not None
        ctrl_name = "CTRL_s%d" % seed
        candidate: Dict[str, float] = {}
        control: Dict[str, float] = {}
        for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
            candidate_row = grouped.get((name, round_index, mix))
            if candidate_row is None:
                continue
            control_row = grouped.get((ctrl_name, round_index, mix))
            if control_row is None:
                continue
            if metric == "episode_survival":
                candidate_values, control_values = (
                    _survival_maps(candidate_row)[0],
                    _survival_maps(control_row)[0],
                )
            elif metric == "alive_survival":
                candidate_values, control_values = (
                    _survival_maps(candidate_row)[1],
                    _survival_maps(control_row)[1],
                )
            else:
                candidate_values = _number_map(candidate_row, metric)
                control_values = _number_map(control_row, metric)
            candidate.update(candidate_values)
            control.update(control_values)
        shared = sorted(set(candidate) & set(control))
        if not shared:
            continue
        diffs = [candidate[world] - control[world] for world in shared]
        seed_values.append(_mean(diffs))
        world_values.extend(diffs)
        seeds.append(seed)
    expected = list(range(10, 15))
    complete = seeds == expected and len(seed_values) == len(expected)
    return {
        "mean": _mean(seed_values),
        "seed_se": _se(seed_values),
        "pooled_world_se": _se(world_values),
        "seeds": seeds,
        "seed_n": len(seed_values),
        "world_n": len(world_values),
        "complete": complete,
    }


def _versus_baseline(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str
) -> Dict[str, Any]:
    baseline = _joined_maps(grouped, "baseline", mix, "mass")
    per_seed: List[float] = []
    worlds: List[float] = []
    for name in sorted({name for name, _, _ in grouped if name.startswith(arm + "_s")}):
        candidate = _joined_maps(grouped, name, mix, "mass")
        shared = sorted(set(candidate) & set(baseline))
        if shared:
            diffs = [candidate[world] - baseline[world] for world in shared]
            per_seed.append(_mean(diffs))
            worlds.extend(diffs)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(worlds),
        "seed_n": len(per_seed),
        "world_n": len(worlds),
    }


def _telemetry_for(telemetry: Mapping[str, Mapping[str, Any]], arm: str) -> Dict[str, Any]:
    items = [value for name, value in telemetry.items() if name.startswith(arm + "_s")]
    holdouts = [
        float(item["holdout_agreement"])
        for item in items
        if item.get("holdout_agreement") is not None
    ]
    boosts = [
        float(item["greedy_boost_rate"])
        for item in items
        if item.get("greedy_boost_rate") is not None
    ]
    return {
        "holdout_agreement_mean": _mean(holdouts),
        "holdout_agreement_min": min(holdouts) if holdouts else float("nan"),
        "holdout_n": len(holdouts),
        "greedy_boost_rate_mean": _mean(boosts),
        "greedy_boost_n": len(boosts),
    }


def _decision(metrics: Mapping[str, Any], extended: bool) -> str:
    primary = metrics["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    mass = metrics["vs_ctrl"]["scripted"]["mass"]["mean"]
    other = [metrics["vs_ctrl"][mix]["episode_survival"]["mean"] for mix in ("frozen", "mixed")]
    holdout = metrics["telemetry"]["holdout_agreement_min"]
    # Missing holdout data blocks GO but is not an invented collapse observation.
    complete = all(
        metrics["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    )
    if not complete:
        return "INCOMPLETE"
    tripwire = (math.isfinite(holdout) and holdout < 0.6) or any(
        metrics["vs_ctrl"][mix]["mass"]["mean"] < -60.0 for mix in MIXES
    )
    go = (
        primary >= 0.06
        and mass >= -30.0
        and all(value >= -0.05 for value in other)
        and metrics["telemetry"]["holdout_n"] == 5
        and holdout >= 0.6
    )
    if tripwire or primary <= 0.0:
        return "KILL"
    if go:
        return "GO"
    return "EXTEND"


def _interaction(grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]]) -> Dict[str, Any]:
    per_seed: List[float] = []
    pooled: List[float] = []
    for seed in range(10, 15):
        names = {
            "ha": "HUBER_ANCHOR0_s%d" % seed,
            "h": "HUBER_s%d" % seed,
            "a": "ANCHOR0_s%d" % seed,
            "c": "CTRL_s%d" % seed,
        }
        maps = {
            key: _joined_maps(grouped, name, "scripted", "episode_survival")
            for key, name in names.items()
        }
        shared = sorted(set.intersection(*(set(value) for value in maps.values()))) if maps else []
        if shared:
            values = [maps["ha"][w] - maps["h"][w] - maps["a"][w] + maps["c"][w] for w in shared]
            per_seed.append(_mean(values))
            pooled.extend(values)
    return {
        "metric": "scripted_episode_survival",
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(pooled),
        "seed_n": len(per_seed),
        "world_n": len(pooled),
    }


def analyze(
    rows: Sequence[Mapping[str, Any]], telemetry: Optional[Mapping[str, Mapping[str, Any]]] = None
) -> Dict[str, Any]:
    """Return the complete T2 analysis for probe ``rows`` and final telemetry."""
    grouped = _group(rows)
    telemetry = telemetry or {}
    names = [f"{arm}_s{seed}" for arm in ("CTRL",) + ARMS for seed in range(10, 15)]
    required = {(name, INITIAL_ROUND, mix) for name in names + ["baseline"] for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("initial T2 analysis requires all 21 x 3 rows")
    for row in grouped.values():
        fields = [_number_map(row, key) for key in ("mass", "deaths", "alive_frames")]
        if not fields[0] or any(set(field) != set(fields[0]) for field in fields):
            raise ValueError("probe world metrics are missing or inconsistent")
        if row.get("frames") != FRAMES:
            raise ValueError("T2 requires the complete H5000 horizon")
    arms: Dict[str, Any] = {}
    for arm in ARMS:
        by_mix: Dict[str, Any] = {}
        for mix in MIXES:
            by_mix[mix] = {
                metric: _paired_summary(grouped, arm, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
        rounds = sorted(
            {round_index for name, round_index, _ in grouped if name.startswith(arm + "_s")}
        )
        extended = EXTENSION_ROUND in rounds
        metrics = {
            "vs_ctrl": by_mix,
            "mass_vs_baseline": {mix: _versus_baseline(grouped, arm, mix) for mix in MIXES},
            "telemetry": _telemetry_for(telemetry, arm),
            "round_indices": rounds,
        }
        metrics["decision"] = _decision(metrics, extended)
        metrics["extension_exhausted"] = extended and metrics["decision"] == "EXTEND"
        arms[arm] = metrics
    result = {
        "study": "T2 learner-regime screen",
        "evidence": "development screen, not gate evidence",
        "initial_round_index": INITIAL_ROUND,
        "extension_round_index": EXTENSION_ROUND,
        "arms": arms,
        "interaction": _interaction(grouped),
    }
    if any(EXTENSION_ROUND in data["round_indices"] for data in arms.values()):
        initial = analyze(
            [row for row in rows if int(row["round_index"]) == INITIAL_ROUND], telemetry
        )
        for arm, data in arms.items():
            data["initial_decision"] = initial["arms"][arm]["decision"]
    result["telemetry_by_run"] = dict(telemetry)
    return result


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Describe exactly the one allowed extra-bank work, without launching it."""
    grouped = _group(rows)
    needs = [arm for arm, data in result["arms"].items() if data["decision"] == "EXTEND"]
    requirements: List[Dict[str, Any]] = []
    seen = set()
    for arm in needs:
        for seed in result["arms"][arm]["vs_ctrl"]["scripted"]["episode_survival"]["seeds"]:
            for name in ("%s_s%d" % (arm, seed), "CTRL_s%d" % seed):
                key = (name, EXTENSION_ROUND, "scripted")
                if key not in grouped and key not in seen:
                    requirements.append(
                        {
                            "name": name,
                            "round_index": EXTENSION_ROUND,
                            "mix": "scripted",
                            "worlds": 96,
                        }
                    )
                    seen.add(key)
    return requirements


def read_telemetry(run_dirs: Sequence[Path]) -> Dict[str, Dict[str, Any]]:
    """Read final agreement/boost telemetry, using summary agreement as documented fallback."""
    out: Dict[str, Dict[str, Any]] = {}
    for root in run_dirs:
        for summary_path in Path(root).glob("*_s*/summary.json"):
            name = summary_path.parent.name
            summary = json.loads(summary_path.read_text())
            data: Dict[str, Any] = {}
            log_path = summary_path.parent / "log.jsonl"
            log = (
                [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
                if log_path.exists()
                else []
            )
            holds = [
                line.get("holdout_agreement")
                for line in log
                if line.get("holdout_agreement") is not None
            ]
            if holds:
                data["holdout_agreement"] = holds[-1]
                data["holdout_source"] = "log.jsonl:last"
            elif summary.get("holdout_agreement_final") is not None:
                data["holdout_agreement"] = summary["holdout_agreement_final"]
                data["holdout_source"] = "summary.json:holdout_agreement_final"
            boosts = [
                line.get("greedy_boost_rate")
                for line in log
                if line.get("greedy_boost_rate") is not None
            ]
            if boosts:
                data["greedy_boost_rate"] = boosts[-1]
                data["greedy_boost_source"] = "log.jsonl:last"
            out[name] = data
    return out


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# T2 learner-regime screen result (2026-10-15)",
        "",
        "**development screen, not gate evidence.**",
        "",
        "## Design",
        "",
        "Five paired B8 fine-tunes per cell, evaluated against same-seed CTRL on fresh round-60 "
        "scripted (96), frozen (32), and mixed (32) worlds. An EXTEND decision adds one "
        "round-61 scripted bank (96 worlds) for that arm and CTRL only.",
        "",
        "## Decisions",
        "",
        "| arm | decision | scripted episode Δ (seed SE; pooled-world SE) | "
        "alive-frame Δ (seed SE; pooled-world SE) | scripted mass Δ vs CTRL | "
        "holdout agreement (min) |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        ep = data["vs_ctrl"]["scripted"]["episode_survival"]
        alive = data["vs_ctrl"]["scripted"]["alive_survival"]
        mass = data["vs_ctrl"]["scripted"]["mass"]
        hold = data["telemetry"]["holdout_agreement_min"]
        lines.append(
            "| %s | %s | %+.3f (%.3f; %.3f) | %+.3f (%.3f; %.3f) | %+.1f | %s |"
            % (
                arm,
                data["decision"],
                ep["mean"],
                ep["seed_se"],
                ep["pooled_world_se"],
                alive["mean"],
                alive["seed_se"],
                alive["pooled_world_se"],
                mass["mean"],
                f"{hold:.3f}" if math.isfinite(hold) else "missing",
            )
        )
    lines += [
        "",
        "Decision rule: GO requires scripted episode survival at least +0.06, scripted mass "
        "at least −30 versus CTRL, no frozen/mixed episode-survival drop below −0.05, and "
        "observed holdout agreement at least 0.6 in all five runs. KILL applies at primary <= 0, "
        "observed agreement < 0.6, or any mix mass below −60; otherwise one scripted extension "
        "is allowed. An EXTEND after round 61 is marked extension exhausted and stops.",
        "",
        "## Secondary outcomes",
        "",
        "| arm | mix | episode Δ vs CTRL | alive-frame Δ vs CTRL | mass Δ vs CTRL | "
        "mass Δ vs baseline |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        for mix in MIXES:
            metrics = data["vs_ctrl"][mix]
            lines.append(
                "| %s | %s | %+.3f | %+.3f | %+.1f | %+.1f |"
                % (
                    arm,
                    mix,
                    metrics["episode_survival"]["mean"],
                    metrics["alive_survival"]["mean"],
                    metrics["mass"]["mean"],
                    data["mass_vs_baseline"][mix]["mean"],
                )
            )
    interaction = result["interaction"]
    lines += [
        "",
        "## Interaction",
        "",
        "HUBER × ANCHOR0 scripted episode-survival interaction: %+.3f "
        "(seed SE %.3f; pooled-world SE %.3f)."
        % (interaction["mean"], interaction["seed_se"], interaction["pooled_world_se"]),
        "",
        "## Caveats",
        "",
        "The outcome is a development screen on a finite dev bank. It does not qualify a serving "
        "checkpoint or replace the promotion gate. Pooled-world SE treats paired world "
        "observations as independent and is reported beside the seed-level paired SE, "
        "which remains the cell-level decision uncertainty. Missing holdout telemetry is "
        "not treated as an observed collapse.",
        "",
    ]
    lines += [
        "## Bank sizes and stopping",
        "",
        "Each cell uses five training seeds (10–14), paired to the same B8 initializations. "
        "HUBER uses TD beta 100 and anchor floor 0.1; ANCHOR0 uses beta 1 and floor 0; "
        "HUBER_ANCHOR0 combines beta 100 and floor 0. CTRL reuses M6 beta 1/floor 0.1 finals. "
        "All student probes have no veto; the incumbent is frp3-s12 plus v8.",
        "",
        "| arm | initial decision | final decision | scripted paired observations | "
        "scripted observations vs incumbent | extension exhausted |",
        "|---|---|---|---:|---:|---|",
    ]
    for arm, data in result["arms"].items():
        lines.append(
            f"| {arm} | {data.get('initial_decision', data['decision'])} | {data['decision']} | "
            f"{data['vs_ctrl']['scripted']['episode_survival']['world_n']} | "
            f"{data['mass_vs_baseline']['scripted']['world_n']} | "
            f"{data.get('extension_exhausted', False)} |"
        )
    lines += [
        "",
        "Incumbent contrasts always use round 60 only. Scripted CTRL contrasts use 96 "
        "worlds per training seed, or 192 for extended arms; frozen and mixed retain 32. "
        "Interaction uses only the worlds shared by all four cells "
        f"({result['interaction']['world_n']} paired observations). An exhausted EXTEND "
        "stops without adopting that arm.",
        "",
        "## Final training telemetry",
        "",
        "| run | final holdout agreement | last greedy boost rate | holdout source |",
        "|---|---:|---:|---|",
    ]
    for name, data in sorted(result.get("telemetry_by_run", {}).items()):
        hold = data.get("holdout_agreement")
        boost = data.get("greedy_boost_rate")
        hold_text = "missing" if hold is None else f"{hold:.4f}"
        boost_text = "missing" if boost is None else f"{boost:.6f}"
        source = data.get("holdout_agreement_source", data.get("holdout_source", "unspecified"))
        lines.append(f"| {name} | {hold_text} | {boost_text} | {source} |")
    missing = [
        name
        for name, data in result.get("telemetry_by_run", {}).items()
        if data.get("holdout_agreement") is None
    ]
    lines += [
        "",
        f"Missing holdout agreement: {len(missing)} runs. Missing values cannot "
        "satisfy GO; no substitute agreement metric was invented.",
        "",
        "## Provenance and failures",
        "",
        f"Seed-bank audit passed: {result.get('seed_bank_audit', {}).get('pass', 'unavailable')}. "
        f"Failed probe attempts: {len(result.get('failed_probes', []))}. "
        "Commands, checkpoint hashes, world seeds, per-world death diagnostics and "
        "failure logs are retained in the external probe directory. Optional deaths per "
        "1,000 decisions at length >=800 was not measured. The screen does not establish "
        "the proposed Huber mechanism; that requires the separate T1 readout.",
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


def _rows_from_path(path: Path) -> List[Dict[str, Any]]:
    text = Path(path).read_text().strip()
    if not text:
        return []
    if text.startswith("["):
        return json.loads(text)
    return [json.loads(line) for line in text.splitlines() if line.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--rows", type=Path, required=True, help="probe JSONL or JSON list")
    parser.add_argument("--telemetry", type=Path)
    parser.add_argument("--run-dir", action="append", type=Path, default=[])
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    telemetry = (
        json.loads(args.telemetry.read_text()) if args.telemetry else read_telemetry(args.run_dir)
    )
    result = analyze(_rows_from_path(args.rows), telemetry)
    result["extension_requirements"] = extension_requirements(result, _rows_from_path(args.rows))
    write_results(result, args.json_out, args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
