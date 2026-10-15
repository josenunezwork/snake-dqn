#!/usr/bin/env python3
"""Analysis for the T2b Huber-delta development screen.

The input is deliberately probe rows plus terminal training statuses.  This keeps
the decision calculation replayable and lets the watcher describe an incomplete
cell after a trainer failure without silently treating it as a successful run.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from research.t2_screen_20261015.analyze_t2 import (
    FRAMES,
    INITIAL_ROUND,
    MIXES,
    _mean,
    _number_map,
    _se,
    _survival_maps,
)

ARMS = ("HUBER10", "HUBER30")
HISTORICAL_ARMS = ("HUBER", "HUBER_ANCHOR0")
EXTENSION_ROUND = 61
SEEDS = tuple(range(10, 15))


def _group(rows: Iterable[Mapping[str, Any]]) -> Dict[Tuple[str, int, str], Mapping[str, Any]]:
    grouped: Dict[Tuple[str, int, str], Mapping[str, Any]] = {}
    for row in rows:
        try:
            key = (str(row["name"]), int(row["round_index"]), str(row["mix"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every row needs name, round_index, and mix") from exc
        if key in grouped:
            raise ValueError("duplicate probe row: %r" % (key,))
        grouped[key] = row
    return grouped


def _metric(row: Mapping[str, Any], metric: str) -> Dict[str, float]:
    if metric == "episode_survival":
        return _survival_maps(row)[0]
    if metric == "alive_survival":
        return _survival_maps(row)[1]
    return _number_map(row, metric)


def _joined(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], name: str, mix: str, metric: str
) -> Dict[str, float]:
    output: Dict[str, float] = {}
    for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
        row = grouped.get((name, round_index, mix))
        if row is None:
            continue
        values = _metric(row, metric)
        overlap = set(output) & set(values)
        if overlap:
            raise ValueError("worlds repeated across rounds for %s/%s" % (name, mix))
        output.update(values)
    return output


def _paired(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str, metric: str
) -> Dict[str, Any]:
    seed_values: List[float] = []
    world_values: List[float] = []
    seeds: List[int] = []
    for seed in SEEDS:
        candidate_name = "%s_s%d" % (arm, seed)
        control_name = "CTRL_s%d" % seed
        for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
            candidate_row = grouped.get((candidate_name, round_index, mix))
            control_row = grouped.get((control_name, round_index, mix))
            if (
                candidate_row is not None
                and control_row is not None
                and set(_metric(candidate_row, metric)) != set(_metric(control_row, metric))
            ):
                raise ValueError("candidate and CTRL worlds must match for paired contrasts")
        candidate = _joined(grouped, candidate_name, mix, metric)
        control = _joined(grouped, control_name, mix, metric)
        shared = sorted(set(candidate) & set(control))
        if not shared:
            continue
        values = [candidate[world] - control[world] for world in shared]
        seed_values.append(_mean(values))
        world_values.extend(values)
        seeds.append(seed)
    return {
        "mean": _mean(seed_values),
        "seed_se": _se(seed_values),
        "pooled_world_se": _se(world_values),
        "seeds": seeds,
        "seed_n": len(seed_values),
        "world_n": len(world_values),
        "complete": seeds == list(SEEDS),
    }


def _versus_incumbent(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str, mix: str, metric: str
) -> Dict[str, Any]:
    """Compare only round-60 candidate observations to the shared incumbent row."""
    baseline_row = grouped.get(("baseline", INITIAL_ROUND, mix))
    baseline = _metric(baseline_row, metric) if baseline_row else {}
    per_seed: List[float] = []
    worlds: List[float] = []
    for seed in SEEDS:
        row = grouped.get(("%s_s%d" % (arm, seed), INITIAL_ROUND, mix))
        candidate = _metric(row, metric) if row else {}
        if candidate and baseline and set(candidate) != set(baseline):
            raise ValueError("candidate and incumbent worlds must match")
        shared = sorted(set(candidate) & set(baseline))
        if shared:
            values = [candidate[world] - baseline[world] for world in shared]
            per_seed.append(_mean(values))
            worlds.extend(values)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(worlds),
        "seed_n": len(per_seed),
        "world_n": len(worlds),
    }


def _successful(statuses: Mapping[str, int], arm: str) -> List[int]:
    return [seed for seed in SEEDS if statuses.get("%s_s%d" % (arm, seed)) == 0]


def _decision(metrics: Mapping[str, Any], statuses: Mapping[str, int], arm: str) -> str:
    """Apply T2b's rule; failed runs forbid GO, but clear primary harm is still KILL."""
    primary = metrics["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    # A terminal nonzero exit has no endpoint.  It blocks GO and yields INCOMPLETE,
    # except that available paired data with non-positive primary already establishes KILL.
    if not math.isfinite(primary):
        return "INCOMPLETE"
    if primary <= 0.0:
        return "KILL"
    if len(_successful(statuses, arm)) != len(SEEDS):
        return "INCOMPLETE"
    complete = all(
        metrics["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    )
    if not complete:
        return "INCOMPLETE"
    masses = [metrics["vs_ctrl"][mix]["mass"]["mean"] for mix in MIXES]
    other_survival = [
        metrics["vs_ctrl"][mix]["episode_survival"]["mean"] for mix in ("frozen", "mixed")
    ]
    if (
        all(value >= -30.0 for value in masses)
        and primary >= 0.06
        and all(value >= -0.05 for value in other_survival)
    ):
        return "GO"
    return "EXTEND"


def _validate_initial(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    expected_statuses = {"%s_s%d" % (arm, seed) for arm in ARMS for seed in SEEDS}
    if set(statuses) != expected_statuses:
        raise ValueError("T2b analysis requires terminal statuses for all 10 runs")
    if any(not isinstance(value, int) for value in statuses.values()):
        raise ValueError("terminal statuses must be integer exits")
    required = {("CTRL_s%d" % seed, INITIAL_ROUND, mix) for seed in SEEDS for mix in MIXES}
    required |= {("baseline", INITIAL_ROUND, mix) for mix in MIXES}
    for arm in ARMS:
        for seed in _successful(statuses, arm):
            required |= {("%s_s%d" % (arm, seed), INITIAL_ROUND, mix) for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("initial T2b analysis requires all successful 5-seed initial rows")
    for row in grouped.values():
        values = [_number_map(row, key) for key in ("mass", "deaths", "alive_frames")]
        if not values[0] or any(set(value) != set(values[0]) for value in values):
            raise ValueError("probe world metrics are missing or inconsistent")
        if row.get("frames") != FRAMES:
            raise ValueError("T2b requires the complete H5000 horizon")


def _arm_metrics(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], arm: str
) -> Dict[str, Any]:
    vs_ctrl = {
        mix: {
            metric: _paired(grouped, arm, mix, metric)
            for metric in ("episode_survival", "alive_survival", "mass")
        }
        for mix in MIXES
    }
    return {
        "vs_ctrl": vs_ctrl,
        "vs_incumbent": {
            mix: {
                metric: _versus_incumbent(grouped, arm, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        },
        "round_indices": sorted(
            {round_index for name, round_index, _ in grouped if name.startswith(arm + "_s")}
        ),
    }


def _extension_complete(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]],
    successful_seeds: Sequence[int],
    arm: str,
) -> bool:
    return bool(successful_seeds) and all(
        ("%s_s%d" % (arm, seed), EXTENSION_ROUND, "scripted") in grouped
        and ("CTRL_s%d" % seed, EXTENSION_ROUND, "scripted") in grouped
        for seed in successful_seeds
    )


def analyze(rows: Sequence[Mapping[str, Any]], statuses: Mapping[str, int]) -> Dict[str, Any]:
    """Analyze the T2b rows. ``statuses`` maps each new training run to its exit code."""
    failed_names = {name for name, exit_code in statuses.items() if exit_code != 0}
    grouped = _group([row for row in rows if str(row.get("name")) not in failed_names])
    _validate_initial(grouped, statuses)
    arms: Dict[str, Any] = {}
    for arm in ARMS:
        data = _arm_metrics(grouped, arm)
        data["successful_seeds"] = _successful(statuses, arm)
        data["failed_runs"] = [
            {"name": "%s_s%d" % (arm, seed), "exit": statuses["%s_s%d" % (arm, seed)]}
            for seed in SEEDS
            if statuses["%s_s%d" % (arm, seed)] != 0
        ]
        data["extension_complete"] = _extension_complete(grouped, data["successful_seeds"], arm)
        data["extension_started"] = any(
            ("%s_s%d" % (arm, seed), EXTENSION_ROUND, "scripted") in grouped
            for seed in data["successful_seeds"]
        )
        data["decision"] = _decision(data, statuses, arm)
        # A partially arrived extension cannot change the registered decision.  In
        # particular, a favorable first seed must not be called GO before all five
        # paired r61 banks have arrived.
        if data["extension_started"] and not data["extension_complete"] and not data["failed_runs"]:
            data["decision"] = "EXTEND"
        data["extension_exhausted"] = data["extension_complete"] and data["decision"] == "EXTEND"
        arms[arm] = data
    dose: Dict[str, Any] = {}
    for arm in ARMS + HISTORICAL_ARMS:
        if any(name.startswith(arm + "_s") for name, _, _ in grouped):
            source = (
                grouped
                if arm in ARMS
                else {key: value for key, value in grouped.items() if key[1] == INITIAL_ROUND}
            )
            dose[arm] = {
                "anchor_label": "0" if arm == "HUBER_ANCHOR0" else "0.1",
                **_arm_metrics(source, arm),
            }
    result: Dict[str, Any] = {
        "study": "T2b Huber-delta sweep",
        "evidence": "development screen, not gate evidence",
        "initial_round_index": INITIAL_ROUND,
        "extension_round_index": EXTENSION_ROUND,
        "terminal_statuses": dict(statuses),
        "arms": arms,
        "dose_response": dose,
        "telemetry": (
            "unavailable: multi-actor trainer does not log holdout agreement; not a blocker"
        ),
        "telemetry_by_run": {},
    }
    if any(EXTENSION_ROUND in value["round_indices"] for value in arms.values()):
        initial_rows = [row for row in rows if int(row["round_index"]) == INITIAL_ROUND]
        initial = analyze(initial_rows, statuses)
        for arm, data in arms.items():
            data["initial_decision"] = initial["arms"][arm]["decision"]
    go_arms = [arm for arm, data in arms.items() if data["decision"] == "GO"]
    result["preferred_arm"] = (
        max(
            go_arms,
            key=lambda arm: _mean(
                [result["arms"][arm]["vs_ctrl"][mix]["mass"]["mean"] for mix in MIXES]
            ),
        )
        if len(go_arms) == 2
        else None
    )
    return result


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Return compatible probe jobs for the single permitted scripted extension."""
    grouped = _group(rows)
    jobs: List[Dict[str, Any]] = []
    seen = set()
    for arm, data in result["arms"].items():
        if data["decision"] != "EXTEND" or data.get("extension_complete"):
            continue
        for seed in data["successful_seeds"]:
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


def _fmt(metric: Mapping[str, Any]) -> str:
    return "%+.3f (SE %.3f; world SE %.3f)" % (
        metric["mean"],
        metric["seed_se"],
        metric["pooled_world_se"],
    )


def markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# T2b Huber-delta sweep (2026-10-15)",
        "",
        "**development screen, not gate evidence.**",
        "",
        "## Decision rule",
        "",
        "GO iff paired scripted episode-survival delta vs CTRL >= +0.06 AND mass delta vs CTRL "
        ">= -30 on every mix AND no other mix's episode survival down > 0.05; KILL iff "
        "primary <= 0; else EXTEND once (96 more scripted worlds, round_index 61, arm + CTRL; "
        "reuse the ANCHOR0 extension's CTRL rows if they exist on that bank).",
        "",
        "A halted run (exit != 0) is dropped and blocks GO for that arm. This report records "
        "INCOMPLETE for a failed arm unless its available primary paired endpoint is <= 0, which "
        "is KILL. Holdout agreement is unavailable because the multi-actor trainer does not log "
        "it and is not a blocker.",
        "",
        "## Decisions",
        "",
        "| arm | initial / final decision | scripted episode Δ vs CTRL | "
        "scripted alive Δ vs CTRL | "
        "scripted mass Δ vs CTRL | failed runs |",
        "|---|---|---:|---:|---:|---|",
    ]
    for arm, data in result["arms"].items():
        failed = (
            ", ".join("%s exit=%s" % (item["name"], item["exit"]) for item in data["failed_runs"])
            or "none"
        )
        scripted = data["vs_ctrl"]["scripted"]
        lines.append(
            "| %s | %s / %s | %s | %s | %s | %s |"
            % (
                arm,
                data.get("initial_decision", data["decision"]),
                data["decision"],
                _fmt(scripted["episode_survival"]),
                _fmt(scripted["alive_survival"]),
                _fmt(scripted["mass"]),
                failed,
            )
        )
    lines += [
        "",
        "## Dose response",
        "",
        "| arm | anchor floor | mix | episode Δ CTRL/inc | alive Δ CTRL/inc | mass Δ CTRL/inc |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for arm, data in result["dose_response"].items():
        for mix in MIXES:
            ctrl = data["vs_ctrl"][mix]
            incumbent = data["vs_incumbent"][mix]
            lines.append(
                "| %s | %s | %s | %+.3f / %+.3f | %+.3f / %+.3f | %+.1f / %+.1f |"
                % (
                    arm,
                    data["anchor_label"],
                    mix,
                    ctrl["episode_survival"]["mean"],
                    incumbent["episode_survival"]["mean"],
                    ctrl["alive_survival"]["mean"],
                    incumbent["alive_survival"]["mean"],
                    ctrl["mass"]["mean"],
                    incumbent["mass"]["mean"],
                )
            )
    lines += [
        "",
        "## Provenance and failures",
        "",
        "All contrasts are paired by training seed where CTRL exists; seed SE is the decision SE "
        "and pooled-world SE is descriptive. Historical T2 Huber arms are round-60 only, so "
        "shared CTRL extension rows do not extend their dose-response contrasts.",
        "",
        "Each initial cell has five paired seeds, 96 scripted worlds and 32 frozen/mixed worlds; "
        "an eligible extension adds 96 scripted worlds. H5000 uses rowwise ego2s forward, advisory "
        "mask, numba vector61 backend, fast anchors, and no veto for students.",
        "",
        "Delta-100 historical HUBER has four available seeds because HUBER_s12 halted; its "
        "dose-response values are descriptive, not a five-seed decision.",
        "",
        "Terminal statuses: "
        + ", ".join("%s=%d" % pair for pair in sorted(result["terminal_statuses"].items()))
        + ".",
        "",
    ]
    lines += [
        "## Bank sizes and stopping",
        "",
        "| arm | paired seeds | scripted worlds per seed | extension exhausted |",
        "|---|---:|---:|---|",
    ]
    for arm, data in result["arms"].items():
        primary = data["vs_ctrl"]["scripted"]["episode_survival"]
        per_seed = primary["world_n"] // primary["seed_n"] if primary["seed_n"] else 0
        lines.append(
            f"| {arm} | {primary['seed_n']} | {per_seed} | " f"{data['extension_exhausted']} |"
        )
    lines += [
        "",
        "An exhausted EXTEND stops after round 61 without adopting the arm. Incumbent contrasts "
        "use round 60 only, including for extended arms; their CTRL scripted contrasts use 192 "
        "worlds per seed. Frozen/mixed retain 32. Episode survival means no death in H5000; "
        "alive-frame survival counts post-transition alive frames divided by 5000. Mass is the "
        "full-horizon mass integral with dead frames zero.",
        "",
        "Preferred arm if both GO: " + (result.get("preferred_arm") or "not applicable") + ". "
        "The preference compares the equal-weight mean mass delta vs CTRL across the three mixes.",
        "",
        "## T2b secondary contrasts and uncertainty",
        "",
        "| arm | mix | comparator | episode Δ (seed SE; world SE) | "
        "alive Δ (seed SE; world SE) | mass Δ (seed SE; world SE) |",
        "|---|---|---|---:|---:|---:|",
    ]
    for arm, data in result["arms"].items():
        for mix in MIXES:
            for key, label in (("vs_ctrl", "CTRL"), ("vs_incumbent", "incumbent")):
                metrics = data[key][mix]
                lines.append(
                    f"| {arm} | {mix} | {label} | "
                    f"{_fmt(metrics['episode_survival'])} | "
                    f"{_fmt(metrics['alive_survival'])} | {_fmt(metrics['mass'])} |"
                )
    lines += [
        "",
        "Probe failures: " + str(len(result.get("failed_probes", []))) + ". "
        "Seed-bank audit passed: "
        + str(result.get("seed_bank_audit", {}).get("pass", "unknown"))
        + ". Checkpoint hashes, commands, row files and failure logs are retained in "
        "the existing redesign-t2-probes-20261015 artifact directory. This screen does not "
        "change serving or qualify a checkpoint for promotion.",
        "",
    ]
    telemetry = result.get("telemetry_by_run", {})
    if telemetry:
        lines += [
            "## Final training telemetry",
            "",
            "Holdout agreement is unavailable in this trainer and was not used as a blocker.",
            "",
            "| run | last greedy boost rate | observation |",
            "|---|---:|---|",
        ]
        for name, data in sorted(telemetry.items()):
            boost = data.get("greedy_boost_rate")
            lines.append(
                "| %s | %s | %s |"
                % (
                    name,
                    "missing" if boost is None else "%.6f" % float(boost),
                    data.get("observation", "holdout agreement unavailable"),
                )
            )
        lines.append("")
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
    parser.add_argument("--json-out", type=Path, required=True)
    parser.add_argument("--markdown-out", type=Path, required=True)
    args = parser.parse_args()
    text = args.rows.read_text().strip()
    rows = (
        json.loads(text)
        if text.startswith("[")
        else [json.loads(line) for line in text.splitlines() if line.strip()]
    )
    statuses = json.loads(args.statuses.read_text())
    result = analyze(rows, statuses)
    result["extension_requirements"] = extension_requirements(result, rows)
    write_results(result, args.json_out, args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
