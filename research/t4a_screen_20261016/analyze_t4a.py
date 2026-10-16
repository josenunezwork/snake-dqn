"""Analyze the registered T4a two-sided development screen.

The gameplay decision is deliberately separate from the bank-agreement
diagnostic.  This module consumes the T2/T3-compatible row format so the
watcher can make the registered round-65 decision before the optional round-66
extension, then produce the same result from the complete evidence set.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

from research.t2_screen_20261015.analyze_t2 import _mean, _number_map, _se, _survival_maps

ARM = "TWO_SIDED"
MIXES = ("scripted", "frozen", "mixed")
SEEDS = tuple(range(10, 15))
INITIAL_ROUND = 65
EXTENSION_ROUND = 66
FRAMES = 5000
WINDOW = 100_000


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
    if metric == "death_length":
        raw = row.get("death_lengths", row.get("length_at_death", {}))
        return {str(key): float(value) for key, value in raw.items() if value is not None}
    return _number_map(row, metric)


def _paired(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], mix: str, metric: str
) -> Dict[str, Any]:
    per_seed: List[float] = []
    worlds: List[float] = []
    seen: List[int] = []
    for seed in SEEDS:
        candidate: Dict[str, float] = {}
        control: Dict[str, float] = {}
        for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
            candidate_row = grouped.get(("%s_s%d" % (ARM, seed), round_index, mix))
            control_row = grouped.get(("CTRL_s%d" % seed, round_index, mix))
            if candidate_row is None or control_row is None:
                continue
            candidate_values = _metric(candidate_row, metric)
            control_values = _metric(control_row, metric)
            if set(candidate_values) != set(control_values):
                raise ValueError("candidate and CTRL worlds must match for paired contrasts")
            overlap = set(candidate) & set(candidate_values)
            if overlap:
                raise ValueError("worlds repeated across T4a rounds")
            candidate.update(candidate_values)
            control.update(control_values)
        if candidate or control:
            if set(candidate) != set(control):
                raise ValueError("candidate and CTRL worlds must match for paired contrasts")
            values = [candidate[key] - control[key] for key in sorted(candidate)]
            if values:
                per_seed.append(_mean(values))
                worlds.extend(values)
                seen.append(seed)
    return {
        "mean": _mean(per_seed),
        "seed_se": _se(per_seed),
        "pooled_world_se": _se(worlds),
        "seeds": seen,
        "seed_n": len(per_seed),
        "world_n": len(worlds),
        "complete": seen == list(SEEDS),
    }


def _incumbent(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], mix: str
) -> Dict[str, Any]:
    baseline_row = grouped.get(("baseline", INITIAL_ROUND, mix))
    baseline = _metric(baseline_row, "mass") if baseline_row else {}
    per_seed: List[float] = []
    worlds: List[float] = []
    for seed in SEEDS:
        candidate_row = grouped.get(("%s_s%d" % (ARM, seed), INITIAL_ROUND, mix))
        candidate = _metric(candidate_row, "mass") if candidate_row else {}
        if candidate and baseline and set(candidate) != set(baseline):
            raise ValueError("candidate and incumbent worlds must match")
        if candidate and baseline:
            values = [candidate[key] - baseline[key] for key in sorted(candidate)]
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
    values: List[float] = []
    for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
        row = grouped.get((name, round_index, mix))
        if row is None:
            continue
        deaths, lengths = _number_map(row, "deaths"), _metric(row, "death_length")
        values.extend(lengths[key] for key, death in deaths.items() if death and key in lengths)
    values.sort()
    if not values:
        return dict(
            death_count=0,
            mean=float("nan"),
            median=float("nan"),
            q25=float("nan"),
            q75=float("nan"),
        )

    def quantile(fraction: float) -> float:
        return values[round((len(values) - 1) * fraction)]

    return dict(
        death_count=len(values),
        mean=_mean(values),
        median=quantile(0.5),
        q25=quantile(0.25),
        q75=quantile(0.75),
    )


def _death_lengths(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], mix: str
) -> Dict[str, Any]:
    candidate = [_death_distribution(grouped, "%s_s%d" % (ARM, seed), mix) for seed in SEEDS]
    ctrl = [_death_distribution(grouped, "CTRL_s%d" % seed, mix) for seed in SEEDS]
    return {
        "candidate": candidate,
        "ctrl": ctrl,
        "candidate_mean": _mean([x["mean"] for x in candidate if math.isfinite(x["mean"])]),
        "ctrl_mean": _mean([x["mean"] for x in ctrl if math.isfinite(x["mean"])]),
    }


def _telemetry_for(telemetry: Mapping[str, Mapping[str, Any]]) -> Dict[str, Any]:
    runs = {"%s_s%d" % (ARM, seed): telemetry.get("%s_s%d" % (ARM, seed), {}) for seed in SEEDS}

    def window(key: str, point: str) -> List[float]:
        return [
            float(value[point][key]["value"])
            for value in runs.values()
            if value.get(point, {}).get(key) is not None
        ]

    def final(key: str) -> List[float]:
        return [float(value[key]) for value in runs.values() if value.get(key) is not None]

    values = {
        "greedy_boost_rate_mean": _mean(final("greedy_boost_rate")),
        "greedy_boost_rate_n": len(final("greedy_boost_rate")),
        "curves": {name: value.get("greedy_boost_rate_curve", []) for name, value in runs.items()},
        "per_run": runs,
    }
    for point in ("2m", "8m"):
        for key in ("bank_agree", "bank_agree_decisive", "bank_agree_counter"):
            samples = window(key, "bank_agree_" + point)
            values["%s_%s_mean" % (key, point)] = _mean(samples)
            values["%s_%s_n" % (key, point)] = len(samples)
    values["bank_agreement_per_100k_means"] = _per_100k_means(runs)
    decisive, counter = values["bank_agree_decisive_8m_mean"], values["bank_agree_counter_8m_mean"]
    if values["bank_agree_decisive_8m_n"] != len(SEEDS) or values["bank_agree_counter_8m_n"] != len(
        SEEDS
    ):
        verdict = "inconclusive: one or both registered 8M bank-agreement sides are missing."
    elif decisive > 0.9 and counter > 0.9:
        verdict = (
            "both >0.9: supports the inputs carrying the distinction and labels being learnable."
        )
    elif decisive < 0.8 and counter > 0.9:
        verdict = (
            "decisive <0.8 while counter holds >0.9: points to an input limit; "
            "input changes are the next research direction."
        )
    else:
        verdict = "inconclusive."
    values["bank_agree_readout_verdict"] = verdict
    return values


def _per_100k_means(runs: Mapping[str, Mapping[str, Any]]) -> Dict[str, List[Dict[str, float]]]:
    """Pool observed side agreement by nominal 100k telemetry bins.

    Actor packets may place an observation slightly either side of its nominal
    boundary.  Each observed side remains its own sample; absent sides do not
    contribute a zero.
    """
    output: Dict[str, List[Dict[str, float]]] = {}
    for key in ("bank_agree_decisive", "bank_agree_counter"):
        bins: Dict[int, List[float]] = {}
        for run in runs.values():
            for point in run.get("bank_agreement_curves", {}).get(key, []):
                nominal = int((int(point["transitions"]) + WINDOW // 2) // WINDOW) * WINDOW
                bins.setdefault(nominal, []).append(float(point["value"]))
        output[key] = [
            {"transitions": transition, "mean": _mean(samples), "n": len(samples)}
            for transition, samples in sorted(bins.items())
        ]
    return output


def _decision(data: Mapping[str, Any], statuses: Mapping[str, int], extended: bool) -> str:
    if any(statuses[name] != 0 for name in statuses):
        return "KILL"
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    if not math.isfinite(primary):
        return "INCOMPLETE"
    if primary <= 0.0:
        return "KILL"
    if not all(
        data["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    ):
        return "INCOMPLETE"
    go = (
        primary >= 0.06
        and all(data["vs_ctrl"][mix]["mass"]["mean"] >= -30.0 for mix in MIXES)
        and all(
            data["vs_ctrl"][mix]["episode_survival"]["mean"] >= -0.05 for mix in ("frozen", "mixed")
        )
    )
    if go:
        return "GO"
    return "INCONCLUSIVE" if extended else "EXTEND"


def _validate(
    grouped: Mapping[Tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    expected = {"%s_s%d" % (ARM, seed) for seed in SEEDS}
    halted = any(value != 0 for value in statuses.values())
    if not set(statuses).issubset(expected) or any(
        not isinstance(value, int) for value in statuses.values()
    ):
        raise ValueError("T4a analysis requires integer terminal statuses for all five runs")
    if not halted and set(statuses) != expected:
        raise ValueError("T4a analysis requires statuses for all five runs unless a run halted")
    required = set()
    if not halted:
        required = {("CTRL_s%d" % seed, INITIAL_ROUND, mix) for seed in SEEDS for mix in MIXES}
        required |= {("baseline", INITIAL_ROUND, mix) for mix in MIXES}
        for seed in SEEDS:
            required |= {("%s_s%d" % (ARM, seed), INITIAL_ROUND, mix) for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("initial T4a analysis requires all successful initial rows")
    allowed_names = (
        {"baseline"}
        | {"CTRL_s%d" % seed for seed in SEEDS}
        | {"%s_s%d" % (ARM, seed) for seed in SEEDS}
    )
    for row in grouped.values():
        values = [_number_map(row, key) for key in ("mass", "deaths", "alive_frames")]
        if not values[0] or any(set(value) != set(values[0]) for value in values):
            raise ValueError("probe world metrics are missing or inconsistent")
        if row.get("frames") != FRAMES:
            raise ValueError("T4a requires the complete H5000 horizon")
    for (name, round_index, mix), row in grouped.items():
        if name not in allowed_names or round_index not in (INITIAL_ROUND, EXTENSION_ROUND):
            raise ValueError("unexpected T4a probe identity")
        if round_index == EXTENSION_ROUND and (mix != "scripted" or name == "baseline"):
            raise ValueError("the T4a extension permits scripted candidate and CTRL rows only")
        expected_worlds = 96 if mix == "scripted" else 32
        if len(_number_map(row, "mass")) != expected_worlds:
            raise ValueError("T4a probe row has the wrong registered world count")


def analyze(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, int],
    telemetry: Optional[Mapping[str, Mapping[str, Any]]] = None,
    rerolls: Optional[Mapping[str, Mapping[str, Any]]] = None,
    t3_context: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Apply T4a's one-extension decision rule and required secondary readouts."""
    failed = {name for name, code in statuses.items() if code != 0}
    grouped = _group(row for row in rows if str(row.get("name")) not in failed)
    _validate(grouped, statuses)
    telemetry, rerolls = telemetry or {}, rerolls or {}
    extension_started = any(
        name.startswith(ARM + "_s") and rnd == EXTENSION_ROUND for name, rnd, _ in grouped
    )
    extension_complete = (
        all(
            ("%s_s%d" % (ARM, seed), EXTENSION_ROUND, "scripted") in grouped
            and ("CTRL_s%d" % seed, EXTENSION_ROUND, "scripted") in grouped
            for seed in SEEDS
        )
        and not failed
    )
    initial_grouped = {key: row for key, row in grouped.items() if key[1] == INITIAL_ROUND}
    initial_metrics = {
        "vs_ctrl": {
            mix: {
                metric: _paired(initial_grouped, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        }
    }
    initial_decision = _decision(initial_metrics, statuses, False)
    if extension_started and initial_decision != "EXTEND":
        raise ValueError("round-66 rows require an initial T4a EXTEND decision")
    # A partial extension is operational evidence only.  It must not change
    # the initial decision or the reported paired estimates.
    analysis_grouped = (
        grouped
        if extension_complete
        else {key: row for key, row in grouped.items() if key[1] == INITIAL_ROUND}
    )
    data: Dict[str, Any] = {
        "vs_ctrl": {
            mix: {
                metric: _paired(analysis_grouped, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        },
        "mass_vs_incumbent": {mix: _incumbent(analysis_grouped, mix) for mix in MIXES},
        "death_lengths": {mix: _death_lengths(analysis_grouped, mix) for mix in MIXES},
        "telemetry": _telemetry_for(telemetry),
        "reroll": _reroll_summary(rerolls),
        "failed_runs": [{"name": name, "exit": statuses[name]} for name in sorted(failed)],
        "unstarted_runs": sorted({"%s_s%d" % (ARM, seed) for seed in SEEDS} - set(statuses)),
        "round_indices": sorted({rnd for name, rnd, _ in grouped if name.startswith(ARM + "_s")}),
        "extension_started": extension_started,
        "extension_complete": extension_complete,
    }
    data["decision"] = _decision(data, statuses, extension_complete)
    if extension_started and not extension_complete and not failed:
        data["decision"] = "EXTEND"
    data["extension_exhausted"] = extension_complete and data["decision"] == "INCONCLUSIVE"
    result: Dict[str, Any] = {
        "study": "T4a two-sided development screen",
        "evidence": "development screen, not gate evidence",
        "initial_round_index": INITIAL_ROUND,
        "extension_round_index": EXTENSION_ROUND,
        "terminal_statuses": dict(statuses),
        "arms": {ARM: data},
        "telemetry_by_run": dict(telemetry),
        "rerolls_by_run": dict(rerolls),
        "t3_context": dict(t3_context or {}),
    }
    if extension_started:
        data["initial_decision"] = initial_decision
    return result


def _reroll_summary(rerolls: Mapping[str, Mapping[str, Any]]) -> Dict[str, float]:
    """Average T3-compatible or T4a preferred-set re-roll fields by final."""
    aliases = {
        "labelled_survivor_action_survival_fraction": (
            "labelled_survivor_action_survival_fraction",
            "labelled_preferred_action_survival_fraction",
            "target_action_survival_fraction",
        ),
        "final_raw_argmax_in_labelled_surviving_set_fraction": (
            "final_raw_argmax_in_labelled_surviving_set_fraction",
            "final_raw_argmax_in_labelled_preferred_set_fraction",
        ),
        "final_advisory_argmax_in_labelled_surviving_set_fraction": (
            "final_advisory_argmax_in_labelled_surviving_set_fraction",
            "final_advisory_argmax_in_labelled_preferred_set_fraction",
        ),
    }
    output: Dict[str, float] = {}
    for output_key, field_names in aliases.items():
        values = []
        for name, value in rerolls.items():
            if not name.startswith(ARM + "_s"):
                continue
            for field in field_names:
                if value.get(field) is not None:
                    values.append(float(value[field]))
                    break
        output[output_key] = _mean(values)
    for prefix, source_key in (
        ("raw_argmax", "raw_argmax_matches_t4a_target_by_side"),
        ("advisory_argmax", "advisory_argmax_matches_t4a_target_by_side"),
    ):
        for side in ("decisive", "counter"):
            values = [
                float(value[source_key][side])
                for name, value in rerolls.items()
                if name.startswith(ARM + "_s")
                and isinstance(value.get(source_key), Mapping)
                and value[source_key].get(side) is not None
            ]
            output["%s_%s_mean" % (prefix, side)] = _mean(values)
            output["%s_%s_n" % (prefix, side)] = len(values)
    return output


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> List[Dict[str, Any]]:
    """Return the single registered round-66 scripted extension, if eligible."""
    data = result["arms"][ARM]
    if data["decision"] != "EXTEND" or data["extension_started"]:
        return []
    grouped = _group(rows)
    jobs: List[Dict[str, Any]] = []
    for seed in SEEDS:
        for name in ("%s_s%d" % (ARM, seed), "CTRL_s%d" % seed):
            key = (name, EXTENSION_ROUND, "scripted")
            if key not in grouped:
                jobs.append(dict(name=name, round_index=EXTENSION_ROUND, mix="scripted", worlds=96))
    return jobs


def t3_context_from_result(t3_result: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    """Extract only the final T3 arm endpoints for a context-only T4a table."""
    output: Dict[str, Dict[str, Any]] = {}
    for arm, data in t3_result.get("arms", {}).items():
        scripted = data.get("vs_ctrl", {}).get("scripted", {})
        output[str(arm)] = {
            "scripted_episode_delta": scripted.get("episode_survival", {}).get("mean"),
            "scripted_mass_delta": scripted.get("mass", {}).get("mean"),
            "decision": (
                "EXTEND exhausted / stop"
                if data.get("extension_exhausted")
                else data.get("decision")
            ),
        }
    return output


def _nearest(
    lines: Sequence[Mapping[str, Any]], target: int, field: str, *, crossing: bool = False
) -> Optional[Dict[str, float]]:
    candidates = [
        row for row in lines if row.get(field) is not None and row.get("transitions") is not None
    ]
    if crossing:
        candidates = [
            row for row in candidates if target <= int(row["transitions"]) < target + WINDOW
        ]
    else:
        candidates = [row for row in candidates if abs(int(row["transitions"]) - target) <= WINDOW]
    if not candidates:
        return None
    row = min(
        candidates,
        key=(
            (lambda value: int(value["transitions"]))
            if crossing
            else (lambda value: abs(int(value["transitions"]) - target))
        ),
    )
    return {
        "transitions": int(row["transitions"]),
        "value": float(row[field]),
        "target": target,
        "bound": WINDOW,
    }


def read_telemetry(
    run_dirs: Union[Sequence[Path], Mapping[str, Path]],
) -> Dict[str, Dict[str, Any]]:
    """Read retained raw log records plus required derived telemetry summaries."""
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
            source_text = log_path.read_text()
            lines = [json.loads(line) for line in source_text.splitlines() if line.strip()]
            boost = [
                {"transitions": int(row["transitions"]), "value": float(row["greedy_boost_rate"])}
                for row in lines
                if row.get("greedy_boost_rate") is not None
            ]
            bank_curves = {
                key: [
                    {"transitions": int(row["transitions"]), "value": float(row[key])}
                    for row in lines
                    if row.get(key) is not None
                ]
                for key in ("bank_agree", "bank_agree_decisive", "bank_agree_counter")
            }

            def point(target: int, crossing: bool) -> Dict[str, Optional[Dict[str, float]]]:
                return {
                    key: _nearest(lines, target, key, crossing=crossing)
                    for key in ("bank_agree", "bank_agree_decisive", "bank_agree_counter")
                }

            final_record = lines[-1] if lines else {}
            final_metrics = {
                key: final_record.get(key)
                for key in (
                    "td",
                    "anchor",
                    "qmax",
                    "bank_margin_loss",
                    "bank_rows_mean",
                    "tight_frac",
                )
            }

            output[supplied_name or log_path.parent.name] = {
                "source_log": {
                    "path": str(log_path),
                    "sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                },
                "telemetry_records": lines,
                "final_telemetry": final_metrics,
                "greedy_boost_rate": boost[-1]["value"] if boost else None,
                "greedy_boost_rate_curve": boost,
                "bank_agreement_curves": bank_curves,
                "bank_agree_2m": point(2_000_000, False),
                "bank_agree_8m": point(8_000_000, True),
            }
    return output


def _fmt(value: Mapping[str, Any]) -> str:
    return "%+.3f (seed SE %.3f; world SE %.3f)" % (
        value["mean"],
        value["seed_se"],
        value["pooled_world_se"],
    )


def markdown(result: Mapping[str, Any]) -> str:
    """Render the complete required T4a development-screen receipt."""
    data = result["arms"][ARM]
    script = data["vs_ctrl"]["scripted"]
    lines = [
        "# T4a two-sided development screen, not gate evidence (2026-10-16)",
        "",
        "**Development screen, not gate evidence.** TWO_SIDED is evaluated without a veto; "
        "it does not promote or change serving.",
        "",
        "## Decision",
        "",
        "| arm | initial / final | scripted episode delta vs CTRL | scripted mass delta CTRL | "
        "failed runs |",
        "|---|---|---:|---:|---|",
        "| %s | %s / %s | %s | %s | %s |"
        % (
            ARM,
            data.get("initial_decision", data["decision"]),
            data["decision"],
            _fmt(script["episode_survival"]),
            _fmt(script["mass"]),
            ", ".join("%s exit=%s" % (item["name"], item["exit"]) for item in data["failed_runs"])
            or "none",
        ),
        "",
        "GO requires primary >= +0.06, mass delta versus CTRL >= -30 on every mix, and neither "
        "frozen nor mixed episode-survival delta below -0.05. KILL applies to any halt or "
        "primary <= 0. Otherwise round 66 is the single 96-world scripted extension; an "
        "intermediate final result is inconclusive.",
        "",
        "## Secondary outcomes",
        "",
        "| mix | episode delta CTRL | alive-frame delta CTRL | mass delta CTRL | mass delta "
        "incumbent | death mean TWO_SIDED / CTRL |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for mix in MIXES:
        ctrl = data["vs_ctrl"][mix]
        lengths = data["death_lengths"][mix]
        lines.append(
            "| %s | %s | %s | %s | %s | %.1f / %.1f |"
            % (
                mix,
                _fmt(ctrl["episode_survival"]),
                _fmt(ctrl["alive_survival"]),
                _fmt(ctrl["mass"]),
                _fmt(data["mass_vs_incumbent"][mix]),
                lengths["candidate_mean"],
                lengths["ctrl_mean"],
            )
        )
    lines += [
        "",
        "### Death-length distributions",
        "",
        "| mix | policy | deaths | mean | median | q25 | q75 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for mix in MIXES:
        for policy in ("candidate", "ctrl"):
            distributions = data["death_lengths"][mix][policy]
            finite = {
                key: _mean([item[key] for item in distributions if math.isfinite(item[key])])
                for key in ("mean", "median", "q25", "q75")
            }
            lines.append(
                "| %s | %s | %d | %.1f | %.1f | %.1f | %.1f |"
                % (
                    mix,
                    policy,
                    sum(item["death_count"] for item in distributions),
                    finite["mean"],
                    finite["median"],
                    finite["q25"],
                    finite["q75"],
                )
            )
    telemetry = data["telemetry"]
    lines += [
        "",
        "## Registered bank-agreement readout",
        "",
        "| seed | decisive at 8M | counter at 8M |",
        "|---|---:|---:|",
    ]
    for seed in SEEDS:
        run = telemetry["per_run"].get("%s_s%d" % (ARM, seed), {})
        point = run.get("bank_agree_8m", {})
        decisive, counter = point.get("bank_agree_decisive"), point.get("bank_agree_counter")
        lines.append(
            "| %d | %s | %s |"
            % (
                seed,
                "missing" if decisive is None else "%.4f" % decisive["value"],
                "missing" if counter is None else "%.4f" % counter["value"],
            )
        )
    lines += [
        "| mean | %.4f | %.4f |"
        % (telemetry["bank_agree_decisive_8m_mean"], telemetry["bank_agree_counter_8m_mean"]),
        "",
        telemetry["bank_agree_readout_verdict"],
        "",
        "### Greedy boost rate per 100k",
        "",
        "| run | transition:value |",
        "|---|---|",
    ]
    for name, item in sorted(result.get("telemetry_by_run", {}).items()):
        curve = ", ".join(
            "%s:%.6f" % (point["transitions"], point["value"])
            for point in item.get("greedy_boost_rate_curve", [])
        )
        lines.append("| %s | %s |" % (name, curve or "missing"))
    lines += [
        "",
        "### Final and 2M telemetry",
        "",
        "| run | final greedy boost | bank agree 2M / 8M | decisive 2M / 8M | counter 2M / 8M |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in sorted(result.get("telemetry_by_run", {}).items()):

        def read(point: str, key: str) -> str:
            value = item.get(point, {}).get(key)
            return "missing" if value is None else "%.4f" % value["value"]

        final_boost = item.get("greedy_boost_rate")
        lines.append(
            "| %s | %s | %s / %s | %s / %s | %s / %s |"
            % (
                name,
                "missing" if final_boost is None else "%.4f" % final_boost,
                read("bank_agree_2m", "bank_agree"),
                read("bank_agree_8m", "bank_agree"),
                read("bank_agree_2m", "bank_agree_decisive"),
                read("bank_agree_8m", "bank_agree_decisive"),
                read("bank_agree_2m", "bank_agree_counter"),
                read("bank_agree_8m", "bank_agree_counter"),
            )
        )
    lines += [
        "",
        "### Final retained learner telemetry",
        "",
        "| run | TD | anchor | Q max | bank margin loss | bank rows | tight fraction |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in sorted(result.get("telemetry_by_run", {}).items()):
        final = item.get("final_telemetry", {})
        rendered = [
            "missing" if final.get(key) is None else "%.6g" % float(final[key])
            for key in ("td", "anchor", "qmax", "bank_margin_loss", "bank_rows_mean", "tight_frac")
        ]
        lines.append("| %s | %s | %s | %s | %s | %s | %s |" % (name, *rendered))
    lines += [
        "",
        "### Bank agreement per 100k",
        "",
        "| run | ordinary | decisive | counter |",
        "|---|---|---|---|",
    ]
    for name, item in sorted(result.get("telemetry_by_run", {}).items()):
        curves = item.get("bank_agreement_curves", {})
        rendered = []
        for key in ("bank_agree", "bank_agree_decisive", "bank_agree_counter"):
            rendered.append(
                ", ".join(
                    "%s:%.6f" % (point["transitions"], point["value"])
                    for point in curves.get(key, [])
                )
                or "missing"
            )
        lines.append("| %s | %s | %s | %s |" % (name, *rendered))
    lines += ["", "### Five-seed side means per nominal 100k", ""]
    for key, label in (("bank_agree_decisive", "decisive"), ("bank_agree_counter", "counter")):
        lines += ["#### %s" % label, "", "| nominal transitions | mean | n |", "|---:|---:|---:|"]
        for point in telemetry["bank_agreement_per_100k_means"].get(key, []):
            lines.append("| %d | %.6f | %d |" % (point["transitions"], point["mean"], point["n"]))
        lines.append("")
    reroll = data["reroll"]
    lines += [
        "",
        "## Post-hoc re-roll",
        "",
        "| target action survival | raw argmax decisive / counter | advisory argmax decisive / "
        "counter |",
        "|---:|---:|---:|",
        "| %.4f | %.4f / %.4f | %.4f / %.4f |"
        % (
            reroll["labelled_survivor_action_survival_fraction"],
            reroll["raw_argmax_decisive_mean"],
            reroll["raw_argmax_counter_mean"],
            reroll["advisory_argmax_decisive_mean"],
            reroll["advisory_argmax_counter_mean"],
        ),
        "",
        "### Per-final re-roll",
        "",
        "| final | target survival | raw decisive / counter | advisory decisive / counter | "
        "states decisive / counter |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, item in sorted(result.get("rerolls_by_run", {}).items()):
        raw, advisory, counts = (
            item.get("raw_argmax_matches_t4a_target_by_side", {}),
            item.get("advisory_argmax_matches_t4a_target_by_side", {}),
            item.get("states_by_side", {}),
        )

        def side(pair: Mapping[str, Any], key: str) -> str:
            value = pair.get(key)
            return "missing" if value is None else "%.4f" % float(value)

        lines.append(
            "| %s | %s | %s / %s | %s / %s | %s / %s |"
            % (
                name,
                side(item, "target_action_survival_fraction"),
                side(raw, "decisive"),
                side(raw, "counter"),
                side(advisory, "decisive"),
                side(advisory, "counter"),
                counts.get("decisive", "missing"),
                counts.get("counter", "missing"),
            )
        )
    lines += [
        "",
        "The re-roll uses the T4a target: v2 preferred set for decisive rows and the stored B8 "
        "action for counters, classified by `surviving_original`. It is descriptive and cannot "
        "change the decision.",
        "",
        "The per-side agreement denominators exclude the other side; missing sides are not "
        "treated as zero. Seed-level paired SE is the decision uncertainty; pooled-world SE is "
        "descriptive. The 8M values are the first final 100k telemetry window at or after 8M, "
        "allowing normal actor packet overshoot.",
        "",
    ]
    context = result.get("t3_context", {})
    if context:
        lines += [
            "## T3 final-arm context only",
            "",
            "The following T3 values are context only and do not enter the T4a decision.",
            "",
            "| arm | scripted episode delta CTRL | scripted mass delta CTRL | final decision |",
            "|---|---:|---:|---|",
            "| TWO_SIDED | %+.6f | %+.6f | %s |"
            % (
                data["vs_ctrl"]["scripted"]["episode_survival"]["mean"],
                data["vs_ctrl"]["scripted"]["mass"]["mean"],
                data["decision"],
            ),
        ]
        for arm, value in sorted(context.items()):
            lines.append(
                "| %s | %s | %s | %s |"
                % (
                    arm,
                    value.get("scripted_episode_delta", "missing"),
                    value.get("scripted_mass_delta", "missing"),
                    value.get("decision", "missing"),
                )
            )
        lines.append("")
        binding = result.get("t3_context_binding", {})
        if binding:
            lines += [
                "T3 source: [final screen](t3_screen_result_2026-10-15.md), commit `%s`; "
                "analysis SHA-256 `%s`. T3 uses rounds 62/63 and T4a uses rounds 65/66; "
                "these are separate development screens, not a paired T3-vs-T4a comparison."
                % (binding["source_commit"], binding["sha256"]),
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
