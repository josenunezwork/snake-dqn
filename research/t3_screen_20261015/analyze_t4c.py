"""Registered round-66/67 T4c screen analysis.

Gameplay decision logic is deliberately isolated from the descriptive bank,
drift, and context readouts.  This accepts the established T2 probe-row schema
plus ``length_at_5000`` supplied by the T4c probe wrapper.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Sequence

from research.t2_screen_20261015.analyze_t2 import _mean, _number_map, _se, _survival_maps

ARM = "TWO_SIDED_T"
MIXES = ("scripted", "frozen", "mixed")
SEEDS = tuple(range(10, 15))
INITIAL_ROUND, EXTENSION_ROUND, FRAMES = 66, 67, 5000


def _group(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[str, int, str], Mapping[str, Any]]:
    grouped = {}
    for row in rows:
        try:
            key = str(row["name"]), int(row["round_index"]), str(row["mix"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("every row needs name, round_index, and mix") from exc
        if key in grouped:
            raise ValueError("duplicate T4c probe row: %r" % (key,))
        grouped[key] = row
    return grouped


def _metric(row: Mapping[str, Any], metric: str) -> dict[str, float]:
    if metric == "episode_survival":
        return _survival_maps(row)[0]
    if metric == "alive_survival":
        return _survival_maps(row)[1]
    return _number_map(row, metric)


def _paired(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], mix: str, metric: str
) -> dict[str, Any]:
    values, worlds, seen = [], [], []
    for seed in SEEDS:
        candidate, ctrl = {}, {}
        for round_index in (INITIAL_ROUND, EXTENSION_ROUND):
            left = grouped.get((f"{ARM}_s{seed}", round_index, mix))
            right = grouped.get((f"CTRL_s{seed}", round_index, mix))
            if left is None or right is None:
                continue
            a, b = _metric(left, metric), _metric(right, metric)
            if set(a) != set(b) or set(candidate) & set(a):
                raise ValueError("T4c paired worlds must match and not repeat across rounds")
            candidate.update(a)
            ctrl.update(b)
        if candidate:
            diff = [candidate[key] - ctrl[key] for key in sorted(candidate)]
            values.append(_mean(diff))
            worlds.extend(diff)
            seen.append(seed)
    return {
        "mean": _mean(values),
        "seed_se": _se(values),
        "pooled_world_se": _se(worlds),
        "seeds": seen,
        "seed_n": len(values),
        "world_n": len(worlds),
        "complete": seen == list(SEEDS),
    }


def _incumbent(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], mix: str
) -> dict[str, Any]:
    baseline = grouped.get(("baseline", INITIAL_ROUND, mix))
    base = _number_map(baseline, "mass") if baseline else {}
    values, worlds = [], []
    for seed in SEEDS:
        row = grouped.get((f"{ARM}_s{seed}", INITIAL_ROUND, mix))
        candidate = _number_map(row, "mass") if row else {}
        if candidate and base and set(candidate) != set(base):
            raise ValueError("candidate and incumbent worlds must match")
        if candidate and base:
            diff = [candidate[key] - base[key] for key in sorted(base)]
            values.append(_mean(diff))
            worlds.extend(diff)
    return {
        "mean": _mean(values),
        "seed_se": _se(values),
        "pooled_world_se": _se(worlds),
        "seed_n": len(values),
        "world_n": len(worlds),
    }


def _shared_lengths(grouped: Mapping[tuple[str, int, str], Mapping[str, Any]]) -> dict[str, Any]:
    """H5000 lengths among worlds where both candidate and CTRL survived."""
    by_seed, all_deltas = [], []
    for seed in SEEDS:
        left = grouped.get((f"{ARM}_s{seed}", INITIAL_ROUND, "scripted"))
        right = grouped.get((f"CTRL_s{seed}", INITIAL_ROUND, "scripted"))
        if left is None or right is None:
            continue
        deaths_a, deaths_b = _number_map(left, "deaths"), _number_map(right, "deaths")
        lengths_a, lengths_b = _number_map(left, "length_at_5000"), _number_map(
            right, "length_at_5000"
        )
        if (
            set(deaths_a) != set(deaths_b)
            or set(lengths_a) != set(deaths_a)
            or set(lengths_b) != set(deaths_b)
        ):
            raise ValueError("T4c length_at_5000 must align with paired scripted worlds")
        shared = [key for key in sorted(deaths_a) if not deaths_a[key] and not deaths_b[key]]
        delta = [lengths_a[key] - lengths_b[key] for key in shared]
        by_seed.append({"seed": seed, "shared_survivors": len(shared), "mean_delta": _mean(delta)})
        all_deltas.extend(delta)
    return {
        "frame": 5000,
        "per_seed": by_seed,
        "shared_survivors": len(all_deltas),
        "candidate_minus_ctrl_mean_length": _mean(all_deltas),
        "five_seed_mean_delta": _mean(
            [item["mean_delta"] for item in by_seed if math.isfinite(item["mean_delta"])]
        ),
        "five_seed_n": len([item for item in by_seed if math.isfinite(item["mean_delta"])]),
        "threshold": -50.0,
    }


def _decision(metrics: Mapping[str, Any], statuses: Mapping[str, int], extended: bool) -> str:
    if any(code != 0 for code in statuses.values()):
        return "KILL"
    primary = metrics["vs_ctrl"]["scripted"]["episode_survival"]["mean"]
    if not math.isfinite(primary):
        return "INCOMPLETE"
    if primary <= 0:
        return "KILL"
    complete = all(
        metrics["vs_ctrl"][mix][metric]["complete"]
        for mix in MIXES
        for metric in ("episode_survival", "alive_survival", "mass")
    )
    if not complete:
        return "INCOMPLETE"
    if (
        primary >= 0.06
        and all(metrics["vs_ctrl"][mix]["mass"]["mean"] >= -30 for mix in MIXES)
        and all(
            metrics["vs_ctrl"][mix]["episode_survival"]["mean"] >= -0.05
            for mix in ("frozen", "mixed")
        )
    ):
        return "GO"
    return "INCONCLUSIVE" if extended else "EXTEND"


def _validate(
    grouped: Mapping[tuple[str, int, str], Mapping[str, Any]], statuses: Mapping[str, int]
) -> None:
    expected = {f"{ARM}_s{seed}" for seed in SEEDS}
    if not set(statuses).issubset(expected) or any(
        type(code) is not int for code in statuses.values()
    ):
        raise ValueError("T4c requires integer statuses only for its five candidate runs")
    halted = any(code != 0 for code in statuses.values())
    if not halted and set(statuses) != expected:
        raise ValueError("T4c requires all five statuses unless a candidate halted")
    if halted:
        return
    required = {(f"CTRL_s{seed}", INITIAL_ROUND, mix) for seed in SEEDS for mix in MIXES}
    required |= {(f"{ARM}_s{seed}", INITIAL_ROUND, mix) for seed in SEEDS for mix in MIXES}
    required |= {("baseline", INITIAL_ROUND, mix) for mix in MIXES}
    if not required.issubset(grouped):
        raise ValueError("round-66 requires all candidate, CTRL, and incumbent rows")
    for row in grouped.values():
        fields = [_number_map(row, name) for name in ("mass", "deaths", "alive_frames")]
        if (
            row.get("frames") != FRAMES
            or not fields[0]
            or any(set(item) != set(fields[0]) for item in fields)
        ):
            raise ValueError("T4c requires complete H5000 world metrics")
        expected_worlds = 96 if row["mix"] == "scripted" else 32
        if len(fields[0]) != expected_worlds:
            raise ValueError("T4c probe row has wrong registered world count")


def analyze(
    rows: Sequence[Mapping[str, Any]],
    statuses: Mapping[str, int],
    telemetry: Mapping[str, Any] | None = None,
    diagnostics: Mapping[str, Any] | None = None,
    context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Apply the preregistered decision rule; diagnostics never change it."""
    failed = {name for name, code in statuses.items() if code != 0}
    grouped = _group(row for row in rows if row.get("name") not in failed)
    _validate(grouped, statuses)
    extension_started = any(round_index == EXTENSION_ROUND for _, round_index, _ in grouped)
    extension_complete = (
        all(
            (f"{ARM}_s{seed}", EXTENSION_ROUND, "scripted") in grouped
            and (f"CTRL_s{seed}", EXTENSION_ROUND, "scripted") in grouped
            for seed in SEEDS
        )
        and not failed
    )
    initial = {key: value for key, value in grouped.items() if key[1] == INITIAL_ROUND}
    initial_metrics = {
        "vs_ctrl": {
            mix: {
                metric: _paired(initial, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        }
    }
    initial_decision = _decision(initial_metrics, statuses, False)
    if extension_started and initial_decision != "EXTEND":
        raise ValueError("round-67 rows require a round-66 EXTEND decision")
    active = grouped if extension_complete else initial
    data = {
        "vs_ctrl": {
            mix: {
                metric: _paired(active, mix, metric)
                for metric in ("episode_survival", "alive_survival", "mass")
            }
            for mix in MIXES
        },
        "mass_vs_incumbent": {mix: _incumbent(active, mix) for mix in MIXES},
        "shared_survivor_length_h5000": _shared_lengths(active),
        "telemetry": dict(telemetry or {}),
        "diagnostics": dict(diagnostics or {}),
        "failed_runs": [{"name": name, "exit": statuses[name]} for name in sorted(failed)],
        "unstarted_runs": sorted({f"{ARM}_s{seed}" for seed in SEEDS} - set(statuses)),
        "extension_started": extension_started,
        "extension_complete": extension_complete,
    }
    data["decision"] = _decision(data, statuses, extension_complete)
    if extension_started and not extension_complete and not failed:
        data["decision"] = "EXTEND"
    if extension_started:
        data["initial_decision"] = initial_decision
    return _json_safe(
        {
            "study": "T4c tightened two-sided development screen",
            "evidence": "development screen, not promotion evidence",
            "initial_round_index": INITIAL_ROUND,
            "extension_round_index": EXTENSION_ROUND,
            "terminal_statuses": dict(statuses),
            "arms": {ARM: data},
            "context": dict(context or {}),
        }
    )


def _json_safe(value: Any) -> Any:
    """Replace nonfinite descriptive values with JSON null, never an invented zero."""
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    return value


def extension_requirements(
    result: Mapping[str, Any], rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    data = result["arms"][ARM]
    if data["decision"] != "EXTEND" or data["extension_started"]:
        return []
    return [
        {"name": f"{arm}_s{seed}", "round_index": EXTENSION_ROUND, "mix": "scripted", "worlds": 96}
        for seed in SEEDS
        for arm in (ARM, "CTRL")
    ]


def markdown(result: Mapping[str, Any]) -> str:
    data = result["arms"][ARM]
    primary = data["vs_ctrl"]["scripted"]["episode_survival"]
    lines = [
        "# T4c tightened two-sided screen result",
        "",
        f"**Decision: {data['decision']}**. This is development-screen evidence only.",
        "",
        (
            "Primary scripted episode-survival delta vs CTRL: "
            f"{_show(primary['mean'])} "
            f"(seed SE {_show(primary['seed_se'])}; n={primary['seed_n']})."
        ),
        "",
        "## Mass delta vs CTRL by mix",
        "",
        "| mix | mean | seed SE |",
        "|---|---:|---:|",
    ]
    lines += [
        (
            f"| {mix} | {_show(data['vs_ctrl'][mix]['mass']['mean'])} "
            f"| {_show(data['vs_ctrl'][mix]['mass']['seed_se'])} |"
        )
        for mix in MIXES
    ]
    length = data["shared_survivor_length_h5000"]
    lines += [
        "",
        "## Required diagnostics",
        "",
        "### Final raw bank agreement",
        "",
        "| family | five-seed mean | n |",
        "|---|---:|---:|",
    ]
    diagnostics = data["diagnostics"]
    bank_means = diagnostics.get("bank_agreement_five_seed_means", {})
    lines += [
        f"| {name} | {_show(value.get('mean'))} | {value.get('n', 0)} |"
        for name, value in bank_means.items()
    ]
    lines += [
        "",
        "| seed | decisive rows/agreement | in-window rows/agreement | outside rows/agreement |",
        "|---:|---:|---:|---:|",
    ]
    for seed, value in diagnostics.get("bank_agreement_raw_six_action", {}).items():
        if value is not None:
            decisive = value["retained_decisive_all"]
            counter = value["in_window_counter"]
            outside = value["outside_counter"]
            lines.append(
                f"| {seed} | {decisive['rows']} / {_show(decisive['agreement'])} | "
                f"{counter['rows']} / {_show(counter['agreement'])} | "
                f"{outside['rows']} / {_show(outside['agreement'])} |"
            )
    lines += [
        "",
        "### Final 100k sampled telemetry",
        "",
        "| family | mean | n |",
        "|---|---:|---:|",
    ]
    telemetry = diagnostics.get("telemetry_final_100k", {}).get("five_seed_means", {})
    lines += [
        f"| {name} | {_show(value.get('mean'))} | {value.get('n', 0)} |"
        for name, value in telemetry.items()
    ]
    drift = diagnostics.get("drift_candidate_readout", {})
    lines += [
        "",
        "### Same-init drift readout (r62)",
        "",
        "| seed | raw | consequential |",
        "|---:|---:|---:|",
    ]
    lines += [
        f"| {seed} | {_show(value.get('raw_change_rate'))} | "
        f"{_show(value.get('consequential_change_rate'))} |"
        for seed, value in drift.get("per_seed", {}).items()
    ]
    lines += [
        "",
        "### Shared-survivor H5000 length",
        "",
        "| seed | shared survivors | candidate minus CTRL |",
        "|---:|---:|---:|",
    ]
    lines += [
        f"| {item['seed']} | {item['shared_survivors']} | {_show(item['mean_delta'])} |"
        for item in length["per_seed"]
    ]
    lines += ["", "## Context: T4a TWO_SIDED and T3", ""]
    for name in ("t4a", "t3"):
        value = result.get("context", {}).get(name)
        if not value:
            lines.append(f"{name}: unavailable.")
            continue
        lines.append(f"{name}: retained contextual development-screen analysis artifact.")
        for arm_name, arm_value in value.get("arms", {}).items():
            scripted = arm_value.get("vs_ctrl", {}).get("scripted", {})
            survival = scripted.get("episode_survival", {}).get("mean")
            mass = scripted.get("mass", {}).get("mean")
            lines.append(
                f"- {arm_name}: {arm_value.get('decision', 'unavailable')}; "
                f"scripted survival {_show(survival)}, mass {_show(mass)}."
            )
    lines += [
        "",
        "## Declared fork",
        "",
        str(
            data["diagnostics"].get(
                "declared_fork", "Diagnostic inputs incomplete; no declared fork can be assigned."
            )
        ),
    ]
    return "\n".join(lines) + "\n"


def _show(value: Any) -> str:
    return "null" if value is None else f"{float(value):.6f}"
