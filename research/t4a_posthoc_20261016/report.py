"""Common-distribution summaries for the descriptive T4a post-hoc report."""

from __future__ import annotations

from typing import Any, Mapping

SETS = (
    "a_decisive_bank_rows",
    "b_counter_bank_rows",
    "c_ctrl_trajectory_outside_bank_rows",
)


def standardize(summary: Mapping[str, Any]) -> dict[str, Any]:
    """Weight a/b/c to one pooled-bank length/decile distribution on common support.

    Restricting support alone leaves different length mixtures in the three
    populations. This descriptive standardization also matches their weights.
    It does not infer unobserved outcomes or create a significance threshold.
    """
    matched = summary["matched_common_support"]
    support = matched[SETS[0]]["support"]
    if any(matched[name]["support"] != support for name in SETS):
        raise ValueError("a/b/c must use identical common support")
    if not support:
        raise ValueError("no common support to standardize")

    def row(name: str, stratum: str) -> Mapping[str, Any]:
        bucket, decile = stratum.split(":D")
        return matched[name]["rates"]["by_length_and_decile"][bucket][decile]

    counts = {
        stratum: sum(row(name, stratum)["decisions"] for name in SETS[:2]) for stratum in support
    }

    def group(strata: list[str]) -> dict[str, Any]:
        total = sum(counts[key] for key in strata)
        if not total:
            return {"bank_weight_count": 0, "sets": None}
        weights = {key: counts[key] / total for key in strata}
        rates = {}
        for name in SETS:
            selected = [row(name, key) for key in strata]
            if any(item["decisions"] <= 0 for item in selected):
                raise ValueError("empty row within declared common support")
            change = sum(weights[key] * row(name, key)["argmax_change_rate"] for key in strata)
            cross = sum(
                weights[key] * row(name, key)["cross_boost_rate_all_decisions"] for key in strata
            )
            rates[name] = {
                "observed_decisions": sum(item["decisions"] for item in selected),
                "argmax_change_rate": change,
                "cross_boost_rate_all_decisions": cross,
                "cross_boost_share_of_argmax_changes": cross / change if change else None,
            }
        return {"bank_weight_count": total, "stratum_weights": weights, "sets": rates}

    return {
        "method": (
            "a/b/c standardized to identical pooled decisive+counter bank stratum weights "
            "on their common length-bucket x CTRL raw -max-Q-decile support; "
            "within-decile weights renormalized over supported length buckets"
        ),
        "overall": group(support),
        "by_decile": {
            str(decile): group([key for key in support if key.endswith(f":D{decile}")])
            for decile in range(1, 11)
        },
    }


def _percent(value: float | None) -> str:
    return "—" if value is None else f"{value * 100:.2f}%"


def render(summary: Mapping[str, Any], interpretation: str) -> str:
    """Render completed diagnostic tables and the verbatim declared interpretation."""
    lines = [
        "# T4a post-hoc spillover and growth diagnostics — 2026-10-16",
        "",
        "Diagnostic only. These results do not change GO/KILL/EXTEND or authorize promotion.",
        "",
        "## Scope and measurement",
        "",
        "All five TWO_SIDED finals and corresponding CTRL, LABELS and LABELS_KEEPV8 finals "
        "(training seeds 10–14) are included. The shared diagnostic worlds are the 96 "
        "round-62 scripted H5000 worlds used in the T3 report. These are separate from "
        "the registered round-65 gameplay evaluation. The same worlds recur across "
        "training seeds; the 480 seed/world pairs are not 480 independent worlds.",
        "",
        "Action changes use raw six-action first argmax for all three sets. Decile scores "
        "are −max of raw CTRL Q; common cutpoints pool all five CTRL trajectories once. "
        "D1 is the lowest score and D10 the highest; ties enter the upper bin. Bank sides "
        "use original rollout survival of the stored B8 action. The bank is the complete "
        "v2 bank, with preferred surviving labels for decisive agreement and exact stored "
        "B8 action for counter agreement. Set (c) excludes exact canonical bank-observation "
        "fingerprints. These are action comparisons, not counterfactual survival outcomes.",
        "",
        "The tables standardize (a), (b) and (c) to identical pooled-bank weights within "
        "common length-bucket × CTRL-Q-decile strata. Length buckets are <400, 400–<700, "
        "700–<1000 and ≥1000. Within each decile the common length weights are renormalized. "
        "Unsupported strata are excluded without extrapolation; raw counts, exclusions, "
        "stratum rates and weights are preserved in the JSON evidence. Cross-boost share "
        "is the standardized cross-speed change rate divided by the standardized total "
        "action-change rate, not a mean of stratum percentages.",
        "",
        "## Matched spillover",
        "",
        "| Arm | State set | Retained decisions | Argmax change | Cross-boost share of changes |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    labels = ("(a) decisive bank", "(b) counter bank", "(c) outside-bank CTRL")
    for arm, result in summary["arms"].items():
        for key, label in zip(SETS, labels):
            row = result["standardized"]["overall"]["sets"][key]
            lines.append(
                f"| {arm} | {label} | {row['observed_decisions']} | "
                f"{_percent(row['argmax_change_rate'])} | "
                f"{_percent(row['cross_boost_share_of_argmax_changes'])} |"
            )
    lines += [
        "",
        "Each decile cell is **argmax-change rate (cross-boost share of changes)**.",
        "",
        "| Decile | Arm | (a) Decisive bank | (b) Counter bank | (c) Outside-bank CTRL |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for decile in range(1, 11):
        for arm, result in summary["arms"].items():
            sets = result["standardized"]["by_decile"][str(decile)]["sets"]
            cells = [
                (
                    "—"
                    if sets is None
                    else (
                        f"{_percent(sets[key]['argmax_change_rate'])} "
                        f"({_percent(sets[key]['cross_boost_share_of_argmax_changes'])})"
                    )
                )
                for key in SETS
            ]
            lines.append(f"| {decile} | {arm} | " + " | ".join(cells) + " |")
    lines += [
        "",
        "## Growth on shared survivors",
        "",
        "The primary curve uses worlds alive under both policies at each named frame. "
        "The cohort can change with time and differs by arm. Means pool matched survivor "
        "pairs across training seeds. The fixed-H5000 curve below uses the same cohort "
        "at every timepoint and is directly comparable to the T3 report.",
    ]
    for field, label in (
        ("frames", "Frame-specific shared survivors"),
        ("fixed_h5000_shared_survivor_frames", "Fixed H5000 shared survivors"),
    ):
        lines += [
            "",
            f"### {label}",
            "",
            "| Arm | Frame | Shared survivors | Arm mean length | CTRL mean length | Δ length |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for arm, result in summary["arms"].items():
            for frame, row in result["growth"][field].items():
                values = [
                    row[key]
                    for key in (
                        "candidate_mean_alive_length",
                        "ctrl_mean_alive_length",
                        "candidate_minus_ctrl_mean_alive_length",
                    )
                ]
                cells = ["—" if value is None else f"{value:.2f}" for value in values]
                lines.append(
                    f"| {arm} | {frame} | {row['shared_survivors']} | " + " | ".join(cells) + " |"
                )
    lines += [
        "",
        "## Bank target agreement",
        "",
        "These full-bank final-checkpoint agreements are distinct from agreement with "
        "CTRL's argmax. Decisive targets use the v2 preferred surviving set; counter "
        "targets require the stored B8 action. All arms use the same v2 rows for contrast.",
        "",
        "| Arm | Seed | Decisive target agreement | Counter target agreement |",
        "| --- | --- | ---: | ---: |",
    ]
    for arm, result in summary["arms"].items():
        for seed, row in result["per_seed"].items():
            agree = row["bank_agreement"]
            lines.append(
                f"| {arm} | {seed} | {_percent(agree['decisive'])} | "
                f"{_percent(agree['counter'])} |"
            )
        agree = result["bank_agreement"]
        lines.append(
            f"| {arm} | five-seed mean | {_percent(agree['decisive'])} | "
            f"{_percent(agree['counter'])} |"
        )
    lines += [
        "",
        "Registered TWO_SIDED final-100k training-window readout:",
        "",
        "| Seed | Transitions | Decisive agreement | Counter agreement | Greedy boost |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    windows = []
    for seed, row in summary["arms"]["TWO_SIDED"]["per_seed"].items():
        window = row["final_training_window"]
        windows.append(window)
        lines.append(
            f"| {seed} | {window['transitions']} | "
            f"{_percent(window['bank_agree_decisive'])} | "
            f"{_percent(window['bank_agree_counter'])} | "
            f"{_percent(window['greedy_boost_rate'])} |"
        )
    means = [
        sum(row[key] for row in windows) / len(windows)
        for key in (
            "bank_agree_decisive",
            "bank_agree_counter",
            "greedy_boost_rate",
        )
    ]
    lines.append("| five-seed mean | — | " + " | ".join(map(_percent, means)) + " |")
    lines += ["", "## Declared interpretation (verbatim)", "", interpretation.strip(), ""]
    return "\n".join(lines)
