"""Render the T3 development diagnostic report from preserved numerical evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def number(value: float | None, digits: int = 3) -> str:
    """Format a finite metric or explicitly mark an empty event population."""
    return "—" if value is None else f"{value:.{digits}f}"


def quartiles(values: dict[str, Any], digits: int = 3) -> str:
    """Format Q25 / median / Q75 in a single table cell."""
    return " / ".join(number(values[key], digits) for key in ("q25", "q50", "q75"))


def table(headers: list[str], rows: list[list[Any]]) -> list[str]:
    """Return a Markdown table with explicit surrounding whitespace."""
    return [
        "",
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows],
        "",
    ]


def render(bank: dict[str, Any], replay: dict[str, Any]) -> str:
    """Build tables from complete evidence; do not infer a new T3 decision."""
    arms = ("LABELS", "LABELS_KEEPV8", "LABELS_KEEPV8_LO")
    counts = bank["sample"]["counts"]
    margin = bank["inputs"]["margin"]
    lines = [
        "# T3 label-arm diagnostics — 2026-10-16",
        "",
        "Development diagnostics, not gate evidence. These measurements do not change the T3 "
        "decision or authorize promotion. All three label arms are bare final networks at "
        "evaluation; KEEPV8 describes training anchors, not a serving veto.",
        "",
        "## Scope and evidence",
        "",
        "All five M6 CTRL finals and all 15 completed T3 finals (training seeds 10–14) are "
        "included. Every T3 run had `exit=0` in `status.txt` before inclusion, including seed 14. "
        "Trajectory diagnostics use only round 62, scripted mix, 96 identical world seeds per "
        "checkpoint, H=5000. Round 63 is excluded. Pairing is by training seed and world seed; "
        "the 480 seed/world pairs reuse the same 96 worlds, "
        "so they are not 480 independent worlds.",
        "",
        "All numerical work ran on CPU with one thread per process and at most four numerical "
        "workers. No MPS, tmux session control, or interaction with "
        "other owners' processes was used.",
        "",
        "The companion `t3_diagnostics_2026-10-16_evidence/` directory preserves bank results, "
        "replay summaries, sampled indices, input hashes, and replay receipts. Full per-frame "
        "NPZ series remain in the local artifact paths recorded in those receipts.",
        "",
        "## 1. Separation on the bank",
        "",
        f"The effective margin is **{margin['base_value']:.10f} × "
        f"{margin['label_bank_margin_scale']:.1f} = {margin['effective_value']:.10f} Q units**. "
        "This is well below the T1 median fatal-versus-survivor gap of 0.45 Q. The advisory "
        "note's proposed oversized-margin explanation is ruled out; the reported separation "
        "quartiles are also sub-unit rather than several Q units. This does not establish "
        "whether a different "
        "margin would improve gameplay.",
        "",
        f"Fixed sample seed `{bank['sample']['seed']}` selects 2,000 rows without replacement: "
        f"{counts['decisive_sample']:,} decisive and {counts['counter_sample']:,} counter rows, "
        f"preserving the bank's {counts['decisive_total']:,}/{counts['counter_total']:,} "
        "proportion "
        "to integer rounding. The same rows are used for every final. Separation is "
        "`max Q(stored surviving set) − Q(stored B8 action)`. Agreement uses raw six-action "
        "argmax, matching training `bank_agree`. The surviving sets are B8-continuation labels, "
        "not freshly established survival outcomes under these students.",
        "",
        "Pooled quartiles below are computed over all 5 × 2,000 model/row evaluations per arm, "
        "not averages of seed quartiles. They are descriptive; "
        "bank fit is not held-out performance.",
    ]
    rows = []
    for arm in ("CTRL", *arms):
        for group in ("overall", "decisive", "counter"):
            metric = bank["arm_pooled"][arm][group]
            rows.append(
                [
                    arm,
                    group,
                    metric["n"],
                    quartiles(metric["separation_quartiles"]),
                    f"{100 * metric['raw_agreement']:.2f}%",
                ]
            )
    lines += table(
        [
            "Arm",
            "Rows",
            "Model/row evaluations",
            "Separation Q25 / Q50 / Q75",
            "Argmax in surviving set",
        ],
        rows,
    )
    lines += ["Per-final results on the same 2,000 rows:"]
    lines += table(
        ["Final", "Separation Q25 / Q50 / Q75", "Raw agreement"],
        [
            [
                name,
                quartiles(metric["overall"]["separation_quartiles"]),
                f"{100 * metric['overall']['raw_agreement']:.2f}%",
            ]
            for name, metric in sorted(bank["results"].items())
        ],
    )
    lines += [
        "## 2. Scripted mass decomposition",
        "",
        "For each paired world and post-transition frame, dead mass is zero. The exact "
        "arm-minus-CTRL mass delta is the sum of (a) mass differences on frames when both are "
        "alive and (b) arm-only surviving mass minus CTRL-only surviving mass, divided by 5000. "
        "The second term captures death timing/incidence; its sign can favor an arm even when "
        "that arm loses overall mass. This is an accounting decomposition, not a causal estimate.",
    ]
    rows = []
    for arm in arms:
        m = replay["mass"][arm]["mean_components"]
        rows.append(
            [
                arm,
                number(m["total_delta"]),
                number(m["common_alive_growth"]),
                number(m["arm_only_survival"] - m["ctrl_only_survival"]),
                number(m["arm_only_survival"]),
                number(m["ctrl_only_survival"]),
            ]
        )
    lines += table(
        [
            "Arm vs CTRL",
            "Total Δ mass",
            "Both-alive size term",
            "Death timing/incidence term",
            "Arm-only mass",
            "CTRL-only mass",
        ],
        rows,
    )
    lines += ["Seed-paired decomposition (each row averages its 96 paired worlds):"]
    rows = []
    for arm in arms:
        for seed in replay["mass"][arm]["per_seed"]:
            rows.append(
                [
                    arm,
                    seed["seed"],
                    number(seed["total_delta"]),
                    number(seed["common_alive_growth"]),
                    number(seed["arm_only_survival"] - seed["ctrl_only_survival"]),
                ]
            )
    lines += table(["Arm", "Seed", "Total Δ", "Both-alive size", "Death timing/incidence"], rows)
    lines += [
        "Death-frame quartiles describe deaths only, with survivors censored at H5000. "
        "The fatal frame is one-based and contributes zero mass/alive frames."
    ]
    rows = []
    for arm in arms:
        d = replay["mass"][arm]["deaths"]
        rows.append(
            [
                arm,
                d["arm_count"],
                d["ctrl_count"],
                quartiles(d["arm_frame_quartiles"], 1),
                quartiles(d["ctrl_frame_quartiles"], 1),
            ]
        )
    lines += table(
        [
            "Arm",
            "Arm deaths / 480",
            "CTRL deaths / 480",
            "Arm death Q25/Q50/Q75",
            "CTRL death Q25/Q50/Q75",
        ],
        rows,
    )
    rows = []
    for arm in arms:
        for seed in replay["mass"][arm]["per_seed"]:
            d = seed["deaths"]
            rows.append(
                [
                    arm,
                    seed["seed"],
                    d["arm_count"],
                    d["ctrl_count"],
                    quartiles(d["arm_frame_quartiles"], 1),
                    quartiles(d["ctrl_frame_quartiles"], 1),
                ]
            )
    lines += table(
        [
            "Arm",
            "Seed",
            "Arm deaths / 96",
            "CTRL deaths / 96",
            "Arm death quartiles",
            "CTRL death quartiles",
        ],
        rows,
    )
    lines += [
        "The length table uses a **fixed paired cohort where both policies survive to frame "
        "5000** for all four timepoints. Means are post-transition hero logical length. "
        "The cohort can differ across label arms; its count is reported. "
        "These descriptive means pool matched survivor world pairs across seeds; "
        "seeds with more shared survivors contribute more pairs. Per-seed cohort means "
        "are retained in the evidence."
    ]
    rows = []
    for arm in arms:
        for frame in ("2000", "3000", "4000", "5000"):
            row = replay["mass"][arm]["common_survivor_length_delta"][frame]
            rows.append(
                [
                    arm,
                    frame,
                    row["worlds"],
                    number(row["arm_mean"], 1),
                    number(row["ctrl_mean"], 1),
                    number(row["arm_minus_ctrl_mean"], 1),
                ]
            )
    lines += table(
        ["Arm", "Frame", "Paired H5000 survivors", "Arm length", "CTRL length", "Δ length"], rows
    )
    lines += [
        "## 3. Where the final arms disagree",
        "",
        "Every living-hero decision on each CTRL final's round-62 scripted trajectory is "
        "evaluated by the matching-seed LABELS and LABELS_KEEPV8 finals on the same observation. "
        "CPU rowwise inference and advisory action masks match the original probes. "
        "These are action disagreements, not measured counterfactual outcomes. "
        "Boost disagreement means the actions differ in boost mode; remaining disagreements "
        "change direction within the same speed mode.",
    ]
    d = replay["disagreements"]
    rows = []
    for arm, result in d["arms"].items():
        for bucket in ("all", "lt400", "400_699", "700_999", "ge1000"):
            metric = result["overall"] if bucket == "all" else result["by_length"][bucket]
            label = {
                "all": "all",
                "lt400": "<400",
                "400_699": "400–<700",
                "700_999": "700–<1000",
                "ge1000": "≥1000",
            }[bucket]
            n, diff, boost = (
                metric["decisions"],
                metric["disagreements"],
                metric["boost_disagreements"],
            )
            rows.append(
                [
                    arm,
                    label,
                    n,
                    diff,
                    number(None if metric["rate"] is None else 100 * metric["rate"], 2) + "%",
                    boost,
                    diff - boost,
                ]
            )
    lines += table(
        [
            "Arm",
            "CTRL length",
            "Decisions",
            "Disagreements",
            "Rate",
            "Cross-boost disagreements",
            "Same-speed direction disagreements",
        ],
        rows,
    )
    lines += [
        "Q deciles use `−max` over all six raw CTRL Q outputs. Cutpoints pool all five CTRL "
        "trajectories once (not once per shadow model); D1 is lowest score and D10 highest. "
        "Ties go to the upper bin, so counts need not be exactly equal. Cross-seed Q scale "
        "differences can affect the pooled bins; per-seed counts/rates "
        "are retained in the evidence.",
        "",
        "Cutpoints: " + ", ".join(number(x, 5) for x in d["global_decile_cutpoints"]) + ".",
    ]
    rows = []
    for decile in map(str, range(1, 11)):
        for arm, result in d["arms"].items():
            metric = result["by_decile"][decile]
            rows.append(
                [
                    decile,
                    arm,
                    metric["decisions"],
                    metric["disagreements"],
                    number(None if metric["rate"] is None else 100 * metric["rate"], 2),
                    metric["boost_disagreements"],
                ]
            )
    lines += table(
        ["Decile", "Arm", "Decisions", "Disagreements", "Rate (%)", "Cross-boost disagreements"],
        rows,
    )
    lines += [
        "## Reading",
        "",
        "<!-- READING -->",
        "",
        "## Reproduction and validation",
        "",
        "<!-- VALIDATION -->",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rendered = render(json.loads(args.bank.read_text()), json.loads(args.replay.read_text()))
    args.out.write_text(rendered)
    print(
        json.dumps(
            {"output": str(args.out), "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}
        )
    )


if __name__ == "__main__":
    main()
