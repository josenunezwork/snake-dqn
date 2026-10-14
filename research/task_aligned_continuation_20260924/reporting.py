"""Presentation of saved continuation results during the admitted analysis stage.

No artifact, plotting package, evaluator or model is opened on import. The caller
supplies the already analyzed cells in a presentation-only ``cell_reports`` key.
This module does not repeat the scientific screen or reconstruct raw trajectories.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

PROFILES = ("solo", "food_pressure", "mixed")
POLICIES = ("parent", "candidate", "greedy_food", "random_safe")
METRICS = (
    "mass_integral",
    "ambient_food",
    "survival_fraction",
    "endpoint",
    "boost_fraction",
    "encounter_frames",
)
LINEAGE = 2026096102


def _hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _number(value: Any) -> float:
    if isinstance(value, bool):
        return float(value)
    if not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("Presentation requires finite saved numeric values")
    return float(value)


def _cells(spec: Mapping[str, Any], analysis: Mapping[str, Any]) -> dict:
    reports = analysis["cell_reports"]
    if (
        spec["lineages"] != [LINEAGE]
        or spec["horizon"] != 3000
        or analysis["kind"] != "paired_continuation_decision"
        or analysis["status"] != "PASS"
        or analysis["n_trained_lineages"] != 1
        or analysis["spec_sha256"] != _hash(spec)
        or [_hash(report) for report in reports] != analysis["source_report_sha256"]
    ):
        raise ValueError("Presentation inputs differ from the saved single-child decision")
    cells = {}
    for report in reports:
        policy = report["policy"]
        role = policy["name"] if policy["role"] == "anchor" else policy["role"]
        key = (report["profile"], role)
        if key in cells or key not in {(p, r) for p in PROFILES for r in POLICIES}:
            raise ValueError("Duplicate or unknown presentation cell")
        if report["spec_sha256"] != analysis["spec_sha256"] or report["status"] != "PASS":
            raise ValueError("Cell does not match completed analysis")
        if role in ("parent", "candidate") and (
            policy["lineage"] != LINEAGE
            or policy["checkpoint_sha256"] != spec["policy_sha256"][role]
        ):
            raise ValueError("Cell policy differs from sealed endpoint")
        cases = report["cases"]
        if len(cases) != 32 or [c["world_seed"] for c in cases] != spec["world_seeds"][key[0]]:
            raise ValueError("Presentation requires every fixed world in saved order")
        for row in [report["summary"], *cases]:
            for metric in METRICS:
                _number(row[metric])
        cells[key] = report
    if set(cells) != {(p, r) for p in PROFILES for r in POLICIES}:
        raise ValueError("Presentation requires the complete twelve-cell matrix")
    for profile in PROFILES:
        reference = [c["initial_sha256"] for c in cells[profile, "parent"]["cases"]]
        if any(
            [c["initial_sha256"] for c in cells[profile, role]["cases"]] != reference
            for role in POLICIES
        ):
            raise ValueError("Saved world identities are not paired")
    return cells


def _table(path: Path, rows: list[dict], title: str, note: str) -> list[Path]:
    csv_path, md_path = path.with_suffix(".csv"), path.with_suffix(".md")
    fields = list(rows[0])
    with csv_path.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    lines = [f"# {title}", "", note, "", "| " + " | ".join(fields) + " |"]
    lines.append("| " + " | ".join("---" for _ in fields) + " |")
    for row in rows:
        values = [
            f"{value:.6g}" if isinstance(value, float) else str(value) for value in row.values()
        ]
        lines.append("| " + " | ".join(values) + " |")
    with md_path.open("x") as stream:
        stream.write("\n".join(lines) + "\n")
    return [csv_path, md_path]


def _training(path: Path) -> dict[str, list[float]]:
    names = ("new_rollouts", "loss", "grad_norm", "mean_abs_q", "max_abs_q")
    series = {name: [] for name in names}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            for name in names:
                series[name].append(_number(row[name]))
    steps = series["new_rollouts"]
    if not steps or any(a >= b for a, b in zip(steps, steps[1:])):
        raise ValueError("Training telemetry is empty or rollout ordering changed")
    if steps[-1] != 31005:
        raise ValueError("Training telemetry does not reach the fixed continuation endpoint")
    return series


def _figure(figure: Any, path: Path, pyplot: Any) -> Path:
    try:
        with path.open("xb") as stream:
            figure.savefig(stream, format="png", dpi=140)
    finally:
        pyplot.close(figure)
    return path


def build_presentation(
    spec: Mapping[str, Any],
    analysis_report: Mapping[str, Any],
    training_log_path: Path,
    out: Path,
) -> list[Path]:
    """Publish complete saved tables and figures once, without scientific reruns.

    ``analysis_report`` is the completed analysis result augmented only for this
    call with ``cell_reports``: its ordered twelve already parsed input reports.
    The caller should not persist that duplicate corpus in the scientific report.
    All output is create-only; any partial failure is retained, with no retry.
    Returned paths are intended for the analysis stage's artifact manifest.
    """
    cells = _cells(spec, analysis_report)
    training = _training(training_log_path)
    out = Path(out)
    stems = ("all-cells", "all-cases", "paired-world-differences")
    names = [f"{stem}.{extension}" for stem in stems for extension in ("csv", "md")]
    names += [f"paired-{profile}.png" for profile in PROFILES] + ["training-telemetry.png"]
    if any((out / name).exists() for name in names):
        raise FileExistsError("Presentation output exists; no overwrite or partial restart")
    out.mkdir(parents=True, exist_ok=True)
    summaries, cases, differences = [], [], []
    for profile in PROFILES:
        for role in POLICIES:
            cell = cells[profile, role]
            summaries.append(
                {
                    "profile": profile,
                    "policy": role,
                    "worlds": 32,
                    **{metric: cell["summary"][metric] for metric in METRICS},
                    "checkpoint_sha256": cell["policy"].get("checkpoint_sha256", ""),
                }
            )
            for row in cell["cases"]:
                cases.append(
                    {
                        "profile": profile,
                        "policy": role,
                        "world_seed": row["world_seed"],
                        **{metric: row[metric] for metric in METRICS},
                        "initial_sha256": row["initial_sha256"],
                    }
                )
        for source, child in zip(
            cells[profile, "parent"]["cases"], cells[profile, "candidate"]["cases"]
        ):
            differences.append(
                {
                    "profile": profile,
                    "world_seed": source["world_seed"],
                    **{
                        metric: _number(child[metric]) - _number(source[metric])
                        for metric in METRICS
                    },
                    "initial_sha256": source["initial_sha256"],
                }
            )
    note = (
        f"Saved decision: **{analysis_report['decision']}**. One fixed child versus its source; "
        "no peak selection, champion comparison or population-of-trained-models claim. "
        "H3000 mass and alive-time denominators include post-death zeros. "
        "Parent means source; candidate means child. CSV preserves saved numeric precision."
    )
    paths = _table(out / "all-cells", summaries, "All twelve evaluation cells", note)
    paths += _table(out / "all-cases", cases, "All 384 evaluated cases", note)
    paths += _table(
        out / "paired-world-differences",
        differences,
        "All 96 paired world differences",
        "Descriptive child minus source differences; no new hypothesis test. " + note,
    )
    # Plotting is loaded only inside the admitted saved-data analysis invocation.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    displays = (
        ("mass_integral", "Mass per horizon frame", 1.0),
        ("ambient_food", "Ambient food", 1.0),
        ("survival_fraction", "Time alive (%)", 100.0),
    )
    with plt.rc_context({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}):
        for profile in PROFILES:
            figure, axes = plt.subplots(1, 3, figsize=(12, 4.8))
            figure.suptitle(f"6102 fixed continuation · {profile} · H3000 · 32 paired worlds")
            for axis, (metric, label, scale) in zip(axes, displays):
                for source, child in zip(
                    cells[profile, "parent"]["cases"], cells[profile, "candidate"]["cases"]
                ):
                    axis.plot(
                        [0, 1],
                        [source[metric] * scale, child[metric] * scale],
                        color="#6B8295",
                        alpha=0.22,
                        linewidth=0.8,
                    )
                means = [
                    cells[profile, role]["summary"][metric] * scale
                    for role in ("parent", "candidate")
                ]
                axis.scatter(
                    [0, 1], means, s=75, color=["#315A7D", "#C96728"], zorder=3, label="Mean"
                )
                pair = analysis_report["per_lineage"][str(LINEAGE)][profile][metric]
                delta, low, high = (
                    _number(pair[key]) * scale for key in ("mean_delta", "ci_low", "ci_high")
                )
                axis.set_title(
                    f"{label}\nΔ {delta:+.2f}; 95% CI [{low:+.2f}, {high:+.2f}]", fontsize=10
                )
                axis.set_xticks([0, 1], ["Source", "Child"])
                axis.set_xlim(-0.25, 1.25)
                axis.set_ylim(bottom=0)
                axis.grid(axis="y", alpha=0.15)
            figure.text(
                0.5,
                0.02,
                "Lines: paired worlds. Dots: means. Intervals copied from saved analysis; one trained child.",
                ha="center",
                fontsize=9,
            )
            figure.tight_layout(rect=(0, 0.06, 1, 0.92))
            paths.append(_figure(figure, out / f"paired-{profile}.png", plt))
        figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
        figure.suptitle("Training telemetry only · fixed 31,005-rollout continuation")
        for axis, metrics, title in zip(
            axes,
            (("loss",), ("grad_norm",), ("mean_abs_q", "max_abs_q")),
            ("Training loss", "Gradient norm", "Training absolute Q"),
        ):
            for metric in metrics:
                axis.plot(training["new_rollouts"], training[metric], linewidth=0.7, label=metric)
            axis.set_title(title)
            axis.set_xlabel("New continuation rollouts")
            axis.grid(alpha=0.15)
            axis.legend(fontsize=8)
        figure.text(
            0.5,
            0.02,
            "Saved training updates, without smoothing or endpoint selection. Not greedy evaluation or gameplay improvement.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.06, 1, 0.92))
        paths.append(_figure(figure, out / "training-telemetry.png", plt))
    return paths
