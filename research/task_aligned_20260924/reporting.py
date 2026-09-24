"""Saved-data presentation for the admitted task-aligned analysis stage.

Importing this module does not import matplotlib, parse artifacts, or calculate
scientific results. render_reports consumes the completed decision and cell
summaries; it never invokes an evaluator, analyzer, simulator, or model loader.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

PROFILES = ("solo", "food_pressure", "mixed")
ANCHORS = ("greedy_food", "random_safe")
COUNTERS = (
    "native_frames",
    "hero_transitions",
    "model_forwards",
    "checkpoint_loads",
    "optimizer_updates",
)
METRICS = (
    "mass_integral",
    "ambient_food",
    "survival_fraction",
    "endpoint",
    "boost_fraction",
    "encounter_frames",
)
COLORS = ("#315A7D", "#C96728", "#728779", "#92989F")


def _finite(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"Non-numeric saved presentation field: {label}")
    if not math.isfinite(value):
        raise ValueError(f"Nonfinite saved presentation field: {label}")
    return float(value)


def _sha(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _json_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _cells(
    spec: Mapping[str, Any], reports: Sequence[Mapping[str, Any]], decision: Mapping[str, Any]
) -> dict[tuple[str, Any, str], Mapping[str, Any]]:
    """Check presentation coverage/identity, without repeating scientific analysis."""
    if [_json_hash(report) for report in reports] != decision["source_report_sha256"]:
        raise ValueError("Presentation cells differ from the already analyzed report corpus")
    expected = {
        (role, seed, profile)
        for role in ("parent", "candidate")
        for seed in spec["lineages"]
        for profile in PROFILES
    }
    expected |= {("anchor", name, profile) for name in ANCHORS for profile in PROFILES}
    cells = {}
    for report in reports:
        policy = report["policy"]
        identity = policy.get("name") if policy["role"] == "anchor" else policy["lineage"]
        key = (policy["role"], identity, report["profile"])
        if key in cells or report["spec_sha256"] != decision["spec_sha256"]:
            raise ValueError("Duplicate cell or differing spec identity in presentation")
        if len(report["cases"]) != 32 or report["status"] != "PASS":
            raise ValueError("Presentation requires completed fixed 32-world cells")
        for name in METRICS:
            _finite(report["summary"][name], name)
        cells[key] = report
    if set(cells) != expected or len(spec["lineages"]) != 3:
        raise ValueError("Presentation requires all three lineages, profiles, and shared anchors")
    if decision["spec_sha256"] != _json_hash(spec):
        raise ValueError("Presentation spec differs from the saved decision")
    if set(decision["per_lineage"]) != {str(seed) for seed in spec["lineages"]}:
        raise ValueError("Saved decision omits a trained lineage")
    return cells


def _rows(spec: Mapping[str, Any], cells: Mapping[tuple[str, Any, str], Mapping[str, Any]]) -> list:
    rows = []
    for profile in PROFILES:
        identities = [(role, seed) for seed in spec["lineages"] for role in ("parent", "candidate")]
        identities += [("anchor", name) for name in ANCHORS]
        for role, identity in identities:
            report = cells[(role, identity, profile)]
            rows.append(
                {
                    "profile": profile,
                    "role": role,
                    "lineage": identity if role != "anchor" else "",
                    "anchor": identity if role == "anchor" else "",
                    "world_count": 32,
                    "horizon": spec["horizon"],
                    **{name: report["summary"][name] for name in METRICS},
                    "checkpoint_sha256": report["policy"].get("checkpoint_sha256", ""),
                }
            )
    return rows


def _write_tables(
    rows: Sequence[Mapping[str, Any]], decision: Mapping[str, Any], out: Path
) -> None:
    with (out / "all-cells.csv").open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "# Final fixed-endpoint evaluation",
        "",
        f"Saved decision: **{decision['decision']}**. This is a package-utility screen, "
        "not promotion or a causal factor result.",
        "",
        "All means use the complete 3,000-frame horizon; post-death mass and alive time "
        "contribute zero. Scripted anchors were evaluated once per profile and are shared "
        "across trained lineages. No best checkpoint or seed was selected.",
        "",
        "| Policy | Profile | Mass/frame | Ambient food | Alive % | Endpoint % | "
        "Boost % | Encounter frames |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        label = row["anchor"] or f"{row['role']} {row['lineage']}"
        lines.append(
            f"| {label} | {row['profile']} | {row['mass_integral']:.3f} | "
            f"{row['ambient_food']:.3f} | {100 * row['survival_fraction']:.2f} | "
            f"{100 * row['endpoint']:.2f} | {100 * row['boost_fraction']:.2f} | "
            f"{row['encounter_frames']:.2f} |"
        )
    lines += [
        "",
        "The CSV preserves all saved numeric summary values. Paired confidence "
        "intervals in the figures are copied from the completed decision: 32 world "
        "families for each fixed lineage/profile, not uncertainty over a population "
        "of trained models. Training telemetry is shown separately and does not "
        "establish greedy gameplay improvement.",
        "",
    ]
    with (out / "all-cells.md").open("x") as stream:
        stream.write("\n".join(lines))


def _training_paths(
    training_logs: Mapping[int, str | Path] | Sequence[str | Path], lineages: Sequence[int]
) -> dict[int, Path]:
    if isinstance(training_logs, Mapping):
        paths = {int(seed): Path(path) for seed, path in training_logs.items()}
    else:
        if len(training_logs) != len(lineages):
            raise ValueError("Exactly three named training logs required")
        paths = {seed: Path(path) for seed, path in zip(lineages, training_logs)}
    if set(paths) != set(lineages) or len(set(path.resolve() for path in paths.values())) != 3:
        raise ValueError("Training logs must bind each lineage to a distinct exact path")
    return paths


def _training_series(path: Path) -> dict[str, list[float]]:
    result = {"rollouts": [], "loss": [], "valid_rows": []}
    with path.open() as stream:
        for line in stream:
            row = json.loads(line)
            result["rollouts"].append(_finite(row["new_rollouts"], "new_rollouts"))
            result["loss"].append(_finite(row["loss"], "loss"))
            result["valid_rows"].append(
                _finite(row["counters"]["hero_transitions"], "actual hero transitions")
            )
    if not result["rollouts"]:
        raise ValueError("Training log is empty")
    if any(a >= b for a, b in zip(result["rollouts"], result["rollouts"][1:])):
        raise ValueError("Training log rollout ordering differs")
    if any(a > b for a, b in zip(result["valid_rows"], result["valid_rows"][1:])):
        raise ValueError("Actual cumulative valid hero rows went backwards")
    return result


def _save_figure(figure: Any, path: Path, pyplot: Any) -> None:
    try:
        with path.open("xb") as stream:
            figure.savefig(
                stream, format="png", dpi=140, metadata={"Software": "task-aligned-saved-report/v1"}
            )
    finally:
        pyplot.close(figure)


def render_reports(
    spec: Mapping[str, Any],
    cell_reports: Sequence[Mapping[str, Any]],
    decision: Mapping[str, Any],
    training_logs: Mapping[int, str | Path] | Sequence[str | Path],
    out: str | Path,
) -> dict[str, Any]:
    """Render existing summaries and telemetry, with zero scientific operations.

    Args:
        spec: The final augmented evaluation specification used by all cells.
        cell_reports: All 24 completed reports, already scientifically analyzed.
        decision: The existing analyze_reports result; never recalculated here.
        training_logs: Three explicit training.jsonl paths, keyed by source
            lineage or ordered exactly like spec['lineages'].
        out: Presentation directory. Existing output files cause a hard stop.

    Returns:
        A manifest with artifact hashes, source identities, and zero counters.
    """
    cells = _cells(spec, cell_reports, decision)
    lineages = tuple(spec["lineages"])
    paths = _training_paths(training_logs, lineages)
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    figures = [f"paired-{seed}-{profile}.png" for seed in lineages for profile in PROFILES]
    names = ["all-cells.csv", "all-cells.md", *figures, "training-telemetry.png"]
    if any((destination / name).exists() for name in [*names, "presentation.json"]):
        raise FileExistsError("Presentation output exists; do not replace completed evidence")
    training = {seed: _training_series(path) for seed, path in paths.items()}
    # Presentation imports are deliberately confined to the admitted invocation.
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    _write_tables(_rows(spec, cells), decision, destination)
    labels = ("Parent", "Candidate", "Greedy", "RandomSafe")
    display = (
        ("mass_integral", "Living mass per frame", 1.0),
        ("ambient_food", "Ambient foods per episode", 1.0),
        ("survival_fraction", "Time alive (%)", 100.0),
    )
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        for seed in lineages:
            for profile in PROFILES:
                figure, axes = plt.subplots(1, 3, figsize=(12, 4.5))
                figure.suptitle(f"Lineage {seed} · {profile} · fixed H{spec['horizon']} endpoint")
                keys = [
                    ("parent", seed, profile),
                    ("candidate", seed, profile),
                    ("anchor", "greedy_food", profile),
                    ("anchor", "random_safe", profile),
                ]
                for axis, (metric, title, scale) in zip(axes, display):
                    values = [cells[key]["summary"][metric] * scale for key in keys]
                    axis.bar(range(4), values, color=COLORS, width=0.68)
                    axis.set_xticks(range(4), labels, rotation=15, ha="right")
                    axis.set_title(title, fontsize=11)
                    axis.set_ylim(0, max(1.0, max(values)) * 1.32)
                    axis.grid(axis="y", alpha=0.18)
                    axis.set_axisbelow(True)
                    pair = decision["per_lineage"][str(seed)][profile][metric]
                    delta, low, high = (
                        _finite(pair[name], name) * scale
                        for name in ("mean_delta", "ci_low", "ci_high")
                    )
                    axis.text(
                        0.02,
                        0.98,
                        f"Δ {delta:+.2f}; 95% CI [{low:+.2f}, {high:+.2f}]",
                        transform=axis.transAxes,
                        va="top",
                        fontsize=9,
                    )
                figure.text(
                    0.5,
                    0.025,
                    "Δ and interval: candidate minus parent, paired over 32 worlds. "
                    "Anchors are shared controls. Package utility only; no promotion.",
                    ha="center",
                    fontsize=9,
                )
                figure.tight_layout(rect=(0, 0.07, 1, 0.91))
                _save_figure(figure, destination / f"paired-{seed}-{profile}.png", plt)
        figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        for seed, color in zip(lineages, COLORS):
            series = training[seed]
            axes[0].plot(
                series["rollouts"],
                series["loss"],
                color=color,
                alpha=0.75,
                linewidth=0.75,
                label=str(seed),
                rasterized=True,
            )
            axes[1].plot(
                series["rollouts"],
                series["valid_rows"],
                color=color,
                linewidth=1.3,
                label=str(seed),
            )
        axes[0].set(title="Recorded native TD loss", xlabel="New completed rollouts", ylabel="Loss")
        axes[1].set(
            title="Actual valid hero experience",
            xlabel="New completed rollouts",
            ylabel="Cumulative valid hero transitions",
        )
        for axis in axes:
            axis.grid(alpha=0.18)
            axis.legend(title="Source lineage", fontsize=9)
        figure.suptitle("Training telemetry — every saved rollout, all three lineages")
        figure.text(
            0.5,
            0.025,
            "Loss and collected experience are training measurements. "
            "They are separate from fixed-endpoint greedy gameplay outcomes.",
            ha="center",
            fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.07, 1, 0.92))
        _save_figure(figure, destination / "training-telemetry.png", plt)
    manifest = {
        "schema_version": 1,
        "study_id": spec["study_id"],
        "kind": "saved_presentation",
        "status": "PASS",
        "decision": decision["decision"],
        "spec_sha256": decision["spec_sha256"],
        "source_decision_sha256": _json_hash(decision),
        "source_cell_report_sha256": decision["source_report_sha256"],
        "training_logs": [
            {"lineage": seed, "path": str(paths[seed]), "sha256": _sha(paths[seed])}
            for seed in lineages
        ],
        "artifacts": [{"path": name, "sha256": _sha(destination / name)} for name in names],
        "counters": dict.fromkeys(COUNTERS, 0),
        "promotion_eligible": False,
        "new_scientific_analysis": False,
    }
    with (destination / "presentation.json").open("x") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    return manifest
