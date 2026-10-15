"""Render the dense T3 ceiling receipt without rewriting the sparse result."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

from research.t1_commitment_20261015.parity import write_json
from research.t3_rollout_labels_20261015.trigger import ROOT

ARMS = ("b8_s0", "b8_s3")
ENVS = range(32)
GO_FRACTION = 0.65
MARKER = "<!-- t3-dense-ceiling-addendum:start -->"
REPO = Path(__file__).resolve().parents[2]
REPORT_PATH = REPO / "docs/research/t3_ceiling_test_2026-10-15.md"


def _load(path: Path) -> dict[str, Any]:
    """Load one JSON receipt, retaining a path in malformed-receipt errors."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read receipt: {path}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"receipt must be a JSON object: {path}")
    return value


def _mean(rows: list[dict[str, Any]], key: str) -> float:
    return sum(float(row[key]) for row in rows) / len(rows)


def _metric_delta(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    return {
        "episode_survival": float(
            int(bool(left["episode_survival"])) - int(bool(right["episode_survival"]))
        ),
        "alive_frame_fraction": float(left["alive_frame_fraction"])
        - float(right["alive_frame_fraction"]),
        "mass_integral": float(left["mass_integral"]) - float(right["mass_integral"]),
    }


def _read_rows(root: Path, label: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    directory = root / "ceiling"
    summary = _load(directory / "summary.json")
    if summary.get("status") != "PASS" or summary.get("episodes") != 64:
        raise ValueError(f"{label} ceiling requires a PASS 64-episode summary")
    rows: list[dict[str, Any]] = []
    for arm in ARMS:
        for env in ENVS:
            row = _load(directory / f"{arm}_{env}.json")
            if row.get("status") != "PASS" or not row.get("prefix_identity"):
                raise ValueError(f"{label} identity check failed: {arm}_{env}")
            if row.get("arm") != arm or row.get("env") != env:
                raise ValueError(f"{label} receipt address mismatch: {arm}_{env}")
            if row.get("world_seed") is None or not row.get("checkpoint", {}).get("sha256"):
                raise ValueError(f"{label} receipt lacks world or checkpoint identity: {arm}_{env}")
            rows.append(row)
    return summary, rows


def _validate_pairing(dense: list[dict[str, Any]], sparse: list[dict[str, Any]]) -> None:
    """Require exact world/checkpoint identity and internally consistent endpoints."""
    sparse_index = {(row["arm"], row["env"]): row for row in sparse}
    if len(sparse_index) != 64:
        raise ValueError("sparse receipt keys are not unique")
    dense_source_identity = dense[0].get("source_identity")
    if not dense_source_identity:
        raise ValueError("dense receipts lack a common source identity")
    if any(row.get("source_identity") != dense_source_identity for row in dense):
        raise ValueError("dense receipts have inconsistent source identity")
    for dense_row in dense:
        key = (dense_row["arm"], dense_row["env"])
        sparse_row = sparse_index.get(key)
        if sparse_row is None:
            raise ValueError(f"dense receipt has no sparse pair: {key}")
        if dense_row["world_seed"] != sparse_row["world_seed"]:
            raise ValueError(f"world seed mismatch for {key}")
        if dense_row["checkpoint"]["sha256"] != sparse_row["checkpoint"]["sha256"]:
            raise ValueError(f"checkpoint SHA256 mismatch for {key}")
        dense_baseline = dense_row.get("baseline", {})
        sparse_baseline = sparse_row.get("baseline", {})
        for metric in ("survived", "survival_fraction", "mass_integral"):
            if dense_baseline.get(metric) != sparse_baseline.get(metric):
                raise ValueError(f"baseline endpoint mismatch for {key}: {metric}")
        for label, row in (("dense", dense_row), ("sparse", sparse_row)):
            baseline = row.get("baseline", {})
            if int(baseline.get("world_seed", row["world_seed"])) != int(row["world_seed"]):
                raise ValueError(f"{label} baseline world mismatch for {key}")
            expected = _metric_delta(
                row,
                {
                    **row,
                    "episode_survival": baseline["survived"],
                    "alive_frame_fraction": baseline["survival_fraction"],
                    "mass_integral": baseline["mass_integral"],
                },
            )
            actual = row.get("paired_delta", {})
            for metric, value in expected.items():
                actual_value = actual.get(metric)
                if actual_value is None or not math.isfinite(float(actual_value)):
                    raise ValueError(
                        f"{label} baseline delta missing or nonfinite for {key}: {metric}"
                    )
                if abs(float(actual_value) - value) > 1e-12:
                    raise ValueError(f"{label} baseline delta mismatch for {key}: {metric}")


def _groups(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | int]]:
    groups: dict[str, dict[str, float | int]] = {}
    for arm in (*ARMS, "pooled"):
        selected = rows if arm == "pooled" else [row for row in rows if row["arm"] == arm]
        baseline = [row["baseline"] for row in selected]
        groups[arm] = {
            "episodes": len(selected),
            "baseline_survivors": sum(int(bool(row["survived"])) for row in baseline),
            "dense_survivors": sum(int(bool(row["episode_survival"])) for row in selected),
            "baseline_episode_survival": _mean(baseline, "survived"),
            "dense_episode_survival": _mean(selected, "episode_survival"),
            "baseline_alive_frame_survival": _mean(baseline, "survival_fraction"),
            "dense_alive_frame_survival": _mean(selected, "alive_frame_fraction"),
            "baseline_mass": _mean(baseline, "mass_integral"),
            "dense_mass": _mean(selected, "mass_integral"),
        }
    return groups


def _append_report(section: str) -> None:
    """Replace only our marker block, preserving all sparse-report bytes exactly."""
    original = REPORT_PATH.read_text()
    prefix = original.split(MARKER, 1)[0]
    separator = "" if prefix.endswith("\n") else "\n"
    REPORT_PATH.write_text(prefix + separator + MARKER + "\n" + section.rstrip() + "\n")


def render_dense(root: Path = ROOT, bank_status: str | None = None) -> dict[str, Any]:
    """Render the declared dense ceiling result and dense-vs-sparse pairing artifacts."""
    dense_root = root / "dense"
    dense_summary, dense_rows = _read_rows(dense_root, "dense")
    sparse_summary, sparse_rows = _read_rows(root, "sparse")
    if any(row.get("trigger_mode") != "dense" for row in dense_rows):
        raise ValueError("dense receipts must declare trigger_mode=dense")
    if any(row.get("trigger_mode", "sparse") != "sparse" for row in sparse_rows):
        raise ValueError("sparse receipts must declare trigger_mode=sparse")
    _validate_pairing(dense_rows, sparse_rows)
    groups = _groups(dense_rows)
    pooled_survival = float(groups["pooled"]["dense_episode_survival"])
    verdict = "GO" if pooled_survival >= GO_FRACTION else "NO_GO"
    if bool(dense_summary.get("go")) != (verdict == "GO"):
        raise ValueError("dense summary verdict mismatch")
    sparse_index = {(row["arm"], row["env"]): row for row in sparse_rows}
    pairs = []
    for dense_row in dense_rows:
        sparse_row = sparse_index[(dense_row["arm"], dense_row["env"])]
        baseline = dense_row["baseline"]
        baseline_row = {
            "episode_survival": baseline["survived"],
            "alive_frame_fraction": baseline["survival_fraction"],
            "mass_integral": baseline["mass_integral"],
        }
        dense_vs_baseline = _metric_delta(dense_row, baseline_row)
        dense_vs_sparse = _metric_delta(dense_row, sparse_row)
        pairs.append(
            {
                "arm": dense_row["arm"],
                "env": dense_row["env"],
                "world_seed": dense_row["world_seed"],
                "baseline_episode_survival": int(bool(baseline["survived"])),
                "sparse_episode_survival": int(bool(sparse_row["episode_survival"])),
                "dense_episode_survival": int(bool(dense_row["episode_survival"])),
                **{f"dense_vs_baseline_{key}": value for key, value in dense_vs_baseline.items()},
                **{f"dense_vs_sparse_{key}": value for key, value in dense_vs_sparse.items()},
                "sparse_death_decision": sparse_row["death_t"],
                "sparse_death_length": sparse_row["death_length"],
                "dense_death_decision": dense_row["death_t"],
                "dense_death_length": dense_row["death_length"],
            }
        )
    total_decisions = sum(int(row["decisions"]) for row in dense_rows)
    scoped = sum(int(row["trigger_scope_decisions"]) for row in dense_rows)
    raw = sum(int(row["raw_trigger_count"]) for row in dense_rows)
    batches = sum(int(row["batch_count"]) for row in dense_rows)
    changes = sum(int(row["action_changes"]) for row in dense_rows)
    core_frames = sum(int(row.get("core_rollout_env_frames", 0)) for row in dense_rows)
    core_seconds = sum(float(row.get("core_rollout_wall_seconds", 0)) for row in dense_rows)
    status = bank_status or (
        "Part D eligible; generation pending" if verdict == "GO" else "Part D not run: dense NO_GO"
    )
    result: dict[str, Any] = {
        "schema": "t3-dense-ceiling-result/v1",
        "status": "PASS",
        "verdict": verdict,
        "gate": {"episodes": 64, "minimum_pooled_episode_survival": GO_FRACTION},
        "groups": groups,
        "total_decisions": total_decisions,
        "length_eligible_decisions": scoped,
        "raw_triggers": raw,
        "batches": batches,
        "action_changes": changes,
        "core_rollout_env_frames": core_frames,
        "core_rollout_worker_seconds": core_seconds,
        "rollout_env_frames_per_worker_second": core_frames / core_seconds if core_seconds else 0.0,
        "rollout_batches_per_worker_second": batches / core_seconds if core_seconds else 0.0,
        "raw_rate_all_decisions": raw / max(1, total_decisions),
        "capped_rate_all_decisions": batches / max(1, total_decisions),
        "raw_rate_length_eligible": raw / max(1, scoped),
        "capped_rate_length_eligible": batches / max(1, scoped),
        "wall_seconds": float(dense_summary["wall_seconds"]),
        "sparse_wall_seconds": float(sparse_summary["wall_seconds"]),
        "bank_status": status,
    }
    write_json(dense_root / "ceiling-result.json", result)
    with (dense_root / "ceiling-pairs.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(pairs[0]))
        writer.writeheader()
        writer.writerows(pairs)
    deaths = {
        arm: [
            row["death_length"]
            for row in dense_rows
            if row["arm"] == arm and not row["episode_survival"]
        ]
        for arm in ARMS
    }
    lines = [
        "## Dense ceiling addendum (declared before running)",
        "",
        f"**Dense gate verdict: {verdict}.** This is the ceiling of record: pooled scripted "
        f"episode survival is {groups['pooled']['dense_survivors']}/64 "
        f"({pooled_survival:.6f}) against the fixed GO threshold >= {GO_FRACTION:.2f}.",
        "",
        "The dense trigger is every fifth episode-local decision beginning with the first "
        "decision at length >=600; its clock advances across later length dips and it has no Q "
        "threshold. Every batch evaluates advisory-legal actions for D=300 or to H5000, with "
        "B8 greedy continuation, eps=0 and no veto; ties prefer B8, then Q, then lowest index. "
        "All 64 dense receipts PASS, and every "
        "arm/env pair matches sparse and B8 baseline world seeds and checkpoint SHA256s.",
        "",
        "| Arm | B8 episode survival | Sparse episode survival | Dense episode survival | "
        "B8 alive-frame | Sparse alive-frame | Dense alive-frame | B8 mass | Sparse mass | "
        "Dense mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for arm in (*ARMS, "pooled"):
        group = groups[arm]
        selected_sparse = (
            sparse_rows if arm == "pooled" else [row for row in sparse_rows if row["arm"] == arm]
        )
        sparse_base = [row["baseline"] for row in selected_sparse]
        lines.append(
            f"| {arm} | {sum(int(bool(row['survived'])) for row in sparse_base)}/"
            f"{len(selected_sparse)} "
            f"({_mean(sparse_base, 'survived'):.6f}) | "
            f"{sum(int(bool(row['episode_survival'])) for row in selected_sparse)}/"
            f"{len(selected_sparse)} "
            f"({_mean(selected_sparse, 'episode_survival'):.6f}) | "
            f"{group['dense_survivors']}/{group['episodes']} "
            f"({group['dense_episode_survival']:.6f}) | "
            f"{group['baseline_alive_frame_survival']:.6f} | "
            f"{_mean(selected_sparse, 'alive_frame_fraction'):.6f} | "
            f"{group['dense_alive_frame_survival']:.6f} | {group['baseline_mass']:.3f} | "
            f"{_mean(selected_sparse, 'mass_integral'):.3f} | {group['dense_mass']:.3f} |"
        )
    lines += [
        "",
        f"Dense wall time: {result['wall_seconds']:.2f}s. {batches:,} rollout batches, "
        f"{changes:,} changed actions, {raw:,} dense triggers on {total_decisions:,} decisions "
        f"({scoped:,} length-eligible): raw/capped trigger rates "
        f"{result['raw_rate_all_decisions']:.4%}/"
        f"{result['capped_rate_all_decisions']:.4%} overall and "
        f"{result['raw_rate_length_eligible']:.4%}/{result['capped_rate_length_eligible']:.4%} "
        "among eligible decisions.",
        "",
        "Raw triggers include terminal decisions with no advisory-legal action; those "
        "cannot execute a rollout batch.",
        "",
        f"Continuation core: {core_frames:,} env-frames in {core_seconds:.2f} "
        "summed worker-seconds "
        f"({result['rollout_env_frames_per_worker_second']:.1f} env-frames/s and "
        f"{result['rollout_batches_per_worker_second']:.4f} decision batches/s per worker). "
        "This includes clone construction; ordinary episode stepping and startup are outside "
        "that core timing denominator.",
        "",
        "Dense death lengths are actual pre-action lengths: "
        + "; ".join(f"{arm}: {len(lengths)} deaths {lengths}" for arm, lengths in deaths.items())
        + ".",
        "",
        f"Part D status: {status}.",
        "",
        "### Dense per-world paired changes",
        "",
        "| Arm | Env | World seed | B8 alive | Sparse alive | Dense alive | "
        "Dense-B8 episode | "
        "Dense-B8 alive-frame | Dense-B8 mass | Dense-Sparse episode | "
        "Dense-Sparse alive-frame | Dense-Sparse mass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for pair in pairs:
        lines.append(
            f"| {pair['arm']} | {pair['env']} | {pair['world_seed']} | "
            f"{pair['baseline_episode_survival']} | {pair['sparse_episode_survival']} | "
            f"{pair['dense_episode_survival']} | "
            f"{pair['dense_vs_baseline_episode_survival']:+.0f} | "
            f"{pair['dense_vs_baseline_alive_frame_fraction']:+.6f} | "
            f"{pair['dense_vs_baseline_mass_integral']:+.3f} | "
            f"{pair['dense_vs_sparse_episode_survival']:+.0f} | "
            f"{pair['dense_vs_sparse_alive_frame_fraction']:+.6f} | "
            f"{pair['dense_vs_sparse_mass_integral']:+.3f} |"
        )
    _append_report("\n".join(lines))
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--bank-status", type=str, default=None)
    args = parser.parse_args()
    render_dense(args.root, args.bank_status)


if __name__ == "__main__":
    main()
