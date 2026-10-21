"""Publish the measured Mac tables without granting production qualification."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from research.bl_mac_latency.l1 import distribution
from research.bl_served_search.exports import export_bytes, export_json, verify_export


def _load(path: Path) -> dict[str, Any]:
    verify_export(path)
    return json.loads(path.read_text())


def _number(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def write_report(root: Path, y1_path: Path, cells_path: Path) -> dict[str, Any]:
    """Recompute all summary numbers from sealed receipts and raw samples."""
    y1, cells = _load(y1_path), _load(cells_path)
    lines = [
        "# Mac Y1 and L1′ measurements — 2026-10-21",
        "",
        "**Nominated cell: none. L1′ verdict: NOT_READY.**",
        "",
        "These are Mac measurements. The requested production reachable-root provider is absent; "
        "the stress roots have no valid production consumption path. No serving or "
        "release is authorized.",
        "",
        "## Y1 Mac dense batches",
        "",
        f"Receipt: `{y1_path}`; SHA256 `{verify_export(y1_path)}`. "
        f"{y1['completed_batches']} actual batches, CPU W={y1['workers']}, 30 per fixture. "
        "Quantiles use nearest rank. All 30 include the first cold sample; warm-29 "
        "values are retained in the receipt. "
        "Operator timing excludes guard/export overhead; raw ruler wall timing retains it.",
        "",
        "| Fixture | n | p50 (s) | p95 (s) | p99 (s) | max (s) |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, row in sorted(y1["latency_seconds"].items()):
        d = row["all30"]
        lines.append(
            f"| {name} | {d['n']} | {_number(d['p50'])} | {_number(d['p95'])} | "
            f"{_number(d['p99'])} | {_number(d['max'])} |"
        )
    lines += [
        "",
        "## Per-cell stress diagnostics",
        "",
        "S below is measured admission to first decoded coordinator visibility for "
        "frozen stress jobs. "
        "It is distinct from valid consumption latency C. CPU Watch6 game updates continue on the "
        "absolute 24fps schedule while workers run. Each cell retains per-length "
        "and per-burst results. "
        "K8 is a configured capacity; actual pending counts are reported separately. "
        "Cold worker initialization is included. These samples cannot qualify "
        "registered L1′ tails or coverage.",
        "",
        "| Cell | n | S p50 (s) | S p95 (s) | S p99 (s) | Tick overruns | Valid "
        "consumptions | Disposition |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    table = []
    for cell in cells["cells"]:
        attempts = cell.get("diagnostic_stress_attempts", [])
        samples = [
            sample
            for attempt in attempts
            for sample in attempt.get("samples", [])
            if "S_diagnostic_seconds" in sample
        ]
        dist = distribution(sample["S_diagnostic_seconds"] for sample in samples)
        overrun = sum(
            attempt.get("tick", {}).get("tick_overruns", 0)
            for attempt in attempts
            if isinstance(attempt.get("tick"), dict)
        )
        invalid = any(a.get("status") == "INVALID" for a in attempts)
        state = (
            "INVALID backend"
            if invalid
            else (
                "STOPPED"
                if any(a.get("status") == "STOPPED" for a in attempts)
                else (
                    "INCOMPLETE"
                    if any(a.get("status") == "INCOMPLETE" for a in attempts)
                    else "NOT_READY" if samples else "NOT_MEASURED"
                )
            )
        )
        by_length = {}
        for length in (800, 1200, 1500):
            subset = [
                sample["S_diagnostic_seconds"]
                for attempt in attempts
                if attempt.get("target_length") == length
                for sample in attempt.get("samples", [])
                if "S_diagnostic_seconds" in sample
            ]
            by_length[str(length)] = distribution(subset)
        table.append(
            {
                "cell_id": cell["cell_id"],
                "S_diagnostic": dist,
                "per_length": by_length,
                "tick_overruns": overrun,
                "valid_consumptions": 0,
                "disposition": state,
                "production_S_C_coverage": "NOT_MEASURED",
            }
        )
        lines.append(
            f"| {cell['cell_id']} | {dist['count']} | {_number(dist['p50_seconds'])} | "
            f"{_number(dist['p95_seconds'])} | {_number(dist['p99_seconds'])} | "
            f"{overrun if attempts else '—'} | 0 | {state} |"
        )
    lines += [
        "",
        "The governing coverage bar is 2.4 roots/s (at least 29 valid consumptions "
        "per 12s and 36 per 15s). "
        "The 1.6 sensitivity requires at least 20/24. X8 has not authorized the easier bar. "
        "These probes observed zero eligible consumptions because they have no "
        "production root provider; "
        "registered coverage is NOT_MEASURED, rather than an estimate of zero "
        "production capability.",
        "",
        "W1/K1 C p99 ceilings are 333/542 ms (unrounded contract values 333.333/541.667 ms). "
        "Expanded S p99 ceilings: W2 0.830s, W3 1.250s, W4 1.458s; W1/K8 retains "
        "the 1.458s tail cap. "
        "Zero overruns and Watch6 update p95 ≤11.688ms are required. Raw diagnostic S is not C. "
        "A matching stress result supplies neither exact-state consumption nor a production PASS.",
        "",
        "## Evidence and limits",
        "",
        f"- Cells receipt: `{cells_path}`; SHA256 `{verify_export(cells_path)}`.",
        "- Owner grants and canonical scoped probe authorization: "
        "`owner-authority.json`, `PROBE_ONLY.json`.",
        "- Phase-0 and quiet snapshots: `phase0-idle*.json`, Y1 host snapshots, and "
        "`root-occupancy/`.",
        "- First Y1 run overlapped local bl_info_value/reconstruction workers; its "
        "receipt and all samples remain retained. "
        "A separately completed quiet repetition, if present, is never spliced with it.",
        "- Complete controller/replay/114-root semantic acceptance, two repetitions "
        "in all four invalidation/start modes, "
        "saturated K8 trace, valid consumption C, and client latency/cadence remain unmeasured. "
        "Terminal-hero fixtures can lose hero inference after death; raw per-tick "
        "alive state is retained.",
        "- No Colab call, serving change, checkpoint/pin edit, prohibited-study "
        "access, or process/tmux termination occurred.",
        "",
    ]
    result = {
        "schema": "bl-mac-l1-result/v1",
        "verdict": "NOT_READY",
        "nominated_cell": None,
        "y1_receipt": str(y1_path),
        "y1_sha256": verify_export(y1_path),
        "cells_receipt": str(cells_path),
        "cells_sha256": verify_export(cells_path),
        "cell_table": table,
    }
    export_json(root / "result.json", result)
    export_bytes(root / "RESULT_L1.md", ("\n".join(lines)).encode())
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--y1", type=Path, required=True)
    parser.add_argument("--cells", type=Path, required=True)
    args = parser.parse_args()
    write_report(args.root, args.y1, args.cells)


if __name__ == "__main__":
    main()
