"""Paired summaries with death censoring and explicit spatial-divergence criteria."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

OFFSETS = (0, 10, 25, 50, 100)
METRICS = (
    "head_region",
    "largest_region",
    "head_in_largest",
    "tail_bfs",
    "tail_reachable",
    "tail_manhattan",
    "own_cells_r15",
    "own_cells_r30",
    "opponent_cells_r15",
    "opponent_cells_r30",
    "boundary_own",
    "boundary_opponent",
    "boundary_wall",
    "toward_food",
    "food_in_smaller_component",
    "race_16",
    "race_32",
    "race_64",
    "race_128",
    "race_256",
)


def paired_summary(rows: list[dict], offset: int, metric: str) -> dict:
    """Exclude missing/dead endpoints; never substitute zero for unreachable BFS."""
    deltas, fatal, surviving = [], [], []
    alive_pairs = 0
    for row in rows:
        f = {t["offset"]: t for t in row["fatal"]["trace"]}.get(offset)
        s = {t["offset"]: t for t in row["surviving"]["trace"]}.get(offset)
        if f is None or s is None:
            continue
        alive_pairs += 1
        fv = f["tail_bfs"] is not None if metric == "tail_reachable" else f[metric]
        sv = s["tail_bfs"] is not None if metric == "tail_reachable" else s[metric]
        if fv is None or sv is None:
            continue
        fatal.append(float(fv))
        surviving.append(float(sv))
        deltas.append(float(fv) - float(sv))
    a = np.asarray(deltas)
    return {
        "n": len(deltas),
        "alive_pairs": alive_pairs,
        "quartiles": np.quantile(a, [0.25, 0.5, 0.75]).tolist() if len(a) else None,
        "fatal_median": float(np.median(fatal)) if fatal else None,
        "surviving_median": float(np.median(surviving)) if surviving else None,
        "negative": int((a < 0).sum()),
        "zero": int((a == 0).sum()),
        "positive": int((a > 0).sum()),
        "direction_consistency": (
            float(max((a < 0).sum(), (a > 0).sum()) / len(a)) if len(a) else None
        ),
    }


def spatial_divergence(row: dict) -> dict:
    """Locate the first smaller fatal pocket; this is a descriptive proxy, not causality."""
    surviving = {t["offset"]: t for t in row["surviving"]["trace"]}
    for f in row["fatal"]["trace"]:
        s = surviving.get(f["offset"])
        if (
            f["offset"] > 0
            and s is not None
            and not f["head_in_largest"]
            and f["head_region"] < s["head_region"]
        ):
            return {
                "offset": f["offset"],
                "outside15": f["boundary_max_chebyshev"] > 15,
                "boundary_max_chebyshev": f["boundary_max_chebyshev"],
                "boundary_outside15_fraction": f["boundary_outside15_fraction"],
                "survivor_in_largest": s["head_in_largest"],
            }
    return {"offset": None, "outside15": None}


def summarize(rows: list[dict]) -> dict:
    """Retain all per-offset distributions and a reproducible descriptive ranking."""
    assert len({(r["arm"], r["env"]) for r in rows}) == len(rows)
    by_offset = {
        str(offset): {m: paired_summary(rows, offset, m) for m in METRICS} for offset in range(101)
    }
    ranking = []
    for metric in METRICS:
        first_median = None
        first_consistent = None
        negative_runs: list[list[int]] = []
        for offset in range(101):
            result = by_offset[str(offset)][metric]
            if result["n"] >= 20 and result["quartiles"][1] != 0 and first_median is None:
                first_median = offset
            if (
                result["n"] >= 20
                and result["direction_consistency"] >= 0.75
                and first_consistent is None
            ):
                first_consistent = offset
            if result["n"] >= 20 and result["negative"] / result["n"] >= 0.75:
                if not negative_runs or offset != negative_runs[-1][-1] + 1:
                    negative_runs.append([offset])
                else:
                    negative_runs[-1].append(offset)
        ranking.append(
            {
                "metric": metric,
                "first_nonzero_median_n20": first_median,
                "first_75pct_same_sign_n20": first_consistent,
                "negative_75pct_offsets": sum(len(run) for run in negative_runs),
                "negative_75pct_runs": [[run[0], run[-1]] for run in negative_runs],
            }
        )
    spatial = [{"arm": r["arm"], "env": r["env"], **spatial_divergence(r)} for r in rows]
    resolved = [s for s in spatial if s["offset"] is not None]
    outside = sum(s["outside15"] for s in resolved)
    branch_counts = {}
    for offset in OFFSETS:
        pairs = []
        for row in rows:
            f = {t["offset"]: t for t in row["fatal"]["trace"]}.get(offset)
            s = {t["offset"]: t for t in row["surviving"]["trace"]}.get(offset)
            if f is not None and s is not None:
                pairs.append((f, s))
        counts: dict = {"n": len(pairs)}
        for index, branch in enumerate(("fatal", "surviving")):
            counts[branch] = {
                "head_in_largest": sum(p[index]["head_in_largest"] for p in pairs),
                "tail_reachable": sum(p[index]["tail_bfs"] is not None for p in pairs),
                "boundary_majority_own": sum(p[index]["boundary_own"] > 0.5 for p in pairs),
                "toward_food_in_smaller_component": sum(
                    bool(p[index]["toward_food"] and p[index]["food_in_smaller_component"])
                    for p in pairs
                ),
            }
        branch_counts[str(offset)] = counts
    return {
        "authority": "development diagnostic, not gate evidence",
        "n": len(rows),
        "offsets": by_offset,
        "ranking": ranking,
        "branch_counts": branch_counts,
        "spatial": {
            "criterion": "first D1..D100 with fatal head outside largest component and "
            "fatal head region strictly smaller than surviving head region",
            "resolved": len(resolved),
            "unresolved": len(rows) - len(resolved),
            "outside15": outside,
            "outside15_share_resolved": outside / len(resolved) if resolved else None,
            "outside15_share_all": outside / len(rows),
            "rows": spatial,
            "lad_boundary_outside15": sum(
                r["fatal"]["trace"][0]["boundary_max_chebyshev"] > 15 for r in rows
            ),
        },
        "verification": {
            "fatal_deaths_reproduced": sum(
                r["fatal"]["death_decision"] == r["fatal_t"] for r in rows
            ),
            "surviving_endpoints_reproduced": sum(r["surviving"]["endpoint_alive"] for r in rows),
            "exact_lad_q_and_geometry": sum(r["lad_q_and_action_geometry_exact"] for r in rows),
            "unmutated_clone_sources": sum(r["source_clone_unchanged"] for r in rows),
            "length_values_checked": sum(
                r["warmup_length_values_checked"] + r["fatal"]["length_values_checked"]
                for r in rows
            ),
        },
    }


def markdown_tables(summary: dict) -> str:
    """Render the complete requested offset table, including changing denominators."""
    lines = [
        "| Quantity | Offset | Paired n | Fatal − surviving median [Q1, Q3] | − / 0 / + |",
        "|---|---:|---:|---|---|",
    ]
    for metric in METRICS:
        for offset in OFFSETS:
            s = summary["offsets"][str(offset)][metric]
            q = s["quartiles"]
            value = "unavailable" if q is None else f"{q[1]:.4g} [{q[0]:.4g}, {q[2]:.4g}]"
            lines.append(
                f"| {metric} | {offset} | {s['n']} | {value} | "
                f"{s['negative']} / {s['zero']} / {s['positive']} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    rows = [json.loads(line) for line in (args.out / "rows.jsonl").read_text().splitlines()]
    receipt = json.loads((args.out / "completion.json").read_text())
    from research.b8_death_census_20261014.census import sha256_file

    assert receipt["status"] == "PASS" and receipt["full_cohort"] and len(rows) == 46
    assert receipt["rows_sha256"] == sha256_file(args.out / "rows.jsonl")
    result = summarize(rows)
    result["report_source_sha256"] = sha256_file(Path(__file__))
    result["rows_sha256"] = receipt["rows_sha256"]
    (args.out / "summary.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    (args.out / "tables.md").write_text(markdown_tables(result))
    print(
        json.dumps(
            {
                "ranking": result["ranking"],
                "spatial": result["spatial"],
                "verification": result["verification"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
