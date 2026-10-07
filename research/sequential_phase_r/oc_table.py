"""Render an OC result JSON (``simulate.py --out``) as Markdown tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

SHORT = {"fixed": "fixed", "sequential": "seq"}


def _diff(row: Dict[str, Any], label: str) -> str:
    if row["design"] == "fixed":
        return "-"
    d = row["vs_fixed"][label]
    return f"{d['diff']:+.3f} ({d['paired_se']:.3f})"


def table(result: Dict[str, Any], family: str) -> str:
    rows = [r for r in result["rows"] if "scenario" in r and r["scenario"]["family"] == family]
    labels = [k for k in rows[0] if isinstance(rows[0][k], dict) and "p" in rows[0][k]]
    labels = [k for k in labels if not k.startswith("efficacy")]
    go, kill = labels[0], [k for k in labels if k.startswith("KILL")][0]
    head = [
        "scenario",
        "tau2",
        "design",
        *labels,
        "eff. any look (non-binding)",
        "GO - fixed (SE)",
        "KILL - fixed (SE)",
        "E[compute]",
        "saving",
        "stops",
    ]
    out: List[str] = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        sc = r["scenario"]
        name = sc.get("label") or f"{sc['effect']:+g}"
        cells = [
            name,
            f"{sc['tau2']:g}",
            SHORT.get(r["design"], r["design"].replace("sequential", "seq")),
            *[f"{r[k]['p']:.3f}" for k in labels],
            f"{r['efficacy_any_look_nonbinding']['p']:.3f}",
            _diff(r, go),
            _diff(r, kill),
            f"{r['expected_fraction']['mean']:.3f}",
            f"{r['saving']:.0%}",
            "/".join(f"{x:.2f}" for x in r["stop_by_look"]),
        ]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("result", type=Path)
    args = parser.parse_args()
    result = json.loads(args.result.read_text())
    for family in ("v4", "v5"):
        print(f"\n### {family} family ({result['reps']} replicates per row)\n")
        print(table(result, family))


if __name__ == "__main__":
    main()
