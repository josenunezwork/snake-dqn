"""Print a compact table from a diagnostics JSON (pooled held-out folds)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for item in report.get("free_q", []):
        out.append(
            {
                "cell": f"freeQ:{item['loss']}",
                "updates": item["steps"],
                "passes": None,
                "flip": item["flip"],
                "flip_ww": item["flip_world_weighted"],
                "bare": item["argmax_bare"],
                "not_forbidden": item["argmax_not_forbidden"],
                "margin_med": item["margin_median"],
                "anchor_agree": None,
                "max_decile": None,
                "boost_ratio": None,
                "train_flip": None,
            }
        )
    for cell in report.get("cells", []):
        name = cell["cell"]["name"]
        for point in cell["pooled"]:
            dec = point["heldout_decisive"]
            u = point["updates"]
            train = [
                next(c for c in f["curve"] if c["updates"] == u)["train_decisive"]
                for f in cell["folds"]
            ]
            n = sum(t["n"] for t in train)
            out.append(
                {
                    "cell": name,
                    "updates": u,
                    "passes": point["decisive_passes_mean"],
                    "flip": dec["flip"],
                    "flip_ww": dec["flip_world_weighted"],
                    "bare": dec["argmax_bare"],
                    "not_forbidden": dec["argmax_not_forbidden"],
                    "margin_med": dec["margin_median"],
                    "anchor_agree": point["heldout_anchor_agree"],
                    "max_decile": point["heldout_anchor_max_decile_disagree"],
                    "boost_ratio": point["heldout_boost_ratio_max"],
                    "train_flip": sum(t["flip"] * t["n"] for t in train) / n,
                }
            )
    return out


def fmt(value: Any) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report", type=Path, nargs="+")
    args = parser.parse_args()
    keys = [
        "cell",
        "updates",
        "passes",
        "train_flip",
        "flip",
        "flip_ww",
        "bare",
        "not_forbidden",
        "margin_med",
        "anchor_agree",
        "max_decile",
        "boost_ratio",
    ]
    print("| " + " | ".join(keys) + " |")
    print("|" + "---|" * len(keys))
    for path in args.report:
        for row in rows(json.loads(path.read_text())):
            print("| " + " | ".join(fmt(row[k]) for k in keys) + " |")


if __name__ == "__main__":
    main()
