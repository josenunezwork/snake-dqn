"""Fixed T34 development analysis.  It fails closed on incomplete banks."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Iterable

from src.scripts.eval_stats import student_t_isf


def _ci(values: Iterable[float], alpha: float, side: str) -> dict[str, Any]:
    xs = [float(x) for x in values]
    if len(xs) < 2:
        return {"status": "NOT_ESTIMABLE", "n": len(xs)}
    mean = sum(xs) / len(xs)
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (len(xs) - 1))
    margin = student_t_isf(alpha, len(xs) - 1) * sd / math.sqrt(len(xs))
    return {"status": "OK", "n": len(xs), "mean": mean,
            "lower": mean - margin, "upper": mean + margin}


def _metric(records: list[dict[str, Any]], arm: str, name: str) -> list[float]:
    values = []
    for record in records:
        data = record.get("arms", {})
        if "bare" not in data or arm not in data:
            continue
        values.append(float(data[arm].get(name, 0.0)) - float(data["bare"].get(name, 0.0)))
    return values


def analyze(root: Path) -> dict[str, Any]:
    """Analyze only a complete registered bank; never infer missing clauses."""
    root = Path(root)
    intent = json.loads((root / "manifest.json").read_text())
    paths = sorted((root / "records").glob("*.json"))
    records = [json.loads(path.read_text()) for path in paths]
    expected = int(intent["expected_records"])
    complete = len(records) == expected and all(r.get("status") == "complete" for r in records)
    report: dict[str, Any] = {"schema": "t34-pathb-analysis/v1", "complete": complete,
                              "expected_records": expected, "records": len(records),
                              "development_only": True,
                              "client_latency": "NOT_MEASURED"}
    if not complete:
        report.update(decision="INCOMPLETE", clauses={"all": "NOT_ESTIMABLE"})
        return report
    terminal = [r for r in records if r.get("kind") == "terminal"]
    product = [r for r in records if r.get("kind") == "product"]
    report["terminal"] = {"async_mass_gain": _ci(_metric(terminal, "async", "mass_integral"), .10, "lower"),
                          "sync_mass_gain_descriptive": _ci(_metric(terminal, "sync", "mass_integral"), .10, "lower"),
                          "async_no_death_gain": _ci(_metric(terminal, "async", "no_death"), .10, "lower")}
    report["product"] = {"death_rate_delta": _ci(_metric(product, "async", "deaths_per_1k"), .025, "upper"),
                         "relative_mass_delta": _ci(_metric(product, "async", "relative_mass"), .025, "lower")}
    timing = [r.get("timing") for r in records if r.get("timing")]
    report["timing"] = timing or "NOT_MEASURED"
    # Passing is intentionally impossible without every registered terminal/product/timing clause.
    report["decision"] = "DEVELOPMENT_REPORT_ONLY"
    return report


def main() -> int:
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    report = analyze(args.out)
    target = args.out / "analysis.json"
    if target.exists():
        raise SystemExit("analysis.json already exists (create-only)")
    target.write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    print(json.dumps({"analysis": str(target), "decision": report["decision"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
