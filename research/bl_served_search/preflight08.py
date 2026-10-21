"""Independently reconstruct fixture mass and compare one/two-worker outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import digest, export_json, verify_export


def without_timings(value: Any) -> Any:
    """Remove measured wall times only; keep every semantic field."""
    if isinstance(value, dict):
        return {
            key: without_timings(item)
            for key, item in value.items()
            if not key.endswith(("_seconds", "_minutes"))
        }
    if isinstance(value, list):
        return [without_timings(item) for item in value]
    return value


def audit(serial: Path, parallel: Path, out: Path) -> dict:
    """Require identical semantic rows and reconstruct each mass component."""
    receipts = []
    for path in (serial, parallel):
        verify_export(path)
        value = json.loads(path.read_text())
        if value.get("schema") != "bl-served-search-l2-fixture-shard/v1":
            raise ValueError("only excluded fixture outputs may enter Mac preflight")
        if value.get("status") != "completed" or not value.get("clean_exit"):
            raise ValueError("fixture did not complete cleanly")
        if not 1 <= value["identity"]["workers"] <= 2:
            raise ValueError("local preflight exceeds two workers")
        for row in value["rows"]:
            horizon = row["horizon"]
            traces = row["trajectories"]
            base, dense = traces["baseline_mass"], traces["dense_mass"]
            if len(base) != horizon or len(dense) != horizon:
                raise ValueError("trajectory horizon mismatch")
            for arm, trace in (("baseline", base), ("dense", dense)):
                if any(type(mass) is not int or mass < 0 for mass in trace):
                    raise ValueError("invalid logical mass")
                if not math.isclose(
                    sum(trace) / horizon, row[arm]["mass_integral"], abs_tol=1e-12, rel_tol=0
                ):
                    raise ValueError("individual arm mass reconstruction failed")
            common = sum(d - b for b, d in zip(base, dense) if b and d) / horizon
            dense_only = sum(d for b, d in zip(base, dense) if d and not b) / horizon
            base_only = -sum(b for b, d in zip(base, dense) if b and not d) / horizon
            for name, expected in (
                ("common_alive_productivity_delta", common),
                ("dense_only_survival_mass", dense_only),
                ("baseline_only_survival_mass", base_only),
                ("total", common + dense_only + base_only),
            ):
                if not math.isclose(row["mass_ledger"][name], expected, abs_tol=1e-12, rel_tol=0):
                    raise ValueError(f"mass ledger reconstruction failed: {name}")
        receipts.append(value)
    rows = [without_timings(value["rows"]) for value in receipts]
    if rows[0] != rows[1]:
        raise ValueError("serial/two-worker semantic rows differ")
    semantic = json.dumps(rows[0], sort_keys=True, separators=(",", ":"), allow_nan=False)
    result = {
        "schema": "bl-served-search-amendment08-preflight/v1",
        "status": "PASS",
        "inputs": {str(path): digest(path) for path in (serial, parallel)},
        "semantic_rows_sha256": hashlib.sha256(semantic.encode()).hexdigest(),
        "fixture_worlds": len(rows[0]),
        "selection_worlds": 0,
        "individual_mass_and_components_reconstructed": True,
        "wall_seconds": [value["wall_seconds"] for value in receipts],
        "workers": [value["identity"]["workers"] for value in receipts],
        "colab_calls": 0,
        "qualifies_l4_or_mac_latency": False,
    }
    export_json(out, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--serial", type=Path, required=True)
    parser.add_argument("--parallel", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.serial, args.parallel, args.out)))


if __name__ == "__main__":
    main()
