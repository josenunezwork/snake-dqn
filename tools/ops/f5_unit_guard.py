"""Prospective, offline repair of F5's inadmissible technical unit placeholder.

Never overwrite a sealed manifest. Apply this to a fresh successor after any
historical F5 builder, then bind fresh dispatch identities, account evidence and
publication. This is a runaway guard, not scientific or dispatch authority.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_json

BASE_UNIT_CAP = 50.0
EFFECTIVE_UNIT_CAP = 100.0
GUARD_DESCRIPTION = (
    "Technical runaway guard only: base50/effective100 units, reserve0/0; "
    "100 units = 64.935 hours at observed L4 1.54 u/h (50 hours at priced 2 u/h). "
    "Existing wall timeout, science caps and forecasts unchanged. "
    "Fresh balance must exceed cap plus committed reservations plus reserve."
)


def repair_guard(manifest: dict[str, Any]) -> dict[str, Any]:
    """Copy an F5 template and replace only its 9999-unit technical guard."""
    if manifest.get("script") != "research/bl_residual/f5_colab.sh":
        raise ValueError("only F5 manifests are eligible")
    f5 = manifest.get("f5", {})
    guards = f5.get("technical_guards", {})
    if guards.get("units_are_scientific_budget_gate") is not False:
        raise ValueError("explicit non-scientific technical guard required")
    budget = manifest.get("budget", {})
    if budget.get("unit_cap") != 9999.0 or guards.get("effective_unit_cap") != 9999.0:
        raise ValueError("only the historical 9999-unit placeholder is eligible")
    if guards.get("base_unit_cap") != 4999.5:
        raise ValueError("historical base/effective 2x guard required")
    if any(
        value != 0
        for value in (
            budget.get("reserve_units"),
            guards.get("base_reserve_units"),
            guards.get("effective_reserve_units"),
        )
    ):
        raise ValueError("reserve changes require a separate successor")
    result = copy.deepcopy(manifest)
    result["budget"]["unit_cap"] = EFFECTIVE_UNIT_CAP
    result["f5"]["technical_guards"].update(
        base_unit_cap=BASE_UNIT_CAP, effective_unit_cap=EFFECTIVE_UNIT_CAP
    )
    result["f5"]["technical_budget"] = GUARD_DESCRIPTION
    return result


def require_admissible(balance: float, committed: float, reserve: float = 0.0) -> None:
    """Mirror supervisor's strict balance inequality without provider calls."""
    values = (balance, committed, reserve)
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in values):
        raise ValueError("finite numeric account evidence required")
    if any(value < 0 for value in values) or balance <= EFFECTIVE_UNIT_CAP + committed + reserve:
        raise ValueError("insufficient balance for cap plus study reserve")


def main() -> None:
    """Write a create-only template successor; never allocate or dispatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--balance", type=float, required=True)
    parser.add_argument("--committed", type=float, required=True)
    args = parser.parse_args()
    require_admissible(args.balance, args.committed)
    export_json(args.output, repair_guard(json.loads(args.manifest.read_text())))


if __name__ == "__main__":
    main()
