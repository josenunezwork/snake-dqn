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
from research.colab_cli.accounting import budget_warning, warning_threshold

BASE_UNIT_CAP = 50.0
EFFECTIVE_UNIT_CAP = 100.0
GUARD_DESCRIPTION = (
    "Legacy measurement thresholds only: base50/effective100 units, reserve0/0; "
    "100 units = 64.935 hours at observed L4 1.54 u/h (50 hours at priced 2 u/h). "
    "Existing wall timeout, science caps and forecasts unchanged. "
    "Unit caps and reserves are warning thresholds only (owner 2026-10-29)."
)


def repair_guard(manifest: dict[str, Any]) -> dict[str, Any]:
    """Copy an F5 template and label its legacy unit thresholds informational."""
    if manifest.get("script") != "research/bl_residual/f5_colab.sh":
        raise ValueError("only F5 manifests are eligible")
    f5 = manifest.get("f5", {})
    guards = f5.get("technical_guards", {})
    if guards.get("units_are_scientific_budget_gate") is not False:
        raise ValueError("explicit non-scientific technical guard required")
    budget = manifest.get("budget", {})
    budget_warning(
        "F5_UNIT_THRESHOLDS_INFORMATIONAL",
        unit_cap=budget.get("unit_cap"),
        reserve=budget.get("reserve_units"),
    )
    result = copy.deepcopy(manifest)
    result.setdefault("budget", {})["unit_cap"] = EFFECTIVE_UNIT_CAP
    result["f5"]["technical_guards"].update(
        base_unit_cap=BASE_UNIT_CAP, effective_unit_cap=EFFECTIVE_UNIT_CAP
    )
    result["f5"]["technical_budget"] = GUARD_DESCRIPTION
    return result


def require_admissible(balance: float, committed: float, reserve: float = 0.0) -> None:
    """Warn about historical thresholds; only depleted balance prevents admission."""
    if type(balance) not in (int, float) or not math.isfinite(balance):
        raise ValueError("finite numeric account evidence required")
    if balance <= 0:
        raise ValueError("account balance is depleted")
    committed = warning_threshold(committed, "committed", 0)
    reserve = warning_threshold(reserve, "reserve", 0)
    if balance <= EFFECTIVE_UNIT_CAP + committed + reserve:
        budget_warning(
            "BALANCE_BELOW_LEGACY_RESERVATIONS",
            balance=balance,
            unit_cap=EFFECTIVE_UNIT_CAP,
            committed=committed,
            reserve=reserve,
        )


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
