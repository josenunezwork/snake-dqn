"""Example rule specs: the FRP-v4 and FRP-v5-H/S2 Phase R rules in sequential form.

These are templates for a FUTURE study's pre-registration (and the simulator's two families);
they do not change FRP-v4, v5-H or v5-S2, whose fixed-N packages stay authoritative.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

from src.evaluation.sequential_phase_r import RULE_SCHEMA

MIXES = ("frozen", "mixed", "scripted")
SURVIVAL_TOLERANCE = 1e-9


def _point(
    name: str, stat: dict, op: str, threshold: float, tol: float = 0.0, protective: bool = False
) -> dict:
    out = {
        "name": name,
        "kind": "point",
        "stat": stat,
        "op": op,
        "threshold": threshold,
        "tolerance": tol,
    }
    if protective:
        out["protective"] = True
    return out


def _hk(cell: str, metric: str) -> dict:
    return {"kind": "hk", "cell": cell, "metric": metric}


def _mix(cell: str, metric: str, mix: str) -> dict:
    return {"kind": "mix_mean", "cell": cell, "metric": metric, "mix": mix}


def _common_clauses(primary: str, guard: str, min_positive: int, cap: int) -> list:
    return [
        _point("c1_point", _hk(primary, "mi5"), ">=", 20.0),
        _point("c2_scripted_mean", _mix(primary, "mi5", "scripted"), ">=", 0.0, protective=True),
        _point(
            "c2_scripted_survival",
            _mix(primary, "surv5", "scripted"),
            ">=",
            -0.03,
            SURVIVAL_TOLERANCE,
            protective=True,
        ),
        _point(
            "c3_frozen_survival",
            _mix(primary, "surv5", "frozen"),
            ">=",
            -0.03,
            SURVIVAL_TOLERANCE,
            protective=True,
        ),
        _point(
            "c3_mixed_survival",
            _mix(primary, "surv5", "mixed"),
            ">=",
            -0.03,
            SURVIVAL_TOLERANCE,
            protective=True,
        ),
        {
            "name": "c4_positive_seeds",
            "kind": "positive_seeds",
            "cell": primary,
            "metric": "mi5",
            "min": min_positive,
        },
        {"name": "c5_prefix_controls", "kind": "flag", "flag": "prefix_controls"},
        _point("c5_mi10", _hk(primary, "mi10"), ">=", 20.0),
        _point(
            "c6_guard_point",
            {"kind": "stratified", "cell": guard, "metric": "mi5"},
            ">",
            0.0,
            protective=True,
        ),
        _point(
            "c6_guard_scripted_survival",
            _mix(guard, "surv5", "scripted"),
            ">=",
            -0.05,
            SURVIVAL_TOLERANCE,
            protective=True,
        ),
        {
            "name": "c7_feasibility",
            "kind": "feasibility",
            "cap": cap,
            "inflation": 1.02,
            "shrink": 0.6,
            "per_mix_alpha": 0.05 / 3,
            "power": 0.90,
        },
    ]


def frp_v4_like(
    seeds: Sequence[int] = (20, 21, 22, 23, 24),
    hero: str = "R4@60000",
    cap: int = 600,
    no_go: Optional[float] = None,
    study_id: str = "example-frp-v4-like",
) -> dict[str, Any]:
    """FRP-v4's rule: GO_R (1)-(7), GO_R_UNGATEABLE ((1)-(6), not (7)), KILL_V4 (HK UB < 20)."""
    seeds = [int(s) for s in seeds]
    clauses = _common_clauses("primary", "guard", 4 if len(seeds) == 5 else 6, cap)
    rule = {
        "schema": RULE_SCHEMA,
        "study_id": study_id,
        "mixes": list(MIXES),
        "seeds": seeds,
        "cells": {
            "primary": {"hero": hero, "baseline": "incumbent", "seeds": seeds},
            "guard": {"hero": hero, "baseline": "champion", "seeds": [seeds[0]]},
        },
        "statuses": {"go": "GO_R", "kill": "KILL_V4", "partial": "PARTIAL"},
        "efficacy": {"stat": _hk("primary", "mi5"), "lower_bound_above": 0.0},
        "go_clauses": clauses,
        "kill": {
            "all_of": [{"name": "kill_h5000", "stat": _hk("primary", "mi5"), "upper_below": 20.0}]
        },
        "final_outcomes": [
            {
                "status": "GO_R_UNGATEABLE",
                "all_of": ["c1_point"] + [c["name"] for c in clauses[1:-1]],
                "none_of": ["c7_feasibility"],
                "extra": [],
                "requires_efficacy": True,
            }
        ],
        "candidate": {"cell": "primary", "metric": "mi5", "confidence": 0.80},
        "no_go": None,
    }
    if no_go is not None:
        rule["no_go"] = {
            "status": "NO_GO_EARLY",
            "all_of": [
                {"name": "no_go_h5000", "stat": _hk("primary", "mi5"), "upper_below": no_go}
            ],
        }
    return rule


def frp_v5_like(
    seeds: Sequence[int] = (30, 31, 32, 33, 34, 35, 36, 37),
    hero: str = "H5@60000",
    control: str = "C@60000",
    cap: int = 600,
    no_go: Optional[float] = None,
    study_id: str = "example-frp-v5-like",
) -> dict[str, Any]:
    """FRP-v5-H/S2's rule: GO (1)-(8), GO_RECIPE, KILL on the H - C Welch contrast."""
    seeds = [int(s) for s in seeds]
    clauses = _common_clauses("primary", "guard", 6 if len(seeds) == 8 else 4, cap)
    welch5 = {"kind": "welch", "cells": ["primary", "control"], "metric": "mi5"}
    welch10 = {"kind": "welch", "cells": ["primary", "control"], "metric": "mi10"}
    clauses += [
        _point("c8_attribution_h5000", welch5, ">=", 0.0),
        _point("c8_attribution_mi10", welch10, ">", 0.0),
    ]
    rule = {
        "schema": RULE_SCHEMA,
        "study_id": study_id,
        "mixes": list(MIXES),
        "seeds": seeds,
        "cells": {
            "primary": {"hero": hero, "baseline": "incumbent", "seeds": seeds},
            "control": {"hero": control, "baseline": "incumbent", "seeds": seeds},
            "guard": {"hero": hero, "baseline": "champion", "seeds": [seeds[0]]},
        },
        "statuses": {"go": "GO_H", "kill": "KILL_H", "partial": "PARTIAL"},
        "efficacy": {"stat": _hk("primary", "mi5"), "lower_bound_above": 0.0},
        "go_clauses": clauses,
        "kill": {
            "all_of": [
                {"name": "kill_contrast_h5000", "stat": welch5, "upper_below": 25.0},
                {"name": "kill_contrast_mi10", "stat": welch10, "upper_below": 40.0},
            ]
        },
        "final_outcomes": [
            {
                "status": "GO_H_RECIPE",
                "all_of": [
                    "c2_scripted_mean",
                    "c2_scripted_survival",
                    "c3_frozen_survival",
                    "c3_mixed_survival",
                    "c5_prefix_controls",
                    "c6_guard_point",
                    "c6_guard_scripted_survival",
                ],
                "none_of": [],
                "extra": [
                    _point("recipe_mi10", welch10, ">=", 20.0),
                    _point("recipe_h5000", welch5, ">=", 0.0),
                ],
                "requires_efficacy": False,
            }
        ],
        "candidate": {"cell": "primary", "metric": "mi5", "confidence": 0.80},
        "no_go": None,
    }
    if no_go is not None:
        rule["no_go"] = {
            "status": "NO_GO_EARLY",
            "all_of": [
                {"name": "no_go_h5000", "stat": _hk("primary", "mi5"), "upper_below": no_go}
            ],
        }
    return rule
