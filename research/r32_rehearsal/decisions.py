"""Independent synthetic decision cases for the R32 rehearsal.

The payload builder deliberately reuses the existing F4 analyzer fixture.  This
module only composes mutations; the expected labels live in ``ORACLE.json`` so
the reducer and its hand-calculated acceptance table remain separate.
"""

from __future__ import annotations

import copy
import importlib.util
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from research.bl_residual import analyze

_HELPER_PATH = Path(__file__).resolve().parents[2] / "tests/test_bl_residual_f4_analyzer.py"
_helper_spec = importlib.util.spec_from_file_location("r32_f4_analyzer_fixture", _HELPER_PATH)
if _helper_spec is None or _helper_spec.loader is None:
    raise ImportError(f"cannot load synthetic F4 fixture helper: {_HELPER_PATH}")
_fixture = importlib.util.module_from_spec(_helper_spec)
_helper_spec.loader.exec_module(_fixture)
_ORACLE = json.loads((Path(__file__).with_name("ORACLE.json")).read_text())


def _payload(**kwargs: Any) -> dict[str, Any]:
    return _fixture._payload(**kwargs)


def _reseal(value: dict[str, Any]) -> dict[str, Any]:
    return _fixture._reseal(value)


def _r9_invalid() -> dict[str, Any]:
    value = _payload()
    value["identity"]["sealed"] = False
    return value


def _r9_incomplete() -> dict[str, Any]:
    value = _payload()
    value.pop("gameplay")
    return value


def _r9_insufficient() -> dict[str, Any]:
    value = _payload()
    value["staging"]["D4096"]["selected_support"] = False
    return value


def _r9_kill_fit() -> dict[str, Any]:
    value = _payload()
    value["staging"]["D256"]["unconditional_fault"] = True
    return value


def _r9_stop_funding() -> dict[str, Any]:
    value = _payload()
    value["staging"]["funding"]["verdict"] = "STOP_FUNDING"
    return value


def _r9_stop_admission() -> dict[str, Any]:
    value = _payload()
    value["staging"]["admission"]["harm_screen_pass"] = False
    return value


def _r9_kill_dose_closed() -> dict[str, Any]:
    return _payload(top=60.0, dose=-1.0)


def _r9_kill_dose_futile() -> dict[str, Any]:
    return _payload(top=60.0, dose=10.0)


def _r9_go_top_dose_dev() -> dict[str, Any]:
    return _payload()


def _r9_inconclusive() -> dict[str, Any]:
    return _payload(top=30.0, dose=30.0)


def _equality_bars() -> dict[str, Any]:
    return _payload(top=40.0, dose=20.0)


def _not_run_by_rule() -> dict[str, Any]:
    value = _payload(top=30.0, dose=30.0)
    value.pop("watch6")
    return value


def _missing_required_watch() -> dict[str, Any]:
    value = _payload()
    value.pop("watch6")
    return value


def _top_passed_dose_failed() -> dict[str, Any]:
    value = _payload(top=60.0, dose=19.0)
    # Keep the unadjusted dose upper bound above +20 while the point bar fails.
    for row in value["gameplay"]["observations"][:2]:
        row["arms"]["D1024"]["mass_integral"] -= 150.0
    return _reseal(value)


def _mass_pass_survival_fail() -> dict[str, Any]:
    value = _payload()
    for row in value["gameplay"]["observations"]:
        row["arms"]["D4096"]["alive_frame_fraction"] = 0.80
    return _reseal(value)


def _mass_pass_watch_fail() -> dict[str, Any]:
    return _payload(product=False)


_BUILDERS: dict[str, Callable[[], dict[str, Any]]] = {
    "r9_invalid": _r9_invalid,
    "r9_incomplete": _r9_incomplete,
    "r9_insufficient": _r9_insufficient,
    "r9_kill_fit": _r9_kill_fit,
    "r9_stop_funding": _r9_stop_funding,
    "r9_stop_admission": _r9_stop_admission,
    "r9_kill_dose_closed": _r9_kill_dose_closed,
    "r9_kill_dose_futile": _r9_kill_dose_futile,
    "r9_go_top_dose_dev": _r9_go_top_dose_dev,
    "r9_inconclusive": _r9_inconclusive,
    "equality_bars": _equality_bars,
    "not_run_by_rule": _not_run_by_rule,
    "missing_required_watch": _missing_required_watch,
    "top_passed_dose_failed": _top_passed_dose_failed,
    "mass_pass_survival_fail": _mass_pass_survival_fail,
    "mass_pass_watch_fail": _mass_pass_watch_fail,
}


def case_names() -> list[str]:
    """Return the frozen case order used by the portable runtime."""
    return list(_BUILDERS)


def payload_case(name: str) -> dict[str, Any]:
    """Build one fresh synthetic analyzer input by frozen case name."""
    try:
        builder = _BUILDERS[name]
    except KeyError as exc:
        raise KeyError(f"unknown R32 synthetic case: {name}") from exc
    return copy.deepcopy(builder())


def run_case(name: str) -> dict[str, Any]:
    """Build and reduce one fresh case, returning an auditable synthetic result.

    The reducer's observed label and full decision are returned beside the
    hand-authored expected label. This lets a portable worker publish one small
    case result while the oracle remains independent of reducer execution.
    """
    payload = payload_case(name)
    actual = analyze.analyze(payload)
    expected = copy.deepcopy(_ORACLE["cases"][name])
    passed = actual.get("label") == expected.get("expected_label")
    if "expected_flag" in expected:
        passed = passed and actual.get("flag") == expected["expected_flag"]
    if "expected_watch6" in expected:
        passed = passed and actual.get("watch6") == expected["expected_watch6"]
    if "expected_reason" in expected:
        passed = passed and expected["expected_reason"] in actual.get("reasons", [])
    return {
        "schema": "r32-decision-case/v1",
        "synthetic": True,
        "scientific_result": False,
        "case": name,
        "observed_label": actual.get("label"),
        "expected_label": expected.get("expected_label"),
        "pass": bool(passed),
        "decision": actual,
    }
