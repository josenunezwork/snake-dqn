"""Final-only S1 transport receipt validation, without remote allocation."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from research.s1_mature_20261018 import final_artifacts, price
from tests.test_colab_cli_admission import passing_result as _passing_result_fixture


@pytest.fixture
def passing_result(tmp_path: Path) -> dict:
    return _passing_result_fixture.__wrapped__(tmp_path)


def test_final_artifact_transport_charges_enforced_cap_at_measured_rate(
    tmp_path: Path, passing_result: dict
) -> None:
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(passing_result))
    value = final_artifacts.acceptance_transport(path)
    result = final_artifacts.transport_bound(value)
    rate = passing_result["final_artifact_transfer"]["bytes_per_second_lower_bound"]
    for arm in ("n3", "n1"):
        assert result["qualified_arms"][arm]["transport_seconds_upper_bound"] == pytest.approx(
            64 * 1024**2 / rate
        )
    assert result["cli_acceptance_receipt"] == final_artifacts.binding(path)
    root = final_artifacts.ROOT / "research/s1_mature_20261018"
    forecast = price.forecast(
        root / "EVAL_PREFLIGHT_FINAL.json",
        root / "TRAIN_PREFLIGHT_DISPATCH03.json",
        acceptance_path=path,
    )
    assert forecast["status"] == "PASS"
    assert forecast["refusal_reasons"] == []
    assert forecast["complete_upper_bound"] == forecast["primary_hard_reservation"]
    assert forecast["complete_upper_bound"]["gpu_allocation_hours"] == pytest.approx(19 + 1 / 6)


@pytest.mark.parametrize(
    "fault",
    [
        "wrong-profile",
        "stale-source",
        "missing-gate",
        "unobserved-gate",
        "short-artifact",
        "missing-result",
        "changed-artifact",
        "nonfinite-rate",
        "not-qualified",
        "plan-source",
    ],
)
def test_s1_forecast_refuses_invalid_acceptance(
    tmp_path: Path, passing_result: dict, fault: str
) -> None:
    value = copy.deepcopy(passing_result)
    if fault == "wrong-profile":
        value["profile"] = "full-state"
    elif fault == "stale-source":
        value["source_manifest"]["research/colab_cli/supervisor.py"] = "0" * 64
    elif fault == "missing-gate":
        value["final_plan"]["gates"].pop()
    elif fault == "unobserved-gate":
        value["final_plan"]["gates"][0]["status"] = "INCOMPLETE"
    elif fault == "short-artifact":
        value["final_artifact_transfer"]["exported_bytes"] -= 1
    elif fault == "changed-artifact":
        Path(value["final_artifact_transfer"]["path"]).write_bytes(b"changed")
    elif fault == "nonfinite-rate":
        value["final_artifact_transfer"]["bytes_per_second_lower_bound"] = float("nan")
    elif fault == "not-qualified":
        value["qualified"] = False
    elif fault == "plan-source":
        value["final_plan"]["source_manifest"] = {}
    path = tmp_path / "acceptance.json"
    if fault != "missing-result":
        path.write_text(json.dumps(value))
    root = final_artifacts.ROOT / "research/s1_mature_20261018"
    forecast = price.forecast(
        root / "EVAL_PREFLIGHT_FINAL.json",
        root / "TRAIN_PREFLIGHT_DISPATCH03.json",
        acceptance_path=path,
    )
    assert forecast["status"] == "REFUSED"
    assert forecast["complete_upper_bound"] is None


def test_s1_acceptance_requires_absolute_path() -> None:
    with pytest.raises(ValueError, match="absolute"):
        final_artifacts.acceptance_result(Path("acceptance.json"))


def test_s1_forecast_refuses_usable_cap_overrun(
    tmp_path: Path, passing_result: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(passing_result))
    monkeypatch.setitem(price.ledger.USABLE_CAPS, "gpu_allocation_hours", 19)
    root = final_artifacts.ROOT / "research/s1_mature_20261018"
    forecast = price.forecast(
        root / "EVAL_PREFLIGHT_FINAL.json",
        root / "TRAIN_PREFLIGHT_DISPATCH03.json",
        acceptance_path=path,
    )
    assert forecast["status"] == "REFUSED"
    assert forecast["complete_upper_bound"] is None
    assert any("exceeds usable gpu_allocation_hours" in r for r in forecast["refusal_reasons"])
