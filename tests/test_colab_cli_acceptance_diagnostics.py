"""Offline diagnostics regressions for F4 synthetic staging failures."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from research.colab_cli import acceptance, supervisor


class _Observer:
    def __init__(self, _oauth_python: str, profile: str = "final-only") -> None:
        self.records: dict[str, dict[str, object]] = {}

    def _record(self, name: str, **evidence: object) -> None:
        self.records[name] = dict(evidence)

    def write(self, _output: Path) -> None:
        return None


def _clean_receipt() -> dict[str, object]:
    return {
        "status": "PASS",
        "stop_verified": True,
        "cleanup_error": None,
        "accounting": {
            "calibration": {
                "status": "PASS",
                "priced_rate_units_per_hour": 2.0,
                "observed_rate_units_per_hour": 1.54,
                "samples": [1.54, 1.54, 1.54],
            }
        },
    }


@pytest.mark.parametrize("watchdog_refusal", [False, True])
def test_staging_bar_refusal_preserves_cleanup_and_calibration_diagnostics(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, watchdog_refusal: bool
) -> None:
    """The ruling's 207-second replay stays staging failure in both broker paths."""
    run_dir = tmp_path / "acceptance"
    manifests = run_dir / "manifests"
    manifests.mkdir(parents=True)
    jobs = []
    for name, gpu in (("cpu", "cpu"), ("l4", "L4")):
        output = run_dir / "outputs" / name
        manifest = manifests / f"{name}.json"
        manifest.write_text(json.dumps({"output_dir": str(output), "requirements": {"gpu": gpu}}))
        jobs.append({"name": name, "manifest": str(manifest)})
    plan: dict[str, object] = {
        "profile": "final-only",
        "jobs": jobs,
        "d1024_synthetic_inventory": {"sha256": "a" * 64},
        "gates": [
            {"id": "rate-calibration", "required": True, "status": "INCOMPLETE"},
            {
                "id": "d1024-synthetic-full-inventory-staging",
                "required": True,
                "status": "INCOMPLETE",
            },
        ],
        "source_manifest": {"frozen": "source"},
        "commit": "frozen",
        "tree_hash": "frozen-tree",
    }

    class _WatchdogBroker:
        def __init__(self, _state: Path, _binary: str, observer: _Observer) -> None:
            self.observer = observer

        def run(self, payload: dict[str, object]) -> None:
            output = Path(str(payload["output_dir"]))
            output.mkdir(parents=True)
            (output / "receipt.json").write_text(json.dumps(_clean_receipt()))
            if payload["requirements"] == {"gpu": "L4"}:
                (output / "upload-measurement.json").write_text(
                    json.dumps(
                        {
                            "elapsed_seconds": 220.22179129195865,
                            "setup_seconds": 1.3687461250228807,
                            "source_delivery_seconds": 11.504017250030302,
                            "transfer_seconds": 207.348055665032,
                        }
                    )
                )
            if payload["requirements"] == {"gpu": "L4"} and watchdog_refusal:
                self.observer.records["mac-loss-watchdog-unassign"] = {
                    "assignment_disappeared_within_bound": True
                }
                raise supervisor.Refused("expected watchdog renewal refusal")

    def _failed_staging(
        _plan: dict[str, object],
        _job: dict[str, object],
        _receipt: dict[str, object],
        output: Path,
    ) -> dict[str, object]:
        acceptance._d1024_staging_measurement(output / "upload-measurement.json")
        raise AssertionError("the failed ruling measurement unexpectedly passed")

    monkeypatch.setattr(acceptance, "AcceptanceObserver", _Observer)
    monkeypatch.setattr(acceptance.supervisor, "Broker", _WatchdogBroker)
    monkeypatch.setattr(acceptance, "verify_plan_sources", lambda *_args: None)
    monkeypatch.setattr(acceptance, "source_manifest", lambda _repo: {"frozen": "source"})
    monkeypatch.setattr(acceptance, "d1024_synthetic_staging_evidence", _failed_staging)

    rc = acceptance.execute(
        plan,
        state_dir=tmp_path / "state",
        colab="fake-colab",
        review_token=acceptance.REVIEW_TOKEN,
        oauth_python="fake-python",
    )

    result = json.loads((run_dir / "acceptance-result.json").read_text())
    l4 = result["final_plan"]["execution"][1]
    assert rc == 1
    assert l4["status"] == "INCOMPLETE"
    assert l4["rate_calibration"]["status"] == "PASS"
    assert l4["measured_rate_units_per_hour"] == 1.54
    assert l4["staging_error"] == {
        "status": "INCOMPLETE",
        "error": "synthetic D1024 transfer exceeds the 150-second bar",
        "exception_type": "Refused",
    }
    assert ("error" in l4) is watchdog_refusal
    assert result["final_plan"]["gates"][0]["status"] == "OBSERVED"
    staging_gate = result["final_plan"]["gates"][1]
    assert staging_gate["status"] == "INCOMPLETE"
    assert staging_gate["evidence_record"] == l4["staging_error"]
