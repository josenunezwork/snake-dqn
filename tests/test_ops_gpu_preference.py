"""H100 preference admission and provider-free dispatch boundary proofs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from research.compute.gpu_preference import prepare_preference, sibling_id, variants
from tests import test_ops_dispatch as shared

fixture = shared.fixture
wait_for = shared.wait_for


def preferred(f: dict, mode: str = "gpu_success") -> None:
    packet = json.loads(f["manifest"].read_text())
    packet.update(
        test_mode=mode,
        timeout_minutes=10,
        requirements={
            "gpu": "H100",
            "allowed_gpus": ["H100"],
            "prefer": ["H100", "A100"],
            "est_minutes": 5,
            "min_vcpu": 2,
        },
        budget={
            "rate_units_per_hour": 12.0,
            "unit_cap": 2.0,
            "reserve_units": 0,
            "other_assignments": {},
        },
    )
    f["manifest"].write_text(json.dumps(packet))


def trace(f: dict, action: str) -> list[dict]:
    path = Path(f["env"]["TRACE"])
    return [
        row
        for line in path.read_text().splitlines()
        if (row := json.loads(line))["action"] == action
    ]


def test_h100_healthy_path_seals_both_before_submit(fixture):
    f = fixture
    preferred(f)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "requested_gpu=H100" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job"]
    all_rows = [json.loads(line) for line in Path(f["env"]["TRACE"]).read_text().splitlines()]
    first_run = next(i for i, row in enumerate(all_rows) if row["action"] == "run")
    assert {row["job"] for row in all_rows[:first_run]} == {"job", "job-a100-fallback"}
    admission = json.loads((f["ops"] / "dispatch/lane/gpu-preference.json").read_text())
    assert [row["gpu"] for row in admission["variants"]] == ["H100", "A100"]
    for row in admission["variants"]:
        assert hashlib.sha256(Path(row["manifest"]).read_bytes()).hexdigest() == row["sha256"]
    receipt_path = f["ops"] / "dispatch/lane/gpu-receipt.json"
    wait_for(receipt_path.exists)
    receipt = json.loads(receipt_path.read_text())
    assert receipt["requested_gpu"] == "H100"
    assert receipt["observed_hardware"]["hardware"]["kind"] == "H100"
    assert not (f["tmp"] / "output-a100-fallback").exists()


def test_allocation_unavailable_reconciles_then_uses_a100(fixture):
    f = fixture
    preferred(f, "gpu_unavailable")
    f["env"]["EXIT_DELAY"] = ".15"
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "requested_gpu=A100" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job", "job-a100-fallback"]
    before = json.loads((f["tmp"] / "output/receipt.json").read_text())
    assert before["stop_verified"] is False  # No invented stop of an unassigned endpoint.
    proof = json.loads((f["ops"] / "dispatch/lane/unavailable.json").read_text())
    assert proof["classification"] == "ALLOCATION_UNAVAILABLE"
    assert proof["endpoint_ever_assigned"] is False
    assert not Path(proof["pending"]["path"]).exists()
    fallback = json.loads((f["ops"] / "manifests/lane-a100-fallback.live.json").read_text())
    assert fallback["requirements"]["allowed_gpus"] == ["A100"]
    assert fallback["budget"]["rate_units_per_hour"] == 8.0
    receipt_path = f["ops"] / "dispatch/lane-a100-fallback/gpu-receipt.json"
    wait_for(receipt_path.exists)
    assert json.loads(receipt_path.read_text())["observed_hardware"]["hardware"]["kind"] == "A100"


@pytest.mark.parametrize("mode", ["gpu_endpoint_failure", "gpu_timeout", "gpu_entitlement"])
def test_failure_or_uncertainty_never_falls_back(fixture, mode):
    f = fixture
    preferred(f, mode)
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert len(trace(f, "run")) == 1
    assert not (f["ops"] / "dispatch/lane/unavailable.json").exists()


def test_recovered_endpoint_keeps_pending_and_refuses_fallback(fixture):
    f = fixture
    preferred(f, "gpu_unavailable")
    f["env"]["RECOVER_ENDPOINT"] = "real-endpoint"
    result = f["invoke"]()
    assert result.returncode != 0
    assert "endpoint recovered" in result.stderr
    assert len(trace(f, "run")) == 1
    assert list((Path(f["env"]["HOME"]) / ".config/colab-cli/snake").glob("pending-*.json"))


def test_both_unavailable_never_attempts_a_third_allocation(fixture):
    f = fixture
    preferred(f, "gpu_both_unavailable")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job", "job-a100-fallback"]


def test_post_endpoint_accounting_failure_never_uses_legacy_retry(fixture):
    f = fixture
    preferred(f, "retry_success")
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert "allocation=failed" in result.stdout
    assert [row["job"] for row in trace(f, "run")] == ["job"]


def test_fallback_existing_output_refuses_primary(fixture):
    f = fixture
    preferred(f)
    (f["tmp"] / "output-a100-fallback").mkdir()
    result = f["invoke"]()
    assert result.returncode != 0
    assert "identity already used" in result.stderr
    assert not Path(f["env"]["TRACE"]).exists()


@pytest.mark.parametrize(
    "field,value", [("unit_cap", 1.99), ("rate_units_per_hour", 7.5), ("unit_cap", True)]
)
def test_pricing_refuses_before_reservation(fixture, field, value):
    f = fixture
    preferred(f)
    packet = json.loads(f["manifest"].read_text())
    packet["budget"][field] = value
    f["manifest"].write_text(json.dumps(packet))
    result = f["invoke"]()
    assert result.returncode != 0
    assert not (f["ops"] / "dispatch").exists()


@pytest.mark.parametrize(
    "relative",
    [
        "logs/lane-a100-fallback.log",
        "dispatch/lane-a100-fallback",
        "manifests/lane-a100-fallback.json",
    ],
)
def test_fallback_identity_reuse_refuses_primary(fixture, relative):
    f = fixture
    preferred(f)
    path = f["ops"] / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    if path.suffix == ".log":
        path.write_text("EXIT=0\n")
    result = f["invoke"]()
    assert result.returncode != 0
    assert "identity already used" in result.stderr
    assert not Path(f["env"]["TRACE"]).exists()


def test_preference_dry_run_validates_both_without_reservation(fixture):
    f = fixture
    preferred(f)
    result = f["invoke"](dry_run=True)
    assert result.returncode == 0, result.stderr
    assert {row["job"] for row in trace(f, "validate")} == {"job", "job-a100-fallback"}
    assert not trace(f, "run")
    assert not (f["ops"] / "dispatch").exists()


def test_live_refresh_admits_calibrated_h100(fixture):
    f = fixture
    f["inventory"](
        {"h": "H100"},
        [
            {
                "session": "other",
                "endpoint": "h",
                "priced_rate_units_per_hour": 12,
                "observed_rate_units_per_hour": 10.3,
            }
        ],
    )
    result = f["invoke"]()
    assert result.returncode == 0, result.stderr
    assert trace(f, "run")[0]["other"] == {"h": 10.3}


def test_preparation_changes_only_new_operational_envelope():
    packet = {
        "job_id": "x" * 64,
        "output_dir": "/tmp/never-dispatch/",
        "timeout_minutes": 20,
        "requirements": {"gpu": "A100", "est_minutes": 10},
        "budget": {"rate_units_per_hour": 8, "unit_cap": 3, "reserve_units": 2},
        "env": {"SEED": "unchanged"},
        "inputs": [],
    }
    prepare_preference(packet)
    h100, a100 = variants(packet)
    assert len(a100["job_id"]) == 64 and a100["job_id"] != h100["job_id"]
    assert sibling_id(packet["job_id"]) == a100["job_id"]
    assert a100["env"] == h100["env"] == {"SEED": "unchanged"}
    assert a100["output_dir"] == "/tmp/never-dispatch-a100-fallback"
    assert h100["budget"]["unit_cap"] == 4
    assert h100["technical_envelope"]["base_unit_cap"] == 2
