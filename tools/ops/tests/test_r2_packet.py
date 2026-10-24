"""R2 canary fresh-attempt manifest and pre-work failure accounting."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tools.ops import r2_packet

ART = r2_packet.ARTIFACT
FIRST = {
    "job_id": "gpu-r2-canary-e66a701fd439-01",
    "commit": "e" * 40,
    "output_dir": "/x/packet/canary-output",
    "budget": {"rate_units_per_hour": 10.0, "unit_cap": 2.0, "reserve_units": 1 / 3},
    "timeout_minutes": 12,
    "requirements": {"gpu": "A100", "est_minutes": 6},
    "env": {"R2_BENCH_SECONDS": "420", "R2_BENCH_CANARY": "1"},
    "inputs": [{"dest": "metadata.tar", "sha256": "a" * 64}],
    "native_cuda_compile": {"receipt": {"path": "/r.json", "sha256": "b" * 64}},
}


def _changed(left: dict, right: dict, prefix: str = "") -> set[str]:
    keys = set(left) | set(right)
    out = set()
    for key in keys:
        a, b = left.get(key), right.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            out |= _changed(a, b, prefix + key + ".")
        elif a != b:
            out.add(prefix + key)
    return out


@pytest.mark.parametrize("prior", (0.0, 0.0568, 0.25, 0.3))
def test_attempt_two_stays_inside_the_two_unit_canary_allowance(prior):
    job = r2_packet.attempt_manifest(FIRST, prior, Path("/x/packet/canary-output-02"))
    cap, timeout = job["budget"]["unit_cap"], job["timeout_minutes"]
    rate = job["budget"]["rate_units_per_hour"]
    assert prior + cap <= r2_packet.CANARY_CAP_UNITS + 1e-12
    assert timeout * rate / 60 <= cap + 1e-12  # qualified supervisor pricing rule
    assert timeout >= 2 * job["requirements"]["est_minutes"]  # qualified 2x rule
    assert timeout >= 420 / 60 + 3  # unchanged producer deadline still fits
    assert job["job_id"] == "gpu-r2-canary-e66a701fd439-02"
    assert set(job["canary_attempt"]["changed_fields"]) >= _changed(FIRST, job) - {"canary_attempt"}
    assert job["env"] == FIRST["env"] and job["inputs"] == FIRST["inputs"]
    assert job["native_cuda_compile"] == FIRST["native_cuda_compile"]


def test_attempt_two_for_the_observed_failure_charge():
    job = r2_packet.attempt_manifest(FIRST, 20.451188 * 10 / 3600, Path("/o"))
    assert job["budget"]["unit_cap"] == 1.94 and job["timeout_minutes"] == 11.6


def test_attempt_limits_refuse():
    with pytest.raises(ValueError):
        r2_packet.attempt_manifest(FIRST, 0.0, Path("/o"), attempt=3)
    with pytest.raises(ValueError):
        r2_packet.attempt_manifest(FIRST, 1.0, Path("/o"))  # 1 u cannot fit 7+3 min
    bad = copy.deepcopy(FIRST)
    bad["job_id"] = "gpu-r2-canary-x-02"
    with pytest.raises(ValueError):
        r2_packet.attempt_manifest(bad, 0.0, Path("/o"))


@pytest.mark.skipif(
    not (ART / "packet/canary-output/receipt.json").is_file(), reason="R2 artifacts absent"
)
def test_real_attempt_one_is_a_released_prework_kernel_creation_failure():
    proof = r2_packet.prior_failure(ART)
    receipt = json.loads((ART / "packet/canary-output/receipt.json").read_text())
    assert proof["workload_gpu_seconds"] == 0
    assert proof["observed_balance_drop_units"] == 0
    assert proof["charged_units"] == pytest.approx(receipt["elapsed_seconds"] * 10 / 3600)


def test_main_preparation_never_regenerates_the_executed_canary(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(r2_packet, "frozen", lambda broker, code, *args: calls.append(code))
    r2_packet._main_manifest(tmp_path, tmp_path, "e" * 40, tmp_path / "receipt.json")
    (code,) = calls
    assert "prepare(" not in code and "r2-canary" not in code
    assert "r2-main.json" in code and "canary=False" in code and "validate(m" in code


def test_r6_family_uses_its_own_names_producer_key_and_main_builder(tmp_path, monkeypatch):
    from tools.ops import r6_packet

    monkeypatch.setattr(r2_packet, "FAMILY", r6_packet.R6)
    first = copy.deepcopy(FIRST)
    first["job_id"] = "gpu-r6-canary-da09cba17c64-01"
    first["env"] = {"R6_BENCH_SECONDS": "360"}
    job = r2_packet.attempt_manifest(first, 0.05, Path("/o"))
    assert job["job_id"].endswith("-02") and job["timeout_minutes"] >= 360 / 60 + 3
    calls = []
    monkeypatch.setattr(r2_packet, "frozen", lambda broker, code, *args: calls.append(code))
    r2_packet._main_manifest(tmp_path, tmp_path, "d" * 40, tmp_path / "receipt.json")
    assert len(calls) == 1 and "r6_pack import prepare_main" in calls[0]
    assert "r2" not in calls[0] and "r6-canary" not in calls[0]
