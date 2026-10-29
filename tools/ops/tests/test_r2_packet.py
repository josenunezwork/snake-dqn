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
    assert prior + cap <= r2_packet.CANARY_CAP_UNITS + 1e-12
    assert timeout == FIRST["timeout_minutes"]  # Wall time is independent of units.
    assert timeout >= 2 * job["requirements"]["est_minutes"]  # qualified 2x rule
    assert timeout >= 420 / 60 + 3  # unchanged producer deadline still fits
    assert job["job_id"] == "gpu-r2-canary-e66a701fd439-02"
    assert set(job["canary_attempt"]["changed_fields"]) >= _changed(FIRST, job) - {"canary_attempt"}
    assert job["env"] == FIRST["env"] and job["inputs"] == FIRST["inputs"]
    assert job["native_cuda_compile"] == FIRST["native_cuda_compile"]


def test_attempt_two_for_the_observed_failure_charge():
    job = r2_packet.attempt_manifest(FIRST, 20.451188 * 10 / 3600, Path("/o"))
    assert job["budget"]["unit_cap"] == 1.94 and job["timeout_minutes"] == 12


def test_attempt_limits_refuse():
    with pytest.raises(ValueError):
        r2_packet.attempt_manifest(FIRST, 0.0, Path("/o"), attempt=3)
    assert r2_packet.attempt_manifest(FIRST, 1.0, Path("/o"))["timeout_minutes"] == 12
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


def _staging_package(tmp_path, *, error=None, returncode=0, staging=362.0, edit=None):
    import hashlib

    from tools.ops.repack_gpu_upload import publish

    root = tmp_path / "packet/native-compile"
    manifest = {
        "job_id": "native-compile-l4-da09cba1-r6",
        "commit": "d" * 40,
        "script": "research/gpu_sim_spike/native_compile.sh",
        "budget": {"other_assignments": {}, "unit_cap": 1.2},
        "output_dir": str(root / "output"),
    }
    publish(root / "native-compile.json", manifest)
    output = root / "output"
    output.mkdir()
    executed = copy.deepcopy(manifest)
    if edit:
        edit(executed)
    (output / "manifest.json").write_text(json.dumps(executed))
    (output / "upload-measurement.json").write_text(json.dumps({"elapsed_seconds": staging}))
    receipt = {
        "status": "INCOMPLETE",
        "failure_class": "SUPERVISOR",
        "error": error or r2_packet.STAGING_ERROR,
        "stop_verified": True,
        "cleanup_error": None,
        "job_id": manifest["job_id"],
        "manifest_sha256": hashlib.sha256((output / "manifest.json").read_bytes()).hexdigest(),
        "terminal": {"returncode": returncode},
        "elapsed_seconds": 1058.757,
        "accounting": {"own_rate_units_per_hour": 3.0},
        "generation": "g",
        "session": "s",
        "usage_before": {"balance": 1080.09},
        "usage_after": {"balance": 1079.77},
    }
    (output / "receipt.json").write_text(json.dumps(receipt))
    return tmp_path, manifest


def test_compile_staging_failure_is_charged_and_freshly_identified(tmp_path):
    artifact, manifest = _staging_package(tmp_path)
    proof = r2_packet.compile_staging_failure(artifact)
    assert proof["charged_units"] == pytest.approx(1058.757 * 3 / 3600)
    assert proof["observed_balance_drop_units"] == pytest.approx(0.32)
    job = r2_packet.compile_attempt_manifest(manifest, proof, Path("/o-02"))
    assert job["job_id"] == "native-compile-l4-da09cba1-r6-02" and job["output_dir"] == "/o-02"
    rest = {k: v for k, v in job.items() if k not in ("job_id", "output_dir", "compile_attempt")}
    assert rest == {k: v for k, v in manifest.items() if k not in ("job_id", "output_dir")}
    with pytest.raises(ValueError):
        r2_packet.compile_attempt_manifest(manifest, proof, Path("/o"), attempt=3)


@pytest.mark.parametrize(
    "kwargs",
    (
        {"error": "Refused('other')"},
        {"returncode": 1},
        {"staging": 120.0},
        {"edit": lambda m: m.update(script="research/gpu_sim_spike/other.sh")},
    ),
)
def test_compile_staging_failure_refuses_other_shapes(tmp_path, kwargs):
    artifact, _ = _staging_package(tmp_path, **kwargs)
    with pytest.raises(ValueError):
        r2_packet.compile_staging_failure(artifact)


@pytest.mark.skipif(
    not (
        Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp")
        / "gpu-r6-escaling-20261024/packet/native-compile/output/receipt.json"
    ).is_file(),
    reason="R6 artifacts absent",
)
def test_real_r6_compile_attempt_one_is_a_staging_time_limit_failure(monkeypatch):
    from tools.ops import r6_packet

    monkeypatch.setattr(r2_packet, "FAMILY", r6_packet.R6)
    proof = r2_packet.compile_staging_failure(r6_packet.R6.artifact)
    assert proof["terminal_returncode"] == 0 and proof["staging_seconds"]["elapsed_seconds"] > 300


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
