"""Provider-free C/T H100 preference, adapter, hardware and continuation proofs."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch

from research.compute.gpu_preference import prepare_preference, variants
from research.sandbox_a_resident.ct_preflight import CT_FOCUSED_TESTS, validate_uploaded_ledger
from research.sandbox_a_resident.ct_resume import CTResumeRefused, write_continuation_packet
from research.sandbox_a_resident.ct_task import CTControl
from research.sandbox_a_resident.ct_units import allocation, publish_stage_snapshot
from research.sandbox_a_resident.test_ct_runner_and_resume import (
    _clean_supervisor_log,
    _rebind_forged_packet,
    _save_tiny_checkpoints,
    _write_tiny_initial_packet,
)
from research.sandbox_a_resident.test_runtime_hardware import (
    test_launchers_compile_for_requested_hardware as _check_launcher,
)
from tools.ops.gpu_receipt import publish
from tools.sandbox.registry import digest


def test_preflight_keeps_review14_and_watchdog_regressions() -> None:
    for name in ("test_ct_review14.py", "test_stall_watchdog.py", Path(__file__).name):
        assert f"research/sandbox_a_resident/{name}" in CT_FOCUSED_TESTS


@pytest.mark.parametrize("gpu,arch", [("H100", "90"), ("A100", "80")])
@pytest.mark.parametrize("missing_hopper", [False, True])
def test_ct_adapter_compiles_for_requested_hardware(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gpu: str, arch: str, missing_hopper: bool
) -> None:
    monkeypatch.setenv("SANDBOX_CT_CONTROL_JSON", "{}")
    _check_launcher(tmp_path, "ct_job.sh", gpu, arch, missing_hopper)


def test_preference_prices_whole_timeout_and_preserves_science() -> None:
    packet = Path(__file__).parent / "packet-ct-1028-14/reseal-02/dispatch-manifest.json"
    original = json.loads(packet.read_text())
    preferred = copy.deepcopy(original)
    prepare_preference(preferred)
    primary, fallback = variants(preferred)
    assert primary["requirements"]["prefer"] == ["H100", "A100"]
    for item, gpu, rate, cap in ((primary, "H100", 12, 288), (fallback, "A100", 8, 192)):
        assert item["requirements"]["gpu"] == gpu
        assert item["requirements"]["allowed_gpus"] == [gpu]
        assert item["requirements"]["est_minutes"] == 720
        assert item["timeout_minutes"] == 1440
        assert item["budget"]["rate_units_per_hour"] == rate
        assert item["budget"]["unit_cap"] == cap
        assert item["technical_envelope"]["base_unit_cap"] == cap / 2
        assert item["technical_envelope"]["effective_unit_cap"] == cap
        assert item["budget"]["reserve_units"] == 0
        science_keys = ("env", "inputs", "resume", "sandbox_a", "source_paths", "required_exports")
        for key in science_keys:
            assert item[key] == original[key]
    assert fallback["job_id"] == original["job_id"] + "-a100-fallback"
    assert fallback["output_dir"] == original["output_dir"] + "-a100-fallback"
    primary["budget"]["unit_cap"] -= 0.01
    with pytest.raises(ValueError, match="whole timeout"):
        variants(primary)


@pytest.mark.parametrize("gpu", ["H100", "A100"])
def test_hardware_receipt_and_two_manual_continuations(tmp_path: Path, gpu: str) -> None:
    """Exercise writer/ledger admission with hardware evidence and fault precedence."""
    checkpoints = _save_tiny_checkpoints(tmp_path)
    ctrl = CTControl(source_commit="a" * 40, worlds=6, collection_ticks=12, tiny=True)
    initial = tmp_path / "initial"
    _write_tiny_initial_packet(initial, ctrl, checkpoints, tmp_path / "attempt-0")
    manifest = json.loads((initial / "dispatch-manifest.json").read_text())
    prepare_preference(manifest)
    manifest = variants(manifest)[0 if gpu == "H100" else 1]
    # Standalone canonical fixture for the actual selected hardware. The dispatcher
    # separately binds preference variants; do not infer observed hardware from prefer.
    manifest["requirements"].pop("prefer")
    _rebind_forged_packet(initial, manifest)
    seal = json.loads((initial / "SEAL.json").read_text())
    seal["sandbox_sha256"] = digest(initial / "sandbox.json")
    (initial / "SEAL.json").write_text(json.dumps(seal))
    control = ctrl.to_dict()
    validate_uploaded_ledger(initial, copy.deepcopy(manifest), control, require_committed=False)
    prior, roots = initial, []
    for index in (1, 2):
        manifest_path = prior / "dispatch-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        output = Path(manifest["output_dir"])
        exports = output / "exports/ct"
        publish_stage_snapshot(
            exports,
            index,
            lambda dst: torch.save({"kind": "collection", "stage": index}, dst),
            metadata={"allocation": allocation(control), "source_commit": ctrl.source_commit},
        )
        roots.append(exports)
        hardware = {
            "ready": True,
            "hardware": {
                "kind": gpu,
                "name": "NVIDIA H100 80GB HBM3" if gpu == "H100" else "NVIDIA A100-SXM4-40GB",
                "compute_cap": "9.0" if gpu == "H100" else "8.0",
            },
        }
        (output / "hardware.json").write_text(json.dumps(hardware))
        receipt = {
            "job_id": manifest["job_id"],
            "commit": manifest["commit"],
            "manifest_sha256": digest(manifest_path),
            "status": "INCOMPLETE",
            "failure_class": "CHILD_NONZERO_EXIT",
            "stop_verified": True,
            "cleanup_error": None,
            "generation": f"generation-{index}",
            "terminal": {
                "job_id": manifest["job_id"],
                "generation": f"generation-{index}",
                "status": "failed",
                "reason": "timeout",
                "returncode": 138,
            },
        }
        (output / "receipt.json").write_text(json.dumps(receipt))
        evidence = tmp_path / f"gpu-receipt-{index}.json"
        publish(manifest_path, evidence)
        actual = json.loads(evidence.read_text())
        assert actual["requested_gpu"] == gpu
        assert actual["observed_hardware"] == hardware
        assert actual["hardware_evidence"]["sha256"] == digest(output / "hardware.json")
        assert actual["supervisor_receipt"]["sha256"] == digest(output / "receipt.json")
        assert actual["stop_verified"] is True
        with pytest.raises(FileExistsError):
            publish(manifest_path, evidence)
        successor = tmp_path / f"completion-{index}"
        kwargs = dict(
            prior_packet_dir=prior,
            terminal_receipt=receipt,
            export_root=roots,
            completion_index=index,
            job_id=f"ct-{gpu.lower()}-completion-{index}",
            output_dir=tmp_path / f"attempt-{index}",
            use_compact_xz_for_stages=None,
            supervisor_log_path=_clean_supervisor_log(tmp_path, f"supervisor-{index}.log"),
            validate_registry_inputs=False,
        )
        fault = exports / "fault.json"
        fault.write_text('{"scientific_fault":true}')
        with pytest.raises(CTResumeRefused):
            write_continuation_packet(tmp_path / f"refused-{index}", **kwargs)
        assert not (tmp_path / f"refused-{index}").exists()
        fault.unlink()
        result = write_continuation_packet(successor, **kwargs)
        assert result["verified_stages"] == list(range(1, index + 1))
        next_manifest = json.loads((successor / "dispatch-manifest.json").read_text())
        assert next_manifest["requirements"]["gpu"] == gpu
        assert next_manifest["env"] == manifest["env"]
        validate_uploaded_ledger(
            successor, copy.deepcopy(next_manifest), control, require_committed=False
        )
        provenance = json.loads((successor / "provenance.json").read_text())
        retained = {row["path"]: row for row in provenance["retained"][str(output)]["files"]}
        assert retained["hardware.json"]["sha256"] == digest(output / "hardware.json")
        prior = successor


def test_gpu_receipt_does_not_invent_observed_hardware(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "job_id": "not-allocated",
                "output_dir": str(tmp_path / "absent-output"),
                "requirements": {"gpu": "H100"},
                "budget": {},
            }
        )
    )
    destination = tmp_path / "gpu-receipt.json"
    publish(manifest, destination)
    actual = json.loads(destination.read_text())
    assert actual["observed_hardware"] is None
    assert actual["hardware_evidence"] is None
    assert actual["stop_verified"] is False
