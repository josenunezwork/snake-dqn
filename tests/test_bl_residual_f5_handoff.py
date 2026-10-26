"""End-to-end synthetic F5 environment and pre-fit handoff checks.

The host-dependent capture and selected-package payload seams are synthetic so
this test can run without a CUDA host or a science package.  Source closure,
environment admission, assignment comparison, plan materialization,
timing-plan validation, and the fit-admission receipt checker are exercised
directly.
"""

from __future__ import annotations

import base64
import copy
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from research.bl_ppo.exports import export_json
from research.bl_residual import f5_environment, f5_job, f5_qualification, f5_timing
from research.bl_residual.f5_package import source_inventory
from research.bl_residual.f5_r10a_annex import allocate


def _sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _package(name: str, root: str) -> dict[str, Any]:
    assignment = root.rsplit("/", 1)[-1]
    wheel = {
        "filename": f"{name}-1.0.whl",
        "raw_url": f"file:///wheelhouse/{assignment}/{name}-1.0.whl",
        "canonical_url": f"file://<WHEELHOUSE>/{name}-1.0.whl",
        "sha256": hashlib.sha256(f"wheel:{name}".encode()).hexdigest(),
    }
    content_raw = f"fixed installed library bytes:{name}".encode()
    contents = {
        f"lib/{name}.py": {
            "raw_sha256": hashlib.sha256(content_raw).hexdigest(),
            "canonical_sha256": hashlib.sha256(content_raw).hexdigest(),
        }
    }
    metadata = {}
    for metadata_name in ("METADATA", "WHEEL", "RECORD", "direct_url.json"):
        if metadata_name in {"METADATA", "WHEEL"}:
            raw = metadata_name.encode()
        elif metadata_name == "direct_url.json":
            raw = (
                f'{{"url":"file:///wheelhouse/{root.rsplit("/", 1)[-1]}/{name}-1.0.whl"}}'.encode()
            )
        else:
            raw = f"{root}/{metadata_name}".encode()
        metadata[metadata_name] = {
            "raw_base64": base64.b64encode(raw).decode(),
            "raw_sha256": hashlib.sha256(raw).hexdigest(),
            "canonical_sha256": hashlib.sha256(
                f5_environment._canonical_metadata_bytes(raw, root, metadata_name, contents, wheel)
            ).hexdigest(),
        }
    return {
        "version": f"{name}-1.0",
        "installation_root": root,
        "installed_contents_sha256": hashlib.sha256(
            f5_environment.canonical_json(
                {key: item["canonical_sha256"] for key, item in contents.items()}
            ).encode()
        ).hexdigest(),
        "installed_contents": contents,
        "installed_metadata": metadata,
    }


def _environment(
    *, uuid: str = "GPU-A", node: str = "host-a", affinity: list[int] | None = None
) -> dict[str, Any]:
    """Return a complete synthetic receipt accepted by the frozen schema."""
    affinity = [0, 1, 2, 3, 4, 5] if affinity is None else affinity
    packages = {
        name: _package(name, f"/opt/{name}/{uuid.lower()}")
        for name in ("torch", "numpy", "numba", "llvmlite", "scipy")
    }
    packages["torch"]["version"] = "2.9.1+cu126"
    packages["numpy"]["version"] = "2.2.0"
    return {
        "schema": f5_environment.SCHEMA,
        "schema_sha256": hashlib.sha256(
            (Path(f5_environment.__file__).with_name("f5_environment_schema_v3.json")).read_bytes()
        ).hexdigest(),
        "platform": "Linux-6.1-x86_64-with-glibc2.35",
        "python": "3.12.3 (main, synthetic)",
        "cpu": "Synthetic Xeon",
        "cpu_count": 8,
        "torch": "2.9.1+cu126",
        "cuda": "12.6",
        "gpu": "L4 Synthetic",
        "gpu_uuid": uuid,
        "gpu_capability": [8, 9],
        "gpu_memory": 24576 * f5_environment.MIB_BYTES,
        "memory_units": dict(f5_environment.MEMORY_UNITS),
        "os_kernel": {
            "system": "Linux",
            "node": node,
            "release": "6.1.0-synthetic",
            "version": "#1 synthetic",
            "machine": "x86_64",
            "processor": "Synthetic Xeon",
        },
        "python_build": ["main", "2026-01-01"],
        "python_compiler": "GCC 12.2",
        "python_cache_tag": "cpython-312",
        "cpu_isa": {"model name": "Synthetic Xeon", "flags": "avx2"},
        "allocated_cpu_affinity": affinity,
        "gpu_driver_query": {
            "name": "L4 Synthetic",
            "uuid": uuid,
            "driver_version": "550.54.15",
            "compute_cap": "8.9",
            "memory_total": "24576",
        },
        "gpu_driver_query_raw": (f"L4 Synthetic, {uuid}, 550.54.15, 8.9, 24576"),
        "cudnn": "9.8",
        "torch_git_version": "synthetic-git",
        "torch_build": "USE_CUDA=1; USE_ROCM=0",
        "numpy": "2.2.0",
        "numpy_blas_lapack_build": "openblas-synthetic",
        "installed_numeric_packages": packages,
        "installed_packages": copy.deepcopy(packages),
        "installed_versions_sha256": "b" * 64,
        "environment_lock_sha256": "c" * 64,
        "environment_descriptor_sha256": "d" * 64,
        "locked_wheels": {
            name: {
                "filename": f"{name}-1.0.whl",
                "raw_url": f"file:///wheelhouse/{uuid.lower()}/{name}-1.0.whl",
                "canonical_url": f"file://<WHEELHOUSE>/{name}-1.0.whl",
                "sha256": hashlib.sha256(f"wheel:{name}".encode()).hexdigest(),
            }
            for name in ("torch", "numpy", "numba", "llvmlite", "scipy")
        },
        "torch_threads": {"intra": 1, "inter": 1},
        "numeric_controls": {
            "deterministic": True,
            "cudnn_deterministic": True,
            "cudnn_benchmark": False,
            "matmul_tf32": False,
            "cudnn_tf32": False,
            "default_dtype": "torch.float32",
            "amp": False,
            "bf16": False,
        },
        "thread_env": {
            name: (":4096:8" if name == "CUBLAS_WORKSPACE_CONFIG" else "1")
            for name in (
                "OMP_NUM_THREADS",
                "OMP_THREAD_LIMIT",
                "MKL_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "NUMBA_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "CUBLAS_WORKSPACE_CONFIG",
            )
        },
    }


def _runtime_capture(monkeypatch: pytest.MonkeyPatch, receipt: dict[str, Any]) -> None:
    """Satisfy the pinned runtime checks while replacing only host observation."""
    monkeypatch.setattr(f5_job, "capture_environment", lambda: copy.deepcopy(receipt))
    monkeypatch.setattr(f5_job.sys, "platform", "linux")
    monkeypatch.setattr(f5_job.sys, "version_info", (3, 12, 3))
    monkeypatch.setattr(f5_job.torch, "__version__", f5_job.L4_TORCH)
    monkeypatch.setattr(f5_job.torch.version, "cuda", f5_job.L4_CUDA)
    monkeypatch.setattr(f5_job.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(f5_job.torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(
        f5_job.torch.cuda,
        "get_device_properties",
        lambda _: SimpleNamespace(name="NVIDIA L4 Synthetic"),
    )
    monkeypatch.setattr(f5_job.os, "sched_getaffinity", lambda _: set(range(6)), raising=False)
    monkeypatch.setattr(f5_job.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(f5_job.torch, "get_num_interop_threads", lambda: 1)
    monkeypatch.setattr(f5_job.torch, "set_num_interop_threads", lambda _: None)
    monkeypatch.setattr(f5_job.torch, "use_deterministic_algorithms", lambda _: None)
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        monkeypatch.setenv(name, "1")
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def _control_inputs(tmp_path: Path, environment: dict[str, Any]) -> Path:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    package = inputs / "selected-training.tar.xz"
    package.write_bytes(b"synthetic selected package boundary")
    source_root = Path(f5_job.__file__).resolve().parents[2]
    source_hashes = source_inventory(source_root)
    control: dict[str, Any] = {
        "schema": f5_job.SCHEMA,
        "namespace": f5_job.NAMESPACE,
        "identity": {
            "source_sha256": _sha(source_hashes),
            "loss_sha256": f5_job.digest(source_root / "research/bl_residual/f5_fit.py"),
            "environment_sha256": (
                "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5"
            ),
            "allocation_sha256": (
                "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
            ),
            "corpus_sha256": "4" * 64,
        },
        "input_hashes": {package.name: f5_job.digest(package)},
        "environment_capture": environment,
        "environment_policy": f5_environment.policy_binding(),
        "source_hashes": source_hashes,
    }
    control["allocation"] = _sha(control)
    (inputs / "control.json").write_text(json.dumps(control, sort_keys=True))
    return inputs


def _timing_allocation() -> list[dict[str, Any]]:
    rows = []
    index = 672
    for stream in f5_timing.TIMING_GROUPS:
        for group_index in range(3):
            rows.append(
                {
                    "allocation_index": index,
                    "group": f"timing/{stream}",
                    "group_index": group_index,
                    "parent_id": (
                        "bl-f5-predat-v2/20261025/r=0/" f"timing/{stream}/{group_index:04d}"
                    ),
                    "timing_only": True,
                }
            )
            index += 1
    return rows


def test_environment_schema_is_exhaustive_and_nested_unknown_fields_refuse() -> None:
    receipt = _environment()
    f5_environment.validate_environment(receipt)
    extra = copy.deepcopy(receipt)
    extra["unexpected"] = True
    with pytest.raises(f5_environment.F5EnvironmentRefusal, match="unknown or missing"):
        f5_environment.validate_environment(extra)
    nested = copy.deepcopy(receipt)
    nested["gpu_driver_query"]["unexpected"] = "drift"
    with pytest.raises(f5_environment.F5EnvironmentRefusal):
        f5_environment.validate_environment(nested)


def test_legitimate_distinct_assignment_provenance_passes_invariant_comparison() -> None:
    first = _environment()
    second = _environment(uuid="GPU-B", node="host-b", affinity=[6, 7, 8, 9, 10, 11])
    result = f5_environment.compare_environment(first, second, mode="cross-assignment")
    assert result["mode"] == "cross-assignment"
    assert result["expected_raw_sha256"] != result["actual_raw_sha256"]
    assert result["expected_canonical_sha256"] == result["actual_canonical_sha256"]
    assert result["assignment_provenance"]["actual"]["gpu_uuid"] == "GPU-B"


@pytest.mark.parametrize(
    "path",
    [
        ("torch_build",),
        ("numeric_controls", "amp"),
        ("gpu_driver_query", "driver_version"),
        ("installed_numeric_packages", "torch", "installed_contents_sha256"),
    ],
)
def test_cross_assignment_build_or_numerical_drift_refuses(path: tuple[str, ...]) -> None:
    first = _environment()
    second = _environment(uuid="GPU-B", node="host-b", affinity=[6, 7, 8, 9, 10, 11])
    cursor: Any = second
    for key in path[:-1]:
        cursor = cursor[key]
    if path[-1].endswith("sha256"):
        cursor[path[-1]] = "e" * 64
    else:
        cursor[path[-1]] = "drift" if isinstance(cursor[path[-1]], str) else not cursor[path[-1]]
    with pytest.raises(f5_environment.F5EnvironmentRefusal):
        f5_environment.compare_environment(first, second, mode="cross-assignment")


def test_same_assignment_requires_raw_identity() -> None:
    first = _environment()
    f5_environment.compare_environment(first, copy.deepcopy(first), mode="same-assignment")
    changed = copy.deepcopy(first)
    changed["os_kernel"]["node"] = "host-replaced"
    with pytest.raises(f5_environment.F5EnvironmentRefusal, match="raw receipt"):
        f5_environment.compare_environment(first, changed, mode="same-assignment")


def test_assignment_boundary_refuses_numeric_control_mutation_without_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference = _environment()
    mutated = copy.deepcopy(reference)
    mutated["numeric_controls"]["amp"] = True
    monkeypatch.setattr(f5_job, "capture_environment", lambda: copy.deepcopy(mutated))
    monkeypatch.setattr(f5_job.torch, "set_num_threads", lambda *_: pytest.fail("boundary reset"))
    monkeypatch.setattr(
        f5_job.torch, "use_deterministic_algorithms", lambda *_: pytest.fail("boundary reset")
    )
    with pytest.raises(f5_environment.F5EnvironmentRefusal, match="raw receipt"):
        f5_job._assignment_boundary(reference)


def test_qualification_runner_records_synthetic_cpu_backend_and_broker_exports(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment()
    _runtime_capture(monkeypatch, environment)
    inputs = _control_inputs(tmp_path, environment)
    checkpoint = inputs / "b8.pth"
    checkpoint.write_bytes(b"synthetic pinned checkpoint bytes")
    fixture_path = tmp_path / "timing-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema": f5_qualification.FIXTURE_SCHEMA,
                "fixtures": [row for row in allocate(0) if row["timing_only"] is True],
            }
        )
    )
    control_path = inputs / "control.json"
    control = json.loads(control_path.read_text())
    checkpoint_sha = f5_job.digest(checkpoint)
    monkeypatch.setattr(f5_job, "B8_SHA256", checkpoint_sha)
    control["input_hashes"][checkpoint.name] = checkpoint_sha
    control["qualification_fixture_sha256"] = hashlib.sha256(fixture_path.read_bytes()).hexdigest()
    control["qualification_b8_sha256"] = checkpoint_sha
    control["allocation"] = _sha(
        {key: value for key, value in control.items() if key != "allocation"}
    )
    control_path.write_text(json.dumps(control, sort_keys=True))

    class SyntheticNet:
        def eval(self) -> "SyntheticNet":
            return self

        def cuda(self) -> "SyntheticNet":
            return self

    q_values = np.arange(6, dtype=np.float32)
    # This exercises broker/receipt orchestration only; it is explicitly not
    # observer evidence for the genuine CUDA qualification gate.
    monkeypatch.setattr(f5_qualification, "_require_cuda", lambda _: None)
    monkeypatch.setattr(
        f5_qualification,
        "_native_observation",
        lambda seed: ({"synthetic_seed": np.asarray([seed])}, np.ones(6, dtype=bool)),
    )
    monkeypatch.setattr(f5_qualification, "_select_cuda", lambda *_: (q_values, 5))
    monkeypatch.setattr(f5_qualification, "_cpu_action", lambda *_: (q_values, 5))
    import src.model.ego2s_network as ego2s_network

    monkeypatch.setattr(ego2s_network, "load_ego2s_checkpoint", lambda *_: SyntheticNet())
    monkeypatch.setattr(ego2s_network, "spec_for", lambda _: "ego2s-bv8")

    qualification_output = tmp_path / "qualification"
    qualification_output.mkdir()
    report = f5_qualification.run(
        inputs,
        fixture_path,
        qualification_output,
        device="synthetic-cpu",
        transport_only=True,
    )
    assert report["status"] == "COMPLETE_TRANSPORT_ONLY_NOT_TIMING_QUALIFIED"
    assert report["device"] == "synthetic-cpu"
    assert report["action_count"] == 6
    assert report["fit_or_optimizer_executed"] is False
    assert report["mac_timing_status"] == "NOT_MEASURED_TRANSPORT_ONLY"
    assert len(report["samples"]) == 12
    assert all(sample["parity"] == "PASS" for sample in report["samples"])
    assert len(report["producer_exports"]) == 6
    assert all(Path(item["path"]).is_file() for item in report["producer_exports"])
    assignment_exports = [
        Path(item["path"])
        for item in report["producer_exports"]
        if Path(item["path"]).name == "assignment-environment.json"
    ]
    assert len(assignment_exports) == 1
    assert json.loads(assignment_exports[0].read_text()) == environment


def test_fit_pair_refuses_cross_assignment_build_drift_before_fitter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment()
    _runtime_capture(monkeypatch, environment)
    inputs = _control_inputs(tmp_path, environment)
    control_path = inputs / "control.json"
    control = json.loads(control_path.read_text())
    control["environment_capture"]["torch_build"] = "DRIFTED-BUILD"
    control["allocation"] = _sha(
        {key: value for key, value in control.items() if key != "allocation"}
    )
    control_path.write_text(json.dumps(control, sort_keys=True))
    monkeypatch.setattr(
        f5_job, "FrozenB8Residual", lambda *_: pytest.fail("fit reached after environment drift")
    )
    with pytest.raises(f5_job.F5FitRefusal, match="invariant environment differs"):
        f5_job.fit_pair(
            inputs,
            tmp_path / "missing-plan.pt",
            tmp_path / "missing-acceptance.json",
            tmp_path / "missing-ledger.json",
            tmp_path / "fit-output",
        )


def test_fit_pair_positive_cw_handoff_and_cross_assignment_resume_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment()
    _runtime_capture(monkeypatch, environment)
    inputs = _control_inputs(tmp_path, environment)
    metadata = {
        "status": torch.ones(8, 6, dtype=torch.bool),
        "legal": torch.ones(8, 6, dtype=torch.bool),
        "action_mask": torch.ones(8, 6, dtype=torch.bool),
        "died": torch.tensor([[False, True, False, True, False, True]] * 8),
        "ordinary": torch.tensor([False, True, False, True, False, True, False, True]),
        "life_id": torch.arange(8),
    }
    monkeypatch.setattr(f5_job, "_payload_metadata", lambda *args: (metadata, {}))
    prepare_output = tmp_path / "prepare"
    f5_job.prepare(inputs, prepare_output)

    checkpoint = inputs / "b8.pth"
    checkpoint.write_bytes(b"synthetic paired-fit checkpoint bytes")
    checkpoint_sha = f5_job.digest(checkpoint)
    monkeypatch.setattr(f5_job, "B8_SHA256", checkpoint_sha)
    control_path = inputs / "control.json"
    control = json.loads(control_path.read_text())
    control["input_hashes"][checkpoint.name] = checkpoint_sha
    control["allocation"] = _sha(
        {key: value for key, value in control.items() if key != "allocation"}
    )
    ledger_path = inputs / "verified-ledger.json"
    ledger = {
        "schema": "bl-f5-verified-ledger/v1",
        "namespace": f5_job.NAMESPACE,
        "allocation": control["allocation"],
        "units": [],
    }
    ledger_path.write_text(json.dumps(ledger, sort_keys=True))
    control["ledger_sha256"] = f5_job.digest(ledger_path)
    control_path.write_text(json.dumps(control, sort_keys=True))

    plan = f5_job._load_plan(prepare_output / f5_job.PLAN_NAME)
    acceptance = {
        "decision": "ACCEPT_F5_FIT",
        "external": True,
        "source_sha256": plan.identity.source_sha256,
        "loss_sha256": plan.identity.loss_sha256,
        "environment_sha256": plan.identity.environment_sha256,
        "allocation_sha256": plan.identity.allocation_sha256,
        "corpus_sha256": plan.identity.corpus_sha256,
        "batch_plan_sha256": plan.sha256,
        "control_sha256": _sha(json.loads(control_path.read_text())),
        "reviewer": "synthetic-independent-reviewer",
        "reviewer_reference_sha256": "5" * 64,
        "one_difference_sha256": "6" * 64,
        "timing_sha256": "7" * 64,
    }
    acceptance["acceptance_sha256"] = _sha(acceptance)
    acceptance_path = tmp_path / "acceptance.json"
    export_json(acceptance_path, acceptance)

    class SyntheticBase(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.anchor = torch.nn.Parameter(torch.zeros(1))

    class SyntheticResidualModel:
        def __init__(self, base: SyntheticBase) -> None:
            self.b8 = base
            self.residual = torch.nn.Linear(1, 1)

        def to(self, _: str) -> "SyntheticResidualModel":
            return self

    def synthetic_state(recipe: Any, plan_value: Any) -> SimpleNamespace:
        model = SyntheticResidualModel(SyntheticBase())
        return SimpleNamespace(
            model=model,
            optimizer_state={},
            updates_completed=2000,
            losses=(0.0,),
            identity=plan_value.identity,
            recipe=recipe,
            plan_sha256=plan_value.sha256,
            python_rng_state=(3, (1, 2), None),
            numpy_rng_state=("MT19937", np.zeros(624, dtype=np.uint32), 0, 0.0, 0),
            torch_cpu_rng_state=torch.zeros(1, dtype=torch.uint8),
            torch_cuda_rng_states=(),
            plan_generator_state=torch.zeros(1, dtype=torch.uint8),
        )

    monkeypatch.setattr(f5_job, "load_ego2s_checkpoint", lambda *_: SyntheticBase())
    monkeypatch.setattr(f5_job, "FrozenB8Residual", lambda base: SyntheticResidualModel(base))
    monkeypatch.setattr(f5_job, "selected_tensor_cache", lambda *_: {})
    monkeypatch.setattr(f5_job, "fuse_residual", lambda model: model.b8)
    monkeypatch.setattr(
        f5_job,
        "fit_f5_accepted",
        lambda model, cache, data, plan_value, recipe, acceptance_value, prior, callback: (
            callback(state := synthetic_state(recipe, plan_value)) or state
        ),
    )
    monkeypatch.setattr(
        f5_job,
        "_checkpoint_blob",
        lambda net, arm, plan: {"synthetic": True, "arm": arm, "plan_sha256": plan.sha256},
    )

    output = tmp_path / "fit-output"
    receipt = f5_job.fit_pair(
        inputs, prepare_output / f5_job.PLAN_NAME, acceptance_path, ledger_path, output
    )
    assert receipt["kind"] == "COMPLETE_PAIRED_FIT_UNIT"
    assert set(receipt["arms"]) == {"C", "W"}
    assert all(receipt["arms"][arm]["updates_completed"] == 2000 for arm in ("C", "W"))
    boundaries = receipt["same_assignment_boundaries"]
    assert boundaries["before_C"]["mode"] == "same-assignment"
    assert boundaries["after_C"]["mode"] == "same-assignment"
    assert boundaries["after_W"]["mode"] == "same-assignment"
    assert receipt["environment"]["reference_raw"] == environment
    assert receipt["environment"]["assignment_raw"] == environment
    assignment_receipt = output / "assignment-environment.json"
    assert assignment_receipt.is_file()
    assert json.loads(assignment_receipt.read_text()) == environment
    assert (output / "boundaries/C/2000.pt").is_file()
    state = torch.load(output / "boundaries/C/2000.pt", map_location="cpu", weights_only=False)
    assert state["assignment_environment_sha256"] == f5_environment.receipt_sha256(environment)

    cross_assignment = copy.deepcopy(environment)
    cross_assignment["os_kernel"]["node"] = "host-cross-assignment"
    with pytest.raises(f5_job.F5FitRefusal, match="raw assignment environment differs"):
        f5_job._resume_state(output / "boundaries/C/2000.pt", None, cross_assignment)


def test_capture_prepare_qualification_fit_admission_handoff_uses_real_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment()
    _runtime_capture(monkeypatch, environment)
    inputs = _control_inputs(tmp_path, environment)

    metadata = {
        "status": torch.ones(8, 6, dtype=torch.bool),
        "legal": torch.ones(8, 6, dtype=torch.bool),
        "action_mask": torch.ones(8, 6, dtype=torch.bool),
        "died": torch.tensor([[False, True, False, True, False, True]] * 8),
        "ordinary": torch.tensor([False, True, False, True, False, True, False, True]),
        "life_id": torch.arange(8),
    }
    monkeypatch.setattr(f5_job, "_payload_metadata", lambda *args: (metadata, {}))
    prepared = f5_job.prepare(inputs, tmp_path / "prepare")
    assert prepared["rows"] == 512000
    receipt = json.loads((tmp_path / "prepare" / "prepare-receipt.json").read_text())
    assert receipt["environment"]["comparison"]["mode"] == "cross-assignment"
    assert receipt["closing_environment"]["mode"] == "same-assignment"

    timing = f5_timing.qualification_plan(_timing_allocation())
    assert timing["status"] == "PREPARED_NOT_MEASURED"
    assert len(timing["timing_parents"]) == 12
    fixture_path = tmp_path / "timing-fixtures.json"
    fixture_path.write_text(
        json.dumps(
            {
                "schema": f5_qualification.FIXTURE_SCHEMA,
                "fixtures": [row for row in allocate(0) if row["timing_only"] is True],
            }
        )
    )
    fixture_rows, fixture_sha = f5_qualification.load_fixtures(fixture_path)
    assert len(fixture_rows) == 12 and len(fixture_sha) == 64

    control = f5_job._control(inputs)
    plan = f5_job._load_plan(tmp_path / "prepare" / f5_job.PLAN_NAME)
    acceptance = {
        "decision": "ACCEPT_F5_FIT",
        "external": True,
        "source_sha256": plan.identity.source_sha256,
        "loss_sha256": plan.identity.loss_sha256,
        "environment_sha256": plan.identity.environment_sha256,
        "allocation_sha256": plan.identity.allocation_sha256,
        "corpus_sha256": plan.identity.corpus_sha256,
        "batch_plan_sha256": plan.sha256,
        "control_sha256": _sha(control),
        "reviewer": "synthetic-independent-reviewer",
        "reviewer_reference_sha256": "5" * 64,
        "one_difference_sha256": "6" * 64,
        "timing_sha256": "7" * 64,
    }
    acceptance["acceptance_sha256"] = _sha(acceptance)
    acceptance_path = tmp_path / "acceptance.json"
    export_json(acceptance_path, acceptance)
    admitted = f5_job._acceptance(acceptance_path, control, plan)
    assert admitted["decision"] == "ACCEPT_F5_FIT"
