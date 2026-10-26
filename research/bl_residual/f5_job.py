"""Offline F5 plan preparation and deliberately gated paired-fit runtime.

The module consumes only a hash-bound selected-training package. ``prepare``
materializes the common plan from the permitted training labels without a fit
or efficacy look. ``fit-pair`` requires an independent acceptance binding the
same source, environment, corpus, allocation, and plan identities before it
executes either optimizer.

This is a runtime entry point, not a dispatcher.  It has no Colab API calls and
does not discover artifact paths or print losses/outcomes.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_json, export_torch, verify_export
from research.bl_residual.f5_data import (
    read_selected_package,
    selected_corpus_identity,
    selected_tensor_cache,
)
from research.bl_residual.f5_fit import (
    ARM_CONTROL,
    ARM_WEIGHTED,
    BatchPlan,
    F5FitRefusal,
    F5FitState,
    F5Identity,
    F5Recipe,
    fit_f5_accepted,
    materialize_batch_plan,
    verify_batch_plan,
)
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from src.model.ego2s_network import ARCH, load_ego2s_checkpoint, spec_for

SCHEMA = "bl-residual-f5-runtime/v1"
NAMESPACE = "bl-residual-f5-w-20261025-v1"
PLAN_NAME = "f5-batch-plan.pt"
STATE_NAME = "f5-fit-state.pt"
SCAN_SHA256 = "8b4910213475266038c8a6a65ad9aea00cd301eba08311f44bb59bc23c312ac4"
SELECTED_RULE_SHA256 = "638372aab7eddcfd7eba5f1725ecb7824b25031c2c2232fbcf43e58efe17410f"
INVENTORY_SHA256 = "cc20e1e7f5c321de5a1f63b94945c92c15c852a8499e88ab56659d1d2d7c1615"
SOURCE_SHA256 = "6906bffc7241de568965c1a91f4faebe799895ee0b4650e59b20e43db1dac494"
SOURCE_BYTES = 45_397_643
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
L4_TORCH = "2.9.1+cu126"
L4_CUDA = "12.6"


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError("expected JSON object")
    return value


def _sha256(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def _identity(control: Mapping[str, Any]) -> F5Identity:
    raw = control.get("identity")
    if not isinstance(raw, dict) or set(raw) != {
        "source_sha256",
        "loss_sha256",
        "environment_sha256",
        "allocation_sha256",
        "corpus_sha256",
    }:
        raise ValueError("F5 control requires the five sealed identities")
    if any(not isinstance(value, str) or len(value) != 64 for value in raw.values()):
        raise ValueError("invalid F5 identity digest")
    return F5Identity(**raw)


def _control(inputs: Path) -> dict[str, Any]:
    path = inputs / "control.json"
    value = _json(path)
    if (
        value.get("schema") != SCHEMA
        or value.get("namespace") != NAMESPACE
        or not isinstance(value.get("allocation"), str)
        or not isinstance(value.get("input_hashes"), dict)
        or value.get("allocation")
        != _sha256({k: v for k, v in value.items() if k not in ("allocation", "ledger_sha256")})
    ):
        raise ValueError("foreign or mutable F5 control")
    _identity(value)
    return value


def _source_admission(control: Mapping[str, Any]) -> None:
    """Refuse source drift instead of accepting an opaque source claim."""
    pins = control.get("source_hashes")
    root = Path(__file__).resolve().parents[2]
    if not isinstance(pins, dict) or not pins:
        raise F5FitRefusal("F5 requires sealed runtime source pins")
    for relative, expected in pins.items():
        component = Path(str(relative))
        if (
            not isinstance(relative, str)
            or not component.parts
            or component.is_absolute()
            or ".." in component.parts
            or component.parts[0] not in {"research", "src"}
        ):
            raise F5FitRefusal("F5 source path escapes the repository closure")
        path = root / str(relative)
        if path.is_symlink() or not path.is_file() or digest(path) != expected:
            raise F5FitRefusal("F5 runtime source differs from its sealed pin")
    from research.bl_residual.f5_package import source_inventory

    if source_inventory(root) != pins:
        raise F5FitRefusal("F5 source map omits or adds runtime dependencies")
    identity = _identity(control)
    if identity.source_sha256 != _sha256(pins) or identity.loss_sha256 != digest(
        Path(__file__).with_name("f5_fit.py")
    ):
        raise F5FitRefusal("F5 source/loss identity linkage differs")
    if (
        identity.environment_sha256
        != "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5"
        or identity.allocation_sha256
        != "1b9fc89e0a814d68ad075bfc5afb961b5c910cbe37365c0bd678142996ed21d7"
    ):
        raise F5FitRefusal("F5 fixed environment/allocation identity differs")


def capture_environment() -> dict[str, Any]:
    """Capture hardware/build evidence without opening a package or control."""
    import numpy as np

    device = torch.cuda.get_device_properties(0) if torch.cuda.is_available() else None
    cpuinfo = Path("/proc/cpuinfo")
    fields: dict[str, str] = {}
    if cpuinfo.is_file():
        for line in cpuinfo.read_text().splitlines():
            key, sep, value = line.partition(":")
            if sep and key.strip() in {"model name", "flags", "Features", "Hardware"}:
                fields.setdefault(key.strip(), value.strip())
    packages = {}
    for name in ("torch", "numpy", "numba", "llvmlite", "scipy"):
        try:
            dist = importlib.metadata.distribution(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
            continue
        metadata_hashes = {}
        for item in dist.files or ():
            if item.name in {
                "METADATA",
                "WHEEL",
                "RECORD",
                "direct_url.json",
            } and ".dist-info" in str(item):
                metadata_hashes[str(item)] = digest(Path(dist.locate_file(item)))
        packages[name] = {"version": dist.version, "installed_metadata": metadata_hashes}
    numpy_build = io.StringIO()
    with contextlib.redirect_stdout(numpy_build):
        np.__config__.show()
    gpu_driver = None
    if device is not None:
        gpu_driver = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid,driver_version,compute_cap,memory.total",
                "--format=csv,noheader",
            ],
            text=True,
            timeout=15,
        ).strip()
    installed = sorted(
        (dist.metadata["Name"], dist.version) for dist in importlib.metadata.distributions()
    )
    environment_folder = Path(__file__).parent / "f5_adoption_environment_20261026_v6"
    lock = environment_folder / "requirements-f5-l4-cu126.lock"
    environment_json = environment_folder / "f5-common-environment-v6.json"
    lock_sha = digest(lock)
    if lock_sha != "2577ed9dd9fe34468f5562262def7bccb352c9c0bdce06b293bb4b12a0ea5301":
        raise F5FitRefusal("F5 environment lock differs")
    if (
        digest(environment_json)
        != "bc05167c188680478663235f2b3dcfb70d7959e9d392071384409ef958b0f0f5"
    ):
        raise F5FitRefusal("F5 common environment descriptor differs")
    return {
        "platform": platform.platform(),
        "python": sys.version,
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": device.name if device else None,
        "gpu_uuid": str(device.uuid) if device else None,
        "gpu_capability": list(torch.cuda.get_device_capability(0)) if device else None,
        "gpu_memory": device.total_memory if device else None,
        "numpy": np.__version__,
        "os_kernel": platform.uname()._asdict(),
        "python_build": platform.python_build(),
        "python_compiler": platform.python_compiler(),
        "python_cache_tag": sys.implementation.cache_tag,
        "cpu_isa": fields,
        "allocated_cpu_affinity": (
            sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
        ),
        "gpu_driver_query": gpu_driver,
        "cudnn": torch.backends.cudnn.version(),
        "torch_git_version": torch.version.git_version,
        "torch_build": torch.__config__.show(),
        "numpy_blas_lapack_build": numpy_build.getvalue(),
        "installed_numeric_packages": packages,
        "installed_versions_sha256": _sha256(installed),
        "environment_lock_sha256": lock_sha,
        "environment_descriptor_sha256": digest(environment_json),
        "torch_wheel_install_binding": {
            "url": (
                "https://download-r2.pytorch.org/whl/cu126/"
                "torch-2.9.1%2Bcu126-cp312-cp312-manylinux_2_28_x86_64.whl"
            ),
            "sha256": "67e9b1054f435d33af6fa67343f93d73dc2d37013623672d6ffb24ce39b666c2",
            "verification": (
                "hash-locked installation; installed METADATA/WHEEL/RECORD " "captured separately"
            ),
        },
        "torch_threads": {
            "intra": torch.get_num_threads(),
            "inter": torch.get_num_interop_threads(),
        },
        "numeric_controls": {
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cudnn_deterministic": torch.backends.cudnn.deterministic,
            "cudnn_benchmark": torch.backends.cudnn.benchmark,
            "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
            "cudnn_tf32": torch.backends.cudnn.allow_tf32,
            "default_dtype": str(torch.get_default_dtype()),
            "amp": False,
            "bf16": False,
        },
        "thread_env": {
            name: os.environ.get(name)
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


def _bound_inputs(inputs: Path, control: Mapping[str, Any], *, names: set[str]) -> None:
    hashes = control["input_hashes"]
    if set(hashes) != names:
        raise ValueError("F5 input inventory differs from the sealed runtime contract")
    for name, expected in hashes.items():
        path = inputs / name
        if Path(name).name != name or not isinstance(expected, str) or digest(path) != expected:
            raise ValueError("F5 input hash mismatch")


def _environment_admission(control: Mapping[str, Any] | None) -> dict[str, Any]:
    """Set and attest the prospective common L4 fp32 environment before any work."""
    if sys.platform != "linux" or sys.version_info[:3] != (3, 12, 3):
        raise F5FitRefusal("F5 requires the pinned Linux CPython 3.12.3 L4 environment")
    if (
        torch.__version__ != L4_TORCH
        or torch.version.cuda != L4_CUDA
        or not torch.cuda.is_available()
    ):
        raise F5FitRefusal("F5 requires torch 2.9.1+cu126 with CUDA 12.6")
    device = torch.cuda.get_device_properties(0)
    available = (
        len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 0)
    )
    if "L4" not in device.name or available < 6 or torch.cuda.device_count() != 1:
        raise F5FitRefusal("F5 requires one L4 and at least six vCPUs")
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        raise F5FitRefusal("F5 requires CUBLAS_WORKSPACE_CONFIG=:4096:8")
    for name in (
        "OMP_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(name) != "1":
            raise F5FitRefusal(f"F5 requires {name}=1")
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    capture = json.loads(json.dumps(capture_environment(), sort_keys=True))
    if control is None:
        return capture
    expected = control.get("environment_capture")
    if (
        not isinstance(expected, dict)
        or set(expected) != set(capture)
        or any(expected[key] != capture[key] for key in capture)
    ):
        raise F5FitRefusal("F5 actual environment capture differs from sealed control")
    return capture


def _checkpoint_blob(net: Any, *, arm: str, plan: BatchPlan) -> dict[str, Any]:
    """Preserve the normal Ego2s checkpoint contract for reload and serving tools."""
    return {
        "obs_spec": spec_for(net),
        "arch": ARCH,
        "config": dict(net.config),
        "state_dict": {key: value.detach().cpu() for key, value in net.state_dict().items()},
        "meta": {
            "f5_arm": arm,
            "batch_plan_sha256": plan.sha256,
            "identity": asdict(plan.identity),
        },
    }


def _payload_metadata(
    inputs: Path, control: Mapping[str, Any]
) -> tuple[Mapping[str, torch.Tensor], dict[str, np.ndarray]]:
    """Open only the compact Mac-verified selected training package on L4."""
    table, metadata, receipt = read_selected_package(inputs / "selected-training.tar.xz")
    if (
        receipt.get("corpus_sha256") != _identity(control).corpus_sha256
        or selected_corpus_identity(metadata, receipt["selected_payload_sha256"])
        != _identity(control).corpus_sha256
    ):
        raise F5FitRefusal("F5 corpus identity differs from compact package projection")
    life_order = {life: index for index, life in enumerate(sorted(set(table["life_id"].tolist())))}
    return {
        "status": torch.as_tensor(table["status"] == "EXACT"),
        "legal": torch.as_tensor(table["legal"]),
        "action_mask": torch.as_tensor(table["action_mask"]),
        "died": torch.as_tensor(table["died"]),
        "ordinary": torch.as_tensor(table["ordinary"]),
        "life_id": torch.as_tensor(
            [life_order[life] for life in table["life_id"]], dtype=torch.int64
        ),
    }, table


def _load_plan(path: Path) -> BatchPlan:
    verify_export(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, BatchPlan):
        raise ValueError("foreign F5 batch plan")
    return value


def _acceptance(path: Path, control: Mapping[str, Any], plan: BatchPlan) -> dict[str, Any]:
    """Require a real independent acceptance, never an implementer placeholder."""
    verify_export(path)
    value = _json(path)
    expected = {
        "decision": "ACCEPT_F5_FIT",
        "external": True,
        "source_sha256": plan.identity.source_sha256,
        "loss_sha256": plan.identity.loss_sha256,
        "environment_sha256": plan.identity.environment_sha256,
        "allocation_sha256": plan.identity.allocation_sha256,
        "corpus_sha256": plan.identity.corpus_sha256,
        "batch_plan_sha256": plan.sha256,
        "control_sha256": None,
    }
    # The accepting reviewer has to name its own identity and bind the package,
    # but this runtime must not manufacture any of those fields.
    required = set(expected) | {
        "reviewer",
        "reviewer_reference_sha256",
        "one_difference_sha256",
        "timing_sha256",
        "acceptance_sha256",
    }
    if (
        set(value) != required
        or not isinstance(value.get("reviewer"), str)
        or not value["reviewer"]
    ):
        raise F5FitRefusal("independent F5 acceptance receipt required")
    expected["control_sha256"] = _sha256(dict(control))
    if any(value[key] != expected[key] for key in expected):
        raise F5FitRefusal("F5 acceptance does not bind this source/environment/plan")
    for name in ("reviewer_reference_sha256", "one_difference_sha256", "timing_sha256"):
        if not isinstance(value[name], str) or len(value[name]) != 64:
            raise F5FitRefusal("F5 acceptance lacks independent review pins")
    if value["acceptance_sha256"] != _sha256(
        {key: value[key] for key in value if key != "acceptance_sha256"}
    ):
        raise F5FitRefusal("F5 acceptance self-digest mismatch")
    return value


def prepare(inputs: Path, output: Path) -> dict[str, Any]:
    """Materialize the common plan from selected training rows without fitting."""
    inputs, output = inputs.resolve(), output.resolve()
    control = _control(inputs)
    _source_admission(control)
    _environment_admission(control)
    _bound_inputs(
        inputs,
        control,
        names={"selected-training.tar.xz"},
    )
    if output.exists():
        raise FileExistsError("F5 plan output must be fresh")
    metadata, _ = _payload_metadata(inputs, control)
    plan = materialize_batch_plan(metadata, _identity(control))
    output.mkdir(parents=True)
    plan_receipt = export_torch(output / PLAN_NAME, plan)
    receipt = {
        "schema": SCHEMA,
        "kind": "PREPARED_COMMON_BATCH_PLAN",
        "namespace": NAMESPACE,
        "allocation": control["allocation"],
        "compact_package_sha256": digest(inputs / "selected-training.tar.xz"),
        "batch_plan_sha256": plan.sha256,
        "plan_export_sha256": plan_receipt["sha256"],
        "rows": 512000,
    }
    export_json(output / "prepare-receipt.json", receipt)
    return receipt


def _state_payload(state: F5FitState) -> dict[str, Any]:
    """Keep the complete residual, optimizer and all captured RNG states losslessly."""
    return {
        "model_residual": state.model.residual.state_dict(),
        "optimizer_state": state.optimizer_state,
        "updates_completed": state.updates_completed,
        "losses": state.losses,
        "identity": asdict(state.identity),
        "recipe": asdict(state.recipe),
        "plan_sha256": state.plan_sha256,
        "python_rng_state": state.python_rng_state,
        "numpy_rng_state": state.numpy_rng_state,
        "torch_cpu_rng_state": state.torch_cpu_rng_state,
        "torch_cuda_rng_states": state.torch_cuda_rng_states,
        "plan_generator_state": state.plan_generator_state,
    }


def _resume_state(path: Path, base: Any) -> F5FitState:
    """Rehydrate a lossless boundary snapshot against the already sealed B8 base."""
    verify_export(path)
    value = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(value, dict) or not {
        "model_residual",
        "optimizer_state",
        "updates_completed",
        "losses",
        "identity",
        "recipe",
        "plan_sha256",
        "python_rng_state",
        "numpy_rng_state",
        "torch_cpu_rng_state",
        "torch_cuda_rng_states",
        "plan_generator_state",
    } <= set(value):
        raise F5FitRefusal("malformed F5 resume snapshot")
    model = FrozenB8Residual(base).to("cuda:0")
    model.residual.load_state_dict(value["model_residual"])
    return F5FitState(
        model=model,
        optimizer_state=value["optimizer_state"],
        updates_completed=value["updates_completed"],
        losses=tuple(value["losses"]),
        identity=F5Identity(**value["identity"]),
        recipe=F5Recipe(**value["recipe"]),
        plan_sha256=value["plan_sha256"],
        python_rng_state=value["python_rng_state"],
        numpy_rng_state=value["numpy_rng_state"],
        torch_cpu_rng_state=value["torch_cpu_rng_state"],
        torch_cuda_rng_states=tuple(value["torch_cuda_rng_states"]),
        plan_generator_state=value["plan_generator_state"],
    )


def _verified_pair(ledger_path: Path, control: Mapping[str, Any]) -> dict[str, Any] | None:
    """Skip exactly a Mac-verified completed C/T unit without loading its payloads."""
    if digest(ledger_path) != control.get("ledger_sha256"):
        raise F5FitRefusal("F5 verified ledger differs from the sealed control")
    value = _json(ledger_path)
    if value.get("schema") != "bl-f5-verified-ledger/v1" or value.get("namespace") != NAMESPACE:
        raise F5FitRefusal("foreign F5 verified ledger")
    if value.get("allocation") != control["allocation"] or not isinstance(value.get("units"), list):
        raise F5FitRefusal("F5 ledger allocation differs")
    if len(value["units"]) > 1:
        raise F5FitRefusal("duplicate F5 paired-fit units")
    if not value["units"]:
        return None
    unit = value["units"][0]
    if (
        not isinstance(unit, dict)
        or unit.get("unit_id") != f"{NAMESPACE}/{control['allocation']}/0"
        or unit.get("kind") != "COMPLETE_PAIRED_FIT_UNIT"
        or not isinstance(unit.get("receipt_sha256"), str)
        or len(unit["receipt_sha256"]) != 64
        or not isinstance(unit.get("receipt_path"), str)
        or not isinstance(unit.get("receipt_size"), int)
    ):
        raise F5FitRefusal("malformed paired-fit resume unit")
    relative = Path(unit["receipt_path"])
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise F5FitRefusal("verified paired receipt must use a safe ledger-relative path")
    receipt_path = ledger_path.parent / relative
    if receipt_path.is_symlink() or receipt_path.stat().st_size != unit["receipt_size"]:
        raise F5FitRefusal("verified paired receipt identity mismatch")
    if verify_export(receipt_path) != unit["receipt_sha256"]:
        raise F5FitRefusal("verified paired receipt digest mismatch")
    receipt = _json(receipt_path)
    if (
        receipt.get("kind") != "COMPLETE_PAIRED_FIT_UNIT"
        or receipt.get("allocation") != control["allocation"]
        or receipt.get("namespace") != NAMESPACE
        or receipt.get("identity") != asdict(_identity(control))
        or set(receipt.get("arms", {})) != {"C", "W"}
        or any(receipt["arms"][arm].get("updates_completed") != 2000 for arm in ("C", "W"))
    ):
        raise F5FitRefusal("verified paired receipt is foreign")
    return unit


def fit_pair(
    inputs: Path,
    plan_path: Path,
    acceptance_path: Path,
    ledger_path: Path,
    output: Path,
    resume_c: Path | None = None,
    resume_w: Path | None = None,
) -> dict[str, Any]:
    """Run C then W in one process and publish one receipt only after both complete."""
    inputs, plan_path, acceptance_path, ledger_path, output = (
        path.resolve() for path in (inputs, plan_path, acceptance_path, ledger_path, output)
    )
    control = _control(inputs)
    _source_admission(control)
    _environment_admission(control)
    _bound_inputs(
        inputs,
        control,
        names={
            "b8.pth",
            "selected-training.tar.xz",
        },
    )
    if digest(inputs / "b8.pth") != B8_SHA256:
        raise F5FitRefusal("F5 requires the fixed B8 input")
    verified = _verified_pair(ledger_path, control)
    if verified is not None:
        return {"schema": SCHEMA, "kind": "SKIPPED_VERIFIED_PAIRED_FIT_UNIT", "unit": verified}
    if output.exists():
        raise FileExistsError("F5 fit output must be fresh")
    metadata, table = _payload_metadata(inputs, control)
    plan = _load_plan(plan_path)
    if plan.identity != _identity(control):
        raise F5FitRefusal("F5 plan identity differs from control")
    verify_batch_plan(plan, metadata)
    acceptance = _acceptance(acceptance_path, control, plan)
    base = load_ego2s_checkpoint(str(inputs / "b8.pth"))
    model = FrozenB8Residual(base).to("cuda:0")
    cache = selected_tensor_cache(table, base)
    output.mkdir(parents=True)

    def boundary(arm_name: str):
        def save(state: F5FitState) -> None:
            export_torch(
                output / "boundaries" / arm_name / f"{state.updates_completed:04d}.pt",
                _state_payload(state),
            )

        return save

    c_prior = _resume_state(resume_c.resolve(), base) if resume_c is not None else None
    if resume_w is not None and (c_prior is None or c_prior.updates_completed != 2000):
        raise F5FitRefusal("W resume requires an exactly completed C boundary")
    c_state = fit_f5_accepted(
        model, cache, metadata, plan, F5Recipe(arm=ARM_CONTROL), acceptance, c_prior, boundary("C")
    )
    # W begins from the same zero residual, on the same plan, in the same pinned runtime.
    w_model = FrozenB8Residual(load_ego2s_checkpoint(str(inputs / "b8.pth"))).to("cuda:0")
    w_prior = _resume_state(resume_w.resolve(), base) if resume_w is not None else None
    w_state = fit_f5_accepted(
        w_model,
        cache,
        metadata,
        plan,
        F5Recipe(arm=ARM_WEIGHTED),
        acceptance,
        w_prior,
        boundary("W"),
    )
    c_state_export = export_torch(output / "f5-state-C.pt", _state_payload(c_state))
    w_state_export = export_torch(output / "f5-state-W.pt", _state_payload(w_state))
    c_checkpoint = export_torch(
        output / "fit-C.pth", _checkpoint_blob(fuse_residual(c_state.model), arm="C", plan=plan)
    )
    w_checkpoint = export_torch(
        output / "fit-W.pth", _checkpoint_blob(fuse_residual(w_state.model), arm="W", plan=plan)
    )
    receipt = {
        "schema": SCHEMA,
        "kind": "COMPLETE_PAIRED_FIT_UNIT",
        "namespace": NAMESPACE,
        "allocation": control["allocation"],
        "identity": asdict(plan.identity),
        "acceptance_sha256": digest(acceptance_path),
        "batch_plan_sha256": plan.sha256,
        "arms": {
            "C": {
                "state_sha256": c_state_export["sha256"],
                "checkpoint_sha256": c_checkpoint["sha256"],
                "updates_completed": c_state.updates_completed,
            },
            "W": {
                "state_sha256": w_state_export["sha256"],
                "checkpoint_sha256": w_checkpoint["sha256"],
                "updates_completed": w_state.updates_completed,
            },
        },
    }
    export_json(output / "fit-receipt.json", receipt)
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description="offline F5 plan/fit runtime")
    sub = parser.add_subparsers(dest="mode", required=True)
    capture_parser = sub.add_parser("capture-environment")
    capture_parser.add_argument("--output", type=Path, required=True)
    prepare_parser = sub.add_parser("prepare")
    prepare_parser.add_argument("--inputs", type=Path, required=True)
    prepare_parser.add_argument("--output", type=Path, required=True)
    fit_parser = sub.add_parser("fit-pair")
    fit_parser.add_argument("--inputs", type=Path, required=True)
    fit_parser.add_argument("--plan", type=Path, required=True)
    fit_parser.add_argument("--acceptance", type=Path, required=True)
    fit_parser.add_argument("--ledger", type=Path, required=True)
    fit_parser.add_argument("--output", type=Path, required=True)
    fit_parser.add_argument("--resume-c", type=Path)
    fit_parser.add_argument("--resume-w", type=Path)
    args = parser.parse_args()
    if args.mode == "capture-environment":
        print(json.dumps(export_json(args.output, _environment_admission(None)), sort_keys=True))
        return
    value = (
        prepare(args.inputs, args.output)
        if args.mode == "prepare"
        else fit_pair(
            args.inputs,
            args.plan,
            args.acceptance,
            args.ledger,
            args.output,
            args.resume_c,
            args.resume_w,
        )
    )
    print(json.dumps({"kind": value["kind"], "status": "COMPLETE"}, sort_keys=True))


if __name__ == "__main__":
    main()
