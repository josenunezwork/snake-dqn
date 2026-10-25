"""R30 training-only worker. No validation outcomes are uploaded or opened."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_bytes, export_json, export_torch, verify_export
from research.bl_residual.data import tensor_cache, validate_rows
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from research.bl_round30.protocol import (
    AMENDMENT_SHA,
    B8_SHA,
    JUDGMENT_SHA,
    NAMESPACE,
    bound_json,
    require_amendment_binding,
    require_series,
    require_successor_gate,
)
from src.model.ego2s_network import ARCH, Ego2sNet, load_ego2s_checkpoint, spec_for


def preflight_inputs(input_dir: Path, *, mode: str, commit: str | None = None) -> dict:
    """Check metadata admission before reading or hashing any training outcomes."""
    control = json.loads((input_dir / "control.json").read_text())
    if (
        control.get("schema") != "r30-control/v2"
        or control.get("stage") != mode
        or control.get("ruling_sha256") != JUDGMENT_SHA
        or control.get("amendment_sha256") != AMENDMENT_SHA
        or control.get("namespace") != NAMESPACE
        or control.get("synthetic") is not False
    ):
        raise ValueError("foreign control/ruling/namespace")
    try:
        require_amendment_binding(control)
    except ValueError as exc:
        raise ValueError("foreign control/ruling/amendment") from exc
    if commit is not None and control.get("commit") != commit:
        raise ValueError("stale source pin")
    from research.bl_round30.admission import freeze_inventory, require_codex_v_approval

    inventory = bound_json(input_dir / "inventory.json", control["inventory_sha256"])
    approval = bound_json(input_dir / "approval.json", control["approval_sha256"])
    admitted = freeze_inventory(inventory)
    require_codex_v_approval(admitted, approval)
    if admitted.allocation_sha256 != control.get("allocation"):
        raise ValueError("frozen corpus allocation differs")
    if len(admitted.validation_parents) < 32 or len(admitted.validation_rescue_parents) < 16:
        raise ValueError("INSUFFICIENT: validation support below frozen minimum")
    if inventory.get("split_salt") != "t34-r30-parent-split-20261023-v1":
        raise ValueError("frozen split salt mismatch")
    if control.get("salt") != admitted.salt:
        raise ValueError("control split salt differs from admitted corpus")
    if control.get("horizon") != 300 or control.get("mass_convention") != "full-horizon-dead-zero":
        raise ValueError("H300 full-horizon dead-zero labels required")
    if not control.get("continuation_sha256"):
        raise ValueError("frozen continuation binding required")
    manifest = bound_json(
        input_dir / "partition-manifest.json", control["partition_manifest_sha256"]
    )
    coverage = bound_json(input_dir / "coverage-approval.json", control["coverage_approval_sha256"])
    life_id_map = bound_json(input_dir / "life-id-map.json", control["life_id_map_sha256"])
    if (
        manifest.get("schema") != "r30-partition-manifest/v1"
        or manifest.get("inventory_sha256") != admitted.inventory_sha256
        or manifest.get("allocation_sha256") != admitted.allocation_sha256
        or manifest.get("life_id_map_sha256") != control["life_id_map_sha256"]
        or coverage.get("schema") != "r30-partition-coverage-approval/v1"
        or coverage.get("approved") is not True
        or coverage.get("reviewer_role") != "Codex-V"
        or coverage.get("partition_manifest_sha256") != control["partition_manifest_sha256"]
        or coverage.get("life_id_map_sha256") != control["life_id_map_sha256"]
        or life_id_map.get("schema") != "r30-global-life-map/v1"
        or life_id_map.get("inventory_sha256") != admitted.inventory_sha256
    ):
        raise ValueError("partition manifest/coverage approval/global life map differs")
    sampler_plan = bound_json(input_dir / "sampler-plan.json", control["sampler_plan_sha256"])
    if (
        sampler_plan.get("schema") != "r30-scientific-sampler-plan/v1"
        or sampler_plan.get("source_commit") != control.get("commit")
        or sampler_plan.get("train_sha256") != control.get("input_hashes", {}).get("train.npz")
        or sampler_plan.get("plan_sha256") != control.get("sampler_plan_bytes_sha256")
        or sampler_plan.get("fit_seed") != 3403
        or sampler_plan.get("audit_seed") != 3404
        or sampler_plan.get("batches_per_plan") != 2000
        or sampler_plan.get("features_constructed") is not False
        or sampler_plan.get("fit_performed") is not False
    ):
        raise ValueError("frozen scientific sampler metadata differs")
    allowed = {
        "b8.pth",
        "train.npz",
        "inventory.json",
        "approval.json",
        "partition-manifest.json",
        "coverage-approval.json",
        "life-id-map.json",
        "sampler-plan.npz",
        "sampler-plan.json",
        "successor-gate.json",
        "diagnosis.json",
    }
    if (
        not {
            "b8.pth",
            "train.npz",
            "sampler-plan.npz",
            "sampler-plan.json",
        }
        <= control.get("input_hashes", {}).keys()
    ):
        raise ValueError("bound training partition, sampler plan, and B8 required")
    if not set(control["input_hashes"]) <= allowed:
        raise ValueError("foreign input: validation/archive outcomes must stay on Mac")
    if mode == "r30-2":
        if admitted.r30_2_status != "READY_R30_2":
            raise ValueError("HOLD: natural ordinary archive/support not admitted")
        gate = bound_json(input_dir / "successor-gate.json", control["successor_gate_sha256"])
        diagnosis = bound_json(input_dir / "diagnosis.json", gate["diagnosis_sha256"])
        require_successor_gate(gate, control["allocation"], diagnosis)
    # Metadata guards above intentionally precede even a checksum of label bytes.
    for name, sha in control.get("input_hashes", {}).items():
        if Path(name).name != name or digest(input_dir / name) != sha:
            raise ValueError("bound input hash mismatch")
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 identity differs")
    return control


def _load_training(input_dir: Path, control: dict, updates: int = 2000) -> dict[str, np.ndarray]:
    """Only the admitted train file is permitted at this boundary."""
    if "train.npz" not in control["input_hashes"]:
        raise ValueError("training partition is not bound")
    with np.load(input_dir / "train.npz", allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    validate_rows(table)
    from research.bl_round30.analyze import require_global_life_identity

    require_global_life_identity(table, json.loads((input_dir / "life-id-map.json").read_text()))
    from research.bl_round30.admission import split_parent

    if any(
        split_parent(str(parent), control["salt"]) != "train" for parent in table["parent_world"]
    ):
        raise ValueError("INVALID: validation ancestry in training partition")
    inventory = json.loads((input_dir / "inventory.json").read_text())
    from research.bl_round30.admission import freeze_inventory

    catalog = set(freeze_inventory(inventory).train_parents)
    if not set(table["parent_world"].tolist()) <= catalog:
        raise ValueError("foreign parent in training partition")
    manifest = json.loads((input_dir / "partition-manifest.json").read_text())
    expected_sha = manifest.get("partitions", {}).get("train", {}).get("sha256")
    if expected_sha != digest(input_dir / "train.npz"):
        raise ValueError("training partition outside reuse authority")
    source_life_ids = table.get("source_life_id")
    global_life_indices = table.get("global_life_index")
    if source_life_ids is None or global_life_indices is None:
        raise ValueError("globally unique source life IDs and integer map are required")
    source_to_life: dict[object, tuple[object, object, object]] = {}
    for source_life, global_index, parent, life in zip(
        source_life_ids.tolist(),
        global_life_indices.tolist(),
        table["parent_world"].tolist(),
        table["life_id"].tolist(),
    ):
        if source_life != life or not isinstance(global_index, (int, np.integer)):
            raise ValueError("global life identity/map schema differs from materialization")
        identity = (parent, life, int(global_index))
        if source_life in source_to_life and source_to_life[source_life] != identity:
            raise ValueError("global source life ID maps to multiple parent/life identities")
        source_to_life[source_life] = identity
    if len(source_to_life) != len({(parent, life) for parent, life, _ in source_to_life.values()}):
        raise ValueError("tranche-local parent/life identity collision")
    if len(source_to_life) != len({index for _, _, index in source_to_life.values()}):
        raise ValueError("global life integer map collision")
    from research.bl_round30.batch_plan import verify_plan_bytes

    # Identity validation precedes the runtime sampler reproduction.  The worker
    # consumes the exact G1-frozen list prefix for a native three-update canary
    # and the entire list for a scientific fit.
    verify_plan_bytes(input_dir / "sampler-plan.npz", table, updates=updates)
    return table


def _checkpoint(base: Any, parameters: torch.Tensor, state: dict) -> dict:
    model = FrozenB8Residual(base)
    with torch.no_grad():
        model.residual.weight.copy_(parameters[:1536].reshape(6, 256).float().cpu())
        model.residual.bias.copy_(parameters[1536:].float().cpu())
    fused = fuse_residual(model)
    return {
        "obs_spec": spec_for(fused),
        "arch": ARCH,
        "config": fused.config,
        "state_dict": fused.state_dict(),
        "fit_state": state,
        "parameters": parameters.detach().cpu(),
        "execution_dtype": "float32",
    }


def _table_for_mode(input_dir: Path, control: dict, *, synthetic: bool, base: Any) -> dict:
    """Keep an exact-input canary on real rows while retaining its short fit."""
    if synthetic and control.get("synthetic") is True:
        from research.bl_round30.fixture import synthetic_table

        return synthetic_table(base)
    return _load_training(input_dir, control, 3 if synthetic else 2000)


def reloaded_canary_actions(endpoint: dict, table: dict, device: str) -> list[int]:
    """Execute a real fp32 fused-policy decision, attributable to this package.

    The qualified CUDA observer records the actual six-action argmax. This is
    synthetic consumer evidence only; Mac rowwise decisions remain authoritative.
    """
    model = Ego2sNet(**endpoint["config"])
    model.load_state_dict(endpoint["state_dict"], strict=True)
    model = model.to(device).eval()
    mask = torch.as_tensor(table["action_mask"][:1], dtype=torch.bool, device=device)
    fallback = torch.zeros_like(mask)
    fallback[:, :3] = True
    mask = torch.where(mask.any(dim=1, keepdim=True), mask, fallback)
    with torch.no_grad():
        q = model(
            *[
                torch.as_tensor(table[key][:1], device=device)
                for key in ("local", "global", "scalars")
            ]
        )
        if q.shape != (1, 6) or not torch.isfinite(q).all():
            raise ValueError("nonfinite/foreign canary fused Q output")
        actions = q.masked_fill(~mask, -torch.inf).argmax(dim=1)
    return actions.cpu().tolist()


def run(input_dir: Path, output: Path, *, mode: str, device: str = "cpu") -> dict:
    """Execute one indivisible deterministic fit unit, skipping Mac-verified units."""
    torch.set_num_threads(1)
    if torch.get_num_interop_threads() != 1:
        torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    synthetic = mode in ("smoke", "canary")
    control = json.loads((input_dir / "control.json").read_text())
    if synthetic:
        if control.get("synthetic") is False and mode == "canary":
            preflight_inputs(input_dir, mode=control["stage"])
        elif (
            control.get("synthetic") is not True
            or control.get("ruling_sha256") != JUDGMENT_SHA
            or control.get("namespace") != NAMESPACE + "-synthetic"
        ):
            raise ValueError("foreign synthetic control")
    else:
        control = preflight_inputs(input_dir, mode=mode)
        require_series(json.loads((input_dir / "series.json").read_text()), mode)
    if digest(input_dir / "b8.pth") != B8_SHA:
        raise ValueError("B8 binding mismatch")
    allocation = digest(input_dir / "control.json")
    ledger = json.loads((input_dir / "verified-ledger.json").read_text())
    if (
        ledger.get("schema") != "r30-job-ledger/v1"
        or ledger.get("namespace") != control["namespace"]
        or ledger.get("allocation") != allocation
        or not isinstance(ledger.get("units"), list)
    ):
        raise ValueError("foreign ledger")
    skipped = False
    if ledger["units"] and not (mode == "canary" and control.get("synthetic") is False):
        if len(ledger["units"]) != 1:
            raise ValueError("duplicate/unallocated completed units")
        unit = ledger["units"][0]
        if (
            unit.get("unit_id") != f"{control['namespace']}/{allocation}/0"
            or unit.get("index") != 0
            or unit.get("name") != "unit-0000.pth"
            or type(unit.get("size")) is not int
            or unit["size"] <= 0
            or len(unit.get("sha256", "")) != 64
        ):
            raise ValueError("foreign ledger unit")
        # Require uploaded Mac-verified immutable payload to substantiate the skip.
        previous = input_dir / "unit-0000.pth"
        identity = bound_json(input_dir / "allocation.json", unit.get("identity_sha256", ""))
        if (
            identity.get("synthetic") != synthetic
            or identity.get("stage") != (control["stage"] if not synthetic else mode)
            or identity.get("allocation") != allocation
            or identity.get("namespace") != control["namespace"]
        ):
            raise ValueError("foreign/canary skip identity")
        if verify_export(previous) != unit["sha256"] or previous.stat().st_size != unit["size"]:
            raise ValueError("unverified skip")
        skipped = True
    # The supervisor creates this empty directory before starting the worker.
    # Accept that layout, but never follow a symlink or reuse attempt evidence.
    if output.is_symlink():
        raise FileExistsError(f"unsafe attempt output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise FileExistsError(f"attempt output directory is not empty: {output}")
    export_json(
        output / "allocation.json",
        {
            "schema": "r30-unit-identity/v1",
            "namespace": control["namespace"],
            "allocation": allocation,
            "source_commit": control["commit"],
            "index": 0,
            "synthetic": synthetic,
            "stage": mode,
        },
    )
    started = time.monotonic()
    stop = threading.Event()

    def pulse() -> None:
        while not stop.wait(110):
            print(
                json.dumps(
                    {
                        "done": 0,
                        "total": 1,
                        "stage": mode,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                ),
                flush=True,
            )

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    try:
        if skipped:
            export_bytes(output / "unit-0000.pth", previous.read_bytes())
        else:
            from research.bl_round30.objective import execute_packet

            base = load_ego2s_checkpoint(str(input_dir / "b8.pth"))
            table = _table_for_mode(input_dir, control, synthetic=synthetic, base=base)
            cache_started = time.monotonic()
            cache = tensor_cache(table, base.to(device), batch_size=1)
            feature_cache_seconds = time.monotonic() - cache_started
            # Real-valued solve uses the L4 host CPU; Adam can use CUDA.
            gate = (
                json.loads((input_dir / "successor-gate.json").read_text())
                if mode == "r30-2"
                else {}
            )
            fit_started = time.monotonic()
            result = execute_packet(
                cache, synthetic=synthetic, successor=mode == "r30-2", gate=gate, device=device
            )
            fit_and_certificate_seconds = time.monotonic() - fit_started
            sampler_packet: dict[str, Any] | None = None
            if control.get("synthetic") is False:
                row_ids = table.get("row_id")
                source_life_ids = table.get("source_life_id")
                if row_ids is None or source_life_ids is None:
                    raise ValueError("real packet lacks immutable row/source-life identities")
                fit_indices = result["batch_indices"]
                audit_indices = result["audit_indices"]
                sampler_packet = {
                    "source_commit": control["commit"],
                    "train_sha256": control["input_hashes"]["train.npz"],
                    "row_ids_by_index": row_ids.tolist(),
                    "source_life_ids_by_index": source_life_ids.tolist(),
                    "fit_indices": fit_indices,
                    "audit_indices": audit_indices,
                    "fit_multiplicities": dict(
                        Counter(index for batch in fit_indices for index in batch)
                    ),
                    "audit_multiplicities": dict(Counter(audit_indices)),
                    "cache_sha256": result["convergence_bindings"]
                    .get("adam", {})
                    .get("cache_sha256"),
                }
            fuse_started = time.monotonic()
            endpoints = {
                name: _checkpoint(base.cpu(), endpoint["parameters"], endpoint["state"])
                for name, endpoint in result.pop("endpoints").items()
            }
            fused_checkpoint_seconds = time.monotonic() - fuse_started
            if mode == "r30-2":
                from research.bl_round30.analyze import endpoint_sha256

                if endpoint_sha256(endpoints["control"]) != gate["control_endpoint_sha256"]:
                    raise ValueError("R30-2 refit fused control differs from diagnosed endpoint")
            unit = {
                "schema": "r30-fit-unit/v1",
                "namespace": control["namespace"],
                "allocation": allocation,
                "index": 0,
                "source_commit": control["commit"],
                "ruling_sha256": JUDGMENT_SHA,
                "amendment_sha256": AMENDMENT_SHA,
                "synthetic": synthetic,
                "stage": mode,
                "endpoints": endpoints,
                "diagnostics": result,
                "heldout_outcomes_opened": False,
                "stage_timings_seconds": {
                    "feature_cache": feature_cache_seconds,
                    "fit_and_certificate": fit_and_certificate_seconds,
                    "fused_checkpoint": fused_checkpoint_seconds,
                },
                "sampler_packet": sampler_packet,
            }
            export_torch(output / "unit-0000.pth", unit, max_payload_bytes=120 * 1024**2)
        canary_actions = None
        if synthetic and not skipped:
            # Reload the actual immutable export before choosing on its target device.
            verify_export(output / "unit-0000.pth")
            saved = torch.load(output / "unit-0000.pth", map_location="cpu", weights_only=False)
            canary_actions = reloaded_canary_actions(saved["endpoints"]["adam"], table, device)
        receipt = {
            "schema": "r30-worker-receipt/v1",
            "synthetic_fused_action_device": device if canary_actions is not None else None,
            "synthetic_fused_actions": canary_actions,
            "namespace": control["namespace"],
            "allocation": allocation,
            "source_commit": control["commit"],
            "synthetic": synthetic,
            "done": 1,
            "skipped": [0] if skipped else [],
            "elapsed_seconds": time.monotonic() - started,
            "execution_nice": os.getpriority(os.PRIO_PROCESS, 0),
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
            "heldout_outcomes_opened": False,
            "scientific_result": False,
        }
        export_json(output / "receipt.json", receipt)
        return receipt
    finally:
        stop.set()
        heartbeat.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "canary", "r30-1", "r30-2"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.input_dir, args.output, mode=args.mode, device=args.device)))


if __name__ == "__main__":
    main()
