"""R30 training-only worker. No validation outcomes are uploaded or opened."""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.bl_ppo.exports import digest, export_bytes, export_json, export_torch, verify_export
from research.bl_residual.data import tensor_cache, validate_rows
from research.bl_residual.model import FrozenB8Residual, fuse_residual
from research.bl_round30.protocol import (
    B8_SHA,
    JUDGMENT_SHA,
    NAMESPACE,
    bound_json,
    require_series,
    require_successor_gate,
)
from src.model.ego2s_network import ARCH, Ego2sNet, load_ego2s_checkpoint, spec_for


def preflight_inputs(input_dir: Path, *, mode: str, commit: str | None = None) -> dict:
    """Check metadata admission before reading or hashing any training outcomes."""
    control = json.loads((input_dir / "control.json").read_text())
    if (
        control.get("schema") != "r30-control/v1"
        or control.get("stage") != mode
        or control.get("ruling_sha256") != JUDGMENT_SHA
        or control.get("namespace") != NAMESPACE
        or control.get("synthetic") is not False
    ):
        raise ValueError("foreign control/ruling/namespace")
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
    allowed = {
        "b8.pth",
        "train.npz",
        "inventory.json",
        "approval.json",
        "successor-gate.json",
        "diagnosis.json",
    }
    if not {"b8.pth", "train.npz"} <= control.get("input_hashes", {}).keys():
        raise ValueError("bound training partition and B8 required")
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


def _load_training(input_dir: Path, control: dict) -> dict[str, np.ndarray]:
    """Only the admitted train file is permitted at this boundary."""
    if "train.npz" not in control["input_hashes"]:
        raise ValueError("training partition is not bound")
    with np.load(input_dir / "train.npz", allow_pickle=False) as archive:
        table = {key: archive[key] for key in archive.files}
    validate_rows(table)
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
    expected_sha = inventory.get("partition_sha256", {}).get("train")
    if expected_sha != digest(input_dir / "train.npz"):
        raise ValueError("training partition outside reuse authority")
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
    if output.exists():
        raise FileExistsError("fresh output required")
    output.mkdir(parents=True)
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
            if synthetic:
                from research.bl_residual.job import synthetic_table

                table = synthetic_table(base)
            else:
                table = _load_training(input_dir, control)
            cache = tensor_cache(table, base.to(device), batch_size=1)
            # Real-valued solve uses the L4 host CPU; Adam can use CUDA.
            gate = (
                json.loads((input_dir / "successor-gate.json").read_text())
                if mode == "r30-2"
                else {}
            )
            result = execute_packet(
                cache, synthetic=synthetic, successor=mode == "r30-2", gate=gate, device=device
            )
            endpoints = {
                name: _checkpoint(base.cpu(), endpoint["parameters"], endpoint["state"])
                for name, endpoint in result.pop("endpoints").items()
            }
            unit = {
                "schema": "r30-fit-unit/v1",
                "namespace": control["namespace"],
                "allocation": allocation,
                "index": 0,
                "source_commit": control["commit"],
                "ruling_sha256": JUDGMENT_SHA,
                "synthetic": synthetic,
                "stage": mode,
                "endpoints": endpoints,
                "diagnostics": result,
                "heldout_outcomes_opened": False,
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
