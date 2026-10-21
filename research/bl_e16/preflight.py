"""Tiny Mac proof of the execution harness; never a scientific efficacy look."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any

import torch

from research.bl_e16.checkpoint import restore_snapshot
from research.bl_e16.evaluate import execute_worlds
from research.bl_e16.model import E16Policy
from research.bl_e16.runner import (
    NaturalLifeCollector,
    _snapshot,
    collect_natural_batch,
    load_audited_bank,
)
from research.bl_e16.training import E16Trainer
from research.bl_ppo.exports import digest, export_json, export_torch, verify_export


def _tensor_sha(values: dict[str, torch.Tensor]) -> str:
    sha = hashlib.sha256()
    for name, value in sorted(values.items()):
        array = value.detach().cpu().contiguous().numpy()
        sha.update(name.encode())
        sha.update(str((array.shape, array.dtype.str)).encode())
        sha.update(array.tobytes())
    return sha.hexdigest()


def resume_proof(path: Path, bank: dict[str, Any]) -> dict[str, Any]:
    """Two fresh deserializations yield identical next collection AND optimizer state."""
    verify_export(path)
    results = []
    frozen_wire = None
    for attempt in range(2):
        payload = torch.load(
            path if attempt == 0 else io.BytesIO(frozen_wire),
            map_location="cpu",
            weights_only=False,
        )
        policy = E16Policy.from_portable_state(payload["policy"])
        trainer = E16Trainer(
            policy, critic_refit_transitions=1, history_qualified=policy.use_history
        )
        collectors = {phase: NaturalLifeCollector(bank, phase) for phase in payload["collector"]}
        counters = restore_snapshot(policy, trainer, collectors, payload)
        if attempt == 0:
            boundary = _snapshot(policy, trainer, counters, payload["identity"], collectors)
            wire = io.BytesIO()
            torch.save(boundary, wire)
            frozen_wire = wire.getvalue()
        batch, _, _ = collect_natural_batch(
            policy, bank, phase="preflight", cursor=0, steps=3, collector=collectors["preflight"]
        )
        before = _tensor_sha(
            {key: value for key, value in vars(batch).items() if isinstance(value, torch.Tensor)}
        )
        trainer.trainer_step(batch)
        after = _tensor_sha(policy.state_dict())
        # Include all Adam moment tensors, not only final actor weights.
        optimizer_tensors = {}
        for label, optimizer in (
            ("actor", trainer.actor_optimizer),
            ("critic", trainer.critic_optimizer),
        ):
            for parameter, state in optimizer.state_dict()["state"].items():
                for name, value in state.items():
                    if isinstance(value, torch.Tensor):
                        optimizer_tensors[f"{label}/{parameter}/{name}"] = value
        results.append((before, after, _tensor_sha(optimizer_tensors)))
    if results[0] != results[1]:
        raise RuntimeError("on-platform exact next-collection/update resume failed")
    return {
        "status": "PASS",
        "comparison": "uninterrupted boundary vs serialized restore",
        "next_transitions": 3,
        "batch_sha256": results[0][0],
        "updated_policy_sha256": results[0][1],
        "optimizer_sha256": results[0][2],
    }


def main() -> None:
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--label", default="02")
    args = parser.parse_args()
    torch.set_num_threads(1)
    bank = load_audited_bank(args.bank, args.audit)
    endpoints = {"B8": args.b8}
    resume = {}
    for arm in ("A", "G"):
        folder = args.root / f"preflight-{arm.lower()}-01"
        resume[arm] = resume_proof(folder / "full-state.pt", bank)
        endpoints[arm] = folder / "endpoint.pt"
    # Restore the embedded teacher-warm-start F parent for the tiny paired proof.
    snapshot = torch.load(
        args.root / "preflight-a-01/full-state.pt", map_location="cpu", weights_only=False
    )
    policy = E16Policy.from_portable_state(snapshot["policy"])
    policy.actor.residual_policy.weight.data.zero_()
    policy.actor.residual_policy.bias.data.zero_()
    parent = args.root / f"preflight-parent-{args.label}.pt"
    export_torch(parent, policy.actor.checkpoint_dict())
    endpoints["parent"] = parent
    worlds = bank["excluded_preflight_rows"]
    started = time.monotonic()
    serial = execute_worlds(worlds, endpoints, workers=1)
    serial_seconds = time.monotonic() - started
    started = time.monotonic()
    parallel = execute_worlds(worlds, endpoints, workers=2)
    parallel_seconds = time.monotonic() - started

    def deterministic(rows: list[dict[str, Any]]) -> str:
        stripped = []
        for row in rows:
            stripped.append(
                {
                    **row,
                    "controllers": {
                        name: {k: v for k, v in value.items() if k != "wall_seconds"}
                        for name, value in row["controllers"].items()
                    },
                }
            )
        return hashlib.sha256(
            json.dumps(stripped, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    if deterministic(serial) != deterministic(parallel):
        raise RuntimeError("serial/parallel whole-life rows differ")
    active = sum(c["active_frames"] for row in serial for c in row["controllers"].values())
    result = {
        "schema": "bl-e16-mac-preflight/v1",
        "status": "PASS",
        "scientific": False,
        "bank_sha256": digest(args.bank),
        "audit_sha256": digest(args.audit),
        "worlds": len(worlds),
        "horizon": 40,
        "controllers": 4,
        "active_world_steps": active,
        "serial_seconds": serial_seconds,
        "parallel_seconds": parallel_seconds,
        "serial_world_steps_per_second": active / serial_seconds,
        "parallel_world_steps_per_second": active / parallel_seconds,
        "serial_parallel_rows_sha256": deterministic(serial),
        "resume": resume,
        "qualification": (
            "tiny natural-prefix smoke only; " "not full-policy H5000 timing or E13 transfer"
        ),
    }
    export_json(args.root / f"preflight-{args.label}.json", result)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
