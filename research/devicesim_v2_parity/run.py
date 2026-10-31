"""CLI runner for Watch6 mechanics-v2 parity across DeviceSim, CTDeviceSim, and GridBatchSim."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import torch

from research.devicesim_v2_parity.harness import derive_world_seeds, run_v2_parity_rollout
from research.devicesim_v2_parity.units import (
    SCHEMA_LEDGER,
    SCHEMA_UNIT,
    UNIT_KIND,
    aggregate_verified_units,
    allocation_for_lane,
    load_verified_ledger,
    namespace_for,
    publish_immutable_json,
    specs_for_lane,
    validate_ledger_document,
    verify_published_json,
)


def _emit(event: str, **payload: Any) -> None:
    record = {
        "event": event,
        "ts_epoch": round(time.time(), 3),
        **payload,
    }
    print(json.dumps(record, sort_keys=True), flush=True)


def run_parity_job(
    *,
    commit: str,
    lane: str,
    device: str,
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    ledger_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    if not b8_checkpoint.is_file():
        raise FileNotFoundError(f"missing B8 checkpoint: {b8_checkpoint}")
    if not frp3_checkpoint.is_file():
        raise FileNotFoundError(f"missing FRP3 checkpoint: {frp3_checkpoint}")
    if not ledger_path.is_file():
        raise FileNotFoundError(f"missing verified ledger: {ledger_path}")

    target_device = torch.device(device)
    if target_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested for v2 parity job, but CUDA is unavailable")

    namespace = namespace_for(commit, lane)
    specs = specs_for_lane(lane)
    allocation = allocation_for_lane(lane)
    verified_in = load_verified_ledger(
        ledger_path,
        expected_namespace=namespace,
        expected_lane=lane,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    shards_dir = output_dir / "shards"
    shards_dir.mkdir(parents=True, exist_ok=True)

    _emit(
        "start",
        namespace=namespace,
        lane=lane,
        device=str(target_device),
        gpu_name=(
            torch.cuda.get_device_name(target_device) if target_device.type == "cuda" else "cpu"
        ),
        allocation=list(allocation),
        pre_verified=sorted(verified_in),
    )

    completed_docs: list[dict[str, Any]] = []
    merged_entries: dict[str, dict[str, Any]] = dict(verified_in)

    for spec in specs:
        if spec.unit_id in verified_in:
            _emit("skip_verified_unit", unit_id=spec.unit_id)
            continue

        seeds = derive_world_seeds(spec.shard_index, spec.num_envs)
        t0 = time.monotonic()
        last_beat = t0

        def _on_tick(done_ticks: int, total_ticks: int, uid: str = spec.unit_id) -> None:
            nonlocal last_beat
            now = time.monotonic()
            if (now - last_beat) >= 30.0 or done_ticks == total_ticks or (done_ticks % 250 == 0):
                last_beat = now
                _emit(
                    "shard_progress",
                    unit_id=uid,
                    done_ticks=done_ticks,
                    total_ticks=total_ticks,
                    elapsed_s=round(now - t0, 2),
                )

        _emit(
            "shard_start",
            unit_id=spec.unit_id,
            num_envs=spec.num_envs,
            ticks=spec.ticks,
            seed_first=seeds[0],
            seed_last=seeds[-1],
        )
        report = run_v2_parity_rollout(
            seeds=seeds,
            ticks=spec.ticks,
            device=target_device,
            b8_checkpoint=b8_checkpoint,
            frp3_checkpoint=frp3_checkpoint,
            check_ct_sim=spec.check_ct_sim,
            ego_check_stride=spec.ego_check_stride,
            mature_fraction=spec.mature_fraction,
            max_capacity=spec.max_capacity,
            progress_cb=_on_tick,
        )
        elapsed_s = round(time.monotonic() - t0, 4)
        unit_doc = {
            "schema": SCHEMA_UNIT,
            "namespace": namespace,
            "commit": commit,
            "lane": lane,
            "unit_id": spec.unit_id,
            "shard_index": spec.shard_index,
            "seeds": seeds,
            "elapsed_seconds": elapsed_s,
            "report": report,
        }
        shard_path = shards_dir / f"{spec.unit_id}.json"
        relpath = f"shards/{spec.unit_id}.json"
        publish_immutable_json(
            shard_path,
            unit_doc,
            unit_id=spec.unit_id,
            namespace=namespace,
            relpath=relpath,
        )
        entry, verified_doc = verify_published_json(
            shard_path,
            expected_unit_id=spec.unit_id,
            expected_namespace=namespace,
            expected_relpath=relpath,
        )
        merged_entries[spec.unit_id] = entry
        completed_docs.append(verified_doc)
        _emit(
            "shard_complete",
            unit_id=spec.unit_id,
            verdict=report["verdict"],
            unexplained_mismatches=report["unexplained_mismatches"],
            tie_explained_decisions=report["tie_explained_decisions"],
            elapsed_s=elapsed_s,
        )

    ordered_ledger_entries = [merged_entries[uid] for uid in allocation if uid in merged_entries]
    out_ledger_doc = {
        "schema": SCHEMA_LEDGER,
        "namespace": namespace,
        "lane": lane,
        "unit_kind": UNIT_KIND,
        "allocation": list(allocation),
        "skip_verified": True,
        "verified_units": ordered_ledger_entries,
    }
    validate_ledger_document(
        out_ledger_doc,
        expected_namespace=namespace,
        expected_lane=lane,
    )
    publish_immutable_json(output_dir / "verified-ledger.json", out_ledger_doc)

    summary = aggregate_verified_units(
        completed_docs,
        namespace=namespace,
        lane=lane,
        commit=commit,
        skipped_unit_ids=sorted(verified_in),
        require_complete_allocation=(len(verified_in) == 0),
    )
    publish_immutable_json(output_dir / "summary.json", summary)
    _emit(
        "complete",
        status=summary["status"],
        world_ticks=summary["world_ticks"],
        snake_ticks=summary["snake_ticks"],
        unexplained_mismatches=summary["unexplained_mismatches"],
        tie_explained_decisions=summary["tie_explained_decisions"],
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--lane", choices=["preflight", "canary", "main"], required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--b8-checkpoint", type=Path, required=True)
    parser.add_argument("--frp3-checkpoint", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    summary = run_parity_job(
        commit=args.commit,
        lane=args.lane,
        device=args.device,
        b8_checkpoint=args.b8_checkpoint,
        frp3_checkpoint=args.frp3_checkpoint,
        ledger_path=args.ledger,
        output_dir=args.output_dir,
    )
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
