"""Run frozen C/T branches, three-arm fit and registered gates on the H5000 bank."""

from __future__ import annotations

import argparse
import json
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from research.sandbox_a_resident.ct_branching import (
    ScoredRootChunk,
    merge_scored_chunks,
    partition_roots_into_chunks,
    score_root_chunk,
)
from research.sandbox_a_resident.ct_fit import ArmEvaluationPredictions, fit_all_three_arms
from research.sandbox_a_resident.ct_h5000.task import CTControl
from research.sandbox_a_resident.ct_liveness import print_status, set_phase
from research.sandbox_a_resident.ct_mechanics import forced_state_determinism_suite
from research.sandbox_a_resident.ct_run import (
    B8_S3_SHA256,
    _load_vector61,
    _Sigusr1Fence,
    _write_progress,
)
from research.sandbox_a_resident.ct_stage2.bank import load_bank, small_real_slice
from research.sandbox_a_resident.ct_stage2.units import (
    STUDY,
    TAIL_BYTES,
    allocation,
    load_ledger,
    publish_bounded_export,
    publish_stage,
)
from research.sandbox_a_resident.ct_stats import evaluate_ct_gates
from research.sandbox_a_resident.ct_topology import validate_geometric_fixtures
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from research.sandbox_a_resident.units import digest
from src.model.ego2s_network import load_ego2s_checkpoint


def run(
    config: dict[str, Any],
    *,
    bank_root: Path,
    ledger: Path,
    input_root: Path,
    checkpoint: Path,
    frp3: Path,
    champion: Path,
    output: Path,
    device: str,
    watchdog: StallWatchdog,
) -> tuple[int, dict[str, Any]]:
    """Skip complete units without recomputation; never collect replacement data."""
    from research.sandbox_a_resident.ct_stage2.config import expected_science

    fixture = config.get("fixture") is True
    if config.get("study") != STUDY or config["science"] != expected_science(
        config["source_commit"], fixture=fixture
    ):
        raise ValueError("pipeline differs from frozen downstream recipe")
    if fixture and device != "cpu":
        raise ValueError("small real-bank fixture is Mac CPU only")
    ctrl = CTControl.from_dict(config["science"])
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    fence = _Sigusr1Fence()
    fence.install()
    completed: list[int] = []

    def emit(phase: str, info: dict[str, Any] | None = None) -> None:
        set_phase(phase)
        record = {
            "study": STUDY,
            "stage": phase,
            "fixture_only": fixture,
            "completed_stages": completed,
            "elapsed_sec": round(time.perf_counter() - started, 3),
            **(info or {}),
        }
        _write_progress(output, record)
        print_status("CT_PROGRESS " + json.dumps(record, sort_keys=True), output)
        watchdog.progress(phase, record)

    def heartbeat(tag: str) -> None:
        watchdog.progress(tag, {"completed_stages": completed})

    def publish_json(name: str, value: dict[str, Any]) -> None:
        publish_bounded_export(
            output / name,
            lambda p: p.write_text(json.dumps(value, sort_keys=True) + "\n"),
            export_root=output,
            max_total_bytes=ctrl.max_total_export_bytes,
            reserved_tail_bytes=TAIL_BYTES,
        )

    def stopped() -> tuple[int, dict[str, Any]]:
        marker = {
            "study": STUDY,
            "allocation": allocation(config),
            "status": "PREEMPTED_SIGUSR1",
            "completed_stages": completed,
            "shutdown_elapsed_sec": fence.elapsed_since_signal(),
        }
        publish_json("sigusr1-marker.json", marker)
        return 138, marker

    try:
        emit("input-validation")
        for path, sha in (
            (checkpoint, B8_S3_SHA256),
            (frp3, FRP3_S12_SHA256),
            (champion, CHAMPION_SHA256),
        ):
            if digest(path) != sha:
                raise ValueError("checkpoint differs from released identity")
        verified = load_ledger(ledger, config, input_root)
        completed.extend(sorted(verified))
        data = load_bank(bank_root, config["bank"])
        if fixture:
            data = small_real_slice(data)
        emit("bank-verified", {"roots": len(data.store), "anchors": data.anchors.num_rows})
        dev = torch.device("cuda:0" if device == "cuda" else device)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")
        emit("forced-state-checks")
        if forced_state_determinism_suite(dev)["status"] != "PASS":
            raise ValueError("execution-device forced-state checks failed")
        if validate_geometric_fixtures()["status"] != "PASS":
            raise ValueError("topology fixtures failed")
        b8 = load_ego2s_checkpoint(str(checkpoint), "cpu").eval()
        opponents = {
            FRP3_S12_SHA256: _load_vector61(frp3, dev),
            CHAMPION_SHA256: _load_vector61(champion, dev),
        }
        chunks = []
        for i, indices in enumerate(partition_roots_into_chunks(len(data.store), 8)):
            stage = i + 1
            if stage in verified:
                blob = torch.load(verified[stage], map_location="cpu", weights_only=False)
                chunk = ScoredRootChunk.from_state_dict(blob)
                if chunk.chunk_index != i or not np.array_equal(chunk.root_indices, indices):
                    raise ValueError("verified chunk coverage differs from frozen bank")
                emit("branch-resumed", {"chunk": i})
            else:
                emit("branch-start", {"chunk": i, "roots": len(indices)})
                chunk = score_root_chunk(
                    chunk_index=i,
                    root_indices=indices,
                    store=data.store,
                    control=ctrl,
                    b8_network=b8,
                    checkpoint_networks=opponents,
                    device=dev,
                    stop_requested=fence.stop_requested,
                    heartbeat=heartbeat,
                    log_progress=lambda info: emit("branch-progress", info),
                    progress_interval_seconds=45,
                )
                if chunk is None:
                    return stopped()
                publish_stage(output, stage, lambda p: torch.save(chunk.to_state_dict(), p), config)
                completed.append(stage)
            chunks.append(chunk)
            if fence.stop_requested():
                return stopped()
        scored = merge_scored_chunks(chunks, expected_num_roots=len(data.store))
        if 9 in verified:
            fit = torch.load(verified[9], map_location="cpu", weights_only=False)
            predictions = {k: ArmEvaluationPredictions(**v) for k, v in fit["predictions"].items()}
            emit("all-fits-resumed")
        else:
            emit("all-fits-start")
            fitted = fit_all_three_arms(
                control=ctrl,
                store=data.store,
                scored=scored,
                anchors=data.anchors,
                world_partitions=data.world_partitions,
                b8_model=b8,
                device=dev,
                stop_requested=fence.stop_requested,
                heartbeat=heartbeat,
                log_progress=lambda info: emit("fit-progress", info),
                progress_interval_seconds=45,
            )
            if fitted is None:
                return stopped()
            predictions, meta = fitted
            fit = {
                "schema": "ct-stage2-fits/v1",
                "fit_shared_meta": meta,
                "predictions": {k: asdict(v) for k, v in predictions.items()},
            }
            publish_stage(output, 9, lambda p: torch.save(fit, p), config)
            completed.append(9)
        if fence.stop_requested():
            return stopped()
        if 10 in verified:
            evaluation = torch.load(verified[10], map_location="cpu", weights_only=False)
            emit("evaluation-resumed")
        else:
            emit("registered-evaluation")
            evaluation = evaluate_ct_gates(
                control=ctrl,
                support_report=config["bank"]["support_report"],
                store=data.store,
                scored=scored,
                anchors=data.anchors,
                arm_predictions=predictions,
            )
            publish_stage(output, 10, lambda p: torch.save(evaluation, p), config)
            completed.append(10)
        summary = {
            "schema": "ct-stage2-summary/v1",
            "study": STUDY,
            "allocation": allocation(config),
            "source_commit": config["source_commit"],
            "bank_sha256": config["bank"]["payload"]["sha256"],
            "fixture_only": fixture,
            "verdict": "PREFLIGHT_ONLY" if fixture else evaluation["verdict"],
            "evaluation": {"sealed_fixture_diagnostic": True} if fixture else evaluation,
            "fit_shared_meta": fit["fit_shared_meta"],
            "resumed_stages": sorted(verified),
            "completed_stages": completed,
            "scoring": {
                "logical_branch_steps": scored.total_logical_branch_steps,
                "executed_sim_steps": scored.total_executed_sim_steps,
                "wall_sec": scored.total_scoring_wall_sec,
                "selection_freeze_sha256s": scored.chunk_sha256s,
            },
            "wall_sec": round(time.perf_counter() - started, 3),
        }
        publish_json("summary.json", summary)
        emit("completed")
        return 0, summary
    except BaseException as exc:
        if not (output / "fault.json").exists():
            publish_json(
                "fault.json",
                {
                    "error_type": type(exc).__name__,
                    "message": str(exc)[:8192],
                    "traceback": traceback.format_exc()[-16384:],
                },
            )
        raise
    finally:
        fence.restore()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in (
        "config",
        "bank-root",
        "ledger",
        "input-root",
        "checkpoint",
        "frp3",
        "champion",
        "output",
    ):
        parser.add_argument("--" + flag, type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stall-timeout-seconds", type=float, default=900)
    parser.add_argument("--stall-diagnostics", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    watchdog = StallWatchdog(
        args.stall_diagnostics or args.output / "stall-watchdog", args.stall_timeout_seconds
    )
    watchdog.start("startup")
    try:
        code, _ = run(
            json.loads(args.config.read_text()),
            bank_root=args.bank_root,
            ledger=args.ledger,
            input_root=args.input_root,
            checkpoint=args.checkpoint,
            frp3=args.frp3,
            champion=args.champion,
            output=args.output,
            device=args.device,
            watchdog=watchdog,
        )
        return code
    finally:
        watchdog.close()


if __name__ == "__main__":
    raise SystemExit(main())
