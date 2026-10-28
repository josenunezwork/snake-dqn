"""Durable C/T counterfactual learnability + dynamic-topology probe runner.

Implements Ruling §4 & §5:
- Startup forced-state determinism suite + geometric topology fixture self-check
  before any collection or scoring.
- Resumable stage-unit execution (``stage-0001`` collection, ``stage-0002..0009``
  scoring chunks, ``stage-0010`` 3-arm distillation fit + ordered gates) skipping
  all Mac-verified ledger stages (including compact ``.pt.xz`` stages).
- Clean ``< 15 s`` ``SIGUSR1`` graceful stop writing completion-last
  ``sigusr1-marker.json`` pointing to the latest durable stage snapshot.
- Scientific-fault diagnostic publication (``fault.json``) on unhandled errors and
  immediate terminal ``INCOMPLETE_SUPPORT`` summary without extra collection.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from research.sandbox_a_resident.ct_branching import (
    ScoredRootChunk,
    merge_scored_chunks,
    partition_roots_into_chunks,
    score_root_chunk,
)
from research.sandbox_a_resident.ct_fit import fit_all_three_arms
from research.sandbox_a_resident.ct_mechanics import (
    RNG_CONSUMER_INVENTORY,
    forced_state_determinism_suite,
    task_separation_report,
)
from research.sandbox_a_resident.ct_stats import evaluate_ct_gates, validate_ct_support
from research.sandbox_a_resident.ct_task import (
    CollectedCTDataset,
    CTControl,
    collect_ct_dataset,
)
from research.sandbox_a_resident.ct_topology import TOPOLOGY_SPEC, validate_geometric_fixtures
from research.sandbox_a_resident.ct_units import (
    STALL_DIAGNOSTIC_SEAL_RESERVE_BYTES,
    TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES,
    allocation,
    load_ledger,
    load_verified_stage_payload,
    publish_bounded_export,
    publish_stage_snapshot,
)
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from research.sandbox_a_resident.units import digest, publish
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint
from src.model.inference_agent import InferenceAgent

B8_S3_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
SIGUSR1_EXIT_CODE = 138


def _load_vector61(path: Path, device: torch.device) -> torch.nn.Module:
    agent = InferenceAgent.from_checkpoint(str(path), device=device)
    if getattr(agent, "obs_spec", None) != "vector61":
        raise ValueError(f"{path} is not a vector61 checkpoint")
    return agent.network.eval()


def _write_progress(output: Path, payload: Mapping[str, Any]) -> None:
    tmp = output / "progress.json.partial"
    tmp.write_text(json.dumps(dict(payload), sort_keys=True) + "\n")
    os.replace(tmp, output / "progress.json")


def _stage_tail_reserve_bytes(control: CTControl) -> int:
    """Return reserved summary/diagnostic tail bytes under ``control.max_total_export_bytes``."""
    if int(control.max_total_export_bytes) >= 2 * TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES:
        return TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES
    return 0


def _summary_tail_reserve_bytes(control: CTControl) -> int:
    """Return reserved stall-diagnostic seal bytes when publishing terminal markers/summaries."""
    if int(control.max_total_export_bytes) >= 2 * TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES:
        return STALL_DIAGNOSTIC_SEAL_RESERVE_BYTES
    return 0


class _Sigusr1Fence:
    """Signal fence tracking SIGUSR1 receipt for clean <15s graceful shutdown."""

    def __init__(self) -> None:
        self.triggered = False
        self.received_monotonic: float | None = None
        self.received_unix: float | None = None
        self._prev_handler: Any = None

    def install(self) -> None:
        if not hasattr(signal, "SIGUSR1"):
            return

        def _handler(_signum: int, _frame: Any) -> None:
            if not self.triggered:
                self.triggered = True
                self.received_monotonic = time.monotonic()
                self.received_unix = time.time()

        self._prev_handler = signal.signal(signal.SIGUSR1, _handler)

    def restore(self) -> None:
        if hasattr(signal, "SIGUSR1") and self._prev_handler is not None:
            signal.signal(signal.SIGUSR1, self._prev_handler)
            self._prev_handler = None

    def stop_requested(self) -> bool:
        return bool(self.triggered)

    def elapsed_since_signal(self) -> float:
        if self.received_monotonic is None:
            return 0.0
        return max(0.0, time.monotonic() - self.received_monotonic)


def run_ct_experiment(
    control: CTControl,
    *,
    b8_checkpoint: Path,
    frp3_checkpoint: Path,
    champion_checkpoint: Path,
    output: Path,
    device: str = "cpu",
    ledger_path: Path | None = None,
    input_root: Path | None = None,
    watchdog: StallWatchdog | None = None,
    test_inject_delay_per_tick_sec: float = 0.0,
) -> tuple[int, dict[str, Any]]:
    """Execute or resume C/T package; returns ``(exit_code, terminal_payload)``."""
    t_run0 = time.perf_counter()
    output.mkdir(parents=True, exist_ok=True)
    control_dict = control.to_dict()
    alloc_sha = allocation(control_dict)

    if not control.tiny:
        if digest(b8_checkpoint) != B8_S3_SHA256:
            raise ValueError("C/T requires pinned released B8 s3 checkpoint")
        if digest(frp3_checkpoint) != FRP3_S12_SHA256:
            raise ValueError("C/T requires pinned released FRP3 s12 checkpoint")
        if digest(champion_checkpoint) != CHAMPION_SHA256:
            raise ValueError("C/T requires pinned released champion checkpoint")

    fence = _Sigusr1Fence()
    fence.install()
    latest_durable_stage = 0

    def emit(stage_label: str, extra: Mapping[str, Any] | None = None) -> None:
        rec = {
            "schema": "t34-cf-progress/v1",
            "study": control.study,
            "stage": stage_label,
            "latest_durable_stage": latest_durable_stage,
            "elapsed_sec": round(time.perf_counter() - t_run0, 2),
            **dict(extra or {}),
        }
        _write_progress(output, rec)
        print(f"CT_PROGRESS {json.dumps(rec, sort_keys=True)}", flush=True)
        if watchdog is not None:
            watchdog.progress(stage_label, rec)

    def heartbeat(tag: str) -> None:
        if test_inject_delay_per_tick_sec > 0.0:
            time.sleep(test_inject_delay_per_tick_sec)
        if watchdog is not None:
            watchdog.progress(tag, {"latest_durable_stage": latest_durable_stage})

    def graceful_sigusr1_exit(interrupted_phase: str) -> tuple[int, dict[str, Any]]:
        elapsed_sig = round(fence.elapsed_since_signal(), 4)
        marker = {
            "schema": "t34-cf-sigusr1-marker/v1",
            "study": control.study,
            "allocation": alloc_sha,
            "status": "PREEMPTED_SIGUSR1",
            "interrupted_phase": interrupted_phase,
            "latest_durable_stage": latest_durable_stage,
            "received_unix": fence.received_unix,
            "shutdown_elapsed_sec": elapsed_sig,
            "within_15s_sla": bool(elapsed_sig < 15.0),
        }
        publish_bounded_export(
            output / "sigusr1-marker.json",
            lambda p: p.write_text(json.dumps(marker, indent=2, sort_keys=True) + "\n"),
            export_root=output,
            max_total_bytes=control.max_total_export_bytes,
            reserved_tail_bytes=_summary_tail_reserve_bytes(control),
        )
        print(f"CT_SIGUSR1_CLEAN_EXIT {json.dumps(marker, sort_keys=True)}", flush=True)
        return SIGUSR1_EXIT_CODE, marker

    try:
        emit("initializing")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        dev = torch.device("cuda:0" if device == "cuda" else device)

        # Mandatory deterministic pre-scoring mechanics + topology fixture checks
        # qualified directly on the active runtime device before any scoring.
        det_report = forced_state_determinism_suite(dev)
        if det_report.get("status") != "PASS":
            raise RuntimeError("forced-state determinism suite failed")
        topo_fixture_report = validate_geometric_fixtures()
        if topo_fixture_report.get("status") != "PASS":
            raise RuntimeError("topology geometric fixture validation failed")
        emit(
            "startup-self-checks-passed",
            {
                "forced_state_cases": len(det_report["cases"]),
                "forced_state_device": det_report["device"],
                "topology_fixtures": topo_fixture_report["status"],
            },
        )

        b8_model: Ego2sNet = load_ego2s_checkpoint(str(b8_checkpoint), "cpu")
        b8_model.eval()
        if control.tiny:
            # Allow synthetic test vector61 checkpoints or raw nn.Linear state dicts in tiny tests.
            ckpt_nets: dict[str, torch.nn.Module] = {}
            for sha_key, p_ckpt in (
                (FRP3_S12_SHA256, frp3_checkpoint),
                (CHAMPION_SHA256, champion_checkpoint),
            ):
                blob = torch.load(p_ckpt, map_location="cpu", weights_only=False)
                if isinstance(blob, dict) and blob.get("tiny_linear_vector61"):
                    lin = torch.nn.Linear(61, 6)
                    lin.load_state_dict(blob["state_dict"])
                    ckpt_nets[sha_key] = lin.to(dev).eval()
                else:
                    ckpt_nets[sha_key] = _load_vector61(p_ckpt, dev)
        else:
            ckpt_nets = {
                FRP3_S12_SHA256: _load_vector61(frp3_checkpoint, dev),
                CHAMPION_SHA256: _load_vector61(champion_checkpoint, dev),
            }

        verified_stages: dict[int, dict[str, Any]] = {}
        compact_manifests: dict[str, dict[str, Any]] = {}
        if ledger_path is not None and ledger_path.exists():
            verified_stages, compact_manifests = load_ledger(
                ledger_path, control_dict, input_root=input_root
            )
            if verified_stages:
                latest_durable_stage = max(verified_stages)
                emit(
                    "ledger-verified",
                    {
                        "verified_stages": sorted(verified_stages),
                        "compact_units": sorted(compact_manifests),
                    },
                )

        base_meta = {
            "allocation": alloc_sha,
            "source_commit": control.source_commit,
            "study": control.study,
        }

        # --- STAGE 1: Collection + Whole-World Split + Support Audit ---
        if 1 in verified_stages:
            s1_blob = load_verified_stage_payload(
                verified_stages[1], compact_manifests=compact_manifests
            )
            collected = CollectedCTDataset.from_state_dict(s1_blob["dataset"])
            support_report = dict(s1_blob["support_report"])
            emit("stage-0001-resumed", {"roots": len(collected.store)})
        else:
            emit("stage-0001-collection-start")
            collected_opt = collect_ct_dataset(
                control,
                b8_network=b8_model,
                checkpoint_networks=ckpt_nets,
                device=dev,
                stop_requested=fence.stop_requested,
                heartbeat=heartbeat,
                log_progress=lambda info: emit("collection-progress", info),
            )
            if collected_opt is None or fence.stop_requested():
                return graceful_sigusr1_exit("stage-0001-collection")
            collected = collected_opt
            support_report = validate_ct_support(
                control=control,
                store=collected.store,
                anchors=collected.anchors,
                world_partitions=collected.world_partitions,
            )
            s1_payload = {
                "schema": "ct-stage-0001/v1",
                "dataset": collected.to_state_dict(),
                "support_report": support_report,
            }
            publish_stage_snapshot(
                output,
                1,
                lambda p: torch.save(s1_payload, p),
                metadata={
                    **base_meta,
                    "phase": "collection",
                    "support_status": support_report["status"],
                    "roots": len(collected.store),
                    "anchors": collected.anchors.num_rows,
                },
                max_stage_bytes=control.max_stage_export_bytes,
                max_total_bytes=control.max_total_export_bytes,
                reserved_tail_bytes=_stage_tail_reserve_bytes(control),
            )
            latest_durable_stage = 1
            emit(
                "stage-0001-published",
                {
                    "support_status": support_report["status"],
                    "roots": len(collected.store),
                    "anchors": collected.anchors.num_rows,
                },
            )

        # Ordered Gate 0: INVALID or INCOMPLETE_SUPPORT terminates immediately without extra runs.
        if support_report["status"] != "PASS":
            gate_eval = evaluate_ct_gates(
                control=control,
                support_report=support_report,
                store=collected.store,
                scored=None,
                anchors=collected.anchors,
                arm_predictions=None,
            )
            summary = {
                "schema": "t34-cf-summary/v1",
                "study": control.study,
                "allocation": alloc_sha,
                "source_commit": control.source_commit,
                "verdict": gate_eval["verdict"],
                "task_separation": task_separation_report(),
                "topology_spec": TOPOLOGY_SPEC,
                "rng_consumer_inventory": list(RNG_CONSUMER_INVENTORY),
                "startup_self_checks": {
                    "forced_state_determinism": det_report,
                    "topology_geometric_fixtures": topo_fixture_report,
                },
                "collection": collected.collection_telemetry,
                "evaluation": gate_eval,
                "resumed_stages": sorted(verified_stages),
                "total_wall_sec": round(time.perf_counter() - t_run0, 3),
            }
            publish_bounded_export(
                output / "summary.json",
                lambda p: p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"),
                export_root=output,
                max_total_bytes=control.max_total_export_bytes,
                reserved_tail_bytes=_summary_tail_reserve_bytes(control),
            )
            emit("completed-early-support-gate", {"verdict": summary["verdict"]})
            return 0, summary

        # --- STAGES 2 .. 1 + scoring_chunks: Counterfactual H300 Scoring ---
        chunk_slices = partition_roots_into_chunks(len(collected.store), control.scoring_chunks)
        scored_chunks: list[ScoredRootChunk] = []

        for chunk_idx, root_slice in enumerate(chunk_slices):
            stage_num = 2 + chunk_idx
            if stage_num in verified_stages:
                sc_blob = load_verified_stage_payload(
                    verified_stages[stage_num], compact_manifests=compact_manifests
                )
                chunk_obj = ScoredRootChunk.from_state_dict(sc_blob)
                scored_chunks.append(chunk_obj)
                emit(
                    f"stage-{stage_num:04d}-resumed",
                    {"chunk_index": chunk_idx, "roots": int(chunk_obj.root_indices.size)},
                )
                continue

            emit(
                f"stage-{stage_num:04d}-scoring-start",
                {"chunk_index": chunk_idx, "roots": int(root_slice.size)},
            )
            chunk_opt = score_root_chunk(
                chunk_index=chunk_idx,
                root_indices=root_slice,
                store=collected.store,
                control=control,
                b8_network=b8_model,
                checkpoint_networks=ckpt_nets,
                device=dev,
                stop_requested=fence.stop_requested,
                heartbeat=heartbeat,
                log_progress=lambda info: emit("scoring-progress", info),
            )
            if chunk_opt is None or fence.stop_requested():
                return graceful_sigusr1_exit(f"stage-{stage_num:04d}-scoring")
            scored_chunks.append(chunk_opt)
            chunk_state = chunk_opt.to_state_dict()
            publish_stage_snapshot(
                output,
                stage_num,
                lambda p, st=chunk_state: torch.save(st, p),
                metadata={
                    **base_meta,
                    "phase": f"scoring_chunk_{chunk_idx}",
                    "chunk_index": chunk_idx,
                    "roots": int(chunk_opt.root_indices.size),
                    "selection_freeze_sha256": chunk_opt.selection_freeze_sha256,
                },
                max_stage_bytes=control.max_stage_export_bytes,
                max_total_bytes=control.max_total_export_bytes,
                reserved_tail_bytes=_stage_tail_reserve_bytes(control),
            )
            latest_durable_stage = stage_num
            emit(
                f"stage-{stage_num:04d}-published",
                {
                    "chunk_index": chunk_idx,
                    "wall_sec": chunk_opt.telemetry["wall_sec"],
                },
            )

        merged_scored = merge_scored_chunks(scored_chunks, expected_num_roots=len(collected.store))

        # --- FINAL STAGE (2 + scoring_chunks): 3-Arm Fit + Detached Probe + Ordered Gates ---
        final_stage = 2 + control.scoring_chunks
        emit(f"stage-{final_stage:04d}-fit-start")
        fit_opt = fit_all_three_arms(
            control=control,
            store=collected.store,
            scored=merged_scored,
            anchors=collected.anchors,
            world_partitions=collected.world_partitions,
            b8_model=b8_model,
            device=dev,
            stop_requested=fence.stop_requested,
            heartbeat=heartbeat,
            log_progress=lambda info: emit("fit-progress", info),
        )
        if fit_opt is None or fence.stop_requested():
            return graceful_sigusr1_exit(f"stage-{final_stage:04d}-fit")

        arm_preds, fit_shared_meta = fit_opt
        gate_eval = evaluate_ct_gates(
            control=control,
            support_report=support_report,
            store=collected.store,
            scored=merged_scored,
            anchors=collected.anchors,
            arm_predictions=arm_preds,
        )

        final_stage_state = {
            "schema": "ct-stage-final/v1",
            "fit_shared_meta": fit_shared_meta,
            "evaluation": gate_eval,
            "arm_greedy_actions": {
                k: torch.from_numpy(v.root_greedy_actions.copy()) for k, v in arm_preds.items()
            },
        }
        publish_stage_snapshot(
            output,
            final_stage,
            lambda p: torch.save(final_stage_state, p),
            metadata={
                **base_meta,
                "phase": "fit_and_evaluation",
                "verdict": gate_eval["verdict"],
            },
            max_stage_bytes=control.max_stage_export_bytes,
            max_total_bytes=control.max_total_export_bytes,
            reserved_tail_bytes=_stage_tail_reserve_bytes(control),
        )
        latest_durable_stage = final_stage

        summary = {
            "schema": "t34-cf-summary/v1",
            "study": control.study,
            "allocation": alloc_sha,
            "source_commit": control.source_commit,
            "verdict": gate_eval["verdict"],
            "task_separation": task_separation_report(),
            "topology_spec": TOPOLOGY_SPEC,
            "rng_consumer_inventory": list(RNG_CONSUMER_INVENTORY),
            "startup_self_checks": {
                "forced_state_determinism": det_report,
                "topology_geometric_fixtures": topo_fixture_report,
            },
            "collection": collected.collection_telemetry,
            "scoring": {
                "chunks": control.scoring_chunks,
                "chunk_selection_freeze_sha256s": merged_scored.chunk_sha256s,
                "total_logical_branch_steps": merged_scored.total_logical_branch_steps,
                "total_executed_sim_steps": merged_scored.total_executed_sim_steps,
                "total_scoring_wall_sec": merged_scored.total_scoring_wall_sec,
                "total_charged_env_steps": (
                    int(collected.collection_telemetry["charged_collection_rows"])
                    + int(merged_scored.total_logical_branch_steps)
                ),
            },
            "fit": fit_shared_meta,
            "evaluation": gate_eval,
            "resumed_stages": sorted(verified_stages),
            "total_wall_sec": round(time.perf_counter() - t_run0, 3),
        }
        publish_bounded_export(
            output / "summary.json",
            lambda p: p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"),
            export_root=output,
            max_total_bytes=control.max_total_export_bytes,
            reserved_tail_bytes=_summary_tail_reserve_bytes(control),
        )
        emit("completed", {"verdict": summary["verdict"]})
        return 0, summary

    except BaseException as exc:
        fault_record = {
            "schema": "t34-cf-fault/v1",
            "study": control.study,
            "allocation": alloc_sha,
            "latest_durable_stage": latest_durable_stage,
            "error_type": type(exc).__qualname__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }
        fault_path = output / "fault.json"
        if not fault_path.exists():
            try:
                publish(
                    fault_path,
                    lambda p: p.write_text(
                        json.dumps(fault_record, indent=2, sort_keys=True) + "\n"
                    ),
                )
            except BaseException:
                pass
        raise
    finally:
        fence.restore()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--champion", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--input-root", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--stall-timeout-seconds", type=float, default=900.0)
    parser.add_argument("--stall-diagnostics", type=Path, default=None)
    parser.add_argument("--test-delay-per-heartbeat", type=float, default=0.0)
    args = parser.parse_args()

    torch.set_num_threads(1)
    control = CTControl.from_dict(json.loads(args.control.read_text()))
    diag_dir = (
        args.stall_diagnostics
        if args.stall_diagnostics is not None
        else (args.output / "stall-watchdog")
    )
    watchdog = StallWatchdog(diag_dir, args.stall_timeout_seconds)
    watchdog.start("startup")
    try:
        code, _payload = run_ct_experiment(
            control,
            b8_checkpoint=args.checkpoint,
            frp3_checkpoint=args.frp3,
            champion_checkpoint=args.champion,
            output=args.output,
            device=args.device,
            ledger_path=args.ledger,
            input_root=args.input_root,
            watchdog=watchdog,
            test_inject_delay_per_tick_sec=args.test_delay_per_heartbeat,
        )
        return code
    finally:
        watchdog.close()


if __name__ == "__main__":
    raise SystemExit(main())
