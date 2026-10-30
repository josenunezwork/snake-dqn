"""Collection-only H5000 successor with immutable stage units and honest cap outcome.

Keep the prior C/T startup parity checks, early liveness, SIGUSR1 fence and
completion-last exports. Never score branches or fit models in this packet.
"""

from __future__ import annotations

import argparse
import os
import signal
import time
import traceback
from pathlib import Path
from typing import Any, Mapping

import torch

from research.sandbox_a_resident import ct_json as json
from research.sandbox_a_resident.ct_h5000.diagnostics import DiagnosticJournal
from research.sandbox_a_resident.ct_h5000.task import (
    CollectedCTDataset,
    CollectionCapReached,
    CTControl,
    collect_ct_dataset,
)
from research.sandbox_a_resident.ct_h5000.units import (
    STALL_DIAGNOSTIC_SEAL_RESERVE_BYTES,
    TAIL_SUMMARY_AND_DIAGNOSTIC_RESERVE_BYTES,
    allocation,
    load_ledger,
    load_verified_stage_payload,
    publish_bounded_export,
    publish_stage_snapshot,
)
from research.sandbox_a_resident.ct_liveness import print_status, set_phase
from research.sandbox_a_resident.ct_mechanics import (
    forced_state_determinism_suite,
    task_separation_report,
)
from research.sandbox_a_resident.ct_stats import validate_ct_support
from research.sandbox_a_resident.ct_topology import validate_geometric_fixtures
from research.sandbox_a_resident.stall_watchdog import StallWatchdog
from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256
from research.sandbox_a_resident.units import digest
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


def _print_progress(output: Path, line: str) -> None:
    """Retain complete emitter lines so byte-truncated poll tails have a witness."""
    print_status(line, output)


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
    progress_interval_seconds: float = 45.0,
    fixture_stage_padding_bytes: int = 0,
) -> tuple[int, dict[str, Any]]:
    """Execute or resume C/T package; returns ``(exit_code, terminal_payload)``."""
    if fixture_stage_padding_bytes and (
        not control.tiny or not 0 < fixture_stage_padding_bytes <= 255 * 1024 * 1024
    ):
        raise ValueError("checkpoint padding is restricted to tiny fixtures, at most 255 MiB")
    if not 0 <= progress_interval_seconds <= 60:
        raise ValueError("progress interval must be between zero and 60 seconds")
    set_phase("input-validation")
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
    diagnostics = DiagnosticJournal()

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
        _print_progress(output, f"CT_PROGRESS {json.dumps(rec, sort_keys=True)}")
        if watchdog is not None:
            watchdog.progress(stage_label, rec)

    def heartbeat(tag: str) -> None:
        if test_inject_delay_per_tick_sec > 0.0:
            time.sleep(test_inject_delay_per_tick_sec)
        if watchdog is not None:
            watchdog.progress(tag, {"latest_durable_stage": latest_durable_stage})

    def graceful_sigusr1_exit(interrupted_phase: str) -> tuple[int, dict[str, Any]]:
        diagnostics.publish(output, control.max_total_export_bytes)
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
        _print_progress(output, f"CT_SIGUSR1_CLEAN_EXIT {json.dumps(marker, sort_keys=True)}")
        return SIGUSR1_EXIT_CODE, marker

    try:
        emit("initializing")
        set_phase("cuda-configuration")
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_float32_matmul_precision("highest")

        dev = torch.device("cuda:0" if device == "cuda" else device)

        # Mandatory deterministic pre-scoring mechanics + topology fixture checks
        # qualified directly on the active runtime device before any scoring.
        set_phase("forced-state-checks")
        det_report = forced_state_determinism_suite(dev)
        if det_report.get("status") != "PASS":
            raise RuntimeError("forced-state determinism suite failed")
        set_phase("topology-fixtures")
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

        set_phase("model-load")
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

        set_phase("ledger-validation")
        verified_stages: dict[int, dict[str, Any]] = {}
        compact_manifests: dict[str, dict[str, Any]] = {}
        if ledger_path is not None and ledger_path.exists():
            verified_stages, compact_manifests = load_ledger(
                ledger_path, control_dict, input_root=input_root
            )
            if set(verified_stages) - {1}:
                raise ValueError("support-only successor accepts only collection stage 1")
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

        set_phase("collection")
        # --- STAGE 1: Collection + Whole-World Split + Support Audit ---
        if 1 in verified_stages:
            s1_blob = load_verified_stage_payload(
                verified_stages[1], compact_manifests=compact_manifests
            )
            collected = CollectedCTDataset.from_state_dict(s1_blob["dataset"])
            telemetry = collected.collection_telemetry
            if (
                telemetry.get("completed_ticks") != control.collection_ticks
                or telemetry.get("charged_collection_rows")
                != control.worlds * control.collection_ticks
                or telemetry.get("collection_complete") is not True
            ):
                raise ValueError("resumed collection is incomplete")
            if collected.control.to_dict() != control_dict:
                raise ValueError("resumed dataset control differs from allocation")
            support_report = validate_ct_support(
                control=control,
                store=collected.store,
                anchors=collected.anchors,
                world_partitions=collected.world_partitions,
            )
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
                progress_interval_seconds=progress_interval_seconds,
                diagnostic=diagnostics.record,
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
            diagnostic_report = diagnostics.publish(output, control.max_total_export_bytes)
            s1_payload = {
                "diagnostics": diagnostic_report,
                "schema": "ct-stage-0001/v1",
                "dataset": collected.to_state_dict(),
                "support_report": support_report,
            }
            if fixture_stage_padding_bytes:
                s1_payload["fixture_padding"] = torch.zeros(
                    fixture_stage_padding_bytes, dtype=torch.uint8, device=dev
                )
                if dev.type == "cuda":
                    torch.cuda.synchronize(dev)

            def write_collection_snapshot(path: Path) -> None:
                if fixture_stage_padding_bytes:
                    # The canary controller signals during the real CUDA->host
                    # torch.save + atomic SHA/completion-last publication.
                    (output / "fixture-checkpoint-start").write_text("ready\n")
                    time.sleep(0.25)  # fixture-only, deterministic controller rendezvous
                torch.save(s1_payload, path)

            publish_stage_snapshot(
                output,
                1,
                write_collection_snapshot,
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

        if fence.stop_requested():
            return graceful_sigusr1_exit("stage-0001-collection")

        diagnostic_report = diagnostics.publish(output, control.max_total_export_bytes)
        if 1 in verified_stages:
            diagnostic_report = {
                "status": "NOT_RECOLLECTED_VERIFIED_STAGE",
                "original_exports": s1_blob.get("diagnostics", {}),
            }
        summary = {
            "diagnostics": diagnostic_report,
            "schema": "t34-ct-h5000-summary/v1",
            **base_meta,
            "verdict": (
                "SUPPORT_COMPLETE"
                if support_report["status"] == "PASS"
                else support_report["status"]
            ),
            "collection": collected.collection_telemetry,
            "support_report": support_report,
            "scope": "collection-support-only; no C/T gate, fit, transfer or release claim",
            "task_separation": task_separation_report(),
            "startup_self_checks": {
                "forced_state_determinism": det_report,
                "topology_geometric_fixtures": topo_fixture_report,
            },
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
        emit("completed-support-only", {"verdict": summary["verdict"]})
        return 0, summary

    except CollectionCapReached as exc:
        diagnostic_report = diagnostics.publish(output, control.max_total_export_bytes)
        summary = {
            "diagnostics": diagnostic_report,
            "schema": "t34-ct-h5000-summary/v1",
            "study": control.study,
            "allocation": alloc_sha,
            "source_commit": control.source_commit,
            "verdict": "INCOMPLETE_COLLECTION",
            "reason": "COLLECTION_WALL_CAP",
            "collection": {
                **exc.telemetry,
                "target_ticks": control.collection_ticks,
                "collection_cap_seconds": control.collection_cap_seconds,
            },
            "support_report": {"status": "NOT_EVALUATED_INCOMPLETE_COLLECTION"},
            "scope": "collection-support-only; no partial support look or automatic continuation",
            "latest_durable_stage": latest_durable_stage,
            "total_wall_sec": round(time.perf_counter() - t_run0, 3),
        }
        publish_bounded_export(
            output / "summary.json",
            lambda p: p.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n"),
            export_root=output,
            max_total_bytes=control.max_total_export_bytes,
            reserved_tail_bytes=_summary_tail_reserve_bytes(control),
        )
        emit("incomplete-collection-cap", {"verdict": summary["verdict"]})
        return 0, summary

    except BaseException as exc:
        diagnostics.publish(output, control.max_total_export_bytes)
        raw_message, raw_traceback = str(exc), traceback.format_exc()
        fault_record = {
            "schema": "t34-cf-fault/v1",
            "study": control.study,
            "allocation": alloc_sha,
            "latest_durable_stage": latest_durable_stage,
            "error_type": type(exc).__qualname__,
            "error_message": raw_message[:8192],
            "traceback": raw_traceback[-16384:],
            "diagnostic_text_truncated": len(raw_message) > 8192 or len(raw_traceback) > 16384,
        }
        fault_path = output / "fault.json"
        if not fault_path.exists():
            try:
                publish_bounded_export(
                    fault_path,
                    lambda p: p.write_text(
                        json.dumps(fault_record, indent=2, sort_keys=True) + "\n"
                    ),
                    export_root=output,
                    max_total_bytes=control.max_total_export_bytes,
                    reserved_tail_bytes=0,
                )
            except BaseException:
                pass
        raise
    finally:
        diagnostics.close()
        if control.tiny:
            facts = {"device": str(dev) if "dev" in locals() else device, "fixture_only": True}
            if device.startswith("cuda") and torch.cuda.is_available():
                facts.update(
                    gpu=torch.cuda.get_device_name(),
                    total_vram_bytes=torch.cuda.get_device_properties(0).total_memory,
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            (output / "fixture-runtime.json").write_text(json.dumps(facts, sort_keys=True) + "\n")
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
    parser.add_argument("--progress-interval-seconds", type=float, default=45.0)
    parser.add_argument("--fixture-stage-padding-bytes", type=int, default=0)
    args = parser.parse_args()

    set_phase("runner-configuration")
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
            progress_interval_seconds=args.progress_interval_seconds,
            fixture_stage_padding_bytes=args.fixture_stage_padding_bytes,
        )
        return code
    finally:
        watchdog.close()


if __name__ == "__main__":
    raise SystemExit(main())
