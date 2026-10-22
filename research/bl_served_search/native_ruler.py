"""Prospective paired kernel rulers; no science bank or provider calls."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import multiprocessing
import os
import platform
import statistics
import time
from pathlib import Path
from typing import Any

from research.bl_served_search import ruler
from research.bl_served_search.exports import digest, export_json, verify_export
from research.bl_served_search.native_provenance import OLD_RULER, RULER_PATH, compatible
from research.bl_served_search.provenance_v2 import CHUNKS_PATH, LEGACY_CHUNKS_SHA256
from research.bl_served_search.ruler_timing import STAGES

SCHEMA = "bl-native-paired-ruler/v1"
THREAD_ENV = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
MODES = ("all", "exact_stop")
_OPERATORS: dict[str, Any] = {}
# Exact pre-amendment bytes, not a blanket compatibility exemption.
LEGACY = {
    RULER_PATH: {OLD_RULER},
    "research/bl_served_search/ceiling.py": {
        "b617043b0ede1dc40e82e8c39262fdf73b476f665d48ad8afcd77c1642a2645e"
    },
    "research/bl_served_search/ruler_operator.py": {
        "ba7facc66b05d7917c3acda5c4c048abcedd463949962263b089648bc0d32b79",
        "896ae6b806c5ada71298bd6c28809ab9d394eab0307d809dd2eaf091e1c14006",
    },
    CHUNKS_PATH: {LEGACY_CHUNKS_SHA256},
    "research/t3_rollout_labels_20261015/bank_engine.py": {
        "c4446dad0b2323b5ce6efd1b53b6266fdca64dc3407bc2ea957e3dd77082d157"
    },
}


def source_identity() -> dict[str, str]:
    """Bind the full new ruler path, including the native routing implementation."""
    root = Path(__file__).resolve().parents[2]
    names = (
        "research/bl_served_search/native_ruler.py",
        "research/bl_served_search/native_provenance.py",
        "research/bl_served_search/mac_lease.py",
        "research/bl_served_search/AMENDMENT_STOP.md",
        "research/bl_k1/native.py",
        "research/t34_speed_s2/native_v8.py",
        "src/simd_env/v8_nb.py",
        "research/bl_served_search/run_native_ruler.sh",
        "research/bl_served_search/run_stop_mac.sh",
        "research/bl_mac_latency/host.py",
    )
    return {**ruler.source_identity(), **{name: digest(root / name) for name in names}}


def validate_source(saved: dict[str, str]) -> None:
    """Accept frozen roots with only the explicitly amended source bytes."""
    current = ruler.source_identity()
    omitted = set(current) - set(saved)
    if (
        (omitted - {"research/bl_served_search/exact_stop.py"})
        or set(saved) - set(current)
        or any(
            checksum != current[name] and checksum not in LEGACY.get(name, set())
            for name, checksum in saved.items()
        )
    ):
        raise ValueError("fixture source differs outside prospective STOP amendment")


def _count(value: Any) -> bool:
    return type(value) is int and value >= 0


def validate_batch(batch: dict[str, Any], fixture: dict[str, Any], mode: str) -> None:
    """Execution counters, not enable flags, establish the native/stop path."""
    semantic = batch.get("semantic", {})
    legal = set(fixture["legal_actions"])
    outcomes = semantic.get("outcomes", {})
    order = semantic.get("evaluation_order", [])
    if (
        semantic.get("search_mode") != mode
        or semantic.get("fired") is not True
        or semantic.get("chosen") not in legal
        or semantic.get("chosen") not in order
        or semantic.get("bare") not in legal
        or not order
        or len(order) != len(set(order))
        or not set(order) <= legal
        or set(outcomes) != {str(action) for action in legal}
        or semantic.get("branches_evaluated") != len(order)
        or semantic.get("branches_skipped") != len(legal) - len(order)
        or (mode == "all" and set(order) != legal)
        or (mode == "exact_stop" and order[0] != semantic["bare"])
    ):
        raise ValueError("INVALID branch execution attestation")
    total = 0
    for action in legal:
        outcome = outcomes[str(action)]
        if action in order:
            frames = outcome.get("frames_executed", outcome.get("env_frames"))
            if not _count(frames) or not 1 <= frames <= ruler.HORIZON:
                raise ValueError("INVALID branch frames")
            if type(outcome.get("survives")) is not bool:
                raise ValueError("INVALID evaluated outcome")
            if outcome["survives"] and frames != ruler.HORIZON:
                raise ValueError("INVALID survivor did not execute H300")
            total += frames
        elif outcome != {"status": "NOT_EVALUATED"}:
            raise ValueError("skipped branch must be NOT_EVALUATED")
    if semantic.get("rollout_branch_frames") != total:
        raise ValueError("INVALID frame reconciliation")
    stats = batch.get("native_v8", {})
    if stats.get("mode") != "kernel" or any(
        not _count(stats.get(key))
        for key in ("kernel_rows", "stock_rows", "noncandidate_rows", "fallback_rows", "calls")
    ):
        raise ValueError("INVALID native execution counters")
    if any(stats[key] != 0 for key in ("stock_rows", "noncandidate_rows", "fallback_rows")):
        raise ValueError("INVALID native fallback rows")
    if fixture["mix"] != "scripted" and (stats["kernel_rows"] <= 0 or stats["calls"] <= 0):
        raise ValueError("INVALID missing native branch rows")
    if fixture["mix"] == "scripted" and (stats["kernel_rows"] or stats["calls"]):
        raise ValueError("INVALID scripted native exemption")
    stages = batch.get("stage_seconds", {})
    if set(stages) != {*STAGES, "submit_visible", "submit_consumed"} or any(
        type(value) not in (int, float) or not math.isfinite(value) or value < 0
        for value in stages.values()
    ):
        raise ValueError("INVALID stage timers")
    threads = batch.get("worker_threads", {})
    if (
        threads.get("environment") != dict.fromkeys(THREAD_ENV, "1")
        or threads.get("torch") != 1
        or threads.get("numba") != 1
    ):
        raise ValueError("INVALID worker thread attestation")
    seconds = batch.get("worker_wall_seconds")
    if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError("INVALID worker wall timer")


def _cell(job: tuple[dict, int, str]) -> tuple[str, int, dict]:
    from research.bl_served_search.ruler_operator import ExactT3Operator

    fixture, repeat, mode = job
    if mode not in _OPERATORS:
        _OPERATORS[mode] = ExactT3Operator(native_mode="kernel", search_mode=mode)
    started = time.perf_counter()
    output = _OPERATORS[mode].batch(fixture)
    output["worker_wall_seconds"] = time.perf_counter() - started
    import numba
    import torch

    output["worker_threads"] = {
        "environment": {name: os.environ.get(name) for name in THREAD_ENV},
        "torch": torch.get_num_threads(),
        "numba": numba.get_num_threads(),
    }
    validate_batch(output, fixture, mode)
    output["worker_pid"] = os.getpid()
    return fixture["fixture_id"], repeat, output


def _distribution(samples: list[float]) -> dict[str, float]:
    warm = sorted(samples[1:])
    return {
        "cold_seconds": samples[0],
        "warm_median_seconds": statistics.median(warm) if warm else samples[0],
        "warm_p99_seconds": warm[math.ceil(0.99 * len(warm)) - 1] if warm else samples[0],
    }


def validate_paired_receipt(receipt: dict[str, Any], *, canary: bool = False) -> None:
    """Reject any changed action or unattested/partial paired ruler before sizing."""
    if (
        receipt.get("schema") != SCHEMA
        or receipt.get("status") != ("CANARY_PASS" if canary else "PAIRED_PASS")
        or receipt.get("native_mode") != "kernel"
        or not compatible(receipt.get("source_identity", {}), source_identity())
        or set(receipt.get("cells", {})) != set(MODES)
    ):
        raise ValueError("INVALID paired native ruler")
    for field in ("k1_sha256",) if canary else ("k1_sha256", "canary_sha256"):
        ruler._sha(receipt.get(field), field)
    elapsed = receipt.get("elapsed_seconds")
    if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
        raise ValueError("INVALID paired elapsed time")
    fixtures = receipt["fixtures"]
    rows = ruler.validate_fixtures(fixtures)
    validate_source(fixtures["source_identity"])
    expected = 1 if canary else 30
    by_mode = {}
    label = receipt.get("platform", {}).get("label")
    if label not in ("mac", "l4"):
        raise ValueError("INVALID target platform")
    if receipt["platform"].get("system") != ("Darwin" if label == "mac" else "Linux"):
        raise ValueError("INVALID target host")
    for mode in MODES:
        cell = receipt["cells"][mode]
        if (
            not compatible(cell.get("source_identity", {}), ruler.source_identity())
            or cell.get("fixture_sha256") != ruler.sha256_bytes(ruler.canonical(fixtures).encode())
            or cell.get("platform") != receipt["platform"]
            or cell.get("min_batches_per_fixture") != expected
            or cell.get("workers") != (1 if label == "mac" else 12)
            or cell.get("completed_batches") != expected * 9
        ):
            raise ValueError("INVALID cell identity/count")
        records = cell.get("records", [])
        if len(records) != 9 or {item["fixture_id"] for item in records} != {
            row["fixture_id"] for row in rows
        }:
            raise ValueError("INCOMPLETE paired fixtures")
        indexed = {row["fixture_id"]: row["batches"] for row in records}
        for fixture in rows:
            batches = indexed[fixture["fixture_id"]]
            if len(batches) != expected:
                raise ValueError("INCOMPLETE paired repeats")
            for batch in batches:
                validate_batch(batch, fixture, mode)
        elapsed = cell.get("elapsed_seconds")
        rate = cell.get("aggregate_batches_per_second")
        if (
            type(elapsed) not in (int, float)
            or not math.isfinite(elapsed)
            or elapsed <= 0
            or type(rate) not in (int, float)
            or not math.isfinite(rate)
            or not math.isclose(rate, expected * 9 / elapsed, rel_tol=1e-12)
        ):
            raise ValueError("INVALID aggregate rate")
        by_mode[mode] = indexed
    comparisons = 0
    for fixture in rows:
        key = fixture["fixture_id"]
        for left, right in zip(by_mode["all"][key], by_mode["exact_stop"][key]):
            comparisons += 1
            if left["semantic"]["chosen"] != right["semantic"]["chosen"]:
                raise ValueError("INVALID chosen-action mismatch; exact stop fails")
    if receipt.get("decision_identity") != {
        "status": "PASS",
        "compared_batches": comparisons,
        "mismatches": [],
    }:
        raise ValueError("INVALID decision identity attestation")
    if label == "mac" and not receipt.get("mac_lease", {}).get("id"):
        raise ValueError("INVALID missing Mac lease")
    if label == "l4":
        hardware = receipt.get("hardware", {})
        if (
            (hardware.get("reported") or {}).get("hardware", {}).get("kind") != "L4"
            or hardware.get("vcpu_count", 0) < 12
            or not hardware.get("sha256")
        ):
            raise ValueError("INVALID L4 hardware")


def run_pairs(
    fixtures: dict,
    *,
    label: str,
    workers: int,
    repeats: int,
    guard: Any,
    out: Path,
    max_seconds: float,
    lease: dict | None = None,
    canary: bool = False,
) -> dict:
    """Alternate homogeneous slabs, measuring mode wall time without rate scaling."""
    rows = ruler.validate_fixtures(fixtures)
    validate_source(fixtures["source_identity"])
    records = {mode: {row["fixture_id"]: {} for row in rows} for mode in MODES}
    started = time.monotonic()
    jobs = [(row, repeat) for repeat in range(repeats) for row in rows]
    context = multiprocessing.get_context("spawn")
    # Mac stays W1, allowing read-only host evidence in the coordinating process.
    mode_seconds = dict.fromkeys(MODES, 0.0)
    pool = concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context)
    try:
        # Homogeneous slabs give each mode its OWN measured wall intervals.
        # Alternate slab order, keep exact pairs, and never multiply a hook gain.
        for offset in range(0, len(jobs), workers):
            slab = jobs[offset : offset + workers]
            for mode in MODES if (offset // workers) % 2 == 0 else tuple(reversed(MODES)):
                guard()
                if time.monotonic() - started >= max_seconds:
                    raise RuntimeError("INCOMPLETE native ruler wall cap before submit")
                cell_started = time.monotonic()
                pending = {pool.submit(_cell, (fixture, repeat, mode)) for fixture, repeat in slab}
                while pending:
                    observation = guard()
                    if observation is not None:
                        export_json(out / "host" / f"{time.monotonic_ns()}.json", observation)
                    if time.monotonic() - started > max_seconds:
                        raise RuntimeError("INCOMPLETE native ruler wall cap")
                    done, pending = concurrent.futures.wait(
                        pending,
                        timeout=min(5.0, max(0.01, max_seconds - (time.monotonic() - started))),
                        return_when=concurrent.futures.FIRST_COMPLETED,
                    )
                    for future in done:
                        key, repeat, result = future.result()
                        export_json(out / "batches" / f"{key}-{repeat:02d}-{mode}.json", result)
                        records[mode][key][repeat] = result
                        other = "exact_stop" if mode == "all" else "all"
                        previous = records[other][key].get(repeat)
                        if (
                            previous
                            and result["semantic"]["chosen"] != previous["semantic"]["chosen"]
                        ):
                            raise RuntimeError("INVALID chosen-action mismatch; no speed claim")
                    print(
                        json.dumps(
                            {
                                "completed_cells": sum(
                                    len(x) for values in records.values() for x in values.values()
                                ),
                                "total_cells": len(jobs) * 2,
                                "elapsed_seconds": time.monotonic() - started,
                            }
                        ),
                        flush=True,
                    )
                mode_seconds[mode] += time.monotonic() - cell_started
    except BaseException:
        # Only these exact child processes belong to this pool. Python 3.12 has
        # no public terminate_workers(); do not leave workers running after a
        # lease, cap or identity failure, and never signal a name/PID pattern.
        for process in list((pool._processes or {}).values()):
            process.terminate()
        pool.shutdown(wait=True, cancel_futures=True)
        raise
    else:
        pool.shutdown(wait=True)
    guard()
    elapsed = time.monotonic() - started
    if elapsed > max_seconds:
        raise RuntimeError("INCOMPLETE native ruler wall cap after final pair")
    cells = {}
    for mode in MODES:
        cells[mode] = {
            "schema": ruler.RECEIPT_SCHEMA,
            "status": "PRECHECK_ONLY" if canary else "RULER_PASS",
            "platform": ruler.platform_identity(label),
            "source_identity": ruler.source_identity(),
            "fixture_sha256": ruler.sha256_bytes(ruler.canonical(fixtures).encode()),
            "workers": workers,
            "min_batches_per_fixture": repeats,
            "completed_batches": len(jobs),
            "elapsed_seconds": mode_seconds[mode],
            "aggregate_batches_per_second": len(jobs) / mode_seconds[mode],
            "rate_denominator": "sum of actual homogeneous slab wall intervals; no scaling",
            "records": [
                {"fixture_id": key, "batches": [value[i] for i in sorted(value)]}
                for key, value in sorted(records[mode].items())
            ],
            "timing_summary": {
                key: _distribution([value[i]["worker_wall_seconds"] for i in sorted(value)])
                for key, value in records[mode].items()
            },
        }
    receipt = {
        "schema": SCHEMA,
        "status": "CANARY_PASS" if canary else "PAIRED_PASS",
        "native_mode": "kernel",
        "platform": ruler.platform_identity(label),
        "source_identity": source_identity(),
        "fixtures": fixtures,
        "hardware": {**ruler.hardware_provenance(), "vcpu_count": os.cpu_count()},
        "host_node": platform.node(),
        "cells": cells,
        "mac_lease": lease,
        "decision_identity": {"status": "PASS", "compared_batches": len(jobs), "mismatches": []},
        "authority": "AMENDMENT_STOP.md; no dispatch or release authority",
        "elapsed_seconds": elapsed,
    }
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--platform", choices=("mac", "l4"), required=True)
    parser.add_argument("--k1", type=Path, required=True)
    parser.add_argument("--lease", type=Path)
    parser.add_argument("--lease-owner")
    parser.add_argument("--lease-id")
    parser.add_argument("--canary", action="store_true")
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--max-seconds", type=float, default=7200)
    args = parser.parse_args()
    if not math.isfinite(args.max_seconds) or not 0 < args.max_seconds <= (
        7200 if args.platform == "mac" else 5760
    ):
        parser.error("wall cap exceeds Mac 2h / L4 96min envelope")
    if (
        any(os.environ.get(name) != "1" for name in THREAD_ENV)
        or os.environ.get("SNAKE_DQN_DEVICE") != "cpu"
    ):
        parser.error("CPU and all five math thread limits must be explicit")
    verify_export(args.k1)
    k1 = json.loads(args.k1.read_text())
    if k1.get("status") != "K1_PASS" or k1.get("speedup", 0) < 5:
        parser.error("NOT_READY(K-1 D2 PASS)")
    verify_export(args.fixtures)
    fixtures = json.loads(args.fixtures.read_text())
    validate_source(fixtures["source_identity"])
    ruler.validate_fixtures(fixtures)
    lease = None
    if args.platform == "mac":
        if not all((args.lease, args.lease_owner, args.lease_id)):
            parser.error("Mac lease path/owner/id required")
        from research.bl_mac_latency.host import check_guard, process_screen, snapshot
        from research.bl_served_search.mac_lease import require

        def guard() -> dict:
            require(args.lease, args.lease_owner, args.lease_id)
            return {
                **check_guard(),
                "load_average": list(os.getloadavg()),
                "process_screen": process_screen(),
            }

        lease = require(args.lease, args.lease_owner, args.lease_id)
        guard()
        idle = snapshot()
        if not idle["quiet"]:
            parser.error("host not idle; no daemon killing")
    else:
        if platform.system() != "Linux" or not os.environ.get("COLAB_CLI_ROOT"):
            parser.error("L4 requires qualified CLI runtime")

        def guard() -> None:
            pass

        hardware = ruler.hardware_provenance()
        if (
            (hardware.get("reported") or {}).get("hardware", {}).get("kind") != "L4"
            or not hardware.get("sha256")
            or (os.cpu_count() or 0) < 12
        ):
            parser.error("L4 ruler requires qualified ready.json L4 and >=12 vCPUs")
        guard()
        idle = None
    canary_seconds = 0.0
    if not args.canary:
        if args.canary_receipt is None:
            parser.error("same-source target-platform canary required")
        verify_export(args.canary_receipt)
        canary = json.loads(args.canary_receipt.read_text())
        validate_paired_receipt(canary, canary=True)
        canary_seconds = float(canary["elapsed_seconds"])
        if (
            canary["platform"] != ruler.platform_identity(args.platform)
            or canary["fixtures"] != fixtures
            or (args.platform == "mac" and canary.get("host_node") != platform.node())
        ):
            parser.error("canary binds another platform/fixture")
    remaining = min(args.max_seconds, (7200 if args.platform == "mac" else 5760) - canary_seconds)
    if remaining <= 0:
        parser.error("canary exhausted the ruler envelope")
    args.out.mkdir(parents=True, exist_ok=False)
    try:
        if idle is not None:
            export_json(args.out / "idle.json", idle)
        receipt = run_pairs(
            fixtures,
            label=args.platform,
            workers=1 if args.platform == "mac" else 12,
            repeats=1 if args.canary else 30,
            guard=guard,
            out=args.out,
            max_seconds=remaining,
            lease=lease,
            canary=args.canary,
        )
        receipt["k1_sha256"] = digest(args.k1)
        receipt["canary_sha256"] = digest(args.canary_receipt) if args.canary_receipt else None
        validate_paired_receipt(receipt, canary=args.canary)
        export_json(args.out / "ruler-receipt.json", receipt)
    except Exception as exc:
        export_json(
            args.out / "stopped.json",
            {"status": "INVALID" if "INVALID" in str(exc) else "INCOMPLETE", "error": str(exc)},
        )
        raise


if __name__ == "__main__":
    main()
