"""Short Mac resident lockstep profile with explicit conditional L4 projections."""

from __future__ import annotations

import argparse
import gc
import json
import os
import platform
import time
from dataclasses import asdict
from pathlib import Path

import numba
import psutil

from research.phase0_search_20261019 import teacher as phase0
from research.phase0_search_20261019.preflight import fixture
from research.phase0_search_20261019.runner import operational_guard
from research.t33_batched_teacher.ablation import B8, calibration_rows
from research.t33_batched_teacher.benchmark import complete, memory_admission, prepared
from research.t33_batched_teacher.evidence import identity
from research.t33_batched_teacher.inference import FixedInference
from research.t33_batched_teacher.teacher import BatchedTeacher, SearchConfig


def projection(cell: dict, threads: int) -> dict:
    """Expose assumptions; never confuse a CPU/GPU model with an L4 measurement."""
    phases, inference = cell["phases"], cell["inference"]
    host = max(0.0, cell["seconds"] - phases["hero_forward_s"] - inference["vector_wall_s"])
    # Six physical cores for 12-vCPU host. 1x and historical 3.7x CPU slowdown
    # bracket an illustrative range. All host work is assumed to scale, which
    # is optimistic for Python dispatch/assembly. This is not a confidence CI.
    hero_rows = inference["calls"] * (inference["rows"] or 0)
    gpu_seconds = hero_rows * (0.8 + 0.6) * 1e-6 + inference["vector_rows"] * 0.086e-6
    rates = {}
    for slowdown in (1.0, 3.7):
        host_seconds = host * threads / 6 * slowdown
        # A/B ideal overlap is a ceiling; no overlap adds the modeled GPU leg.
        for overlap in (False, True):
            seconds = max(host_seconds, gpu_seconds) if overlap else host_seconds + gpu_seconds
            rate = cell["worlds"] / max(seconds, 1e-12)
            rates[f"cpu_slowdown_{slowdown:g}_{'ideal_AB' if overlap else 'serial'}"] = {
                "one_L4_decisions_per_s": rate,
                "three_L4_decisions_per_s": 3 * rate,
            }
    return {
        "status": "CONDITIONAL_NOT_MEASURED",
        "host_seconds_on_Mac": host,
        "modeled_L4_device_seconds": gpu_seconds,
        "rates": rates,
        "assumptions": [
            "12 vCPU = 6 physical cores, perfect host scaling from configured Mac pool",
            "x86 core slowdown vs Mac between 1 and historical 3.7; workload not matched",
            "B8 0.8 us/row plus H2D 0.6 us/row from Fable throughput model",
            "Vector61 0.86 MFLOP/row at assumed 10 TFLOP/s; transfer omitted",
            "three independent hosts scale linearly; orchestration/setup omitted",
            "ideal A/B max(host,device) is a ceiling, not observed overlap",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worlds", default="256,1024")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=32)
    parser.add_argument("--rollouts", type=int, default=4)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.threads <= 4 or platform.system() != "Darwin":
        parser.error("Mac only, 1..4 native threads")
    if args.out.exists():
        raise FileExistsError(args.out)
    phase0.configure_cpu()
    os.environ["NUMBA_NUM_THREADS"] = str(args.threads)
    numba.config.reload_config()
    numba.set_num_threads(args.threads)
    operational_guard()
    templates = [prepared(fixture(row, B8, 800)) for row in calibration_rows(smoke=True)]
    config = SearchConfig(horizon=args.horizon, rollouts=args.rollouts, length_gate=0)
    # Warm all kernels/types on a small shape. Large-shape assembly, allocations
    # and first CPU resident forward remain in the measured wall time (conservative).
    warm = BatchedTeacher(config, inference=FixedInference(templates[0].hero.net, "cpu"))
    warm.choose(templates)
    del warm
    gc.collect()
    receipt = {
        "schema": "t33-Mac-lockstep/v1",
        "source": identity(),
        "threads": args.threads,
        "search": asdict(config),
        "cells": [],
        "warmup": "3 source worlds, full H/N; large-shape first allocation included",
        "platform": platform.platform(),
        "scope": "repeated L800 snapshots; timing only",
    }
    for count in map(int, args.worlds.split(",")):
        operational_guard()
        sources = [templates[i % len(templates)] for i in range(count)]
        host_available = int(psutil.virtual_memory().available * 0.7)
        admission = memory_admission(
            max(templates, key=lambda w: w.sim.seg_count.max()),
            count,
            config.rollouts,
            "cpu",
            host_available,
            None,
            horizon=config.horizon,
        )
        if not admission["admitted"]:
            receipt["cells"].append({"worlds": count, **admission})
            continue
        adapter = FixedInference(templates[0].hero.net, "cpu")
        search = BatchedTeacher(config, inference=adapter, opponent_inference=adapter)
        start = time.perf_counter()
        actions, diagnostics = search.choose(sources)
        elapsed = time.perf_counter() - start
        phases = search.profile()
        cell = {
            "status": "complete",
            "worlds": count,
            "admission": admission,
            "seconds": elapsed,
            "decisions_per_s": count / elapsed,
            "searched": sum(d["searched"] for d in diagnostics),
            "phases": phases,
            "inference": adapter.profile(),
            "stage_us_per_resident_row_step": {
                k: 1e6 * v / max(1, phases["resident_row_steps"])
                for k, v in phases.items()
                if k.endswith("_s")
            },
            "stage_us_per_active_row_step": {
                k: 1e6 * v / max(1, phases["active_row_steps"])
                for k, v in phases.items()
                if k.endswith("_s")
            },
        }
        cell["L4_projection"] = projection(cell, args.threads)
        receipt["cells"].append(cell)
        print(
            json.dumps(
                {
                    "event": "lockstep_cell",
                    "worlds": count,
                    "seconds": elapsed,
                    "decisions_per_s": count / elapsed,
                }
            ),
            flush=True,
        )
        del search, adapter, actions, diagnostics, sources
        gc.collect()
    complete(args.out, receipt)


if __name__ == "__main__":
    main()
