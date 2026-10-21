"""Real local/Colab E3 entrypoint, deterministic reconstruction then offline fits."""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_info_value.catalog import PROTOCOL, check_path, prepare, unpack
from research.bl_info_value.io import sha256, write_json, write_npz
from research.bl_info_value.resources import (
    PROGRESS_SECONDS,
    available_cpus,
    lower_priority,
    worker_count,
)


def threads() -> None:
    """One math thread per worker; process count supplies parallelism."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _reconstruct(task: dict[str, Any]) -> tuple[dict, dict]:
    lower_priority()
    threads()
    from research.bl_info_value.reconstruct import reconstruct_world

    arrays, metrics = reconstruct_world(task)
    return arrays, {**metrics, "worker_nice": os.nice(0)}


def merge(results: list[tuple[dict, dict]]) -> dict[str, np.ndarray]:
    """Merge in input task order, never process completion order."""
    if not results:
        raise ValueError("no reconstructed worlds")
    names = set(results[0][0])
    if any(set(arrays) != names for arrays, _ in results):
        raise ValueError("reconstructed schema differs across worlds")
    return {name: np.concatenate([arrays[name] for arrays, _ in results]) for name in sorted(names)}


def execute(args: argparse.Namespace) -> dict:
    """Run the exact entrypoint on a fresh archive and preserve terminal evidence."""
    started = time.perf_counter()
    work, out = check_path(args.work), check_path(args.out)
    if work.exists() or out.exists():
        raise FileExistsError("work and output directories must both be fresh")
    requested_workers = args.workers
    cpus = available_cpus()
    args.workers = worker_count(requested_workers, cpus)
    if args.expected_workers is not None and args.workers != args.expected_workers:
        raise ValueError("usable CPU headroom cannot supply the manifest's priced worker count")
    nice = lower_priority()
    threads()
    catalog = unpack(args.archive, work / "inputs", args.archive_sha256)
    if catalog["smoke"] != args.smoke or not 1 <= args.workers <= 12:
        raise ValueError("smoke/production input mismatch or invalid workers")
    if args.smoke and args.workers > 2:
        raise ValueError("Mac smoke permits at most two workers")
    if not args.smoke and (args.epochs != 60 or args.device != "cuda"):
        raise ValueError("production fits require frozen 60 epochs on CUDA")
    inputs = work / "inputs"
    if catalog["schema"] == "bl-e3-input/v2":
        from research.bl_info_value.compact import hydrate

        if args.checkpoint_archive is None:
            raise ValueError("compact inputs require the hash-bound checkpoint archive")
        hydrate(catalog, inputs, args.checkpoint_archive)
    os.environ["SNAKE_CKPT_DIR"] = str(inputs / "checkpoints")
    out.mkdir(parents=True)
    tasks = []
    for task in catalog["tasks"]:
        item = json.loads(json.dumps(task))
        item["checkpoint"] = str(inputs / item["checkpoint"])
        item["source_npz"] = str(inputs / item["source_npz"])
        for spec in item["plan"]["specs"]:
            if spec[0] == "checkpoint":
                spec[1] = str(inputs / spec[1])
        tasks.append(item)
    write_json(
        out / "start.json",
        dict(
            schema="bl-e3-start/v1",
            protocol_sha256=sha256(PROTOCOL),
            input_sha256=args.archive_sha256,
            source_archive_sha256=catalog.get("source_archive_sha256", args.archive_sha256),
            input_schema=catalog["schema"],
            smoke=args.smoke,
            workers=args.workers,
            requested_workers=requested_workers,
            available_cpus=cpus,
            nice=nice,
            worlds=len(tasks),
            roots=sum(len(t["indices"]) for t in tasks),
            authority="development only; excluded from every registered verdict",
        ),
    )
    try:
        reconstruction_start = time.perf_counter()
        last_progress = reconstruction_start
        results = []
        if args.workers == 1:
            iterator = map(_reconstruct, tasks)
            pool = None
        else:
            pool = ProcessPoolExecutor(
                args.workers, mp_context=multiprocessing.get_context("spawn")
            )
            iterator = pool.map(_reconstruct, tasks)
        try:
            for index, result in enumerate(iterator):
                arrays, metrics = result
                results.append(result)
                # Keep incremental scratch out of the supervisor inventory:
                # hundreds of exports cause repeated transfer/renewal RPCs.
                # dataset.npz and timing.json retain all arrays and metrics.
                write_npz(work / "worlds" / f"{index:03d}.npz", arrays)
                write_json(work / "worlds" / f"{index:03d}.json", metrics)
                now = time.perf_counter()
                if index == 0 or index + 1 == len(tasks) or now - last_progress >= PROGRESS_SECONDS:
                    elapsed = now - reconstruction_start
                    print(
                        json.dumps(
                            dict(
                                phase="reconstruction",
                                done=index + 1,
                                total=len(tasks),
                                elapsed_seconds=elapsed,
                                eta_seconds=elapsed / (index + 1) * (len(tasks) - index - 1),
                            )
                        ),
                        flush=True,
                    )
                    last_progress = now
        finally:
            if pool:
                pool.shutdown(wait=True, cancel_futures=True)
        reconstruction_seconds = time.perf_counter() - reconstruction_start
        arrays = merge(results)
        parity_seconds = 0.0
        if args.check_parallel:
            if not args.smoke:
                raise ValueError("serial/parallel qualification is smoke-only")
            parity_start = time.perf_counter()
            other_workers = 2 if args.workers == 1 else 1
            if worker_count(other_workers, cpus) != other_workers:
                raise ValueError("parallel qualification requires four usable CPUs")
            if other_workers == 1:
                repeated = list(map(_reconstruct, tasks))
            else:
                with ProcessPoolExecutor(
                    2, mp_context=multiprocessing.get_context("spawn")
                ) as parity_pool:
                    repeated = list(parity_pool.map(_reconstruct, tasks))
            other = merge(repeated)
            if any(
                arrays[name].dtype != other[name].dtype
                or arrays[name].shape != other[name].shape
                or arrays[name].tobytes(order="C") != other[name].tobytes(order="C")
                for name in arrays
            ):
                raise ValueError("serial/parallel array-byte identity failed")
            parity_seconds = time.perf_counter() - parity_start
        write_npz(out / "dataset.npz", arrays)
        from research.bl_info_value.fit import fit_students

        fit_start = time.perf_counter()
        fits = fit_students(
            arrays, out / "fits", device=args.device, epochs=args.epochs, smoke=args.smoke
        )
        fit_seconds = time.perf_counter() - fit_start
        report = dict(
            schema="bl-e3-timing/v1",
            smoke=args.smoke,
            workers=args.workers,
            requested_workers=requested_workers,
            available_cpus=cpus,
            nice=nice,
            worlds=len(tasks),
            roots=len(arrays["action"]),
            reconstruction_seconds=reconstruction_seconds,
            fit_seconds=fit_seconds,
            parity_seconds=parity_seconds,
            total_seconds=time.perf_counter() - started,
            world_metrics=[m for _, m in results],
            fits=fits,
            smoke_epochs=args.epochs if args.smoke else None,
            research_sources=sorted(
                {
                    str(Path(module.__file__).resolve().relative_to(Path.cwd()))
                    for name, module in sys.modules.items()
                    if name.startswith("research.")
                    and getattr(module, "__file__", None)
                    and Path(module.__file__).resolve().is_relative_to(Path.cwd())
                }
            ),
        )
        write_json(out / "timing.json", report)
        write_json(
            out / "terminal.json",
            dict(
                status="SMOKE_COMPLETE" if args.smoke else "COMPLETE",
                timing_sha256=sha256(out / "timing.json"),
            ),
        )
        return report
    except Exception as exc:
        write_json(
            out / "terminal.json",
            dict(
                status="INCOMPLETE", reason=str(exc), elapsed_seconds=time.perf_counter() - started
            ),
        )
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--bank", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--smoke", action="store_true")
    run = commands.add_parser("run")
    run.add_argument("--archive", type=Path, required=True)
    run.add_argument("--archive-sha256", required=True)
    run.add_argument("--checkpoint-archive", type=Path)
    run.add_argument("--work", type=Path, required=True)
    run.add_argument("--out", type=Path, required=True)
    run.add_argument("--workers", type=int, default=1)
    run.add_argument("--expected-workers", type=int, help="refuse fewer than the priced workers")
    run.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    run.add_argument("--epochs", type=int, default=60)
    run.add_argument("--smoke", action="store_true")
    run.add_argument("--check-parallel", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        catalog = prepare(args.bank, args.out, smoke=args.smoke)
        print(
            json.dumps(
                dict(
                    worlds=len(catalog["tasks"]),
                    roots=sum(len(t["indices"]) for t in catalog["tasks"]),
                )
            )
        )
    else:
        report = execute(args)
        print(json.dumps({k: v for k, v in report.items() if k not in {"world_metrics", "fits"}}))


if __name__ == "__main__":
    main()
