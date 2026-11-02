"""Resumable paired-world shard runner for Colab CLI jobs (COLAB_INPUT_DIR / COLAB_OUTPUT_DIR)."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import multiprocessing
import os
import signal
import sys
import threading
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.engine import ScientificFault
from research.watch6_l1_confirm.run import (
    DEFAULT_LOCAL_SAVED_DIR,
    _publish_world_unit,
    branch_fixture,
)
from research.world_checkpoint.checkpoint import (
    configure_cpu,
    find_latest_checkpoint,
    verify_checkpoint,
)
from research.world_checkpoint.runner import (
    DEFAULT_CHECKPOINT_INTERVAL,
    GRACE_SECONDS,
    CheckpointInterrupted,
    run_paired_world_resumable,
)
from tools.ops.package_spec import publish_immutable, publish_json, read_ledger

configure_cpu()


def _unit_tag(index: int, *, fixture: bool) -> str:
    prefix = "fixture-world" if fixture else "world"
    return f"{prefix}-{int(index):02d}"


def _resolve_world_resume_dirs(
    input_checkpoint_dir: Path | None,
    index: int,
    *,
    fixture: bool,
) -> list[Path]:
    if input_checkpoint_dir is None or not input_checkpoint_dir.is_dir():
        return []
    tag = _unit_tag(index, fixture=fixture)
    candidates = [
        input_checkpoint_dir / tag,
        input_checkpoint_dir / "checkpoints" / tag,
        input_checkpoint_dir / "efficacy" / "checkpoints" / tag,
        input_checkpoint_dir,
    ]
    return [d for d in candidates if d.is_dir()]


def stage_exported_checkpoints(
    prior_exports_dir: str | Path,
    target_input_checkpoint_dir: str | Path,
) -> list[dict[str, Any]]:
    """Stage the newest verified checkpoint per world from prior exports into inputs."""
    src_root = Path(prior_exports_dir)
    dst_root = Path(target_input_checkpoint_dir)
    if not src_root.is_dir():
        return []

    search_roots = [
        src_root / "efficacy" / "checkpoints",
        src_root / "checkpoints",
        src_root,
    ]
    world_dirs: list[Path] = []
    seen_resolved: set[Path] = set()
    for sroot in search_roots:
        if not sroot.is_dir():
            continue
        for sub in sorted(sroot.iterdir()):
            if sub.is_dir() and (
                sub.name.startswith("world-") or sub.name.startswith("fixture-world-")
            ):
                res = sub.resolve()
                if res not in seen_resolved:
                    seen_resolved.add(res)
                    world_dirs.append(sub)

    staged: list[dict[str, Any]] = []
    for wdir in world_dirs:
        latest = find_latest_checkpoint(wdir, verify_identity=False)
        if latest is None:
            continue
        sha = verify_checkpoint(latest)
        target_ckpt = dst_root / wdir.name / latest.name
        meta = publish_immutable(target_ckpt, latest.read_bytes())
        staged.append(
            {
                "world_tag": wdir.name,
                "source": str(latest),
                "dest": str(target_ckpt.relative_to(dst_root)),
                "sha256": sha,
                "size_bytes": int(meta["size"]),
            }
        )
    return staged


def resumable_world_unit(
    row: Mapping[str, Any],
    fixture: bool = False,
    *,
    output_checkpoint_root: str | Path | None = None,
    input_checkpoint_root: str | Path | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    stop_at: tuple[str, int] | None = None,
) -> dict[str, Any]:
    """Execute or resume all 3 arms of one paired world, exporting atomic mid-world checkpoints."""
    configure_cpu()
    os.environ["SNAKE_WATCH_SOURCE"] = "live"
    index = int(row["index"])
    seed = int(row["seed"])
    horizon = int(row["horizon"])
    arm_order = list(row.get("arm_order") or p.arm_order_for_index(index))
    tag = _unit_tag(index, fixture=fixture)

    out_ckpt_dir = (
        Path(output_checkpoint_root) / tag if output_checkpoint_root is not None else None
    )
    in_ckpt_dirs = _resolve_world_resume_dirs(
        Path(input_checkpoint_root) if input_checkpoint_root is not None else None,
        index,
        fixture=fixture,
    )

    try:
        payload = run_paired_world_resumable(
            index=index,
            seed=seed,
            arm_order=arm_order,
            horizon=horizon,
            fixture=fixture,
            checkpoint_dir=out_ckpt_dir,
            resume_checkpoint_dirs=in_ckpt_dirs,
            checkpoint_interval=int(checkpoint_interval),
            install_signal_handlers=True,
            stop_at=stop_at,
        )
        return {"__interrupted__": False, "payload": payload}
    except CheckpointInterrupted as exc:
        return {
            "__interrupted__": True,
            "index": index,
            "checkpoint_path": str(exc.checkpoint_path),
            "arm": exc.arm,
            "frame_index": int(exc.frame_index),
            "signal_number": exc.signal_number,
            "reason": str(exc.reason),
        }


def _read_verified_skip_indices(ledger_path: Path | None, *, fixture: bool) -> set[int]:
    if ledger_path is None or not ledger_path.is_file():
        return set()
    if fixture:
        rows = read_ledger(ledger_path, p.NAMESPACE, "watch6-fixture", {"science": p.identity()})
        return {int(row["index"]) for row in rows}
    from research.watch6_l1_confirm.admission import completed_ledger

    verified = completed_ledger(ledger_path)
    return {int(row["index"]) for row in verified}


def run(
    *,
    out: Path,
    ledger_path: Path | None,
    workers: int,
    shards: int = p.SHARDS,
    shard: int = 0,
    fixture: bool = False,
    fixture_horizon: int = 120,
    fixture_count: int = 2,
    canary_performance_out: Path | None = None,
    local_smoke: bool = False,
    input_checkpoint_dir: Path | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    stop_at_by_index: Mapping[int, tuple[str, int]] | None = None,
) -> dict[str, Any]:
    """Execute or resume a shard of worlds, publishing checkpoints under ``out/checkpoints``."""
    configure_cpu()
    if local_smoke:
        if not fixture:
            raise ValueError("--local-smoke only runs excluded operational fixture seeds")
        if "SNAKE_CKPT_DIR" not in os.environ and DEFAULT_LOCAL_SAVED_DIR.is_dir():
            os.environ["SNAKE_CKPT_DIR"] = str(DEFAULT_LOCAL_SAVED_DIR)
        os.environ.setdefault("SNAKE_WATCH_SOURCE", "live")
        skip = _read_verified_skip_indices(ledger_path, fixture=fixture)
    else:
        from research.watch6_l1_confirm.admission import completed_ledger, runtime_admit

        runtime_admit(workers=workers, shards=shards, shard=shard)
        if ledger_path is None:
            raise ValueError("admitted runtime execution requires --ledger")
        verified = completed_ledger(ledger_path)
        skip = set() if fixture else {int(row["index"]) for row in verified}

    if fixture:
        if type(fixture_count) is not int or not 1 <= fixture_count <= len(p.FIXTURE_SEEDS):
            raise ValueError(f"fixture_count must be in 1..{len(p.FIXTURE_SEEDS)}")
        if type(fixture_horizon) is not int or fixture_horizon <= 0:
            raise ValueError("fixture_horizon must be a positive integer")
        all_rows = [
            {
                "index": i,
                "seed": p.FIXTURE_SEEDS[i],
                "horizon": fixture_horizon,
                "arm_order": p.arm_order_for_index(i),
            }
            for i in range(fixture_count)
        ]
        rows = [r for r in all_rows if int(r["index"]) not in skip]
    else:
        alloc = p.allocation()
        owned = p.shard_indices(shard, shards)
        rows = [alloc[i] for i in owned if i not in skip]

    started = time.monotonic()
    last_log = started
    completed_now: list[int] = []
    interrupted_units: list[dict[str, Any]] = []
    out.mkdir(parents=True, exist_ok=True)
    output_checkpoint_root = out / "checkpoints"
    output_checkpoint_root.mkdir(parents=True, exist_ok=True)

    workload_started_path = out.parent / "workload-started.json"
    if not workload_started_path.exists():
        publish_json(
            workload_started_path,
            {
                "schema": "watch6-l1-workload-started/v1",
                "namespace": p.NAMESPACE,
                "science": p.identity(),
                "shard": shard,
                "shards": shards,
                "workers": workers,
                "fixture": bool(fixture),
                "local_smoke": bool(local_smoke),
                "pending_units": len(rows),
                "skipped_verified_total": len(skip),
                "checkpoint_interval": int(checkpoint_interval),
            },
        )

    if rows:
        pool_workers = min(max(1, int(workers)), len(rows))
        if pool_workers == 1:
            for row in rows:
                idx = int(row["index"])
                stop_spec = (
                    stop_at_by_index.get(idx) if isinstance(stop_at_by_index, Mapping) else None
                )
                unit_out = resumable_world_unit(
                    row,
                    fixture,
                    output_checkpoint_root=output_checkpoint_root,
                    input_checkpoint_root=input_checkpoint_dir,
                    checkpoint_interval=checkpoint_interval,
                    stop_at=stop_spec,
                )
                if unit_out.get("__interrupted__"):
                    interrupted_units.append(unit_out)
                    break
                payload = unit_out["payload"]
                _publish_world_unit(out, row, payload, fixture=fixture)
                completed_now.append(idx)
                print(
                    f"WATCH6_L1_UNIT index={idx:02d} fixture={fixture} "
                    f"horizon={row['horizon']} order={','.join(row['arm_order'])} "
                    f"unit_s={payload['elapsed_seconds']:.1f} "
                    f"done={len(completed_now)}/{len(rows)}",
                    flush=True,
                )
        else:
            context = multiprocessing.get_context("spawn")
            forwarded_signal: dict[str, int | None] = {"signum": None}
            previous_handlers: dict[int, Any] = {}
            is_main_thread = threading.current_thread() is threading.main_thread()

            with concurrent.futures.ProcessPoolExecutor(pool_workers, mp_context=context) as pool:

                def _forward_to_children(signum: int, _frame: Any) -> None:
                    if forwarded_signal["signum"] is None:
                        forwarded_signal["signum"] = int(signum)
                        for child in list(getattr(pool, "_processes", {}).values()):
                            if child is not None and child.pid and child.is_alive():
                                try:
                                    os.kill(child.pid, int(signum))
                                except OSError:
                                    pass

                if is_main_thread:
                    for sig in (signal.SIGUSR1, signal.SIGTERM):
                        previous_handlers[sig] = signal.getsignal(sig)
                        signal.signal(sig, _forward_to_children)

                try:
                    pending = {
                        pool.submit(
                            resumable_world_unit,
                            row,
                            fixture,
                            output_checkpoint_root=output_checkpoint_root,
                            input_checkpoint_root=input_checkpoint_dir,
                            checkpoint_interval=checkpoint_interval,
                            stop_at=(
                                stop_at_by_index.get(int(row["index"]))
                                if isinstance(stop_at_by_index, Mapping)
                                else None
                            ),
                        ): row
                        for row in rows
                    }
                    while pending:
                        wait_timeout = (
                            GRACE_SECONDS if forwarded_signal["signum"] is not None else 30.0
                        )
                        done, _ = concurrent.futures.wait(
                            pending,
                            timeout=wait_timeout,
                            return_when=concurrent.futures.FIRST_COMPLETED,
                        )
                        now = time.monotonic()
                        if not done:
                            if forwarded_signal["signum"] is not None:
                                break
                            if now - last_log >= 120:
                                print(
                                    f"WATCH6_L1_PROGRESS shard={shard}/{shards} "
                                    f"done={len(completed_now)}/{len(rows)} "
                                    f"skipped_total={len(skip)} elapsed_s={now - started:.1f}",
                                    flush=True,
                                )
                                last_log = now
                            continue
                        for future in done:
                            row = pending.pop(future)
                            unit_out = future.result()
                            if unit_out.get("__interrupted__"):
                                interrupted_units.append(unit_out)
                                continue
                            payload = unit_out["payload"]
                            _publish_world_unit(out, row, payload, fixture=fixture)
                            completed_now.append(int(row["index"]))
                            print(
                                f"WATCH6_L1_UNIT index={row['index']:02d} fixture={fixture} "
                                f"horizon={row['horizon']} order={','.join(row['arm_order'])} "
                                f"unit_s={payload['elapsed_seconds']:.1f} "
                                f"done={len(completed_now)}/{len(rows)}",
                                flush=True,
                            )
                            last_log = time.monotonic()
                except BaseException:
                    for child in list(getattr(pool, "_processes", {}).values()):
                        if child is not None and child.is_alive():
                            child.terminate()
                    raise
                finally:
                    if is_main_thread:
                        for sig, handler in previous_handlers.items():
                            signal.signal(sig, handler)

    if interrupted_units:
        first_int = interrupted_units[0]
        interrupted_summary = {
            "schema": "watch6-l1-checkpoint-interrupted/v1",
            "namespace": p.NAMESPACE,
            "science": p.identity(),
            "shard": shard,
            "shards": shards,
            "completed_indices": sorted(completed_now),
            "interrupted_units": interrupted_units,
            "elapsed_seconds": time.monotonic() - started,
        }
        int_path = out / f"checkpoint-interrupted-{int(time.time() * 1000)}.json"
        publish_json(int_path, interrupted_summary)
        raise CheckpointInterrupted(
            f"shard {shard} interrupted after saving checkpoint for world {first_int['index']}",
            checkpoint_path=Path(first_int["checkpoint_path"]),
            arm=first_int.get("arm"),
            frame_index=int(first_int.get("frame_index", 0)),
            signal_number=first_int.get("signal_number"),
            reason=str(first_int.get("reason", "interrupted")),
        )

    if fixture and canary_performance_out is not None and not canary_performance_out.exists():
        perf = branch_fixture(p.FIXTURE_SEEDS[0])
        publish_json(canary_performance_out, perf)

    summary = {
        "schema": "watch6-l1-operational/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "fixture": bool(fixture),
        "local_smoke": bool(local_smoke),
        "shard": shard,
        "shards": shards,
        "workers": workers,
        "skipped_verified_total": len(skip),
        "completed_indices": sorted(completed_now),
        "elapsed_seconds": time.monotonic() - started,
    }
    publish_json(out / "operational.json", summary)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, default=None)
    parser.add_argument("--input-checkpoint-dir", type=Path, default=None)
    parser.add_argument("--checkpoint-interval", type=int, default=DEFAULT_CHECKPOINT_INTERVAL)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--shards", type=int, default=p.SHARDS)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--fixture", action="store_true")
    parser.add_argument("--fixture-horizon", type=int, default=120)
    parser.add_argument("--fixture-count", type=int, default=2)
    parser.add_argument("--canary-performance-out", type=Path, default=None)
    parser.add_argument("--local-smoke", action="store_true")
    args = parser.parse_args(argv)
    try:
        result = run(
            out=args.out.resolve(),
            ledger_path=args.ledger.resolve() if args.ledger is not None else None,
            workers=args.workers,
            shards=args.shards,
            shard=args.shard,
            fixture=args.fixture or args.local_smoke,
            fixture_horizon=args.fixture_horizon,
            fixture_count=args.fixture_count,
            canary_performance_out=(
                args.canary_performance_out.resolve()
                if args.canary_performance_out is not None
                else None
            ),
            local_smoke=args.local_smoke,
            input_checkpoint_dir=(
                args.input_checkpoint_dir.resolve()
                if args.input_checkpoint_dir is not None
                else None
            ),
            checkpoint_interval=int(args.checkpoint_interval),
        )
        print(json.dumps(result, sort_keys=True), flush=True)
        return 0
    except CheckpointInterrupted as exc:
        print(
            json.dumps(
                {
                    "status": "checkpoint_interrupted",
                    "checkpoint_path": str(exc.checkpoint_path),
                    "arm": exc.arm,
                    "frame_index": exc.frame_index,
                    "signal": exc.signal_number,
                    "reason": exc.reason,
                },
                sort_keys=True,
            ),
            file=sys.stderr,
            flush=True,
        )
        return 75
    except ScientificFault as exc:
        print(f"WATCH6_L1_SCIENTIFIC_FAULT: {exc}", file=sys.stderr, flush=True)
        return 86


if __name__ == "__main__":
    raise SystemExit(main())
