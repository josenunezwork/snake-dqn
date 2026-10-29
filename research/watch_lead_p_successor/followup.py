"""Missing-cell and natural long-body timing canary; no scientific look."""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import json
import queue
import time
from pathlib import Path
from unittest.mock import patch

from research.bl_served_search.exports import export_json
from research.watch_lead_p import engine
from research.watch_lead_p.live import LiveWorld
from research.watch_lead_p_successor.prepare import bank_record
from research.watch_lead_p_successor.profile import (
    WINDOWS,
    Workers,
    coordinator_signals,
    host_record,
    measurement,
    prepare_roots,
    sweep,
    task,
)

BASE_WORK_SECONDS = 600
WORK_SECONDS = 2 * BASE_WORK_SECONDS


def scan(seed: int, frames: int, *, scratch: Path) -> dict:
    """Time a full bare session; select roots only by frozen engineering size/time rules.

    Sample each 250 ticks. Peak = first largest alive body; late = latest alive
    L600 sample. Outcomes are never returned. Roots remain local to this runtime.
    """
    begun, cpu = time.monotonic(), time.process_time()
    world = LiveWorld(seed)
    construction = time.monotonic() - begun
    original_step = world.step
    payloads, selected = {}, {}
    snapshot_s = 0.0
    chunks = []
    chunk_started, chunk_cpu = time.monotonic(), time.process_time()
    snapshot_at_chunk = 0.0

    def step(forced: int | None = None) -> dict:
        nonlocal snapshot_s, chunk_started, chunk_cpu, snapshot_at_chunk
        event = original_step(forced)
        if (world.frame % 250 == 0 or world.frame == frames) and world.hero.is_alive:
            length = int(world.hero.length)
            labels = []
            if length > selected.get("peak", {}).get("length", -1):
                labels.append("peak")
            if length >= 600:
                labels.append("late")
            if labels:
                started = time.monotonic()
                # Do not serialize this observer's closure into the root.
                del world.step
                try:
                    payload = world.snapshot()
                finally:
                    world.step = step
                snapshot_s += time.monotonic() - started
                if len(payload) > 16 * 1024 * 1024:
                    raise ValueError("tail root exceeds original P transport limit")
                for label in labels:
                    payloads[label] = payload
                    selected[label] = {
                        "frame": world.frame,
                        "length": length,
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
        if world.frame % 1000 == 0 or world.frame == frames:
            now, used = time.monotonic(), time.process_time()
            chunks.append(
                {
                    "end_frame": world.frame,
                    "wall_seconds": now - chunk_started,
                    "cpu_seconds": used - chunk_cpu,
                    "snapshot_seconds": snapshot_s - snapshot_at_chunk,
                }
            )
            chunk_started, chunk_cpu, snapshot_at_chunk = now, used, snapshot_s
            print(f"P_TAIL_SCAN seed={seed} frame={world.frame}/{frames}", flush=True)
        return event

    class Factory:
        def __new__(cls, value: int) -> LiveWorld:
            if value != seed:
                raise ValueError("foreign scan seed")
            return world

    world.step = step
    with (
        patch.object(engine, "LiveWorld", Factory),
        patch.object(engine, "NAMESPACE", bank_record()["namespace"]),
    ):
        row = engine.run_arm(seed, "bare", frames=frames)
    value = measurement(row, time.monotonic() - begun, time.process_time() - cpu, 0, 0)
    value.update(
        construction_seconds=construction,
        snapshot_seconds=snapshot_s,
        chunks=chunks,
        selected=selected,
        full_horizon=frames == 6000,
    )
    paths = {}
    for label, payload in payloads.items():
        path = scratch / f"tail-{seed}-{label}.bin"
        with path.open("xb") as handle:
            handle.write(payload)
        paths[label] = str(path)
    return {"paths": paths, "metadata": value}


def detail_task(seed: int, arm: str, frames: int, payload: bytes) -> dict:
    """Exact P path with separate restore/final-digest and exhaustive CPU-tail timing.

    The extra all-legal 300-tick continuation is engineering stress, separate
    from the unchanged exact-P window. It deliberately continues after deaths;
    no branch result, action, survival, Q, mass or death count is exported.
    """
    restores = []
    final_digest_s = 0.0
    original_restore, original_digest = LiveWorld.restore, LiveWorld.digest

    def restore(value: bytes) -> LiveWorld:
        started = time.monotonic()
        result = original_restore(value)
        restores.append(time.monotonic() - started)
        return result

    def digest(world: LiveWorld) -> str:
        nonlocal final_digest_s
        started = time.monotonic()
        result = original_digest(world)
        final_digest_s = time.monotonic() - started
        return result

    with patch.object(LiveWorld, "restore", restore), patch.object(LiveWorld, "digest", digest):
        row = task(seed, arm, frames, payload)
    row.update(
        construction_seconds=restores[0],
        restore_total_seconds=sum(restores),
        final_digest_seconds=final_digest_s,
    )
    return row


def stress_task(seed: int, _arm: str, frames: int, payload: bytes) -> dict:
    """Separate full-continuation load; never substitute for exact P cost."""
    started, cpu = time.monotonic(), time.process_time()
    root = LiveWorld.restore(payload)
    initial = root.digest()
    proposal = root.proposal()
    clone_s = 0.0
    branches = 0
    trace = hashlib.sha256()
    for action, legal in enumerate(proposal["mask"]):
        if not legal:
            continue
        before = time.monotonic()
        branch = root.clone()
        clone_s += time.monotonic() - before
        for frame in range(frames):
            branch.step(action if frame == 0 else None)
        branches += 1
        trace.update(branch.digest().encode())
    if not branches:
        raise ValueError("no supported legal branch at tail root")
    ended = time.monotonic()
    row = measurement(
        {
            "seed": seed,
            "arm": "all-legal-stress",
            "frames": frames * branches,
            "initial_state": initial,
            "final_state": root.digest(),
            "trace_sha256": trace.hexdigest(),
        },
        ended - started,
        time.process_time() - cpu,
        0,
        0,
    )
    return {
        **row,
        "clone_seconds": clone_s,
        "requested_frames_per_branch": frames,
        "legal_branches": branches,
        "continues_after_death": True,
        "exact_P_search": False,
        "search_calls": 0,
        "search_seconds": 0,
        "started_monotonic": started,
        "ended_monotonic": ended,
    }


def tail_roots(
    seeds: list[int], frames: int, output: Path, deadline: float, workers: int
) -> tuple[dict, dict]:
    pool = Workers(workers, {}, seeds[0], deadline)
    roots = {"peak": {}, "late": {}}
    metadata = {}
    started = last_print = time.monotonic()
    try:
        for seed in seeds:
            pool.tasks.put((seed, "tail-scan", (seed, frames)))
        while len(metadata) < len(seeds):
            pool.check(deadline)
            try:
                seed, ok, value = pool.results.get(timeout=1)
            except queue.Empty:
                value = None
            if value is not None:
                if not ok:
                    raise RuntimeError(value)
                meta = value["metadata"]
                export_json(output / f"scan-{seed}.json", meta)
                metadata[seed] = meta
                for label, path in value["paths"].items():
                    payload = Path(path).read_bytes()
                    if hashlib.sha256(payload).hexdigest() != meta["selected"][label]["sha256"]:
                        raise ValueError("tail root hash mismatch")
                    roots[label][seed] = payload
            if time.monotonic() - last_print >= 30:
                print(f"P_TAIL_SCAN done={len(metadata)}/{len(seeds)}", flush=True)
                last_print = time.monotonic()
        export_json(
            output / "scan-summary.json",
            {
                "wall_seconds": time.monotonic() - started,
                "pool_spawn_seconds": pool.spawn_seconds,
                "worker_boot": pool.boot,
            },
        )
        return roots, metadata
    finally:
        pool.close()


def execute(bank: Path, output: Path, *, smoke: bool = False) -> None:
    if json.loads(bank.read_text()) != bank_record():
        raise ValueError("foreign engineering bank")
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "host.json", host_record())
    deadline = time.monotonic() + (120 if smoke else WORK_SECONDS)
    seeds = bank_record()["seeds"][:1] if smoke else bank_record()["seeds"]
    with coordinator_signals():
        if not smoke:
            roots = prepare_roots(seeds, output, deadline)
            jobs = [(seeds[i % len(seeds)], "P0", 1) for i in range(11)]
            for count in (8, 11):
                sweep(jobs, roots, count, output, deadline, "missing-contention")
        roots, metadata = tail_roots(
            seeds, 2 if smoke else 6000, output, deadline, 1 if smoke else 4
        )
        results = []
        for label in ("peak", "late"):
            if not roots[label]:
                continue
            # No replacement: missing roots explicitly prevent qualification.
            jobs = [
                (seed, arm, 2 if smoke else WINDOWS[arm])
                for _repeat in range(2)
                for seed in seeds
                if seed in roots[label]
                for arm in ("P0", "P48", "bare")
            ]
            results.append(
                sweep(
                    jobs,
                    roots[label],
                    1 if smoke else 4,
                    output,
                    deadline,
                    f"tail-{label}",
                    kind="tail-window",
                )
            )
            stress_jobs = [
                (seed, "all-legal-stress", 2 if smoke else 300)
                for seed in seeds
                if seed in roots[label]
            ]
            sweep(
                stress_jobs,
                roots[label],
                1 if smoke else 4,
                output,
                deadline,
                f"stress-{label}",
                kind="tail-stress",
            )
        for result in results:
            rows = result["rows"]
            half = len(rows) // 2
            keys = ("seed", "arm", "initial_state", "final_state", "trace_sha256")
            if [tuple(r[k] for k in keys) for r in rows[:half]] != [
                tuple(r[k] for k in keys) for r in rows[half:]
            ]:
                raise ValueError("repeated engineering windows differ")
        covered = not smoke and all(
            "late" in m["selected"] and m["selected"]["peak"]["length"] >= 1000
            for m in metadata.values()
        )
        export_json(
            output / "followup.json",
            {
                "engineering_only": True,
                "complete": True,
                "smoke": smoke,
                "long_body_and_late_root_support": covered,
                "repeated_window_identity": True,
                "successor_envelope_qualified": False,
                "reason": "Full searched sessions and sustained tail-pressure remain unmeasured; "
                "stress timings must be reviewed before filling envelopes.",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    try:
        execute(args.bank, args.output, smoke=args.smoke)
    finally:
        faulthandler.cancel_dump_traceback_later()


if __name__ == "__main__":
    main()
