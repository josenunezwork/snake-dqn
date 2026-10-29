"""Missing-cell and natural long-body timing canary; no scientific look."""

from __future__ import annotations

import argparse
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
        try:
            row = engine.run_arm(seed, "bare", frames=frames)
        finally:
            del world.step
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
    """Exact P path with separate restore and digest timings."""
    restores = []
    digests = []
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
        digests.append(final_digest_s)
        return result

    with (
        patch.object(LiveWorld, "restore", staticmethod(restore)),
        patch.object(LiveWorld, "digest", digest),
    ):
        row = task(seed, arm, frames, payload)
    row.update(
        construction_seconds=restores[0],
        restore_total_seconds=sum(restores),
        final_digest_seconds=final_digest_s,
        digest_calls=len(digests),
        digest_total_seconds=sum(digests),
    )
    return row


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


def execute(bank: Path, output: Path, *, smoke: bool = False, phase: str = "tail") -> None:
    if phase not in ("contention", "bare", "tail"):
        raise ValueError("foreign follow-up phase")
    if json.loads(bank.read_text()) != bank_record():
        raise ValueError("foreign engineering bank")
    output.mkdir(parents=True, exist_ok=False)
    export_json(output / "host.json", host_record())
    deadline = time.monotonic() + (120 if smoke else WORK_SECONDS)
    with coordinator_signals():
        if phase == "contention":
            seeds = bank_record()["seeds"]
            if smoke:
                from research.watch_lead_p.uploaded import uploaded_checkpoints

                with uploaded_checkpoints():
                    roots = {seeds[0]: LiveWorld(seeds[0]).snapshot()}
                seeds = seeds[:1]
            else:
                roots = prepare_roots(seeds, output, deadline)
            jobs = [(seeds[i % len(seeds)], "P0", 1) for i in range(1 if smoke else 11)]
            values = [
                sweep(jobs, roots, count, output, deadline, "missing-contention")
                for count in ((1,) if smoke else (8, 11))
            ]
            keys = ("seed", "arm", "initial_state", "final_state", "trace_sha256")
            identities = [[tuple(r[k] for k in keys) for r in v["rows"]] for v in values]
            if any(x != identities[0] for x in identities[1:]):
                raise ValueError("contention trajectory identity differs")
            covered = False
        else:
            # A fixed FIRST seed, selected before any new tail measurement.
            # This is a bounded support diagnostic, never a representative N estimate.
            seeds = bank_record()["seeds"][:1]
            roots, metadata = tail_roots(
                seeds, 2 if smoke else (6000 if phase == "bare" else 5000), output, deadline, 1
            )
            results = []
            for label in (() if phase == "bare" else ("peak", "late")):
                if not roots[label]:
                    export_json(
                        output / f"tail-{label}-w4.json",
                        {"supported": False, "rows": [], "workers": 4},
                    )
                    continue
                seed = seeds[0]
                jobs = [
                    (seed, arm, 2 if smoke else WINDOWS[arm])
                    for arm in ("P0", "P48", "bare", "bare")
                ]
                result = sweep(
                    jobs,
                    roots[label],
                    1 if smoke else 4,
                    output,
                    deadline,
                    f"tail-{label}",
                    kind="tail-window",
                )
                results.append(result)
                keys = ("initial_state", "final_state", "trace_sha256")
                if [result["rows"][2][k] for k in keys] != [result["rows"][3][k] for k in keys]:
                    raise ValueError("repeated bare engineering window differs")
            covered = (
                phase == "tail"
                and not smoke
                and all(
                    "late" in m["selected"] and m["selected"]["peak"]["length"] >= 1000
                    for m in metadata.values()
                )
                and all(
                    r["search_calls"] > 0
                    for value in results
                    for r in value["rows"]
                    if r["arm"] != "bare"
                )
            )
        export_json(
            output / "followup.json",
            {
                "engineering_only": True,
                "complete": True,
                "smoke": smoke,
                "phase": phase,
                "long_body_and_late_root_support": covered,
                "repeated_window_identity": True,
                "successor_envelope_qualified": False,
                "reason": "Full searched sessions, exhaustive branch stress and sustained "
                "tail-pressure remain unmeasured; no successor envelope qualification.",
            },
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--phase", choices=("contention", "bare", "tail"), required=True)
    args = parser.parse_args()
    execute(args.bank, args.output, smoke=args.smoke, phase=args.phase)


if __name__ == "__main__":
    main()
