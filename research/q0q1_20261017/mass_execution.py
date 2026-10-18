"""Deterministic, durable episode execution for the Q0 mass packet.

Only the parent writes shards/indexes. Receipt assembly always uses plan order,
never completion order. A fresh destination can import a hash-pinned manifest;
queue output admission remains create-only.
"""

from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping

from research.q0q1_20261017.audit_common import canonical, sha256_file

SCHEMA = "q0-mass-episode-partials/v1"


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def default_workers() -> int:
    return max(1, min((os.cpu_count() or 1) - 1, 10))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def load_partials(
    manifest: Path, expected_sha256: str, binding: Mapping[str, Any], tasks: list[dict[str, Any]]
) -> dict[int, dict[str, Any]]:
    """Reject any changed index, source, runtime, plan, identity, or episode bytes."""
    if sha256_file(manifest) != expected_sha256:
        raise ValueError("Q0 partial manifest SHA-256 mismatch")
    index = json.loads(manifest.read_text())
    if (
        index.get("schema") != SCHEMA
        or index.get("binding") != binding
        or index.get("plan_sha256") != digest(tasks)
    ):
        raise ValueError("Q0 partial source/runtime/input/plan binding mismatch")
    results = {}
    for key, entry in index["episodes"].items():
        number = int(key)
        if str(number) != key or not 0 <= number < len(tasks):
            raise ValueError("Q0 partial episode index outside plan")
        path = manifest.parent / f"episode-{number:04d}.json"
        if path.is_symlink() or sha256_file(path) != entry["sha256"]:
            raise ValueError("Q0 partial episode SHA-256 mismatch")
        packet = json.loads(path.read_text())
        if packet.get("task") != tasks[number] or packet.get("binding_sha256") != digest(binding):
            raise ValueError("Q0 partial episode identity mismatch")
        if not isinstance(packet.get("result"), dict):
            raise ValueError("Q0 partial episode result missing")
        results[number] = packet["result"]
    return results


def execute(
    tasks: list[dict[str, Any]],
    *,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    binding: Mapping[str, Any],
    workers: int,
    deadline: float,
    partials_dir: Path | None = None,
    resume_partials: Path | None = None,
    resume_sha256: str | None = None,
    progress_seconds: float = 120.0,
) -> list[dict[str, Any]]:
    """Execute independent worlds with bounded in-flight work and a live heartbeat."""
    if workers < 1 or progress_seconds <= 0:
        raise ValueError("workers and progress interval must be positive")
    if (resume_partials is None) != (resume_sha256 is None):
        raise ValueError("resuming requires both the prior manifest and its SHA-256")
    results = (
        load_partials(resume_partials, resume_sha256, binding, tasks)
        if resume_partials is not None
        else {}
    )
    index = {
        "schema": SCHEMA,
        "binding": binding,
        "plan_sha256": digest(tasks),
        "episodes": {},
    }
    if partials_dir is not None:
        partials_dir.mkdir(parents=True, exist_ok=False)

    def save(number: int, result: dict[str, Any]) -> None:
        if partials_dir is None:
            return
        path = partials_dir / f"episode-{number:04d}.json"
        atomic_json(
            path, {"task": tasks[number], "binding_sha256": digest(binding), "result": result}
        )
        index["episodes"][str(number)] = {"sha256": sha256_file(path)}
        atomic_json(partials_dir / "manifest.json", index)

    for number, result in sorted(results.items()):
        save(number, result)
    if partials_dir is not None and not results:
        atomic_json(partials_dir / "manifest.json", index)
    started = time.monotonic()
    reused = len(results)

    def progress() -> None:
        elapsed = time.monotonic() - started
        completed = len(results) - reused
        eta = elapsed / completed * (len(tasks) - len(results)) if completed else None
        print(
            f"Q0 mass episodes {len(results)}/{len(tasks)} done (reused={reused}) "
            f"elapsed={elapsed:.1f}s ETA={f'{eta:.1f}s' if eta is not None else 'pending'}",
            flush=True,
        )

    progress()
    pending_tasks = iter(number for number in range(len(tasks)) if number not in results)
    if workers == 1:
        initializer(*initargs)
        # A heartbeat thread also reports while a single long world runs.
        import threading

        stop = threading.Event()

        def heartbeat() -> None:
            while not stop.wait(progress_seconds):
                progress()

        thread = threading.Thread(target=heartbeat, daemon=True)
        thread.start()
        try:
            for number in pending_tasks:
                if time.monotonic() >= deadline:
                    raise TimeoutError("Q0 CPU runtime cap exhausted")
                result = worker(tasks[number])
                save(number, result)
                results[number] = result
        finally:
            stop.set()
            thread.join()
    else:
        # Spawn avoids inheriting Torch/Numba thread pools and scoped selector hooks.
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initializer,
            initargs=initargs,
        )
        running = {}
        next_progress = time.monotonic() + progress_seconds
        try:

            def submit() -> None:
                for _ in range(workers - len(running)):
                    number = next(pending_tasks, None)
                    if number is None:
                        break
                    running[pool.submit(worker, tasks[number])] = number

            submit()
            while running:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("Q0 CPU runtime cap exhausted")
                done, _ = wait(
                    running,
                    timeout=max(0.0, min(remaining, next_progress - time.monotonic())),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    number = running.pop(future)
                    result = future.result()
                    save(number, result)
                    results[number] = result
                if time.monotonic() >= next_progress:
                    progress()
                    next_progress = time.monotonic() + progress_seconds
                submit()
        finally:
            # Workers' frame observers enforce the same absolute deadline. Cancel
            # queued work on failure; no task beyond the bounded active set starts.
            for future in running:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    progress()
    return [results[number] for number in range(len(tasks))]
