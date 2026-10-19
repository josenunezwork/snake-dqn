"""Q1 CPU H5000 evaluator: one immutable bank shared by A--D and B8.

Each shard owns one identity (``A`` through ``D`` or ``B8``).  Consequently a
completed B8 shard is reusable for every arm only when the exact bank and
identity receipt match; analysis refuses duplicate or substituted B8 rows.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import multiprocessing
import os
import platform
import subprocess
import tempfile
import threading
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from research.q0q1_20261017 import q1_bank, q1_parity
from research.t23_r0_screen_20261017 import evaluate as t23

SCHEMA = "q1-eval-shard/v1"
ROW_SCHEMA = "q1-eval-row/v1"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
ARMS = ("A", "B", "C", "D", "B8")
PARTIAL_SCHEMA = "q1-eval-world-partials/v1"
PROGRESS_SECONDS = 120.0

_WORKER_CONTEXT: dict[str, Any] | None = None
_WORKER_CHECKPOINT: Path | None = None
_WORKER_CARRIER: tuple[str, str] | None = None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_manifest() -> dict[str, str]:
    """Pin the transitive simulator/model execution plus Q1-owned adapters."""
    root = Path(__file__).resolve().parents[2]
    result = t23.source_manifest()
    for name in ("q1_bank.py", "q1_evaluate.py", "q1_parity.py"):
        relative = "research/q0q1_20261017/" + name
        result[relative] = sha256(root / relative)
    return result


def validate_provenance(value: dict[str, Any], arm: str, seed: int, checkpoint_sha256: str) -> None:
    required = {
        "schema": "q1-training-provenance/v1",
        "arm": arm,
        "training_seed": seed,
        "endpoint": 8_000_000,
        "clean_exit": True,
        "checkpoint_sha256": checkpoint_sha256,
        "parity_pass": True,
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise ValueError("candidate provenance lacks Q1 arm/seed/8M/clean/parity binding")
    hashes = ("checkpoint_sha256", "environment_sha256", "runtime_parity_receipt_sha256")
    if value.get("tripwire_triggered") is not False or any(
        not isinstance(value.get(key), str)
        or len(value[key]) != 64
        or any(c not in "0123456789abcdef" for c in value[key])
        for key in hashes
    ):
        raise ValueError(
            "candidate provenance has tripwire or missing environment/runtime parity binding"
        )
    counters = value.get("counters")
    if not isinstance(counters, dict):
        raise ValueError("candidate provenance lacks actual counter receipt")
    decisions = counters.get("actual_learner_decisions")
    overshoot = counters.get("transition_budget_overshoot")
    bound = counters.get("registered_overshoot_bound")
    if (
        type(decisions) is not int
        or type(overshoot) is not int
        or type(bound) is not int
        or not 8_000_000 <= decisions <= 8_000_000 + bound
        or overshoot != decisions - 8_000_000
        or not 0 <= bound <= 1_000_000
    ):
        raise ValueError("candidate actual learner counter or overshoot bound is invalid")
    actors, worlds, seats = (counters.get(k) for k in ("actors", "worlds", "learner_seats"))
    if (
        any(type(v) is not int or v <= 0 for v in (actors, worlds, seats))
        or worlds != 128
        or seats != (3 if arm in ("A", "B") else 1)
        or bound != actors * worlds * seats - 1
    ):
        raise ValueError("candidate overshoot bound differs from registered actor packet")
    accounting_keys = (
        "emitted_replay_rows",
        "completed_updates",
        "pending_update_debt",
        "truncation_rows_dropped",
        "pending_nstep_rows",
    )
    if any(type(counters.get(k)) is not int or counters[k] < 0 for k in accounting_keys):
        raise ValueError("candidate replay/update/tail counters are missing or invalid")
    emitted, updates, debt, dropped, pending = (counters[k] for k in accounting_keys)
    if (
        updates == 0
        or debt >= 256
        or pending > actors * worlds * seats * 5
        or emitted != 256 * updates + debt
        or decisions - emitted != dropped + pending
    ):
        raise ValueError("candidate decision/replay/update/tail accounting does not reconcile")
    for name in ("environment", "config"):
        content = value.get(name)
        if (
            not isinstance(content, dict)
            or value.get(name + "_sha256")
            != hashlib.sha256(canonical(content).encode()).hexdigest()
        ):
            raise ValueError(f"candidate {name} hash does not bind recorded content")
    root = Path(__file__).resolve().parents[2]
    if value.get("source_sha256") != sha256(root / "research/redesign_m3c_20261008/train_m3c.py"):
        raise ValueError("candidate trainer differs from registered source")
    config = value["config"]
    required_config = {
        "seed": str(seed),
        "transitions": "8000000",
        "worlds": "128",
        "learner_seats": str(seats),
        "actors": str(actors),
        "reward": "mass",
        "kill_scale": "0.0",
        "batch_scale": "2",
        "obs_version": "bv8",
        "anchor_mode": "v8margin",
        "namespace": q1_bank.training_namespace(arm),
        "schedule_package": "mature" if arm in ("B", "D") else "restart",
        "collection_profile": "legacy" if arm in ("A", "B") else "task-aligned-n1",
    }
    if any(str(config.get(k)) != expected for k, expected in required_config.items()):
        raise ValueError("candidate recipe differs from registered arm")


def validate_candidate_provenance(
    path: Path, arm: str, seed: int, checkpoint: Path
) -> dict[str, Any]:
    value = json.loads(path.read_text())
    validate_provenance(value, arm, seed, sha256(checkpoint))
    return value


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def default_workers() -> int:
    """Leave one CPU free while using the bounded Colab CPU allocation."""
    return max(1, min((os.cpu_count() or 1) - 1, 10))


def _single_thread_math() -> None:
    """Constrain each evaluator process before it constructs its simulator context."""
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMBA_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        # PyTorch permits this only before its inter-op pool is first used.  The
        # process still has one intra-op math thread, which is the contract the
        # evaluator relies on.
        pass


def _initialize_world_worker(checkpoint: str, carrier: tuple[str, str]) -> None:
    """Build isolated one-thread state in each spawned evaluator process."""
    global _WORKER_CARRIER, _WORKER_CHECKPOINT, _WORKER_CONTEXT
    _single_thread_math()
    _WORKER_CHECKPOINT = Path(checkpoint)
    _WORKER_CARRIER = carrier
    _WORKER_CONTEXT = t23._context(1)


def _evaluate_world_worker(world: dict[str, Any]) -> dict[str, Any]:
    """Evaluate one world from a spawned process; all inputs are immutable."""
    if _WORKER_CONTEXT is None or _WORKER_CHECKPOINT is None or _WORKER_CARRIER is None:
        raise RuntimeError("Q1 evaluator worker was not initialized")
    return t23.evaluate_world(world, _WORKER_CHECKPOINT, _WORKER_CARRIER, _WORKER_CONTEXT)


def _timed_world_worker(world: dict[str, Any]) -> dict[str, Any]:
    """Return timing metadata for a reserved-world capacity probe only."""
    started = time.monotonic()
    row = _evaluate_world_worker(world)
    return {"row": row, "elapsed_seconds": time.monotonic() - started}


def _partial_paths(receipt: Path) -> tuple[Path, Path]:
    directory = receipt.with_name(receipt.stem + ".partials")
    return directory, directory / "manifest.json"


def _partial_binding(identity: Mapping[str, Any], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "world_hashes": [world["world_semantic_sha256"] for world in worlds],
    }


def _load_partials(
    manifest: Path, binding: Mapping[str, Any], count: int
) -> dict[int, dict[str, Any]]:
    """Load only hash-bound completed world units from this exact shard identity."""
    if not manifest.exists():
        return {}
    value = json.loads(manifest.read_text())
    if value.get("schema") != PARTIAL_SCHEMA or value.get("binding") != binding:
        raise RuntimeError("refusing Q1 partials with changed shard identity")
    rows: dict[int, dict[str, Any]] = {}
    for key, entry in value.get("worlds", {}).items():
        number = int(key)
        if str(number) != key or not 0 <= number < count:
            raise RuntimeError("Q1 partial world index is outside the shard plan")
        path = manifest.parent / f"world-{number:04d}.json"
        if path.is_symlink() or sha256(path) != entry.get("sha256"):
            raise RuntimeError("Q1 partial world receipt SHA-256 mismatch")
        packet = json.loads(path.read_text())
        if packet.get("binding") != binding or not isinstance(packet.get("row"), dict):
            raise RuntimeError("Q1 partial world receipt identity mismatch")
        rows[number] = packet["row"]
    return rows


def _execute_ordered(
    worlds: list[dict[str, Any]],
    *,
    worker: Callable[[dict[str, Any]], dict[str, Any]],
    initializer: Callable[..., None],
    initargs: tuple[Any, ...],
    workers: int,
    partials_dir: Path,
    binding: Mapping[str, Any],
    progress_seconds: float = PROGRESS_SECONDS,
    export_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Persist each completed world and return its rows in bank order.

    Completion order is deliberately discarded: it cannot influence any row or
    receipt byte.  ``spawn`` also prevents child processes from inheriting a
    parent Torch/Numba thread pool.
    """
    if workers < 1 or progress_seconds <= 0:
        raise ValueError("workers and progress interval must be positive")
    if export_dir is not None and not export_dir.is_dir():
        raise ValueError("partial export directory must be precreated by the queue")
    partials_dir.mkdir(parents=True, exist_ok=True)
    manifest = partials_dir / "manifest.json"
    results = _load_partials(manifest, binding, len(worlds))
    index: dict[str, Any] = {
        "schema": PARTIAL_SCHEMA,
        "binding": dict(binding),
        "worlds": {
            str(number): {"sha256": sha256(partials_dir / f"world-{number:04d}.json")}
            for number in sorted(results)
        },
    }

    def save(number: int, row: dict[str, Any]) -> None:
        path = partials_dir / f"world-{number:04d}.json"
        if path.exists():
            packet = json.loads(path.read_text())
            if packet.get("binding") != binding or packet.get("row") != row:
                raise RuntimeError("refusing to replace a Q1 completed world unit")
        else:
            _write(path, {"binding": binding, "row": row})
        index["worlds"][str(number)] = {"sha256": sha256(path)}
        _write(manifest, index)
        if export_dir is not None:
            _write(export_dir / path.name, {"binding": binding, "row": row})
            _write(export_dir / "manifest.json", index)

    if not manifest.exists():
        _write(manifest, index)
    if export_dir is not None:
        # A resumed attempt exports its validated retained units before fresh
        # work. Its durable manifest must never reference absent world files.
        for number, row in sorted(results.items()):
            _write(export_dir / f"world-{number:04d}.json", {"binding": binding, "row": row})
        _write(export_dir / "manifest.json", index)

    started = time.monotonic()
    reused = len(results)

    def progress() -> None:
        elapsed = time.monotonic() - started
        fresh = len(results) - reused
        eta = elapsed / fresh * (len(worlds) - len(results)) if fresh else None
        print(
            f"Q1 evaluation worlds {len(results)}/{len(worlds)} done (reused={reused}) "
            f"elapsed={elapsed:.1f}s ETA={f'{eta:.1f}s' if eta is not None else 'pending'}",
            flush=True,
        )

    progress()
    pending = iter(number for number in range(len(worlds)) if number not in results)
    if workers == 1:
        initializer(*initargs)
        stop = threading.Event()
        heartbeat = threading.Thread(
            target=lambda: _heartbeat(stop, progress, progress_seconds), daemon=True
        )
        heartbeat.start()
        try:
            for number in pending:
                row = worker(worlds[number])
                save(number, row)
                results[number] = row
        finally:
            stop.set()
            heartbeat.join()
    else:
        pool = ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=initializer,
            initargs=initargs,
        )
        running: dict[Any, int] = {}
        next_progress = time.monotonic() + progress_seconds
        try:

            def submit() -> None:
                for _ in range(workers - len(running)):
                    number = next(pending, None)
                    if number is None:
                        return
                    running[pool.submit(worker, worlds[number])] = number

            submit()
            while running:
                done, _ = wait(
                    running,
                    timeout=max(0.0, next_progress - time.monotonic()),
                    return_when=FIRST_COMPLETED,
                )
                for future in done:
                    number = running.pop(future)
                    row = future.result()
                    save(number, row)
                    results[number] = row
                if time.monotonic() >= next_progress:
                    progress()
                    next_progress = time.monotonic() + progress_seconds
                submit()
        finally:
            for future in running:
                future.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
    progress()
    return [results[number] for number in range(len(worlds))]


def _heartbeat(stop: threading.Event, progress: Callable[[], None], seconds: float) -> None:
    while not stop.wait(seconds):
        progress()


def _split(text: str) -> tuple[int, int]:
    a, b = text.split("/", 1)
    index, count = int(a), int(b)
    if not 0 <= index < count:
        raise ValueError("shard index is outside shard count")
    return index, count


def shard_worlds(worlds: list[dict[str, Any]], index: int, count: int) -> list[dict[str, Any]]:
    return [
        x
        for i, x in enumerate(sorted(worlds, key=lambda x: (x["mix"], x["world_seed"])))
        if i % count == index
    ]


def receipt_path(root: Path, seed: int, arm: str, index: int, count: int) -> Path:
    return root / "evaluation" / f"seed-{seed}" / arm / f"shard-{index}-of-{count}.json"


def valid_completed_shard(path: Path, identity: dict[str, Any]) -> bool:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    rows = value.get("rows")
    return (
        value.get("schema") == SCHEMA
        and value.get("status") == "completed"
        and value.get("clean_exit") is True
        and value.get("identity") == identity
        and isinstance(rows, list)
        and len(rows) == identity["world_count"]
        and value.get("rows_sha256") == hashlib.sha256(canonical(rows).encode()).hexdigest()
        and all(x.get("schema") == ROW_SCHEMA for x in rows)
    )


def timing_probe(
    *,
    out: Path,
    bank_path: Path,
    b8: Path,
    parity_path: Path,
    workers: int,
    repeats: int = 2,
) -> Path:
    """Time the reserved four-world B8 parity bank without creating Q1 evidence.

    The result deliberately contains only duration and row hashes.  It is a
    capacity measurement, never a Q1 score or a reusable scientific shard.
    """
    if out.exists():
        raise RuntimeError("refusing to replace a Q1 timing probe receipt")
    if out.with_name(out.stem + ".partials").exists():
        raise RuntimeError("timing requires a fresh partial directory; resumed time is invalid")
    if workers < 1 or repeats < 1:
        raise ValueError("Q1 evaluation workers and timing-probe repeats must be positive integers")
    _single_thread_math()
    bank = q1_bank.load_world_bank(bank_path)
    base_worlds = list(bank["parity_worlds"])
    # The repeated reserved worlds are capacity-only work.  Keep their task
    # indices so persistence and ordered merge cannot conflate duplicate IDs.
    worlds = [
        {**world, "timing_probe_index": repeat}
        for repeat in range(repeats)
        for world in base_worlds
    ]
    audit_path = bank_path.with_name("Q1_SEED_AUDIT.json")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank["allocation_semantic_sha256"]
        or audit.get("world_bank_file", {}).get("sha256") != sha256(bank_path)
    ):
        raise ValueError("Q1 collision audit is missing or does not bind the supplied bank")
    bank_sha = hashlib.sha256(canonical(bank).encode()).hexdigest()
    b8_sha = sha256(b8)
    if b8_sha != B8_SHA256:
        raise ValueError("B8 checkpoint does not match the released B8 SHA")
    parity = json.loads(parity_path.read_text())
    q1_parity.validate_runtime_receipt(
        parity, bank_sha256=bank_sha, b8_sha256=b8_sha, source_manifest=source_manifest()
    )
    if parity["runtime_artifact"]["runtime"] != q1_parity.runtime_environment():
        raise ValueError("parity receipt belongs to a different evaluation runtime")
    context = t23._context(1)
    profile = str(context["profile"].digest)
    if bank["profile"].get("base_profile_digest") != profile:
        raise ValueError("Q1 bank profile differs from active evaluation profile")
    carrier = context["lookup"][
        __import__(
            "research.redesign_scope_20261007.grid_h5000_identity", fromlist=["FRP3_S12"]
        ).FRP3_S12[1]
    ]
    # Execute the exact pre-A4 serial loop, loaded from the registered Git
    # revision, against reserved worlds only. Never rewrite historical evidence.
    original = subprocess.run(
        ["git", "show", "17a927c9:research/q0q1_20261017/q1_evaluate.py"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        check=True,
    ).stdout
    historical_manifest = source_manifest()
    historical_manifest["research/q0q1_20261017/q1_evaluate.py"] = hashlib.sha256(
        original
    ).hexdigest()
    with tempfile.TemporaryDirectory(prefix="q1-a3-oracle-") as temporary:
        oracle_path = Path(temporary) / "oracle.py"
        oracle_path.write_bytes(original)
        spec = importlib.util.spec_from_file_location("q1_a3_oracle", oracle_path)
        oracle = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(oracle)
        old_context = oracle.t23._context(1)
        old_rows = [
            oracle.t23.evaluate_world(world, b8, carrier, old_context) for world in base_worlds
        ]
    serial = _execute_ordered(
        base_worlds,
        worker=_timed_world_worker,
        initializer=_initialize_world_worker,
        initargs=(str(b8), carrier),
        workers=1,
        partials_dir=out.with_name(out.stem + ".serial-partials"),
        binding={"kind": "timing-serial", "source_manifest": source_manifest()},
    )
    serial_rows = [row["row"] for row in serial] * repeats
    historical_rows = old_rows * repeats
    started = time.monotonic()
    rows = _execute_ordered(
        worlds,
        worker=_timed_world_worker,
        initializer=_initialize_world_worker,
        initargs=(str(b8), carrier),
        workers=workers,
        partials_dir=out.with_name(out.stem + ".partials"),
        binding={
            "kind": "timing-probe",
            "bank_sha256": bank_sha,
            "b8_sha256": b8_sha,
            "world_hashes": [
                {
                    "probe_index": world["timing_probe_index"],
                    "sha256": world["world_semantic_sha256"],
                }
                for world in worlds
            ],
            "parity_sha256": hashlib.sha256(canonical(parity).encode()).hexdigest(),
            "source_manifest": source_manifest(),
        },
    )
    elapsed = time.monotonic() - started
    parallel_rows = [row["row"] for row in rows]
    if canonical(parallel_rows) != canonical(serial_rows) or canonical(serial_rows) != canonical(
        historical_rows
    ):
        raise RuntimeError("real historical serial/current serial/spawn evaluator bytes differ")
    _write(
        out,
        {
            "schema": "q1-eval-timing-probe/v1",
            "status": "completed",
            "purpose": "capacity measurement only; not Q1 decision evidence",
            "world_count": len(worlds),
            "parity_world_count": len(base_worlds),
            "repeats": repeats,
            "workers": workers,
            "elapsed_seconds": elapsed,
            "runtime": q1_parity.runtime_environment(),
            "source_manifest": source_manifest(),
            "historical_source_manifest": historical_manifest,
            "serial_row_sha256": [
                hashlib.sha256(canonical(row).encode()).hexdigest() for row in serial_rows
            ],
            "historical_row_sha256": [
                hashlib.sha256(canonical(row).encode()).hexdigest() for row in historical_rows
            ],
            "serial_unit_elapsed_seconds": [row["elapsed_seconds"] for row in serial] * repeats,
            "bank_sha256": bank_sha,
            "b8_sha256": b8_sha,
            "parity_sha256": hashlib.sha256(canonical(parity).encode()).hexdigest(),
            "row_sha256": [
                hashlib.sha256(canonical(result["row"]).encode()).hexdigest() for result in rows
            ],
            "unit_elapsed_seconds": [result["elapsed_seconds"] for result in rows],
        },
    )
    return out


def run(
    *,
    root: Path,
    bank_path: Path,
    b8: Path,
    checkpoint: Path,
    arm: str,
    seed: int,
    shard: str,
    workers: int | None = None,
    partials_export: Path | None = None,
    provenance_path: Path | None = None,
    parity_path: Path | None = None,
) -> Path:
    if arm not in ARMS or seed not in q1_bank.TRAINING_SEEDS:
        raise ValueError("unregistered Q1 arm or seed")
    index, count = _split(shard)
    bank = q1_bank.load_world_bank(bank_path)
    worlds = shard_worlds(q1_bank.worlds_for_seed(bank, seed), index, count)
    if workers is None:
        workers = default_workers()
    if workers < 1:
        raise ValueError("Q1 evaluation workers must be a positive integer")
    _single_thread_math()
    context = t23._context(1)
    profile = str(context["profile"].digest)
    if bank["profile"].get("base_profile_digest") != profile:
        raise ValueError("Q1 bank profile differs from active evaluation profile")
    audit_path = bank_path.with_name("Q1_SEED_AUDIT.json")
    audit = json.loads(audit_path.read_text())
    if (
        audit.get("pass") is not True
        or audit.get("world_bank_semantic_sha256") != bank["allocation_semantic_sha256"]
        or audit.get("world_bank_file", {}).get("sha256") != sha256(bank_path)
    ):
        raise ValueError("Q1 collision audit is missing or does not bind the supplied bank")
    q1_bank_digest = hashlib.sha256(canonical(bank).encode()).hexdigest()
    checkpoint_sha = sha256(checkpoint)
    b8_sha = sha256(b8)
    if b8_sha != B8_SHA256:
        raise ValueError("B8 checkpoint does not match the released B8 SHA")
    if arm == "B8" and checkpoint_sha != b8_sha:
        raise ValueError("B8 identity must use the pinned B8 checkpoint bytes")
    if parity_path is None:
        raise ValueError("every Q1 evaluation identity requires --parity-receipt")
    parity = json.loads(parity_path.read_text())
    q1_parity.validate_runtime_receipt(
        parity, bank_sha256=q1_bank_digest, b8_sha256=b8_sha, source_manifest=source_manifest()
    )
    if parity["runtime_artifact"]["runtime"] != q1_parity.runtime_environment():
        raise ValueError("parity receipt belongs to a different evaluation runtime")
    provenance = (
        None
        if arm == "B8"
        else validate_candidate_provenance(Path(provenance_path or ""), arm, seed, checkpoint)
    )
    identity = {
        "training_seed": seed,
        "arm": arm,
        "checkpoint_sha256": checkpoint_sha,
        "b8_sha256": b8_sha,
        "endpoint": 8_000_000,
        "shard": index,
        "shards": count,
        "world_count": len(worlds),
        "world_hashes": [x["world_semantic_sha256"] for x in worlds],
        "bank_sha256": q1_bank_digest,
        "profile_digest": profile,
        "evaluator_source_sha256": sha256(Path(__file__)),
        "evaluator_source_manifest": source_manifest(),
        "runtime_parity_receipt_sha256": hashlib.sha256(canonical(parity).encode()).hexdigest(),
        "environment_sha256": parity["environment"]["sha256"],
        "training_provenance_sha256": (
            None
            if provenance is None
            else hashlib.sha256(canonical(provenance).encode()).hexdigest()
        ),
    }
    path = receipt_path(root, seed, arm, index, count)
    if valid_completed_shard(path, identity):
        return path
    if path.exists():
        raise RuntimeError("refusing to replace a Q1 shard with changed identity")
    carrier = context["lookup"][
        __import__(
            "research.redesign_scope_20261007.grid_h5000_identity", fromlist=["FRP3_S12"]
        ).FRP3_S12[1]
    ]

    def decorate(world: dict[str, Any], value: dict[str, Any]) -> dict[str, Any]:
        value.update(
            {
                "schema": ROW_SCHEMA,
                "arm": arm,
                "training_seed": seed,
                "checkpoint_step": 8_000_000,
                "checkpoint_sha256": checkpoint_sha,
                "b8_checkpoint_sha256": identity["b8_sha256"],
                "world_bank_sha256": q1_bank_digest,
                "world_profile_sha256": profile,
                "initial_world_sha256": world["initial_world_sha256"],
                "evaluator_source_sha256": identity["evaluator_source_sha256"],
                "clean_exit": True,
            }
        )
        return value

    partials_dir, _ = _partial_paths(path)
    raw_rows = _execute_ordered(
        worlds,
        worker=_evaluate_world_worker,
        initializer=_initialize_world_worker,
        initargs=(str(checkpoint), carrier),
        workers=workers,
        partials_dir=partials_dir,
        binding=_partial_binding(identity, worlds),
        export_dir=partials_export,
    )
    rows = [decorate(world, value) for world, value in zip(worlds, raw_rows)]
    _write(
        path,
        {
            "schema": SCHEMA,
            "status": "completed",
            "clean_exit": True,
            "identity": identity,
            "runtime_parity": parity,
            "training_provenance": provenance,
            "rows": rows,
            "rows_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "engine": {"cpu_only": True, "horizon": 5000, "terminal_hero": True},
            "platform": platform.platform(),
        },
    )
    return path


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bank", type=Path, default=Path(__file__).with_name("Q1_WORLD_BANK.json"))
    p.add_argument("--b8-checkpoint", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--shard")
    p.add_argument("--workers", type=int, default=default_workers())
    p.add_argument("--partials-export", type=Path)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--parity-receipt", type=Path, required=True)
    p.add_argument(
        "--timing-probe-out",
        type=Path,
        help="measure only the four reserved B8 parity worlds; writes no scored Q1 shard",
    )
    p.add_argument("--timing-probe-repeats", type=int, default=2)
    a = p.parse_args(argv)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    if a.timing_probe_out is not None:
        if a.arm != "B8" or a.checkpoint != a.b8_checkpoint:
            p.error(
                "--timing-probe-out requires --arm B8 and --checkpoint equal to --b8-checkpoint"
            )
        print(
            timing_probe(
                out=a.timing_probe_out,
                bank_path=a.bank,
                b8=a.b8_checkpoint,
                parity_path=a.parity_receipt,
                workers=a.workers,
                repeats=a.timing_probe_repeats,
            )
        )
    else:
        if a.seed is None or a.shard is None:
            p.error("--seed and --shard are required for a scored Q1 shard")
        print(
            run(
                root=a.root,
                bank_path=a.bank,
                b8=a.b8_checkpoint,
                checkpoint=a.checkpoint,
                arm=a.arm,
                seed=a.seed,
                shard=a.shard,
                workers=a.workers,
                partials_export=a.partials_export,
                provenance_path=a.provenance,
                parity_path=a.parity_receipt,
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
