#!/usr/bin/env python3
"""Mac-only, resumable terminal-hero evaluator for S1.

Every invocation evaluates one ``(arm, training seed)`` identity on all 128
registered H5000 worlds.  It writes one atomic receipt per completed world and
always merges those units in bank order; therefore the two-worker result is
byte-identical to the serial result.  This is development evidence only.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import multiprocessing
import os
import platform
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from typing import Any, Mapping, Sequence

from research.s1_mature_20261018 import bank
from research.t23_r0_screen_20261017 import evaluate as t23

SCHEMA = "s1-eval-shard/v1"
ROW_SCHEMA = "s1-eval-row/v1"
PARTIAL_SCHEMA = "s1-eval-world-partials/v1"
ARMS = ("N1", "N3", "B8")
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
MAX_WORKERS = 2
PROGRESS_SECONDS = 120.0

_CONTEXT: dict[str, Any] | None = None
_CHECKPOINT: Path | None = None
_CARRIER: tuple[str, str] | None = None


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False, suffix=".tmp") as handle:
        handle.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")
        temporary = Path(handle.name)
    temporary.replace(path)


def source_manifest() -> dict[str, str]:
    root = Path(__file__).resolve().parents[2]
    result = t23.source_manifest()
    for relative in (
        "research/s1_mature_20261018/evaluate.py",
        "research/s1_mature_20261018/bank.py",
    ):
        result[relative] = sha256(root / relative)
    return result


def _single_thread_math() -> None:
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _init(checkpoint: str, carrier: tuple[str, str]) -> None:
    global _CONTEXT, _CHECKPOINT, _CARRIER
    _single_thread_math()
    _CHECKPOINT, _CARRIER, _CONTEXT = Path(checkpoint), carrier, t23._context(1)


def _world(world: dict[str, Any]) -> dict[str, Any]:
    if _CONTEXT is None or _CHECKPOINT is None or _CARRIER is None:
        raise RuntimeError("S1 evaluator worker was not initialized")
    return t23.evaluate_world(world, _CHECKPOINT, _CARRIER, _CONTEXT)


def _timed_world(world: dict[str, Any]) -> dict[str, Any]:
    started = time.monotonic()
    return {"row": _world(world), "elapsed_seconds": time.monotonic() - started}


def _audit_path(bank_path: Path) -> Path:
    # The study-owned allocator exports the fixed name; retaining a sibling
    # lookup makes a copied immutable bank usable in an operator result root.
    sibling = bank_path.with_name(bank.AUDIT_OUTPUT.name)
    return sibling if sibling.exists() else Path(bank.AUDIT_OUTPUT)


def _validate_bank(bank_path: Path, value: dict[str, Any]) -> None:
    audit = json.loads(_audit_path(bank_path).read_text())
    if audit.get("pass") is not True:
        raise ValueError("S1 collision audit is absent or failed")
    if audit.get("world_bank_semantic_sha256") != value.get("allocation_semantic_sha256"):
        raise ValueError("S1 collision audit does not bind this allocation")
    if audit.get("world_bank_file", {}).get("sha256") != sha256(bank_path):
        raise ValueError("S1 collision audit does not bind bank bytes")
    context = t23._context(1)
    if value.get("profile", {}).get("base_profile_digest") != str(context["profile"].digest):
        raise ValueError("S1 bank is not for the active H5000 profile")


def _partial_binding(identity: Mapping[str, Any], worlds: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "identity": dict(identity),
        "world_hashes": [w["world_semantic_sha256"] for w in worlds],
    }


def _load_partials(
    manifest: Path, binding: Mapping[str, Any], count: int
) -> dict[int, dict[str, Any]]:
    if not manifest.exists():
        return {}
    value = json.loads(manifest.read_text())
    if value.get("schema") != PARTIAL_SCHEMA or value.get("binding") != binding:
        raise RuntimeError("refusing S1 partials with changed identity")
    results: dict[int, dict[str, Any]] = {}
    declared = value.get("worlds", {})
    for key, entry in declared.items():
        index = int(key)
        path = manifest.parent / f"world-{index:04d}.json"
        if (
            str(index) != key
            or not 0 <= index < count
            or path.is_symlink()
            or sha256(path) != entry.get("sha256")
        ):
            raise RuntimeError("S1 partial hash or index is invalid")
        packet = json.loads(path.read_text())
        if packet.get("binding") != binding or not isinstance(packet.get("row"), dict):
            raise RuntimeError("S1 partial binding is invalid")
        results[index] = packet["row"]
    # A world file is published before the manifest update.  A process crash
    # in that narrow interval leaves an immutable, hash-bound orphan.  Adopt
    # only a correctly named and correctly bound orphan; arbitrary files are
    # never silently ignored or overwritten on resume.
    for path in manifest.parent.glob("world-*.json"):
        stem = path.stem.removeprefix("world-")
        if len(stem) != 4 or not stem.isdecimal():
            raise RuntimeError("S1 partial directory contains malformed world filename")
        number = int(stem)
        if number in results:
            continue
        if not 0 <= number < count or path.is_symlink():
            raise RuntimeError("S1 orphan partial world index is invalid")
        packet = json.loads(path.read_text())
        if packet.get("binding") != binding or not isinstance(packet.get("row"), dict):
            raise RuntimeError("S1 orphan partial does not match this identity")
        results[number] = packet["row"]
    return results


def execute_ordered(
    worlds: list[dict[str, Any]],
    *,
    checkpoint: Path,
    carrier: tuple[str, str],
    workers: int,
    partials_dir: Path,
    binding: Mapping[str, Any],
    timed: bool = False,
) -> list[dict[str, Any]]:
    """Run/reuse worlds and return results in registered order, never completion order."""
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError(f"S1 Mac evaluation permits 1..{MAX_WORKERS} workers")
    partials_dir.mkdir(parents=True, exist_ok=True)
    # One writer per identity.  flock is released automatically if a crashed
    # process dies, unlike a marker file that could deadlock recovery.
    lock_handle = (partials_dir / ".writer.lock").open("a+")
    try:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock_handle.close()
        raise RuntimeError("another S1 evaluator owns these partials") from exc
    manifest = partials_dir / "manifest.json"
    results = _load_partials(manifest, binding, len(worlds))
    index = {"schema": PARTIAL_SCHEMA, "binding": dict(binding), "worlds": {}}
    for number in sorted(results):
        index["worlds"][str(number)] = {"sha256": sha256(partials_dir / f"world-{number:04d}.json")}
    if not manifest.exists():
        _write(manifest, index)

    def save(number: int, row: dict[str, Any]) -> None:
        path = partials_dir / f"world-{number:04d}.json"
        if path.exists():
            raise RuntimeError("refusing to overwrite a completed S1 world")
        _write(path, {"binding": binding, "row": row})
        index["worlds"][str(number)] = {"sha256": sha256(path)}
        _write(manifest, index)

    pending = iter(i for i in range(len(worlds)) if i not in results)
    worker = _timed_world if timed else _world
    if workers == 1:
        _init(str(checkpoint), carrier)
        for number in pending:
            results[number] = worker(worlds[number])
            save(number, results[number])
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=multiprocessing.get_context("spawn"),
            initializer=_init,
            initargs=(str(checkpoint), carrier),
        ) as pool:
            running: dict[Any, int] = {}

            def submit() -> None:
                while len(running) < workers:
                    number = next(pending, None)
                    if number is None:
                        return
                    running[pool.submit(worker, worlds[number])] = number

            submit()
            while running:
                done, _ = wait(running, return_when=FIRST_COMPLETED)
                for future in done:
                    number = running.pop(future)
                    results[number] = future.result()
                    save(number, results[number])
                submit()
    return [results[i] for i in range(len(worlds))]


def _carrier() -> tuple[str, str]:
    context = t23._context(1)
    from research.redesign_scope_20261007.grid_h5000_identity import FRP3_S12

    return context["lookup"][FRP3_S12[1]]


def _validate_provenance(
    provenance: Mapping[str, Any], arm: str, seed: int, checkpoint: Path
) -> None:
    if arm == "B8":
        if sha256(checkpoint) != B8_SHA256:
            raise ValueError("B8 checkpoint SHA differs from registered comparator")
        return
    expected_actors, expected_seats = (3, 1) if arm == "N1" else (6, 3)
    required = {
        "schema": "s1-training-provenance/v1",
        "arm": arm,
        "training_seed": seed,
        "endpoint": 8_000_000,
        "clean_exit": True,
        "tripwire_triggered": False,
        "checkpoint_sha256": sha256(checkpoint),
        "trainer_source_sha256": provenance.get("trainer_source_sha256"),
        "runtime_parity_receipt_sha256": provenance.get("runtime_parity_receipt_sha256"),
    }
    if any(provenance.get(k) != v for k, v in required.items()):
        raise ValueError("S1 final training provenance does not bind arm/seed/endpoint/checkpoint")
    if any(
        not isinstance(provenance.get(k), str) or len(provenance[k]) != 64
        for k in ("trainer_source_sha256", "runtime_parity_receipt_sha256")
    ):
        raise ValueError("S1 final provenance lacks trainer/runtime hash bindings")
    recipe = provenance.get("recipe")
    expected_recipe = {
        "worlds": 128,
        "actors": expected_actors,
        "learner_seats": expected_seats,
        "schedule_package": "mature",
        "collection_profile": "task-aligned-n1" if arm == "N1" else "legacy",
        "transitions": 8_000_000,
        "namespace": bank.training_namespace(arm, seed),
    }
    if not isinstance(recipe, Mapping) or any(
        recipe.get(k) != v for k, v in expected_recipe.items()
    ):
        raise ValueError("S1 final provenance recipe differs from registration")
    counters = provenance.get("counters")
    if not isinstance(counters, Mapping):
        raise ValueError("S1 final provenance lacks actual training counters")
    decisions = counters.get("actual_learner_decisions")
    overshoot = counters.get("transition_budget_overshoot")
    bound = counters.get("registered_overshoot_bound")
    if (
        type(decisions) is not int
        or type(overshoot) is not int
        or type(bound) is not int
        or not 8_000_000 <= decisions <= 8_000_000 + bound
        or overshoot != decisions - 8_000_000
        or bound != expected_actors * 128 * expected_seats - 1
    ):
        raise ValueError("S1 final provenance learner counter/overshoot bound is invalid")


def run(
    *,
    root: Path,
    bank_path: Path,
    checkpoint: Path,
    arm: str,
    seed: int,
    workers: int = 2,
    provenance_path: Path | None = None,
) -> Path:
    if arm not in ARMS or seed not in bank.TRAINING_SEEDS:
        raise ValueError("unregistered S1 arm or seed")
    if not 1 <= workers <= MAX_WORKERS:
        raise ValueError("S1 permits at most two single-thread workers")
    _single_thread_math()
    value = bank.load_world_bank(bank_path)
    _validate_bank(bank_path, value)
    worlds = bank.worlds_for_seed(value, seed)
    provenance = None if arm == "B8" else json.loads(Path(provenance_path or "").read_text())
    _validate_provenance(provenance or {}, arm, seed, checkpoint)
    identity = {
        "arm": arm,
        "training_seed": seed,
        "endpoint": 8_000_000,
        "checkpoint_sha256": sha256(checkpoint),
        "b8_sha256": B8_SHA256,
        "bank_sha256": hashlib.sha256(canonical(value).encode()).hexdigest(),
        "world_hashes": [w["world_semantic_sha256"] for w in worlds],
        "world_count": len(worlds),
        "source_manifest": source_manifest(),
    }
    out = root / "evaluation" / f"seed-{seed}" / f"{arm}.json"
    if out.exists():
        old = json.loads(out.read_text())
        if old.get("identity") == identity and old.get("status") == "completed":
            return out
        raise RuntimeError("refusing to replace an S1 evaluation receipt")
    raw = execute_ordered(
        worlds,
        checkpoint=checkpoint,
        carrier=_carrier(),
        workers=workers,
        partials_dir=out.with_name(out.stem + ".partials"),
        binding=_partial_binding(identity, worlds),
    )
    rows = []
    for world, row in zip(worlds, raw):
        rows.append(
            {
                **row,
                "schema": ROW_SCHEMA,
                "arm": arm,
                "training_seed": seed,
                "checkpoint_step": 8_000_000,
                "checkpoint_sha256": identity["checkpoint_sha256"],
                "b8_checkpoint_sha256": B8_SHA256,
                "world_bank_sha256": identity["bank_sha256"],
                "evaluator_source_sha256": source_manifest()[
                    "research/s1_mature_20261018/evaluate.py"
                ],
                "clean_exit": True,
            }
        )
    _write(
        out,
        {
            "schema": SCHEMA,
            "status": "completed",
            "clean_exit": True,
            "identity": identity,
            "training_provenance": provenance,
            "rows": rows,
            "rows_sha256": hashlib.sha256(canonical(rows).encode()).hexdigest(),
            "engine": {"cpu_only": True, "terminal_hero": True, "horizon": 5000},
            "platform": platform.platform(),
        },
    )
    return out


def timing_probe(*, out: Path, bank_path: Path, checkpoint: Path, workers: int = 2) -> Path:
    """Measure >=10 reserved disjoint worlds; receipt contains hashes/times only."""
    if out.exists():
        raise RuntimeError("refusing to overwrite S1 timing receipt")
    _single_thread_math()
    value = bank.load_world_bank(bank_path)
    _validate_bank(bank_path, value)
    worlds = list(value.get("timing_worlds", []))
    if len(worlds) < 10:
        raise ValueError("S1 timing fixture requires >=10 registered disjoint worlds")
    identity = {
        "kind": "timing",
        "bank_sha256": hashlib.sha256(canonical(value).encode()).hexdigest(),
        "checkpoint_sha256": sha256(checkpoint),
        "world_hashes": [w["world_semantic_sha256"] for w in worlds],
        "source_manifest": source_manifest(),
    }
    parallel = execute_ordered(
        worlds,
        checkpoint=checkpoint,
        carrier=_carrier(),
        workers=workers,
        partials_dir=out.with_name(out.stem + ".partials"),
        binding=identity,
        timed=True,
    )
    serial = execute_ordered(
        worlds,
        checkpoint=checkpoint,
        carrier=_carrier(),
        workers=1,
        partials_dir=out.with_name(out.stem + ".serial-partials"),
        binding={**identity, "mode": "serial"},
        timed=True,
    )
    if canonical([x["row"] for x in parallel]) != canonical([x["row"] for x in serial]):
        raise RuntimeError("S1 spawned merge differs from serial reference")
    _write(
        out,
        {
            "schema": "s1-eval-timing/v1",
            "status": "completed",
            "purpose": "capacity measurement only; no scientific outcomes",
            "identity": identity,
            "workers": workers,
            "episode_count": len(worlds),
            "parallel_elapsed_seconds": [x["elapsed_seconds"] for x in parallel],
            "serial_elapsed_seconds": [x["elapsed_seconds"] for x in serial],
            "row_sha256": [
                hashlib.sha256(canonical(x["row"]).encode()).hexdigest() for x in parallel
            ],
        },
    )
    return out


def main(argv: Sequence[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--bank", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--seed", type=int)
    p.add_argument("--provenance", type=Path)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--timing-probe-out", type=Path)
    a = p.parse_args(argv)
    if a.timing_probe_out:
        print(
            timing_probe(
                out=a.timing_probe_out, bank_path=a.bank, checkpoint=a.checkpoint, workers=a.workers
            )
        )
        return 0
    if a.seed is None:
        p.error("--seed is required for a scored identity")
    print(
        run(
            root=a.root,
            bank_path=a.bank,
            checkpoint=a.checkpoint,
            arm=a.arm,
            seed=a.seed,
            workers=a.workers,
            provenance_path=a.provenance,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
