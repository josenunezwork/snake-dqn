"""Mac-side action tapes for replaying the selected E3 T3 roots.

The original T3 archive binds observations at selected decisions, but does not
contain every action which led to them.  Re-evaluating B8 on another CPU is not
an identity-preserving substitute: a small float32 Q drift can change a tie and
therefore every later state.  This module creates a compact, hash-bound tape on
the Mac.  Consumers replay it through :func:`trace_step` and never dispatch a
policy before a selected root.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_info_value.catalog import check_path, unpack
from research.bl_info_value.compact import _tar_gzip, hydrate
from research.bl_info_value.io import export_bytes, sha256, write_json
from research.bl_info_value.reconstruct import ReconstructionError, _check_row

_TRACE_LIMIT = 2_000_000
_RUNTIME_ARRAYS = (
    "controlled",
    "_carry_state",
    "_carry_mask",
    "_carry_valid",
    "_alive_post",
    "sel_state",
    "sel_mask",
    "sel_ready",
    "sel_fresh",
)


def _digest_array(digest: "hashlib._Hash", name: str, value: np.ndarray) -> None:
    array = np.ascontiguousarray(np.asarray(value))
    digest.update(name.encode("utf-8") + b"\0")
    digest.update(array.dtype.str.encode("ascii") + b"\0")
    digest.update(repr(tuple(array.shape)).encode("ascii") + b"\0")
    digest.update(array.tobytes(order="C"))


def state_binding(world: Any) -> str:
    """Hash complete decision-state semantics, including Vector61 carry state.

    ``gid.env_digest`` owns the simulator's observable state, rings, event
    latches and PRNG.  The vector runtime's mutable arrays and trend memory are
    separate from the simulator, so bind those explicitly.  Neural weights,
    compiled kernels, feature objects, traces and scratch/cache objects are
    deliberately excluded: none carries decision-time state.
    """
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    runtime = world.runtime
    digest = hashlib.sha256()
    digest.update(b"bl-e3/trace-state/v1\0")
    digest.update(gid.env_digest(world.sim, 0).encode("ascii"))
    _digest_array(digest, "sim.frame", np.asarray(getattr(world.sim, "frame", [0]))[0:1])
    for name in ("_policy_phase_prepared", "_policy_step_in_progress"):
        digest.update(name.encode("ascii") + b"=")
        digest.update(str(bool(getattr(world.sim, name, False))).encode("ascii"))
    for name in _RUNTIME_ARRAYS:
        value = getattr(runtime, name, None)
        if value is not None:
            _digest_array(digest, name, value)
    memory = getattr(runtime, "memory", None)
    if memory is not None:
        for name in ("prev_dist", "prev_id"):
            value = getattr(memory, name, None)
            if value is None:
                raise ReconstructionError(f"runtime memory lacks {name}")
            _digest_array(digest, "memory." + name, value)
    return digest.hexdigest()


def trace_step(
    world: Any, actions: np.ndarray, *, capture_obs: bool = True
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray] | None]:
    """Advance one prepared Watch frame with a recorded six-seat action row.

    The selector intentionally contains no policy dispatch.  It retains the
    normal runtime prepare/observe hooks and B8 raster construction so a
    replayed root has precisely the same observation boundary as collection.
    """
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    recorded = np.asarray(actions, dtype=np.int64)
    if recorded.shape == (int(world.sim.S),):
        recorded = recorded.reshape(1, -1)
    if recorded.shape != (1, int(world.sim.S)):
        raise ValueError(f"trace actions must have shape {(1, int(world.sim.S))}")
    if np.any(recorded < 0) or np.any(recorded > 5):
        raise ValueError("trace actions must lie in [0, 5]")
    selected: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        # Match bank_engine.choose_and_step's Watch boundary exactly: resolving
        # the mask may refresh lazy simulator state before runtime preparation.
        prepared.get_resolved_action_mask()
        world.runtime.prepare(prepared)
        selected["advisory"] = prepared.get_advisory_action_mask()[0, 0].astype(bool).copy()
        selected["obs"] = (
            build_ego_raster(
                prepared, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
            )
            if capture_obs
            else None
        )
        # Return an owned copy so neither the simulator nor caller can mutate
        # the stored tape through an alias.
        return recorded.copy()

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    return recorded.copy(), selected["advisory"], selected["obs"]


def record_world_trace(task: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Run one Mac world to its last selected root and bind its action tape."""
    from research.bl_info_value.reconstruct import _mass_continuation
    from research.t3_rollout_labels_20261015.bank_engine import build_world, choose_and_step

    required = {"arm", "plan", "checkpoint", "row_bindings", "world"}
    missing = sorted(required.difference(task))
    if missing:
        raise ValueError(f"trace task lacks {missing}")
    rows = {int(row["decision"]): row for row in task["row_bindings"]}
    if not rows:
        raise ValueError("trace task has no selected roots")
    if len(rows) != len(task["row_bindings"]):
        raise ValueError("trace task repeats a selected decision")
    world = build_world(
        str(task["arm"]), dict(task["plan"]), hero_checkpoint=Path(task["checkpoint"])
    )
    original_prepare = world.runtime.prepare
    root_hashes: dict[str, str] = {}
    prepared_frames: dict[int, int] = {}
    root_clones: dict[int, Any] = {}
    decision = 0

    def capture(sim: Any) -> None:
        original_prepare(sim)
        if decision in rows:
            root_hashes[str(decision)] = state_binding(world)
            prepared_frames[decision] = int(sim.frame[0])
            # Capture after the runtime's carried selector state is prepared,
            # before the recorded source action mutates simulator or memory.
            root_clones[decision] = world.clone()

    world.runtime.prepare = capture
    started = time.perf_counter()
    tape: list[np.ndarray] = []
    active_controls: list[dict[str, Any]] = []
    control_hashes: dict[str, str] = {}
    control_deaths: dict[str, int | None] = {}
    max_root = max(rows)
    try:
        # Continue after the last root only to traverse its already-defined
        # reactive B8 control.  Those extra source actions are deliberately
        # not added to the tape: x86 needs roots, not historical continuations.
        while bool(world.sim.alive[0, 0]) and (decision <= max_root or active_controls):
            actions, _advisory, obs = choose_and_step(world, capture_obs=decision in rows)
            action_row = np.asarray(actions[0], dtype=np.int64)
            if np.any(action_row < 0) or np.any(action_row > 5):
                raise ReconstructionError(f"invalid action in recorded decision {decision}")
            if decision <= max_root:
                tape.append(action_row.astype(np.uint8, copy=True))
            after_mass = float(world.sim.length[0, 0]) if bool(world.sim.alive[0, 0]) else 0.0
            for control in active_controls:
                offset = decision - control["decision"]
                if 0 <= offset < len(control["prefix"]) and after_mass != control["prefix"][offset]:
                    raise ReconstructionError(
                        f"Mac B8 control mass mismatch at decision {decision}: "
                        f"{after_mass} != {control['prefix'][offset]}"
                    )
            active_controls = [
                item
                for item in active_controls
                if decision - item["decision"] + 1 < len(item["prefix"])
            ]
            task_death = task.get("death_decision")
            if task_death is not None and decision >= int(task_death):
                if bool(world.sim.alive[0, 0]) or decision != int(task_death):
                    raise ReconstructionError(
                        f"recorded source death mismatch: observed decision {decision}, "
                        f"expected {task_death}"
                    )
            if decision in rows:
                if obs is None or decision not in prepared_frames:
                    raise ReconstructionError(f"trace capture missing at decision {decision}")
                _check_row(
                    rows[decision], obs, int(action_row[0]), decision, prepared_frames[decision]
                )
                value, prefix, death = _mass_continuation(
                    root_clones.pop(decision), int(action_row[0]), int(task.get("depth", 300))
                )
                del value
                if not prefix or prefix[0] != after_mass:
                    raise ReconstructionError(
                        f"Mac B8 forced-first mass mismatch at decision {decision}"
                    )
                saved = rows[decision].get("outcome_death_frame")
                if saved is not None:
                    saved_values = np.asarray(saved).reshape(-1)
                    expected_death = int(
                        saved_values[0]
                        if saved_values.size == 1
                        else saved_values[int(action_row[0])]
                    )
                    observed_death = -1 if death is None else decision + death
                    if expected_death != observed_death:
                        raise ReconstructionError(
                            f"recorded B8 death mismatch at decision {decision}"
                        )
                prefix_bytes = np.asarray(prefix, dtype=np.float64).tobytes(order="C")
                control_hashes[str(decision)] = hashlib.sha256(prefix_bytes).hexdigest()
                control_deaths[str(decision)] = death
                active_controls.append({"decision": decision, "prefix": prefix})
            decision += 1
    finally:
        world.runtime.prepare = original_prepare
    if set(root_hashes) != {str(value) for value in rows}:
        raise ReconstructionError("recorded world terminated before every selected root")
    task_death = task.get("death_decision")
    if task_death is not None and not bool(world.sim.alive[0, 0]):
        observed_death = decision - 1
        if observed_death != int(task_death):
            raise ReconstructionError(
                f"recorded source death mismatch: {observed_death} != {task_death}"
            )
    # If the source hero died before a control's horizon, all remaining control
    # entries must be the specified zero-dead suffix.
    for control in active_controls:
        offset = decision - control["decision"]
        if any(value != 0.0 for value in control["prefix"][max(offset, 0) :]):
            raise ReconstructionError("Mac control survived after source terminal death")
    return np.stack(tape).astype(np.uint8, copy=False), {
        "world": int(task["world"]),
        "roots": len(rows),
        "replayed_decisions": len(tape),
        "tape_seconds": time.perf_counter() - started,
        "root_state_hashes": root_hashes,
        "control_prefix_hashes": control_hashes,
        "control_deaths": control_deaths,
        "mac_controls_verified": len(control_hashes),
        "provenance": (
            "regenerated on Mac by live collection; verified against original selected rows"
        ),
    }


def _record_task(task: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Spawn-safe, low-priority single-thread trace worker."""
    from research.bl_info_value.resources import lower_priority
    from research.bl_info_value.run import threads

    lower_priority()
    threads()
    tape, metric = record_world_trace(task)
    metric["worker_nice"] = os.nice(0)
    return tape, metric


def build_trace_bank(
    archive: Path,
    archive_sha256: str,
    checkpoint_archive: Path,
    out: Path,
    *,
    smoke: bool = False,
    workers: int = 1,
) -> dict[str, Any]:
    """Create a bounded v3 tape archive from a verified compact-v2 input.

    This runs on the Mac with one smoke worker or up to three production
    workers.  It creates no D300 targets or student fits; it only records the
    prefix needed to recreate each root.
    """
    archive, checkpoint_archive, out = (
        Path(archive),
        Path(checkpoint_archive),
        check_path(Path(out)),
    )
    if out.exists():
        raise FileExistsError(out)
    if not 1 <= workers <= 3 or (smoke and workers > 1):
        raise ValueError("trace bank permits one smoke worker or up to three production workers")
    if os.uname().machine != "arm64":
        raise RuntimeError("trace bank creation is Mac arm64 only")
    # Set these before any model imports.  The original collection is a
    # single-threaded CPU identity path; this operation must retain that seam.
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    for name in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        os.environ[name] = "1"
    from research.bl_info_value.resources import lower_priority
    from research.bl_info_value.run import threads

    lower_priority()
    threads()
    with __import__("tempfile").TemporaryDirectory(prefix="bl-e3-traces-") as temporary:
        inputs = Path(temporary) / "inputs"
        catalog = unpack(archive, inputs, archive_sha256)
        if catalog.get("schema") != "bl-e3-input/v2" or bool(catalog.get("smoke")) != smoke:
            raise ValueError("trace bank requires matching compact v2 inputs")
        hydrate(catalog, inputs, checkpoint_archive)
        previous_ckpt_dir = os.environ.get("SNAKE_CKPT_DIR")
        os.environ["SNAKE_CKPT_DIR"] = str(inputs / "checkpoints")
        tasks: list[dict[str, Any]] = []
        trace_payloads: dict[str, bytes] = {}
        metrics: list[dict[str, Any]] = []
        try:
            recording_tasks = []
            for source_task in catalog["tasks"]:
                # Keep ``task`` portable in v3.  Only the ephemeral recording copy
                # receives absolute paths beneath this temporary input directory.
                task = json.loads(json.dumps(source_task))
                recording_task = json.loads(json.dumps(task))
                recording_task["checkpoint"] = str(inputs / recording_task["checkpoint"])
                for spec in recording_task["plan"]["specs"]:
                    if spec[0] == "checkpoint":
                        spec[1] = str(inputs / spec[1])
                recording_tasks.append((task, recording_task))
            started = time.perf_counter()
            last_progress = started
            if workers == 1:
                iterator = map(lambda pair: _record_task(pair[1]), recording_tasks)
                pool = None
            else:
                pool = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))
                iterator = pool.map(_record_task, [pair[1] for pair in recording_tasks])
            try:
                for index, ((task, _recording_task), (tape, metric)) in enumerate(
                    zip(recording_tasks, iterator, strict=True)
                ):
                    stream = io.BytesIO()
                    np.savez_compressed(stream, actions=tape)
                    name = f"traces/{index:03d}.npz"
                    payload = stream.getvalue()
                    trace_payloads[name] = payload
                    task["trace_npz"] = name
                    task["trace_sha256"] = hashlib.sha256(payload).hexdigest()
                    task["root_state_hashes"] = metric.pop("root_state_hashes")
                    task["control_prefix_hashes"] = metric.pop("control_prefix_hashes")
                    task["control_deaths"] = metric.pop("control_deaths")
                    task["mac_controls_verified"] = metric.pop("mac_controls_verified")
                    tasks.append(task)
                    metrics.append(metric)
                    now = time.perf_counter()
                    if (
                        index == 0
                        or index + 1 == len(recording_tasks)
                        or now - last_progress >= 120.0
                    ):
                        print(
                            json.dumps(
                                {
                                    "phase": "trace_bank",
                                    "done": index + 1,
                                    "total": len(recording_tasks),
                                    "elapsed_seconds": now - started,
                                }
                            ),
                            flush=True,
                        )
                        last_progress = now
            finally:
                if pool is not None:
                    pool.shutdown(wait=True, cancel_futures=True)
        finally:
            if previous_ckpt_dir is None:
                os.environ.pop("SNAKE_CKPT_DIR", None)
            else:
                os.environ["SNAKE_CKPT_DIR"] = previous_ckpt_dir
        result = dict(catalog)
        result.update(
            schema="bl-e3-input/v3",
            tasks=tasks,
            files={
                name: hashlib.sha256(payload).hexdigest()
                for name, payload in sorted(trace_payloads.items())
            },
        )
        catalog_bytes = (json.dumps(result, sort_keys=True) + "\n").encode()
        archive_bytes = _tar_gzip({"catalog.json": catalog_bytes, **trace_payloads})
        if len(archive_bytes) > _TRACE_LIMIT:
            raise ValueError(f"trace archive exceeds {_TRACE_LIMIT} byte Z-9 cap")
        out.mkdir(parents=True)
        write_json(out / "catalog.json", result)
        export_bytes(out / "inputs.tar.gz", archive_bytes)
        timing = {
            "schema": "bl-e3-trace-timing/v1",
            "smoke": smoke,
            "worlds": len(tasks),
            "tape_seconds": sum(float(item["tape_seconds"]) for item in metrics),
            "replayed_decisions": sum(int(item["replayed_decisions"]) for item in metrics),
            "world_metrics": metrics,
            "archive_bytes": len(archive_bytes),
            "source_archive_sha256": sha256(archive),
            "checkpoint_archive_sha256": sha256(checkpoint_archive),
            "source_manifest_sha256": catalog.get("source_manifest_sha256"),
            "protocol_sha256": catalog.get("protocol_sha256"),
            "trace_bank_sha256": sha256(Path(__file__)),
            "reconstruct_sha256": sha256(Path(__file__).with_name("reconstruct.py")),
            "root_hash_note": (
                "Mac-regenerated roots checked against original selected-row identities; "
                "original full-state hashes were unavailable."
            ),
        }
        write_json(out / "timing.json", timing)
    return result


def main() -> None:
    """Create one fresh Mac trace bank without launching any remote work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--archive-sha256", required=True)
    parser.add_argument("--checkpoint-archive", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    result = build_trace_bank(
        args.archive,
        args.archive_sha256,
        args.checkpoint_archive,
        args.out,
        smoke=args.smoke,
        workers=args.workers,
    )
    print(json.dumps({"worlds": len(result["tasks"]), "schema": result["schema"]}))


if __name__ == "__main__":
    main()
