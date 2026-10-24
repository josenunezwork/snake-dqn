"""Device producer for the GPU-vs-CPU closed-loop identity falsifier.

Each development world is played closed-loop on the integrated I0 engine (arm
``I``: indexed MOVE, native vector61 geometry, static flood, no indexed v8) at
the pinned runtime, with ``replicas`` identical copies forming the batch.  The
producer exports one immutable tape per world: replica-0 seam states, actions,
hero Q/mask, observation digests and opponent Q/masks/state digests, plus a
per-seam flag that every replica is bit-identical to replica 0.

Nothing here compares against the CPU or reads an outcome; the Mac regenerator
(``identity_cpu``) and comparator (``identity_compare``) own that.  Worlds are
independent units published atomically; a verified ledger skips completed
units, and a worker that reaches the producer deadline abandons its partial
world rather than publishing a truncated tape.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import math
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from research.gpu_sim_spike import identity_tape as tape

SCHEMA = "gpu-identity-device/v1"
REQUIRED_ENV = {
    "GPU_SIM_VECTOR_FLOOD": "static",
    "GPU_SIM_VECTOR_GEOMETRY": "native",
    "GPU_SIM_V8_INDEXED": "0",
}


class DeviceIdentityError(RuntimeError):
    """The producer cannot preserve the pinned integrated-engine contract."""


def _bits(value: Any) -> Any:
    """Bit view for float tensors so -0.0/NaN payloads are compared exactly."""
    import torch

    if value.dtype == torch.float32:
        return value.view(torch.int32)
    if value.dtype == torch.float64:
        return value.view(torch.int64)
    return value


def replicas_identical(tensors: list[Any]) -> bool:
    """One synchronizing reduction: every row equals row 0 bit-for-bit."""
    import torch

    checks = []
    for value in tensors:
        if not isinstance(value, torch.Tensor) or value.ndim == 0 or value.shape[0] < 2:
            continue
        bits = _bits(value)
        checks.append((bits[1:] == bits[:1]).reshape(-1).all())
    if not checks:
        return True
    return bool(torch.stack(checks).all().item())


def _state_tensors(sim: Any, policy: Any) -> list[Any]:
    import torch

    rows = int(sim.E)
    values = [v for v in vars(sim).values() if isinstance(v, torch.Tensor)]
    runtime = getattr(policy, "vector_runtime", None)
    if runtime is not None:
        values += [v for v in vars(runtime).values() if isinstance(v, torch.Tensor)]
    return [v for v in values if v.ndim and v.shape[0] == rows]


class Capture:
    """Non-invasive hooks recording the engine's own hero/opponent intermediates."""

    def __init__(self, policy: Any) -> None:
        self.policy = policy
        self.hero_inputs: tuple[Any, ...] | None = None
        self.events: list[tuple[str, Any]] = []
        self.handles = [policy.b8_network.register_forward_pre_hook(self._hero)]
        seen: set[int] = set()
        for net in policy.opponent_networks:
            if net is not None and id(net) not in seen:
                seen.add(id(net))
                self.handles.append(net.register_forward_hook(self._opponent))
        runtime = policy.vector_runtime
        original = runtime.selection

        def selection(rows: Any) -> Any:
            states, masks = original(rows)
            self.events.append(("selection", (rows, states, masks)))
            return states, masks

        runtime.selection = selection  # instance attribute; class method untouched
        self._runtime = runtime

    def _hero(self, _module: Any, args: tuple[Any, ...]) -> None:
        self.hero_inputs = tuple(args)

    def _opponent(self, _module: Any, _args: Any, output: Any) -> None:
        self.events.append(("network", output))

    def reset(self) -> None:
        self.hero_inputs = None
        self.events = []

    def opponents(self, slots: int) -> dict[str, np.ndarray]:
        """Pair each network forward with its preceding selection (slot order)."""
        q = np.full((slots, 6), np.nan, dtype=np.float32)
        mask = np.zeros((slots, 6), dtype=bool)
        digest = np.zeros((slots, 32), dtype=np.uint8)
        recorded = np.zeros(slots, dtype=bool)
        tensors: list[Any] = []
        pending = None
        for kind, value in self.events:
            if kind == "selection":
                pending = value
                continue
            if pending is None:
                raise DeviceIdentityError("opponent forward without a selection")
            rows, states, masks = pending
            pending = None
            slot = int(rows[0, 1].item())
            q[slot] = value[0].detach().float().cpu().numpy()
            mask[slot] = masks[0].detach().cpu().numpy().astype(bool)
            digest[slot] = tape.tensor_digest(
                states[0].detach().cpu().numpy().astype(np.float32, copy=False)
            )
            recorded[slot] = True
            tensors += [value, states, masks]
        return {
            "opp_q": q,
            "opp_mask": mask,
            "opp_state_digest": digest,
            "opp_recorded": recorded,
            "_tensors": tensors,
        }

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        if "selection" in vars(self._runtime):
            del self._runtime.selection


def obs_digests(inputs: tuple[Any, ...]) -> np.ndarray:
    if inputs is None or len(inputs) != 3:
        raise DeviceIdentityError("hero network did not receive (local, global, scalars)")
    rows = [tape.tensor_digest(item[0:1].detach().cpu().contiguous().numpy()) for item in inputs]
    return np.stack(rows)


@contextlib.contextmanager
def _arm(arm: str) -> Iterator[None]:
    from research.gpu_sim_spike import integration_engine

    with integration_engine._arm(arm):
        yield


def build_case(world: Any, device: str, arm: str, replicas: int) -> tuple[Any, Any]:
    """The I0 arm's engine on one natural world, replicated to the batch width."""
    from research.gpu_sim_spike import integration_engine
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim

    spec = integration_engine.arm_spec(arm)
    one_sim = DeviceSim.from_reference(world.sim, device=device)
    one_policy = DevicePolicy.from_world(world, device=device, sim=one_sim)
    sim = one_sim.repeat_interleave(replicas)
    policy = one_policy.repeat_interleave(replicas)
    integration_engine._set_move_backend(sim, spec.move_backend)
    integration_engine._install_geometry(policy, spec.geometry)
    if spec.v8_indexed or os.environ.get("GPU_SIM_V8_INDEXED", "0") != "0":
        raise DeviceIdentityError("identity test runs without the pending indexed v8 route")
    if not bool(getattr(world.sim, "_policy_phase_prepared", False)) or not sim._prepared:
        raise DeviceIdentityError("device start must be the CPU world's prepared seam")
    return sim, policy


def play_world(
    row: dict[str, Any],
    checkpoint: Path,
    *,
    device: str,
    arm: str,
    replicas: int,
    horizon: int,
    deadline: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    """Closed-loop device play; ``None`` if the deadline interrupts the world."""
    import torch

    world = tape.build_cpu_world(row, checkpoint)
    start_cpu = tape.cpu_seam_state(world.sim)
    began = time.monotonic()
    with _arm(arm), torch.inference_mode():
        sim, policy = build_case(world, device, arm, replicas)
        capture = Capture(policy)
        writer = tape.TapeWriter(int(sim.S))
        terminal = "horizon"
        try:
            for step in range(horizon):
                if time.time() >= deadline:  # wall-clock epoch, shared with the parent
                    return None
                sim.prepare()
                writer.state(
                    tape.device_seam_state(sim, 0),
                    replicas_identical(_state_tensors(sim, policy)),
                )
                capture.reset()
                q, legal, hero = policy.evaluate(sim)
                actions = policy.actions(sim, hero)
                opponents = capture.opponents(int(sim.S))
                agree = replicas_identical(
                    [q, legal, actions, *(capture.hero_inputs or ()), *opponents.pop("_tensors")]
                )
                writer.decision(
                    {
                        "hero_q": q[0].detach().float().cpu().numpy(),
                        "hero_mask": legal[0].detach().cpu().numpy().astype(bool),
                        "actions": actions[0].detach().cpu().numpy().astype(np.int64),
                        "obs_digest": obs_digests(capture.hero_inputs),
                        "replica_decision_agree": np.asarray(agree),
                        **opponents,
                    }
                )
                sim.step(actions)
                policy.after_step(sim)
                if not bool(sim.alive[0, 0].item()):
                    terminal = f"hero_dead_after_decision_{step}"
                    break
            sim.prepare()
            writer.state(
                tape.device_seam_state(sim, 0),
                replicas_identical(_state_tensors(sim, policy)),
            )
        finally:
            capture.close()
    arrays = writer.arrays()
    native = _native_evidence(device, arm)
    meta = {
        "terminal": terminal,
        "decisions": int(arrays["actions"].shape[0]),
        "replicas": int(replicas),
        "arm": arm,
        "device": device,
        "device_name": torch.cuda.get_device_name() if device == "cuda" else "cpu",
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "tf32": {
            "matmul": bool(torch.backends.cuda.matmul.allow_tf32),
            "cudnn": bool(torch.backends.cudnn.allow_tf32),
        },
        "native": native,
        "start_matches_cpu_constructor": bool(
            all(
                np.array_equal(np.asarray(start_cpu[key]), np.asarray(arrays_value))
                for key, arrays_value in (
                    ("snake", arrays["state_snake"][0]),
                    ("body", arrays["state_body"][0]),
                    ("frame", arrays["state_frame"][0]),
                )
            )
            and np.array_equal(
                start_cpu["food"], arrays["state_food"][0, : int(arrays["state_food_count"][0])]
            )
        ),
        "wall_seconds": time.monotonic() - began,
        "world_id": row["world_id"],
        "mix": row["mix"],
        "seed_sha256": hashlib.sha256(str(int(row["world_seed"])).encode()).hexdigest(),
    }
    return arrays, meta


def _native_evidence(device: str, arm: str) -> dict[str, Any]:
    """Prove the admitted native routes were actually loaded (no silent fallback)."""
    from research.gpu_sim_spike import move_cuda, vector_geometry

    return {
        "move_extension_loaded": move_cuda._EXTENSION is not None,
        "geometry_extension_loaded": vector_geometry._EXTENSION is not None,
        "required": device == "cuda" and arm == "I",
    }


def run_worker(args: argparse.Namespace) -> int:
    """Play the assigned worlds in order and publish each completed unit."""
    _configure_torch(args.device)
    rows = tape.world_rows(args.mode)
    spec = tape.MODES[args.mode]
    allocation = tape.allocation(args.mode)
    ns = tape.namespace(args.mode, args.commit)
    status: dict[str, Any] = {"worker": args.worker, "units": {}}
    for index in args.worlds:
        row = rows[index]
        unit = tape.unit_id(index, row)
        played = play_world(
            row,
            args.checkpoint,
            device=args.device,
            arm=args.arm,
            replicas=int(spec["replicas"]),
            horizon=int(spec["horizon"]),
            deadline=args.deadline,
        )
        if played is None:
            status["units"][unit] = "DEADLINE_ABANDONED"
            break
        arrays, meta = played
        if args.device == "cuda" and not (
            meta["native"]["move_extension_loaded"] and meta["native"]["geometry_extension_loaded"]
        ):
            status["units"][unit] = "NATIVE_ROUTE_NOT_LOADED"
            break
        meta.update(unit_id=unit, namespace=ns, commit=args.commit, schema=tape.SCHEMA)
        tape.publish_unit(
            args.output,
            ns,
            allocation,
            unit,
            tape.encode_tape(arrays, meta),
            {
                "terminal": meta["terminal"],
                "decisions": meta["decisions"],
                "wall_seconds": meta["wall_seconds"],
                "native": meta["native"],
            },
        )
        status["units"][unit] = "PUBLISHED"
        print(json.dumps({"event": "identity_unit", "unit": unit}), flush=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    args.status.write_text(json.dumps(status, sort_keys=True) + "\n")
    return 0 if all(value == "PUBLISHED" for value in status["units"].values()) else 3


def _configure_torch(device: str) -> None:
    import torch

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device == "cuda" and (
        not torch.cuda.is_available() or "A100" not in torch.cuda.get_device_name()
    ):
        raise DeviceIdentityError("identity device run requires an A100")


def partition(indices: list[int], workers: int) -> list[list[int]]:
    """Round-robin keeps mixes spread across workers; order is frozen."""
    groups = [indices[offset::workers] for offset in range(workers)]
    return [group for group in groups if group]


def run(args: argparse.Namespace) -> dict[str, Any]:
    from research.gpu_sim_spike.benchmark import source_commit

    if args.device == "cuda":
        for key, value in REQUIRED_ENV.items():
            if os.environ.get(key) != value:
                raise DeviceIdentityError(f"{key} must be {value!r} for the I0 arm")
        if args.arm != "I":
            raise DeviceIdentityError("CUDA identity run uses the admitted I0 arm only")
    elif args.mode != "local":
        raise DeviceIdentityError("CPU device runs are the local Mac proof only")
    if not math.isfinite(args.seconds) or not 0 < args.seconds <= 3000:
        raise DeviceIdentityError("producer deadline must be finite, positive and bounded")
    if not 1 <= args.workers <= 6:
        raise DeviceIdentityError("workers must lie in [1, 6]")
    started = time.monotonic()
    deadline = time.time() + args.seconds
    commit = args.commit or source_commit()
    allocation = tape.allocation(args.mode)
    ns = tape.namespace(args.mode, commit)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    skip = tape.load_ledger(args.ledger, ns, allocation)
    pending = [index for index, unit in enumerate(allocation) if unit not in skip]
    work = args.work_dir or output.parent / "identity-work"
    work.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(110):
            print(
                json.dumps(
                    {"event": "identity_heartbeat", "elapsed_seconds": time.monotonic() - started}
                ),
                flush=True,
            )

    threading.Thread(target=heartbeat, daemon=True).start()
    processes = []
    for number, group in enumerate(partition(pending, args.workers)):
        status = work / f"worker-{number}.json"
        command = [
            sys.executable,
            "-m",
            "research.gpu_sim_spike.identity_device",
            "worker",
            "--mode",
            args.mode,
            "--commit",
            commit,
            "--checkpoint",
            str(args.checkpoint),
            "--output",
            str(output),
            "--device",
            args.device,
            "--arm",
            args.arm,
            "--deadline",
            repr(deadline),
            "--worker",
            str(number),
            "--status",
            str(status),
            "--worlds",
            *[str(index) for index in group],
        ]
        processes.append((number, group, status, subprocess.Popen(command)))
    workers = []
    for number, group, status, process in processes:
        code = process.wait()
        detail = json.loads(status.read_text()) if status.is_file() else None
        workers.append({"worker": number, "worlds": group, "returncode": code, "status": detail})
    stop.set()
    final = tape.build_ledger(output, ns, allocation)
    published = sorted(row["unit_id"] for row in final["units"])
    complete = sorted(set(published) | set(skip))
    result = {
        "schema": SCHEMA,
        "status": "COMPLETE" if complete == sorted(allocation) else "INCOMPLETE",
        "mode": args.mode,
        "commit": commit,
        "namespace": ns,
        "allocation": list(allocation),
        "skipped_verified": sorted(skip),
        "published_units": published,
        "workers": workers,
        "producer_seconds": time.monotonic() - started,
        "arm": args.arm,
        "device": args.device,
        "single_thread_env": tape.single_thread_environment(),
        "claim": "engineering identity tapes only; no outcome is computed on the device",
    }
    tape.publish_json(output / "identity-device.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    produce = sub.add_parser("run")
    produce.add_argument("--mode", choices=sorted(tape.MODES), required=True)
    produce.add_argument("--checkpoint", type=Path, required=True)
    produce.add_argument("--ledger", type=Path, required=True)
    produce.add_argument("--output-dir", type=Path, required=True)
    produce.add_argument("--seconds", type=float, required=True)
    produce.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    produce.add_argument("--arm", choices=("I", "R"), default="I")
    produce.add_argument("--workers", type=int, default=4)
    produce.add_argument("--commit")
    produce.add_argument("--work-dir", type=Path)
    worker = sub.add_parser("worker")
    worker.add_argument("--mode", choices=sorted(tape.MODES), required=True)
    worker.add_argument("--commit", required=True)
    worker.add_argument("--checkpoint", type=Path, required=True)
    worker.add_argument("--output", type=Path, required=True)
    worker.add_argument("--device", choices=("cpu", "cuda"), required=True)
    worker.add_argument("--arm", choices=("I", "R"), required=True)
    worker.add_argument("--deadline", type=float, required=True)
    worker.add_argument("--worker", type=int, required=True)
    worker.add_argument("--status", type=Path, required=True)
    worker.add_argument("--worlds", type=int, nargs="+", required=True)
    args = parser.parse_args()
    if args.command == "worker":
        raise SystemExit(run_worker(args))
    result = run(args)
    print(json.dumps({"status": result["status"], "units": result["published_units"]}))
    raise SystemExit(0 if result["status"] == "COMPLETE" else 3)


if __name__ == "__main__":
    main()
