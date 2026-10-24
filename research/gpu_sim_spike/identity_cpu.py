"""Mac CPU regenerator for the closed-loop identity falsifier.

The pinned CPU authority independently plays each development world from the
same constructor seam: ``trajectory.build_world`` + ``prepare_world``, hero =
``ExitTeacher.evaluate(...).greedy_action()`` (bare B8, fp32, batch one), and
every opponent through ``ExitTeacher.actions`` (the T3/T34 dispatcher with the
frozen v8 hooks), then ``sim.step`` and ``runtime.observe_step``.  This is the
natural-play loop of ``trajectory.collect_world`` without search.  No device
action is ever replayed.

Two passes per world:

``natural``  the registered Tier-L regeneration.  It stops one transition after
             the first divergence (only the first divergence is scored).
``bridged``  a recorded-only diagnostic.  When a seam differs *only* by draws of
             the device counter RNG (fresh ambient food, respawn head cells),
             those device draws are written into the CPU world before its
             policy runtime prepares, and play continues.  It stops at the first
             divergence that is not RNG-explained.  It never decides an outcome.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from research.gpu_sim_spike import identity_compare as compare
from research.gpu_sim_spike import identity_tape as tape

SCHEMA = "gpu-identity-cpu/v1"
RECORD_SCHEMA = "gpu-identity-cpu-record/v1"
KINDS = ("natural", "bridged")


class CpuCapture:
    """Record the CPU authority's own hero/opponent intermediates (no behaviour change)."""

    def __init__(self, world: Any, teacher: Any) -> None:
        self.obs: Mapping[str, np.ndarray] | None = None
        self.events: list[tuple[str, Any]] = []
        policy = teacher.policy
        original_forward = policy._forward

        def forward(net: Any, obs: Mapping[str, np.ndarray]) -> np.ndarray:
            self.obs = obs
            return original_forward(net, obs)

        policy._forward = forward
        runtime = world.runtime
        original_selection = runtime.selection

        def selection(slots: Any) -> Any:
            states, exact = original_selection(slots)
            self.events.append(("selection", (np.asarray(slots), states, exact)))
            return states, exact

        runtime.selection = selection
        self._patched = [(policy, "_forward"), (runtime, "selection")]
        for opponent in world.opponents:
            if hasattr(opponent, "q_values") and hasattr(opponent, "network"):
                original_q = opponent.q_values

                def q_values(states: Any, _original: Any = original_q) -> Any:
                    value = _original(states)
                    self.events.append(("network", value))
                    return value

                opponent.q_values = q_values
                self._patched.append((opponent, "q_values"))

    def reset(self) -> None:
        self.obs = None
        self.events = []

    def obs_digests(self) -> np.ndarray:
        if self.obs is None:
            raise RuntimeError("CPU hero forward was not observed")
        return np.stack(
            [
                tape.tensor_digest(np.ascontiguousarray(self.obs[key][0:1]))
                for key in ("local", "global", "scalars")
            ]
        )

    def opponents(self, slots: int) -> dict[str, np.ndarray]:
        q = np.full((slots, 6), np.nan, dtype=np.float32)
        mask = np.zeros((slots, 6), dtype=bool)
        digest = np.zeros((slots, 32), dtype=np.uint8)
        recorded = np.zeros(slots, dtype=bool)
        pending = None
        for kind, value in self.events:
            if kind == "selection":
                pending = value
                continue
            if pending is None:
                raise RuntimeError("CPU opponent forward without a selection")
            rows, states, exact = pending
            pending = None
            slot = int(rows[0, 1])
            q[slot] = np.asarray(value.detach().float().cpu().numpy()[0], dtype=np.float32)
            mask[slot] = np.asarray(exact[0], dtype=bool)
            digest[slot] = tape.tensor_digest(np.asarray(states[0], dtype=np.float32))
            recorded[slot] = True
        return {"opp_q": q, "opp_mask": mask, "opp_state_digest": digest, "opp_recorded": recorded}

    def close(self) -> None:
        for owner, name in self._patched:
            if name in vars(owner):
                delattr(owner, name)


def _device_state(dev: Mapping[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
    count = int(dev["state_food_count"][t])
    return {
        "snake": np.asarray(dev["state_snake"][t], dtype=np.int64),
        "food": np.asarray(dev["state_food"][t, :count], dtype=np.int64),
    }


def apply_rng_bridge(sim: Any, dev: Mapping[str, np.ndarray], t: int, respawned: set[int]) -> None:
    """Supply the device's draws: ordered food (+corpse flags) and respawn heads."""
    target = _device_state(dev, t)
    cells = [(int(x), int(y)) for x, y, _ in target["food"]]
    sim.food_cells[0] = list(cells)
    sim.food_set[0] = set(cells)
    sim.corpse_cells[0] = {(int(x), int(y)) for x, y, c in target["food"] if int(c)}
    hx, hy = tape.SNAKE_FIELDS.index("head_x"), tape.SNAKE_FIELDS.index("head_y")
    for slot in respawned:
        head = int(sim.head_ptr[0, slot])
        sim.bodies[0, slot, head] = (int(target["snake"][slot, hx]), int(target["snake"][slot, hy]))
    sim._rebuild_env_grids(0, food=True)
    sim._rebuild_traversed_from_heads(np.ones(1, dtype=bool))
    sim._refresh_action_masks(np.ones(1, dtype=bool))


def _window(states: list[dict], decisions: list[dict], t: int, slots: int) -> dict[str, np.ndarray]:
    """Two-seam view (t-1, t) of running CPU rows in tape layout."""
    writer = tape.TapeWriter(slots)
    start = max(t - 1, 0)
    for index in range(start, t + 1):
        writer.states.append(states[index])
        writer.replica_agree.append(True)
        if index < len(decisions):
            writer.decisions.append(decisions[index])
    if len(writer.decisions) == len(writer.states):
        writer.states.append(states[t])  # placeholder end seam; never indexed
        writer.replica_agree.append(True)
    return writer.arrays()


def _device_window(dev: Mapping[str, np.ndarray], t: int) -> dict[str, np.ndarray]:
    """The same (t-1, t) view of the device tape; index ``min(t, 1)`` is seam t."""
    start = max(t - 1, 0)
    return {key: value[start : t + 1] for key, value in dev.items()}


def regenerate_world(
    row: Mapping[str, Any],
    dev: Mapping[str, np.ndarray],
    checkpoint: Path,
    *,
    horizon: int,
    kind: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Independent CPU closed loop; compares online only to know when to stop."""
    from research.t34_exit_b8.engine import ExitTeacher
    from research.t34_exit_b8.trajectory import prepare_world

    if kind not in KINDS:
        raise ValueError(f"unknown regeneration kind: {kind}")
    began = time.monotonic()
    world = tape.build_cpu_world(row, checkpoint)
    teacher = ExitTeacher(device="cpu")
    capture = CpuCapture(world, teacher)
    kinds = compare.opponent_kinds(row)
    slots = int(world.sim.S)
    states: list[dict[str, np.ndarray]] = []
    decisions: list[dict[str, np.ndarray]] = []
    bridges: list[int] = []
    first_non_rng = None
    stopped = "horizon"
    terminal = "horizon"
    dev_seams = len(dev["state_frame"])

    def prepare_bridged(t: int) -> None:
        class Prepared(Exception):
            pass

        def select(sim: Any) -> np.ndarray:
            if t < dev_seams:
                current = tape.cpu_seam_state(sim)
                probe = _window(states + [current], decisions, t, slots)
                diffs = compare.state_differences(_device_window(dev, t), probe, min(t, 1))
                if (
                    diffs
                    and compare.classify_state(_device_window(dev, t), probe, min(t, 1), diffs)
                    == "RNG"
                ):
                    alive = tape.SNAKE_FIELDS.index("alive")
                    respawned = {
                        s
                        for s in range(slots)
                        if states[t - 1]["snake"][s, alive] == 0 and current["snake"][s, alive] == 1
                    }
                    apply_rng_bridge(sim, dev, t, respawned)
                    bridges.append(t)
            world.runtime.prepare(sim)
            raise Prepared

        try:
            world.sim.step_with_policy(select)
        except Prepared:
            world.sim._policy_phase_prepared = True
            world.sim._policy_step_in_progress = False

    try:
        for t in range(horizon):
            if t > 0:
                if kind == "bridged":
                    prepare_bridged(t)
                else:
                    prepare_world(world)
            states.append(tape.cpu_seam_state(world.sim))
            capture.reset()
            evaluation = teacher.evaluate(world)
            hero = evaluation.greedy_action()
            grid = teacher.actions(world, hero)
            decisions.append(
                {
                    "hero_q": np.asarray(evaluation.q, dtype=np.float32),
                    "hero_mask": np.asarray(
                        world.sim.get_advisory_action_mask()[0, 0], dtype=bool
                    ).copy(),
                    "actions": np.asarray(grid[0], dtype=np.int64),
                    "obs_digest": capture.obs_digests(),
                    "replica_decision_agree": np.asarray(True),
                    **capture.opponents(slots),
                }
            )
            world.sim.step(grid)
            world.runtime.observe_step(world.sim)
            found = None
            if t < dev_seams:
                window = _window(states, decisions, t, slots)
                found = compare.seam_divergence(_device_window(dev, t), window, min(t, 1), kinds)
            if found is None and t >= len(dev["actions"]):
                stopped = "device_tape_end"
            if found is not None:
                found = {**found, "tick": t}
                first_non_rng = found
                stopped = "first_divergence"
            if not bool(world.sim.alive[0, 0]):
                terminal = f"hero_dead_after_decision_{t}"
            if found is not None or stopped == "device_tape_end" or terminal != "horizon":
                break
        if kind == "bridged" and len(states) < dev_seams:
            prepare_bridged(len(states))
        else:
            prepare_world(world)
        states.append(tape.cpu_seam_state(world.sim))
    finally:
        capture.close()
    writer = tape.TapeWriter(slots)
    for index, state in enumerate(states):
        writer.state(state)
        if index < len(decisions):
            writer.decision(decisions[index])
    meta = {
        "kind": kind,
        "stopped": stopped,
        "terminal": terminal,
        "decisions": len(decisions),
        "wall_seconds": time.monotonic() - began,
        "world_id": row["world_id"],
        "mix": row["mix"],
        "authority": "trajectory.build_world/prepare_world + ExitTeacher(cpu) greedy + actions",
    }
    if kind == "bridged":
        meta["rng_bridges"] = len(bridges)
        meta["first_rng_bridge"] = bridges[0] if bridges else None
        meta["first_non_rng_divergence"] = first_non_rng
    return writer.arrays(), meta


def _paths(cpu_root: Path, ns: str, unit: str, kind: str) -> tuple[Path, Path]:
    base = Path(cpu_root) / ns / kind
    return base / f"{unit}.npz", base / f"{unit}.json"


def load_cpu_tape(
    cpu_root: Path, ns: str, unit: str, kind: str, device_record: Mapping[str, Any]
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    tape_path, record_path = _paths(cpu_root, ns, unit, kind)
    if not record_path.exists():
        raise FileNotFoundError(record_path)
    tape.verify(record_path)
    record = json.loads(record_path.read_text())
    digest = tape.verify(tape_path)
    if (
        record.get("schema") != RECORD_SCHEMA
        or record.get("namespace") != ns
        or record.get("unit_id") != unit
        or record.get("kind") != kind
        or record.get("tape_sha256") != digest
        or record.get("device_tape_sha256") != device_record["tape"]["sha256"]
    ):
        raise ValueError(f"CPU tape binding differs: {record_path}")
    arrays, meta = tape.decode_tape(tape_path.read_bytes())
    return arrays, meta


def run(args: argparse.Namespace) -> dict[str, Any]:
    rows = tape.world_rows(args.mode)
    allocation = tape.allocation(args.mode)
    ns = tape.namespace(args.mode, args.commit)
    horizon = int(tape.MODES[args.mode]["horizon"])
    checkpoint_sha = tape.sha256_file(args.checkpoint)
    if args.expected_checkpoint_sha256 and checkpoint_sha != args.expected_checkpoint_sha256:
        raise ValueError("Mac B8 checkpoint differs from the public input hash")
    selected = set(args.units or allocation)
    summary: dict[str, Any] = {"schema": SCHEMA, "namespace": ns, "units": {}}
    for index, unit in enumerate(allocation):
        if unit not in selected:
            continue
        record, dev, _meta = tape.load_unit(args.device_exports, ns, allocation, unit)
        for kind in args.kinds:
            tape_path, record_path = _paths(args.cpu_root, ns, unit, kind)
            if record_path.exists():
                load_cpu_tape(args.cpu_root, ns, unit, kind, record)
                summary["units"].setdefault(unit, {})[kind] = "VERIFIED_EXISTING"
                continue
            arrays, meta = regenerate_world(
                rows[index], dev, args.checkpoint, horizon=horizon, kind=kind
            )
            meta.update(
                unit_id=unit,
                namespace=ns,
                checkpoint_sha256=checkpoint_sha,
                single_thread_env=tape.single_thread_environment(),
            )
            digest = tape.publish_bytes(tape_path, tape.encode_tape(arrays, meta))
            tape.publish_json(
                record_path,
                {
                    "schema": RECORD_SCHEMA,
                    "namespace": ns,
                    "unit_id": unit,
                    "kind": kind,
                    "tape_sha256": digest,
                    "device_tape_sha256": record["tape"]["sha256"],
                    "stopped": meta["stopped"],
                    "decisions": meta["decisions"],
                },
            )
            summary["units"].setdefault(unit, {})[kind] = meta["stopped"]
            print(
                json.dumps(
                    {
                        "event": "cpu_regenerated",
                        "unit": unit,
                        "kind": kind,
                        "stopped": meta["stopped"],
                        "decisions": meta["decisions"],
                    }
                ),
                flush=True,
            )
    summary["checkpoint_sha256"] = checkpoint_sha
    summary["seed_digest"] = hashlib.sha256(
        ",".join(str(row["world_seed"]) for row in rows).encode()
    ).hexdigest()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device-exports", type=Path, required=True)
    parser.add_argument("--cpu-root", type=Path, required=True)
    parser.add_argument("--mode", choices=sorted(tape.MODES), required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-checkpoint-sha256")
    parser.add_argument("--kinds", nargs="+", choices=KINDS, default=list(KINDS))
    parser.add_argument("--units", nargs="+")
    args = parser.parse_args()
    print(json.dumps(run(args), sort_keys=True))


if __name__ == "__main__":
    main()
