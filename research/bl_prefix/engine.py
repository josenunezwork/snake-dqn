"""Exact GridBatchSim root capture and counterfactual continuations for E6/E7."""

from __future__ import annotations

import copy
import hashlib
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.bl_prefix.controllers import B8Controller, Controller
from research.t4_snapshot_restart_20261016 import snapshots

HORIZON = 5000
PREFIX_HISTORY = 128


class IncompleteRoot(RuntimeError):
    """A root lacking the registered state/history requirements."""


@dataclass(frozen=True)
class PrefixRecord:
    observation: dict[str, np.ndarray]
    action: int
    legal: np.ndarray
    advisory: np.ndarray | None = None


@dataclass
class EngineWorld:
    sim: Any
    runtime: Any
    opponents: list[Any]
    seed: int
    mix: str
    specs: list[list[str]]
    b8_path: str
    b8: B8Controller | None = None


@dataclass(frozen=True)
class Root:
    snapshot: dict[str, Any]
    prefix: tuple[PrefixRecord, ...]
    seed: int
    mix: str
    frame: int
    b8_path: str
    checkpoint_sha256: Mapping[str, str]


@dataclass(frozen=True)
class StepResult:
    observation: dict[str, np.ndarray]
    action: int
    legal: np.ndarray
    advisory: np.ndarray
    b8_q: np.ndarray
    pre_frame: int
    pre_length: int
    pre_alive: bool
    frame: int
    length: int
    alive: bool
    mass: float
    forced_fallback: bool = False

    @property
    def died(self) -> bool:
        return self.pre_alive and not self.alive


@dataclass(frozen=True)
class RolloutResult:
    mass_sum: float
    terminal_survival: bool
    action_tape: tuple[int, ...]
    steps: int
    fallback_count: int
    truncated: bool = False
    mass_trace: tuple[float, ...] = ()
    death_frame: int | None = None
    last_living_length: int | None = None


def initial_world(
    *, arm: str, plan: Mapping[str, Any], b8_path: str | Path | None = None
) -> EngineWorld:
    """Build one exact terminal-hero all-mix world through the T3 public seam."""
    from research.t3_rollout_labels_20261015 import bank_engine

    built = bank_engine.build_world(
        arm, dict(plan), hero_checkpoint=Path(b8_path) if b8_path else None
    )
    path = str(b8_path) if b8_path is not None else str(built.hero.checkpoint_path)
    return EngineWorld(
        built.sim,
        built.runtime,
        list(built.opponents),
        built.seed,
        built.mix,
        list(plan["specs"]),
        path,
        B8Controller(path),
    )


def clone_world(world: EngineWorld) -> EngineWorld:
    """Exact in-memory branch clone, used only to avoid retaining every source frame."""
    # T4's codec is intentionally the clone authority: it preserves raw grids,
    # RNG and vector-opponent carry state without marking the world prepared.
    snapshot = snapshots.capture(world.sim, world.runtime, world.specs)
    sim, runtime, policies, _ = snapshots.restore(snapshot)
    opponents = [policies[tuple(spec)] for spec in snapshot["specs"]]
    return EngineWorld(
        sim,
        runtime,
        opponents,
        world.seed,
        world.mix,
        copy.deepcopy(world.specs),
        world.b8_path,
        world.b8,
    )


def _observation(sim: Any) -> dict[str, np.ndarray]:
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    raw = build_ego_raster(
        sim, EgoRasterConfig(version="bv8"), backend="numba", rows=np.array([[0, 0]])
    )
    return {key: np.asarray(raw[key][0]).copy() for key in ("local", "global", "scalars")}


def _dispatch_opponents(world: EngineWorld, actions: np.ndarray, masks: np.ndarray) -> None:
    alive = world.sim.get_alive()
    # Reuse the evaluator's exact per-slot order, retaining stateful scripted policy RNG behaviour.
    for slot, policy in enumerate(world.opponents, start=1):
        if alive[0, slot]:
            actions[0, slot] = int(
                policy.actions(masks[0, slot : slot + 1], world.sim, np.array([[0, slot]]))[0]
            )


def step(
    world: EngineWorld, controller: Controller, *, forced_action: int | None = None
) -> StepResult:
    """Step a real world once, including reactive opponents and the exact bv8 input."""
    captured: dict[str, Any] = {}

    def choose(prepared: Any) -> np.ndarray:
        pre_alive = bool(prepared.alive[0, 0])
        world.runtime.prepare(prepared)
        masks = prepared.get_resolved_action_mask()
        # E7's frontier probe deliberately ranges over the raw mechanics-legal
        # action space.  The evaluator's resolved mask remains for opponents.
        legal = np.asarray(prepared.get_legal_action_mask()[0, 0], dtype=bool).copy()
        advisory = np.asarray(prepared.get_advisory_action_mask()[0, 0], dtype=bool).copy()
        if not advisory.any():
            advisory[:3] = True
        if legal.shape != (6,) or not legal.any():
            raise IncompleteRoot("real simulator did not resolve a legal six-action hero mask")
        obs = _observation(prepared)
        b8 = world.b8 or B8Controller(world.b8_path)
        world.b8 = b8
        b8_q = b8.q_values(obs)
        if forced_action is not None:
            action = int(forced_action)
            if not 0 <= action < 6:
                raise ValueError("forced action must be in [0, 5]")
            fallback = not bool(legal[action])
            if fallback:
                # The E7 rule explicitly says a stale tape action becomes the
                # deployed bare B8 action, not a second stochastic sample.
                action = b8.choose(obs, advisory)
        else:
            action = controller.choose(
                obs,
                advisory if getattr(controller, "mask_mode", "resolved") == "advisory" else legal,
            )
            fallback = False
        if not 0 <= action < 6 or not legal[action]:
            raise ValueError("controller selected a non-resolved action")
        actions = np.ones((1, prepared.S), dtype=np.int64)
        _dispatch_opponents(world, actions, masks)
        actions[0, 0] = action
        captured.update(
            obs=obs,
            legal=legal,
            advisory=advisory,
            b8_q=b8_q,
            action=action,
            fallback=fallback,
            pre_alive=pre_alive,
            pre_frame=int(prepared.frame[0]),
            pre_length=int(prepared.length[0, 0]),
        )
        return actions

    world.sim.step_with_policy(choose)
    world.runtime.observe_step(world.sim)
    alive = bool(world.sim.alive[0, 0])
    length = int(world.sim.length[0, 0]) if alive else 0
    return StepResult(
        captured["obs"],
        captured["action"],
        captured["legal"],
        captured["advisory"],
        captured["b8_q"],
        captured["pre_frame"],
        captured["pre_length"],
        captured["pre_alive"],
        int(world.sim.frame[0]),
        length,
        alive,
        float(length),
        captured["fallback"],
    )


def capture_root(world: EngineWorld, history: Sequence[PrefixRecord]) -> Root:
    """Capture an unprepared full state with its actual preceding 128 decisions."""
    if len(history) < PREFIX_HISTORY:
        raise IncompleteRoot("root lacks the required actual 128-decision prefix")
    prefix = tuple(history[-PREFIX_HISTORY:])
    for row in prefix:
        if (
            row.legal.shape != (6,)
            or not np.asarray(row.legal, dtype=bool).any()
            or not 0 <= row.action < 6
        ):
            raise IncompleteRoot("malformed actual prefix record")
    try:
        snapshot = snapshots.capture(world.sim, world.runtime, world.specs)
    except ValueError as exc:
        raise IncompleteRoot(str(exc)) from exc
    bound = _checkpoint_digests(world.b8_path, world.specs)
    if world.b8_path not in bound:
        raise IncompleteRoot("B8 root checkpoint is absent and cannot be identity-bound")
    return Root(
        snapshot, prefix, world.seed, world.mix, int(world.sim.frame[0]), world.b8_path, bound
    )


def restore_root(root: Root) -> EngineWorld:
    """Restore the data-only T4 state, RNG, and carried opponents from a root."""
    sim, runtime, policies, _ = snapshots.restore(root.snapshot)
    opponents = [policies[tuple(spec)] for spec in root.snapshot["specs"]]
    return EngineWorld(
        sim,
        runtime,
        opponents,
        root.seed,
        root.mix,
        copy.deepcopy(root.snapshot["specs"]),
        root.b8_path,
        B8Controller(root.b8_path),
    )


def _checkpoint_digests(b8_path: str, specs: Sequence[Sequence[Any]]) -> dict[str, str]:
    paths = {str(b8_path)}
    for spec in specs:
        paths.update(
            str(value) for value in spec if isinstance(value, str) and Path(value).is_file()
        )
    return {path: snapshots.sha256(Path(path)) for path in sorted(paths) if Path(path).is_file()}


def save_root(root: Root, path: str | Path) -> dict[str, str]:
    """Persist an immutable root as data-only T4 NPZ plus actual prefix arrays.

    The caller publishes these ordinary root files through the supervisor's
    immutable export convention; this helper deliberately does not create a
    `.complete` marker itself.
    """
    from research.bl_ppo.exports import export_bytes, export_json

    path = Path(path)
    if any(Path(str(path) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError(path)
    prefix_path = path.with_suffix(path.suffix + ".prefix.npz")
    meta_path = path.with_suffix(path.suffix + ".root.json")
    if any(
        Path(str(candidate) + suffix).exists()
        for candidate in (prefix_path, meta_path)
        for suffix in ("", ".sha256", ".complete")
    ):
        raise FileExistsError("root companion path already exists")
    with tempfile.TemporaryDirectory(dir=path.parent) as temporary:
        raw_path = Path(temporary) / "root.npz"
        snapshots.save_snapshot(raw_path, root.snapshot)
        snapshot_entry = export_bytes(path, raw_path.read_bytes())
    arrays = {
        "local": np.stack([record.observation["local"] for record in root.prefix]),
        "global": np.stack([record.observation["global"] for record in root.prefix]),
        "scalars": np.stack([record.observation["scalars"] for record in root.prefix]),
        "actions": np.asarray([record.action for record in root.prefix], np.int8),
        "legal": np.stack([record.legal for record in root.prefix]).astype(bool),
        "advisory": np.stack(
            [
                record.advisory if record.advisory is not None else record.legal
                for record in root.prefix
            ]
        ).astype(bool),
    }
    import io

    buffer = io.BytesIO()
    np.savez_compressed(buffer, **arrays)
    prefix_entry = export_bytes(prefix_path, buffer.getvalue())
    meta = {
        "schema": "bl-prefix-root/v1",
        "snapshot": path.name,
        "prefix": prefix_path.name,
        "seed": root.seed,
        "mix": root.mix,
        "frame": root.frame,
        "b8_path": root.b8_path,
        "checkpoint_sha256": dict(root.checkpoint_sha256),
        "snapshot_sha256": snapshot_entry["sha256"],
        "prefix_sha256": prefix_entry["sha256"],
    }
    meta_entry = export_json(meta_path, meta)
    return {
        "snapshot": path.name,
        "prefix": prefix_path.name,
        "metadata": meta_path.name,
        "snapshot_sha256": snapshot_entry["sha256"],
        "prefix_sha256": prefix_entry["sha256"],
        "metadata_sha256": meta_entry["sha256"],
    }


def load_root(path: str | Path) -> Root:
    """Load a portable root written by :func:`save_root`, refusing malformed history."""
    from research.bl_ppo.exports import verify_export

    path = Path(path)
    verify_export(path)
    verify_export(path.with_suffix(path.suffix + ".prefix.npz"))
    verify_export(path.with_suffix(path.suffix + ".root.json"))
    meta = json.loads(path.with_suffix(path.suffix + ".root.json").read_text())
    if meta.get("schema") != "bl-prefix-root/v1" or meta.get("snapshot") != path.name:
        raise IncompleteRoot("invalid root metadata")
    if meta.get("snapshot_sha256") != snapshots.sha256(path) or meta.get(
        "prefix_sha256"
    ) != snapshots.sha256(path.with_suffix(path.suffix + ".prefix.npz")):
        raise IncompleteRoot("root metadata hash mismatch")
    snapshot = snapshots.load_snapshot(path)
    with np.load(path.with_suffix(path.suffix + ".prefix.npz"), allow_pickle=False) as z:
        required = {"local", "global", "scalars", "actions", "legal", "advisory"}
        if set(z.files) != required or len(z["actions"]) != PREFIX_HISTORY:
            raise IncompleteRoot("root has no complete actual 128-decision prefix")
        prefix = tuple(
            PrefixRecord(
                {key: z[key][i].copy() for key in ("local", "global", "scalars")},
                int(z["actions"][i]),
                z["legal"][i].astype(bool, copy=True),
                z["advisory"][i].astype(bool, copy=True),
            )
            for i in range(PREFIX_HISTORY)
        )
    if int(snapshot["sim"]["frame"][0]) != int(meta["frame"]):
        raise IncompleteRoot("root snapshot frame differs from metadata")
    if int(snapshot["sim"]["_seeds"][0]) != int(meta["seed"]):
        raise IncompleteRoot("root snapshot seed differs from metadata")
    bound = meta.get("checkpoint_sha256")
    if not isinstance(bound, dict) or str(meta["b8_path"]) not in bound:
        raise IncompleteRoot("root has no bound checkpoint identities")
    return Root(
        snapshot,
        prefix,
        int(meta["seed"]),
        str(meta["mix"]),
        int(meta["frame"]),
        str(meta["b8_path"]),
        {str(k): str(v) for k, v in bound.items()},
    )


def relocate_root(root: Root, replacements: Mapping[str, str | Path]) -> Root:
    """Relocate shipped checkpoint paths after verifying each replacement's bytes."""

    def replace(value: str) -> str:
        target = replacements.get(value)
        target = Path(target) if target is not None else Path(value)
        expected = root.checkpoint_sha256.get(value)
        if expected is None:
            return value
        if not target.is_file() or snapshots.sha256(target) != expected:
            raise IncompleteRoot(f"checkpoint relocation identity failed: {value}")
        return str(target)

    snapshot = copy.deepcopy(root.snapshot)
    snapshot["specs"] = [
        [replace(str(item)) if isinstance(item, str) else item for item in spec]
        for spec in snapshot["specs"]
    ]
    return Root(
        snapshot,
        root.prefix,
        root.seed,
        root.mix,
        root.frame,
        replace(root.b8_path),
        {replace(key): value for key, value in root.checkpoint_sha256.items()},
    )


def _record_digest(records: Sequence[PrefixRecord]) -> str:
    digest = hashlib.sha256()
    for record in records:
        for key in ("local", "global", "scalars"):
            value = np.ascontiguousarray(record.observation[key])
            digest.update(key.encode())
            digest.update(value.tobytes())
        digest.update(np.ascontiguousarray(record.legal).tobytes())
        if record.advisory is not None:
            digest.update(np.ascontiguousarray(record.advisory).tobytes())
        digest.update(bytes([record.action]))
    return digest.hexdigest()


def verify_root_parity(root: Root, frames: int = 300) -> dict[str, Any]:
    """Prove a root against an independently reconstructed T3 source world.

    This intentionally does not call :func:`step` for the source side.  The
    source uses the frozen T3 `choose_and_step` oracle; only the restored side
    exercises this package's continuation machinery.
    """
    if type(frames) is not int or frames < 1:
        raise ValueError("parity frames must be a positive integer")
    if len(root.prefix) != PREFIX_HISTORY:
        raise IncompleteRoot("parity requires the complete actual 128-decision prefix")
    from research.t3_rollout_labels_20261015 import bank_engine
    from research.t4_snapshot_restart_20261016.build import assert_state_equal

    plan = {"seed": root.seed, "mix": root.mix, "specs": copy.deepcopy(root.snapshot["specs"])}
    original = bank_engine.build_world("s3", plan, hero_checkpoint=Path(root.b8_path))
    actual: list[PrefixRecord] = []
    while int(original.sim.frame[0]) < root.frame:
        actions, advisory, obs = bank_engine.choose_and_step(original, capture_obs=True)
        if obs is None:
            raise AssertionError("T3 source omitted a real B8 observation")
        # T3's public seam returns advisory state and full actions.  The
        # mechanics-legal mask is bound inside the restored snapshot itself.
        actual.append(
            PrefixRecord(
                {key: np.asarray(obs[key][0]).copy() for key in ("local", "global", "scalars")},
                int(actions[0, 0]),
                np.asarray(advisory, bool).copy(),
                np.asarray(advisory, bool).copy(),
            )
        )
    if int(original.sim.frame[0]) != root.frame:
        raise AssertionError("source did not reach root frame")
    if len(actual) < PREFIX_HISTORY:
        raise IncompleteRoot("source reconstruction has insufficient prefix history")
    for expected, observed in zip(root.prefix, actual[-PREFIX_HISTORY:]):
        if expected.action != observed.action or expected.advisory is None:
            raise AssertionError("stored prefix action/advisory differs from independent source")
        for key in ("local", "global", "scalars"):
            np.testing.assert_array_equal(expected.observation[key], observed.observation[key])
        np.testing.assert_array_equal(expected.advisory, observed.advisory)

    restored = restore_root(root)
    assert_state_equal(original.sim, original.runtime, restored.sim, restored.runtime)
    b8 = B8Controller(root.b8_path)
    trace = hashlib.sha256()
    for _ in range(frames):
        original_actions, original_advisory, original_obs = bank_engine.choose_and_step(
            original, capture_obs=True
        )
        if original_obs is None:
            raise AssertionError("T3 source omitted B8 observation during parity")
        original_q = bank_engine.prepared_q(original, original_obs)
        restored_step = step(restored, b8)
        if int(original_actions[0, 0]) != restored_step.action:
            raise AssertionError("restored hero action differs from independent source")
        np.testing.assert_array_equal(np.asarray(original_advisory, bool), restored_step.advisory)
        np.testing.assert_array_equal(original_q, restored_step.b8_q)
        assert_state_equal(original.sim, original.runtime, restored.sim, restored.runtime)
        trace.update(np.asarray(original_actions, np.int8).tobytes())
        trace.update(np.asarray(original_q, np.float32).tobytes())
    return {
        "schema": "bl-prefix-root-parity/v1",
        "seed": root.seed,
        "mix": root.mix,
        "root_frame": root.frame,
        "frames": frames,
        "prefix_sha256": _record_digest(root.prefix),
        "continuation_sha256": trace.hexdigest(),
    }


def replace_future_rng(world: EngineWorld, future_seed: int) -> None:
    """Replace only post-root environment RNG; policy sampling stays controller-local."""
    from src.simd_env.rng import EnvRng

    cfg = world.sim.cfg
    world.sim._rngs[0] = EnvRng(
        int(future_seed), cfg.game_width, cfg.game_height, cfg.segment_size, cfg.wall_thickness
    )


def rollout(
    root: Root,
    controller: Controller,
    b8: B8Controller,
    *,
    horizon: int = HORIZON,
    prefix_steps: int = 64,
    future_seed: int,
    forced_actions: Sequence[int] | None = None,
    max_steps: int | None = None,
    preflight: bool = False,
) -> RolloutResult:
    """Restore, reconstruct recurrent state, steer a prefix, then finish bare B8."""
    if horizon != HORIZON or not 0 <= prefix_steps <= horizon - root.frame:
        raise ValueError("continuations must retain the registered H5000 horizon")
    if max_steps is not None and (not preflight or max_steps < 1):
        raise ValueError("max_steps is permitted only for an explicit positive preflight")
    if forced_actions is not None and not 0 < len(forced_actions) <= prefix_steps:
        raise ValueError("forced prefix must be nonempty and no longer than prefix_steps")
    world = restore_root(root)
    replace_future_rng(world, future_seed)
    controller.reset()
    # Only recurrent E2 needs a real prefix burn-in.  Stochastic B8 must not
    # spend its registered exploration RNG draws reconstructing history.
    if getattr(controller, "requires_prefix_reconstruction", False):
        for record in root.prefix:
            controller.choose(
                record.observation,
                (
                    record.advisory
                    if getattr(controller, "mask_mode", "resolved") == "advisory"
                    and record.advisory is not None
                    else record.legal
                ),
            )
    b8.reset()
    mass, tape, fallbacks, trace = 0.0, [], 0, []
    last_living_length: int | None = None
    effective_prefix = len(forced_actions) if forced_actions is not None else prefix_steps
    while int(world.sim.frame[0]) < horizon and bool(world.sim.alive[0, 0]):
        if max_steps is not None and len(tape) >= max_steps:
            break
        n = len(tape)
        chosen: Controller = controller if n < effective_prefix else b8
        forced = None if forced_actions is None or n >= effective_prefix else int(forced_actions[n])
        if forced is not None:
            result = step(world, b8, forced_action=forced)
            fallbacks += int(result.forced_fallback)
        else:
            result = step(world, chosen)
        tape.append(result.action)
        mass += result.mass
        trace.append(result.mass)
        if result.pre_alive:
            last_living_length = result.pre_length
    truncated = bool(
        max_steps is not None
        and len(tape) >= max_steps
        and int(world.sim.frame[0]) < horizon
        and bool(world.sim.alive[0, 0])
    )
    return RolloutResult(
        mass,
        bool(world.sim.alive[0, 0]),
        tuple(tape),
        len(tape),
        fallbacks,
        truncated,
        tuple(trace),
        None if bool(world.sim.alive[0, 0]) else int(world.sim.frame[0]),
        last_living_length,
    )


def mass_ledger(
    candidate: RolloutResult, baseline: RolloutResult, *, prefix_steps: int = 64
) -> dict[str, float]:
    """Return the three exact paired mass terms and prefix common-alive growth.

    A rollout trace begins at the same root frame and is intentionally kept in
    memory only.  Missing tail entries are dead-frame zeros, never imputed
    observations, so the terms retain the H5000 dead-frame convention.
    """
    width = max(len(candidate.mass_trace), len(baseline.mass_trace))
    cand = candidate.mass_trace + (0.0,) * (width - len(candidate.mass_trace))
    base = baseline.mass_trace + (0.0,) * (width - len(baseline.mass_trace))
    both = candidate_only = b8_only = prefix_growth = 0.0
    for index, (candidate_mass, b8_mass) in enumerate(zip(cand, base)):
        if candidate_mass > 0.0 and b8_mass > 0.0:
            both += candidate_mass - b8_mass
            if index < prefix_steps:
                prefix_growth += candidate_mass - b8_mass
        elif candidate_mass > 0.0:
            candidate_only += candidate_mass
        elif b8_mass > 0.0:
            b8_only += b8_mass
    return {
        "both_alive_delta": both / HORIZON,
        "candidate_only_mass": candidate_only / HORIZON,
        "b8_only_mass": b8_only / HORIZON,
        "common_alive_prefix_growth": prefix_growth / HORIZON,
    }
