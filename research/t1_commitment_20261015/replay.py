"""Census world construction and isolated, exact B8 replay execution."""

from __future__ import annotations

import hashlib
import random
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np

from research.b8_death_census_20261014 import census

ARTIFACTS = census.ARTIFACTS / "t1-commitment-20261015"
INPUTS = census.OUT_ROOT / "runs"


def configure_cpu() -> None:
    """Pin each numerical process to one CPU thread."""
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass


def _hash_value(h: Any, value: Any) -> None:
    if isinstance(value, np.ndarray):
        h.update(str((value.dtype.str, value.shape)).encode())
        if value.dtype.hasobject:
            _hash_value(h, value.tolist())
        else:
            h.update(np.ascontiguousarray(value).tobytes())
    elif isinstance(value, random.Random):
        _hash_value(h, value.getstate())
    elif isinstance(value, np.random.Generator):
        _hash_value(h, value.bit_generator.state)
    elif isinstance(value, dict):
        for key in sorted(value, key=repr):
            _hash_value(h, key)
            _hash_value(h, value[key])
    elif isinstance(value, (set, frozenset)):
        _hash_value(h, sorted(value, key=repr))
    elif isinstance(value, (tuple, list)):
        h.update(type(value).__name__.encode())
        for element in value:
            _hash_value(h, element)
    elif isinstance(value, types.ModuleType):
        h.update(value.__name__.encode())
    elif hasattr(value, "__dict__"):
        _hash_value(h, vars(value))
    else:
        h.update(repr(value).encode())
    h.update(b"\0")


def state_fields(sim: Any, opponents: Sequence[Any]) -> dict:
    """Hash every simulator field, including raw grids, storage, RNG and guards."""
    out = {}
    for name, value in {**vars(sim), "opponent_policies": list(opponents)}.items():
        h = hashlib.sha256()
        _hash_value(h, value)
        out[name] = h.hexdigest()
    return out


def assert_same_world(left: "Replay", right: "Replay") -> None:
    """Give exact field names on any full-state divergence."""
    a = state_fields(left.sim, left.opponents)
    b = state_fields(right.sim, right.opponents)
    different = [k for k in sorted(a.keys() | b.keys()) if a.get(k) != b.get(k)]
    assert not different, f"full-state divergence: {different}"


class _Captured(Exception):
    pass


class _ForcedPolicy:
    batched_rows = True

    def __init__(self, actions: Sequence[int]) -> None:
        self.forced = np.asarray(actions, dtype=np.int64)

    def actions(self, masks: Any, sim: Any, slots: np.ndarray) -> np.ndarray:
        return np.maximum(0, self.forced[slots[:, 0]])


@dataclass
class Replay:
    """B8 executor; stateless networks and scripted anchors are safely shared."""

    sim: Any
    hero: Any
    opponents: list
    seeds: list
    kinds: list
    env_frames: int = 0

    def clone(self, prepared: bool = False) -> "Replay":
        sim = census.clone_sim(self.sim)
        if prepared:
            # A selector snapshot precedes step_with_policy's guard cleanup. Match
            # census.pre_step's already-prepared snapshot, without preparing twice.
            sim._policy_phase_prepared = True
            sim._policy_step_in_progress = False
            if hasattr(sim, "_active_env_mask"):
                del sim._active_env_mask
        return Replay(sim, self.hero, self.opponents, self.seeds, self.kinds)

    def q(self) -> np.ndarray:
        from src.simd_env.ego2s_policy import student_q

        return student_q(self.hero.net, self.sim, np.array([[0, 0]]), "rowwise").numpy()[0]

    def actions(self, forced: Optional[Sequence[int]] = None) -> np.ndarray:
        from src.simd_env.eval_engine import _dispatch_actions

        masks = self.sim.get_resolved_action_mask()
        actions = np.ones((self.sim.E, self.sim.S), np.int64)
        heroes = (
            [self.hero] * self.sim.E if forced is None else [_ForcedPolicy(forced)] * self.sim.E
        )
        _dispatch_actions(self.sim, masks, actions, heroes, self.opponents)
        return actions

    def step(
        self,
        forced: Optional[Sequence[int]] = None,
        observer: Optional[Callable[["Replay"], None]] = None,
    ) -> None:
        def choose(sim: Any) -> np.ndarray:
            if observer is not None:
                observer(self)
            return self.actions(forced)

        self.sim.step_with_policy(choose)
        self.env_frames += int(self.sim.E)

    def first(self, action: int) -> None:
        """Apply one intervention to an already prepared single-world clone."""
        assert self.sim.E == 1 and getattr(self.sim, "_policy_phase_prepared", False)
        self.sim.step(self.actions([action]))
        self.env_frames += 1


def build_replay(arm: str, envs: Sequence[int], context: Any = None) -> Replay:
    """Capture the census evaluator's initial sim and actual policy objects.

    No engine configuration is duplicated, and no transition is consumed. Reference
    arms are deliberately rejected: their stateful vector61 runtime needs a separate
    executor and is optional in this Part A spec.
    """
    from src.simd_env import eval_engine as ee
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    if arm not in census.B8_ARMS:
        raise ValueError("this executor supports B8 only; references are optional")
    ctx, all_seeds, all_rosters = context or census.load_context()
    seeds = [all_seeds[int(e)] for e in envs]
    rosters = {s: all_rosters[s] for s in seeds}
    cls = ee._terminal_hero_sim_class("grid")
    original = cls.step_with_policy
    captured = {}

    def capture(sim: Any, selector: Any, *args: Any, **kwargs: Any) -> None:
        values = dict(
            zip(selector.__code__.co_freevars, (c.cell_contents for c in selector.__closure__))
        )
        captured.update(sim=sim, **values)
        raise _Captured

    cls.step_with_policy = capture
    try:
        census.run_eval(ctx, seeds, rosters, arm, census.HORIZON, "numba", "fast")
    except _Captured:
        pass
    finally:
        cls.step_with_policy = original
    assert captured, "census evaluator did not reach its first transition"
    opponents = captured["opp_policies"]
    assert all(isinstance(p, FastProfileAnchorSimdPolicy) for row in opponents for p in row)
    assert not np.any(captured["vector61_runtime"].controlled)
    kinds = [[census.spec_kind(s) for s in rosters[seed]] for seed in seeds]
    return Replay(captured["sim"], captured["hero_policies"][0], opponents, seeds, kinds)


def advance_recorded(replay: Replay, series: Any, envs: Sequence[int], stop: int) -> int:
    """Replay up to (but excluding) decision stop; assert every pre-action length."""
    compared = 0
    while int(replay.sim.frame[0]) < stop:
        t = int(replay.sim.frame[0])

        def check(r: Replay) -> None:
            nonlocal compared
            for local, source in enumerate(envs):
                expected = int(series["length"][source, t])
                actual = int(r.sim.length[local, 0]) if r.sim.alive[local, 0] else 0
                assert actual == expected, (source, t, actual, expected)
                compared += 1

        replay.step(series["action"][list(envs), t], check)
    return compared


def load_inputs(arm: str) -> tuple:
    import json

    data = json.loads((INPUTS / f"{arm}.json").read_text())
    with np.load(INPUTS / f"{arm}_series.npz") as archive:
        series = {k: archive[k].copy() for k in archive.files}
    checkpoint = Path(data["checkpoint"]["path"])
    assert census.sha256_file(checkpoint) == data["checkpoint"]["sha256"]
    assert checkpoint.resolve() == census.b8_path(arm).resolve()
    return data, series
