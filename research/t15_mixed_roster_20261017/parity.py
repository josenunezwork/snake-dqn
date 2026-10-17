"""T15 mixed-Watch identity probe.

This is deliberately an evidence producer, rather than a serving hook.  It creates a
default-constructed ``GameSession`` with the opt-in ``mixed`` environment, then runs the
same world in the GridBatch evaluator.  The evaluator only exposes a hero-veto public
argument, so the probe temporarily extends its *constructed* FRP3 simulator policy with
the already implemented v8 rule on slots ``1..N-1``.  No evaluator or serving source is
changed.  The report records that local instrumentation explicitly.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence
from unittest.mock import patch

import numpy as np

CHECKPOINT_B8 = "b8-s3"
CHECKPOINT_FRP3 = "frp3-s12"
V8_LAMBDA = 8.0
_ENV = (
    "SNAKE_SERVE_WATCH_ROSTER",
    "SNAKE_SERVE_CHECKPOINT",
    "SNAKE_SERVE_VETO_VARIANT",
    "SNAKE_SERVE_VETO_WATCH_HERO",
    "SNAKE_SERVE_VETO_PLAY_AI",
)


def _digest(value: Any) -> str:
    """Stable digest for numerical arrays and JSON-safe values."""
    if isinstance(value, np.ndarray):
        return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()
    canonical = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _first_difference(left: Sequence[Any], right: Sequence[Any]) -> int | None:
    """First unequal item, including an unequal length as a final difference."""
    for index, (a, b) in enumerate(zip(left, right)):
        if a != b:
            return index
    return None if len(left) == len(right) else min(len(left), len(right))


def _comparison(left: Sequence[Any], right: Sequence[Any]) -> dict[str, Any]:
    first = _first_difference(left, right)
    return {
        "live_count": len(left),
        "eval_count": len(right),
        "first_difference": first,
        "live_sha256": _digest(list(left)),
        "eval_sha256": _digest(list(right)),
        "identical": first is None and _digest(list(left)) == _digest(list(right)),
    }


@contextlib.contextmanager
def _mixed_environment() -> Iterator[None]:
    """Set exactly the public mixed-roster selector and restore all caller overrides."""
    saved = {key: os.environ.get(key) for key in _ENV}
    for key in _ENV:
        os.environ.pop(key, None)
    os.environ["SNAKE_SERVE_WATCH_ROSTER"] = "mixed"
    try:
        yield
    finally:
        for key, value in saved.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _roster_count(num_snakes: int) -> Iterator[None]:
    """Constructor-local Watch count override for the registered 6/8/12 probes."""
    if num_snakes not in (6, 8, 12):
        raise ValueError("T15 identity supports only 6, 8, or 12 snakes")
    if num_snakes == 6:
        yield
        return
    from web.backend import session as session_module

    loader = session_module.load_config

    def counted(path: str) -> Any:
        config = loader(path)
        return replace(config, game=replace(config.game, num_snakes=int(num_snakes)))

    with patch("web.backend.session.load_config", side_effect=counted):
        yield


def _packet(
    slot: int, state: np.ndarray, q: np.ndarray, action: int | None = None
) -> dict[str, Any]:
    return {
        "slot": int(slot),
        "state_sha256": _digest(np.asarray(state, dtype=np.float32)),
        "q_sha256": _digest(np.asarray(q, dtype=np.float32)),
        "action": None if action is None else int(action),
    }


def _assert_product_roster(session: Any, count: int) -> dict[str, Any]:
    """Fail closed unless this is the product-owned mixed roster, never T13 mutation."""
    snakes = list(session.game.snakes)
    state = session.safety_veto_state() or {}
    opponent = dict(getattr(session, "watch_opponent_checkpoint", None) or {})
    expected = list(range(1, count))
    if getattr(session, "watch_roster", None) != "mixed":
        raise RuntimeError(
            f"mixed selector did not activate: {getattr(session, 'watch_roster', None)!r}"
        )
    if (session.served_checkpoint or {}).get("name") != CHECKPOINT_B8:
        raise RuntimeError(f"mixed hero is not B8: {session.served_checkpoint!r}")
    if opponent.get("name") != CHECKPOINT_FRP3:
        raise RuntimeError(f"mixed opponent pin failed: {opponent!r}")
    if len(snakes) != count or int(session.hero_id) != 0:
        raise RuntimeError(
            f"mixed roster shape differs from T15 request: {len(snakes)}, {session.hero_id}"
        )
    if type(snakes[0].policy).__name__ != "Ego2sServingPolicy" or snakes[0].safety_veto is not None:
        raise RuntimeError("mixed hero must be the unwrapped ego2s B8 policy")
    if state.get("scope") != "watch_opponents" or state.get("wrapped_snake_ids") != expected:
        raise RuntimeError(f"mixed v8 scope differs from slots 1..N-1: {state!r}")
    if any(s.safety_veto is None for s in snakes[1:]):
        raise RuntimeError("every mixed FRP3 opponent must have a v8 wrapper")
    return {"hero": dict(session.served_checkpoint or {}), "opponent": opponent, "veto": state}


def _live_side(seed: int, horizon: int, num_snakes: int) -> dict[str, Any]:
    """Run the real product roster and capture decision-time role evidence."""
    from research.apex_veto_serving_20261001 import serving_run as v2run
    from research.apex_veto_serving_20261001.serving_run import set_world_seed
    from web.backend.session import GameSession

    hero_packets: list[dict[str, Any] | None] = []
    vector_packets: list[list[dict[str, Any]]] = []
    hero_active: list[bool] = []
    active_opponent_slots: list[list[int]] = []
    frames: list[str] = []
    with (
        v2run.parity_config(),
        _mixed_environment(),
        _roster_count(num_snakes),
        patch("web.backend.session.served_resolution"),
        patch("web.backend.session.watch_tick"),
    ):
        set_world_seed(int(seed))
        session = GameSession()
    session._release_monitoring = False
    identity = _assert_product_roster(session, num_snakes)
    hero = session.game.snakes[0]
    # Eval is terminal-hero; this local qualification alignment never changes serving.
    hero.auto_respawn = False
    session.policy.forward = "rowwise"
    opponent_policy = session.game.snakes[1].policy
    original_forward = opponent_policy.dqn.forward
    original_hero_dispatch = session.policy._next_dispatch_q
    active_slot: list[int | None] = [None]
    dispatched: dict[int, int] = {}
    original_updates = {id(snake): snake.update for snake in session.game.snakes}

    def observed_forward(state: Any, *args: Any, **kwargs: Any) -> Any:
        q = original_forward(state, *args, **kwargs)
        states = state.detach().cpu().numpy()
        values = q.detach().cpu().numpy()
        slot = active_slot[0]
        if slot is None:
            raise RuntimeError("vector61 forward occurred outside an AI snake dispatch")
        for value, result in zip(states, values):
            vector_packets[-1].append(_packet(slot, value, result))
        return q

    def observed_hero_dispatch(state: Any) -> Any:
        """Capture the actual GameState decision phase, after its frame increment/respawns."""
        obs = session.policy._obs
        if not obs or not len(obs["rows"]):
            raise RuntimeError("live ego2s hero has no decision observation")
        hero_row = int(np.flatnonzero(np.asarray(obs["rows"]) == 0)[0])
        packed = np.concatenate(
            [
                obs["local"][hero_row].reshape(-1),
                obs["global"][hero_row].reshape(-1),
                obs["scalars"][hero_row].reshape(-1),
            ]
        )
        hero_packets[-1] = _packet(0, packed, session.policy.q_for(0))
        return original_hero_dispatch(state)

    opponent_policy.dqn.forward = observed_forward
    session.policy._next_dispatch_q = observed_hero_dispatch
    for snake in session.game.snakes:

        def observed_update(*args: Any, _snake: Any = snake, **kwargs: Any) -> Any:
            active_slot[0] = int(_snake.id)
            if int(_snake.id) == 0:
                # This marks the actual GameState update dispatch, independently
                # from the ego2s Q/observation hook above.
                hero_active[-1] = True
            try:
                result = original_updates[id(_snake)](*args, **kwargs)
            finally:
                active_slot[0] = None
            action = getattr(_snake, "_pre_collision_action", None)
            if action is not None:
                dispatched[int(_snake.id)] = int(action)
            return result

        snake.update = observed_update
    try:
        for _ in range(int(horizon)):
            hero_packets.append(None)
            vector_packets.append([])
            hero_active.append(False)
            active_opponent_slots.append([])
            dispatched.clear()
            before = int(session.game.frame)
            session.step()
            if int(session.game.frame) != before + 1 or session.last_error:
                raise RuntimeError(f"mixed Watch failed at frame {before}: {session.last_error!r}")
            if hero_packets[-1] is not None:
                if 0 not in dispatched:
                    raise RuntimeError("live ego2s observation had no executed action")
                hero_packets[-1]["action"] = dispatched[0]
            for packet in vector_packets[-1]:
                slot = int(packet["slot"])
                if slot not in dispatched:
                    raise RuntimeError(
                        f"vector61 observation for slot {slot} had no executed action"
                    )
                packet["action"] = dispatched[slot]
            vector_packets[-1].sort(key=lambda packet: int(packet["slot"]))
            active_opponent_slots[-1] = sorted(slot for slot in dispatched if slot != 0)
            from research.frp3_checkpoint_serving_20261005.serving_run import live_frame_signature

            frames.append(_digest(live_frame_signature(session.game)))
    finally:
        opponent_policy.dqn.forward = original_forward
        session.policy._next_dispatch_q = original_hero_dispatch
        for snake in session.game.snakes:
            snake.update = original_updates[id(snake)]
    return {
        "identity": identity,
        "hero_packets": hero_packets,
        "vector_packets": vector_packets,
        "hero_active": hero_active,
        "active_opponent_slots": active_opponent_slots,
        "frame_digests": frames,
        "terminal_alignment": {"hero_auto_respawn": False, "serving_default_continuous": True},
        "parity_config": {
            "path": str(v2run.PARITY_CONFIG),
            "sha256": hashlib.sha256(Path(v2run.PARITY_CONFIG).read_bytes()).hexdigest(),
            "session_config_path": str(session.config_path),
        },
    }


@contextlib.contextmanager
def _eval_recorders(
    opponent_path: str,
    num_snakes: int,
    hero_packets: dict[int, dict[str, Any]],
    vector_packets: dict[int, list],
    hero_active: dict[int, bool],
    active_opponent_slots: dict[int, list[int]],
):
    """Instrument only constructed evaluator policies; source bytes remain untouched."""
    import src.simd_env.ego2s_policy as ep
    import src.simd_env.eval_engine as ee

    base_vector, base_ego = ee.Vector61SimdPolicy, ep.Ego2sSimdPolicy

    class RecordedVector(base_vector):
        def __init__(self, checkpoint_path: str, *args: Any, **kwargs: Any) -> None:
            if str(checkpoint_path) == str(opponent_path) and not kwargs.get("veto_slots"):
                kwargs.update(
                    veto_slots=tuple(range(1, num_snakes)), veto_variant="v8", veto_lambda=V8_LAMBDA
                )
            super().__init__(checkpoint_path, *args, **kwargs)

        def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
            state, _mask = self.runtime.selection(slots)
            q = self.q_values(state).detach().cpu().numpy()
            action = super().actions(masks, sim, slots)
            vector_packets[int(sim.frame[0])] = [
                _packet(int(slot[1]), s, value, a)
                for slot, s, value, a in zip(slots, state, q, action)
            ]
            active_opponent_slots[int(sim.frame[0])] = sorted(int(slot[1]) for slot in slots)
            return action

    class RecordedEgo(base_ego):
        def actions(self, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
            from src.simd_env.ego2s_policy import student_q

            # The evaluator invoked this selected role before any Q/packet capture.
            hero_active[int(sim.frame[0])] = True
            q = student_q(self.net, sim, slots, self.forward).numpy()
            action = super().actions(masks, sim, slots)
            # Hash all three tensors in the same packed row order as live evidence.
            from src.simd_env.ego_raster import build_ego_raster

            obs = build_ego_raster(
                sim,
                self.net
                and __import__(
                    "src.simd_env.ego2s_policy", fromlist=["ego_config_for"]
                ).ego_config_for(self.net),
                backend="numba",
                rows=slots,
            )
            packed = np.concatenate(
                [
                    obs["local"][0].reshape(-1),
                    obs["global"][0].reshape(-1),
                    obs["scalars"][0].reshape(-1),
                ]
            )
            hero_packets[int(sim.frame[0])] = _packet(
                int(slots[0][1]), packed, q[0], int(action[0])
            )
            return action

    with (
        patch.object(ee, "Vector61SimdPolicy", RecordedVector),
        patch.object(ep, "Ego2sSimdPolicy", RecordedEgo),
    ):
        yield


def _eval_side(
    seed: int, horizon: int, num_snakes: int, live_identity: Mapping[str, Any]
) -> dict[str, Any]:
    """Run GridBatchSim with the same role policy identities and opponent-v8 instrumentation."""
    from research.apex_veto_serving_20261001 import serving_run as v2run
    from research.t12_b8_serving_20261017 import parity as t12
    from src.core.config_loader import load_and_initialize_config
    from src.simd_env import eval_engine as ee

    hero_packets: dict[int, dict[str, Any]] = {}
    vector_packets: dict[int, list[dict[str, Any]]] = {}
    hero_active: dict[int, bool] = {}
    active_opponent_slots: dict[int, list[int]] = {}
    frames: list[str] = []
    hero_path = str(live_identity["hero"]["path"])
    opponent_path = str(live_identity["opponent"]["path"])
    profile = t12._context(int(horizon))["profile"]
    # T12's profile is six-slot.  The T15 sibling protocol also qualifies its
    # count-specific Watch8/Watch12 worlds; roster size is the only profile field
    # altered here, and the returned profile digest consequently binds that fact.
    if int(profile.world.num_snakes) != int(num_snakes):
        profile = replace(profile, world=replace(profile.world, num_snakes=int(num_snakes)))
    original = ee._hero_post_frame_snapshot

    def traced(sim: Any, events: Any) -> Any:
        from research.frp3_checkpoint_serving_20261005.serving_run import simd_frame_signature

        frames.append(_digest(simd_frame_signature(sim, 0)))
        return original(sim, events)

    load_and_initialize_config(str(v2run.PARITY_CONFIG))
    ee._hero_post_frame_snapshot = traced
    try:
        with _eval_recorders(
            opponent_path,
            num_snakes,
            hero_packets,
            vector_packets,
            hero_active,
            active_opponent_slots,
        ):
            record = ee.run_simd_eval(
                ("checkpoint", opponent_path),
                [("checkpoint", opponent_path)] * (num_snakes - 1),
                int(horizon),
                [int(seed)],
                profile=profile,
                mix_id="t15-mixed-watch",
                vector61=True,
                vector61_forward="rowwise",
                sim_engine="grid",
                hero_ego2s=hero_path,
                ego2s_forward="rowwise",
                ego2s_mask="advisory",
                frame_observer=lambda _frame: None,
            )[0]
    finally:
        ee._hero_post_frame_snapshot = original
    # ``sim.frame`` is incremented before the evaluator's choice phase.  Preserve
    # missing decisions as explicit inactive entries, including after terminal hero death.
    hero_stream = [hero_packets.get(frame) for frame in range(1, int(horizon) + 1)]
    vector_stream = [vector_packets.get(frame, []) for frame in range(1, int(horizon) + 1)]
    hero_active_stream = [
        bool(hero_active.get(frame, False)) for frame in range(1, int(horizon) + 1)
    ]
    active_opponent_stream = [
        active_opponent_slots.get(frame, []) for frame in range(1, int(horizon) + 1)
    ]
    return {
        "record": record,
        "hero_packets": hero_stream,
        "vector_packets": vector_stream,
        "hero_active": hero_active_stream,
        "active_opponent_slots": active_opponent_stream,
        "frame_digests": frames,
        "instrumentation": {
            "opponent_veto": "harness-local Vector61SimdPolicy slots 1..N-1 v8 lambda=8",
            "source_mutation": False,
        },
        "profile": {
            "name": profile.name,
            "digest": profile.digest,
            # ``EffectiveWorldConfig.normalization`` is a mappingproxy, so
            # dataclasses.asdict cannot deepcopy it on Python 3.12.
            "world": {
                field.name: (
                    dict(getattr(profile.world, field.name))
                    if field.name == "normalization"
                    else getattr(profile.world, field.name)
                )
                for field in dataclasses.fields(profile.world)
            },
        },
        "record_metadata": {
            key: record.get(key)
            for key in (
                "world_identity",
                "world_runtime_spec",
                "world_runtime_spec_digest",
                "vector61_policy",
                "ego2s_hero",
            )
            if key in record
        },
    }


def run_identity(seed: int, horizon: int, num_snakes: int = 6) -> dict[str, Any]:
    """Return full, hash-bound T15 role identity evidence for one fresh world."""
    if int(horizon) <= 0:
        raise ValueError("horizon must be positive")
    live = _live_side(int(seed), int(horizon), int(num_snakes))
    simd = _eval_side(int(seed), int(horizon), int(num_snakes), live["identity"])
    checks = {
        "world": _comparison(live["frame_digests"], simd["frame_digests"]),
        "hero_ego2s": _comparison(live["hero_packets"], simd["hero_packets"]),
        "opponent_vector61": _comparison(live["vector_packets"], simd["vector_packets"]),
        "active_roles": _comparison(
            list(zip(live["hero_active"], live["active_opponent_slots"])),
            list(zip(simd["hero_active"], simd["active_opponent_slots"])),
        ),
    }
    return {
        "schema_version": "t15-mixed-watch-identity/v1",
        "seed": int(seed),
        "horizon": int(horizon),
        "num_snakes": int(num_snakes),
        "product_session": (
            "GameSession() with SNAKE_SERVE_WATCH_ROSTER=mixed and no checkpoint argument"
        ),
        "identity": live["identity"],
        "terminal_alignment": live["terminal_alignment"],
        "parity_config": live["parity_config"],
        "eval_profile": simd["profile"],
        "eval_record_metadata": simd["record_metadata"],
        "checks": checks,
        # These slot-labelled packets are the audit material behind the chain hashes
        # above. ``None`` means the terminal hero had no decision that frame; an
        # empty list means no vector opponent dispatched that frame.
        "evidence": {
            "live": {
                "frames": live["frame_digests"],
                "hero_ego2s": live["hero_packets"],
                "opponent_vector61": live["vector_packets"],
                "hero_active": live["hero_active"],
                "active_opponent_slots": live["active_opponent_slots"],
            },
            "eval": {
                "frames": simd["frame_digests"],
                "hero_ego2s": simd["hero_packets"],
                "opponent_vector61": simd["vector_packets"],
                "hero_active": simd["hero_active"],
                "active_opponent_slots": simd["active_opponent_slots"],
            },
        },
        "eval_instrumentation": simd["instrumentation"],
        "pass": all(bool(value["identical"]) for value in checks.values()),
    }
