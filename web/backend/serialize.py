"""Serialize the live GameState into a JSON frame for the browser.

This is the single serialization boundary: the frontend never computes game
state, a state vector, or a Q-value — it renders whatever this produces.
PyQt-free on purpose (the server must run without a display).
"""

from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np
import torch

from src.core.game_config import GameConfig
from web.backend.state_labels import ACTION_LABELS, state_groups

# Hidden layer is 512-D; downsample to this many buckets for display.
HIDDEN_DISPLAY_BUCKETS = 96


def _to_numpy(x):
    """Coerce a torch tensor or array-like to a numpy array (CPU, detached)."""
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _downsample(vec: np.ndarray, buckets: int) -> List[float]:
    """Mean-pool a 1-D activation vector into `buckets` values for display."""
    if vec.size == 0:
        return []
    if vec.size <= buckets:
        return [float(v) for v in vec]
    idx = np.linspace(0, vec.size, buckets + 1).astype(int)
    return [float(vec[idx[i] : idx[i + 1]].mean()) for i in range(buckets)]


def _activation_summary(
    inp: np.ndarray, hidden: np.ndarray, out: np.ndarray, labels: List[str]
) -> Dict[str, object]:
    """Compact summary mirroring the PyQt visualizer (reimplemented PyQt-free)."""
    if out.size == 0:
        return {
            "status": "WAITING",
            "top_action": "--",
            "top_index": -1,
            "top_q": 0.0,
            "margin": 0.0,
            "input_count": 0,
            "hidden_count": 0,
            "output_count": 0,
            "input_activity": 0.0,
            "hidden_activity": 0.0,
            "architecture": "APEX DQN",
        }
    usable = out[: len(labels)]
    top_index = int(np.argmax(usable))
    top_q = float(usable[top_index])
    sorted_q = np.sort(usable)[::-1]
    runner_up = float(sorted_q[1]) if sorted_q.size > 1 else top_q
    return {
        "status": "READY",
        "top_action": labels[top_index] if top_index < len(labels) else f"A{top_index}",
        "top_index": top_index,
        "top_q": top_q,
        "margin": top_q - runner_up,
        "input_count": int(inp.size),
        "hidden_count": int(hidden.size),
        "output_count": int(out.size),
        "input_activity": float(np.abs(inp).mean()) if inp.size else 0.0,
        "hidden_activity": float(np.abs(hidden).mean()) if hidden.size else 0.0,
        "architecture": "APEX DQN",
    }


def _hero_state_tensor(hero, others, food) -> Optional[torch.Tensor]:
    """The state the hero acted on this frame (fallback: recompute, no side effects)."""
    state = getattr(hero, "last_state", None)
    if state is not None:
        return state
    try:
        return hero.get_state(others, food, update_enemy_memory=False)
    except Exception:
        return None


def _raster_activations(session, hero):
    """Activations for a raster (``raster31v2``) hero via its serving policy.

    Returns ``(input, hidden, output, acts)`` where ``acts`` carries the same
    keys (``value``/``advantages``) the vector path exposes, or ``None`` if the
    policy could not produce an observation for this snake this frame.
    """
    hero_activations = getattr(session.policy, "hero_activations", None)
    if not callable(hero_activations):
        return None
    result = hero_activations(hero.id)
    if result is None:
        return None
    _q, acts = result  # acts already numpy 1-D per key (see InferenceAgent)
    inp = np.asarray(acts["input"]).reshape(-1)
    hidden = np.asarray(acts["hidden"]).reshape(-1)
    out = np.asarray(acts["output"]).reshape(-1)
    return inp, hidden, out, acts


def _vector_activations(session, hero, others, food):
    """Activations for a vector (``vector61``) hero via ``policy.dqn``.

    Returns ``(input, hidden, output, acts)`` or ``None`` when the hero state
    cannot be built. ``acts`` tensors are detached CPU tensors.
    """
    state = _hero_state_tensor(hero, others, food)
    if state is None:
        return None
    net = session.policy.dqn
    with torch.no_grad():
        x = state.to(session.policy.device)
        if x.dim() > 1:
            x = x.view(-1)
        _q, acts = net.forward_with_activations(x)
    inp = acts["input"].view(-1).numpy()
    hidden = acts["hidden"].view(-1).numpy()
    out = acts["output"].view(-1).numpy()
    return inp, hidden, out, acts


def _build_inspector_and_netviz(session) -> tuple[Optional[dict], Optional[dict]]:
    game = session.game
    hero = next((s for s in game.snakes if s.id == session.hero_id), None)
    if hero is None or not getattr(hero, "is_alive", False):
        return None, None
    others = [s for s in game.snakes if s is not hero]

    obs_spec = str(getattr(session, "obs_spec", "vector61"))
    if obs_spec == "raster31v2":
        bundle = _raster_activations(session, hero)
    else:
        bundle = _vector_activations(session, hero, others, game.food)
    if bundle is None:
        return None, None
    inp, hidden, out, acts = bundle

    input_size = int(inp.size)
    state_list = [float(v) for v in inp]
    free_space = state_list[58:61] if input_size >= 61 else None

    inspector = {
        "input_size": input_size,
        "state": state_list,
        "groups": state_groups(input_size),
        "q_values": [float(v) for v in out],
        "action_labels": ACTION_LABELS,
        "chosen": int(np.argmax(out)) if out.size else -1,
        "free_space": free_space,
    }
    # Dueling decomposition, when the head exposes it: scalar V(s) and per-action
    # advantages A(s,a) (Q = V + (A - mean A)).
    value = None
    advantages = None
    if "value" in acts and "advantages" in acts:
        # acts values are torch tensors (vector path) or numpy arrays (raster
        # path). np.asarray(...).reshape(-1) normalizes both.
        vv = np.asarray(_to_numpy(acts["value"])).reshape(-1)
        aa = np.asarray(_to_numpy(acts["advantages"])).reshape(-1)
        if vv.size:
            value = float(vv[0])
        advantages = [float(v) for v in aa]

    netviz = {
        "input": state_list,
        "hidden_sample": _downsample(hidden, HIDDEN_DISPLAY_BUCKETS),
        "hidden_count": int(hidden.size),
        "output": [float(v) for v in out],
        "value": value,
        "advantages": advantages,
        "summary": _activation_summary(inp, hidden, out, ACTION_LABELS),
    }
    return inspector, netviz


def _hero_raster(session, hero_id) -> Optional[dict]:
    """Compact ego-raster payload for the hero, for the frontend raster viewer.

    Emits the two tactical uint8 planes (type-code + value byte, 31x31 each),
    the three strategic density planes (25x25 uint8), the scalar vector, the
    6-bit action mask, and channel legends so the frontend can expand/label the
    planes without re-deriving the featurizer contract. Returns ``None`` for the
    vector serving path (no raster policy) or when no observation is available.

    Shapes (all row-major nested int/float lists):
        - ``tactical_code``   ``(31, 31)`` uint8 type codes (see ``tactical_codes``)
        - ``tactical_value``  ``(31, 31)`` uint8 value byte
        - ``strategic``       ``(3, 25, 25)`` uint8 density planes
        - ``scalars``         ``(26,)`` float32
        - ``mask``            ``(6,)`` bool
    """
    hero_observation = getattr(session.policy, "hero_observation", None)
    if not callable(hero_observation):
        return None
    obs = hero_observation(hero_id)
    if obs is None:
        return None

    from src.simd_env.featurizer import STRATEGIC_CHANNEL_NAMES, TACTICAL_CHANNEL_NAMES

    tactical = np.asarray(obs["tactical_uint8"])  # (2, 31, 31)
    strategic = np.asarray(obs["strategic_uint8"])  # (3, 25, 25)
    scalars = np.asarray(obs["scalars"]).reshape(-1)  # (26,)
    mask = np.asarray(obs["mask"]).reshape(-1)  # (6,)

    tactical_codes = {
        "empty": 0,
        "wall": 1,
        "ambient_food": 2,
        "corpse_food": 3,
        "enemy_pred": 4,
        "own_body": 5,
        "enemy_body": 6,
        "enemy_head": 7,
        "own_head": 8,
    }
    return {
        "tactical_size": int(tactical.shape[-1]),
        "tactical_code": tactical[0].astype(int).tolist(),
        "tactical_value": tactical[1].astype(int).tolist(),
        "tactical_channels": list(TACTICAL_CHANNEL_NAMES),
        "tactical_codes": tactical_codes,
        "strategic_size": int(strategic.shape[-1]),
        "strategic": strategic.astype(int).tolist(),
        "strategic_channels": list(STRATEGIC_CHANNEL_NAMES),
        "scalars": [round(float(v), 4) for v in scalars],
        "mask": [bool(v) for v in mask],
    }


def build_frame(session) -> dict:
    """Build the full per-frame DTO sent over the WebSocket."""
    game = session.game

    snakes = []
    for s in game.snakes:
        color = getattr(s, "color", (180, 180, 180))
        snakes.append(
            {
                "id": int(s.id),
                "color": [int(color[0]), int(color[1]), int(color[2])],
                "name": getattr(s, "color_name", f"Snake {s.id}"),
                "alive": bool(getattr(s, "is_alive", False)),
                "boosting": bool(getattr(s, "is_boosting", False)),
                "length": int(getattr(s, "length", len(s.segments))),
                "segments": [[int(x), int(y)] for (x, y) in s.segments],
                "head": [int(s.head[0]), int(s.head[1])],
                "is_hero": bool(s.id == session.hero_id),
            }
        )

    loss = None
    for s in game.snakes:
        cl = getattr(s, "current_loss", None)
        if cl:
            loss = float(cl)
            break

    inspector, netviz = _build_inspector_and_netviz(session)
    obs_spec = str(getattr(session, "obs_spec", "vector61"))
    hero_raster = _hero_raster(session, session.hero_id) if obs_spec == "raster31v2" else None

    return {
        "type": "frame",
        "frame": int(game.frame),
        "obs_spec": obs_spec,
        "mechanics_version": int(GameConfig.MECHANICS_VERSION),
        "hero_raster": hero_raster,
        "arena": {
            "width": int(getattr(game, "_game_width", GameConfig.WIDTH)),
            "height": int(getattr(game, "_game_height", GameConfig.HEIGHT)),
            "segment": int(GameConfig.SEGMENT_SIZE),
            "wall": int(GameConfig.WALL_THICKNESS),
            "arena_type": str(GameConfig.ARENA_TYPE),
        },
        "snakes": snakes,
        "food": [[int(x), int(y)] for (x, y) in game.food],
        "stats": {
            "alive": int(game.alive_snakes),
            "frame": int(game.frame),
            "food_eaten": int(game.episode_food_eaten),
            "deaths": int(game.episode_deaths),
            "kills": int(game.episode_kills),
            "best_length": int(game.episode_best_length),
            "loss": loss,
            "epsilon": float(getattr(session.policy, "epsilon", 0.0)),
        },
        "session": session.control_state(),
        "play": session.play_state(),
        "inspector": inspector,
        "netviz": netviz,
    }
