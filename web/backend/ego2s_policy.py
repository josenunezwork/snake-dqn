"""Serve an ``ego2s-draft`` / ``ego2s-b`` student behind the live web game (Watch / Play).

Mirrors :class:`web.backend.raster_policy.RasterServingPolicy`: it duck-types the
``ApexPolicy`` surface the game and serializer use (``training`` / ``epsilon`` /
``device`` / ``dqn`` / ``select_action``) so ``AISnake`` and ``GameState`` run unchanged.

Once per frame (``prepare_frame``, called by ``GameState.update`` step 6 after respawns and
before any snake moves; or lazily on the first ``dqn`` call) it rasterises the live game
with :func:`~src.simd_env.ego_live_adapter.game_state_to_ego_view` and runs the SAME
featurizer the student was trained on (numba backend, version from the checkpoint) for
every living snake, then one batched forward. Serve-time observations are bitwise the
training observations (``tests/test_ego_live_identity.py`` and the gate-world harness
``research/redesign_scope_20261007/ego_live_identity.py --obs-version b``).

``dqn(vector_state)`` returns the calling snake's Q row; identity follows the per-frame
dispatch order of ``GameState.update`` (alive AI snakes using this policy, in
``game.snakes`` order), exactly as the raster serving policy does, so epsilon is pinned to
0. ``AISnake`` applies its own safe-action mask (advisory + normal-only fallback) to the
Q row, as it does for every served policy. Forward-only: training is never enabled.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model.ego2s_network import load_ego2s_checkpoint, obs_spec_of, obs_tensors
from src.simd_env.ego2s_policy import ego_config_for
from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_raster import build_ego_raster

__all__ = ["Ego2sServingPolicy"]


class _Ego2sDQNShim:
    def __init__(self, owner: "Ego2sServingPolicy") -> None:
        self._owner = owner

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self._owner._next_dispatch_q(state)

    def eval(self) -> "_Ego2sDQNShim":
        return self

    def parameters(self):  # pragma: no cover - passthrough
        return self._owner.net.parameters()


class Ego2sServingPolicy:
    """Forward-only ego2s student serving policy for the live web game."""

    def __init__(self, checkpoint_path: str, device: Optional[torch.device] = None) -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.obs_spec = obs_spec_of(checkpoint_path)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.net = load_ego2s_checkpoint(checkpoint_path, self.device)
        self.ego_cfg = ego_config_for(self.net)
        self.output_size = 6
        self.training = False
        self.total_reward = 0.0
        self.memory = None
        self.dqn = _Ego2sDQNShim(self)
        self._game = None
        self._invalidate()

    # epsilon pinned to 0 (see RasterServingPolicy: positional dispatch forbids skips)
    @property
    def epsilon(self) -> float:
        return 0.0

    @epsilon.setter
    def epsilon(self, value) -> None:  # noqa: D401 - intentional no-op
        return

    def attach_game(self, game) -> None:
        self._game = game
        self._invalidate()

    def prepare_frame(self) -> None:
        self._ensure_frame()

    def _invalidate_cache(self) -> None:
        """The name ``GameSession.reset_game`` calls (as on the raster serving policy)."""
        self._invalidate()

    def _invalidate(self) -> None:
        self._frame: Optional[int] = None
        self._q: Optional[np.ndarray] = None  # (S, 6), rows in game.snakes order
        self._obs: Optional[Dict[str, np.ndarray]] = None
        self._id_to_row: Dict[int, int] = {}
        self._dispatch: List[int] = []

    def _ensure_frame(self) -> None:
        game = self._game
        if game is None:
            return
        frame = int(getattr(game, "frame", 0))
        if self._frame == frame and self._q is not None:
            return
        snakes = list(game.snakes)
        view = game_state_to_ego_view(game)
        rows = np.array(
            [[0, i] for i in range(len(snakes)) if view.alive[0, i]], dtype=np.int64
        ).reshape(-1, 2)
        q = np.zeros((len(snakes), self.output_size), dtype=np.float32)
        obs = None
        if len(rows):
            obs = build_ego_raster(view, self.ego_cfg, backend="numba", rows=rows)
            with torch.no_grad():
                q_rows = self.net(*obs_tensors(obs, self.device)).float().cpu().numpy()
            q[rows[:, 1]] = q_rows
        self._frame = frame
        self._q = q
        self._obs = (
            {"rows": rows[:, 1], **{k: np.asarray(v) for k, v in obs.items()}} if obs else None
        )
        self._id_to_row = {int(s.id): row for row, s in enumerate(snakes)}
        from src.game.ai_snake import AISnake

        self._dispatch = [
            int(s.id)
            for s in snakes
            if isinstance(s, AISnake)
            and bool(getattr(s, "is_alive", False))
            and getattr(s, "policy", None) is self
        ]

    def _next_dispatch_q(self, state: torch.Tensor) -> torch.Tensor:
        self._ensure_frame()
        if self._q is None:
            n = 1 if state is None or state.dim() == 1 else int(state.shape[0])
            return torch.zeros((n, self.output_size), device=self.device)
        row = self._id_to_row.get(self._dispatch.pop(0), 0) if self._dispatch else 0
        return torch.as_tensor(self._q[row], dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state, snake_id: Optional[int] = None, action_mask=None) -> int:
        self._ensure_frame()
        if self._q is None:
            return 0
        if snake_id is not None:
            row = self._id_to_row.get(int(snake_id), 0)
        elif self._dispatch:
            row = self._id_to_row.get(self._dispatch.pop(0), 0)
        else:
            row = 0
        q = self._q[row].astype(np.float64)
        if action_mask is not None:
            m = np.asarray(action_mask, dtype=bool).reshape(-1)
            if m.size == q.size and m.any():
                q = np.where(m, q, -np.inf)
        return int(np.argmax(q))

    # -- inspector helpers (serializer) -------------------------------------
    def hero_observation(self, snake_id: int):
        """``None``: the raster viewer's planes are raster31-specific (not shown)."""
        return None

    def q_for(self, snake_id: int) -> Optional[np.ndarray]:
        self._ensure_frame()
        if self._q is None or int(snake_id) not in self._id_to_row:
            return None
        return self._q[self._id_to_row[int(snake_id)]].copy()

    def hero_activations(self, snake_id: int) -> Optional[Tuple[np.ndarray, Dict[str, object]]]:
        """``(q, acts)`` with the serializer's keys (input = the scalar band)."""
        self._ensure_frame()
        if self._obs is None or int(snake_id) not in self._id_to_row:
            return None
        row = self._id_to_row[int(snake_id)]
        pos = np.flatnonzero(self._obs["rows"] == row)
        if not len(pos):
            return None
        i = int(pos[0])
        one = {k: self._obs[k][i : i + 1] for k in ("local", "global", "scalars")}
        net = self.net
        with torch.no_grad():
            loc, glob, sca = obs_tensors(one, self.device)
            h = net.fuse(
                torch.cat(
                    [
                        net.local(loc.float() * (1.0 / 255.0)),
                        net.glob(glob.float() * (1.0 / 255.0)),
                        sca.float(),
                    ],
                    dim=1,
                )
            )
            value = net.value(h)
            adv = net.adv(h)
            q = value + adv - adv.mean(dim=1, keepdim=True)
        acts = {
            "input": one["scalars"].reshape(-1),
            "hidden": h.cpu().numpy().reshape(-1),
            "output": q.cpu().numpy().reshape(-1),
            "value": value.cpu().numpy().reshape(-1),
            "advantages": adv.cpu().numpy().reshape(-1),
        }
        return acts["output"], acts
