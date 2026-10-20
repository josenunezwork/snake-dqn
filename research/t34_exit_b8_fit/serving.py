"""Opt-in live policy for an ExIt tower candidate.

This duck-types the narrow ``ApexPolicy`` surface the game consumes, but it is
not registered in ``web.backend.served_checkpoint`` and cannot change a default.
It exists only for a future candidate-specific serving qualification.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import numpy as np
import torch

from research.t34_exit_b8_fit.full_sight import full_sight_from_ego_view
from research.t34_exit_b8_fit.network import load_exit_fit_tower_checkpoint
from src.model.ego2s_network import obs_tensors
from src.simd_env.ego2s_policy import ego_config_for
from src.simd_env.ego_live_adapter import game_state_to_ego_view
from src.simd_env.ego_raster import build_ego_raster

__all__ = ["ExitFitTowerServingPolicy"]

_LOG = logging.getLogger(__name__)


class _DQNShim:
    def __init__(self, owner: "ExitFitTowerServingPolicy") -> None:
        self._owner = owner

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self._owner._next_dispatch_q(state)

    def eval(self) -> "_DQNShim":
        return self

    def parameters(self):  # pragma: no cover - standard policy passthrough
        return self._owner.net.parameters()


class ExitFitTowerServingPolicy:
    """Forward-only, explicit-candidate full-sight policy for a live GameState.

    No registry, environment variable, pin, or released checkpoint is touched.
    The caller must construct this class directly during a separately approved
    candidate qualification.
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: Optional[torch.device] = None,
        *,
        forward: str = "batched",
    ) -> None:
        if forward not in ("batched", "rowwise"):
            raise ValueError("forward must be 'batched' or 'rowwise'")
        self.checkpoint_path = str(checkpoint_path)
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.net = load_exit_fit_tower_checkpoint(checkpoint_path, self.device)
        self.ego_cfg = ego_config_for(self.net.base)
        self.forward = forward
        self.obs_spec = "ego2s-bv8-full-sight"
        self.output_size = 6
        self.training = False
        self.total_reward = 0.0
        self.memory = None
        self.dqn = _DQNShim(self)
        self._game: Any = None
        self._invalidate()

    @property
    def epsilon(self) -> float:
        return 0.0

    @epsilon.setter
    def epsilon(self, value: object) -> None:
        return

    def attach_game(self, game: Any) -> None:
        self._game = game
        self._invalidate()

    def prepare_frame(self) -> None:
        self._ensure_frame()

    def _invalidate_cache(self) -> None:
        self._invalidate()

    def _invalidate(self) -> None:
        self._frame: int | None = None
        self._q: np.ndarray | None = None
        self._id_to_row: dict[int, int] = {}
        self._dispatch: list[int] = []

    @torch.inference_mode()
    def _ensure_frame(self) -> None:
        game = self._game
        if game is None:
            return
        frame = int(getattr(game, "frame", 0))
        if frame == self._frame and self._q is not None:
            return
        from src.game.ai_snake import AISnake

        snakes = list(game.snakes)
        view = game_state_to_ego_view(game)
        driven = [
            slot
            for slot, snake in enumerate(snakes)
            if isinstance(snake, AISnake)
            and bool(getattr(snake, "is_alive", False))
            and getattr(snake, "policy", None) is self
        ]
        q = np.zeros((len(snakes), self.output_size), dtype=np.float32)
        if driven:
            rows = np.asarray([[0, slot] for slot in driven], dtype=np.int64)
            v8_source = None
            if self.ego_cfg.version == "bv8":
                from src.simd_env.ego_live_bv8 import game_state_to_v8_source

                v8_source = game_state_to_v8_source(game, mask_slots=driven)
            obs = build_ego_raster(
                view, self.ego_cfg, backend="numba", rows=rows, v8_source=v8_source
            )
            sight = full_sight_from_ego_view(view, rows).planes
            local, glob, scalars = obs_tensors(obs, self.device)
            full = torch.from_numpy(np.ascontiguousarray(sight)).to(self.device)
            if self.forward == "rowwise":
                q_rows = torch.cat(
                    [
                        self.net(
                            local[i : i + 1], glob[i : i + 1], scalars[i : i + 1], full[i : i + 1]
                        )
                        for i in range(len(rows))
                    ]
                )
            else:
                q_rows = self.net(local, glob, scalars, full)
            q[rows[:, 1]] = q_rows.cpu().numpy()
        self._frame = frame
        self._q = q
        self._id_to_row = {int(snake.id): row for row, snake in enumerate(snakes)}
        self._dispatch = [int(snakes[slot].id) for slot in driven]

    def _next_dispatch_q(self, state: torch.Tensor) -> torch.Tensor:
        self._ensure_frame()
        if self._q is None:
            batch = 1 if state is None or state.dim() == 1 else int(state.shape[0])
            return torch.zeros((batch, self.output_size), device=self.device)
        if self._dispatch:
            row = self._id_to_row.get(self._dispatch.pop(0), 0)
        else:
            _LOG.warning("exit-fit tower: dqn called with an empty dispatch queue; row 0 served")
            row = 0
        return torch.as_tensor(self._q[row], dtype=torch.float32, device=self.device).unsqueeze(0)

    def select_action(self, state: object, snake_id: int | None = None, action_mask=None) -> int:
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
            mask = np.asarray(action_mask, dtype=bool).reshape(-1)
            if mask.size == len(q) and mask.any():
                q = np.where(mask, q, -np.inf)
        return int(np.argmax(q))

    def q_for(self, snake_id: int) -> np.ndarray | None:
        self._ensure_frame()
        if self._q is None or int(snake_id) not in self._id_to_row:
            return None
        return self._q[self._id_to_row[int(snake_id)]].copy()

    def hero_observation(self, snake_id: int):
        return None

    def hero_activations(self, snake_id: int):
        return None
