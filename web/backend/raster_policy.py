"""Serve an ``obs_spec='raster31v2'`` checkpoint behind the live web game.

The web backend drives a single live :class:`~src.game.game_state.GameState`
whose AI snakes share one policy object. The legacy path shares an
:class:`~src.training.apex_policy.ApexPolicy` (a 61-D vector MLP): each snake
builds a 61-D vector state and the game reads Q-values via ``policy.dqn(state)``.

A raster checkpoint's network (``RasterDuelingNetwork``) cannot consume that
vector — it needs the dual-scale ego raster built by the SAME featurizer the
trainer uses. :class:`RasterServingPolicy` bridges the two worlds *without*
touching the vector serving path or the game/snake code:

- It duck-types the exact ``ApexPolicy`` surface the game and
  :mod:`web.backend.serialize` touch (``training`` / ``epsilon`` / ``device`` /
  ``dqn`` / ``select_action``), so ``AISnake`` and the central game loop run
  unchanged.
- Once per frame it converts the live ``GameState`` into an
  :class:`~src.simd_env.featurizer.ObsInputs` (E = 1) via
  :func:`~src.simd_env.live_adapter.game_state_to_obs_inputs`, builds the raster
  observation for every snake with
  :func:`~src.simd_env.featurizer.build_observations`, and runs the raster net
  once for the whole ``(1, S)`` grid.
- ``dqn(vector_state)`` returns *the calling snake's* raster Q-row. Identity is
  resolved by the deterministic per-frame call order of ``GameState.update``
  step 6 (alive snakes, in ``game.snakes`` order, each calling ``dqn`` exactly
  once), validated so a desync degrades gracefully rather than crashing.

Because the serving observation IS the training observation (one featurizer,
:mod:`src.simd_env.live_adapter`), the served snake behaves like the trained
one by construction. Training is intentionally unsupported here — this is a
read-only serving surface (``training`` is always ``False``), so the game's
central ``train_step`` / replay-memory paths are skipped.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model.inference_agent import InferenceAgent
from src.model.raster_network import raster_tensors_from_obs
from src.simd_env.featurizer import build_observations
from src.simd_env.live_adapter import game_state_to_obs_inputs

__all__ = ["RasterServingPolicy"]


class _RasterDQNShim:
    """A ``policy.dqn``-compatible callable backed by the raster network.

    ``AISnake`` and :mod:`web.backend.serialize` call ``policy.dqn(state)`` with
    a batched vector state and expect an ``(N, output_size)`` Q tensor. This shim
    ignores the vector payload and instead returns the raster Q-row for the snake
    the game is currently processing, resolved from the owning policy's per-frame
    dispatch queue.
    """

    def __init__(self, owner: "RasterServingPolicy") -> None:
        self._owner = owner

    def __call__(self, state: torch.Tensor) -> torch.Tensor:
        return self._owner._next_dispatch_q(state)

    def eval(self) -> "_RasterDQNShim":  # parity with nn.Module.eval()
        return self

    def parameters(self):  # pragma: no cover - trivial passthrough
        return self._owner.agent.network.parameters()


class RasterServingPolicy:
    """Drive a live game from a raster checkpoint via the shared featurizer.

    Presents the minimal ``ApexPolicy`` surface the web game and serializer use.
    It is forward-only: :attr:`training` is always ``False``.

    Attributes:
        agent: The :class:`InferenceAgent` wrapping the raster network.
        obs_spec: The served checkpoint's obs_spec (``"raster31v2"``).
        dqn: The :class:`_RasterDQNShim` the game reads Q-values through.
        device: The inference device.
        output_size: Number of discrete actions.
    """

    def __init__(self, agent: InferenceAgent) -> None:
        """Wrap an already-built raster :class:`InferenceAgent`.

        Args:
            agent: A raster (``obs_spec='raster31v2'``) inference agent.
        """
        self.agent = agent
        self.obs_spec = agent.obs_spec
        self.device = agent.device
        self.output_size = int(agent.output_size)
        self.training = False
        self.total_reward = 0.0
        self.memory = None  # AISnake skips replay when training is False.
        self.dqn = _RasterDQNShim(self)

        # Live game reference, set by the session after the game is built.
        self._game = None
        # Per-frame observation cache: (frame, obs dict, dispatch queue of snake ids).
        self._cache_frame: Optional[int] = None
        self._cache_obs: Optional[dict] = None
        self._cache_id_to_row: Dict[int, int] = {}
        self._cache_q: Optional[np.ndarray] = None  # (S, output_size)
        self._dispatch_queue: List[int] = []

    # -- epsilon (forced to 0) ---------------------------------------------
    @property
    def epsilon(self) -> float:
        """Always ``0.0``: raster serving is greedy and forward-only.

        The per-frame dispatch queue in :meth:`_next_dispatch_q` assumes every
        alive snake calls ``dqn`` exactly once, in ``game.snakes`` order. But
        ``AISnake.update`` *skips* the ``dqn`` call for any snake that explores
        (``random.random() < effective_epsilon`` with ``record_q_values`` off,
        which is the case for headless serving). A single skipped call would
        shift the queue and hand every subsequent greedy snake a foreign
        snake's raster Q-row. Because identity is resolved positionally rather
        than by the caller's id, we forbid exploration entirely by clamping
        epsilon to ``0.0`` — writes (from session init or ``set_epsilon``) are
        accepted but ignored, so no snake can ever skip its ``dqn`` call.
        """
        return 0.0

    @epsilon.setter
    def epsilon(self, value) -> None:  # noqa: D401 - intentional no-op
        # Ignore all writes; see the property docstring for why serving must
        # stay strictly greedy. Kept as a settable attribute for surface parity
        # with ApexPolicy (session.py and set_epsilon assign to it).
        return

    # -- construction -------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None
    ) -> "RasterServingPolicy":
        """Build from a raster checkpoint path.

        Args:
            checkpoint_path: Path to an ``obs_spec='raster31v2'`` checkpoint.
            device: Optional inference device.

        Returns:
            A ready :class:`RasterServingPolicy`.
        """
        agent = InferenceAgent.from_checkpoint(checkpoint_path, device=device)
        return cls(agent)

    def attach_game(self, game) -> None:
        """Attach the live game whose snakes this policy serves.

        Args:
            game: The :class:`~src.game.game_state.GameState` being served.
        """
        self._game = game
        self._invalidate_cache()

    # -- per-frame observation build ---------------------------------------
    def _invalidate_cache(self) -> None:
        self._cache_frame = None
        self._cache_obs = None
        self._cache_id_to_row = {}
        self._cache_q = None
        self._dispatch_queue = []

    def _ensure_frame(self) -> None:
        """Build (once per game frame) the all-snake raster obs and Q matrix.

        Rebuilds when the game's frame counter advances. Also (re)seeds the
        per-frame dispatch queue with the alive snakes in ``game.snakes`` order,
        matching ``GameState.update`` step 6's iteration so successive
        ``dqn`` calls map to the right snakes.
        """
        game = self._game
        if game is None:
            return
        frame = int(getattr(game, "frame", 0))
        if self._cache_frame == frame and self._cache_obs is not None:
            return

        snakes = list(game.snakes)
        inp = game_state_to_obs_inputs(game)
        obs = build_observations(inp)  # (1, S, ...) + mask
        # (1, S, ...) uint8 -> (S, ...) float tensors the raster net consumes.
        tensors = raster_tensors_from_obs(obs, device=self.device)
        with torch.no_grad():
            q = self.agent.network(tensors).cpu().numpy()  # (S, output_size)

        self._cache_frame = frame
        self._cache_obs = obs
        self._cache_q = q
        self._cache_id_to_row = {int(s.id): row for row, s in enumerate(snakes)}
        # Dispatch queue: alive snakes, in the exact order game.update iterates.
        self._dispatch_queue = [int(s.id) for s in snakes if bool(getattr(s, "is_alive", False))]

    def _row_for_snake_id(self, snake_id: int) -> int:
        """Return the observation row index for a snake id (0 if unknown)."""
        return int(self._cache_id_to_row.get(int(snake_id), 0))

    def _next_dispatch_q(self, state: torch.Tensor) -> torch.Tensor:
        """Return the next queued snake's raster Q-row as an ``(N, out)`` tensor.

        Called by ``AISnake`` (via ``policy.dqn``) once per alive snake per frame,
        in game-loop order. Pops the dispatch queue seeded by :meth:`_ensure_frame`.
        If the queue is exhausted (unexpected extra call), falls back to row 0 so
        serving degrades gracefully instead of raising mid-frame.

        Args:
            state: The batched vector state the caller passed (payload ignored;
                only its batch size shapes the returned tensor).

        Returns:
            ``(N, output_size)`` Q tensor on this policy's device.
        """
        self._ensure_frame()
        if self._cache_q is None:
            n = 1 if state is None or state.dim() == 1 else int(state.shape[0])
            return torch.zeros((n, self.output_size), device=self.device)
        if self._dispatch_queue:
            snake_id = self._dispatch_queue.pop(0)
            row = self._row_for_snake_id(snake_id)
        else:
            row = 0
        q_row = torch.as_tensor(self._cache_q[row], dtype=torch.float32, device=self.device)
        return q_row.unsqueeze(0)

    # -- inspector / serialize helpers -------------------------------------
    def hero_observation(self, snake_id: int) -> Optional[Dict[str, np.ndarray]]:
        """Return the raster obs (single-agent uint8 planes + scalars) for a snake.

        Slices the cached ``(1, S, ...)`` observation to the snake's row. Used by
        :mod:`web.backend.serialize` both to feed the inspector activations and to
        expose the ego-raster planes to the frontend.

        Args:
            snake_id: The snake whose observation to return.

        Returns:
            Dict with ``tactical_uint8`` ``(2, 31, 31)``, ``strategic_uint8``
            ``(3, 25, 25)``, ``scalars`` ``(26,)`` and ``mask`` ``(6,)``; or
            ``None`` when no live game / observation is available.
        """
        self._ensure_frame()
        if self._cache_obs is None:
            return None
        row = self._row_for_snake_id(snake_id)
        obs = self._cache_obs
        return {
            "tactical_uint8": np.asarray(obs["tactical_uint8"])[0, row],
            "strategic_uint8": np.asarray(obs["strategic_uint8"])[0, row],
            "scalars": np.asarray(obs["scalars"])[0, row],
            "mask": np.asarray(obs["mask"])[0, row],
        }

    def hero_activations(self, snake_id: int) -> Optional[Tuple[np.ndarray, Dict[str, np.ndarray]]]:
        """Return ``(q_values, activations)`` for a snake's current raster obs.

        Mirrors :meth:`InferenceAgent.activations`; the activations dict carries
        the same keys (``input``/``hidden``/``output``/``value``/``advantages``)
        the vector path produces, so the serializer's inspector/net-viz panels
        render both specs identically.

        Args:
            snake_id: The snake to inspect.

        Returns:
            ``(q_values, activations_dict)`` or ``None`` when unavailable.
        """
        obs = self.hero_observation(snake_id)
        if obs is None:
            return None
        return self.agent.activations(obs)

    # -- ApexPolicy surface the game uses ----------------------------------
    def select_action(
        self,
        state: torch.Tensor,
        snake_id: Optional[int] = None,
        action_mask=None,
    ) -> int:
        """Greedy raster action for the snake identified by the dispatch order.

        ``AISnake`` only reaches this fallback when it could not read Q-values via
        ``dqn``; it is provided for surface completeness. The returned action is
        the argmax of the same raster Q-row ``dqn`` would have produced.

        Args:
            state: The vector state the caller passed (payload ignored).
            snake_id: Optional snake id for direct row lookup.
            action_mask: Optional length-``output_size`` mask.

        Returns:
            The chosen action index.
        """
        self._ensure_frame()
        if self._cache_q is None:
            return 0
        if snake_id is not None:
            row = self._row_for_snake_id(snake_id)
        elif self._dispatch_queue:
            row = self._row_for_snake_id(self._dispatch_queue.pop(0))
        else:
            row = 0
        q = torch.as_tensor(self._cache_q[row], dtype=torch.float32, device=self.device)
        if action_mask is not None:
            mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=q.device).view(
                -1
            )
            if mask.numel() == q.numel() and bool(mask.any()):
                q = q.masked_fill(~mask, float("-inf"))
        return int(torch.argmax(q).item())
