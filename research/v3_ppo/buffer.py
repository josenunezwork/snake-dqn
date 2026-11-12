"""Compact ``uint8`` multi-scale rollout buffer and GAE for Game V3.1 PPO."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from research.game_v3.obs import LearnerIndex, ObservationBatch
from research.v3_ppo.config import DEFAULT_OBS_LAYOUT, ObsLayout


@dataclass
class RolloutMinibatch:
    """Gathered learner minibatch slice from ``Uint8RolloutBuffer`` (`uint8` rasters preserved)."""

    raster_s1: torch.Tensor  # (B, C, fine_size, fine_size) uint8
    raster_s4: torch.Tensor  # (B, C, mid_size, mid_size) uint8
    raster_s16: torch.Tensor  # (B, C, wide_size, wide_size) uint8
    scalars: torch.Tensor  # (B, scalar_dim) float32
    action_mask: torch.Tensor  # (B, num_actions) bool
    actions: torch.Tensor  # (B,) int64
    old_log_probs: torch.Tensor  # (B,) float32
    old_values: torch.Tensor  # (B,) float32
    advantages: torch.Tensor  # (B,) float32
    returns: torch.Tensor  # (B,) float32


def assert_obs_batch_matches_layout(
    obs: ObservationBatch,
    layout: ObsLayout,
    *,
    batch_rows: int,
    context: str = "ObservationBatch",
) -> None:
    """Assert that ``obs`` tensor shapes and dtypes match ``layout`` and ``batch_rows``."""
    expected_rasters = (
        (
            "raster_s1",
            obs.raster_s1,
            (batch_rows, layout.channels, layout.fine_size, layout.fine_size),
        ),
        (
            "raster_s4",
            obs.raster_s4,
            (batch_rows, layout.channels, layout.mid_size, layout.mid_size),
        ),
        (
            "raster_s16",
            obs.raster_s16,
            (batch_rows, layout.channels, layout.wide_size, layout.wide_size),
        ),
    )
    for field_name, tensor, exp_shape in expected_rasters:
        if tuple(tensor.shape) != exp_shape or tensor.dtype != torch.uint8:
            raise ValueError(
                f"{context}.{field_name} shape/dtype mismatch: expected uint8 {exp_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
    exp_scalars = (batch_rows, layout.scalar_dim)
    if tuple(obs.scalars.shape) != exp_scalars or obs.scalars.dtype != torch.float32:
        raise ValueError(
            f"{context}.scalars shape/dtype mismatch: expected float32 {exp_scalars}, "
            f"got {obs.scalars.dtype} {tuple(obs.scalars.shape)}"
        )
    exp_mask = (batch_rows, layout.num_actions)
    if tuple(obs.action_mask.shape) != exp_mask or obs.action_mask.dtype != torch.bool:
        raise ValueError(
            f"{context}.action_mask shape/dtype mismatch: expected bool {exp_mask}, "
            f"got {obs.action_mask.dtype} {tuple(obs.action_mask.shape)}"
        )


def compute_gae(
    *,
    rewards: torch.Tensor,
    values: torch.Tensor,
    died: torch.Tensor,
    truncated: torch.Tensor,
    acted: torch.Tensor,
    last_values: torch.Tensor,
    gamma: float,
    gae_lambda: float,
    trunc_bootstrap: torch.Tensor | None = None,
    advantages_out: torch.Tensor | None = None,
    returns_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Generalized Advantage Estimation with exact death vs truncation semantics.

    Transition rules per learner slot ``m`` at step ``t``:
      - ``~acted[t, m]`` (dead / in respawn delay at decision time):
        ``advantages[t, m] = 0``, ``returns[t, m] = 0``, and running GAE trace is reset to ``0``.
      - ``acted[t, m] & died[t, m]`` (snake died this tick):
        zero bootstrap value (`next_value = 0`), GAE trace from ``t + 1`` is cut (`0`).
      - ``acted[t, m] & (~died[t, m]) & truncated[t, m]`` (rollout-end or life truncation):
        bootstraps ``next_value`` (`last_values[m]` at ``t == T - 1``, `values[t + 1, m]` at
        ``t < T - 1`` when ``trunc_bootstrap is None``, or ``trunc_bootstrap[t, m]`` when given),
        and cuts GAE trace from ``t + 1`` so new episodes never bleed into truncated ones.
      - ``acted[t, m] & (~died[t, m]) & (~truncated[t, m])`` (continuing life step):
        ``delta = r_t + gamma * V(s_{t+1}) - V(s_t)`` and ``gae = delta + gamma * lambda * gae``.
    """
    if rewards.ndim != 2:
        raise ValueError(f"rewards must have shape (T, M), got {tuple(rewards.shape)}")
    t_steps, m_learners = rewards.shape
    for name, ten in (
        ("values", values),
        ("died", died),
        ("truncated", truncated),
        ("acted", acted),
    ):
        if ten.shape != (t_steps, m_learners):
            raise ValueError(f"{name} shape {tuple(ten.shape)} != expected {(t_steps, m_learners)}")
    if last_values.shape != (m_learners,):
        raise ValueError(
            f"last_values shape {tuple(last_values.shape)} != expected {(m_learners,)}"
        )
    if trunc_bootstrap is not None and trunc_bootstrap.shape != (t_steps, m_learners):
        raise ValueError(
            f"trunc_bootstrap shape {tuple(trunc_bootstrap.shape)} != {(t_steps, m_learners)}"
        )

    dev = rewards.device
    adv = (
        advantages_out
        if advantages_out is not None
        else torch.empty((t_steps, m_learners), dtype=torch.float32, device=dev)
    )
    ret = (
        returns_out
        if returns_out is not None
        else torch.empty((t_steps, m_learners), dtype=torch.float32, device=dev)
    )

    gae = torch.zeros((m_learners,), dtype=torch.float32, device=dev)
    zeros = torch.zeros((m_learners,), dtype=torch.float32, device=dev)
    g = float(gamma)
    gl = float(gamma) * float(gae_lambda)

    for t in range(t_steps - 1, -1, -1):
        acted_t = acted[t]
        died_t = died[t]
        trunc_t = truncated[t]
        step_next_v = last_values if t == t_steps - 1 else values[t + 1]
        boot_v = (
            trunc_bootstrap[t]
            if trunc_bootstrap is not None
            else (last_values if t == t_steps - 1 else values[t + 1])
        )
        next_v = torch.where(
            died_t,
            zeros,
            torch.where(trunc_t, boot_v, step_next_v),
        )
        cont = acted_t & (~died_t) & (~trunc_t)
        delta = rewards[t] + g * next_v - values[t]
        gae = torch.where(
            acted_t,
            delta + gl * torch.where(cont, gae, zeros),
            zeros,
        )
        adv[t] = gae
        ret[t] = torch.where(acted_t, gae + values[t], zeros)

    return adv, ret


class Uint8RolloutBuffer:
    """Preallocated ``(T, M, ...)`` rollout buffer storing ``uint8`` rasters directly.

    Exposes zero-copy per-tick :class:`ObservationBatch` views bound to ``learner_index`` so
    ``build_obs(sim, learner_index=idx, out=buffer.obs_view(t))`` writes straight into the
    buffer with zero extra allocations, copies, or host synchronizations.
    """

    def __init__(
        self,
        *,
        rollout_len: int,
        learner_index: LearnerIndex,
        sim_shape: tuple[int, int],
        device: torch.device,
        obs_layout: ObsLayout = DEFAULT_OBS_LAYOUT,
    ) -> None:
        if int(rollout_len) < 1:
            raise ValueError(f"rollout_len must be >= 1, got {rollout_len}")
        if int(learner_index.count) < 1:
            raise ValueError("learner_index must select at least 1 learner slot")
        self.T = int(rollout_len)
        self.M = int(learner_index.count)
        self.learner_index = learner_index
        self.sim_shape = (int(sim_shape[0]), int(sim_shape[1]))
        self.device = device
        self.obs_layout = obs_layout

        c = self.obs_layout.channels
        s1_hw = self.obs_layout.fine_size
        s4_hw = self.obs_layout.mid_size
        s16_hw = self.obs_layout.wide_size

        self.raster_s1 = torch.empty(
            (self.T, self.M, c, s1_hw, s1_hw),
            dtype=torch.uint8,
            device=device,
        )
        self.raster_s4 = torch.empty(
            (self.T, self.M, c, s4_hw, s4_hw),
            dtype=torch.uint8,
            device=device,
        )
        self.raster_s16 = torch.empty(
            (self.T, self.M, c, s16_hw, s16_hw),
            dtype=torch.uint8,
            device=device,
        )
        self.scalars = torch.empty(
            (self.T, self.M, self.obs_layout.scalar_dim),
            dtype=torch.float32,
            device=device,
        )
        self.action_mask = torch.empty(
            (self.T, self.M, self.obs_layout.num_actions),
            dtype=torch.bool,
            device=device,
        )

        self.actions = torch.zeros((self.T, self.M), dtype=torch.uint8, device=device)
        self.log_probs = torch.zeros((self.T, self.M), dtype=torch.float32, device=device)
        self.values = torch.zeros((self.T, self.M), dtype=torch.float32, device=device)
        self.rewards = torch.zeros((self.T, self.M), dtype=torch.float32, device=device)
        self.advantages = torch.zeros((self.T, self.M), dtype=torch.float32, device=device)
        self.returns = torch.zeros((self.T, self.M), dtype=torch.float32, device=device)
        self.died = torch.zeros((self.T, self.M), dtype=torch.bool, device=device)
        self.truncated = torch.zeros((self.T, self.M), dtype=torch.bool, device=device)
        self.acted = torch.zeros((self.T, self.M), dtype=torch.bool, device=device)

        self._obs_views: tuple[ObservationBatch, ...] = tuple(
            ObservationBatch(
                raster_s1=self.raster_s1[t],
                raster_s4=self.raster_s4[t],
                raster_s16=self.raster_s16[t],
                scalars=self.scalars[t],
                action_mask=self.action_mask[t],
                _learner_index=learner_index,
                _sim_shape=self.sim_shape,
            )
            for t in range(self.T)
        )

    def obs_view(self, t: int) -> ObservationBatch:
        """Return prebound ``ObservationBatch`` slice view into rollout step ``t``."""
        if not (0 <= t < self.T):
            raise IndexError(f"step index {t} out of range [0, {self.T})")
        return self._obs_views[t]

    def copy_obs_into_step(self, t: int, src: ObservationBatch) -> ObservationBatch:
        """Copy an already-built ``ObservationBatch`` (e.g. bootstrap obs) into step ``t``."""
        dst = self.obs_view(t)
        dst.raster_s1.copy_(src.raster_s1)
        dst.raster_s4.copy_(src.raster_s4)
        dst.raster_s16.copy_(src.raster_s16)
        dst.scalars.copy_(src.scalars)
        dst.action_mask.copy_(src.action_mask)
        return dst

    def raster_nbytes(self) -> int:
        """Return total bytes allocated for the 3 ``uint8`` raster tensors."""
        total = int(self.raster_s1.nbytes + self.raster_s4.nbytes + self.raster_s16.nbytes)
        assert total == self.T * self.M * self.obs_layout.raster_bytes_per_step
        return total

    def nbytes_total(self) -> int:
        """Return exact total bytes allocated across all buffer tensors."""
        tensors = (
            self.raster_s1,
            self.raster_s4,
            self.raster_s16,
            self.scalars,
            self.action_mask,
            self.actions,
            self.log_probs,
            self.values,
            self.rewards,
            self.advantages,
            self.returns,
            self.died,
            self.truncated,
            self.acted,
        )
        total = sum(int(t.nbytes) for t in tensors)
        assert total == self.T * self.M * self.obs_layout.bytes_per_learner_step
        return total

    def compute_gae(
        self,
        *,
        last_values: torch.Tensor,
        gamma: float,
        gae_lambda: float,
        trunc_bootstrap: torch.Tensor | None = None,
    ) -> None:
        """Compute ``self.advantages`` and ``self.returns`` in-place."""
        compute_gae(
            rewards=self.rewards,
            values=self.values,
            died=self.died,
            truncated=self.truncated,
            acted=self.acted,
            last_values=last_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
            trunc_bootstrap=trunc_bootstrap,
            advantages_out=self.advantages,
            returns_out=self.returns,
        )

    def valid_flat_indices(self) -> torch.Tensor:
        """Return 1D ``int64`` indices in ``[0, T * M)`` where ``acted == True``."""
        return torch.nonzero(self.acted.reshape(-1), as_tuple=False).squeeze(-1)

    def gather_minibatch(
        self,
        flat_indices: torch.Tensor,
        *,
        normalized_advantages_flat: torch.Tensor | None = None,
        include_old_values: bool = True,
    ) -> RolloutMinibatch:
        """Gather a learner minibatch by flat ``(T * M)`` row indices (`uint8` preserved)."""
        tm = self.T * self.M
        adv_source = (
            normalized_advantages_flat
            if normalized_advantages_flat is not None
            else self.advantages.reshape(tm)
        )
        old_vals = (
            self.values.reshape(tm).index_select(0, flat_indices)
            if include_old_values
            else self.values.new_empty((0,))
        )
        c = self.obs_layout.channels
        s1_hw = self.obs_layout.fine_size
        s4_hw = self.obs_layout.mid_size
        s16_hw = self.obs_layout.wide_size
        return RolloutMinibatch(
            raster_s1=self.raster_s1.reshape(tm, c, s1_hw, s1_hw).index_select(0, flat_indices),
            raster_s4=self.raster_s4.reshape(tm, c, s4_hw, s4_hw).index_select(0, flat_indices),
            raster_s16=self.raster_s16.reshape(tm, c, s16_hw, s16_hw).index_select(0, flat_indices),
            scalars=self.scalars.reshape(tm, self.obs_layout.scalar_dim).index_select(
                0, flat_indices
            ),
            action_mask=self.action_mask.reshape(tm, self.obs_layout.num_actions).index_select(
                0, flat_indices
            ),
            actions=self.actions.reshape(tm).index_select(0, flat_indices).to(torch.long),
            old_log_probs=self.log_probs.reshape(tm).index_select(0, flat_indices),
            old_values=old_vals,
            advantages=adv_source.index_select(0, flat_indices),
            returns=self.returns.reshape(tm).index_select(0, flat_indices),
        )
