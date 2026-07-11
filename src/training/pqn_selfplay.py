"""Pool self-play for PQN (blueprint P3 §3).

Multi-agent PQN trains one *hero* policy against a rotating pool of *frozen*
past copies of itself. Each arena slot ``(env, snake)`` is assigned a
``policy_id``: with probability 80% the LATEST (hero) policy, otherwise a
uniformly-drawn frozen pool member. Only hero-slot transitions are used to
update the learner; frozen slots exist purely to provide non-stationary
opponents.

Acting is done with a **K+1 batched forward**: the ``S`` slots per env are
grouped by their assigned ``policy_id`` and each *distinct* policy runs exactly
one forward over its own slot subset (hero + up to ``pool_size`` frozen nets, so
at most ``K + 1`` forwards for a ``K``-net pool). Actions scatter back into the
``(E, S)`` grid. This keeps acting cost independent of the number of slots.

The :class:`OpponentPool` is a FIFO ring of eval-mode
:class:`~src.model.raster_network.RasterDuelingNetwork` snapshots. Snapshots are
deep state-dict copies (device-resident) so a later hero update never mutates a
frozen opponent.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from src.model.raster_network import RasterDuelingNetwork

__all__ = ["OpponentPool", "assign_policy_ids", "batched_act", "HERO_POLICY_ID"]

# Reserved policy id for the (trainable) hero. Frozen pool members use ids 0..K-1
# as returned by :meth:`OpponentPool.policy_ids`.
HERO_POLICY_ID = -1


class OpponentPool:
    """FIFO pool of frozen hero snapshots for self-play.

    Holds up to ``capacity`` eval-mode ``RasterDuelingNetwork`` copies. Adding a
    snapshot past capacity evicts the oldest (FIFO). Each resident snapshot has a
    stable integer id (its slot index ``0..capacity-1``); ids are reused when a
    slot is overwritten, which is fine because ids are only meaningful for the
    current acting step.

    Args:
        capacity: Maximum number of frozen nets kept resident (blueprint: <=10).
        device: Device the frozen nets live on.
    """

    def __init__(self, capacity: int = 10, device: Optional[torch.device] = None) -> None:
        if capacity < 0:
            raise ValueError("capacity must be >= 0")
        self.capacity = capacity
        self.device = device or torch.device("cpu")
        self._nets: List[RasterDuelingNetwork] = []
        self._next_slot = 0

    def __len__(self) -> int:
        """Number of frozen nets currently resident."""
        return len(self._nets)

    def policy_ids(self) -> List[int]:
        """Ids of the resident frozen nets (``0..len-1``)."""
        return list(range(len(self._nets)))

    def add_snapshot(self, hero: RasterDuelingNetwork) -> None:
        """Freeze a deep copy of ``hero`` into the pool (FIFO eviction).

        The snapshot is a fresh ``RasterDuelingNetwork`` loaded with a detached
        CPU->device clone of ``hero``'s state dict, set to eval mode with
        ``requires_grad=False`` so it never trains and never shares storage with
        the live hero.

        Args:
            hero: The current trainable hero network to snapshot.
        """
        if self.capacity == 0:
            return
        snap = RasterDuelingNetwork(output_size=hero.output_size)
        state = {k: v.detach().clone() for k, v in hero.state_dict().items()}
        snap.load_state_dict(state)
        snap.to(self.device).eval()
        for p in snap.parameters():
            p.requires_grad_(False)

        if len(self._nets) < self.capacity:
            self._nets.append(snap)
        else:
            self._nets[self._next_slot] = snap
            self._next_slot = (self._next_slot + 1) % self.capacity

    def get(self, policy_id: int) -> RasterDuelingNetwork:
        """Return the frozen net for ``policy_id`` (``0..len-1``)."""
        return self._nets[policy_id]


def assign_policy_ids(
    E: int,
    S: int,
    pool_ids: List[int],
    hero_frac: float,
    rng: np.random.Generator,
    hero_slot0: bool = True,
) -> np.ndarray:
    """Assign a ``policy_id`` to every ``(env, snake)`` slot (80/20 default).

    Each slot is independently the hero with probability ``hero_frac``, else a
    uniformly-chosen frozen pool id. When the pool is empty every slot is the
    hero. When ``hero_slot0`` is set, slot 0 of every env is forced to the hero
    so there is always at least one trainable transition per env (mirrors the
    eval engine's "hero is arena slot 0" convention).

    Args:
        E: Number of environments.
        S: Snakes per environment.
        pool_ids: Ids of resident frozen nets (from
            :meth:`OpponentPool.policy_ids`).
        hero_frac: Probability a slot is the hero (blueprint: 0.8).
        rng: NumPy random generator.
        hero_slot0: Force slot 0 of every env to the hero.

    Returns:
        ``(E, S)`` int array of policy ids (``HERO_POLICY_ID`` for hero slots).
    """
    ids = np.full((E, S), HERO_POLICY_ID, dtype=np.int64)
    if pool_ids:
        use_hero = rng.random((E, S)) < hero_frac
        frozen_choice = rng.integers(0, len(pool_ids), size=(E, S))
        pool_arr = np.asarray(pool_ids, dtype=np.int64)
        frozen_ids = pool_arr[frozen_choice]
        ids = np.where(use_hero, HERO_POLICY_ID, frozen_ids)
    if hero_slot0:
        ids[:, 0] = HERO_POLICY_ID
    return ids


def _greedy_masked_actions(q_values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Argmax over VALID actions per row (``q`` masked by the 6-bit ``mask``).

    Invalid actions get ``-inf`` before the argmax. Rows with no valid action
    fall back to the unmasked argmax (the sim resolves the death).

    Args:
        q_values: ``(M, 6)`` Q-values.
        mask: ``(M, 6)`` bool valid-action mask.

    Returns:
        ``(M,)`` long action indices.
    """
    neg_inf = torch.finfo(q_values.dtype).min
    any_valid = mask.any(dim=1)
    masked = torch.where(mask, q_values, torch.full_like(q_values, neg_inf))
    actions = masked.argmax(dim=1)
    if not bool(any_valid.all()):
        fallback = q_values.argmax(dim=1)
        actions = torch.where(any_valid, actions, fallback)
    return actions


def batched_act(
    hero: RasterDuelingNetwork,
    pool: OpponentPool,
    policy_ids: np.ndarray,
    obs: Dict[str, torch.Tensor],
    mask: torch.Tensor,
    epsilon: float,
    rng: np.random.Generator,
    device: torch.device,
) -> Tuple[np.ndarray, torch.Tensor]:
    """K+1 batched greedy-masked acting over all slots grouped by policy.

    Groups the ``E*S`` flattened slots by ``policy_id`` and runs ONE forward per
    distinct policy (hero + each resident frozen id present in ``policy_ids``),
    then scatters greedy-masked actions back to the ``(E, S)`` grid. ε-greedy
    exploration is applied to HERO slots only (frozen opponents act greedily),
    with the random action drawn from that slot's valid mask.

    Args:
        hero: Trainable hero network (used for ``HERO_POLICY_ID`` slots).
        pool: Frozen opponent pool.
        policy_ids: ``(E, S)`` policy id per slot (from
            :func:`assign_policy_ids`).
        obs: Network-input dict with ``tactical`` ``(E, S, 9, 31, 31)``,
            ``strategic`` ``(E, S, 3, 25, 25)`` and ``scalars`` ``(E, S, 26)``
            tensors on ``device``.
        mask: ``(E, S, 6)`` bool valid-action mask on ``device``.
        epsilon: Hero exploration rate.
        rng: NumPy random generator (for ε and random-action draws).
        device: Compute device.

    Returns:
        ``(actions, hero_q)``:
            actions: ``(E, S)`` int64 chosen action per slot.
            hero_q: ``(E, S, 6)`` Q-values from the HERO net for EVERY slot
                (used by the learner; frozen-slot rows are ignored downstream but
                computed once for the whole grid so the hero's targets are
                available without a second forward).
    """
    E, S = policy_ids.shape
    flat_ids = policy_ids.reshape(-1)
    tac = obs["tactical"].reshape(E * S, *hero.tactical_shape)
    strat = obs["strategic"].reshape(E * S, *hero.strategic_shape)
    scal = obs["scalars"].reshape(E * S, hero.scalars_dim)
    mask_flat = mask.reshape(E * S, 6)

    actions = np.zeros(E * S, dtype=np.int64)

    # Hero forward over the WHOLE grid: needed both to act hero slots and to
    # expose hero Q-values for every slot to the learner in one pass.
    with torch.no_grad():
        hero_q_flat = hero(tac, strat, scal)  # (E*S, 6)

    # --- Hero slots: ε-greedy masked ---
    hero_sel = flat_ids == HERO_POLICY_ID
    if np.any(hero_sel):
        idx = np.nonzero(hero_sel)[0]
        idx_t = torch.as_tensor(idx, device=device)
        greedy = _greedy_masked_actions(hero_q_flat[idx_t], mask_flat[idx_t])
        greedy_np = greedy.cpu().numpy()
        explore = rng.random(idx.shape[0]) < epsilon
        if np.any(explore):
            valid_np = mask_flat[idx_t].cpu().numpy()
            rand_actions = _sample_valid(valid_np, rng)
            greedy_np = np.where(explore, rand_actions, greedy_np)
        actions[idx] = greedy_np

    # --- Frozen slots: greedy masked, one forward per resident policy id ---
    for pid in np.unique(flat_ids):
        if pid == HERO_POLICY_ID:
            continue
        sel = flat_ids == pid
        idx = np.nonzero(sel)[0]
        idx_t = torch.as_tensor(idx, device=device)
        net = pool.get(int(pid))
        with torch.no_grad():
            q = net(tac[idx_t], strat[idx_t], scal[idx_t])
        greedy = _greedy_masked_actions(q, mask_flat[idx_t])
        actions[idx] = greedy.cpu().numpy()

    return actions.reshape(E, S), hero_q_flat.reshape(E, S, 6)


def _sample_valid(valid_np: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample one valid action index per row of a ``(M, 6)`` bool mask.

    Rows with no valid action fall back to a uniform draw over all 6 actions.

    Args:
        valid_np: ``(M, 6)`` bool array.
        rng: NumPy random generator.

    Returns:
        ``(M,)`` int64 sampled action indices.
    """
    m = valid_np.shape[0]
    out = np.zeros(m, dtype=np.int64)
    for i in range(m):
        opts = np.nonzero(valid_np[i])[0]
        if opts.size == 0:
            out[i] = rng.integers(0, valid_np.shape[1])
        else:
            out[i] = opts[rng.integers(0, opts.size)]
    return out
