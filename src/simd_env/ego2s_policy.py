"""SIMD policies for an ``ego2s-draft`` student (redesign M2 evaluation).

* :class:`Ego2sSimdPolicy`: greedy over the engine's resolved action mask (normal-only
  fallback when nothing is safe), observation from the numba ego2s featurizer for exactly
  the rows it controls, one batched forward per frame.
* :class:`Ego2sV8Policy`: the same student wrapped in the policy-agnostic v8 veto. It reuses
  :class:`~src.simd_env.vector61_policy.Vector61SimdPolicy` for the veto plumbing (the
  vector61 runtime supplies the v8 free-space counts and the advisory mask, exactly as for
  the teacher); only the Q values come from the student. The vector61 ``carrier``
  checkpoint is loaded only to satisfy the featurizer-width check, never run.

Both need :class:`~src.simd_env.grid_sim.GridBatchSim` (the featurizer reads its grids).
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch

from src.model.ego2s_network import OBS_SPEC_B, load_ego2s_checkpoint, obs_tensors
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.vector61_policy import Vector61SimdPolicy

__all__ = ["Ego2sSimdPolicy", "Ego2sV8Policy", "student_q"]


def ego_config_for(net) -> EgoRasterConfig:
    """The featurizer config a student was trained on (from its input shape)."""
    from src.model.ego2s_network import spec_for

    return EgoRasterConfig(version="b" if spec_for(net) == OBS_SPEC_B else "draft")


def student_q(net, sim, slots: np.ndarray) -> torch.Tensor:
    """``(N, 6)`` student Q for ``(N, 2)`` rows of ``sim`` (CPU tensor)."""
    slots = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
    if not len(slots):
        return torch.zeros((0, 6))
    obs = build_ego_raster(sim, ego_config_for(net), backend="numba", rows=slots)
    device = next(net.parameters()).device
    with torch.no_grad():
        return net(*obs_tensors(obs, device)).float().cpu()


class Ego2sSimdPolicy:
    """Greedy masked ego2s student (duck-types ``SimdPolicy.actions``)."""

    batched_rows = True  # eval_engine groups every row of a frame into one call

    def __init__(self, checkpoint_path: str, device: str = "cpu") -> None:
        self.checkpoint_path = str(checkpoint_path)
        self.net = load_ego2s_checkpoint(checkpoint_path, device)

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        q = student_q(self.net, sim, slots).numpy()
        m = np.asarray(masks, dtype=bool).copy()
        m[~m.any(axis=1), :3] = True
        return np.where(m, q, -np.inf).argmax(axis=1).astype(np.int64)


class Ego2sV8Policy(Vector61SimdPolicy):
    """The student inside the v8 veto (Q from the student, everything else as the teacher)."""

    def __init__(
        self,
        checkpoint_path: str,
        carrier_vector61_checkpoint: str,
        runtime,
        *,
        veto_slots: Sequence[int] = (0,),
        veto_variant: str = "v8",
        veto_lambda: Optional[float] = 8.0,
        veto_reference_lambda: Optional[float] = None,
        device: str = "cpu",
    ) -> None:
        super().__init__(
            carrier_vector61_checkpoint,
            runtime,
            veto_slots=veto_slots,
            forward="batched",
            veto_variant=veto_variant,
            veto_lambda=veto_lambda,
            veto_reference_lambda=veto_reference_lambda,
        )
        self.network = None  # the carrier network is never run
        self.checkpoint_path = str(checkpoint_path)
        self.student = load_ego2s_checkpoint(checkpoint_path, device)
        self._pending = None

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        self._pending = (sim, np.asarray(slots, dtype=np.int64).reshape(-1, 2))
        try:
            return super().actions(masks, sim, slots)
        finally:
            self._pending = None

    def q_values(self, states):
        if self._pending is None:
            raise RuntimeError("Ego2sV8Policy.q_values outside actions()")
        sim, slots = self._pending
        return student_q(self.student, sim, slots)
