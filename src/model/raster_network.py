"""Dual-scale raster Dueling DQN network (blueprint P3, obs_spec ``raster31v2``).

The production successor to :class:`~src.model.apex_network.ApexNetwork`. Where
``ApexNetwork`` is a 61-D vector MLP, this network consumes the dual-scale ego
raster produced by :mod:`src.simd_env.featurizer`:

- ``tactical``  ``(N, 9, 31, 31)`` float32 — forward-biased close-range raster.
- ``strategic`` ``(N, 3, 25, 25)`` float32 — coarse whole-arena density raster.
- ``scalars``   ``(N, 26)`` float32 — ego-frame Markov-completing scalars.

Two conv trunks (tactical / strategic) plus the scalar vector are fused and fed
to the SAME dueling head :class:`ApexNetwork` uses (``dueling_q``): a shared
value stream ``V(s)`` and advantage stream ``A(s, a)`` combined as
``Q = V + (A - mean_a A)``. LayerNorm follows each fused/trunk FC (blueprint §3
calls for LayerNorm in the net for the target-network-free PQN algorithm).

The activations contract is byte-for-byte the one the web UI depends on
(:mod:`web.backend.serialize`): :meth:`forward_with_activations` returns
``(q_values, {"input", "hidden", "output", "value", "advantages"})`` with the
same tensor semantics as ``ApexNetwork.forward_with_activations``, so the
inspector / net-viz panels render unchanged.

Input contract: :meth:`forward` accepts either a mapping with keys
``tactical`` / ``strategic`` / ``scalars`` (float tensors on the network's
device), or the three tensors positionally. A leading batch dim is required
(``(N, ...)``); single observations should be unsqueezed by the caller. The
helper :func:`raster_tensors_from_obs` converts a featurizer ``uint8`` obs dict
into the float tensors this network consumes (reusing ``expand_tactical``).
"""

from __future__ import annotations

from typing import Dict, Mapping, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from src.simd_env.featurizer import (
    SCALARS_DIM,
    STRATEGIC_CHANNELS,
    STRATEGIC_SIZE,
    TACTICAL_CHANNELS,
    TACTICAL_SIZE,
    expand_tactical,
)

from .base_network import dueling_q, init_weights_orthogonal

__all__ = ["RasterDuelingNetwork", "raster_tensors_from_obs"]

# Input-contract constants (blueprint §2 raster geometry, re-exported for tests
# and checkpoint metadata). These MUST match the featurizer.
TACTICAL_SHAPE: Tuple[int, int, int] = (TACTICAL_CHANNELS, TACTICAL_SIZE, TACTICAL_SIZE)
STRATEGIC_SHAPE: Tuple[int, int, int] = (STRATEGIC_CHANNELS, STRATEGIC_SIZE, STRATEGIC_SIZE)
SCALARS_SHAPE: Tuple[int] = (SCALARS_DIM,)

# Fused-trunk / head sizes (blueprint P3 topology).
_TACTICAL_FC = 256
_STRATEGIC_FC = 64
_FUSED = 256
_OUTPUT_SIZE = 6


def _conv_out_size(size: int, kernel: int, stride: int, padding: int) -> int:
    """Spatial edge length after one Conv2d (floor division, no dilation)."""
    return (size + 2 * padding - kernel) // stride + 1


class RasterDuelingNetwork(nn.Module):
    """Dual-scale raster Dueling DQN with a LayerNorm fusion trunk.

    Architecture (blueprint P3):
        tactical  (9, 31, 31)
            Conv3x3 9->32 s1 p1  -> (32, 31, 31)  ReLU
            Conv3x3 32->64 s2 p1 -> (64, 16, 16)  ReLU
            Conv3x3 64->64 s2 p1 -> (64, 8, 8)    ReLU
            flatten -> FC 256 -> LayerNorm -> ReLU
        strategic (3, 25, 25)
            Conv3x3 3->16 s2 p1  -> (16, 13, 13)  ReLU
            Conv3x3 16->32 s2 p1 -> (32, 7, 7)    ReLU
            flatten -> FC 64 -> LayerNorm -> ReLU
        scalars   (26,)  (passed through)
        fuse concat(256 + 64 + 26 = 346) -> FC 256 -> LayerNorm -> ReLU
            -> value_stream     FC 256 -> ReLU -> FC 1
            -> advantage_stream FC 256 -> ReLU -> FC 6
        Q = V + (A - mean_a A)

    The value/advantage streams reuse the ``value_stream`` / ``advantage_stream``
    attribute names ``ApexNetwork`` uses so shared visualization/serialization
    code finds them.

    Attributes:
        output_size: Number of actions (6: 3 dirs x 2 speed modes).
        tactical_shape / strategic_shape / scalars_dim: Input-contract sizes.
    """

    def __init__(self, output_size: int = _OUTPUT_SIZE) -> None:
        """Build the network with orthogonal init.

        Args:
            output_size: Number of discrete actions (default 6).
        """
        super().__init__()
        self.output_size = output_size
        self.tactical_shape = TACTICAL_SHAPE
        self.strategic_shape = STRATEGIC_SHAPE
        self.scalars_dim = SCALARS_DIM

        # --- Tactical conv trunk ---
        self.tactical_conv = nn.Sequential(
            nn.Conv2d(TACTICAL_CHANNELS, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        t = TACTICAL_SIZE
        t = _conv_out_size(t, 3, 1, 1)
        t = _conv_out_size(t, 3, 2, 1)
        t = _conv_out_size(t, 3, 2, 1)
        self._tactical_flat = 64 * t * t
        self.tactical_fc = nn.Sequential(
            nn.Linear(self._tactical_flat, _TACTICAL_FC),
            nn.LayerNorm(_TACTICAL_FC),
            nn.ReLU(),
        )

        # --- Strategic conv trunk ---
        self.strategic_conv = nn.Sequential(
            nn.Conv2d(STRATEGIC_CHANNELS, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        g = STRATEGIC_SIZE
        g = _conv_out_size(g, 3, 2, 1)
        g = _conv_out_size(g, 3, 2, 1)
        self._strategic_flat = 32 * g * g
        self.strategic_fc = nn.Sequential(
            nn.Linear(self._strategic_flat, _STRATEGIC_FC),
            nn.LayerNorm(_STRATEGIC_FC),
            nn.ReLU(),
        )

        # --- Fusion trunk ---
        fused_in = _TACTICAL_FC + _STRATEGIC_FC + SCALARS_DIM
        self._fused_in = fused_in
        self.fuse_fc = nn.Sequential(
            nn.Linear(fused_in, _FUSED),
            nn.LayerNorm(_FUSED),
            nn.ReLU(),
        )

        # --- Dueling head (same names/shape contract as ApexNetwork) ---
        self.value_stream = nn.Sequential(
            nn.Linear(_FUSED, _FUSED), nn.ReLU(), nn.Linear(_FUSED, 1)
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(_FUSED, _FUSED), nn.ReLU(), nn.Linear(_FUSED, output_size)
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Orthogonal init for all conv/linear layers (biases zeroed)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=2.0**0.5)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)
        # Output layers of the dueling streams use gain 1.0 (match ApexNetwork).
        init_weights_orthogonal(self.value_stream[-1], gain=1.0)
        init_weights_orthogonal(self.advantage_stream[-1], gain=1.0)

    # -- input normalization ------------------------------------------------
    def _unpack(
        self,
        obs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        strategic: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Resolve the (tactical, strategic, scalars) triple from flexible args.

        Accepts either a mapping with keys ``tactical`` / ``strategic`` /
        ``scalars`` (first positional arg), or the three tensors positionally.

        Args:
            obs: Mapping or the tactical tensor.
            strategic: Strategic tensor when ``obs`` is the tactical tensor.
            scalars: Scalars tensor when ``obs`` is the tactical tensor.

        Returns:
            ``(tactical, strategic, scalars)`` float tensors, batched.
        """
        if isinstance(obs, Mapping):
            tactical = obs["tactical"]
            strategic = obs["strategic"]
            scalars = obs["scalars"]
        else:
            tactical = obs
            if strategic is None or scalars is None:
                raise ValueError("positional call requires (tactical, strategic, scalars) tensors")
        return tactical, strategic, scalars

    def _features(
        self, tactical: torch.Tensor, strategic: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Compute the fused 256-D feature vector (post-LayerNorm, post-ReLU).

        Args:
            tactical: ``(N, 9, 31, 31)`` float tensor.
            strategic: ``(N, 3, 25, 25)`` float tensor.
            scalars: ``(N, 26)`` float tensor.

        Returns:
            ``(N, 256)`` fused feature tensor.
        """
        n = tactical.shape[0]
        t = self.tactical_conv(tactical).reshape(n, -1)
        t = self.tactical_fc(t)
        g = self.strategic_conv(strategic).reshape(n, -1)
        g = self.strategic_fc(g)
        fused = torch.cat([t, g, scalars], dim=1)
        return self.fuse_fc(fused)

    def forward(
        self,
        obs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        strategic: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute dueling Q-values ``(N, 6)``.

        Args:
            obs: Mapping with ``tactical`` / ``strategic`` / ``scalars`` keys, or
                the tactical tensor (with ``strategic`` / ``scalars`` positional).
            strategic: Strategic tensor when ``obs`` is the tactical tensor.
            scalars: Scalars tensor when ``obs`` is the tactical tensor.

        Returns:
            ``(N, output_size)`` Q-values.
        """
        tactical, strategic, scalars = self._unpack(obs, strategic, scalars)
        features = self._features(tactical, strategic, scalars)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        return dueling_q(value, advantages)

    def forward_with_activations(
        self,
        obs: Union[Mapping[str, torch.Tensor], torch.Tensor],
        strategic: Optional[torch.Tensor] = None,
        scalars: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Forward pass returning Q-values and the UI activations dict.

        The activations contract EXACTLY matches
        :meth:`ApexNetwork.forward_with_activations` so
        :mod:`web.backend.serialize` renders both networks identically:

        - ``input``: the flattened scalar vector (the raster planes are large and
          spatial; the inspector's "input" band is the scalar Markov vector).
        - ``hidden``: the fused 256-D feature vector.
        - ``value``: ``V(s)`` ``(N, 1)``.
        - ``advantages``: ``A(s, a)`` ``(N, 6)``.
        - ``output``: the fused Q-values ``(N, 6)``.

        All activation tensors are detached and moved to CPU (matching the base
        contract), while the returned Q-values stay on-device and grad-attached.

        Args:
            obs: Mapping or the tactical tensor.
            strategic: Strategic tensor when ``obs`` is the tactical tensor.
            scalars: Scalars tensor when ``obs`` is the tactical tensor.

        Returns:
            ``(q_values, activations_dict)``.
        """
        tactical, strategic, scalars = self._unpack(obs, strategic, scalars)
        features = self._features(tactical, strategic, scalars)
        value = self.value_stream(features)
        advantages = self.advantage_stream(features)
        q_values = dueling_q(value, advantages)

        activations = {
            "input": scalars.detach().cpu(),
            "hidden": features.detach().cpu(),
            "value": value.detach().cpu(),
            "advantages": advantages.detach().cpu(),
            "output": q_values.detach().cpu(),
        }
        return q_values, activations

    def get_num_parameters(self) -> Dict[str, int]:
        """Parameter counts per component and total.

        Returns:
            Dict with ``total`` and per-trunk/stream counts.
        """

        def count(module: nn.Module) -> int:
            return sum(p.numel() for p in module.parameters())

        return {
            "total": count(self),
            "tactical_conv": count(self.tactical_conv),
            "tactical_fc": count(self.tactical_fc),
            "strategic_conv": count(self.strategic_conv),
            "strategic_fc": count(self.strategic_fc),
            "fuse_fc": count(self.fuse_fc),
            "value_stream": count(self.value_stream),
            "advantage_stream": count(self.advantage_stream),
        }

    def __repr__(self) -> str:
        """String representation with total parameter count."""
        total = self.get_num_parameters()["total"]
        return (
            f"RasterDuelingNetwork(tactical={self.tactical_shape}, "
            f"strategic={self.strategic_shape}, scalars={self.scalars_dim}, "
            f"output_size={self.output_size}, total_params={total:,})"
        )


def raster_tensors_from_obs(
    obs: Mapping[str, np.ndarray],
    device: Optional[torch.device] = None,
    dtype: torch.dtype = torch.float32,
) -> Dict[str, torch.Tensor]:
    """Convert a featurizer ``uint8`` obs dict into float network-input tensors.

    Reuses :func:`~src.simd_env.featurizer.expand_tactical` for the tactical
    plane expansion (9 channels) so this stays the single source of truth for
    the uint8 -> float mapping. Leading ``(E, S)`` or ``(N,)`` dims are flattened
    into one batch dim ``N`` and the last three (tactical) / two (scalars) axes
    are preserved.

    Args:
        obs: Dict with ``tactical_uint8`` ``(..., 2, 31, 31)``,
            ``strategic_uint8`` ``(..., 3, 25, 25)`` and ``scalars``
            ``(..., 26)`` (as produced by
            :func:`~src.simd_env.featurizer.build_observations`).
        device: Target device for the returned tensors.
        dtype: Floating dtype for the returned tensors (default float32).

    Returns:
        Dict with ``tactical`` ``(N, 9, 31, 31)``, ``strategic``
        ``(N, 3, 25, 25)`` and ``scalars`` ``(N, 26)`` float tensors.
    """
    tactical = expand_tactical(obs["tactical_uint8"])  # (..., 9, 31, 31) float32
    strategic = np.asarray(obs["strategic_uint8"]).astype(np.float32) / 255.0
    scalars = np.asarray(obs["scalars"]).astype(np.float32)

    tactical = tactical.reshape((-1,) + TACTICAL_SHAPE)
    strategic = strategic.reshape((-1,) + STRATEGIC_SHAPE)
    scalars = scalars.reshape((-1, SCALARS_DIM))

    return {
        "tactical": torch.as_tensor(tactical, dtype=dtype, device=device),
        "strategic": torch.as_tensor(strategic, dtype=dtype, device=device),
        "scalars": torch.as_tensor(scalars, dtype=dtype, device=device),
    }
