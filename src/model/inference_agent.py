"""Forward-only inference agent for a trained Apex DQN checkpoint.

``ApexPolicy`` is the training vehicle — it owns an optimizer, a target network,
a replay buffer, epsilon schedules, and the whole learning loop. None of that is
needed to *use* a trained snake: watching it play, evaluating it, or serving it
behind the web UI only requires a forward pass and an argmax.

``InferenceAgent`` is that minimal path. It holds a single eval-mode network and
answers three questions — "what are the Q-values here", "what action do you take",
and "show me your activations" — with optional action masking so it never has to
pick a move that walks into a wall. It is intentionally cheap to build and free of
training state, which makes it the right tool for eval scripts, tournaments, and
any read-only serving surface.

Design note: this does not replace ``ApexPolicy`` in the live training loop; it
sits beside it as the clean, dependency-light "just run the model" interface.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple, Union

import numpy as np
import torch

from src.core.device_manager import DeviceManager
from src.model.apex_network import ApexNetwork
from src.model.obs_spec import (
    DEFAULT_OBS_SPEC,
    KNOWN_OBS_SPECS,
    OBS_SPEC_KEY,
    RASTER31V2,
    VECTOR61,
    RasterObsShapes,
)

__all__ = ["InferenceAgent"]

# A raster observation is a dict carrying either the uint8 featurizer outputs
# (``tactical_uint8``/``strategic_uint8``/``scalars``) or the float network
# tensors (``tactical``/``strategic``/``scalars``). ``act``/``q_values``/
# ``activations`` accept either form for a single agent.
RasterObs = Dict[str, Union[np.ndarray, torch.Tensor]]

# Checkpoint state-dict tensors whose shapes reveal the architecture, so we can
# rebuild a matching network even when the checkpoint omits the size metadata.
_FEATURE_W = "feature_layer.0.weight"  # (hidden_size, input_size)
_ADVANTAGE_W = "advantage_stream.2.weight"  # (output_size, hidden_size)

_STATE_DICT_KEYS = ("dqn_state_dict", "model_state_dict", "state_dict")


class InferenceAgent:
    """A trained Apex network wrapped for greedy, no-grad action selection.

    Build one from a checkpoint::

        agent = InferenceAgent.from_checkpoint("saved_snakes/champion.pth")
        action = agent.act(state_vector)                 # greedy
        action = agent.act(state_vector, action_mask=m)  # masked greedy
        action = agent.act_safe(state_vector)            # mask derived from state

    The agent is stateless across calls and holds no optimizer or replay buffer.
    """

    def __init__(
        self,
        network: torch.nn.Module,
        device: Optional[torch.device] = None,
        metadata: Optional[dict] = None,
        obs_spec: str = DEFAULT_OBS_SPEC,
    ) -> None:
        """Wrap an already-constructed network.

        Args:
            network: A network with a dueling ``forward``/``forward_with_activations``
                contract — an ``ApexNetwork`` (``vector61``) or a
                ``RasterDuelingNetwork`` (``raster31v2``), weights already loaded.
            device: Torch device; defaults to :class:`DeviceManager` selection.
            metadata: Optional provenance (e.g. total_reward, update_counter).
            obs_spec: Which observation the network consumes; selects the
                ``act``/``q_values``/``activations`` input path. Defaults to
                :data:`~src.model.obs_spec.VECTOR61`.
        """
        if obs_spec not in KNOWN_OBS_SPECS:
            raise ValueError(f"Unknown obs_spec {obs_spec!r}; expected one of {KNOWN_OBS_SPECS}.")
        self.device = device or DeviceManager.get_device()
        self.network = network.to(self.device).eval()
        self.metadata: dict = dict(metadata or {})
        self._obs_spec = obs_spec

    # -- construction -------------------------------------------------------
    @classmethod
    def from_checkpoint(
        cls, checkpoint_path: str, device: Optional[torch.device] = None
    ) -> "InferenceAgent":
        """Build an agent from an Apex checkpoint file.

        Architecture dimensions come from the checkpoint metadata when present,
        otherwise they are inferred from the weight-tensor shapes — so a
        checkpoint from any training run loads without a matching config.

        Args:
            checkpoint_path: Path to a ``.pth`` Apex checkpoint.
            device: Optional target device.

        Returns:
            A ready-to-use :class:`InferenceAgent`.

        Raises:
            ValueError: If no recognizable network state dict is found.
        """
        device = device or DeviceManager.get_device()
        blob = torch.load(checkpoint_path, map_location=device, weights_only=False)

        state_dict = cls._extract_state_dict(blob)
        obs_spec = cls._detect_obs_spec(blob)

        if obs_spec == RASTER31V2:
            network = cls._build_raster_network(blob, state_dict)
        else:
            input_size, hidden_size, output_size = cls._infer_dims(blob, state_dict)
            network = ApexNetwork(
                input_size=input_size, hidden_size=hidden_size, output_size=output_size
            )
            network.load_state_dict(state_dict)

        metadata = {
            key: blob[key]
            for key in ("total_reward", "update_counter", "epsilon", "policy_type")
            if isinstance(blob, dict) and key in blob
        }
        metadata["checkpoint_path"] = checkpoint_path
        metadata[OBS_SPEC_KEY] = obs_spec
        return cls(network, device=device, metadata=metadata, obs_spec=obs_spec)

    @staticmethod
    def _detect_obs_spec(blob) -> str:
        """Read the checkpoint's obs_spec, defaulting to ``vector61`` when absent.

        The default keeps every pre-contract-v2 champion loadable unchanged.

        Args:
            blob: The loaded checkpoint mapping.

        Returns:
            A validated obs_spec string.

        Raises:
            ValueError: If the checkpoint records an unrecognized obs_spec.
        """
        if not isinstance(blob, dict):
            return DEFAULT_OBS_SPEC
        obs_spec = blob.get(OBS_SPEC_KEY, DEFAULT_OBS_SPEC)
        # Contract v2 may nest metadata under apex_config/config.
        if OBS_SPEC_KEY not in blob:
            for cfg_key in ("apex_config", "config"):
                cfg = blob.get(cfg_key)
                if isinstance(cfg, dict) and OBS_SPEC_KEY in cfg:
                    obs_spec = cfg[OBS_SPEC_KEY]
                    break
        obs_spec = str(obs_spec)
        if obs_spec not in KNOWN_OBS_SPECS:
            raise ValueError(
                f"Checkpoint declares unknown obs_spec {obs_spec!r}; "
                f"this build understands {KNOWN_OBS_SPECS}."
            )
        return obs_spec

    @staticmethod
    def _build_raster_network(blob, state_dict: dict) -> torch.nn.Module:
        """Construct a ``RasterDuelingNetwork`` sized from checkpoint metadata.

        Imported lazily so this module still imports if ``raster_network.py``
        lands slightly later than this one.

        Args:
            blob: The loaded checkpoint mapping (holds shape metadata).
            state_dict: The network state dict to load.

        Returns:
            A ``RasterDuelingNetwork`` with weights loaded.
        """
        from src.model.raster_network import RasterDuelingNetwork

        meta = dict(blob) if isinstance(blob, dict) else {}
        # RasterDuelingNetwork fixes tactical/strategic/scalar shapes to the
        # featurizer geometry; only output_size varies. We still validate the
        # recorded shape descriptor against the build so a stale/mismatched
        # checkpoint fails loudly rather than silently loading a wrong-sized net.
        shapes = RasterObsShapes.from_metadata(meta)
        output_size = int(meta.get("output_size", 6))
        network = RasterDuelingNetwork(output_size=output_size)

        expected = RasterObsShapes(
            tactical_channels=network.tactical_shape[0],
            tactical_size=network.tactical_shape[1],
            strategic_channels=network.strategic_shape[0],
            strategic_size=network.strategic_shape[1],
            scalars=network.scalars_dim,
        )
        if shapes != expected:
            raise ValueError(
                "raster31v2 checkpoint shape metadata "
                f"{shapes} does not match this build's RasterDuelingNetwork "
                f"input contract {expected}."
            )
        network.load_state_dict(state_dict)
        return network

    @staticmethod
    def _extract_state_dict(blob) -> dict:
        if not isinstance(blob, dict):
            raise ValueError("Checkpoint is not a dict; cannot read a network state.")
        for key in _STATE_DICT_KEYS:
            if key in blob and isinstance(blob[key], dict):
                return blob[key]
        # A bare state dict (tensor values) is also acceptable.
        if all(isinstance(v, torch.Tensor) for v in blob.values()) and blob:
            return blob
        raise ValueError(
            f"No network state dict found under keys {_STATE_DICT_KEYS} in checkpoint."
        )

    @staticmethod
    def _infer_dims(blob, state_dict: dict) -> Tuple[int, int, int]:
        # Prefer explicit metadata; fall back to weight shapes.
        feat = state_dict.get(_FEATURE_W)
        adv = state_dict.get(_ADVANTAGE_W)
        input_size = hidden_size = output_size = None
        if feat is not None:
            hidden_size, input_size = int(feat.shape[0]), int(feat.shape[1])
        if adv is not None:
            output_size = int(adv.shape[0])
        if isinstance(blob, dict):
            input_size = int(blob.get("input_size", input_size or 61))
            hidden_size = int(blob.get("hidden_size", hidden_size or 512))
            output_size = int(blob.get("output_size", output_size or 6))
        if not (input_size and hidden_size and output_size):
            raise ValueError("Could not determine network dimensions from checkpoint.")
        return input_size, hidden_size, output_size

    # -- properties ---------------------------------------------------------
    @property
    def obs_spec(self) -> str:
        """Which observation this agent's network consumes (see :mod:`obs_spec`)."""
        return self._obs_spec

    @property
    def input_size(self) -> int:
        """Vector input width. Only meaningful for the ``vector61`` spec.

        Raises:
            AttributeError: For a raster agent, whose input is a raster obs dict,
                not a flat vector.
        """
        if self._obs_spec != VECTOR61:
            raise AttributeError(
                f"input_size is only defined for the {VECTOR61!r} obs_spec; "
                f"this agent is {self._obs_spec!r}. Use obs_spec instead."
            )
        return int(self.network.input_size)

    @property
    def output_size(self) -> int:
        return int(self.network.output_size)

    # -- raster input path --------------------------------------------------
    def _prepare_raster(self, obs: RasterObs) -> Dict[str, torch.Tensor]:
        """Coerce a single raster observation into ``(1, ...)`` device tensors.

        Accepts either the featurizer's uint8 dict (``tactical_uint8`` /
        ``strategic_uint8`` / ``scalars``) or the already-expanded float dict
        (``tactical`` / ``strategic`` / ``scalars``). A single agent's leading
        ``(E, S)`` dims are flattened to a batch of exactly one row.

        Args:
            obs: A raster observation dict for one agent.

        Returns:
            Dict of ``tactical`` ``(1, 9, 31, 31)``, ``strategic``
            ``(1, 3, 25, 25)``, ``scalars`` ``(1, 26)`` float tensors on device.

        Raises:
            ValueError: If ``obs`` is not a mapping, is missing keys, or resolves
                to more than one agent.
        """
        from src.model.raster_network import raster_tensors_from_obs

        if not isinstance(obs, dict):
            raise ValueError(
                "raster InferenceAgent expects an observation dict "
                "(tactical[_uint8]/strategic[_uint8]/scalars); got "
                f"{type(obs).__name__}."
            )
        if "tactical_uint8" in obs:
            tensors = raster_tensors_from_obs(obs, device=self.device)
        elif "tactical" in obs:
            tensors = {
                k: torch.as_tensor(
                    np.asarray(obs[k]) if not isinstance(obs[k], torch.Tensor) else obs[k],
                    dtype=torch.float32,
                    device=self.device,
                )
                for k in ("tactical", "strategic", "scalars")
            }
            # Flatten any leading (E, S) dims down to N, keep the trailing shape.
            tensors["tactical"] = tensors["tactical"].reshape(
                (-1,) + tensors["tactical"].shape[-3:]
            )
            tensors["strategic"] = tensors["strategic"].reshape(
                (-1,) + tensors["strategic"].shape[-3:]
            )
            tensors["scalars"] = tensors["scalars"].reshape((-1, tensors["scalars"].shape[-1]))
        else:
            raise ValueError("raster observation dict must contain 'tactical_uint8' or 'tactical'.")
        n = tensors["tactical"].shape[0]
        if n != 1:
            raise ValueError(
                f"InferenceAgent expects a single raster observation; resolved {n} "
                "agents. Call it once per agent."
            )
        return tensors

    def _raster_mask(self, obs: RasterObs) -> Optional[np.ndarray]:
        """Extract a single-agent 6-bit action mask from a raster obs, if present."""
        if isinstance(obs, dict) and obs.get("mask") is not None:
            mask = np.asarray(obs["mask"]).reshape(-1)
            if mask.size == self.output_size:
                return mask
        return None

    # -- inference ----------------------------------------------------------
    def _prepare(self, state) -> torch.Tensor:
        """Coerce a single state into a ``(1, input_size)`` float tensor on the device.

        This agent is single-state by design (``act`` returns one action). A
        multi-row batch is rejected explicitly rather than silently flattened —
        otherwise ``act``'s ``argmax`` over the flattened tensor could return an
        index outside ``[0, output_size)``.
        """
        if isinstance(state, torch.Tensor):
            x = state.detach().to(self.device, dtype=torch.float32)
        else:
            x = torch.as_tensor(np.asarray(state), dtype=torch.float32, device=self.device)
        if x.dim() == 1:
            x = x.unsqueeze(0)
        if x.dim() != 2 or x.shape[0] != 1:
            raise ValueError(
                "InferenceAgent expects a single state (1-D, or shape (1, input_size)); "
                f"got shape {tuple(x.shape)}. Call it once per state."
            )
        return x

    @torch.no_grad()
    def q_values(self, state) -> np.ndarray:
        """Return the Q-value for every action as a 1-D numpy array.

        Args:
            state: A ``vector61`` state vector, or a ``raster31v2`` observation
                dict (matching this agent's :attr:`obs_spec`).
        """
        if self._obs_spec == RASTER31V2:
            q = self.network(self._prepare_raster(state))
        else:
            q = self.network(self._prepare(state))
        return q.view(-1).cpu().numpy()

    @torch.no_grad()
    def act(self, state, action_mask=None) -> int:
        """Return the greedy action, optionally restricted to valid actions.

        Args:
            state: State vector (list / numpy / tensor of length ``input_size``).
            action_mask: Optional length-``output_size`` mask; entries that are
                falsy mark actions as invalid. If every action is masked out the
                mask is ignored (better a wall than a crash).

        For a ``raster31v2`` agent ``state`` is a raster observation dict; when
        no explicit ``action_mask`` is passed, the 6-bit ``mask`` embedded in the
        obs dict (if any) is used automatically.

        Returns:
            The chosen action index.
        """
        if self._obs_spec == RASTER31V2:
            q = self.network(self._prepare_raster(state)).view(-1)
            if action_mask is None:
                action_mask = self._raster_mask(state)
        else:
            q = self.network(self._prepare(state)).view(-1)
        if action_mask is not None:
            mask = torch.as_tensor(np.asarray(action_mask), dtype=torch.bool, device=q.device).view(
                -1
            )
            if mask.numel() == q.numel() and bool(mask.any()):
                q = q.masked_fill(~mask, float("-inf"))
        return int(torch.argmax(q).item())

    @torch.no_grad()
    def act_safe(self, state) -> int:
        """Greedy action with a "don't-trap-yourself" mask derived from the state.

        Uses the per-action danger features baked into the state vector to avoid
        immediately-fatal moves when a safe alternative exists. Falls back to a
        plain greedy pick if the mask cannot be built.

        For a ``raster31v2`` agent the safety mask travels with the observation
        (the featurizer's 6-bit ``mask``), so this defers to :meth:`act`, which
        applies that embedded mask automatically.
        """
        if self._obs_spec == RASTER31V2:
            return self.act(state)
        try:
            from src.training.action_mask import valid_action_mask_from_states

            x = self._prepare(state)
            mask = valid_action_mask_from_states(x).view(-1)
            return self.act(state, action_mask=mask)
        except Exception:
            return self.act(state)

    @torch.no_grad()
    def activations(self, state) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """Return ``(q_values, {input, hidden, output, value, advantages})`` for viz.

        ``value``/``advantages`` are the dueling decomposition (Q = V + (A − mean A)).
        Both obs_specs expose the same activation keys so ``web.backend.serialize``
        is untouched. For ``raster31v2`` the ``input`` band is the scalar vector.
        """
        if self._obs_spec == RASTER31V2:
            prepared = self._prepare_raster(state)
        else:
            prepared = self._prepare(state)
        q, acts = self.network.forward_with_activations(prepared)
        np_acts = {k: v.reshape(-1).cpu().numpy() for k, v in acts.items()}
        return q.view(-1).cpu().numpy(), np_acts

    def __repr__(self) -> str:
        if self._obs_spec == VECTOR61:
            head = f"input={self.input_size}"
        else:
            head = f"obs_spec={self._obs_spec!r}"
        return f"InferenceAgent({head}, output={self.output_size}, device={self.device})"
