"""Student-checkpoint rollout policy for the T34 teacher.

``StudentPolicy`` subclasses the ratified :class:`B8Policy` and replaces only
the hero network: the world's B8 carrier still builds/replays the world and the
frozen opponents keep B8Policy's exact dispatch.  For a plain ego2s student
(arm S1b) only ``_forward`` is replaced, so observation building, batching and
tie handling are byte-for-byte B8Policy's.  A tower student (arm S1a) also
paints the full-sight planes from the same simulator rows.

Factory reference for ``--policy-factory``: ``research.t34_claim_a.student:student_policy``.
"""

from __future__ import annotations

import contextlib
import hashlib
import time
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
import torch

from research.t34_exit_b8.policy import B8Policy, PolicyEvaluation

FACTORY = "research.t34_claim_a.student:student_policy"


def sha256_file(path: str | Path) -> str:
    """Bounded-memory SHA-256 of one file."""
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


@contextlib.contextmanager
def full_fp32() -> Iterator[None]:
    """Disable CUDA TF32 for one forward block (labels were recorded in full fp32).

    cuDNN convolutions default to TF32 on Ampere+/Ada GPUs (the L4), which would
    perturb previous-student Q values by ~1e-3 relative.  CPU is unaffected.
    """
    matmul = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = False
    try:
        with torch.backends.cudnn.flags(
            enabled=torch.backends.cudnn.enabled,
            benchmark=False,
            deterministic=True,
            allow_tf32=False,
        ):
            yield
    finally:
        torch.backends.cuda.matmul.allow_tf32 = matmul


def checkpoint_arch(path: str | Path) -> str:
    """Return the ``arch`` field of a student checkpoint without building it."""
    blob = torch.load(path, map_location="cpu", weights_only=False)
    return str(blob.get("arch", ""))


def load_student_model(path: str | Path, device: str | torch.device = "cpu") -> torch.nn.Module:
    """Load a plain ego2s (S1b) or ``exit-fit-tower/1`` (S1a) student in eval mode."""
    from research.t34_exit_b8_fit.network import ARCH as TOWER_ARCH
    from research.t34_exit_b8_fit.network import load_exit_fit_tower_checkpoint
    from src.model.ego2s_network import load_ego2s_checkpoint

    if checkpoint_arch(path) == TOWER_ARCH:
        return load_exit_fit_tower_checkpoint(path, torch.device(device))
    return load_ego2s_checkpoint(str(path), torch.device(device))


def needs_full_sight(model: torch.nn.Module) -> bool:
    """Whether a model's forward consumes the S1a full-sight planes."""
    return bool(getattr(model, "requires_full_sight", False))


def full_sight_planes(sim: Any, rows: np.ndarray) -> np.ndarray:
    """Paint uint8 ``(N, 4, 88, 152)`` planes for simulator ``rows``."""
    from research.t34_exit_b8_fit.full_sight import full_sight_from_ego_view

    return full_sight_from_ego_view(sim.ego_view(), np.asarray(rows, dtype=np.int64)).planes


class StudentPolicy(B8Policy):
    """B8Policy with the hero forward served by an explicit student checkpoint."""

    def __init__(
        self, device: str = "cpu", *, checkpoint: str | Path, batch_device_vectors: bool = False
    ) -> None:
        super().__init__(device, batch_device_vectors=batch_device_vectors)
        self.checkpoint_path = str(checkpoint)
        self.checkpoint_sha256 = sha256_file(checkpoint)
        model = load_student_model(checkpoint, "cpu")
        self.student = model.to(self._models.device, dtype=torch.float32).eval()
        self.requires_full_sight = needs_full_sight(self.student)

    # -- forward ---------------------------------------------------------------
    def _forward(self, net: Any, obs: dict[str, np.ndarray]) -> np.ndarray:
        """Ignore the carrier ``net``; evaluate the student on the same rows."""
        del net
        from src.model.ego2s_network import obs_tensors

        started = time.perf_counter()
        with torch.inference_mode():
            tensors = obs_tensors(obs, self._models.device)
            if self.requires_full_sight:
                sight = torch.from_numpy(np.ascontiguousarray(obs["full_sight"]))
                value = self.student(*tensors, sight.to(self._models.device))
            else:
                value = self.student(*tensors)
            value = value.float().cpu().numpy()
        self.hero_forward_seconds += time.perf_counter() - started
        return np.asarray(value, dtype=np.float32)

    def _observe(self, sim: Any, rows: np.ndarray) -> dict[str, np.ndarray]:
        from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

        obs = dict(
            build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows)
        )
        obs["full_sight"] = full_sight_planes(sim, rows)
        return obs

    # -- the RolloutPolicy surface (tower students only diverge) ----------------
    def evaluate(self, world: Any) -> PolicyEvaluation:
        if not self.requires_full_sight:
            return super().evaluate(world)
        sim = world.sim
        q = self._forward(None, self._observe(sim, np.array([[0, 0]])))[0]
        return _evaluation(sim, q)

    def evaluate_rows(self, sim: Any, rows: np.ndarray, net: Any) -> np.ndarray:
        if not self.requires_full_sight:
            return super().evaluate_rows(sim, rows, net)
        return self._forward(net, self._observe(sim, rows))

    def evaluate_worlds(
        self, worlds: Sequence[Any], *, bucket_size: int = 64
    ) -> list[PolicyEvaluation]:
        if not self.requires_full_sight:
            return super().evaluate_worlds(worlds, bucket_size=bucket_size)
        if type(bucket_size) is not int or not 1 <= bucket_size <= 1536:
            raise ValueError("inference bucket size must lie in [1, 1536]")
        results: list[PolicyEvaluation] = []
        root = np.asarray([[0, 0]], dtype=np.int64)
        for start in range(0, len(worlds), bucket_size):
            chunk = worlds[start : start + bucket_size]
            parts = [self._observe(world.sim, root) for world in chunk]
            obs = {key: np.concatenate([part[key] for part in parts], axis=0) for key in parts[0]}
            values = self._forward(None, obs)
            results.extend(_evaluation(world.sim, q) for world, q in zip(chunk, values))
        return results


def _evaluation(sim: Any, q: np.ndarray) -> PolicyEvaluation:
    """Exactly B8Policy's legal/advisory-mask fallback rules."""
    legal = np.asarray(sim.get_legal_action_mask()[0, 0], dtype=bool).copy()
    if not legal.any():
        legal[:3] = True
    select = np.asarray(sim.get_advisory_action_mask()[0, 0], dtype=bool).copy()
    if not select.any():
        select[:3] = legal[:3]
    return PolicyEvaluation(q=q, legal=legal, select=select)


def student_policy(*, device: str, checkpoint: str | Path) -> StudentPolicy:
    """``--policy-factory`` entry: the ratified label seam passes device + checkpoint."""
    return StudentPolicy(str(device), checkpoint=Path(checkpoint))
