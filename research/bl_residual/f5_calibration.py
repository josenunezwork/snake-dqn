"""Disjoint-fixture F5 optimiser calibration, never a science fit or acceptance."""

from __future__ import annotations

import copy
import hashlib
import io
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from research.bl_residual.f5_fit import F5Identity, F5Recipe, _fit, materialize_batch_plan
from research.bl_residual.f5_progress import QualificationProgress
from research.bl_residual.f5_r10a_annex import allocate
from research.bl_residual.model import FrozenB8Residual
from src.model.ego2s_network import obs_tensors

SCHEMA = "bl-f5-disjoint-calibration/v1"
SCHEMA_ASSET = "f5_calibration_schema_v1.json"
NAMESPACE = "bl-f5-disjoint-timing-calibration/20261025"
ROWS = 512
BOUNDARY_UPDATE = 1000


class CalibrationRefusal(ValueError):
    """The timing-only surrogate cannot safely exercise the F5 fit core."""


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _equal_saved_value(left: Any, right: Any) -> bool:
    """Compare full saved state values, including NumPy RNG arrays, exactly."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.detach().cpu().contiguous().numpy().tobytes()
            == right.detach().cpu().contiguous().numpy().tobytes()
        )
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        return (
            left.dtype == right.dtype
            and left.shape == right.shape
            and left.tobytes(order="C") == right.tobytes(order="C")
        )
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            _equal_saved_value(left[key], right[key]) for key in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            _equal_saved_value(a, b) for a, b in zip(left, right)
        )
    if isinstance(left, float):
        return repr(left) == repr(right)
    return left == right


def _cache(base: Any, observations: Sequence[Mapping[str, Any]]) -> dict[str, torch.Tensor]:
    """Build an in-memory calibration cache from exactly twelve source observations."""
    if len(observations) != 12:
        raise CalibrationRefusal("requires exactly twelve sealed timing observations")
    device = torch.device("cuda:0")
    rows = [observations[index % len(observations)] for index in range(ROWS)]
    merged = {
        name: np.concatenate([value[name] for value in rows], axis=0)
        for name in ("local", "global", "scalars")
    }
    from research.a100_a2.model import b8_features

    with torch.inference_mode():
        tensors = obs_tensors(merged, device)
        q = base(*tensors).float().detach().cpu()
        features = b8_features(base, *tensors).float().detach().cpu()
    if tuple(q.shape) != (ROWS, 6) or tuple(features.shape) != (ROWS, 256):
        raise CalibrationRefusal("B8 calibration feature/Q shape differs")
    return _surrogate_cache(features, q)


def _surrogate_cache(features: torch.Tensor, q: torch.Tensor) -> dict[str, torch.Tensor]:
    """Build the fixed, disclosed timing-only strata from source-derived features."""
    if tuple(features.shape) != (ROWS, 256) or tuple(q.shape) != (ROWS, 6):
        raise CalibrationRefusal("B8 calibration feature/Q shape differs")
    bare = q.argmax(dim=1)
    died = torch.zeros((ROWS, 6), dtype=torch.bool)
    died[torch.arange(ROWS), (bare + 1) % 6] = True
    return {
        "features": features,
        "q": q,
        "legal": torch.ones((ROWS, 6), dtype=torch.bool),
        "action_mask": torch.ones((ROWS, 6), dtype=torch.bool),
        "bare": bare,
        "died": died,
        "status": torch.ones((ROWS, 6), dtype=torch.bool),
        "mass": torch.zeros((ROWS, 6), dtype=torch.float32),
        "ordinary": torch.arange(ROWS).remainder(2).bool(),
        "natural_stream": torch.zeros(ROWS, dtype=torch.bool),
        "life_id": torch.arange(ROWS, dtype=torch.int64).remainder(12),
        # The timing-only schema fixes an unweighted surrogate; this is not an
        # estimate of scientific archive prevalence or a production cache rate.
        "sampling_prob": torch.ones(ROWS, dtype=torch.float32),
    }


def _digest_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _fixture_rows(rows: Sequence[Mapping[str, Any]], fixture_sha256: str) -> list[dict[str, Any]]:
    """Require precisely the immutable twelve timing-only allocation rows and bytes."""
    expected = [row for row in allocate(0) if row["timing_only"] is True]
    actual = [dict(row) for row in rows]
    expected_document = {"schema": "bl-f5-sealed-timing-fixtures/v1", "fixtures": expected}
    expected_digest = _digest_bytes(
        (json.dumps(expected_document, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    if actual != expected or fixture_sha256 != expected_digest:
        raise CalibrationRefusal(
            "calibration fixture rows or digest differ from the frozen sealed allocation"
        )
    return actual


def _schema_sha256() -> str:
    path = Path(__file__).with_name(SCHEMA_ASSET)
    if path.is_symlink() or not path.is_file():
        raise CalibrationRefusal("calibration schema asset is absent or unsafe")
    return _digest_bytes(path.read_bytes())


def _source_identity(source_hashes: Mapping[str, str]) -> str:
    from research.bl_residual.f5_package import ROOT, source_inventory

    if source_hashes != source_inventory(ROOT):
        raise CalibrationRefusal("source inventory differs from the complete current source pin")
    if not source_hashes or any(
        not isinstance(name, str)
        or not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
        for name, value in source_hashes.items()
    ):
        raise CalibrationRefusal("source inventory binding is malformed")
    return _digest_bytes(
        json.dumps(dict(source_hashes), sort_keys=True, separators=(",", ":")).encode()
    )


def _cache_identity(cache: Mapping[str, torch.Tensor]) -> str:
    hasher = hashlib.sha256()
    for name in sorted(cache):
        value = cache[name].detach().cpu().contiguous()
        hasher.update(name.encode("ascii"))
        hasher.update(str(value.dtype).encode("ascii"))
        hasher.update(repr(tuple(value.shape)).encode("ascii"))
        hasher.update(value.numpy().tobytes())
    return hasher.hexdigest()


def _source_observations(
    checkpoint: Path,
    rows: Sequence[Mapping[str, Any]],
    progress: QualificationProgress | None = None,
) -> list[dict[str, np.ndarray]]:
    """Build the exact source-root observations internally; callers cannot inject rows."""
    from research.bl_residual.f5_qualification import _source_root_observation

    result: list[dict[str, np.ndarray]] = []
    for row in rows:
        if progress is not None:
            progress.phase("calibration-source")
        _, observation, _ = _source_root_observation(row, checkpoint)
        result.append(observation)
    return result


def run(
    base: Any,
    checkpoint: Path,
    fixture_rows: Sequence[Mapping[str, Any]],
    fixture_sha256: str,
    *,
    source_hashes: Mapping[str, str],
    environment_sha256: str,
    progress: QualificationProgress | None = None,
) -> dict[str, Any]:
    """Time C's exact 2,000x256 core and a serialized continuation replay.

    This qualification-only private core invocation is submitted for independent
    implementation review. It is possible only with the fixed timing-fixture
    surrogate cache below and it never returns model state, loss values, Q
    values, labels, or an acceptance artifact.
    """
    started = time.monotonic()
    if not isinstance(fixture_sha256, str) or len(fixture_sha256) != 64:
        raise CalibrationRefusal("fixture digest is malformed")
    rows = _fixture_rows(fixture_rows, fixture_sha256)
    from research.bl_ppo.exports import digest
    from research.bl_residual import f5_job

    if (
        checkpoint.is_symlink()
        or not checkpoint.is_file()
        or digest(checkpoint) != f5_job.B8_SHA256
    ):
        raise CalibrationRefusal("calibration checkpoint differs from the fixed B8 pin")
    if not isinstance(environment_sha256, str) or len(environment_sha256) != 64:
        raise CalibrationRefusal("environment receipt digest is malformed")
    source_sha256 = _source_identity(source_hashes)
    observations = _source_observations(checkpoint, rows, progress)
    if progress is not None:
        progress.phase("calibration-cache")
    cache = _cache(base, observations)
    identity = F5Identity(
        source_sha256=source_sha256,
        loss_sha256=_digest_bytes(Path(__file__).with_name("f5_fit.py").read_bytes()),
        environment_sha256=environment_sha256,
        allocation_sha256=fixture_sha256,
        corpus_sha256=_cache_identity(cache),
    )
    if progress is not None:
        progress.phase("calibration-plan")
    plan = materialize_batch_plan(
        {
            name: cache[name]
            for name in ("status", "legal", "action_mask", "died", "ordinary", "life_id")
        },
        identity,
    )

    def state_equal(left: Any, right: Any) -> bool:
        return (
            left.updates_completed == right.updates_completed
            and _equal_saved_value(left.losses, right.losses)
            and left.identity == right.identity
            and left.recipe == right.recipe
            and left.plan_sha256 == right.plan_sha256
            and _equal_saved_value(
                left.model.residual.state_dict(), right.model.residual.state_dict()
            )
            and _equal_saved_value(left.optimizer_state, right.optimizer_state)
            and _equal_saved_value(left.python_rng_state, right.python_rng_state)
            and _equal_saved_value(left.numpy_rng_state, right.numpy_rng_state)
            and _equal_saved_value(left.torch_cpu_rng_state, right.torch_cpu_rng_state)
            and _equal_saved_value(left.torch_cuda_rng_states, right.torch_cuda_rng_states)
            and _equal_saved_value(left.plan_generator_state, right.plan_generator_state)
        )

    def measure(arm: str) -> dict[str, Any]:
        recipe = F5Recipe(arm=arm)  # type: ignore[arg-type]
        boundary: list[Any] = []

        def save(state: Any) -> None:
            if state.updates_completed == BOUNDARY_UPDATE:
                boundary.append(state)

        if progress is not None:
            progress.phase(f"calibration-{arm.lower()}-full")
        started = time.monotonic()
        complete = _fit(
            FrozenB8Residual(copy.deepcopy(base)).cuda(),
            cache,
            plan,
            recipe,
            None,
            synthetic=False,
            boundary_callback=save,
        )
        full_seconds = time.monotonic() - started
        if len(boundary) != 1:
            raise CalibrationRefusal("missing exact calibration continuation boundary")
        serialized = io.BytesIO()
        torch.save(boundary[0], serialized)
        serialized.seek(0)
        restored = torch.load(serialized, map_location="cpu", weights_only=False)
        if progress is not None:
            progress.phase(f"calibration-{arm.lower()}-resumed")
        resumed_started = time.monotonic()
        resumed = _fit(
            FrozenB8Residual(copy.deepcopy(base)).cuda(),
            cache,
            plan,
            recipe,
            restored,
            synthetic=False,
            boundary_callback=None,
        )
        continuation_seconds = time.monotonic() - resumed_started
        if not state_equal(complete, resumed):
            raise CalibrationRefusal(
                "serialized CUDA continuation differs from uninterrupted calibration"
            )
        return {
            "full_seconds": full_seconds,
            "serialized_continuation_seconds": continuation_seconds,
            "serialized_continuation_exact": True,
        }

    arm_measurements = {arm: measure(arm) for arm in ("C", "W")}

    return {
        "schema": SCHEMA,
        "namespace": NAMESPACE,
        "schema_asset": SCHEMA_ASSET,
        "schema_sha256": _schema_sha256(),
        "fixture_sha256": fixture_sha256,
        "surrogate": "timing-only labels derived from disjoint source features; not science data",
        "rows": ROWS,
        "updates": 2000,
        "batch_size": 256,
        "doom_batch_size": 128,
        "ordinary_batch_size": 128,
        "arms": arm_measurements,
        "state_export_timing": "HOLD_NOT_A_PRODUCER_EXPORT_MEASUREMENT",
        "fit_usable": False,
        "production_cache_feasibility": "HOLD_REQUIRES_ACTUAL_PLAN_AND_CACHE",
        "elapsed_seconds": time.monotonic() - started,
    }
