"""CPU-only refusal contracts for the disjoint F5 timing calibration."""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from research.bl_residual import f5_calibration as subject
from research.bl_residual import f5_qualification as qualification
from research.bl_residual.f5_r10a_annex import allocate


def test_calibration_refuses_anything_other_than_twelve_source_observations() -> None:
    with pytest.raises(subject.CalibrationRefusal, match="twelve"):
        subject._cache(object(), [])


def test_calibration_refuses_a_malformed_fixture_digest_before_compute() -> None:
    with pytest.raises(subject.CalibrationRefusal, match="fixture digest"):
        subject.run(
            object(),
            Path("missing"),
            [],
            "not-a-digest",
            source_hashes={},
            environment_sha256="0" * 64,
        )


def test_calibration_refuses_noncanonical_fixture_rows_before_compute() -> None:
    with pytest.raises(subject.CalibrationRefusal, match="frozen sealed allocation"):
        subject.run(
            object(), Path("missing"), [], "0" * 64, source_hashes={}, environment_sha256="0" * 64
        )


def test_saved_state_value_compares_numpy_rng_arrays_exactly() -> None:
    import numpy as np

    left = ("MT19937", np.asarray([1, 2], dtype=np.uint32), 3)
    assert subject._equal_saved_value(left, ("MT19937", np.asarray([1, 2], dtype=np.uint32), 3))
    assert not subject._equal_saved_value(left, ("MT19937", np.asarray([1, 3], dtype=np.uint32), 3))


def test_saved_state_equality_refuses_dtype_and_signed_zero_changes() -> None:
    import numpy as np
    import torch

    assert not subject._equal_saved_value(torch.tensor([1.0]), torch.tensor([1.0]).double())
    assert not subject._equal_saved_value(torch.tensor([0.0]), torch.tensor([-0.0]))
    assert not subject._equal_saved_value(np.asarray([0.0]), np.asarray([-0.0]))
    assert not subject._equal_saved_value(0.0, -0.0)


def test_surrogate_cache_satisfies_the_actual_fit_cache_contract_on_cpu() -> None:
    from research.bl_residual.fit import _cache as fit_cache

    cache = subject._surrogate_cache(torch.zeros((512, 256)), torch.zeros((512, 6)))
    resolved = fit_cache(cache, torch.device("cpu"))
    assert resolved["sampling_prob"].shape == (512,)
    assert torch.equal(resolved["sampling_weight"], torch.ones(512))


def _timing_rows() -> tuple[list[dict[str, object]], str]:
    rows = [dict(row) for row in allocate(0) if row["timing_only"] is True]
    document = {"schema": "bl-f5-sealed-timing-fixtures/v1", "fixtures": rows}
    fixture_sha = subject._digest_bytes(
        (json.dumps(document, sort_keys=True, allow_nan=False) + "\n").encode()
    )
    return subject._fixture_rows(rows, fixture_sha), fixture_sha


def _rng_snapshot() -> tuple[object, tuple[object, ...], torch.Tensor]:
    return random.getstate(), np.random.get_state(), torch.get_rng_state().clone()


def _same_rng_state(left: tuple[object, ...], right: tuple[object, ...]) -> bool:
    return subject._equal_saved_value(left, right)


def test_reconstructed_calibration_inputs_reuse_ordered_source_and_cache_after_fixture_skips(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A restarted calibration rebuilds all twelve source inputs independently.

    The native GridBatchSim observation constructor is retained so its per-world
    deterministic RNG is exercised.  The registered world/checkpoint and CUDA
    model boundaries are replaced with a source-observation seam and a fixed
    CPU feature/Q projection; neither replacement can hide row-order or cache
    identity drift.
    """

    rows, _ = _timing_rows()
    allocation_order = [int(row["allocation_index"]) for row in rows]
    calls: list[int] = []

    def source_root(row: dict[str, object], checkpoint: Path):
        del checkpoint
        key = int(row["allocation_index"])
        calls.append(key)
        observation, _ = qualification._native_observation(int(row["seed"]))
        return object(), observation, 0.0

    monkeypatch.setattr(qualification, "_source_root_observation", source_root)

    class DeterministicBase:
        def __call__(self, *tensors: torch.Tensor) -> torch.Tensor:
            n = tensors[0].shape[0]
            signal = tensors[0].float().reshape(n, -1).mean(dim=1, keepdim=True)
            return torch.cat([signal + value for value in range(6)], dim=1)

    def source_tensors(observation: dict[str, np.ndarray], device: torch.device):
        del device
        return tuple(
            torch.from_numpy(np.ascontiguousarray(observation[name]))
            for name in ("local", "global", "scalars")
        )

    def deterministic_features(
        base: DeterministicBase,
        local: torch.Tensor,
        glob: torch.Tensor,
        scalars: torch.Tensor,
    ) -> torch.Tensor:
        del base, glob, scalars
        n = local.shape[0]
        signal = local.float().reshape(n, -1).mean(dim=1, keepdim=True)
        return torch.cat([signal + value for value in range(256)], dim=1)

    monkeypatch.setattr(subject, "obs_tensors", source_tensors)
    import research.a100_a2.model as a2_model

    monkeypatch.setattr(a2_model, "b8_features", deterministic_features)

    def build_inputs():
        observations = subject._source_observations(Path("sealed-b8.pth"), rows)
        cache = subject._cache(DeterministicBase(), observations)
        return observations, cache

    original_rng = _rng_snapshot()
    try:
        random.seed(113)
        np.random.seed(227)
        torch.manual_seed(331)
        baseline_rng = _rng_snapshot()

        uninterrupted_observations, uninterrupted_cache = build_inputs()
        uninterrupted_rng = _rng_snapshot()
        assert calls == allocation_order

        # A prior attempt may have completed these fixtures in any order.  The
        # calibration unit still reconstructs the complete canonical source
        # sequence from the sealed allocation rather than inheriting that order.
        skipped = {672, 676, 681}
        calls.clear()
        for row in reversed([row for row in rows if int(row["allocation_index"]) in skipped]):
            subject._source_observations(Path("sealed-b8.pth"), [row])
        reconstructed_observations, reconstructed_cache = build_inputs()
        reconstructed_rng = _rng_snapshot()

        assert calls == [681, 676, 672, *allocation_order]
        for left, right in zip(uninterrupted_observations, reconstructed_observations):
            assert left.keys() == right.keys()
            for name in left:
                np.testing.assert_array_equal(left[name], right[name])
        assert set(uninterrupted_cache) == set(reconstructed_cache)
        for name in uninterrupted_cache:
            assert torch.equal(uninterrupted_cache[name], reconstructed_cache[name]), name
        assert subject._cache_identity(uninterrupted_cache) == subject._cache_identity(
            reconstructed_cache
        )
        assert _same_rng_state(uninterrupted_rng, reconstructed_rng)
        assert _same_rng_state(baseline_rng, uninterrupted_rng)
    finally:
        random.setstate(original_rng[0])
        np.random.set_state(original_rng[1])
        torch.set_rng_state(original_rng[2])


@pytest.mark.skipif(
    not __import__("torch").cuda.is_available()
    or not __import__("os").environ.get("F5_QUALIFICATION_B8"),
    reason="requires a real CUDA runtime, uploaded checkpoint pool, and F5_QUALIFICATION_B8",
)
def test_cuda_full_calibration_runs_source_bound_continuation() -> None:
    from research.bl_residual.f5_package import ROOT, source_inventory
    from research.bl_residual.f5_r10a_annex import allocate
    from src.model.ego2s_network import load_ego2s_checkpoint

    checkpoint = Path(__import__("os").environ["F5_QUALIFICATION_B8"])
    fixture_rows = [row for row in allocate(0) if row["timing_only"] is True]
    fixture_document = {"schema": "bl-f5-sealed-timing-fixtures/v1", "fixtures": fixture_rows}
    fixture_sha = subject._digest_bytes(
        (__import__("json").dumps(fixture_document, sort_keys=True) + "\n").encode()
    )
    result = subject.run(
        load_ego2s_checkpoint(str(checkpoint), "cuda:0").eval(),
        checkpoint,
        fixture_rows,
        fixture_sha,
        source_hashes=source_inventory(ROOT),
        environment_sha256="0" * 64,
    )
    assert result["arms"]["C"]["serialized_continuation_exact"] is True
    assert result["arms"]["W"]["serialized_continuation_exact"] is True
