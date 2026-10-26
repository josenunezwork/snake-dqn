"""CPU-only refusal contracts for the disjoint F5 timing calibration."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_residual import f5_calibration as subject


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
    import torch

    from research.bl_residual.fit import _cache as fit_cache

    cache = subject._surrogate_cache(torch.zeros((512, 256)), torch.zeros((512, 6)))
    resolved = fit_cache(cache, torch.device("cpu"))
    assert resolved["sampling_prob"].shape == (512,)
    assert torch.equal(resolved["sampling_weight"], torch.ones(512))


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
