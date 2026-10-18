#!/usr/bin/env python3
"""Execute Q0b's retained T22/T24 short/long fixtures on a CUDA host.

The six-policy differential runs execute real frozen B8 inference on CUDA while
vector policies and retained-body kernels follow the registered CPU actor path.
Each side owns independent simulators, policies and RNGs. Missing checkpoints or
CUDA fail closed; these commands never turn pytest skips into PASS receipts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from pathlib import Path


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fixture_sources(fixture: str) -> dict[str, str]:
    paths = [
        __file__,
        "tests/test_t22_frozen_pool.py",
        "tests/fixtures/t22/league_pool_pre_t22.py",
        "tests/fixtures/t24/v8_nb_pre_t24.py",
        "research/redesign_m3c_20261008/league_pool.py",
        "src/simd_env/v8_nb.py",
        "src/simd_env/ego_raster.py",
        "src/simd_env/ego_raster_bv8.py",
        "src/simd_env/vector61_policy.py",
        "research/redesign_m3c_20261008/league_random_safe.py",
        "tests/test_t22_random_safe.py" if fixture == "t22" else "tests/test_t24_v8_bodymap.py",
    ]
    return {
        str(Path(path).relative_to(Path.cwd()) if Path(path).is_absolute() else Path(path)): sha(
            Path(path)
        )
        for path in paths
    }


def run_cuda_pool() -> None:
    import os

    import numpy as np
    import torch

    from research.redesign_m3c_20261008 import league_pool
    from tests.test_t22_frozen_pool import (
        reference_module,
        run_recorded_pair,
        test_compiled_v8_keeps_tied_q_actions_on_all_slots,
    )

    inputs = Path(os.environ["Q0Q1_INPUT_ROOT"])
    checkpoints, b8 = inputs / "checkpoints", inputs / "init/B8_s3_final.pth"
    for length, frames in ((0, 96), (1000, 12)):
        _, _, exposure = run_recorded_pair(
            seed=743 if length else 741,
            frames=frames,
            long_body_length=length,
            force_lifetime_boundary=not bool(length),
            device="cuda",
            checkpoint_dir=checkpoints,
            b8_checkpoint=b8,
        )
        if length:
            assert all(value >= length for value in exposure.values()), exposure
    # Independent CUDA selection for exact ties and an all-false normal fallback.
    q = torch.tensor([[3, 3, 2, 3, 3, 2], [1, 1, 1, 1, 1, 1]], device="cuda", dtype=torch.float32)
    masks = np.array([[True] * 6, [False] * 6])
    expected_mask = reference_module()._normal_fallback_mask(
        torch.from_numpy(masks.copy()).to("cuda")
    )
    expected = torch.where(expected_mask, q, torch.full_like(q, -torch.inf)).argmax(1)
    actual_mask = league_pool._normal_fallback_mask(torch.from_numpy(masks).to("cuda"))
    actual = torch.where(actual_mask, q, torch.full_like(q, -torch.inf)).argmax(1)
    assert torch.equal(actual, expected)
    assert actual.cpu().tolist() == [0, 0]
    test_compiled_v8_keeps_tied_q_actions_on_all_slots()
    torch.cuda.synchronize()


def run_t22() -> None:
    from tests.test_t22_random_safe import (
        test_batched_random_safe_is_row_order_free_and_handles_frame_zero_transition,
        test_batched_random_safe_matches_hashlib_for_every_normal_mask,
        test_batched_random_safe_uses_two_sha_blocks_for_long_signed_key,
        test_invalid_prepared_frame_retains_canonical_rejection,
        test_rebind_assignment_rebuilds_prefixes_and_out_of_range_seed_falls_back,
        test_sha_kernel_matches_hashlib_across_padding_boundaries,
    )

    for seed in (0, 17, -17, -(1 << 63), (1 << 63) - 1):
        test_batched_random_safe_matches_hashlib_for_every_normal_mask(seed)
    for width in (54, 55, 56, 63, 64, 95):
        test_sha_kernel_matches_hashlib_across_padding_boundaries(width)
    test_batched_random_safe_is_row_order_free_and_handles_frame_zero_transition()
    test_batched_random_safe_uses_two_sha_blocks_for_long_signed_key()
    test_rebind_assignment_rebuilds_prefixes_and_out_of_range_seed_falls_back()
    test_invalid_prepared_frame_retains_canonical_rejection()


def run_t24() -> None:
    from tests.test_t24_v8_bodymap import (
        test_bodymap_empty_and_dead_rows_match_prechange_contract,
        test_bodymap_repeated_own_cells_at_normal_and_boost_landings,
        test_bodymap_synthetic_full_result_parity,
        test_t24_fixture_bytes_are_pinned,
    )

    test_t24_fixture_bytes_are_pinned()
    for mechanics in (1, 2):
        for length in (5, 150, 1000, 1200, 1500):  # registered short and long bodies
            test_bodymap_synthetic_full_result_parity(mechanics, length)
        test_bodymap_repeated_own_cells_at_normal_and_boost_landings(mechanics)
    test_bodymap_empty_and_dead_rows_match_prechange_contract()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", choices=("t22", "t24"), required=True)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    import torch

    result = {"schema": "q0b-cuda-fixture/v1", "fixture": args.fixture, "status": "INCOMPLETE"}
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable")
        # Materialize an actual CUDA operation before testing actor-side NumPy/Numba contracts.
        if not bool(torch.tensor([1.0], device="cuda").is_cuda):
            raise RuntimeError("CUDA tensor allocation failed")
        run_cuda_pool()
        (run_t22 if args.fixture == "t22" else run_t24)()
        source = Path(
            "tests/test_t22_random_safe.py"
            if args.fixture == "t22"
            else "tests/test_t24_v8_bodymap.py"
        )
        result.update(
            status="PASS",
            target_runtime="cuda",
            short_body=True,
            long_body=True,
            source_sha256=sha(source),
            sources=fixture_sources(args.fixture),
            environment=__import__(
                "research.q0q1_20261017.q0b_measure", fromlist=["_environment_identity"]
            )._environment_identity(),
            execution={
                "b8_forward": "cuda",
                "vector61_forward": "cpu",
                "v8_kernel": "numba-cpu",
                "independent_old_new_pools": True,
            },
            runtime={
                "python": sys.version,
                "platform": platform.platform(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
            },
        )
    except Exception as exc:  # retain an admission-blocking receipt
        result["error"] = f"{type(exc).__name__}: {exc}"
    args.out.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n")
    return 0 if result["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
