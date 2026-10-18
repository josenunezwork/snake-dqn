#!/usr/bin/env python3
"""Execute Q0b's retained T22/T24 short/long fixtures on a CUDA host.

The fixture kernels are NumPy/Numba contracts, so this records the CUDA host and
Torch runtime that owns the qualification while exercising the exact retained
CPU-side contracts used by its actors.  It refuses a host without an available
CUDA device and writes a hash-bound PASS/INCOMPLETE receipt.
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


def run_t22() -> None:
    from tests.test_t22_random_safe import (
        test_batched_random_safe_matches_hashlib_for_every_normal_mask,
        test_batched_random_safe_uses_two_sha_blocks_for_long_signed_key,
        test_batched_random_safe_is_row_order_free_and_handles_frame_zero_transition,
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
        for length in (5, 1500):  # registered short and long bodies
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
