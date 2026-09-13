"""Independent parity tests for the tactical raster vectorization.

The candidate featurizer is compared with an embedded copy of the original
44acc33 paint primitive.  Full builds run once with the candidate primitive and
once with the embedded primitive temporarily installed in the candidate module,
then restore the candidate implementation.

These tests focus on the source-major paint stream owned by the tactical
renderer.  They use fixed worlds that include overlaps, tie values, walls,
dead slots, boost predictions, and a body buffer at its 150-segment capacity.
The full observation dictionary is compared for both raster contracts because
an optimization in the tactical producer must not alter strategic, scalar, or
mask output as a side effect.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pytest

import src.simd_env.featurizer as candidate_featurizer
from src.model.obs_spec import RASTER31V2, RASTER31V3
from src.simd_env.featurizer import ObsInputs

TACTICAL_HEAD_ROW = 23
TACTICAL_HEAD_COL = 15


_REFERENCE_PRIORITY = {
    0: 0,  # empty
    2: 1,  # ambient food
    3: 2,  # corpse food
    4: 3,  # enemy prediction
    5: 4,  # own body
    6: 5,  # enemy body
    7: 6,  # enemy head
    8: 7,  # own head
    1: 8,  # wall
}


def reference_paint(
    code_plane: np.ndarray,
    value_plane: np.ndarray,
    ei: np.ndarray,
    si: np.ndarray,
    row: np.ndarray,
    col: np.ndarray,
    code: np.ndarray | int,
    value: np.ndarray | int,
    size: int,
) -> None:
    """Independent copy of the original 44acc33 `_paint` implementation.

    This intentionally retains the original corrected-v3 Python loop and the
    legacy v2 stable-sort/last-write behavior. It does not import candidate
    constants or implementation helpers, so monkeypatching the candidate with
    this function creates an independent full-render oracle.
    """
    corrected = code_plane.dtype == np.int32
    scalar_code = np.ndim(code) == 0
    inb = (row >= 0) & (row < size) & (col >= 0) & (col < size)
    if not inb.all():
        if not inb.any():
            return
        ei, si, row, col = ei[inb], si[inb], row[inb], col[inb]
        if not scalar_code:
            code = code[inb]
        if np.ndim(value):
            value = value[inb]
    if corrected:
        codes = np.full(len(ei), int(code), dtype=np.int32) if scalar_code else np.asarray(code)
        values = (
            np.full(len(ei), int(value), dtype=np.uint16)
            if np.ndim(value) == 0
            else np.asarray(value)
        )
        ranks = np.asarray([_REFERENCE_PRIORITY[int(item)] for item in codes], dtype=np.int32)
        for index in range(len(ei)):
            current = code_plane[ei[index], si[index], row[index], col[index]]
            current_value = value_plane[ei[index], si[index], row[index], col[index]]
            if ranks[index] > current or (
                ranks[index] == current and values[index] > current_value
            ):
                code_plane[ei[index], si[index], row[index], col[index]] = ranks[index]
                value_plane[ei[index], si[index], row[index], col[index]] = values[index]
        return
    if scalar_code:
        code_plane[ei, si, row, col] = code
        value_plane[ei, si, row, col] = value
        return
    order = np.argsort(code, kind="stable")
    code_plane[ei[order], si[order], row[order], col[order]] = code[order]
    value_plane[ei[order], si[order], row[order], col[order]] = value[order]


def reference_build(inp: ObsInputs, mask: np.ndarray, obs_spec: str) -> dict:
    """Build through the candidate module with only `_paint` replaced by oracle."""
    original_paint = candidate_featurizer._paint
    candidate_featurizer._paint = reference_paint
    try:
        return candidate_featurizer.build_observations(inp, mask=mask, obs_spec=obs_spec)
    finally:
        candidate_featurizer._paint = original_paint


def _stress_inputs() -> tuple[ObsInputs, np.ndarray]:
    """Build a two-environment world exercising every tactical paint family."""
    envs, snakes, maxlen, food_slots = 2, 4, 150, 6
    heads = np.array(
        [
            [[40, 40], [43, 40], [42, 38], [110, 70]],
            [[0, 0], [5, 3], [8, 4], [100, 50]],
        ],
        dtype=np.int64,
    )
    body_len = np.array([[3, 4, 150, 5], [2, 3, 4, 150]], dtype=np.int64)
    lengths = np.array([[3, 5, 150, 7], [2, 4, 6, 150]], dtype=np.int64)
    alive = np.array([[True, True, True, False], [True, True, True, False]], dtype=bool)
    heading = np.array([[0, 1, 2, 3], [0, 1, 2, 3]], dtype=np.int64)
    boosting = np.array([[True, False, True, False], [False, True, False, False]], dtype=bool)

    # Unused and dead slots contain hostile sentinels.  A renderer that reads
    # beyond body_len, or fails to apply the alive mask, will differ from the
    # frozen output instead of accidentally passing with zero padding.
    bodies = np.full((envs, snakes, maxlen, 2), -777, dtype=np.int64)
    for env in range(envs):
        for snake in range(snakes):
            if not alive[env, snake]:
                continue
            bodies[env, snake, 0] = heads[env, snake]
            for segment in range(1, int(body_len[env, snake])):
                bodies[env, snake, segment] = (
                    heads[env, snake, 0] + 6 + (segment % 11),
                    heads[env, snake, 1] + 5 + (segment // 11),
                )

    # One cell is deliberately claimed by own body, enemy body, enemy head,
    # and stacked food.  The later source loop and the v3 rank/value tie rule
    # are both observable at this location.
    target = np.array((42, 38), dtype=np.int64)
    bodies[0, 0, 1] = target  # own body
    bodies[0, 0, 2] = (40, 42)
    bodies[0, 1, 1] = target  # enemy body
    bodies[0, 1, 2] = (44, 40)
    bodies[0, 1, 3] = (45, 40)
    bodies[0, 2, 1] = (61, 46)
    bodies[0, 2, 2] = (62, 46)

    food_cells = np.full((envs, food_slots, 2), -888, dtype=np.int64)
    food_mass = np.zeros((envs, food_slots), dtype=np.float64)
    food_is_corpse = np.zeros((envs, food_slots), dtype=bool)
    food_cells[0, 0] = target
    food_mass[0, 0] = 1.0
    food_cells[0, 1] = target
    food_mass[0, 1] = 2.0
    food_is_corpse[0, 1] = True
    food_cells[0, 2] = (42, 40)  # prediction target for the boosting enemy
    food_mass[0, 2] = 1.0
    food_cells[0, 3] = (80, 30)
    food_mass[0, 3] = 300.0  # value-byte saturation
    food_cells[1, 0] = (1, 1)
    food_mass[1, 0] = 1.0
    food_cells[1, 1] = (7, 4)
    food_mass[1, 1] = 1.0
    food_is_corpse[1, 1] = True

    masks = np.array(
        [
            [[True, False, True, False, True, True]] * snakes,
            [[False, True, True, True, False, True]] * snakes,
        ],
        dtype=bool,
    )
    inp = ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=body_len,
        lengths=lengths,
        alive=alive,
        heading=heading,
        boost_frames=np.array([[2, 1, 0, 9], [0, 3, 7, 11]], dtype=np.int64),
        frames_since_food=np.array([[10, 20, 30, 40], [50, 60, 70, 80]], dtype=np.int64),
        boosting=boosting,
        food_cells=food_cells,
        food_mass=food_mass,
        food_is_corpse=food_is_corpse,
        grid_w=145,
        grid_h=83,
        max_snakes=snakes,
        starvation_max=500,
        max_length=150,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([91, 17], dtype=np.int64),
        max_frames=5000,
        arena_type_flag=0.0,
    )
    return inp, masks


def _tie_inputs() -> ObsInputs:
    """Build same-class enemy-body ties with distinct TTL bytes."""
    heads = np.array([[[20, 20], [25, 15], [30, 15]]], dtype=np.int64)
    bodies = np.full((1, 3, 4, 2), -999, dtype=np.int64)
    bodies[0, 0, 0] = heads[0, 0]
    bodies[0, 1, 0] = heads[0, 1]
    bodies[0, 1, 1] = (25, 14)  # TTL round((2-1)/2*255) = 128
    bodies[0, 2, 0] = heads[0, 2]
    bodies[0, 2, 1] = (30, 16)
    bodies[0, 2, 2] = (30, 17)
    bodies[0, 2, 3] = (25, 14)  # TTL round((4-3)/4*255) = 64
    return ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=np.array([[1, 2, 4]], dtype=np.int64),
        lengths=np.array([[4, 3, 5]], dtype=np.int64),
        alive=np.ones((1, 3), dtype=bool),
        heading=np.zeros((1, 3), dtype=np.int64),
        boost_frames=np.zeros((1, 3), dtype=np.int64),
        frames_since_food=np.zeros((1, 3), dtype=np.int64),
        boosting=np.zeros((1, 3), dtype=bool),
        food_cells=np.full((1, 1, 2), -999, dtype=np.int64),
        food_mass=np.zeros((1, 1), dtype=np.float64),
        food_is_corpse=np.zeros((1, 1), dtype=bool),
        grid_w=40,
        grid_h=40,
        max_snakes=3,
        starvation_max=500,
        max_length=100,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([0], dtype=np.int64),
        max_frames=5000,
        arena_type_flag=0.0,
    )


def _wall_order_inputs() -> ObsInputs:
    """Place an enemy head and food on the outside of the observer's wall."""
    heads = np.array([[[0, 20], [-1, 20]]], dtype=np.int64)
    bodies = np.array([[[[0, 20]], [[-1, 20]]]], dtype=np.int64)
    return ObsInputs(
        heads=heads,
        bodies=bodies,
        body_len=np.ones((1, 2), dtype=np.int64),
        lengths=np.array([[5, 7]], dtype=np.int64),
        alive=np.ones((1, 2), dtype=bool),
        heading=np.array([[1, 1]], dtype=np.int64),
        boost_frames=np.zeros((1, 2), dtype=np.int64),
        frames_since_food=np.zeros((1, 2), dtype=np.int64),
        boosting=np.zeros((1, 2), dtype=bool),
        food_cells=np.array([[[-1, 20]]], dtype=np.int64),
        food_mass=np.array([[1.0]], dtype=np.float64),
        food_is_corpse=np.array([[False]], dtype=bool),
        grid_w=40,
        grid_h=40,
        max_snakes=2,
        starvation_max=500,
        max_length=100,
        min_boost_length=5,
        boost_cost_frames=3,
        frame=np.array([0], dtype=np.int64),
        max_frames=5000,
        arena_type_flag=0.0,
    )


def _assert_observation_bytes(actual: dict, expected: dict, context: str) -> None:
    """Compare every output plane and report a first differing coordinate."""
    for key in ("tactical_uint8", "strategic_uint8", "scalars", "mask"):
        left, right = actual[key], expected[key]
        assert left.dtype == right.dtype, f"{context}: {key} dtype changed"
        if np.array_equal(left, right):
            continue
        mismatch = np.argwhere(left != right)[0]
        index = tuple(int(item) for item in mismatch)
        raise AssertionError(
            f"{context}: {key} differs at {index}: "
            f"candidate={left[index]!r} frozen={right[index]!r}"
        )


def _paint_case(
    paint_fn: Callable[..., None],
    corrected: bool,
    operations: list[dict],
    initial: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Run private paint operations against a fresh small accumulator."""
    dtype = np.int32 if corrected else np.uint16
    code_plane = np.zeros((1, 1, 5, 5), dtype=dtype)
    value_plane = np.zeros((1, 1, 5, 5), dtype=np.uint16)
    if initial is not None:
        code_plane[...] = initial[0]
        value_plane[...] = initial[1]
    _paint_operations(paint_fn, code_plane, value_plane, operations)
    return code_plane, value_plane


def _paint_operations(
    paint_fn: Callable[..., None],
    code_plane: np.ndarray,
    value_plane: np.ndarray,
    operations: list[dict],
) -> None:
    """Apply operations to caller-owned planes, preserving their memory order."""
    for operation in operations:
        paint_fn(code_plane, value_plane, size=5, **operation)


def _direct_paint_operations() -> list[list[dict]]:
    """Return paint calls covering scalar/vector, ties, bounds, and uint16 values."""
    ei = np.zeros(3, dtype=np.int64)
    si = np.zeros(3, dtype=np.int64)
    row = np.full(3, 2, dtype=np.int64)
    col = np.full(3, 2, dtype=np.int64)
    return [
        # Scalar code with vector values: v2's last-write value differs from
        # v3's max-value tie rule, and both paths must match the frozen source.
        [
            {
                "ei": ei,
                "si": si,
                "row": row,
                "col": col,
                "code": 5,
                "value": np.array([7, 9, 3], dtype=np.uint16),
            }
        ],
        # Mixed type codes are sorted by source priority before duplicate-cell
        # scatter.  Corrected-v3 also accepts scalar values for this vector
        # call; legacy v2 intentionally retains its original TypeError here.
        [
            {
                "ei": ei,
                "si": si,
                "row": row,
                "col": col,
                "code": np.array([2, 4, 3], dtype=np.uint16),
                "value": 17,
            }
        ],
        # Equal rank, different values: v3 stores the maximum value, while v2
        # retains its stable input-order last write.
        [
            {
                "ei": np.array([0, 0], dtype=np.int64),
                "si": np.array([0, 0], dtype=np.int64),
                "row": np.array([1, 1], dtype=np.int64),
                "col": np.array([1, 1], dtype=np.int64),
                "code": np.array([6, 6], dtype=np.uint16),
                "value": np.array([12, 200], dtype=np.uint16),
            }
        ],
        # Resident lower/higher rank behavior across calls.  A later lower-rank
        # scalar must not replace a v3 resident; a higher rank must replace it.
        [
            {
                "ei": np.array([0]),
                "si": np.array([0]),
                "row": np.array([3]),
                "col": np.array([3]),
                "code": 6,
                "value": 31,
            },
            {
                "ei": np.array([0]),
                "si": np.array([0]),
                "row": np.array([3]),
                "col": np.array([3]),
                "code": 4,
                "value": 250,
            },
            {
                "ei": np.array([0]),
                "si": np.array([0]),
                "row": np.array([3]),
                "col": np.array([3]),
                "code": 7,
                "value": 42,
            },
        ],
        # Empty code, zero value, and off-grid rows/columns must not write.
        [
            {
                "ei": np.array([0, 0, 0, 0]),
                "si": np.array([0, 0, 0, 0]),
                "row": np.array([-1, 0, 5, 2]),
                "col": np.array([2, 6, 1, 2]),
                "code": np.array([0, 8, 1, 6], dtype=np.uint16),
                "value": np.array([0, 65535, 123, 0], dtype=np.uint16),
            }
        ],
        # Keep the accumulator's uint16 value contract intact at both endpoints.
        [
            {
                "ei": np.array([0, 0]),
                "si": np.array([0, 0]),
                "row": np.array([0, 4]),
                "col": np.array([0, 4]),
                "code": np.array([1, 8], dtype=np.uint16),
                "value": np.array([0, 65535], dtype=np.uint16),
            }
        ],
    ]


def test_candidate_paint_matches_reference_for_direct_edge_cases() -> None:
    """The optimized scatter preserves every private paint edge case in v2/v3."""
    for corrected in (False, True):
        for case_index, operations in enumerate(_direct_paint_operations()):
            if not corrected and case_index == 1:
                # The original v2 vector-code/scalar-value path raises; retain
                # that legacy behavior and exercise this combination in v3.
                continue
            candidate = _paint_case(candidate_featurizer._paint, corrected, operations)
            frozen = _paint_case(reference_paint, corrected, operations)
            for plane_index, (actual, expected) in enumerate(zip(candidate, frozen)):
                np.testing.assert_array_equal(
                    actual,
                    expected,
                    err_msg=f"paint corrected={corrected} case={case_index} plane={plane_index}",
                )

    # A larger duplicate-heavy stream exercises the O(number-of-points) scratch
    # path used by vectorized corrected-v3 paint without creating a world-sized
    # fixture or changing the product implementation.
    count = 4096
    large = [
        {
            "ei": np.zeros(count, dtype=np.int64),
            "si": np.zeros(count, dtype=np.int64),
            "row": np.arange(count, dtype=np.int64) % 5,
            "col": (np.arange(count, dtype=np.int64) * 3) % 5,
            "code": (np.arange(count, dtype=np.uint16) % 9),
            "value": (np.arange(count, dtype=np.uint32) * 257 % 65536).astype(np.uint16),
        }
    ]
    candidate = _paint_case(candidate_featurizer._paint, True, large)
    frozen = _paint_case(reference_paint, True, large)
    for plane_index, (actual, expected) in enumerate(zip(candidate, frozen)):
        np.testing.assert_array_equal(actual, expected, err_msg=f"large-paint plane={plane_index}")


def test_candidate_paint_matches_reference_for_prepopulated_rank_value_cases() -> None:
    """Incoming lower/higher/equal rank and full uint16 values keep source state."""
    point = {
        "ei": np.array([0]),
        "si": np.array([0]),
        "row": np.array([2]),
        "col": np.array([2]),
    }
    incoming = (
        (4, 65535, "lower-rank"),
        (6, 100, "equal-lower-value"),
        (6, 65535, "equal-higher-value"),
        (6, 65535, "equal-value"),
        (7, 0, "higher-rank-zero-value"),
    )
    for corrected in (False, True):
        resident_code = 5 if corrected else 6
        resident = (
            np.full((1, 1, 5, 5), resident_code, dtype=np.int32 if corrected else np.uint16),
            np.full((1, 1, 5, 5), 30000, dtype=np.uint16),
        )
        for code, value, label in incoming:
            operation = [{**point, "code": code, "value": value}]
            candidate = _paint_case(
                candidate_featurizer._paint, corrected, operation, initial=resident
            )
            frozen = _paint_case(reference_paint, corrected, operation, initial=resident)
            for plane_index, (actual, expected) in enumerate(zip(candidate, frozen)):
                np.testing.assert_array_equal(
                    actual,
                    expected,
                    err_msg=f"resident corrected={corrected} case={label} plane={plane_index}",
                )


def _accumulator_for_layout(dtype: np.dtype, layout: str) -> np.ndarray:
    """Allocate a logical 5x5 accumulator with the requested backing layout."""
    if layout == "fortran":
        return np.asfortranarray(np.zeros((1, 1, 5, 5), dtype=dtype))
    if layout == "strided":
        base = np.zeros((1, 1, 5, 10), dtype=dtype)
        return base[..., ::2]
    raise AssertionError(f"unknown accumulator layout {layout!r}")


@pytest.mark.parametrize("layout", ["fortran", "strided"])
def test_candidate_paint_matches_reference_on_non_c_accumulators(layout: str) -> None:
    """Corrected v3 updates F-contiguous and strided planes through C scratch."""
    operation = {
        "ei": np.array([0]),
        "si": np.array([0]),
        "row": np.array([2]),
        "col": np.array([2]),
        "code": 7,
        "value": 65535,
    }
    candidate_code = _accumulator_for_layout(np.dtype(np.int32), layout)
    candidate_value = _accumulator_for_layout(np.dtype(np.uint16), layout)
    reference_code = _accumulator_for_layout(np.dtype(np.int32), layout)
    reference_value = _accumulator_for_layout(np.dtype(np.uint16), layout)
    # Existing rank 1/value 100 is lower than incoming rank 6/value 65535, so
    # this proves an actual update rather than only exercising a no-op copy.
    candidate_code[0, 0, 2, 2] = 1
    candidate_value[0, 0, 2, 2] = 100
    reference_code[0, 0, 2, 2] = 1
    reference_value[0, 0, 2, 2] = 100
    assert not candidate_code.flags.c_contiguous
    assert not candidate_value.flags.c_contiguous
    candidate_featurizer._paint(candidate_code, candidate_value, size=5, **operation)
    reference_paint(reference_code, reference_value, size=5, **operation)
    np.testing.assert_array_equal(candidate_code, reference_code)
    np.testing.assert_array_equal(candidate_value, reference_value)
    assert candidate_code[0, 0, 2, 2] == 6
    assert candidate_value[0, 0, 2, 2] == 65535


@pytest.mark.parametrize("code", [-1, 9])
def test_candidate_paint_rejects_invalid_scalar_code_without_mutation(code: int) -> None:
    """Invalid corrected-v3 scalar codes fail closed before any write."""
    for paint_fn in (reference_paint, candidate_featurizer._paint):
        code_plane = np.zeros((1, 1, 5, 5), dtype=np.int32)
        value_plane = np.zeros((1, 1, 5, 5), dtype=np.uint16)
        code_plane[0, 0, 2, 2] = 5
        value_plane[0, 0, 2, 2] = 123
        before = (code_plane.copy(), value_plane.copy())
        with pytest.raises((KeyError, ValueError)):
            paint_fn(
                code_plane,
                value_plane,
                np.array([0]),
                np.array([0]),
                np.array([2]),
                np.array([2]),
                code,
                65535,
                5,
            )
        np.testing.assert_array_equal(code_plane, before[0])
        np.testing.assert_array_equal(value_plane, before[1])


@pytest.mark.parametrize("codes", [np.array([-1, 0]), np.array([8, 9])])
def test_candidate_paint_rejects_invalid_vector_code_without_mutation(codes: np.ndarray) -> None:
    """A mixed valid/invalid vector is validated before corrected-v3 writes."""
    for paint_fn in (reference_paint, candidate_featurizer._paint):
        code_plane = np.zeros((1, 1, 5, 5), dtype=np.int32)
        value_plane = np.zeros((1, 1, 5, 5), dtype=np.uint16)
        code_plane[0, 0, 2, 2] = 5
        value_plane[0, 0, 2, 2] = 123
        before = (code_plane.copy(), value_plane.copy())
        with pytest.raises((KeyError, ValueError)):
            paint_fn(
                code_plane,
                value_plane,
                np.array([0, 0]),
                np.array([0, 0]),
                np.array([2, 2]),
                np.array([2, 3]),
                codes,
                np.array([65535, 1], dtype=np.uint16),
                5,
            )
        np.testing.assert_array_equal(code_plane, before[0])
        np.testing.assert_array_equal(value_plane, before[1])


def test_candidate_paint_filters_invalid_off_grid_codes_before_validation() -> None:
    """Invalid codes on wholly off-grid points remain a no-op in both paths."""
    operation = {
        "ei": np.array([0, 0]),
        "si": np.array([0, 0]),
        "row": np.array([-1, 5]),
        "col": np.array([2, 2]),
        "code": np.array([-1, 9]),
        "value": np.array([65535, 65535], dtype=np.uint16),
    }
    for paint_fn in (reference_paint, candidate_featurizer._paint):
        code_plane = np.zeros((1, 1, 5, 5), dtype=np.int32)
        value_plane = np.zeros((1, 1, 5, 5), dtype=np.uint16)
        before = (code_plane.copy(), value_plane.copy())
        paint_fn(code_plane, value_plane, size=5, **operation)
        np.testing.assert_array_equal(code_plane, before[0])
        np.testing.assert_array_equal(value_plane, before[1])


@pytest.mark.parametrize("obs_spec", [RASTER31V2, RASTER31V3])
def test_candidate_matches_reference_full_observation_on_maxlen_stress(obs_spec: str) -> None:
    """v2 and v3 bytes match with dead slots, boosts, overlaps, and MAXLEN=150."""
    inp, mask = _stress_inputs()
    candidate = candidate_featurizer.build_observations(inp, mask=mask, obs_spec=obs_spec)
    frozen = reference_build(inp, mask=mask, obs_spec=obs_spec)
    _assert_observation_bytes(candidate, frozen, f"maxlen-stress/{obs_spec}")

    # Explicit witnesses prove that the fixture reached the intended branches.
    tac = candidate["tactical_uint8"]
    assert tac[0, 0, 0, TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL] == 8
    assert tac[0, 0, 1, TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL] == 255
    assert tac[0, 0, 0, TACTICAL_HEAD_ROW - 1, TACTICAL_HEAD_COL + 2] == 4
    assert tac[0, 0, 0, TACTICAL_HEAD_ROW, TACTICAL_HEAD_COL + 2] == 4
    assert np.count_nonzero(tac[0, 3]) == 0
    assert np.count_nonzero(candidate["strategic_uint8"][0, 3]) == 0


def test_candidate_preserves_source_major_tie_behavior() -> None:
    """Same-code duplicate cells retain v2 last-write and v3 max-value ties."""
    inp = _tie_inputs()
    row, col = TACTICAL_HEAD_ROW - 6, TACTICAL_HEAD_COL + 5
    for obs_spec in (RASTER31V2, RASTER31V3):
        mask = np.ones((1, 3, 6), dtype=bool)
        candidate = candidate_featurizer.build_observations(inp, mask=mask, obs_spec=obs_spec)
        frozen = reference_build(inp, mask=mask, obs_spec=obs_spec)
        _assert_observation_bytes(candidate, frozen, f"body-tie/{obs_spec}")
        tac = candidate["tactical_uint8"][0, 0]
        assert tac[0, row, col] == 6  # enemy body
        expected_value = 64 if obs_spec == RASTER31V2 else 128
        assert tac[1, row, col] == expected_value


def test_candidate_preserves_v2_paint_order_and_v3_wall_priority() -> None:
    """A wall collision distinguishes v2 paint order from v3 ranked priority."""
    inp = _wall_order_inputs()
    cell = (TACTICAL_HEAD_ROW + 1, TACTICAL_HEAD_COL)
    mask = np.ones((1, 2, 6), dtype=bool)
    candidate_v2 = candidate_featurizer.build_observations(inp, mask=mask, obs_spec=RASTER31V2)
    frozen_v2 = reference_build(inp, mask=mask, obs_spec=RASTER31V2)
    _assert_observation_bytes(candidate_v2, frozen_v2, "wall-order/raster31v2")
    # v2 paints wall first, then outside food and enemy head overwrite it.
    assert candidate_v2["tactical_uint8"][0, 0, 0, cell[0], cell[1]] == 7
    assert candidate_v2["tactical_uint8"][0, 0, 1, cell[0], cell[1]] == 178

    candidate_v3 = candidate_featurizer.build_observations(inp, mask=mask, obs_spec=RASTER31V3)
    frozen_v3 = reference_build(inp, mask=mask, obs_spec=RASTER31V3)
    _assert_observation_bytes(candidate_v3, frozen_v3, "wall-order/raster31v3")
    # v3 gives the actual wall rank above food, bodies, and heads.
    assert candidate_v3["tactical_uint8"][0, 0, 0, cell[0], cell[1]] == 1
    assert candidate_v3["tactical_uint8"][0, 0, 1, cell[0], cell[1]] == 255


@pytest.mark.parametrize("obs_spec", [RASTER31V2, RASTER31V3])
def test_candidate_and_reference_agree_on_circular_geometry_rejection(obs_spec: str) -> None:
    """Both producers reject circular input until circular raster geometry exists."""
    inp, mask = _stress_inputs()
    circular = ObsInputs(**{**inp.__dict__, "arena_type_flag": 1.0})
    with pytest.raises(ValueError, match="rectangular"):
        reference_build(circular, mask=mask, obs_spec=obs_spec)
    with pytest.raises(ValueError, match="rectangular"):
        candidate_featurizer.build_observations(circular, mask=mask, obs_spec=obs_spec)
