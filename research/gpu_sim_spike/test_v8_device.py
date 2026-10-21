"""Focused CPU/CUDA-shape parity checks for the device v8 port.

The CPU reference is intentionally the frozen public v8 entry point; this
test does not share implementation helpers with the candidate.
"""

from __future__ import annotations

import numpy as np
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.v8_device import v8_decide
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.simd_env.v8_nb import v8_decide as cpu_v8_decide


def _fixture() -> BatchSim:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=160,
        game_height=120,
        initial_food=0,
        max_food=0,
        min_boost_length=5,
        boost_length_cost_frames=3,
        mechanics_version=2,
    )
    sim = BatchSim(cfg, seeds=[19], train_mode=False)
    sim.alive[:, :] = True
    # Ring bodies are deliberately coiled enough to exercise post-move release;
    # slots 1/2 put head-reach candidates around the controlled row.
    bodies = (
        ((7, 6), (6, 6), (5, 6), (5, 7), (5, 8), (6, 8), (7, 8), (7, 7)),
        ((9, 6), (10, 6), (11, 6), (11, 7), (10, 7), (9, 7)),
        ((7, 4), (7, 3), (6, 3), (6, 4), (6, 5)),
    )
    for s, cells in enumerate(bodies):
        sim.head_ptr[0, s] = len(cells) - 1
        sim.seg_count[0, s] = len(cells)
        sim.length[0, s] = len(cells)
        sim.direction[0, s] = (1, 3, 2)[s]
        for k, cell in enumerate(cells):
            sim.bodies[0, s, len(cells) - 1 - k] = cell
    return sim


def _self_overlap_fixture() -> BatchSim:
    """A straight boost lands on a retained own-body cell (v5 landing = 0)."""
    sim = _fixture()
    cells = ((7, 6), (6, 6), (5, 6), (5, 7), (6, 7), (7, 7), (8, 7), (8, 6), (9, 6))
    sim.head_ptr[0, 0] = len(cells) - 1
    sim.seg_count[0, 0] = len(cells)
    sim.length[0, 0] = len(cells)
    sim.direction[0, 0] = 1
    for k, cell in enumerate(cells):
        sim.bodies[0, 0, len(cells) - 1 - k] = cell
    return sim


def _assert_same(sim: BatchSim, q_np: np.ndarray, mask_np: np.ndarray) -> None:
    rows_np = np.array([[0, 0]], dtype=np.int64)
    ref = cpu_v8_decide(sim, rows_np, q_np, mask_np, features=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds = DeviceSim.from_reference(sim, device=device)
    got = v8_decide(
        ds,
        torch.as_tensor(rows_np, device=device),
        torch.as_tensor(q_np, device=device),
        torch.as_tensor(mask_np, device=device),
        features=True,
    )
    for name in (
        "action",
        "base",
        "v5_action",
        "v7_action",
        "outcome",
        "v5_reason",
        "counts",
        "cap",
        "need",
        "area_cap",
        "area",
        "landing",
        "head_risky",
        "head_waived",
        "v5_eligible",
    ):
        assert np.array_equal(getattr(got, name).detach().cpu().numpy(), getattr(ref, name)), name
    assert got.feature_planes().shape == (1, 6, 5)


def test_device_v8_matches_reference_fixture_and_cuda_shapes() -> None:
    sim = _fixture()
    # Base boost, constrained masks, all-illegal normal fallback and exact Q
    # ties cover the frozen base/v5/v7 tie and eligibility branches.
    for q, mask in (
        (np.array([[0.7, 0.4, 0.6, 0.8, 0.2, 0.5]], np.float32), np.ones((1, 6), bool)),
        (
            np.array([[0.0, 1.0, 0.0, 0.0, 9.0, 0.0]], np.float32),
            np.array([[True, False, True, True, False, True]]),
        ),
        (np.ones((1, 6), np.float32), np.zeros((1, 6), bool)),
    ):
        _assert_same(sim, q, mask)


def test_device_v8_retained_body_landing_and_same_direction_mask_parity() -> None:
    sim = _self_overlap_fixture()
    q = np.array([[0.0, 0.0, 0.0, 0.0, 9.0, 0.0]], dtype=np.float32)
    # The high-Q straight boost's normal replacement is illegal.  The CPU must
    # keep its retained-body landing blocked rather than clearing the whole
    # head coordinate from the blocker map.
    mask = np.array([[True, False, True, True, True, True]])
    ref = cpu_v8_decide(sim, np.array([[0, 0]], dtype=np.int64), q, mask, features=True)
    assert ref.landing[0, 1] == 0
    _assert_same(sim, q, mask)
