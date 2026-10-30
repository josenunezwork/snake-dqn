"""Independent live/GPU observation and standalone spatial actor checks."""

from __future__ import annotations

import random
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.gpu_sim_spike.device_sim import DeviceSim
from research.spatial_student_pilot.model import (
    SpatialStudent,
    choose_actions,
    export_checkpoint,
    load_checkpoint,
)
from research.spatial_student_pilot.observation import (
    build_observation,
    build_observation_cpu_reference,
    build_observation_numpy,
    compact_cat,
    compact_from_live,
    compact_from_sim,
    compact_take,
)
from src.core.game_config import get_config, initialize_config
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.parity import (
    PyRefGame,
    SurvivorPolicy,
    _install_v2_config,
    build_parity_config,
)


def _assert_reference(compact: dict, device: str = "cpu") -> None:
    actual = build_observation(compact, device)
    expected = build_observation_cpu_reference(compact)
    for tensor, array in zip(actual, expected):
        value = tensor.detach().cpu().numpy()
        assert value.dtype == array.dtype
        assert value.tobytes() == array.tobytes()
    for tensor, array in zip(build_observation_numpy(compact), expected):
        assert tensor.numpy().tobytes() == array.tobytes()


def _sim(envs: int = 2, cap: int = 1200) -> GridBatchSim:
    cfg = BatchSimConfig(
        num_envs=envs,
        num_snakes=6,
        game_width=500,
        game_height=300,
        max_capacity=cap,
        initial_food=10,
        max_food=20,
        mechanics_version=1,
    )
    return GridBatchSim(cfg, seeds=list(range(envs)), train_mode=True)


def test_sparse_ring_growth_overlap_and_duplicate_row_order() -> None:
    sim = _sim()
    # A wrapped ring, body > old capacity400, repeated cells, two overlapping
    # owners and outstanding growth. Never infer body order from owner grids.
    n = 997
    for env in range(2):
        for slot in range(6):
            sim.head_ptr[env, slot] = 13 + slot
            sim.seg_count[env, slot] = n - slot
            sim.length[env, slot] = n + 27
            sim.alive[env, slot] = slot != 5
            sim.direction[env, slot] = slot % 4
            sim.boost_frames[env, slot] = slot % 3
            sim.frames_since_food[env, slot] = slot * 17
            for k in range(n - slot):
                sim.bodies[env, slot, (13 + slot - k) % sim.cap] = (k % 48, k // 48 + 1)
    compact = compact_from_sim(sim, [1, 0, 1], hero=2)
    assert compact["body_cells"].dtype == torch.int16
    assert compact["body_cells"].shape == (3, 6, n, 2)
    _assert_reference(compact)
    p, s, heads = build_observation(compact)
    assert torch.equal(heads, torch.tensor([[0, 1]] * 3))
    assert p[0, 0, 1, 0] == 1 and p[0, 1, 1, 0] == 1
    assert p[0, 3].count_nonzero() == 0  # cyclic reordered dead slot 5
    assert s[0, 2] == torch.tensor((n + 27 - (n - 2)) / 10400)
    assert compact["body_cells"].shape[2] < sim.cap


def test_torch_device_sim_and_numpy_sim_independent_reconstruction() -> None:
    sim = _sim()
    for step in range(9):
        sim.step(np.full((2, 6), step % 3, dtype=np.int64))
    device_sim = DeviceSim.from_reference(sim, device="cpu")
    cpu = compact_from_sim(sim, [1, 0, 1], hero=4)
    device = compact_from_sim(device_sim, [1, 0, 1], hero=4)
    for a, b in zip(build_observation(cpu), build_observation(device)):
        assert torch.equal(a, b)
    _assert_reference(device)


@pytest.mark.parametrize("mechanics", [1, 2])
def test_real_snake_food_manager_live_adapter_parity(mechanics: int) -> None:
    cfg = replace(
        build_parity_config(
            num_snakes=6, game_width=500, game_height=300, initial_food=30, max_food=40
        ),
        mechanics_version=mechanics,
        max_capacity=1200,
    )
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(81)
        live = PyRefGame(cfg, allow_respawn=True)
        sim = GridBatchSim(cfg, seeds=[81], train_mode=False, allow_respawn=True)
        policy = SurvivorPolicy(15, 6, boost_prob=0.3)
        masks = np.array(live.action_masks(), dtype=bool)
        checked = 0
        for tick in range(100):
            actions = policy.actions(masks)
            live.step(actions)
            sim.step(actions[None])
            masks = np.array(live.action_masks(), dtype=bool)
            # Scripted live snakes do not execute reward bookkeeping. A learned
            # wrapper must advance hunger, exactly as the pre-existing adapter
            # documents. Synchronize this one wrapper-owned counter only.
            for slot, snake in enumerate(live.snakes):
                snake.frames_since_food = int(sim.frames_since_food[0, slot])
            hero = tick % 6
            left = compact_from_sim(sim, hero=hero)
            right = compact_from_live(live, hero=hero)
            for a, b in zip(build_observation(left), build_observation(right)):
                assert torch.equal(a, b), (tick, hero)
            if tick % 25 == 0:
                _assert_reference(right)
            checked += 1
        assert checked == 100
    finally:
        initialize_config(saved)


def test_sparse_shard_concat_padding_and_selection() -> None:
    sim = _sim()
    early = compact_from_sim(sim, [0])
    for _ in range(8):
        sim.step(np.ones((2, 6), dtype=np.int64))
    later = compact_from_sim(sim, [1])
    joined = compact_cat([early, later])
    selected = compact_take(joined, [1, 0, 1])
    for row, original in ((0, later), (1, early), (2, later)):
        for a, b in zip(
            build_observation(compact_take(selected, [row])), build_observation(original)
        ):
            assert torch.equal(a, b)
    bad = dict(later, grid_w=999)
    with pytest.raises(ValueError, match="metadata"):
        compact_cat([early, bad])


def test_long_body_max_span_10400_and_growth_unsaturated() -> None:
    sim = GridBatchSim(
        BatchSimConfig(
            num_envs=1,
            num_snakes=6,
            max_capacity=10432,
            initial_food=0,
            max_food=0,
            mechanics_version=1,
        ),
        seeds=[7],
    )
    sim.seg_count[0, 0] = 10400
    sim.length[0, 0] = 10420
    sim.head_ptr[0, 0] = 2
    for k in range(10400):
        sim.bodies[0, 0, (2 - k) % sim.cap] = (k % 145, k // 145)
    compact = compact_from_sim(sim)
    _assert_reference(compact)
    p, _, _ = build_observation(compact)
    assert p[0, 6, 0, 0] > 1  # no L800/L1500 clipping
    assert p[0, 6, 0, 1] > p[0, 6, 71, 104]


def test_invalid_body_and_food_rejected_dead_body_ignored() -> None:
    compact = compact_from_sim(_sim())
    compact["body_cells"][0, 0, 0] = torch.tensor([-1, 4])
    with pytest.raises(ValueError, match="outside"):
        build_observation(compact)
    compact["alive"][0, 0] = False
    build_observation(compact)
    compact["food_cells"][0, 0] = torch.tensor([999, 4])
    with pytest.raises(ValueError, match="outside"):
        build_observation(compact)


def test_head_gather_is_not_board_centre() -> None:
    features = torch.arange(2 * 3 * 7 * 11).reshape(2, 3, 7, 11)
    heads = torch.tensor([[0, 6], [10, 0]])
    actual = SpatialStudent.gather_head(features, heads)
    assert torch.equal(actual[0], features[0, :, 6, 0])
    assert torch.equal(actual[1], features[1, :, 0, 10])


def test_actor_train_export_reload_and_legal_masks(tmp_path) -> None:
    torch.set_num_threads(1)
    torch.manual_seed(1901)
    model = SpatialStudent()
    assert sum(p.numel() for p in model.parameters()) == 2299430
    x = torch.randn(2, 16, 17, 25)
    s = torch.randn(2, 36)
    heads = torch.tensor([[1, 15], [23, 2]])
    optim = torch.optim.Adam(model.parameters(), lr=3e-4)
    before = model(x, s, heads).detach()
    loss = torch.nn.functional.cross_entropy(model(x, s, heads), torch.tensor([2, 4]))
    loss.backward()
    optim.step()
    after = model(x, s, heads).detach()
    assert not torch.equal(before, after)
    path = tmp_path / "student.pt"
    export_checkpoint(model, path, metadata={"seed": 1901, "stage": "unit-test"})
    restored, meta = load_checkpoint(path)
    assert meta == {"seed": 1901, "stage": "unit-test"}
    assert torch.equal(after, restored(x, s, heads))
    with pytest.raises(FileExistsError):
        export_checkpoint(model, path)
    logits = torch.tensor([[9.0, 8.0, 7.0, 6.0, 5.0, 4.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    mask = torch.tensor([[0, 0, 1, 0, 1, 0], [1, 1, 0, 0, 0, 0]], dtype=torch.bool)
    assert choose_actions(logits, mask).tolist() == [2, 0]
    with pytest.raises(ValueError, match="no legal"):
        choose_actions(logits, torch.zeros_like(mask))
    with pytest.raises(ValueError, match="nonfinite"):
        choose_actions(logits * float("nan"), mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="real CUDA required")
def test_cuda_observation_scatter_matches_independent_cpu() -> None:
    sim = DeviceSim.from_reference(_sim(), device="cuda")
    compact = compact_from_sim(sim)
    _assert_reference(compact, "cuda")
