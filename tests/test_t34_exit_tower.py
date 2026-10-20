"""Focused P5 checks: full-sight identity, zero-Q anchor, and stripped export."""

from __future__ import annotations

import random
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from research.t34_exit_b8_fit.full_sight import LiveFullSight, full_sight_from_ego_view
from research.t34_exit_b8_fit.network import (
    ARCH,
    ExitFitTowerNet,
    load_exit_fit_tower_checkpoint,
    save_exit_fit_tower_checkpoint,
)
from research.t34_exit_b8_fit.check_a3 import _checked_delta
from research.t34_exit_b8_fit.serving import ExitFitTowerServingPolicy
from research.t34_exit_b8_fit import serving as tower_serving
from src.core.game_config import get_config, initialize_config
from src.model.ego2s_network import Ego2sNet
from src.simd_env.grid_sim import GridBatchSim
from src.simd_env.parity import PyRefGame, SurvivorPolicy, _install_v2_config, build_parity_config


def _tower() -> ExitFitTowerNet:
    torch.manual_seed(37)
    return ExitFitTowerNet(Ego2sNet(local_channels=7, n_scalars=60))


def _batch(rows: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(9)
    return (
        torch.from_numpy(rng.integers(0, 256, (rows, 7, 31, 31), dtype=np.uint8)),
        torch.from_numpy(rng.integers(0, 256, (rows, 4, 37, 37), dtype=np.uint8)),
        torch.from_numpy(rng.normal(size=(rows, 60)).astype(np.float32)),
        torch.from_numpy(rng.integers(0, 256, (rows, 4, 88, 152), dtype=np.uint8)),
    )


def test_zero_fuse_is_exact_b8_and_first_step_trains_new_fuse() -> None:
    torch.set_num_threads(1)
    net = _tower()
    local, glob, scalars, sight = _batch()
    net.eval()
    with torch.inference_mode():
        assert (
            net(local, glob, scalars, sight).numpy().tobytes()
            == net.base(local, glob, scalars).numpy().tobytes()
        )
    net.train()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)
    optimizer.zero_grad()
    (net(local, glob, scalars, sight).square().mean()).backward()
    assert net.tower_fuse.weight.grad is not None
    assert torch.count_nonzero(net.tower_fuse.weight.grad) > 0
    optimizer.step()
    assert torch.count_nonzero(net.tower_fuse.weight) > 0


def test_full_sight_live_and_sim_planes_are_byte_identical() -> None:
    torch.set_num_threads(1)
    cfg = build_parity_config(
        num_snakes=6, game_width=400, game_height=300, initial_food=40, max_food=50
    )
    saved = get_config()
    _install_v2_config(cfg)
    try:
        random.seed(19)
        live = PyRefGame(cfg, allow_respawn=True)
        sim = GridBatchSim(
            replace(cfg, num_envs=1), seeds=[19], train_mode=False, allow_respawn=True
        )
        policy = SurvivorPolicy(19, cfg.num_snakes, boost_prob=0.3)
        masks = np.asarray(live.action_masks(), dtype=bool)
        checks = 0
        for _ in range(24):
            actions = policy.actions(masks)
            live.step(actions)
            sim.step(actions.reshape(1, cfg.num_snakes))
            masks = np.asarray(live.action_masks(), dtype=bool)
            for hero in np.flatnonzero(sim.get_alive()[0]):
                sim_planes = full_sight_from_ego_view(sim.ego_view(), [[0, int(hero)]]).planes[0]
                live_planes = LiveFullSight.build(live, int(hero))
                assert sim_planes.tobytes() == live_planes.tobytes()
                checks += 1
        assert checks > 0
    finally:
        initialize_config(saved)


def test_export_strips_auxiliary_head_and_requires_explicit_arch(tmp_path) -> None:
    torch.set_num_threads(1)
    path = tmp_path / "candidate.pth"
    save_exit_fit_tower_checkpoint(path, _tower())
    blob = torch.load(path, map_location="cpu", weights_only=False)
    assert blob["arch"] == ARCH and blob["aux_heads_stripped"] is True
    assert not any(key.startswith("free_region_head.") for key in blob["state_dict"])
    restored = load_exit_fit_tower_checkpoint(path)
    assert restored.free_region_head is None


def test_nonzero_fuse_export_reloads_the_same_q(tmp_path) -> None:
    torch.set_num_threads(1)
    net = _tower().train()
    local, glob, scalars, sight = _batch()
    loss = net(local, glob, scalars, sight).square().mean()
    loss.backward()
    torch.optim.SGD(net.parameters(), lr=0.01).step()
    net.eval()
    with torch.inference_mode():
        expected = net(local, glob, scalars, sight).clone()
    path = tmp_path / "trained-candidate.pth"
    save_exit_fit_tower_checkpoint(path, net)
    with torch.inference_mode():
        actual = load_exit_fit_tower_checkpoint(path)(local, glob, scalars, sight)
    assert actual.numpy().tobytes() == expected.numpy().tobytes()


def test_a3_nonfinite_q_is_not_accepted() -> None:
    with np.testing.assert_raises_regex(ValueError, "non-finite"):
        _checked_delta(torch.tensor([[float("nan")]]), torch.zeros((1, 1)))


def test_opt_in_serving_policy_consumes_direct_selection_dispatch(tmp_path) -> None:
    torch.set_num_threads(1)
    path = tmp_path / "candidate.pth"
    save_exit_fit_tower_checkpoint(path, _tower())
    policy = ExitFitTowerServingPolicy(str(path), forward="rowwise")
    policy._frame = 0
    policy._q = np.array([[0, 1, 2, 3, 4, 5], [5, 4, 3, 2, 1, 0]], dtype=np.float32)
    policy._id_to_row = {11: 0, 22: 1}
    policy._dispatch = [11, 22]
    assert policy.select_action(None) == 5
    assert policy.select_action(None) == 0


def test_opt_in_serving_policy_builds_and_dispatches_a_live_style_frame(tmp_path, monkeypatch) -> None:
    """Exercise the policy's live-frame contract without opening a web server."""
    torch.set_num_threads(1)
    path = tmp_path / "candidate.pth"
    save_exit_fit_tower_checkpoint(path, _tower())
    policy = ExitFitTowerServingPolicy(str(path))

    class FakeSnake:
        def __init__(self, snake_id: int) -> None:
            self.id, self.is_alive, self.policy = snake_id, True, policy

    game = SimpleNamespace(frame=4, snakes=[FakeSnake(71), FakeSnake(72)])
    local, glob, scalars, _ = _batch(2)
    obs = {"local": local.numpy(), "global": glob.numpy(), "scalars": scalars.numpy()}
    monkeypatch.setattr("src.game.ai_snake.AISnake", FakeSnake)
    monkeypatch.setattr(tower_serving, "game_state_to_ego_view", lambda _game: object())
    monkeypatch.setattr(tower_serving, "build_ego_raster", lambda *args, **kwargs: obs)
    monkeypatch.setattr(
        tower_serving,
        "full_sight_from_ego_view",
        lambda *args, **kwargs: SimpleNamespace(planes=np.zeros((2, 4, 88, 152), dtype=np.uint8)),
    )
    monkeypatch.setattr("src.simd_env.ego_live_bv8.game_state_to_v8_source", lambda *args, **kwargs: object())
    policy.attach_game(game)
    policy.prepare_frame()
    first, second = policy.q_for(71), policy.q_for(72)
    assert first is not None and second is not None
    assert policy.dqn(torch.zeros(61)).cpu().numpy()[0].tobytes() == first.tobytes()
    assert policy.dqn(torch.zeros(61)).cpu().numpy()[0].tobytes() == second.tobytes()
