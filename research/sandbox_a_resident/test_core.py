"""Independent tiny CPU correctness and exact-resume proofs (one process/thread)."""

from __future__ import annotations

import copy
import math
from dataclasses import replace

import numpy as np
import pytest
import torch

from research.sandbox_a_resident.core import Recipe, ResidentAssay, distribution, gae, seed64
from src.model.ego2s_network import Ego2sNet


@pytest.fixture(autouse=True)
def cpu_threads() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(17)


def assay(**kwargs: object) -> ResidentAssay:
    recipe = Recipe(worlds=2, horizon=3, minibatch=3, tiny=True, max_episodes=24, **kwargs)
    return ResidentAssay(recipe, Ego2sNet(local_channels=7, n_scalars=60))


def assert_exact(left: object, right: object) -> None:
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert left.dtype == right.dtype and left.shape == right.shape
        assert (
            left.cpu().contiguous().numpy().tobytes() == right.cpu().contiguous().numpy().tobytes()
        )
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            assert_exact(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert_exact(a, b)
    else:
        assert left == right


def test_distribution_logprob_entropy_gradient_and_legality() -> None:
    logits = torch.tensor([[math.log(2), 99, math.log(3), -20, 3, 1]], requires_grad=True)
    mask = torch.tensor([[True, False, True, False, False, False]])
    dist = distribution(logits, mask)
    assert dist.probs[0].tolist() == pytest.approx([0.4, 0, 0.6, 0, 0, 0])
    assert dist.log_prob(torch.tensor([2])).item() == pytest.approx(math.log(0.6))
    assert dist.entropy().item() == pytest.approx(-0.4 * math.log(0.4) - 0.6 * math.log(0.6))
    assert set(dist.sample((1000,)).flatten().tolist()) <= {0, 2}
    (dist.entropy().sum() - dist.log_prob(torch.tensor([0])).sum()).backward()
    assert torch.isfinite(logits.grad).all()
    assert logits.grad[0, ~mask[0]].eq(0).all()
    fallback = distribution(torch.zeros(2, 6), torch.zeros(2, 6, dtype=torch.bool))
    assert fallback.probs[:, :3].eq(1 / 3).all()
    assert fallback.probs[:, 3:].eq(0).all()


def test_gae_matches_independent_numpy_death_truncation_horizon() -> None:
    rng = np.random.default_rng(8)
    rewards, values, next_values = [rng.normal(size=(5, 3)).astype("float32") for _ in range(3)]
    terminal = np.zeros((5, 3), dtype=bool)
    truncated = terminal.copy()
    terminal[1, 0] = True
    truncated[2, 1] = True
    expected = np.empty_like(values)
    for world in range(3):
        carry = np.float32(0)
        for t in reversed(range(5)):
            delta = rewards[t, world] - values[t, world]
            if not terminal[t, world]:
                delta += np.float32(0.99) * next_values[t, world]
            if terminal[t, world] or truncated[t, world]:
                carry = delta
            else:
                carry = delta + np.float32(0.99 * 0.95) * carry
            expected[t, world] = carry
    advantage, returns = gae(
        *(torch.tensor(x) for x in (rewards, values, next_values, terminal, truncated)), 0.99, 0.95
    )
    np.testing.assert_allclose(advantage.numpy(), expected, atol=2e-7, rtol=1e-6)
    np.testing.assert_allclose(returns.numpy(), expected + values, atol=2e-7, rtol=1e-6)


def test_selective_reset_preserves_every_other_world_byte() -> None:
    runner = assay()
    runner.sim.prepare()
    runner.sim.capacity_overflow[:] = True
    before = {k: v.clone() for k, v in runner.sim.__dict__.items() if isinstance(v, torch.Tensor)}
    runner._reset(torch.tensor([True, False]))
    for name, value in before.items():
        assert_exact(value[1], getattr(runner.sim, name)[1])
    assert runner.episode.tolist() == [1, 0]
    assert not runner.sim._prepared
    assert runner.sim.alive.all()
    assert not runner.sim.capacity_overflow[0].any()
    assert runner.sim.capacity_overflow[1].all()


def test_collect_legal_rows_and_update_changes_actor() -> None:
    runner = assay()
    before = copy.deepcopy(runner.actor.state_dict())
    rollout = runner.collect()
    assert rollout["obs"][2].shape == (3, 2, 60)
    from research.sandbox_a_resident.core import allowed_mask

    assert allowed_mask(rollout["mask"]).gather(-1, rollout["action"][..., None]).all()
    result = runner.update(rollout)
    assert torch.isfinite(result["loss"])
    assert runner.cycle == 1
    assert any(not torch.equal(v, runner.actor.state_dict()[k]) for k, v in before.items())


def test_exact_resume_through_collection_reset_and_optimizer(tmp_path: object) -> None:
    runner = assay(episode_limit=2)
    runner.update(runner.collect())
    path = tmp_path / "state.pt"
    runner.save(path)
    expected_rollout = runner.collect()
    runner.update(expected_rollout)
    expected = runner.snapshot()
    restored = assay(episode_limit=2)
    restored.restore(torch.load(path, weights_only=False))
    actual_rollout = restored.collect()
    assert_exact(expected_rollout, actual_rollout)
    restored.update(actual_rollout)
    assert_exact(expected, restored.snapshot())


def test_terminal_and_truncation_have_distinct_bootstrap() -> None:
    runner = assay(episode_limit=1)
    with torch.no_grad():
        runner.actor.critic.bias.fill_(3)
    data = runner.collect()
    assert (data["terminal"] | data["truncated"]).all()
    expected = data["reward"] + 0.99 * data["next_value"] * ~data["terminal"] - data["value"]
    torch.testing.assert_close(data["advantage"], expected)
    assert (data["next_value"][data["truncated"]] == 3).all()


def test_seed_namespace_and_restore_drift_refused() -> None:
    assert seed64("a100-dev/sandbox-a", 0) != seed64("a100-dev/sandbox-a", 1)
    with pytest.raises(ValueError):
        seed64("other-study", 0)
    runner = assay()
    state = runner.snapshot()
    state["recipe"] = {**state["recipe"], "horizon": 4}
    with pytest.raises(ValueError, match="differs"):
        runner.restore(state)
    with pytest.raises(ValueError):
        replace(runner.recipe, seed_namespace="foreign")


def test_cached_full_state_reconstruction_and_next_update(tmp_path: object) -> None:
    from research.sandbox_a_resident.run import package_unit, restore_unit_checkpoint, synthetic

    runner = assay()
    cached = synthetic(runner)
    runner.update(cached)
    path = tmp_path / "cached.tar"
    package_unit(path, runner, {"phase": "cached", "cycles": [0]}, cached)
    runner.update(cached)
    expected = runner.snapshot()
    restored = assay()
    actual_cached = restore_unit_checkpoint(path, restored)
    assert_exact(cached, actual_cached)
    restored.update(actual_cached)
    assert_exact(expected, restored.snapshot())


def test_reset_compacts_spawn_and_never_prepares_unselected_worlds(monkeypatch) -> None:
    from research.gpu_sim_spike.device_sim import DeviceSim

    # E == S used to corrupt slot constants through a shape[0] heuristic.
    runner = ResidentAssay(
        Recipe(worlds=6, horizon=2, tiny=True), Ego2sNet(local_channels=7, n_scalars=60)
    )
    runner.sim.slot_constant = torch.arange(6)
    before = runner.sim.slot_constant.clone()
    spawn_sizes, prepared_sizes = [], []
    original_spawn, original_prepare = DeviceSim._respawn, DeviceSim.prepare

    def spawn(sim):
        spawn_sizes.append(sim.E)
        return original_spawn(sim)

    def prepare(sim):
        prepared_sizes.append(sim.E)
        return original_prepare(sim)

    monkeypatch.setattr(DeviceSim, "_respawn", spawn)
    monkeypatch.setattr(DeviceSim, "prepare", prepare)
    runner._reset(torch.tensor([False, False, True, False, False, False]))
    assert spawn_sizes == [1]
    assert prepared_sizes == []
    assert_exact(before, runner.sim.slot_constant)
    runner.observe()
    assert prepared_sizes == [6]
    assert runner.sim.frame.tolist() == [1] * 6


def test_bootstrap_compacts_only_live_truncations_on_simultaneous_death(monkeypatch) -> None:
    runner = assay(episode_limit=1)
    observed, forwarded = [], []
    original_observe, original_forward = runner.observe, runner.actor.forward

    def observe(sim=None):
        active = runner.sim if sim is None else sim
        assert active.alive[:, 0].all(), "dead hero reached observation"
        observed.append(active.E)
        return original_observe(sim)

    def forward(*obs):
        forwarded.append(obs[0].shape[0])
        return original_forward(*obs)

    def simultaneous(actions):
        runner.sim._last_done.zero_()
        runner.sim._last_done[0, 0] = True
        runner.sim.alive[0, 0] = False
        runner.sim.length[0, 0] = 0
        runner.sim._prepared = False

    monkeypatch.setattr(runner, "observe", observe)
    monkeypatch.setattr(runner.actor, "forward", forward)
    monkeypatch.setattr(runner.sim, "step", simultaneous)
    data = runner.collect()
    assert observed == [2, 1, 2, 1, 2, 1, 2]
    assert forwarded == observed
    assert data["terminal"][:, 0].all()
    assert data["truncated"][:, 1].all()
    assert data["next_value"][:, 0].eq(0).all()


def test_default_time_limit_is_reached_inside_real_horizon(monkeypatch) -> None:
    runner = ResidentAssay(Recipe(worlds=2, tiny=True), Ego2sNet(local_channels=7, n_scalars=60))

    # Isolate the time-limit path from stochastic deaths; use real reset/GAE code.
    def survive(actions):
        runner.sim._last_done.zero_()
        runner.sim._last_reward.zero_()

    def observe(sim=None):
        active = runner.sim if sim is None else sim
        return (torch.zeros(active.E, 1),) * 3, torch.ones(active.E, 6, dtype=torch.bool)

    def forward(obs, *_):
        return torch.zeros(obs.shape[0], 6), torch.full((obs.shape[0],), 3.0)

    monkeypatch.setattr(runner.sim, "step", survive)
    monkeypatch.setattr(runner, "observe", observe)
    monkeypatch.setattr(runner.actor, "forward", forward)
    monkeypatch.setattr(runner.policy, "actions", lambda sim, action: action)
    monkeypatch.setattr(runner.policy, "after_step", lambda sim: None)
    data = runner.collect()
    assert data["truncated"].sum() == 4
    assert data["truncated"][63].all() and data["truncated"][127].all()
    assert data["next_value"][data["truncated"]].eq(3).all()


def test_collect_fault_checks_only_at_boundary_and_do_not_lose_overflow(monkeypatch) -> None:
    runner = assay(episode_limit=1)
    count = []
    original_check, original_step = runner._check_collection_faults, runner.sim.step

    def check():
        count.append(True)
        original_check()

    def overflow(actions):
        original_step(actions)
        runner.sim.capacity_overflow[0, 0] = True

    monkeypatch.setattr(runner, "_check_collection_faults", check)
    monkeypatch.setattr(runner.sim, "step", overflow)
    with pytest.raises(RuntimeError, match="capacity exceeded"):
        runner.collect()
    assert count == [True]
    assert runner.reset_count > 0  # Reset clearing cannot erase the sticky failure.


def test_update_checks_nonfinite_once_after_all_minibatches(monkeypatch) -> None:
    import inspect

    runner = assay()
    data = runner.collect()
    scalar_checks, clips = [], []
    original_bool, original_clip = torch.Tensor.__bool__, torch.nn.utils.clip_grad_norm_

    def tensor_bool(value):
        if inspect.currentframe().f_back.f_code.co_name == "update":
            scalar_checks.append(True)
        return original_bool(value)

    def clip(*args, **kwargs):
        clips.append(kwargs["error_if_nonfinite"])
        return original_clip(*args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "__bool__", tensor_bool)
    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", clip)
    runner.update(data)
    assert scalar_checks == [True]
    assert clips == [False, False]
    # The relaxed inner checks still reject invalid gradients/loss at the boundary.
    data["returns"].fill_(float("nan"))
    with pytest.raises(RuntimeError, match="nonfinite PPO"):
        runner.update(data)
    assert runner.cycle == 1


def test_restore_relocates_cpu_loaded_world_and_counters_to_target_device() -> None:
    runner = assay()
    state = runner.snapshot()
    # Meta provides an actual distinct device on CPU-only test hosts; no CUDA claim.
    state["device"] = "meta"
    runner.device = torch.device("meta")
    runner.restore(state)
    for value in runner.sim.__dict__.values():
        if isinstance(value, torch.Tensor):
            assert value.device.type == "meta"
    for name in ("keys", "episode", "age", "reset_count", "collection_faults"):
        assert getattr(runner, name).device.type == "meta"


def test_reset_matches_frozen_pre_argon_world_bytes_after_prepare() -> None:
    """Compare the compact path with the published full-batch oracle, not a twin helper."""
    import subprocess
    import sys
    import types

    source = subprocess.check_output(
        ["git", "show", "fb5948d2:research/sandbox_a_resident/core.py"], text=True
    )
    reference = types.ModuleType("_sandbox_a_pre_argon")
    sys.modules[reference.__name__] = reference
    try:
        exec(compile(source, "<published pre-Argon core>", "exec"), reference.__dict__)
        b8 = Ego2sNet(local_channels=7, n_scalars=60)
        recipe = dict(worlds=6, horizon=3, minibatch=3, episode_limit=2, tiny=True)
        current = ResidentAssay(Recipe(**recipe), b8)
        oracle = reference.ResidentAssay(reference.Recipe(**recipe), b8)
        # The old reset eagerly prepared; the new path defers it to observation.
        current.observe()
        oracle.observe()
        for mask in (
            torch.tensor([False, False, True, False, False, False]),
            torch.tensor([True, False, False, True, True, False]),
        ):
            current._reset(mask)
            oracle._reset(mask)
            # Align the unchanged live-world prepare phase before comparison.
            oracle.sim._prepared = False
            oracle.sim.prepare()
            # The oracle already prepared only reset rows once during _reset.
            # Restore those rows from the oracle's scratch after advancing live rows.
            for name, value in oracle.scratch.__dict__.items():
                if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == 6:
                    getattr(oracle.sim, name)[mask] = value[mask]
            current.observe()
            for name, value in current.sim.__dict__.items():
                if isinstance(value, torch.Tensor):
                    assert_exact(value, getattr(oracle.sim, name))
            assert_exact(current.episode, oracle.episode)
    finally:
        del sys.modules[reference.__name__]
