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
    assert runner.sim._prepared
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
