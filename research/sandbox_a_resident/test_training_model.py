"""Focused CPU fixtures for the independent actor/critic/export package."""

from __future__ import annotations

import copy

import pytest
import torch

from research.sandbox_a_resident.training_model import B8Actor, ClockedPopArtCritic, categorical
from src.model.ego2s_network import Ego2sNet, load_ego2s_checkpoint
from web.backend.ego2s_policy import Ego2sServingPolicy


def b8() -> Ego2sNet:
    torch.manual_seed(61)
    return Ego2sNet(local_channels=7, n_scalars=60)


def observation(n: int = 3) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randint(0, 256, (n, 7, 31, 31), dtype=torch.uint8),
        torch.randint(0, 256, (n, 4, 37, 37), dtype=torch.uint8),
        torch.rand(n, 60),
    )


def test_actor_is_full_b8_advantage_temperature_and_reference_is_frozen() -> None:
    torch.set_num_threads(1)
    source = b8()
    actor = B8Actor(b8_model=source, temperature=1.7)
    local, glob, scalars = observation()
    with torch.no_grad():
        expected = (
            source.adv(
                actor.reference.fuse(
                    torch.cat(
                        (
                            actor.reference.local(local.float() / 255),
                            actor.reference.glob(glob.float() / 255),
                            scalars,
                        ),
                        1,
                    )
                )
            )
            / 1.7
        )
    torch.testing.assert_close(actor(local, glob, scalars), expected)
    torch.testing.assert_close(actor.reference_logits(local, glob, scalars), expected)
    assert all(not parameter.requires_grad for parameter in actor.reference.parameters())
    assert any(parameter.requires_grad for parameter in actor.local.parameters())


def test_export_round_trip_served_loader_preserves_logit_differences_and_masked_actions(
    tmp_path: object,
) -> None:
    torch.set_num_threads(1)
    actor = B8Actor(b8_model=b8(), temperature=1.3)
    local, glob, scalars = observation(5)
    masks = torch.eye(6, dtype=torch.bool)
    masks = torch.cat((masks, torch.zeros(1, 6, dtype=torch.bool)))
    local, glob, scalars = (
        torch.cat((value, value[:2]), dim=0) for value in (local, glob, scalars)
    )
    # Verify an untouched B8-initialized actor as well as a trained actor.
    initial = tmp_path / "ppo-initial.pth"
    actor.export_ego2s(initial)
    initial_served = Ego2sServingPolicy(str(initial), torch.device("cpu"))
    torch.testing.assert_close(
        initial_served.net(local, glob, scalars),
        load_ego2s_checkpoint(str(initial))(local, glob, scalars),
    )
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    loss = actor(local, glob, scalars).square().mean()
    loss.backward()
    optimizer.step()  # Exercise a noninitial export too.
    path = tmp_path / "ppo-export.pth"
    actor.export_ego2s(path, {"fixture": "round-trip"})
    served = Ego2sServingPolicy(str(path), torch.device("cpu"))
    logits, q = actor(local, glob, scalars), served.net(local, glob, scalars)
    torch.testing.assert_close(logits - logits.mean(-1, keepdim=True), q - q.mean(-1, keepdim=True))
    assert torch.equal(
        categorical(logits, masks).probs.argmax(-1), categorical(q, masks).probs.argmax(-1)
    )
    # Export changes only the per-row additive dueling constant, so margins
    # (including small cross-platform tie diagnostics) are lossless.
    torch.testing.assert_close(
        torch.topk(logits, 2).values.diff(dim=-1), torch.topk(q, 2).values.diff(dim=-1)
    )


def test_popart_preserves_outputs_and_maps_adam_moments() -> None:
    torch.set_num_threads(1)
    critic = ClockedPopArtCritic(b8(), beta=0.5)
    local, glob, scalars = observation(4)
    clock = torch.tensor([1.0, 0.8, 0.4, 0.0])
    optimizer = torch.optim.Adam(critic.parameters(), lr=1e-3)
    critic(local, glob, scalars, clock).sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before = critic(local, glob, scalars, clock).detach()
    old_scale = critic.scale.detach().clone()
    state_before = {
        parameter: {
            key: value.clone()
            for key, value in optimizer.state[parameter].items()
            if key in ("exp_avg", "exp_avg_sq")
        }
        for parameter in critic.output.parameters()
    }
    critic.update_targets(torch.tensor([100.0, 120.0, 180.0, 220.0]), optimizer)
    after = critic(local, glob, scalars, clock).detach()
    # The affine identity is exact mathematically; float32 GEMM accumulation
    # after a large target-scale change is expected to differ by a few ulps.
    torch.testing.assert_close(before, after, rtol=5e-5, atol=5e-6)
    coordinate = old_scale / critic.scale
    for parameter in critic.output.parameters():
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg"], state_before[parameter]["exp_avg"] * coordinate
        )
        torch.testing.assert_close(
            optimizer.state[parameter]["exp_avg_sq"],
            state_before[parameter]["exp_avg_sq"] * coordinate.square(),
        )
    expected_mean = torch.tensor(77.5)  # beta=.5 and targets have mean 155
    expected_second = 0.5 + 0.5 * torch.tensor([100.0, 120.0, 180.0, 220.0]).square().mean()
    torch.testing.assert_close(critic.mean, expected_mean)
    torch.testing.assert_close(critic.second_moment, expected_second)
    torch.testing.assert_close(critic.scale.square(), critic.second_moment - critic.mean.square())


def test_masked_categorical_falls_back_to_normal_actions() -> None:
    dist = categorical(torch.zeros(2, 6), torch.zeros(2, 6, dtype=torch.bool))
    assert dist.probs[:, :3].eq(1 / 3).all()
    assert dist.probs[:, 3:].eq(0).all()


def test_same_device_compiled_actor_matches_eager_logits_and_gradients() -> None:
    torch.set_num_threads(1)
    eager = B8Actor(b8_model=b8())
    compiled = torch.compile(copy.deepcopy(eager), backend="eager")
    local, glob, scalars = observation(2)
    eager_output = eager(local, glob, scalars)
    compiled_output = compiled(local, glob, scalars)
    torch.testing.assert_close(eager_output, compiled_output)
    eager_output.square().mean().backward()
    compiled_output.square().mean().backward()
    compiled_module = compiled._orig_mod
    for eager_parameter, compiled_parameter in zip(
        eager.parameters(), compiled_module.parameters()
    ):
        if eager_parameter.requires_grad:
            torch.testing.assert_close(eager_parameter.grad, compiled_parameter.grad)


def test_export_preserves_rng_and_temperature_state_roundtrip(tmp_path: object) -> None:
    torch.set_num_threads(1)
    actor = B8Actor(b8_model=b8(), temperature=1.7)
    local, glob, scalars = observation(2)
    torch.manual_seed(404)
    before = torch.get_rng_state().clone()
    actor.export_ego2s(tmp_path / "export.pth")
    assert torch.equal(before, torch.get_rng_state())
    restored = B8Actor(b8_model=b8(), temperature=1.0)
    restored.load_state_dict(actor.state_dict())
    assert float(restored.temperature) == pytest.approx(1.7)
    torch.testing.assert_close(
        restored.reference_logits(local, glob, scalars),
        actor.reference_logits(local, glob, scalars),
    )
