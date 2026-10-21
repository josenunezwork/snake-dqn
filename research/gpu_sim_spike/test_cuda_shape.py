"""CUDA tensor syntax/shape checks without CUDA hardware or any Colab call."""

from __future__ import annotations

import torch
from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.rollout import dense_decision
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


class _ShapePolicy:
    """Fixed fake observations isolate real simulator/branch CUDA tensor ops."""

    def __init__(self, model: torch.nn.Module) -> None:
        self.model = model

    def evaluate(self, sim: DeviceSim) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n, device = sim.E, sim.device
        q = self.model(
            torch.empty((n, 7, 31, 31), dtype=torch.uint8, device=device),
            torch.empty((n, 4, 37, 37), dtype=torch.uint8, device=device),
            torch.empty((n, 60), device=device),
        )
        return q, torch.ones((n, 6), dtype=torch.bool, device=device), q.argmax(-1)

    def repeat_interleave(self, repeats: int) -> "_ShapePolicy":
        return self

    def actions(self, sim: DeviceSim, hero: torch.Tensor) -> torch.Tensor:
        actions = torch.ones((sim.E, sim.S), dtype=torch.long, device=sim.device)
        actions[:, 0] = hero
        return actions


def test_device_program_meta_step_dense_branches_and_b8_shapes() -> None:
    cfg = BatchSimConfig(
        num_envs=2,
        num_snakes=2,
        game_width=200,
        game_height=200,
        initial_food=0,
        max_food=1,
        max_capacity=8,
    )
    with torch.device("meta"):
        sim = DeviceSim(cfg, device="meta", allow_respawn=True)
        model = Ego2sNet(local_channels=7, n_scalars=60).to("meta").eval()
        result = dense_decision(sim, _ShapePolicy(model), depth=2)
        assert result["action"].shape == (2,)
        assert result["alive_steps"].shape == (2, 6)
        assert result["mass_integral"].shape == (2, 6)
        assert result["q"].shape == (2, 6)
        assert result["q"].device == torch.device("meta")


def test_actual_b8_forward_fake_cuda_shapes_without_hardware() -> None:
    # Full device tensor program is checked on meta above. PyTorch's macOS
    # build cannot index a CUDA FakeTensor (C++ indexing enters a CUDA guard
    # before dispatch); B8 forward has no such indexing and can use fake CUDA.
    with torch.device("meta"):
        model = Ego2sNet(local_channels=7, n_scalars=60).eval()
    with FakeTensorMode() as mode:
        for module in model.modules():
            for name, value in tuple(module._parameters.items()):
                if value is not None:
                    module._parameters[name] = FakeTensor(mode, value, torch.device("cuda:0"))
        with torch.inference_mode():
            result = model(
                torch.empty((12, 7, 31, 31), dtype=torch.uint8, device="cuda:0"),
                torch.empty((12, 4, 37, 37), dtype=torch.uint8, device="cuda:0"),
                torch.empty((12, 60), device="cuda:0"),
            )
        assert result.shape == (12, 6)
        assert result.device == torch.device("cuda:0")


def test_full_resident_policy_meta_dense_loop_shapes() -> None:
    """Exercise the real v8, ego, vector, scripted and simulator tensor paths."""
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
    from src.simd_env.vector61_featurizer import Vector61Params

    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=3,
        game_width=160,
        game_height=120,
        initial_food=0,
        max_food=2,
        max_capacity=16,
    )
    with torch.device("meta"):
        sim = DeviceSim(cfg, device="meta", allow_respawn=True)
        runtime = DeviceVectorRuntime(Vector61Params.from_game_config(cfg))
        runtime.bind(sim)
        policy = DevicePolicy(
            torch.device("meta"),
            Ego2sNet(local_channels=7, n_scalars=60).eval(),
            opponent_networks=(torch.nn.Linear(61, 6), None),
            anchor_kinds=(None, "random_safe"),
            anchor_tables=(None, torch.empty((5000, 3), dtype=torch.long, device="meta")),
            vector_runtime=runtime,
            vector_prepared=False,
        )
        result = dense_decision(sim, policy, depth=2)
        assert result["q"].shape == (1, 6)
        assert result["action_trace"].shape == (1, 6, 2, 3)
        assert result["mass_integral"].shape == (1, 6)
