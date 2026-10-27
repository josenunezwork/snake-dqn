from __future__ import annotations

import numpy as np
import pytest

from research.sandbox_a_resident.training_acceptance import (
    distribution_screen_descriptor,
    forced_action_common_state,
    vector_carry_common_state,
    vector_v8_common_state,
)
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def test_forced_actions_match_cpu_discrete_common_state_fixture() -> None:
    assert forced_action_common_state(((1, 1), (4, 1), (1, 0)))["status"] == "PASS"


def test_v8_mask_and_action_match_cpu_on_explicit_state() -> None:
    cfg = BatchSimConfig(num_envs=1, num_snakes=2, game_width=120, game_height=100, max_capacity=32)
    sim = GridBatchSim(cfg, seeds=[9], train_mode=True, allow_respawn=False, jit=False)
    q = np.array(((0.0, 2.0, 1.0, 0.0, 0.0, 0.0), (1.0, 0.0, 2.0, 0.0, 0.0, 0.0)), dtype=np.float32)
    assert vector_v8_common_state(sim, q)["status"] == "PASS"
    assert vector_carry_common_state(sim)["status"] == "PASS"


def test_distribution_screen_is_explicitly_nongating() -> None:
    report = distribution_screen_descriptor()
    assert report["worlds_per_mix"] == 32
    assert "not a gate" in report["status_meaning"]


def test_cpu_reference_uses_real_frozen_roster_short_horizon() -> None:
    import torch

    from research.sandbox_a_resident.training_cpu_reference import CpuRosterTask

    task = CpuRosterTask("frozen", 3, 5000, seed=19)
    assert task.observe()[2].shape == (3, 60)
    assert task.step(torch.ones(3, dtype=torch.long)).mass.shape == (3,)


def test_distribution_screen_records_first_lives_and_gross_support(monkeypatch) -> None:
    """Exercise both complete screen arms with a tiny explicit event tape."""
    from types import SimpleNamespace

    import torch

    from research.sandbox_a_resident import training_acceptance as acceptance

    monkeypatch.setattr(acceptance, "distribution_screen_descriptor", lambda **kw: kw)

    class Task:
        def __init__(self, mix, worlds, horizon):
            self.tick = 0
            bodies = torch.zeros((worlds, 2, 1, 2), dtype=torch.long)
            bodies[:, 0, 0] = torch.tensor([1, 1])
            bodies[:, 1, 0] = torch.tensor([4, 1])
            self.sim = SimpleNamespace(
                bodies=bodies,
                head_ptr=torch.zeros((worlds, 2), dtype=torch.long),
                S=2,
                food_cells=torch.tensor([[[2, 2], [3, 3]]] * worlds),
                food_present=torch.ones((worlds, 2), dtype=torch.bool),
                direction=torch.ones((worlds, 2), dtype=torch.long),
                alive=torch.ones((worlds, 2), dtype=torch.bool),
                _last_food_ate=torch.zeros((worlds, 2), dtype=torch.bool),
                _last_death_cause=torch.zeros((worlds, 2), dtype=torch.long),
            )

        def observe(self):
            return (
                torch.zeros(3, 1),
                torch.zeros(3, 1),
                torch.zeros(3, 1),
                torch.ones(3, 6, dtype=torch.bool),
            )

        def step(self, actions):
            self.tick += 1
            terminal = torch.tensor([self.tick == 1, False, False])
            self.sim.alive[0, 0] = False
            return SimpleNamespace(
                mass=torch.tensor([0.0, 10.0, 10.0]),
                terminal=terminal,
                reset=torch.zeros(3, dtype=torch.bool),
            )

    report = acceptance.run_distribution_screen(
        cpu_factory=Task,
        device_factory=Task,
        cpu_actor=lambda *args: torch.zeros(3, 6),
        device_actor=lambda *args: torch.zeros(3, 6),
        worlds_per_mix=3,
        horizon=4,
    )
    assert report["material_gaps"] == []
    for mix in ("scripted", "frozen", "mixed"):
        row = report["results"][mix]["cpu"]
        assert row["death_events"] == 1
        assert row["no_death_fraction"] == pytest.approx(2 / 3)
        assert row["mean_mass_integral"] == pytest.approx(20 / 3)
        assert row["nearest_spawn_mean_cells"] == 3
