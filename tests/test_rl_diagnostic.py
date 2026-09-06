"""Focused contracts for the experimental PQN reward diagnostic."""

import random

import numpy as np
import torch

from src.scripts import rl_diagnostic
from src.simd_env.batch_sim import BatchSim, BatchSimConfig
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def test_diagnostic_control_matches_base_reward_exactly():
    cfg = BatchSimConfig(num_envs=1, num_snakes=2)
    base = BatchSim(cfg, seeds=[3], train_mode=True)
    control = rl_diagnostic.DiagnosticBatchSim(
        cfg, seeds=[3], train_mode=True, mass_reward_coefficient=0.0
    )
    actions = np.array([[1, 1]], dtype=np.int64)
    base.step(actions)
    control.step(actions)
    np.testing.assert_array_equal(control.get_reward(), base.get_reward())


def test_diagnostic_mass_reward_uses_only_poststep_alive_length_after_death():
    sim = rl_diagnostic.DiagnosticBatchSim(
        BatchSimConfig(num_envs=1, num_snakes=2),
        seeds=[3],
        train_mode=True,
        mass_reward_coefficient=0.003,
    )
    sim._last_reward[:] = 1.0
    sim.length[:] = [[10, 20]]
    sim.alive[:] = [[True, False]]  # second slot died on the just-finished step
    np.testing.assert_allclose(sim.get_reward(), [[1.03, 1.0]])


def test_initial_seed_reproduces_pqn_tensor_hash():
    cfg = PQNConfig(num_envs=1, num_snakes=2, rollout_len=1, seed=17)

    def build_hash() -> str:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        return rl_diagnostic._tensor_hash(
            PQNTrainer(cfg, device=torch.device("cpu")).network.state_dict()
        )

    assert build_hash() == build_hash()


def test_source_hashes_include_pqn_trainer():
    assert "src/training/pqn_trainer.py" in rl_diagnostic._source_hashes()
