import random

import numpy as np
import torch

from research.bl_e16.checkpoint import restore_snapshot


class Collector:
    def __init__(self, value: int) -> None:
        self.bank, self.value = {"bank": 1}, value

    @classmethod
    def restore(cls, bank, state):
        return cls(state["value"])


class Trainer:
    def __init__(self):
        self.model = torch.nn.Linear(1, 1)
        self.actor_optimizer = torch.optim.Adam(self.model.parameters())
        self.critic_optimizer = torch.optim.Adam(self.model.parameters())
        self.refit_transitions, self.e12_passed = 0, False


def test_restore_restores_rng_optimizers_and_collectors() -> None:
    trainer, policy = Trainer(), torch.nn.Linear(1, 1)
    collectors = {"actor": Collector(1)}
    random.seed(4)
    np.random.seed(5)
    torch.manual_seed(6)
    payload = {
        "policy": {"state_dict": policy.state_dict()},
        "trainer": {
            "actor": trainer.actor_optimizer.state_dict(),
            "critic": trainer.critic_optimizer.state_dict(),
            "refit_transitions": 7,
            "e12_passed": True,
        },
        "collector": {"actor": {"value": 9}},
        "counters": {"n": 3},
        "rng": {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
        },
    }
    expected = (random.random(), float(np.random.rand()), float(torch.rand(())))
    restore_snapshot(policy, trainer, collectors, payload)
    assert (random.random(), float(np.random.rand()), float(torch.rand(()))) == expected
    assert collectors["actor"].value == 9 and trainer.refit_transitions == 7 and trainer.e12_passed
