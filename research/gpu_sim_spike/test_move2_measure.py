"""An exact reset must undo persistent inventory/dtype changes from warmup."""

from types import SimpleNamespace

import torch

from research.gpu_sim_spike.move2_measure import restore
from research.gpu_sim_spike.move2_parity import compare


def test_restore_removes_new_fields_and_restores_original_reward_dtype() -> None:
    template = SimpleNamespace(reward=torch.ones(2, dtype=torch.float64), _prepared=False)
    current = SimpleNamespace(
        reward=torch.zeros(2, dtype=torch.float32),
        _prepared=True,
        capacity_overflow=torch.ones(2, dtype=torch.bool),
    )
    restore(current, template)
    assert compare(current, template)["match"]
    assert not hasattr(current, "capacity_overflow")
    current.reward[0] = 9
    assert template.reward[0] == 1
