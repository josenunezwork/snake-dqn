"""Focused contracts for the isolated E8 monitor."""

from __future__ import annotations

import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from research.bl_ppo.diagnostics import E8Diagnostics
from research.bl_ppo.monitor import (
    FixedOrdinaryBank,
    ProbeRoot,
    run_fixed_bank,
    run_paired_mass_probe,
)


def _raw(n=3):
    return SimpleNamespace(
        local=np.zeros((n, 1, 2, 2), np.float32),
        global_=np.zeros((n, 1, 2, 2), np.float32),
        scalars=np.zeros((n, 2), np.float32),
        mask=np.ones((n, 6), bool),
        reset=np.zeros(n, bool),
        frame=np.arange(n),
        length=np.arange(800, 800 + n),
    )


def test_bank_requires_actual_r_context_and_freezes_deterministic_reservoir() -> None:
    bank = FixedOrdinaryBank(capacity=3, seed=1)
    raw = _raw()
    try:
        bank.offer(
            raw,
            hidden_before=None,
            actor_mode="frozen_initial_stochastic",
            actor_sha256="a",
            context_kind="actual_collector_hidden",
        )
        assert False, "missing R context accepted"
    except ValueError:
        pass
    bank.offer(
        raw,
        hidden_before=np.zeros((1, 3, 256), np.float32),
        actor_mode="frozen_initial_stochastic",
        actor_sha256="a",
        context_kind="actual_collector_hidden",
    )
    first = bank.freeze()
    assert first["selected_rows"] == 3 and len(first["sha256"]) == 64
    assert bank.state_dict()["metadata"] == first


class _Policy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.temperature = 1.0

    def forward(self, local, global_, scalars, hidden=None, reset=None, *, remaining_horizon=None):
        logits = torch.zeros((len(local), 6), device=local.device)
        logits[:, 1] = self.weight
        return logits, torch.zeros(len(local), device=local.device), hidden

    def teacher(self, local, global_, scalars):
        return torch.zeros((len(local), 6), device=local.device)


def test_fixed_bank_uses_stored_context_and_updates_diagnostics() -> None:
    bank = FixedOrdinaryBank(capacity=2)
    bank.offer(
        _raw(2),
        hidden_before=np.zeros((1, 2, 256), np.float32),
        actor_mode="frozen_initial_stochastic",
        actor_sha256="x",
        context_kind="actual_collector_hidden",
    )
    bank.freeze()
    policy = _Policy()
    policy.train()
    result = run_fixed_bank(policy, bank, E8Diagnostics(), batch_size=1)
    assert result["evaluated_rows"] == 2 and policy.training


class _Driver:
    def __init__(self, horizon):
        # Construction represents third-party simulator imports that might
        # otherwise perturb the three training RNG streams.
        random.random()
        np.random.random()
        torch.rand(())
        self.horizon, self.frame = horizon, 0

    def observe_hero(self):
        return {
            "local": np.zeros((1, 1, 2, 2), np.float32),
            "global": np.zeros((1, 1, 2, 2), np.float32),
            "scalars": np.zeros((1, 2), np.float32),
            "mask": np.ones((1, 6), bool),
        }

    def step(self, action):
        self.frame += 1
        return {"mass": float(action + 1), "done": self.frame >= self.horizon}


def test_paired_probe_is_actual_driver_return_and_reports_charge() -> None:
    py_before, np_before, torch_before = (
        random.getstate(),
        np.random.get_state(),
        torch.get_rng_state().clone(),
    )
    out = run_paired_mass_probe(
        _Policy(),
        [ProbeRoot(1, (("scripted", "greedy_food"),), "scripted")],
        lambda seed, roster, horizon: _Driver(horizon),
        horizon=3,
        rng_seed=5,
    )
    assert out["excluded_from_training"] and out["charged_simulator_transitions"] == 6
    assert out["rows"][0]["greedy_mass_integral"] == 2.0
    assert random.getstate() == py_before
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert torch.equal(torch.get_rng_state(), torch_before)


def test_monitor_nonfinite_outputs_preserve_scientific_tripwire() -> None:
    policy = _Policy()
    with torch.no_grad():
        policy.weight.fill_(float("nan"))
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_paired_mass_probe(
            policy,
            [ProbeRoot(1, (("scripted", "greedy_food"),), "scripted")],
            lambda seed, roster, horizon: _Driver(horizon),
            horizon=3,
        )
    bank = FixedOrdinaryBank(capacity=2)
    bank.offer(
        _raw(2),
        hidden_before=None,
        actor_mode="frozen_initial_stochastic",
        actor_sha256="x",
        context_kind="feedforward_no_hidden",
    )
    bank.freeze()
    with pytest.raises(FloatingPointError, match="nonfinite"):
        run_fixed_bank(policy, bank, E8Diagnostics())
