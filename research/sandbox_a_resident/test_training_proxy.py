"""Deterministic proxy behavior independent of the resident engine."""

import torch

from research.sandbox_a_resident.training_proxy import (
    MIXES,
    ProxyConfig,
    evaluate,
    masked_argmax,
    paired,
)


class Actor(torch.nn.Module):
    def forward(self, local, glob, scalars):
        return torch.stack(
            (
                scalars[:, 0],
                1 - scalars[:, 0],
                torch.zeros_like(scalars[:, 0]),
                torch.zeros_like(scalars[:, 0]),
                torch.zeros_like(scalars[:, 0]),
                torch.zeros_like(scalars[:, 0]),
            ),
            1,
        )


class Event:
    def __init__(self, mass, terminal):
        self.mass, self.terminal = mass, terminal


class Task:
    def __init__(self, worlds, horizon, seed):
        self.worlds, self.tick, self.seed = worlds, 0, seed
        self.sim = type("Sim", (), {"length": torch.ones((worlds, 1), dtype=torch.long)})()

    def hero_length(self):
        return self.sim.length[:, 0]

    def observe(self):
        x = torch.full((self.worlds,), float(self.tick % 2))
        return (
            torch.zeros((self.worlds, 7, 1, 1), dtype=torch.uint8),
            torch.zeros((self.worlds, 4, 1, 1), dtype=torch.uint8),
            x[:, None],
            torch.tensor([[True, True, True, False, False, False]]).expand(self.worlds, -1),
        )

    def step(self, actions):
        self.tick += 1
        return Event(actions.float() + 1, torch.zeros(self.worlds, dtype=torch.bool))


def test_masked_argmax_and_paired_proxy_are_deterministic() -> None:
    assert (
        masked_argmax(torch.tensor([[1.0] * 6]), torch.zeros((1, 6), dtype=torch.bool)).item() == 0
    )
    seen = []

    def factory(mix, worlds, horizon, seed):
        seen.append((mix, seed))
        return Task(worlds, horizon, seed)

    config = ProxyConfig(horizon=3, worlds_per_mix=2, seed_base=9)
    baseline, candidate = evaluate(Actor(), factory, config), evaluate(Actor(), factory, config)
    assert [mix for mix, _ in seen[:3]] == list(MIXES)
    assert [seed for _, seed in seen[:3]] == [9, 11, 13]
    assert seen[:3] == seen[3:]
    summary = paired(candidate, baseline)
    assert summary["pooled"]["mass_delta"] == 0.0
    assert all(row.boost_rate_under_700 == 0.0 for row in baseline.values())
