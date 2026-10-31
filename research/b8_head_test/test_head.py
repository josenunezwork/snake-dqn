"""Leakage, action-rule and cluster-statistic checks for the exploratory head."""

import numpy as np
import pytest
import torch

from research.b8_head_test.run import choose, decide, fit_fold, folds_for, summarize


def test_positive_legal_and_b8_ties() -> None:
    values = torch.tensor(
        [
            [1.0, 2.0, 2.0, 90.0, 0.0, 0.0],
            [-2.0, -1.0, -3.0, 0.0, 0.0, 0.0],
            [0.0, 4.0, 2.0, 0.0, 0.0, 0.0],
            [2.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    legal = torch.tensor([[True, True, True, False, False, False]] * 4)
    b8 = torch.tensor([2, 0, 0, 0])
    assert choose(values, legal, b8).tolist() == [2, 0, 1, 0]
    values[0, 3] = torch.nan
    assert choose(values, legal, b8)[0] == 2
    values[0, 0] = torch.nan
    with pytest.raises(ValueError, match="nonfinite"):
        choose(values, legal, b8)


def test_world_folds_deterministic_and_disjoint() -> None:
    world = torch.arange(524).repeat_interleave(4)
    folds = folds_for(world)
    assert torch.equal(folds, folds_for(world))
    assert sorted(torch.bincount(folds).tolist()) == [416, 420, 420, 420, 420]
    for f in range(5):
        assert not set(world[folds == f].tolist()) & set(world[folds != f].tolist())


def test_heldout_labels_features_and_illegal_targets_cannot_change_fit() -> None:
    torch.set_num_threads(1)
    rng = torch.Generator().manual_seed(10)
    phi = torch.randn(10, 256, generator=rng)
    targets = torch.randn(10, 6, generator=rng)
    legal = torch.ones(10, 6, dtype=torch.bool)
    legal[:, 5] = False
    targets[:, 5] = torch.nan
    train, test = torch.arange(8), torch.arange(8, 10)
    p1, s1 = fit_fold(phi, targets, legal, train, test, updates=3)
    targets[test] = 1e20
    phi[test] = 500
    _, s2 = fit_fold(phi, targets, legal, train, test, updates=3)
    assert torch.isfinite(p1).all()
    assert all(torch.equal(s1["model"][k], s2["model"][k]) for k in s1["model"])
    assert torch.equal(s1["mean"], s2["mean"])
    assert torch.equal(s1["std"], s2["std"])


def test_cluster_intervals_and_rescue_use_paired_suffixes() -> None:
    # Constant effect in 40 worlds: cluster bootstrap has exact zero-width CIs.
    n = 80
    world = np.repeat(np.arange(40), 2)
    mass = np.ones((n, 6, 4)) * 100
    mass[:, 1] = 110
    mass[:, 2] = 120
    survival = np.ones((n, 6, 4))
    survival[::2, 0] = [0, 0, 0, 1]
    b8, chosen, teacher = np.zeros(n, int), np.ones(n, int), np.ones(n, int) * 2
    row = summarize(world, mass, survival, b8, chosen, teacher)
    assert row["chosen_mass"] == dict(point=10.0, lb90=10.0, ci95=[10.0, 10.0])
    assert row["relative_mass"]["point"] == 0.1
    assert row["teacher_retention"]["point"] == 0.5
    assert row["rescue"]["point"] == 1
    assert row["paired_suffix_rescue"]["point"] == 1
    assert row["nondeath_change"]["point"] == 1
    assert row["chosen_survival"]["point"] == 0.375
    with pytest.raises(ValueError, match="32 parent worlds"):
        summarize(world[:20], mass[:20], survival[:20], b8[:20], chosen[:20], teacher[:20])


def test_go_and_kill_boundaries() -> None:
    late = dict(
        chosen_mass=dict(point=5, lb90=0.01), teacher_mass=dict(point=10), rescue=dict(point=0.35)
    )
    ordinary = dict(relative_mass=dict(lb90=-0.05), chosen_survival=dict(point=-0.01))
    stats = dict(late=late, ordinary=ordinary)
    assert decide(stats)[0] == "GO"
    late["rescue"]["point"] = 0.1999
    assert decide(stats)[0] == "KILL"
    late["rescue"]["point"] = 0.20
    late["chosen_mass"]["lb90"] = 0
    assert decide(stats)[0] == "GRAY"
    late["teacher_mass"]["point"] = 0
    assert decide(stats)[0] == "KILL"
