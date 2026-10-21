"""Hermetic tests for the T35 outcome-head probe (no exports, no checkpoint)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.t35_outcome_head import metrics as M
from research.t35_outcome_head.data import assign_folds, own_continuation, world_targets
from research.t35_outcome_head.features import trunk_forward
from research.t35_outcome_head.model import (
    L2_GRID,
    LITERAL,
    PRIMARY,
    outcome_loss,
    predict,
    train_head,
)
from research.t35_outcome_head.probe import select_l2
from src.model.ego2s_network import Ego2sNet


def _label(d, length, searched=False, surv=(1, 0, 1, 0, 0, 0), a_star=0, decisive=True):
    lab = {
        "decision": d,
        "length": length,
        "student_action": 1,
        "q": [0.0, 1.0, 0.5, 0.0, 0.0, 0.0],
        "legal": [True] * 6,
        "outcomes": [{"survive": False, "alive_steps": 0, "mean_mass": 0.0}] * 6,
        "tested": [False] * 6,
        "decisive": False,
    }
    if searched:
        lab["tested"] = [True] * 6
        lab["outcomes"] = [
            {"survive": bool(s), "alive_steps": 300 if s else 5, "mean_mass": 10.0 * (a + 1)}
            for a, s in enumerate(surv)
        ]
        lab["decisive"] = decisive
        lab["a_star"] = a_star
    return lab


def test_own_continuation_died_world():
    valid, surv, mass = own_continuation([10] * 400, frames=5000, horizon=300)
    assert valid.all()
    # Died during step 399: from t it completed 399 - t steps.
    assert surv[99] == 1.0 and surv[100] == 0.0
    assert mass[0] == pytest.approx(10.0)
    assert mass[399] == pytest.approx(0.0)
    assert mass[299] == pytest.approx(10.0 * 100 / 300)


def test_own_continuation_survived_world_censors_tail():
    valid, surv, mass = own_continuation(list(range(1, 1001)), frames=1000, horizon=300)
    assert valid[:701].all() and not valid[701:].any()
    assert surv[valid].min() == 1.0
    assert mass[0] == pytest.approx(np.mean(np.arange(2, 302)))


def test_world_targets_extracts_teacher_and_own():
    labels = [_label(d, 5) for d in range(10)]
    labels[3] = _label(3, 5, searched=True, a_star=2)
    world = {
        "world_id": "w/0",
        "frames": 10,
        "labels": labels,
        "actions": [[d % 6, 0, 0, 0, 0, 0] for d in range(10)],
    }
    t = world_targets(world)
    assert t.searched.sum() == 1 and t.decisive[3] and t.a_star[3] == 2
    assert t.t_survive[3].tolist() == [1, 0, 1, 0, 0, 0]
    assert t.t_mass[3, 5] == pytest.approx(60.0)
    assert t.executed_action.tolist() == [d % 6 for d in range(10)]
    assert not t.own_valid.any()  # 10-frame world: every 300-step window is censored


def test_world_targets_rejects_gaps():
    world = {"world_id": "w", "frames": 5, "labels": [_label(0, 1), _label(2, 1)], "actions": []}
    with pytest.raises(ValueError):
        world_targets(world)


def test_folds_are_world_level_deterministic_and_balanced():
    ids = [f"w{i}" for i in range(39)]
    has = [i < 21 for i in range(39)]
    a, b = assign_folds(ids, has), assign_folds(list(reversed(ids)), list(reversed(has)))
    assert a == b
    per_fold = [sum(1 for w, s in zip(ids, has) if s and a[w] == k) for k in range(5)]
    assert max(per_fold) - min(per_fold) <= 1


def test_auc_matches_bruteforce_with_ties():
    rng = np.random.default_rng(0)
    s = rng.integers(0, 5, 200).astype(float)
    y = rng.random(200) < 0.4
    brute = np.mean([(sp > sn) + 0.5 * (sp == sn) for sp in s[y] for sn in s[~y]])
    assert M.auc(s, y) == pytest.approx(brute)
    assert np.isnan(M.auc(s, np.ones(200, bool)))


def test_outcome_argmax_mirrors_teacher_rule():
    p = np.array([[0.9, 0.2, 0.7, 0.1, 0.1, 0.1], [0.1, 0.2, 0.3, 0.1, 0.1, 0.4]])
    q = np.array([[0.0, 5.0, 1.0, 9.0, 0, 0], [0.0] * 6])
    legal = np.ones((2, 6), bool)
    assert M.outcome_argmax(p, q, legal).tolist() == [2, 5]
    legal[1, 5] = False
    assert M.outcome_argmax(p, q, legal).tolist() == [2, 2]


def test_verdict_rule():
    assert M.verdict(0.85, 0.79, 0.90, 0.05) == "GO"
    assert M.verdict(0.85, 0.76, 0.90, 0.05) == "INCONCLUSIVE"
    assert M.verdict(0.85, 0.79, 0.90, 0.01) == "INCONCLUSIVE"
    assert M.verdict(0.77, 0.70, 0.95, 0.10) == "KILL"
    assert M.verdict(0.90, 0.85, 0.65, 0.10) == "KILL"
    assert M.verdict(float("nan"), 0.0, 0.9, 0.1) == "ABORT"


def test_loss_ignores_masked_entries():
    s = torch.zeros(2, 6)
    m = torch.zeros(2, 6)
    mask = torch.zeros(2, 6, dtype=torch.bool)
    mask[0, 0] = True
    y1 = torch.zeros(2, 6)
    y2 = y1.clone()
    y2[1] = 1.0
    a = outcome_loss(s, m, y1, y1, mask, 0.1, 1.0)
    b = outcome_loss(s, m, y2, y2 * 5, mask, 0.1, 1.0)
    assert torch.allclose(a, b)


def test_train_head_learns_separable_signal():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(1400, 256)).astype(np.float32)
    y = (x[:, :6] > 0).astype(np.float32)
    mass = x[:, 6:12].copy()
    mask = np.ones((1400, 6), bool)
    cpu = torch.device("cpu")
    head = train_head(x[:1000], y[:1000], mass[:1000], mask[:1000], PRIMARY, 0, cpu, l2=1e-2)
    p, mz = predict(head, x[1000:], cpu)
    assert M.auc(p.ravel(), y[1000:].ravel()) > 0.9
    with pytest.raises(ValueError):
        train_head(x, y, mass, mask, PRIMARY, 0, cpu)  # primary needs inner-CV l2
    train_head(x, y, mass, mask, LITERAL, 0, cpu)  # expert-literal recipe runs


def test_select_l2_uses_only_training_worlds():
    rng = np.random.default_rng(1)
    n = 600
    x = rng.normal(size=(n, 256)).astype(np.float32)
    y = (x[:, :6] > 0).astype(np.float32)
    mask = np.ones((n, 6), bool)
    world_names = np.asarray([f"w{i % 12}" for i in range(n)])
    rows = np.arange(n)
    lam, scores = select_l2(x, y, x[:, 6:12], mask, rows, world_names, torch.device("cpu"))
    assert lam in L2_GRID and set(scores) == set(L2_GRID)


def test_trunk_forward_reproduces_net_q():
    torch.manual_seed(0)
    net = Ego2sNet(7, 60).eval()
    local = torch.randint(0, 256, (3, 7, 31, 31), dtype=torch.uint8)
    glob = torch.randint(0, 256, (3, 4, 37, 37), dtype=torch.uint8)
    sc = torch.randn(3, 60)
    with torch.no_grad():
        h, q = trunk_forward(net, local, glob, sc)
        assert h.shape == (3, 256)
        assert torch.equal(q, net(local, glob, sc))
