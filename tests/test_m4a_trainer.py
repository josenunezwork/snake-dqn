"""M4-A trainer options (research/redesign_m3c_20261008/train_m3c.py)."""

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008 import train_m3c as T
from tests.t14_fixtures import evaluator_context


def test_shield_prob_schedule():
    assert T.shield_prob(0, 8, 16) == 1.0
    assert T.shield_prob(8, 8, 16) == 1.0
    assert T.shield_prob(12, 8, 16) == pytest.approx(0.5)
    assert T.shield_prob(16, 8, 16) == 0.0
    assert T.shield_prob(99, 8, 16) == 0.0


def test_gamma_schedule_uses_m4a_endpoints():
    args = T.build_parser().parse_args(
        ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x", "--gamma-end", "0.998"]
    )
    sched = T.schedules_from_args(args)
    assert sched.gamma(0) == 0.99
    assert sched.gamma(10**9) == 0.998


def test_margin_anchor_zero_when_label_leads_by_margin():
    q = torch.tensor([[2.0, 0.0, 0.5, -1.0, -1.0, -1.0]])
    a = torch.tensor([0])
    mask = torch.tensor([[True, True, True, False, False, False]])
    assert float(T.anchor_loss_margin(torch, q, a, mask, margin=1.0)) == pytest.approx(0.0)
    # label not ahead: loss = (best other + margin) - q[label]
    q2 = torch.tensor([[0.0, 1.0, 0.5, 9.0, 9.0, 9.0]])
    assert float(T.anchor_loss_margin(torch, q2, a, mask, margin=0.5)) == pytest.approx(1.5)


class _Ring:
    def __init__(self, valid_from):
        self.valid_from = valid_from

    def valid(self, ids):
        return ids >= self.valid_from


def test_label_buffer_wraps_and_skips_overwritten_rows():
    buf = T.LabelBuffer(4)
    buf.add(np.arange(6, dtype=np.int64), np.arange(6, dtype=np.int64) % 6)
    assert buf.size == 4
    assert sorted(buf.ids.tolist()) == [2, 3, 4, 5]
    idx = buf.sample(np.random.default_rng(0), 50, _Ring(valid_from=4))
    assert len(idx) and set(buf.ids[idx].tolist()) <= {4, 5}
    assert np.array_equal(buf.a[idx], buf.ids[idx] % 6)


def _main_args(*extra):
    return ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x", *extra]


@pytest.mark.parametrize(
    "extra",
    [
        ["--shield", "v8", "--anchor-mode", "v8margin"],
        ["--kill-scale", "0"],
        ["--gamma-end", "0.998"],
    ],
)
def test_m4a_options_require_actors(extra):
    with pytest.raises(SystemExit, match="multi-actor"):
        T.main(_main_args(*extra))


def test_onpolicy_labels_require_margin_anchor():
    with pytest.raises(SystemExit, match="v8margin"):
        T.main(_main_args("--actors", "1", "--onpolicy-labels"))


def test_m4a_flags_recorded_only_when_set():
    d = T.default_args()
    assert T.m3c_flags(d) == {}
    args = T.build_parser().parse_args(
        _main_args("--kill-scale", "0", "--shield", "v8", "--anchor-mode", "v8margin")
    )
    flags = T.m3c_flags(args)
    assert flags["kill_scale"] == 0.0 and flags["shield"] == "v8"
    assert flags["anchor_mode"] == "v8margin"


def test_hero_selection_mask_equals_vector61_runtime_selection(monkeypatch, tmp_path):
    from types import SimpleNamespace

    from research.apex_safety_20260926 import dev_screen

    ctx, _ = evaluator_context(tmp_path)
    monkeypatch.setattr(dev_screen, "_design_rows", ctx["ds"]._design_rows)
    args = SimpleNamespace(
        worlds=8,
        seed=0,
        frozen_frac=0.25,
        mixed_frac=0.25,
        namespace=T.TRAIN_NAMESPACE,
        kill_scale=None,
        death_value=None,
    )
    sim, rt, pol, groups = T.build_segment(args, ctx, 0, 0.99, hero_v61=True)
    rng = np.random.default_rng(0)
    checked = []

    def choose(p):
        masks, alive = p.get_resolved_action_mask(), p.get_alive()
        acts = np.ones((sim.E, p.S), np.int64)
        rt.prepare(p)
        hero = np.argwhere(alive[:, :1])
        if len(hero):
            _, sel = rt.selection(hero)
            assert np.array_equal(sel, T.hero_selection_mask(p, hero))
            checked.append(len(hero))
            m = masks[hero[:, 0], 0].copy()
            m[~m.any(1), :3] = True
            acts[hero[:, 0], 0] = [rng.choice(np.flatnonzero(r)) for r in m]
        for spec, cells in groups.items():
            live = cells[alive[cells[:, 0], cells[:, 1]]]
            if len(live):
                mk = masks[live[:, 0], live[:, 1]]
                acts[live[:, 0], live[:, 1]] = pol[spec].actions(mk, p, live)
        return acts

    for _ in range(200):
        sim.step_with_policy(choose)
        rt.observe_step(sim)
    assert sum(checked) > 1000
