"""Eval-only ego2s seed ensemble (research/ensemble_20261014): selection-rule invariants."""

import numpy as np
import torch

from research.ensemble_20261014.ensemble_policy import (
    EnsembleEgo2sPolicy,
    masked_first_argmax,
    vote_actions,
    zmean_actions,
)
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.ego2s_policy import Ego2sSimdPolicy
from src.simd_env.grid_sim import GridBatchSim


def _ckpt(tmp_path, seed, name):
    torch.manual_seed(seed)
    path = tmp_path / name
    save_ego2s_checkpoint(path, Ego2sNet(local_channels=7, n_scalars=60))
    return str(path)


def _play(policies, frames=25):
    cfg = BatchSimConfig(num_envs=3, num_snakes=4, mechanics_version=2)
    sim = GridBatchSim(cfg, seeds=[1, 2, 3])
    for _ in range(frames):
        masks = sim.get_resolved_action_mask()
        rows = np.argwhere(sim.get_alive())
        m = masks[rows[:, 0], rows[:, 1]]
        outs = [p.actions(m, sim, rows) for p in policies]
        yield sim, rows, outs
        acts = np.ones((sim.E, sim.S), dtype=np.int64)
        acts[rows[:, 0], rows[:, 1]] = outs[0]
        sim.step(acts)


def test_single_member_ensemble_is_the_member(tmp_path):
    a = _ckpt(tmp_path, 0, "a.pth")
    ref = Ego2sSimdPolicy(a, forward="rowwise", mask_mode="advisory")
    mean1 = EnsembleEgo2sPolicy([a], rule="mean")
    vote1 = EnsembleEgo2sPolicy([a], rule="vote")
    dup = EnsembleEgo2sPolicy([a, a, a], rule="mean")
    for _, _, (r, m1, v1, d) in _play([ref, mean1, vote1, dup]):
        assert np.array_equal(r, m1)
        assert np.array_equal(r, v1)
        assert np.array_equal(r, d)
    cal = mean1.calib.summary()
    n = sum(b["decisions"] for b in cal.values())
    assert n > 0
    for b in cal.values():
        if b["decisions"]:
            assert b["unanimous_frac"] == 1.0 and b["mean_ne_vote_frac"] == 0.0


def test_mean_rule_is_argmax_of_mean_q_inside_the_advisory_mask(tmp_path):
    paths = [_ckpt(tmp_path, s, f"m{s}.pth") for s in range(3)]
    ens = EnsembleEgo2sPolicy(paths, rule="mean")
    for sim, rows, (acts,) in _play([ens], frames=10):
        q = ens.member_q(sim, rows)
        m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
        m[~m.any(axis=1), :3] = True
        assert np.array_equal(acts, masked_first_argmax(q.mean(axis=0), m))
    diag = ens.diagnostics([1, 2, 3])
    assert set(diag["per_world"]) == {1, 2, 3}
    assert all(w["last_len"] is not None for w in diag["per_world"].values())


def test_vote_majority_and_tie_break():
    m = np.ones((2, 6), dtype=bool)
    qmean = np.array([[0, 0, 5, 0, 0, 0], [0, 3, 1, 0, 0, 0]], dtype=np.float64)
    # Row 0: votes 0,0,2 -> 0 wins despite mean Q favouring 2.
    # Row 1: votes 1,2 tie (plus one for 2 masked out below) -> highest mean Q among tied.
    greedy = np.array([[0, 1], [0, 2], [2, 4]])
    m[1, 4] = False
    out = vote_actions(greedy, qmean, m)
    assert out.tolist() == [0, 1]


def test_zmean_is_scale_free():
    rng = np.random.default_rng(0)
    q = rng.normal(size=(3, 5, 6))
    m = np.ones((5, 6), dtype=bool)
    m[0, 3:] = False
    scaled = q * np.array([1.0, 10.0, 0.1])[:, None, None] + 7.0
    assert np.array_equal(zmean_actions(q, m), zmean_actions(scaled, m))


def test_calibration_pooled_weights_mixes_by_decisions():
    from research.ensemble_20261014.analyze import calibration_pooled

    def bucket(n, vmax, adv, agree):
        return {
            "decisions": n,
            "vmax_mean": vmax,
            "adv_sd": adv,
            "member_agree_chosen": agree,
            "pair_agree": [[1.0, agree[0]], [agree[0], 1.0]],
            "vmax_corr": [[1.0, 0.9], [0.9, 1.0]],
            "adv_corr": [[1.0, 0.95], [0.95, 1.0]],
            "gap_mean": [0.1, 0.3],
            "vmax_member_spread": 0.5,
            "unanimous_frac": agree[0],
            "mean_ne_vote_frac": 0.0,
            "mean_ne_zmean_frac": 0.01,
        }

    cal = {
        "frozen": {"lt800": bucket(1, [1.0, 2.0], [1.0, 1.0], [1.0, 0.5]), "ge800": {}},
        "mixed": {"lt800": bucket(3, [3.0, 2.0], [1.0, 2.0], [0.5, 0.5])},
    }
    out = calibration_pooled(cal)
    assert set(out) == {"lt800"}
    c = out["lt800"]
    assert c["decisions"] == 4
    assert c["vmax_mean_per_member"] == [2.5, 2.0]
    assert c["adv_sd_max_over_min"] == 1.75
    assert c["chosen_ne_member_frac"] == [0.375, 0.5]
    assert abs(c["top2_gap_mean"] - 0.2) < 1e-12
