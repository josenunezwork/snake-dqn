"""Eval-only seed ensemble of ego2s students (development information only, no candidate).

:class:`EnsembleEgo2sPolicy` duck-types :class:`~src.simd_env.ego2s_policy.Ego2sSimdPolicy`
(advisory mask, normal-only fallback, float64 Q, first maximum) but plays K students of the
same obs spec on ONE shared featurizer output:

* ``rule="mean"``: argmax of the float64 mean of the members' Q (``K == 1`` is exactly the
  member's own selection, which the dev-probe identity check pins).
* ``rule="vote"``: each member's own advisory-masked greedy action casts one vote; the most
  voted action wins and ties go to the highest mean Q among the tied actions (then the first).

It also accumulates read-only diagnostics per hero decision (they never change an action):
the hero's length at its last decision (length at death for terminal heroes), decisions at
length >= ``long_len``, and Q-calibration statistics split by length bucket (per-member
scale of masked max Q / centered Q, pairwise correlation, greedy agreement, and how often
the mean-Q, vote and per-member-normalized ("z-mean") rules would pick different actions).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from src.model.ego2s_network import load_ego2s_checkpoint, spec_for
from src.simd_env.ego2s_policy import ego_config_for, forward_q
from src.simd_env.ego_raster import build_ego_raster

RULES = ("mean", "vote")
LONG_LEN = 800
BUCKETS = ("lt800", "ge800")


def masked_first_argmax(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    """``(N,)`` first maximum of ``q`` over ``m`` (float64 ``q``; ``m`` has a True per row)."""
    return np.where(m, q, -np.inf).argmax(axis=1).astype(np.int64)


def selection_mask(sim, rows: np.ndarray) -> np.ndarray:
    """The live advisory mask with the normal-only fallback (as ``Ego2sSimdPolicy``)."""
    m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
    m[~m.any(axis=1), :3] = True
    return m


def vote_actions(greedy: np.ndarray, qmean: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Majority vote of ``(K, N)`` member actions; ties -> highest mean Q, then first action."""
    k, n = greedy.shape
    counts = np.zeros((n, qmean.shape[1]), dtype=np.int64)
    for i in range(k):
        counts[np.arange(n), greedy[i]] += 1
    top = counts == counts.max(axis=1, keepdims=True)
    return masked_first_argmax(qmean, top & m)


def zmean_actions(q: np.ndarray, m: np.ndarray) -> np.ndarray:
    """Argmax of the mean of per-member, per-row standardized masked Q (scale-free rule)."""
    mf = m[None].astype(np.float64)
    cnt = mf.sum(axis=2, keepdims=True)
    mu = (q * mf).sum(axis=2, keepdims=True) / cnt
    var = (((q - mu) ** 2) * mf).sum(axis=2, keepdims=True) / cnt
    z = (q - mu) / np.sqrt(var + 1e-12)
    return masked_first_argmax(z.mean(axis=0), m)


class CalibrationStats:
    """Online Q-calibration / agreement accumulators, one set per length bucket."""

    def __init__(self, k: int) -> None:
        self.k = k
        self.b: Dict[str, Dict[str, Any]] = {}
        for name in BUCKETS:
            self.b[name] = {
                "decisions": 0,
                "unanimous": 0,
                "distinct_sum": 0,
                "mean_ne_vote": 0,
                "mean_ne_zmean": 0,
                "vote_ne_zmean": 0,
                "member_eq_chosen": np.zeros(k, dtype=np.int64),
                "pair_agree": np.zeros((k, k), dtype=np.int64),
                # masked max Q ("value") moments
                "v_s": np.zeros(k),
                "v_ss": np.zeros((k, k)),
                # centered masked Q (advantage-like) moments over (row, valid action)
                "a_n": 0,
                "a_s": np.zeros(k),
                "a_ss": np.zeros((k, k)),
                # masked top-2 gap per member (rows with >= 2 valid actions)
                "gap_s": np.zeros(k),
                "gap_n": 0,
                # cross-member spread of each row's max Q and of the chosen action's Q
                "vspread_s": 0.0,
                "qchosen_spread_s": 0.0,
            }

    def add(
        self,
        q: np.ndarray,
        m: np.ndarray,
        greedy: np.ndarray,
        chosen: np.ndarray,
        a_mean: np.ndarray,
        a_vote: np.ndarray,
        a_z: np.ndarray,
        long_rows: np.ndarray,
    ) -> None:
        for name, sel in (("lt800", ~long_rows), ("ge800", long_rows)):
            if not sel.any():
                continue
            self._add(
                self.b[name],
                q[:, sel],
                m[sel],
                greedy[:, sel],
                chosen[sel],
                a_mean[sel],
                a_vote[sel],
                a_z[sel],
            )

    def _add(self, b, q, m, greedy, chosen, a_mean, a_vote, a_z) -> None:
        k, n, _ = q.shape
        b["decisions"] += n
        b["unanimous"] += int((greedy == greedy[:1]).all(axis=0).sum())
        b["distinct_sum"] += int(sum(len(set(greedy[:, i].tolist())) for i in range(n)))
        b["mean_ne_vote"] += int((a_mean != a_vote).sum())
        b["mean_ne_zmean"] += int((a_mean != a_z).sum())
        b["vote_ne_zmean"] += int((a_vote != a_z).sum())
        b["member_eq_chosen"] += (greedy == chosen[None]).sum(axis=1)
        b["pair_agree"] += (greedy[:, None, :] == greedy[None, :, :]).sum(axis=2)
        qm = np.where(m[None], q, -np.inf)
        v = qm.max(axis=2)  # (k, n)
        b["v_s"] += v.sum(axis=1)
        b["v_ss"] += v @ v.T
        mu = (q * m[None]).sum(axis=2, keepdims=True) / m.sum(axis=1)[None, :, None]
        c = (q - mu)[:, m]  # (k, n_valid)
        b["a_n"] += c.shape[1]
        b["a_s"] += c.sum(axis=1)
        b["a_ss"] += c @ c.T
        multi = m.sum(axis=1) >= 2
        if multi.any():
            srt = np.sort(qm[:, multi], axis=2)
            b["gap_s"] += (srt[:, :, -1] - srt[:, :, -2]).sum(axis=1)
            b["gap_n"] += int(multi.sum())
        b["vspread_s"] += float(v.std(axis=0).sum())
        qc = q[:, np.arange(n), chosen]
        b["qchosen_spread_s"] += float(qc.std(axis=0).sum())

    def summary(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name, b in self.b.items():
            n = b["decisions"]
            if not n:
                out[name] = {"decisions": 0}
                continue
            vm = b["v_s"] / n
            vcov = b["v_ss"] / n - np.outer(vm, vm)
            vsd = np.sqrt(np.clip(np.diag(vcov), 0, None))
            an = max(b["a_n"], 1)
            am = b["a_s"] / an
            acov = b["a_ss"] / an - np.outer(am, am)
            asd = np.sqrt(np.clip(np.diag(acov), 0, None))
            out[name] = {
                "decisions": n,
                "unanimous_frac": b["unanimous"] / n,
                "mean_distinct_actions": b["distinct_sum"] / n,
                "mean_ne_vote_frac": b["mean_ne_vote"] / n,
                "mean_ne_zmean_frac": b["mean_ne_zmean"] / n,
                "vote_ne_zmean_frac": b["vote_ne_zmean"] / n,
                "member_agree_chosen": (b["member_eq_chosen"] / n).tolist(),
                "pair_agree": (b["pair_agree"] / n).tolist(),
                "vmax_mean": vm.tolist(),
                "vmax_sd": vsd.tolist(),
                "vmax_corr": (vcov / np.outer(vsd, vsd).clip(1e-12)).tolist(),
                "adv_sd": asd.tolist(),
                "adv_corr": (acov / np.outer(asd, asd).clip(1e-12)).tolist(),
                "gap_mean": (b["gap_s"] / max(b["gap_n"], 1)).tolist(),
                "vmax_member_spread": b["vspread_s"] / n,
                "qchosen_member_spread": b["qchosen_spread_s"] / n,
            }
        return out


class EnsembleEgo2sPolicy:
    """K ego2s students on one shared observation; advisory-mask greedy, no veto."""

    batched_rows = True

    def __init__(
        self,
        members: Sequence[str],
        rule: str = "mean",
        forward: str = "rowwise",
        mask_mode: str = "advisory",
        device: str = "cpu",
        long_len: int = LONG_LEN,
        calibrate: bool = True,
    ) -> None:
        if rule not in RULES:
            raise ValueError(f"rule must be one of {RULES}, got {rule!r}")
        if mask_mode != "advisory":
            raise ValueError("the ensemble replicates the advisory (no-veto) selection only")
        if not members:
            raise ValueError("at least one member is required")
        self.members = [str(p) for p in members]
        self.nets = [load_ego2s_checkpoint(p, device) for p in self.members]
        specs = {spec_for(n) for n in self.nets}
        if len(specs) != 1:
            raise ValueError(f"members must share one obs spec, got {sorted(specs)}")
        self.cfg = ego_config_for(self.nets[0])
        self.rule = rule
        self.forward = forward
        self.mask_mode = mask_mode
        self.long_len = int(long_len)
        self.calib: Optional[CalibrationStats] = (
            CalibrationStats(len(self.nets)) if calibrate else None
        )
        self.last_len: Dict[int, int] = {}
        self.last_frame: Dict[int, int] = {}
        self.long_decisions: Dict[int, int] = {}
        self.decisions: Dict[int, int] = {}

    def member_q(self, sim, rows: np.ndarray) -> np.ndarray:
        """``(K, N, 6)`` float64 member Q on one shared featurizer output."""
        obs = build_ego_raster(sim, self.cfg, backend="numba", rows=rows)
        return np.stack(
            [forward_q(net, obs, self.forward).numpy().astype(np.float64) for net in self.nets]
        )

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        if not len(rows):
            return np.zeros(0, dtype=np.int64)
        q = self.member_q(sim, rows)
        m = selection_mask(sim, rows)
        qmean = q.mean(axis=0) if len(self.nets) > 1 else q[0]
        a_mean = masked_first_argmax(qmean, m)
        greedy = np.stack([masked_first_argmax(q[i], m) for i in range(len(self.nets))])
        a_vote = vote_actions(greedy, qmean, m)
        chosen = a_mean if self.rule == "mean" else a_vote
        lengths = sim.get_lengths()[rows[:, 0], rows[:, 1]].astype(np.int64)
        frames = np.asarray(sim.frame)[rows[:, 0]]
        for (env, _slot), length, frame in zip(rows.tolist(), lengths.tolist(), frames.tolist()):
            self.last_len[env] = int(length)
            self.last_frame[env] = int(frame)
            self.decisions[env] = self.decisions.get(env, 0) + 1
            if length >= self.long_len:
                self.long_decisions[env] = self.long_decisions.get(env, 0) + 1
        if self.calib is not None:
            self.calib.add(
                q, m, greedy, chosen, a_mean, a_vote, zmean_actions(q, m), lengths >= self.long_len
            )
        return chosen

    def diagnostics(self, seeds: Sequence[int]) -> Dict[str, Any]:
        """Per-world (by seed) last-decision length / frame and decision counts + calibration."""
        per = {
            int(seed): {
                "last_len": self.last_len.get(e),
                "last_frame": self.last_frame.get(e),
                "decisions": self.decisions.get(e, 0),
                "decisions_ge_long": self.long_decisions.get(e, 0),
            }
            for e, seed in enumerate(seeds)
        }
        return {
            "per_world": per,
            "long_len": self.long_len,
            "calibration": self.calib.summary() if self.calib is not None else None,
        }


def members_from_globs(paths: Sequence[str]) -> List[str]:
    """Sorted, de-duplicated member paths (globs allowed)."""
    import glob

    out: List[str] = []
    for p in paths:
        hits = sorted(glob.glob(p)) or [p]
        for h in hits:
            if h not in out:
                out.append(h)
    return out
