#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC (M7 study A): do long-horizon escape features predict death better?

Reads the per-decision captures of ``collect.py`` (5 B8 finals + 4 M6 CTRL finals on the
32 census worlds) and fits held-out death predictors at hero length ``L >= 800``.

Pre-set rule (written before any fit; threshold from the M6 lever analysis):

* Label: the hero dies by ``self`` or ``enemy_body`` within the next ``H`` decisions
  (``H`` in 50 / 100 / 300; the fatal decision counts as within). Decisions followed by a
  head-on / wall death within ``H`` are dropped, as are survivors' decisions whose window
  runs past the horizon (censored).
* Rows: every decision with ``L >= 800`` (all arms pooled), thinned to every
  ``--stride``-th frame (consecutive decisions are near duplicates).
* Feature sets: (i) EXISTING = what the student already observes that is scalar: its 60
  ``ego2s-bv8`` scalars (incl. the 30 v8 per-action features and the ego2s-b region
  scalars), 10 summaries of its local / global planes, and simple aggregates of those
  (best advisory-safe area_g, taken-action area_g and region, safe-action count, the
  ``tight`` flag); (ii) EXISTING + CANDIDATES (families A time-aware flood, B frontier
  vacate times, C global region structure, D body shape, H history / trends; see
  ``escape_nb`` and :func:`history_features`).
* Models: 5-fold cross-validation split by WORLD (all arms of a world in one fold);
  logistic regression (standardized, L2, C = 0.1) and histogram gradient boosting. PRIMARY model:
  gradient boosting. AUC of the out-of-fold predictions, CIs by bootstrap over worlds
  (2000 resamples, the same resample for both sets so the difference is paired).
* Verdict: ``d_H = AUC_ii - AUC_i`` (primary model). GO if ``max(d_100, d_300) >= 0.10``;
  NO_GO if ``max(d_100, d_300) < 0.05``; else INCONCLUSIVE.

Secondary (descriptive, no rule): logistic-regression deltas, per-family additions, the
student's own Q added to EXISTING, the AUC among decisions that are not yet tight, per-arm
group AUCs, single-feature AUCs and the per-decision compute cost.

Runs with any Python that has numpy + scikit-learn (the project venv has no sklearn; the
study used /opt/anaconda3/bin/python3, read-only, nothing installed).
"""

from __future__ import annotations

import os

# one knob for every native thread pool (OpenMP for the boosting, BLAS for the logistic
# regression); must be set before numpy / sklearn load
_THREADS = os.environ.get("M7_THREADS", "3")
for _k in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = _THREADS

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Dict, List, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/m7-diagnostics-20261014/feature_auc")
ARMS = tuple(f"B8_s{i}" for i in range(5)) + tuple(f"CTRL_s1{i}" for i in range(4))
HS = (50, 100, 300)
MIN_L = 800
FRAMES = 5000
SELF, ENEMY_BODY = 2, 4
GO, NO_GO = 0.10, 0.05

# names (kept in sync with escape_nb / ego_raster / collect; asserted against shapes)
ACTION_FEATURES = (
    "A_nq4", "A_nq2", "A_nq1", "A_tstar", "A_own_first", "A_own_frac", "A_arr_largest",
    "B_front_own", "B_front_le_half", "B_front_le_R", "B_min_ttl", "B_slack", "B_own_share",
    "C_in_largest",
)  # fmt: skip
STATE_FEATURES = (
    "C_largest_over_L", "C_largest_frac", "C_ncomp32", "C_free_frac", "C_second_over_largest",
    "D_own_dens8", "D_own_dens16", "D_bbox_ratio", "D_fill",
)  # fmt: skip
SCALAR_NAMES = (
    "log_length", "length_per_1000", "pending_growth", "boost_available", "boost_phase",
    "hunger", "wall_ahead", "wall_right", "wall_behind", "wall_left", "alive_fraction",
    "reserved", "nearest_enemy_ahead", "nearest_enemy_right", "nearest_enemy_size",
    "nearest_enemy_heading_ahead", "nearest_enemy_heading_right", "nearest_enemy_boosting",
    "nearest_enemy_approach", "kill_opportunity", "second_enemy_ahead", "second_enemy_right",
    "second_enemy_size", "region_frac_left", "region_frac_straight", "region_frac_right",
    "region_log_left", "region_log_straight", "region_log_right", "enemies_alive_fraction",
) + tuple(
    f"v8_{f}_a{a}"
    for a in range(6)
    for f in ("area_g", "head_risky", "landing_ok", "v5_eligible", "spacious")
)  # fmt: skip
PLANE_NAMES = (
    "P_reach_cnt", "P_reach_le12", "P_own_cnt", "P_own_255", "P_own_min", "P_enemy_cnt",
    "P_wall_cnt", "G_open_blocks", "G_own_near", "G_mass_near",
)  # fmt: skip
FAMILIES = ("A", "B", "C", "D", "H")


# ----------------------------------------------------------------------------- features
def lagged(series: np.ndarray, lag: int) -> np.ndarray:
    """``series[t - lag]`` along the last axis (NaN before the start)."""
    out = np.full(series.shape, np.nan)
    if lag < series.shape[-1]:
        out[..., lag:] = series[..., :-lag]
    return out


def window_mean(x: np.ndarray, w: int) -> np.ndarray:
    """Trailing mean over the last ``w`` decisions (fewer at the start); NaN-free input."""
    c = np.cumsum(np.concatenate([np.zeros(x.shape[:-1] + (1,)), x], axis=-1), axis=-1)
    t = np.arange(1, x.shape[-1] + 1)
    lo = np.maximum(t - w, 0)
    return (c[..., t] - np.take(c, lo, axis=-1)) / (t - lo)


def history_features(
    grid: Dict[str, np.ndarray], alive: np.ndarray
) -> Tuple[List[str], np.ndarray]:
    """(names, (E, T, k)) per-env history features from (E, T) decision series."""
    rel = grid["rel_action"]
    left = np.where(alive, (rel == 0).astype(float), 0.0)
    right = np.where(alive, (rel == 2).astype(float), 0.0)
    boost = np.where(alive, grid["boost"].astype(float), 0.0)
    tight = np.where(alive, grid["tight"].astype(float), 0.0)
    feats, names = [], []
    for w in (50, 200):
        b = window_mean(left, w) - window_mean(right, w)
        feats += [b, np.abs(b)]
        names += [f"H_turn_bias{w}", f"H_turn_abs{w}"]
    for w in (100, 300):
        feats.append(window_mean(tight, w))
        names.append(f"H_tight_frac{w}")
    feats.append(window_mean(boost, 200))
    names.append("H_boost_frac200")
    for key in ("best_region_log", "best_area_g", "best_tstar", "best_nq1"):
        for lag in (10, 30, 100):
            feats.append(grid[key] - lagged(grid[key], lag))
            names.append(f"H_d_{key}_{lag}")
    feats.append((grid["length"] - lagged(grid["length"], 100)) / 100.0)
    names.append("H_growth100")
    return names, np.stack(feats, axis=-1)


def load_arm(arm: str, root: Path, stride: int) -> Dict[str, object]:
    z = np.load(root / "runs" / f"{arm}.npz")
    meta = json.loads((root / "runs" / f"{arm}.json").read_text())
    if meta["checks"].get("observer_action_mismatch", 0) or not (
        meta.get("reference_match") or {}
    ).get("match"):
        raise SystemExit(
            f"{arm}: capture checks failed {meta['checks']} {meta.get('reference_match')}"
        )
    env, frame = z["env"].astype(np.int64), z["frame"].astype(np.int64)
    n = len(env)
    S = z["scalars"].astype(np.float64)
    P = z["planes"].astype(np.float64)
    adv = z["adv"].astype(bool)
    act = z["action"].astype(np.int64)
    if S.shape[1] != len(SCALAR_NAMES) or P.shape[1] != len(PLANE_NAMES):
        raise SystemExit("feature name drift")
    area = S[:, 30:60].reshape(n, 6, 5)[:, :, 0]
    safe_area = np.where(adv, area, 0.0)
    best_area = safe_area.max(axis=1)
    taken_area = area[np.arange(n), act] * adv[np.arange(n), act]
    region_log = S[:, 26:29]
    region_frac = S[:, 23:26]
    rel = act % 3
    ex_derived = np.stack(
        [
            best_area,
            taken_area,
            adv.sum(axis=1),
            (best_area < 0.5).astype(float),
            region_log.max(axis=1),
            region_log[np.arange(n), rel],
            region_frac.max(axis=1),
            region_frac[np.arange(n), rel],
        ],
        axis=1,
    )
    ex_names = (
        [f"S_{s}" for s in SCALAR_NAMES]
        + list(PLANE_NAMES)
        + ["X_best_area_g", "X_taken_area_g", "X_n_safe", "X_tight", "X_best_region_log",
           "X_taken_region_log", "X_best_region_frac", "X_taken_region_frac"]
    )  # fmt: skip
    X_exist = np.concatenate([S, P, ex_derived], axis=1)
    q = z["q"].astype(np.float64)
    qm = np.where(adv | ~adv.any(axis=1, keepdims=True), q, -np.inf)
    qs = np.sort(qm, axis=1)
    X_q = np.stack([qs[:, -1], qs[:, -1] - np.where(np.isfinite(qs[:, -2]), qs[:, -2], qs[:, -1]),
                    q.mean(axis=1)], axis=1)  # fmt: skip
    q_names = ["Q_max", "Q_gap", "Q_mean"]
    # candidates (rows with L >= min_len)
    na, ns = len(ACTION_FEATURES), len(STATE_FEATURES)
    ca = np.full((n, 3, na), np.nan)
    cs = np.full((n, ns), np.nan)
    idx = z["cand_dec_index"]
    ca[idx] = z["cand_act"].astype(np.float64)
    cs[idx] = z["cand_state"].astype(np.float64)
    if z["cand_act"].shape[1:] != (3, na) or z["cand_state"].shape[1] != ns:
        raise SystemExit("candidate shape drift")
    safe3 = adv[:, :3]
    anysafe = safe3.any(axis=1, keepdims=True)
    pool = safe3 | ~anysafe  # no safe normal move: aggregate over all three
    agg_max = np.where(pool[:, :, None], ca, -np.inf).max(axis=1)
    agg_max = np.where(np.isfinite(agg_max), agg_max, np.nan)
    agg_taken = ca[np.arange(n), rel]
    cand_cols = [ca.reshape(n, 3 * na), agg_max, agg_taken, cs]
    cand_names = (
        [f"{f}_rel{a}" for a in range(3) for f in ACTION_FEATURES]
        + [f"{f}_maxsafe" for f in ACTION_FEATURES]
        + [f"{f}_taken" for f in ACTION_FEATURES]
        + list(STATE_FEATURES)
    )
    X_cand = np.concatenate(cand_cols, axis=1)
    # per-env (E, T) series for history features
    E = len(meta["seeds"])
    alive = np.zeros((E, FRAMES), bool)
    alive[env, frame] = True
    grid: Dict[str, np.ndarray] = {}
    for key, val in (
        ("rel_action", rel),
        ("boost", (act >= 3).astype(float)),
        ("tight", (best_area < 0.5).astype(float)),
        ("best_region_log", region_log.max(axis=1)),
        ("best_area_g", best_area),
        ("best_tstar", agg_max[:, ACTION_FEATURES.index("A_tstar")]),
        ("best_nq1", agg_max[:, ACTION_FEATURES.index("A_nq1")]),
        ("length", z["length"].astype(float)),
    ):
        g = np.full((E, FRAMES), np.nan if key not in ("rel_action",) else -1.0)
        g[env, frame] = val
        grid[key] = g
    h_names, H = history_features(grid, alive)
    X_hist = H[env, frame]
    death_frame = z["death_frame"].astype(np.int64)
    death_cause = z["death_cause"].astype(np.int64)
    seeds = np.asarray(meta["seeds"], np.int64)
    length = z["length"].astype(np.int64)
    k = (length >= MIN_L) & (frame % stride == 0)
    return {
        "arm": arm,
        "meta": meta,
        "world": seeds[env][k],
        "env": env[k],
        "frame": frame[k],
        "length": length[k],
        "fd": death_frame[env][k],
        "cause": death_cause[env][k],
        "X_exist": X_exist[k],
        "ex_names": ex_names,
        "X_cand": X_cand[k],
        "cand_names": cand_names,
        "X_hist": X_hist[k],
        "h_names": h_names,
        "X_q": X_q[k],
        "q_names": q_names,
        "tight": (best_area < 0.5)[k],
    }


def labels(fd: np.ndarray, cause: np.ndarray, frame: np.ndarray, H: int):
    """(y, keep) for death by self / enemy_body within the next H decisions."""
    died = fd >= 0
    within = died & (fd - frame < H)
    target = (cause == SELF) | (cause == ENEMY_BODY)
    y = within & target
    keep = ~(within & ~target) & ~(~died & (frame > FRAMES - H))
    return y, keep


# ----------------------------------------------------------------------------- AUC
class WAUC:
    """Weighted Mann-Whitney AUC with ties, for repeated world-bootstrap reweighting."""

    def __init__(self, score: np.ndarray, y: np.ndarray) -> None:
        uniq, self.inv = np.unique(score, return_inverse=True)
        self.g = len(uniq)
        self.y = y.astype(float)

    def __call__(self, w: np.ndarray) -> float:
        wp = np.bincount(self.inv, weights=w * self.y, minlength=self.g)
        wn = np.bincount(self.inv, weights=w * (1.0 - self.y), minlength=self.g)
        P, N = wp.sum(), wn.sum()
        if P <= 0 or N <= 0:
            return float("nan")
        below = np.cumsum(wn) - wn
        return float((wp * (below + 0.5 * wn)).sum() / (P * N))


def folds_by_world(worlds: np.ndarray, k: int = 5, seed: int = 0) -> np.ndarray:
    uw = np.unique(worlds)
    perm = np.random.default_rng(seed).permutation(len(uw))
    fold_of = {int(uw[p]): i % k for i, p in enumerate(perm)}
    return np.array([fold_of[int(w)] for w in worlds])


def make_model(kind: str, threads: int):
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.impute import SimpleImputer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if kind == "lr":
        return make_pipeline(
            SimpleImputer(strategy="constant", fill_value=0.0),
            StandardScaler(),
            LogisticRegression(C=0.1, max_iter=4000),
        )
    return HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=400,
        l2_regularization=1.0,
        early_stopping=False,
        random_state=0,
    )


def oof(X: np.ndarray, y: np.ndarray, fold: np.ndarray, kind: str, threads: int) -> np.ndarray:
    pred = np.zeros(len(y))
    X = np.where(np.isinf(X), np.nan, X)
    for k in np.unique(fold):
        tr, te = fold != k, fold == k
        m = make_model(kind, threads)
        m.fit(X[tr], y[tr])
        pred[te] = m.predict_proba(X[te])[:, 1]
    return pred


def boot_weights(worlds: np.ndarray, B: int, seed: int = 1) -> np.ndarray:
    uw, inv = np.unique(worlds, return_inverse=True)
    rng = np.random.default_rng(seed)
    counts = np.stack([np.bincount(rng.integers(0, len(uw), len(uw)), minlength=len(uw))
                       for _ in range(B)])  # fmt: skip
    return counts, inv


def ci(a: np.ndarray, lo: float, hi: float) -> List[float]:
    a = a[np.isfinite(a)]
    return [float(np.percentile(a, lo)), float(np.percentile(a, hi))]


# ----------------------------------------------------------------------------- main
def main(argv: Sequence[str] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=int(_THREADS), help="set M7_THREADS instead")
    ap.add_argument("--families", action="store_true", help="per-family additions (slower)")
    args = ap.parse_args(argv)
    root = Path(args.root)
    t_start = time.time()
    arms = [load_arm(a, root, args.stride) for a in ARMS if (root / "runs" / f"{a}.npz").exists()]
    if len(arms) != len(ARMS):
        print(f"WARNING: {len(arms)} of {len(ARMS)} arms present", flush=True)
    keys = ("world", "frame", "length", "fd", "cause", "X_exist", "X_cand", "X_hist", "X_q")
    cat = {k: np.concatenate([a[k] for a in arms]) for k in keys + ("tight",)}
    group = np.concatenate([np.full(len(a["frame"]), a["arm"][:2]) for a in arms])
    arm_id = np.concatenate([np.full(len(a["frame"]), a["arm"]) for a in arms])
    ex_names, cand_names = arms[0]["ex_names"], arms[0]["cand_names"]
    h_names, q_names = arms[0]["h_names"], arms[0]["q_names"]
    rows = (cat["length"] >= MIN_L) & (cat["frame"] % args.stride == 0)
    all_cand_names = cand_names + h_names
    X_all_cand = np.concatenate([cat["X_cand"], cat["X_hist"]], axis=1)
    if np.isnan(cat["X_cand"][rows]).any():
        raise SystemExit("missing candidate features at L >= 800")
    fam_cols = {
        f: [i for i, n in enumerate(all_cand_names) if n.startswith(f + "_")] for f in FAMILIES
    }
    sets = {
        "existing": cat["X_exist"],
        "existing+candidates": np.concatenate([cat["X_exist"], X_all_cand], axis=1),
        "existing+Q": np.concatenate([cat["X_exist"], cat["X_q"]], axis=1),
    }
    if args.families:
        for f in FAMILIES:
            sets[f"existing+{f}"] = np.concatenate(
                [cat["X_exist"], X_all_cand[:, fam_cols[f]]], axis=1
            )
        sets["existing+ABCD (no history)"] = np.concatenate(
            [cat["X_exist"], X_all_cand[:, sum((fam_cols[f] for f in "ABCD"), [])]], axis=1
        )
    world = cat["world"]
    fold_all = folds_by_world(world)
    counts, winv = boot_weights(world[rows], args.boot)
    out: Dict[str, object] = {
        "schema": "m7-feature-auc/v1",
        "authority": "development diagnostic only (not a screen, not gate evidence)",
        "rule": {
            "primary_model": "gbm",
            "delta": "AUC(existing+candidates) - AUC(existing), out-of-fold, world-grouped 5-fold",
            "GO": f"max(d_100, d_300) >= {GO}",
            "NO_GO": f"max(d_100, d_300) < {NO_GO}",
            "else": "INCONCLUSIVE",
        },
        "arms": [a["arm"] for a in arms],
        "stride": args.stride,
        "min_length": MIN_L,
        "n_features": {"existing": len(ex_names), "candidates": len(all_cand_names),
                       "candidate_families": {f: len(c) for f, c in fam_cols.items()}},  # fmt: skip
        "feature_names": {"existing": ex_names, "candidates": all_cand_names, "q": q_names},
        "by_H": {},
    }
    for H in HS:
        y_all, keep_all = labels(cat["fd"], cat["cause"], cat["frame"], H)
        sel = rows & keep_all
        y = y_all[sel]
        w_idx = winv[keep_all[rows]]  # bootstrap world index of every selected row
        ccounts = counts[:, :]
        fold = fold_all[sel]
        res: Dict[str, object] = {
            "rows": int(sel.sum()),
            "positives": int(y.sum()),
            "death_episodes": int(
                len({(a, w) for a, w, yy in zip(arm_id[sel], world[sel], y) if yy})
            ),
            "worlds_with_positive": int(len(np.unique(world[sel][y]))),
            "models": {},
        }
        preds: Dict[Tuple[str, str], np.ndarray] = {}
        for kind in ("gbm", "lr"):
            for name, X in sets.items():
                if (
                    kind == "lr"
                    and name.startswith("existing+")
                    and name not in ("existing+candidates", "existing+Q")
                ):
                    continue
                t0 = time.time()
                preds[(kind, name)] = oof(X[sel], y, fold, kind, args.threads)
                print(f"H{H} {kind} {name}: {time.time() - t0:.0f}s", flush=True)
        boots: Dict[Tuple[str, str], np.ndarray] = {}
        for key, p in preds.items():
            wa = WAUC(p, y)
            point = wa(np.ones(len(y)))
            bs = np.array([wa(ccounts[b][w_idx].astype(float)) for b in range(len(ccounts))])
            boots[key] = bs
            nt = ~cat["tight"][sel]
            res["models"].setdefault(key[0], {})[key[1]] = {
                "auc": point,
                "ci90": ci(bs, 5, 95),
                "ci95": ci(bs, 2.5, 97.5),
                "auc_not_tight": WAUC(p[nt], y[nt])(np.ones(int(nt.sum()))),
                "auc_by_group": {
                    g: WAUC(p[group[sel] == g], y[group[sel] == g])(
                        np.ones(int((group[sel] == g).sum()))
                    )
                    for g in ("B8", "CT")
                },
            }
        deltas = {}
        for kind in ("gbm", "lr"):
            base = boots[(kind, "existing")]
            pb = res["models"][kind]["existing"]["auc"]
            for name in sets:
                if name == "existing" or (kind, name) not in boots:
                    continue
                d = boots[(kind, name)] - base
                deltas.setdefault(kind, {})[name] = {
                    "delta": res["models"][kind][name]["auc"] - pb,
                    "ci90": ci(d, 5, 95),
                    "ci95": ci(d, 2.5, 97.5),
                    "delta_not_tight": res["models"][kind][name]["auc_not_tight"]
                    - res["models"][kind]["existing"]["auc_not_tight"],
                }
        res["deltas"] = deltas
        # warning lead time (descriptive): alarm = score above the 95th percentile of the
        # negatives; per death episode, the sustained lead = decisions before death since
        # which the alarm has been on at every sampled decision; also any alarm >= 100 early.
        lead: Dict[str, Dict[str, object]] = {}
        ep_key = np.char.add(arm_id[sel].astype(str), world[sel].astype(str))
        fr, fds = cat["frame"][sel], cat["fd"][sel]
        for kind, name in (("gbm", "existing"), ("gbm", "existing+candidates")):
            p = preds[(kind, name)]
            tau = float(np.percentile(p[~y], 95))
            sustained, early = [], []
            for key in np.unique(ep_key[y]):
                m = (ep_key == key) & y
                order = np.argsort(-(fds[m] - fr[m]))  # earliest decision first
                before = (fds[m] - fr[m])[order]
                on = (p[m] >= tau)[order]
                run = 0
                for b, o in zip(before[::-1], on[::-1]):  # from the fatal end backwards
                    if not o:
                        break
                    run = int(b)
                sustained.append(run)
                early.append(bool((on & (before >= 100)).any()))
            lead[f"{kind}:{name}"] = {
                "fpr": 0.05,
                "sustained_lead_median": float(np.median(sustained)) if sustained else None,
                "sustained_lead_quartiles": (
                    [float(x) for x in np.percentile(sustained, [25, 75])] if sustained else None
                ),
                "any_alarm_ge100_before_death_frac": float(np.mean(early)) if early else None,
                "episodes": len(sustained),
            }
        res["lead_time"] = lead
        # univariate AUCs (sign-free: max(auc, 1 - auc)), candidates vs existing, top 12
        uni = {}
        for names, X in ((ex_names, cat["X_exist"]), (all_cand_names, X_all_cand)):
            for j, nm in enumerate(names):
                v = X[sel, j]
                ok = np.isfinite(v)
                if ok.sum() < 100 or np.nanstd(v) == 0:
                    continue
                a = WAUC(v[ok], y[ok])(np.ones(int(ok.sum())))
                uni[nm] = {
                    "auc": a,
                    "direction": "+" if a >= 0.5 else "-",
                    "auc_abs": max(a, 1 - a),
                }
        top = sorted(uni.items(), key=lambda kv: -kv[1]["auc_abs"])
        res["univariate_top_existing"] = [
            [k, v["auc_abs"], v["direction"]] for k, v in top if k in ex_names
        ][:12]
        res["univariate_top_candidates"] = [
            [k, v["auc_abs"], v["direction"]] for k, v in top if k not in ex_names
        ][:15]
        out["by_H"][str(H)] = res
        print(
            json.dumps(
                {H: {k: v for k, v in res.items() if k in ("rows", "positives", "deltas")}},
                default=float,
            )[:3000],
            flush=True,
        )
    d = {
        H: out["by_H"][str(H)]["deltas"]["gbm"]["existing+candidates"]["delta"] for H in (100, 300)
    }
    best = max(d.values())
    verdict = "GO" if best >= GO else ("NO_GO" if best < NO_GO else "INCONCLUSIVE")
    out["verdict"] = {"verdict": verdict, "d_100": d[100], "d_300": d[300], "max": best}
    # compute cost (from the capture runs)
    cost = {a["arm"]: a["meta"]["per_decision_ms"] for a in arms}
    out["compute_ms_per_decision"] = {
        "per_arm": cost,
        "median_kernel_full_abcd": float(np.median([c["kernel_full_abcd"] for c in cost.values()])),
        "median_kernel_static_bcd": float(
            np.median([c["kernel_static_bcd"] for c in cost.values()])
        ),
        "median_incl_python": float(
            np.median([c["candidates_all_families_incl_python"] for c in cost.values()])
        ),
    }
    out["wall_seconds"] = time.time() - t_start
    path = root / "results.json"
    path.write_text(json.dumps(out, indent=1, default=float))
    print(json.dumps(out["verdict"]), str(path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
