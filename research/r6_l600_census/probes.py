"""5-fold world-disjoint OOF separability probes and R6 kill-rule evaluation.

Evaluates whether causal ``t=0`` observations (``ego2s-bv8`` alone vs ``ego2s-bv8``
augmented with ``+2`` global 37x37 FIFO-weighted corpse/trail planes + 4 scalars)
can select ``L<600`` safe-harvest overrides out-of-fold that simultaneously satisfy
R6's three pre-declared bars:
1. Qualifying-root recall ``>= 15%`` (``death_delta_A500 >= delta_A500 >= 5.0`` & survive).
2. Mean selected ``delta_A500 >= +8.0``.
3. H500 lethal-flip rate ``<= 0.5%`` (``0.005``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, RidgeCV
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler

from research.r6_l600_census import R6_KILL_THRESHOLDS


@dataclass(frozen=True)
class CandidateActionSample:
    """One non-bare advisory-legal candidate action at a captured L<600 root."""

    root_idx: int
    world_idx: int
    world_seed: int
    action: int
    bare_action: int
    bare_survives: bool
    action_survives: bool
    delta_A500: float
    death_delta_A500: float
    qualifies: bool
    lethal_flip: bool


def assign_world_folds(world_indices: Sequence[int], n_folds: int = 5) -> np.ndarray:
    """Deterministically partition unique worlds into ``n_folds`` disjoint folds."""
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    unique_worlds = sorted({int(w) for w in world_indices})
    if len(unique_worlds) < n_folds:
        raise ValueError(f"need >= {n_folds} unique worlds for {n_folds}-fold world-disjoint CV")
    world_to_fold = {w: idx % n_folds for idx, w in enumerate(unique_worlds)}
    return np.array([world_to_fold[int(w)] for w in world_indices], dtype=np.int64)


def _safe_auc(y_true: np.ndarray, y_score: np.ndarray) -> tuple[float | None, float | None]:
    y_b = np.asarray(y_true, dtype=int)
    if y_b.size == 0 or int(y_b.sum()) == 0 or int(y_b.sum()) == int(y_b.size):
        return None, None
    return float(roc_auc_score(y_b, y_score)), float(average_precision_score(y_b, y_score))


def build_candidate_dataset(
    roots: Sequence[Mapping[str, Any]],
) -> tuple[list[CandidateActionSample], np.ndarray, np.ndarray, np.ndarray, list[bool]]:
    """Construct aligned candidate-action feature matrices from evaluated roots.

    Returns:
        Tuple ``(samples, x_bv8, x_extra_tabular, x_planes_flat, root_qualifies)``.
    """
    samples: list[CandidateActionSample] = []
    rows_bv8: list[np.ndarray] = []
    rows_extra: list[np.ndarray] = []
    rows_planes: list[np.ndarray] = []
    root_qualifies: list[bool] = []

    for r_idx, root in enumerate(roots):
        world_idx = int(root["world_idx"])
        world_seed = int(root["world_seed"])
        bare = int(root["bare_action"])
        branches = root["branches"]
        bare_br = branches[str(bare)]
        bare_surv = bool(bare_br["survives"])

        phi = np.asarray(root["phi"], dtype=np.float32)  # (256,)
        q_b8 = np.asarray(root["q_b8"], dtype=np.float32)  # (6,)
        scalars_bv8 = np.asarray(root["scalars_bv8"], dtype=np.float32)  # (60,)
        planes = np.asarray(root["corpse_trail_planes"], dtype=np.float32)  # (2, 37, 37)
        pooled = np.asarray(root["corpse_trail_pooled"], dtype=np.float32)  # (28,)
        s4 = np.asarray(root["corpse_fifo_scalars"], dtype=np.float32)  # (4,)
        aux4 = np.asarray(root["corpse_fifo_aux"], dtype=np.float32)  # (4,)
        hero_len_norm = float(root["hero_length"]) / 600.0

        q_centered = q_b8 - float(q_b8[bare])
        v8_bare = scalars_bv8[30 + 5 * bare : 30 + 5 * (bare + 1)]
        planes_flat = planes.reshape(-1)

        any_qual = False
        for a_str, br in sorted(branches.items(), key=lambda kv: int(kv[0])):
            act = int(a_str)
            if act == bare:
                continue
            act_surv = bool(br["survives"])
            delta_a500 = float(br["delta_A500"])
            death_delta_a500 = float(br["death_delta_A500"])
            qual = bool(
                act_surv
                and bare_surv
                and death_delta_a500
                >= delta_a500
                >= R6_KILL_THRESHOLDS["min_qualifying_delta_A500"]
            )
            any_qual = any_qual or qual
            lflip = bool(bare_surv and not act_surv)

            one_hot = np.zeros(6, dtype=np.float32)
            one_hot[act] = 1.0
            is_boost = 1.0 if act >= 3 else 0.0
            v8_act = scalars_bv8[30 + 5 * act : 30 + 5 * (act + 1)]
            v8_diff = v8_act - v8_bare

            x_base = np.concatenate(
                [
                    phi,
                    scalars_bv8,
                    q_centered,
                    np.array([hero_len_norm, is_boost, float(q_centered[act])], dtype=np.float32),
                    one_hot,
                    v8_act,
                    v8_diff,
                ]
            )
            # Action-aligned cone indices in pooled (11, 12, 13 for drop; 25, 26, 27 for trail).
            d_rel = act % 3
            drop_cone = float(pooled[11 + d_rel])
            trail_cone = float(pooled[25 + d_rel])
            fifo_fill = float(s4[0])
            action_extra = np.array(
                [
                    drop_cone,
                    trail_cone,
                    is_boost * drop_cone,
                    is_boost * float(s4[1]),
                    is_boost * fifo_fill,
                    is_boost * float(s4[3]),
                    float(v8_act[0]) * drop_cone,  # landing_area * drop_cone
                    (1.0 - float(v8_act[1])) * drop_cone,  # !head_risky * drop_cone
                ],
                dtype=np.float32,
            )
            x_ext = np.concatenate(
                [
                    s4,
                    np.array(
                        [aux4[0], aux4[1], aux4[2] / 150.0, aux4[3] / 150.0], dtype=np.float32
                    ),
                    pooled,
                    action_extra,
                ]
            )

            samples.append(
                CandidateActionSample(
                    root_idx=r_idx,
                    world_idx=world_idx,
                    world_seed=world_seed,
                    action=act,
                    bare_action=bare,
                    bare_survives=bare_surv,
                    action_survives=act_surv,
                    delta_A500=delta_a500,
                    death_delta_A500=death_delta_a500,
                    qualifies=qual,
                    lethal_flip=lflip,
                )
            )
            rows_bv8.append(x_base)
            rows_extra.append(x_ext)
            rows_planes.append(planes_flat)
        root_qualifies.append(any_qual)

    if not samples:
        return (
            [],
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            np.zeros((0, 1), dtype=np.float32),
            root_qualifies,
        )
    return (
        samples,
        np.stack(rows_bv8, axis=0),
        np.stack(rows_extra, axis=0),
        np.stack(rows_planes, axis=0),
        root_qualifies,
    )


def _fit_binary_prob(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, *, model_kind: str, seed: int
) -> np.ndarray:
    pos = int(y_tr.sum())
    if pos == 0:
        return np.zeros(x_te.shape[0], dtype=np.float64)
    if pos == y_tr.size:
        return np.ones(x_te.shape[0], dtype=np.float64)
    if model_kind == "linear_l2":
        clf = LogisticRegression(
            C=0.05,
            max_iter=1000,
            class_weight="balanced",
            random_state=seed,
        )
        clf.fit(x_tr, y_tr)
        return clf.predict_proba(x_te)[:, 1].astype(np.float64)
    if model_kind == "gbdt_shallow":
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            min_samples_leaf=max(5, min(20, x_tr.shape[0] // 10)),
            max_iter=80,
            l2_regularization=5.0,
            random_state=seed,
        )
        clf.fit(x_tr, y_tr)
        return clf.predict_proba(x_te)[:, 1].astype(np.float64)
    raise ValueError(f"unknown model_kind: {model_kind}")


def _fit_regression(
    x_tr: np.ndarray, y_tr: np.ndarray, x_te: np.ndarray, *, model_kind: str, seed: int
) -> np.ndarray:
    if model_kind == "linear_l2":
        reg = RidgeCV(alphas=np.array([1.0, 10.0, 100.0, 500.0, 2000.0], dtype=np.float64))
        reg.fit(x_tr, y_tr)
        return reg.predict(x_te).astype(np.float64)
    if model_kind == "gbdt_shallow":
        reg = HistGradientBoostingRegressor(
            max_depth=3,
            min_samples_leaf=max(5, min(20, x_tr.shape[0] // 10)),
            max_iter=80,
            l2_regularization=5.0,
            random_state=seed,
        )
        reg.fit(x_tr, y_tr)
        return reg.predict(x_te).astype(np.float64)
    raise ValueError(f"unknown model_kind: {model_kind}")


def evaluate_threshold_grid(
    *,
    num_roots: int,
    root_qualifies: Sequence[bool],
    samples: Sequence[CandidateActionSample],
    score_oof: np.ndarray,
    prob_die_oof: np.ndarray,
    thresholds: Mapping[str, float] = R6_KILL_THRESHOLDS,
) -> dict[str, Any]:
    """Sweep ``(p_die_max, tau)`` on OOF scores and check R6's three simultaneous bars."""
    n_qual_roots = int(sum(1 for q in root_qualifies if q))
    by_root: list[list[int]] = [[] for _ in range(num_roots)]
    for idx, s in enumerate(samples):
        by_root[s.root_idx].append(idx)

    p_die_grid = (0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20, 0.35, 0.50, 1.0)
    min_rec = float(thresholds["min_oof_qualifying_root_recall"])
    min_delta = float(thresholds["min_mean_selected_delta_A500"])
    max_lflip = float(thresholds["max_H500_lethal_flip_rate"])

    passing_strict: list[dict[str, Any]] = []
    passing_all_root_denom: list[dict[str, Any]] = []
    best_safe_point: dict[str, Any] | None = None
    best_recall_at_delta8: dict[str, Any] | None = None

    for p_max in p_die_grid:
        root_best_idx = np.full(num_roots, -1, dtype=np.int64)
        root_best_score = np.full(num_roots, -np.inf, dtype=np.float64)
        for r in range(num_roots):
            cand = [i for i in by_root[r] if prob_die_oof[i] <= p_max]
            if cand:
                best_i = max(cand, key=lambda i: (float(score_oof[i]), -samples[i].action))
                root_best_idx[r] = best_i
                root_best_score[r] = float(score_oof[best_i])

        finite_scores = np.sort(np.unique(root_best_score[np.isfinite(root_best_score)]))[::-1]
        if finite_scores.size > 160:
            qs = np.linspace(0, finite_scores.size - 1, 160, dtype=np.int64)
            tau_candidates = finite_scores[np.unique(qs)]
        else:
            tau_candidates = finite_scores

        for tau in tau_candidates.tolist():
            sel_roots = np.flatnonzero(root_best_score >= tau)
            n_sel = int(sel_roots.size)
            if n_sel == 0:
                continue
            chosen_sample_idxs = root_best_idx[sel_roots]
            deltas = np.array([samples[int(i)].delta_A500 for i in chosen_sample_idxs])
            lflips = int(sum(1 for i in chosen_sample_idxs if samples[int(i)].lethal_flip))
            any_dies = int(
                sum(1 for i in chosen_sample_idxs if not samples[int(i)].action_survives)
            )
            qual_exact_hits = int(sum(1 for i in chosen_sample_idxs if samples[int(i)].qualifies))
            qual_any_hits = int(sum(1 for r in sel_roots if root_qualifies[int(r)]))

            recall_exact = float(qual_exact_hits / n_qual_roots) if n_qual_roots > 0 else 0.0
            recall_any = float(qual_any_hits / n_qual_roots) if n_qual_roots > 0 else 0.0
            mean_delta = float(deltas.mean())
            lflip_rate_sel = float(lflips / n_sel)
            lflip_rate_all = float(lflips / max(1, num_roots))

            point = {
                "p_die_max": float(p_max),
                "tau": float(tau),
                "selected_roots": n_sel,
                "qualifying_roots_total": n_qual_roots,
                "qualifying_exact_hits": qual_exact_hits,
                "qualifying_root_any_hits": qual_any_hits,
                "recall_qualifying_exact": round(recall_exact, 6),
                "recall_qualifying_root_any": round(recall_any, 6),
                "mean_selected_delta_A500": round(mean_delta, 4),
                "lethal_flips": lflips,
                "selected_deaths_total": any_dies,
                "lethal_flip_rate_per_selected": round(lflip_rate_sel, 6),
                "lethal_flip_rate_all_roots": round(lflip_rate_all, 6),
            }

            if recall_exact >= min_rec and mean_delta >= min_delta and lflip_rate_sel <= max_lflip:
                passing_strict.append(point)
            if recall_any >= min_rec and mean_delta >= min_delta and lflip_rate_all <= max_lflip:
                passing_all_root_denom.append(point)

            if lflip_rate_sel <= max_lflip:
                if best_safe_point is None or (
                    point["recall_qualifying_exact"],
                    point["mean_selected_delta_A500"],
                ) > (
                    best_safe_point["recall_qualifying_exact"],
                    best_safe_point["mean_selected_delta_A500"],
                ):
                    best_safe_point = point

            if mean_delta >= min_delta:
                if best_recall_at_delta8 is None or (
                    -point["lethal_flip_rate_per_selected"],
                    point["recall_qualifying_exact"],
                ) > (
                    -best_recall_at_delta8["lethal_flip_rate_per_selected"],
                    best_recall_at_delta8["recall_qualifying_exact"],
                ):
                    best_recall_at_delta8 = point

    passing_strict.sort(
        key=lambda p: (-p["recall_qualifying_exact"], -p["mean_selected_delta_A500"])
    )
    passing_all_root_denom.sort(
        key=lambda p: (-p["recall_qualifying_root_any"], -p["mean_selected_delta_A500"])
    )
    return {
        "passes_strict": len(passing_strict) > 0,
        "passes_all_root_denom": len(passing_all_root_denom) > 0,
        "num_passing_strict": len(passing_strict),
        "num_passing_all_root_denom": len(passing_all_root_denom),
        "best_passing_strict": passing_strict[0] if passing_strict else None,
        "best_passing_all_root_denom": (
            passing_all_root_denom[0] if passing_all_root_denom else None
        ),
        "best_safe_point_lflip_le_0p5pct": best_safe_point,
        "best_point_mean_delta_ge_8": best_recall_at_delta8,
    }


def run_world_disjoint_oof_probes(
    roots: Sequence[Mapping[str, Any]],
    *,
    n_folds: int = 5,
    thresholds: Mapping[str, float] = R6_KILL_THRESHOLDS,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Run 5-fold world-disjoint OOF probes on ``ego2s-bv8`` vs ``ego2s-bv8 + corpse/FIFO``."""
    num_roots = len(roots)
    if num_roots == 0:
        raise ValueError("roots must be nonempty")
    root_worlds = [int(r["world_idx"]) for r in roots]
    root_folds = assign_world_folds(root_worlds, n_folds=n_folds)

    samples, x_bv8, x_extra, x_planes, root_qualifies = build_candidate_dataset(roots)
    n_samples = len(samples)
    if n_samples == 0:
        raise ValueError("no non-bare legal candidate actions found across roots")

    sample_folds = np.array([root_folds[s.root_idx] for s in samples], dtype=np.int64)
    sample_worlds = np.array([s.world_seed for s in samples], dtype=np.int64)
    y_qual = np.array([1 if s.qualifies else 0 for s in samples], dtype=np.int64)
    y_die = np.array([1 if not s.action_survives else 0 for s in samples], dtype=np.int64)
    y_lflip = np.array([1 if s.lethal_flip else 0 for s in samples], dtype=np.int64)
    y_delta = np.array([s.delta_A500 for s in samples], dtype=np.float64)

    probe_results: dict[str, Any] = {}
    any_strict_pass = False
    any_lenient_pass = False

    for feat_name in ("ego2s_bv8", "ego2s_bv8_plus_corpse_fifo"):
        for model_kind in ("linear_l2", "gbdt_shallow"):
            prob_qual_oof = np.zeros(n_samples, dtype=np.float64)
            prob_die_oof = np.zeros(n_samples, dtype=np.float64)
            pred_delta_oof = np.zeros(n_samples, dtype=np.float64)

            for fold in range(n_folds):
                te_mask = sample_folds == fold
                tr_mask = ~te_mask
                if not np.any(te_mask) or not np.any(tr_mask):
                    raise ValueError(f"degenerate fold {fold} in world-disjoint CV")
                # Explicitly verify zero world-seed overlap between train and test folds.
                tr_worlds = set(sample_worlds[tr_mask].tolist())
                te_worlds = set(sample_worlds[te_mask].tolist())
                if not tr_worlds.isdisjoint(te_worlds):
                    raise AssertionError("world leakage detected across CV fold boundary")

                if feat_name == "ego2s_bv8":
                    x_tr_raw = x_bv8[tr_mask]
                    x_te_raw = x_bv8[te_mask]
                else:
                    n_comp = min(12, x_planes[tr_mask].shape[0] - 1, x_planes.shape[1])
                    if n_comp >= 1 and float(np.std(x_planes[tr_mask])) > 1e-8:
                        pca = PCA(n_components=n_comp, random_state=seed + fold)
                        p_tr = pca.fit_transform(x_planes[tr_mask]).astype(np.float32)
                        p_te = pca.transform(x_planes[te_mask]).astype(np.float32)
                    else:
                        p_tr = np.zeros((int(tr_mask.sum()), 1), dtype=np.float32)
                        p_te = np.zeros((int(te_mask.sum()), 1), dtype=np.float32)
                    x_tr_raw = np.concatenate([x_bv8[tr_mask], x_extra[tr_mask], p_tr], axis=1)
                    x_te_raw = np.concatenate([x_bv8[te_mask], x_extra[te_mask], p_te], axis=1)

                scaler = StandardScaler()
                x_tr = scaler.fit_transform(x_tr_raw)
                x_te = scaler.transform(x_te_raw)

                fold_seed = seed + 100 * fold
                prob_qual_oof[te_mask] = _fit_binary_prob(
                    x_tr, y_qual[tr_mask], x_te, model_kind=model_kind, seed=fold_seed
                )
                prob_die_oof[te_mask] = _fit_binary_prob(
                    x_tr, y_die[tr_mask], x_te, model_kind=model_kind, seed=fold_seed + 1
                )
                pred_delta_oof[te_mask] = _fit_regression(
                    x_tr, y_delta[tr_mask], x_te, model_kind=model_kind, seed=fold_seed + 2
                )

            qual_roc, qual_pr = _safe_auc(y_qual, prob_qual_oof)
            die_roc, die_pr = _safe_auc(y_die, prob_die_oof)
            lflip_roc, lflip_pr = _safe_auc(y_lflip, prob_die_oof)

            score_variants = {
                "prob_qual": prob_qual_oof,
                "pred_delta": pred_delta_oof,
                "risk_penalized_delta": pred_delta_oof - 40.0 * prob_die_oof,
            }
            sweeps: dict[str, Any] = {}
            for s_name, s_vec in score_variants.items():
                res = evaluate_threshold_grid(
                    num_roots=num_roots,
                    root_qualifies=root_qualifies,
                    samples=samples,
                    score_oof=s_vec,
                    prob_die_oof=prob_die_oof,
                    thresholds=thresholds,
                )
                sweeps[s_name] = res
                if feat_name == "ego2s_bv8_plus_corpse_fifo":
                    any_strict_pass = any_strict_pass or bool(res["passes_strict"])
                    any_lenient_pass = any_lenient_pass or bool(res["passes_all_root_denom"])

            key = f"{feat_name}__{model_kind}"
            probe_results[key] = {
                "feature_set": feat_name,
                "model_kind": model_kind,
                "qualifies_oof_roc_auc": qual_roc,
                "qualifies_oof_pr_auc": qual_pr,
                "die_H500_oof_roc_auc": die_roc,
                "die_H500_oof_pr_auc": die_pr,
                "lethal_flip_oof_roc_auc": lflip_roc,
                "lethal_flip_oof_pr_auc": lflip_pr,
                "delta_A500_oof_rmse": float(np.sqrt(np.mean((pred_delta_oof - y_delta) ** 2))),
                "threshold_sweeps": sweeps,
            }

    n_qual_roots = int(sum(1 for q in root_qualifies if q))
    n_qual_actions = int(y_qual.sum())
    n_lflip_actions = int(y_lflip.sum())
    boost_mask = np.array([s.action >= 3 for s in samples], dtype=bool)
    n_boost_actions = int(boost_mask.sum())
    n_boost_lflips = int(y_lflip[boost_mask].sum()) if n_boost_actions > 0 else 0

    verdict = "PASS" if any_strict_pass else "KILL"
    return {
        "schema": "t34-r6-oof-separability-summary/v1",
        "verdict": verdict,
        "verdict_all_root_denom_sensitivity": "PASS" if any_lenient_pass else "KILL",
        "frozen_kill_thresholds": dict(thresholds),
        "cv_folds": n_folds,
        "world_disjoint_verified": True,
        "num_worlds": len(set(root_worlds)),
        "num_roots": num_roots,
        "num_candidate_nonbare_actions": n_samples,
        "qualifying_roots": n_qual_roots,
        "qualifying_root_fraction": round(n_qual_roots / max(1, num_roots), 6),
        "qualifying_actions": n_qual_actions,
        "lethal_flip_actions": n_lflip_actions,
        "lethal_flip_action_fraction": round(n_lflip_actions / max(1, n_samples), 6),
        "legal_nonbare_boost_actions": n_boost_actions,
        "legal_nonbare_boost_lethal_flips": n_boost_lflips,
        "legal_nonbare_boost_lethal_flip_rate": (
            round(n_boost_lflips / n_boost_actions, 6) if n_boost_actions > 0 else 0.0
        ),
        "probes": probe_results,
    }
