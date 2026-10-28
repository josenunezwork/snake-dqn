"""Ordered decision gates, parent-world cluster bootstrap, AUC, and Holm for C/T.

Implements Ruling §4:
- Ordered evaluation: ``INVALID`` or ``INCOMPLETE_SUPPORT`` first; then on
  held-out assessment worlds using independent scoring suffixes ``{4..7}``:
  (a) ``>= 25%`` of held-out death events have ``>= 1`` lookback root where
      ``S_score(s, a*) >= 0.75`` and ``S_score(s, a_B8) <= 0.25``
      (``< 15%`` -> ``STOP_RESCUE_TARGET_PROGRAM``; ``[15%, 25%)`` -> ``INCONCLUSIVE``);
  (b) selected-action ``H300`` mass benefit ``M_score(s, a*) - M_score(s, a_B8)``
      has parent-world bootstrap ``LB90 > 0``;
  (c) ``full_bv8`` greedy rescue retention ``S_score(s, a_theta) >= 0.75``
      (also reporting ``a_theta == a*`` diagnostic) ``>= 30%`` and ``>= +15 pp``
      above ``frozen_phi``;
  (d) ordinary greedy flips ``<= 1.0%`` on held-out archive (``>= 10,000`` rows)
      and ``<= 1`` fresh-scored lethal flip (``S_score(s, a_B8) - S_score(s, a_theta) >= 0.50``)
      among ``>= 256`` searched held-out ordinary roots.
- T-arm (``full_bv8_plus_topology``) requires (a), (b), retention ``>= 30%``,
  ``>= +15 pp`` over ``frozen_phi``, ``>= +15 pp`` over ``full_bv8``, Gate (d),
  and world-grouped ``D100-D300`` separability ``AUC >= 0.83`` when both classes
  have support (``AUC alone never buys training``).
- Parent-world bootstrap intervals (10,000 resamples, seed 3432) for mass benefit
  and retention differences; Holm step-down (``m = 2``) applied only when claiming
  statistical superiority across trainable arms.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from research.sandbox_a_resident.ct_branching import MergedScoredRoots
from research.sandbox_a_resident.ct_fit import D100_D300_LOOKBACKS, ArmEvaluationPredictions
from research.sandbox_a_resident.ct_snapshots import LosslessBoundedSnapshotStore, RootMetadata
from research.sandbox_a_resident.ct_task import (
    ORDINARY_LENGTH_BIN_EDGES,
    ORDINARY_LENGTH_BIN_LABELS,
    AnchorArchive,
    CTControl,
    _length_bin_index,
    compute_required_ordinary_cell_quotas,
)
from research.sandbox_a_resident.training_rosters import MIXES


def validate_ct_support(
    *,
    control: CTControl,
    store: LosslessBoundedSnapshotStore,
    anchors: AnchorArchive,
    world_partitions: Sequence[str],
) -> dict[str, Any]:
    """Audit held-out and train support counts and per-mix breakdowns before fitting."""
    mix_names = tuple(MIXES)
    assess_worlds = {w for w, p in enumerate(world_partitions) if p == "assessment"}
    train_worlds = {w for w, p in enumerate(world_partitions) if p == "train"}

    # Verify strict whole-world partition integrity across every root and anchor row.
    partition_leak = False
    for meta in store.metadata:
        expected_part = world_partitions[meta.world_id]
        if meta.partition != expected_part:
            partition_leak = True
            break
    if not partition_leak and anchors.num_rows > 0:
        anc_is_assess = np.array(
            [world_partitions[int(w)] == "assessment" for w in anchors.world_id],
            dtype=bool,
        )
        if not np.array_equal(anc_is_assess, anchors.is_assessment):
            partition_leak = True

    per_mix: dict[str, dict[str, dict[str, int]]] = {
        part: {
            m: {
                "worlds_allocated": 0,
                "death_events": 0,
                "distinct_death_worlds": 0,
                "death_lookback_roots": 0,
                "ordinary_searched_roots": 0,
                "anchor_rows": 0,
            }
            for m in mix_names
        }
        for part in ("train", "assessment")
    }

    from research.sandbox_a_resident.training_rosters import training_rosters

    rosters = training_rosters(control.worlds)
    for w_idx, part in enumerate(world_partitions):
        per_mix[part][rosters[w_idx].mix]["worlds_allocated"] += 1

    seen_deaths: dict[str, set[int]] = {"train": set(), "assessment": set()}
    seen_death_worlds: dict[str, set[int]] = {"train": set(), "assessment": set()}
    seen_death_worlds_mix: dict[tuple[str, str], set[int]] = {
        (part, m): set() for part in ("train", "assessment") for m in mix_names
    }
    seen_deaths_mix: dict[tuple[str, str], set[int]] = {
        (part, m): set() for part in ("train", "assessment") for m in mix_names
    }

    n_bins = len(ORDINARY_LENGTH_BIN_EDGES)
    death_cells: dict[str, dict[tuple[str, int], int]] = {
        part: {(m, b): 0 for m in mix_names for b in range(n_bins)}
        for part in ("train", "assessment")
    }
    ordinary_cells: dict[str, dict[tuple[str, int], int]] = {
        part: {(m, b): 0 for m in mix_names for b in range(n_bins)}
        for part in ("train", "assessment")
    }

    train_searched_count = 0
    assess_ordinary_count = 0
    for meta in store.metadata:
        part = meta.partition
        m = meta.mix
        b = _length_bin_index(int(meta.hero_length))
        if part == "train":
            train_searched_count += 1
        if meta.kind == "death_lookback":
            per_mix[part][m]["death_lookback_roots"] += 1
            death_cells[part][(m, b)] += 1
            if meta.death_event_id is not None:
                seen_deaths[part].add(meta.death_event_id)
                seen_deaths_mix[(part, m)].add(meta.death_event_id)
                seen_death_worlds[part].add(meta.world_id)
                seen_death_worlds_mix[(part, m)].add(meta.world_id)
        elif meta.kind == "ordinary":
            per_mix[part][m]["ordinary_searched_roots"] += 1
            ordinary_cells[part][(m, b)] += 1
            if part == "assessment":
                assess_ordinary_count += 1

    for part in ("train", "assessment"):
        for m in mix_names:
            per_mix[part][m]["death_events"] = len(seen_deaths_mix[(part, m)])
            per_mix[part][m]["distinct_death_worlds"] = len(seen_death_worlds_mix[(part, m)])

    for idx_m, m in enumerate(mix_names):
        if anchors.num_rows > 0:
            mask_m = anchors.mix_id == idx_m
            per_mix["train"][m]["anchor_rows"] = int(np.sum(mask_m & (~anchors.is_assessment)))
            per_mix["assessment"][m]["anchor_rows"] = int(np.sum(mask_m & anchors.is_assessment))

    held_out_deaths = len(seen_deaths["assessment"])
    held_out_death_worlds = len(seen_death_worlds["assessment"])
    held_out_ordinary_roots = int(assess_ordinary_count)
    held_out_anchors = int(np.sum(anchors.is_assessment))
    train_anchors = int(np.sum(~anchors.is_assessment))

    target_assess_ord = int(round(0.20 * control.max_ordinary_roots))
    target_train_ord = int(control.max_ordinary_roots) - target_assess_ord
    ordinary_matching: dict[str, Any] = {
        "length_bin_edges": list(ORDINARY_LENGTH_BIN_EDGES),
        "length_bin_labels": list(ORDINARY_LENGTH_BIN_LABELS),
        "cross_mix_or_bin_spill_allowed": False,
        "partitions": {},
    }
    matching_reasons: list[str] = []
    for part, quota, min_ord in (
        ("train", target_train_ord, 0 if control.tiny else 1),
        ("assessment", target_assess_ord, int(control.min_held_out_ordinary_roots)),
    ):
        d_map = death_cells[part]
        o_map = ordinary_cells[part]
        total_d = sum(d_map.values())
        if total_d > 0:
            req_map = compute_required_ordinary_cell_quotas(
                d_map, quota=quota, min_partition_ordinary=min_ord
            )
        else:
            req_map = {k: 0 for k in d_map}
        part_shortfall = 0
        part_unmatched = 0
        cells_out: dict[str, dict[str, dict[str, int]]] = {m: {} for m in mix_names}
        for m in mix_names:
            for b in range(n_bins):
                req_c = int(req_map[(m, b)])
                act_c = int(o_map[(m, b)])
                sf_c = max(0, req_c - act_c)
                unm_c = act_c if (total_d > 0 and req_c == 0 and act_c > 0) else 0
                part_shortfall += sf_c
                part_unmatched += unm_c
                cells_out[m][ORDINARY_LENGTH_BIN_LABELS[b]] = {
                    "death_lookback_roots": int(d_map[(m, b)]),
                    "required_ordinary_roots": req_c,
                    "achieved_ordinary_roots": act_c,
                    "shortfall": sf_c,
                    "unmatched_in_zero_death_cell": unm_c,
                }
        ordinary_matching["partitions"][part] = {
            "partition_quota_cap": int(quota),
            "death_lookback_roots_total": int(total_d),
            "required_ordinary_roots_total": int(sum(req_map.values())),
            "achieved_ordinary_roots_total": int(sum(o_map.values())),
            "shortfall_total": int(part_shortfall),
            "unmatched_in_zero_death_cell_total": int(part_unmatched),
            "per_mix_bin": cells_out,
        }
        if part_shortfall > 0:
            matching_reasons.append(f"ordinary_matching_shortfall[{part}]={part_shortfall}")
        if part_unmatched > 0:
            matching_reasons.append(f"ordinary_unmatched_substitution[{part}]={part_unmatched}")

    reasons: list[str] = []
    if partition_leak:
        reasons.append("partition_leak_detected")
    if not control.tiny and len(seen_deaths["train"]) < 1:
        reasons.append("train_deaths=0")
    if held_out_deaths < control.min_held_out_deaths:
        reasons.append(f"held_out_deaths={held_out_deaths} < min={control.min_held_out_deaths}")
    if held_out_death_worlds < control.min_held_out_death_worlds:
        reasons.append(
            f"held_out_death_worlds={held_out_death_worlds} < "
            f"min={control.min_held_out_death_worlds}"
        )
    if held_out_ordinary_roots < control.min_held_out_ordinary_roots:
        reasons.append(
            f"held_out_ordinary_roots={held_out_ordinary_roots} < "
            f"min={control.min_held_out_ordinary_roots}"
        )
    if held_out_anchors < control.min_held_out_anchor_rows:
        reasons.append(
            f"held_out_anchor_rows={held_out_anchors} < min={control.min_held_out_anchor_rows}"
        )
    if train_searched_count < 1:
        reasons.append("train_searched_roots=0")
    if train_anchors < 1:
        reasons.append("train_anchor_rows=0")
    reasons.extend(matching_reasons)

    if partition_leak:
        status = "INVALID"
    elif reasons:
        status = "INCOMPLETE_SUPPORT"
    else:
        status = "PASS"

    return {
        "status": status,
        "reasons": reasons,
        "whole_world_split_disjoint": len(train_worlds & assess_worlds) == 0
        and (not partition_leak),
        "held_out_deaths": held_out_deaths,
        "held_out_distinct_death_worlds": held_out_death_worlds,
        "held_out_ordinary_roots": held_out_ordinary_roots,
        "held_out_anchor_rows": held_out_anchors,
        "train_deaths": len(seen_deaths["train"]),
        "train_distinct_death_worlds": len(seen_death_worlds["train"]),
        "train_searched_roots": train_searched_count,
        "train_anchor_rows": train_anchors,
        "minimum_bars": {
            "min_held_out_deaths": int(control.min_held_out_deaths),
            "min_held_out_death_worlds": int(control.min_held_out_death_worlds),
            "min_held_out_ordinary_roots": int(control.min_held_out_ordinary_roots),
            "min_held_out_anchor_rows": int(control.min_held_out_anchor_rows),
        },
        "per_mix": per_mix,
        "ordinary_matching": ordinary_matching,
    }


def parent_world_cluster_bootstrap(
    values: np.ndarray,
    world_ids: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 3432,
) -> dict[str, Any]:
    """Cluster-bootstrap a root/event mean by resampling distinct parent worlds."""
    vals = np.asarray(values, dtype=np.float64).reshape(-1)
    wids = np.asarray(world_ids, dtype=np.int64).reshape(-1)
    if vals.size == 0:
        return {
            "point_estimate": 0.0,
            "num_units": 0,
            "num_worlds": 0,
            "lb90_one_sided": 0.0,
            "ci90_two_sided": [0.0, 0.0],
            "ci95_two_sided": [0.0, 0.0],
            "p_one_sided_gt_zero": 1.0,
        }
    point = float(np.mean(vals))
    unique_w, inv = np.unique(wids, return_inverse=True)
    G = int(unique_w.size)
    if G <= 1:
        return {
            "point_estimate": point,
            "num_units": int(vals.size),
            "num_worlds": G,
            "lb90_one_sided": point,
            "ci90_two_sided": [point, point],
            "ci95_two_sided": [point, point],
            "p_one_sided_gt_zero": 0.0 if point > 0.0 else 1.0,
        }

    cluster_sums = np.bincount(inv, weights=vals, minlength=G).astype(np.float64)
    cluster_counts = np.bincount(inv, minlength=G).astype(np.float64)
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(resamples), dtype=np.float64)
    chunk_b = 1000
    for start in range(0, int(resamples), chunk_b):
        end = min(int(resamples), start + chunk_b)
        idx = rng.integers(0, G, size=(end - start, G))
        num = cluster_sums[idx].sum(axis=1)
        den = cluster_counts[idx].sum(axis=1)
        draws[start:end] = num / np.maximum(den, 1.0)

    lb90 = float(np.percentile(draws, 10.0))
    ci90 = [float(x) for x in np.percentile(draws, [5.0, 95.0])]
    ci95 = [float(x) for x in np.percentile(draws, [2.5, 97.5])]
    p_gt0 = float((1 + int(np.sum(draws <= 0.0))) / (int(resamples) + 1))
    return {
        "point_estimate": point,
        "num_units": int(vals.size),
        "num_worlds": G,
        "lb90_one_sided": lb90,
        "ci90_two_sided": ci90,
        "ci95_two_sided": ci95,
        "p_one_sided_gt_zero": p_gt0,
    }


def paired_parent_world_diff_bootstrap(
    values_a: np.ndarray,
    values_b: np.ndarray,
    world_ids: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 3432,
) -> dict[str, Any]:
    """Paired parent-world bootstrap for ``mean(values_a) - mean(values_b)`` on identical roots."""
    diffs = np.asarray(values_a, dtype=np.float64) - np.asarray(values_b, dtype=np.float64)
    return parent_world_cluster_bootstrap(diffs, world_ids, resamples=resamples, seed=seed)


def world_grouped_mann_whitney_auc(
    scores: np.ndarray,
    is_pos: np.ndarray,
    is_neg: np.ndarray,
    world_ids: np.ndarray,
    *,
    resamples: int = 10_000,
    seed: int = 3432,
) -> dict[str, Any]:
    """Compute world-grouped cross-world ROC AUC and root-level diagnostic AUC.

    Each distinct positive held-out world ``w_p`` and distinct negative held-out
    world ``w_n`` (``w_p != w_n`` when disjoint pairs exist) receives equal cluster
    weight through its pairwise Mann-Whitney kernel
    ``K(w_p, w_n) = mean_{i in S_p(w_p), j in S_n(w_n)} [1(s_i > s_j) + 0.5 * 1(s_i == s_j)]``,
    eliminating within-world root autocorrelation.
    """
    sc = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos_m = np.asarray(is_pos, dtype=bool).reshape(-1)
    neg_m = np.asarray(is_neg, dtype=bool).reshape(-1)
    wids = np.asarray(world_ids, dtype=np.int64).reshape(-1)

    pos_scores_all = sc[pos_m]
    neg_scores_all = sc[neg_m]
    if pos_scores_all.size == 0 or neg_scores_all.size == 0:
        return {
            "both_classes_supported": False,
            "pos_roots": int(pos_scores_all.size),
            "neg_roots": int(neg_scores_all.size),
            "pos_worlds": 0,
            "neg_worlds": 0,
            "world_grouped_auc": None,
            "root_level_auc": None,
            "ci95_two_sided": None,
        }

    # Exact root-level Mann-Whitney U diagnostic.
    diff_all = pos_scores_all[:, None] - neg_scores_all[None, :]
    root_auc = float(np.mean((diff_all > 0.0) + 0.5 * (diff_all == 0.0)))

    pos_worlds = np.unique(wids[pos_m])
    neg_worlds = np.unique(wids[neg_m])
    Gp = int(pos_worlds.size)
    Gn = int(neg_worlds.size)
    pos_by_w = [sc[pos_m & (wids == int(w))] for w in pos_worlds]
    neg_by_w = [sc[neg_m & (wids == int(w))] for w in neg_worlds]

    kernel = np.zeros((Gp, Gn), dtype=np.float64)
    valid_pair = np.ones((Gp, Gn), dtype=bool)
    for i_p, w_p in enumerate(pos_worlds.tolist()):
        sp = pos_by_w[i_p]
        for j_n, w_n in enumerate(neg_worlds.tolist()):
            sn = neg_by_w[j_n]
            d = sp[:, None] - sn[None, :]
            kernel[i_p, j_n] = float(np.mean((d > 0.0) + 0.5 * (d == 0.0)))
            if w_p == w_n:
                valid_pair[i_p, j_n] = False

    if not np.any(valid_pair):
        valid_pair[:, :] = True

    world_auc = float(np.mean(kernel[valid_pair]))

    # Parent-world bootstrap over all distinct held-out worlds participating in either class.
    all_worlds = np.unique(np.concatenate((pos_worlds, neg_worlds)))
    Gall = int(all_worlds.size)
    p_map = {int(w): idx for idx, w in enumerate(pos_worlds.tolist())}
    n_map = {int(w): idx for idx, w in enumerate(neg_worlds.tolist())}

    if Gall <= 1 or int(resamples) < 1:
        ci95 = [world_auc, world_auc]
    else:
        rng = np.random.default_rng(int(seed))
        draws: list[float] = []
        for _ in range(int(resamples)):
            sampled_w = all_worlds[rng.integers(0, Gall, size=Gall)]
            p_idxs = [p_map[int(w)] for w in sampled_w if int(w) in p_map]
            n_idxs = [n_map[int(w)] for w in sampled_w if int(w) in n_map]
            if not p_idxs or not n_idxs:
                continue
            sub_k = kernel[np.ix_(p_idxs, n_idxs)]
            sub_v = valid_pair[np.ix_(p_idxs, n_idxs)]
            if np.any(sub_v):
                draws.append(float(np.mean(sub_k[sub_v])))
            else:
                draws.append(float(np.mean(sub_k)))
        ci95 = (
            [float(x) for x in np.percentile(draws, [2.5, 97.5])]
            if draws
            else [world_auc, world_auc]
        )

    return {
        "both_classes_supported": True,
        "pos_roots": int(pos_scores_all.size),
        "neg_roots": int(neg_scores_all.size),
        "pos_worlds": Gp,
        "neg_worlds": Gn,
        "world_grouped_auc": world_auc,
        "root_level_auc": root_auc,
        "ci95_two_sided": ci95,
    }


def holm_step_down(
    p_values: Mapping[str, float], *, family_alpha: float = 0.10
) -> dict[str, dict[str, Any]]:
    """Apply Holm-Bonferroni step-down across ``m`` arms for superiority claims."""
    ordered = sorted(p_values.items(), key=lambda kv: (float(kv[1]), kv[0]))
    m = len(ordered)
    out: dict[str, dict[str, Any]] = {}
    still_rejecting = True
    running_adj = 0.0
    for rank_idx, (arm, p_raw) in enumerate(ordered):
        denom = m - rank_idx
        threshold = float(family_alpha) / float(denom)
        running_adj = min(1.0, max(running_adj, float(p_raw) * float(denom)))
        reject = bool(still_rejecting and (float(p_raw) <= threshold))
        if not reject:
            still_rejecting = False
        out[arm] = {
            "rank": rank_idx + 1,
            "p_raw_one_sided": float(p_raw),
            "holm_threshold": threshold,
            "p_holm_adjusted": running_adj,
            "statistically_superior": reject,
        }
    return out


def evaluate_ct_gates(
    *,
    control: CTControl,
    support_report: Mapping[str, Any],
    store: LosslessBoundedSnapshotStore,
    scored: MergedScoredRoots | None,
    anchors: AnchorArchive,
    arm_predictions: Mapping[str, ArmEvaluationPredictions] | None,
    fault_reason: str | None = None,
) -> dict[str, Any]:
    """Evaluate ordered C/T decision gates strictly on held-out worlds and scoring suffixes."""
    if fault_reason is not None or support_report.get("status") == "INVALID":
        return {
            "verdict": "INVALID",
            "fault_reason": fault_reason or ",".join(support_report.get("reasons", [])),
            "support": dict(support_report),
            "gates": {},
            "arms": {},
            "statistical_superiority_holm": {},
        }

    if support_report.get("status") != "PASS":
        return {
            "verdict": "INCOMPLETE_SUPPORT",
            "support": dict(support_report),
            "gates": {},
            "arms": {},
            "statistical_superiority_holm": {},
        }

    if scored is None or arm_predictions is None:
        return {
            "verdict": "INVALID",
            "fault_reason": "missing_scored_roots_or_arm_predictions",
            "support": dict(support_report),
            "gates": {},
            "arms": {},
            "statistical_superiority_holm": {},
        }

    # Verify numerical finiteness on legal actions across scored roots and arm logits.
    for r_i, meta in enumerate(store.metadata):
        leg = scored.legal_mask[r_i]
        if not np.any(leg):
            return {
                "verdict": "INVALID",
                "fault_reason": f"empty_legal_mask_at_root_{meta.root_uid}",
                "support": dict(support_report),
                "gates": {},
                "arms": {},
                "statistical_superiority_holm": {},
            }
        if not (
            np.all(np.isfinite(scored.s_score[r_i, leg]))
            and np.all(np.isfinite(scored.m_score[r_i, leg]))
            and np.all(np.isfinite(scored.s_sel[r_i, leg]))
        ):
            return {
                "verdict": "INVALID",
                "fault_reason": f"non_finite_branch_score_at_root_{meta.root_uid}",
                "support": dict(support_report),
                "gates": {},
                "arms": {},
                "statistical_superiority_holm": {},
            }

    for arm_name, pred in arm_predictions.items():
        if not np.all(np.isfinite(pred.root_logits)):
            return {
                "verdict": "INVALID",
                "fault_reason": f"non_finite_arm_logits_{arm_name}",
                "support": dict(support_report),
                "gates": {},
                "arms": {},
                "statistical_superiority_holm": {},
            }

    # Index held-out assessment roots strictly.
    assess_death_roots_by_event: dict[int, list[tuple[int, RootMetadata]]] = {}
    assess_all_death_root_indices: list[int] = []
    assess_causal_rescue_indices: list[int] = []
    assess_ordinary_indices: list[int] = []
    assess_d100_d300_mask = np.zeros(len(store), dtype=bool)
    assess_ord_mask = np.zeros(len(store), dtype=bool)
    all_root_worlds = np.array([m.world_id for m in store.metadata], dtype=np.int64)

    for r_i, meta in enumerate(store.metadata):
        if meta.partition != "assessment":
            continue
        if meta.kind == "death_lookback":
            assert meta.death_event_id is not None
            assess_death_roots_by_event.setdefault(meta.death_event_id, []).append((r_i, meta))
            assess_all_death_root_indices.append(r_i)
            if meta.lookback in D100_D300_LOOKBACKS:
                assess_d100_d300_mask[r_i] = True
            a_st = int(scored.a_star[r_i])
            a_b8 = int(meta.b8_action)
            if (
                float(scored.s_score[r_i, a_st]) >= 0.75
                and float(scored.s_score[r_i, a_b8]) <= 0.25
            ):
                assess_causal_rescue_indices.append(r_i)
        elif meta.kind == "ordinary":
            assess_ordinary_indices.append(r_i)
            assess_ord_mask[r_i] = True

    causal_set = set(assess_causal_rescue_indices)

    # --- GATE (a): Held-out death event causal rescue rate ---
    event_rescued: list[float] = []
    event_worlds: list[int] = []
    for ev_id in sorted(assess_death_roots_by_event):
        ev_roots = assess_death_roots_by_event[ev_id]
        w_id = ev_roots[0][1].world_id
        has_rescue = any(r_i in causal_set for r_i, _m in ev_roots)
        event_rescued.append(1.0 if has_rescue else 0.0)
        event_worlds.append(w_id)

    event_boot = parent_world_cluster_bootstrap(
        np.asarray(event_rescued, dtype=np.float64),
        np.asarray(event_worlds, dtype=np.int64),
        resamples=control.bootstrap_resamples,
        seed=control.bootstrap_seed,
    )
    n_ev = len(event_rescued)
    k_ev = int(round(sum(event_rescued)))
    rescue_event_rate = float(event_boot["point_estimate"])
    if n_ev > 0 and (100 * k_ev >= 25 * n_ev):
        gate_a_status = "PASS"
    elif n_ev == 0 or (100 * k_ev < 15 * n_ev):
        gate_a_status = "FAIL_STOP_RESCUE_TARGET"
    else:
        gate_a_status = "INCONCLUSIVE_15_TO_25_PCT"

    gate_a = {
        "status": gate_a_status,
        "held_out_deaths": n_ev,
        "rescuable_held_out_deaths": k_ev,
        "rescuable_death_rate": rescue_event_rate,
        "held_out_causal_rescue_roots": len(assess_causal_rescue_indices),
        "pass_threshold": 0.25,
        "stop_threshold": 0.15,
        "world_bootstrap": event_boot,
    }

    # --- GATE (b): Selected-action H300 mass benefit world-bootstrap LB90 > 0 ---
    death_idx_arr = np.asarray(assess_all_death_root_indices, dtype=np.int64)
    if death_idx_arr.size > 0:
        a_star_d = scored.a_star[death_idx_arr]
        a_b8_d = scored.b8_action[death_idx_arr]
        row_arange = np.arange(death_idx_arr.size)
        mass_benefit_all = (
            scored.m_score[death_idx_arr, a_star_d] - scored.m_score[death_idx_arr, a_b8_d]
        )
        mass_worlds_all = all_root_worlds[death_idx_arr]
        _ = row_arange
    else:
        mass_benefit_all = np.zeros(0, dtype=np.float64)
        mass_worlds_all = np.zeros(0, dtype=np.int64)

    mass_boot_all = parent_world_cluster_bootstrap(
        mass_benefit_all,
        mass_worlds_all,
        resamples=control.bootstrap_resamples,
        seed=control.bootstrap_seed,
    )

    causal_idx_arr = np.asarray(assess_causal_rescue_indices, dtype=np.int64)
    if causal_idx_arr.size > 0:
        mass_benefit_causal = (
            scored.m_score[causal_idx_arr, scored.a_star[causal_idx_arr]]
            - scored.m_score[causal_idx_arr, scored.b8_action[causal_idx_arr]]
        )
        mass_boot_causal = parent_world_cluster_bootstrap(
            mass_benefit_causal,
            all_root_worlds[causal_idx_arr],
            resamples=control.bootstrap_resamples,
            seed=control.bootstrap_seed,
        )
    else:
        mass_boot_causal = parent_world_cluster_bootstrap(np.zeros(0), np.zeros(0, dtype=np.int64))

    gate_b_pass = bool(float(mass_boot_all["lb90_one_sided"]) > 0.0)
    gate_b = {
        "status": "PASS" if gate_b_pass else "FAIL",
        "lb90_threshold": 0.0,
        "all_held_out_death_roots_mass_benefit": mass_boot_all,
        "causal_rescue_roots_mass_benefit_diagnostic": mass_boot_causal,
    }

    # --- Per-Arm Evaluation for Gate (c), Gate (d), and T-Arm Separability ---
    assess_anc_idx = anchors.assessment_indices().astype(np.int64)
    assess_anc_b8 = anchors.b8_action[assess_anc_idx].astype(np.int64)
    assess_anc_worlds = anchors.world_id[assess_anc_idx].astype(np.int64)
    ord_idx_arr = np.asarray(assess_ordinary_indices, dtype=np.int64)

    arm_retained_vec: dict[str, np.ndarray] = {}
    arms_report: dict[str, dict[str, Any]] = {}

    for arm_name, pred in arm_predictions.items():
        if causal_idx_arr.size > 0:
            chosen_c = pred.root_greedy_actions[causal_idx_arr]
            s_chosen = scored.s_score[causal_idx_arr, chosen_c]
            retained_vec = (s_chosen >= 0.75).astype(np.float64)
            exact_astar_vec = (chosen_c == scored.a_star[causal_idx_arr]).astype(np.float64)
            ret_boot = parent_world_cluster_bootstrap(
                retained_vec,
                all_root_worlds[causal_idx_arr],
                resamples=control.bootstrap_resamples,
                seed=control.bootstrap_seed,
            )
            exact_rate = float(np.mean(exact_astar_vec))
        else:
            retained_vec = np.zeros(0, dtype=np.float64)
            ret_boot = parent_world_cluster_bootstrap(np.zeros(0), np.zeros(0, dtype=np.int64))
            exact_rate = 0.0
        arm_retained_vec[arm_name] = retained_vec

        # Gate (d): held-out unsearched archive flips + held-out searched ordinary lethal flips
        flips_anc = (pred.assessment_anchor_greedy_actions != assess_anc_b8).astype(np.float64)
        flip_boot = parent_world_cluster_bootstrap(
            flips_anc,
            assess_anc_worlds,
            resamples=control.bootstrap_resamples,
            seed=control.bootstrap_seed,
        )
        archive_flip_rate = float(flip_boot["point_estimate"])

        if ord_idx_arr.size > 0:
            ord_chosen = pred.root_greedy_actions[ord_idx_arr]
            ord_b8 = scored.b8_action[ord_idx_arr]
            ord_flipped = ord_chosen != ord_b8
            ord_s_drop = (
                scored.s_score[ord_idx_arr, ord_b8] - scored.s_score[ord_idx_arr, ord_chosen]
            )
            lethal_mask = ord_flipped & (ord_s_drop >= 0.50)
            lethal_count = int(np.sum(lethal_mask))
            ord_flip_count = int(np.sum(ord_flipped))
        else:
            lethal_count = 0
            ord_flip_count = 0

        anc_flip_count = int(round(float(np.sum(flips_anc))))
        anc_rows = int(flips_anc.size)
        gate_d_pass = bool((1000 * anc_flip_count <= 10 * anc_rows) and lethal_count <= 1)

        # World-grouped D100-D300 separability AUC (detached train-only probe + policy surprise)
        probe_auc_info = world_grouped_mann_whitney_auc(
            pred.root_danger_probe_scores,
            assess_d100_d300_mask,
            assess_ord_mask,
            all_root_worlds,
            resamples=control.bootstrap_resamples,
            seed=control.bootstrap_seed,
        )
        surprise_auc_info = world_grouped_mann_whitney_auc(
            pred.root_b8_neg_log_prob,
            assess_d100_d300_mask,
            assess_ord_mask,
            all_root_worlds,
            resamples=control.bootstrap_resamples,
            seed=control.bootstrap_seed,
        )

        arms_report[arm_name] = {
            "rescue_retention": {
                "rate": float(ret_boot["point_estimate"]),
                "retained_count": int(np.sum(retained_vec)),
                "causal_rescue_denominator": int(retained_vec.size),
                "exact_a_star_match_rate_diagnostic": exact_rate,
                "world_bootstrap": ret_boot,
            },
            "gate_d_ordinary_safety": {
                "status": "PASS" if gate_d_pass else "FAIL",
                "archive_flip_rate": archive_flip_rate,
                "archive_flip_count": int(np.sum(flips_anc)),
                "archive_rows": int(flips_anc.size),
                "archive_flip_max_bar": 0.010,
                "archive_flip_world_bootstrap": flip_boot,
                "searched_ordinary_roots": int(ord_idx_arr.size),
                "searched_ordinary_flip_count": ord_flip_count,
                "searched_ordinary_lethal_flip_count": lethal_count,
                "searched_ordinary_lethal_flip_max_bar": 1,
            },
            "d100_d300_separability": {
                "probe_fitted_on_detached_train_only": bool(pred.danger_probe_fitted),
                "detached_linear_probe_auc": probe_auc_info,
                "policy_b8_surprise_auc_diagnostic": surprise_auc_info,
                "t_arm_auc_bar": 0.83,
                "auc_alone_buys_training": False,
            },
            "train_telemetry": pred.train_telemetry,
        }

    # --- Retention Differences & Holm Step-Down ---
    causal_worlds = (
        all_root_worlds[causal_idx_arr] if causal_idx_arr.size > 0 else np.zeros(0, dtype=np.int64)
    )
    diff_bv8_vs_frozen = paired_parent_world_diff_bootstrap(
        arm_retained_vec["full_bv8"],
        arm_retained_vec["frozen_phi"],
        causal_worlds,
        resamples=control.bootstrap_resamples,
        seed=control.bootstrap_seed,
    )
    diff_topo_vs_frozen = paired_parent_world_diff_bootstrap(
        arm_retained_vec["full_bv8_plus_topology"],
        arm_retained_vec["frozen_phi"],
        causal_worlds,
        resamples=control.bootstrap_resamples,
        seed=control.bootstrap_seed,
    )
    diff_topo_vs_bv8 = paired_parent_world_diff_bootstrap(
        arm_retained_vec["full_bv8_plus_topology"],
        arm_retained_vec["full_bv8"],
        causal_worlds,
        resamples=control.bootstrap_resamples,
        seed=control.bootstrap_seed,
    )

    holm_vs_frozen = holm_step_down(
        {
            "full_bv8": float(diff_bv8_vs_frozen["p_one_sided_gt_zero"]),
            "full_bv8_plus_topology": float(diff_topo_vs_frozen["p_one_sided_gt_zero"]),
        },
        family_alpha=0.10,
    )

    ret_frozen = arms_report["frozen_phi"]["rescue_retention"]["rate"]
    ret_bv8 = arms_report["full_bv8"]["rescue_retention"]["rate"]
    ret_topo = arms_report["full_bv8_plus_topology"]["rescue_retention"]["rate"]
    causal_denom = int(causal_idx_arr.size)
    k_frozen = int(arms_report["frozen_phi"]["rescue_retention"]["retained_count"])
    k_bv8 = int(arms_report["full_bv8"]["rescue_retention"]["retained_count"])
    k_topo = int(arms_report["full_bv8_plus_topology"]["rescue_retention"]["retained_count"])

    # Exact integer comparisons over common denominator causal_denom prevent IEEE-754
    # boundary rejection (e.g. 7/20 - 4/20 == 0.14999999999999997 < 0.15 in float64).
    gate_c_bv8_pass = bool(
        causal_denom > 0
        and (100 * k_bv8 >= 30 * causal_denom)
        and (100 * (k_bv8 - k_frozen) >= 15 * causal_denom)
    )
    gate_d_bv8_pass = arms_report["full_bv8"]["gate_d_ordinary_safety"]["status"] == "PASS"

    topo_auc_info = arms_report["full_bv8_plus_topology"]["d100_d300_separability"][
        "detached_linear_probe_auc"
    ]
    if topo_auc_info["both_classes_supported"]:
        topo_auc_pass = bool(
            topo_auc_info["world_grouped_auc"] is not None
            and float(topo_auc_info["world_grouped_auc"]) >= 0.83
        )
    else:
        topo_auc_pass = True

    topo_ret_pass = bool(
        causal_denom > 0
        and (100 * k_topo >= 30 * causal_denom)
        and (100 * (k_topo - k_frozen) >= 15 * causal_denom)
        and (100 * (k_topo - k_bv8) >= 15 * causal_denom)
    )
    gate_d_topo_pass = (
        arms_report["full_bv8_plus_topology"]["gate_d_ordinary_safety"]["status"] == "PASS"
    )

    gate_c = {
        "full_bv8_status": "PASS" if gate_c_bv8_pass else "FAIL",
        "min_retention_bar": 0.30,
        "min_gain_over_frozen_phi_pp": 0.15,
        "frozen_phi_retention": ret_frozen,
        "full_bv8_retention": ret_bv8,
        "full_bv8_minus_frozen_phi": diff_bv8_vs_frozen,
    }
    t_arm_gate = {
        "status": (
            "PASS"
            if (
                gate_a_status == "PASS"
                and gate_b_pass
                and topo_ret_pass
                and gate_d_topo_pass
                and topo_auc_pass
            )
            else "FAIL"
        ),
        "retention_rate": ret_topo,
        "min_gain_over_full_bv8_pp": 0.15,
        "topo_minus_frozen_phi": diff_topo_vs_frozen,
        "topo_minus_full_bv8": diff_topo_vs_bv8,
        "retention_gate_pass": topo_ret_pass,
        "gate_d_pass": gate_d_topo_pass,
        "d100_d300_auc_gate_pass": topo_auc_pass,
        "d100_d300_world_grouped_auc": topo_auc_info["world_grouped_auc"],
        "auc_alone_buys_training": False,
    }

    # --- Ordered Overall Verdict ---
    if gate_a_status == "FAIL_STOP_RESCUE_TARGET":
        verdict = "STOP_RESCUE_TARGET_PROGRAM"
    elif gate_a_status == "INCONCLUSIVE_15_TO_25_PCT":
        verdict = "INCONCLUSIVE"
    elif not gate_b_pass:
        verdict = "FAIL_GATE_B_MASS_BENEFIT"
    else:
        bv8_all_pass = gate_c_bv8_pass and gate_d_bv8_pass
        topo_all_pass = topo_ret_pass and gate_d_topo_pass and topo_auc_pass
        if bv8_all_pass and topo_all_pass:
            verdict = "GO_BV8_AND_TOPOLOGY"
        elif bv8_all_pass:
            verdict = "GO_BV8_ONLY"
        elif topo_all_pass:
            verdict = "GO_TOPOLOGY_ONLY"
        else:
            verdict = "FAIL_DISTILLATION_RETENTION_OR_SAFETY"

    return {
        "verdict": verdict,
        "support": dict(support_report),
        "gates": {
            "gate_a_causal_rescue_existence": gate_a,
            "gate_b_selected_mass_benefit": gate_b,
            "gate_c_bv8_rescue_retention": gate_c,
            "gate_t_dynamic_topology": t_arm_gate,
        },
        "arms": arms_report,
        "statistical_superiority_holm": {
            "note": (
                "Holm step-down (m=2, familywise one-sided alpha=0.10) applied only for "
                "statistical superiority claims across trainable arms."
            ),
            "vs_frozen_phi": holm_vs_frozen,
            "topo_vs_full_bv8_paired_bootstrap": diff_topo_vs_bv8,
        },
    }
