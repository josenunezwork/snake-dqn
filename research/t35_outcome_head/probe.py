"""T35 outcome-head probe entrypoint (registered in PREREGISTRATION.md).

Example (Mac, registered run)::

    OMP_NUM_THREADS=1 NUMBA_NUM_THREADS=1 nice -n 10 ./venv/bin/python \
        -m research.t35_outcome_head.probe --exports-root EXPORTS --checkpoint B8 \
        --out OUT --workers 2 --threads 1
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import platform
import time
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from scipy.stats import spearmanr

from research.t34_exit_b8.exports import load_world

from . import metrics as M
from .data import assign_folds, discover_worlds, world_targets
from .features import build_world_features, load_world_features
from .model import L2_GRID, LITERAL, PRIMARY, Recipe, predict, train_head

N_FOLDS = 5
INNER_FOLDS = 4
Q_TOL = 1e-3
Q_BAD_FRACTION = 0.01
MAX_DROP_RATE = 0.02


def _worker(args: tuple[str, str, str, int]) -> dict[str, Any]:
    return build_world_features(*args)


def _zscore(values: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    sel = values[mask]
    mu = float(sel.mean()) if sel.size else 0.0
    sd = float(sel.std()) if sel.size else 1.0
    return mu, (sd if sd > 1e-6 else 1.0)


def _fit(
    feats: np.ndarray,
    y_s: np.ndarray,
    y_m: np.ndarray,
    mask: np.ndarray,
    train_rows: np.ndarray,
    recipe: Recipe,
    seed: int,
    device: torch.device,
    l2: float | None = None,
):
    mu, sd = _zscore(y_m[train_rows], mask[train_rows])
    return train_head(
        feats[train_rows],
        y_s[train_rows],
        (y_m[train_rows] - mu) / sd,
        mask[train_rows],
        recipe,
        seed,
        device,
        l2=l2,
    )


def select_l2(
    feats: np.ndarray,
    y_s: np.ndarray,
    y_m: np.ndarray,
    mask: np.ndarray,
    train_rows: np.ndarray,
    world_names: np.ndarray,
    device: torch.device,
) -> tuple[float, dict[float, float]]:
    """Pick the L2 strength by inner world-level CV over the outer-training worlds only.

    Score = pooled survival AUC over masked (row, action) entries of the inner
    held-out worlds; ties go to the larger (more conservative) L2.
    """
    train_rows = np.asarray(train_rows)
    names = world_names[train_rows]
    uniq = sorted(set(names.tolist()))
    inner = assign_folds(uniq, [True] * len(uniq), INNER_FOLDS)
    inner_fold = np.asarray([inner[w] for w in names])
    scores: dict[float, float] = {}
    for lam in L2_GRID:
        pred = np.full((train_rows.size, 6), np.nan, dtype=np.float64)
        for k in range(INNER_FOLDS):
            fit_rows, held = train_rows[inner_fold != k], np.flatnonzero(inner_fold == k)
            if fit_rows.size == 0 or held.size == 0:
                continue
            head = _fit(feats, y_s, y_m, mask, fit_rows, PRIMARY, 0, device, l2=lam)
            pred[held] = predict(head, feats[train_rows[held]], device)[0]
        ok = np.isfinite(pred).all(axis=1)
        msk = mask[train_rows][ok]
        scores[lam] = M.auc(pred[ok][msk], y_s[train_rows][ok][msk])
    finite = {k: v for k, v in scores.items() if np.isfinite(v)}
    if not finite:
        return max(L2_GRID), scores
    best = max(finite.values())
    return max(k for k, v in finite.items() if v == best), scores


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.perf_counter()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    cache = out / "features"
    cache.mkdir()
    paths = sorted(p for root in args.exports_root for p in discover_worlds(root))
    if args.max_worlds:
        paths = paths[: args.max_worlds]
    if args.expect_worlds and len(paths) != args.expect_worlds:
        raise SystemExit(f"expected {args.expect_worlds} worlds, found {len(paths)}")
    if not paths:
        raise SystemExit("no complete world exports found")
    targets = {}
    for p in paths:
        t = world_targets(load_world(p))
        if t.world_id in targets:
            raise ValueError(f"duplicate world id {t.world_id}")
        targets[t.world_id] = t
    t_targets = time.perf_counter() - started

    jobs = [(str(p), str(args.checkpoint), str(cache), int(args.threads)) for p in paths]
    t0 = time.perf_counter()
    if args.workers > 1:
        with mp.get_context("spawn").Pool(args.workers) as pool:
            receipts = pool.map(_worker, jobs, chunksize=1)
    else:
        receipts = [_worker(j) for j in jobs]
    t_features = time.perf_counter() - t0
    dropped = [r for r in receipts if not r["accepted"]]
    drop_rate = len(dropped) / len(receipts)
    report: dict[str, Any] = {
        "schema": "t35-outcome-head-probe/v1",
        "mode": "preflight" if args.preflight else "registered",
        "worlds": len(receipts),
        "dropped": dropped,
        "drop_rate": drop_rate,
        "receipts": receipts,
        "host": {"platform": platform.platform(), "cpu_count": os.cpu_count()},
        "workers": args.workers,
        "threads": args.threads,
    }
    if drop_rate > MAX_DROP_RATE:
        report["verdict"] = "ABORT"
        report["abort_reason"] = "replay drop rate above 2%"
        return report

    # Assemble accepted worlds in a fixed order.
    ids = [r["world_id"] for r in receipts if r["accepted"]]
    cols: dict[str, list[np.ndarray]] = {}
    for wi, wid in enumerate(ids):
        f = load_world_features(cache, wid)
        t = targets[wid]
        if not np.array_equal(f["decisions"], t.decisions):
            raise ValueError(f"{wid}: cached decisions differ from targets")
        parts = {
            "feats": f["feats"],
            "q": f["q"],
            "adv_mask": f["mask"],
            "world": np.full(t.n, wi, dtype=np.int64),
            "searched": t.searched,
            "decisive": t.decisive,
            "a_star": t.a_star,
            "executed": t.executed_action,
            "legal": t.legal,
            "q_export": t.q_export,
            "t_tested": t.t_tested,
            "t_survive": t.t_survive,
            "t_mass": t.t_mass,
            "t_alive": t.t_alive_steps,
            "own_valid": t.own_valid,
            "own_survive": t.own_survive,
            "own_mass": t.own_mass,
        }
        for k, v in parts.items():
            cols.setdefault(k, []).append(v)
    D = {k: np.concatenate(v) for k, v in cols.items()}
    n = D["feats"].shape[0]

    # Integrity: replayed B8 Q must match the exported teacher-time Q.
    has_q = np.isfinite(D["q_export"]).all(axis=1)
    dq = np.abs(D["q"][has_q] - D["q_export"][has_q]).max(axis=1)
    q_bad = float((dq > Q_TOL).mean()) if dq.size else 1.0
    report["integrity"] = {
        "rows": int(n),
        "rows_with_export_q": int(has_q.sum()),
        "q_max_abs_diff": float(dq.max()) if dq.size else None,
        "q_fraction_above_tol": q_bad,
        "searched_rows": int(D["searched"].sum()),
        "decisive_rows": int(D["decisive"].sum()),
        "worlds_with_decisive": int(np.unique(D["world"][D["decisive"]]).size),
    }
    if q_bad > Q_BAD_FRACTION:
        report["verdict"] = "ABORT"
        report["abort_reason"] = "replayed B8 Q does not match exported Q"
        return report

    folds_by_world = assign_folds(ids, [bool(targets[w].searched.any()) for w in ids], N_FOLDS)
    fold = np.asarray([folds_by_world[ids[w]] for w in D["world"]], dtype=np.int64)
    world_names = np.asarray([ids[w] for w in D["world"]])
    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    report["device"] = str(device)

    # Control-arm target: executed action only.
    own_mask = np.zeros((n, 6), dtype=bool)
    own_mask[np.arange(n), D["executed"]] = True
    own_mask &= D["own_valid"][:, None]
    own_s = np.where(own_mask, D["own_survive"][:, None], 0.0).astype(np.float32)
    own_m = np.where(own_mask, D["own_mass"][:, None], 0.0).astype(np.float32)

    arms = {
        "teacher_primary_s0": ("teacher", PRIMARY, 0),
        "control_primary_s0": ("control", PRIMARY, 0),
        "teacher_primary_s1": ("teacher", PRIMARY, 1),
        "teacher_primary_s2": ("teacher", PRIMARY, 2),
        "teacher_literal_s0": ("teacher", LITERAL, 0),
    }
    preds = {a: (np.full((n, 6), np.nan), np.full((n, 6), np.nan)) for a in arms}
    fold_info = []
    t1 = time.perf_counter()
    for k in range(N_FOLDS):
        test = np.flatnonzero((fold == k) & D["decisive"])
        tr_teacher = np.flatnonzero((fold != k) & D["searched"])
        tr_control = np.flatnonzero((fold != k) & D["own_valid"])
        fold_info.append(
            {
                "fold": k,
                "worlds": sorted(w for w in ids if folds_by_world[w] == k),
                "test_decisive_rows": int(test.size),
                "train_teacher_rows": int(tr_teacher.size),
                "train_control_rows": int(tr_control.size),
                "train_control_deaths": int((D["own_survive"][tr_control] == 0).sum()),
            }
        )
        if test.size == 0:
            continue
        chosen: dict[str, Any] = {}
        for arm, (kind, recipe, seed) in arms.items():
            if kind == "teacher":
                ys, ym, mk, tr = D["t_survive"], D["t_mass"], D["t_tested"], tr_teacher
            else:
                ys, ym, mk, tr = own_s, own_m, own_mask, tr_control
            if tr.size == 0:
                continue
            lam = None
            if recipe is PRIMARY:
                key = kind
                if key not in chosen:
                    chosen[key] = select_l2(D["feats"], ys, ym, mk, tr, world_names, device)
                lam = chosen[key][0]
            head = _fit(D["feats"], ys, ym, mk, tr, recipe, seed, device, l2=lam)
            p, m = predict(head, D["feats"][test], device)
            preds[arm][0][test], preds[arm][1][test] = p, m
        fold_info[-1]["l2_selection"] = {
            arm_kind: {"chosen": v[0], "inner_auc": {str(a): b for a, b in v[1].items()}}
            for arm_kind, v in chosen.items()
        }
    t_fit = time.perf_counter() - t1
    report["folds"] = fold_info
    report["timing_seconds"] = {
        "targets": t_targets,
        "features_wall": t_features,
        "fit_eval": t_fit,
        "total": time.perf_counter() - started,
    }
    if args.preflight:
        report["verdict"] = "PREFLIGHT_ONLY"
        return report

    E = np.flatnonzero(D["decisive"] & np.isfinite(preds["teacher_primary_s0"][0]).all(axis=1))
    tested = D["t_tested"]
    y = D["t_survive"]

    def pooled_auc(scores: np.ndarray, rows: np.ndarray, sel: np.ndarray | None = None) -> float:
        msk = tested[rows] if sel is None else (tested[rows] & sel[rows])
        return M.auc(scores[rows][msk], y[rows][msk])

    def agree(arm_p: np.ndarray, rows: np.ndarray) -> float:
        a = M.outcome_argmax(arm_p[rows], D["q"][rows], D["legal"][rows])
        return float(np.mean(a == D["a_star"][rows]))

    non_immediate = ~((y == 0) & (D["t_alive"] <= 1))
    results: dict[str, Any] = {"eval_rows": int(E.size), "eval_pairs": int(tested[E].sum())}
    for arm in arms:
        p, m = preds[arm]
        if not np.isfinite(p[E]).all():
            results[arm] = {"missing": True}
            continue
        argm = M.outcome_argmax(p[E], D["q"][E], D["legal"][E])
        surv_set = (y[E] > 0)[np.arange(E.size), argm]
        mz_true = D["t_mass"][E][tested[E]]
        results[arm] = {
            "auc_pooled": pooled_auc(p, E),
            "auc_non_immediate": pooled_auc(p, E, non_immediate),
            "auc_within_row": M.within_row_auc(p[E], y[E], tested[E]),
            "agree_a_star": float(np.mean(argm == D["a_star"][E])),
            "argmax_in_teacher_survivors": float(np.mean(surv_set)),
            "agree_pure_argmax_p": float(np.mean(np.argmax(p[E], axis=1) == D["a_star"][E])),
            "mass_spearman": float(spearmanr(m[E][tested[E]], mz_true).statistic),
        }
    results["baseline_b8_q"] = {
        "auc_pooled": pooled_auc(D["q"], E),
        "auc_non_immediate": pooled_auc(D["q"], E, non_immediate),
        "agree_a_star_argmax_q": float(np.mean(np.argmax(D["q"][E], 1) == D["a_star"][E])),
    }
    results["baseline_advisory_mask"] = {
        "auc_pooled": pooled_auc(D["adv_mask"].astype(np.float64), E),
    }
    pt, pc = preds["teacher_primary_s0"][0], preds["control_primary_s0"][0]
    widx = D["world"][E]
    auc_t = results["teacher_primary_s0"]["auc_pooled"]
    auc_c = results["control_primary_s0"]["auc_pooled"]
    ag = results["teacher_primary_s0"]["agree_a_star"]
    reps = int(args.bootstrap_reps)
    ci_auc = M.world_bootstrap(widx, lambda r: pooled_auc(pt, E[r]), reps)
    ci_margin = M.world_bootstrap(widx, lambda r: pooled_auc(pt, E[r]) - pooled_auc(pc, E[r]), reps)
    ci_agree = M.world_bootstrap(widx, lambda r: agree(pt, E[r]), reps)
    primary = {
        "auc_teacher": auc_t,
        "auc_teacher_ci95": ci_auc,
        "agree_a_star": ag,
        "agree_a_star_ci95": ci_agree,
        "auc_control": auc_c,
        "margin_teacher_minus_control": auc_t - auc_c,
        "margin_ci95": ci_margin,
        "eval_worlds": int(np.unique(widx).size),
    }
    report["primary"] = primary
    report["secondary"] = results
    report["verdict"] = M.verdict(auc_t, ci_auc[0], ag, auc_t - auc_c)
    report["timing_seconds"]["total"] = time.perf_counter() - started
    return report


def main(argv: Iterable[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exports-root", required=True, nargs="+")
    ap.add_argument("--expect-worlds", type=int, default=0)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--out", required=True, help="fresh output directory (create-only)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--device", default="auto")
    ap.add_argument("--max-worlds", type=int, default=0)
    ap.add_argument("--bootstrap-reps", type=int, default=2000)
    ap.add_argument("--preflight", action="store_true", help="plumbing/timing only, no metrics")
    args = ap.parse_args(argv)
    torch.set_num_threads(int(args.threads))
    report = run(args)
    path = Path(args.out) / "result.json"
    with path.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True, default=float)
        stream.write("\n")
    summary = {
        k: report.get(k)
        for k in ("mode", "verdict", "abort_reason", "worlds", "drop_rate", "timing_seconds")
    }
    if report.get("primary"):
        summary["primary"] = report["primary"]
    print(json.dumps(summary, indent=2, default=float))
    return 0 if report["verdict"] not in ("ABORT",) else 2


if __name__ == "__main__":
    raise SystemExit(main())
