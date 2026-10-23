"""The sole A2 scoring look: complete Mac-verified union, paired world bootstrap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

from research.a100_a2.protocol import (
    DOSES,
    ENCODERS,
    HORIZONS,
    NAMESPACE,
    SEEDS,
    TARGETS,
    allocation,
    read_ledger,
)
from research.bl_ppo.exports import digest, export_json, verify_export

TABLE_COUNTS = {"t5": (19976, 342), "a5": (296, None)}
A5_DECISIVE_COUNTS = (225, 21)


def _metric(y: np.ndarray, p: np.ndarray) -> dict[str, float | None]:
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError("invalid probabilities")
    return {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None,
        "brier": float(np.mean((y - p) ** 2)),
    }


def kill_decisions(
    delta_auc: float, lower_bound: float, slope: float, relative_brier_improvement: float
) -> dict[str, Any]:
    """Frozen strict inequalities; equality is never silently rounded across a bar."""
    return {
        "scale_closed": delta_auc < 0.02 and lower_bound <= 0 and slope < 0.01,
        "target": "KEEP_HAZARDS" if relative_brier_improvement >= 0.1 else "KEEP_SINGLE_HORIZON",
    }


def _agreement(row: dict[str, np.ndarray]) -> np.ndarray:
    p = 1 - row["p_d300"]
    mask = row["legal"] & row["observed"]
    survivors = mask & (p >= 0.5)
    action = np.where(
        survivors.any(1),
        np.where(survivors, row["q"], -np.inf).argmax(1),
        np.where(mask, p, -np.inf).argmax(1),
    )
    return (action == row["a_star"])[row["decisive"].astype(bool)]


def _bootstrap(rows: dict[str, dict[str, np.ndarray]], n: int) -> tuple[dict, dict]:
    """Sort each score once; resample whole worlds with common paired weights."""
    first = next(iter(rows.values()))
    mask = first["observed"].astype(bool) & first["legal"].astype(bool)
    worlds, wi = np.unique(first["world"].astype(str), return_inverse=True)
    branch_world = np.broadcast_to(wi[:, None], mask.shape)[mask]
    y = first["died"][mask].astype(float)
    rng = np.random.default_rng(20261022)
    weights = rng.multinomial(len(worlds), np.full(len(worlds), 1 / len(worlds)), n)
    summaries, draws = {}, {}
    for key, row in rows.items():
        p = row["p_d300"][mask].astype(float)
        point = _metric(y, p)
        order = np.argsort(p, kind="stable")
        sy = y[order]
        starts = np.r_[0, np.flatnonzero(np.diff(p[order])) + 1]
        errors = (y - p) ** 2
        aucs, briers = [], []
        for count in weights:
            w = count[branch_world]
            sw = w[order]
            pos = np.add.reduceat(sw * sy, starts)
            neg = np.add.reduceat(sw * (1 - sy), starts)
            denom = pos.sum() * neg.sum()
            aucs.append(
                float(np.sum(pos * (np.cumsum(neg) - neg / 2)) / denom) if denom else np.nan
            )
            briers.append(float(np.dot(w, errors) / w.sum()))
        a = np.asarray(aucs)
        b = np.asarray(briers)
        draws[key] = {"auc": a, "brier": b}
        point["auc_ci95"] = np.nanpercentile(a, [2.5, 97.5]).tolist()
        point["brier_ci95"] = np.percentile(b, [2.5, 97.5]).tolist()
        point["undefined_auc_bootstraps"] = int(np.isnan(a).sum())
        point["parent_worlds"] = len(worlds)
        point["branches"] = int(mask.sum())
        point["decisive_rows"] = int(row["decisive"].sum())
        decisive = row["decisive"].astype(bool)
        point["rescue_bearing_worlds"] = len(np.unique(row["world"][decisive]))
        teacher_choices = (
            decisive.any()
            and (row["a_star"][decisive] >= 0).all()
            and np.isfinite(row["q"][decisive]).all()
        )
        if teacher_choices:
            agree = _agreement(row)
            dw = wi[decisive]
            agreement_draws = np.asarray(
                [np.dot(w[dw], agree) / w[dw].sum() if w[dw].sum() else np.nan for w in weights]
            )
            point["decisive_agreement"] = float(agree.mean())
            point["agreement_ci95"] = np.nanpercentile(agreement_draws, [2.5, 97.5]).tolist()
        else:
            point["decisive_agreement"] = None
            point["agreement_status"] = "unavailable: no recorded teacher choice/Q on this table"
        if "p_hazards" in row:
            scores = []
            for h, horizon in enumerate(HORIZONS):
                truth = row["died"].astype(bool) & (row["alive_steps"] < horizon)
                scores.append(
                    {"horizon": horizon, **_metric(truth[mask], row["p_hazards"][:, :, h][mask])}
                )
            point["multi_horizon"] = scores
        else:
            point["multi_horizon"] = "unavailable for single-horizon head"
        summaries[key] = point
    return summaries, draws


def analyze(ledger: Path, alloc: str, output: Path, n_boot: int = 2000) -> dict[str, Any]:
    """Reject incomplete/mixed data before the only scoring look."""
    units = read_ledger(ledger, NAMESPACE, alloc)
    specs = allocation()
    if set(units) != set(range(len(specs))):
        raise ValueError("INCOMPLETE: all 480 frozen fit units required before scoring")
    loaded = {}
    lineage = None
    for root in {Path(item["path"]).parent for item in units.values()}:
        verify_export(root / "materialization.json")
        meta = json.loads((root / "materialization.json").read_text())
        if meta.get("synthetic"):
            raise ValueError("synthetic materialization cannot enter registered look")
        if lineage is None:
            lineage = meta
        elif lineage != meta:
            raise ValueError("materialization lineage differs across attempts")
    for spec in specs:
        i = spec["index"]
        item = units[i]
        path = Path(item["path"])
        if verify_export(path) != item["sha256"] or path.stat().st_size != item["size"]:
            raise ValueError("verified union changed")
        with np.load(path, allow_pickle=False) as z:
            row = {k: z[k] for k in z.files}
        if (
            str(row["namespace"]) != NAMESPACE
            or str(row["allocation"]) != alloc
            or int(row["index"]) != i
        ):
            raise ValueError("foreign namespace/allocation/unit in predictions")
        for key in ("table", "encoder", "target", "dose", "seed", "fold"):
            value = row[key]
            if value.ndim == 0:
                ok = value.item() == spec[key]
            else:
                ok = np.all(value == spec[key])
            if not ok:
                raise ValueError("unit factorial identity mismatch")
        loaded[i] = row
    report: dict[str, Any] = {
        "schema": "a2-one-look/v1",
        "allocation": alloc,
        "ledger_sha256": digest(ledger),
        "n_boot": n_boot,
        "evidence": "exploratory historical bank; no served-gain claim",
        "materialization": lineage,
        "tables": {},
    }
    for table in ("t5", "a5"):
        grouped = {}
        reference = None
        for encoder in ENCODERS:
            for target in TARGETS:
                for dose in DOSES:
                    seed_rows = []
                    for seed in SEEDS:
                        parts = [
                            loaded[s["index"]]
                            for s in specs
                            if s["table"] == table
                            and s["encoder"] == encoder
                            and s["target"] == target
                            and s["dose"] == dose
                            and s["seed"] == seed
                        ]
                        keys = [
                            "row_index",
                            "world",
                            "fold",
                            "legal",
                            "observed",
                            "died",
                            "alive_steps",
                            "decisive",
                            "a_star",
                            "q",
                            "p_d300",
                        ]
                        if target == "hazard":
                            keys.append("p_hazards")
                        merged = {k: np.concatenate([p[k] for p in parts]) for k in keys}
                        order = np.argsort(merged["row_index"])
                        merged = {k: v[order] for k, v in merged.items()}
                        expected_n, expected_worlds = TABLE_COUNTS[table]
                        if not np.array_equal(merged["row_index"], np.arange(expected_n)):
                            raise ValueError("INCOMPLETE: exact registered row coverage required")
                        if (
                            expected_worlds is not None
                            and len(np.unique(merged["world"])) != expected_worlds
                        ):
                            raise ValueError("INCOMPLETE: registered parent-world count differs")
                        if set(merged["fold"].tolist()) != set(range(5)):
                            raise ValueError("INCOMPLETE: frozen fold coverage differs")
                        for world in np.unique(merged["world"]):
                            if len(np.unique(merged["fold"][merged["world"] == world])) != 1:
                                raise ValueError("ancestry crosses held-out folds")
                        if table == "a5":
                            decisive = merged["decisive"].astype(bool)
                            if (
                                int(decisive.sum()) != A5_DECISIVE_COUNTS[0]
                                or len(np.unique(merged["world"][decisive]))
                                != A5_DECISIVE_COUNTS[1]
                            ):
                                raise ValueError("INCOMPLETE: A5 decisive rows/lives differ")
                        if reference is None:
                            reference = merged
                        for k in keys:
                            if k.startswith("p_"):
                                continue
                            if not np.array_equal(merged[k], reference[k], equal_nan=k == "q"):
                                raise ValueError("paired rows/labels differ across arms")
                        seed_rows.append(merged)
                    row = seed_rows[0].copy()
                    for key in ("p_d300", "p_hazards"):
                        if key in row:
                            row[key] = (seed_rows[0][key] + seed_rows[1][key]) / 2
                    grouped[f"{encoder}/{target}/{dose}"] = row
        summaries, draws = _bootstrap(grouped, n_boot)
        slopes, hazards = {}, {}
        for encoder in ENCODERS:
            for target in TARGETS:
                key = f"{encoder}/{target}"
                lo, hi = key + "/50", key + "/100"
                a, b = summaries[lo]["auc"], summaries[hi]["auc"]
                if a is None or b is None:
                    raise ValueError("INVALID: undefined primary AUC")
                slopes[key] = {
                    "auc_delta": b - a,
                    "ci95": np.nanpercentile(
                        draws[hi]["auc"] - draws[lo]["auc"], [2.5, 97.5]
                    ).tolist(),
                }
            for dose in DOSES:
                single, hazard = f"{encoder}/d300/{dose}", f"{encoder}/hazard/{dose}"
                improvement = 1 - summaries[hazard]["brier"] / summaries[single]["brier"]
                hazards[f"{encoder}/{dose}"] = {
                    "relative_brier_improvement": improvement,
                    "decision": kill_decisions(0, 0, 0, improvement)["target"],
                    "ci95": np.percentile(
                        1 - draws[hazard]["brier"] / draws[single]["brier"], [2.5, 97.5]
                    ).tolist(),
                }
        m, s = "m_random/d300/100", "s_random/d300/100"
        delta = summaries[m]["auc"] - summaries[s]["auc"]
        ci = np.nanpercentile(draws[m]["auc"] - draws[s]["auc"], [2.5, 97.5]).tolist()
        closed = kill_decisions(delta, ci[0], slopes["m_random/d300"]["auc_delta"], 0)[
            "scale_closed"
        ]
        report["tables"][table] = {
            "arms": summaries,
            "slopes": slopes,
            "hazard_pairs": hazards,
            "scale_contrast": {"delta_auc": delta, "ci95": ci, "closed": closed},
        }
    report["scale_decision"] = (
        "CLOSED_WP4_ONLY_WP2_COUNTERFACTUAL_LABELS"
        if report["tables"]["t5"]["scale_contrast"]["closed"]
        else "NOT_CLOSED_EXPLORATORY"
    )
    report["target_decision"] = report["tables"]["t5"]["hazard_pairs"]["s_random/100"]["decision"]
    export_json(output, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--allocation", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.ledger, args.allocation, args.out)
    print(json.dumps({k: result[k] for k in ("scale_decision", "target_decision")}))


if __name__ == "__main__":
    main()
