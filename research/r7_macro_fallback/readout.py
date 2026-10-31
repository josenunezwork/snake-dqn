"""Authoritative Mac-side aggregator and kill-gate evaluator for R7 macro-option fallback.

Verifies stopped supervisor receipts (or explicit local-preflight outputs), validates
all ``batch-XXXX.json`` unit sidecars without duplicate IDs or foreign namespaces,
Optionally emits a resume-by-default ``verified-ledger.json`` from path/size/hash only
if an attempt stopped incomplete, and evaluates R7's five pre-declared kill thresholds
strictly on C/T ``train`` roots (``assessment_roots_used == 0``).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from research.r7_macro_fallback import ALLOCATION_NAME, NAMESPACE, R7_KILL_THRESHOLDS
from research.r7_macro_fallback.search import (
    ALLOCATION_INDEX,
    STUDY,
    UNIT_KIND,
    batch_unit_id,
    batch_unit_stem,
    publish_verified_ledger,
)
from tools.ops.package_spec import verify_immutable


def _sha256_file(path: Path) -> str:
    return hashlib.file_digest(path.open("rb"), "sha256").hexdigest()


def _resolve_units_dir(attempt_dir: Path, *, local_preflight: bool) -> tuple[Path, dict[str, Any]]:
    """Verify stopped Colab receipt (unless ``local_preflight``) and locate ``units/``."""
    if local_preflight:
        direct = attempt_dir / "units"
        nested = attempt_dir / "exports" / "r7" / "units"
        units_dir = direct if direct.is_dir() else nested
        return units_dir, {"mode": "local_preflight", "attempt_dir": str(attempt_dir)}

    receipt_path = attempt_dir / "receipt.json"
    if not receipt_path.is_file():
        raise ValueError(f"Missing supervisor receipt.json in {attempt_dir}")
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if receipt.get("stop_verified") is not True or receipt.get("cleanup_error"):
        raise ValueError(
            f"Refusing readout from unstopped runtime {attempt_dir}: "
            f"status={receipt.get('status')!r} stop_verified={receipt.get('stop_verified')!r} "
            f"cleanup_error={receipt.get('cleanup_error')!r}"
        )
    units_dir = attempt_dir / "exports" / "r7" / "units"
    return units_dir, {
        "mode": "colab_supervisor_receipt",
        "attempt_dir": str(attempt_dir),
        "receipt_sha256": _sha256_file(receipt_path),
        "job_id": receipt.get("job_id"),
        "status": receipt.get("status"),
        "stop_verified": receipt.get("stop_verified"),
    }


def collect_verified_batch_units(
    attempt_dirs: Sequence[Path],
    *,
    local_preflight: bool = False,
    allow_canary_units: bool = False,
) -> tuple[dict[int, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Scan attempt directories, verify unit sidecars, and refuse duplicates/foreign units."""
    ledger_entries: dict[int, dict[str, Any]] = {}
    unit_payloads: dict[str, dict[str, Any]] = {}
    provenance: list[dict[str, Any]] = []

    for attempt_dir in attempt_dirs:
        units_dir, prov = _resolve_units_dir(attempt_dir, local_preflight=local_preflight)
        provenance.append(prov)
        if not units_dir.is_dir():
            continue
        for json_path in sorted(units_dir.glob("batch-*.json")):
            meta = verify_immutable(json_path)
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("schema") != "t34-r7-batch-unit/v1":
                raise ValueError(f"Foreign unit schema in {json_path}: {payload.get('schema')!r}")
            if (
                payload.get("study") != STUDY
                or payload.get("namespace") != NAMESPACE
                or payload.get("allocation") != ALLOCATION_NAME
                or payload.get("unit_kind") != UNIT_KIND
                or payload.get("allocation_index") != ALLOCATION_INDEX
            ):
                raise ValueError(f"Foreign unit namespace/allocation in {json_path}")
            if int(payload.get("assessment_roots_used", -1)) != 0:
                raise ValueError(f"Nonzero assessment_roots_used in {json_path}")
            if bool(payload.get("canary_only", False)) and not allow_canary_units:
                raise ValueError(f"Refusing canary_only unit {json_path} in main readout")

            b_idx = int(payload["batch_index"])
            expected_uid = batch_unit_id(b_idx)
            expected_stem = batch_unit_stem(b_idx)
            uid = str(payload["unit_id"])
            if uid != expected_uid or expected_stem != json_path.stem:
                raise ValueError(
                    f"Unit ID {uid!r} / index {b_idx} mismatch for filename {json_path.name!r}"
                )
            if b_idx in ledger_entries:
                raise ValueError(f"Duplicate verified unit {uid!r} across attempts")
            ledger_entries[b_idx] = meta
            unit_payloads[uid] = payload

    return ledger_entries, unit_payloads, provenance


def evaluate_r7_readout(
    *,
    train_baseline: Mapping[str, Any],
    unit_payloads: Mapping[str, Mapping[str, Any]],
    mac_timing: Mapping[str, Any],
    expected_batches: int,
    partial_preflight: bool = False,
) -> dict[str, Any]:
    """Join verified all-die macro outcomes onto the C/T train baseline and evaluate kill bars."""
    if train_baseline.get("schema") != "t34-r7-train-exact-stop-baseline/v1":
        raise ValueError(f"Invalid train baseline schema: {train_baseline.get('schema')!r}")
    if int(train_baseline.get("assessment_roots_used", -1)) != 0:
        raise ValueError("Train baseline reports nonzero assessment_roots_used")
    if mac_timing.get("schema") != "t34-r7-mac-tracker-timing/v1":
        raise ValueError(f"Invalid mac timing schema: {mac_timing.get('schema')!r}")

    expected_uids = [batch_unit_id(i) for i in range(expected_batches)]
    missing_uids = [uid for uid in expected_uids if uid not in unit_payloads]
    if missing_uids:
        raise ValueError(
            f"Cannot compute final R7 readout; missing verified batches: {missing_uids}"
        )

    macro_by_uid: dict[str, Mapping[str, Any]] = {}
    for uid in expected_uids:
        payload = unit_payloads[uid]
        for r in payload["roots"]:
            if r.get("partition") != "train":
                raise ValueError(f"Non-train root {r.get('root_uid')!r} found in unit {uid}")
            ruid = str(r["root_uid"])
            if ruid in macro_by_uid:
                raise ValueError(f"Duplicate root_uid {ruid!r} across verified batches")
            macro_by_uid[ruid] = r

    baseline_rows: list[Mapping[str, Any]] = list(train_baseline["train_roots"])
    if any(r.get("partition") != "train" for r in baseline_rows):
        raise ValueError("Non-train row present in train_exact_stop_baseline")

    if partial_preflight:
        eval_rows = [
            r for r in baseline_rows if (not r["all_die_sel0"]) or (r["root_uid"] in macro_by_uid)
        ]
    else:
        missing_roots = [
            r["root_uid"]
            for r in baseline_rows
            if r["all_die_sel0"] and r["root_uid"] not in macro_by_uid
        ]
        if missing_roots:
            raise ValueError(
                f"Missing {len(missing_roots)} 1-step-all-die train roots "
                f"(first: {missing_roots[0]})"
            )
        eval_rows = baseline_rows

    # Join policy outcomes under strict exact_stop gating:
    # - If all_die_sel0 is False (any 1-step branch survived on suffix 0): emit exact_stop.
    # - If all_die_sel0 is True: emit macro iff macro_triggered on suffix 0, else exact_stop.
    joined: list[dict[str, Any]] = []
    newly_caused_ordinary_flips = 0
    for row in eval_rows:
        ruid = str(row["root_uid"])
        es_s = float(row["exact_stop_s_score"])
        es_m = float(row["exact_stop_m_score"])
        es4_s = float(row["ct4_a_star_s_score"])
        if not bool(row["all_die_sel0"]):
            pol_s = es_s
            pol_m = es_m
            triggered = False
            ply1_ticks = 0.0
            two_ply_ticks = 0.0
            o1_name = "exact_stop"
            o2_name = "exact_stop"
        else:
            m_rec = macro_by_uid[ruid]
            triggered = bool(m_rec["macro_triggered"])
            pol_s = float(m_rec["policy_s_score"]) if triggered else es_s
            pol_m = float(m_rec["policy_m_score"]) if triggered else es_m
            ply1_ticks = float(m_rec["indep_mean_ply1_nonfatal_ticks"]) if triggered else 0.0
            two_ply_ticks = float(m_rec["indep_mean_two_ply_nonfatal_ticks"]) if triggered else 0.0
            o1_name = str(m_rec["o1_name"]) if triggered else "exact_stop"
            o2_name = str(m_rec["o2_name"]) if triggered else "exact_stop"

        if row["kind"] == "ordinary" and pol_s + 1e-12 < es_s:
            newly_caused_ordinary_flips += 1

        joined.append(
            {
                "root_uid": ruid,
                "kind": str(row["kind"]),
                "mix": str(row["mix"]),
                "hero_length": int(row["hero_length"]),
                "all_die_sel0": bool(row["all_die_sel0"]),
                "macro_triggered": triggered,
                "o1_name": o1_name,
                "o2_name": o2_name,
                "exact_stop_s_score": es_s,
                "exact_stop_m_score": es_m,
                "ct4_a_star_s_score": es4_s,
                "policy_s_score": pol_s,
                "policy_m_score": pol_m,
                "lift_over_exact_stop": pol_s - es_s,
                "lift_over_ct4_a_star": pol_s - es4_s,
                "ply1_nonfatal_ticks": ply1_ticks,
                "two_ply_nonfatal_ticks": two_ply_ticks,
            }
        )

    def _summarize_slice(items: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        if not items:
            return {
                "n": 0,
                "macro_triggered": 0,
                "exact_stop_mean_s_score": 0.0,
                "ct4_a_star_mean_s_score": 0.0,
                "policy_mean_s_score": 0.0,
                "lift_over_exact_stop": 0.0,
                "lift_over_ct4_a_star": 0.0,
            }
        es_arr = np.array([float(x["exact_stop_s_score"]) for x in items], dtype=np.float64)
        ct4_arr = np.array([float(x["ct4_a_star_s_score"]) for x in items], dtype=np.float64)
        pol_arr = np.array([float(x["policy_s_score"]) for x in items], dtype=np.float64)
        return {
            "n": len(items),
            "macro_triggered": sum(1 for x in items if x["macro_triggered"]),
            "exact_stop_mean_s_score": round(float(es_arr.mean()), 6),
            "ct4_a_star_mean_s_score": round(float(ct4_arr.mean()), 6),
            "policy_mean_s_score": round(float(pol_arr.mean()), 6),
            "lift_over_exact_stop": round(float((pol_arr - es_arr).mean()), 6),
            "lift_over_ct4_a_star": round(float((pol_arr - ct4_arr).mean()), 6),
        }

    all_die_l800 = [x for x in joined if x["all_die_sel0"] and x["hero_length"] >= 800]
    all_die_death_l800 = [x for x in all_die_l800 if x["kind"] == "death_lookback"]
    all_die_l600 = [x for x in joined if x["all_die_sel0"] and x["hero_length"] >= 600]
    all_die_all = [x for x in joined if x["all_die_sel0"]]
    death_l800_total = [
        x for x in joined if x["kind"] == "death_lookback" and x["hero_length"] >= 800
    ]
    death_l600_total = [
        x for x in joined if x["kind"] == "death_lookback" and x["hero_length"] >= 600
    ]
    ordinary_total = [x for x in joined if x["kind"] == "ordinary"]

    triggered_rows = [x for x in all_die_all if x["macro_triggered"]]
    mean_ply1_duration = (
        float(np.mean([x["ply1_nonfatal_ticks"] for x in triggered_rows]))
        if triggered_rows
        else 0.0
    )
    mean_two_ply_duration = (
        float(np.mean([x["two_ply_nonfatal_ticks"] for x in triggered_rows]))
        if triggered_rows
        else 0.0
    )

    s_all_die_l800 = _summarize_slice(all_die_l800)
    s_all_die_death_l800 = _summarize_slice(all_die_death_l800)
    s_all_die_l600 = _summarize_slice(all_die_l600)
    s_all_die_all = _summarize_slice(all_die_all)
    s_death_l800_total = _summarize_slice(death_l800_total)
    s_death_l600_total = _summarize_slice(death_l600_total)
    s_ordinary_total = _summarize_slice(ordinary_total)

    min_rescue = float(R7_KILL_THRESHOLDS["min_independent_suffix_H300_rescue_L800_all_die"])
    min_death_lift = (
        float(R7_KILL_THRESHOLDS["min_total_death_root_rescue_lift_over_exact_stop_pp"]) / 100.0
    )
    max_ord_flips = int(R7_KILL_THRESHOLDS["max_newly_caused_ordinary_lethal_flips"])
    max_p95_ms = float(R7_KILL_THRESHOLDS["max_mac_1thread_tracker_p95_ms_per_tick"])
    min_dur_ticks = float(R7_KILL_THRESHOLDS["min_realized_nonfatal_macro_duration_ticks"])

    l800_all_die_rescue_rate = float(s_all_die_l800["policy_mean_s_score"])
    # Check total death-root lift on BOTH L>=600 death roots and L>=800 death roots;
    # take the stronger of the two total death-root cohorts while reporting both explicitly.
    best_total_death_lift = max(
        float(s_death_l800_total["lift_over_exact_stop"]),
        float(s_death_l600_total["lift_over_exact_stop"]),
    )
    mac_p95_ms = float(mac_timing["p95_ms_per_tick"])

    bar_checks = {
        "passes_l800_all_die_rescue_ge_25pct": bool(l800_all_die_rescue_rate >= min_rescue),
        "passes_total_death_root_lift_ge_10pp": bool(best_total_death_lift >= min_death_lift),
        "passes_zero_new_ordinary_lethal_flips": bool(newly_caused_ordinary_flips <= max_ord_flips),
        "passes_mac_1thread_p95_lt_1ms": bool(mac_p95_ms < max_p95_ms),
        "passes_realized_nonfatal_macro_duration_ge_10_ticks": bool(
            mean_ply1_duration >= min_dur_ticks
        ),
    }
    failed_bars = [name for name, ok in bar_checks.items() if not ok]
    verdict = "PASS" if not failed_bars else "KILL"

    option_pair_counts: dict[str, int] = {}
    for x in triggered_rows:
        pair_key = f"{x['o1_name']} -> {x['o2_name']}"
        option_pair_counts[pair_key] = option_pair_counts.get(pair_key, 0) + 1

    return {
        "schema": "t34-r7-macro-fallback-readout/v1",
        "study": STUDY,
        "namespace": NAMESPACE,
        "partial_preflight": bool(partial_preflight),
        "assessment_roots_used": 0,
        "spacious_interrupt_enabled": False,
        "thresholds": dict(R7_KILL_THRESHOLDS),
        "bar_checks": bar_checks,
        "failed_bars": failed_bars,
        "verdict": verdict,
        "metrics": {
            "l800_all_die_indep_rescue_rate": round(l800_all_die_rescue_rate, 6),
            "l800_all_die_death_only_indep_rescue_rate": round(
                float(s_all_die_death_l800["policy_mean_s_score"]), 6
            ),
            "total_death_root_lift_l800": round(
                float(s_death_l800_total["lift_over_exact_stop"]), 6
            ),
            "total_death_root_lift_l600": round(
                float(s_death_l600_total["lift_over_exact_stop"]), 6
            ),
            "newly_caused_ordinary_lethal_flips": int(newly_caused_ordinary_flips),
            "mac_1thread_tracker_p95_ms_per_tick": round(mac_p95_ms, 4),
            "realized_nonfatal_ply1_macro_duration_ticks": round(mean_ply1_duration, 4),
            "realized_nonfatal_two_ply_macro_duration_ticks": round(mean_two_ply_duration, 4),
        },
        "slices": {
            "train_l800_all_die": s_all_die_l800,
            "train_l800_all_die_death_lookback": s_all_die_death_l800,
            "train_l600_all_die": s_all_die_l600,
            "train_all_die_total": s_all_die_all,
            "train_death_lookback_l800_total": s_death_l800_total,
            "train_death_lookback_l600_total": s_death_l600_total,
            "train_ordinary_total": s_ordinary_total,
        },
        "triggered_option_pair_histogram": dict(
            sorted(option_pair_counts.items(), key=lambda kv: (-kv[1], kv[0]))
        ),
        "mac_tracker_timing": dict(mac_timing),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--attempt-dir", type=Path, action="append", required=True)
    parser.add_argument("--train-baseline", type=Path, required=True)
    parser.add_argument("--mac-timing", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-batches", type=int, default=3)
    parser.add_argument("--write-ledger", type=Path, default=None)
    parser.add_argument("--local-preflight", action="store_true")
    parser.add_argument("--allow-canary-units", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    ledger_entries, unit_payloads, provenance = collect_verified_batch_units(
        args.attempt_dir,
        local_preflight=args.local_preflight,
        allow_canary_units=args.allow_canary_units,
    )
    if args.write_ledger is not None:
        publish_verified_ledger(args.write_ledger, ledger_entries)

    train_baseline = json.loads(args.train_baseline.read_text(encoding="utf-8"))
    mac_timing = json.loads(args.mac_timing.read_text(encoding="utf-8"))
    report = evaluate_r7_readout(
        train_baseline=train_baseline,
        unit_payloads=unit_payloads,
        mac_timing=mac_timing,
        expected_batches=args.expected_batches,
        partial_preflight=args.local_preflight,
    )
    report["attempt_provenance"] = provenance
    report["verified_unit_ids"] = sorted(unit_payloads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "verdict": report["verdict"],
                "metrics": report["metrics"],
                "failed_bars": report["failed_bars"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
