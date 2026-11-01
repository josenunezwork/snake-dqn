"""Fixed-sequence 3-arm confirmation analyzer for Watch6-L1-V1 with parameterized alpha."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from scipy.stats import t as student_t

from research.bl_served_search.b2_engine import FRP3
from research.bl_served_search.ceiling import B8_SHA256
from research.watch6_l1_confirm import protocol as p
from research.watch6_l1_confirm.resume import build as build_verified_ledger
from tools.ops.package_spec import publish_json

DEFAULT_ALPHA = 0.05
GATE1_MASS_LB_MIN = -0.05
GATE2_MASS_LB_MIN = -0.025
GATE2_D_GE800_PER_1K_UB_MAX = 0.05


def _validate_alpha(alpha: float) -> float:
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)) or not math.isfinite(alpha):
        raise ValueError(f"alpha must be a finite float in (0, 1), got {alpha!r}")
    value = float(alpha)
    if not 0.0 < value < 1.0:
        raise ValueError(f"alpha must satisfy 0 < alpha < 1, got {value!r}")
    return value


def paired_t_ci(diffs: np.ndarray, *, alpha: float = DEFAULT_ALPHA) -> dict:
    """Two-sided (1 - alpha) paired Student-t confidence interval on world-level differences."""
    alpha_val = _validate_alpha(alpha)
    arr = np.asarray(diffs, dtype=np.float64)
    if arr.ndim != 1 or arr.size < 2 or not np.isfinite(arr).all():
        raise ValueError("paired_t_ci requires finite 1-D sample with n >= 2")
    n = int(arr.size)
    df = n - 1
    crit = float(student_t.ppf(1.0 - alpha_val / 2.0, df))
    mean = float(np.mean(arr))
    if float(np.ptp(arr)) == 0.0:
        return {
            "mean": mean,
            "ci": [mean, mean],
            "sd": 0.0,
            "se": 0.0,
            "df": df,
            "alpha": alpha_val,
            "t_critical": crit,
            "degenerate_zero_variance": True,
        }
    sd = float(np.std(arr, ddof=1))
    se = sd / math.sqrt(n)
    half = crit * se
    return {
        "mean": mean,
        "ci": [mean - half, mean + half],
        "sd": sd,
        "se": se,
        "df": df,
        "alpha": alpha_val,
        "t_critical": crit,
        "degenerate_zero_variance": False,
    }


def paired_fieller_relative_ci(
    candidate_means: np.ndarray,
    reference_means: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Two-sided (1 - alpha) paired Fieller CI on (mean(cand) - mean(ref)) / mean(ref)."""
    alpha_val = _validate_alpha(alpha)
    cand = np.asarray(candidate_means, dtype=np.float64)
    ref = np.asarray(reference_means, dtype=np.float64)
    if (
        cand.ndim != 1
        or ref.ndim != 1
        or cand.size != ref.size
        or cand.size < 2
        or not np.isfinite(cand).all()
        or not np.isfinite(ref).all()
    ):
        raise ValueError("paired_fieller_relative_ci requires equal-length finite samples (n >= 2)")
    n = int(cand.size)
    df = n - 1
    crit = float(student_t.ppf(1.0 - alpha_val / 2.0, df))
    mx = float(np.mean(ref))
    my = float(np.mean(cand))
    if mx <= 0.0:
        return {
            "relative_difference": None,
            "ci": None,
            "bounded": False,
            "status": "nonpositive_reference_mean",
            "df": df,
            "alpha": alpha_val,
            "t_critical": crit,
        }
    rel_diff = (my - mx) / mx
    vx = float(np.var(ref, ddof=1)) / n
    vy = float(np.var(cand, ddof=1)) / n
    vxy = float(np.cov(ref, cand, ddof=1)[0, 1]) / n
    t2 = crit * crit
    a = mx * mx - t2 * vx
    b = mx * my - t2 * vxy
    c = my * my - t2 * vy
    disc = b * b - a * c
    tol = 64.0 * float(np.finfo(np.float64).eps) * max(1.0, b * b, abs(a * c))
    if -tol <= disc < 0.0:
        disc = 0.0
    if a <= 0.0 or disc < 0.0:
        return {
            "relative_difference": rel_diff,
            "ci": None,
            "bounded": False,
            "status": "unbounded_or_disconnected",
            "df": df,
            "alpha": alpha_val,
            "t_critical": crit,
        }
    root = math.sqrt(disc)
    r_low = (b - root) / a
    r_high = (b + root) / a
    low, high = sorted((r_low - 1.0, r_high - 1.0))
    return {
        "relative_difference": rel_diff,
        "ci": [low, high],
        "bounded": True,
        "status": "bounded",
        "df": df,
        "alpha": alpha_val,
        "t_critical": crit,
    }


def _validate_world_payload(
    unit: dict,
    *,
    expected_alloc_by_index: dict[int, dict] | None,
    expected_horizon: int,
) -> None:
    if (
        not isinstance(unit, dict)
        or unit.get("schema") != "watch6-l1-world/v1"
        or unit.get("namespace") != p.NAMESPACE
        or unit.get("seed_kind") != "primary"
        or unit.get("fixture") is not False
        or unit.get("record_mode") is not None
        or type(unit.get("index")) is not int
        or type(unit.get("seed")) is not int
        or unit.get("horizon") != expected_horizon
    ):
        raise ValueError(f"foreign, fixture, or malformed Watch6-L1-V1 world payload: {unit!r}")

    idx = unit["index"]
    if expected_alloc_by_index is not None:
        alloc_row = expected_alloc_by_index.get(idx)
        if (
            alloc_row is None
            or unit["seed"] != alloc_row["seed"]
            or unit.get("arm_order") != alloc_row["arm_order"]
        ):
            raise ValueError(f"world {idx} seed or arm_order differs from frozen allocation")

    roster = unit.get("roster")
    if not isinstance(roster, dict) or (
        roster.get("hero_checkpoint_sha256") != B8_SHA256
        or roster.get("opponent_checkpoint_sha256") != FRP3
        or roster.get("opponent_count") != 5
        or roster.get("veto_variant") != "v8"
        or roster.get("veto_lambda") != 8.0
        or roster.get("mechanics_version") != 2
        or roster.get("num_snakes") != 6
    ):
        raise ValueError(f"world {idx} roster mismatch against shipped Watch6-v2 pins")

    arms = unit.get("arms")
    if not isinstance(arms, dict) or set(arms) != set(p.ARMS):
        raise ValueError(f"world {idx} missing required 3-arm results {p.ARMS}")

    for arm in p.ARMS:
        row = arms[arm]
        if (
            not isinstance(row, dict)
            or row.get("arm") != arm
            or row.get("frames") != expected_horizon
            or row.get("mirror_identity_frames") != expected_horizon
        ):
            raise ValueError(f"world {idx} arm {arm} failed full-horizon live/Grid mirror check")
        batches = int(row.get("batches", -1))
        kernel_rows = int(row.get("native_v8_kernel_rows", -1))
        if arm == "bare":
            if batches != 0 or kernel_rows != 0 or int(row.get("interventions", -1)) != 0:
                raise ValueError(f"world {idx} bare arm executed search batches")
        else:
            if (
                batches < 0
                or (batches > 0 and kernel_rows <= 0)
                or (batches == 0 and kernel_rows != 0)
            ):
                raise ValueError(f"world {idx} arm {arm} invalid native v8 kernel attestation")


def _arm_aggregate(units: list[dict], arm: str, horizon: int) -> dict:
    rows = [u["arms"][arm] for u in units]
    total_frames = len(rows) * horizon
    total_deaths = sum(int(r["deaths"]) for r in rows)
    total_short = sum(int(r["short_body_deaths_lt600"]) for r in rows)
    total_long = sum(int(r["long_body_deaths_ge600"]) for r in rows)
    total_ge800 = sum(int(r["late_large_body_deaths"]) for r in rows)
    total_decisions = sum(int(r["decisions"]) for r in rows)
    total_boost = sum(int(r["boost_decisions"]) for r in rows)
    branches = [int(r["branches_evaluated"]) for r in rows]
    return {
        "arm": arm,
        "label": p.ARM_SPECS[arm]["label"],
        "sessions": len(rows),
        "frames_total": total_frames,
        "deaths_total": total_deaths,
        "deaths_per_1k": total_deaths * 1000.0 / total_frames,
        "short_body_deaths_lt600_total": total_short,
        "short_body_deaths_lt600_per_1k": total_short * 1000.0 / total_frames,
        "long_body_deaths_ge600_total": total_long,
        "long_body_deaths_ge600_per_1k": total_long * 1000.0 / total_frames,
        "late_large_body_deaths_ge800_total": total_ge800,
        "late_large_body_deaths_ge800_per_1k": total_ge800 * 1000.0 / total_frames,
        "pooled_mean_mass_all_frames": sum(int(r["mass_sum"]) for r in rows) / total_frames,
        "session_median_mass_mean": float(np.mean([float(r["length_median"]) for r in rows])),
        "session_alive_median_length_mean": float(
            np.mean([float(r["alive_length_median"]) for r in rows])
        ),
        "session_p95_length_mean": float(np.mean([float(r["length_p95"]) for r in rows])),
        "max_length_max": max(int(r["max_length"]) for r in rows),
        "alive_frame_share": sum(int(r["alive_frames"]) for r in rows) / total_frames,
        "boost_decision_share": (total_boost / total_decisions) if total_decisions > 0 else 0.0,
        "batches_mean": float(np.mean([int(r["batches"]) for r in rows])),
        "batches_lt600_mean": float(np.mean([int(r["batches_lt600"]) for r in rows])),
        "batches_ge600_mean": float(np.mean([int(r["batches_ge600"]) for r in rows])),
        "interventions_mean": float(np.mean([int(r["interventions"]) for r in rows])),
        "interventions_lt600_mean": float(np.mean([int(r["interventions_lt600"]) for r in rows])),
        "interventions_ge600_mean": float(np.mean([int(r["interventions_ge600"]) for r in rows])),
        "branches_evaluated_mean": float(np.mean(branches)),
        "branches_evaluated_p95": float(np.percentile(branches, 95)),
        "branches_evaluated_max": max(branches),
        "branches_skipped_mean": float(np.mean([int(r["branches_skipped"]) for r in rows])),
        "native_v8_kernel_rows_total": sum(int(r["native_v8_kernel_rows"]) for r in rows),
        "mirror_identity_frames_total": sum(int(r["mirror_identity_frames"]) for r in rows),
        "main_seconds_mean": float(np.mean([float(r["timing"]["main_seconds"]) for r in rows])),
        "search_seconds_mean": float(np.mean([float(r["timing"]["search_seconds"]) for r in rows])),
    }


def analyze_worlds(
    units: list[dict],
    *,
    alpha: float = DEFAULT_ALPHA,
    expected_count: int = p.COUNT,
    expected_horizon: int = p.HORIZON,
    enforce_frozen_allocation: bool = True,
) -> dict:
    """Evaluate fixed-sequence Gate 1 (l1 vs bare) and Gate 2 (l1 vs l2_prime) at level alpha."""
    alpha_val = _validate_alpha(alpha)
    if type(expected_count) is not int or expected_count < 2:
        raise ValueError("expected_count must be an integer >= 2")
    if len(units) != expected_count:
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: expected {expected_count} paired worlds, got {len(units)}"
        )

    alloc_map = {row["index"]: row for row in p.allocation()} if enforce_frozen_allocation else None
    ordered = sorted(units, key=lambda item: int(item.get("index", -1)))
    if [u.get("index") for u in ordered] != list(range(expected_count)):
        raise ValueError("INVALID_OR_INCOMPLETE: world indices do not form exact 0..N-1 bijection")

    for unit in ordered:
        _validate_world_payload(
            unit,
            expected_alloc_by_index=alloc_map,
            expected_horizon=expected_horizon,
        )

    deaths_bare = np.array(
        [u["arms"]["bare"]["deaths"] * 1000.0 / expected_horizon for u in ordered]
    )
    deaths_l2p = np.array(
        [u["arms"]["l2_prime"]["deaths"] * 1000.0 / expected_horizon for u in ordered]
    )
    deaths_l1 = np.array([u["arms"]["l1"]["deaths"] * 1000.0 / expected_horizon for u in ordered])

    mass_bare = np.array([u["arms"]["bare"]["mass_sum"] / expected_horizon for u in ordered])
    mass_l2p = np.array([u["arms"]["l2_prime"]["mass_sum"] / expected_horizon for u in ordered])
    mass_l1 = np.array([u["arms"]["l1"]["mass_sum"] / expected_horizon for u in ordered])

    ge800_bare = np.array(
        [u["arms"]["bare"]["late_large_body_deaths"] * 1000.0 / expected_horizon for u in ordered]
    )
    ge800_l2p = np.array(
        [
            u["arms"]["l2_prime"]["late_large_body_deaths"] * 1000.0 / expected_horizon
            for u in ordered
        ]
    )
    ge800_l1 = np.array(
        [u["arms"]["l1"]["late_large_body_deaths"] * 1000.0 / expected_horizon for u in ordered]
    )

    # Gate 1: l1 vs bare (tested first at full level alpha under fixed-sequence hierarchy).
    g1_deaths = paired_t_ci(deaths_l1 - deaths_bare, alpha=alpha_val)
    g1_mass = paired_fieller_relative_ci(mass_l1, mass_bare, alpha=alpha_val)
    g1_ge800_diag = paired_t_ci(ge800_l1 - ge800_bare, alpha=alpha_val)

    g1_deaths_pass = (not g1_deaths["degenerate_zero_variance"]) and (g1_deaths["ci"][1] < 0.0)
    g1_mass_pass = bool(g1_mass["bounded"] and g1_mass["ci"][0] >= GATE1_MASS_LB_MIN)
    gate1_pass = bool(g1_deaths_pass and g1_mass_pass)
    gate1_fail = bool(
        (
            (not g1_deaths["degenerate_zero_variance"])
            and g1_deaths["ci"][0] >= 0.0
            and g1_deaths["ci"][1] > 0.0
        )
        or (g1_mass["bounded"] and g1_mass["ci"][1] < GATE1_MASS_LB_MIN)
    )
    gate1_outcome = (
        "PASS_GATE1" if gate1_pass else ("FAIL_GATE1" if gate1_fail else "INCONCLUSIVE_GATE1")
    )

    # Gate 2: l1 vs l2_prime (tested for confirmation ONLY IF Gate 1 passes).
    g2_deaths = paired_t_ci(deaths_l1 - deaths_l2p, alpha=alpha_val)
    g2_mass = paired_fieller_relative_ci(mass_l1, mass_l2p, alpha=alpha_val)
    g2_ge800 = paired_t_ci(ge800_l1 - ge800_l2p, alpha=alpha_val)

    g2_deaths_bar = (not g2_deaths["degenerate_zero_variance"]) and (g2_deaths["ci"][1] < 0.0)
    g2_mass_bar = bool(g2_mass["bounded"] and g2_mass["ci"][0] >= GATE2_MASS_LB_MIN)
    # Zero variance (e.g., 0 vs 0 L>=800 deaths in every world) gives CI [0.0, 0.0] <= +0.05.
    g2_ge800_bar = bool(g2_ge800["ci"][1] <= GATE2_D_GE800_PER_1K_UB_MAX)
    g2_all_bars_met = bool(g2_deaths_bar and g2_mass_bar and g2_ge800_bar)

    if gate1_pass:
        if g2_all_bars_met:
            gate2_outcome = "PASS_GATE2"
            verdict = "GO_L1_SUPERIOR"
            serving_decision = (
                "ELIGIBLE_FOR_L1_BANK_EXPANSION_AND_MAC_24FPS_SERVING_QUAL_PENDING_OWNER_OK"
            )
        else:
            g2_fail = bool(
                (
                    (not g2_deaths["degenerate_zero_variance"])
                    and g2_deaths["ci"][0] >= 0.0
                    and g2_deaths["ci"][1] > 0.0
                )
                or (g2_mass["bounded"] and g2_mass["ci"][1] < GATE2_MASS_LB_MIN)
                or (g2_ge800["ci"][0] > GATE2_D_GE800_PER_1K_UB_MAX)
            )
            gate2_outcome = "FAIL_GATE2" if g2_fail else "INCONCLUSIVE_GATE2"
            verdict = "KEEP_L2_PRIME_GATE1_ONLY"
            serving_decision = "RETAIN_CONFIRMED_EXACT_STOP_L2_PRIME_BANK"
    else:
        gate2_outcome = "NOT_TESTED_GATE1_DID_NOT_PASS"
        verdict = "FAIL_GATE1" if gate1_fail else "INCONCLUSIVE_GATE1"
        serving_decision = "RETAIN_CONFIRMED_EXACT_STOP_L2_PRIME_BANK"

    # Descriptive replication diagnostic (never gating; V3 already confirmed l2_prime vs bare).
    rep_deaths = paired_t_ci(deaths_l2p - deaths_bare, alpha=alpha_val)
    rep_mass = paired_fieller_relative_ci(mass_l2p, mass_bare, alpha=alpha_val)
    rep_ge800 = paired_t_ci(ge800_l2p - ge800_bare, alpha=alpha_val)

    return {
        "schema": "watch6-l1-confirm-decision/v1",
        "namespace": p.NAMESPACE,
        "science": p.identity(),
        "verdict": verdict,
        "replay_serving_decision": serving_decision,
        "multiplicity_procedure": "fixed_sequence_gatekeeping",
        "alpha": alpha_val,
        "confidence_level": 1.0 - alpha_val,
        "n": expected_count,
        "df": expected_count - 1,
        "t_critical": float(student_t.ppf(1.0 - alpha_val / 2.0, expected_count - 1)),
        "horizon_frames": expected_horizon,
        "reserve_substitutions_count": 0,
        "gate1_l1_vs_bare": {
            "tested_in_fixed_sequence": True,
            "outcome": gate1_outcome,
            "passed": gate1_pass,
            "bars": {
                "deaths_per_1k_ub_lt_0": {
                    "threshold_ub_exclusive": 0.0,
                    "passed": g1_deaths_pass,
                    "estimate": g1_deaths["mean"],
                    "ci": g1_deaths["ci"],
                    "se": g1_deaths["se"],
                    "degenerate_zero_variance": g1_deaths["degenerate_zero_variance"],
                },
                "pooled_mass_fieller_lb_ge_neg_0p05": {
                    "threshold_lb_inclusive": GATE1_MASS_LB_MIN,
                    "passed": g1_mass_pass,
                    "relative_difference": g1_mass["relative_difference"],
                    "ci": g1_mass["ci"],
                    "bounded": g1_mass["bounded"],
                    "status": g1_mass["status"],
                },
            },
            "descriptive_late_large_body_deaths_ge800_delta_per_1k": {
                "estimate": g1_ge800_diag["mean"],
                "ci": g1_ge800_diag["ci"],
            },
        },
        "gate2_l1_vs_l2_prime": {
            "tested_in_fixed_sequence": gate1_pass,
            "outcome": gate2_outcome,
            "passed": bool(gate1_pass and g2_all_bars_met),
            "unconditional_bars_met_diagnostic": g2_all_bars_met,
            "bars": {
                "deaths_per_1k_ub_lt_0": {
                    "threshold_ub_exclusive": 0.0,
                    "bar_met": g2_deaths_bar,
                    "estimate": g2_deaths["mean"],
                    "ci": g2_deaths["ci"],
                    "se": g2_deaths["se"],
                    "degenerate_zero_variance": g2_deaths["degenerate_zero_variance"],
                },
                "pooled_mass_fieller_lb_ge_neg_0p025": {
                    "threshold_lb_inclusive": GATE2_MASS_LB_MIN,
                    "bar_met": g2_mass_bar,
                    "relative_difference": g2_mass["relative_difference"],
                    "ci": g2_mass["ci"],
                    "bounded": g2_mass["bounded"],
                    "status": g2_mass["status"],
                },
                "session_d_ge800_per_1k_ub_le_pos_0p05": {
                    "threshold_ub_inclusive": GATE2_D_GE800_PER_1K_UB_MAX,
                    "bar_met": g2_ge800_bar,
                    "estimate": g2_ge800["mean"],
                    "ci": g2_ge800["ci"],
                    "se": g2_ge800["se"],
                    "degenerate_zero_variance": g2_ge800["degenerate_zero_variance"],
                },
            },
        },
        "descriptive_replication_l2_prime_vs_bare": {
            "gating": False,
            "note": "Descriptive replication only; Watch6 V3 already confirmed l2_prime vs bare.",
            "deaths_per_1k_delta": {"estimate": rep_deaths["mean"], "ci": rep_deaths["ci"]},
            "pooled_mass_fieller": {
                "relative_difference": rep_mass["relative_difference"],
                "ci": rep_mass["ci"],
                "bounded": rep_mass["bounded"],
            },
            "late_large_body_deaths_ge800_delta_per_1k": {
                "estimate": rep_ge800["mean"],
                "ci": rep_ge800["ci"],
            },
        },
        "arms": {arm: _arm_aggregate(ordered, arm, expected_horizon) for arm in p.ARMS},
    }


def analyze_verified_union(
    unit_paths: list[Path],
    receipt_paths: list[Path],
    *,
    alpha: float = DEFAULT_ALPHA,
) -> dict:
    """Verify stopped receipts, sidecars, and 96/96 bijection outcome-blind before opening."""
    alpha_val = _validate_alpha(alpha)
    ledger = build_verified_ledger(unit_paths, receipt_paths)
    if len(ledger["units"]) != p.COUNT:
        raise ValueError(
            f"INVALID_OR_INCOMPLETE: verified ledger has {len(ledger['units'])}/{p.COUNT} worlds"
        )
    payloads = [
        json.loads(path.resolve(strict=True).read_text(encoding="utf-8")) for path in unit_paths
    ]
    return analyze_worlds(
        payloads,
        alpha=alpha_val,
        expected_count=p.COUNT,
        expected_horizon=p.HORIZON,
        enforce_frozen_allocation=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--units", type=Path, nargs="+", required=True)
    parser.add_argument("--receipts", type=Path, nargs="+", required=True)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    decision = analyze_verified_union(args.units, args.receipts, alpha=args.alpha)
    publish_json(args.out.resolve(), decision)
    print(
        json.dumps(
            {
                "verdict": decision["verdict"],
                "alpha": decision["alpha"],
                "gate1_outcome": decision["gate1_l1_vs_bare"]["outcome"],
                "gate2_outcome": decision["gate2_l1_vs_l2_prime"]["outcome"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
