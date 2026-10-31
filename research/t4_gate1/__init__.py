"""Track-4 Gate 1: privileged factored critic go/no-go gate on fresh Watch6-v2 DeviceSim harvest.

Trains a clean-slate privileged full-arena factored critic (separate long-horizon competing-bin
survival-hazard head and short-gamma alive foraging mass-flux head, zero B8 weights embedded)
on a fresh Watch6-v2 harvest generated with exact GPU ``DeviceSim`` (``exact_mechanics=True``,
``hero_respawn=True``; ``B8 s3`` hero vs ``5x frp3-s12+v8``) in a verified-disjoint exploration
seed namespace.

Evaluates once on a held-out world-disjoint test split:
  (a) Residual explained variance over a rich board-blind ``(L_t, H_rem)`` baseline fitted
      strictly on ``fit`` worlds, primary stratum ``L >= 600``;
  (b) Counterfactual ``H = 300`` rescue value on held-out mature (``L >= 600``) death-lookback
      and ordinary roots using independent CRN selection and scoring suffixes.
"""

from __future__ import annotations

from typing import Any, Mapping

STUDY = "t34-t4-gate1-privileged-critic-v1"
ALLOCATION_NAME = "watch6-v2-world-disjoint-2rep"
ALLOCATION_INDEX = 0
UNIT_KIND = "t4_gate1_replicate"

MAIN_NAMESPACE = "t34-explore-t4-gate1-v2-1031-r4"
CANARY_NAMESPACE = "t34-explore-t4-gate1-canary-1031-r4"
PREFLIGHT_NAMESPACE = "t34-explore-t4-gate1-preflight-1031-r4"

B8_S3_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
FRP3_S12_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"

# Pre-stated Gate 1 kill/pass thresholds frozen before any Colab execution.
# Primary bars follow OPUS_META (§3 row 3, §4 R4 bullet) and review-4 (§2):
#   - KILL T4 if held-out world-disjoint L>=600 residual EV over (L_t, H_rem) < +0.25 (Opus),
#     or 95% world-clustered bootstrap CI lower bound <= +0.10, or either replicate < +0.15.
#   - KILL T4 if held-out world-disjoint L>=600 death-root H300 rescue mass delta < +40.0 mass,
#     or 95% world-clustered bootstrap CI lower bound <= 0.0 mass.
#   - Guardrails: per-action H300 doom AUC >= 0.80, ordinary-root H300 mass delta >= -15.0 mass,
#     and ordinary-root newly caused lethal-flip rate <= 0.03 (3.0%).
T4_GATE1_THRESHOLDS: dict[str, float] = {
    "min_l600_residual_ev": 0.25,
    "min_l600_residual_ev_ci95_low": 0.10,
    "min_per_replicate_l600_residual_ev": 0.15,
    "min_death_root_h300_rescue_mass_delta": 40.0,
    "min_death_root_h300_rescue_mass_ci95_low": 0.0,
    "min_h300_doom_auc": 0.80,
    "min_ordinary_root_h300_mass_delta": -15.0,
    "max_ordinary_lethal_flip_rate": 0.03,
}

# Pre-stated downstream requirements needed for full Track-4 GO (500M-1B v2 GPU RL)
# even if Gate 1 passes (OPUS_META §4 + review-4 §2-3).
T4_FULL_GO_PREREQUISITES: tuple[str, ...] = (
    "Gate 1 PASS on all pre-stated held-out world-disjoint critic bars above.",
    "Watch6 V3 confirms late-game search/value ceiling >= +40 Watch6 session mass.",
    "Quiet-Mac 1-thread served bare spatial actor (zero 7-source H=320 BFS) meets "
    "Watch6 paced p95 <= 11.688 ms (hero forward+featurizer <= 4.0 ms).",
    "A100 resident v2 training loop demonstrates >= 8,500 combined hero steps/s.",
    "BC/ExIt warm-started bare spatial actor distilled against the frozen Gate-1 critic "
    "retains >= 50% of critic rescue gain and beats B8 s3 on a paired 64-world Watch6-v2 "
    "screen before scaling past 50M RL steps (no cold-start scratch PPO).",
)


def replicate_unit_stem(replicate_index: int) -> str:
    if replicate_index < 0:
        raise ValueError(f"replicate_index must be >= 0, got {replicate_index}")
    return f"replicate-{int(replicate_index):04d}"


def replicate_unit_id(namespace: str, replicate_index: int) -> str:
    return (
        f"{namespace}:{ALLOCATION_NAME}:{ALLOCATION_INDEX}:"
        f"{UNIT_KIND}:{replicate_unit_stem(replicate_index)}"
    )


def evaluate_gate1_bars(
    metrics: Mapping[str, Any],
    *,
    thresholds: Mapping[str, float] = T4_GATE1_THRESHOLDS,
) -> dict[str, Any]:
    """Evaluate pre-stated Gate 1 kill/GO bars on pooled held-out world-disjoint metrics."""
    ev_l600 = float(metrics["l_ge_600"]["residual_ev"])
    ev_l600_ci_low = float(metrics["l_ge_600"]["residual_ev_ci95"][0])
    rep_evs = [float(v) for v in metrics.get("per_replicate_l_ge_600_residual_ev", [])]
    min_rep_ev = min(rep_evs) if rep_evs else float("nan")

    rescue_delta = float(metrics["h300_rescue"]["death_lookback"]["critic_vs_b8_mass_delta"])
    rescue_ci_low = float(
        metrics["h300_rescue"]["death_lookback"]["critic_vs_b8_mass_delta_ci95"][0]
    )
    doom_auc = float(metrics["h300_rescue"]["all_mature"]["per_action_doom_auc"])
    ord_mass_delta = float(metrics["h300_rescue"]["ordinary"]["critic_vs_b8_mass_delta"])
    ord_lethal_flip = float(metrics["h300_rescue"]["ordinary"]["newly_caused_lethal_flip_rate"])

    checks = {
        "opus_primary_l600_residual_ev_ge_0p25": {
            "passed": bool(ev_l600 >= thresholds["min_l600_residual_ev"]),
            "observed": round(ev_l600, 6),
            "threshold": thresholds["min_l600_residual_ev"],
            "sense": ">=",
        },
        "l600_residual_ev_ci95_low_gt_0p10": {
            "passed": bool(ev_l600_ci_low > thresholds["min_l600_residual_ev_ci95_low"]),
            "observed": round(ev_l600_ci_low, 6),
            "threshold": thresholds["min_l600_residual_ev_ci95_low"],
            "sense": ">",
        },
        "both_replicates_l600_residual_ev_ge_0p15": {
            "passed": bool(
                len(rep_evs) > 0 and min_rep_ev >= thresholds["min_per_replicate_l600_residual_ev"]
            ),
            "observed": round(min_rep_ev, 6) if rep_evs else None,
            "threshold": thresholds["min_per_replicate_l600_residual_ev"],
            "sense": ">=",
        },
        "opus_death_root_h300_rescue_mass_ge_40": {
            "passed": bool(rescue_delta >= thresholds["min_death_root_h300_rescue_mass_delta"]),
            "observed": round(rescue_delta, 4),
            "threshold": thresholds["min_death_root_h300_rescue_mass_delta"],
            "sense": ">=",
        },
        "death_root_h300_rescue_mass_ci95_low_gt_0": {
            "passed": bool(rescue_ci_low > thresholds["min_death_root_h300_rescue_mass_ci95_low"]),
            "observed": round(rescue_ci_low, 4),
            "threshold": thresholds["min_death_root_h300_rescue_mass_ci95_low"],
            "sense": ">",
        },
        "mature_per_action_h300_doom_auc_ge_0p80": {
            "passed": bool(doom_auc >= thresholds["min_h300_doom_auc"]),
            "observed": round(doom_auc, 6),
            "threshold": thresholds["min_h300_doom_auc"],
            "sense": ">=",
        },
        "ordinary_root_h300_mass_delta_ge_neg15": {
            "passed": bool(ord_mass_delta >= thresholds["min_ordinary_root_h300_mass_delta"]),
            "observed": round(ord_mass_delta, 4),
            "threshold": thresholds["min_ordinary_root_h300_mass_delta"],
            "sense": ">=",
        },
        "ordinary_newly_caused_lethal_flip_le_0p03": {
            "passed": bool(ord_lethal_flip <= thresholds["max_ordinary_lethal_flip_rate"]),
            "observed": round(ord_lethal_flip, 6),
            "threshold": thresholds["max_ordinary_lethal_flip_rate"],
            "sense": "<=",
        },
    }
    failed = [name for name, item in checks.items() if not item["passed"]]
    verdict = "GATE1_PASS" if not failed else "KILL_T4"
    return {
        "verdict": verdict,
        "all_bars_passed": not failed,
        "failed_bars": failed,
        "checks": checks,
        "full_track4_go_prerequisites": list(T4_FULL_GO_PREREQUISITES),
    }


__all__ = [
    "ALLOCATION_INDEX",
    "ALLOCATION_NAME",
    "B8_S3_SHA256",
    "CANARY_NAMESPACE",
    "FRP3_S12_SHA256",
    "MAIN_NAMESPACE",
    "PREFLIGHT_NAMESPACE",
    "STUDY",
    "T4_FULL_GO_PREREQUISITES",
    "T4_GATE1_THRESHOLDS",
    "UNIT_KIND",
    "evaluate_gate1_bars",
    "replicate_unit_id",
    "replicate_unit_stem",
]
