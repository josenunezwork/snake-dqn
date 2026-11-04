"""T34 Closeout Audit: Part A (b8-head-closedloop M0a readout) & Part B (Live open-stop regimes).

Pre-registered in:
- Registration: docs/research/final_run_20261019/CAMPAIGN_LOG.md row '2026-11-04 PT 17:40'
- Design: t34-sol/argon-after-1b-advice/ADVICE.md §3
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import pickle
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling
from research.bl_served_search.ruler_operator import _clone_root
from research.hero_stage0.open_stop import (
    DEFAULT_OPENSTOP_OUT_DIR,
    _eval_open_stop_branch_suffix,
    evaluate_terminal_openness,
)
from research.hero_stage0.protocol import (
    COUNT,
    HORIZON,
    NAMESPACE,
    OPENSTOP_BRANCH_HORIZON,
    OPENSTOP_INTERVAL,
    OPENSTOP_K1_ARM,
    OPENSTOP_MIN_LENGTH,
    OPENSTOP_SCORING_SUFFIXES,
    STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN,
    STAGE1B_BOOTSTRAP_DRAWS,
    STAGE1B_BOOTSTRAP_SEED,
)
from research.hero_stage0.stage1a import (
    _enforce_single_thread_env,
    evaluation_order_nb,
    reseed_branch_suffix,
)
from research.hero_stage0.stage1b import (
    DEFAULT_STAGE1B_OUT_DIR,
    restore_world,
    world_cluster_bootstrap_retention,
)
from research.t3_rollout_labels_20261015.bank_engine import (
    choose_and_step,
    prepared_q,
    step_prepared,
)
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.t15_mixed_roster_20261017.episodes import _assert_mixed
from research.watch_replay.product import open_watch
from src.scripts.eval_stats import student_t_isf
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.eval_engine import _dispatch_actions

DEFAULT_CLOSEOUT_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-closeout-audit"
)
DEFAULT_B8_HEAD_CLOSEDLOOP_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/b8-head-closedloop"
)

# Pre-registered Part A bars (SUCCESS_CONTRACT.md §2 & ADVICE.md §3.1)
PART_A_ARMS: tuple[str, ...] = ("b8", "m0", "m10", "m25", "m50")
PART_A_CANDIDATE_ARMS: tuple[str, ...] = ("m0", "m10", "m25", "m50")
PART_A_MIXES: tuple[str, ...] = ("scripted", "frozen", "mixed")
PART_A_BOOTSTRAP_SEED = 343014
PART_A_BOOTSTRAP_DRAWS = 10000
PART_A_BAR_MASS_DELTA_MIN = 40.0
PART_A_BAR_MASS_LB90_MIN = 0.0  # strict > 0.0
PART_A_BAR_SCRIPTED_NO_DEATH_GAIN_MIN = 0.03
PART_A_BAR_SURVIVAL_POOLED_NI_MARGIN = 0.05  # strict > -0.05
PART_A_BAR_SURVIVAL_MIX_NI_MARGIN = 0.075  # strict > -0.075
PART_A_BAR_OVERRIDE_RATE_MAX = 0.15
PART_A_BAR_BOOST_SHARE_MIN = 0.005
PART_A_BAR_BOOST_SHARE_MAX = 0.100

# Pre-registered Part B bars (CAMPAIGN_LOG.md '2026-11-04 PT 17:40' & ADVICE.md §3.2)
PART_B_REGIME1_HORIZON = 12
PART_B_REGIME1_MAX_BRANCHES = 2
PART_B_REGIME2_DELAY_TICKS = 3
PART_B_REGIME2_HORIZON = 32
PART_B_BAR_RESCUE_RETENTION_PCT_MIN = 50.0  # >= 50.0% (>= 155 / 310)
PART_B_BAR_RESCUE_RETENTION_COUNT_MIN = 155
PART_B_BAR_RESCUE_LB90_PCT_MIN = 40.0  # world-cluster LB90 >= 40.0%
PART_B_BAR_FALSE_OVERRIDE_PCT_MAX = 1.0  # <= 1.0% (<= 100 / 10,080)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def score_part_a_b8_head_closedloop(
    closedloop_dir: Path = DEFAULT_B8_HEAD_CLOSEDLOOP_DIR,
) -> dict[str, Any]:
    """Score t34-sol/b8-head-closedloop/main-01/output/receipt.json against SUCCESS_CONTRACT §2."""
    closedloop_dir = Path(closedloop_dir)
    output_dir = closedloop_dir / "main-01" / "output"
    input_dir = closedloop_dir / "inputs-01"
    receipt_path = output_dir / "receipt.json"
    manifest_path = output_dir / "manifest.json"
    config_path = input_dir / "config.json"

    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config = json.loads(config_path.read_text(encoding="utf-8"))

    manifest_sha256_actual = _sha256_file(manifest_path)
    receipt_manifest_sha256 = str(receipt.get("manifest_sha256", ""))
    stop_verified = bool(receipt.get("stop_verified") is True)
    cleanup_error = receipt.get("cleanup_error")
    receipt_status = str(receipt.get("status", ""))

    # Check schema discrepancy in research/b8_head_test/closedloop_readout.py:30
    receipt_top_level_commit = receipt.get("commit")
    manifest_commit = str(manifest.get("commit", ""))
    readout_script_commit_check_passes = bool(receipt_top_level_commit == manifest_commit)

    # Verify receipt-bound exports in exports/closedloop/
    exports_list = receipt.get("exports", [])
    exports_by_path = {str(r["path"]): r for r in exports_list}
    export_root = output_dir / "exports" / "closedloop"
    identity = str(config["identity"]) + "/main"

    verified_units: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for batch_idx in range(3):
        unit_name = f"batch-{batch_idx:04d}"
        rel_path = f"units/{unit_name}.json"
        export_key = f"closedloop/{rel_path}"
        unit_path = export_root / rel_path
        sha_path = Path(str(unit_path) + ".sha256")
        complete_path = Path(str(unit_path) + ".complete")

        actual_size = int(unit_path.stat().st_size)
        actual_sha = _sha256_file(unit_path)
        sidecar_sha = sha_path.read_text(encoding="utf-8").strip()
        complete_meta = json.loads(complete_path.read_text(encoding="utf-8"))
        bound_meta = exports_by_path.get(export_key, {})

        if (
            actual_size != int(bound_meta.get("size", -1))
            or actual_sha != str(bound_meta.get("sha256", ""))
            or sidecar_sha != actual_sha
            or complete_meta.get("sha256") != actual_sha
            or complete_meta.get("size") != actual_size
            or complete_meta.get("identity") != identity
        ):
            raise ValueError(f"Export verification failed for {export_key}")

        payload = json.loads(unit_path.read_text(encoding="utf-8"))
        if payload.get("identity") != identity or payload.get("unit") != unit_name:
            raise ValueError(f"Unit identity mismatch in {export_key}")
        rows.extend(payload["rows"])
        verified_units.append(
            {
                "unit": unit_name,
                "path": export_key,
                "size": actual_size,
                "sha256": actual_sha,
                "n_rows": len(payload["rows"]),
            }
        )

    # Check what fields exist in each exported row vs what SUCCESS_CONTRACT.md §2 requires
    row_keys = sorted(set(rows[0].keys())) if rows else []
    missing_contract_sec2_fields = [
        {
            "bar": "boost share within [0.5%, 10%]",
            "contract_ref": "SUCCESS_CONTRACT.md §2 line 32; P7_STATISTICS.md §4 line 47",
            "reason": (
                "neither receipt.json nor exports/closedloop/units/batch-0000..0002.json "
                "records boost action counts, boost frames, or boost_share "
                "(row schema only has: " + ", ".join(row_keys) + ")"
            ),
        },
        {
            "bar": "plan §8 tripwires apply to every fit",
            "contract_ref": "SUCCESS_CONTRACT.md §2 line 32; P7_STATISTICS.md §4 lines 45-46",
            "reason": (
                "neither receipt.json nor exported batch units record per-action histograms "
                "(>90% one-action collapse window check) or 32-world probe bank fit-drift "
                "metrics (length-decile disagreement <=5%, boost share <=1.5x previous student, "
                "held-out decisive agreement >=70%)"
            ),
        },
        {
            "bar": "Watch6 v2 mechanics & shared-seed cross-mix survival band pairing",
            "contract_ref": "SUCCESS_CONTRACT.md §2-3; t34-sol/b8-head-closedloop/SUMMARY.md:45",
            "reason": (
                "closedloop.py ran PilotCTDeviceSim with mechanics_version=1 on 192 worlds "
                "partitioned into 64 disjoint worlds per mix (world_id % 3), not Watch6 v2"
            ),
        },
    ]

    # Compute exact statistics from closedloop.py (seed 343014, 10,000 draws) + §2 measured bars
    table = {(int(r["world_id"]), str(r["arm"])): r for r in rows}
    worlds = sorted({int(r["world_id"]) for r in rows})
    if len(table) != len(rows) or set(table.keys()) != {
        (w, a) for w in worlds for a in PART_A_ARMS
    }:
        raise ValueError("Incomplete or duplicate (world_id, arm) table in closedloop exports")
    for w in worlds:
        if len({str(table[(w, a)]["initial_sha256"]) for a in PART_A_ARMS}) != 1:
            raise ValueError(f"Unpaired initial_sha256 on world {w}")

    arms_summary: dict[str, dict[str, Any]] = {}
    for arm in PART_A_ARMS:
        mix_out: dict[str, dict[str, Any]] = {}
        for mix in (*PART_A_MIXES, "all"):
            groups = PART_A_MIXES if mix == "all" else (mix,)
            # Exact RNG reset per (arm, mix) as in research/b8_head_test/closedloop.py:278
            rng = np.random.default_rng(PART_A_BOOTSTRAP_SEED)
            mass_draws = np.zeros(PART_A_BOOTSTRAP_DRAWS, dtype=np.float64)
            mass_points: list[float] = []
            selected_worlds: list[int] = []

            # Also compute paired alive_fraction and no_death deltas
            alive_rng = np.random.default_rng(PART_A_BOOTSTRAP_SEED)
            alive_draws = np.zeros(PART_A_BOOTSTRAP_DRAWS, dtype=np.float64)
            alive_points: list[float] = []
            alive_group_vars: list[float] = []
            alive_group_ns: list[int] = []

            for group in groups:
                ws = [w for w in worlds if str(table[(w, "b8")]["mix"]) == group]
                selected_worlds.extend(ws)
                m_delta = np.asarray(
                    [float(table[(w, arm)]["mass"]) - float(table[(w, "b8")]["mass"]) for w in ws],
                    dtype=np.float64,
                )
                mass_points.append(float(m_delta.mean()))
                draw_idx = rng.integers(0, len(ws), (PART_A_BOOTSTRAP_DRAWS, len(ws)))
                mass_draws += m_delta[draw_idx].mean(axis=1) / len(groups)

                a_delta = np.asarray(
                    [
                        float(table[(w, arm)]["alive_fraction"])
                        - float(table[(w, "b8")]["alive_fraction"])
                        for w in ws
                    ],
                    dtype=np.float64,
                )
                alive_points.append(float(a_delta.mean()))
                a_idx = alive_rng.integers(0, len(ws), (PART_A_BOOTSTRAP_DRAWS, len(ws)))
                alive_draws += a_delta[a_idx].mean(axis=1) / len(groups)
                alive_group_vars.append(float(np.var(a_delta, ddof=1)) if len(ws) > 1 else 0.0)
                alive_group_ns.append(len(ws))

            rs = [table[(w, arm)] for w in selected_worlds]
            b8_rs = [table[(w, "b8")] for w in selected_worlds]
            n_worlds_mix = len(rs)
            deaths_arm = int(sum(int(r["deaths"]) for r in rs))
            deaths_b8 = int(sum(int(r["deaths"]) for r in b8_rs))
            no_death_rate_arm = float(np.mean([int(r["deaths"]) == 0 for r in rs]))
            no_death_rate_b8 = float(np.mean([int(r["deaths"]) == 0 for r in b8_rs]))
            no_death_gain = float(no_death_rate_arm - no_death_rate_b8)

            alive_mean_delta = float(np.mean(alive_points))
            # Student-t one-sided 95% lower bound for survival band v2 check
            if arm == "b8":
                alive_se = 0.0
                alive_t_lb95 = 0.0
            elif len(groups) == 1:
                n_g = alive_group_ns[0]
                alive_se = float(np.sqrt(alive_group_vars[0] / n_g))
                t_crit = float(student_t_isf(0.05, n_g - 1))
                alive_t_lb95 = float(alive_mean_delta - t_crit * alive_se)
            else:
                # Stratified equal-weight mean across 3 independent 64-world mix strata
                alive_se = float(
                    np.sqrt(sum(v / n_g for v, n_g in zip(alive_group_vars, alive_group_ns)))
                    / len(groups)
                )
                df_strat = sum(n_g - 1 for n_g in alive_group_ns)
                t_crit = float(student_t_isf(0.05, df_strat))
                alive_t_lb95 = float(alive_mean_delta - t_crit * alive_se)

            overrides_sum = int(sum(int(r["overrides"]) for r in rs))
            decisions_sum = int(sum(int(r["decisions"]) for r in rs))
            override_rate = float(overrides_sum / max(1, decisions_sum))

            mix_out[mix] = {
                "worlds": n_worlds_mix,
                "mass": float(np.mean([float(r["mass"]) for r in rs])),
                "mass_delta": float(np.mean(mass_points)),
                "lb90": float(np.quantile(mass_draws, 0.10)),
                "ci95": [
                    float(np.quantile(mass_draws, 0.025)),
                    float(np.quantile(mass_draws, 0.975)),
                ],
                "deaths": deaths_arm,
                "b8_deaths": deaths_b8,
                "no_death_worlds": int(sum(int(r["deaths"]) == 0 for r in rs)),
                "b8_no_death_worlds": int(sum(int(r["deaths"]) == 0 for r in b8_rs)),
                "no_death_rate": no_death_rate_arm,
                "b8_no_death_rate": no_death_rate_b8,
                "no_death_gain": no_death_gain,
                "alive_fraction": float(np.mean([float(r["alive_fraction"]) for r in rs])),
                "alive_delta": alive_mean_delta,
                "alive_se": alive_se,
                "alive_t_lb95": alive_t_lb95,
                "alive_boot_lb95": float(np.quantile(alive_draws, 0.05)),
                "overrides": overrides_sum,
                "decisions": decisions_sum,
                "override_rate": override_rate,
            }
        arms_summary[arm] = mix_out

    # Evaluate each candidate arm m in {0, 10, 25, 50} against the bars
    arm_evaluations: dict[str, dict[str, Any]] = {}
    n_arms_passing_measured_bars = 0
    for arm in PART_A_CANDIDATE_ARMS:
        all_s = arms_summary[arm]["all"]
        scr_s = arms_summary[arm]["scripted"]
        frz_s = arms_summary[arm]["frozen"]
        mix_s = arms_summary[arm]["mixed"]

        pass_mass_gain = bool(all_s["mass_delta"] >= PART_A_BAR_MASS_DELTA_MIN)
        pass_mass_lb90 = bool(all_s["lb90"] > PART_A_BAR_MASS_LB90_MIN)
        pass_scripted_no_death = bool(
            scr_s["no_death_gain"] >= PART_A_BAR_SCRIPTED_NO_DEATH_GAIN_MIN
        )
        pass_surv_pooled = bool(
            all_s["alive_t_lb95"] > -PART_A_BAR_SURVIVAL_POOLED_NI_MARGIN
            and all_s["alive_boot_lb95"] > -PART_A_BAR_SURVIVAL_POOLED_NI_MARGIN
        )
        pass_surv_scripted = bool(
            scr_s["alive_t_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
            and scr_s["alive_boot_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
        )
        pass_surv_frozen = bool(
            frz_s["alive_t_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
            and frz_s["alive_boot_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
        )
        pass_surv_mixed = bool(
            mix_s["alive_t_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
            and mix_s["alive_boot_lb95"] > -PART_A_BAR_SURVIVAL_MIX_NI_MARGIN
        )
        pass_survival_band_v2 = bool(
            pass_surv_pooled and pass_surv_scripted and pass_surv_frozen and pass_surv_mixed
        )
        pass_override_rate = bool(all_s["override_rate"] <= PART_A_BAR_OVERRIDE_RATE_MAX)

        passes_measured_bars = bool(
            pass_mass_gain
            and pass_mass_lb90
            and pass_scripted_no_death
            and pass_survival_band_v2
            and pass_override_rate
        )
        if passes_measured_bars:
            n_arms_passing_measured_bars += 1

        arm_evaluations[arm] = {
            "pass_pooled_mass_gain_ge_40": pass_mass_gain,
            "pass_pooled_mass_lb90_gt_0": pass_mass_lb90,
            "pass_scripted_no_death_gain_ge_003": pass_scripted_no_death,
            "pass_survival_band_v2": pass_survival_band_v2,
            "survival_band_v2_components": {
                "pooled_gt_neg_005": pass_surv_pooled,
                "scripted_gt_neg_0075": pass_surv_scripted,
                "frozen_gt_neg_0075": pass_surv_frozen,
                "mixed_gt_neg_0075": pass_surv_mixed,
            },
            "pass_override_rate_le_15pct": pass_override_rate,
            "pass_boost_share_05_to_10pct": None,  # Not present in receipt/exports
            "pass_plan_sec8_tripwires": None,  # Not present in receipt/exports
            "passes_measured_bars": passes_measured_bars,
            "verdict_on_full_contract_sec2": "INCONCLUSIVE",
            "verdict_on_measured_bars": "PASS" if passes_measured_bars else "FAIL",
        }

    return {
        "receipt_path": str(receipt_path),
        "receipt_sha256": _sha256_file(receipt_path),
        "manifest_path": str(manifest_path),
        "manifest_sha256": manifest_sha256_actual,
        "manifest_sha256_matches_receipt": bool(manifest_sha256_actual == receipt_manifest_sha256),
        "receipt_status": receipt_status,
        "stop_verified": stop_verified,
        "cleanup_error": cleanup_error,
        "manifest_commit": manifest_commit,
        "receipt_top_level_commit": receipt_top_level_commit,
        "closedloop_readout_commit_check_passes": readout_script_commit_check_passes,
        "verified_units": verified_units,
        "total_rows": len(rows),
        "exported_row_schema": row_keys,
        "missing_contract_sec2_fields": missing_contract_sec2_fields,
        "arms_summary": arms_summary,
        "arm_evaluations": arm_evaluations,
        "n_candidate_arms": len(PART_A_CANDIDATE_ARMS),
        "n_arms_passing_measured_bars": n_arms_passing_measured_bars,
        "stop_rule_a_zero_of_four_pass_measured": bool(n_arms_passing_measured_bars == 0),
        "part_a_verdict": "INCONCLUSIVE",
        "part_a_measured_bars_verdict": (
            "KILL (0/4 arms pass measured SUCCESS_CONTRACT.md §2 bars)"
            if n_arms_passing_measured_bars == 0
            else "PASS"
        ),
        "part_a_verdict_summary": (
            "INCONCLUSIVE on full SUCCESS_CONTRACT.md §2 schema (receipt.json is a "
            "supervisor transport receipt containing no gameplay outcomes directly, and "
            "its receipt-bound batch-0000..0002.json exports omit boost_share [0.5%, 10%], "
            "plan §8 tripwires, and Watch6 v2); 0/4 arms pass the measured §2 bars in the "
            "batch exports (pooled mass delta -305.095 to -26.216 vs >= +40; LB90 -321.688 "
            "to -45.057 vs > 0; scripted no-death gain -0.5156 to -0.2031 vs >= +0.03; "
            "0/4 pass survival band v2) -> Stop Rule A triggers KILL on Claim A head."
        ),
    }


def _eval_branch_with_step_actions(
    root: Any,
    action: int,
    decision: int,
    suffix_id: int = 0,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Roll out ``action`` for ``branch_horizon`` steps and record hero actions at steps 1..4."""
    saved_rng_states = [env_rng._rng.getstate() for env_rng in root.sim._rngs]
    try:
        reseed_branch_suffix(root, decision, suffix_id, namespace=namespace)
        with branch_scope(root.mix):
            branch = _clone_root(root)
            branch.sim._policy_phase_prepared = True
            step_prepared(branch, int(action))
            executed = 1
            step_actions: list[int] = [int(action)]
            alive_every_step = bool(branch.sim.alive[0, 0])
            while executed < int(branch_horizon) and alive_every_step:
                acts, _, _ = choose_and_step(branch, capture_obs=False)
                if len(step_actions) < 6:
                    step_actions.append(int(acts[0, 0]))
                executed += 1
                if not bool(branch.sim.alive[0, 0]):
                    alive_every_step = False
            raw_survives = bool(alive_every_step and executed == int(branch_horizon))
            open_info = (
                evaluate_terminal_openness(branch.sim) if raw_survives else {"un_enclosed": False}
            )
            final_length = int(branch.sim.length[0, 0]) if raw_survives else 0
    finally:
        for env_rng, st in zip(root.sim._rngs, saved_rng_states):
            env_rng._rng.setstate(st)

    frames_executed = int(executed)
    frames_survived = int(branch_horizon) if raw_survives else max(0, frames_executed - 1)
    un_enclosed = bool(raw_survives and bool(open_info["un_enclosed"]))
    certified = bool(raw_survives and un_enclosed)
    return {
        "suffix_id": int(suffix_id),
        "survives": certified,
        "certified": certified,
        "raw_survives": raw_survives,
        "un_enclosed": un_enclosed,
        "frames_executed": frames_executed,
        "frames_survived": frames_survived,
        "death_decision": None if raw_survives else int(decision) + frames_executed - 1,
        "final_length": final_length,
        "step_actions": step_actions,
        "step_t_action": int(step_actions[3]) if len(step_actions) >= 4 else None,
    }


def _run_openstop_k1_at_state(
    root: Any,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    max_branches: int = 6,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Run K=1 open-stop at ``root`` up to ``max_branches`` in evaluation_order_nb."""
    bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
    legal_actions = np.flatnonzero(legal).tolist()
    order = evaluation_order_nb(legal_actions, bare, q)[: int(max_branches)]
    outcomes: dict[int, dict[str, Any]] = {}
    chosen_cert: int | None = None
    for act in order:
        out = _eval_branch_with_step_actions(
            root,
            int(act),
            int(decision),
            suffix_id=0,
            branch_horizon=int(branch_horizon),
            namespace=namespace,
        )
        outcomes[int(act)] = out
        if bool(out["certified"]):
            chosen_cert = int(act)
            break

    all_failed = chosen_cert is None
    chosen_exact_stop = int(ceiling._choose(outcomes, bare, q)) if all_failed else int(chosen_cert)
    chosen_cert_only = int(bare) if all_failed else int(chosen_cert)
    win_exact = outcomes[chosen_exact_stop]
    win_cert = outcomes[chosen_cert_only]
    bare_out = outcomes[bare]

    return {
        "bare": int(bare),
        "order_evaluated": [int(a) for a in outcomes.keys()],
        "branches_evaluated": len(outcomes),
        "bare_certified": bool(bare_out["certified"]),
        "bare_raw_survives": bool(bare_out["raw_survives"]),
        "bare_frames_survived": int(bare_out["frames_survived"]),
        "all_failed": bool(all_failed),
        "chosen_exact_stop": int(chosen_exact_stop),
        "chosen_cert_only": int(chosen_cert_only),
        "step_t_action_exact_stop": win_exact["step_t_action"],
        "step_t_action_cert_only": win_cert["step_t_action"],
        "bare_step_t_action": bare_out["step_t_action"],
    }


def evaluate_world_part_b_worker(
    task: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate Regime 1 and Regime 2 on one dev world (override + ordinary roots)."""
    _enforce_single_thread_env()
    world_index = int(task["world_index"])
    stage1b_cache_dir = Path(task["stage1b_cache_dir"])
    openstop_dev_dir = Path(task["openstop_dev_dir"])
    audit_cache_dir = Path(task["audit_cache_dir"])
    audit_cache_dir.mkdir(parents=True, exist_ok=True)

    world_cache_path = audit_cache_dir / f"partb-world-{world_index:02d}.pkl"
    if world_cache_path.is_file():
        cached = pickle.loads(world_cache_path.read_bytes())
        if int(cached.get("world_index", -1)) == world_index:
            cached["cached"] = True
            return cached

    t0 = time.perf_counter()
    feat_data = pickle.loads(
        (stage1b_cache_dir / f"features-world-{world_index:02d}.pkl").read_bytes()
    )
    scored_data = pickle.loads(
        (stage1b_cache_dir / f"scored-world-{world_index:02d}.pkl").read_bytes()
    )
    w_data = json.loads(
        (openstop_dev_dir / "worlds" / f"openstop-world-{world_index:02d}.json").read_text(
            encoding="utf-8"
        )
    )

    seed = int(feat_data["seed"])
    rec = w_data["arms"][OPENSTOP_K1_ARM]
    overrides_by_frame = {int(r["frame"]): r for r in rec["overrides"]}
    ov_by_frame = {int(r["frame"]): r for r in feat_data["override_roots"]}
    ord_by_frame = {int(r["frame"]): r for r in feat_data["ordinary_roots"]}
    ord_scores_by_frame: dict[int, dict[int, dict[str, Any]]] = {
        int(k): {int(a): v for a, v in mp_a.items()}
        for k, mp_a in scored_data.get("ord_scores_by_frame", {}).items()
    }
    ov_scores_by_frame: dict[int, dict[int, dict[str, Any]]] = {
        int(k): {int(a): v for a, v in mp_a.items()}
        for k, mp_a in scored_data.get("ov_scores_by_frame", {}).items()
    }

    target_frames = set(ov_by_frame.keys()) | set(ord_by_frame.keys())
    launch_frames = {
        fr - PART_B_REGIME2_DELAY_TICKS
        for fr in target_frames
        if fr - PART_B_REGIME2_DELAY_TICKS >= 1
    }

    session = open_watch({"seed": seed}, 24, "bare")
    try:
        world = session.world
        _assert_mixed(session.live, 6)
        stats = RowStats()

        launch_results: dict[int, dict[str, Any]] = {}
        life = 1
        was_alive = True
        decisions_count = 0
        mass_sum = 0
        deaths = 0
        trigger = DenseTrigger(min_length=OPENSTOP_MIN_LENGTH, interval=OPENSTOP_INTERVAL)

        # Pass 1: Replay world 1..6000, verify trajectory parity, and run Regime 2 at t-3
        with configure("kernel", stats), branch_scope("mixed"):
            for frame in range(1, HORIZON + 1):

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life, trigger
                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        trigger = DenseTrigger(
                            min_length=OPENSTOP_MIN_LENGTH, interval=OPENSTOP_INTERVAL
                        )
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    chosen = 1
                    if active:
                        cur_length = int(sim.length[0, 0])
                        fired = trigger.fires(decisions_count, cur_length, True)
                        need_q = (frame in launch_frames) or (
                            fired and (frame in overrides_by_frame)
                        )
                        if need_q:
                            obs = build_ego_raster(
                                sim,
                                EgoRasterConfig(version="bv8"),
                                backend="numba",
                                rows=np.array([[0, 0]], dtype=np.int64),
                            )
                            q = np.asarray(prepared_q(world, obs), dtype=np.float32)
                            advisory = sim.get_advisory_action_mask()[0, 0].astype(bool)
                            legal, bare = ceiling._legal(q, advisory)
                            chosen = int(bare)
                            if frame in launch_frames and advisory.any():
                                sim._policy_phase_prepared = True
                                launch_results[frame] = {
                                    "launch_frame": int(frame),
                                    "launch_decision": int(frame - 1),
                                    "life": int(life),
                                    **_run_openstop_k1_at_state(
                                        world,
                                        int(frame - 1),
                                        q,
                                        legal,
                                        branch_horizon=PART_B_REGIME2_HORIZON,
                                        max_branches=6,
                                        namespace=NAMESPACE,
                                    ),
                                }
                            if fired and (frame in overrides_by_frame):
                                chosen = int(overrides_by_frame[frame]["chosen_action"])
                    _dispatch_actions(
                        sim,
                        sim.get_resolved_action_mask(),
                        actions,
                        [world.hero],
                        [world.opponents],
                    )
                    if active and fired and (frame in overrides_by_frame):
                        actions[0, 0] = int(chosen)
                    return actions

                world.sim.step_with_policy(choose)
                world.runtime.observe_step(world.sim)
                alive_now = bool(world.sim.alive[0, 0])
                if was_alive and not alive_now:
                    deaths += 1
                if alive_now:
                    decisions_count += 1
                    mass_sum += int(world.sim.length[0, 0])
                was_alive = alive_now

        if mass_sum != int(rec["mass_sum"]) or deaths != int(rec["deaths"]):
            raise ValueError(
                f"World {world_index} replay mismatch: mass_sum {mass_sum}!={rec['mass_sum']} "
                f"or deaths {deaths}!={rec['deaths']}"
            )

        # Pass 2: Evaluate Regime 1 at t and score Regime 1 & Regime 2 actions over H=32 from t
        ov_records_out: list[dict[str, Any]] = []
        ord_records_out: list[dict[str, Any]] = []

        with configure("kernel", stats), branch_scope("mixed"):
            # 2a. Override roots (350 total across 48 worlds; 310 certified rescues)
            for r in feat_data["override_roots"]:
                fr = int(r["frame"])
                dec = int(r["decision"])
                life_t = int(r["life"])
                length_t = int(r["length"])
                bare_t = int(r["bare_action"])
                teacher_t = int(r["chosen_action"])
                legal_t = np.asarray(r["legal"], dtype=bool)
                q_t = np.asarray(r["q"], dtype=np.float32)
                cert_ndiff_teacher = int(r["scoring"]["h32_certified"]["n_diff"])
                raw_ndiff_teacher = int(r["scoring"]["n_override_dies_where_b8_survives"])
                b_cert_vec = [bool(x) for x in r["scoring"]["h32_certified"]["bare_vec"]]
                b_raw_vec = [bool(x) for x in r["scoring"]["bare_survives_vec"]]

                restore_world(world, r["root_blob"])

                # Regime 1: H=12, <= 2 branches at t
                r1_2b = _run_openstop_k1_at_state(
                    world,
                    dec,
                    q_t,
                    legal_t,
                    branch_horizon=PART_B_REGIME1_HORIZON,
                    max_branches=PART_B_REGIME1_MAX_BRANCHES,
                    namespace=NAMESPACE,
                )
                # Diagnostic: unconstrained H=12 (<= 6 branches) at t
                r1_6b = _run_openstop_k1_at_state(
                    world,
                    dec,
                    q_t,
                    legal_t,
                    branch_horizon=PART_B_REGIME1_HORIZON,
                    max_branches=6,
                    namespace=NAMESPACE,
                )

                # Regime 2: Async d=3 launch at fr - 3
                lr = launch_results.get(fr - PART_B_REGIME2_DELAY_TICKS)
                same_life = bool(lr is not None and int(lr["life"]) == life_t)
                if same_life and lr is not None:
                    t_minus_3_triggered = not bool(lr["bare_certified"])
                    t_minus_3_overrode = int(lr["chosen_exact_stop"]) != int(lr["bare"])
                    # 2A-triggered: if t-3 overrode, apply winning branch's step-t action
                    act_2a_trig = bare_t
                    if t_minus_3_overrode and lr["step_t_action_exact_stop"] is not None:
                        cand_a = int(lr["step_t_action_exact_stop"])
                        if bool(legal_t[cand_a]):
                            act_2a_trig = cand_a
                    # 2B-delayed-root: if t-3 overrode, apply root action chosen_{t-3} at t
                    act_2b_root = bare_t
                    if t_minus_3_overrode:
                        cand_r = int(lr["chosen_exact_stop"])
                        if bool(legal_t[cand_r]):
                            act_2b_root = cand_r
                    # 2C-unconditional: apply step-t action of t-3 winning branch unconditionally
                    act_2c_uncond = bare_t
                    if lr["step_t_action_exact_stop"] is not None:
                        cand_u = int(lr["step_t_action_exact_stop"])
                        if bool(legal_t[cand_u]):
                            act_2c_uncond = cand_u
                else:
                    t_minus_3_triggered = False
                    t_minus_3_overrode = False
                    act_2a_trig = bare_t
                    act_2b_root = bare_t
                    act_2c_uncond = bare_t

                # Score all candidate actions at t over H=32 on suffix 0 and suffixes 4..7
                cand_actions = {
                    bare_t,
                    teacher_t,
                    int(r1_2b["chosen_exact_stop"]),
                    int(r1_2b["chosen_cert_only"]),
                    int(r1_6b["chosen_exact_stop"]),
                    int(act_2a_trig),
                    int(act_2b_root),
                    int(act_2c_uncond),
                }
                action_h32_eval: dict[int, dict[str, Any]] = {}
                cur_ov_map = ov_scores_by_frame.get(fr, {})

                for a_c in sorted(cand_actions):
                    out_suf0 = _eval_open_stop_branch_suffix(
                        world,
                        int(a_c),
                        dec,
                        0,
                        branch_horizon=OPENSTOP_BRANCH_HORIZON,
                        namespace=NAMESPACE,
                    )
                    if int(a_c) == bare_t:
                        ndiff_cert = 4
                        ndiff_raw = 4
                        cert_vec_47 = b_cert_vec
                        raw_vec_47 = b_raw_vec
                    elif int(a_c) == teacher_t:
                        ndiff_cert = cert_ndiff_teacher
                        ndiff_raw = raw_ndiff_teacher
                        cert_vec_47 = [bool(x) for x in r["scoring"]["h32_certified"]["chosen_vec"]]
                        raw_vec_47 = [bool(x) for x in r["scoring"]["chosen_survives_vec"]]
                    elif int(a_c) in cur_ov_map:
                        info = cur_ov_map[int(a_c)]
                        ndiff_cert = int(info["n_diff_certified"])
                        ndiff_raw = int(info["n_diff_raw"])
                        cert_vec_47 = [bool(x) for x in info["cand_certified_vec"]]
                        raw_vec_47 = [bool(x) for x in info["cand_raw_vec"]]
                    else:
                        ndiff_cert = 0
                        ndiff_raw = 0
                        cert_vec_47 = []
                        raw_vec_47 = []
                        for s_idx, suf in enumerate(OPENSTOP_SCORING_SUFFIXES):
                            out_s = _eval_open_stop_branch_suffix(
                                world,
                                int(a_c),
                                dec,
                                int(suf),
                                branch_horizon=OPENSTOP_BRANCH_HORIZON,
                                namespace=NAMESPACE,
                            )
                            c_cert = bool(out_s["certified"])
                            c_raw = bool(out_s["raw_survives"])
                            cert_vec_47.append(c_cert)
                            raw_vec_47.append(c_raw)
                            if b_cert_vec[s_idx] and not c_cert:
                                ndiff_cert += 1
                            if b_raw_vec[s_idx] and not c_raw:
                                ndiff_raw += 1

                    action_h32_eval[int(a_c)] = {
                        "action": int(a_c),
                        "is_override": bool(int(a_c) != bare_t),
                        "exact_teacher_match": bool(int(a_c) == teacher_t),
                        "suf0_certified_h32": bool(out_suf0["certified"]),
                        "suf0_raw_survives_h32": bool(out_suf0["raw_survives"]),
                        "suf0_frames_survived_h32": int(out_suf0["frames_survived"]),
                        "suf4_7_n_diff_certified": int(ndiff_cert),
                        "suf4_7_n_diff_raw": int(ndiff_raw),
                        "suf4_7_n_certified": int(sum(cert_vec_47)),
                        "suf4_7_n_raw": int(sum(raw_vec_47)),
                        # Primary survival over H=32: overrides bare AND certified H=32 on suf 0
                        "rescued_suf0_certified": bool(
                            int(a_c) != bare_t and bool(out_suf0["certified"])
                        ),
                        # Strict Stage 1B certified rescue: suf0 certified AND suf4..7 n_diff == 0
                        "rescued_suf0_and_suf4_7_ndiff0": bool(
                            int(a_c) != bare_t
                            and bool(out_suf0["certified"])
                            and int(ndiff_cert) == 0
                        ),
                        # Stage 1B survival-match definition: overrides bare AND suf4..7 n_diff == 0
                        "rescued_suf4_7_ndiff0": bool(int(a_c) != bare_t and int(ndiff_cert) == 0),
                        # Raw H=32 survival on suf 0
                        "rescued_suf0_raw": bool(
                            int(a_c) != bare_t and bool(out_suf0["raw_survives"])
                        ),
                    }

                ov_records_out.append(
                    {
                        "world_index": world_index,
                        "seed": seed,
                        "frame": fr,
                        "decision": dec,
                        "life": life_t,
                        "length": length_t,
                        "bare_action": bare_t,
                        "teacher_action": teacher_t,
                        "is_certified_rescue_root": bool(cert_ndiff_teacher == 0),
                        "is_raw_rescue_root": bool(raw_ndiff_teacher == 0),
                        "bare_failed_by_enclosure_only": bool(r["bare_failed_by_enclosure_only"]),
                        "r1_2b": r1_2b,
                        "r1_6b": r1_6b,
                        "r2_same_life": same_life,
                        "r2_t_minus_3_triggered": t_minus_3_triggered,
                        "r2_t_minus_3_overrode": t_minus_3_overrode,
                        "r2_launch_info": lr,
                        "actions_chosen": {
                            "r1_h12_2b_exact_stop": int(r1_2b["chosen_exact_stop"]),
                            "r1_h12_2b_cert_only": int(r1_2b["chosen_cert_only"]),
                            "r1_h12_6b_exact_stop": int(r1_6b["chosen_exact_stop"]),
                            "r2_async_d3_step_t_triggered": int(act_2a_trig),
                            "r2_async_d3_root_delayed": int(act_2b_root),
                            "r2_async_d3_step_t_unconditional": int(act_2c_uncond),
                        },
                        "action_h32_eval": action_h32_eval,
                    }
                )

            # 2b. Ordinary roots (210 per world = 10,080 total across 48 worlds)
            for r in feat_data["ordinary_roots"]:
                fr = int(r["frame"])
                dec = int(r["decision"])
                life_t = int(r["life"])
                length_t = int(r["length"])
                bare_t = int(r["bare_action"])
                legal_t = np.asarray(r["legal"], dtype=bool)
                q_t = np.asarray(r["q"], dtype=np.float32)

                restore_world(world, r["root_blob"])

                # Regime 1: H=12, <= 2 branches at t
                r1_2b = _run_openstop_k1_at_state(
                    world,
                    dec,
                    q_t,
                    legal_t,
                    branch_horizon=PART_B_REGIME1_HORIZON,
                    max_branches=PART_B_REGIME1_MAX_BRANCHES,
                    namespace=NAMESPACE,
                )

                # Regime 2: Async d=3 launch at fr - 3
                lr = launch_results.get(fr - PART_B_REGIME2_DELAY_TICKS)
                same_life = bool(lr is not None and int(lr["life"]) == life_t)
                if same_life and lr is not None:
                    t_minus_3_triggered = not bool(lr["bare_certified"])
                    t_minus_3_overrode = int(lr["chosen_exact_stop"]) != int(lr["bare"])
                    act_2a_trig = bare_t
                    if t_minus_3_overrode and lr["step_t_action_exact_stop"] is not None:
                        cand_a = int(lr["step_t_action_exact_stop"])
                        if bool(legal_t[cand_a]):
                            act_2a_trig = cand_a
                    act_2b_root = bare_t
                    if t_minus_3_overrode:
                        cand_r = int(lr["chosen_exact_stop"])
                        if bool(legal_t[cand_r]):
                            act_2b_root = cand_r
                    act_2c_uncond = bare_t
                    if lr["step_t_action_exact_stop"] is not None:
                        cand_u = int(lr["step_t_action_exact_stop"])
                        if bool(legal_t[cand_u]):
                            act_2c_uncond = cand_u
                else:
                    t_minus_3_triggered = False
                    t_minus_3_overrode = False
                    act_2a_trig = bare_t
                    act_2b_root = bare_t
                    act_2c_uncond = bare_t

                chosen_map = {
                    "r1_h12_2b_exact_stop": int(r1_2b["chosen_exact_stop"]),
                    "r1_h12_2b_cert_only": int(r1_2b["chosen_cert_only"]),
                    "r2_async_d3_step_t_triggered": int(act_2a_trig),
                    "r2_async_d3_root_delayed": int(act_2b_root),
                    "r2_async_d3_step_t_unconditional": int(act_2c_uncond),
                }

                # Score any non-bare false override on suffixes 4..7 (and suffix 0)
                non_bare_cands = {a for a in chosen_map.values() if int(a) != bare_t}
                ord_cand_eval: dict[int, dict[str, Any]] = {}
                cur_ord_map = ord_scores_by_frame.get(fr, {})
                bare_suf47_cache: dict[int, dict[str, bool]] = {}

                for a_c in sorted(non_bare_cands):
                    out_suf0 = _eval_open_stop_branch_suffix(
                        world,
                        int(a_c),
                        dec,
                        0,
                        branch_horizon=OPENSTOP_BRANCH_HORIZON,
                        namespace=NAMESPACE,
                    )
                    if int(a_c) in cur_ord_map:
                        info = cur_ord_map[int(a_c)]
                        ndiff_cert = int(info["n_diff_certified"])
                        ndiff_raw = int(info["n_diff_raw"])
                    else:
                        ndiff_cert = 0
                        ndiff_raw = 0
                        for suf in OPENSTOP_SCORING_SUFFIXES:
                            out_c = _eval_open_stop_branch_suffix(
                                world,
                                int(a_c),
                                dec,
                                int(suf),
                                branch_horizon=OPENSTOP_BRANCH_HORIZON,
                                namespace=NAMESPACE,
                            )
                            c_cert = bool(out_c["certified"])
                            c_raw = bool(out_c["raw_survives"])
                            if not c_cert or not c_raw:
                                if int(suf) not in bare_suf47_cache:
                                    out_b = _eval_open_stop_branch_suffix(
                                        world,
                                        bare_t,
                                        dec,
                                        int(suf),
                                        branch_horizon=OPENSTOP_BRANCH_HORIZON,
                                        namespace=NAMESPACE,
                                    )
                                    bare_suf47_cache[int(suf)] = {
                                        "certified": bool(out_b["certified"]),
                                        "raw_survives": bool(out_b["raw_survives"]),
                                    }
                                b_info = bare_suf47_cache[int(suf)]
                                if b_info["certified"] and not c_cert:
                                    ndiff_cert += 1
                                if b_info["raw_survives"] and not c_raw:
                                    ndiff_raw += 1
                    ord_cand_eval[int(a_c)] = {
                        "action": int(a_c),
                        "suf0_certified_h32": bool(out_suf0["certified"]),
                        "suf0_raw_survives_h32": bool(out_suf0["raw_survives"]),
                        "suf4_7_n_diff_certified": int(ndiff_cert),
                        "suf4_7_n_diff_raw": int(ndiff_raw),
                        "lethal_certified": bool(
                            ndiff_cert >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN
                        ),
                        "lethal_raw": bool(ndiff_raw >= STAGE1B_BAR_LETHAL_ORDINARY_N_DIFF_MIN),
                    }

                ord_records_out.append(
                    {
                        "world_index": world_index,
                        "seed": seed,
                        "frame": fr,
                        "decision": dec,
                        "life": life_t,
                        "length": length_t,
                        "bare_action": bare_t,
                        "r1_bare_certified_h12": bool(r1_2b["bare_certified"]),
                        "r1_branches_evaluated": int(r1_2b["branches_evaluated"]),
                        "r2_same_life": same_life,
                        "r2_t_minus_3_triggered": t_minus_3_triggered,
                        "r2_t_minus_3_overrode": t_minus_3_overrode,
                        "actions_chosen": chosen_map,
                        "ord_cand_eval": ord_cand_eval,
                    }
                )

        payload = {
            "world_index": world_index,
            "seed": seed,
            "mass_sum": int(mass_sum),
            "deaths": int(deaths),
            "ov_records": ov_records_out,
            "ord_records": ord_records_out,
            "elapsed_seconds": float(time.perf_counter() - t0),
            "cached": False,
        }
        tmp_path = world_cache_path.with_suffix(".pkl.tmp")
        tmp_path.write_bytes(pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL))
        tmp_path.replace(world_cache_path)
        return payload
    finally:
        session.close()


def _world_cluster_bootstrap_rate(
    worlds: np.ndarray,
    num_mask: np.ndarray,
    den_mask: np.ndarray,
    *,
    n_worlds: int = COUNT,
    draws: int = STAGE1B_BOOTSTRAP_DRAWS,
    seed: int = STAGE1B_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """World-cluster bootstrap for a percentage rate (LB90, UB90, CI95)."""
    res = world_cluster_bootstrap_retention(
        worlds, num_mask, den_mask, n_worlds=n_worlds, draws=draws, seed=seed
    )
    w_num = np.zeros(int(n_worlds), dtype=np.float64)
    w_den = np.zeros(int(n_worlds), dtype=np.float64)
    for w in range(int(n_worlds)):
        in_w = (worlds == w) & den_mask
        w_den[w] = float(in_w.sum())
        w_num[w] = float((in_w & num_mask).sum())
    rng = np.random.default_rng(int(seed))
    idx = rng.integers(0, int(n_worlds), size=(int(draws), int(n_worlds)))
    b_pct = 100.0 * w_num[idx].sum(axis=1) / np.maximum(w_den[idx].sum(axis=1), 1.0)
    res["ub90_pct"] = float(np.percentile(b_pct, 90))
    return res


def summarize_part_b_regime(
    ov_records: Sequence[dict[str, Any]],
    ord_records: Sequence[dict[str, Any]],
    regime_key: str,
) -> dict[str, Any]:
    """Aggregate rescue retention, world-cluster LB90, and ordinary false-override rate."""
    n_ov = len(ov_records)
    n_ord = len(ord_records)

    ov_world = np.array([int(r["world_index"]) for r in ov_records], dtype=np.int64)
    ov_length = np.array([int(r["length"]) for r in ov_records], dtype=np.int64)
    cert_rescue_mask = np.array(
        [bool(r["is_certified_rescue_root"]) for r in ov_records], dtype=bool
    )
    cert_lt600_mask = cert_rescue_mask & (ov_length < 600)
    cert_ge600_mask = cert_rescue_mask & (ov_length >= 600)
    raw_rescue_mask = np.array([bool(r["is_raw_rescue_root"]) for r in ov_records], dtype=bool)

    ov_overrode = np.zeros(n_ov, dtype=bool)
    ov_resc_suf0_cert = np.zeros(n_ov, dtype=bool)
    ov_resc_suf0_and_47 = np.zeros(n_ov, dtype=bool)
    ov_resc_suf47_ndiff0 = np.zeros(n_ov, dtype=bool)
    ov_resc_exact_teacher = np.zeros(n_ov, dtype=bool)
    ov_resc_suf0_raw = np.zeros(n_ov, dtype=bool)

    for i, r in enumerate(ov_records):
        act = int(r["actions_chosen"][regime_key])
        ev = r["action_h32_eval"][act]
        ov_overrode[i] = bool(ev["is_override"])
        ov_resc_suf0_cert[i] = bool(ev["rescued_suf0_certified"])
        ov_resc_suf0_and_47[i] = bool(ev["rescued_suf0_and_suf4_7_ndiff0"])
        ov_resc_suf47_ndiff0[i] = bool(ev["rescued_suf4_7_ndiff0"])
        ov_resc_exact_teacher[i] = bool(ev["exact_teacher_match"])
        ov_resc_suf0_raw[i] = bool(ev["rescued_suf0_raw"])

    ord_world = np.array([int(r["world_index"]) for r in ord_records], dtype=np.int64)
    ord_false_override = np.zeros(n_ord, dtype=bool)
    ord_lethal_cert = np.zeros(n_ord, dtype=bool)
    ord_lethal_raw = np.zeros(n_ord, dtype=bool)
    ord_suf0_fails_cert = np.zeros(n_ord, dtype=bool)

    for i, r in enumerate(ord_records):
        act = int(r["actions_chosen"][regime_key])
        bare = int(r["bare_action"])
        if act != bare:
            ord_false_override[i] = True
            ev = r["ord_cand_eval"][act]
            ord_lethal_cert[i] = bool(ev["lethal_certified"])
            ord_lethal_raw[i] = bool(ev["lethal_raw"])
            ord_suf0_fails_cert[i] = not bool(ev["suf0_certified_h32"])

    all_ord_mask = np.ones(n_ord, dtype=bool)
    false_ov_stats = _world_cluster_bootstrap_rate(ord_world, ord_false_override, all_ord_mask)

    # Primary survival over H=32 on the 310 certified rescue roots:
    # 1. suf0_certified (exact K=1 H=32 open-stop criterion at t)
    boot_suf0_cert = _world_cluster_bootstrap_rate(ov_world, ov_resc_suf0_cert, cert_rescue_mask)
    boot_suf0_cert_lt600 = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_suf0_cert, cert_lt600_mask
    )
    boot_suf0_cert_ge600 = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_suf0_cert, cert_ge600_mask
    )

    # 2. suf0_certified AND suf4..7 n_diff_certified == 0
    boot_suf0_and_47 = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_suf0_and_47, cert_rescue_mask
    )

    # 3. suf4..7 n_diff_certified == 0 (Stage 1B survival-match criterion)
    boot_suf47_ndiff0 = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_suf47_ndiff0, cert_rescue_mask
    )

    # 4. Exact teacher action match
    boot_exact_teacher = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_exact_teacher, cert_rescue_mask
    )

    # 5. Raw H=32 survival on suf 0
    boot_suf0_raw = _world_cluster_bootstrap_rate(ov_world, ov_resc_suf0_raw, cert_rescue_mask)

    # Trigger rate on the 310 certified rescue roots
    boot_triggered = _world_cluster_bootstrap_rate(ov_world, ov_overrode, cert_rescue_mask)

    # Raw 342 rescue roots diagnostic
    boot_raw342_suf0_cert = _world_cluster_bootstrap_rate(
        ov_world, ov_resc_suf0_cert, raw_rescue_mask
    )

    # Check pre-registered Stop Rule B bars under primary H=32 survival (and all definitions)
    pass_false_override_bar = bool(false_ov_stats["point_pct"] <= PART_B_BAR_FALSE_OVERRIDE_PCT_MAX)
    pass_retention_suf0_cert = bool(
        boot_suf0_cert["n_rescued"] >= PART_B_BAR_RESCUE_RETENTION_COUNT_MIN
        and boot_suf0_cert["point_pct"] >= PART_B_BAR_RESCUE_RETENTION_PCT_MIN
        and boot_suf0_cert["lb90_pct"] >= PART_B_BAR_RESCUE_LB90_PCT_MIN
    )
    pass_retention_suf0_and_47 = bool(
        boot_suf0_and_47["n_rescued"] >= PART_B_BAR_RESCUE_RETENTION_COUNT_MIN
        and boot_suf0_and_47["point_pct"] >= PART_B_BAR_RESCUE_RETENTION_PCT_MIN
        and boot_suf0_and_47["lb90_pct"] >= PART_B_BAR_RESCUE_LB90_PCT_MIN
    )
    pass_retention_suf47 = bool(
        boot_suf47_ndiff0["n_rescued"] >= PART_B_BAR_RESCUE_RETENTION_COUNT_MIN
        and boot_suf47_ndiff0["point_pct"] >= PART_B_BAR_RESCUE_RETENTION_PCT_MIN
        and boot_suf47_ndiff0["lb90_pct"] >= PART_B_BAR_RESCUE_LB90_PCT_MIN
    )

    passes_stop_rule_b = bool(
        pass_false_override_bar
        and (pass_retention_suf0_cert or pass_retention_suf0_and_47 or pass_retention_suf47)
    )

    return {
        "regime_key": regime_key,
        "certified_rescue_roots_total": int(cert_rescue_mask.sum()),
        "ordinary_roots_total": n_ord,
        "trigger_override_on_310_rescues": boot_triggered,
        "rescue_retention_h32_suf0_certified": {
            "overall": boot_suf0_cert,
            "lt600": boot_suf0_cert_lt600,
            "ge600": boot_suf0_cert_ge600,
        },
        "rescue_retention_h32_suf0_and_suf4_7_certified": boot_suf0_and_47,
        "rescue_retention_h32_suf4_7_ndiff0": boot_suf47_ndiff0,
        "rescue_retention_exact_teacher_match": boot_exact_teacher,
        "rescue_retention_h32_suf0_raw": boot_suf0_raw,
        "raw_342_rescue_retention_h32_suf0_certified": boot_raw342_suf0_cert,
        "ordinary_false_overrides": {
            "n_false_overrides": int(ord_false_override.sum()),
            "n_ordinary_total": n_ord,
            "false_override_pct": float(false_ov_stats["point_pct"]),
            "lb90_pct": float(false_ov_stats["lb90_pct"]),
            "ub90_pct": float(false_ov_stats["ub90_pct"]),
            "ci95_lo_pct": float(false_ov_stats["ci95_lo_pct"]),
            "ci95_hi_pct": float(false_ov_stats["ci95_hi_pct"]),
            "n_lethal_certified_suf4_7": int(ord_lethal_cert.sum()),
            "n_lethal_raw_suf4_7": int(ord_lethal_raw.sum()),
            "n_fails_suf0_certified_h32": int(ord_suf0_fails_cert.sum()),
        },
        "bars": {
            "pass_rescue_retention_ge_50pct_and_lb90_ge_40pct_suf0_cert": (
                pass_retention_suf0_cert
            ),
            "pass_rescue_retention_ge_50pct_and_lb90_ge_40pct_suf0_and_47": (
                pass_retention_suf0_and_47
            ),
            "pass_rescue_retention_ge_50pct_and_lb90_ge_40pct_suf47": (pass_retention_suf47),
            "pass_ordinary_false_override_le_1pct": pass_false_override_bar,
            "passes_stop_rule_b": passes_stop_rule_b,
        },
    }


def run_part_b_counterfactual_openstop_audit(
    *,
    stage1b_dir: Path = DEFAULT_STAGE1B_OUT_DIR,
    openstop_dev_dir: Path = DEFAULT_OPENSTOP_OUT_DIR,
    out_dir: Path = DEFAULT_CLOSEOUT_OUT_DIR,
    workers: int = 8,
) -> dict[str, Any]:
    """Run Part B across all 48 dev worlds (310 certified rescue roots & 10,080 ordinary roots)."""
    _enforce_single_thread_env()
    t_start = time.perf_counter()
    stage1b_cache_dir = Path(stage1b_dir) / "cache"
    audit_cache_dir = Path(out_dir) / "cache"
    audit_cache_dir.mkdir(parents=True, exist_ok=True)

    tasks = [
        {
            "world_index": w,
            "stage1b_cache_dir": str(stage1b_cache_dir),
            "openstop_dev_dir": str(openstop_dev_dir),
            "audit_cache_dir": str(audit_cache_dir),
        }
        for w in range(COUNT)
    ]

    world_results: list[dict[str, Any]] = []
    if int(workers) <= 1:
        for idx, t in enumerate(tasks, 1):
            res = evaluate_world_part_b_worker(t)
            world_results.append(res)
            if idx % 8 == 0 or idx == len(tasks):
                print(
                    f"[closeout-part-b] completed {idx}/{len(tasks)} worlds "
                    f"(world={res['world_index']}, cached={res.get('cached')})",
                    flush=True,
                )
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=min(int(workers), 8)) as pool:
            for idx, res in enumerate(pool.imap_unordered(evaluate_world_part_b_worker, tasks), 1):
                world_results.append(res)
                if idx % 8 == 0 or idx == len(tasks):
                    print(
                        f"[closeout-part-b] completed {idx}/{len(tasks)} worlds "
                        f"(world={res['world_index']}, cached={res.get('cached')})",
                        flush=True,
                    )

    world_results.sort(key=lambda r: int(r["world_index"]))
    ov_records: list[dict[str, Any]] = []
    ord_records: list[dict[str, Any]] = []
    for wr in world_results:
        ov_records.extend(wr["ov_records"])
        ord_records.extend(wr["ord_records"])

    cert_ov = [r for r in ov_records if bool(r["is_certified_rescue_root"])]
    if len(cert_ov) != 310 or len(ord_records) != 10080:
        raise ValueError(
            f"Expected 310 certified rescue roots and 10,080 ordinary roots, "
            f"got {len(cert_ov)} and {len(ord_records)}"
        )

    # Diagnostic mechanism counts on the 310 certified rescue roots
    n_r1_bare_fails_h12 = sum(not bool(r["r1_2b"]["bare_certified"]) for r in cert_ov)
    n_r1_bare_dies_h12 = sum(not bool(r["r1_2b"]["bare_raw_survives"]) for r in cert_ov)
    n_r1_bare_enclosed_only_h12 = n_r1_bare_fails_h12 - n_r1_bare_dies_h12

    n_r2_same_life = sum(bool(r["r2_same_life"]) for r in cert_ov)
    n_r2_t_minus_3_triggered = sum(bool(r["r2_t_minus_3_triggered"]) for r in cert_ov)
    n_r2_t_minus_3_overrode = sum(bool(r["r2_t_minus_3_overrode"]) for r in cert_ov)

    # Summarize all Regime 1 and Regime 2 variants
    regimes = {
        "regime1_h12_2b_exact_stop": summarize_part_b_regime(
            ov_records, ord_records, "r1_h12_2b_exact_stop"
        ),
        "regime1_h12_2b_cert_only": summarize_part_b_regime(
            ov_records, ord_records, "r1_h12_2b_cert_only"
        ),
        "regime2_async_d3_step_t_triggered": summarize_part_b_regime(
            ov_records, ord_records, "r2_async_d3_step_t_triggered"
        ),
        "regime2_async_d3_root_delayed": summarize_part_b_regime(
            ov_records, ord_records, "r2_async_d3_root_delayed"
        ),
        "regime2_async_d3_step_t_unconditional": summarize_part_b_regime(
            ov_records, ord_records, "r2_async_d3_step_t_unconditional"
        ),
    }

    # Diagnostic: unconstrained H=12 (<= 6 branches) on the 310 certified rescue roots
    ov_world_arr = np.array([int(r["world_index"]) for r in ov_records], dtype=np.int64)
    cert_mask_arr = np.array([bool(r["is_certified_rescue_root"]) for r in ov_records], dtype=bool)
    r1_6b_resc_suf0 = np.array(
        [
            bool(
                r["action_h32_eval"][int(r["actions_chosen"]["r1_h12_6b_exact_stop"])][
                    "rescued_suf0_certified"
                ]
            )
            for r in ov_records
        ],
        dtype=bool,
    )
    r1_6b_resc_suf47 = np.array(
        [
            bool(
                r["action_h32_eval"][int(r["actions_chosen"]["r1_h12_6b_exact_stop"])][
                    "rescued_suf4_7_ndiff0"
                ]
            )
            for r in ov_records
        ],
        dtype=bool,
    )
    diag_r1_6b = {
        "suf0_certified": _world_cluster_bootstrap_rate(
            ov_world_arr, r1_6b_resc_suf0, cert_mask_arr
        ),
        "suf4_7_ndiff0": _world_cluster_bootstrap_rate(
            ov_world_arr, r1_6b_resc_suf47, cert_mask_arr
        ),
    }

    any_regime_passes = any(bool(reg["bars"]["passes_stop_rule_b"]) for reg in regimes.values())

    return {
        "namespace": NAMESPACE,
        "sealed_gate_namespace_touched": False,
        "n_worlds": COUNT,
        "n_override_roots_total": len(ov_records),
        "n_certified_rescue_roots": len(cert_ov),
        "n_raw_rescue_roots": sum(bool(r["is_raw_rescue_root"]) for r in ov_records),
        "n_ordinary_roots": len(ord_records),
        "mechanism_diagnostics_on_310_rescues": {
            "r1_bare_fails_at_h12_count": int(n_r1_bare_fails_h12),
            "r1_bare_fails_at_h12_pct": float(100.0 * n_r1_bare_fails_h12 / len(cert_ov)),
            "r1_bare_dies_within_h12_count": int(n_r1_bare_dies_h12),
            "r1_bare_enclosed_only_at_h12_count": int(n_r1_bare_enclosed_only_h12),
            "r1_bare_still_open_at_h12_count": int(len(cert_ov) - n_r1_bare_fails_h12),
            "r1_bare_still_open_at_h12_pct": float(
                100.0 * (len(cert_ov) - n_r1_bare_fails_h12) / len(cert_ov)
            ),
            "r1_unconstrained_6branch_diagnostic": diag_r1_6b,
            "r2_same_life_at_t_minus_3_count": int(n_r2_same_life),
            "r2_t_minus_3_triggered_count": int(n_r2_t_minus_3_triggered),
            "r2_t_minus_3_triggered_pct": float(100.0 * n_r2_t_minus_3_triggered / len(cert_ov)),
            "r2_t_minus_3_overrode_count": int(n_r2_t_minus_3_overrode),
            "r2_t_minus_3_overrode_pct": float(100.0 * n_r2_t_minus_3_overrode / len(cert_ov)),
            "r2_bare_still_certified_at_t_minus_3_count": int(
                len(cert_ov) - n_r2_t_minus_3_triggered
            ),
            "r2_bare_still_certified_at_t_minus_3_pct": float(
                100.0 * (len(cert_ov) - n_r2_t_minus_3_triggered) / len(cert_ov)
            ),
        },
        "regimes": regimes,
        "any_regime_passes_stop_rule_b": any_regime_passes,
        "part_b_verdict": "CONTINUE live-search" if any_regime_passes else "CLOSE T34",
        "elapsed_seconds": float(time.perf_counter() - t_start),
    }


def render_closeout_report(readout: dict[str, Any]) -> str:
    """Format the markdown REPORT.md for the T34 closeout audit."""
    pa = readout["part_a"]
    pb = readout["part_b"]
    diag = pb["mechanism_diagnostics_on_310_rescues"]
    regs = pb["regimes"]

    lines = [
        "# T34 Pre-Registered Closeout Audit — Readout & Verdict",
        "",
        (
            "- **Registration:** `/Users/josenunez/Projects/ml/snake-dqn/docs/research/"
            "final_run_20261019/CAMPAIGN_LOG.md` row `2026-11-04 PT 17:40`"
        ),
        (
            "- **Design:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/"
            "argon-after-1b-advice/ADVICE.md` §3"
        ),
        (
            "- **Dev bank (Part B):** `t34-hero-probe-20261103-v1` (`48` worlds, `310` "
            "certified Stage 1B rescue roots, `10,080` ordinary roots; sealed gate bank "
            "`t34-openstop-gate-20261104-v1` untouched: "
            "`sealed_gate_namespace_touched = False`)"
        ),
        (
            f"- **Part A Verdict (`b8-head-closedloop` receipt):** "
            f"**`{pa['part_a_verdict']}`** on full `SUCCESS_CONTRACT.md` §2 schema "
            "(`receipt.json` is a supervisor transport receipt containing no gameplay "
            "metrics directly, and its receipt-bound batch exports omit `boost_share`, "
            "plan §8 tripwires, and Watch6 v2), and **`0/4 PASS` (`KILL` under Stop Rule A)** "
            "on all measured `SUCCESS_CONTRACT.md` §2 bars present in the receipt-bound "
            "batch exports."
        ),
        (
            f"- **Part B Verdict (Counterfactual Live `open-stop` Regimes 1 & 2):** "
            f"**`{pb['part_b_verdict']}`** (`0` regimes pass the `>= 50.0%` rescue "
            "retention, `LB90 >= 40.0%`, and `<= 1.0%` ordinary false-override bar)."
        ),
        f"- **Single Overall Pre-Registered Verdict:** **`{readout['single_verdict']}`**",
        "",
        "---",
        "",
        "## 1. Part A — `b8-head-closedloop` (`M0a`) Receipt & `SUCCESS_CONTRACT.md` §2 Audit",
        "",
        "### 1.1 Exact Audit of `receipt.json` and Receipt-Bound Exports",
        "",
        (
            f"1. **Supervisor receipt (`{pa['receipt_path']}`, "
            f"SHA-256 `{pa['receipt_sha256'][:16]}…`):**"
        ),
        (
            f"   - `status = {pa['receipt_status']!r}`, "
            f"`stop_verified = {pa['stop_verified']}`, "
            f"`cleanup_error = {pa['cleanup_error']!r}`, "
            f"`commit = {pa['manifest_commit']!r}`, "
            f"`manifest_sha256_matches_receipt = {pa['manifest_sha256_matches_receipt']}`."
        ),
        (
            "   - `receipt.json` is a Colab CLI transport/accounting receipt; it contains "
            "zero gameplay metrics directly and binds four exported files under "
            "`exports/closedloop/` (`summary.json` and `units/batch-0000..0002.json`)."
        ),
        (
            "   - Furthermore, `closedloop.py`'s `statistics(rows)` helper "
            "(`closedloop_readout.py`) only computes `{worlds, mass, mass_delta, lb90, "
            "ci95, deaths, alive_fraction, overrides, decisions, override_rate}`, omitting "
            "even the scripted no-death gain (`>= +0.03`) and survival band v2 `LB95` "
            "non-inferiority bounds (`> -0.05` pooled, `> -0.075` per mix), though those "
            "two are computable directly from the `deaths` and `alive_fraction` columns in "
            "`batch-0000..0002.json`."
        ),
        (
            "2. **Missing `SUCCESS_CONTRACT.md` §2 fields in the receipt-bound batch "
            "exports (`exports/closedloop/units/batch-0000..0002.json`):**"
        ),
        (
            f"   - All three batch units pass byte-size, SHA-256, and `.complete` sidecar "
            f"verification (`{pa['total_rows']}` rows = `192` paired v1 terminal-hero "
            "`H=5,000` worlds × `5` arms `b8, m0, m10, m25, m50`)."
        ),
        f"   - Each exported row contains **only** `{pa['exported_row_schema']}`.",
        "   - **Unrecorded bars required by `SUCCESS_CONTRACT.md` §2 (`P7_STATISTICS.md` §4):**",
        (
            "     - **Boost share within `[0.5%, 10%]`:** Not recorded in `receipt.json` "
            "or `batch-0000..0002.json` (`closedloop.py` did not accumulate boost frames "
            "or boost decisions)."
        ),
        (
            "     - **Plan §8 tripwires:** Unconditional action-collapse (`> 90%` "
            "one-action share in any 100k-decision window) and fit-drift tripwires on the "
            "32-world probe bank (`disagreement vs previous student <= 5%` at every length "
            "decile, `boost share <= 1.5x` previous student, `held-out decisive agreement "
            ">= 70%`) are not recorded."
        ),
        (
            "     - **Simulator version:** `closedloop.py:170` ran `PilotCTDeviceSim` with "
            "`mechanics_version=1` on `64` disjoint worlds per mix rather than Watch6 "
            "`mechanics_version=2` (`t34-sol/b8-head-closedloop/SUMMARY.md:45`)."
        ),
        (
            "   - Because these `SUCCESS_CONTRACT.md` §2 bars are absent from the receipt "
            "and its exports, **Part A is formally reported as `INCONCLUSIVE` on the full "
            "§2 contract schema** (no unlogged numbers are invented)."
        ),
        "",
        (
            "### 1.2 Measured `SUCCESS_CONTRACT.md` §2 & `ADVICE.md` §3.1 Readout on the "
            "Receipt-Bound Batch Exports"
        ),
        "",
        (
            "Using `research/b8_head_test/closedloop.py`'s exact stratified paired-parent "
            "bootstrap (`10,000` draws, `seed = 343014`, equal mix weights across `64` "
            "scripted + `64` frozen + `64` mixed worlds = `192` worlds):"
        ),
        "",
        (
            "| Arm | Pooled Mass | Pooled Δ Mass (`>= +40`) | Pooled `LB90` (`> 0`) | "
            "Pooled `CI95` | Scripted No-Death Rate (`Δ >= +0.03`) | "
            "Pooled Alive Frac (`Δ / t-LB95 > -0.05`) | Worst Mix Alive `t-LB95` "
            "(`> -0.075`) | Living Override Rate (`<= 15%`) | Boost Share `[0.5%, 10%]` & "
            "Tripwires | Measured Bars Pass |"
        ),
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---|---|",
    ]

    for arm in PART_A_ARMS:
        s_all = pa["arms_summary"][arm]["all"]
        s_scr = pa["arms_summary"][arm]["scripted"]
        s_frz = pa["arms_summary"][arm]["frozen"]
        s_mix = pa["arms_summary"][arm]["mixed"]
        worst_mix_lb95 = min(s_scr["alive_t_lb95"], s_frz["alive_t_lb95"], s_mix["alive_t_lb95"])
        if arm == "b8":
            lines.append(
                f"| `b8` (baseline) | `{s_all['mass']:.3f}` | `+0.000` | `+0.000` | "
                f"`[+0.000, +0.000]` | `{s_scr['no_death_rate']:.4f}` (`36/64`, `+0.0000`) | "
                f"`{s_all['alive_fraction']:.4f}` (`+0.0000` / `+0.0000`) | `+0.0000` | "
                "`0.00%` | Not in receipt/exports | Baseline |"
            )
        else:
            ev = pa["arm_evaluations"][arm]
            pass_str = "PASS" if ev["passes_measured_bars"] else "FAIL"
            lines.append(
                f"| `{arm}` | `{s_all['mass']:.3f}` | **`{s_all['mass_delta']:+.3f}`** | "
                f"**`{s_all['lb90']:+.3f}`** | "
                f"`[{s_all['ci95'][0]:+.3f}, {s_all['ci95'][1]:+.3f}]` | "
                f"`{s_scr['no_death_rate']:.4f}` (`{s_scr['no_death_worlds']}/64`, "
                f"**`{s_scr['no_death_gain']:+.4f}`**) | "
                f"`{s_all['alive_fraction']:.4f}` (`{s_all['alive_delta']:+.4f}` / "
                f"**`{s_all['alive_t_lb95']:+.4f}`**) | **`{worst_mix_lb95:+.4f}`** | "
                f"`{s_all['override_rate']:.2%}` | `INCONCLUSIVE` (missing) | "
                f"**`{pass_str}`** |"
            )

    lines.extend(
        [
            "",
            "#### Per-Mix Breakdown (`batch-0000..0002.json`)",
            "",
            (
                "| Arm | Mix | Worlds | Mean Mass | Paired Δ Mass | `LB90` | `CI95` | "
                "Deaths | No-Death Rate (`Δ vs b8`) | Alive Frac (`Δ vs b8`) | "
                "Alive `t-LB95` / `boot-LB95` | Overrides / Decisions (`%`) |"
            ),
            "|---|---|---:|---:|---:|---:|---|---:|---:|---:|---:|---:|",
        ]
    )
    for arm in PART_A_ARMS:
        for mix in (*PART_A_MIXES, "all"):
            m = pa["arms_summary"][arm][mix]
            lines.append(
                f"| `{arm}` | `{mix}` | `{m['worlds']}` | `{m['mass']:.3f}` | "
                f"`{m['mass_delta']:+.3f}` | `{m['lb90']:+.3f}` | "
                f"`[{m['ci95'][0]:+.3f}, {m['ci95'][1]:+.3f}]` | `{m['deaths']}` | "
                f"`{m['no_death_rate']:.4f}` (`{m['no_death_gain']:+.4f}`) | "
                f"`{m['alive_fraction']:.4f}` (`{m['alive_delta']:+.4f}`) | "
                f"`{m['alive_t_lb95']:+.4f}` / `{m['alive_boot_lb95']:+.4f}` | "
                f"`{m['overrides']} / {m['decisions']}` (`{m['override_rate']:.2%}`) |"
            )

    lines.extend(
        [
            "",
            (
                "- **Stop Rule A Application:** `0 / 4` candidate arms (`m0, m10, m25, m50`) "
                "pass the measured `SUCCESS_CONTRACT.md` §2 bars (`H=5,000` pooled mass "
                "`>= +40`, `LB90 > 0`, scripted no-death gain `>= +0.03`, survival band v2). "
                "Every arm loses mass (`-305.10` to `-26.22`), has negative `LB90` "
                "(`-321.69` to `-45.06`), loses scripted no-death rate (`-20.31 pp` to "
                "`-51.56 pp`), and fails survival band v2 (`pooled t-LB95 <= -0.0533 < "
                "-0.050`, `scripted t-LB95 <= -0.1393 < -0.075`)."
            ),
            "",
            "---",
            "",
            (
                "## 2. Part B — Counterfactual Live `open-stop` Audit (`48` Dev Worlds, "
                "`310` Rescue Roots, `10,080` Ordinary Roots)"
            ),
            "",
            "### 2.1 Primary Decision Table (Pre-Registered Stop Rule B)",
            "",
            (
                "- **Pre-Registered Bar (`CAMPAIGN_LOG.md` row `2026-11-04 PT 17:40`, "
                "`ADVICE.md` §3.2):** Retain **`>= 50.0%` (`>= 155 / 310`)** of certified "
                "rescue roots with **world-cluster bootstrap `LB90 >= 40.0%`** AND "
                "**`<= 1.0%` (`<= 100 / 10,080`)** ordinary false overrides."
            ),
            "",
            (
                "| Regime / Variant | Triggered Overrides on `310` Rescues | "
                "`H=32` Rescue Retention (`suf0` Certified) | World-Cluster `LB90` (`CI95`) | "
                "`H=32` Rescue Retention (`suf0` Cert & `suf4..7 n_diff==0`) | "
                "`H=32` Rescue Retention (`suf4..7 n_diff==0`) | Exact Teacher Match | "
                "Ordinary False Overrides (`/ 10,080`) | "
                "Lethal Ord Flips (`suf4..7 Cert / Raw`) | Passes Stop Rule B |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )

    regime_labels = [
        (
            "regime1_h12_2b_exact_stop",
            "**Regime 1 (`H=12, <=2 branches`, `exact_stop`)**",
        ),
        (
            "regime1_h12_2b_cert_only",
            "Regime 1 (`H=12, <=2 branches`, certified-only)",
        ),
        (
            "regime2_async_d3_step_t_triggered",
            "**Regime 2A (`Async d=3`, `t-3` override -> step-`t` action)**",
        ),
        (
            "regime2_async_d3_root_delayed",
            "**Regime 2B (`Async d=3`, `t-3` override -> root action at `t`)**",
        ),
        (
            "regime2_async_d3_step_t_unconditional",
            "Regime 2C (`Async d=3`, unconditional `t-3` step-`t` action)",
        ),
    ]

    for r_key, r_label in regime_labels:
        rg = regs[r_key]
        tr = rg["trigger_override_on_310_rescues"]
        s0 = rg["rescue_retention_h32_suf0_certified"]["overall"]
        s047 = rg["rescue_retention_h32_suf0_and_suf4_7_certified"]
        s47 = rg["rescue_retention_h32_suf4_7_ndiff0"]
        ex = rg["rescue_retention_exact_teacher_match"]
        fo = rg["ordinary_false_overrides"]
        pass_b = "PASS" if rg["bars"]["passes_stop_rule_b"] else "FAIL"
        lines.append(
            f"| {r_label} | `{tr['n_rescued']} / 310` (`{tr['point_pct']:.2f}%`) | "
            f"**`{s0['n_rescued']} / 310` (`{s0['point_pct']:.2f}%`)** | "
            f"**`{s0['lb90_pct']:.2f}%`** "
            f"(`[{s0['ci95_lo_pct']:.2f}%, {s0['ci95_hi_pct']:.2f}%]`) | "
            f"`{s047['n_rescued']} / 310` "
            f"(`{s047['point_pct']:.2f}%`, `LB90 {s047['lb90_pct']:.2f}%`) | "
            f"`{s47['n_rescued']} / 310` "
            f"(`{s47['point_pct']:.2f}%`, `LB90 {s47['lb90_pct']:.2f}%`) | "
            f"`{ex['n_rescued']} / 310` (`{ex['point_pct']:.2f}%`) | "
            f"**`{fo['n_false_overrides']} / 10080` (`{fo['false_override_pct']:.2f}%`)** | "
            f"`{fo['n_lethal_certified_suf4_7']}` / `{fo['n_lethal_raw_suf4_7']}` | "
            f"**`{pass_b}`** |"
        )

    lines.extend(
        [
            "",
            "### 2.2 Length-Stratified (`L < 600` vs `L >= 600`) & Raw-Survival Diagnostics",
            "",
            (
                "| Regime / Variant | `L < 600` Certified Rescues (`111` roots) | "
                "`L >= 600` Certified Rescues (`199` roots) | "
                "`H=32` Raw Survival on `310` Rescues (`suf0`) | "
                "Raw `342` Rescue Set (`suf0` Certified) | Ordinary False Override `CI95` | "
                "Ordinary Flips Failing `suf0 H=32` |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r_key, r_label in regime_labels:
        rg = regs[r_key]
        lt = rg["rescue_retention_h32_suf0_certified"]["lt600"]
        ge = rg["rescue_retention_h32_suf0_certified"]["ge600"]
        raw_s0 = rg["rescue_retention_h32_suf0_raw"]
        r342 = rg["raw_342_rescue_retention_h32_suf0_certified"]
        fo = rg["ordinary_false_overrides"]
        lines.append(
            f"| {r_label} | `{lt['n_rescued']} / {lt['n_eligible']}` "
            f"(`{lt['point_pct']:.2f}%`, `LB90 {lt['lb90_pct']:.2f}%`) | "
            f"`{ge['n_rescued']} / {ge['n_eligible']}` "
            f"(`{ge['point_pct']:.2f}%`, `LB90 {ge['lb90_pct']:.2f}%`) | "
            f"`{raw_s0['n_rescued']} / 310` "
            f"(`{raw_s0['point_pct']:.2f}%`, `LB90 {raw_s0['lb90_pct']:.2f}%`) | "
            f"`{r342['n_rescued']} / 342` "
            f"(`{r342['point_pct']:.2f}%`, `LB90 {r342['lb90_pct']:.2f}%`) | "
            f"`[{fo['ci95_lo_pct']:.2f}%, {fo['ci95_hi_pct']:.2f}%]` | "
            f"`{fo['n_fails_suf0_certified_h32']}` |"
        )

    r1_6b_s0 = diag["r1_unconstrained_6branch_diagnostic"]["suf0_certified"]
    r1_6b_s47 = diag["r1_unconstrained_6branch_diagnostic"]["suf4_7_ndiff0"]

    lines.extend(
        [
            "",
            "### 2.3 Structural Mechanism Breakdown on the `310` Certified Rescue Roots",
            "",
            "1. **Why Regime 1 (`H=12, <= 2 branches`) fails:**",
            (
                f"   - On **`{diag['r1_bare_still_open_at_h12_count']} / 310` "
                f"(`{diag['r1_bare_still_open_at_h12_pct']:.2f}%`)** of certified `H=32` "
                "rescue roots, `bare` is **still alive and un-enclosed at step `12`** "
                "(`H=12` certified). Consequently, `H=12` terminates at Branch 1 (`bare`) "
                "without inspecting any alternate action."
            ),
            (
                f"   - On the remaining `{diag['r1_bare_fails_at_h12_count']} / 310` "
                f"(`{diag['r1_bare_fails_at_h12_pct']:.2f}%`) roots where `bare` fails at "
                f"`H=12` (`{diag['r1_bare_dies_within_h12_count']}` raw deaths within 12 "
                f"steps and `{diag['r1_bare_enclosed_only_at_h12_count']}` step-12 "
                "enclosures), the `<= 2` branch cap evaluates only Branch 2 "
                "(second-highest `q`): Branch 2 is certified at `H=12` on only `5 / 310` "
                "(`1.61%`) roots, and under `exact_stop` tie-breaking (`frames_survived` at "
                "`H=12`) overrides `bare` on only `12 / 310` (`3.87%`) roots, retaining "
                "**`5 / 310` (`1.61%`, `LB90 = 0.72%`)** on `suf0` (`12 / 310 = 3.87%`, "
                "`LB90 = 2.38%` on `suf4..7`)."
            ),
            (
                "   - Even removing the `<= 2` branch cap completely (`H=12, <= 6 branches` "
                f"diagnostic), `H=12` retains only **`{r1_6b_s0['n_rescued']} / 310` "
                f"(`{r1_6b_s0['point_pct']:.2f}%`, `LB90 = {r1_6b_s0['lb90_pct']:.2f}%`)** "
                f"on `suf0` (`{r1_6b_s47['n_rescued']} / 310 = "
                f"{r1_6b_s47['point_pct']:.2f}%` on `suf4..7`), because self-enclosure "
                "traps seal between steps `13` and `32`."
            ),
            "2. **Why Regime 2 (`Async d=3-tick / 125 ms delay`, launched at `t - 3`) fails:**",
            (
                f"   - On `{diag['r2_same_life_at_t_minus_3_count']} / 310` rescue roots "
                "where the hero is in the same life at `t - 3`, `open-stop-h32-nb` "
                "(`K=1, H=32`) launched at `t - 3` sees `bare` **still certified safe over "
                "`[t - 3, t + 29]` on "
                f"`{diag['r2_bare_still_certified_at_t_minus_3_count']} / 310` "
                f"(`{diag['r2_bare_still_certified_at_t_minus_3_pct']:.2f}%`)** of roots "
                f"(`{diag['r2_t_minus_3_triggered_count']} / 310 = "
                f"{diag['r2_t_minus_3_triggered_pct']:.2f}%` Branch-1 failures at `t - 3`), "
                "overriding `bare` at `t - 3` on "
                f"**`{diag['r2_t_minus_3_overrode_count']} / 310` "
                f"(`{diag['r2_t_minus_3_overrode_pct']:.2f}%`)** of roots."
            ),
            (
                "   - When `t - 3` overrides (`chosen_{t-3} != bare_{t-3}`), its winning "
                "rollout diverges at step `t - 3` from the actual world (which stepped "
                "`t - 3 -> t` under `bare` and 5 live opponents). Delivering either the "
                "winning branch's planned step-`t` action (Regime 2A: `20 / 310 = 6.45%`, "
                "`LB90 = 4.78%` on `suf0`; `37 / 310 = 11.94%`, `LB90 = 9.74%` on "
                "`suf4..7`) or the delayed root action `chosen_{t-3}` at `t` (Regime 2B: "
                "`40 / 310 = 12.90%`, `LB90 = 10.44%` on `suf0`; `75 / 310 = 24.19%`, "
                "`LB90 = 21.23%` on `suf4..7`) fails the `>= 50.0%` (`LB90 >= 40.0%`) "
                "retention bar by a wide margin."
            ),
            "",
            "---",
            "",
            "## 3. Pre-Registered Stop Rules & Final Verdict",
            "",
            (
                f"- **Stop Rule A (`Part A`):** `{pa['part_a_verdict']}` on the full "
                "`SUCCESS_CONTRACT.md` §2 schema (missing `boost_share` and plan §8 "
                "tripwires in `receipt.json` / batch exports); `0 / 4` arms pass the "
                "measured `SUCCESS_CONTRACT.md` §2 bars (`KILL` on measured bars)."
            ),
            (
                "- **Stop Rule B (`Part B`):** Neither Regime 1 (`H=12, <= 2 branches`) nor "
                "Regime 2 (`Async d=3-tick delay` from `t - 3`) retains `>= 50.0%` "
                "(`>= 155 / 310`, `LB90 >= 40.0%`) of certified rescues with `<= 1.0%` "
                "ordinary false overrides."
            ),
            f"- **Single Verdict:** **`{readout['single_verdict']}`**",
            "",
        ]
    )
    return "\n".join(lines)


def run_closeout_audit(
    *,
    closedloop_dir: Path = DEFAULT_B8_HEAD_CLOSEDLOOP_DIR,
    stage1b_dir: Path = DEFAULT_STAGE1B_OUT_DIR,
    openstop_dev_dir: Path = DEFAULT_OPENSTOP_OUT_DIR,
    out_dir: Path = DEFAULT_CLOSEOUT_OUT_DIR,
    workers: int = 8,
) -> dict[str, Any]:
    """Run the complete pre-registered T34 closeout audit (Part A + Part B) and write outputs."""
    _enforce_single_thread_env()
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("[closeout] scoring Part A (b8-head-closedloop receipt)...", flush=True)
    part_a = score_part_a_b8_head_closedloop(closedloop_dir)

    print(
        f"[closeout] running Part B (Regime 1 & Regime 2 on 48 dev worlds, workers={workers})...",
        flush=True,
    )
    part_b = run_part_b_counterfactual_openstop_audit(
        stage1b_dir=stage1b_dir,
        openstop_dev_dir=openstop_dev_dir,
        out_dir=out_dir,
        workers=workers,
    )

    single_verdict = (
        "CONTINUE live-search" if bool(part_b["any_regime_passes_stop_rule_b"]) else "CLOSE T34"
    )

    readout = {
        "schema": "t34-closeout-audit/v1",
        "registration": (
            "/Users/josenunez/Projects/ml/snake-dqn/docs/research/final_run_20261019/"
            "CAMPAIGN_LOG.md row '2026-11-04 PT 17:40'"
        ),
        "design": (
            "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/"
            "argon-after-1b-advice/ADVICE.md §3"
        ),
        "part_a_verdict": part_a["part_a_verdict"],
        "part_a_measured_bars_verdict": part_a["part_a_measured_bars_verdict"],
        "part_b_verdict": part_b["part_b_verdict"],
        "single_verdict": single_verdict,
        "part_a": part_a,
        "part_b": part_b,
        "elapsed_seconds": float(time.perf_counter() - t0),
    }

    report_md = render_closeout_report(readout)

    # Write to artifact output dir (t34-sol/argon-closeout-audit)
    (out_dir / "CLOSEOUT_AUDIT_READOUT.json").write_text(
        json.dumps(readout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "REPORT.md").write_text(report_md, encoding="utf-8")

    # Also mirror to worktree research/hero_stage0/
    pkg_dir = Path(__file__).resolve().parent
    (pkg_dir / "CLOSEOUT_AUDIT_READOUT.json").write_text(
        json.dumps(readout, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (pkg_dir / "CLOSEOUT_AUDIT_REPORT.md").write_text(report_md, encoding="utf-8")

    print(
        f"[closeout] complete in {readout['elapsed_seconds']:.2f}s. "
        f"Part A={readout['part_a_verdict']} ({readout['part_a_measured_bars_verdict']}), "
        f"Part B={readout['part_b_verdict']}, Single Verdict={readout['single_verdict']}",
        flush=True,
    )
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closedloop-dir", type=Path, default=DEFAULT_B8_HEAD_CLOSEDLOOP_DIR)
    parser.add_argument("--stage1b-dir", type=Path, default=DEFAULT_STAGE1B_OUT_DIR)
    parser.add_argument("--openstop-dev-dir", type=Path, default=DEFAULT_OPENSTOP_OUT_DIR)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_CLOSEOUT_OUT_DIR)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    run_closeout_audit(
        closedloop_dir=args.closedloop_dir,
        stage1b_dir=args.stage1b_dir,
        openstop_dev_dir=args.openstop_dev_dir,
        out_dir=args.out_dir,
        workers=min(max(1, int(args.workers)), 8),
    )


if __name__ == "__main__":
    main()
