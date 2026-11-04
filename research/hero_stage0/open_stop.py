"""In-sample dev-filter evaluation of open-stop-h32-nb (K=1 and K=2) on the 48 probe worlds."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import patch

import numpy as np

from research.bl_k1.native import RowStats, branch_scope, configure
from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.hero_stage0.protocol import (
    CAUSE_CLASSES,
    COUNT,
    HORIZON,
    LIVE_CAUSE_TO_CAUSE_CLASS,
    NAMESPACE,
    OPENSTOP_BRANCH_HORIZON,
    OPENSTOP_CONTROLLER,
    OPENSTOP_DEV_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN,
    OPENSTOP_DEV_BAR_OVERRIDE_RATE_PCT_MAX,
    OPENSTOP_DEV_BAR_SELF_ENCLOSURE_REMOVED_PCT_MIN,
    OPENSTOP_K1_ARM,
    OPENSTOP_K1_SELECTION_SUFFIXES,
    OPENSTOP_K2_ARM,
    OPENSTOP_K2_SELECTION_SUFFIXES,
    OPENSTOP_POCKET_MIN_LENGTH,
    OPENSTOP_REGION_FRAC_MIN,
    OPENSTOP_SCORING_SUFFIXES,
    OPENSTOP_V8_NEED_CAP,
    OPENSTOP_V8_NEED_FLOOR,
    SIM_CODE_TO_CAUSE_CLASS,
    STAGE1A_ARM_SPECS,
    STAGE1A_BRANCH_HORIZON,
    STAGE1A_CONTROLLER,
    allocation,
)
from research.hero_stage0.stage1a import (
    _enforce_single_thread_env,
    _fieller_ratio_ci95,
    _paired_ci,
    _summarize_arm,
    evaluation_order_nb,
    reseed_branch_suffix,
    run_stage1a_arm,
)
from research.t3_rollout_labels_20261015.bank_engine import (
    choose_and_step,
    prepared_q,
    step_prepared,
)
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.t15_mixed_roster_20261017.episodes import _assert_mixed, roster_identity
from research.watch_replay.engine import step_guard
from research.watch_replay.product import open_watch
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.ego_raster_b import region_sizes
from src.simd_env.eval_engine import _dispatch_actions
from src.simd_env.v8_nb import v8_decide

DEFAULT_STAGE1A_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a"
)
DEFAULT_OPENSTOP_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-openstop-dev"
)
V8_LAMBDA = 8.0


def evaluate_terminal_openness(sim: Any) -> dict[str, Any]:
    """Evaluate the terminal-step un-enclosed certificate on ``sim`` (hero slot 0).

    Certificate definition:
        un_enclosed = (v8 best_safe_area >= min(max(4, L), 160))
                      AND (for L >= 160: max over valid actions of region_frac >= 0.25,
                           i.e. pocket R >= L).
    """
    if not bool(sim.alive[0, 0]):
        return {
            "length": 0,
            "advisory_any": False,
            "v8_need": 0,
            "v8_best_safe_area": 0,
            "v8_open": False,
            "region_sizes_dirs": [0, 0, 0],
            "region_frac_dirs": [0.0, 0.0, 0.0],
            "max_pocket_r": 0,
            "max_region_frac": 0.0,
            "pocket_r_ge_l": False,
            "region_frac_ge_025": False,
            "pocket_open": False,
            "un_enclosed": False,
        }

    length = int(sim.length[0, 0])
    v8_need = min(max(OPENSTOP_V8_NEED_FLOOR, length), OPENSTOP_V8_NEED_CAP)
    advisory = sim.get_advisory_action_mask()[0, 0].astype(bool)
    if not advisory.any():
        return {
            "length": length,
            "advisory_any": False,
            "v8_need": v8_need,
            "v8_best_safe_area": 0,
            "v8_open": False,
            "region_sizes_dirs": [0, 0, 0],
            "region_frac_dirs": [0.0, 0.0, 0.0],
            "max_pocket_r": 0,
            "max_region_frac": 0.0,
            "pocket_r_ge_l": False,
            "region_frac_ge_025": False,
            "pocket_open": False,
            "un_enclosed": False,
        }

    hero_row = np.array([[0, 0]], dtype=np.int64)
    v8_res = v8_decide(
        sim,
        hero_row,
        np.zeros((1, 6), dtype=np.float32),
        advisory.reshape(1, 6),
        V8_LAMBDA,
        features=True,
    )
    best_safe_area = int(np.where(advisory, v8_res.area[0], 0).max())
    v8_open = bool(best_safe_area >= v8_need)

    # Exact uncapped tail-aware component sizes R and region_frac = min(R / L, 4.0) / 4.0
    # from src/simd_env/ego_raster_b.py (line 246).
    reg = region_sizes(sim.ego_view(), hero_row, backend="numba")[0]
    denom = float(max(length, 1))
    reg_frac_dirs = (np.minimum(reg.astype(np.float64) / denom, 4.0) / 4.0).astype(np.float32)

    valid_r = [int(reg[a % 3]) for a in range(6) if bool(advisory[a])]
    valid_frac = [float(reg_frac_dirs[a % 3]) for a in range(6) if bool(advisory[a])]
    max_pocket_r = int(max(valid_r)) if valid_r else 0
    max_region_frac = float(max(valid_frac)) if valid_frac else 0.0

    pocket_r_ge_l = bool(max_pocket_r >= length)
    region_frac_ge_025 = bool(max_region_frac >= OPENSTOP_REGION_FRAC_MIN)
    if pocket_r_ge_l != region_frac_ge_025:
        raise RuntimeError(
            f"R >= L ({max_pocket_r} >= {length}) disagreed with "
            f"region_frac >= 0.25 ({max_region_frac})"
        )

    pocket_open = (
        bool(pocket_r_ge_l and region_frac_ge_025) if length >= OPENSTOP_POCKET_MIN_LENGTH else True
    )
    un_enclosed = bool(v8_open and pocket_open)

    return {
        "length": length,
        "advisory_any": True,
        "v8_need": int(v8_need),
        "v8_best_safe_area": int(best_safe_area),
        "v8_open": v8_open,
        "region_sizes_dirs": [int(x) for x in reg.tolist()],
        "region_frac_dirs": [float(x) for x in reg_frac_dirs.tolist()],
        "max_pocket_r": int(max_pocket_r),
        "max_region_frac": float(max_region_frac),
        "pocket_r_ge_l": pocket_r_ge_l,
        "region_frac_ge_025": region_frac_ge_025,
        "pocket_open": pocket_open,
        "un_enclosed": un_enclosed,
    }


def _eval_open_stop_branch_suffix(
    root: Any,
    action: int,
    decision: int,
    suffix_id: int,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Roll out ``action`` for ``branch_horizon`` steps; check alive 1..H + un-enclosed at H."""
    saved_rng_states = [env_rng._rng.getstate() for env_rng in root.sim._rngs]
    try:
        reseed_branch_suffix(root, decision, suffix_id, namespace=namespace)
        with branch_scope(root.mix):
            branch = _clone_root(root)
            branch.sim._policy_phase_prepared = True
            step_prepared(branch, int(action))
            executed = 1
            alive_every_step = bool(branch.sim.alive[0, 0])
            while executed < int(branch_horizon) and alive_every_step:
                choose_and_step(branch, capture_obs=False)
                executed += 1
                if not bool(branch.sim.alive[0, 0]):
                    alive_every_step = False

            raw_survives = bool(alive_every_step and executed == int(branch_horizon))
            open_info = (
                evaluate_terminal_openness(branch.sim)
                if raw_survives
                else {
                    "length": 0,
                    "advisory_any": False,
                    "v8_need": 0,
                    "v8_best_safe_area": 0,
                    "v8_open": False,
                    "region_sizes_dirs": [0, 0, 0],
                    "region_frac_dirs": [0.0, 0.0, 0.0],
                    "max_pocket_r": 0,
                    "max_region_frac": 0.0,
                    "pocket_r_ge_l": False,
                    "region_frac_ge_025": False,
                    "pocket_open": False,
                    "un_enclosed": False,
                }
            )
            final_length = int(branch.sim.length[0, 0]) if raw_survives else 0
    finally:
        for env_rng, st in zip(root.sim._rngs, saved_rng_states):
            env_rng._rng.setstate(st)

    frames_executed = int(executed)
    frames_survived = int(branch_horizon) if raw_survives else max(0, frames_executed - 1)
    un_enclosed = bool(raw_survives and open_info["un_enclosed"])
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
        "terminal_openness": open_info,
    }


def _eval_scoring_branch_suffix(
    root: Any,
    action: int,
    decision: int,
    suffix_id: int,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    extended_horizon: int = STAGE1A_BRANCH_HORIZON,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Evaluate a scoring suffix at both H=32 (raw & certified) and H=48 (Stage 1A horizon)."""
    saved_rng_states = [env_rng._rng.getstate() for env_rng in root.sim._rngs]
    try:
        reseed_branch_suffix(root, decision, suffix_id, namespace=namespace)
        with branch_scope(root.mix):
            branch = _clone_root(root)
            branch.sim._policy_phase_prepared = True
            step_prepared(branch, int(action))
            executed = 1
            alive_every_step = bool(branch.sim.alive[0, 0])
            while executed < int(branch_horizon) and alive_every_step:
                choose_and_step(branch, capture_obs=False)
                executed += 1
                if not bool(branch.sim.alive[0, 0]):
                    alive_every_step = False

            raw_survives_h32 = bool(alive_every_step and executed == int(branch_horizon))
            frames_executed_h32 = int(executed)
            frames_survived_h32 = (
                int(branch_horizon) if raw_survives_h32 else max(0, frames_executed_h32 - 1)
            )
            open_h32 = (
                evaluate_terminal_openness(branch.sim)
                if raw_survives_h32
                else {"un_enclosed": False, "v8_best_safe_area": 0, "max_pocket_r": 0}
            )
            certified_h32 = bool(raw_survives_h32 and open_h32["un_enclosed"])
            final_length_h32 = int(branch.sim.length[0, 0]) if raw_survives_h32 else 0

            # Continue to extended_horizon (H=48) if still alive at H=32
            while executed < int(extended_horizon) and alive_every_step:
                choose_and_step(branch, capture_obs=False)
                executed += 1
                if not bool(branch.sim.alive[0, 0]):
                    alive_every_step = False

            raw_survives_h48 = bool(alive_every_step and executed == int(extended_horizon))
            frames_executed_h48 = int(executed)
            frames_survived_h48 = (
                int(extended_horizon) if raw_survives_h48 else max(0, frames_executed_h48 - 1)
            )
            open_h48 = (
                evaluate_terminal_openness(branch.sim)
                if raw_survives_h48
                else {"un_enclosed": False, "v8_best_safe_area": 0, "max_pocket_r": 0}
            )
            certified_h48 = bool(raw_survives_h48 and open_h48["un_enclosed"])
            final_length_h48 = int(branch.sim.length[0, 0]) if raw_survives_h48 else 0
    finally:
        for env_rng, st in zip(root.sim._rngs, saved_rng_states):
            env_rng._rng.setstate(st)

    return {
        "suffix_id": int(suffix_id),
        "survives": raw_survives_h32,
        "raw_survives_h32": raw_survives_h32,
        "certified_h32": certified_h32,
        "frames_executed": frames_executed_h32,
        "frames_survived": frames_survived_h32,
        "final_length": final_length_h32,
        "v8_best_safe_area_h32": int(open_h32.get("v8_best_safe_area", 0)),
        "max_pocket_r_h32": int(open_h32.get("max_pocket_r", 0)),
        "raw_survives_h48": raw_survives_h48,
        "certified_h48": certified_h48,
        "frames_executed_h48": frames_executed_h48,
        "frames_survived_h48": frames_survived_h48,
        "final_length_h48": final_length_h48,
    }


def _compute_harmful_flags(
    bare_vec: Sequence[bool],
    chosen_vec: Sequence[bool],
) -> dict[str, Any]:
    n_score = len(bare_vec)
    diff_vec = [(not c) and b for b, c in zip(bare_vec, chosen_vec)]
    n_bare = int(sum(bare_vec))
    n_chosen = int(sum(chosen_vec))
    n_diff = int(sum(diff_vec))
    return {
        "bare_vec": [bool(x) for x in bare_vec],
        "chosen_vec": [bool(x) for x in chosen_vec],
        "diff_vec": [bool(x) for x in diff_vec],
        "n_bare": n_bare,
        "n_chosen": n_chosen,
        "n_diff": n_diff,
        "s_score_bare": float(n_bare / n_score) if n_score > 0 else 0.0,
        "s_score_chosen": float(n_chosen / n_score) if n_score > 0 else 0.0,
        "harmful_conditional_majority": bool(n_bare > 0 and n_diff > (n_bare / 2.0)),
        "harmful_majority_of_4": bool(n_diff > (n_score / 2.0)),
        "harmful_ge2_of_4": bool(n_diff >= 2),
        "harmful_ge1_of_4": bool(n_diff >= 1),
    }


def score_open_stop_override_independent_suffixes(
    root: Any,
    decision: int,
    bare: int,
    chosen: int,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    scoring_suffixes: Sequence[int] = OPENSTOP_SCORING_SUFFIXES,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Independently score an override (chosen != bare) across the 4 scoring suffixes (4..7)."""
    if int(chosen) == int(bare):
        raise ValueError("score_open_stop_override_independent_suffixes requires chosen != bare")
    if set(scoring_suffixes) & set(OPENSTOP_K2_SELECTION_SUFFIXES):
        raise ValueError("scoring_suffixes must be disjoint from selection_suffixes")

    t0 = time.perf_counter()
    bare_by_suffix: dict[str, dict[str, Any]] = {}
    chosen_by_suffix: dict[str, dict[str, Any]] = {}
    scoring_env_frames = 0

    for suf in scoring_suffixes:
        b_out = _eval_scoring_branch_suffix(
            root,
            bare,
            decision,
            int(suf),
            branch_horizon=branch_horizon,
            extended_horizon=STAGE1A_BRANCH_HORIZON,
            namespace=namespace,
        )
        c_out = _eval_scoring_branch_suffix(
            root,
            chosen,
            decision,
            int(suf),
            branch_horizon=branch_horizon,
            extended_horizon=STAGE1A_BRANCH_HORIZON,
            namespace=namespace,
        )
        bare_by_suffix[str(suf)] = b_out
        chosen_by_suffix[str(suf)] = c_out
        scoring_env_frames += int(b_out["frames_executed"]) + int(c_out["frames_executed"])

    n_score = len(scoring_suffixes)
    h32_raw = _compute_harmful_flags(
        [bool(bare_by_suffix[str(s)]["raw_survives_h32"]) for s in scoring_suffixes],
        [bool(chosen_by_suffix[str(s)]["raw_survives_h32"]) for s in scoring_suffixes],
    )
    h32_cert = _compute_harmful_flags(
        [bool(bare_by_suffix[str(s)]["certified_h32"]) for s in scoring_suffixes],
        [bool(chosen_by_suffix[str(s)]["certified_h32"]) for s in scoring_suffixes],
    )
    h48_raw = _compute_harmful_flags(
        [bool(bare_by_suffix[str(s)]["raw_survives_h48"]) for s in scoring_suffixes],
        [bool(chosen_by_suffix[str(s)]["raw_survives_h48"]) for s in scoring_suffixes],
    )

    return {
        "scoring_suffixes": [int(s) for s in scoring_suffixes],
        "bare_action": int(bare),
        "chosen_action": int(chosen),
        "bare_survives_vec": h32_raw["bare_vec"],
        "chosen_survives_vec": h32_raw["chosen_vec"],
        "override_dies_where_b8_survives_vec": h32_raw["diff_vec"],
        "n_scoring_suffixes": n_score,
        "n_bare_survives": h32_raw["n_bare"],
        "n_chosen_survives": h32_raw["n_chosen"],
        "n_chosen_dies": int(n_score - h32_raw["n_chosen"]),
        "n_override_dies_where_b8_survives": h32_raw["n_diff"],
        "s_score_bare": h32_raw["s_score_bare"],
        "s_score_chosen": h32_raw["s_score_chosen"],
        "harmful": h32_raw["harmful_conditional_majority"],
        "harmful_conditional_majority": h32_raw["harmful_conditional_majority"],
        "harmful_majority_of_4": h32_raw["harmful_majority_of_4"],
        "harmful_ge2_of_4": h32_raw["harmful_ge2_of_4"],
        "harmful_ge1_of_4": h32_raw["harmful_ge1_of_4"],
        "h32_certified": h32_cert,
        "h48_raw": h48_raw,
        "bare_outcomes": bare_by_suffix,
        "chosen_outcomes": chosen_by_suffix,
        "scoring_rollouts_evaluated": 2 * n_score,
        "scoring_env_frames": scoring_env_frames,
        "scoring_wall_seconds": time.perf_counter() - t0,
    }


def dense_batch_open_stop_h32_nb(
    world: Any,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = OPENSTOP_BRANCH_HORIZON,
    selection_suffixes: Sequence[int] = OPENSTOP_K1_SELECTION_SUFFIXES,
    scoring_suffixes: Sequence[int] = OPENSTOP_SCORING_SUFFIXES,
    namespace: str = NAMESPACE,
    score_overrides: bool = True,
) -> dict[str, Any]:
    """Run open-stop-h32-nb on K in {1, 2} selection suffixes and score overrides on 4 suffixes."""
    if len(selection_suffixes) not in (1, 2):
        raise ValueError("open-stop-h32-nb requires K=1 or K=2 deterministic selection suffixes")

    root = _clone_root(world)
    before = _root_bytes(root)
    bare = int(np.where(legal, q.astype(np.float64), -np.inf).argmax())
    legal_actions = np.flatnonzero(legal).tolist()

    stats = RowStats()
    selection_rollouts = 0
    selection_env_frames = 0

    def evaluate_action_selection(action: int) -> dict[str, Any]:
        nonlocal selection_rollouts, selection_env_frames
        suffix_outcomes: dict[str, dict[str, Any]] = {}
        certified_all = True
        raw_survives_all = True
        min_frames_executed: int | None = None
        min_frames_survived: int | None = None
        action_env_frames = 0

        for suf in selection_suffixes:
            out_s = _eval_open_stop_branch_suffix(
                root,
                int(action),
                int(decision),
                int(suf),
                branch_horizon=branch_horizon,
                namespace=namespace,
            )
            suffix_outcomes[str(suf)] = out_s
            selection_rollouts += 1
            f_exec = int(out_s["frames_executed"])
            f_surv = int(out_s["frames_survived"])
            selection_env_frames += f_exec
            action_env_frames += f_exec
            certified_all = bool(certified_all and bool(out_s["certified"]))
            raw_survives_all = bool(raw_survives_all and bool(out_s["raw_survives"]))
            if min_frames_executed is None or f_exec < min_frames_executed:
                min_frames_executed = f_exec
            if min_frames_survived is None or f_surv < min_frames_survived:
                min_frames_survived = f_surv
            if not certified_all:
                break

        return {
            "survives": certified_all,
            "certified": certified_all,
            "raw_survives": raw_survives_all,
            "frames_executed": int(min_frames_executed or 0),
            "frames_survived": int(min_frames_survived or 0),
            "selection_suffixes": suffix_outcomes,
            "rollouts_evaluated": len(suffix_outcomes),
            "env_frames_executed": action_env_frames,
        }

    def choose_all_fail(
        outcomes: dict[int, dict[str, Any]], bare_act: int, q_vec: np.ndarray
    ) -> int:
        nonlocal selection_rollouts, selection_env_frames
        for act, out in outcomes.items():
            suf_map = out["selection_suffixes"]
            for suf in selection_suffixes:
                s_key = str(suf)
                if s_key not in suf_map:
                    out_s = _eval_open_stop_branch_suffix(
                        root,
                        int(act),
                        int(decision),
                        int(suf),
                        branch_horizon=branch_horizon,
                        namespace=namespace,
                    )
                    suf_map[s_key] = out_s
                    selection_rollouts += 1
                    f_exec = int(out_s["frames_executed"])
                    f_surv = int(out_s["frames_survived"])
                    selection_env_frames += f_exec
                    out["env_frames_executed"] = int(out["env_frames_executed"]) + f_exec
                    out["rollouts_evaluated"] = len(suf_map)
                    out["raw_survives"] = bool(out["raw_survives"] and bool(out_s["raw_survives"]))
                    out["frames_executed"] = min(int(out["frames_executed"]), f_exec)
                    out["frames_survived"] = min(int(out["frames_survived"]), f_surv)
        return ceiling._choose(outcomes, bare_act, q_vec)

    t_sel0 = time.perf_counter()
    with (
        configure("kernel", stats),
        patch.object(exact_stop, "evaluation_order", evaluation_order_nb),
    ):
        result = exact_stop.run(
            legal_actions,
            bare,
            q,
            evaluate_action_selection,
            choose_all_fail,
        )
        selection_wall_seconds = time.perf_counter() - t_sel0

        chosen = int(result["chosen"])
        override_scoring: dict[str, Any] | None = None
        if score_overrides and chosen != bare:
            override_scoring = score_open_stop_override_independent_suffixes(
                root,
                decision,
                bare,
                chosen,
                branch_horizon=branch_horizon,
                scoring_suffixes=scoring_suffixes,
                namespace=namespace,
            )

    if _root_bytes(root) != before:
        raise ValueError("branch mutated its parent RNG/carry")

    counters = stats.as_dict()
    if (
        counters["kernel_rows"] <= 0
        or counters["stock_rows"] != 0
        or counters["fallback_rows"] != 0
        or counters["noncandidate_rows"] != 0
    ):
        raise ValueError("missing native kernel per-decision coverage")

    return {
        **result,
        "selection_rollouts_evaluated": selection_rollouts,
        "selection_env_frames": selection_env_frames,
        "selection_wall_seconds": selection_wall_seconds,
        "override_scoring": override_scoring,
        "native_v8": counters,
        "root_sha256": ceiling.digest_value(before.hex()),
    }


def run_open_stop_arm(
    seed: int,
    arm: str,
    check: Callable[[], None] = lambda: None,
    *,
    world_index: int = 0,
    horizon: int = HORIZON,
    audit_live_mirror: bool = False,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Run one Watch6-v2 session for arm in {'open-stop-h32-nb-k1', 'open-stop-h32-nb-k2'}."""
    if arm not in (OPENSTOP_K1_ARM, OPENSTOP_K2_ARM):
        raise ValueError(f"unknown open-stop arm: {arm!r}")
    if not 1 <= horizon <= 14400:
        raise ValueError("horizon must be 1..14400")

    spec = STAGE1A_ARM_SPECS[arm]
    min_length: int = int(spec["min_length"])
    interval: int = int(spec["interval"])
    branch_horizon: int = int(spec["branch_horizon"])
    selection_suffixes: tuple[int, ...] = tuple(spec["selection_suffixes"])
    scoring_suffixes: tuple[int, ...] = tuple(spec["scoring_suffixes"])

    check()
    started = time.monotonic()
    session = open_watch({"seed": int(seed)}, 24, "bare")
    world = session.world
    if not world.sim.allow_respawn or any(
        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
    ):
        session.close()
        raise ValueError("open-stop requires continuous respawn in real Watch and mirror")
    if getattr(session.live, "pathb", None) is not None:
        session.close()
        raise ValueError("open-stop refuses Path B runtime assistance")

    _assert_mixed(session.live, 6)
    roster = roster_identity(session.live)

    life = 1
    was_alive = True
    trigger = DenseTrigger(min_length=min_length, interval=interval)
    prev_grid_alive = [bool(x) for x in world.sim.alive[0].tolist()]
    grid_respawns = [0] * int(world.sim.S)

    alive_frames = 0
    alive_frames_lt600 = 0
    alive_frames_ge600 = 0
    mass_sum = 0
    deaths = 0
    decisions_count = 0
    decisions_lt600 = 0
    decisions_ge600 = 0
    boost_decisions = 0
    boost_decisions_lt600 = 0
    boost_decisions_ge600 = 0
    max_length = 0
    alive_lengths: list[int] = []
    death_lengths: list[int] = []
    death_records: list[dict[str, Any]] = []

    search_roots = 0
    search_roots_lt600 = 0
    search_roots_ge600 = 0
    bare_survived_both_roots = 0
    bare_raw_survived_roots = 0
    bare_failed_by_enclosure_roots = 0
    bare_failed_by_death_roots = 0
    all_die_roots = 0
    interventions = 0
    interventions_lt600 = 0
    interventions_ge600 = 0
    interventions_to_nonboost = 0
    interventions_to_boost = 0
    interventions_from_enclosure_only = 0
    interventions_from_raw_death = 0
    branches_evaluated = 0
    branches_skipped = 0
    selection_rollouts_evaluated = 0
    selection_env_frames = 0
    selection_wall_seconds = 0.0
    scoring_rollouts_evaluated = 0
    scoring_env_frames = 0
    scoring_wall_seconds = 0.0
    native_v8_kernel_rows = 0
    override_records: list[dict[str, Any]] = []

    try:
        with step_guard(check):
            for frame in range(1, horizon + 1):
                check()
                selected: dict[str, Any] = {}

                def choose(sim: Any) -> np.ndarray:
                    nonlocal life, trigger
                    nonlocal search_roots, search_roots_lt600, search_roots_ge600
                    nonlocal bare_survived_both_roots, bare_raw_survived_roots
                    nonlocal bare_failed_by_enclosure_roots, bare_failed_by_death_roots
                    nonlocal all_die_roots
                    nonlocal interventions, interventions_lt600, interventions_ge600
                    nonlocal interventions_to_nonboost, interventions_to_boost
                    nonlocal interventions_from_enclosure_only, interventions_from_raw_death
                    nonlocal branches_evaluated, branches_skipped
                    nonlocal selection_rollouts_evaluated, selection_env_frames
                    nonlocal selection_wall_seconds
                    nonlocal scoring_rollouts_evaluated, scoring_env_frames, scoring_wall_seconds
                    nonlocal native_v8_kernel_rows

                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        trigger = DenseTrigger(min_length=min_length, interval=interval)
                    actions = np.ones((1, sim.S), dtype=np.int64)
                    if active:
                        obs = build_ego_raster(
                            sim,
                            EgoRasterConfig(version="bv8"),
                            backend="numba",
                            rows=np.array([[0, 0]]),
                        )
                        q = np.asarray(prepared_q(world, obs), dtype=np.float32)
                        advisory = sim.get_advisory_action_mask()[0, 0]
                        legal, bare = ceiling._legal(q, advisory)
                        chosen = bare
                        cur_length = int(sim.length[0, 0])
                        fired = trigger.fires(decisions_count, cur_length, True)
                        if fired and advisory.any():
                            batch = dense_batch_open_stop_h32_nb(
                                world,
                                frame - 1,
                                q,
                                legal,
                                branch_horizon=branch_horizon,
                                selection_suffixes=selection_suffixes,
                                scoring_suffixes=scoring_suffixes,
                                namespace=namespace,
                                score_overrides=True,
                            )
                            chosen = int(batch["chosen"])
                            overrode = chosen != bare
                            search_roots += 1
                            if cur_length < 600:
                                search_roots_lt600 += 1
                                interventions_lt600 += int(overrode)
                            else:
                                search_roots_ge600 += 1
                                interventions_ge600 += int(overrode)
                            interventions += int(overrode)
                            if overrode:
                                if chosen < 3:
                                    interventions_to_nonboost += 1
                                else:
                                    interventions_to_boost += 1

                            outcomes_map = batch["outcomes"]
                            bare_out = outcomes_map.get(bare, {})
                            bare_cert = bool(bare_out.get("survives", False))
                            bare_raw = bool(bare_out.get("raw_survives", False))
                            if bare_cert:
                                bare_survived_both_roots += 1
                                bare_raw_survived_roots += 1
                            elif bare_raw:
                                bare_raw_survived_roots += 1
                                bare_failed_by_enclosure_roots += 1
                                if overrode:
                                    interventions_from_enclosure_only += 1
                            else:
                                bare_failed_by_death_roots += 1
                                if overrode:
                                    interventions_from_raw_death += 1

                            if not any(
                                bool(v.get("survives", False))
                                for v in outcomes_map.values()
                                if isinstance(v, dict)
                            ):
                                all_die_roots += 1

                            branches_evaluated += int(batch["branches_evaluated"])
                            branches_skipped += int(batch["branches_skipped"])
                            selection_rollouts_evaluated += int(
                                batch["selection_rollouts_evaluated"]
                            )
                            selection_env_frames += int(batch["selection_env_frames"])
                            selection_wall_seconds += float(batch["selection_wall_seconds"])
                            native_v8_kernel_rows += int(batch["native_v8"]["kernel_rows"])

                            if overrode and batch["override_scoring"] is not None:
                                sc = batch["override_scoring"]
                                scoring_rollouts_evaluated += int(sc["scoring_rollouts_evaluated"])
                                scoring_env_frames += int(sc["scoring_env_frames"])
                                scoring_wall_seconds += float(sc["scoring_wall_seconds"])
                                bare_sel = outcomes_map.get(bare, {}).get("selection_suffixes", {})
                                chosen_sel = outcomes_map.get(chosen, {}).get(
                                    "selection_suffixes", {}
                                )
                                override_records.append(
                                    {
                                        "world_index": int(world_index),
                                        "seed": int(seed),
                                        "frame": int(frame),
                                        "decision_index": int(decisions_count),
                                        "life": int(life),
                                        "length": int(cur_length),
                                        "length_regime": "L<600" if cur_length < 600 else "L>=600",
                                        "bare_action": int(bare),
                                        "chosen_action": int(chosen),
                                        "bare_is_boost": bool(bare >= 3),
                                        "chosen_is_boost": bool(chosen >= 3),
                                        "bare_raw_survived_selection": bare_raw,
                                        "bare_failed_by_enclosure_only": bool(
                                            (not bare_cert) and bare_raw
                                        ),
                                        "q": q.astype(float).tolist(),
                                        "advisory": np.asarray(advisory, dtype=bool).tolist(),
                                        "evaluation_order": list(batch["evaluation_order"]),
                                        "branches_evaluated": int(batch["branches_evaluated"]),
                                        "chosen_survived_selection_both": bool(
                                            outcomes_map.get(chosen, {}).get("survives", False)
                                        ),
                                        "bare_selection_survives": [
                                            bool(bare_sel.get(str(s), {}).get("survives", False))
                                            for s in selection_suffixes
                                        ],
                                        "bare_selection_raw_survives": [
                                            bool(
                                                bare_sel.get(str(s), {}).get("raw_survives", False)
                                            )
                                            for s in selection_suffixes
                                        ],
                                        "chosen_selection_survives": [
                                            bool(chosen_sel.get(str(s), {}).get("survives", False))
                                            for s in selection_suffixes
                                        ],
                                        "chosen_selection_raw_survives": [
                                            bool(
                                                chosen_sel.get(str(s), {}).get(
                                                    "raw_survives", False
                                                )
                                            )
                                            for s in selection_suffixes
                                        ],
                                        "scoring": sc,
                                    }
                                )

                        if not np.isfinite(q).all() or not 0 <= chosen < 6:
                            raise ValueError("nonfinite Q or illegal action")
                        selected.update(
                            frame=frame,
                            life=life,
                            length=cur_length,
                            bare=bare,
                            chosen=chosen,
                            q=q.astype(float).tolist(),
                            advisory=np.asarray(advisory, dtype=bool).tolist(),
                        )
                    _dispatch_actions(
                        sim,
                        sim.get_resolved_action_mask(),
                        actions,
                        [world.hero],
                        [world.opponents],
                    )
                    if active:
                        if int(actions[0, 0]) != selected["bare"]:
                            raise ValueError("bare dispatch/Q mismatch")
                        actions[0, 0] = selected["chosen"]
                    return actions

                actions_holder: dict[str, np.ndarray] = {}

                def choose_record(sim: Any) -> np.ndarray:
                    actions = choose(sim)
                    if audit_live_mirror:
                        actions_holder["actions"] = actions.copy()
                    return actions

                world.sim.step_with_policy(choose_record)
                world.runtime.observe_step(world.sim)

                cur_grid_alive = [bool(x) for x in world.sim.alive[0].tolist()]
                for slot in range(int(world.sim.S)):
                    if cur_grid_alive[slot] and not prev_grid_alive[slot]:
                        grid_respawns[slot] += 1
                prev_grid_alive = cur_grid_alive

                grid_death = bool(
                    world.sim.get_transition_valid()[0, 0] and not world.sim.alive[0, 0]
                )
                grid_alive_now = bool(world.sim.alive[0, 0])
                grid_mass_now = int(world.sim.length[0, 0]) if grid_alive_now else 0
                sim_cause_code = int(world.sim.get_death_cause()[0, 0]) if grid_death else 0
                live_cause_str: str | None = None

                if audit_live_mirror:
                    actual_decisions: list[int] = []
                    hero = session.live.game.snakes[0]
                    original_update = hero.update
                    original_q = session.live.policy._next_dispatch_q

                    def observed_update(*args: Any, **kwargs: Any) -> Any:
                        actual_decisions.append(int(hero.length))
                        return original_update(*args, **kwargs)

                    def observed_q(*args: Any, **kwargs: Any) -> Any:
                        value = original_q(*args, **kwargs)
                        if not selected or not np.array_equal(
                            value.detach().cpu().numpy().reshape(6),
                            np.asarray(selected["q"], dtype=np.float32),
                        ):
                            raise ValueError("live/mirror B8 Q mismatch")
                        return value

                    hero.update = observed_update
                    session.live.policy._next_dispatch_q = observed_q
                    try:
                        session._step_live(actions_holder["actions"])
                    finally:
                        hero.update = original_update
                        session.live.policy._next_dispatch_q = original_q
                    if actual_decisions != ([selected["length"]] if selected else []):
                        raise ValueError("live/mirror actual hero decision mismatch")
                    hero = session.live.game.snakes[0]
                    live_death = int(hero.id) in session.live.game.frame_death_causes
                    if live_death != grid_death:
                        raise ValueError("Watch/mirror death event mismatch")
                    if selected and int(hero.last_action) != selected["chosen"]:
                        raise ValueError("live delivered action mismatch")
                    if bool(hero.is_alive) != grid_alive_now:
                        raise ValueError("Watch/mirror alive state mismatch")
                    live_mass = int(hero.length) if hero.is_alive else 0
                    if live_mass != grid_mass_now:
                        raise ValueError("Watch/mirror hero length mismatch")
                    if grid_death:
                        live_cause_str = str(session.live.game.frame_death_causes[int(hero.id)])
                        if SIM_CODE_TO_CAUSE_CLASS.get(
                            sim_cause_code
                        ) != LIVE_CAUSE_TO_CAUSE_CLASS.get(live_cause_str):
                            raise ValueError("Watch/mirror death cause mismatch")

                if selected:
                    decisions_count += 1
                    is_boost = int(selected["chosen"] >= 3)
                    boost_decisions += is_boost
                    if int(selected["length"]) < 600:
                        decisions_lt600 += 1
                        boost_decisions_lt600 += is_boost
                    else:
                        decisions_ge600 += 1
                        boost_decisions_ge600 += is_boost

                death_len = int(selected.get("length", 0)) if grid_death else 0
                if grid_death:
                    deaths += 1
                    death_lengths.append(death_len)
                    cause_class = SIM_CODE_TO_CAUSE_CLASS.get(sim_cause_code, "unknown")
                    death_records.append(
                        {
                            "world_index": int(world_index),
                            "seed": int(seed),
                            "frame": int(frame),
                            "life": int(life),
                            "length": int(death_len),
                            "length_regime": "L<600" if death_len < 600 else "L>=600",
                            "sim_cause_code": int(sim_cause_code),
                            "live_cause": live_cause_str,
                            "cause_class": cause_class,
                            "bare_action": int(selected.get("bare", -1)),
                            "chosen_action": int(selected.get("chosen", -1)),
                        }
                    )
                if grid_alive_now:
                    alive_frames += 1
                    if grid_mass_now < 600:
                        alive_frames_lt600 += 1
                    else:
                        alive_frames_ge600 += 1
                    mass_sum += grid_mass_now
                    alive_lengths.append(grid_mass_now)
                    if grid_mass_now > max_length:
                        max_length = grid_mass_now

                was_alive = grid_alive_now
    finally:
        session.close()

    arr_lengths = np.asarray(alive_lengths, dtype=np.float64)
    short_deaths = sum(1 for length in death_lengths if length < 600)
    long_deaths = sum(1 for length in death_lengths if length >= 600)
    late_deaths = sum(1 for length in death_lengths if length >= 800)

    return {
        "world_index": int(world_index),
        "seed": int(seed),
        "arm": arm,
        "controller": OPENSTOP_CONTROLLER,
        "k_selection_suffixes": len(selection_suffixes),
        "selection_suffixes": list(selection_suffixes),
        "branch_horizon": branch_horizon,
        "min_length": min_length,
        "interval": interval,
        "frames": int(horizon),
        "alive_frames": int(alive_frames),
        "alive_frames_lt600": int(alive_frames_lt600),
        "alive_frames_ge600": int(alive_frames_ge600),
        "dead_frames": int(horizon - alive_frames),
        "mass_sum": int(mass_sum),
        "mass_integral": float(mass_sum / horizon),
        "deaths": int(deaths),
        "deaths_per_1k": float(deaths * 1000.0 / horizon),
        "short_body_deaths_lt600": int(short_deaths),
        "long_body_deaths_ge600": int(long_deaths),
        "late_large_body_deaths_ge800": int(late_deaths),
        "deaths_lt600_per_1k_frames": float(short_deaths * 1000.0 / horizon),
        "deaths_ge600_per_1k_frames": float(long_deaths * 1000.0 / horizon),
        "deaths_lt600_per_1k_alive_lt600": (
            float(short_deaths * 1000.0 / alive_frames_lt600) if alive_frames_lt600 > 0 else 0.0
        ),
        "deaths_ge600_per_1k_alive_ge600": (
            float(long_deaths * 1000.0 / alive_frames_ge600) if alive_frames_ge600 > 0 else 0.0
        ),
        "lives": int(life),
        "decisions": int(decisions_count),
        "decisions_lt600": int(decisions_lt600),
        "decisions_ge600": int(decisions_ge600),
        "boost_decisions": int(boost_decisions),
        "boost_decisions_lt600": int(boost_decisions_lt600),
        "boost_decisions_ge600": int(boost_decisions_ge600),
        "boost_share": float(boost_decisions / decisions_count) if decisions_count > 0 else 0.0,
        "max_length": int(max_length),
        "alive_length_median": float(np.median(arr_lengths)) if alive_lengths else 0.0,
        "alive_length_p95": float(np.percentile(arr_lengths, 95)) if alive_lengths else 0.0,
        "death_lengths": death_lengths,
        "death_records": death_records,
        "search_roots": int(search_roots),
        "search_roots_lt600": int(search_roots_lt600),
        "search_roots_ge600": int(search_roots_ge600),
        "bare_survived_both_roots": int(bare_survived_both_roots),
        "bare_raw_survived_roots": int(bare_raw_survived_roots),
        "bare_failed_by_enclosure_roots": int(bare_failed_by_enclosure_roots),
        "bare_failed_by_death_roots": int(bare_failed_by_death_roots),
        "all_die_roots": int(all_die_roots),
        "interventions": int(interventions),
        "interventions_lt600": int(interventions_lt600),
        "interventions_ge600": int(interventions_ge600),
        "interventions_to_nonboost": int(interventions_to_nonboost),
        "interventions_to_boost": int(interventions_to_boost),
        "interventions_from_enclosure_only": int(interventions_from_enclosure_only),
        "interventions_from_raw_death": int(interventions_from_raw_death),
        "branches_evaluated": int(branches_evaluated),
        "branches_skipped": int(branches_skipped),
        "selection_rollouts_evaluated": int(selection_rollouts_evaluated),
        "selection_env_frames": int(selection_env_frames),
        "selection_wall_seconds": float(selection_wall_seconds),
        "scoring_rollouts_evaluated": int(scoring_rollouts_evaluated),
        "scoring_env_frames": int(scoring_env_frames),
        "scoring_wall_seconds": float(scoring_wall_seconds),
        "native_v8_kernel_rows": int(native_v8_kernel_rows),
        "overrides": override_records,
        "respawns_per_slot": grid_respawns,
        "mirror_identity_frames": int(horizon if audit_live_mirror else 0),
        "wall_seconds": float(time.monotonic() - started),
        "roster": roster,
    }


def _load_stage1a_world(
    stage1a_dir: Path | None,
    world_index: int,
    seed: int,
    horizon: int,
    namespace: str,
) -> dict[str, Any] | None:
    if stage1a_dir is None:
        return None
    p = stage1a_dir / "worlds" / f"stage1a-world-{world_index:02d}.json"
    if not p.is_file():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
        if (
            data.get("schema") == "t34-hero-stage1a-paired-world/v1"
            and data.get("namespace") == namespace
            and int(data.get("world_index", -1)) == world_index
            and int(data.get("seed", -1)) == seed
            and int(data.get("horizon", -1)) == horizon
            and int(data["arms"]["bare"]["mirror_identity_frames"]) == horizon
            and int(data["arms"][STAGE1A_CONTROLLER]["mirror_identity_frames"]) == horizon
        ):
            return data
    except Exception:
        return None
    return None


def run_open_stop_world_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Run K=1 and K=2 open-stop-h32-nb on one Watch6-v2 world, reusing Stage 1A bare."""
    _enforce_single_thread_env()
    world_index = int(job["world_index"])
    seed = int(job["seed"])
    horizon = int(job.get("horizon", HORIZON))
    namespace = str(job.get("namespace", NAMESPACE))
    out_dir = Path(job["out_dir"]) if job.get("out_dir") else None
    stage1a_dir = Path(job["stage1a_dir"]) if job.get("stage1a_dir") else None
    audit_live_mirror = bool(job.get("audit_live_mirror", False))

    t0 = time.monotonic()
    stage1a_cached = _load_stage1a_world(stage1a_dir, world_index, seed, horizon, namespace)

    arms_out: dict[str, dict[str, Any]] = {}
    reused_bare = False
    reused_h48 = False
    if stage1a_cached is not None:
        arms_out["bare"] = stage1a_cached["arms"]["bare"]
        arms_out[STAGE1A_CONTROLLER] = stage1a_cached["arms"][STAGE1A_CONTROLLER]
        reused_bare = True
        reused_h48 = True
    else:
        arms_out["bare"] = run_stage1a_arm(
            seed,
            "bare",
            world_index=world_index,
            horizon=horizon,
            audit_live_mirror=audit_live_mirror,
            namespace=namespace,
        )

    # Balanced execution order between K=1 and K=2 across even/odd worlds
    k_order = (
        (OPENSTOP_K1_ARM, OPENSTOP_K2_ARM)
        if (world_index % 2 == 0)
        else (OPENSTOP_K2_ARM, OPENSTOP_K1_ARM)
    )
    for arm in k_order:
        arms_out[arm] = run_open_stop_arm(
            seed,
            arm,
            world_index=world_index,
            horizon=horizon,
            audit_live_mirror=audit_live_mirror,
            namespace=namespace,
        )
        if arms_out["bare"]["roster"] != arms_out[arm]["roster"]:
            raise ValueError(f"roster identity mismatch on world {world_index} for {arm}")

    paired_delta: dict[str, dict[str, Any]] = {}
    for arm in (OPENSTOP_K1_ARM, OPENSTOP_K2_ARM):
        paired_delta[arm] = {
            "mass_integral": float(
                arms_out[arm]["mass_integral"] - arms_out["bare"]["mass_integral"]
            ),
            "deaths": int(arms_out[arm]["deaths"] - arms_out["bare"]["deaths"]),
            "deaths_lt600": int(
                arms_out[arm]["short_body_deaths_lt600"]
                - arms_out["bare"]["short_body_deaths_lt600"]
            ),
            "deaths_ge600": int(
                arms_out[arm]["long_body_deaths_ge600"] - arms_out["bare"]["long_body_deaths_ge600"]
            ),
            "deaths_per_1k": float(
                arms_out[arm]["deaths_per_1k"] - arms_out["bare"]["deaths_per_1k"]
            ),
            "boost_share_pp": float(
                (arms_out[arm]["boost_share"] - arms_out["bare"]["boost_share"]) * 100.0
            ),
        }

    payload = {
        "schema": "t34-hero-openstop-dev-world/v1",
        "in_sample": True,
        "namespace": namespace,
        "world_index": world_index,
        "seed": seed,
        "horizon": horizon,
        "reused_stage1a_bare": reused_bare,
        "reused_stage1a_h48_nb": reused_h48,
        "k_arm_execution_order": list(k_order),
        "arms": arms_out,
        "paired_delta": paired_delta,
        "wall_seconds": float(time.monotonic() - t0),
    }

    if out_dir is not None:
        worlds_dir = out_dir / "worlds"
        worlds_dir.mkdir(parents=True, exist_ok=True)
        world_path = worlds_dir / f"openstop-world-{world_index:02d}.json"
        tmp_path = world_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(world_path)

    return payload


def _summarize_open_stop_overrides(overrides: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(overrides)

    def _mode_stats(subset: Sequence[dict[str, Any]], mode: str) -> dict[str, Any]:
        m = len(subset)
        if m == 0:
            return {
                "overrides_scored": 0,
                "harmful_conditional_majority_count": 0,
                "harmful_conditional_majority_pct": 0.0,
                "harmful_majority_of_4_count": 0,
                "harmful_majority_of_4_pct": 0.0,
                "harmful_ge2_of_4_count": 0,
                "harmful_ge2_of_4_pct": 0.0,
                "harmful_ge1_of_4_count": 0,
                "harmful_ge1_of_4_pct": 0.0,
                "mean_s_score_bare": 0.0,
                "mean_s_score_chosen": 0.0,
                "n_diff_histogram": {str(k): 0 for k in range(5)},
                "n_bare_histogram": {str(k): 0 for k in range(5)},
                "n_chosen_histogram": {str(k): 0 for k in range(5)},
            }

        if mode == "h32_raw":
            blocks = [
                {
                    "harmful_conditional_majority": bool(
                        r["scoring"]["harmful_conditional_majority"]
                    ),
                    "harmful_majority_of_4": bool(r["scoring"]["harmful_majority_of_4"]),
                    "harmful_ge2_of_4": bool(r["scoring"]["harmful_ge2_of_4"]),
                    "harmful_ge1_of_4": bool(r["scoring"]["harmful_ge1_of_4"]),
                    "s_score_bare": float(r["scoring"]["s_score_bare"]),
                    "s_score_chosen": float(r["scoring"]["s_score_chosen"]),
                    "n_diff": int(r["scoring"]["n_override_dies_where_b8_survives"]),
                    "n_bare": int(r["scoring"]["n_bare_survives"]),
                    "n_chosen": int(r["scoring"]["n_chosen_survives"]),
                }
                for r in subset
            ]
        else:
            blocks = [r["scoring"][mode] for r in subset]

        n_cond = sum(1 for b in blocks if bool(b["harmful_conditional_majority"]))
        n_maj4 = sum(1 for b in blocks if bool(b["harmful_majority_of_4"]))
        n_ge2 = sum(1 for b in blocks if bool(b["harmful_ge2_of_4"]))
        n_ge1 = sum(1 for b in blocks if bool(b["harmful_ge1_of_4"]))
        diff_hist = Counter(int(b["n_diff"]) for b in blocks)
        bare_hist = Counter(int(b["n_bare"]) for b in blocks)
        chosen_hist = Counter(int(b["n_chosen"]) for b in blocks)

        return {
            "overrides_scored": m,
            "harmful_conditional_majority_count": n_cond,
            "harmful_conditional_majority_pct": float(n_cond * 100.0 / m),
            "harmful_majority_of_4_count": n_maj4,
            "harmful_majority_of_4_pct": float(n_maj4 * 100.0 / m),
            "harmful_ge2_of_4_count": n_ge2,
            "harmful_ge2_of_4_pct": float(n_ge2 * 100.0 / m),
            "harmful_ge1_of_4_count": n_ge1,
            "harmful_ge1_of_4_pct": float(n_ge1 * 100.0 / m),
            "mean_s_score_bare": float(np.mean([b["s_score_bare"] for b in blocks])),
            "mean_s_score_chosen": float(np.mean([b["s_score_chosen"] for b in blocks])),
            "n_diff_histogram": {str(k): int(diff_hist.get(k, 0)) for k in range(5)},
            "n_bare_histogram": {str(k): int(bare_hist.get(k, 0)) for k in range(5)},
            "n_chosen_histogram": {str(k): int(chosen_hist.get(k, 0)) for k in range(5)},
        }

    lt600 = [r for r in overrides if int(r["length"]) < 600]
    ge600 = [r for r in overrides if int(r["length"]) >= 600]
    encl_only = [r for r in overrides if bool(r.get("bare_failed_by_enclosure_only", False))]
    raw_death = [r for r in overrides if not bool(r.get("bare_failed_by_enclosure_only", False))]

    return {
        "total_overrides_scored": total,
        "overrides_from_enclosure_only": len(encl_only),
        "overrides_from_raw_death": len(raw_death),
        "overall": _mode_stats(overrides, "h32_raw"),
        "h32_raw": {
            "overall": _mode_stats(overrides, "h32_raw"),
            "L<600": _mode_stats(lt600, "h32_raw"),
            "L>=600": _mode_stats(ge600, "h32_raw"),
        },
        "h32_certified": {
            "overall": _mode_stats(overrides, "h32_certified"),
            "L<600": _mode_stats(lt600, "h32_certified"),
            "L>=600": _mode_stats(ge600, "h32_certified"),
        },
        "h48_raw": {
            "overall": _mode_stats(overrides, "h48_raw"),
            "L<600": _mode_stats(lt600, "h48_raw"),
            "L>=600": _mode_stats(ge600, "h48_raw"),
        },
    }


def _subband_cause_table(arm_rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    all_deaths = [d for a in arm_rows for d in a["death_records"]]
    bands = {
        "L<200": [d for d in all_deaths if int(d["length"]) < 200],
        "200<=L<600": [d for d in all_deaths if 200 <= int(d["length"]) < 600],
        "L<600": [d for d in all_deaths if int(d["length"]) < 600],
        "600<=L<800": [d for d in all_deaths if 600 <= int(d["length"]) < 800],
        "L>=800": [d for d in all_deaths if int(d["length"]) >= 800],
        "L>=600": [d for d in all_deaths if int(d["length"]) >= 600],
        "ALL": all_deaths,
    }
    out: dict[str, Any] = {}
    for b_name, d_list in bands.items():
        cnt = Counter(d["cause_class"] for d in d_list)
        opp_total = int(cnt.get("opponent body", 0) + cnt.get("opponent head-on", 0))
        out[b_name] = {
            "total": len(d_list),
            "causes": {c: int(cnt.get(c, 0)) for c in CAUSE_CLASSES},
            "opponent_collision_total": opp_total,
        }
    return out


def _compare_variant(
    ordered: Sequence[dict[str, Any]],
    arm: str,
    bare_summary: dict[str, Any],
    h48_summary: dict[str, Any] | None,
    horizon: int,
) -> dict[str, Any]:
    cand_summary = _summarize_arm(ordered, arm, horizon)
    cand_arm_rows = [r["arms"][arm] for r in ordered]
    bare_arm_rows = [r["arms"]["bare"] for r in ordered]

    cand_summary["deaths"]["subbands"] = _subband_cause_table(cand_arm_rows)
    cand_summary["search"]["bare_raw_survived_roots"] = sum(
        int(a.get("bare_raw_survived_roots", 0)) for a in cand_arm_rows
    )
    cand_summary["search"]["bare_failed_by_enclosure_roots"] = sum(
        int(a.get("bare_failed_by_enclosure_roots", 0)) for a in cand_arm_rows
    )
    cand_summary["search"]["bare_failed_by_death_roots"] = sum(
        int(a.get("bare_failed_by_death_roots", 0)) for a in cand_arm_rows
    )
    cand_summary["search"]["overrides_from_enclosure_only"] = sum(
        int(a.get("interventions_from_enclosure_only", 0)) for a in cand_arm_rows
    )
    cand_summary["search"]["overrides_from_raw_death"] = sum(
        int(a.get("interventions_from_raw_death", 0)) for a in cand_arm_rows
    )

    all_overrides = [ov for a in cand_arm_rows for ov in a["overrides"]]
    scoring_summary = _summarize_open_stop_overrides(all_overrides)

    base_mass = np.array([a["mass_integral"] for a in bare_arm_rows], dtype=np.float64)
    cand_mass = np.array([a["mass_integral"] for a in cand_arm_rows], dtype=np.float64)
    mass_diff = cand_mass - base_mass
    rel_mass, rel_mass_ci95 = _fieller_ratio_ci95(cand_mass, base_mass)

    base_deaths_1k = np.array([a["deaths_per_1k"] for a in bare_arm_rows], dtype=np.float64)
    cand_deaths_1k = np.array([a["deaths_per_1k"] for a in cand_arm_rows], dtype=np.float64)
    deaths_1k_diff = cand_deaths_1k - base_deaths_1k

    base_lt600_1k = np.array(
        [a["deaths_lt600_per_1k_frames"] for a in bare_arm_rows], dtype=np.float64
    )
    cand_lt600_1k = np.array(
        [a["deaths_lt600_per_1k_frames"] for a in cand_arm_rows], dtype=np.float64
    )
    lt600_1k_diff = cand_lt600_1k - base_lt600_1k

    base_lt600_deaths = int(bare_summary["deaths"]["lt600"])
    cand_lt600_deaths = int(cand_summary["deaths"]["lt600"])
    base_ge600_deaths = int(bare_summary["deaths"]["ge600"])
    cand_ge600_deaths = int(cand_summary["deaths"]["ge600"])
    base_total_deaths = int(bare_summary["deaths"]["total"])
    cand_total_deaths = int(cand_summary["deaths"]["total"])

    base_self_all = int(bare_summary["deaths"]["cause_breakdown"]["all"]["self-body enclosure"])
    cand_self_all = int(cand_summary["deaths"]["cause_breakdown"]["all"]["self-body enclosure"])
    base_self_lt600 = int(bare_summary["deaths"]["cause_breakdown"]["lt600"]["self-body enclosure"])
    cand_self_lt600 = int(cand_summary["deaths"]["cause_breakdown"]["lt600"]["self-body enclosure"])
    base_self_ge600 = int(bare_summary["deaths"]["cause_breakdown"]["ge600"]["self-body enclosure"])
    cand_self_ge600 = int(cand_summary["deaths"]["cause_breakdown"]["ge600"]["self-body enclosure"])

    lt600_removed_pct = (
        float((base_lt600_deaths - cand_lt600_deaths) * 100.0 / base_lt600_deaths)
        if base_lt600_deaths > 0
        else 0.0
    )
    ge600_removed_pct = (
        float((base_ge600_deaths - cand_ge600_deaths) * 100.0 / base_ge600_deaths)
        if base_ge600_deaths > 0
        else 0.0
    )
    total_removed_pct = (
        float((base_total_deaths - cand_total_deaths) * 100.0 / base_total_deaths)
        if base_total_deaths > 0
        else 0.0
    )
    self_all_removed_pct = (
        float((base_self_all - cand_self_all) * 100.0 / base_self_all) if base_self_all > 0 else 0.0
    )
    self_lt600_removed_pct = (
        float((base_self_lt600 - cand_self_lt600) * 100.0 / base_self_lt600)
        if base_self_lt600 > 0
        else 0.0
    )
    self_ge600_removed_pct = (
        float((base_self_ge600 - cand_self_ge600) * 100.0 / base_self_ge600)
        if base_self_ge600 > 0
        else 0.0
    )

    boost_share_delta_pooled_pp = float(
        cand_summary["boost"]["boost_share_pooled_pct"]
        - bare_summary["boost"]["boost_share_pooled_pct"]
    )
    boost_share_world_diff_pp = np.array(
        [
            (float(c["boost_share"]) - float(b["boost_share"])) * 100.0
            for b, c in zip(bare_arm_rows, cand_arm_rows)
        ],
        dtype=np.float64,
    )

    override_rate_per_decision_pct = float(
        cand_summary["search"]["override_rate_per_decision_pct"]["overall"]
    )
    override_rate_per_root_pct = float(
        cand_summary["search"]["override_rate_per_root_pct"]["overall"]
    )

    # Pre-registered Dev Filter Rule evaluation:
    # continue to a fresh-world gate only if:
    # 1. override rate <= 0.5%
    # 2. self-enclosure deaths removed >= 55%
    # 3. L<600 deaths removed >= 70%
    bar_ov_decision_pass = bool(
        override_rate_per_decision_pct <= OPENSTOP_DEV_BAR_OVERRIDE_RATE_PCT_MAX
    )
    bar_ov_root_pass = bool(override_rate_per_root_pct <= OPENSTOP_DEV_BAR_OVERRIDE_RATE_PCT_MAX)
    bar_self_encl_pass = bool(
        self_all_removed_pct >= OPENSTOP_DEV_BAR_SELF_ENCLOSURE_REMOVED_PCT_MIN
    )
    bar_lt600_pass = bool(lt600_removed_pct >= OPENSTOP_DEV_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN)

    dev_filter_pass_decision_rate = bool(
        bar_ov_decision_pass and bar_self_encl_pass and bar_lt600_pass
    )
    dev_filter_pass_root_rate = bool(bar_ov_root_pass and bar_self_encl_pass and bar_lt600_pass)

    vs_h48: dict[str, Any] | None = None
    if h48_summary is not None:
        h48_arm_rows = [r["arms"][STAGE1A_CONTROLLER] for r in ordered]
        h48_mass = np.array([a["mass_integral"] for a in h48_arm_rows], dtype=np.float64)
        mass_diff_h48 = cand_mass - h48_mass
        h48_self_all = int(h48_summary["deaths"]["cause_breakdown"]["all"]["self-body enclosure"])
        vs_h48 = {
            "mass_delta_mean": float(mass_diff_h48.mean()),
            "mass_delta_ci95": _paired_ci(mass_diff_h48, 0.95),
            "deaths_total_delta": int(cand_total_deaths - int(h48_summary["deaths"]["total"])),
            "deaths_lt600_delta": int(cand_lt600_deaths - int(h48_summary["deaths"]["lt600"])),
            "deaths_ge600_delta": int(cand_ge600_deaths - int(h48_summary["deaths"]["ge600"])),
            "self_enclosure_deaths_delta": int(cand_self_all - h48_self_all),
            "overrides_delta": int(
                cand_summary["search"]["total_overrides"]
                - int(h48_summary["search"]["total_overrides"])
            ),
            "ms_per_root_delta": float(
                cand_summary["search"]["search_wall_seconds_per_world"]["mean_ms_per_root"]
                - float(h48_summary["search"]["search_wall_seconds_per_world"]["mean_ms_per_root"])
            ),
            "search_wall_s_per_world_delta": float(
                cand_summary["search"]["search_wall_seconds_per_world"]["mean"]
                - float(h48_summary["search"]["search_wall_seconds_per_world"]["mean"])
            ),
        }

    return {
        "arm": arm,
        "summary": cand_summary,
        "override_scoring": scoring_summary,
        "dev_filter_bars": {
            "override_rate_pct": {
                "required_max_pct": OPENSTOP_DEV_BAR_OVERRIDE_RATE_PCT_MAX,
                "observed_per_decision_pct": override_rate_per_decision_pct,
                "observed_per_root_pct": override_rate_per_root_pct,
                "total_overrides": int(cand_summary["search"]["total_overrides"]),
                "total_decisions": int(cand_summary["boost"]["total_decisions"]),
                "total_search_roots": int(cand_summary["search"]["total_search_roots"]),
                "pass_per_decision": bar_ov_decision_pass,
                "pass_per_root": bar_ov_root_pass,
            },
            "self_enclosure_deaths_removed_pct": {
                "required_min_pct": OPENSTOP_DEV_BAR_SELF_ENCLOSURE_REMOVED_PCT_MIN,
                "observed_pct": self_all_removed_pct,
                "bare_self_enclosure_deaths": base_self_all,
                "variant_self_enclosure_deaths": cand_self_all,
                "removed_count": int(base_self_all - cand_self_all),
                "lt600_removed_pct": self_lt600_removed_pct,
                "ge600_removed_pct": self_ge600_removed_pct,
                "pass": bar_self_encl_pass,
            },
            "l_lt_600_deaths_removed_pct": {
                "required_min_pct": OPENSTOP_DEV_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN,
                "observed_pct": lt600_removed_pct,
                "bare_l_lt_600_deaths": base_lt600_deaths,
                "variant_l_lt_600_deaths": cand_lt600_deaths,
                "removed_count": int(base_lt600_deaths - cand_lt600_deaths),
                "pass": bar_lt600_pass,
            },
            "pass_all_per_decision_override_rate": dev_filter_pass_decision_rate,
            "pass_all_per_root_override_rate": dev_filter_pass_root_rate,
        },
        "paired_vs_bare": {
            "mass_delta_mean": float(mass_diff.mean()),
            "mass_delta_sd": float(mass_diff.std(ddof=1)) if len(ordered) > 1 else 0.0,
            "mass_delta_se": (
                float(mass_diff.std(ddof=1) / np.sqrt(len(ordered))) if len(ordered) > 1 else 0.0
            ),
            "mass_delta_ci95": _paired_ci(mass_diff, 0.95),
            "mass_relative_difference": rel_mass,
            "mass_relative_ci95_fieller": rel_mass_ci95,
            "deaths_total_removed_count": int(base_total_deaths - cand_total_deaths),
            "deaths_total_removed_pct": total_removed_pct,
            "deaths_lt600_removed_count": int(base_lt600_deaths - cand_lt600_deaths),
            "deaths_lt600_removed_pct": lt600_removed_pct,
            "deaths_ge600_removed_count": int(base_ge600_deaths - cand_ge600_deaths),
            "deaths_ge600_removed_pct": ge600_removed_pct,
            "self_enclosure_all_removed_count": int(base_self_all - cand_self_all),
            "self_enclosure_all_removed_pct": self_all_removed_pct,
            "self_enclosure_lt600_removed_count": int(base_self_lt600 - cand_self_lt600),
            "self_enclosure_lt600_removed_pct": self_lt600_removed_pct,
            "self_enclosure_ge600_removed_count": int(base_self_ge600 - cand_self_ge600),
            "self_enclosure_ge600_removed_pct": self_ge600_removed_pct,
            "deaths_per_1k_delta_mean": float(deaths_1k_diff.mean()),
            "deaths_per_1k_delta_ci95": _paired_ci(deaths_1k_diff, 0.95),
            "deaths_lt600_per_1k_delta_mean": float(lt600_1k_diff.mean()),
            "deaths_lt600_per_1k_delta_ci95": _paired_ci(lt600_1k_diff, 0.95),
            "boost_share_delta_pooled_pp": boost_share_delta_pooled_pp,
            "boost_share_delta_world_mean_pp": float(boost_share_world_diff_pp.mean()),
            "boost_share_delta_world_ci95_pp": _paired_ci(boost_share_world_diff_pp, 0.95),
            "wins_mass": int(np.sum(mass_diff > 0)),
            "ties_mass": int(np.sum(mass_diff == 0)),
            "losses_mass": int(np.sum(mass_diff < 0)),
            "wins_fewer_deaths": int(np.sum(deaths_1k_diff < 0)),
            "ties_deaths": int(np.sum(deaths_1k_diff == 0)),
            "losses_more_deaths": int(np.sum(deaths_1k_diff > 0)),
            "wins_fewer_lt600_deaths": int(np.sum(lt600_1k_diff < 0)),
            "ties_lt600_deaths": int(np.sum(lt600_1k_diff == 0)),
            "losses_more_lt600_deaths": int(np.sum(lt600_1k_diff > 0)),
        },
        "paired_vs_exact_stop_h48_nb": vs_h48,
    }


def aggregate_open_stop_dev(
    world_results: Sequence[dict[str, Any]],
    *,
    wall_seconds: float,
    workers: int,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Aggregate in-sample dev filter evaluation of open-stop-h32-nb (K=1 and K=2)."""
    ordered = sorted(world_results, key=lambda r: int(r["world_index"]))
    n = len(ordered)
    if n == 0:
        raise ValueError("world_results must be non-empty")
    horizon = int(ordered[0]["horizon"])
    total_frames = n * horizon

    bare_summary = _summarize_arm(ordered, "bare", horizon)
    bare_summary["deaths"]["subbands"] = _subband_cause_table([r["arms"]["bare"] for r in ordered])

    has_h48 = all(STAGE1A_CONTROLLER in r["arms"] for r in ordered)
    h48_summary: dict[str, Any] | None = None
    if has_h48:
        h48_summary = _summarize_arm(ordered, STAGE1A_CONTROLLER, horizon)
        h48_summary["deaths"]["subbands"] = _subband_cause_table(
            [r["arms"][STAGE1A_CONTROLLER] for r in ordered]
        )

    k1_eval = _compare_variant(ordered, OPENSTOP_K1_ARM, bare_summary, h48_summary, horizon)
    k2_eval = _compare_variant(ordered, OPENSTOP_K2_ARM, bare_summary, h48_summary, horizon)

    any_pass_decision_rate = bool(
        k1_eval["dev_filter_bars"]["pass_all_per_decision_override_rate"]
        or k2_eval["dev_filter_bars"]["pass_all_per_decision_override_rate"]
    )
    any_pass_root_rate = bool(
        k1_eval["dev_filter_bars"]["pass_all_per_root_override_rate"]
        or k2_eval["dev_filter_bars"]["pass_all_per_root_override_rate"]
    )
    decision_str = "CONTINUE" if any_pass_decision_rate else "STOP"

    return {
        "schema": "t34-hero-openstop-dev-readout/v1",
        "in_sample": True,
        "in_sample_notice": (
            "IN-SAMPLE DEV FILTER ONLY — these 48 worlds (t34-hero-probe-20261103-v1) "
            "were used in Stage 0b and Stage 1A to diagnose self-enclosure and design "
            "open-stop-h32-nb; results are not decisive release/gate evidence."
        ),
        "namespace": namespace,
        "world_count": n,
        "horizon_per_world": horizon,
        "total_frames_per_arm": total_frames,
        "reused_stage1a_bare_worlds": sum(
            1 for r in ordered if bool(r.get("reused_stage1a_bare", False))
        ),
        "reused_stage1a_h48_nb_worlds": sum(
            1 for r in ordered if bool(r.get("reused_stage1a_h48_nb", False))
        ),
        "workers": int(workers),
        "wall_seconds": round(float(wall_seconds), 3),
        "decision": decision_str,
        "decision_under_per_decision_override_rate": (
            "CONTINUE" if any_pass_decision_rate else "STOP"
        ),
        "decision_under_per_root_override_rate": ("CONTINUE" if any_pass_root_rate else "STOP"),
        "continue": any_pass_decision_rate,
        "comparators": {
            "bare": bare_summary,
            STAGE1A_CONTROLLER: h48_summary,
        },
        "variants": {
            OPENSTOP_K1_ARM: k1_eval,
            OPENSTOP_K2_ARM: k2_eval,
        },
        "per_world_summary": [
            {
                "world_index": int(r["world_index"]),
                "seed": int(r["seed"]),
                "bare_mass": float(r["arms"]["bare"]["mass_integral"]),
                "h48_nb_mass": (
                    float(r["arms"][STAGE1A_CONTROLLER]["mass_integral"])
                    if STAGE1A_CONTROLLER in r["arms"]
                    else None
                ),
                "k1_mass": float(r["arms"][OPENSTOP_K1_ARM]["mass_integral"]),
                "k2_mass": float(r["arms"][OPENSTOP_K2_ARM]["mass_integral"]),
                "bare_deaths": int(r["arms"]["bare"]["deaths"]),
                "h48_nb_deaths": (
                    int(r["arms"][STAGE1A_CONTROLLER]["deaths"])
                    if STAGE1A_CONTROLLER in r["arms"]
                    else None
                ),
                "k1_deaths": int(r["arms"][OPENSTOP_K1_ARM]["deaths"]),
                "k2_deaths": int(r["arms"][OPENSTOP_K2_ARM]["deaths"]),
                "bare_deaths_lt600": int(r["arms"]["bare"]["short_body_deaths_lt600"]),
                "h48_nb_deaths_lt600": (
                    int(r["arms"][STAGE1A_CONTROLLER]["short_body_deaths_lt600"])
                    if STAGE1A_CONTROLLER in r["arms"]
                    else None
                ),
                "k1_deaths_lt600": int(r["arms"][OPENSTOP_K1_ARM]["short_body_deaths_lt600"]),
                "k2_deaths_lt600": int(r["arms"][OPENSTOP_K2_ARM]["short_body_deaths_lt600"]),
                "bare_deaths_ge600": int(r["arms"]["bare"]["long_body_deaths_ge600"]),
                "h48_nb_deaths_ge600": (
                    int(r["arms"][STAGE1A_CONTROLLER]["long_body_deaths_ge600"])
                    if STAGE1A_CONTROLLER in r["arms"]
                    else None
                ),
                "k1_deaths_ge600": int(r["arms"][OPENSTOP_K1_ARM]["long_body_deaths_ge600"]),
                "k2_deaths_ge600": int(r["arms"][OPENSTOP_K2_ARM]["long_body_deaths_ge600"]),
                "k1_search_roots": int(r["arms"][OPENSTOP_K1_ARM]["search_roots"]),
                "k1_overrides": int(r["arms"][OPENSTOP_K1_ARM]["interventions"]),
                "k1_selection_wall_seconds": float(
                    r["arms"][OPENSTOP_K1_ARM]["selection_wall_seconds"]
                ),
                "k2_search_roots": int(r["arms"][OPENSTOP_K2_ARM]["search_roots"]),
                "k2_overrides": int(r["arms"][OPENSTOP_K2_ARM]["interventions"]),
                "k2_selection_wall_seconds": float(
                    r["arms"][OPENSTOP_K2_ARM]["selection_wall_seconds"]
                ),
            }
            for r in ordered
        ],
    }


def run_open_stop_dev(
    *,
    count: int = COUNT,
    horizon: int = HORIZON,
    workers: int = 8,
    out_dir: Path = DEFAULT_OPENSTOP_OUT_DIR,
    stage1a_dir: Path | None = DEFAULT_STAGE1A_DIR,
    audit_live_mirror: bool = False,
    resume_cached_worlds: bool = True,
) -> dict[str, Any]:
    """Execute in-sample dev evaluation of open-stop-h32-nb (K=1 and K=2) across 48 worlds."""
    if not 1 <= workers <= 8:
        raise ValueError("workers must be in 1..8")
    _enforce_single_thread_env()
    out_dir.mkdir(parents=True, exist_ok=True)
    worlds_dir = out_dir / "worlds"
    worlds_dir.mkdir(parents=True, exist_ok=True)

    alloc = allocation(NAMESPACE, count)
    jobs: list[dict[str, Any]] = []
    completed: list[dict[str, Any]] = []

    for w in alloc["worlds"]:
        idx = int(w["world_index"])
        seed = int(w["seed"])
        cached = worlds_dir / f"openstop-world-{idx:02d}.json"
        if resume_cached_worlds and cached.is_file():
            try:
                data = json.loads(cached.read_text(encoding="utf-8"))
                if (
                    data.get("schema") == "t34-hero-openstop-dev-world/v1"
                    and int(data.get("world_index", -1)) == idx
                    and int(data.get("seed", -1)) == seed
                    and int(data.get("horizon", -1)) == horizon
                    and OPENSTOP_K1_ARM in data.get("arms", {})
                    and OPENSTOP_K2_ARM in data.get("arms", {})
                ):
                    completed.append(data)
                    continue
            except Exception:
                pass
        jobs.append(
            {
                "world_index": idx,
                "seed": seed,
                "horizon": horizon,
                "namespace": NAMESPACE,
                "out_dir": str(out_dir),
                "stage1a_dir": str(stage1a_dir) if stage1a_dir is not None else None,
                "audit_live_mirror": audit_live_mirror,
            }
        )

    t0 = time.monotonic()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {
                ex.submit(run_open_stop_world_worker, job): job["world_index"] for job in jobs
            }
            for fut in as_completed(fut_map):
                res = fut.result()
                completed.append(res)
                b_arm = res["arms"]["bare"]
                k1_arm = res["arms"][OPENSTOP_K1_ARM]
                k2_arm = res["arms"][OPENSTOP_K2_ARM]
                print(
                    f"[openstop-dev {len(completed):02d}/{count:02d}] "
                    f"world={res['world_index']:02d} seed={res['seed']} "
                    f"bare(d={b_arm['deaths']}[<600:{b_arm['short_body_deaths_lt600']}], "
                    f"m={b_arm['mass_integral']:.1f}) "
                    f"K1(d={k1_arm['deaths']}[<600:{k1_arm['short_body_deaths_lt600']}], "
                    f"m={k1_arm['mass_integral']:.1f}, ov={k1_arm['interventions']}, "
                    f"t={k1_arm['selection_wall_seconds']:.1f}s) "
                    f"K2(d={k2_arm['deaths']}[<600:{k2_arm['short_body_deaths_lt600']}], "
                    f"m={k2_arm['mass_integral']:.1f}, ov={k2_arm['interventions']}, "
                    f"t={k2_arm['selection_wall_seconds']:.1f}s) "
                    f"wall={res['wall_seconds']:.1f}s",
                    flush=True,
                )

    total_wall = time.monotonic() - t0
    summary = aggregate_open_stop_dev(completed, wall_seconds=total_wall, workers=workers)

    readout_path = out_dir / "OPENSTOP_DEV_READOUT.json"
    readout_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OPENSTOP_OUT_DIR)
    parser.add_argument("--stage1a-dir", type=Path, default=DEFAULT_STAGE1A_DIR)
    parser.add_argument("--audit-live-mirror", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    summary = run_open_stop_dev(
        count=args.count,
        horizon=args.horizon,
        workers=args.workers,
        out_dir=args.out_dir,
        stage1a_dir=args.stage1a_dir,
        audit_live_mirror=args.audit_live_mirror,
        resume_cached_worlds=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "in_sample": summary["in_sample"],
                "decision": summary["decision"],
                "k1_bars": summary["variants"][OPENSTOP_K1_ARM]["dev_filter_bars"],
                "k2_bars": summary["variants"][OPENSTOP_K2_ARM]["dev_filter_bars"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
