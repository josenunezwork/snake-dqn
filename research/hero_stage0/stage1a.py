"""Stage 1A paired evaluation of bare vs exact-stop-h48-nb on the 48 probe worlds."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Sequence
from unittest.mock import patch

import numpy as np
import torch
from scipy.stats import t as student_t

from research.bl_k1.native import RowStats, configure
from research.bl_served_search import ceiling, exact_stop
from research.bl_served_search.ruler_operator import _clone_root, _root_bytes
from research.hero_stage0.protocol import (
    CAUSE_CLASSES,
    COUNT,
    HORIZON,
    LIVE_CAUSE_TO_CAUSE_CLASS,
    NAMESPACE,
    SIM_CODE_TO_CAUSE_CLASS,
    STAGE1A_ARM_SPECS,
    STAGE1A_BAR_BOOST_SHARE_DELTA_PP_MAX,
    STAGE1A_BAR_HARMFUL_OVERRIDE_PCT_MAX,
    STAGE1A_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN,
    STAGE1A_BRANCH_HORIZON,
    STAGE1A_CONTROLLER,
    STAGE1A_SCORING_SUFFIXES,
    STAGE1A_SELECTION_SUFFIXES,
    allocation,
)
from research.t3_rollout_labels_20261015.bank_engine import prepared_q, rollout_all
from research.t3_rollout_labels_20261015.trigger import DenseTrigger
from research.t15_mixed_roster_20261017.episodes import _assert_mixed, roster_identity
from research.watch_replay.engine import step_guard
from research.watch_replay.product import open_watch
from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
from src.simd_env.eval_engine import _dispatch_actions

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a"
)


def _enforce_single_thread_env() -> None:
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "TORCH_NUM_THREADS",
    ):
        os.environ[var] = "1"
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
    os.environ.setdefault("SNAKE_CKPT_DIR", str(REPO_ROOT / "saved_snakes"))
    torch.set_num_threads(1)


def evaluation_order_nb(legal: Sequence[int], bare: int, q: np.ndarray) -> list[int]:
    """Return bare-first, then legal non-boost (0..2) in Q-order, then boost (3..5) in Q-order."""
    actions = [int(action) for action in legal]
    if not actions or len(set(actions)) != len(actions) or bare not in actions:
        raise ValueError("exact stop requires unique legal actions containing bare")
    if q.shape != (6,) or not np.isfinite(q).all():
        raise ValueError("exact stop requires finite six-action Q values")
    return [
        int(bare),
        *sorted(
            (action for action in actions if action != bare),
            key=lambda a: (int(a) >= 3, -float(q[a]), int(a)),
        ),
    ]


def branch_suffix_seed(
    namespace: str,
    world_seed: int,
    decision: int,
    suffix_id: int,
) -> int:
    """Derive a deterministic 64-bit CRN seed for a branch rollout suffix (action-independent)."""
    tag = (
        f"{namespace}:crn-branch:seed-{int(world_seed)}:"
        f"decision-{int(decision)}:suffix-{int(suffix_id)}"
    ).encode("utf-8")
    return int.from_bytes(hashlib.sha256(tag).digest()[:8], "big")


def reseed_branch_suffix(
    branch: Any,
    decision: int,
    suffix_id: int,
    *,
    namespace: str = NAMESPACE,
) -> int:
    """Reseed all environment RNGs on ``branch.sim`` with the deterministic CRN suffix seed."""
    seed64 = branch_suffix_seed(namespace, int(branch.seed), int(decision), int(suffix_id))
    for env_idx, env_rng in enumerate(branch.sim._rngs):
        env_rng._rng.seed(seed64 ^ env_idx)
    return seed64


def _eval_single_branch_suffix(
    root: Any,
    action: int,
    decision: int,
    suffix_id: int,
    *,
    branch_horizon: int = STAGE1A_BRANCH_HORIZON,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Roll out ``action`` for ``branch_horizon`` steps on cloned ``root`` under ``suffix_id``."""
    saved_rng_states = [env_rng._rng.getstate() for env_rng in root.sim._rngs]
    try:
        reseed_branch_suffix(root, decision, suffix_id, namespace=namespace)
        out = rollout_all(
            root,
            [int(action)],
            int(branch_horizon),
            int(decision),
            record_outcomes=True,
        )[int(action)]
    finally:
        for env_rng, st in zip(root.sim._rngs, saved_rng_states):
            env_rng._rng.setstate(st)
    survives = bool(out["survives"])
    frames_executed = int(out["frames_executed"])
    frames_survived = int(branch_horizon) if survives else max(0, frames_executed - 1)
    return {
        "suffix_id": int(suffix_id),
        "survives": survives,
        "frames_executed": frames_executed,
        "frames_survived": frames_survived,
        "death_decision": out.get("death_decision"),
        "final_length": int(out.get("final_length", 0)),
    }


def score_override_independent_suffixes(
    root: Any,
    decision: int,
    bare: int,
    chosen: int,
    *,
    branch_horizon: int = STAGE1A_BRANCH_HORIZON,
    scoring_suffixes: Sequence[int] = STAGE1A_SCORING_SUFFIXES,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Independently score an override (chosen != bare) across the 4 scoring suffixes."""
    if int(chosen) == int(bare):
        raise ValueError("score_override_independent_suffixes requires chosen != bare")
    if set(scoring_suffixes) & set(STAGE1A_SELECTION_SUFFIXES):
        raise ValueError("scoring_suffixes must be disjoint from selection_suffixes")

    t0 = time.perf_counter()
    bare_by_suffix: dict[str, dict[str, Any]] = {}
    chosen_by_suffix: dict[str, dict[str, Any]] = {}
    bare_survives_vec: list[bool] = []
    chosen_survives_vec: list[bool] = []
    diff_dies_vec: list[bool] = []
    scoring_env_frames = 0

    for suf in scoring_suffixes:
        b_out = _eval_single_branch_suffix(
            root,
            bare,
            decision,
            int(suf),
            branch_horizon=branch_horizon,
            namespace=namespace,
        )
        c_out = _eval_single_branch_suffix(
            root,
            chosen,
            decision,
            int(suf),
            branch_horizon=branch_horizon,
            namespace=namespace,
        )
        b_surv = bool(b_out["survives"])
        c_surv = bool(c_out["survives"])
        diff_dies = (not c_surv) and b_surv

        bare_by_suffix[str(suf)] = b_out
        chosen_by_suffix[str(suf)] = c_out
        bare_survives_vec.append(b_surv)
        chosen_survives_vec.append(c_surv)
        diff_dies_vec.append(diff_dies)
        scoring_env_frames += int(b_out["frames_executed"]) + int(c_out["frames_executed"])

    n_score = len(scoring_suffixes)
    n_bare_survives = int(sum(bare_survives_vec))
    n_chosen_survives = int(sum(chosen_survives_vec))
    n_chosen_dies = int(n_score - n_chosen_survives)
    n_diff_dies = int(sum(diff_dies_vec))

    # Primary & alternative definitions of "the override dies within H=48 on a
    # majority of scoring suffixes where the b8-s3 action would not":
    # 1. Conditional majority among scoring suffixes where b8-s3 survives:
    harmful_conditional_majority = bool(
        n_bare_survives > 0 and n_diff_dies > (n_bare_survives / 2.0)
    )
    # 2. Unconditional strict majority (> n_score / 2, i.e. >= 3 of 4) of all scoring suffixes:
    harmful_majority_of_4 = bool(n_diff_dies > (n_score / 2.0))
    # 3. Half-or-more (>= 2 of 4, C/T s_drop >= 0.50 convention):
    harmful_ge2_of_4 = bool(n_diff_dies >= 2)
    # 4. Any scoring suffix (>= 1 of 4) where override dies and b8-s3 survives:
    harmful_ge1_of_4 = bool(n_diff_dies >= 1)

    return {
        "scoring_suffixes": [int(s) for s in scoring_suffixes],
        "bare_action": int(bare),
        "chosen_action": int(chosen),
        "bare_survives_vec": bare_survives_vec,
        "chosen_survives_vec": chosen_survives_vec,
        "override_dies_where_b8_survives_vec": diff_dies_vec,
        "n_scoring_suffixes": n_score,
        "n_bare_survives": n_bare_survives,
        "n_chosen_survives": n_chosen_survives,
        "n_chosen_dies": n_chosen_dies,
        "n_override_dies_where_b8_survives": n_diff_dies,
        "s_score_bare": float(n_bare_survives / n_score) if n_score > 0 else 0.0,
        "s_score_chosen": float(n_chosen_survives / n_score) if n_score > 0 else 0.0,
        "harmful": harmful_conditional_majority,
        "harmful_conditional_majority": harmful_conditional_majority,
        "harmful_majority_of_4": harmful_majority_of_4,
        "harmful_ge2_of_4": harmful_ge2_of_4,
        "harmful_ge1_of_4": harmful_ge1_of_4,
        "bare_outcomes": bare_by_suffix,
        "chosen_outcomes": chosen_by_suffix,
        "scoring_rollouts_evaluated": 2 * n_score,
        "scoring_env_frames": scoring_env_frames,
        "scoring_wall_seconds": time.perf_counter() - t0,
    }


def dense_batch_h48_nb(
    world: Any,
    decision: int,
    q: np.ndarray,
    legal: np.ndarray,
    *,
    branch_horizon: int = STAGE1A_BRANCH_HORIZON,
    selection_suffixes: Sequence[int] = STAGE1A_SELECTION_SUFFIXES,
    scoring_suffixes: Sequence[int] = STAGE1A_SCORING_SUFFIXES,
    namespace: str = NAMESPACE,
    score_overrides: bool = True,
) -> dict[str, Any]:
    """Run exact-stop-h48-nb on K=2 selection suffixes and score overrides on 4 scoring suffixes."""
    if len(selection_suffixes) != 2:
        raise ValueError("exact-stop-h48-nb requires K=2 deterministic selection suffixes")

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
        survives_all = True
        min_frames_executed: int | None = None
        min_frames_survived: int | None = None
        action_env_frames = 0

        for suf in selection_suffixes:
            out_s = _eval_single_branch_suffix(
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
            survives_all = bool(survives_all and bool(out_s["survives"]))
            if min_frames_executed is None or f_exec < min_frames_executed:
                min_frames_executed = f_exec
            if min_frames_survived is None or f_surv < min_frames_survived:
                min_frames_survived = f_surv
            if not survives_all:
                break

        return {
            "survives": survives_all,
            "frames_executed": int(min_frames_executed or 0),
            "frames_survived": int(min_frames_survived or 0),
            "selection_suffixes": suffix_outcomes,
            "rollouts_evaluated": len(suffix_outcomes),
            "env_frames_executed": action_env_frames,
        }

    def choose_all_die(
        outcomes: dict[int, dict[str, Any]], bare_act: int, q_vec: np.ndarray
    ) -> int:
        nonlocal selection_rollouts, selection_env_frames
        for act, out in outcomes.items():
            suf_map = out["selection_suffixes"]
            for suf in selection_suffixes:
                s_key = str(suf)
                if s_key not in suf_map:
                    out_s = _eval_single_branch_suffix(
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
            choose_all_die,
        )
        selection_wall_seconds = time.perf_counter() - t_sel0

        chosen = int(result["chosen"])
        override_scoring: dict[str, Any] | None = None
        if score_overrides and chosen != bare:
            override_scoring = score_override_independent_suffixes(
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


def run_stage1a_arm(
    seed: int,
    arm: str,
    check: Callable[[], None] = lambda: None,
    *,
    world_index: int = 0,
    horizon: int = HORIZON,
    audit_live_mirror: bool = True,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Run one Watch6-v2 session for arm in {'bare', 'exact-stop-h48-nb'} with live/Grid mirror."""
    if arm not in STAGE1A_ARM_SPECS:
        raise ValueError(f"unknown Stage 1A arm: {arm!r}")
    if not 1 <= horizon <= 14400:
        raise ValueError("horizon must be 1..14400")

    spec = STAGE1A_ARM_SPECS[arm]
    min_length: int | None = spec["min_length"]
    interval: int = int(spec["interval"])

    check()
    started = time.monotonic()
    session = open_watch({"seed": int(seed)}, 24, "bare")
    world = session.world
    if not world.sim.allow_respawn or any(
        not getattr(s, "auto_respawn", True) for s in session.live.game.snakes
    ):
        session.close()
        raise ValueError("Stage 1A requires continuous respawn in real Watch and mirror")
    if getattr(session.live, "pathb", None) is not None:
        session.close()
        raise ValueError("Stage 1A refuses Path B runtime assistance")

    _assert_mixed(session.live, 6)
    roster = roster_identity(session.live)

    life = 1
    was_alive = True
    trigger = (
        DenseTrigger(min_length=min_length, interval=interval) if min_length is not None else None
    )
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
    all_die_roots = 0
    interventions = 0
    interventions_lt600 = 0
    interventions_ge600 = 0
    interventions_to_nonboost = 0
    interventions_to_boost = 0
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
                    nonlocal bare_survived_both_roots, all_die_roots
                    nonlocal interventions, interventions_lt600, interventions_ge600
                    nonlocal interventions_to_nonboost, interventions_to_boost
                    nonlocal branches_evaluated, branches_skipped
                    nonlocal selection_rollouts_evaluated, selection_env_frames
                    nonlocal selection_wall_seconds
                    nonlocal scoring_rollouts_evaluated, scoring_env_frames, scoring_wall_seconds
                    nonlocal native_v8_kernel_rows

                    world.runtime.prepare(sim)
                    active = bool(sim.alive[0, 0])
                    if active and not was_alive:
                        life += 1
                        if min_length is not None:
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
                        fired = trigger is not None and trigger.fires(
                            decisions_count, cur_length, True
                        )
                        if fired and advisory.any():
                            batch = dense_batch_h48_nb(
                                world,
                                frame - 1,
                                q,
                                legal,
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
                            if bool(bare_out.get("survives", False)):
                                bare_survived_both_roots += 1
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
                                        "q": q.astype(float).tolist(),
                                        "advisory": np.asarray(advisory, dtype=bool).tolist(),
                                        "evaluation_order": list(batch["evaluation_order"]),
                                        "branches_evaluated": int(batch["branches_evaluated"]),
                                        "chosen_survived_selection_both": bool(
                                            outcomes_map.get(chosen, {}).get("survives", False)
                                        ),
                                        "bare_selection_survives": [
                                            bool(bare_sel.get(str(s), {}).get("survives", False))
                                            for s in STAGE1A_SELECTION_SUFFIXES
                                        ],
                                        "chosen_selection_survives": [
                                            bool(chosen_sel.get(str(s), {}).get("survives", False))
                                            for s in STAGE1A_SELECTION_SUFFIXES
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
        "all_die_roots": int(all_die_roots),
        "interventions": int(interventions),
        "interventions_lt600": int(interventions_lt600),
        "interventions_ge600": int(interventions_ge600),
        "interventions_to_nonboost": int(interventions_to_nonboost),
        "interventions_to_boost": int(interventions_to_boost),
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


def run_paired_world_worker(job: dict[str, Any]) -> dict[str, Any]:
    """Run paired bare and exact-stop-h48-nb on one Watch6-v2 world with live/Grid mirror."""
    _enforce_single_thread_env()
    world_index = int(job["world_index"])
    seed = int(job["seed"])
    horizon = int(job.get("horizon", HORIZON))
    namespace = str(job.get("namespace", NAMESPACE))
    out_dir = Path(job["out_dir"]) if job.get("out_dir") else None

    t0 = time.monotonic()
    # Balanced arm execution order across even/odd worlds
    arm_order = (
        ("bare", STAGE1A_CONTROLLER) if (world_index % 2 == 0) else (STAGE1A_CONTROLLER, "bare")
    )
    arms_out: dict[str, dict[str, Any]] = {}
    for arm in arm_order:
        arms_out[arm] = run_stage1a_arm(
            seed,
            arm,
            world_index=world_index,
            horizon=horizon,
            audit_live_mirror=True,
            namespace=namespace,
        )

    if arms_out["bare"]["roster"] != arms_out[STAGE1A_CONTROLLER]["roster"]:
        raise ValueError(f"roster identity mismatch on world {world_index}")

    pair_payload = {
        "schema": "t34-hero-stage1a-paired-world/v1",
        "namespace": namespace,
        "world_index": world_index,
        "seed": seed,
        "horizon": horizon,
        "arm_execution_order": list(arm_order),
        "arms": {
            "bare": arms_out["bare"],
            STAGE1A_CONTROLLER: arms_out[STAGE1A_CONTROLLER],
        },
        "paired_delta": {
            "mass_integral": float(
                arms_out[STAGE1A_CONTROLLER]["mass_integral"] - arms_out["bare"]["mass_integral"]
            ),
            "deaths": int(arms_out[STAGE1A_CONTROLLER]["deaths"] - arms_out["bare"]["deaths"]),
            "deaths_lt600": int(
                arms_out[STAGE1A_CONTROLLER]["short_body_deaths_lt600"]
                - arms_out["bare"]["short_body_deaths_lt600"]
            ),
            "deaths_ge600": int(
                arms_out[STAGE1A_CONTROLLER]["long_body_deaths_ge600"]
                - arms_out["bare"]["long_body_deaths_ge600"]
            ),
            "deaths_per_1k": float(
                arms_out[STAGE1A_CONTROLLER]["deaths_per_1k"] - arms_out["bare"]["deaths_per_1k"]
            ),
            "boost_share_pp": float(
                (arms_out[STAGE1A_CONTROLLER]["boost_share"] - arms_out["bare"]["boost_share"])
                * 100.0
            ),
        },
        "wall_seconds": float(time.monotonic() - t0),
    }

    if out_dir is not None:
        worlds_dir = out_dir / "worlds"
        worlds_dir.mkdir(parents=True, exist_ok=True)
        world_path = worlds_dir / f"stage1a-world-{world_index:02d}.json"
        tmp_path = world_path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(pair_payload, indent=2) + "\n", encoding="utf-8")
        tmp_path.replace(world_path)

    return pair_payload


def _paired_ci(diff: np.ndarray, confidence: float = 0.95) -> list[float]:
    n = int(diff.size)
    if n < 2:
        val = float(diff[0]) if n == 1 else 0.0
        return [val, val]
    alpha = 1.0 - confidence
    crit = float(student_t.ppf(1.0 - alpha / 2.0, n - 1))
    mean = float(diff.mean())
    se = float(diff.std(ddof=1) / np.sqrt(n))
    return [mean - crit * se, mean + crit * se]


def _fieller_ratio_ci95(
    candidate: np.ndarray, baseline: np.ndarray
) -> tuple[float | None, list[float] | None]:
    """Ratio of pooled means (candidate - baseline) / baseline with Fieller 95% CI."""
    n = int(candidate.size)
    base_mean = float(baseline.mean())
    d = candidate - baseline
    ratio = float(d.mean() / base_mean) if base_mean > 0 else None
    if n < 2 or base_mean <= 0:
        return ratio, None
    crit = float(student_t.ppf(0.975, n - 1))
    c = crit**2 / n
    a = float(base_mean**2 - c * baseline.var(ddof=1))
    b = float(-2.0 * d.mean() * base_mean + 2.0 * c * np.cov(d, baseline, ddof=1)[0, 1])
    q = float(d.mean() ** 2 - c * d.var(ddof=1))
    disc = b * b - 4.0 * a * q
    tol = 64.0 * np.finfo(float).eps * max(abs(b * b), abs(4.0 * a * q), 1.0)
    if -tol <= disc < 0:
        disc = 0.0
    bounds = (
        sorted([float((-b - np.sqrt(disc)) / (2.0 * a)), float((-b + np.sqrt(disc)) / (2.0 * a))])
        if a > 0 and disc >= 0
        else None
    )
    return ratio, bounds


def _summarize_arm(rows: Sequence[dict[str, Any]], arm: str, horizon: int) -> dict[str, Any]:
    n = len(rows)
    total_frames = n * horizon
    arm_rows = [r["arms"][arm] for r in rows]

    mass_arr = np.array([a["mass_integral"] for a in arm_rows], dtype=np.float64)
    deaths_arr = np.array([a["deaths"] for a in arm_rows], dtype=np.float64)
    deaths_1k_arr = np.array([a["deaths_per_1k"] for a in arm_rows], dtype=np.float64)
    deaths_lt600_arr = np.array([a["short_body_deaths_lt600"] for a in arm_rows], dtype=np.float64)
    deaths_ge600_arr = np.array([a["long_body_deaths_ge600"] for a in arm_rows], dtype=np.float64)
    deaths_ge800_arr = np.array(
        [a["late_large_body_deaths_ge800"] for a in arm_rows], dtype=np.float64
    )
    deaths_lt600_1k_arr = deaths_lt600_arr * 1000.0 / horizon
    deaths_ge600_1k_arr = deaths_ge600_arr * 1000.0 / horizon

    total_alive = sum(int(a["alive_frames"]) for a in arm_rows)
    total_alive_lt600 = sum(int(a["alive_frames_lt600"]) for a in arm_rows)
    total_alive_ge600 = sum(int(a["alive_frames_ge600"]) for a in arm_rows)

    total_deaths = int(deaths_arr.sum())
    total_deaths_lt600 = int(deaths_lt600_arr.sum())
    total_deaths_ge600 = int(deaths_ge600_arr.sum())
    total_deaths_ge800 = int(deaths_ge800_arr.sum())

    total_decisions = sum(int(a["decisions"]) for a in arm_rows)
    total_decisions_lt600 = sum(int(a["decisions_lt600"]) for a in arm_rows)
    total_decisions_ge600 = sum(int(a["decisions_ge600"]) for a in arm_rows)
    total_boost = sum(int(a["boost_decisions"]) for a in arm_rows)
    total_boost_lt600 = sum(int(a["boost_decisions_lt600"]) for a in arm_rows)
    total_boost_ge600 = sum(int(a["boost_decisions_ge600"]) for a in arm_rows)

    boost_share_world_pct = np.array(
        [float(a["boost_share"]) * 100.0 for a in arm_rows], dtype=np.float64
    )

    total_roots = sum(int(a["search_roots"]) for a in arm_rows)
    total_roots_lt600 = sum(int(a["search_roots_lt600"]) for a in arm_rows)
    total_roots_ge600 = sum(int(a["search_roots_ge600"]) for a in arm_rows)
    total_bare_survived_both = sum(int(a["bare_survived_both_roots"]) for a in arm_rows)
    total_all_die_roots = sum(int(a["all_die_roots"]) for a in arm_rows)

    total_interventions = sum(int(a["interventions"]) for a in arm_rows)
    total_interventions_lt600 = sum(int(a["interventions_lt600"]) for a in arm_rows)
    total_interventions_ge600 = sum(int(a["interventions_ge600"]) for a in arm_rows)
    total_interventions_nonboost = sum(int(a["interventions_to_nonboost"]) for a in arm_rows)
    total_interventions_boost = sum(int(a["interventions_to_boost"]) for a in arm_rows)

    total_branches_eval = sum(int(a["branches_evaluated"]) for a in arm_rows)
    total_branches_skip = sum(int(a["branches_skipped"]) for a in arm_rows)
    total_sel_rollouts = sum(int(a["selection_rollouts_evaluated"]) for a in arm_rows)
    total_sel_env_frames = sum(int(a["selection_env_frames"]) for a in arm_rows)
    sel_wall_arr = np.array(
        [float(a["selection_wall_seconds"]) for a in arm_rows], dtype=np.float64
    )
    score_wall_arr = np.array(
        [float(a["scoring_wall_seconds"]) for a in arm_rows], dtype=np.float64
    )
    session_wall_arr = np.array([float(a["wall_seconds"]) for a in arm_rows], dtype=np.float64)

    all_death_records = [d for a in arm_rows for d in a["death_records"]]
    cause_counts_all = Counter(d["cause_class"] for d in all_death_records)
    cause_counts_lt600 = Counter(d["cause_class"] for d in all_death_records if d["length"] < 600)
    cause_counts_ge600 = Counter(d["cause_class"] for d in all_death_records if d["length"] >= 600)

    return {
        "arm": arm,
        "label": STAGE1A_ARM_SPECS[arm]["label"],
        "worlds": n,
        "horizon_per_world": horizon,
        "total_session_frames": total_frames,
        "total_alive_frames": total_alive,
        "total_alive_frames_lt600": total_alive_lt600,
        "total_alive_frames_ge600": total_alive_ge600,
        "alive_frame_fraction": float(total_alive / total_frames) if total_frames > 0 else 0.0,
        "mass_mean": float(mass_arr.mean()),
        "mass_sd": float(mass_arr.std(ddof=1)) if n > 1 else 0.0,
        "mass_se": float(mass_arr.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
        "mass_ci95": _paired_ci(mass_arr, 0.95),
        "mean_max_length": float(np.mean([a["max_length"] for a in arm_rows])),
        "deaths": {
            "total": total_deaths,
            "lt600": total_deaths_lt600,
            "ge600": total_deaths_ge600,
            "ge800": total_deaths_ge800,
            "per_1k_session_frames": {
                "total": float(total_deaths * 1000.0 / total_frames) if total_frames > 0 else 0.0,
                "lt600": (
                    float(total_deaths_lt600 * 1000.0 / total_frames) if total_frames > 0 else 0.0
                ),
                "ge600": (
                    float(total_deaths_ge600 * 1000.0 / total_frames) if total_frames > 0 else 0.0
                ),
                "total_ci95": _paired_ci(deaths_1k_arr, 0.95),
                "lt600_ci95": _paired_ci(deaths_lt600_1k_arr, 0.95),
                "ge600_ci95": _paired_ci(deaths_ge600_1k_arr, 0.95),
            },
            "per_1k_alive_frames": {
                "total": float(total_deaths * 1000.0 / total_alive) if total_alive > 0 else 0.0,
                "lt600_in_band": (
                    float(total_deaths_lt600 * 1000.0 / total_alive_lt600)
                    if total_alive_lt600 > 0
                    else 0.0
                ),
                "ge600_in_band": (
                    float(total_deaths_ge600 * 1000.0 / total_alive_ge600)
                    if total_alive_ge600 > 0
                    else 0.0
                ),
            },
            "cause_breakdown": {
                "all": {c: int(cause_counts_all.get(c, 0)) for c in CAUSE_CLASSES},
                "lt600": {c: int(cause_counts_lt600.get(c, 0)) for c in CAUSE_CLASSES},
                "ge600": {c: int(cause_counts_ge600.get(c, 0)) for c in CAUSE_CLASSES},
            },
        },
        "boost": {
            "total_decisions": total_decisions,
            "decisions_lt600": total_decisions_lt600,
            "decisions_ge600": total_decisions_ge600,
            "boost_decisions": total_boost,
            "boost_decisions_lt600": total_boost_lt600,
            "boost_decisions_ge600": total_boost_ge600,
            "boost_share_pooled_pct": (
                float(total_boost * 100.0 / total_decisions) if total_decisions > 0 else 0.0
            ),
            "boost_share_lt600_pct": (
                float(total_boost_lt600 * 100.0 / total_decisions_lt600)
                if total_decisions_lt600 > 0
                else 0.0
            ),
            "boost_share_ge600_pct": (
                float(total_boost_ge600 * 100.0 / total_decisions_ge600)
                if total_decisions_ge600 > 0
                else 0.0
            ),
            "boost_share_world_mean_pct": float(boost_share_world_pct.mean()),
            "boost_share_world_sd_pct": (
                float(boost_share_world_pct.std(ddof=1)) if n > 1 else 0.0
            ),
        },
        "search": {
            "total_search_roots": total_roots,
            "search_roots_lt600": total_roots_lt600,
            "search_roots_ge600": total_roots_ge600,
            "mean_search_roots_per_world": float(total_roots / n) if n > 0 else 0.0,
            "bare_survived_both_roots": total_bare_survived_both,
            "bare_survived_both_pct": (
                float(total_bare_survived_both * 100.0 / total_roots) if total_roots > 0 else 0.0
            ),
            "all_die_roots": total_all_die_roots,
            "all_die_roots_pct": (
                float(total_all_die_roots * 100.0 / total_roots) if total_roots > 0 else 0.0
            ),
            "total_overrides": total_interventions,
            "overrides_lt600": total_interventions_lt600,
            "overrides_ge600": total_interventions_ge600,
            "overrides_to_nonboost": total_interventions_nonboost,
            "overrides_to_boost": total_interventions_boost,
            "override_rate_per_root_pct": {
                "overall": (
                    float(total_interventions * 100.0 / total_roots) if total_roots > 0 else 0.0
                ),
                "lt600": (
                    float(total_interventions_lt600 * 100.0 / total_roots_lt600)
                    if total_roots_lt600 > 0
                    else 0.0
                ),
                "ge600": (
                    float(total_interventions_ge600 * 100.0 / total_roots_ge600)
                    if total_roots_ge600 > 0
                    else 0.0
                ),
            },
            "override_rate_per_decision_pct": {
                "overall": (
                    float(total_interventions * 100.0 / total_decisions)
                    if total_decisions > 0
                    else 0.0
                ),
                "lt600": (
                    float(total_interventions_lt600 * 100.0 / total_decisions_lt600)
                    if total_decisions_lt600 > 0
                    else 0.0
                ),
                "ge600": (
                    float(total_interventions_ge600 * 100.0 / total_decisions_ge600)
                    if total_decisions_ge600 > 0
                    else 0.0
                ),
            },
            "branches_evaluated": total_branches_eval,
            "branches_skipped": total_branches_skip,
            "mean_branches_per_root": (
                float(total_branches_eval / total_roots) if total_roots > 0 else 0.0
            ),
            "selection_rollouts_evaluated": total_sel_rollouts,
            "mean_selection_rollouts_per_root": (
                float(total_sel_rollouts / total_roots) if total_roots > 0 else 0.0
            ),
            "selection_env_frames": total_sel_env_frames,
            "search_wall_seconds_per_world": {
                "total": float(sel_wall_arr.sum()),
                "mean": float(sel_wall_arr.mean()),
                "sd": float(sel_wall_arr.std(ddof=1)) if n > 1 else 0.0,
                "min": float(sel_wall_arr.min()),
                "median": float(np.median(sel_wall_arr)),
                "p90": float(np.percentile(sel_wall_arr, 90)),
                "p95": float(np.percentile(sel_wall_arr, 95)),
                "max": float(sel_wall_arr.max()),
                "mean_ms_per_root": (
                    float(sel_wall_arr.sum() * 1000.0 / total_roots) if total_roots > 0 else 0.0
                ),
                "mean_ms_per_session_frame": (
                    float(sel_wall_arr.sum() * 1000.0 / total_frames) if total_frames > 0 else 0.0
                ),
            },
            "scoring_wall_seconds_per_world": {
                "total": float(score_wall_arr.sum()),
                "mean": float(score_wall_arr.mean()),
                "median": float(np.median(score_wall_arr)),
                "p95": float(np.percentile(score_wall_arr, 95)),
                "max": float(score_wall_arr.max()),
            },
            "session_wall_seconds_per_world": {
                "total": float(session_wall_arr.sum()),
                "mean": float(session_wall_arr.mean()),
                "median": float(np.median(session_wall_arr)),
                "p95": float(np.percentile(session_wall_arr, 95)),
                "max": float(session_wall_arr.max()),
            },
        },
    }


def _summarize_override_scoring(overrides: Sequence[dict[str, Any]]) -> dict[str, Any]:
    total = len(overrides)

    def _band_stats(subset: Sequence[dict[str, Any]]) -> dict[str, Any]:
        m = len(subset)
        n_cond_maj = sum(1 for r in subset if bool(r["scoring"]["harmful_conditional_majority"]))
        n_maj_4 = sum(1 for r in subset if bool(r["scoring"]["harmful_majority_of_4"]))
        n_ge2_4 = sum(1 for r in subset if bool(r["scoring"]["harmful_ge2_of_4"]))
        n_ge1_4 = sum(1 for r in subset if bool(r["scoring"]["harmful_ge1_of_4"]))

        diff_dies_hist = Counter(
            int(r["scoring"]["n_override_dies_where_b8_survives"]) for r in subset
        )
        bare_surv_hist = Counter(int(r["scoring"]["n_bare_survives"]) for r in subset)
        chosen_surv_hist = Counter(int(r["scoring"]["n_chosen_survives"]) for r in subset)
        rescues_ge3_of_4 = sum(
            1
            for r in subset
            if int(r["scoring"]["n_chosen_survives"]) >= 3
            and int(r["scoring"]["n_bare_survives"]) <= 1
        )
        rescues_4_of_4 = sum(
            1
            for r in subset
            if int(r["scoring"]["n_chosen_survives"]) == 4
            and int(r["scoring"]["n_bare_survives"]) == 0
        )

        return {
            "overrides_scored": m,
            "harmful_conditional_majority_count": n_cond_maj,
            "harmful_conditional_majority_pct": float(n_cond_maj * 100.0 / m) if m > 0 else 0.0,
            "harmful_majority_of_4_count": n_maj_4,
            "harmful_majority_of_4_pct": float(n_maj_4 * 100.0 / m) if m > 0 else 0.0,
            "harmful_ge2_of_4_count": n_ge2_4,
            "harmful_ge2_of_4_pct": float(n_ge2_4 * 100.0 / m) if m > 0 else 0.0,
            "harmful_ge1_of_4_count": n_ge1_4,
            "harmful_ge1_of_4_pct": float(n_ge1_4 * 100.0 / m) if m > 0 else 0.0,
            "mean_s_score_bare": (
                float(np.mean([r["scoring"]["s_score_bare"] for r in subset])) if m > 0 else 0.0
            ),
            "mean_s_score_chosen": (
                float(np.mean([r["scoring"]["s_score_chosen"] for r in subset])) if m > 0 else 0.0
            ),
            "rescues_4_of_4_count": rescues_4_of_4,
            "rescues_4_of_4_pct": float(rescues_4_of_4 * 100.0 / m) if m > 0 else 0.0,
            "rescues_ge3_of_4_count": rescues_ge3_of_4,
            "rescues_ge3_of_4_pct": float(rescues_ge3_of_4 * 100.0 / m) if m > 0 else 0.0,
            "n_override_dies_where_b8_survives_histogram": {
                str(k): int(diff_dies_hist.get(k, 0)) for k in range(5)
            },
            "n_bare_survives_histogram": {str(k): int(bare_surv_hist.get(k, 0)) for k in range(5)},
            "n_chosen_survives_histogram": {
                str(k): int(chosen_surv_hist.get(k, 0)) for k in range(5)
            },
        }

    lt600 = [r for r in overrides if int(r["length"]) < 600]
    ge600 = [r for r in overrides if int(r["length"]) >= 600]
    lt200 = [r for r in overrides if int(r["length"]) < 200]
    l200_599 = [r for r in overrides if 200 <= int(r["length"]) < 600]
    l600_799 = [r for r in overrides if 600 <= int(r["length"]) < 800]
    ge800 = [r for r in overrides if int(r["length"]) >= 800]

    harmful_events = [
        r
        for r in overrides
        if bool(r["scoring"]["harmful_conditional_majority"])
        or bool(r["scoring"]["harmful_majority_of_4"])
        or bool(r["scoring"]["harmful_ge2_of_4"])
        or bool(r["scoring"]["harmful_ge1_of_4"])
    ]

    return {
        "total_overrides_scored": total,
        "overall": _band_stats(overrides),
        "by_regime": {
            "L<600": _band_stats(lt600),
            "L>=600": _band_stats(ge600),
            "L<200": _band_stats(lt200),
            "200<=L<600": _band_stats(l200_599),
            "600<=L<800": _band_stats(l600_799),
            "L>=800": _band_stats(ge800),
        },
        "harmful_or_discrepant_overrides": harmful_events,
    }


def aggregate_stage1a(
    world_results: Sequence[dict[str, Any]],
    *,
    wall_seconds: float,
    workers: int,
    namespace: str = NAMESPACE,
) -> dict[str, Any]:
    """Aggregate Stage 1A paired results across the 48 probe worlds and evaluate PASS/FAIL bars."""
    ordered = sorted(world_results, key=lambda r: int(r["world_index"]))
    n = len(ordered)
    if n == 0:
        raise ValueError("world_results must be non-empty")
    horizon = int(ordered[0]["horizon"])
    total_frames = n * horizon

    bare_summary = _summarize_arm(ordered, "bare", horizon)
    cand_summary = _summarize_arm(ordered, STAGE1A_CONTROLLER, horizon)

    all_overrides = [ov for r in ordered for ov in r["arms"][STAGE1A_CONTROLLER]["overrides"]]
    scoring_summary = _summarize_override_scoring(all_overrides)

    # Paired comparisons across the N worlds
    base_mass = np.array([r["arms"]["bare"]["mass_integral"] for r in ordered], dtype=np.float64)
    cand_mass = np.array(
        [r["arms"][STAGE1A_CONTROLLER]["mass_integral"] for r in ordered], dtype=np.float64
    )
    mass_diff = cand_mass - base_mass
    rel_mass, rel_mass_ci95 = _fieller_ratio_ci95(cand_mass, base_mass)

    base_deaths_1k = np.array(
        [r["arms"]["bare"]["deaths_per_1k"] for r in ordered], dtype=np.float64
    )
    cand_deaths_1k = np.array(
        [r["arms"][STAGE1A_CONTROLLER]["deaths_per_1k"] for r in ordered], dtype=np.float64
    )
    deaths_1k_diff = cand_deaths_1k - base_deaths_1k

    base_lt600_1k = np.array(
        [r["arms"]["bare"]["deaths_lt600_per_1k_frames"] for r in ordered], dtype=np.float64
    )
    cand_lt600_1k = np.array(
        [r["arms"][STAGE1A_CONTROLLER]["deaths_lt600_per_1k_frames"] for r in ordered],
        dtype=np.float64,
    )
    lt600_1k_diff = cand_lt600_1k - base_lt600_1k

    base_ge600_1k = np.array(
        [r["arms"]["bare"]["deaths_ge600_per_1k_frames"] for r in ordered], dtype=np.float64
    )
    cand_ge600_1k = np.array(
        [r["arms"][STAGE1A_CONTROLLER]["deaths_ge600_per_1k_frames"] for r in ordered],
        dtype=np.float64,
    )
    ge600_1k_diff = cand_ge600_1k - base_ge600_1k

    base_lt600_deaths = int(bare_summary["deaths"]["lt600"])
    cand_lt600_deaths = int(cand_summary["deaths"]["lt600"])
    base_ge600_deaths = int(bare_summary["deaths"]["ge600"])
    cand_ge600_deaths = int(cand_summary["deaths"]["ge600"])
    base_total_deaths = int(bare_summary["deaths"]["total"])
    cand_total_deaths = int(cand_summary["deaths"]["total"])

    lt600_deaths_removed_count = base_lt600_deaths - cand_lt600_deaths
    lt600_deaths_removed_pct = (
        float(lt600_deaths_removed_count * 100.0 / base_lt600_deaths)
        if base_lt600_deaths > 0
        else 0.0
    )
    ge600_deaths_removed_count = base_ge600_deaths - cand_ge600_deaths
    ge600_deaths_removed_pct = (
        float(ge600_deaths_removed_count * 100.0 / base_ge600_deaths)
        if base_ge600_deaths > 0
        else 0.0
    )
    total_deaths_removed_count = base_total_deaths - cand_total_deaths
    total_deaths_removed_pct = (
        float(total_deaths_removed_count * 100.0 / base_total_deaths)
        if base_total_deaths > 0
        else 0.0
    )

    boost_share_delta_pooled_pp = float(
        cand_summary["boost"]["boost_share_pooled_pct"]
        - bare_summary["boost"]["boost_share_pooled_pct"]
    )
    boost_share_world_diff_pp = np.array(
        [r["paired_delta"]["boost_share_pp"] for r in ordered], dtype=np.float64
    )

    # Bar evaluations
    # Bar 1: L<600 deaths removed >= 70%
    bar1_pass = bool(lt600_deaths_removed_pct >= STAGE1A_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN)
    # Bar 2: delivered boost-share change <= +0.5 pp
    bar2_pass = bool(boost_share_delta_pooled_pp <= STAGE1A_BAR_BOOST_SHARE_DELTA_PP_MAX)
    # Bar 3: harmful overrides <= 1%
    harmful_pct_cond = float(scoring_summary["overall"]["harmful_conditional_majority_pct"])
    harmful_pct_maj4 = float(scoring_summary["overall"]["harmful_majority_of_4_pct"])
    bar3_pass_cond = bool(harmful_pct_cond <= STAGE1A_BAR_HARMFUL_OVERRIDE_PCT_MAX)
    bar3_pass_maj4 = bool(harmful_pct_maj4 <= STAGE1A_BAR_HARMFUL_OVERRIDE_PCT_MAX)
    bar3_pass = bool(bar3_pass_cond and bar3_pass_maj4)

    overall_pass = bool(bar1_pass and bar2_pass and bar3_pass)

    return {
        "schema": "t34-hero-stage1a-readout/v1",
        "namespace": namespace,
        "world_count": n,
        "horizon_per_world": horizon,
        "total_frames_per_arm": total_frames,
        "mirror_identity_frames_bare": sum(
            int(r["arms"]["bare"]["mirror_identity_frames"]) for r in ordered
        ),
        "mirror_identity_frames_exact_stop_h48_nb": sum(
            int(r["arms"][STAGE1A_CONTROLLER]["mirror_identity_frames"]) for r in ordered
        ),
        "workers": int(workers),
        "wall_seconds": round(float(wall_seconds), 3),
        "verdict": "PASS" if overall_pass else "FAIL",
        "pass": overall_pass,
        "bars": {
            "l_lt_600_deaths_removed_pct": {
                "required_min_pct": STAGE1A_BAR_L_LT_600_DEATHS_REMOVED_PCT_MIN,
                "observed_pct": lt600_deaths_removed_pct,
                "bare_l_lt_600_deaths": base_lt600_deaths,
                "exact_stop_h48_nb_l_lt_600_deaths": cand_lt600_deaths,
                "removed_count": lt600_deaths_removed_count,
                "pass": bar1_pass,
            },
            "boost_share_change_pp": {
                "required_max_pp": STAGE1A_BAR_BOOST_SHARE_DELTA_PP_MAX,
                "observed_pooled_pp": boost_share_delta_pooled_pp,
                "observed_world_mean_pp": float(boost_share_world_diff_pp.mean()),
                "bare_boost_share_pct": bare_summary["boost"]["boost_share_pooled_pct"],
                "exact_stop_h48_nb_boost_share_pct": cand_summary["boost"][
                    "boost_share_pooled_pct"
                ],
                "pass": bar2_pass,
            },
            "harmful_overrides_pct": {
                "required_max_pct": STAGE1A_BAR_HARMFUL_OVERRIDE_PCT_MAX,
                "observed_conditional_majority_pct": harmful_pct_cond,
                "observed_conditional_majority_count": int(
                    scoring_summary["overall"]["harmful_conditional_majority_count"]
                ),
                "observed_majority_of_4_pct": harmful_pct_maj4,
                "observed_majority_of_4_count": int(
                    scoring_summary["overall"]["harmful_majority_of_4_count"]
                ),
                "observed_ge2_of_4_pct": float(scoring_summary["overall"]["harmful_ge2_of_4_pct"]),
                "observed_ge2_of_4_count": int(
                    scoring_summary["overall"]["harmful_ge2_of_4_count"]
                ),
                "total_overrides_scored": int(scoring_summary["total_overrides_scored"]),
                "pass": bar3_pass,
                "pass_conditional_majority": bar3_pass_cond,
                "pass_majority_of_4": bar3_pass_maj4,
            },
        },
        "arms": {
            "bare": bare_summary,
            STAGE1A_CONTROLLER: cand_summary,
        },
        "paired_comparison": {
            "mass_delta_mean": float(mass_diff.mean()),
            "mass_delta_sd": float(mass_diff.std(ddof=1)) if n > 1 else 0.0,
            "mass_delta_se": float(mass_diff.std(ddof=1) / np.sqrt(n)) if n > 1 else 0.0,
            "mass_delta_ci95": _paired_ci(mass_diff, 0.95),
            "mass_delta_ci90": _paired_ci(mass_diff, 0.90),
            "mass_relative_difference": rel_mass,
            "mass_relative_ci95_fieller": rel_mass_ci95,
            "deaths_total_removed_count": total_deaths_removed_count,
            "deaths_total_removed_pct": total_deaths_removed_pct,
            "deaths_lt600_removed_count": lt600_deaths_removed_count,
            "deaths_lt600_removed_pct": lt600_deaths_removed_pct,
            "deaths_ge600_removed_count": ge600_deaths_removed_count,
            "deaths_ge600_removed_pct": ge600_deaths_removed_pct,
            "deaths_per_1k_delta_mean": float(deaths_1k_diff.mean()),
            "deaths_per_1k_delta_ci95": _paired_ci(deaths_1k_diff, 0.95),
            "deaths_lt600_per_1k_delta_mean": float(lt600_1k_diff.mean()),
            "deaths_lt600_per_1k_delta_ci95": _paired_ci(lt600_1k_diff, 0.95),
            "deaths_ge600_per_1k_delta_mean": float(ge600_1k_diff.mean()),
            "deaths_ge600_per_1k_delta_ci95": _paired_ci(ge600_1k_diff, 0.95),
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
        "override_scoring": scoring_summary,
        "per_world_summary": [
            {
                "world_index": int(r["world_index"]),
                "seed": int(r["seed"]),
                "arm_execution_order": r["arm_execution_order"],
                "bare_mass": float(r["arms"]["bare"]["mass_integral"]),
                "h48_nb_mass": float(r["arms"][STAGE1A_CONTROLLER]["mass_integral"]),
                "mass_delta": float(r["paired_delta"]["mass_integral"]),
                "bare_deaths": int(r["arms"]["bare"]["deaths"]),
                "h48_nb_deaths": int(r["arms"][STAGE1A_CONTROLLER]["deaths"]),
                "bare_deaths_lt600": int(r["arms"]["bare"]["short_body_deaths_lt600"]),
                "h48_nb_deaths_lt600": int(
                    r["arms"][STAGE1A_CONTROLLER]["short_body_deaths_lt600"]
                ),
                "bare_deaths_ge600": int(r["arms"]["bare"]["long_body_deaths_ge600"]),
                "h48_nb_deaths_ge600": int(r["arms"][STAGE1A_CONTROLLER]["long_body_deaths_ge600"]),
                "bare_boost_share_pct": float(r["arms"]["bare"]["boost_share"] * 100.0),
                "h48_nb_boost_share_pct": float(
                    r["arms"][STAGE1A_CONTROLLER]["boost_share"] * 100.0
                ),
                "search_roots": int(r["arms"][STAGE1A_CONTROLLER]["search_roots"]),
                "overrides": int(r["arms"][STAGE1A_CONTROLLER]["interventions"]),
                "overrides_lt600": int(r["arms"][STAGE1A_CONTROLLER]["interventions_lt600"]),
                "overrides_ge600": int(r["arms"][STAGE1A_CONTROLLER]["interventions_ge600"]),
                "selection_wall_seconds": float(
                    r["arms"][STAGE1A_CONTROLLER]["selection_wall_seconds"]
                ),
                "scoring_wall_seconds": float(
                    r["arms"][STAGE1A_CONTROLLER]["scoring_wall_seconds"]
                ),
                "h48_nb_session_wall_seconds": float(r["arms"][STAGE1A_CONTROLLER]["wall_seconds"]),
                "bare_session_wall_seconds": float(r["arms"]["bare"]["wall_seconds"]),
            }
            for r in ordered
        ],
    }


def run_stage1a(
    *,
    count: int = COUNT,
    horizon: int = HORIZON,
    workers: int = 8,
    out_dir: Path = DEFAULT_OUT_DIR,
    resume_cached_worlds: bool = True,
) -> dict[str, Any]:
    """Execute Stage 1A paired evaluation across all 48 probe worlds (at most 8 workers)."""
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
        cached = worlds_dir / f"stage1a-world-{idx:02d}.json"
        if resume_cached_worlds and cached.is_file():
            try:
                data = json.loads(cached.read_text(encoding="utf-8"))
                if (
                    data.get("schema") == "t34-hero-stage1a-paired-world/v1"
                    and int(data.get("world_index", -1)) == idx
                    and int(data.get("seed", -1)) == seed
                    and int(data.get("horizon", -1)) == horizon
                    and int(data["arms"]["bare"]["mirror_identity_frames"]) == horizon
                    and int(data["arms"][STAGE1A_CONTROLLER]["mirror_identity_frames"]) == horizon
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
            }
        )

    t0 = time.monotonic()
    if jobs:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            fut_map = {ex.submit(run_paired_world_worker, job): job["world_index"] for job in jobs}
            for fut in as_completed(fut_map):
                res = fut.result()
                completed.append(res)
                b_arm = res["arms"]["bare"]
                c_arm = res["arms"][STAGE1A_CONTROLLER]
                b_lt = b_arm["short_body_deaths_lt600"]
                b_ge = b_arm["long_body_deaths_ge600"]
                c_lt = c_arm["short_body_deaths_lt600"]
                c_ge = c_arm["long_body_deaths_ge600"]
                print(
                    f"[stage1a {len(completed):02d}/{count:02d}] "
                    f"world={res['world_index']:02d} seed={res['seed']} "
                    f"bare(mass={b_arm['mass_integral']:.1f}, "
                    f"deaths={b_arm['deaths']}[<600:{b_lt},>=600:{b_ge}]) "
                    f"h48_nb(mass={c_arm['mass_integral']:.1f}, "
                    f"deaths={c_arm['deaths']}[<600:{c_lt},>=600:{c_ge}], "
                    f"roots={c_arm['search_roots']}, ov={c_arm['interventions']}, "
                    f"sel_t={c_arm['selection_wall_seconds']:.1f}s) "
                    f"wall={res['wall_seconds']:.1f}s",
                    flush=True,
                )

    total_wall = time.monotonic() - t0
    summary = aggregate_stage1a(completed, wall_seconds=total_wall, workers=workers)

    readout_path = out_dir / "STAGE1A_READOUT.json"
    readout_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--count", type=int, default=COUNT)
    parser.add_argument("--horizon", type=int, default=HORIZON)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    summary = run_stage1a(
        count=args.count,
        horizon=args.horizon,
        workers=args.workers,
        out_dir=args.out_dir,
        resume_cached_worlds=not args.no_resume,
    )
    print(
        json.dumps(
            {
                "verdict": summary["verdict"],
                "bars": summary["bars"],
                "paired_comparison": summary["paired_comparison"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
