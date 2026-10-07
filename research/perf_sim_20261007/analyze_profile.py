"""Summarize a cProfile dump of the eval/actor harness by component.

Self time (tottime) of every function is attributed to one component by its file (and,
inside snake_state.py / ai_snake.py, by function), so components sum to 100%. Also prints
inclusive (cumulative) time of the main entry points and the top self-time functions.
"""

from __future__ import annotations

import pstats
import sys
from collections import defaultdict


def component(filename: str, func: str) -> str:
    f = filename.replace("\\", "/")
    if "/safety_veto" in f or "/src/training/explore_filter" in f:
        return "veto (v8 chain: BFS/landing/area/head)"
    if f.endswith("snake_state.py"):
        if func in ("_get_free_space_features", "reachable", "in_bounds", "to_cell", "_flood"):
            return "state: free-space BFS (61-D)"
        if func in ("_get_danger_map", "update_danger", "_angle_to_sector", "near_body"):
            return "state: danger map"
        if func in ("_get_per_action_danger", "near_min"):
            return "state: per-action danger"
        return "state: other features"
    if f.endswith("snake.py") and func == "_angle_to_sector":
        return "state: danger map"
    if f.endswith("snake_geometry.py"):
        return (
            "action mask (safe-action sim)"
            if func != "near_obstacle_min_d2"
            else "action mask (safe-action sim)"
        )
    if f.endswith("ai_snake.py") and func in (
        "build_obstacle_index",
        "simulate_relative_action_fatality",
        "_get_safe_actions",
        "_simulate_move_after_action",
        "_segments_collide_after_move",
        "_in_bounds_position",
    ):
        return "action mask (safe-action sim)"
    if f.endswith("game_logic.py") or f.endswith("mechanics_constants.py"):
        return "collision + food cell checks (distance/same_cell)"
    if (
        f.endswith("scripted_snake.py")
        or f.endswith("anchors.py")
        or (f.endswith("tournament_eval.py") and func in ("choose_action", "<genexpr>"))
    ):
        return "scripted opponents"
    if (
        "/torch/" in f
        or "apex_network" in f
        or "base_network" in f
        or f.startswith("<built-in method torch")
    ):
        return "torch forward / tensor ops"
    if "{built-in method torch" in func or "torch._C" in func or "method 'argmax'" in func:
        return "torch forward / tensor ops"
    if any(
        s in f
        for s in ("game_state.py", "food_manager.py", "/snake.py", "metrics.py", "behavior_probes")
    ):
        return "game physics / episode loop"
    if any(s in f for s in ("apex_actor", "multistep", "replay", "td_targets", "action_mask")):
        return "replay / actor bookkeeping"
    if "gate_world" in f or "gate_roster" in f:
        return "replay / actor bookkeeping"
    if f.endswith("snake_reward.py") or f.endswith("reward_contract.py") or "reward" in f:
        return "reward"
    if f.endswith("ai_snake.py"):
        return "ai_snake update glue"
    if f == "~":
        if "math." in func or "built-in method builtins" in func or "method '" in func:
            return "python builtins (attributed below by caller)"
    return "other"


def main(path: str) -> None:
    stats = pstats.Stats(path)
    raw = stats.stats  # type: ignore[attr-defined]
    total = sum(v[2] for v in raw.values())
    by = defaultdict(float)
    # Builtins (math.sqrt, len, list ops): attribute to the dominant caller's component.
    for (fn, line, func), (cc, nc, tt, ct, callers) in raw.items():
        comp = component(fn, func)
        if fn == "~" and callers:
            best = None
            for (cfn, cline, cfunc), cstat in callers.items():
                share = cstat[2]
                if best is None or share > best[0]:
                    best = (share, component(cfn, cfunc))
            if best is not None:
                # split proportionally across callers
                tot_c = sum(c[2] for c in callers.values()) or 1.0
                for (cfn, cline, cfunc), cstat in callers.items():
                    by[component(cfn, cfunc)] += tt * (cstat[2] / tot_c)
                continue
        by[comp] += tt
    print(f"total self time {total:.1f}s ({path})")
    for comp, t in sorted(by.items(), key=lambda kv: -kv[1]):
        print(f"  {100 * t / total:5.1f}%  {t:8.1f}s  {comp}")
    print("\ninclusive time of entry points:")
    keys = [
        "update",
        "get_state",
        "_get_free_space_features",
        "_get_per_action_danger",
        "_get_danger_map",
        "_get_safe_actions",
        "apply",
        "decide",
        "forward",
        "handle_collisions",
        "compute_reward_and_train",
        "_choose_action",
        "_send_experience_batch",
        "_compute_n_step_experience",
    ]
    rows = []
    for (fn, line, func), (cc, nc, tt, ct, callers) in raw.items():
        if func in keys and ("src/" in fn or "research" in fn):
            rows.append((ct, nc, f"{fn.split('snake-dqn')[-1].split('ref_')[-1]}:{line} {func}"))
    for ct, nc, name in sorted(rows, reverse=True)[:25]:
        print(f"  {100 * ct / total:5.1f}%  {ct:8.1f}s  calls={nc:<9d} {name}")
    print("\ntop self time:")
    stats.sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main(sys.argv[1])
