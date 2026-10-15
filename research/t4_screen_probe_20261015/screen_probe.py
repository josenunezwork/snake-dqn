#!/usr/bin/env python3
"""Compare T2's rowwise ego2s hero forward with one batched forward per frame.

This is a development performance probe only.  It deliberately reuses the T2
world construction (round, mix rosters, profile, masks, and anchors) and runs
each path over the *same* ``GridBatchSim`` worlds.  It is not a strict gate:
float tie differences can select a different action and therefore diverge a
world.  A screen must choose one path for every arm.
"""

from __future__ import annotations

import argparse
import contextlib
import dataclasses
import hashlib
import json
import os
import signal
import time
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, Mapping, Sequence

import numpy as np

from research.redesign_m2_20261008 import dev_probe, ni_spec
from research.redesign_scope_20261007 import grid_h5000_identity as gid

FORWARDS = ("rowwise", "batched")
DEVICES = ("auto", "cpu", "mps")
PATHS = ("rowwise", "batched", "compare")
MAX_MPS_WALL_SECONDS = 299


def resolve_device(requested: str) -> str:
    """Resolve the probe device; ``auto`` selects MPS only when it is usable."""
    if requested not in DEVICES:
        raise ValueError("device must be one of %r, got %r" % (DEVICES, requested))
    import torch

    mps = bool(torch.backends.mps.is_available())
    if requested == "mps" and not mps:
        raise ValueError("--device mps requested but MPS is unavailable")
    return "mps" if requested == "auto" and mps else ("cpu" if requested == "auto" else requested)


class DecisionTrace:
    """Hero action trace, keyed by world seed, without modifying evaluation code."""

    def __init__(self, seeds: Sequence[int]) -> None:
        self._seeds = tuple(int(seed) for seed in seeds)
        self.actions: Dict[int, list[int]] = {seed: [] for seed in self._seeds}

    def add(self, slots: np.ndarray, actions: np.ndarray) -> None:
        for slot, action in zip(np.asarray(slots), np.asarray(actions)):
            env, snake = int(slot[0]), int(slot[1])
            if snake == 0:
                self.actions[self._seeds[env]].append(int(action))


@contextlib.contextmanager
def trace_hero_actions(trace: DecisionTrace, device: str) -> Iterator[None]:
    """Temporarily attach a trace and selected device to the new ego2s hero policy.

    ``run_simd_eval`` imports this class inside its call, so this narrow patch
    changes no project default and cannot affect a concurrent process.
    """
    from src.simd_env import ego2s_policy

    original_init = ego2s_policy.Ego2sSimdPolicy.__init__
    original_actions = ego2s_policy.Ego2sSimdPolicy.actions

    def traced_init(self: Any, checkpoint_path: str, **kwargs: Any) -> None:
        kwargs["device"] = device
        original_init(self, checkpoint_path, **kwargs)

    def traced_actions(self: Any, masks: np.ndarray, sim: Any, slots: np.ndarray) -> np.ndarray:
        actions = original_actions(self, masks, sim, slots)
        trace.add(slots, actions)
        return actions

    ego2s_policy.Ego2sSimdPolicy.__init__ = traced_init
    ego2s_policy.Ego2sSimdPolicy.actions = traced_actions
    try:
        yield
    finally:
        ego2s_policy.Ego2sSimdPolicy.__init__ = original_init
        ego2s_policy.Ego2sSimdPolicy.actions = original_actions


@contextlib.contextmanager
def mps_deadline(device: str, seconds: int) -> Iterator[None]:
    """Abort an MPS path before it can exceed the five-minute shared-device budget."""
    if device != "mps":
        yield
        return
    if not 0 < seconds <= MAX_MPS_WALL_SECONDS:
        raise ValueError("MPS max wall seconds must be in [1, %d]" % MAX_MPS_WALL_SECONDS)
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)

    def timeout(_signum: int, _frame: Any) -> None:
        raise TimeoutError("MPS screen path exceeded %d seconds" % seconds)

    signal.signal(signal.SIGALRM, timeout)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)


def materialize_worlds(
    args: argparse.Namespace, ctx: Mapping[str, Any]
) -> tuple[list[int], dict[int, list[Any]]]:
    """Return the exact T2 seed roster for the requested mix."""
    seeds = dev_probe.probe_seeds(args.worlds, args.round_index)
    rows = {
        int(row["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in row["slots"]]
        for row in ctx["ds"]._design_rows(seeds)
        if row["mix"] == args.mix
    }
    if set(rows) != set(seeds):
        raise ValueError("T2 design did not materialize exactly one roster per requested world")
    return seeds, rows


def run_path(
    args: argparse.Namespace,
    forward: str,
    *,
    ctx: Mapping[str, Any] | None = None,
    run_eval: Callable[..., list[dict[str, Any]]] | None = None,
    warmup: bool = True,
    enforce_deadline: bool = True,
) -> dict[str, Any]:
    """Run one forward mode on one all-world ``GridBatchSim`` T2 mix."""
    if forward not in FORWARDS:
        raise ValueError("forward must be one of %r, got %r" % (FORWARDS, forward))
    if args.arm != "student" or not args.student:
        raise ValueError("the screen probe needs --arm student and --student CHECKPOINT")
    ctx = gid._context(1) if ctx is None else ctx
    profile = ctx["profile"]
    frames = args.frames or ni_spec.HORIZON
    if args.worlds <= 0 or frames <= 0:
        raise ValueError("worlds and frames must both be positive")
    if args.frames:
        profile = dataclasses.replace(profile, scored_horizon=frames)
    seeds, rows = materialize_worlds(args, ctx)
    # The historical T2 comparison path is CPU rowwise.  Only the new batched
    # path may occupy MPS, so the reported speedup has an unambiguous baseline.
    device = "cpu" if forward == "rowwise" else resolve_device(args.device)
    trace = DecisionTrace(seeds)
    if run_eval is None:
        from src.simd_env.eval_engine import run_simd_eval

        run_eval = run_simd_eval
    deadline = (
        mps_deadline(device, args.max_wall_seconds)
        if enforce_deadline
        else contextlib.nullcontext()
    )
    with deadline:
        if warmup and args.warmup:
            # Run a small equal warmup for each path outside timing, so compilation,
            # model transfer, and MPS graph setup do not inflate the comparison.
            warm_args = argparse.Namespace(
                **{**vars(args), "worlds": min(2, args.worlds), "frames": 8, "warmup": False}
            )
            run_path(
                warm_args,
                forward,
                ctx=ctx,
                run_eval=run_eval,
                warmup=False,
                enforce_deadline=False,
            )
        started = time.perf_counter()
        with trace_hero_actions(trace, device):
            records = run_eval(
                ctx["lookup"][ni_spec.FRP3_S12[1]],
                rows[seeds[0]],
                frames,
                seeds,
                profile=profile,
                opponent_specs_by_world=rows,
                mix_id=args.mix,
                vector61=True,
                vector61_forward="rowwise",
                sim_engine="grid",
                vector61_backend=dev_probe.effective_v61_backend(args.v61_backend),
                scripted_anchors=args.anchors,
                **{**dev_probe.probe_kwargs(args), "ego2s_forward": forward},
            )
    return {
        "forward": forward,
        "device": device,
        "warmup": {
            "worlds": min(2, args.worlds),
            "frames": 8,
            "performed": bool(warmup and args.warmup),
        },
        "wall_seconds": time.perf_counter() - started,
        "records": {int(record["seed"]): record for record in records},
        "decisions": trace.actions,
    }


def compare_paths(rowwise: Mapping[str, Any], batched: Mapping[str, Any]) -> dict[str, Any]:
    """Compute decision and terminal-outcome agreement without hiding divergence."""
    left, right = rowwise["records"], batched["records"]
    if set(left) != set(right) or set(rowwise["decisions"]) != set(batched["decisions"]):
        raise ValueError("paths must cover the same worlds")
    worlds: dict[int, dict[str, Any]] = {}
    decision_matches = decision_common_total = decision_union_total = missing_tail_total = 0
    for seed in sorted(left):
        a, b = rowwise["decisions"][seed], batched["decisions"][seed]
        common = min(len(a), len(b))
        matches = sum(x == y for x, y in zip(a[:common], b[:common]))
        decision_matches += matches
        decision_common_total += common
        union = max(len(a), len(b))
        missing_tail = abs(len(a) - len(b))
        decision_union_total += union
        missing_tail_total += missing_tail
        lr, br = left[seed], right[seed]
        mass_delta = float(br["mass_integral"]) - float(lr["mass_integral"])
        survival_delta = float(br["survival_fraction"]) - float(lr["survival_fraction"])
        outcome_equal = (
            mass_delta == 0.0 and survival_delta == 0.0 and int(br["deaths"]) == int(lr["deaths"])
        )
        worlds[seed] = {
            "rowwise_decisions": len(a),
            "batched_decisions": len(b),
            "common_decisions": common,
            "matching_decisions": matches,
            "decision_agreement_common_time": float(matches / common) if common else 1.0,
            "decision_union_count": union,
            "missing_tail_decisions": missing_tail,
            "mass_delta_batched_minus_rowwise": mass_delta,
            "survival_delta_batched_minus_rowwise": survival_delta,
            "outcome_agreement": outcome_equal,
        }
    rowwise_wall, batched_wall = float(rowwise["wall_seconds"]), float(batched["wall_seconds"])
    return {
        "decision_agreement_common_time": (
            float(decision_matches / decision_common_total) if decision_common_total else 1.0
        ),
        "decision_agreement_union": (
            float(decision_matches / decision_union_total) if decision_union_total else 1.0
        ),
        "matching_decisions": decision_matches,
        "common_time_decisions": decision_common_total,
        "union_decisions": decision_union_total,
        "missing_tail_decisions": missing_tail_total,
        "outcome_agreement_fraction": float(
            sum(item["outcome_agreement"] for item in worlds.values()) / len(worlds)
        ),
        "max_abs_mass_delta": max(
            abs(item["mass_delta_batched_minus_rowwise"]) for item in worlds.values()
        ),
        "max_abs_survival_delta": max(
            abs(item["survival_delta_batched_minus_rowwise"]) for item in worlds.values()
        ),
        "mean_mass_delta_batched_minus_rowwise": float(
            np.mean([item["mass_delta_batched_minus_rowwise"] for item in worlds.values()])
        ),
        "mean_alive_survival_delta_batched_minus_rowwise": float(
            np.mean([item["survival_delta_batched_minus_rowwise"] for item in worlds.values()])
        ),
        "mean_episode_survival_delta_batched_minus_rowwise": float(
            np.mean(
                [
                    int(right[seed]["deaths"] == 0) - int(left[seed]["deaths"] == 0)
                    for seed in worlds
                ]
            )
        ),
        "survival_only_agreement_fraction": float(
            np.mean(
                [
                    int(right[seed]["deaths"] == 0) == int(left[seed]["deaths"] == 0)
                    for seed in worlds
                ]
            )
        ),
        "rowwise_wall_seconds": rowwise_wall,
        "batched_wall_seconds": batched_wall,
        "speedup_rowwise_over_batched": rowwise_wall / batched_wall if batched_wall else None,
        "worlds": worlds,
    }


def probe(args: argparse.Namespace, ctx: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Run the requested path or comparison and return a JSON-safe result."""
    result: dict[str, Any] = {
        "schema": "t4-screen-probe/v1",
        "authority": "development tooling only; never strict gate evidence",
        "student": str(args.student),
        "student_sha256": hashlib.sha256(Path(args.student).read_bytes()).hexdigest(),
        "round_index": args.round_index,
        "mix": args.mix,
        "worlds_requested": args.worlds,
        "frames": args.frames or ni_spec.HORIZON,
        "path": args.path,
        "batched_device": resolve_device(args.device),
        "rowwise_device": "cpu",
        "anchors": args.anchors,
        "v61_backend": dev_probe.effective_v61_backend(args.v61_backend),
    }
    if args.path == "compare":
        rowwise = run_path(args, "rowwise", ctx=ctx)
        batched = run_path(args, "batched", ctx=ctx)
        result["comparison"] = compare_paths(rowwise, batched)
        result["devices"] = {"rowwise": rowwise["device"], "batched": batched["device"]}
    else:
        path = run_path(args, args.path, ctx=ctx)
        result["run"] = path
        result["mean_mass"] = float(np.mean([r["mass_integral"] for r in path["records"].values()]))
        result["mean_alive_survival"] = float(
            np.mean([r["survival_fraction"] for r in path["records"].values()])
        )
        result["episode_survival_fraction"] = float(
            np.mean([int(r["deaths"] == 0) for r in path["records"].values()])
        )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build an explicit T2-compatible screen-only CLI."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--student", required=True, type=Path)
    parser.add_argument("--arm", choices=("student",), default="student")
    parser.add_argument("--mix", choices=ni_spec.MIXES, default="scripted")
    parser.add_argument("--worlds", type=int, default=32)
    parser.add_argument("--round-index", type=int, default=60)
    parser.add_argument("--frames", type=int, default=None)
    parser.add_argument("--path", choices=PATHS, default="compare")
    parser.add_argument("--device", choices=DEVICES, default="auto")
    parser.add_argument("--max-wall-seconds", type=int, default=295)
    parser.add_argument("--warmup", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--veto", choices=("none",), default="none")
    parser.add_argument("--ego2s-mask", choices=("resolved", "advisory"), default="advisory")
    parser.add_argument("--ego2s-forward", choices=FORWARDS, default="rowwise")
    parser.add_argument("--v61-backend", choices=dev_probe.V61_BACKENDS, default="numba")
    parser.add_argument("--anchors", choices=dev_probe.ANCHORS, default="fast")
    parser.add_argument("--out", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    import torch

    torch.set_num_threads(1)
    if os.environ.get("OMP_NUM_THREADS") != "1":
        raise SystemExit("Set OMP_NUM_THREADS=1")
    result = probe(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, allow_nan=False, indent=2) + "\n")
    print(json.dumps(result.get("comparison", result.get("run")), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
