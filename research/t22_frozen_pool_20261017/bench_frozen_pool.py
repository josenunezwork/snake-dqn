#!/usr/bin/env python3
"""Bounded, single-thread Mac benchmark for the M3-C six-policy frozen pool.

This is deliberately an actor-frame microbenchmark, not a training invocation.  It
constructs the real 128-world GridBatchSim and SHA-pinned static pool, warms all
numba/torch paths, then measures dispatch plus the runtime/mask preparation the
actor does immediately before dispatch.  ``--reference`` loads ``league_pool.py``
from the recorded pre-change commit, so a later optimized dispatcher can be
measured against the same source oracle without restoring the working tree.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.util
import json
import os
import platform
import pstats
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import replace
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterable

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
REFERENCE_COMMIT = "4c1e1b2"
POOL_PATH = "research/redesign_m3c_20261008/league_pool.py"


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--frames", type=int, default=30, help="measured frames after warmup")
    ap.add_argument("--warmup", type=int, default=12, help="unmeasured JIT/warmup frames")
    ap.add_argument("--worlds", type=int, default=128)
    ap.add_argument("--seed", type=int, default=314159)
    ap.add_argument("--checkpoint-dir", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--reference", action="store_true", help="load dispatcher from base commit")
    return ap


def git_text(revision: str, path: str) -> bytes:
    return subprocess.check_output(["git", "show", f"{revision}:{path}"], cwd=REPO)


def load_reference() -> tuple[ModuleType, str]:
    """Load the base dispatcher under its package name so its relative imports work."""
    text = git_text(REFERENCE_COMMIT, POOL_PATH)
    digest = hashlib.sha256(text).hexdigest()
    name = "research.redesign_m3c_20261008._t22_reference_league_pool"
    spec = importlib.util.spec_from_loader(name, loader=None)
    if spec is None:
        raise RuntimeError("could not build reference module spec")
    module = importlib.util.module_from_spec(spec)
    module.__file__ = f"{REFERENCE_COMMIT}:{POOL_PATH}"
    sys.modules[name] = module
    exec(compile(text, module.__file__, "exec"), module.__dict__)
    return module, digest


class TimedPolicy:
    """Thin observer; it does not change masks, rows, or the action call shape."""

    def __init__(self, policy: Any, policy_id: str) -> None:
        self.policy, self.policy_id = policy, policy_id
        self.total_s = 0.0
        self.calls = 0
        self.rows = 0

    def actions(self, masks: np.ndarray, prepared: Any, rows: np.ndarray) -> np.ndarray:
        started = time.perf_counter()
        result = self.policy.actions(masks, prepared, rows)
        self.total_s += time.perf_counter() - started
        self.calls += 1
        self.rows += len(rows)
        return result

    def reset(self) -> None:
        self.total_s = 0.0
        self.calls = 0
        self.rows = 0

    def __getattr__(self, name: str) -> Any:
        return getattr(self.policy, name)


def uniformly_assigned_seats(pool: Any, learner: np.ndarray) -> np.ndarray:
    """Give every identity exactly the same number of frozen rows each frame."""
    seats = np.empty(learner.shape, dtype=object)
    rows = np.argwhere(~learner)
    policy_ids = pool.policy_ids
    for index, (env, slot) in enumerate(rows):
        seats[int(env), int(slot)] = policy_ids[index % len(policy_ids)]
    return seats


def classify_profile(profile: cProfile.Profile) -> Dict[str, float]:
    """Self-time attribution that stays useful when implementations move slightly."""
    out: Dict[str, float] = defaultdict(float)
    for (filename, _line, function), stat in pstats.Stats(profile).stats.items():
        tottime = float(stat[2])
        path = str(filename).replace("\\", "/")
        if "vector61_featurizer" in path or "vector61_nb" in path:
            out["vector61_features_self_s"] += tottime
        elif "safety_veto" in path or "vector61_policy" in path:
            out["vector61_veto_and_dispatch_self_s"] += tottime
        elif "fast_anchors" in path or "anchors.py" in path:
            out["scripted_self_s"] += tottime
        elif "ego_raster" in path:
            out["bv8_raster_self_s"] += tottime
        elif "ego2s" in path:
            out["bv8_forward_self_s"] += tottime
        elif "league_pool" in path:
            out["pool_dispatch_self_s"] += tottime
        elif function == "actions":
            out["other_actions_self_s"] += tottime
    return dict(sorted(out.items()))


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.frames < 1 or args.warmup < 0 or args.worlds < 6:
        raise SystemExit("frames >= 1, warmup >= 0, and worlds >= 6 are required")
    args.checkpoint_dir = args.checkpoint_dir.resolve()
    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"missing checkpoint directory: {args.checkpoint_dir}")
    import torch

    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    from research.redesign_m3c_20261008 import train_m3c
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    module: Any
    if args.reference:
        module, source_digest = load_reference()
        source_label = f"git:{REFERENCE_COMMIT}"
    else:
        import research.redesign_m3c_20261008.league_pool as module

        source_digest = hashlib.sha256((REPO / POOL_PATH).read_bytes()).hexdigest()
        source_label = "working-tree"
    run_args = train_m3c.default_args()
    run_args.worlds = args.worlds
    run_args.seed = args.seed
    run_args.learner_seats = 3
    run_args.actors = 1
    run_args.obs_version = "bv8"
    run_args.v61_backend = "numba"
    run_args.league_checkpoint_dir = str(args.checkpoint_dir)
    run_args.league_b8_checkpoint = str(args.checkpoint_dir / "b8_s3_t7_20261016.pth")
    ctx = gid._context(1)
    sim, runtime, _, _ = train_m3c.build_segment(run_args, ctx, 0, 0.99)
    pool = module.build_default_static_pool(run_args, ctx, sim, runtime, "cpu")
    timed = {key: TimedPolicy(value, key) for key, value in pool.policies.items()}
    pool.policies = timed
    learner = train_m3c.learner_seat_mask(args.seed, 0, sim.E, 3)
    seats = uniformly_assigned_seats(pool, learner)

    prep_s = 0.0
    dispatch_s = 0.0
    observe_s = 0.0
    phases: Dict[str, float] = defaultdict(float)
    action_digest = hashlib.sha256()

    def timed_method(policy_id: str, method: str, phase: str) -> None:
        policy = timed[policy_id].policy
        original = getattr(policy, method)

        def wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return original(*call_args, **call_kwargs)
            finally:
                phases[f"{policy_id}:{phase}"] += time.perf_counter() - started

        setattr(policy, method, wrapped)

    # Exact phase walls include native torch/numba time; cProfile self time does not.
    for policy_id in ("frp3-s12", "frp3-s12-v8", "champion-v8"):
        timed_method(policy_id, "q_values", "forward")
    for policy_id in ("frp3-s12-v8", "champion-v8"):
        timed_method(policy_id, "_apply_veto", "veto")
    import src.simd_env.ego_raster as ego_raster

    original_raster = ego_raster.build_ego_raster

    def raster_wrapped(*call_args: Any, **call_kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return original_raster(*call_args, **call_kwargs)
        finally:
            phases["b8-s3:raster_features"] += time.perf_counter() - started

    ego_raster.build_ego_raster = raster_wrapped
    b8_net = timed["b8-s3"].policy.net

    def forward_pre(_module: Any, _inputs: Any) -> None:
        phases["b8-s3:forward_started"] = time.perf_counter()

    def forward_post(_module: Any, _inputs: Any, _output: Any) -> None:
        started = phases.pop("b8-s3:forward_started", None)
        if started is not None:
            phases["b8-s3:forward"] += time.perf_counter() - started

    b8_hooks = [b8_net.register_forward_pre_hook(forward_pre), b8_net.register_forward_hook(forward_post)]

    def frame(measure: bool) -> None:
        nonlocal prep_s, dispatch_s, observe_s

        def choose(prepared: Any) -> np.ndarray:
            nonlocal prep_s, dispatch_s
            t0 = time.perf_counter()
            masks = prepared.get_resolved_action_mask()
            alive = prepared.get_alive()
            runtime.prepare(prepared)
            prep_s += time.perf_counter() - t0 if measure else 0.0
            actions = np.ones((sim.E, sim.S), dtype=np.int64)
            # Model the learner's existing b-format gather.  The optimized pool
            # consumes its B8 slice from the same call; the reference deliberately
            # performs the identical learner gather separately.  No learner forward
            # is needed to test feature reuse, and its actions stay constant here.
            from src.simd_env.ego_raster import build_ego_raster

            learner_rows = np.argwhere(alive & learner)
            b8_rows = pool.groups_for(seats, alive).get("b8-s3")
            b8_raw = timed["b8-s3"].policy
            t_shared = time.perf_counter()
            if hasattr(b8_raw, "prime_with_learner_rows") and b8_rows is not None:
                b8_raw.prime_with_learner_rows(prepared, learner_rows, b8_rows)
                phases["learner_b8:shared_b_raster"] += time.perf_counter() - t_shared
            else:
                build_ego_raster(
                    prepared,
                    replace(b8_raw.ego_config, version="b"),
                    backend="numba",
                    rows=learner_rows,
                )
                phases["learner_b8:separate_b_raster"] += time.perf_counter() - t_shared
            t1 = time.perf_counter()
            pool.dispatch(prepared, masks, alive, actions, seats)
            dispatch_s += time.perf_counter() - t1 if measure else 0.0
            action_digest.update(np.ascontiguousarray(actions, dtype=np.int64).tobytes())
            return actions

        sim.step_with_policy(choose)
        t2 = time.perf_counter()
        runtime.observe_step(sim)
        observe_s += time.perf_counter() - t2 if measure else 0.0

    for _ in range(args.warmup):
        frame(False)
    for observer in timed.values():
        observer.reset()
    phases.clear()
    # First timed pass gives wall-clock attribution without profiler overhead.
    for _ in range(args.frames):
        frame(True)
    policy = {
        key: {
            "calls": value.calls,
            "rows": value.rows,
            "elapsed_s": value.total_s,
            "ms_per_frame": 1000.0 * value.total_s / args.frames,
            "us_per_row": 1.0e6 * value.total_s / max(1, value.rows),
        }
        for key, value in timed.items()
    }
    measured_phases = dict(phases)
    # A small second pass is attribution-only.  Its cost is not mixed into timings.
    profiler = cProfile.Profile()
    profiler.enable()
    for _ in range(min(8, args.frames)):
        frame(False)
    profiler.disable()
    for hook in b8_hooks:
        hook.remove()
    report: Dict[str, Any] = {
        "schema": "t22-frozen-pool-microbench/v1",
        "reference": args.reference,
        "source": {
            "label": source_label,
            "executed_league_pool_sha256": source_digest,
            "identity_source_pins": (
                "reference module is executed, but identity validation intentionally reads "
                "the current checkout source paths; see pool_identities for those pins"
                if args.reference
                else "executed module and identity-source paths are the current checkout"
            ),
        },
        "argv": vars(args),
        "host": {"platform": platform.platform(), "processor": platform.processor(), "python": sys.version},
        "torch": str(torch.__version__),
        "pool_identities": pool.identity_records(),
        "measurement": {
            "warmup_frames_excluded": args.warmup,
            "measured_frames": args.frames,
            "worlds": sim.E,
            "learner_seats": 3,
            "frozen_rows_per_frame": int((~learner).sum()),
            "runtime_and_mask_ms_per_frame": 1000.0 * prep_s / args.frames,
            "runtime_observe_ms_per_frame": 1000.0 * observe_s / args.frames,
            "dispatch_ms_per_frame": 1000.0 * dispatch_s / args.frames,
            "policy": policy,
            "phase_ms_per_frame": {
                key: 1000.0 * value / args.frames
                for key, value in sorted(measured_phases.items())
                if not key.endswith("_started")
            },
            "action_stream_sha256": action_digest.hexdigest(),
            "profile_self_time_s": classify_profile(profiler),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, sort_keys=True, default=str) + "\n")
    print(json.dumps(report["measurement"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
