#!/usr/bin/env python3
"""Fixed-state CPU microbenchmark for T24 shared learner/B8 v8 features.

This is an actor decision-frame microbenchmark, never a trainer invocation.  It
uses the real N=3 six-policy pool and a deterministic 128-world GridBatchSim
state.  The long case is injected directly (L=1200) so body-map cost is not
hidden behind a long growth run.  Each timed operation rereads the same prepared
state; all setup, JIT warmup, profiling, and report construction are excluded.

The report separates the synthetic CPU B8 forward/dispatch from host feature
work.  In particular, a lower v8 host cost must not be reported as a reduction
of the B8 network forward cost.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

for _key in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_key] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"

import numba  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
DEFAULT_CKPTS = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")


def _sha(path: str) -> str:
    return hashlib.sha256((REPO / path).read_bytes()).hexdigest()


def _timed(fn: Callable[[], Any], repeats: int) -> dict[str, Any]:
    """Warm once, then retain unprofiled fixed-state samples in milliseconds."""
    fn()
    samples: list[float] = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        samples.append(1000.0 * (time.perf_counter() - started))
    return {"samples_ms": samples, "median_ms": float(np.median(samples))}


def _uniform_seats(pool: Any, learner: np.ndarray) -> np.ndarray:
    """Use every pinned policy each frame, preserving the T22 pool mix."""
    seats = np.empty(learner.shape, dtype=object)
    for index, (world, slot) in enumerate(np.argwhere(~learner)):
        seats[int(world), int(slot)] = pool.policy_ids[index % len(pool.policy_ids)]
    return seats


def _bodymap_sources() -> dict[str, str]:
    paths = (
        "src/simd_env/v8_nb.py",
        "src/simd_env/ego_raster_bv8.py",
        "research/redesign_m3c_20261008/league_pool.py",
        "research/redesign_m3c_20261008/train_m3c.py",
    )
    return {path: _sha(path) for path in paths}


def _build(worlds: int, length: int, seed: int, checkpoint_dir: Path) -> tuple[Any, ...]:
    from research.redesign_m3c_20261008 import train_m3c
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.grid_scenarios import inject_serpentines

    args = train_m3c.default_args()
    args.worlds = worlds
    args.seed = seed
    args.actors = 1
    args.learner_seats = 3
    args.obs_version = "bv8"
    args.v61_backend = "numba"
    args.league_checkpoint_dir = str(checkpoint_dir)
    args.league_b8_checkpoint = str(checkpoint_dir / "b8_s3_t7_20261016.pth")
    ctx = gid._context(1)
    sim, runtime, _, _ = train_m3c.build_segment(args, ctx, 0, 0.99)
    # The capacity comes from the profile and is > 1200.  This does not advance
    # RNG state or execute gameplay; it prepares a fixed, valid long-body state.
    inject_serpentines([sim], [length] * sim.S, box_height=9)
    # The fixed post-reset fixture represents a normal decision frame.  The
    # random-safe pool policy intentionally refuses the reset sentinel frame 0.
    sim.frame[...] = 1
    pool = __import__(
        "research.redesign_m3c_20261008.league_pool", fromlist=["build_default_static_pool"]
    ).build_default_static_pool(args, ctx, sim, runtime, "cpu")
    learner = train_m3c.learner_seat_mask(seed, 0, sim.E, 3)
    seats = _uniform_seats(pool, learner)
    vector_ids = tuple(identity.policy_id for identity in pool.identities if identity.kind == "vector61")
    runtime.controlled[...] = np.isin(seats, vector_ids)
    return sim, runtime, pool, learner, seats, train_m3c.hero_selection_mask


def _case(worlds: int, length: int, repeats: int, seed: int, checkpoint_dir: Path) -> dict[str, Any]:
    from src.simd_env.ego_raster_bv8 import v8_scalars

    sim, runtime, pool, learner, seats, selection_mask = _build(
        worlds, length, seed, checkpoint_dir
    )
    masks = sim.get_resolved_action_mask()
    alive = sim.get_alive()
    learner_rows = np.argwhere(alive & learner)
    learner_mask = selection_mask(sim, learner_rows)
    groups = pool.groups_for(seats, alive)
    b8_rows = groups["b8-s3"]
    assert len(learner_rows) == worlds * 3 and len(b8_rows) > 0
    # The frozen pool includes vector61 identities; prepare their independent
    # runtime state once outside the B8-only dispatch measurements.
    runtime.prepare(sim)

    # Contract checks are outside timing: candidate learner slice must equal the
    # independent old call byte-for-byte and its primed B8 dispatch must preserve
    # the baseline action stream on this exact prepared state.
    direct_features, direct_result = v8_scalars(sim, learner_rows, learner_mask)
    supports_shared_v8 = hasattr(pool, "v8_with_learner_rows")

    def old_two_v8_calls() -> tuple[Any, Any]:
        """The pinned pre-T24 shape: independent learner and B8 v8 calls."""
        learner_out = v8_scalars(sim, learner_rows, learner_mask)
        b8_mask = sim.get_advisory_action_mask()[b8_rows[:, 0], b8_rows[:, 1]]
        v8_scalars(sim, b8_rows, b8_mask)
        return learner_out

    def shared_v8_call() -> tuple[Any, Any]:
        if supports_shared_v8:
            return pool.v8_with_learner_rows(sim, learner_rows, learner_mask, alive, seats)
        return old_two_v8_calls()

    shared_features, shared_result = shared_v8_call()
    if not np.array_equal(direct_features, shared_features):
        raise AssertionError("shared learner v8 feature bytes differ from direct v8")
    for field in ("action", "base", "counts", "need", "spacious", "area", "area_cap"):
        if not np.array_equal(getattr(direct_result, field), getattr(shared_result, field)):
            raise AssertionError(f"shared learner V8Result differs for {field}")

    def direct_learner_v8() -> Any:
        return v8_scalars(sim, learner_rows, learner_mask)

    def shared_learner_b8_v8() -> Any:
        # This is the actor's candidate shared host call: it produces learner
        # bytes and primes B8's exact advisory-mask slice for the following dispatch.
        return shared_v8_call()

    def direct_b8_v8() -> Any:
        b8_mask = sim.get_advisory_action_mask()[b8_rows[:, 0], b8_rows[:, 1]]
        return v8_scalars(sim, b8_rows, b8_mask)

    def baseline_b8_dispatch() -> np.ndarray:
        pool.policies["b8-s3"].clear_frame_cache()
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        pool.dispatch(sim, masks, alive, actions, seats)
        return actions

    def candidate_b8_dispatch() -> np.ndarray:
        shared_v8_call()
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        pool.dispatch(sim, masks, alive, actions, seats)
        return actions

    old_actions = baseline_b8_dispatch()
    new_actions = candidate_b8_dispatch()
    if not np.array_equal(old_actions, new_actions):
        raise AssertionError("primed shared-v8 frozen dispatch changed actions")

    def baseline_decision_frame() -> np.ndarray:
        runtime.prepare(sim)
        pool.policies["b8-s3"].clear_frame_cache()
        _ = v8_scalars(sim, learner_rows, learner_mask)
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        pool.dispatch(sim, masks, alive, actions, seats)
        return actions

    def candidate_decision_frame() -> np.ndarray:
        runtime.prepare(sim)
        _ = pool.prime_b8_with_learner_rows(sim, learner_rows, alive, seats)
        _ = shared_v8_call()
        actions = np.ones((sim.E, sim.S), dtype=np.int64)
        pool.dispatch(sim, masks, alive, actions, seats)
        return actions

    # Repeat identity catches frame-local cache misuse and stamps leaking between calls.
    if not np.array_equal(baseline_decision_frame(), candidate_decision_frame()):
        raise AssertionError("whole fixed-state decision frame changed actions")
    b8_net = pool.policies["b8-s3"].net
    return {
        "worlds": worlds,
        "body_length": length,
        "learner_rows": int(len(learner_rows)),
        "frozen_rows": int((~learner).sum()),
        "b8_rows": int(len(b8_rows)),
        "action_stream_sha256": hashlib.sha256(old_actions.tobytes()).hexdigest(),
        "checks": {
            "learner_v8_bytes_identical": True,
            "frozen_dispatch_actions_identical": True,
            "decision_frame_actions_identical": True,
        },
        "shared_v8_implementation": (
            "one learner+B8 call" if supports_shared_v8 else "two independent baseline calls"
        ),
        "timing": {
            "learner_v8_direct_baseline": _timed(direct_learner_v8, repeats),
            "b8_v8_direct_baseline": _timed(direct_b8_v8, repeats),
            "learner_plus_b8_v8_shared_candidate": _timed(shared_learner_b8_v8, repeats),
        },
        "forward": {
            "b8_device": str(next(b8_net.parameters()).device),
            "scope": "CPU synthetic frozen B8 forward is included in dispatch/frame timings",
        },
    }


def _whole_actor_samples(
    length: int, repeats: int, frames: int, warm: int, seed: int, checkpoint_dir: Path
) -> dict[str, Any]:
    """Run real actor_main frames after injecting an exact body fixture.

    ``bench_actor.run_cell`` owns the shared-memory ring and sink accounting.  Its
    local tracked builder calls this outer wrapper, so serpentines are injected
    after normal segment construction but before the first policy decision.  No
    state restore or simulator construction is charged to actor ``env_s`` or wall
    frame timing.
    """
    from research.perf_actor_20261013 import bench_actor
    from research.redesign_m3c_20261008 import train_m3c as train
    from src.simd_env.grid_scenarios import inject_serpentines

    original = train.build_segment
    fixture_before: list[dict[str, int]] = []

    def injected(*args: Any, **kwargs: Any) -> Any:
        output = original(*args, **kwargs)
        sim = output[0]
        inject_serpentines([sim], [length] * sim.S, box_height=9)
        sim.frame[...] = 1
        segs = sim.seg_count[sim.alive]
        fixture_before.append(
            {
                "live_rows": int(sim.alive.sum()),
                "min_segments": int(segs.min()),
                "max_segments": int(segs.max()),
            }
        )
        return output

    train.build_segment = injected
    extra = " ".join(
        (
            "--learner-seats 3 --obs-version bv8 --onpolicy-labels --label-frac 1.0",
            "--anchor-mode v8margin --shield v8 --reward mass",
            f"--league-checkpoint-dir {checkpoint_dir}",
            f"--league-b8-checkpoint {checkpoint_dir / 'b8_s3_t7_20261016.pth'}",
        )
    )
    samples: list[dict[str, Any]] = []
    try:
        # Each invocation creates a separate real actor and is deliberately
        # unprofiled.  Alternating order belongs in an A/B driver; this function
        # retains the raw samples so such a driver does not hide jitter.
        for rep in range(repeats):
            row = bench_actor.run_cell(
                "numba",
                128,
                "cpu",
                frames,
                warm,
                str(checkpoint_dir / "b8_s3_t7_20261016.pth"),
                0,
                0.0,
                seed + rep,
                trainer_args=extra,
            )
            samples.append(row)
    finally:
        train.build_segment = original
    values = [float(sample["ms_per_frame"]) for sample in samples]
    env_values = [float(sample["env_ms_per_frame"]) for sample in samples]
    return {
        "scope": "real actor_main: step, runtime carry, replay/n-step ring writes; no learner updates",
        "frames_per_sample": frames,
        "warm_frames_excluded": warm,
        "fixture_before_first_decision": fixture_before,
        "samples": samples,
        "median_ms_per_frame": float(np.median(values)),
        "median_env_ms_per_frame": float(np.median(env_values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--worlds", type=int, default=128)
    parser.add_argument("--seed", type=int, default=314159)
    parser.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CKPTS)
    parser.add_argument("--actor-frames", type=int, default=8)
    parser.add_argument("--actor-warm", type=int, default=4)
    parser.add_argument(
        "--without-whole-actor", action="store_true", help="skip bounded actor_main rows"
    )
    args = parser.parse_args()
    if args.repeats < 3 or args.worlds != 128:
        raise SystemExit("requires --worlds 128 and --repeats >= 3 for comparable T24 rows")
    if not args.checkpoint_dir.is_dir():
        raise SystemExit(f"missing checkpoint directory: {args.checkpoint_dir}")
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    numba.set_num_threads(1)
    output = {
        "schema": "t24-v8-bodymap-fixed-state-bench/v1",
        "authority": "CPU fixed-state microbenchmark; not a training, GPU, or throughput result",
        "argv": vars(args),
        "host": {"platform": platform.platform(), "python": sys.version},
        "versions": {"torch": torch.__version__, "numpy": np.__version__, "numba": numba.__version__},
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip(),
        "source_sha256": {
            **_bodymap_sources(),
            str(Path(__file__).relative_to(REPO)): hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        },
        "method": {
            "thread_count": 1,
            "timing": "unprofiled perf_counter samples after one warmup each",
            "state": "same injected prepared state for every timed repetition",
            "short_length": 5,
            "long_length": 1200,
            "whole_frame_scope": "runtime prepare plus learner/B8 features and frozen dispatch; no sim.step mutation",
        },
        "cases": [],
    }
    for length in (5, 1200):
        case = _case(args.worlds, length, args.repeats, args.seed, args.checkpoint_dir)
        if not args.without_whole_actor:
            case["whole_actor"] = _whole_actor_samples(
                length,
                args.repeats,
                args.actor_frames,
                args.actor_warm,
                args.seed,
                args.checkpoint_dir,
            )
        output["cases"].append(case)
        print(json.dumps({"length": length, "timing": case["timing"]}, sort_keys=True), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2, sort_keys=True, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
