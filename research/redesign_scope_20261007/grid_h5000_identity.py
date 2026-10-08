#!/usr/bin/env python3
"""DEVELOPMENT-ONLY M1(i) check: frp3-s12 H5000 episodes on GridBatchSim vs BatchSim.

Not a governance screen and not gate evidence. It runs the SAME profiled Watch evaluation
(``run_simd_eval``: pinned deployment config, ``promotion-v2-watch-rect`` profile at H5000,
the strict balanced rosters of ``dev_screen._design_rows`` for the frozen / scripted / mixed
mixes, terminal hero, vector61 rowwise bit-exact forwards) twice per chunk, once on each
engine (``sim_engine="batch"`` and ``sim_engine="grid"``), and compares:

* the whole record (canonical JSON, type-strict; ``veto_diagnostics`` minus wall-clock keys,
  i.e. every key containing ``seconds``), world by world;
* a per-frame whole-world digest stream: per env, every snake's ordered body (ring order from
  the head), head pointer, segment count, length, alive, heading, boost / hunger / respawn
  counters, the advisory / legal / resolved masks, reward, done, death cause, kill credit and
  victim lengths, every step event, the ordered food list, the corpse set and the per-env
  CPython RNG state. The first differing frame is reported.

Arms: ``v8`` (the served frp3-s12 + v8 veto, lambda 8) and ``none`` (frp3-s12 unwrapped, the
M2 non-inferiority comparator). World seeds come from their own namespace
(``redesign-grid-identity/v1``), disjoint from every study.

One invocation runs one (arm, mix) chunk on both engines and writes one create-only JSON
file; ``summarize`` tallies every chunk. Run unlocked, 1 thread, ``nice -n 10``:

  nice -n 10 ./venv/bin/python research/redesign_scope_20261007/grid_h5000_identity.py run \\
      --arm v8 --mix frozen --worlds 8 --out research/redesign_scope_20261007/results/identity
  ./venv/bin/python research/redesign_scope_20261007/grid_h5000_identity.py summarize \\
      --out research/redesign_scope_20261007/results/identity
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

SCHEMA = "redesign-grid-h5000-identity/v1"
AUTHORITY = "development-only identity check (not a screen, not gate evidence)"
DOMAIN = "redesign-grid-identity/v1"
NAMESPACE = "worlds"
CHECKPOINT_DIR = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
FRP3_S12 = (
    "frp3_m3_s12_u60000_20261005.pth",
    "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
)
DEPLOYMENT_CONFIG_SHA256 = "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
ARMS: Dict[str, Dict[str, Any]] = {
    "v8": {"hero_safety_veto": "v8", "hero_safety_veto_lambda": 8.0},
    "none": {"hero_safety_veto": False},
}
ENGINES = ("batch", "grid")
_ARRAYS = (
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
)
_ACCESSORS = (
    "get_action_mask",
    "get_legal_action_mask",
    "get_resolved_action_mask",
    "get_reward",
    "get_done",
    "get_death_cause",
    "get_kill_credit",
    "get_transition_valid",
    "get_boosted_this_step",
)


# ---------------------------------------------------------------------------
# Pure helpers (unit-tested)
# ---------------------------------------------------------------------------
def normalize(value: Any) -> Any:
    """JSON round-trip, as records are saved."""
    return json.loads(json.dumps(value, sort_keys=True, allow_nan=False))


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def strip_timing(value: Any) -> Any:
    """``value`` without wall-clock fields: every key containing ``seconds``, recursively."""
    if isinstance(value, dict):
        return {k: strip_timing(v) for k, v in value.items() if "seconds" not in str(k)}
    if isinstance(value, list):
        return [strip_timing(v) for v in value]
    return value


def _kind(value: Any) -> str:
    if isinstance(value, bool):
        return "bool"
    if isinstance(value, int):
        return "int"
    if isinstance(value, float):
        return "float"
    return type(value).__name__


def diff_paths(a: Any, b: Any, path: str = "") -> List[Tuple[str, Any, Any]]:
    """Every type-strict difference ``(path, a, b)`` (the rule of the v7/v8 H5000 check)."""
    out: List[Tuple[str, Any, Any]] = []
    if isinstance(a, dict) and isinstance(b, dict):
        for key in sorted(set(a) | set(b)):
            sub = f"{path}.{key}" if path else str(key)
            if key not in a:
                out.append((sub, "<missing>", b[key]))
            elif key not in b:
                out.append((sub, a[key], "<missing>"))
            else:
                out.extend(diff_paths(a[key], b[key], sub))
        return out
    if isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append((f"{path}[len]", len(a), len(b)))
        for index, (x, y) in enumerate(zip(a, b)):
            out.extend(diff_paths(x, y, f"{path}[{index}]"))
        return out
    if _kind(a) != _kind(b) or a != b:
        out.append((path, a, b))
    return out


def comparable_record(record: Mapping[str, Any]) -> Any:
    """A record as compared across engines: normalized, veto timing stripped."""
    return strip_timing(normalize(record))


def compare_worlds(
    batch: Mapping[str, Any], grid: Mapping[str, Any], seeds: Sequence[int]
) -> List[Dict[str, Any]]:
    """Per-world comparison of two engine results of the same chunk."""
    rows: List[Dict[str, Any]] = []
    for index, seed in enumerate(seeds):
        ra = comparable_record(batch["records"][index])
        rb = comparable_record(grid["records"][index])
        diffs = diff_paths(ra, rb)
        fa, fb = batch["frame_digests"][index], grid["frame_digests"][index]
        first = next((f for f, (x, y) in enumerate(zip(fa, fb)) if x != y), None)
        if first is None and len(fa) != len(fb):
            first = min(len(fa), len(fb))
        rows.append(
            {
                "world_seed": int(seed),
                "record_identical": not diffs and canonical(ra) == canonical(rb),
                "frames_compared": min(len(fa), len(fb)),
                "first_differing_frame": first,
                "world_chain_batch": chain(fa),
                "world_chain_grid": chain(fb),
                "identical": not diffs and first is None and chain(fa) == chain(fb),
                "record_differences": [
                    {"path": p, "batch": x, "grid": y} for p, x, y in diffs[:25]
                ],
                "mass_integral": ra.get("mass_integral"),
                "survival_fraction": ra.get("survival_fraction"),
                "deaths": ra.get("deaths"),
                "peak_length": ra.get("probes", {}).get("peak_length"),
            }
        )
    return rows


def chain(digests: Sequence[str]) -> str:
    h = hashlib.blake2b(digest_size=16)
    for d in digests:
        h.update(d.encode())
    return h.hexdigest()


# ---------------------------------------------------------------------------
# World digests (installed on both engines' step_with_policy)
# ---------------------------------------------------------------------------
def env_digest(sim: Any, e: int) -> str:
    """Digest of every observable quantity of env ``e`` after a step."""
    import numpy as np

    h = hashlib.blake2b(digest_size=12)
    cap = int(sim.cap)
    for name in _ARRAYS:
        h.update(np.ascontiguousarray(getattr(sim, name)[e]).tobytes())
    for name in _ACCESSORS:
        h.update(np.ascontiguousarray(getattr(sim, name)()[e]).tobytes())
    events = sim.get_step_events()
    for key in sorted(events):
        h.update(key.encode())
        h.update(np.ascontiguousarray(events[key][e]).tobytes())
    for s in range(sim.S):
        n = int(sim.seg_count[e, s])
        rings = (int(sim.head_ptr[e, s]) - np.arange(n)) % cap
        h.update(np.ascontiguousarray(sim.bodies[e, s, rings]).tobytes())
        h.update(repr(list(sim.get_kill_victim_lengths(e, s))).encode())
    h.update(repr(sim.food_cells[e]).encode())
    h.update(repr(sorted(sim.corpse_cells[e])).encode())
    h.update(repr(sim._rngs[e]._rng.getstate()).encode())
    return h.hexdigest()


class DigestRecorder:
    """Wraps a sim class's ``step_with_policy`` to digest every env after each step."""

    def __init__(self) -> None:
        self.per_env: List[List[str]] = []

    def install(self, sim_class: type) -> Any:
        recorder = self
        original = sim_class.step_with_policy

        def step_with_policy(sim, *args, **kwargs):
            out = original(sim, *args, **kwargs)
            if not recorder.per_env:
                recorder.per_env = [[] for _ in range(sim.E)]
            for e in range(sim.E):
                recorder.per_env[e].append(env_digest(sim, e))
            return out

        sim_class.step_with_policy = step_with_policy
        return original


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
def world_seeds(count: int, offset: int = 0) -> List[int]:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.screen_seeds(offset + count, DOMAIN, NAMESPACE)[offset:]


def _context(threads: int) -> Dict[str, Any]:
    import torch

    from research.apex_safety_20260926 import dev_screen as ds
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if ds.sha256_file(ds.DEFAULT_CONFIG) != DEPLOYMENT_CONFIG_SHA256:
        raise SystemExit("deployment config drift")
    load_and_initialize_config(str(ds.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(ds.PROFILE_NAME, ds.HORIZON)
    if profile.digest != ds.PROFILE_DIGEST:
        raise SystemExit("profile digest drift")
    paths = {sha: str(CHECKPOINT_DIR / name) for name, sha in ds.POOL}
    paths[FRP3_S12[1]] = str(CHECKPOINT_DIR / FRP3_S12[0])
    for sha, path in paths.items():
        if ds.sha256_file(Path(path)) != sha:
            raise SystemExit(f"{path} does not hash to {sha}")
    return {"profile": profile, "lookup": ds.agent_lookup(paths), "ds": ds}


def run_engine(
    ctx: Mapping[str, Any],
    arm: str,
    mix: str,
    seeds: Sequence[int],
    engine: str,
    frames: int,
) -> Dict[str, Any]:
    from src.simd_env import eval_engine as ee

    ds = ctx["ds"]
    rows = [r for r in ds._design_rows(list(seeds)) if r["mix"] == mix]
    rosters = {
        int(r["world_seed"]): [ctx["lookup"][slot["member_sha256"]] for slot in r["slots"]]
        for r in rows
    }
    sim_class = ee._terminal_hero_sim_class(engine)
    recorder = DigestRecorder()
    original = recorder.install(sim_class)
    started = time.perf_counter()
    try:
        records = ee.run_simd_eval(
            ctx["lookup"][FRP3_S12[1]],
            rosters[int(seeds[0])],
            frames,
            list(seeds),
            profile=ctx["profile"],
            opponent_specs_by_world=rosters,
            mix_id=mix,
            vector61=True,
            vector61_forward="rowwise",
            sim_engine=engine,
            **ARMS[arm],
        )
    finally:
        sim_class.step_with_policy = original
    wall = time.perf_counter() - started
    return {
        "engine": engine,
        "wall_seconds": wall,
        "records": [normalize(r) for r in records],
        "frame_digests": recorder.per_env,
    }


def _grid_jit() -> bool:
    """Whether the grid engine runs the compiled (numba) mask / collision kernels."""
    from src.simd_env.grid_sim import GridBatchSim, _numba_available

    default = GridBatchSim.JIT_DEFAULT
    return bool(_numba_available() if default is None else default)


def _git() -> Dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty_paths": run("status", "--porcelain")}


def cmd_run(args: argparse.Namespace) -> int:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{args.arm}-{args.mix}-w{args.offset}-{args.offset + args.worlds - 1}.json"
    if path.exists():
        print(f"refusing: {path} exists (create-only)", file=sys.stderr)
        return 2
    ctx = _context(args.threads)
    seeds = world_seeds(args.worlds, args.offset)
    frames = int(ctx["ds"].HORIZON) if args.frames is None else int(args.frames)
    if frames != ctx["ds"].HORIZON:
        raise SystemExit("the profiled path requires frames == H5000")
    results = {e: run_engine(ctx, args.arm, args.mix, seeds, e, frames) for e in ENGINES}
    rows = compare_worlds(results["batch"], results["grid"], seeds)
    import numba
    import numpy
    import torch

    payload = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": _git(),
        "arm": args.arm,
        "arm_kwargs": ARMS[args.arm],
        "mix": args.mix,
        "world_seeds": [int(s) for s in seeds],
        "frames": frames,
        "hero": {"name": FRP3_S12[0], "sha256": FRP3_S12[1]},
        "threads": {"torch": args.threads, "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS")},
        "versions": {
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "torch": torch.__version__,
            "numba": numba.__version__,
        },
        "wall_seconds": {e: results[e]["wall_seconds"] for e in ENGINES},
        "grid_sim_jit": _grid_jit(),
        "worlds": rows,
        "identical": sum(r["identical"] for r in rows),
        "compared": len(rows),
        "records": {e: results[e]["records"] for e in ENGINES},
    }
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")
    print(
        json.dumps(
            {
                "chunk": path.name,
                "identical": payload["identical"],
                "compared": payload["compared"],
                "wall_seconds": payload["wall_seconds"],
                "mass": [r["mass_integral"] for r in rows],
                "peak_length": [r["peak_length"] for r in rows],
            }
        ),
        flush=True,
    )
    return 0 if payload["identical"] == payload["compared"] else 1


def cmd_summarize(args: argparse.Namespace) -> int:
    out = Path(args.out)
    chunks = sorted(out.glob("*-w*.json"))
    table: Dict[str, Dict[str, Any]] = {}
    total = identical = 0
    batch_wall = grid_wall = 0.0
    for path in chunks:
        payload = json.loads(path.read_text())
        key = f"{payload['arm']}/{payload['mix']}"
        row = table.setdefault(key, {"compared": 0, "identical": 0, "frames": 0, "max_peak": 0})
        row["compared"] += payload["compared"]
        row["identical"] += payload["identical"]
        row["frames"] += sum(w["frames_compared"] for w in payload["worlds"])
        row["max_peak"] = max(
            [row["max_peak"]] + [int(w["peak_length"] or 0) for w in payload["worlds"]]
        )
        total += payload["compared"]
        identical += payload["identical"]
        batch_wall += payload["wall_seconds"]["batch"]
        grid_wall += payload["wall_seconds"]["grid"]
    summary = {
        "schema_version": SCHEMA,
        "authority": AUTHORITY,
        "chunks": [p.name for p in chunks],
        "by_arm_mix": table,
        "compared": total,
        "identical": identical,
        "pass": total > 0 and identical == total,
        "wall_seconds": {"batch": batch_wall, "grid": grid_wall},
    }
    print(json.dumps(summary, indent=1, sort_keys=True))
    if args.write:
        (out / "summary.json").write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return 0 if summary["pass"] else 1


def parse_args(argv: Optional[Sequence[str]]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    run = sub.add_parser("run")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    run.add_argument("--mix", choices=("frozen", "scripted", "mixed"), required=True)
    run.add_argument("--worlds", type=int, default=8)
    run.add_argument("--offset", type=int, default=0)
    run.add_argument("--frames", type=int, default=None)
    run.add_argument("--threads", type=int, default=1)
    run.add_argument("--out", required=True)
    summ = sub.add_parser("summarize")
    summ.add_argument("--out", required=True)
    summ.add_argument("--write", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    return cmd_run(args) if args.cmd == "run" else cmd_summarize(args)


if __name__ == "__main__":
    raise SystemExit(main())
