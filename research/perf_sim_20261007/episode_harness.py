"""Gate-world evaluation episodes for the perf-sim byte-identity harness, profiler and bench.

Plays exactly what the strict gates play (``research/frp3_strict_20261005/spec.py``
``episode_runner``): ``tournament_eval.rollout(hero_safety_veto=True)`` with the install routed
through ``dev_screen.hero_veto_installer(install_space_and_head_veto(hero, 8.0))``, the pinned
deployment config, the ``promotion-v2-watch-rect`` profile at H5000, and the strict balanced
rosters of ``dev_screen._design_rows`` (frozen / scripted / mixed). The hero is the served
frp3-s12 checkpoint (``CHAMPION`` with ``--hero champion``).

It imports ``src`` and ``research`` from ``--root`` (a source tree: this worktree, or a frozen
``git archive`` of main), so the SAME driver compares two code versions.

Modes:

* ``identity``: per episode, a per-frame digest stream of the whole world (every snake's
  segments, direction, length, alive, boost state, the food list, the frame counter, the Python
  and NumPy global RNG states) and of every ApexNetwork forward output (raw float32 bytes per
  row), plus the canonical-JSON rollout record and the veto diagnostics. Written as JSONL.
* ``time``: wall time per episode, no hooks (``--profile FILE`` wraps the loop in cProfile).

Seeds come from their own namespace (``perf-sim-identity/v1``) so nothing here overlaps a study.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

CHECKPOINT_DIR = Path(
    os.environ.get("SNAKE_PERF_CKPT_DIR", "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
)


def checkpoint_path(name: str, sha: str) -> str:
    """``CHECKPOINT_DIR/name``, or ``CHECKPOINT_DIR/<sha>.pth`` (the fan-out pod layout)."""
    by_name = CHECKPOINT_DIR / name
    return str(by_name if by_name.exists() else CHECKPOINT_DIR / f"{sha}.pth")


FRP3_S12 = (
    "frp3_m3_s12_u60000_20261005.pth",
    "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
)
DOMAIN = "perf-sim-identity/v1"
VETO_LAMBDA = 8.0


def _bootstrap(root: Path) -> None:
    root = root.resolve()
    sys.path.insert(0, str(root))
    os.chdir(root)
    os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")


def world_seeds(count: int, namespace: str = "eval") -> List[int]:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.screen_seeds(count, DOMAIN, namespace)


def setup(threads: int = 1, frames: int = 5000) -> Dict[str, Any]:
    """Pinned config + profile + roster lookup, exactly as the frp3 strict ``_context``."""
    import torch

    from research.apex_safety_20260926 import dev_screen
    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    torch.set_num_threads(int(threads))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    if dev_screen.sha256_file(dev_screen.DEFAULT_CONFIG) != (
        "4146baa3a06102b8afd627b1fba8384e9a2f47aaac4a9bc96292c3eb71715aa5"
    ):
        raise SystemExit("deployment config drift")
    load_and_initialize_config(str(dev_screen.DEFAULT_CONFIG))
    profile = evaluation_profile_for_name(dev_screen.PROFILE_NAME, dev_screen.HORIZON)
    if profile.digest != dev_screen.PROFILE_DIGEST:
        raise SystemExit("profile digest drift")
    paths = {sha: checkpoint_path(name, sha) for name, sha in dev_screen.POOL}
    paths[FRP3_S12[1]] = checkpoint_path(*FRP3_S12)
    return {"profile": profile, "lookup": dev_screen.agent_lookup(paths)}


def rows_for(seeds: Sequence[int], mixes: Sequence[str]) -> List[Dict[str, Any]]:
    from research.apex_safety_20260926 import dev_screen

    rows = [row for row in dev_screen._design_rows(list(seeds)) if row["mix"] in mixes]
    # Round-robin mix order per world, like the strict gates.
    order = {seed: i for i, seed in enumerate(seeds)}
    rows.sort(key=lambda r: (order[r["world_seed"]], list(mixes).index(r["mix"])))
    return rows


def play(row: Mapping[str, Any], ctx: Mapping[str, Any], hero: str, frames: int) -> Dict[str, Any]:
    from research.apex_safety_20260926 import dev_screen
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.scripts.tournament_eval import rollout

    hero_sha = FRP3_S12[1] if hero == "frp3" else dev_screen.CHAMPION[1]
    lookup = ctx["lookup"]
    opponents = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    with dev_screen.hero_veto_installer(
        lambda h: install_space_and_head_veto(h, VETO_LAMBDA)
    ) as installed:
        record = rollout(
            lookup[hero_sha],
            opponents,
            frames,
            int(row["world_seed"]),
            # Smoke horizons (< H5000) use the legacy profile-free path.
            profile=ctx["profile"] if frames == dev_screen.HORIZON else None,
            world_identity=_expected_world_identity(row) if frames == dev_screen.HORIZON else None,
            mix_id=row["mix"],
            hero_safety_veto=True,
        )
    return {"record": record, "veto_diagnostics": installed[0].diagnostics_record()}


# ----------------------------------------------------------------------------- tracing


class Tracer:
    """Per-frame world digests + network-output digests (identity mode only)."""

    def __init__(self) -> None:
        self.frames: List[str] = []
        self.q_rows: List[str] = []
        self._q = hashlib.blake2b(digest_size=16)
        self.q_count = 0
        self.non_int_coords = 0

    def install(self) -> None:
        import numpy as np

        from src.game.game_state import GameState
        from src.model.apex_network import ApexNetwork

        tracer = self
        original_update = GameState.update
        original_forward = ApexNetwork.forward

        def update(gs, *args, **kwargs):
            out = original_update(gs, *args, **kwargs)
            tracer.frames.append(world_digest(gs))
            tracer.non_int_coords += non_int_coordinates(gs)
            return out

        def forward(net, x, *args, **kwargs):
            out = original_forward(net, x, *args, **kwargs)
            q = out[0] if isinstance(out, tuple) else out
            arr = q.detach().cpu().numpy().astype(np.float32, copy=False)
            arr = arr.reshape(-1, arr.shape[-1])
            inp = x.detach().cpu().numpy().astype(np.float32, copy=False).reshape(arr.shape[0], -1)
            for row_in, row_out in zip(inp, arr):
                tracer._q.update(row_in.tobytes())
                tracer._q.update(row_out.tobytes())
                tracer.q_count += 1
            return out

        GameState.update = update
        ApexNetwork.forward = forward

    def take(self) -> Dict[str, Any]:
        chain = hashlib.blake2b(digest_size=16)
        for d in self.frames:
            chain.update(d.encode())
        result = {
            "frames": len(self.frames),
            "world_chain": chain.hexdigest(),
            "frame_digests": list(self.frames),
            "q_digest": self._q.hexdigest(),
            "q_rows": self.q_count,
            "non_int_coords": self.non_int_coords,
        }
        self.frames = []
        self._q = hashlib.blake2b(digest_size=16)
        self.q_count = 0
        self.non_int_coords = 0
        return result


def world_digest(gs: Any) -> str:
    import random

    import numpy as np

    h = hashlib.blake2b(digest_size=12)
    parts: List[Any] = [int(gs.frame)]
    for s in gs.snakes:
        parts.append(
            (
                type(s).__name__,
                int(s.id),
                bool(s.is_alive),
                tuple(tuple(seg) for seg in s.segments),
                tuple(s.direction),
                int(s.length),
                bool(getattr(s, "is_boosting", False)),
                int(getattr(s, "boost_frames", 0)),
                getattr(s, "_pre_collision_action", None),
            )
        )
    parts.append(tuple(tuple(f) for f in gs.food_manager.food))
    h.update(repr(parts).encode())
    h.update(repr(random.getstate()).encode())
    np_state = np.random.get_state()
    h.update(np_state[1].tobytes())
    h.update(repr(np_state[2:]).encode())
    return h.hexdigest()


def non_int_coordinates(gs: Any) -> int:
    """Count snake / food coordinates that are not Python ints (the fast paths assume ints)."""
    bad = 0
    for s in gs.snakes:
        for x, y in s.segments:
            bad += (type(x) is not int) + (type(y) is not int)
    for x, y in gs.food_manager.food:
        bad += (type(x) is not int) + (type(y) is not int)
    return bad


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=True, default=repr)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--mode", choices=("identity", "time"), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--worlds", type=int, default=10, help="worlds (each x len(mixes))")
    parser.add_argument("--world-offset", type=int, default=0)
    parser.add_argument("--namespace", default="eval")
    parser.add_argument("--mixes", default="frozen,scripted,mixed")
    parser.add_argument("--frames", type=int, default=5000)
    parser.add_argument("--hero", choices=("frp3", "champion"), default="frp3")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--profile", type=Path, default=None, help="cProfile output (time mode)")
    args = parser.parse_args(argv)

    out = args.out.resolve()
    prof_path = args.profile.resolve() if args.profile else None
    _bootstrap(args.root)
    ctx = setup(args.threads, args.frames)
    mixes = [m for m in args.mixes.split(",") if m]
    seeds = world_seeds(args.world_offset + args.worlds, args.namespace)[args.world_offset :]
    rows = rows_for(seeds, mixes)
    tracer = Tracer() if args.mode == "identity" else None
    if tracer:
        tracer.install()
    profiler = None
    if prof_path is not None:
        import cProfile

        profiler = cProfile.Profile()
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as stream:
        for row in rows:
            started = time.perf_counter()
            cpu0 = time.process_time()
            if profiler:
                profiler.enable()
            result = play(row, ctx, args.hero, args.frames)
            if profiler:
                profiler.disable()
            entry = {
                "mix": row["mix"],
                "world_seed": int(row["world_seed"]),
                "wall_seconds": time.perf_counter() - started,
                "cpu_seconds": time.process_time() - cpu0,
            }
            if tracer:
                entry["trace"] = tracer.take()
                entry["record_canonical"] = canonical(result["record"])
                entry["veto_diagnostics_canonical"] = canonical(result["veto_diagnostics"])
            else:
                rec = result["record"]
                entry["mass_integral"] = rec.get("mass_integral")
            stream.write(json.dumps(entry, sort_keys=True) + "\n")
            stream.flush()
            print(
                f"{row['mix']:9s} seed={row['world_seed']:>10d} wall={entry['wall_seconds']:.2f}s",
                file=sys.stderr,
                flush=True,
            )
    if profiler:
        profiler.dump_stats(str(prof_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
