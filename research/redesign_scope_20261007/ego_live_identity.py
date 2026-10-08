#!/usr/bin/env python3
"""DEVELOPMENT-ONLY M1(iii) check: ego2s from a real live GameState == ego2s from GridBatchSim.

Not a screen and not gate evidence. Per world, it plays the same gate-world H5000 episode
(pinned deployment config, ``promotion-v2-watch-rect``, strict balanced roster, frp3-s12 +
v8 lambda 8 hero) twice:

* **live**: ``tournament_eval.rollout`` on a real ``GameState`` (the perf-sim harness's
  ``play``). At every decision point (``GameState.update`` step 6, after respawns and before
  any snake acts: the ``prepare_frame`` hook serving policies use) the observation is built
  with :func:`~src.simd_env.ego_live_adapter.game_state_to_ego_view`;
* **sim**: ``run_simd_eval(sim_engine="grid", vector61=True, hero_safety_veto="v8")``. At every
  decision point (``step_with_policy``'s prepared state) the observation is built from
  :meth:`GridBatchSim.ego_view`.

For every frame it digests, per world, the alive vector and the local planes, global planes
and scalars (raw bytes) of EVERY living snake, from both featurizer backends, and requires
the live and sim digest streams to be equal frame by frame. One field is excluded: the
own-hunger scalar of SCRIPTED rows, because the live ``ScriptedSnake`` never advances
``frames_since_food`` (it is always 0 there) while the sim advances it for every slot. A
scripted anchor never reads an ego2s observation, so nothing a policy reads is excluded.
The two episode records must also be equal (the worlds are the same world).

  nice -n 10 ./venv/bin/python research/redesign_scope_20261007/ego_live_identity.py \\
      --worlds 1 --mixes frozen,scripted,mixed \\
      --out research/redesign_scope_20261007/results/ego_live_identity.json
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SCHEMA = "redesign-ego2s-live-identity/v1"
BACKENDS = ("numpy", "numba")


HUNGER = 5  # SCALAR_NAMES index of the own-hunger scalar


def obs_digest(view: Any, env: int, scripted_slots: Sequence[int] = ()) -> Dict[str, Any]:
    """Digest of env ``env``'s ego2s observation for every living snake, both backends.

    The live ``ScriptedSnake`` never advances ``frames_since_food`` (the sim does, for
    every slot), so the own-hunger scalar of a scripted row is excluded from the
    digest. A scripted snake never consumes an ego2s observation, and a row's hunger
    scalar appears only in that row's own observation, so every observation a policy
    can read is still compared in full.
    """
    from src.simd_env.ego_raster import build_ego_raster

    alive = np.asarray(view.alive[env], dtype=bool)
    rows = np.array([[env, s] for s in np.flatnonzero(alive)], dtype=np.int64).reshape(-1, 2)
    scripted = np.isin(rows[:, 1], np.asarray(list(scripted_slots), dtype=np.int64))
    h = hashlib.blake2b(digest_size=16)
    h.update(alive.tobytes())
    per_backend = {}
    for backend in BACKENDS:
        obs = build_ego_raster(view, backend=backend, rows=rows)
        scalars = obs["scalars"].copy()
        scalars[scripted, HUNGER] = 0.0
        obs["scalars"] = scalars
        hb = hashlib.blake2b(digest_size=16)
        for key in ("local", "global", "scalars"):
            hb.update(key.encode())
            hb.update(np.ascontiguousarray(obs[key]).tobytes())
        per_backend[backend] = hb.hexdigest()
        h.update(per_backend[backend].encode())
    stats = {
        "rows": int(len(rows)),
        "scripted_rows": int(scripted.sum()),
        "max_length": int(view.length[env][alive].max()) if alive.any() else 0,
    }
    return {"digest": h.hexdigest(), "backends": per_backend, "stats": stats}


def scripted_slots(row: Dict[str, Any], lookup: Dict[str, Any]) -> List[int]:
    """Arena slots of the row's scripted anchors (the hero, slot 0, is never one)."""
    return [
        index
        for index, slot in enumerate(row["slots"], start=1)
        if lookup[slot["member_sha256"]][0] == "scripted"
    ]


def run_live(row: Dict[str, Any], ctx: Dict[str, Any]) -> Dict[str, Any]:
    """One live gate episode; returns the record and the per-frame obs digests."""
    from research.perf_sim_20261007 import episode_harness as eh
    from src.scripts import tournament_eval as te
    from src.simd_env.ego_live_adapter import game_state_to_ego_view

    frames: List[Dict[str, Any]] = []
    skip = scripted_slots(row, ctx["lookup"])
    original = te.configure_eval_game_state

    def configure(gs):
        out = original(gs)
        policy = gs.snakes[0].policy

        def prepare_frame() -> None:
            frames.append(obs_digest(game_state_to_ego_view(gs), 0, skip))

        if getattr(policy, "prepare_frame", None) is not None:
            raise RuntimeError("hero policy already has a prepare_frame hook")
        policy.prepare_frame = prepare_frame
        return out

    te.configure_eval_game_state = configure
    started = time.perf_counter()
    try:
        result = eh.play(row, ctx, "frp3", 5000)
    finally:
        te.configure_eval_game_state = original
    return {"result": result, "frames": frames, "wall": time.perf_counter() - started}


def run_sim(row: Dict[str, Any], ctx: Dict[str, Any], prof_ctx: Dict[str, Any]) -> Dict[str, Any]:
    """The same world on GridBatchSim (v8 hero); per-frame obs digests at decision time."""
    from src.simd_env import eval_engine as ee

    sim_class = ee._terminal_hero_sim_class("grid")
    frames: List[Dict[str, Any]] = []
    skip = scripted_slots(row, prof_ctx["lookup"])
    original = sim_class.step_with_policy

    def step_with_policy(sim, policy, *args, **kwargs):
        def wrapped(prepared):
            frames.append(obs_digest(prepared.ego_view(), 0, skip))
            return policy(prepared)

        return original(sim, wrapped, *args, **kwargs)

    sim_class.step_with_policy = step_with_policy
    lookup = prof_ctx["lookup"]
    roster = [lookup[slot["member_sha256"]] for slot in row["slots"]]
    seed = int(row["world_seed"])
    started = time.perf_counter()
    try:
        records = ee.run_simd_eval(
            lookup[gid.FRP3_S12[1]],
            roster,
            5000,
            [seed],
            profile=prof_ctx["profile"],
            opponent_specs_by_world={seed: roster},
            mix_id=row["mix"],
            vector61=True,
            vector61_forward="rowwise",
            sim_engine="grid",
            **gid.ARMS["v8"],
        )
    finally:
        sim_class.step_with_policy = original
    return {"record": records[0], "frames": frames, "wall": time.perf_counter() - started}


def compare(live: Dict[str, Any], sim: Dict[str, Any]) -> Dict[str, Any]:
    lf, sf = live["frames"], sim["frames"]
    first = next((i for i, (a, b) in enumerate(zip(lf, sf)) if a["digest"] != b["digest"]), None)
    if first is None and len(lf) != len(sf):
        first = min(len(lf), len(sf))
    live_record = gid.normalize(live["result"]["record"])
    sim_record = {
        k: v
        for k, v in gid.normalize(sim["record"]).items()
        if k not in ("world_runtime_spec", "world_runtime_spec_digest", "vector61_policy")
    }
    sim_diag = gid.strip_timing(sim_record.pop("veto_diagnostics", None))
    live_diag = gid.strip_timing(gid.normalize(live["result"]["veto_diagnostics"]))
    record_diffs = gid.diff_paths(live_record, sim_record)
    backend_agree = all(f["backends"]["numpy"] == f["backends"]["numba"] for f in lf + sf)
    return {
        "frames_live": len(lf),
        "frames_sim": len(sf),
        "first_differing_frame": first,
        "obs_identical": first is None,
        "backends_agree": backend_agree,
        "record_identical": not record_diffs,
        "veto_diagnostics_identical": live_diag == sim_diag,
        "record_differences": [{"path": p, "live": a, "sim": b} for p, a, b in record_diffs[:10]],
        "rows_compared": int(sum(f["stats"]["rows"] for f in lf)),
        "scripted_rows_hunger_excluded": int(sum(f["stats"]["scripted_rows"] for f in lf)),
        "max_length_seen": int(max([f["stats"]["max_length"] for f in lf] + [0])),
        "live_wall_seconds": live["wall"],
        "sim_wall_seconds": sim["wall"],
        "mass_integral": live_record.get("mass_integral"),
        "chain_live": gid.chain([f["digest"] for f in lf]),
        "chain_sim": gid.chain([f["digest"] for f in sf]),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--worlds", type=int, default=1)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--mixes", default="frozen,scripted,mixed")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.out.exists():
        print(f"refusing: {args.out} exists (create-only)", file=sys.stderr)
        return 2

    from research.perf_sim_20261007 import episode_harness as eh

    prof_ctx = gid._context(1)  # pinned config + profile + verified checkpoint hashes
    ctx = {"profile": prof_ctx["profile"], "lookup": prof_ctx["lookup"]}
    mixes = [m for m in args.mixes.split(",") if m]
    seeds = gid.world_seeds(args.worlds, args.offset)
    rows = eh.rows_for(seeds, mixes)
    results = []
    for row in rows:
        live = run_live(row, ctx)
        sim = run_sim(row, ctx, prof_ctx)
        cmp = compare(live, sim)
        cmp.update({"mix": row["mix"], "world_seed": int(row["world_seed"])})
        results.append(cmp)
        print(json.dumps({k: v for k, v in cmp.items() if k != "record_differences"}), flush=True)
    import numba

    passed = all(
        r["obs_identical"]
        and r["backends_agree"]
        and r["record_identical"]
        and r["veto_diagnostics_identical"]
        for r in results
    )
    summary = {
        "schema_version": SCHEMA,
        "authority": gid.AUTHORITY,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "hero": {"name": gid.FRP3_S12[0], "sha256": gid.FRP3_S12[1], "veto": gid.ARMS["v8"]},
        "versions": {"numba": numba.__version__, "numpy": np.__version__},
        "frames_compared": int(sum(r["frames_live"] for r in results)),
        "rows_compared": int(sum(r["rows_compared"] for r in results)),
        "pass": passed,
        "worlds": results,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x", encoding="utf-8") as stream:
        json.dump(summary, stream, sort_keys=True, indent=1)
        stream.write("\n")
    print(json.dumps({k: summary[k] for k in ("frames_compared", "rows_compared", "pass")}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
