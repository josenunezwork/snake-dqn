#!/usr/bin/env python3
"""M2 distillation data: frp3-s12 + v8 teacher labels on ego2s observations (doc 15.1).

One invocation = one chunk: ``E`` gate-world worlds (pinned deployment config,
``promotion-v2-watch-rect`` world, Watch food branch, everyone respawns, strict balanced
roster of one mix) on :class:`GridBatchSim`, run for ``--frames`` frames. At every hero
(slot 0) decision it records:

* the ego2s observation (numba backend, hero rows): local / global planes, scalars;
* the teacher's raw Q(6) BEFORE the veto (frp3-s12, ``Vector61SimdPolicy`` batched forward);
* ``a_base`` (teacher masked argmax), ``a_v8`` (after the v8 veto, lambda 8), ``a_taken``;
* the resolved mask and the teacher's (advisory + fallback) mask, length, frame, seed, mix.

DAgger (``--student CKPT --beta B``): the action taken is the student's (masked argmax over
the resolved mask, forward on MPS) with probability ``1 - B`` per decision; the teacher
still labels. Without ``--student`` the teacher + v8 acts (round 0).

Shards: ``<out>/<round>/chunk-<k>-part-<j>.npz`` (``np.savez_compressed``, 10k samples) and
``chunk-<k>.json`` (provenance). Create-only. Never gate evidence. Seeds:
``ni_spec.distill_seeds(E, offset=chunk * E, round_index=round)``.
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m2_20261008 import ni_spec  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SCHEMA = "redesign-m2-distill-shard/v1"
PART = 10000
FIELDS = (
    "local",
    "global",
    "scalars",
    "q_teacher",
    "a_base",
    "a_v8",
    "a_taken",
    "mask_resolved",
    "mask_teacher",
    "length",
    "frame",
    "world_seed",
)
MIX_CODE = {"frozen": 0, "scripted": 1, "mixed": 2}


def make_teacher_class():
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    class TeacherPolicy(Vector61SimdPolicy):
        """frp3-s12 + v8 that exposes the raw (pre-mask, pre-veto) Q of its last call."""

        last_q: np.ndarray = np.zeros((0, 6), dtype=np.float32)

        def q_values(self, states):
            q = super().q_values(states)
            self.last_q = q.detach().cpu().numpy().astype(np.float32)
            return q

    return TeacherPolicy


class Writer:
    def __init__(self, out: Path, chunk: int, meta: Dict[str, Any], fields=FIELDS) -> None:
        self.out, self.chunk, self.meta = out, chunk, meta
        self.fields = tuple(fields)
        self.buf: Dict[str, List[np.ndarray]] = {k: [] for k in self.fields}
        self.n = 0
        self.parts: List[Dict[str, Any]] = []

    def add(self, **arrays: np.ndarray) -> None:
        for k in self.fields:
            self.buf[k].append(arrays[k])
        self.n += len(arrays["a_taken"])
        if self.n >= PART:
            self.flush()

    def flush(self) -> None:
        if not self.n:
            return
        data = {k: np.concatenate(v) for k, v in self.buf.items()}
        data["mix"] = np.full(self.n, MIX_CODE[self.meta["mix"]], dtype=np.int8)
        data["round"] = np.full(self.n, self.meta["round"], dtype=np.int8)
        path = self.out / f"chunk-{self.chunk:04d}-part-{len(self.parts):02d}.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **data)
        self.parts.append({"file": path.name, "samples": self.n, "bytes": path.stat().st_size})
        self.buf = {k: [] for k in self.fields}
        self.n = 0


def run_chunk(args: argparse.Namespace) -> Dict[str, Any]:
    import torch

    from src.core.world_runtime import WorldRuntimeSpec
    from src.simd_env import eval_engine as ee
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.vector61_policy import Vector61Runtime, Vector61SimdPolicy

    torch.set_num_threads(1)
    ctx = gid._context(1)
    profile = ctx["profile"]
    E = args.worlds
    mix = ("frozen", "scripted", "mixed")[args.chunk % 3] if args.mix == "cycle" else args.mix
    seeds = ni_spec.distill_seeds(E, offset=args.chunk * E, round_index=args.round)
    base = ee._config_from_game_config(6, 0.99, profile)
    cap = WorldRuntimeSpec.source_exact(profile).body_storage_capacity
    cfg = replace(base, num_envs=E, body_storage_capacity=cap)
    sim = GridBatchSim(cfg, seeds=seeds, train_mode=False, allow_respawn=True)

    lookup = ctx["lookup"]
    runtime = Vector61Runtime()
    teacher = make_teacher_class()(
        lookup[ni_spec.FRP3_S12[1]][1],
        runtime,
        veto_slots=(0,),
        forward="batched",
        veto_variant="v8",
        veto_lambda=8.0,
    )
    rows = {int(r["world_seed"]): r for r in ctx["ds"]._design_rows(seeds) if r["mix"] == mix}
    groups: Dict[Any, List[tuple]] = {}
    for e, seed in enumerate(seeds):
        for slot, member in enumerate(rows[int(seed)]["slots"], start=1):
            groups.setdefault(lookup[member["member_sha256"]], []).append((e, slot))
    policies: Dict[Any, Any] = {}
    controlled = np.zeros((E, 6), dtype=bool)
    controlled[:, 0] = True
    for spec, cells in groups.items():
        if spec[0] == "scripted":
            policies[spec] = FastProfileAnchorSimdPolicy(spec[1], seeds)
        else:
            policies[spec] = Vector61SimdPolicy(spec[1], runtime, forward="batched")
            for e, s in cells:
                controlled[e, s] = True
    groups_arr = {k: np.array(v, dtype=np.int64) for k, v in groups.items()}
    runtime.bind(sim, controlled)

    student = None
    if args.student:
        from src.model.ego2s_network import load_ego2s_checkpoint, obs_tensors

        device = torch.device(args.student_device)
        student = load_ego2s_checkpoint(args.student, device)
    rng = np.random.default_rng([args.round, args.chunk, 77])

    out = Path(args.out) / f"round-{args.round}"
    out.mkdir(parents=True, exist_ok=True)
    manifest = out / f"chunk-{args.chunk:04d}.json"
    if manifest.exists():
        raise SystemExit(f"refusing: {manifest} exists (create-only)")
    meta = {"mix": mix, "round": args.round}
    fields = FIELDS + (("state61",) if args.record_state61 else ())
    writer = Writer(out, args.chunk, meta, fields)
    ego_cfg = EgoRasterConfig(version=args.obs_version)
    stats = {"decisions": 0, "veto_override": 0, "student_acted": 0, "hero_deaths": 0}

    def choose(prepared):
        masks = prepared.get_resolved_action_mask()
        alive = prepared.get_alive()
        actions = np.ones((E, prepared.S), dtype=np.int64)
        runtime.prepare(prepared)
        hero = np.argwhere(alive[:, :1])
        if len(hero):
            obs = build_ego_raster(prepared, ego_cfg, backend="numba", rows=hero)
            res = masks[hero[:, 0], 0]
            a_v8 = teacher.actions(res, prepared, hero)
            q = teacher.last_q
            states61, tmask = runtime.selection(hero)
            tmask = tmask.copy()
            tmask[~tmask.any(axis=1), :3] = True  # the teacher's live fallback
            a_base = np.where(tmask, q, -np.inf).argmax(axis=1)
            taken = a_v8.copy()
            if student is not None:
                with torch.no_grad():
                    sq = student(*obs_tensors(obs, next(student.parameters()).device))
                    sq = sq.cpu().numpy()
                m = res.copy()
                m[~m.any(axis=1), :3] = True
                a_s = np.where(m, sq, -np.inf).argmax(axis=1)
                use = rng.random(len(hero)) >= args.beta
                taken = np.where(use, a_s, a_v8)
                stats["student_acted"] += int(use.sum())
            actions[hero[:, 0], 0] = taken
            writer.add(
                local=obs["local"],
                **{"global": obs["global"]},
                scalars=obs["scalars"],
                q_teacher=q,
                a_base=a_base.astype(np.int8),
                a_v8=a_v8.astype(np.int8),
                a_taken=taken.astype(np.int8),
                mask_resolved=res,
                mask_teacher=tmask,
                length=prepared.length[hero[:, 0], 0].astype(np.int32),
                frame=prepared.frame[hero[:, 0]].astype(np.int32),
                world_seed=np.array([seeds[e] for e in hero[:, 0]], dtype=np.uint32),
                state61=states61.astype(np.float32),
            )
            stats["decisions"] += len(hero)
            stats["veto_override"] += int((a_v8 != a_base).sum())
        for spec, cells in groups_arr.items():
            live = cells[alive[cells[:, 0], cells[:, 1]]]
            if len(live):
                m = masks[live[:, 0], live[:, 1]]
                actions[live[:, 0], live[:, 1]] = policies[spec].actions(m, prepared, live)
        return actions

    started = time.perf_counter()
    for _ in range(args.frames):
        sim.step_with_policy(choose)
        runtime.observe_step(sim)
        stats["hero_deaths"] += int(sim.get_done()[:, 0].sum())
    writer.flush()
    wall = time.perf_counter() - started
    import numba

    summary = {
        "schema_version": SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "round": args.round,
        "chunk": args.chunk,
        "mix": mix,
        "worlds": E,
        "frames": args.frames,
        "world_seeds": [int(s) for s in seeds],
        "teacher": {
            "sha256": ni_spec.FRP3_S12[1],
            "veto": "v8",
            "lambda": 8.0,
            "forward": "batched",
        },
        "student": (
            None
            if not args.student
            else {"path": str(args.student), "sha256": gid_sha(args.student), "beta": args.beta}
        ),
        "versions": {"numba": numba.__version__, "numpy": np.__version__},
        "obs_version": args.obs_version,
        "fields": list(fields),
        "wall_seconds": wall,
        "decisions_per_s": stats["decisions"] / wall,
        "stats": stats,
        "parts": writer.parts,
        "max_length": int(sim.length.max()),
    }
    manifest.write_text(json.dumps(summary, indent=1, sort_keys=True) + "\n")
    return summary


def gid_sha(path: str) -> str:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.sha256_file(Path(path))


def guard_ok() -> List[str]:
    """AC power, lid open, thermal guard clear (empty list = ok)."""
    import subprocess

    from research.apex_safety_20260926 import dev_screen
    from research.compute.thermal_guard import (
        parse_pmset_therm,
        read_pmset_therm,
        thermal_reasons,
    )

    problems = []
    if not dev_screen.on_ac_power():
        problems.append("not on AC power")
    lid = subprocess.run(
        ["ioreg", "-r", "-k", "AppleClamshellState"], capture_output=True, text=True
    ).stdout
    if '"AppleClamshellState" = Yes' in lid:
        problems.append("lid closed")
    reading = read_pmset_therm()
    if reading.get("gated") and reading.get("available"):
        problems += thermal_reasons(parse_pmset_therm(reading["text"]))
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--round", type=int, required=True)
    ap.add_argument("--chunk", type=int, required=True)
    ap.add_argument("--worlds", type=int, default=16)
    ap.add_argument("--frames", type=int, default=5000)
    ap.add_argument("--mix", default="cycle", choices=("cycle", "frozen", "scripted", "mixed"))
    ap.add_argument("--student", default=None)
    ap.add_argument("--student-device", default="mps")
    ap.add_argument("--beta", type=float, default=1.0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-version", choices=("draft", "b"), default="draft")
    ap.add_argument("--record-state61", action="store_true", help="store the teacher's 61-D input")
    args = ap.parse_args()
    problems = guard_ok()
    if problems:
        print(json.dumps({"refused": problems}), flush=True)
        return 3
    summary = run_chunk(args)
    keys = ("round", "chunk", "mix", "wall_seconds", "decisions_per_s", "stats", "max_length")
    print(json.dumps({k: summary[k] for k in keys}), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
