#!/usr/bin/env python3
"""Evaluation-speed switches for dev probes / Phase R: identity, flip rate, timing (perf-eval).

DEVELOPMENT ONLY (not a screen, not gate evidence). Three subcommands around
``research/redesign_m2_20261008/dev_probe.py`` (``dev_probe.probe``, same engine, profile,
worlds and rosters as the probes ``probe_watch.py`` launches):

* ``identity``: run one probe twice in this process, the reference (vector61 featurizer
  ``python``, hero forward ``rowwise``) and a variant (``--variant numba``: compiled vector61
  featurizer; ``batched``: one batched ego2s hero forward per frame; ``numba+batched``), each
  with the per-frame whole-world digest recorder of ``grid_h5000_identity`` installed, and
  compare every world's record (type-strict, wall-clock keys stripped) and digest stream
  (first differing frame). Exit 1 on any difference.
* ``flips``: the reference run with a shadow batched hero forward: per decision, compares the
  batched Q with the rowwise Q actually used (bitwise-equal rows, max |dQ|, and the masked
  greedy argmax flip rate under the mask the policy selects with). The trajectory is the
  reference one (the shadow never acts).
* ``bench``: dev_probe CLI wall per (arm, mix, variant), one subprocess per cell, at most
  ``--procs`` (1-2) at a time. Numbers taken while other jobs run are NOISY.

Arms (``--arm``): ``student-nv`` (no veto, advisory mask: the M3-C/M4 primary probe),
``student-v8`` (the student inside the v8 veto), ``incumbent-v8`` (frp3-s12 + v8),
``incumbent-nv`` (frp3-s12, no veto).

  ./venv/bin/python research/perf_eval_20261013/eval_speed.py identity --arm student-nv \\
      --student S.pth --mix frozen --worlds 4 --frames 5000 --variant numba --out OUT.json
  ./venv/bin/python research/perf_eval_20261013/eval_speed.py flips --arm student-nv \\
      --student S.pth --mix frozen --worlds 4 --frames 2000 --out OUT.json
  ./venv/bin/python research/perf_eval_20261013/eval_speed.py bench --student S.pth \\
      --arms student-nv,incumbent-v8 --mixes frozen,scripted,mixed --variants default,numba \\
      --worlds 16 --frames 5000 --procs 1 --out OUT.json     # quiet machine
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import argparse  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m2_20261008 import dev_probe  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

DEV_PROBE = REPO / "research/redesign_m2_20261008/dev_probe.py"
ARMS: Dict[str, Dict[str, str]] = {
    "student-nv": {"arm": "student", "veto": "none"},
    "student-v8": {"arm": "student", "veto": "v8"},
    "incumbent-v8": {"arm": "baseline", "veto": "v8"},
    "incumbent-nv": {"arm": "baseline", "veto": "none"},
}
#: variant -> (vector61 backend, ego2s hero forward); "default" is the probe_watch reference.
VARIANTS: Dict[str, Dict[str, str]] = {
    "default": {"v61_backend": "python", "ego2s_forward": "rowwise"},
    "numba": {"v61_backend": "numba", "ego2s_forward": "rowwise"},
    "batched": {"v61_backend": "python", "ego2s_forward": "batched"},
    "numba+batched": {"v61_backend": "numba", "ego2s_forward": "batched"},
}


def probe_args(arm: str, variant: str, ns: argparse.Namespace) -> argparse.Namespace:
    """``dev_probe`` arguments for one (arm, variant) cell (probe_watch's settings)."""
    a, v = ARMS[arm], VARIANTS[variant]
    argv = [
        *("--arm", a["arm"], "--veto", a["veto"], "--mix", ns.mix),
        *("--worlds", str(ns.worlds), "--round-index", str(ns.round_index)),
        *("--ego2s-forward", v["ego2s_forward"], "--ego2s-mask", "advisory"),
        *("--v61-backend", v["v61_backend"], "--out", os.devnull),
    ]
    if a["arm"] == "student":
        if not ns.student:
            raise SystemExit(f"--arm {arm} needs --student")
        argv += ["--student", str(ns.student)]
    if ns.frames:
        argv += ["--frames", str(ns.frames)]
    return dev_probe.build_parser().parse_args(argv)


def run_recorded(args: argparse.Namespace, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """One probe with the per-frame world digest recorder installed."""
    from src.simd_env import eval_engine as ee

    sim_class = ee._terminal_hero_sim_class("grid")
    recorder = gid.DigestRecorder()
    original = recorder.install(sim_class)
    started = time.perf_counter()
    try:
        records, row = dev_probe.probe(args, ctx)
    finally:
        sim_class.step_with_policy = original
    return {
        "wall_seconds": time.perf_counter() - started,
        "records": [gid.normalize(r) for r in records],
        "frame_digests": recorder.per_env,
        "row": row,
    }


def compare(ref: Mapping[str, Any], var: Mapping[str, Any], seeds: Sequence[int]):
    """Per-world identity rows (``gid.compare_worlds``: records + digest streams)."""
    rows = gid.compare_worlds(ref, var, seeds)
    for r in rows:  # gid's labels are engine names; here they are reference / variant
        r["world_chain_reference"] = r.pop("world_chain_batch")
        r["world_chain_variant"] = r.pop("world_chain_grid")
        for d in r["record_differences"]:
            d["reference"] = d.pop("batch")
            d["variant"] = d.pop("grid")
    return rows


def _meta(ns: argparse.Namespace) -> Dict[str, Any]:
    import numba
    import torch

    return {
        "authority": "development-only (not a screen, not gate evidence)",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "host": platform.node(),
        "machine": platform.machine(),
        "load_avg": os.getloadavg(),
        "noisy": bool(getattr(ns, "noisy", True)),
        "versions": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "numba": numba.__version__,
        },
        "student": str(ns.student) if getattr(ns, "student", None) else None,
    }


def _write(path: Optional[Path], payload: Mapping[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, sort_keys=True, indent=1, allow_nan=False)
        stream.write("\n")


# ----------------------------------------------------------------------------- identity
def cmd_identity(ns: argparse.Namespace) -> int:
    import torch

    torch.set_num_threads(1)
    ctx = gid._context(1)
    ref_args = probe_args(ns.arm, "default", ns)
    var_args = probe_args(ns.arm, ns.variant, ns)
    seeds = dev_probe.probe_seeds(ns.worlds, ns.round_index)
    ref = run_recorded(ref_args, ctx)
    var = run_recorded(var_args, ctx)
    rows = compare(ref, var, seeds)
    identical = sum(r["identical"] for r in rows)
    payload = {
        **_meta(ns),
        "schema": "perf-eval-identity/v1",
        "arm": ns.arm,
        "mix": ns.mix,
        "variant": ns.variant,
        "frames": ns.frames or dev_probe.ni_spec.HORIZON,
        "world_seeds": [int(s) for s in seeds],
        "wall_seconds_with_digests": {
            "reference": ref["wall_seconds"],
            "variant": var["wall_seconds"],
        },
        "probe_wall_seconds": {"reference": ref["row"]["wall"], "variant": var["row"]["wall"]},
        "rows": {"reference": ref["row"], "variant": var["row"]},
        "worlds": rows,
        "identical": identical,
        "compared": len(rows),
    }
    _write(ns.out, payload)
    print(
        json.dumps(
            {
                "arm": ns.arm,
                "mix": ns.mix,
                "variant": ns.variant,
                "identical": identical,
                "compared": len(rows),
                "first_differing_frames": [r["first_differing_frame"] for r in rows],
                "probe_wall_seconds": payload["probe_wall_seconds"],
                "deaths": list(ref["row"]["deaths"].values()),
            }
        ),
        flush=True,
    )
    return 0 if identical == len(rows) else 1


# ----------------------------------------------------------------------------- flips
class FlipShadow:
    """Shadow batched hero forward: compares it with the rowwise Q the policy acts on."""

    def __init__(self) -> None:
        self.pending: Optional[tuple] = None
        self.rows = 0
        self.bit_equal_rows = 0
        self.max_abs_dq = 0.0
        self.unmasked_flips = 0
        self.masked_flips = 0
        self.masked_rows = 0
        self.min_masked_gap: Optional[float] = None
        self.flip_gaps: List[float] = []

    def forward_q(self, original):
        def wrapped(net, obs, forward="batched"):
            q_r = original(net, obs, "rowwise")
            q_b = original(net, obs, "batched")
            self.pending = (q_r.numpy().copy(), q_b.numpy().copy())
            return q_r if forward == "rowwise" else q_b

        return wrapped

    def observe(self, mask: np.ndarray, float64: bool) -> None:
        """Account the last forward's rows under the policy's selection ``mask``."""
        if self.pending is None:
            return
        q_r, q_b = self.pending
        self.pending = None
        n = len(q_r)
        self.rows += n
        self.bit_equal_rows += int((q_r.view(np.uint32) == q_b.view(np.uint32)).all(axis=1).sum())
        if n:
            self.max_abs_dq = max(self.max_abs_dq, float(np.abs(q_r - q_b).max()))
        self.unmasked_flips += int((q_r.argmax(axis=1) != q_b.argmax(axis=1)).sum())
        m = np.asarray(mask, dtype=bool).copy()
        m[~m.any(axis=1), :3] = True
        dt = np.float64 if float64 else np.float32
        mr = np.where(m, q_r.astype(dt), -np.inf)
        mb = np.where(m, q_b.astype(dt), -np.inf)
        ar, ab = mr.argmax(axis=1), mb.argmax(axis=1)
        self.masked_rows += n
        flips = ar != ab
        self.masked_flips += int(flips.sum())
        srt = np.sort(mr, axis=1)
        gaps = srt[:, -1] - srt[:, -2]
        finite = np.isfinite(gaps)
        if finite.any():
            g = float(gaps[finite].min())
            self.min_masked_gap = g if self.min_masked_gap is None else min(g, self.min_masked_gap)
        self.flip_gaps += [float(x) for x in gaps[flips]]

    def summary(self) -> Dict[str, Any]:
        return {
            "rows": self.rows,
            "bit_equal_rows": self.bit_equal_rows,
            "bit_equal_fraction": self.bit_equal_rows / self.rows if self.rows else None,
            "max_abs_dq": self.max_abs_dq,
            "unmasked_argmax_flips": self.unmasked_flips,
            "masked_argmax_flips": self.masked_flips,
            "masked_flip_rate": self.masked_flips / self.masked_rows if self.masked_rows else None,
            "min_masked_top2_gap": self.min_masked_gap,
            "flip_top2_gaps": self.flip_gaps[:50],
        }


def cmd_flips(ns: argparse.Namespace) -> int:
    import torch

    from src.simd_env import ego2s_policy as ep

    torch.set_num_threads(1)
    if ARMS[ns.arm]["arm"] != "student":
        raise SystemExit("flips: the batched hero forward exists only for the ego2s student")
    ctx = gid._context(1)
    shadow = FlipShadow()
    orig_forward_q = ep.forward_q
    orig_nv, orig_v8 = ep.Ego2sSimdPolicy.actions, ep.Ego2sV8Policy.actions

    def nv_actions(self, masks, sim, slots):
        out = orig_nv(self, masks, sim, slots)
        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        if self.mask_mode == "advisory":
            m = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]]
            shadow.observe(m, float64=True)
        else:
            shadow.observe(np.asarray(masks, dtype=bool), float64=False)
        return out

    def v8_actions(self, masks, sim, slots):
        out = orig_v8(self, masks, sim, slots)
        _, exact = self.runtime.selection(np.asarray(slots, dtype=np.int64).reshape(-1, 2))
        shadow.observe(exact, float64=False)  # the base (pre-veto) masked argmax, float32
        return out

    ep.forward_q = shadow.forward_q(orig_forward_q)
    ep.Ego2sSimdPolicy.actions, ep.Ego2sV8Policy.actions = nv_actions, v8_actions
    try:
        started = time.perf_counter()
        _, row = dev_probe.probe(probe_args(ns.arm, "default", ns), ctx)
        wall = time.perf_counter() - started
    finally:
        ep.forward_q = orig_forward_q
        ep.Ego2sSimdPolicy.actions, ep.Ego2sV8Policy.actions = orig_nv, orig_v8
    payload = {
        **_meta(ns),
        "schema": "perf-eval-flips/v1",
        "arm": ns.arm,
        "mix": ns.mix,
        "frames": ns.frames or dev_probe.ni_spec.HORIZON,
        "worlds": ns.worlds,
        "wall_seconds": wall,
        "flips": shadow.summary(),
        "row": row,
    }
    _write(ns.out, payload)
    print(json.dumps({"arm": ns.arm, "mix": ns.mix, **shadow.summary()}), flush=True)
    return 0


# ----------------------------------------------------------------------------- bench
def cmd_bench(ns: argparse.Namespace) -> int:
    if not 1 <= ns.procs <= 2:
        raise SystemExit("compute policy: 1..2 processes")
    arms = [a for a in ns.arms.split(",") if a]
    mixes = [m for m in ns.mixes.split(",") if m]
    variants = [v for v in ns.variants.split(",") if v]
    tmp = Path(ns.tmp)
    tmp.mkdir(parents=True, exist_ok=True)
    cells = []
    for rep in range(ns.repeats):
        for arm in arms:
            for mix in mixes:
                for variant in variants:
                    if variant in ("batched", "numba+batched") and ARMS[arm]["arm"] != "student":
                        continue
                    cells.append((rep, arm, mix, variant))
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    env.pop("SNAKE_V61_BACKEND", None)
    results: List[Dict[str, Any]] = []
    running: List[tuple] = []
    todo = list(cells)

    def launch(cell):
        rep, arm, mix, variant = cell
        out = tmp / f"bench-{rep}-{arm}-{mix}-{variant}.jsonl"
        out.unlink(missing_ok=True)
        args = probe_args(arm, variant, argparse.Namespace(**{**vars(ns), "mix": mix}))
        argv = [sys.executable, str(DEV_PROBE)]
        for key in ("arm", "veto", "mix", "worlds", "round_index", "ego2s_forward"):
            argv += [f"--{key.replace('_', '-')}", str(getattr(args, key))]
        argv += ["--ego2s-mask", "advisory", "--v61-backend", args.v61_backend, "--out", str(out)]
        if args.student:
            argv += ["--student", args.student]
        if args.frames:
            argv += ["--frames", str(args.frames)]
        prefix = ["nice", "-n", str(ns.nice)] if ns.nice else []
        proc = subprocess.Popen(prefix + argv, cwd=REPO, env=env, stdout=subprocess.DEVNULL)
        return proc, cell, out, time.perf_counter()

    while todo or running:
        while todo and len(running) < ns.procs:
            running.append(launch(todo.pop(0)))
        time.sleep(0.5)
        still = []
        for proc, cell, out, t0 in running:
            if proc.poll() is None:
                still.append((proc, cell, out, t0))
                continue
            if proc.returncode != 0:
                raise SystemExit(f"bench cell {cell} failed ({proc.returncode})")
            row = json.loads(out.read_text().splitlines()[-1])
            rep, arm, mix, variant = cell
            results.append(
                {
                    "repeat": rep,
                    "arm": arm,
                    "mix": mix,
                    "variant": variant,
                    "probe_wall_seconds": row["wall"],
                    "process_wall_seconds": time.perf_counter() - t0,
                    "mass": row["mass"],
                    "deaths": row["deaths"],
                    "load_avg_after": os.getloadavg(),
                }
            )
            print(json.dumps(results[-1] | {"mass": None, "deaths": None}), flush=True)
        running = still
    summary = []
    for arm in arms:
        for mix in mixes:
            base = [
                r["probe_wall_seconds"]
                for r in results
                if (r["arm"], r["mix"], r["variant"]) == (arm, mix, "default")
            ]
            for variant in variants:
                walls = [
                    r["probe_wall_seconds"]
                    for r in results
                    if (r["arm"], r["mix"], r["variant"]) == (arm, mix, variant)
                ]
                if not walls:
                    continue
                same = all(
                    r["mass"] == ref["mass"] and r["deaths"] == ref["deaths"]
                    for r in results
                    for ref in results
                    if (r["arm"], r["mix"], r["variant"]) == (arm, mix, variant)
                    and (ref["arm"], ref["mix"], ref["variant"]) == (arm, mix, "default")
                )
                summary.append(
                    {
                        "arm": arm,
                        "mix": mix,
                        "variant": variant,
                        "probe_wall_seconds_median": float(np.median(walls)),
                        "speedup_vs_default": (
                            float(np.median(base) / np.median(walls)) if base else None
                        ),
                        "mass_deaths_equal_default": same if base else None,
                    }
                )
    payload = {
        **_meta(ns),
        "schema": "perf-eval-bench/v1",
        "worlds": ns.worlds,
        "frames": ns.frames or dev_probe.ni_spec.HORIZON,
        "procs": ns.procs,
        "nice": ns.nice,
        "summary": summary,
        "cells": results,
    }
    _write(ns.out, payload)
    for s in summary:
        print(json.dumps(s), flush=True)
    return 0


# ----------------------------------------------------------------------------- compare-records
#: Phase R record-entry fields that describe the run, not the episode (never compared).
RUN_FIELDS = ("binding", "commit", "unit_wall_seconds", "unit_id", "look", "tier")


def compare_record_dirs(a: Path, b: Path) -> Dict[str, Any]:
    """Phase R records of two roots/looks (same plan banks), file by file.

    Compares every entry field except :data:`RUN_FIELDS`, with ``record`` / ``prefix_h5000``
    normalized and wall-clock keys stripped (``gid.comparable_record``)."""
    fa = {p.name: p for p in Path(a).glob("*.json")}
    fb = {p.name: p for p in Path(b).glob("*.json")}
    rows = []
    for name in sorted(set(fa) & set(fb)):
        ea, eb = json.loads(fa[name].read_text()), json.loads(fb[name].read_text())
        ca = {k: gid.comparable_record(v) for k, v in ea.items() if k not in RUN_FIELDS}
        cb = {k: gid.comparable_record(v) for k, v in eb.items() if k not in RUN_FIELDS}
        diffs = gid.diff_paths(ca, cb)
        rows.append({"file": name, "identical": not diffs, "differences": diffs[:10]})
    return {
        "only_a": sorted(set(fa) - set(fb)),
        "only_b": sorted(set(fb) - set(fa)),
        "compared": len(rows),
        "identical": sum(r["identical"] for r in rows),
        "rows": rows,
    }


def cmd_compare_records(ns: argparse.Namespace) -> int:
    result = compare_record_dirs(ns.a, ns.b)
    _write(
        ns.out,
        {"schema": "perf-eval-compare-records/v1", "a": str(ns.a), "b": str(ns.b), **result},
    )
    print(json.dumps({k: result[k] for k in ("compared", "identical", "only_a", "only_b")}))
    ok = result["compared"] > 0 and result["identical"] == result["compared"]
    return 0 if ok and not result["only_a"] and not result["only_b"] else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--student", default=None)
    common.add_argument("--worlds", type=int, default=4)
    common.add_argument("--frames", type=int, default=None, help="default: H5000")
    common.add_argument("--round-index", type=int, default=52)
    common.add_argument("--out", type=Path, default=None)
    common.add_argument(
        "--quiet-machine",
        dest="noisy",
        action="store_false",
        help="label the results as taken on a quiet machine (default: noisy)",
    )
    p = sub.add_parser("identity", parents=[common])
    p.add_argument("--arm", choices=sorted(ARMS), required=True)
    p.add_argument("--mix", choices=dev_probe.ni_spec.MIXES, required=True)
    p.add_argument("--variant", choices=[v for v in VARIANTS if v != "default"], required=True)
    f = sub.add_parser("flips", parents=[common])
    f.add_argument("--arm", choices=("student-nv", "student-v8"), required=True)
    f.add_argument("--mix", choices=dev_probe.ni_spec.MIXES, required=True)
    b = sub.add_parser("bench", parents=[common])
    b.add_argument("--arms", default="student-nv,incumbent-v8")
    b.add_argument("--mixes", default="frozen,scripted,mixed")
    b.add_argument("--variants", default="default,numba")
    b.add_argument("--repeats", type=int, default=1)
    b.add_argument("--procs", type=int, default=1)
    b.add_argument("--nice", type=int, default=10)
    b.add_argument("--tmp", default="/private/tmp/perfpass/perf-eval/bench")
    c = sub.add_parser("compare-records")
    c.add_argument("--a", type=Path, required=True, help="R1/shards/look-K/records")
    c.add_argument("--b", type=Path, required=True, help="R2/shards/look-K/records")
    c.add_argument("--out", type=Path, default=None)
    ns = ap.parse_args(argv)
    return {
        "identity": cmd_identity,
        "flips": cmd_flips,
        "bench": cmd_bench,
        "compare-records": cmd_compare_records,
    }[ns.cmd](ns)


if __name__ == "__main__":
    raise SystemExit(main())
