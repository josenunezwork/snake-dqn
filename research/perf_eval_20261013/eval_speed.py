#!/usr/bin/env python3
"""Evaluation-speed switches for dev probes / Phase R: identity, flip rate, timing (perf-eval).

DEVELOPMENT ONLY (not a screen, not gate evidence). Subcommands around
``research/redesign_m2_20261008/dev_probe.py`` (``dev_probe.probe``, same engine, profile,
worlds and rosters as the probes ``probe_watch.py`` launches):

* ``identity``: run one probe twice in this process, the reference (vector61 featurizer
  ``python``, hero forward ``rowwise``, Python anchors) and a variant (``--variant numba``:
  compiled vector61 featurizer; ``fast-anchors``: compiled scripted anchors; ``batched``: one
  batched ego2s hero forward per frame; and the ``numba+...`` combinations), each
  with the per-frame whole-world digest recorder of ``grid_h5000_identity`` and a per-frame
  action recorder (every slot's action, as returned to the sim) installed, and compare every
  world's record (type-strict, wall-clock keys stripped), digest stream and action stream
  (first differing frame of each). The batched variants' own provenance field
  (``ego2s_hero.forward``) is listed but not counted. Exit 1 on any difference.
* ``flips``: the reference run with a shadow batched hero forward: per decision, compares the
  batched Q with the rowwise Q actually used (bitwise-equal rows, max |dQ|, and the masked
  greedy argmax flip rate under the mask the policy selects with). The trajectory is the
  reference one (the shadow never acts).
* ``profile``: one probe with stage timers (inclusive wall of the sim step, vector61
  featurizer prepare/observe, Apex forwards, veto + its free-space counts, ego2s raster and
  forward, scripted policies) and the exclusive cost split (sums to the probe wall).
* ``bench``: dev_probe CLI wall per (arm, mix, variant), one subprocess per cell, at most
  ``--procs`` (1-2) at a time. Numbers taken while other jobs run are NOISY.
* ``compare-records``: Phase R record directories of two plan roots (e.g. the same smoke look
  under a ``python`` and a ``numba`` ``phase_r_m4.py`` plan), every entry field except the
  run's own (binding, commit, wall, unit id, look, tier).

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
  ./venv/bin/python research/perf_eval_20261013/eval_speed.py profile --arm student-nv \\
      --student S.pth --mix frozen --worlds 16 --frames 1500 --variant numba --out OUT.json

Measured 2026-10-13 on the busy Mac (NOISY; identity, flips, cost splits, 32-world H5000
walls and a Phase R smoke-subset records comparison): ``results/noisy_mac_20261013.json``.
Re-run ``bench`` with ``--quiet-machine --procs 1`` (and ``--variants
default,numba,numba+fast-anchors``) on an idle machine for numbers to quote.
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
#: variant -> (vector61 backend, ego2s hero forward, scripted anchors); "default" is the
#: probe_watch reference.
VARIANTS: Dict[str, Dict[str, str]] = {
    "default": {"v61_backend": "python", "ego2s_forward": "rowwise", "anchors": "python"},
    "numba": {"v61_backend": "numba", "ego2s_forward": "rowwise", "anchors": "python"},
    "fast-anchors": {"v61_backend": "python", "ego2s_forward": "rowwise", "anchors": "fast"},
    "numba+fast-anchors": {"v61_backend": "numba", "ego2s_forward": "rowwise", "anchors": "fast"},
    "batched": {"v61_backend": "python", "ego2s_forward": "batched", "anchors": "python"},
    "numba+batched": {"v61_backend": "numba", "ego2s_forward": "batched", "anchors": "python"},
}


def probe_args(arm: str, variant: str, ns: argparse.Namespace) -> argparse.Namespace:
    """``dev_probe`` arguments for one (arm, variant) cell (probe_watch's settings)."""
    a, v = ARMS[arm], VARIANTS[variant]
    argv = [
        *("--arm", a["arm"], "--veto", a["veto"], "--mix", ns.mix),
        *("--worlds", str(ns.worlds), "--round-index", str(ns.round_index)),
        *("--ego2s-forward", v["ego2s_forward"], "--ego2s-mask", "advisory"),
        *("--v61-backend", v["v61_backend"], "--anchors", v["anchors"], "--out", os.devnull),
    ]
    if a["arm"] == "student":
        if not ns.student:
            raise SystemExit(f"--arm {arm} needs --student")
        argv += ["--student", str(ns.student)]
    if ns.frames:
        argv += ["--frames", str(ns.frames)]
    return dev_probe.build_parser().parse_args(argv)


class ActionRecorder:
    """Wraps a sim class's ``step_with_policy`` selector: every env's full action row
    (all slots) of every frame, as returned to the sim (``per_env[e][frame]`` = bytes)."""

    def __init__(self) -> None:
        self.per_env: List[List[bytes]] = []

    def install(self, sim_class: type) -> Any:
        recorder = self
        original = sim_class.step_with_policy

        def step_with_policy(sim, action_selector, *args, **kwargs):
            def select(prepared):
                actions = action_selector(prepared)
                grid = np.asarray(actions, dtype=np.int64)
                if not recorder.per_env:
                    recorder.per_env = [[] for _ in range(grid.shape[0])]
                for e in range(grid.shape[0]):
                    recorder.per_env[e].append(grid[e].astype(np.int8).tobytes())
                return actions

            return original(sim, select, *args, **kwargs)

        sim_class.step_with_policy = step_with_policy
        return original


def _restore(sim_class: type, own: bool, original: Any) -> None:
    """Undo a recorder: restore the class's own method, or drop the shadowing attribute."""
    if own:
        sim_class.step_with_policy = original
    else:
        del sim_class.step_with_policy


def run_recorded(args: argparse.Namespace, ctx: Mapping[str, Any]) -> Dict[str, Any]:
    """One probe with the per-frame world digest and action recorders installed."""
    from src.simd_env import eval_engine as ee

    sim_class = ee._terminal_hero_sim_class("grid")
    own = "step_with_policy" in vars(sim_class)
    original = sim_class.step_with_policy
    digests, actions = gid.DigestRecorder(), ActionRecorder()
    digests.install(sim_class)
    actions.install(sim_class)  # outermost: sees the selector's actions before the step
    started = time.perf_counter()
    try:
        records, row = dev_probe.probe(args, ctx)
    finally:
        _restore(sim_class, own, original)
    return {
        "wall_seconds": time.perf_counter() - started,
        "records": [gid.normalize(r) for r in records],
        "frame_digests": digests.per_env,
        "frame_actions": actions.per_env,
        "row": row,
    }


#: Record paths that only name the variant's own setting (provenance), per variant.
PROVENANCE_PATHS: Dict[str, tuple] = {
    "batched": ("ego2s_hero.forward",),
    "numba+batched": ("ego2s_hero.forward",),
}


def compare(
    ref: Mapping[str, Any],
    var: Mapping[str, Any],
    seeds: Sequence[int],
    ignore: Sequence[str] = (),
):
    """Per-world identity rows: records + digest streams (``gid.compare_worlds``) + actions.

    A world is ``identical`` when its record (type-strict, wall-clock keys stripped), its
    per-frame whole-world digest stream and its per-frame action rows (every slot) are all
    equal. ``ignore``: record paths that are provenance of the variant itself (the batched
    hero's ``ego2s_hero.forward``); they are listed per world but are not a difference."""
    rows = gid.compare_worlds(ref, var, seeds)
    for index, r in enumerate(rows):  # gid's labels are engine names; here reference/variant
        r["world_chain_reference"] = r.pop("world_chain_batch")
        r["world_chain_variant"] = r.pop("world_chain_grid")
        kept, ignored = [], []
        for d in r["record_differences"]:
            d["reference"] = d.pop("batch")
            d["variant"] = d.pop("grid")
            (ignored if d["path"] in ignore else kept).append(d)
        r["record_differences"], r["ignored_provenance"] = kept, ignored
        if ignored:  # gid compared the whole record; only the provenance path differs
            r["record_identical"] = not kept
        fa, fb = ref["frame_actions"][index], var["frame_actions"][index]
        first = next((f for f, (x, y) in enumerate(zip(fa, fb)) if x != y), None)
        if first is None and len(fa) != len(fb):
            first = min(len(fa), len(fb))
        r["actions_compared"] = min(len(fa), len(fb))
        r["first_differing_action_frame"] = first
        r["differing_action_frames"] = sum(x != y for x, y in zip(fa, fb)) + abs(len(fa) - len(fb))
        r["identical"] = (
            r["record_identical"]
            and r["first_differing_frame"] is None
            and r["world_chain_reference"] == r["world_chain_variant"]
            and first is None
        )
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
    rows = compare(ref, var, seeds, PROVENANCE_PATHS.get(ns.variant, ()))
    identical = sum(r["identical"] for r in rows)
    payload = {
        **_meta(ns),
        "schema": "perf-eval-identity/v2",
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
                "first_differing_action_frames": [r["first_differing_action_frame"] for r in rows],
                "record_difference_paths": sorted(
                    {d["path"] for r in rows for d in r["record_differences"]}
                ),
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


# ----------------------------------------------------------------------------- profile
class StageTimer:
    """Inclusive wall time + calls of patched callables (``(owner, name)`` attributes)."""

    def __init__(self) -> None:
        self.seconds: Dict[str, float] = {}
        self.calls: Dict[str, int] = {}
        self._undo: List[tuple] = []

    def patch(self, label: str, owner: Any, name: str) -> None:
        own = name in vars(owner)
        original = getattr(owner, name)
        timer = self
        self.seconds.setdefault(label, 0.0)
        self.calls.setdefault(label, 0)

        def timed(*args, **kwargs):
            t0 = time.perf_counter()
            try:
                return original(*args, **kwargs)
            finally:
                timer.seconds[label] += time.perf_counter() - t0
                timer.calls[label] += 1

        setattr(owner, name, timed)
        self._undo.append((owner, name, own, original))

    def restore(self) -> None:
        for owner, name, own, original in reversed(self._undo):
            if own:
                setattr(owner, name, original)
            else:
                delattr(owner, name)
        self._undo.clear()


def profile_split(seconds: Mapping[str, float], wall: float) -> Dict[str, float]:
    """Exclusive cost split (seconds) from :class:`StageTimer`'s inclusive stages.

    Nesting: ``step_with_policy`` > {``v61_prepare``, ``dispatch``}; ``dispatch`` > every
    policy's ``actions``; vector61 ``actions`` > {``v61_q_apex``, ``veto``}; ``veto`` >
    ``v8_free_space``; ego2s ``actions`` > {``ego2s_raster``, ``ego2s_forward``}."""
    s = {k: float(v) for k, v in seconds.items()}
    g = s.get
    v61_actions = g("v61_actions", 0.0)
    ego_actions = g("ego2s_actions", 0.0)
    scripted = g("scripted_actions", 0.0)
    ego_core = g("ego2s_raster", 0.0) + g("ego2s_forward", 0.0)
    split = {
        "v61_featurize_prepare": g("v61_prepare", 0.0),
        "v61_featurize_observe": g("v61_observe", 0.0),
        "v61_apex_forward": g("v61_q_apex", 0.0),
        "veto_rule": g("veto", 0.0) - g("v8_free_space", 0.0),
        "veto_free_space": g("v8_free_space", 0.0),
        "v61_policy_other": v61_actions - g("v61_q_apex", 0.0) - g("veto", 0.0),
        "ego2s_raster": g("ego2s_raster", 0.0),
        "ego2s_forward": g("ego2s_forward", 0.0),
        "ego2s_policy_other": max(ego_actions - ego_core, 0.0),
        "scripted_policies": scripted,
        "dispatch_other": g("dispatch", 0.0) - v61_actions - ego_actions - scripted,
        "sim_step": g("step_with_policy", 0.0) - g("v61_prepare", 0.0) - g("dispatch", 0.0),
        "accounting_other": wall - g("step_with_policy", 0.0) - g("v61_observe", 0.0),
    }
    # An ego2s v8 hero runs its student inside the vector61 veto policy (q_values override):
    # its raster/forward time is inside v61_actions, not ego2s_actions.
    if ego_actions < ego_core:
        split["v61_policy_other"] -= ego_core - ego_actions
    return split


def cmd_profile(ns: argparse.Namespace) -> int:
    """One dev probe with stage timers (inclusive wall per stage) and the exclusive split."""
    import torch

    from src.simd_env import ego2s_policy as ep
    from src.simd_env import eval_engine as ee
    from src.simd_env import vector61_policy as vp
    from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

    torch.set_num_threads(1)
    ctx = gid._context(1)
    args = probe_args(ns.arm, ns.variant, ns)
    timer = StageTimer()
    timer.patch("step_with_policy", ee._terminal_hero_sim_class("grid"), "step_with_policy")
    timer.patch("v61_prepare", vp.Vector61Runtime, "prepare")
    timer.patch("v61_observe", vp.Vector61Runtime, "observe_step")
    timer.patch("v8_free_space", vp.Vector61Runtime, "decision_free_space")
    timer.patch("dispatch", ee, "_dispatch_actions")
    timer.patch("v61_actions", vp.Vector61SimdPolicy, "actions")
    timer.patch("v61_q_apex", vp.Vector61SimdPolicy, "q_values")  # not the ego2s override
    timer.patch("veto", vp.Vector61SimdPolicy, "_apply_veto")
    timer.patch("ego2s_actions", ep.Ego2sSimdPolicy, "actions")
    timer.patch("ego2s_raster", ep, "build_ego_raster")
    timer.patch("ego2s_forward", ep, "forward_q")
    scripted = (ee.GreedyFoodSimdPolicy, ee.RandomSafeSimdPolicy, ee._ProfileAnchorSimdPolicy)
    for cls in scripted + (FastProfileAnchorSimdPolicy,):
        timer.patch("scripted_actions", cls, "actions")
    started = time.perf_counter()
    try:
        _, row = dev_probe.probe(args, ctx)
    finally:
        timer.restore()
    wall = time.perf_counter() - started
    split = profile_split(timer.seconds, wall)
    payload = {
        **_meta(ns),
        "schema": "perf-eval-profile/v1",
        "arm": ns.arm,
        "mix": ns.mix,
        "variant": ns.variant,
        "frames": ns.frames or dev_probe.ni_spec.HORIZON,
        "worlds": ns.worlds,
        "wall_seconds": wall,
        "probe_wall_seconds": row["wall"],
        "inclusive_seconds": timer.seconds,
        "calls": timer.calls,
        "exclusive_seconds": split,
        "exclusive_fraction": {k: v / wall for k, v in split.items()},
        "row": row,
    }
    _write(ns.out, payload)
    print(json.dumps({"arm": ns.arm, "mix": ns.mix, "variant": ns.variant, "wall": wall}))
    for k, v in sorted(split.items(), key=lambda kv: -kv[1]):
        print(f"  {k:24s} {v:8.2f}s {100 * v / wall:5.1f}%")
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
                    batched = VARIANTS[variant]["ego2s_forward"] == "batched"
                    if batched and ARMS[arm]["arm"] != "student":
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
        argv += ["--ego2s-mask", "advisory", "--v61-backend", args.v61_backend]
        argv += ["--anchors", args.anchors, "--out", str(out)]
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
    pr = sub.add_parser("profile", parents=[common])
    pr.add_argument("--arm", choices=sorted(ARMS), required=True)
    pr.add_argument("--mix", choices=dev_probe.ni_spec.MIXES, required=True)
    pr.add_argument("--variant", choices=sorted(VARIANTS), default="default")
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
        "profile": cmd_profile,
        "bench": cmd_bench,
        "compare-records": cmd_compare_records,
    }[ns.cmd](ns)


if __name__ == "__main__":
    raise SystemExit(main())
