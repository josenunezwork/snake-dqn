#!/usr/bin/env python3
"""Component breakdown of ``train_m3c`` profiles (perf-profile, 2026-10-13).

Reads the dumps written by ``M3C_PROFILE`` (:mod:`research.redesign_m3c_20261008.profile_m3c`)
or by ``bench_actor.py --profile --profile-dir``:

* ``<role>.pyisession`` (pyinstrument): every sample's full stack is classified by the
  innermost frame that matches a component rule (:data:`ACTOR_RULES` / :data:`LEARNER_RULES`);
  a few generic leaves (``Tensor.to``, ``Tensor.cpu``, ``sleep``, ...) are *refined* by the
  component that encloses them (e.g. ``Tensor.to`` inside ``choose`` is the hero upload,
  inside ``actor_main`` the weight sync). The components partition the profiled wall time.
* ``<role>.pstats`` (cProfile): inclusive time and call counts of the rule functions (no
  partition: cProfile keeps no stacks; its per-call cost also inflates Python-heavy code).

How time lands where (pyinstrument 5): a sample is taken on the first Python call / return /
builtin-call event after the interval; the elapsed time goes to the stack of that event, a
``call`` event charging the caller. A compiled numba kernel or a blocking device wait that
is entered without a builtin-call event (``tensor[bool_mask] = x``, ``bool(tensor)``, a numba
dispatcher) is therefore charged to the *Python function that called it* (its self time),
which is what the rules key on.

Per actor the report also gives the Amdahl bound of compiling the rest of the Python grid sim
plus the scripted opponents into one kernel (:func:`amdahl`).

Usage::

    ./venv/bin/python research/perf_profile_20261013/analyze_profiles.py DIR [DIR ...] \\
        [--json out.json] [--drill PREFIX [--depth N]]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Union

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

Comp = Union[str, Callable[[str], str]]
#: (function, class or None, file basename or None, component | refine(enclosing) -> component)
Rule = Tuple[str, Optional[str], Optional[str], Comp]

# ----------------------------------------------------------------------------- categories
#: Component prefix -> report category (first match wins; the report tables group by these).
CATEGORIES = (
    ("sim:", "grid sim internals"),
    ("opponents: scripted", "scripted opponents (fast_anchors)"),
    ("opponents:", "vector61 featurization + opponent MLPs"),
    ("vector61 featurize", "vector61 featurization + opponent MLPs"),
    ("hero obs", "ego raster + v8 features"),
    ("v8 label", "ego raster + v8 features"),
    ("hero", "hero forward + transfers"),
    ("policy glue", "policy glue (masks, eps sampling)"),
    ("ring write", "n-step / ring / messaging"),
    ("n-step", "n-step / ring / messaging"),
    ("actor loop", "n-step / ring / messaging"),
    ("queue put", "waiting on the learner (queue back-pressure)"),
    ("weight sync", "weight sync (actor refresh)"),
    ("segment build", "segment build (once per segment)"),
    ("paused", "paused (Mac guard)"),
    ("learner/sample", "sample + gather (CPU)"),
    ("learner/h2d", "transfer: H2D batch upload"),
    ("learner/launch", "forward / backward / loss (host-side launch)"),
    ("learner/optim", "optimizer + target sync (host-side launch)"),
    ("learner/sync", "device sync (host waits for GPU compute)"),
    ("learner/publish", "weight publish (D2H for the actors)"),
    ("learner/idle", "idle: waiting for actor frames"),
    ("learner/loop", "ingest / loop / logging"),
)
#: Actor categories that are not actor work (the actor is blocked on someone else).
ACTOR_IDLE = (
    "waiting on the learner (queue back-pressure)",
    "paused (Mac guard)",
)


def category(comp: str) -> str:
    for prefix, cat in CATEGORIES:
        if comp.startswith(prefix):
            return cat
    return "other"


# ----------------------------------------------------------------------------- actor rules
def _glue_or(target: str) -> Callable[[str], str]:
    """Refine a generic leaf: inside the policy glue it is ``target``, else the enclosure."""

    def refine(enclosing: str) -> str:
        return target if enclosing.startswith("policy glue") else enclosing

    return refine


def _to_refine(enclosing: str) -> str:
    if enclosing.startswith("policy glue"):
        return HERO_H2D
    if enclosing.startswith("actor loop"):
        return "weight sync"
    return enclosing


def _sleep_refine(enclosing: str) -> str:
    return "paused (Mac guard)" if enclosing.startswith("actor loop") else enclosing


SIM_FOOD = "sim: food (maintain / spawn / consume / trail pellets; Python lists)"
SIM_MOVE = "sim: movement + grid writes"
SIM_DEATH = "sim: collision resolve + deaths (corpses, grid rebuild)"
SIM_GLUE = "sim: step glue (frame/active masks, counters, traversal rebuild)"
V61_FOOD = "vector61 featurize: food lists -> arrays (_food_arrays, Python)"
HERO_H2D = "hero H2D (4 blocking Tensor.to: local/global/scalars/mask)"
HERO_FWD = "hero forward (Ego2sNet launch / CPU side)"
HERO_D2H = "hero D2H (Tensor.cpu; waits for the forward)"

ACTOR_RULES: Sequence[Rule] = (
    # ---- grid sim
    ("step_with_policy", None, "batch_sim.py", SIM_GLUE),
    ("step", "GridBatchSim", "batch_sim.py", SIM_GLUE),
    ("_rebuild_traversed_from_heads", None, "batch_sim.py", SIM_GLUE),
    ("_maintain_food", None, "grid_sim.py", SIM_FOOD),
    ("_maintain_food_env", None, "grid_sim.py", SIM_FOOD),
    ("_consume_food", None, "grid_sim.py", SIM_FOOD),
    ("_drop_trail_pellets", None, "grid_sim.py", SIM_FOOD),
    ("_respawn_dead", None, "grid_sim.py", "sim: respawn"),
    ("_refresh_action_masks", None, "batch_sim.py", "sim: action masks (numba kernel + resolve)"),
    ("_move_all", None, "grid_sim.py", SIM_MOVE),
    ("_move_all", None, "batch_sim.py", SIM_MOVE),
    ("_write_new_heads", None, "grid_sim.py", SIM_MOVE),
    ("detect_collisions", None, "grid_sim_nb.py", "sim: collision detect (numba kernel)"),
    ("_detect_collisions", None, "grid_sim.py", "sim: collision detect (numba kernel)"),
    ("_resolve_collisions", None, "grid_sim.py", SIM_DEATH),
    ("_apply_deaths", None, "grid_sim.py", SIM_DEATH),
    ("_compute_rewards", None, "grid_sim.py", "sim: rewards"),
    # ---- opponents
    ("opponents", None, "train_m3c.py", "opponents: loop glue"),
    ("actions", "FastProfileAnchorSimdPolicy", "fast_anchors.py", "opponents: scripted"),
    ("actions", "Vector61SimdPolicy", "vector61_policy.py", "opponents: vector61 mask/argmax"),
    ("q_values", "Vector61SimdPolicy", "vector61_policy.py", "opponents: vector61 MLP forward"),
    # ---- vector61 featurization (opponent rows + the hero row for v8)
    ("prepare", "Vector61Runtime", "vector61_policy.py", "vector61 featurize (pre-step)"),
    ("observe_step", "Vector61Runtime", "vector61_policy.py", "vector61 featurize (post-step)"),
    ("_food_arrays", None, "vector61_nb.py", V61_FOOD),
    # ---- hero observation
    ("build_ego_raster", None, "ego_raster.py", "hero obs: ego raster"),
    ("v8_scalars", None, "ego_raster_bv8.py", "hero obs: v8 features (v8_decide)"),
    ("selection", "Vector61Runtime", "vector61_policy.py", _glue_or("hero obs: v8 features")),
    ("v8_redecide", None, "v8_nb.py", "v8 label redecide"),
    # ---- hero forward + transfers
    ("forward", "Ego2sNet", "ego2s_network.py", HERO_FWD),
    ("_wrapped_call_impl", "Ego2sNet", "module.py", HERO_FWD),
    ("_call_impl", "Ego2sNet", "module.py", HERO_FWD),
    ("Tensor.to", None, "<built-in>", _to_refine),
    ("Tensor.cpu", None, "<built-in>", _glue_or(HERO_D2H)),
    ("Tensor.numpy", None, "<built-in>", _glue_or(HERO_D2H)),
    ("Tensor.float", None, "<built-in>", _glue_or(HERO_D2H)),
    ("choose", None, "train_m3c.py", "policy glue (masks, eps, argmax)"),
    ("sample_legal", None, "train_m3c.py", "policy glue (masks, eps, argmax)"),
    # ---- bookkeeping / messaging
    ("add", "SharedObsRing", "train_m3c.py", "ring write (shared obs ring)"),
    ("step", "NStepBuilder", "train_m3c.py", "n-step builder"),
    ("take", "NStepBuilder", "train_m3c.py", "n-step builder"),
    ("end_segment", "NStepBuilder", "train_m3c.py", "n-step builder"),
    ("put", "Queue", "queues.py", "queue put (back-pressure: learner not keeping up)"),
    ("_read_weights", None, "train_m3c.py", "weight sync"),
    ("to", "Ego2sNet", "module.py", "weight sync"),
    ("build_segment", None, "train_m3c.py", "segment build"),
    ("make_shield", None, "train_m3c.py", "segment build"),
    ("check_live_rule_config", None, "vector61_policy.py", "segment build"),
    ("sleep", None, "<built-in>", _sleep_refine),
    ("actor_main", None, "train_m3c.py", "actor loop (reward, mass term, msg build)"),
)

# ----------------------------------------------------------------------------- learner rules
L_SAMPLE = "learner/sample: replay + label indices, ring rows gathered into staging"
L_H2D = "learner/h2d: one staged batch upload (_Staging.upload)"
L_SYNC_ARGMAX = "learner/sync: _masked_argmax bool index (waits for upload + next-state fwds)"
L_SYNC_FINAL = "learner/sync: _finalize finite flag (waits for the previous fwd/bwd)"


def _learner_to(enclosing: str) -> str:
    if enclosing.startswith("learner/launch: update glue"):
        return "learner/h2d: other Tensor.to in the update"
    return enclosing


LEARNER_RULES: Sequence[Rule] = (
    ("main_multi", None, "train_m3c.py", "learner/loop: main loop glue"),
    ("get", "Queue", "queues.py", "learner/idle: queue get (waiting for actor frames)"),
    ("add_many", "Replay", "train_m3c.py", "learner/loop: ingest (replay / labels)"),
    ("encode", "MultiRing", "train_m3c.py", "learner/loop: ingest (replay / labels)"),
    ("add", "LabelBuffer", "train_m3c.py", "learner/loop: ingest (replay / labels)"),
    ("update", "Learner", "learner_m3c.py", "learner/launch: update glue (gathers, targets)"),
    ("_update_fast", "Learner", "learner_m3c.py", "learner/launch: update glue (gathers, targets)"),
    (
        "_update_reference",
        "Learner",
        "learner_m3c.py",
        "learner/launch: update glue (gathers, targets)",
    ),
    ("_sample_fast", "Learner", "learner_m3c.py", L_SAMPLE),
    ("sample", "Replay", "train_m3c.py", L_SAMPLE),
    ("sample", "LabelBuffer", "train_m3c.py", L_SAMPLE),
    ("get", "MultiRing", "train_m3c.py", L_SAMPLE),
    ("valid", "MultiRing", "train_m3c.py", L_SAMPLE),
    ("upload", "_Staging", "learner_m3c.py", L_H2D),
    ("Tensor.to", None, "<built-in>", _learner_to),
    ("forward", "Ego2sNet", "ego2s_network.py", "learner/launch: forward"),
    ("_wrapped_call_impl", "Ego2sNet", "module.py", "learner/launch: forward"),
    ("backward", "Tensor", "_tensor.py", "learner/launch: backward"),
    ("clip_grad_norm_", None, "clip_grad.py", "learner/launch: grad clip"),
    ("anchor_loss_margin", None, "train_m3c.py", "learner/launch: loss ops"),
    ("smooth_l1_loss", None, "functional.py", "learner/launch: loss ops"),
    ("_masked_argmax", None, "learner_m3c.py", L_SYNC_ARGMAX),
    ("_finalize", "Learner", "learner_m3c.py", L_SYNC_FINAL),
    ("wrapper", None, "optimizer.py", "learner/optim: Adam step (launch)"),
    ("zero_grad", None, "optimizer.py", "learner/optim: zero_grad"),
    ("load_state_dict", "Ego2sNet", "module.py", "learner/optim: target sync (load_state_dict)"),
    ("flush", "Learner", "learner_m3c.py", "learner/loop: flush stats (log rows, syncs)"),
    (
        "flush_stats_only",
        "Learner",
        "learner_m3c.py",
        "learner/loop: flush stats (log rows, syncs)",
    ),
    ("_publish_weights", None, "train_m3c.py", "learner/publish: _publish_weights"),
    ("save_checkpoint_atomic", None, "train_m3c.py", "learner/loop: checkpoint"),
    ("write", "Telemetry", "train_m3c.py", "learner/loop: telemetry rows"),
    ("guard_ok", None, None, "learner/loop: Mac guard"),
)


# ----------------------------------------------------------------------------- matching
def parse_frame(ident: str) -> Tuple[str, str, str]:
    """``(function, class, file basename)`` of a pyinstrument frame identifier."""
    parts = ident.split("\x00")
    func = parts[0]
    path = parts[1] if len(parts) > 1 else ""
    attrs = parts[2].split("\x01")[1:] if len(parts) > 2 else []
    cls = next((a[1:] for a in attrs if a.startswith("c")), "")
    return func, cls, path.rsplit("/", 1)[-1]


def _index(rules: Sequence[Rule]) -> Dict[str, List[Rule]]:
    idx: Dict[str, List[Rule]] = defaultdict(list)
    for r in rules:
        idx[r[0]].append(r)
    return idx


def match(frame: Tuple[str, str, str], idx: Dict[str, List[Rule]]) -> Optional[Comp]:
    func, cls, base = frame
    for _, rcls, rfile, comp in idx.get(func, ()):
        if rcls is not None and rcls != cls:
            continue
        if rfile is not None and rfile != base:
            continue
        return comp
    return None


def classify(stack: Sequence[str], idx: Dict[str, List[Rule]], cache: dict) -> str:
    """The innermost hard rule on the stack, refined by the innermost generic leaf below it."""
    frames = [cache.setdefault(s, parse_frame(s)) for s in stack]
    comps = [match(f, idx) for f in frames]
    hard_i, comp = -1, "other (outside the rules)"
    for i in range(len(frames) - 1, -1, -1):
        if isinstance(comps[i], str):
            hard_i, comp = i, comps[i]
            break
    for j in range(len(frames) - 1, hard_i, -1):
        if callable(comps[j]):
            return comps[j](comp)
    return comp


def _load_session(path: Path):
    os.environ.setdefault("PYINSTRUMENT_IGNORE_OVERHEAD_WARNING", "1")
    from pyinstrument.session import Session

    return Session.load(str(path))


def breakdown_session(path: Path, rules: Sequence[Rule]) -> Dict[str, float]:
    s = _load_session(path)
    idx = _index(rules)
    cache: dict = {}
    out: Dict[str, float] = defaultdict(float)
    for stack, t in s.frame_records:
        out[classify(stack, idx, cache)] += float(t)
    return dict(out)


def drill_session(
    path: Path, rules: Sequence[Rule], prefix: str, depth: int = 2, top: int = 25
) -> Tuple[float, List[Tuple[str, float]]]:
    """The innermost ``depth`` frames of the samples whose component starts with ``prefix``
    (what a component is made of: e.g. ``fromiter`` inside the vector61 food conversion)."""
    s = _load_session(path)
    idx = _index(rules)
    cache: dict = {}
    out: Dict[str, float] = defaultdict(float)
    total = 0.0
    for stack, t in s.frame_records:
        if not classify(stack, idx, cache).startswith(prefix):
            continue
        total += float(t)
        frames = [cache.setdefault(x, parse_frame(x)) for x in stack[-depth:]]
        out[" > ".join(f"{f[0]}({f[2]})" for f in frames)] += float(t)
    return total, sorted(out.items(), key=lambda kv: -kv[1])[:top]


def pstats_inclusive(path: Path, rules: Sequence[Rule]) -> List[dict]:
    """cProfile inclusive time / calls of each hard-rule function (no partition)."""
    import pstats

    st = pstats.Stats(str(path))
    rows = []
    want = {(r[0], r[2]) for r in rules if isinstance(r[3], str)}
    for (file, line, func), (cc, nc, tt, ct, _callers) in st.stats.items():
        base = file.rsplit("/", 1)[-1]
        if (func, base) in want:
            rows.append({"func": func, "file": base, "line": line, "calls": nc, "tt": tt, "ct": ct})
    rows.sort(key=lambda r: -r["ct"])
    return rows


# ----------------------------------------------------------------------------- Amdahl
#: What "one compiled sim step" would absorb: the Python grid sim and the scripted opponents
#: (``compiled``), plus the downstream cost that exists only because the sim keeps its food
#: in Python lists (``induced``: the vector61 featurizer converts them to arrays every frame).
AMDAHL_COMPILED = ("grid sim internals", "scripted opponents (fast_anchors)")
AMDAHL_INDUCED = (V61_FOOD,)


def amdahl(components: Dict[str, float]) -> dict:
    """Actor-side Amdahl bounds for compiling the sim + scripted opponents into one kernel.

    ``busy`` is the actor's own work (profiled time minus queue back-pressure and pauses);
    ``p`` the share of ``busy`` the kernel would replace. ``S_inf`` = 1 / (1 - p) is the
    ceiling (the replaced code takes zero time); ``S_3x`` assumes the replaced code runs 3x
    faster. ``utilization`` = busy / profiled: below 1 the actors already wait for the
    learner, and an actor speed-up does not change the run's throughput at all.
    """
    total = sum(components.values())
    cats: Dict[str, float] = defaultdict(float)
    for c, v in components.items():
        cats[category(c)] += v
    idle = sum(cats[c] for c in ACTOR_IDLE)
    busy = total - idle
    compiled = sum(cats[c] for c in AMDAHL_COMPILED)
    induced = sum(components.get(c, 0.0) for c in AMDAHL_INDUCED)
    out = {"profiled_s": total, "busy_s": busy, "utilization": busy / total if total else 0.0}
    for name, part in (("sim+scripted", compiled), ("sim+scripted+food lists", compiled + induced)):
        p = part / busy if busy else 0.0
        out[name] = {
            "s": part,
            "p": p,
            "S_inf": 1.0 / (1.0 - p) if p < 1 else float("inf"),
            "S_3x": 1.0 / ((1.0 - p) + p / 3.0),
        }
    return out


# ----------------------------------------------------------------------------- report
def summarize(d: Path) -> Dict[str, dict]:
    if not any(d.glob("*.meta.json")) and (d / "profile").is_dir():
        d = d / "profile"  # a train_m3c --out directory
    out: Dict[str, dict] = {}
    for meta_path in sorted(d.glob("*.meta.json")):
        role = meta_path.name[: -len(".meta.json")]
        meta = json.loads(meta_path.read_text())
        rules = LEARNER_RULES if role.startswith("learner") else ACTOR_RULES
        sess = d / f"{role}.pyisession"
        pst = d / f"{role}.pstats"
        row: dict = {"meta": meta}
        if sess.exists():
            comps = breakdown_session(sess, rules)
            total = sum(comps.values())
            row["total_s"] = total
            row["components"] = dict(sorted(comps.items(), key=lambda kv: -kv[1]))
            cats: Dict[str, float] = defaultdict(float)
            for c, v in comps.items():
                cats[category(c)] += v
            row["categories"] = dict(sorted(cats.items(), key=lambda kv: -kv[1]))
            if not role.startswith("learner"):
                row["amdahl"] = amdahl(comps)
        if pst.exists():
            row["pstats_inclusive"] = pstats_inclusive(pst, rules)
        out[role] = row
    return out


def per_unit(role: str, meta: dict) -> Tuple[str, float]:
    if role.startswith("learner"):
        s, e = meta.get("start") or {}, meta.get("end") or {}
        n = (e.get("updates", 0) - s.get("updates", 0)) if s and e else 0
        return "update", float(n)
    return "frame", float(meta.get("frames") or meta.get("ticks") or 0)


def _table(rows: Dict[str, float], total: float, unit: str, n: float, head: str) -> None:
    print(f"| {head} | s | % | ms/{unit} |")
    print("|---|---:|---:|---:|")
    for c, v in rows.items():
        per = 1e3 * v / n if n else float("nan")
        print(f"| {c} | {v:.2f} | {100 * v / total:.1f} | {per:.2f} |")


def print_report(d: Path, res: Dict[str, dict]) -> None:
    print(f"## {d}")
    for role, row in res.items():
        meta = row["meta"]
        unit, n = per_unit(role, meta)
        print(
            f"\n### {role} ({meta.get('mode')}, profiled {meta.get('profiled_s', 0):.1f} s, "
            f"{int(n)} {unit}s)"
        )
        if "components" not in row:
            for r in row.get("pstats_inclusive", [])[:30]:
                print(
                    f"  {r['ct']:8.2f}s incl {r['tt']:8.2f}s own {r['calls']:>9} calls  "
                    f"{r['func']} ({r['file']})"
                )
            continue
        total = row["total_s"]
        _table(row["components"], total, unit, n, "component")
        print()
        _table(row["categories"], total, unit, n, "category")
        am = row.get("amdahl")
        if am:
            print(
                f"\nAmdahl (actor busy {am['busy_s']:.2f} s = {100 * am['utilization']:.0f}% "
                "of profiled):"
            )
            for name in ("sim+scripted", "sim+scripted+food lists"):
                a = am[name]
                print(
                    f"  {name}: p = {a['p']:.3f} of busy -> S_inf = {a['S_inf']:.3f}x, "
                    f"S_3x = {a['S_3x']:.3f}x"
                )


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("dirs", nargs="+", type=Path)
    ap.add_argument("--json", type=Path, default=None)
    ap.add_argument("--drill", default=None, help="component prefix to break down by leaf")
    ap.add_argument("--depth", type=int, default=2, help="--drill: innermost frames shown")
    a = ap.parse_args(argv)
    if a.drill:
        for d in a.dirs:
            if not any(d.glob("*.pyisession")) and (d / "profile").is_dir():
                d = d / "profile"
            for sess in sorted(d.glob("*.pyisession")):
                role = sess.name[: -len(".pyisession")]
                rules = LEARNER_RULES if role.startswith("learner") else ACTOR_RULES
                total, rows = drill_session(sess, rules, a.drill, a.depth)
                print(f"## {sess}: components {a.drill!r}* = {total:.2f} s")
                for k, v in rows:
                    print(f"  {v:8.2f}  {k}")
        return 0
    allres = {}
    for d in a.dirs:
        res = summarize(d)
        print_report(d, res)
        allres[str(d)] = res
    if a.json:
        a.json.parent.mkdir(parents=True, exist_ok=True)
        a.json.write_text(json.dumps(allres, indent=1, default=str) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
