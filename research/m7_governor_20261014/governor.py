#!/usr/bin/env python3
"""DEVELOPMENT DIAGNOSTIC (M7 study B): would capping growth close B8's scripted-survival gap?

Evaluation only, never served, no candidate. The B8 death census
(``research/b8_death_census_20261014``) found that the B8 no-veto student dies on scripted
worlds by self-enclosure once its length reaches ~800+ (49 of 75 deaths at L >= 800, where
champion_a5 + v8 almost never goes). This asks whether a decision-time growth governor
closes the gap, and at what mass cost.

**Governor.** :class:`GovernorEgo2sPolicy` is a drop-in for
``Ego2sSimdPolicy(mask_mode="advisory")`` (the M3-C / Phase R no-veto student). It selects
the student's action exactly as the parent does (float64 Q, first maximum over the advisory
mask, normal-only fallback). Then, if the hero's length ``L > Lc``, it swaps a normal action
``a`` (0..2) for its boost twin ``a + 3`` when that twin is advisory-safe. Otherwise it keeps
the student's action. With ``Lc = None`` ("off") it returns the parent's actions unchanged.

**Boost mechanics** (``BatchSim._move_all``; ``GridBatchSim`` inherits them):

* A boost moves the head two cells.
* Every boosting frame increments ``boost_frames``. The counter is not reset by
  non-boost frames.
* On the ``boost_length_cost_frames``-th boost (3 in the deployment config) the snake pays
  one segment (``length -= 1``). Under mechanics v2 that tail cell becomes a corpse-class
  trail pellet, worth +1 to whoever eats it, the hero included.
* So burning needs sustained boosting: about -1/3 length per boosting frame, before food.
  The governor therefore boosts on every decision while ``L > Lc``, not once.

``min_boost_length`` is 5, so the boost twin is always legal here. Whether the cap holds is
measured, not assumed (``over_cap`` / ``replaced`` / ``max_len`` per world).

**Protocol.** The dev probe (``research/redesign_m2_20261008/dev_probe.probe``) is used
unchanged:

* the first 32 distillation-namespace worlds at ``round_index`` 52;
* mixes ``frozen``, ``scripted`` and ``mixed``;
* H5000, ``promotion-v2-watch-rect``, grid engine;
* ``ego2s_forward="rowwise"``, ``ego2s_mask="advisory"``, no veto.

The opt-in ``--v61-backend numba --anchors fast`` are record-identical by claim
(``tests/test_eval_v61_backend_identity.py``). The ``off`` rows re-check that claim against
the existing M4 / M6 probe rows world by world (``summarize``: ``checks.off_identity``).

Usage::

  ./venv/bin/python research/m7_governor_20261014/governor.py run \\
      --ckpt B8_s0=/.../B8_s0/final.pth --caps off,800,900,1000,1100 \\
      --mixes scripted,frozen,mixed --out OUT/rows_lane0.jsonl
  ./venv/bin/python research/m7_governor_20261014/governor.py summarize --root OUT
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from collections import Counter  # noqa: E402
from contextlib import contextmanager  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, Iterator, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.simd_env import ego2s_policy as ep  # noqa: E402

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
OUT_ROOT = ARTIFACTS / "m7-diagnostics-20261014" / "governor"
INCUMBENT_ROWS = ARTIFACTS / "redesign-m3c-probes-20261008-recovery" / "incumbent__dev__v8.jsonl"
CENSUS_RESULTS = ARTIFACTS / "b8-death-census-20261014" / "results.json"
CENSUS_RUNS = ARTIFACTS / "b8-death-census-20261014" / "runs"
# Existing no-veto advisory dev-probe rows of the same checkpoints (python backend/anchors).
REFERENCE_PROBES = {
    "B8": ARTIFACTS / "redesign-m4-mac-probes-20261013",
    "CTRL": ARTIFACTS / "redesign-m6-mac-probes-20261014",
}
MIXES = ("frozen", "scripted", "mixed")
ROUND_INDEX = 52
WORLDS = 32
BIG = 800  # the census's large-length threshold
LEN_BINS = (800, 1000, 1200)
CHAMPION_SCRIPTED = 0.688  # champion_a5 + v8, census (32 scripted worlds)
GUARD = 0.05  # Phase R: scripted survival within 0.05 of champion_a5 + v8
GO_SCRIPTED = 0.64
NO_GO_SCRIPTED = 0.60
MASS_MARGIN = 60.0
LEVER4_PREDICTION = {"cap": 900, "survival": 0.61, "mass_delta": -44.0}
GOVERNOR_MODES = ("twin", "bestboost", "twinspace")
MODE_LABEL = {
    "twin": "boost-twin-if-advisory-safe",
    "bestboost": "best-q-safe-boost",
    "twinspace": "boost-twin-if-advisory-safe-spacious-and-not-head-risky",
}
SPACE_LAM = 8.0  # v8 lambda for the diagnostics call (areas / head risk do not depend on it)


def parse_cap(text: str) -> Optional[int]:
    return None if text in ("off", "none", "") else int(text)


def cap_label(cap: Optional[int]) -> str:
    return "off" if cap is None else str(int(cap))


def arm_key(row: Mapping[str, Any]) -> str:
    """``"off"``, ``"<Lc>"`` (twin governor) or ``"<Lc>/<mode>"`` (a variant)."""
    mode = row.get("mode", "twin")
    if row["cap"] == "off" or mode == "twin":
        return row["cap"]
    return f"{row['cap']}/{mode}"


# ----------------------------------------------------------------------------- governor
class GovernorEgo2sPolicy(ep.Ego2sSimdPolicy):
    """Advisory-mask student plus the growth governor (see the module docstring).

    ``cap`` is ``Lc`` (``None`` = off). Per-world counters (indexed by env) are kept for the
    probe row. ``over_cap`` counts decisions with ``L > Lc``. ``replaced`` counts those where
    the boost twin was swapped in. ``blocked`` counts those where the student chose a normal
    action whose twin was not advisory-safe. ``student_boost`` counts decisions where the
    student itself chose a boost.
    """

    cap: Optional[int] = None
    mode: str = "twin"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.mask_mode != "advisory":
            raise ValueError("the governor wraps the advisory (Phase R no-veto) selection only")
        self.stats: Dict[str, np.ndarray] = {}

    COUNTERS = (
        "decisions",
        "over_cap",
        "replaced",
        "blocked",
        "student_boost",
        "ge800",
        "ge1000",
        "ge1200",
    )

    def _ensure(self, n_env: int) -> None:
        have = len(self.stats.get("decisions", ()))
        if have >= n_env:
            return
        for k in self.COUNTERS + ("last_len", "max_len", "len_sum", "last_replaced_at"):
            old = self.stats.get(k, np.zeros(0, dtype=np.int64))
            new = np.zeros(n_env, dtype=np.int64)
            new[: len(old)] = old
            self.stats[k] = new

    @staticmethod
    def govern(
        act: np.ndarray,
        adv: np.ndarray,
        length: np.ndarray,
        cap: Optional[int],
        q: Optional[np.ndarray] = None,
        mode: str = "twin",
        space_ok: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(new actions, replaced, blocked)`` for student actions ``act``.

        ``adv`` is the raw ``(N, 6)`` advisory mask (before the normal-only fallback).
        ``mode="twin"`` (the study's governor): a normal ``a`` becomes ``a + 3`` when that is
        advisory-safe. ``mode="bestboost"`` (variant): a normal action becomes the student's
        highest-Q advisory-safe boost action (float64, first maximum), if any.
        ``mode="twinspace"`` (variant): as ``twin``, and the twin must also be ``space_ok``
        (computed by :meth:`twin_space_ok`: spacious landing and not head-risky).
        """
        n = len(act)
        if cap is None:
            z = np.zeros(n, dtype=bool)
            return act, z, z
        if mode not in GOVERNOR_MODES:
            raise ValueError(f"mode must be one of {GOVERNOR_MODES}, got {mode!r}")
        over = length > cap
        normal = act < 3
        if mode in ("twin", "twinspace"):
            pick = act % 3 + 3
            safe = adv[np.arange(n), pick]
            if mode == "twinspace":
                assert space_ok is not None
                safe = safe & np.asarray(space_ok, dtype=bool)
        else:
            assert q is not None
            qb = np.where(adv[:, 3:], np.asarray(q, dtype=np.float64)[:, 3:], -np.inf)
            pick = qb.argmax(axis=1).astype(np.int64) + 3
            safe = adv[:, 3:].any(axis=1)
        replaced = over & normal & safe
        blocked = over & normal & ~safe
        return np.where(replaced, pick, act).astype(np.int64), replaced, blocked

    @staticmethod
    def twin_space_ok(
        sim, rows: np.ndarray, q: np.ndarray, adv: np.ndarray, act: np.ndarray, length
    ) -> np.ndarray:
        """Whether each row's boost twin lands spacious and is not v8 head-risky.

        Spacious: the twin's v8 tail-aware post-move landing area (``v8_nb``, capped at v7's
        ``area_cap`` = ``2L`` bounded by the free cells) is at least ``min(L, area_cap)``,
        i.e. there is room for the whole body after the boost (the census's length-aware
        "not tight" test).
        """
        from src.simd_env.v8_nb import v8_decide

        n = len(rows)
        ok = np.zeros(n, dtype=bool)
        if not n:
            return ok
        res = v8_decide(sim, rows, q, adv, SPACE_LAM, features=True)
        twin = act % 3 + 3
        i = np.arange(n)
        area = res.area[i, twin]
        need = np.minimum(np.asarray(length, dtype=np.int64), res.area_cap)
        return (area >= need) & ~res.head_risky[i, twin]

    def actions(self, masks: np.ndarray, sim, slots: np.ndarray) -> np.ndarray:
        rows = np.asarray(slots, dtype=np.int64).reshape(-1, 2)
        q = ep.student_q(self.net, sim, rows, self.forward).numpy()
        adv = sim.get_advisory_action_mask()[rows[:, 0], rows[:, 1]].astype(bool)
        m = adv.copy()
        m[~m.any(axis=1), :3] = True
        act = np.where(m, q.astype(np.float64), -np.inf).argmax(axis=1).astype(np.int64)
        length = np.asarray(sim.length)[rows[:, 0], rows[:, 1]].astype(np.int64)
        space_ok = None
        if self.mode == "twinspace" and self.cap is not None:
            space_ok = np.zeros(len(rows), dtype=bool)
            cand = np.nonzero((length > self.cap) & (act < 3))[0]
            space_ok[cand] = self.twin_space_ok(
                sim, rows[cand], q[cand], adv[cand], act[cand], length[cand]
            )
        out, replaced, blocked = self.govern(
            act, adv, length, self.cap, q, self.mode, space_ok=space_ok
        )
        # counters (hero rows only: one row per env)
        env = rows[:, 0]
        self._ensure(int(env.max()) + 1 if len(env) else 0)
        s = self.stats
        np.add.at(s["decisions"], env, 1)
        if self.cap is not None:
            np.add.at(s["over_cap"], env, (length > self.cap).astype(np.int64))
        np.add.at(s["replaced"], env, replaced.astype(np.int64))
        np.add.at(s["blocked"], env, blocked.astype(np.int64))
        np.add.at(s["student_boost"], env, (act >= 3).astype(np.int64))
        np.add.at(s["ge800"], env, (length >= 800).astype(np.int64))
        np.add.at(s["ge1000"], env, (length >= 1000).astype(np.int64))
        np.add.at(s["ge1200"], env, (length >= 1200).astype(np.int64))
        np.add.at(s["len_sum"], env, length)
        s["last_len"][env] = length
        # decision index (1-based) of the most recent governor boost, 0 = never
        s["last_replaced_at"][env[replaced]] = s["decisions"][env[replaced]]
        s["max_len"][env] = np.maximum(s["max_len"][env], length)
        return out


@contextmanager
def governed(cap: Optional[int], mode: str = "twin") -> Iterator[List[GovernorEgo2sPolicy]]:
    """Patch ``Ego2sSimdPolicy`` with a governor at ``cap``; yields the built instances."""
    built: List[GovernorEgo2sPolicy] = []

    class _Governor(GovernorEgo2sPolicy):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            built.append(self)

    _Governor.cap = cap
    _Governor.mode = mode
    orig = ep.Ego2sSimdPolicy
    ep.Ego2sSimdPolicy = _Governor  # eval_engine imports it from the module at build time
    try:
        yield built
    finally:
        ep.Ego2sSimdPolicy = orig


# ----------------------------------------------------------------------------- run
def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _git() -> Dict[str, Any]:
    def run(*a: str) -> str:
        return subprocess.run(
            ["git", "-C", str(REPO), *a], capture_output=True, text=True, check=False
        ).stdout.strip()

    return {"commit": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain"))}


def probe_args(ckpt: str, mix: str, worlds: int, backend: str, anchors: str, frames=None):
    from research.redesign_m2_20261008 import dev_probe

    argv = [
        "--student",
        str(ckpt),
        "--arm",
        "student",
        "--mix",
        mix,
        "--worlds",
        str(worlds),
        "--out",
        "/dev/null",
        "--round-index",
        str(ROUND_INDEX),
        "--ego2s-forward",
        "rowwise",
        "--ego2s-mask",
        "advisory",
        "--v61-backend",
        backend,
        "--anchors",
        anchors,
    ]
    if frames:
        argv += ["--frames", str(frames)]
    return dev_probe.build_parser().parse_args(argv)


def run_job(
    ctx: Mapping[str, Any],
    name: str,
    ckpt: str,
    cap: Optional[int],
    mix: str,
    worlds: int,
    backend: str,
    anchors: str,
    frames: Optional[int] = None,
    mode: str = "twin",
) -> Dict[str, Any]:
    from research.redesign_m2_20261008 import dev_probe

    args = probe_args(ckpt, mix, worlds, backend, anchors, frames)
    with governed(cap, mode) as built:
        recs, row = dev_probe.probe(args, ctx)
    if len(built) != 1:
        raise RuntimeError(f"expected one governed hero policy, got {len(built)}")
    st = built[0].stats
    seeds = [int(r["seed"]) for r in recs]
    per_world: Dict[int, Dict[str, Any]] = {}
    for e, (seed, r) in enumerate(zip(seeds, recs)):
        w = {k: int(st[k][e]) for k in st}
        w["boost_frame_fraction"] = r["probes"].get("boost_frame_fraction")
        w["food_eaten"] = r["probes"].get("food_eaten")
        w["max_mass"] = r.get("max_mass")
        w["len_at_death"] = int(st["last_len"][e]) if r["deaths"] else None
        w["cause"] = r["probes"].get("death_cause")
        # decisions from the last governor boost to the fatal decision (0 = boosted on it)
        w["since_governor_boost"] = (
            int(st["decisions"][e] - st["last_replaced_at"][e])
            if r["deaths"] and st["last_replaced_at"][e]
            else None
        )
        per_world[seed] = w
    row.update(
        {
            "study": "m7-governor",
            "name": name,
            "cap": cap_label(cap),
            "governor": MODE_LABEL[mode],
            "mode": mode,
            "per_world": per_world,
            "git": _git(),
        }
    )
    return row


def done_keys(out: Path) -> set:
    got = set()
    if out.exists():
        for line in out.read_text().splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            got.add((r["student_sha256"], arm_key(r), r["mix"], len(r["mass"])))
    return got


def cmd_run(args: argparse.Namespace) -> int:
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    torch.set_num_threads(1)
    ctx = gid._context(1)
    caps = [parse_cap(c) for c in args.caps.split(",")]
    mixes = args.mixes.split(",")
    ckpts = [c.split("=", 1) for c in args.ckpt]
    args.out.parent.mkdir(parents=True, exist_ok=True)
    have = done_keys(args.out)
    jobs = [(n, p, c, m) for n, p in ckpts for c in caps for m in mixes]
    for name, path, cap, mix in jobs:
        key = (
            sha256_file(Path(path)),
            arm_key({"cap": cap_label(cap), "mode": args.mode}),
            mix,
            args.worlds,
        )
        if key in have:
            print(f"skip {name} cap={cap_label(cap)} {mix}", flush=True)
            continue
        t0 = time.perf_counter()
        row = run_job(
            ctx, name, path, cap, mix, args.worlds, args.v61_backend, args.anchors, mode=args.mode
        )
        with args.out.open("a") as fh:
            fh.write(json.dumps(row) + "\n")
        m = list(row["mass"].values())
        surv = sum(1 for d in row["deaths"].values() if d == 0) / len(m)
        print(
            f"{name} cap={cap_label(cap)}/{args.mode} {mix}: mass {sum(m) / len(m):.1f}"
            f" surv {surv:.3f} ({time.perf_counter() - t0:.0f}s)",
            flush=True,
        )
    return 0


# ----------------------------------------------------------------------------- summarize
def load_rows(root: Path) -> List[Dict[str, Any]]:
    rows = []
    for f in sorted(root.glob("rows_*.jsonl")):
        for line in f.read_text().splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def group_of(name: str) -> str:
    return name.split("_")[0]


def reference_row(name: str, mix: str) -> Optional[Dict[str, Any]]:
    d = REFERENCE_PROBES.get(group_of(name))
    if d is None:
        return None
    f = d / f"{name}__final__none.jsonl"
    if not f.exists():
        return None
    for line in f.read_text().splitlines():
        r = json.loads(line)
        if r["mix"] == mix and r.get("ego2s_mask") == "advisory":
            return r
    return None


def _surv(deaths: Mapping[Any, Any]) -> float:
    return sum(1 for d in deaths.values() if d == 0) / len(deaths)


def _mean(xs: Sequence[float]) -> Optional[float]:
    return float(np.mean(xs)) if len(xs) else None


def arm_summary(rows: Sequence[Mapping[str, Any]], names: Sequence[str]) -> Dict[str, Any]:
    """Pooled metrics of one (group, cap) arm: rows of every seed x mix."""
    out: Dict[str, Any] = {"seeds": sorted(names), "per_mix": {}, "per_seed": {}}
    mix_mass = {}
    for mix in MIXES:
        rs = [r for r in rows if r["mix"] == mix]
        if len(rs) != len(names):
            out["per_mix"][mix] = None
            continue
        masses = [v for r in rs for v in r["mass"].values()]
        deaths = [v for r in rs for v in r["deaths"].values()]
        pw = [w for r in rs for w in r["per_world"].values()]
        dec = sum(w["decisions"] for w in pw)
        ge800 = sum(w["ge800"] for w in pw)
        died = [w for w in pw if w["len_at_death"] is not None]
        big_deaths = sum(1 for w in died if w["len_at_death"] >= BIG)
        boost_dec = sum(w["replaced"] + w["student_boost"] for w in pw)
        mix_mass[mix] = float(np.mean(masses))
        out["per_mix"][mix] = {
            "episodes": len(masses),
            "mass": float(np.mean(masses)),
            "survival": sum(1 for d in deaths if d == 0) / len(deaths),
            "deaths": int(sum(1 for d in deaths if d)),
            "deaths_L_ge_800": big_deaths,
            "deaths_L_lt_800": len(died) - big_deaths,
            "median_len_at_death": (
                float(np.median([w["len_at_death"] for w in died])) if died else None
            ),
            "hazard_per_1000_L_ge_800": (1000.0 * big_deaths / ge800) if ge800 else None,
            "hazard_per_1000_L_lt_800": (
                1000.0 * (len(died) - big_deaths) / (dec - ge800) if dec > ge800 else None
            ),
            "frac_decisions_L_ge_800": ge800 / dec if dec else None,
            "frac_decisions_L_ge_1000": sum(w["ge1000"] for w in pw) / dec if dec else None,
            "frac_decisions_L_ge_1200": sum(w["ge1200"] for w in pw) / dec if dec else None,
            "frac_over_cap": sum(w["over_cap"] for w in pw) / dec if dec else None,
            "replaced_frac_of_over_cap": (
                sum(w["replaced"] for w in pw) / max(1, sum(w["over_cap"] for w in pw))
            ),
            "blocked_frac_of_over_cap": (
                sum(w["blocked"] for w in pw) / max(1, sum(w["over_cap"] for w in pw))
            ),
            "boost_decision_frac": boost_dec / dec if dec else None,
            "median_max_len": float(np.median([w["max_len"] for w in pw])),
            "p90_max_len": float(np.percentile([w["max_len"] for w in pw], 90)),
            "causes": dict(sorted(Counter(w["cause"] for w in died).items())),
            "deaths_within_5_of_governor_boost": sum(
                1
                for w in died
                if w.get("since_governor_boost") is not None and w["since_governor_boost"] <= 4
            ),
            "deaths_boosted_on_fatal_decision": sum(
                1 for w in died if w.get("since_governor_boost") == 0
            ),
        }
    if len(mix_mass) == len(MIXES):
        out["mass_stratified"] = float(np.mean(list(mix_mass.values())))
    for name in sorted(names):
        rs = {r["mix"]: r for r in rows if r["name"] == name}
        if len(rs) != len(MIXES):
            continue
        out["per_seed"][name] = {
            "mass_stratified": float(
                np.mean([np.mean(list(rs[m]["mass"].values())) for m in MIXES])
            ),
            **{f"surv_{m}": _surv(rs[m]["deaths"]) for m in MIXES},
        }
    return out


def paired_scripted_delta(
    rows: Sequence[Mapping[str, Any]], off: Sequence[Mapping[str, Any]]
) -> Optional[Dict[str, Any]]:
    """Seed-world paired scripted survival / mass difference (cap - off)."""
    a = {(r["name"], k): v for r in rows if r["mix"] == "scripted" for k, v in r["deaths"].items()}
    b = {(r["name"], k): v for r in off if r["mix"] == "scripted" for k, v in r["deaths"].items()}
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    d = np.array([(a[k] == 0) - (b[k] == 0) for k in keys], dtype=np.float64)
    se = d.std(ddof=1) / np.sqrt(len(d)) if len(d) > 1 else float("nan")
    return {
        "pairs": len(keys),
        "delta_survival": float(d.mean()),
        "se": float(se),
        "lb90": float(d.mean() - 1.645 * se),
        "ub90": float(d.mean() + 1.645 * se),
        "rescued": int((d > 0).sum()),
        "lost": int((d < 0).sum()),
    }


def incumbent_summary() -> Dict[str, Any]:
    out: Dict[str, Any] = {"file": str(INCUMBENT_ROWS), "per_mix": {}}
    masses = []
    for line in INCUMBENT_ROWS.read_text().splitlines():
        r = json.loads(line)
        m = float(np.mean(list(r["mass"].values())))
        masses.append(m)
        out["per_mix"][r["mix"]] = {"mass": m, "survival": _surv(r["deaths"])}
    out["mass_stratified"] = float(np.mean(masses))
    return out


def off_identity(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """``off`` rows vs the existing python/python probe rows (mass, deaths, causes)."""
    checked, mismatches, missing = 0, [], []
    for r in rows:
        if r["cap"] != "off":
            continue
        ref = reference_row(r["name"], r["mix"])
        if ref is None or ref.get("student_sha256") != r["student_sha256"]:
            missing.append([r["name"], r["mix"]])
            continue
        checked += 1
        for k in ("mass", "deaths", "causes"):
            if {str(a): b for a, b in ref[k].items()} != {str(a): b for a, b in r[k].items()}:
                mismatches.append([r["name"], r["mix"], k])
    return {"rows_checked": checked, "mismatches": mismatches, "no_reference": missing}


def governor_off_noop(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Decisions where the governor at a cap changed nothing must not differ from off below
    the cap: here only the counters are checked (replaced == 0 whenever never over cap)."""
    bad = 0
    for r in rows:
        for w in r["per_world"].values():
            if w["over_cap"] == 0 and w["replaced"]:
                bad += 1
            if w["replaced"] + w["blocked"] > w["over_cap"]:
                bad += 1
    return {"violations": bad}


def verdict(arms: Mapping[str, Any], incumbent_mass: float) -> Dict[str, Any]:
    best_go, best_surv = None, None
    for cap, a in arms.items():
        if cap == "off" or a.get("per_mix", {}).get("scripted") is None:
            continue
        s = a["per_mix"]["scripted"]["survival"]
        mass = a.get("mass_stratified")
        if best_surv is None or s > best_surv[1]:
            best_surv = (cap, s)
        if s >= GO_SCRIPTED and mass is not None and mass >= incumbent_mass + MASS_MARGIN:
            if best_go is None or mass > best_go[1]:
                best_go = (cap, mass)
    if best_go:
        v = "GO"
    elif best_surv is None or best_surv[1] < NO_GO_SCRIPTED:
        v = "NO_GO"
    else:
        v = "INCONCLUSIVE"
    return {
        "verdict": v,
        "go_cap": best_go[0] if best_go else None,
        "best_scripted_cap": best_surv[0] if best_surv else None,
        "best_scripted_survival": best_surv[1] if best_surv else None,
        "rule": (
            f"GO if some Lc has pooled scripted survival >= {GO_SCRIPTED} and mix-stratified "
            f"mass >= incumbent + {MASS_MARGIN:.0f}; NO_GO if no Lc reaches scripted >= "
            f"{NO_GO_SCRIPTED}; else INCONCLUSIVE"
        ),
    }


def _arm_order(key: str) -> Tuple[int, int, str]:
    if key == "off":
        return (0, 0, "")
    cap, _, mode = key.partition("/")
    return (1 if not mode else 2, int(cap), mode)


def cmd_summarize(args: argparse.Namespace) -> int:
    rows = load_rows(args.root)
    inc = incumbent_summary()
    groups: Dict[str, Dict[str, Any]] = {}
    for g in sorted({group_of(r["name"]) for r in rows}):
        g_rows = [r for r in rows if group_of(r["name"]) == g and len(r["mass"]) == args.worlds]
        names = sorted({r["name"] for r in g_rows})
        caps = sorted({arm_key(r) for r in g_rows}, key=_arm_order)
        off_rows = [r for r in g_rows if r["cap"] == "off"]
        arms = {}
        for cap in caps:
            c_rows = [r for r in g_rows if arm_key(r) == cap]
            a = arm_summary(c_rows, sorted({r["name"] for r in c_rows}))
            if cap != "off" and off_rows:
                a["paired_scripted_vs_off"] = paired_scripted_delta(c_rows, off_rows)
            arms[cap] = a
        off = arms.get("off", {})
        for cap, a in arms.items():
            ms = a.get("mass_stratified")
            a["vs_incumbent_mass"] = None if ms is None else ms - inc["mass_stratified"]
            a["vs_off_mass"] = (
                None
                if ms is None or off.get("mass_stratified") is None
                else ms - off["mass_stratified"]
            )
            sc = (a.get("per_mix") or {}).get("scripted")
            a["scripted_vs_champion"] = None if sc is None else sc["survival"] - CHAMPION_SCRIPTED
            a["phase_r_guard_scripted"] = (
                None if sc is None else sc["survival"] >= CHAMPION_SCRIPTED - GUARD
            )
        groups[g] = {
            "seeds": names,
            "arms": arms,
            "verdict": verdict(arms, inc["mass_stratified"]),
        }
    summary = {
        "authority": "development diagnostic only (not a screen, not gate evidence, never served)",
        "protocol": {
            "worlds": args.worlds,
            "round_index": ROUND_INDEX,
            "mixes": list(MIXES),
            "horizon": 5000,
            "hero": "ego2s student, rowwise forward, advisory mask, no veto",
            "governor": "L > Lc: swap a normal action for its boost twin if advisory-safe",
            "speed_flags": "--v61-backend numba --anchors fast",
        },
        "incumbent_frp3_s12_v8": inc,
        "champion_a5_v8_scripted_census": CHAMPION_SCRIPTED,
        "lever4_prediction": LEVER4_PREDICTION,
        "groups": groups,
        "checks": {"off_identity": off_identity(rows), "counters": governor_off_noop(rows)},
        "git": _git(),
    }
    out = args.root / "summary.json"
    out.write_text(json.dumps(summary, indent=1) + "\n")
    print_table(summary)
    print(f"wrote {out}")
    return 0


def print_table(summary: Mapping[str, Any]) -> None:
    inc = summary["incumbent_frp3_s12_v8"]
    print(
        f"incumbent frp3-s12+v8: mass {inc['mass_stratified']:.1f} "
        + " ".join(f"{m} {inc['per_mix'][m]['survival']:.3f}" for m in MIXES)
    )
    for g, gd in summary["groups"].items():
        print(f"== {g} {gd['seeds']}  verdict {gd['verdict']['verdict']}")
        print(
            "cap   mass   dOff  dInc | surv fro  scr   mix | scr d L>=800 / all"
            " | hzL>=800 | fracL>=800 | boostdec | med maxlen | scr paired vs off"
        )
        for cap, a in gd["arms"].items():
            pm = a.get("per_mix", {})
            if any(pm.get(m) is None for m in MIXES):
                print(f"{cap:>5} incomplete")
                continue
            sc = pm["scripted"]
            pr = a.get("paired_scripted_vs_off")
            hz = sc["hazard_per_1000_L_ge_800"]
            print(
                f"{cap:>5} {a['mass_stratified']:6.1f} {a['vs_off_mass'] or 0:+6.1f}"
                f" {a['vs_incumbent_mass']:+6.1f} | "
                + " ".join(f"{pm[m]['survival']:.3f}" for m in MIXES)
                + f" | {sc['deaths_L_ge_800']:3d}/{sc['deaths']:3d}"
                + f" | {hz if hz is None else round(hz, 3)}"
                + f" | {sc['frac_decisions_L_ge_800']:.3f}"
                + f" | {sc['boost_decision_frac']:.3f}"
                + f" | {sc['median_max_len']:.0f}"
                + (
                    f" | {pr['delta_survival']:+.3f} [{pr['lb90']:+.3f},{pr['ub90']:+.3f}]"
                    if pr
                    else ""
                )
            )


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run governed dev probes (one process, sequential jobs)")
    r.add_argument("--ckpt", action="append", required=True, help="NAME=PATH (repeatable)")
    r.add_argument("--caps", default="off,800,900,1000,1100")
    r.add_argument("--mode", default="twin", choices=GOVERNOR_MODES)
    r.add_argument("--mixes", default=",".join(MIXES))
    r.add_argument("--worlds", type=int, default=WORLDS)
    r.add_argument("--v61-backend", default="numba", choices=("python", "numba"))
    r.add_argument("--anchors", default="fast", choices=("python", "fast"))
    r.add_argument("--out", type=Path, required=True)
    s = sub.add_parser("summarize")
    s.add_argument("--root", type=Path, default=OUT_ROOT)
    s.add_argument("--worlds", type=int, default=WORLDS)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    return {"run": cmd_run, "summarize": cmd_summarize}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
