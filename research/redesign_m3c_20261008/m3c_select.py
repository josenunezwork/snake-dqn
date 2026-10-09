#!/usr/bin/env python3
"""M3-C arm selection and Phase R entry gates (PREREGISTRATION_M3C_DRAFT.md section 6).

Reads probe_watch result files ``<probes>/<run>__<ckpt>__<veto>.jsonl`` (runs named
``<arm>_s<seed>``), the incumbent's dev probe ``<probes>/incumbent__dev__v8.jsonl``
(``dev_probe.py --arm baseline --veto v8`` on the same worlds) and, per run,
``<runs>/<run>/summary.json``, then applies the pre-registered rules:

* a seed is **eligible** iff its summary says ``halt is None`` and ``transitions >= target``
  and its probe files cover every mix; the arms are compared on the **common seeds** (seeds
  eligible in every arm); fewer than MIN_SEEDS (4) common seeds -> nothing is selected;
* score(arm, veto) = mean over the common seeds of the mix-stratified mean mass of the
  ``final`` checkpoint on the dev worlds (per mix: the worlds every eligible file and the
  incumbent share);
* primary arm = argmax no-veto score (ties: plan order); it ENTERS the primary Phase R iff
  score_nv >= incumbent_v8 - G (G = 50) AND, in every mix, the world-paired dev survival
  difference (candidate: share of its seeds alive at H5000 on the world; incumbent: alive or
  not) has mean >= -0.03 - 1.28 SE (a noise-aware version of the Phase R survival clauses);
* secondary arm = argmax v8 score; it ENTERS the secondary Phase R iff
  score_v8 - incumbent_v8 >= SECONDARY_MARGIN (30) AND the same survival check (v8);
* contrasts (descriptive): B - A, D - B, B - L, E - B, seed-paired, df-4 t interval.

  ./venv/bin/python research/redesign_m3c_20261008/m3c_select.py --probes DIR --runs RUNS \\
      --transitions 20000000 --arms A,B,D,L
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

MIXES = ("frozen", "scripted", "mixed")
SEEDS = (0, 1, 2, 3, 4)
G = 50.0
SECONDARY_MARGIN = 30.0
SURVIVAL_TOL = 0.03
SURVIVAL_Z = 1.28  # one-sided 90% normal quantile for the paired survival difference
MIN_SEEDS = 4
T90 = {4: 1.533, 3: 1.638}  # one-sided 90% t quantiles by df (df 4 matches m3b_slope)
# Pre-registered descriptive contrasts (seed-paired: the arms share seeds and world streams):
# B - A = the metric-aligned reward; D - B = eval-matched worlds; B - L = the short anchor;
# E - B = the boost fix.
CONTRASTS = (("B", "A"), ("D", "B"), ("B", "L"), ("E", "B"))

Probe = Dict[str, Dict[str, Tuple[float, float]]]  # mix -> world -> (mass, deaths)


def load_probe(path: Path) -> Probe:
    """{mix: {world_seed: (mass, deaths)}} from a dev_probe output file (last row per mix)."""
    out: Probe = {}
    if not path.exists():
        return out
    for line in path.read_text().splitlines():
        try:
            row = json.loads(line)
        except ValueError:
            continue
        deaths = row.get("deaths", {})
        out[row["mix"]] = {
            str(k): (float(v), float(deaths.get(str(k), 0) or 0)) for k, v in row["mass"].items()
        }
    return out


def common_worlds(probes: Sequence[Probe]) -> Dict[str, List[str]]:
    """Per mix, the worlds shared by every probe that HAS that mix (a probe missing a mix
    makes its seed incomplete; it does not empty the mix for everyone)."""
    out = {}
    for mix in MIXES:
        sets = [set(p[mix]) for p in probes if mix in p]
        out[mix] = sorted(set.intersection(*sets)) if sets else []
    return out


def stratified(probe: Probe, worlds: Dict[str, List[str]]) -> Optional[float]:
    means = []
    for mix in MIXES:
        ws = worlds.get(mix) or []
        if not ws or mix not in probe or any(w not in probe[mix] for w in ws):
            return None
        means.append(sum(probe[mix][w][0] for w in ws) / len(ws))
    return sum(means) / len(means)


def survival(probes: Sequence[Probe], worlds: Dict[str, List[str]]) -> Dict[str, float]:
    out = {}
    for mix in MIXES:
        alive = [p[mix][w][1] == 0 for p in probes for w in worlds.get(mix) or []]
        out[mix] = sum(alive) / len(alive) if alive else float("nan")
    return out


def survival_check(
    cand: Sequence[Probe], inc: Probe, worlds: Dict[str, List[str]]
) -> Dict[str, Dict[str, float]]:
    """Per mix: world-paired difference (candidate seeds' alive share - incumbent alive)."""
    out = {}
    for mix in MIXES:
        ws = worlds.get(mix) or []
        d = [
            sum(p[mix][w][1] == 0 for p in cand) / len(cand) - float(inc[mix][w][1] == 0)
            for w in ws
        ]
        n = len(d)
        m = sum(d) / n if n else float("nan")
        sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1)) if n > 1 else float("nan")
        se = sd / math.sqrt(n) if n > 1 else float("nan")
        bar = -SURVIVAL_TOL - SURVIVAL_Z * se
        out[mix] = {"mean": m, "se": se, "bar": bar, "ok": bool(n and m >= bar)}
    return out


def t_interval(xs: Sequence[float]) -> Dict[str, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = T90.get(n - 1, float("nan"))
    return {"mean": m, "se": se, "lb90": m - t * se, "ub90": m + t * se, "n": n}


def _sha(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def seed_eligible(runs: Path, run: str, target: int) -> Tuple[bool, str]:
    path = runs / run / "summary.json"
    if not path.exists():
        return False, "no summary.json"
    s = json.loads(path.read_text())
    if s.get("halt") is not None:
        return False, f"halt: {s['halt']}"
    if int(s.get("transitions", 0)) < int(target):
        return False, f"transitions {s.get('transitions')} < {target}"
    if not (runs / run / "final.pth").exists():
        return False, "no final.pth"
    return True, "ok"


def select(
    probes_dir: Path,
    arms: Sequence[str],
    runs: Path,
    target: int,
    ckpt: str = "final",
) -> Dict[str, object]:
    files: Dict[tuple, Probe] = {}
    elig: Dict[tuple, Tuple[bool, str]] = {}
    for arm in arms:
        for s in SEEDS:
            elig[(arm, s)] = seed_eligible(runs, f"{arm}_s{s}", target)
            for veto in ("none", "v8"):
                files[(arm, s, veto)] = load_probe(probes_dir / f"{arm}_s{s}__{ckpt}__{veto}.jsonl")
    inc = load_probe(probes_dir / "incumbent__dev__v8.jsonl")
    usable = [p for (a, s, _), p in files.items() if p and elig[(a, s)][0]]
    worlds = common_worlds([inc] + usable) if inc else {m: [] for m in MIXES}
    # a seed counts only where every arm has it eligible AND fully probed (both vetoes)
    probed = {
        (a, s): elig[(a, s)][0]
        and all(stratified(files[(a, s, v)], worlds) is not None for v in ("none", "v8"))
        for a in arms
        for s in SEEDS
    }
    common = [s for s in SEEDS if all(probed[(a, s)] for a in arms)]
    use = common if len(common) >= MIN_SEEDS else []
    scores: Dict[str, Dict[str, object]] = {}
    for arm in arms:
        row: Dict[str, object] = {
            "seeds": {s: elig[(arm, s)][1] for s in SEEDS},
            "final_sha256": {
                s: _sha(runs / f"{arm}_s{s}" / "final.pth")
                for s in SEEDS
                if (runs / f"{arm}_s{s}" / "final.pth").exists()
            },
            "eligible": [s for s in SEEDS if probed[(arm, s)]],
        }
        for veto in ("none", "v8"):
            per = [stratified(files[(arm, s, veto)], worlds) for s in use]
            complete = bool(use) and all(p is not None for p in per)
            row[veto] = {
                "per_seed": per,
                "complete": complete,
                "score": (sum(per) / len(per)) if complete else None,
                "survival": (
                    survival([files[(arm, s, veto)] for s in use], worlds) if complete else None
                ),
                "survival_check": (
                    survival_check([files[(arm, s, veto)] for s in use], inc, worlds)
                    if complete and inc
                    else None
                ),
            }
        scores[arm] = row
    inc_v8 = stratified(inc, worlds) if inc else None
    inc_surv = survival([inc], worlds) if inc_v8 is not None else None

    def best(veto: str) -> Optional[str]:
        ok = [a for a in arms if scores[a][veto]["complete"]]
        return max(ok, key=lambda a: (scores[a][veto]["score"], -arms.index(a))) if ok else None

    def gate(veto: str, arm: Optional[str], ok_score) -> Dict[str, object]:
        if inc_v8 is None:
            return {"arm": arm, "enter": False, "reason": "incumbent dev probe missing/incomplete"}
        if arm is None:
            return {"arm": None, "enter": False, "reason": "no eligible complete arm"}
        sc = scores[arm][veto]["score"]
        surv = scores[arm][veto]["survival"]
        surv_ok = all(v["ok"] for v in scores[arm][veto]["survival_check"].values())
        return {
            "arm": arm,
            "score": sc,
            "incumbent_v8": inc_v8,
            "delta": sc - inc_v8,
            "survival": surv,
            "incumbent_survival": inc_surv,
            "score_ok": ok_score(sc),
            "survival_ok": surv_ok,
            "seeds_used": use,
            "enter": bool(ok_score(sc) and surv_ok),
        }

    primary = gate("none", best("none"), lambda sc: sc >= inc_v8 - G)
    secondary = gate("v8", best("v8"), lambda sc: sc - inc_v8 >= SECONDARY_MARGIN)
    contrasts: Dict[str, Dict[str, object]] = {}
    for hi, lo in CONTRASTS:
        if hi not in arms or lo not in arms:
            continue
        for veto in ("none", "v8"):
            a, b = scores[hi][veto]["per_seed"], scores[lo][veto]["per_seed"]
            if a and b and all(x is not None for x in a + b):
                diffs = [a[i] - b[i] for i in range(len(a))]
                contrasts.setdefault(f"{hi}-{lo}", {})[veto] = t_interval(diffs)
    return {
        "checkpoint": ckpt,
        "target_transitions": target,
        "common_seeds": common,
        "seeds_used": use,
        "worlds_per_mix": {m: len(v) for m, v in worlds.items()},
        "incumbent_v8": inc_v8,
        "incumbent_survival": inc_surv,
        "scores": scores,
        "primary": primary,
        "secondary": secondary,
        "contrasts": contrasts,
        "rules": {
            "G": G,
            "secondary_margin": SECONDARY_MARGIN,
            "survival_tolerance": SURVIVAL_TOL,
            "survival_z": SURVIVAL_Z,
            "min_seeds": MIN_SEEDS,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, required=True)
    ap.add_argument("--runs", type=Path, required=True, help="pulled run dirs (summary.json)")
    ap.add_argument("--transitions", type=int, required=True)
    ap.add_argument("--arms", default="A,B,D,L")
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if re.fullmatch(r"[A-Za-z0-9]+", a)]
    out = select(args.probes, arms, args.runs, args.transitions, args.ckpt)
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
