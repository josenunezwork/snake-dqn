#!/usr/bin/env python3
"""M4 arm selection and Phase R entry gates (PREREGISTRATION_M4.md section 4). DRAFT.

M3-C's rules (``research/redesign_m3c_20261008/m3c_select.py``: score bar, noise-aware survival
check, common seeds, seed-paired contrasts) for M4-A's arms C / D / S / B8 x seeds 0-2, with
three changes forced by how M4-A runs:

* **Several result directories.** Arm C (pod 1) and D / S / B8 (pod 2) land in different probe
  and run directories: ``--probes`` and ``--runs`` are repeatable. A run (``<arm>_s<seed>``)
  must be found in exactly one ``--runs`` directory (``<dir>/<run>/summary.json``; pass each
  runner's ``attempt<NN>/out``) and its probe file in exactly one ``--probes`` directory, else
  the seed is ineligible ("ambiguous").
* **No-veto final probes only.** The final watchers probe ``--vetoes none`` (32 worlds per mix,
  round 52, ``--ego2s-forward rowwise --ego2s-mask advisory``). Eligibility needs only the
  no-veto probe. The secondary (v8) gate is evaluated only when EVERY compared arm has complete
  v8 final probes on the used seeds; otherwise it is "not entered: no v8 final probes".
* **Exact world set.** The incumbent reference (``--incumbent``, default M3-C's
  ``incumbent__dev__v8.jsonl``: ``dev_probe.py --arm baseline --veto v8``, round 52, 32 worlds
  per mix) must have, per mix, EXACTLY the world set of every final probe file and the same
  round index; a mismatch selects nothing (pass a matching ``--incumbent`` file).

A probe row must carry the run's ``final.pth`` sha256 (``student_sha256``), ``veto``, round
index and the Phase R hero modes (no veto: ``ego2s_forward == "rowwise"`` and ``ego2s_mask ==
"advisory"``), else the seed is ineligible.

Rules (as M3-C, 3 seeds):

* a seed is **eligible** iff ``summary.json`` has ``halt is None`` and ``transitions >=
  target``, ``final.pth`` exists and its no-veto final probe is complete and valid;
* arms are compared on the **common seeds** (eligible in every arm); fewer than MIN_SEEDS (2)
  -> nothing is selected;
* score(arm, veto) = mean over the common seeds of the mix-stratified mean mass;
* primary arm = argmax no-veto score (ties: plan order C, D, S, B8); it ENTERS iff all 3 of its
  seeds are eligible AND score_nv >= incumbent_v8 - G (G = 50) AND, in every mix, the
  world-paired survival difference has mean >= -0.03 - 1.28 SE;
* secondary arm = argmax v8 score; ENTERS iff all 3 seeds eligible AND score_v8 - incumbent_v8
  >= +30 AND the same survival check (v8);
* contrasts (descriptive, seed-paired, df = n - 1 t interval): D - C (on-policy labels),
  S - D (training-time shield), B8 - D (v8 features as inputs).

  ./venv/bin/python research/redesign_m4_20261013/m4_select.py \\
      --probes .../redesign-m4a-probes-20261012 --probes .../redesign-m4a2-probes-20261012 \\
      --runs .../redesign-m4a-gpu-20261012/attempt00/out \\
      --runs .../redesign-m4a2-gpu-20261012/attempt00/out \\
      --transitions 20000000 --arms C,D,S,B8 --out selection.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3c_20261008 import m3c_select as base  # noqa: E402
from research.redesign_m4_20261013 import rule_spec_m4 as rs  # noqa: E402

MIXES = base.MIXES
SEEDS = rs.TRAINING_SEEDS  # the Phase R seeds (five; PREREGISTRATION_M4.md section 2)
ARMS = ("C", "D", "S", "B8")
G = base.G
SECONDARY_MARGIN = base.SECONDARY_MARGIN
SURVIVAL_TOL = base.SURVIVAL_TOL
SURVIVAL_Z = base.SURVIVAL_Z
MIN_SEEDS = 2
T90 = {1: 3.078, 2: 1.886}  # one-sided 90% t quantiles by df
CONTRASTS = (("D", "C"), ("S", "D"), ("B8", "D"))
ROUND_INDEX = 52
INCUMBENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m3c-probes-20261008-recovery/"
    "incumbent__dev__v8.jsonl"
)
TARGET = 20_000_000

Probe = base.Probe


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def rows_of(path: Path) -> List[Dict[str, Any]]:
    out = []
    for line in Path(path).read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def find_one(dirs: Sequence[Path], rel: str) -> Tuple[Optional[Path], str]:
    hits = [Path(d) / rel for d in dirs if (Path(d) / rel).exists()]
    if not hits:
        return None, "missing"
    if len(hits) > 1:
        return None, "ambiguous: " + ", ".join(str(h) for h in hits)
    return hits[0], "ok"


def world_sets(path: Path) -> Dict[str, Tuple[int, frozenset]]:
    """{mix: (round_index, worlds)} of a probe file (last row per mix, as ``load_probe``)."""
    out = {}
    for row in rows_of(path):
        out[row["mix"]] = (row.get("round_index"), frozenset(str(w) for w in row["mass"]))
    return out


def probe_problem(path: Path, veto: str, want_sha: str, round_index: int) -> Optional[str]:
    """None if every row of a final probe file is valid for this run; else the reason."""
    rows = rows_of(path)
    if {r.get("mix") for r in rows} != set(MIXES):
        return f"probe {veto} incomplete (mixes {sorted({str(r.get('mix')) for r in rows})})"
    for r in rows:
        if r.get("student_sha256") != want_sha:
            return f"probe {veto} not made with this run's final.pth"
        if r.get("veto") != veto or r.get("round_index") != round_index:
            return f"probe {veto} has veto {r.get('veto')} / round {r.get('round_index')}"
        if veto == "none" and (
            r.get("ego2s_forward") != "rowwise" or r.get("ego2s_mask") != "advisory"
        ):
            return "probe none not in Phase R hero modes (rowwise forward, advisory mask)"
    return None


def t_interval(xs: Sequence[float]) -> Dict[str, float]:
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1)) if n > 1 else float("nan")
    se = sd / math.sqrt(n) if n > 1 else float("nan")
    t = T90.get(n - 1, float("nan"))
    return {"mean": m, "se": se, "lb90": m - t * se, "ub90": m + t * se, "n": n}


def select(
    probes_dirs: Sequence[Path],
    arms: Sequence[str],
    runs_dirs: Sequence[Path],
    target: int,
    incumbent: Path = INCUMBENT,
    ckpt: str = "final",
) -> Dict[str, Any]:
    arms = list(arms)
    inc_path = Path(incumbent)
    inc = base.load_probe(inc_path)
    inc_sets = world_sets(inc_path) if inc_path.exists() else {}
    inc_ok = set(inc_sets) == set(MIXES) and all(r[0] == ROUND_INDEX for r in inc_sets.values())
    reasons: Dict[Tuple[str, int], str] = {}
    files: Dict[Tuple[str, int, str], Probe] = {}
    run_paths: Dict[Tuple[str, int], Optional[str]] = {}
    shas: Dict[Tuple[str, int], str] = {}
    mismatch: List[Dict[str, Any]] = []
    for arm in arms:
        for s in SEEDS:
            run = f"{arm}_s{s}"
            reason = "ok"
            run_dir, how = find_one(runs_dirs, f"{run}/summary.json")
            run_paths[(arm, s)] = str(run_dir.parent) if run_dir else None
            if run_dir is None:
                reason = "no summary.json" if how == "missing" else f"run {how}"
            else:
                d = run_dir.parent
                if (d / "final.pth").exists():
                    shas[(arm, s)] = sha256_file(d / "final.pth")
                ok, why = base.seed_eligible(d.parent, run, target)
                if not ok:
                    reason = why
            for veto in ("none", "v8"):
                path, how = find_one(probes_dirs, f"{run}__{ckpt}__{veto}.jsonl")
                if path is None:
                    if veto == "none" and reason == "ok":
                        reason = "no final no-veto probe" if how == "missing" else f"probe {how}"
                    continue
                if (arm, s) not in shas:
                    continue
                bad = probe_problem(path, veto, shas[(arm, s)], ROUND_INDEX)
                if bad:
                    if veto == "none" and reason == "ok":
                        reason = bad
                    continue
                if inc_ok:
                    for mix, (_, ws) in world_sets(path).items():
                        if ws != inc_sets[mix][1]:
                            mismatch.append({"file": str(path), "mix": mix})
                files[(arm, s, veto)] = base.load_probe(path)
            reasons[(arm, s)] = reason
    worlds = {m: sorted(inc_sets[m][1]) for m in MIXES} if inc_ok else {m: [] for m in MIXES}
    eligible = {k: v == "ok" and (k[0], k[1], "none") in files for k, v in reasons.items()}
    common = [s for s in SEEDS if all(eligible[(a, s)] for a in arms)]
    world_problem = None
    if not inc_ok:
        world_problem = f"incumbent reference {inc_path} missing / incomplete / not round 52"
    elif mismatch:
        world_problem = "final-probe world set != incumbent world set: pass a matching --incumbent"
    use = common if len(common) >= MIN_SEEDS and world_problem is None else []
    scores: Dict[str, Dict[str, Any]] = {}
    for arm in arms:
        row: Dict[str, Any] = {
            "seeds": {s: reasons[(arm, s)] for s in SEEDS},
            "run_dirs": {s: run_paths[(arm, s)] for s in SEEDS},
            "final_sha256": {s: shas[(arm, s)] for s in SEEDS if (arm, s) in shas},
            "eligible": [s for s in SEEDS if eligible[(arm, s)]],
        }
        for veto in ("none", "v8"):
            have = all((arm, s, veto) in files for s in use)
            per = [base.stratified(files[(arm, s, veto)], worlds) for s in use] if have else []
            complete = bool(use) and have and all(p is not None for p in per)
            probes = [files[(arm, s, veto)] for s in use] if complete else []
            row[veto] = {
                "per_seed": per,
                "complete": complete,
                "score": (sum(per) / len(per)) if complete else None,
                "survival": base.survival(probes, worlds) if complete else None,
                "survival_check": base.survival_check(probes, inc, worlds) if complete else None,
            }
        scores[arm] = row
    inc_v8 = base.stratified(inc, worlds) if inc_ok else None
    inc_surv = base.survival([inc], worlds) if inc_v8 is not None else None

    def best(veto: str) -> Optional[str]:
        ok = [a for a in arms if scores[a][veto]["complete"]]
        return max(ok, key=lambda a: (scores[a][veto]["score"], -arms.index(a))) if ok else None

    def gate(veto: str, ok_score) -> Dict[str, Any]:
        if world_problem is not None:
            return {"arm": None, "enter": False, "reason": world_problem}
        if not use:
            return {"arm": None, "enter": False, "reason": f"fewer than {MIN_SEEDS} common seeds"}
        if veto == "v8" and not all(scores[a]["v8"]["complete"] for a in arms):
            return {"arm": None, "enter": False, "reason": "no complete v8 final probes"}
        arm = best(veto)
        if arm is None:
            return {"arm": None, "enter": False, "reason": "no eligible complete arm"}
        sc = scores[arm][veto]["score"]
        surv_ok = all(v["ok"] for v in scores[arm][veto]["survival_check"].values())
        all_seeds = scores[arm]["eligible"] == list(SEEDS)
        return {
            "arm": arm,
            "score": sc,
            "incumbent_v8": inc_v8,
            "delta": sc - inc_v8,
            "survival": scores[arm][veto]["survival"],
            "incumbent_survival": inc_surv,
            "score_ok": bool(ok_score(sc)),
            "survival_ok": surv_ok,
            "all_seeds_eligible": all_seeds,
            "seeds_used": use,
            "enter": bool(ok_score(sc) and surv_ok and all_seeds),
        }

    primary = gate("none", lambda sc: sc >= inc_v8 - G)
    secondary = gate("v8", lambda sc: sc - inc_v8 >= SECONDARY_MARGIN)
    contrasts: Dict[str, Dict[str, Any]] = {}
    for hi, lo in CONTRASTS:
        if hi not in arms or lo not in arms:
            continue
        for veto in ("none", "v8"):
            a, b = scores[hi][veto]["per_seed"], scores[lo][veto]["per_seed"]
            if a and b and len(a) == len(b) and len(a) > 1:
                contrasts.setdefault(f"{hi}-{lo}", {})[veto] = t_interval(
                    [a[i] - b[i] for i in range(len(a))]
                )
    return {
        "schema": "redesign-m4-select/v1",
        "checkpoint": ckpt,
        "target_transitions": target,
        "arms": arms,
        "probes_dirs": [str(p) for p in probes_dirs],
        "runs_dirs": [str(p) for p in runs_dirs],
        "incumbent_file": {
            "path": str(inc_path),
            "sha256": sha256_file(inc_path) if inc_path.exists() else None,
        },
        "world_check": {"ok": world_problem is None, "problem": world_problem, "files": mismatch},
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
            "round_index": ROUND_INDEX,
        },
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--probes", type=Path, action="append", required=True)
    ap.add_argument(
        "--runs", type=Path, action="append", required=True, help="<runner>/attempt*/out"
    )
    ap.add_argument("--transitions", type=int, required=True)
    ap.add_argument("--arms", default=",".join(ARMS))
    ap.add_argument("--incumbent", type=Path, default=INCUMBENT)
    ap.add_argument("--ckpt", default="final")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    arms = [a for a in args.arms.split(",") if re.fullmatch(r"[A-Za-z0-9]+", a)]
    out = select(args.probes, arms, args.runs, args.transitions, args.incumbent, args.ckpt)
    text = json.dumps(out, indent=1)
    print(text)
    if args.out:
        if args.out.exists():
            raise SystemExit(f"refusing: {args.out} exists (create-only)")
        args.out.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
