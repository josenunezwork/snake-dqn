#!/usr/bin/env python3
"""Run the PRE-REGISTERED M2 non-inferiority check (``ni_spec.py``, ``PREREGISTRATION.md``).

Subcommands (all create-only under ``--out``):

* ``intent --student CKPT``: write ``intent.json`` (student sha256, NI seeds, rule, git)
  BEFORE the first NI episode. Every later ``run`` refuses a different student.
* ``run --arm A --mix M``: one (arm, mix) batch of the 48 NI worlds through
  ``run_simd_eval(sim_engine="grid", vector61=True)`` at H5000 on the pinned profile:

  - ``baseline``: frp3-s12, no veto, rowwise (bit-exact) forwards;
  - ``student``: the ego2s student, no veto (``hero_ego2s``), CPU forward;
  - ``baseline_v8`` / ``student_v8``: the same with the v8 veto (lambda 8): REPORTED ONLY.

* ``decide``: apply ``ni_spec.decide`` to ``student`` vs ``baseline`` and write
  ``verdict.json`` (plus the reported-only v8 comparison when present).

One thread, ``nice -n 10``, no slot locks; the AC / lid / thermal guard is checked before
each batch. Development check, not gate evidence.
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
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m2_20261008 import ni_spec, ni_spec_m2b  # noqa: E402
from research.redesign_scope_20261007 import grid_h5000_identity as gid  # noqa: E402

SPECS = {"m2": (ni_spec, 2), "m2b": (ni_spec_m2b, 12)}
SPEC = ni_spec  # set by --spec (default m2, the original M2 check)
FINAL_ROUND = 2

ARMS = {
    "baseline": {"ego2s": False, "veto": False},
    "student": {"ego2s": True, "veto": False},
    "baseline_v8": {"ego2s": False, "veto": True},
    "student_v8": {"ego2s": True, "veto": True},
}


def _sha(path: str) -> str:
    from research.apex_safety_20260926 import dev_screen

    return dev_screen.sha256_file(Path(path))


def _student_meta(path: str) -> Dict[str, Any]:
    import torch

    meta = torch.load(path, map_location="cpu", weights_only=False).get("meta", {})
    return json.loads(json.dumps(meta, default=str))


def cmd_intent(args: argparse.Namespace) -> int:
    out = Path(args.out)
    meta = _student_meta(args.student)
    if FINAL_ROUND not in meta.get("rounds", []):
        raise SystemExit(f"the pre-registered student is the fit on round {FINAL_ROUND}")
    if SPEC is ni_spec_m2b:
        from src.model.ego2s_network import OBS_SPEC_B, obs_spec_of

        if list(meta.get("rounds", [])) != list(ni_spec_m2b.DATA_ROUNDS):
            raise SystemExit(f"the M2b student must be the fit on rounds {ni_spec_m2b.DATA_ROUNDS}")
        if obs_spec_of(args.student) != OBS_SPEC_B:
            raise SystemExit("the M2b student must be an ego2s-b checkpoint")
    out.mkdir(parents=True, exist_ok=True)
    intent = {
        "schema_version": SPEC.SCHEMA,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git": gid._git(),
        "student": {
            "path": str(Path(args.student).resolve()),
            "sha256": _sha(args.student),
            "meta": meta,
        },
        "baseline": {"name": SPEC.FRP3_S12[0], "sha256": SPEC.FRP3_S12[1]},
        "rule": {
            "margin": SPEC.MARGIN,
            "confidence": SPEC.CONFIDENCE,
            "worlds_per_mix": SPEC.WORLDS_PER_MIX,
            "mixes": list(SPEC.MIXES),
            "namespace": SPEC.NAMESPACE,
        },
        "ni_seeds": SPEC.ni_seeds(),
        "grid_sim_jit": gid._grid_jit(),
    }
    with (out / "intent.json").open("x") as stream:
        json.dump(intent, stream, indent=1, sort_keys=True)
    print(json.dumps({"student_sha256": intent["student"]["sha256"]}))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from research.redesign_m2_20261008.gen_data import guard_ok
    from src.simd_env import eval_engine as ee

    out = Path(args.out)
    intent = json.loads((out / "intent.json").read_text())
    path = out / f"{args.arm}-{args.mix}.json"
    if path.exists():
        print(f"refusing: {path} exists (create-only)", file=sys.stderr)
        return 2
    problems = guard_ok()
    if problems:
        print(json.dumps({"refused": problems}), flush=True)
        return 3
    arm = ARMS[args.arm]
    student = intent["student"]["path"]
    if arm["ego2s"] and _sha(student) != intent["student"]["sha256"]:
        raise SystemExit("student checkpoint differs from intent.json")
    import torch

    torch.set_num_threads(1)
    ctx = gid._context(1)
    seeds = SPEC.ni_seeds()
    if seeds != intent["ni_seeds"]:
        raise SystemExit("NI seeds differ from intent.json")
    rows = {
        int(r["world_seed"]): [ctx["lookup"][s["member_sha256"]] for s in r["slots"]]
        for r in ctx["ds"]._design_rows(seeds)
        if r["mix"] == args.mix
    }
    kwargs: Dict[str, Any] = {}
    if arm["veto"]:
        kwargs.update(hero_safety_veto="v8", hero_safety_veto_lambda=8.0)
    if arm["ego2s"]:
        kwargs["hero_ego2s"] = student
    started = time.perf_counter()
    records = ee.run_simd_eval(
        ctx["lookup"][SPEC.FRP3_S12[1]],
        rows[seeds[0]],
        SPEC.HORIZON,
        seeds,
        profile=ctx["profile"],
        opponent_specs_by_world=rows,
        mix_id=args.mix,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        **kwargs,
    )
    payload = {
        "arm": args.arm,
        "mix": args.mix,
        "git": gid._git(),
        "wall_seconds": time.perf_counter() - started,
        "student_sha256": intent["student"]["sha256"] if arm["ego2s"] else None,
        "records": [gid.normalize(r) for r in records],
    }
    with path.open("x") as stream:
        json.dump(payload, stream, sort_keys=True)
    masses = [r["mass_integral"] for r in records]
    print(json.dumps({"arm": args.arm, "mix": args.mix, "mean": sum(masses) / len(masses)}))
    return 0


def _load(out: Path, arm: str) -> Dict[str, Dict[str, Any]]:
    payloads = {}
    for mix in SPEC.MIXES:
        path = out / f"{arm}-{mix}.json"
        if path.exists():
            payloads[mix] = json.loads(path.read_text())
    return payloads


def _masses(payloads: Dict[str, Dict[str, Any]]) -> Dict[str, Dict[int, float]]:
    return {
        mix: {int(r["seed"]): float(r["mass_integral"]) for r in p["records"]}
        for mix, p in payloads.items()
    }


def audit(out: Path, intent: Dict[str, Any]) -> List[str]:
    """Provenance problems that must block a verdict (empty list = ok)."""
    problems: List[str] = []
    sha = intent["student"]["sha256"]
    seeds = sorted(SPEC.ni_seeds())
    commits, digests = set(), set()
    for arm in ("student", "baseline"):
        payloads = _load(out, arm)
        for mix in SPEC.MIXES:
            p = payloads.get(mix)
            if p is None:
                problems.append(f"missing {arm}-{mix}.json")
                continue
            if p.get("arm") != arm or p.get("mix") != mix:
                problems.append(f"{arm}-{mix}.json holds arm={p.get('arm')} mix={p.get('mix')}")
            if (p.get("student_sha256") == sha) != (arm == "student"):
                problems.append(f"{arm}-{mix}.json student_sha256 mismatch")
            if p["git"]["dirty_paths"].strip():
                problems.append(f"{arm}-{mix}.json ran on a dirty tree")
            commits.add(p["git"]["commit"])
            recs = p["records"]
            if sorted(int(r["seed"]) for r in recs) != seeds:
                problems.append(f"{arm}-{mix}.json seeds differ from the NI seeds")
            for r in recs:
                digests.add(r.get("evaluation_profile_digest"))
                mix_id = r.get("world_identity", {}).get("mix_id")
                if mix_id != mix:
                    problems.append(f"{arm}-{mix}.json record mix_id {mix_id}")
                    break
                hero = r.get("ego2s_hero")
                if arm == "student" and (
                    not hero or hero.get("sha256") != sha or hero.get("veto") is not None
                ):
                    problems.append(f"{arm}-{mix}.json record not played by the student")
                    break
                if arm == "baseline" and hero is not None:
                    problems.append(f"{arm}-{mix}.json baseline record played by a student")
                    break
    if len(commits) > 1:
        problems.append(f"arms ran on different commits {sorted(commits)}")
    if len(digests) != 1:
        problems.append(f"profile digests differ {sorted(map(str, digests))}")
    return problems


def cmd_decide(args: argparse.Namespace) -> int:
    out = Path(args.out)
    intent = json.loads((out / "intent.json").read_text())
    problems = audit(out, intent)
    if problems:
        print(json.dumps({"refused": problems}, indent=1))
        return 2
    student, baseline = _masses(_load(out, "student")), _masses(_load(out, "baseline"))
    verdict = SPEC.decide(student, baseline)
    verdict["identical_arms_flag"] = all(student[m] == baseline[m] for m in SPEC.MIXES)
    reported: Dict[str, Any] = {}
    sv8, bv8 = _masses(_load(out, "student_v8")), _masses(_load(out, "baseline_v8"))
    if sv8 and bv8:
        reported["student_v8_vs_I"] = SPEC.decide(sv8, bv8)
        reported["student_v8_vs_I"].pop("verdict")
    for arm in ARMS:
        m = _masses(_load(out, arm))
        vals = [v for mix in m.values() for v in mix.values()]
        if vals:
            reported[f"mean_{arm}"] = sum(vals) / len(vals)
    verdict["reported_not_gated"] = reported
    verdict["intent_student_sha256"] = json.loads((out / "intent.json").read_text())["student"][
        "sha256"
    ]
    with (out / "verdict.json").open("x") as stream:
        json.dump(verdict, stream, indent=1, sort_keys=True)
    print(json.dumps({"verdict": verdict["verdict"], "pooled": verdict["pooled"]}))
    return 0


def main() -> int:
    global SPEC, FINAL_ROUND
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--spec", choices=sorted(SPECS), required=True)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("intent")
    a.add_argument("--student", required=True)
    a.add_argument("--out", required=True)
    r = sub.add_parser("run")
    r.add_argument("--arm", choices=sorted(ARMS), required=True)
    r.add_argument("--mix", choices=ni_spec.MIXES, required=True)
    r.add_argument("--out", required=True)
    d = sub.add_parser("decide")
    d.add_argument("--out", required=True)
    args = ap.parse_args()
    SPEC, FINAL_ROUND = SPECS[args.spec]
    return {"intent": cmd_intent, "run": cmd_run, "decide": cmd_decide}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
