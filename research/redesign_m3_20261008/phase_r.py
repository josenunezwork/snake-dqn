#!/usr/bin/env python3
"""M3 Tier-1 sequential Phase R harness (PREREGISTRATION_M3.md section 7).

Method ``sequential-phase-r-obf-hk-v1`` through ``research/sequential_phase_r`` (plan, look
gate, look units, look analysis, receipts, audit). The unit runner is a fork of FRP-v5-S2's
``evaluate.run_unit`` (nested H10000 with an exact H5000 prefix accumulator; direct-H5000
prefix controls in look 0), adapted to the ego2s hero, the SIMD grid engine and the pinned
forward modes (``vector61_forward="rowwise"``, ``ego2s_forward="rowwise"``, the modes P2
verified). Mac only; each unit is one subprocess holding one shared CPU slot lock.

Heroes (every one carries the v8 veto, lambda 8):

* ``M3-ego2s-b@20M+v8`` -- seed s's 20M checkpoint (``final.pth``) on seed s's bank (decision);
* ``incumbent`` -- frp3-s12 (decision baseline, every seed);
* ``champion`` -- champion_a5 (guard baseline, seed 0's bank);
* ``M2b-student+v8`` -- the M2b student (control arm, seed 0's bank; reported, not decision).

Prefix controls (look 0, direct H5000 on world index 0): the candidate and the incumbent on
every seed, the champion on seed 0. ``prefix_controls`` = every control equals its nested
partner's H5000 prefix block (computed at look 0, frozen in ``flags.json``, reused).

  ./venv/bin/python research/redesign_m3_20261008/phase_r.py plan  --root R [--smoke]
  ./venv/bin/python research/redesign_m3_20261008/phase_r.py look  --root R --look K [--procs 3]
  python -I research/sequential_phase_r/audit.py --root R --record-dirs R/shards/look-*
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m3_20261008 import rule_spec  # noqa: E402

SCHEMA = "redesign-m3-phase-r-record/v1"
PHASE_R_NS = "redesign-m3-phase-r/v1"
SMOKE_NS = "redesign-m3-phase-r-smoke/v1"  # out of the study namespace (smoke only)
SMOKE_N = 4  # looks (2, 3, 4)
RULE_SHA256 = "fd42a5d5ba898e41c2f4b251ac7833a461ed7773e5c91290f4d5015c00e04267"
PLAN_SHA256 = "5d1ee49d0ce8ce0f764f61909ad1803b542e825c70a08c01c12b83636651304e"
PROFILE_DIGEST_H5000 = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
PROFILE_DIGEST_H10000 = "c882e07fb7179c01feed5380aeae228113975442bb10ab8840bc58a0011f5b15"
H5000, H10000 = 5000, 10000
VETO, VETO_LAMBDA = "v8", 8.0
VECTOR61_FORWARD = "rowwise"
EGO2S_FORWARD = "rowwise"
SIM_ENGINE = "grid"
MAX_BATCH = 8

CANDIDATE = rule_spec.HERO
INCUMBENT, CHAMPION, CONTROL_ARM = "incumbent", "champion", "M2b-student+v8"
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
RUNS = ART / "redesign-m3b-gpu-20261008-r4/attempt01/out"
M2B = ART / "redesign-m2b-20261008/student_m2b_final.pth"
M2B_SHA = "36a92948ff82618b1944431675414301ef27f22bc15c7e12329bf75fc7cac48f"
P2_DIR = ART / "redesign-m3-p2-candidates-20261008"


def candidate_path(seed: int) -> Path:
    return RUNS / f"m3b_seed{int(seed)}" / "final.pth"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_commit() -> Tuple[str, bool]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO, capture_output=True, text=True, check=True
    ).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain"], cwd=REPO, capture_output=True, text=True, check=True
        ).stdout.strip()
    )
    return commit, dirty


# ----------------------------------------------------------------------------- plan
def plan_for(smoke: bool):
    from src.evaluation.sequential_phase_r import make_plan

    if smoke:
        return make_plan(SMOKE_N, rule_spec.m3_rule())
    plan = rule_spec.m3_plan()
    from research.sequential_phase_r.receipts import canonical_sha

    if canonical_sha(rule_spec.m3_rule()) != RULE_SHA256 or plan.sha256() != PLAN_SHA256:
        raise SystemExit("the rule / plan is not the pinned one")
    return plan


def make_banks(n: int, namespace: str) -> Dict[int, List[int]]:
    from research.apex_safety_20260926 import dev_screen

    return {
        s: [int(w) for w in dev_screen.screen_seeds(n, namespace, f"seed{s}")]
        for s in rule_spec.TRAINING_SEEDS
    }


def heroes() -> List[Dict[str, Any]]:
    seeds = list(rule_spec.TRAINING_SEEDS)
    return [
        {"hero": CANDIDATE, "seeds": seeds, "tier": 1},
        {"hero": INCUMBENT, "seeds": seeds, "tier": 1},
        {"hero": CHAMPION, "seeds": [0], "tier": 1},
        {"hero": CONTROL_ARM, "seeds": [0], "tier": 2},
    ]


def controls() -> List[Dict[str, Any]]:
    seeds = list(rule_spec.TRAINING_SEEDS)
    return [
        {"hero": CANDIDATE, "seeds": seeds},
        {"hero": INCUMBENT, "seeds": seeds},
        {"hero": CHAMPION, "seeds": [0]},
    ]


def checkpoints() -> Dict[str, Dict[str, str]]:
    """Every checkpoint a hero plays, with its sha256 (recorded in the plan)."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    out = {
        f"{CANDIDATE}|s{s}": {
            "path": str(candidate_path(s)),
            "sha256": sha256_file(candidate_path(s)),
        }
        for s in rule_spec.TRAINING_SEEDS
    }
    out[INCUMBENT] = {"path": str(gid.CHECKPOINT_DIR / gid.FRP3_S12[0]), "sha256": gid.FRP3_S12[1]}
    out[CHAMPION] = {
        "path": str(gid.CHECKPOINT_DIR / dev_screen.CHAMPION[0]),
        "sha256": dev_screen.CHAMPION[1],
    }
    out[CONTROL_ARM] = {"path": str(M2B), "sha256": M2B_SHA}
    for name, row in out.items():
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"{name}: {row['path']} does not hash to {row['sha256']}")
    return out


def p2_evidence() -> Dict[str, Any]:
    """P2 on every candidate (H5000 3 worlds / mix + H10000 1 world / mix) must have passed."""
    out = {}
    for s in rule_spec.TRAINING_SEEDS:
        want = sha256_file(candidate_path(s))
        for kind in ("p2_v8", "p2_h10000"):
            path = P2_DIR / f"{kind}_seed{s}.json"
            if not path.exists():
                raise SystemExit(f"P2 evidence missing: {path}")
            data = json.loads(path.read_text())
            n = 9 if kind == "p2_v8" else 3
            horizons = {int(w.get("horizon", 5000)) for w in data["worlds"]}
            ok = (
                data.get("pass") is True
                and data.get("arm") == "v8"
                and data["student"]["sha256"] == want
                and data["compared"] == n
                and data["identical"] == n
                and horizons == ({5000} if kind == "p2_v8" else {10000})
                and (kind == "p2_v8" or data.get("profile_digest") == PROFILE_DIGEST_H10000)
            )
            if not ok:
                raise SystemExit(f"P2 did not pass for seed {s} ({path})")
            out[f"{kind}_seed{s}"] = {"path": str(path), "sha256": sha256_file(path)}
    return out


def cmd_plan(args) -> int:
    from research.sequential_phase_r import receipts

    commit, dirty = git_commit()
    if dirty and not args.smoke:
        raise SystemExit("refusing: dirty worktree (the plan binds a clean commit)")
    plan = plan_for(args.smoke)
    banks = make_banks(plan.n_worlds, SMOKE_NS if args.smoke else PHASE_R_NS)
    prereg = REPO / "research/redesign_m3_20261008/PREREGISTRATION_M3.md"
    study = {
        "study_id": rule_spec.STUDY_ID + ("-SMOKE" if args.smoke else ""),
        "smoke": bool(args.smoke),
        "namespace": SMOKE_NS if args.smoke else PHASE_R_NS,
        "commit": commit,
        "dirty": dirty,
        "preregistration": {"path": str(prereg.relative_to(REPO)), "sha256": sha256_file(prereg)},
        "ratification": "PREREGISTRATION_M3.md section 9 (RATIFIED 2026-10-08)",
        "rule_sha256": receipts.canonical_sha(rule_spec.m3_rule()),
        "checkpoints": checkpoints(),
        "p2": None if args.smoke else p2_evidence(),
        "engine": {
            "sim_engine": SIM_ENGINE,
            "vector61_forward": VECTOR61_FORWARD,
            "ego2s_forward": EGO2S_FORWARD,
            "veto": VETO,
            "veto_lambda": VETO_LAMBDA,
            "profile_h5000": PROFILE_DIGEST_H5000,
            "profile_h10000": PROFILE_DIGEST_H10000,
        },
        "heroes": heroes(),
        "controls": controls(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    Path(args.root).mkdir(parents=True, exist_ok=True)
    receipts.write_plan(
        Path(args.root), plan.as_dict(), {str(k): v for k, v in banks.items()}, study
    )
    print(json.dumps({"plan_sha256": plan.sha256(), "look_sizes": plan.look_sizes}))
    return 0


def load_plan(root: Path):
    from research.sequential_phase_r import receipts
    from src.evaluation.sequential_phase_r import make_plan

    payload, _ = receipts.read_plan(root)
    plan = make_plan(int(payload["plan"]["n_worlds"]), payload["plan"]["rule"])
    if plan.sha256() != payload["plan_sha256"]:
        raise SystemExit("the plan on disk does not rebuild to its sha256")
    if not payload["study"]["smoke"] and plan.sha256() != PLAN_SHA256:
        raise SystemExit("the study plan is not the pinned plan")
    banks = {int(s): [int(w) for w in v] for s, v in payload["banks"].items()}
    return payload, plan, banks


def units_for(root: Path, look: int) -> List[Dict[str, Any]]:
    from research.sequential_phase_r import hooks

    payload, plan, banks = load_plan(root)
    hs = payload["study"]["heroes"]
    cs = payload["study"]["controls"]
    return hooks.look_units(plan, look, banks, hs, cs, max_batch=MAX_BATCH)


# ----------------------------------------------------------------------------- unit runner
class PrefixObserver:
    """``frame_observer``: an exact H5000 prefix per env (fork of FRP-v5-S2 ``FrameStats``
    without the hazard table): an ``EvaluationMetricsAccumulator(prefix)`` per env fed with
    exactly the arguments ``run_simd_eval`` gives its own accumulator, for frames < prefix."""

    def __init__(self, n_envs: int, prefix: int):
        from src.evaluation.metrics import EvaluationMetricsAccumulator

        self.n, self.prefix = int(n_envs), int(prefix)
        self.accumulators = [EvaluationMetricsAccumulator(self.prefix) for _ in range(self.n)]
        self.prev_alive: Optional[List[bool]] = None

    def __call__(self, payload: Mapping[str, Any]) -> None:
        from src.evaluation.metrics import PostStepState, StepEvents
        from src.simd_env.eval_engine import _DEATH_CAUSE_LABEL, DEATH_NONE

        frame = int(payload["frame"])
        pre, post = payload["pre"], payload["post"]
        if self.prev_alive is None:
            self.prev_alive = [bool(v) for v in pre["alive"]]
        if frame < self.prefix:
            for e in range(self.n):
                valid = bool(post["transition_valid"][e])
                died = bool(post["done"][e]) if valid else False
                cause = int(post["death_cause"][e]) if valid else DEATH_NONE
                self.accumulators[e].observe(
                    pre_alive=bool(self.prev_alive[e]),
                    post=PostStepState(
                        alive=bool(post["alive"][e]), logical_mass=float(post["logical_mass"][e])
                    ),
                    events=StepEvents(
                        food_eaten=int(post["food_ate"][e]) if valid else 0,
                        boost_executed=bool(post["boosted"][e]) if valid else False,
                        kills=int(post["kills"][e]) if valid else 0,
                        death=died,
                        death_cause=_DEATH_CAUSE_LABEL.get(cause) if died else None,
                    ),
                    acted=valid,
                )
        self.prev_alive = [bool(v) for v in post["alive"]]

    def result(self, env: int) -> Dict[str, Any]:
        return self.accumulators[env].result()


def hero_call(hero: str, seed: int, ctx, study) -> Tuple[Any, Dict[str, Any]]:
    """(hero_spec, extra run_simd_eval kwargs) for one hero on one seed's bank."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    lookup = ctx["lookup"]
    ck = study["checkpoints"]
    if hero == CANDIDATE:
        row = ck[f"{CANDIDATE}|s{int(seed)}"]
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"candidate s{seed} checkpoint changed since the plan")
        return lookup[gid.FRP3_S12[1]], {"hero_ego2s": row["path"], "ego2s_forward": EGO2S_FORWARD}
    if hero == CONTROL_ARM:
        if sha256_file(M2B) != M2B_SHA:
            raise SystemExit("the M2b student changed")
        return lookup[gid.FRP3_S12[1]], {"hero_ego2s": str(M2B), "ego2s_forward": EGO2S_FORWARD}
    if hero == INCUMBENT:
        return lookup[gid.FRP3_S12[1]], {}
    if hero == CHAMPION:
        return lookup[dev_screen.CHAMPION[1]], {}
    raise SystemExit(f"unknown hero {hero}")


def bank_rows(bank: Sequence[int], mix: str) -> Dict[int, Mapping[str, Any]]:
    """Design rows of the FULL bank (never of a batch), filtered by mix."""
    from research.apex_safety_20260926 import dev_screen

    return {
        int(r["world_seed"]): r
        for r in dev_screen._design_rows([int(w) for w in bank])
        if r["mix"] == mix
    }


def hero_checkpoint(hero: str, seed: int, study: Mapping[str, Any]) -> Dict[str, str]:
    ck = study["checkpoints"]
    row = ck[f"{CANDIDATE}|s{int(seed)}"] if hero == CANDIDATE else ck[hero]
    return {"path": row["path"], "sha256": row["sha256"]}


def record_name(entry: Mapping[str, Any]) -> str:
    hero = str(entry["hero"]).replace("@", "-u").replace("+", "p")
    kind = "control-" if entry.get("control") else ""
    return f"{kind}{hero}-s{entry['seed']}-{entry['mix']}-{entry['world_seed']}.json"


def shard_dir(root: Path, look: int) -> Path:
    return Path(root) / "shards" / f"look-{int(look)}"


def run_unit(root: Path, look: int, unit_id: str) -> int:
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.simd_env.eval_engine import run_simd_eval

    torch.set_num_threads(1)
    payload, plan, banks = load_plan(root)
    units = {u["unit_id"]: u for u in units_for(root, look)}
    unit = units[unit_id]
    from research.sequential_phase_r import receipts

    out_dir = shard_dir(root, look) / "records"
    start = json.loads((shard_dir(root, look) / "start.json").read_text())
    commit, dirty = git_commit()
    if commit != start["commit"] or (dirty and not payload["study"]["smoke"]):
        raise SystemExit("refusing: HEAD / worktree differs from the look's start marker")
    names = [record_name(dict(unit, world_seed=w)) for w in unit["worlds"]]
    # Resume: validate every record already written for this unit; write only missing ones.
    for n in names:
        path = out_dir / n
        if path.exists():
            old = json.loads(path.read_text())
            if old.get("unit_id") != unit_id or old.get("binding") != start["binding"]:
                raise SystemExit(f"refusing: {path} belongs to another unit / binding")
    if all((out_dir / n).exists() for n in names):
        return 0
    ctx = gid._context(1)
    profile5 = ctx["profile"]
    if profile5.digest != PROFILE_DIGEST_H5000:
        raise SystemExit("H5000 profile digest drift")
    profile10 = dataclasses.replace(profile5, scored_horizon=H10000)
    if profile10.digest != PROFILE_DIGEST_H10000:
        raise SystemExit("H10000 profile digest drift")
    control = bool(unit["control"])
    horizon = H5000 if control else H10000
    seeds = [int(w) for w in unit["worlds"]]
    rows = bank_rows(banks[int(unit["seed"])], unit["mix"])
    rosters = {
        s: [ctx["lookup"][slot["member_sha256"]] for slot in rows[s]["slots"]] for s in seeds
    }
    identities = {s: _expected_world_identity(rows[s]) for s in seeds}
    hero_spec, extra = hero_call(unit["hero"], unit["seed"], ctx, payload["study"])
    hero_ck = hero_checkpoint(unit["hero"], unit["seed"], payload["study"])
    observer = None if control else PrefixObserver(len(seeds), H5000)
    started = time.monotonic()
    records = run_simd_eval(
        hero_spec,
        rosters[seeds[0]],
        horizon,
        seeds,
        profile=profile5 if control else profile10,
        opponent_specs_by_world=rosters,
        world_identities=identities,
        mix_id=unit["mix"],
        frame_observer=observer,
        vector61=True,
        vector61_forward=VECTOR61_FORWARD,
        hero_safety_veto=VETO,
        hero_safety_veto_lambda=VETO_LAMBDA,
        sim_engine=SIM_ENGINE,
        **extra,
    )
    wall = time.monotonic() - started
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, s in enumerate(seeds):
        entry = {
            "schema": SCHEMA,
            "unit_id": unit_id,
            "look": int(look),
            "binding": start["binding"],
            "commit": commit,
            "hero": unit["hero"],
            "seed": int(unit["seed"]),
            "mix": unit["mix"],
            "world_seed": s,
            "world_index": banks[int(unit["seed"])].index(s),
            "world_index_range": unit["world_index_range"],
            "roster_member_sha256s": [slot["member_sha256"] for slot in rows[s]["slots"]],
            "horizon": horizon,
            "control": control,
            "tier": int(unit["tier"]),
            "hero_checkpoint": hero_ck,
            "engine": {
                "sim_engine": SIM_ENGINE,
                "vector61_forward": VECTOR61_FORWARD,
                "ego2s_forward": extra.get("ego2s_forward"),
                "veto": VETO,
            },
            "record": json.loads(json.dumps(records[i])),
            "prefix_h5000": None if control else observer.result(i),
            "unit_wall_seconds": wall,
        }
        path = out_dir / names[i]
        if not path.exists():  # resume writes only the missing records (validated above)
            receipts.write_once(path, entry)
    return 0


# ----------------------------------------------------------------------------- look
def prefix_identical(direct: Mapping[str, Any], prefix: Mapping[str, Any]) -> bool:
    """Same comparison as the audit (``probes.safety_veto`` excluded)."""
    for field, value in prefix.items():
        a = direct.get(field)
        if field == "probes" and isinstance(a, dict):
            a = {k: v for k, v in a.items() if k != "safety_veto"}
            value = {k: v for k, v in value.items() if k != "safety_veto"}
        if json.dumps(a, sort_keys=True) != json.dumps(value, sort_keys=True):
            return False
    return True


def prefix_controls_flag(root: Path) -> Tuple[bool, List[Dict[str, Any]]]:
    from research.sequential_phase_r import hooks

    entries = hooks.load_entries([shard_dir(root, 0)])
    nested = {
        (e["hero"], e["seed"], e["mix"], e["world_seed"]): e
        for e in entries
        if not e.get("control")
    }
    rows = []
    for e in entries:
        if not e.get("control"):
            continue
        partner = nested.get((e["hero"], e["seed"], e["mix"], e["world_seed"]))
        same = partner is not None and prefix_identical(e["record"], partner["prefix_h5000"])
        rows.append({"key": [e["hero"], e["seed"], e["mix"], e["world_seed"]], "identical": same})
    return bool(rows) and all(r["identical"] for r in rows), rows


def run_units(root: Path, look: int, units: Sequence[Mapping[str, Any]], procs: int) -> List[str]:
    """Each unit = one subprocess (own session) holding one shared CPU slot (``slotrun.py``).
    On any exit of this runner (error, Ctrl-C), still-running unit sessions are terminated."""
    import signal

    log = shard_dir(root, look) / "units.log"
    todo = list(units)
    running: List[Tuple[subprocess.Popen, str]] = []
    failed: List[str] = []
    with log.open("a") as stream:
        try:
            while todo or running:
                still = []
                for proc, uid in running:
                    if proc.poll() is None:
                        still.append((proc, uid))
                    elif proc.returncode != 0:
                        failed.append(uid)
                running = still
                if todo and len(running) < procs:
                    u = todo.pop(0)
                    argv = [
                        sys.executable,
                        str(REPO / "research/redesign_m3_20261008/slotrun.py"),
                        "--",
                        sys.executable,
                        str(Path(__file__).resolve()),
                        "unit",
                        "--root",
                        str(root),
                        "--look",
                        str(look),
                        "--unit-id",
                        u["unit_id"],
                    ]
                    proc = subprocess.Popen(
                        argv, cwd=REPO, stdout=stream, stderr=stream, start_new_session=True
                    )
                    running.append((proc, u["unit_id"]))
                    print(json.dumps({"started": u["unit_id"], "left": len(todo)}), flush=True)
                    continue
                time.sleep(2)
        finally:
            for proc, _ in running:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except OSError:
                        proc.terminate()
    return failed


def cmd_look(args) -> int:
    from research.sequential_phase_r import hooks, receipts

    root, look = Path(args.root), int(args.look)
    payload, plan, banks = load_plan(root)
    commit, dirty = git_commit()
    if commit != payload["study"]["commit"] or (dirty and not payload["study"]["smoke"]):
        raise SystemExit("refusing: HEAD / worktree differs from the plan's commit")
    binding = receipts.look_gate(root, look)
    sd = shard_dir(root, look)
    sd.mkdir(parents=True, exist_ok=True)
    units = units_for(root, look)
    start = {
        "binding": binding,
        "commit": commit,
        "units": [u["unit_id"] for u in units],
        "unit_worlds": {u["unit_id"]: [int(w) for w in u["worlds"]] for u in units},
        "started_utc": datetime.now(timezone.utc).isoformat(),
    }
    marker = sd / "start.json"
    if marker.exists():
        old = json.loads(marker.read_text())
        if (
            old["binding"] != binding
            or old["units"] != start["units"]
            or old.get("unit_worlds") != start["unit_worlds"]
            or old["commit"] != commit
        ):
            raise SystemExit("start marker exists with a different binding (recovery mismatch)")
    else:
        receipts.write_once(marker, start)
    failed = run_units(root, look, units, args.procs)
    if failed:
        print(json.dumps({"look": look, "failed_units": failed}))
        return 1
    missing = [
        n
        for u in units
        for n in (record_name(dict(u, world_seed=w)) for w in u["worlds"])
        if not (sd / "records" / n).exists()
    ]
    if missing:
        print(json.dumps({"look": look, "missing_records": missing[:10], "n": len(missing)}))
        return 1
    flags_path = root / "flags.json"
    if look == 0:
        ok, rows = prefix_controls_flag(root)
        computed = {"prefix_controls": ok, "controls": rows}
        if flags_path.exists():
            if json.loads(flags_path.read_text()) != computed:
                raise SystemExit("flags.json exists and differs from the recomputed flags")
        else:
            receipts.write_once(flags_path, computed)
    flags = {"prefix_controls": bool(json.loads(flags_path.read_text())["prefix_controls"])}
    entries = hooks.load_entries([shard_dir(root, k) for k in range(look + 1)])
    receipt = hooks.analyse_look(
        root,
        plan,
        entries,
        look,
        flags,
        extra={"analysed_utc": datetime.now(timezone.utc).isoformat()},
    )
    d = receipt["decision"]
    print(
        json.dumps(
            {
                "look": look,
                "n_per_seed_mix": receipt["n_per_seed_mix"],
                "status": d.get("status"),
                "action": receipt["action"],
                "valid": d.get("valid"),
                "flags": flags,
            }
        )
    )
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    q = sub.add_parser("look")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--look", type=int, required=True)
    q.add_argument("--procs", type=int, default=3)
    u = sub.add_parser("unit")
    u.add_argument("--root", type=Path, required=True)
    u.add_argument("--look", type=int, required=True)
    u.add_argument("--unit-id", required=True)
    args = ap.parse_args()
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "look":
        if not 1 <= args.procs <= 3:
            raise SystemExit("compute policy: 1..3 processes")
        return cmd_look(args)
    return run_unit(args.root, args.look, args.unit_id)


if __name__ == "__main__":
    raise SystemExit(main())
