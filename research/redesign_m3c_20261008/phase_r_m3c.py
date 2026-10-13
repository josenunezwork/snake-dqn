#!/usr/bin/env python3
"""M3-C Tier-1 sequential Phase R harness (PREREGISTRATION_M3C, section 8). DRAFT.

A fork of the M3 harness (``research/redesign_m3_20261008/phase_r.py``: plan, look gate,
look units, nested H10000 with an exact H5000 prefix, prefix controls, slot-locked unit
subprocesses, create-only records) for the two M3-C studies (``rule_spec_m3c``):

* ``--study nv`` (primary, ``redesign-m3c-phase-r-nv``): the candidate plays WITHOUT a veto,
  SIMD hero with ``ego2s_mask="advisory"`` (the live AISnake selection rule; P2-NV must have
  passed on every candidate); the control arm is the M2b student, also without a veto;
* ``--study v8`` (secondary, ``redesign-m3c-phase-r-v8``): candidate + v8 (as M3).

The incumbent (frp3-s12) and the champion (champion_a5) always carry v8 (lambda 8).
Candidates come from ``--candidates FILE`` (``{"arm": X, "paths": {"0": ..., "4": ...}}``),
hashed into ``plan.json``. A real (non-smoke) plan refuses unless the pre-registration file
says RATIFIED.

  ./venv/bin/python research/redesign_m3c_20261008/phase_r_m3c.py plan --study nv \
      --root R --candidates C.json [--smoke]
  ./venv/bin/python research/redesign_m3c_20261008/phase_r_m3c.py look --root R --look K
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

from research.redesign_m3c_20261008 import rule_spec_m3c as rs  # noqa: E402

SCHEMA = "redesign-m3c-phase-r-record/v1"
SMOKE_NS = "redesign-m3c-phase-r-smoke/v1"  # out of every study namespace (smoke only)
SMOKE_N = 4  # looks (2, 3, 4)
STUDIES = {
    "nv": {
        "rule": rs.primary_rule,
        "plan": rs.primary_plan,
        "id": rs.PRIMARY_ID,
        "hero": rs.HERO_NV,
        "no_veto": True,
        "rule_sha256": rs.PRIMARY_RULE_SHA256,
        "plan_sha256": rs.PRIMARY_PLAN_SHA256,
    },
    "v8": {
        "rule": rs.secondary_rule,
        "plan": rs.secondary_plan,
        "id": rs.SECONDARY_ID,
        "hero": rs.HERO_V8,
        "no_veto": False,
        "rule_sha256": rs.SECONDARY_RULE_SHA256,
        "plan_sha256": rs.SECONDARY_PLAN_SHA256,
    },
}
PREREG = REPO / "research/redesign_m3c_20261008/PREREGISTRATION_M3C.md"
PROFILE_DIGEST_H5000 = "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b"
PROFILE_DIGEST_H10000 = "c882e07fb7179c01feed5380aeae228113975442bb10ab8840bc58a0011f5b15"
H5000, H10000 = 5000, 10000
VETO, VETO_LAMBDA = "v8", 8.0
VECTOR61_FORWARD = "rowwise"
EGO2S_FORWARD = "rowwise"
SIM_ENGINE = "grid"
MAX_BATCH = 8

INCUMBENT, CHAMPION = "incumbent", "champion"
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
M2B = ART / "redesign-m2b-20261008/student_m2b_final.pth"
M2B_SHA = "36a92948ff82618b1944431675414301ef27f22bc15c7e12329bf75fc7cac48f"
P2_DIR = ART / "redesign-m3c-p2-candidates"
TRAINING_TRANSITIONS = 20_000_000  # PREREGISTRATION_M3C section 2


def control_arm(key: str) -> str:
    return "M2b-student" if STUDIES[key]["no_veto"] else "M2b-student+v8"


def study_key(study: Mapping[str, Any]) -> str:
    sid = str(study["study_id"]).replace("-SMOKE", "")
    hits = [k for k, v in STUDIES.items() if v["id"] == sid]
    if len(hits) != 1:
        raise SystemExit(f"unknown study id {sid}")
    return hits[0]


def ratified(prereg: Path) -> bool:
    """A line that STARTS with ``**RATIFIED <yyyy-mm-dd>`` (not a substring: "NOT RATIFIED"
    or a quoted mention does not count)."""
    import re

    if not prereg.exists():
        return False
    return any(
        re.match(r"^\*\*RATIFIED \d{4}-\d{2}-\d{2}", line)
        for line in prereg.read_text().splitlines()
    )


def check_selection(key: str, selection: Optional[Path], candidates: Path) -> Dict[str, Any]:
    """The candidates must be the arm ``m3c_select.py`` selected for this study, with
    ``enter`` true, and exactly that arm's 5 ``final.pth`` (one per seed)."""
    if selection is None:
        raise SystemExit("refusing: --selection (m3c_select.py output) is required")
    sel = json.loads(Path(selection).read_text())
    gate = sel["primary" if STUDIES[key]["no_veto"] else "secondary"]
    cand = json.loads(Path(candidates).read_text())
    if not gate.get("enter") or gate.get("arm") != cand.get("arm"):
        raise SystemExit(f"refusing: the selection does not enter arm {cand.get('arm')}: {gate}")
    if sel.get("target_transitions") != TRAINING_TRANSITIONS:
        raise SystemExit("refusing: the selection was not made with the full training length")
    arm_row = sel["scores"][cand["arm"]]
    if sorted(arm_row.get("eligible", [])) != list(rs.TRAINING_SEEDS):
        raise SystemExit(f"refusing: arm {cand['arm']} does not have 5 eligible seeds")
    if sel.get("checkpoint") != "final":
        raise SystemExit("refusing: the selection was not made on the final checkpoints")
    shas = {str(k): v for k, v in arm_row.get("final_sha256", {}).items()}
    for s, path in load_candidates(candidates).items():
        if shas.get(str(s)) != sha256_file(Path(path)):
            raise SystemExit(
                f"refusing: seed {s} candidate is not the checkpoint the selection probed"
            )
        if Path(path).name != "final.pth" or Path(path).parent.name != f"{cand['arm']}_s{s}":
            raise SystemExit(
                f"refusing: seed {s} candidate {path} is not {cand['arm']}_s{s}/final.pth"
            )
    return {"path": str(selection), "sha256": sha256_file(Path(selection)), "gate": gate}


def load_candidates(path: Path) -> Dict[int, Path]:
    data = json.loads(Path(path).read_text())
    paths = {int(k): Path(v) for k, v in data["paths"].items()}
    if sorted(paths) != list(rs.TRAINING_SEEDS):
        raise SystemExit(f"candidates must name seeds {list(rs.TRAINING_SEEDS)}")
    return paths


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
def plan_for(key: str, smoke: bool):
    from src.evaluation.sequential_phase_r import make_plan

    st = STUDIES[key]
    if smoke:
        return make_plan(SMOKE_N, st["rule"]())
    plan = st["plan"]()
    from research.sequential_phase_r.receipts import canonical_sha

    if canonical_sha(st["rule"]()) != st["rule_sha256"] or plan.sha256() != st["plan_sha256"]:
        raise SystemExit("the rule / plan is not the pinned one")
    return plan


def make_banks(n: int, namespace: str) -> Dict[int, List[int]]:
    from research.apex_safety_20260926 import dev_screen

    return {
        s: [int(w) for w in dev_screen.screen_seeds(n, namespace, f"seed{s}")]
        for s in rs.TRAINING_SEEDS
    }


def heroes(key: str) -> List[Dict[str, Any]]:
    seeds = list(rs.TRAINING_SEEDS)
    return [
        {"hero": STUDIES[key]["hero"], "seeds": seeds, "tier": 1},
        {"hero": INCUMBENT, "seeds": seeds, "tier": 1},
        {"hero": CHAMPION, "seeds": [0], "tier": 1},
        {"hero": control_arm(key), "seeds": [0], "tier": 2},
    ]


def controls(key: str) -> List[Dict[str, Any]]:
    seeds = list(rs.TRAINING_SEEDS)
    return [
        {"hero": STUDIES[key]["hero"], "seeds": seeds},
        {"hero": INCUMBENT, "seeds": seeds},
        {"hero": CHAMPION, "seeds": [0]},
    ]


def checkpoints(key: str, candidates: Mapping[int, Path]) -> Dict[str, Dict[str, str]]:
    """Every checkpoint a hero plays, with its sha256 (recorded in the plan)."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    hero = STUDIES[key]["hero"]
    out = {
        f"{hero}|s{s}": {"path": str(candidates[s]), "sha256": sha256_file(candidates[s])}
        for s in rs.TRAINING_SEEDS
    }
    out[INCUMBENT] = {"path": str(gid.CHECKPOINT_DIR / gid.FRP3_S12[0]), "sha256": gid.FRP3_S12[1]}
    out[CHAMPION] = {
        "path": str(gid.CHECKPOINT_DIR / dev_screen.CHAMPION[0]),
        "sha256": dev_screen.CHAMPION[1],
    }
    out[control_arm(key)] = {"path": str(M2B), "sha256": M2B_SHA}
    for name, row in out.items():
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"{name}: {row['path']} does not hash to {row['sha256']}")
    return out


def p2_evidence(key: str, candidates: Mapping[int, Path]) -> Dict[str, Any]:
    """P2 on every candidate must have passed: nv -> P2-NV (``p2_noveto.py``, 9 worlds H5000,
    arm none); v8 -> M3's P2-v8 (9 worlds H5000) + P2 H10000 (3 worlds)."""
    out = {}
    kinds = ("p2_noveto", "p2_noveto_h10000") if STUDIES[key]["no_veto"] else ("p2_v8", "p2_h10000")
    for s in rs.TRAINING_SEEDS:
        want = sha256_file(candidates[s])
        for kind in kinds:
            path = P2_DIR / f"{kind}_seed{s}.json"
            if not path.exists():
                raise SystemExit(f"P2 evidence missing: {path}")
            data = json.loads(path.read_text())
            long_ = kind.endswith("h10000")
            n = 3 if long_ else 9
            horizons = {int(w.get("horizon", 5000)) for w in data["worlds"]}
            schema = {
                "p2_noveto": "redesign-m3c-student-record-parity-noveto/v1",
                "p2_noveto_h10000": "redesign-m3c-student-record-parity-noveto-h10000/v1",
            }.get(kind)
            ok = (
                data.get("pass") is True
                and data.get("arm") == ("none" if kind.startswith("p2_noveto") else "v8")
                and data["student"]["sha256"] == want
                and data["compared"] == n
                and data["identical"] == n
                and horizons == ({10000} if long_ else {5000})
                and (not long_ or data.get("profile_digest") == PROFILE_DIGEST_H10000)
                and (schema is None or data.get("schema_version") == schema)
            )
            if not ok:
                raise SystemExit(f"P2 did not pass for seed {s} ({path})")
            out[f"{kind}_seed{s}"] = {"path": str(path), "sha256": sha256_file(path)}
    return out


def cmd_plan(args) -> int:
    from research.sequential_phase_r import receipts

    key = args.study
    commit, dirty = git_commit()
    if dirty and not args.smoke:
        raise SystemExit("refusing: dirty worktree (the plan binds a clean commit)")
    prereg = PREREG
    selection = None
    if not args.smoke:
        if not ratified(prereg):
            raise SystemExit("refusing: PREREGISTRATION_M3C.md has no '**RATIFIED <date>' line")
        selection = check_selection(key, args.selection, args.candidates)
    plan = plan_for(key, args.smoke)
    namespace = SMOKE_NS if args.smoke else rs.BANK_NS[STUDIES[key]["id"]]
    banks = make_banks(plan.n_worlds, namespace)
    candidates = load_candidates(args.candidates)
    cand = json.loads(Path(args.candidates).read_text())
    study = {
        "study_id": STUDIES[key]["id"] + ("-SMOKE" if args.smoke else ""),
        "smoke": bool(args.smoke),
        "namespace": namespace,
        "commit": commit,
        "dirty": dirty,
        "candidate_arm": cand.get("arm"),
        "selection": selection,
        "no_veto_candidate": STUDIES[key]["no_veto"],
        "preregistration": (
            {"path": str(prereg.relative_to(REPO)), "sha256": sha256_file(prereg)}
            if prereg.exists()
            else None
        ),
        "ratification": "PREREGISTRATION_M3C.md section 11" if not args.smoke else None,
        "rule_sha256": receipts.canonical_sha(STUDIES[key]["rule"]()),
        "checkpoints": checkpoints(key, candidates),
        "p2": None if args.smoke else p2_evidence(key, candidates),
        "engine": {
            "sim_engine": SIM_ENGINE,
            "vector61_forward": VECTOR61_FORWARD,
            "ego2s_forward": EGO2S_FORWARD,
            "veto": VETO,
            "veto_lambda": VETO_LAMBDA,
            "candidate_veto": None if STUDIES[key]["no_veto"] else VETO,
            "candidate_ego2s_mask": "advisory" if STUDIES[key]["no_veto"] else None,
            "profile_h5000": PROFILE_DIGEST_H5000,
            "profile_h10000": PROFILE_DIGEST_H10000,
        },
        "heroes": heroes(key),
        "controls": controls(key),
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
    key = study_key(payload["study"])
    if not payload["study"]["smoke"] and plan.sha256() != STUDIES[key]["plan_sha256"]:
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
    """(hero_spec, extra run_simd_eval kwargs) for one hero on one seed's bank.

    ``extra["veto"]`` is False for the no-veto candidate / control (nv study) and True for
    everything else (v8, lambda 8); ``run_unit`` turns it into the veto kwargs."""
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    key = study_key(study)
    no_veto = bool(STUDIES[key]["no_veto"])
    lookup = ctx["lookup"]
    ck = study["checkpoints"]
    ego = {"ego2s_forward": EGO2S_FORWARD}
    if no_veto:
        ego["ego2s_mask"] = "advisory"
    if hero == STUDIES[key]["hero"]:
        row = ck[f"{hero}|s{int(seed)}"]
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"candidate s{seed} checkpoint changed since the plan")
        return lookup[gid.FRP3_S12[1]], {"hero_ego2s": row["path"], "veto": not no_veto, **ego}
    if hero == control_arm(key):
        if sha256_file(M2B) != M2B_SHA:
            raise SystemExit("the M2b student changed")
        return lookup[gid.FRP3_S12[1]], {"hero_ego2s": str(M2B), "veto": not no_veto, **ego}
    if hero == INCUMBENT:
        return lookup[gid.FRP3_S12[1]], {"veto": True}
    if hero == CHAMPION:
        return lookup[dev_screen.CHAMPION[1]], {"veto": True}
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
    key = study_key(study)
    cand = STUDIES[key]["hero"]
    row = ck[f"{cand}|s{int(seed)}"] if hero == cand else ck[hero]
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
    vetoed = bool(extra.pop("veto"))
    veto_kwargs = (
        {"hero_safety_veto": VETO, "hero_safety_veto_lambda": VETO_LAMBDA} if vetoed else {}
    )
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
        sim_engine=SIM_ENGINE,
        **veto_kwargs,
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
                "ego2s_mask": extra.get("ego2s_mask"),
                "veto": VETO if vetoed else None,
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
    p.add_argument("--study", choices=sorted(STUDIES), required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=None, help="m3c_select.py output (real plans)")
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
