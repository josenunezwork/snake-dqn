#!/usr/bin/env python3
"""M4 Tier-1 sequential Phase R harness (PREREGISTRATION_M4.md section 7). DRAFT.

A thin wrapper of the ratified M3-C harness ``research/redesign_m3c_20261008/phase_r_m3c.py``
(plan, look gate, look units, nested H10000 with an exact H5000 prefix, prefix controls,
slot-locked unit subprocesses, create-only records). That file is pinned by M3-C's
ratification, so it is NOT edited: this module loads a PRIVATE copy of it (its own module
object, sha256-checked against ``BASE_SHA256``) and rebinds the copy's study globals:

* rule specs -> ``rule_spec_m4`` (seeds 0-2; studies ``redesign-m4-phase-r-nv`` / ``-v8``);
* record ``SCHEMA``, smoke namespace, pre-registration (``PREREGISTRATION_M4.md``), P2
  evidence directory;
* ``__file__`` -> this file, so look units run ``phase_r_m4.py unit`` (with the same rebinding).

Everything else -- hero modes (nv: candidate and the M2b control WITHOUT a veto, SIMD
``ego2s_mask="advisory"``; v8: everyone vetoed; incumbent frp3-s12 and champion_a5 always + v8),
engine, profiles, records, look logic -- is M3-C's code, byte for byte.

``plan`` is this module's own (M3-C's plus): the candidates' observation version comes from
the checkpoint (``obs_spec`` key: ``ego2s-b`` or ``ego2s-bv8``; all three seeds must agree) and
is bound into ``plan.json``. ``ego2s-bv8`` has NO live path (``EGO2S_SIMD_ONLY_SPECS``), so its
P2 (grid == live) cannot run, let alone pass: a real bv8 plan refuses unless
``--bv8-simd-only`` is given (owner decision, PREREGISTRATION_M4 section 6), which records
"P2 not applicable" in the plan. The SIMD Phase R itself plays bv8 (the hero's featurizer
version is read from the network, ``ego_config_for``).

  ./venv/bin/python research/redesign_m4_20261013/phase_r_m4.py plan --study nv \\
      --root R --candidates C.json --selection S.json [--smoke] [--bv8-simd-only]
  ./venv/bin/python research/redesign_m4_20261013/phase_r_m4.py look --root R --look K
  python -I research/sequential_phase_r/audit.py --root R --record-dirs R/shards/look-*
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse  # noqa: E402
import hashlib  # noqa: E402
import importlib.util  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from types import ModuleType  # noqa: E402
from typing import Mapping  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.redesign_m4_20261013 import rule_spec_m4 as rs  # noqa: E402

BASE_PATH = REPO / "research/redesign_m3c_20261008/phase_r_m3c.py"
BASE_SHA256 = "3a7658fd97a023fad44fc96df3b8afc5895b878dced8cd6beec94c075c553e0f"
BASE_MODULE = "research.redesign_m4_20261013._phase_r_m3c_base"
SCHEMA = "redesign-m4-phase-r-record/v1"
SMOKE_NS = "redesign-m4-phase-r-smoke/v1"  # out of every study namespace (smoke only)
PREREG = REPO / "research/redesign_m4_20261013/PREREGISTRATION_M4.md"
RATIFICATION_REF = "PREREGISTRATION_M4.md section 12"
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
P2_DIR = ART / "redesign-m4-p2-candidates"
TRAINING_TRANSITIONS = 20_000_000  # M4-A plans (research/redesign_m4a_20261012/plans)
EGO2S_B, EGO2S_BV8 = "ego2s-b", "ego2s-bv8"
CANDIDATE_SPECS = (EGO2S_B, EGO2S_BV8)
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


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_base() -> ModuleType:
    """A private, rebound copy of the pinned M3-C harness (never the shared module object, so
    ``phase_r_m3c`` imported elsewhere in the same process is untouched)."""
    if BASE_MODULE in sys.modules:
        return sys.modules[BASE_MODULE]
    if sha256_file(BASE_PATH) != BASE_SHA256:
        raise SystemExit(f"{BASE_PATH} is not the pinned M3-C harness ({BASE_SHA256[:12]})")
    spec = importlib.util.spec_from_file_location(BASE_MODULE, BASE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[BASE_MODULE] = mod
    spec.loader.exec_module(mod)
    mod.rs = rs
    mod.SCHEMA = SCHEMA
    mod.SMOKE_NS = SMOKE_NS
    mod.STUDIES = STUDIES
    mod.PREREG = PREREG
    mod.P2_DIR = P2_DIR
    mod.TRAINING_TRANSITIONS = TRAINING_TRANSITIONS
    mod.__file__ = str(Path(__file__).resolve())  # look units run THIS file's ``unit``
    return mod


base = _load_base()
# re-exported (the M3-C behaviour, rebound to M4)
ratified = base.ratified
check_selection = base.check_selection
load_candidates = base.load_candidates
heroes = base.heroes
controls = base.controls
control_arm = base.control_arm
hero_call = base.hero_call
study_key = base.study_key
git_commit = base.git_commit


def candidate_obs_spec(candidates: Mapping[int, Path]) -> str:
    """The candidates' (common) observation spec, read from the checkpoints."""
    from src.model.ego2s_network import obs_spec_of

    specs = {int(s): obs_spec_of(str(p)) for s, p in candidates.items()}
    if len(set(specs.values())) != 1:
        raise SystemExit(f"refusing: the candidates' obs specs differ: {specs}")
    spec = next(iter(specs.values()))
    if spec not in CANDIDATE_SPECS:
        raise SystemExit(f"refusing: candidate obs spec {spec!r} is not one of {CANDIDATE_SPECS}")
    return spec


def p2_for(key: str, candidates: Mapping[int, Path], obs_spec: str, bv8_simd_only: bool):
    """P2 evidence (M3-C's checks, M4's directory), or the recorded bv8 exception."""
    if obs_spec == EGO2S_BV8:
        if not bv8_simd_only:
            raise SystemExit(
                "refusing: ego2s-bv8 has no live path, so P2 (grid == live) cannot pass; "
                "a SIMD-only Phase R needs --bv8-simd-only (PREREGISTRATION_M4 section 6)"
            )
        return {
            "status": "not_applicable",
            "reason": "ego2s-bv8 is SIMD-only (src/model/obs_spec.py EGO2S_SIMD_ONLY_SPECS); "
            "no live serving / tournament path, so grid == live parity cannot be checked",
            "consequence": "Tier-1 SIMD evidence only; live bv8 serving + P2 before LH-1",
        }
    if bv8_simd_only:
        raise SystemExit("refusing: --bv8-simd-only given for a non-bv8 candidate")
    return base.p2_evidence(key, candidates)


def cmd_plan(args) -> int:
    """M3-C's ``cmd_plan`` with M4's pre-registration, obs-spec binding and bv8 P2 rule."""
    from research.sequential_phase_r import receipts

    key = args.study
    commit, dirty = git_commit()
    if dirty and not args.smoke:
        raise SystemExit("refusing: dirty worktree (the plan binds a clean commit)")
    selection = None
    if not args.smoke:
        if not ratified(PREREG):
            raise SystemExit("refusing: PREREGISTRATION_M4.md has no '**RATIFIED <date>' line")
        selection = check_selection(key, args.selection, args.candidates)
    plan = base.plan_for(key, args.smoke)
    namespace = SMOKE_NS if args.smoke else rs.BANK_NS[STUDIES[key]["id"]]
    banks = base.make_banks(plan.n_worlds, namespace)
    candidates = load_candidates(args.candidates)
    cand = json.loads(Path(args.candidates).read_text())
    obs_spec = candidate_obs_spec(candidates)
    study = {
        "study_id": STUDIES[key]["id"] + ("-SMOKE" if args.smoke else ""),
        "smoke": bool(args.smoke),
        "namespace": namespace,
        "commit": commit,
        "dirty": dirty,
        "candidate_arm": cand.get("arm"),
        "candidate_obs_spec": obs_spec,
        "selection": selection,
        "no_veto_candidate": STUDIES[key]["no_veto"],
        "preregistration": (
            {"path": str(PREREG.relative_to(REPO)), "sha256": sha256_file(PREREG)}
            if PREREG.exists()
            else None
        ),
        "ratification": RATIFICATION_REF if not args.smoke else None,
        "base_harness": {"path": str(BASE_PATH.relative_to(REPO)), "sha256": BASE_SHA256},
        "rule_sha256": receipts.canonical_sha(STUDIES[key]["rule"]()),
        "checkpoints": base.checkpoints(key, candidates),
        "p2": None if args.smoke else p2_for(key, candidates, obs_spec, args.bv8_simd_only),
        "engine": {
            "sim_engine": base.SIM_ENGINE,
            "vector61_forward": base.VECTOR61_FORWARD,
            "ego2s_forward": base.EGO2S_FORWARD,
            "veto": base.VETO,
            "veto_lambda": base.VETO_LAMBDA,
            "candidate_veto": None if STUDIES[key]["no_veto"] else base.VETO,
            "candidate_ego2s_mask": "advisory" if STUDIES[key]["no_veto"] else None,
            "profile_h5000": base.PROFILE_DIGEST_H5000,
            "profile_h10000": base.PROFILE_DIGEST_H10000,
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


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("plan")
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--study", choices=sorted(STUDIES), required=True)
    p.add_argument("--candidates", type=Path, required=True)
    p.add_argument("--selection", type=Path, default=None, help="m4_select.py output (real plans)")
    p.add_argument(
        "--bv8-simd-only",
        action="store_true",
        help="ego2s-bv8 candidates: run without P2 (no live path); recorded in plan.json",
    )
    q = sub.add_parser("look")
    q.add_argument("--root", type=Path, required=True)
    q.add_argument("--look", type=int, required=True)
    q.add_argument("--procs", type=int, default=2)
    u = sub.add_parser("unit")
    u.add_argument("--root", type=Path, required=True)
    u.add_argument("--look", type=int, required=True)
    u.add_argument("--unit-id", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        return cmd_plan(args)
    if args.cmd == "look":
        if not 1 <= args.procs <= 3:
            raise SystemExit("compute policy: 1..3 processes")
        return base.cmd_look(args)
    return base.run_unit(args.root, args.look, args.unit_id)


if __name__ == "__main__":
    raise SystemExit(main())
