#!/usr/bin/env python3
"""T7 B8 no-veto Phase R runner.

This is deliberately a small, study-local adapter.  It evaluates exactly the
frozen fresh banks and writes immutable, hash-bound records; it never changes
serving configuration or starts a tmux session itself.
"""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import hashlib
import json
import os
import platform
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NUMBA_NUM_THREADS",
):
    os.environ[_name] = "1"
os.environ["SNAKE_DQN_DEVICE"] = "cpu"
os.environ["SNAKE_V61_BACKEND"] = "python"

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from research.redesign_t7_phase_r_20261016 import seed_audit  # noqa: E402
from research.sequential_phase_r import receipts  # noqa: E402

ART_ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-t7-phase-r-b8-20261016")
PREREG = REPO / "research/redesign_t7_phase_r_20261016/PREREGISTRATION_T7.md"
SEED_AUDIT = REPO / "research/redesign_t7_phase_r_20261016/SEED_AUDIT_T7.json"
ALLOCATION_ERRATUM = REPO / "research/redesign_t7_phase_r_20261016/SEED_ALLOCATION_ERRATUM_T7.md"
REJECTED_ALLOCATION = (
    REPO / "research/redesign_t7_phase_r_20261016/SEED_ALLOCATION_REJECTED_T7.json"
)
FINAL_ERRATUM = PREREG.parent / "SEED_ALLOCATION_ERRATUM_76_T7.md"
REJECTED_75 = PREREG.parent / "SEED_ALLOCATION_REJECTED_75_T7.json"
SCHEMA = "redesign-t7-phase-r-record/v1"
H5000, H10000, MAX_BATCH = 5000, 10000, 8
MIX_COUNTS = {"scripted": 192, "frozen": 96, "mixed": 96}
HERO, INCUMBENT, CHAMPION = "B8-noveto", "incumbent", "champion"
HEROES = (HERO, INCUMBENT, CHAMPION)
B8_SHA256 = {
    0: "bf6a42b387e6d643e943a03620a30128ec6032f984abb5a08559b9b0ac865f4f",
    1: "a7b04a84c94ad6cf90f6461a2ab23b214a28c4c59ac9e7bc69c61ca3ae617753",
    2: "5840ba615efd600325d5c5bcedf87b2cbb452fe0969d954d5797ab12f6917a0e",
    3: "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    4: "6b6d3f37c6db129de941c3fb205ca6af77c6506db0793b1291656f767e4fb28f",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def canonical_sha(value: Any) -> str:
    return receipts.canonical_sha(value)


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _git(args: list[str]) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO, check=True, text=True, capture_output=True
    ).stdout.strip()


def git_commit() -> tuple[str, bool]:
    return _git(["rev-parse", "HEAD"]), bool(_git(["status", "--porcelain"]))


def source_manifest() -> dict[str, str]:
    """Hash the relevant tracked source boundary once into the immutable plan."""
    files = _git(["ls-files"]).splitlines()
    keep = [
        p
        for p in files
        if p.endswith(".py")
        or p.endswith((".yaml", ".yml"))
        or p.startswith("web/")
        or p == "AGENTS.md"
    ]
    return {p: sha256_file(REPO / p) for p in keep if (REPO / p).is_file()}


def candidate_paths() -> dict[int, Path]:
    root = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m4b-mac-20261013")
    return {s: root / f"B8_s{s}" / "final.pth" for s in range(5)}


def checkpoint_manifest() -> dict[str, dict[str, str]]:
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    out = {
        f"{HERO}|s{s}": {"path": str(p), "sha256": B8_SHA256[s]}
        for s, p in candidate_paths().items()
    }
    for s, p in candidate_paths().items():
        if sha256_file(p) != B8_SHA256[s]:
            raise SystemExit(f"B8 seed {s} checkpoint does not match its pre-registration pin")
    for name, item in ((INCUMBENT, gid.FRP3_S12), (CHAMPION, dev_screen.CHAMPION)):
        p = gid.CHECKPOINT_DIR / item[0]
        out[name] = {"path": str(p), "sha256": item[1]}
        if sha256_file(p) != item[1]:
            raise SystemExit(f"checkpoint hash mismatch: {p}")
    for filename, digest in dev_screen.POOL:
        p = gid.CHECKPOINT_DIR / filename
        if sha256_file(p) != digest:
            raise SystemExit(f"opponent checkpoint hash mismatch: {p}")
        out[f"opponent|{digest}"] = {"path": str(p), "sha256": digest}
    return out


def _require_prereg_committed() -> str:
    if (
        not PREREG.exists()
        or PREREG.relative_to(REPO).as_posix() not in _git(["ls-files"]).splitlines()
    ):
        raise SystemExit("pre-registration must be tracked before plan")
    if not _git(["log", "-1", "--format=%H", "--", str(PREREG.relative_to(REPO))]):
        raise SystemExit("pre-registration must be committed before plan")
    return sha256_file(PREREG)


def _banks_from_audit() -> dict[int, list[int]]:
    if not SEED_AUDIT.exists():
        raise SystemExit("missing committed SEED_AUDIT_T7.json")
    audit = _json(SEED_AUDIT)
    if not audit.get("pass") or audit.get("audit_source_sha256") != sha256_file(
        REPO / "research/redesign_t7_phase_r_20261016/seed_audit.py"
    ):
        raise SystemExit("seed audit did not pass")
    generated_all = seed_audit.generate_banks()
    generated = generated_all["banks"]
    if (
        audit.get("banks") != generated
        or audit.get("namespace") != generated_all["namespace"]
        or audit.get("candidate_count") != generated_all["count"]
    ):
        raise SystemExit("committed audit bank identities differ from generated fixed banks")
    out = {s: [int(x) for x in generated[f"s{s}"]["seeds"]] for s in range(5)}
    if any(len(v) != 192 for v in out.values()):
        raise SystemExit("seed audit banks are incomplete")
    return out


def p2_evidence(candidates: Mapping[int, Path]) -> dict[str, Any]:
    """Use the established M4 live-serving P2 validator for the no-veto B8 path."""
    from research.redesign_m4_20261013 import phase_r_m4

    return phase_r_m4.p2_for("nv", candidates, "ego2s-bv8", False, "B8")


def build_plan(root: Path) -> dict[str, Any]:
    from research.redesign_t7_phase_r_20261016 import rule_spec_t7 as rules

    commit, dirty = git_commit()
    if dirty:
        raise SystemExit("refusing plan from a dirty worktree")
    prereg_sha = _require_prereg_committed()
    if (
        not ALLOCATION_ERRATUM.exists()
        or ALLOCATION_ERRATUM.relative_to(REPO).as_posix() not in _git(["ls-files"]).splitlines()
        or not _git(["log", "-1", "--format=%H", "--", str(ALLOCATION_ERRATUM.relative_to(REPO))])
    ):
        raise SystemExit("committed seed-allocation erratum required")
    banks = _banks_from_audit()
    candidates = candidate_paths()
    checks = checkpoint_manifest()
    rule = rules.rule()
    p2 = p2_evidence(candidates)
    from src.evaluation.strict_promotion import _expected_world_identity

    rosters = {}
    world_identities = {}
    for s, bank in banks.items():
        rosters[str(s)] = {}
        world_identities[str(s)] = {}
        for mix, count in MIX_COUNTS.items():
            design = bank_rows(bank, mix)
            rosters[str(s)][mix] = {
                str(w): [x["member_sha256"] for x in design[w]["slots"]] for w in bank[:count]
            }
            world_identities[str(s)][mix] = {
                str(w): _expected_world_identity(design[w]) for w in bank[:count]
            }
    if not REJECTED_ALLOCATION.exists():
        raise SystemExit("missing rejected allocation receipt")
    return {
        "schema": "t7-phase-r-plan/v1",
        "commit": commit,
        "preregistration": {"path": str(PREREG.relative_to(REPO)), "sha256": prereg_sha},
        "allocation_erratum": {
            "path": str(ALLOCATION_ERRATUM.relative_to(REPO)),
            "sha256": sha256_file(ALLOCATION_ERRATUM),
        },
        "rejected_allocation": {
            "path": str(REJECTED_ALLOCATION.relative_to(REPO)),
            "sha256": sha256_file(REJECTED_ALLOCATION),
        },
        "allocation_history": [
            {"path": str(p.relative_to(REPO)), "sha256": sha256_file(p)}
            for p in (FINAL_ERRATUM, REJECTED_75)
        ],
        "seed_audit": {
            "path": str(SEED_AUDIT.relative_to(REPO)),
            "sha256": sha256_file(SEED_AUDIT),
        },
        "rule": rule,
        "rule_sha256": canonical_sha(rule),
        "banks": {str(k): v for k, v in banks.items()},
        "rosters": rosters,
        "world_identities": world_identities,
        "mix_counts": MIX_COUNTS,
        "checkpoints": checks,
        "p2": p2,
        "source_manifest": source_manifest(),
        "heroes": list(HEROES),
        "engine": {
            "sim_engine": "grid",
            "vector61_forward": "rowwise",
            "ego2s_forward": "rowwise",
            "candidate_veto": False,
            "candidate_ego2s_mask": "advisory",
            "baseline_veto": "v8",
            "profile_h5000": "d396d3ed93e3a264d050674887eb47e59de67d3c6696c2c41fa5f8b145ea0e8b",
            "profile_h10000": "c882e07fb7179c01feed5380aeae228113975442bb10ab8840bc58a0011f5b15",
            "horizons": {"nested": H10000, "control": H5000},
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def write_plan(root: Path) -> str:
    root.mkdir(parents=True, exist_ok=True)
    plan = build_plan(root)
    return receipts.write_once(root / "plan.json", plan)


def load_plan(root: Path) -> dict[str, Any]:
    plan = _json(root / "plan.json")
    if (
        plan.get("schema") != "t7-phase-r-plan/v1"
        or canonical_sha(plan["rule"]) != plan["rule_sha256"]
    ):
        raise SystemExit("invalid plan binding")
    return plan


def _verify_plan_runtime(plan: Mapping[str, Any]) -> None:
    from research.redesign_t7_phase_r_20261016 import rule_spec_t7 as rules

    commit, dirty = git_commit()
    if dirty or commit != plan["commit"]:
        raise SystemExit("HEAD/worktree differs from frozen plan")
    if (
        sha256_file(PREREG) != plan["preregistration"]["sha256"]
        or sha256_file(ALLOCATION_ERRATUM) != plan["allocation_erratum"]["sha256"]
        or sha256_file(REJECTED_ALLOCATION) != plan["rejected_allocation"]["sha256"]
        or sha256_file(SEED_AUDIT) != plan["seed_audit"]["sha256"]
    ):
        raise SystemExit("preregistration or seed audit changed")
    registered_rule = rules.rule()
    audited_banks = {str(k): v for k, v in _banks_from_audit().items()}
    if (
        plan.get("banks") != audited_banks
        or plan.get("rule") != registered_rule
        or plan.get("rule_sha256") != canonical_sha(registered_rule)
    ):
        raise SystemExit("plan banks or rule differ from their registered source")
    for rel, digest in plan["source_manifest"].items():
        if sha256_file(REPO / rel) != digest:
            raise SystemExit(f"source changed since plan: {rel}")
    for row in plan["allocation_history"]:
        if sha256_file(REPO / row["path"]) != row["sha256"]:
            raise SystemExit("seed allocation history changed")
    for row in plan["checkpoints"].values():
        if sha256_file(Path(row["path"])) != row["sha256"]:
            raise SystemExit(f"checkpoint changed: {row['path']}")


def bank_rows(bank: Sequence[int], mix: str) -> dict[int, Mapping[str, Any]]:
    from research.apex_safety_20260926 import dev_screen

    rows = {
        int(x["world_seed"]): x
        for x in dev_screen._design_rows([int(w) for w in bank])
        if x["mix"] == mix
    }
    # _design_rows constructs all three rosters for every full-bank identity.  The
    # frozen/mixed study samples are the first 96 of that fixed 192-world order.
    if len(rows) != 192:
        raise SystemExit(f"full bank mix roster mismatch: {mix} has {len(rows)}")
    return rows


def units_for(plan: Mapping[str, Any]) -> list[dict[str, Any]]:
    units = []
    for hero in HEROES:
        for seed in range(5):
            bank = plan["banks"][str(seed)]
            for mix, count in MIX_COUNTS.items():
                worlds = bank[:count]
                if len(worlds) != count:
                    raise SystemExit("bank ordering lost")
                for lo in range(0, count, MAX_BATCH):
                    ws = worlds[lo : lo + MAX_BATCH]
                    units.append(
                        {
                            "unit_id": f"{hero}|s{seed}|{mix}|{lo}",
                            "hero": hero,
                            "seed": seed,
                            "mix": mix,
                            "worlds": ws,
                            "world_index_range": [lo, lo + len(ws)],
                            "control": False,
                        }
                    )
                # Direct-H5000 prefix control on first full-bank world: 45 total.
                units.append(
                    {
                        "unit_id": f"control|{hero}|s{seed}|{mix}",
                        "hero": hero,
                        "seed": seed,
                        "mix": mix,
                        "worlds": [worlds[0]],
                        "world_index_range": [0, 1],
                        "control": True,
                    }
                )
    return units


class PrefixObserver:
    """Exact M4-style H5000 observer, with pre-death length-bin diagnostics."""

    def __init__(self, n_envs: int, prefix: int = H5000):
        from src.evaluation.metrics import EvaluationMetricsAccumulator

        self.prefix, self.accumulators = prefix, [
            EvaluationMetricsAccumulator(prefix) for _ in range(n_envs)
        ]
        self.prev_alive: list[bool] | None = None
        self.length_exposure = [
            {"0-199": 0, "200-399": 0, "400-799": 0, "800+": 0} for _ in range(n_envs)
        ]
        self.deaths_pre_length = [{} for _ in range(n_envs)]

    @staticmethod
    def _bin(value: float) -> str:
        return (
            "0-199"
            if value < 200
            else "200-399" if value < 400 else "400-799" if value < 800 else "800+"
        )

    def __call__(self, payload: Mapping[str, Any]) -> None:
        from src.evaluation.metrics import PostStepState, StepEvents
        from src.simd_env.eval_engine import _DEATH_CAUSE_LABEL, DEATH_NONE

        frame, pre, post = int(payload["frame"]), payload["pre"], payload["post"]
        if self.prev_alive is None:
            self.prev_alive = [bool(x) for x in pre["alive"]]
        if frame < self.prefix:
            for i, acc in enumerate(self.accumulators):
                valid = bool(post["transition_valid"][i])
                died = bool(post["done"][i]) if valid else False
                mass = float(pre.get("logical_mass", post["logical_mass"])[i])
                bucket = self._bin(mass)
                if self.prev_alive[i] and valid:
                    self.length_exposure[i][bucket] += 1
                if died:
                    self.deaths_pre_length[i][bucket] = self.deaths_pre_length[i].get(bucket, 0) + 1
                cause = int(post["death_cause"][i]) if valid else DEATH_NONE
                acc.observe(
                    pre_alive=self.prev_alive[i],
                    post=PostStepState(
                        alive=bool(post["alive"][i]), logical_mass=float(post["logical_mass"][i])
                    ),
                    events=StepEvents(
                        food_eaten=int(post["food_ate"][i]) if valid else 0,
                        boost_executed=bool(post["boosted"][i]) if valid else False,
                        kills=int(post["kills"][i]) if valid else 0,
                        death=died,
                        death_cause=_DEATH_CAUSE_LABEL.get(cause) if died else None,
                    ),
                    acted=valid,
                )
        self.prev_alive = [bool(x) for x in post["alive"]]

    def result(self, i: int) -> dict[str, Any]:
        return self.accumulators[i].result()

    def diagnostic(self, i: int) -> dict[str, Any]:
        return {"exposure": self.length_exposure[i], "deaths_pre_length": self.deaths_pre_length[i]}


def prefix_identical(direct: Mapping[str, Any], prefix: Mapping[str, Any]) -> bool:
    for key, value in prefix.items():
        left = direct.get(key)
        if key == "probes" and isinstance(left, dict):
            left = {k: v for k, v in left.items() if k != "safety_veto"}
            value = {k: v for k, v in value.items() if k != "safety_veto"}
        if json.dumps(left, sort_keys=True) != json.dumps(value, sort_keys=True):
            return False
    return True


def _hero_call(
    hero: str, seed: int, ctx: Mapping[str, Any], plan: Mapping[str, Any]
) -> tuple[Any, dict[str, Any]]:
    from research.apex_safety_20260926 import dev_screen
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    if hero == HERO:
        row = plan["checkpoints"][f"{HERO}|s{seed}"]
        return ctx["lookup"][gid.FRP3_S12[1]], {
            "hero_ego2s": row["path"],
            "ego2s_forward": "rowwise",
            "ego2s_mask": "advisory",
            "veto": False,
        }
    if hero == INCUMBENT:
        return ctx["lookup"][gid.FRP3_S12[1]], {"veto": True}
    if hero == CHAMPION:
        return ctx["lookup"][dev_screen.CHAMPION[1]], {"veto": True}
    raise SystemExit("unknown hero")


def record_name(unit: Mapping[str, Any], world: int) -> str:
    return f"{'control-' if unit['control'] else ''}{unit['hero']}-s{unit['seed']}-{unit['mix']}-{world}.json"


def validate_start(root: Path, plan: Mapping[str, Any]) -> str:
    """Bind every unit and audit to the original, immutable launch marker."""
    start = _json(root / "start.json")
    expected = {
        "schema": "t7-phase-r-start/v1",
        "plan_sha256": sha256_file(root / "plan.json"),
        "commit": plan["commit"],
        "units": [u["unit_id"] for u in units_for(plan)],
    }
    if any(start.get(k) != v for k, v in expected.items()):
        raise SystemExit("start marker differs from plan")
    return sha256_file(root / "start.json")


def _envelope(row: Mapping[str, Any]) -> str:
    plain = dict(row)
    plain.pop("envelope_sha256", None)
    return canonical_sha(plain)


def _validate_existing(
    path: Path,
    unit: Mapping[str, Any],
    plan: Mapping[str, Any],
    *,
    allow_missing_sidecar: bool = False,
) -> None:
    row = _json(path)
    required = {
        "schema": SCHEMA,
        "unit_id": unit["unit_id"],
        "plan_sha256": sha256_file(path.parents[1] / "plan.json"),
        "source_manifest_sha256": canonical_sha(plan["source_manifest"]),
        "start_sha256": validate_start(path.parents[1], plan),
    }
    if any(row.get(k) != v for k, v in required.items()):
        raise SystemExit(f"invalid saved record: {path}")
    if row.get("envelope_sha256") != _envelope(row):
        raise SystemExit(f"record envelope changed: {path}")
    sidecar = path.with_suffix(".sha256.json")
    if not sidecar.exists() and allow_missing_sidecar:
        pass
    elif not sidecar.exists() or _json(sidecar) != {
        "record": path.name,
        "sha256": sha256_file(path),
    }:
        raise SystemExit(f"invalid saved record digest: {path}")
    expected = {
        "hero": unit["hero"],
        "seed": unit["seed"],
        "mix": unit["mix"],
        "control": unit["control"],
        "horizon": H5000 if unit["control"] else H10000,
        "world_index_range": unit["world_index_range"],
    }
    if (
        any(row.get(k) != v for k, v in expected.items())
        or int(row.get("world_seed", -1)) not in unit["worlds"]
    ):
        raise SystemExit(f"record identity mismatch: {path}")
    if (
        path.name != record_name(unit, int(row["world_seed"]))
        or row.get("commit") != plan["commit"]
    ):
        raise SystemExit(f"record filename/commit mismatch: {path}")
    want_roster = plan["rosters"][str(unit["seed"])][unit["mix"]][str(row["world_seed"])]
    if row.get("roster_member_sha256s") != want_roster or not isinstance(row.get("record"), dict):
        raise SystemExit(f"record roster/payload mismatch: {path}")
    ck = plan["checkpoints"].get(
        f"{HERO}|s{unit['seed']}" if unit["hero"] == HERO else unit["hero"]
    )
    if row.get("hero_checkpoint") != ck or row.get("engine", {}).get("sim_engine") != "grid":
        raise SystemExit(f"record hero/engine mismatch: {path}")
    expected_engine = {
        "sim_engine": "grid",
        "vector61_forward": "rowwise",
        "ego2s_forward": "rowwise" if unit["hero"] == HERO else None,
        "ego2s_mask": "advisory" if unit["hero"] == HERO else None,
        "veto": None if unit["hero"] == HERO else "v8",
    }
    if row.get("engine") != expected_engine:
        raise SystemExit(f"record engine modes differ: {path}")
    world = str(row["world_seed"])
    if row.get("world_index") != plan["banks"][str(unit["seed"])].index(int(world)):
        raise SystemExit(f"record world index differs: {path}")
    inner = row["record"]
    if (
        inner.get("seed") != int(world)
        or inner.get("evaluation_profile_digest")
        != plan["engine"]["profile_h5000" if unit["control"] else "profile_h10000"]
        or inner.get("world_identity")
        != plan["world_identities"][str(unit["seed"])][unit["mix"]][world]
        or inner.get("denominators", {}).get("scored_frames") != expected["horizon"]
    ):
        raise SystemExit(f"inner record world/profile/horizon differs: {path}")
    if unit["control"] and row.get("prefix_h5000") is not None:
        raise SystemExit(f"control prefix mismatch: {path}")
    if not unit["control"] and not isinstance(row.get("prefix_h5000"), dict):
        raise SystemExit(f"nested prefix missing: {path}")
    if (
        not unit["control"]
        and row["prefix_h5000"].get("denominators", {}).get("scored_frames") != H5000
    ):
        raise SystemExit(f"prefix horizon differs: {path}")


def run_unit(root: Path, unit_id: str) -> int:
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.evaluation.strict_promotion import _expected_world_identity
    from src.simd_env.eval_engine import run_simd_eval

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    if platform.system() != "Darwin" or not (root / "start.json").exists():
        raise SystemExit("Mac start marker required")
    plan = load_plan(root)
    _verify_plan_runtime(plan)
    start_sha = validate_start(root, plan)
    if (root / "decision.json").exists():
        raise SystemExit("decision already exists; T7 run is complete")
    unit = {u["unit_id"]: u for u in units_for(plan)}.get(unit_id)
    if unit is None:
        raise SystemExit("unknown unit")
    # A recovered coordinator cannot duplicate a still-running unit from an
    # externally interrupted prior coordinator.  This lock belongs only to this
    # unit process and never inspects or signals other processes.
    unit_lock = root / "unit-locks" / f"{hashlib.sha256(unit_id.encode()).hexdigest()}.lock"
    unit_lock.parent.mkdir(parents=True, exist_ok=True)
    lock_stream = unit_lock.open("w")
    try:
        fcntl.flock(lock_stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock_stream.close()
        raise SystemExit(f"unit already owned: {unit_id}")
    out = root / "records"
    names = [out / record_name(unit, w) for w in unit["worlds"]]
    for p in names:
        if p.exists():
            _validate_existing(p, unit, plan)
    if all(p.exists() for p in names):
        return 0
    ctx = gid._context(1)
    profile5 = ctx["profile"]
    if profile5.digest != plan["engine"]["profile_h5000"]:
        raise SystemExit("H5000 profile digest drift")
    profile10 = dataclasses.replace(profile5, scored_horizon=H10000)
    if profile10.digest != plan["engine"]["profile_h10000"]:
        raise SystemExit("H10000 profile digest drift")
    profile = profile5 if unit["control"] else profile10
    rows = bank_rows(plan["banks"][str(unit["seed"])], unit["mix"])
    seeds = [int(x) for x in unit["worlds"]]
    rosters = {
        s: [ctx["lookup"][slot["member_sha256"]] for slot in rows[s]["slots"]] for s in seeds
    }
    identities = {s: _expected_world_identity(rows[s]) for s in seeds}
    spec, extra = _hero_call(unit["hero"], unit["seed"], ctx, plan)
    veto = extra.pop("veto")
    observer = None if unit["control"] else PrefixObserver(len(seeds))
    started = time.monotonic()
    records = run_simd_eval(
        spec,
        rosters[seeds[0]],
        H5000 if unit["control"] else H10000,
        seeds,
        profile=profile,
        opponent_specs_by_world=rosters,
        world_identities=identities,
        mix_id=unit["mix"],
        frame_observer=observer,
        vector61=True,
        vector61_forward="rowwise",
        sim_engine="grid",
        **({"hero_safety_veto": "v8", "hero_safety_veto_lambda": 8.0} if veto else {}),
        **extra,
    )
    wall = time.monotonic() - started
    out.mkdir(parents=True, exist_ok=True)
    for i, (p, world) in enumerate(zip(names, seeds)):
        if p.exists():
            continue
        row = {
            "schema": SCHEMA,
            "unit_id": unit_id,
            "plan_sha256": sha256_file(root / "plan.json"),
            "source_manifest_sha256": canonical_sha(plan["source_manifest"]),
            "start_sha256": start_sha,
            "commit": plan["commit"],
            "hero": unit["hero"],
            "seed": unit["seed"],
            "mix": unit["mix"],
            "world_seed": world,
            "world_index": plan["banks"][str(unit["seed"])].index(world),
            "world_index_range": unit["world_index_range"],
            "roster_member_sha256s": [x["member_sha256"] for x in rows[world]["slots"]],
            "hero_checkpoint": plan["checkpoints"].get(
                f"{HERO}|s{unit['seed']}" if unit["hero"] == HERO else unit["hero"]
            ),
            "horizon": H5000 if unit["control"] else H10000,
            "control": unit["control"],
            "engine": {
                "sim_engine": "grid",
                "vector61_forward": "rowwise",
                "ego2s_forward": extra.get("ego2s_forward"),
                "ego2s_mask": extra.get("ego2s_mask"),
                "veto": "v8" if veto else None,
            },
            "record": json.loads(json.dumps(records[i])),
            "prefix_h5000": None if unit["control"] else observer.result(i),
            "length_hazard": None if unit["control"] else observer.diagnostic(i),
            "unit_wall_seconds": wall,
        }
        row["envelope_sha256"] = _envelope(row)
        receipts.write_once(p, row)
        receipts.write_once(
            p.with_suffix(".sha256.json"), {"record": p.name, "sha256": sha256_file(p)}
        )
    return 0


def guard_ok() -> list[str]:
    from research.redesign_m2_20261008.gen_data import guard_ok as check

    return list(check())


def run(root: Path, procs: int) -> int:
    if not 1 <= procs <= 6:
        raise SystemExit("procs must be 1..6")
    if platform.system() != "Darwin":
        raise SystemExit("T7 Phase R is Mac-only")
    plan = load_plan(root)
    _verify_plan_runtime(plan)
    if (root / "decision.json").exists():
        raise SystemExit("decision already exists; T7 run is complete")
    expected_records = {record_name(u, w) for u in units_for(plan) for w in u["worlds"]}
    existing_records = {
        p.name for p in (root / "records").glob("*.json") if not p.name.endswith(".sha256.json")
    }
    if existing_records - expected_records:
        raise SystemExit("unexpected saved records: refusing launch")
    lock = root / "run.lock"
    lock.parent.mkdir(parents=True, exist_ok=True)
    with lock.open("w") as lockf:
        try:
            fcntl.flock(lockf, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("another T7 coordinator owns run.lock")
        start_marker = {
            "schema": "t7-phase-r-start/v1",
            "plan_sha256": sha256_file(root / "plan.json"),
            "commit": plan["commit"],
            "units": [u["unit_id"] for u in units_for(plan)],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "started_epoch": time.time(),
        }
        if (root / "start.json").exists():
            old = _json(root / "start.json")
            fixed = {
                k: v for k, v in start_marker.items() if k not in ("started_utc", "started_epoch")
            }
            if {k: old.get(k) for k in fixed} != fixed or not isinstance(
                old.get("started_epoch"), (int, float)
            ):
                raise SystemExit("start marker differs from frozen plan")
        else:
            receipts.write_once(root / "start.json", start_marker)
        started = float(_json(root / "start.json")["started_epoch"])
        todo = []
        for u in units_for(plan):
            paths = [root / "records" / record_name(u, w) for w in u["worlds"]]
            for p in paths:
                if p.exists():
                    _validate_existing(p, u, plan, allow_missing_sidecar=True)
                    sidecar = p.with_suffix(".sha256.json")
                    if not sidecar.exists():
                        receipts.write_once(sidecar, {"record": p.name, "sha256": sha256_file(p)})
            if not all(p.exists() for p in paths):
                todo.append(u)
        running: list[tuple[subprocess.Popen, str]] = []
        previous_sigterm = signal.getsignal(signal.SIGTERM)

        def _owned_shutdown(_signum, _frame):
            raise SystemExit("T7 coordinator received SIGTERM")

        signal.signal(signal.SIGTERM, _owned_shutdown)
        try:
            while todo or running:
                bad = guard_ok()
                paused = bool(bad)
                for proc, _ in running:
                    if proc.poll() is None:
                        try:
                            os.killpg(proc.pid, signal.SIGSTOP if paused else signal.SIGCONT)
                        except ProcessLookupError:
                            pass
                if paused:
                    time.sleep(30)
                    continue
                next_running = []
                for proc, uid in running:
                    if proc.poll() is None:
                        next_running.append((proc, uid))
                    elif proc.returncode != 0:
                        for other, _ in running:
                            if other.poll() is None:
                                os.killpg(other.pid, signal.SIGTERM)
                        raise SystemExit(f"unit failed: {uid}")
                running = next_running
                while todo and len(running) < procs:
                    u = todo.pop(0)
                    env = dict(os.environ, SNAKE_DQN_DEVICE="cpu", SNAKE_V61_BACKEND="python")
                    proc = subprocess.Popen(
                        [
                            "nice",
                            "-n",
                            "10",
                            sys.executable,
                            str(Path(__file__).resolve()),
                            "unit",
                            "--root",
                            str(root),
                            "--unit-id",
                            u["unit_id"],
                        ],
                        cwd=REPO,
                        env=env,
                        start_new_session=True,
                    )
                    running.append((proc, u["unit_id"]))
                    print(json.dumps({"started": u["unit_id"], "remaining": len(todo)}), flush=True)
                time.sleep(2)
            result = decide(root, started)
            receipts.write_once(root / "decision.json", result)
        finally:
            for proc, _ in running:
                if proc.poll() is None:
                    try:
                        os.killpg(proc.pid, signal.SIGCONT)
                    except ProcessLookupError:
                        pass
                    try:
                        os.killpg(proc.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
            for proc, _ in running:
                try:
                    proc.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(proc.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                    proc.wait()
            signal.signal(signal.SIGTERM, previous_sigterm)
    return 0


def decide(root: Path, started: float | None = None) -> dict[str, Any]:
    from research.redesign_t7_phase_r_20261016 import rule_spec_t7 as rules

    plan = load_plan(root)
    start_sha = validate_start(root, plan)
    units = units_for(plan)
    by_name = {record_name(u, w): u for u in units for w in u["worlds"]}
    records = []
    for p in sorted((root / "records").glob("*.json")):
        if not p.name.endswith(".sha256.json"):
            if p.name not in by_name:
                raise SystemExit(f"unexpected record {p.name}")
            _validate_existing(p, by_name[p.name], plan)
            records.append(_json(p))
    if len(records) != len(by_name):
        raise SystemExit("incomplete record set: one look requires full N")
    controls = [x for x in records if x["control"]]
    nested = {
        (x["hero"], x["seed"], x["mix"], x["world_seed"]): x for x in records if not x["control"]
    }
    checks = [
        {
            "key": [x["hero"], x["seed"], x["mix"], x["world_seed"]],
            "identical": prefix_identical(
                x["record"],
                nested.get((x["hero"], x["seed"], x["mix"], x["world_seed"]), {}).get(
                    "prefix_h5000", {}
                ),
            ),
        }
        for x in controls
    ]
    decision = rules.decide(
        [x for x in records if not x["control"]],
        {int(k): v for k, v in plan["banks"].items()},
        len(checks) == 45 and all(x["identical"] for x in checks),
    )
    inventory = [
        {"path": str(p.relative_to(root)), "sha256": sha256_file(p)}
        for p in sorted((root / "records").glob("*.json"))
        if not p.name.endswith(".sha256.json")
    ]
    return {
        "schema": "t7-phase-r-decision/v1",
        "plan_sha256": sha256_file(root / "plan.json"),
        "start_sha256": start_sha,
        "records": inventory,
        "prefix_controls": checks,
        "decision": decision,
        "wall_seconds": None if started is None else time.time() - started,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }


def audit(root: Path) -> int:
    plan = load_plan(root)
    _verify_plan_runtime(plan)
    expected = {record_name(u, w) for u in units_for(plan) for w in u["worlds"]}
    found = {
        p.name for p in (root / "records").glob("*.json") if not p.name.endswith(".sha256.json")
    }
    report = {
        "schema": "t7-phase-r-audit/v1",
        "plan_valid": True,
        "coverage": {"expected": len(expected), "found": len(found), "exact": expected == found},
        "records_valid": True,
        "decision_valid": False,
    }
    by_name = {record_name(u, w): u for u in units_for(plan) for w in u["worlds"]}
    for p in (root / "records").glob("*.json"):
        if p.name.endswith(".sha256.json"):
            continue
        try:
            _validate_existing(p, by_name[p.name], plan)
        except Exception as e:
            report["records_valid"] = False
            report.setdefault("errors", []).append(str(e))
    dpath = root / "decision.json"
    if dpath.exists() and report["coverage"]["exact"]:
        old = _json(dpath)
        new = decide(root)
        report["decision_valid"] = (
            old["decision"] == new["decision"]
            and old["records"] == new["records"]
            and old.get("start_sha256") == new["start_sha256"]
            and old.get("plan_sha256") == new["plan_sha256"]
            and new["decision"].get("valid") is True
        )
    receipts.write_once(root / "audit.json", report)
    print(json.dumps(report, sort_keys=True))
    return (
        0
        if all((report["coverage"]["exact"], report["records_valid"], report["decision_valid"]))
        else 1
    )


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("plan", "run", "unit", "audit"):
        p = sub.add_parser(name)
        p.add_argument("--root", type=Path, default=ART_ROOT)
        if name == "run":
            p.add_argument("--procs", type=int, default=6)
        if name == "unit":
            p.add_argument("--unit-id", required=True)
    args = ap.parse_args(argv)
    if args.cmd == "plan":
        print(json.dumps({"plan_sha256": write_plan(args.root)}))
        return 0
    if args.cmd == "unit":
        return run_unit(args.root, args.unit_id)
    if args.cmd == "run":
        return run(args.root, args.procs)
    return audit(args.root)


if __name__ == "__main__":
    raise SystemExit(main())
