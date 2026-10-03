#!/usr/bin/env python3
"""Tier-1 screen (non-authoritative): Apex + v8 (lambda=8) vs Apex + v7 (lambda=4).

Pre-registration: ``protocol.md`` beside this file. Arms (paired on every world):

  A  champion + ``free-space-veto/v7-space-preference(lambda=4.0)`` (incumbent, STRICT_PASS vs v5).
  B  champion + ``free-space-veto/v8-space-and-head(lambda=8.0)`` (reference diagnostic at 4.0).
  C  A repeated on the first 2 worlds per mix, in another shard (determinism control).
  D  B repeated on the first 2 worlds per mix, in another shard (replay control).

Built on ``research/apex_safety_20260926/dev_screen.py`` and the v8 DEV sweep
(``research/apex_veto_v8_lambda_sweep_20261002/sweep.py``: installers, episode loop with the
thermal-guard admission, guard tallies), unchanged. Three shard processes, one pool-3 CPU slot
each (no start barrier); ``merge`` writes one ``summary.json`` with the pre-registered decision.

Usage (see protocol.md for the full operator commands):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v8_screen_20261002/screen.py \\
    run --shard K --out <dir>/shard-K --deadline-utc <now + <= 4 h> \\
    --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power      # K = 0, 1, 2
  ... screen.py merge --shard-dirs <dir>/shard-0 <dir>/shard-1 <dir>/shard-2 --out <dir>/merged
Smoke (plumbing only, smoke namespace, arms A and B on 1 world, <= 500 frames):
  ... screen.py run --shard 0 --shards 1 --out /tmp/x --deadline-utc ... --smoke-frames 500 \\
    --worlds-per-mix 1 --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import argparse  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402
from research.apex_veto_v7_lambda_sweep_20261002 import sweep as v7sweep  # noqa: E402
from research.apex_veto_v8_lambda_sweep_20261002 import sweep  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREEN_ID = "apex-veto-v8-screen-v1"
SCHEMA = "apex-veto-v8-screen/v1"
SHARD_SCHEMA = "apex-veto-v8-screen-shard/v1"
AUTHORITY = "screen (Tier-1, non-authoritative)"
DOMAIN = SCREEN_ID
SMOKE_DOMAIN = "apex-veto-v8-screen-smoke-v1"
NAMESPACE = "worlds"
A_LAMBDA = 4.0
B_LAMBDA = 8.0
V7_ARMS = ("A", "C")
V8_ARMS = ("B", "D")
PAIR_ARMS = ("A", "B")
CONTROLS = {"C": "A", "D": "B"}  # control arm -> the arm it repeats
CONTROL_WORLDS = 2
WORLDS_PER_MIX = 60
SHARDS = 3
SLOT_POOL = 3
MAX_WALL_SECONDS = 4 * 3600
DEFAULT_SLOT_TIMEOUT_SECONDS = 180.0
# Start barrier (non-smoke): the sweep's rendezvous; the 3 shards hold the 3 pool-3 slots.
BARRIER_DIRNAME = sweep.BARRIER_DIRNAME
BARRIER_TIMEOUT_SECONDS = sweep.BARRIER_TIMEOUT_SECONDS
# Fixed operating parameters outside a smoke (protocol.md "Compute cap and operations").
FIXED_RUN_PARAMS = {
    "thermal_backoff_seconds": 60.0,
    "thermal_max_backoffs": 5,
    "min_episode_budget_seconds": 45.0,
}
# Refuse to start with less time than the largest shard (126 episodes) x 35 s x 1.3.
MIN_START_SECONDS = 126 * 35 * 1.3
EXCLUSION_PREFIX = 1000
CONFIDENCE_ONE_SIDED = 0.90
NORMAL_Q90 = 1.2815515655446004

STATUS_ADVANCE = "ADVANCE"
STATUS_NOT_ADVANCED = "NOT_ADVANCED"
DECISION_RULE = (
    "per world and mix d = mass_integral(B) - mass_integral(A) (B = v8 lambda 8, A = v7 "
    "lambda 4). Per mix: mean, sd, one-sided 90% bounds mean -/+ t(0.90, n-1) sd/sqrt(n); a "
    "mix LOSS iff its upper bound < 0. Pooled: equal-weight mean of the 3 mix means, SE^2 = "
    "(1/9) sum s_m^2/n_m, Welch-Satterthwaite df, lower bound mean - t(0.90, df) SE. In order: "
    "SMOKE_NO_DECISION, INVALID_SELF_CHECK_FAILED, INCOMPLETE, INVALID_NONDETERMINISTIC (C vs "
    "A, D vs B); ADVANCE iff pooled mean > 0 AND pooled lower bound > 0 AND no mix LOSS; "
    "else NOT_ADVANCED."
)

# The DEV sweep that selected lambda = 8 (protocol.md, "where lambda = 8 came from").
SWEEP_SUMMARY = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-lambda-sweep-20261002"
    "/run-v1/merged/summary.json"
)
SWEEP_SUMMARY_SHA256 = "2a3eb8b30cb5911e713ec78d4028b5483f78628c0b9dbf31bfedca41814720b2"

ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
# Real-run output roots (the run and its one allowed mechanical-defect recovery).
RUN_ROOT = ARTIFACTS / "apex-veto-v8-screen-20261002" / "run-v1"
ALLOWED_RUN_ROOTS: Tuple[Path, ...] = (RUN_ROOT, RUN_ROOT / "recovery-1")
SLOT_LOCK_ROOT = dev_screen.DEFAULT_SLOT_LOCK_ROOT
PACKAGE_FILES = ("screen.py", "protocol.md")
# Saved artifacts whose observed world seeds must not overlap this screen (fail closed).
OBSERVED_SEED_SOURCES: Tuple[Path, ...] = (
    ARTIFACTS / "apex-veto-v7-strict-20261002/run-v1/output/rosters.json",
    ARTIFACTS / "apex-veto-v8-lambda-sweep-20261002/run-v1/shard-0/intent.json",
    ARTIFACTS / "apex-veto-v8-lambda-sweep-20261002/run-v1/shard-1/intent.json",
    ARTIFACTS / "apex-veto-v8-lambda-sweep-20261002/run-v1/shard-2/intent.json",
    ARTIFACTS / "apex-veto-v8-lambda-sweep-20261002/run-v1/merged/summary.json",
    ARTIFACTS / "apex-veto-v7-lambda-sweep-20261002/run-v1/intent.json",
    ARTIFACTS / "apex-veto-v7-screen-20261002/run-v1/intent.json",
    ARTIFACTS / "apex-veto-v7-serving-20261002/run-v1/intent.json",
)
SEED_KEYS = ("world_seed", "seeds", "paired_seeds", "world_seeds")

V7_WEB_DOMAINS = ("apex-veto-v7-web-serving-v1", "apex-veto-v7-web-serving-smoke-v1")
EARLIER_DOMAINS: Dict[str, Tuple[str, ...]] = sweep._merge_domains(
    sweep.EARLIER_DOMAINS,
    {name: sweep.WEB_PURPOSES for name in V7_WEB_DOMAINS},
    {DOMAIN: (NAMESPACE,), SMOKE_DOMAIN: (NAMESPACE,)},
)


def earlier_namespaces(domain: str, prefix: int = EXCLUSION_PREFIX) -> Dict[str, List[int]]:
    """``domain/purpose[0:prefix]`` -> seeds for every earlier domain except ``domain``."""
    return {
        f"{name}/{purpose}[0:{prefix}]": [
            dev_screen.uint32_seed(name, purpose, index) for index in range(prefix)
        ]
        for name, purposes in EARLIER_DOMAINS.items()
        if name != domain
        for purpose in purposes
    }


def _collect_seeds(value: Any, out: set, under_key: bool = False) -> None:
    """Integers stored under any of :data:`SEED_KEYS` (lists, dicts of lists, scalars)."""
    if isinstance(value, Mapping):
        for key, item in value.items():
            _collect_seeds(item, out, under_key or key in SEED_KEYS)
    elif isinstance(value, list):
        for item in value:
            _collect_seeds(item, out, under_key)
    elif under_key and isinstance(value, int) and not isinstance(value, bool):
        out.add(int(value))


def observed_seed_namespaces(
    sources: Sequence[Path] = OBSERVED_SEED_SOURCES,
) -> Dict[str, List[int]]:
    """``observed:<path>`` -> seeds saved in each source file; a missing file raises."""
    out: Dict[str, List[int]] = {}
    for path in sources:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"observed-seed source missing: {path}")
        seeds: set = set()
        _collect_seeds(json.loads(path.read_text(encoding="utf-8")), seeds)
        if not seeds:
            raise ValueError(f"observed-seed source has no seeds: {path}")
        out[f"observed:{path}"] = sorted(seeds)
    return out


INSTALL_A = sweep.INSTALLERS["A"]  # v7 lambda 4
INSTALL_B = sweep.INSTALLERS["H800"]  # v8 lambda 8, reference_lambda 4
INSTALLERS: Dict[str, Any] = {"A": INSTALL_A, "B": INSTALL_B, "C": INSTALL_A, "D": INSTALL_B}


def arm_method(arm: str) -> str:
    from src.evaluation import safety_veto_v7, safety_veto_v8

    if arm in V7_ARMS:
        return safety_veto_v7.method_for(A_LAMBDA)
    return safety_veto_v8.method_for(B_LAMBDA)


def expected_descriptor(arm: str) -> Dict[str, Any]:
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
    from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

    if arm in V7_ARMS:
        return SpacePreferenceVeto(A_LAMBDA).descriptor()
    return SpaceAndHeadVeto(B_LAMBDA).descriptor()


def make_spec(domain: str, extra_namespaces: Mapping[str, Sequence[int]]) -> dev_screen.ScreenSpec:
    return dev_screen.ScreenSpec(
        name=SCREEN_ID,
        schema=SCHEMA,
        authority=AUTHORITY,
        domain=domain,
        namespace=NAMESPACE,
        protocol=HERE / "protocol.md",
        arm_vetoes=dict(INSTALLERS),
        extra_namespaces=dict(extra_namespaces),
        require_slot_locks=True,
        require_ac_power=True,
        owner="Apex safety lane",
        hypothesis="v8(lambda=8) raises the H5000 mass_integral over v7(lambda=4) (B - A > 0) "
        "with no significant loss in any mix",
        decision_informs="ADVANCE: design a Tier-2 strict gate for v8(8) vs v7(4); "
        "NOT_ADVANCED: stop",
        primary_metric="paired mass_integral B - A per world per mix",
        estimator="stratified mean of mix means, one-sided 90% bounds (see protocol.md)",
        max_wall_seconds=MAX_WALL_SECONDS,
    )


# ---------------------------------------------------------------- sweep binding


class SweepBindingError(ValueError):
    """The DEV sweep summary does not authorize this screen."""


def check_sweep_binding(path: Path = SWEEP_SUMMARY, sha256: str = SWEEP_SUMMARY_SHA256) -> Dict:
    """The pinned v8 DEV sweep summary: same bytes, real run, SELECTED lambda 8."""
    path = Path(path)
    if not path.is_file():
        raise SweepBindingError(f"sweep summary {path} does not exist")
    actual = dev_screen.sha256_file(path)
    if actual != sha256:
        raise SweepBindingError(f"sweep summary sha256 {actual} is not the pinned {sha256}")
    summary = json.loads(path.read_text(encoding="utf-8"))
    selection = summary.get("selection") or {}
    if (
        summary.get("schema_version") != sweep.SCHEMA
        or summary.get("sweep_id") != sweep.SWEEP_ID
        or summary.get("smoke") is not False
        or selection.get("status") != sweep.STATUS_SELECTED
        or selection.get("passes") is not True
        or selection.get("selected_lambda") != B_LAMBDA
    ):
        raise SweepBindingError("the sweep summary does not select lambda 8 in a real run")
    return {
        "summary_path": str(path),
        "summary_sha256": actual,
        "selected_lambda": selection["selected_lambda"],
        "source_commit": (summary.get("source") or {}).get("commit"),
    }


# ---------------------------------------------------------------- plan


def shard_plan(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    mixes: Sequence[str],
    shard: int,
    shards: int,
    smoke: bool,
) -> List[Tuple[str, Mapping[str, Any]]]:
    """One shard's episodes (protocol.md, "Shards")."""
    if smoke:
        return [(arm, row) for row in rows for arm in PAIR_ARMS]
    index = {int(seed): i for i, seed in enumerate(seeds)}
    order = {mix: i for i, mix in enumerate(mixes)}
    mine = [row for j, row in enumerate(rows) if j % shards == shard]
    mine.sort(key=lambda row: (index[int(row["world_seed"])], order[row["mix"]]))
    plan: List[Tuple[str, Mapping[str, Any]]] = [(arm, row) for row in mine for arm in PAIR_ARMS]
    for control in CONTROLS:
        for mix in mixes:
            first = [(j, row) for j, row in enumerate(rows) if row["mix"] == mix][:CONTROL_WORLDS]
            plan.extend((control, row) for j, row in first if (j + 1) % shards == shard)
    return plan


plan_keys = sweep.plan_keys


# ---------------------------------------------------------------- self-check (gating)


def check_entry(entry: Mapping[str, Any], smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one entry (protocol.md, "Self-checks")."""
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        _expected_world_identity,
        _validate_candidate_wrapper_probe,
    )

    arm = entry.get("arm")
    if arm not in INSTALLERS:
        return {"failures": [f"unknown arm {arm!r}"], "warnings": []}
    failures: List[str] = []
    warnings: List[str] = []
    if entry.get("schema_version") != SCHEMA or entry.get("authority") != AUTHORITY:
        failures.append("entry schema/authority is not this screen's")
    if entry.get("screen") != SCREEN_ID or entry.get("hero_sha256") != dev_screen.CHAMPION[1]:
        failures.append("entry screen id or hero sha256 differs")
    if entry.get("safety_veto") is not True or entry.get("safety_veto_method") != arm_method(arm):
        failures.append(f"entry veto flag/method is not arm {arm}'s")
    record = entry.get("record") if isinstance(entry.get("record"), Mapping) else {}
    probe = (record.get("probes") or {}).get("safety_veto")
    try:
        _validate_candidate_wrapper_probe(probe, expected_descriptor(arm))
    except (StrictPromotionArtifactError, TypeError, KeyError) as exc:
        failures.append(f"safety_veto probe is not arm {arm}'s veto: {exc}")
    if record.get("seed") != entry.get("world_seed"):
        failures.append("record seed differs from the entry's world seed")
    mass = record.get("mass_integral")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or not math.isfinite(mass):
        failures.append("mass_integral is not a finite number")
    counters = (probe.get("counters") or {}) if isinstance(probe, Mapping) else {}
    if not smoke:
        if record.get("evaluation_profile_digest") != dev_screen.PROFILE_DIGEST:
            failures.append("evaluation_profile_digest is not promotion-v2-watch-rect")
        row = {
            "world_seed": entry.get("world_seed"),
            "mix": entry.get("mix"),
            "slots": [{"member_sha256": s} for s in entry.get("roster_member_sha256s") or []],
        }
        if record.get("world_identity") != _expected_world_identity(row):
            failures.append("world_identity does not match the materialized roster")
        denominators = record.get("denominators") or {}
        if denominators.get("scored_frames") != dev_screen.HORIZON:
            failures.append("denominators.scored_frames is not the H5000 horizon")
        if counters.get("decisions") != denominators.get("decision_frames"):
            failures.append("veto decisions differ from denominators.decision_frames")
    diagnostics = entry.get("veto_diagnostics")
    if arm in V7_ARMS:
        if not isinstance(diagnostics, Mapping) or "rerank_changes" not in diagnostics:
            warnings.append(f"arm {arm} v7 counters missing")
    elif (
        not isinstance(diagnostics, Mapping)
        or "action_differs_from_reference" not in diagnostics
        or diagnostics.get("reference_lambda") != A_LAMBDA
        or diagnostics.get("reference_decisions") != counters.get("decisions")
    ):
        failures.append(f"arm {arm} v8 reference diagnostic missing or not at lambda {A_LAMBDA}")
    elif not sweep.v8_identities_hold(diagnostics, counters):
        warnings.append(f"arm {arm} v8 counters disagree with the probe's v2 counters")
    return {"failures": failures, "warnings": warnings}


def self_check(entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], smoke: bool) -> Dict:
    """Check every entry; ``passes`` iff there are entries and no gating failure."""
    planned = {int(seed) for seed in seeds}
    failures: List[str] = []
    warnings: List[str] = []
    seen = set()
    for entry in entries:
        name = f"{entry.get('arm')}-{entry.get('mix')}-{entry.get('world_seed')}"
        result = check_entry(entry, smoke)
        if entry.get("world_seed") not in planned:
            result["failures"].append("world seed is not a planned seed")
        if name in seen:
            result["failures"].append("duplicate episode")
        seen.add(name)
        failures.extend(f"{name}: {message}" for message in result["failures"])
        warnings.extend(f"{name}: {message}" for message in result["warnings"])
    return {
        "entries": len(entries),
        "failures": failures,
        "warnings": warnings,
        "passes": not failures and bool(entries),
    }


# ---------------------------------------------------------------- statistics


def t90(df: float) -> float:
    """One-sided 90% Student-t quantile (normal quantile for infinite df)."""
    from src.scripts.eval_stats import student_t_isf

    if math.isinf(df):
        return NORMAL_Q90
    return student_t_isf(1.0 - CONFIDENCE_ONE_SIDED, df)


def mix_bounds(deltas: Sequence[float]) -> Dict[str, Any]:
    """Mean, SD and one-sided 90% bounds of one mix's paired deltas (None below 2)."""
    n = len(deltas)
    mean = v7sweep._mean(deltas)
    sd = v7sweep._sd(deltas)
    if sd is None:
        return {"n": n, "mean": mean, "sd": sd, "se": None, "lower_90": None, "upper_90": None}
    se = sd / math.sqrt(n)
    half = t90(n - 1) * se
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "t": t90(n - 1),
        "lower_90": mean - half,
        "upper_90": mean + half,
    }


def pooled_bound(deltas_by_mix: Mapping[str, Sequence[float]]) -> Dict[str, Any]:
    """Equal-weight mean of mix means, Welch-Satterthwaite SE/df, one-sided 90% lower bound."""
    if not deltas_by_mix or any(len(d) < 2 for d in deltas_by_mix.values()):
        return {"mean": None, "se": None, "df": None, "lower_90": None}
    m = len(deltas_by_mix)
    means = [sum(d) / len(d) for d in deltas_by_mix.values()]
    terms = [v7sweep._sd(d) ** 2 / len(d) for d in deltas_by_mix.values()]
    dfs = [len(d) - 1 for d in deltas_by_mix.values()]
    mu = sum(means) / m
    variance = sum(terms) / (m * m)
    se = math.sqrt(variance)
    denom = sum((term / (m * m)) ** 2 / df for term, df in zip(terms, dfs))
    df = variance**2 / denom if denom > 0 else math.inf
    lower = mu - t90(df) * se if se > 0 else mu
    return {"mean": mu, "se": se, "df": df, "t": t90(df), "lower_90": lower, "strata": m}


def decide(per_mix: Mapping[str, Mapping[str, Any]], pooled: Mapping[str, Any]) -> Dict[str, Any]:
    """Step 5 of the rule: ADVANCE iff pooled mean > 0, pooled LB > 0 and no mix loss."""
    losses = [
        mix
        for mix, row in per_mix.items()
        if row.get("upper_90") is not None and row["upper_90"] < 0
    ]
    mean, lower = pooled.get("mean"), pooled.get("lower_90")
    criteria = {
        "pooled_mean_positive": mean is not None and mean > 0,
        "pooled_lower_90_positive": lower is not None and lower > 0,
        "no_mix_loss": not losses,
    }
    status = STATUS_ADVANCE if all(criteria.values()) else STATUS_NOT_ADVANCED
    return {"status": status, "criteria": criteria, "loss_mixes": losses}


def control_report(entries: Sequence[Mapping[str, Any]], planned: int) -> Dict[str, Any]:
    """C vs A and D vs B: every control record must equal the record it repeats."""
    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e["record"] for e in entries}
    controls = [e for e in entries if e["arm"] in CONTROLS]
    mismatches = [
        {"arm": e["arm"], "mix": e["mix"], "world_seed": e["world_seed"]}
        for e in controls
        if (CONTROLS[e["arm"]], e["mix"], int(e["world_seed"])) not in by_key
        or dev_screen.canonical_json(by_key[(CONTROLS[e["arm"]], e["mix"], int(e["world_seed"]))])
        != dev_screen.canonical_json(e["record"])
    ]
    return {
        "rule": "C repeats A and D repeats B, each in another shard process",
        "planned": planned,
        "compared": len(controls),
        "identical": len(controls) - len(mismatches),
        "mismatches": mismatches,
        "passes": len(controls) == planned and not mismatches,
    }


def pairing_failures(entries: Sequence[Mapping[str, Any]]) -> List[str]:
    """Gating: every arm on one (mix, seed) played the same roster and world identity."""
    groups: Dict[Tuple[str, int], List[Mapping[str, Any]]] = {}
    for entry in entries:
        groups.setdefault((str(entry.get("mix")), int(entry.get("world_seed"))), []).append(entry)
    failures = []
    for (mix, seed), group in sorted(groups.items()):
        rosters = {json.dumps(e.get("roster_member_sha256s")) for e in group}
        worlds = {
            dev_screen.canonical_json((e.get("record") or {}).get("world_identity")) for e in group
        }
        if len(rosters) > 1 or len(worlds) > 1:
            failures.append(f"{mix}-{seed}: arms differ in roster or world_identity (unpaired)")
    return failures


def _cause(record: Mapping[str, Any]) -> str:
    cause = (record.get("probes") or {}).get("death_cause")
    return "survived" if cause is None else str(cause)


V8_REPORTED = (
    "decisions",
    "head_checks",
    "head_risky_decisions",
    "head_risky_vetoes",
    "head_risky_kept_no_alternative",
    "head_risk_waived_hero_wins",
    "head_vetoes_speed_switched",
    "head_vetoes_of_v7_kept",
    "head_vetoes_of_v7_rerank",
    "action_differs_from_reference",
    "apply_seconds_total",
)


def paired_table(
    entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], mixes: Sequence[str]
) -> Dict[str, Any]:
    """Per mix: paired mass deltas with bounds, survival, death causes, B's v8 counters."""
    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e for e in entries}
    per_mix: Dict[str, Any] = {}
    for mix in mixes:
        paired = [s for s in seeds if ("A", mix, s) in by_key and ("B", mix, s) in by_key]
        a = [by_key[("A", mix, s)]["record"] for s in paired]
        b = [by_key[("B", mix, s)]["record"] for s in paired]
        deltas = [rb["mass_integral"] - ra["mass_integral"] for ra, rb in zip(a, b)]
        surv = [rb["survival_fraction"] - ra["survival_fraction"] for ra, rb in zip(a, b)]
        causes = {
            arm: {
                cause: sum(1 for r in records if _cause(r) == cause)
                for cause in sorted({_cause(r) for r in records})
            }
            for arm, records in (("A", a), ("B", b))
        }
        b_diag = [by_key[("B", mix, s)].get("veto_diagnostics") or {} for s in paired]
        per_mix[mix] = {
            "paired_worlds": len(paired),
            "paired_seeds": paired,
            "mass_integral": {
                "mean_A": v7sweep._mean([r["mass_integral"] for r in a]),
                "mean_B": v7sweep._mean([r["mass_integral"] for r in b]),
                "deltas_B_minus_A": deltas,
                **mix_bounds(deltas),
                "wins_B": sum(1 for d in deltas if d > 0),
                "losses_B": sum(1 for d in deltas if d < 0),
                "ties": sum(1 for d in deltas if d == 0),
            },
            "survival_fraction": {
                "mean_A": v7sweep._mean([r["survival_fraction"] for r in a]),
                "mean_B": v7sweep._mean([r["survival_fraction"] for r in b]),
                "mean_delta": v7sweep._mean(surv),
                "survived_A": causes["A"].get("survived", 0),
                "survived_B": causes["B"].get("survived", 0),
            },
            "death_causes": causes,
            "head_on_deaths": {arm: causes[arm].get("head_on", 0) for arm in ("A", "B")},
            "v8_counters_B": {
                key: sum(float(d.get(key) or 0) for d in b_diag) for key in V8_REPORTED
            },
            "episodes_B_differs_from_A_rule": sum(
                1 for d in b_diag if (d.get("action_differs_from_reference") or 0) > 0
            ),
        }
    return per_mix


def summarize(
    entries: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    mixes: Sequence[str],
    *,
    smoke: bool,
    planned_episodes: int,
    control_planned: int,
    source: Mapping[str, Any],
    checker: Callable[..., Dict[str, Any]] | None = None,
) -> Dict[str, Any]:
    """The pre-registered decision (protocol.md) plus reported tables. JSON-safe."""
    check = dict((checker or self_check)(entries, seeds, smoke))
    pairing = pairing_failures(entries)
    if pairing:
        check["failures"] = list(check.get("failures") or []) + pairing
        check["passes"] = False
    per_mix = paired_table(entries, seeds, mixes)
    controls = control_report(entries, control_planned)
    complete = len(entries) == planned_episodes and all(
        per_mix[mix]["paired_worlds"] == len(seeds) for mix in mixes
    )
    deltas = {mix: per_mix[mix]["mass_integral"]["deltas_B_minus_A"] for mix in mixes}
    pooled = pooled_bound(deltas)
    rule = decide({m: per_mix[m]["mass_integral"] for m in mixes}, pooled)
    if smoke:
        decision = "SMOKE_NO_DECISION"
    elif not check["passes"]:
        decision = "INVALID_SELF_CHECK_FAILED"
    elif not complete:
        decision = "INCOMPLETE"
    elif not controls["passes"]:
        decision = "INVALID_NONDETERMINISTIC"
    else:
        decision = rule["status"]
    all_deltas = [d for mix in mixes for d in deltas[mix]]
    informational: Dict[str, Any] = {}
    if not smoke and all(len(deltas[m]) >= 2 for m in mixes):
        informational = dict(
            dev_screen.pooled_summaries(deltas, {m: per_mix[m]["paired_seeds"] for m in mixes})
        )
        informational["role"] = (
            "informational only; the decision uses pooled.lower_90 and each mix's upper_90 "
            "(protocol.md), no Holm test is run"
        )
    head_on = {arm: sum(per_mix[m]["head_on_deaths"][arm] for m in mixes) for arm in ("A", "B")}
    wall = {
        arm: [float(e["wall_seconds"]) for e in entries if e["arm"] == arm]
        for arm in (*PAIR_ARMS, *CONTROLS)
    }
    summary = {
        "schema_version": SCHEMA,
        "screen_id": SCREEN_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "smoke": smoke,
        "complete": complete,
        "decision": decision,
        "decision_rule": DECISION_RULE,
        "rule_outcome": rule,
        "arms": {"A": arm_method("A"), "B": arm_method("B")},
        "per_mix": per_mix,
        "pooled": {
            **pooled,
            "pairs": len(all_deltas),
            "wins_B": sum(1 for d in all_deltas if d > 0),
            "losses_B": sum(1 for d in all_deltas if d < 0),
            "ties": sum(1 for d in all_deltas if d == 0),
            "sd_all_pairs": v7sweep._sd(all_deltas),
        },
        "pooled_informational": informational,
        "head_on_deaths_total": head_on,
        "control": controls,
        "self_check": check,
        "source": {"commit": source.get("commit"), "dirty_paths": source.get("dirty_paths")},
        "wall_seconds": {
            arm: {"episodes": len(v), "mean": v7sweep._mean(v), "total": sum(v)}
            for arm, v in wall.items()
        },
    }
    return dev_screen.json_safe(summary)


# ---------------------------------------------------------------- arguments and run


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="play one shard (needs the slot/guard flags)")
    run.add_argument("--out", required=True, type=Path, help="new output dir (must not exist)")
    run.add_argument("--shard", type=int, required=True)
    run.add_argument("--shards", type=int, default=SHARDS)
    run.add_argument("--worlds-per-mix", type=int, default=WORLDS_PER_MIX)
    run.add_argument("--deadline-utc", required=True, type=dev_screen._utc)
    run.add_argument("--config", type=Path, default=dev_screen.DEFAULT_CONFIG)
    run.add_argument("--checkpoint-dir", type=Path, default=dev_screen.DEFAULT_CHECKPOINT_DIR)
    run.add_argument("--pilot-output", type=Path, default=dev_screen.DEFAULT_PILOT_OUTPUT)
    run.add_argument("--min-episode-budget-seconds", type=float, default=45.0)
    run.add_argument("--smoke-frames", type=int, default=None)
    run.add_argument("--smoke-mixes", default="scripted")
    run.add_argument("--use-slot-locks", action="store_true", help="required")
    run.add_argument("--slot-pool", type=int, choices=sorted(dev_screen.SLOT_POOLS), default=2)
    run.add_argument("--thermal-guard", action="store_true", help="required")
    run.add_argument("--require-ac-power", action="store_true", help="always enforced")
    run.add_argument("--slot-lock-root", type=Path, default=dev_screen.DEFAULT_SLOT_LOCK_ROOT)
    run.add_argument("--slot-timeout-seconds", type=float, default=DEFAULT_SLOT_TIMEOUT_SECONDS)
    run.add_argument("--thermal-backoff-seconds", type=float, default=60.0)
    run.add_argument("--thermal-max-backoffs", type=int, default=5)
    run.add_argument("--barrier-timeout-seconds", type=float, default=BARRIER_TIMEOUT_SECONDS)
    merge = sub.add_parser("merge", help="combine the 3 shard dirs into summary.json")
    merge.add_argument("--shard-dirs", required=True, nargs="+", type=Path)
    merge.add_argument("--out", required=True, type=Path, help="new output dir (must not exist)")
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.thermal_backoff_seconds <= 0 or args.thermal_max_backoffs < 0:
            parser.error("need --thermal-backoff-seconds > 0 and --thermal-max-backoffs >= 0")
        if args.smoke_frames is not None:
            args.mixes = tuple(m for m in args.smoke_mixes.split(",") if m)
            episodes = len(args.mixes) * args.worlds_per_mix * len(PAIR_ARMS)
            if (
                not 0 < args.smoke_frames <= 500
                or not 0 < episodes <= 2
                or not set(args.mixes) <= set(dev_screen.MIXES)
            ):
                parser.error("smoke is limited to <= 500 frames and <= 2 episodes on known mixes")
        else:
            args.mixes = dev_screen.MIXES
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "merge":
        return merge_main(args)
    return run_main(args, argv)


def expected_shard_dirs(shard: int) -> List[Path]:
    """The only ``--out`` paths a real shard may use."""
    return [Path(root).resolve() / f"shard-{int(shard)}" for root in ALLOWED_RUN_ROOTS]


def _git(*args: str) -> Tuple[int, str]:
    import subprocess

    done = subprocess.run(
        ["git", "-C", str(REPO), *args], capture_output=True, text=True, check=False
    )
    return done.returncode, done.stdout


def package_sha256() -> Dict[str, str]:
    """sha256 of this package's files (recorded in every intent, checked at merge)."""
    return {name: dev_screen.sha256_file(HERE / name) for name in PACKAGE_FILES}


def tree_refusal() -> str | None:
    """A real run needs a fully clean tree (also no untracked files) and this package
    tracked and committed."""
    code, porcelain = _git("status", "--porcelain", "--untracked-files=all")
    if code != 0:
        return "refusing: git status failed"
    if porcelain.strip():
        return f"refusing a non-smoke screen with a dirty tree: {porcelain.splitlines()[:5]}"
    rel = HERE.relative_to(REPO)
    for name in PACKAGE_FILES:
        code, _ = _git("ls-files", "--error-unmatch", str(rel / name))
        if code != 0:
            return f"refusing: {rel / name} is not tracked"
    return None


def run_refusal(args: argparse.Namespace, now: datetime) -> str | None:
    """Why this ``run`` must not start (checked before any lock, output or episode)."""
    smoke = args.smoke_frames is not None
    if not args.use_slot_locks or args.slot_pool != SLOT_POOL or not args.thermal_guard:
        return "this screen requires --use-slot-locks --slot-pool 3 --thermal-guard"
    if smoke and (args.shards != 1 or args.shard != 0):
        return "a smoke runs as one shard: --shard 0 --shards 1"
    if not smoke and args.shards != SHARDS:
        return f"refusing --shards {args.shards}; the pre-registered run is {SHARDS} shards"
    if not 0 <= args.shard < args.shards:
        return f"--shard must be in 0..{args.shards - 1}"
    if not smoke and args.worlds_per_mix != WORLDS_PER_MIX:
        return (
            f"refusing a non-smoke run with {args.worlds_per_mix} worlds per mix; the "
            f"pre-registered size is {WORLDS_PER_MIX}"
        )
    if args.deadline_utc <= now:
        return "deadline already passed"
    if not smoke and (args.deadline_utc - now).total_seconds() > MAX_WALL_SECONDS:
        return f"--deadline-utc exceeds the {MAX_WALL_SECONDS} s wall-time cap"
    out = Path(args.out).resolve()
    if out.exists() or dev_screen.FORBIDDEN_OUTPUT_ROOT in out.parts:
        return f"--out {out} exists or is inside a forbidden root"
    if smoke and str(ARTIFACTS) in str(out):
        return "a smoke must write outside snake-dqn-artifacts"
    if not smoke:
        if (args.deadline_utc - now).total_seconds() < MIN_START_SECONDS:
            return (
                f"refusing: less than {MIN_START_SECONDS:.0f} s to the deadline (126 episodes x "
                "35 s x 1.3)"
            )
        if Path(args.slot_lock_root).resolve() != Path(SLOT_LOCK_ROOT).resolve():
            return f"refusing a non-default --slot-lock-root outside a smoke ({SLOT_LOCK_ROOT})"
        if out not in expected_shard_dirs(args.shard):
            return (
                f"refusing --out {out}: a real shard writes only to <root>/shard-{args.shard} "
                f"with <root> in {[str(r) for r in ALLOWED_RUN_ROOTS]}"
            )
        for key, value in FIXED_RUN_PARAMS.items():
            if getattr(args, key) != value:
                return f"refusing non-default --{key.replace('_', '-')} outside a smoke ({value})"
        problem = tree_refusal()
        if problem:
            return problem
        try:
            check_sweep_binding()
        except (SweepBindingError, OSError, ValueError) as exc:
            return f"sweep binding: {exc}"
    return None


def run_main(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Play one shard: refusals, preflight, AC check, one pool-3 slot, then the loop."""
    refusal = run_refusal(args, datetime.now(timezone.utc))
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    smoke = args.smoke_frames is not None
    domain = SMOKE_DOMAIN if smoke else DOMAIN
    try:
        observed = observed_seed_namespaces()
    except (OSError, ValueError) as exc:
        print(f"refusing: {exc}", file=sys.stderr)
        return 2
    spec = make_spec(domain, {**earlier_namespaces(domain), **observed})
    binding = None if smoke else check_sweep_binding()
    out = Path(args.out).resolve()

    from src.core.config_loader import load_and_initialize_config
    from src.scripts.tournament_eval import evaluation_profile_for_name

    dev_screen._configure_torch()
    if dev_screen.sha256_file(args.config) != dev_screen.CONFIG_SHA256:
        print("config bytes differ from the pinned deployment config", file=sys.stderr)
        return 2
    load_and_initialize_config(str(args.config))
    profile = evaluation_profile_for_name(dev_screen.PROFILE_NAME, dev_screen.HORIZON)
    if profile.digest != dev_screen.PROFILE_DIGEST:
        print("resolved profile differs from the strict pilot's profile", file=sys.stderr)
        return 2
    seeds = dev_screen.screen_seeds(args.worlds_per_mix, spec.domain, spec.namespace)
    disjoint = dev_screen.disjointness_report(
        seeds, args.pilot_output, screen_domain=spec.domain, extra_namespaces=spec.extra_namespaces
    )
    parity = dev_screen.roster_parity_report(args.pilot_output)
    failures = dev_screen.preflight_failures(disjoint, parity, smoke)
    if failures:
        print(json.dumps({"preflight_failures": failures}), file=sys.stderr)
        return 2
    if not dev_screen.on_ac_power():
        print("refusing to start on battery power", file=sys.stderr)
        return 2
    waited_from = time.monotonic()
    try:
        slots = dev_screen.acquire_cpu_slots(
            args.slot_lock_root, 1, args.slot_timeout_seconds, pool=SLOT_POOL
        )
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"CPU slot lock not acquired: {exc}", file=sys.stderr)
        return 2
    try:
        slot_wait = time.monotonic() - waited_from
        if not dev_screen.on_ac_power():
            print("refusing to start on battery power (after the slot wait)", file=sys.stderr)
            return 2
        if args.deadline_utc <= datetime.now(timezone.utc):
            print("deadline passed while waiting for a slot", file=sys.stderr)
            return 2
        held = Path(slots[0].name).name
        barrier = None
        if not smoke:
            identity = {
                "commit": dev_screen._git_state().get("commit"),
                "protocol_sha256": dev_screen.sha256_file(Path(spec.protocol)),
                "screen_sha256": package_sha256()["screen.py"],
                "shards": args.shards,
            }
            barrier = sweep.start_barrier(
                out.parent / BARRIER_DIRNAME,
                args.shard,
                identity,
                held,
                args.barrier_timeout_seconds,
            )
            if not barrier["passed"]:
                print(json.dumps({"start_barrier_refusal": barrier}), file=sys.stderr)
                return 2
        return _run_shard(
            args,
            argv,
            spec,
            out,
            seeds,
            disjoint,
            parity,
            profile,
            smoke,
            held,
            slot_wait_seconds=slot_wait,
            binding=binding,
            barrier=barrier,
        )
    finally:
        dev_screen.release_cpu_slots(slots)


def _run_shard(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    spec: dev_screen.ScreenSpec,
    out: Path,
    seeds: Sequence[int],
    disjoint: Mapping[str, Any],
    parity: Mapping[str, Any],
    profile: Any,
    smoke: bool,
    slot_held: str,
    *,
    slot_wait_seconds: float = 0.0,
    binding: Mapping[str, Any] | None = None,
    barrier: Mapping[str, Any] | None = None,
    runner: Callable[..., Dict[str, Any]] | None = None,
) -> int:
    """The shard body (``run_main`` holds the slot lock around it)."""
    from research.compute.thermal_guard import admit_next_episode

    out.mkdir(parents=True)
    records_dir = out / "records"
    records_dir.mkdir()
    snapshots = dev_screen.snapshot_checkpoints(args.checkpoint_dir, out)
    lookup = dev_screen.agent_lookup(snapshots)
    rows = [row for row in dev_screen._design_rows(seeds) if row["mix"] in args.mixes]
    plan = shard_plan(rows, seeds, args.mixes, args.shard, args.shards, smoke)
    guard = dev_screen.make_thermal_guard(dev_screen.on_ac_power)
    git = dev_screen._git_state()
    intent = {
        "schema_version": SCHEMA,
        "screen_id": SCREEN_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "protocol": str(Path(spec.protocol).resolve()),
        "protocol_sha256": dev_screen.sha256_file(Path(spec.protocol)),
        "decision_rule": DECISION_RULE,
        "hypothesis": spec.hypothesis,
        "decision_informs": spec.decision_informs,
        "owner": spec.owner,
        "sweep_binding": binding,
        "argv": list(sys.argv if argv is None else argv),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "git": git,
        "config": {"path": str(Path(args.config).resolve()), "sha256": dev_screen.CONFIG_SHA256},
        "profile": {
            "name": dev_screen.PROFILE_NAME,
            "digest": dev_screen.PROFILE_DIGEST,
            "horizon": dev_screen.HORIZON,
        },
        "smoke_frames": args.smoke_frames,
        "mixes": list(args.mixes),
        "hero": {"name": dev_screen.CHAMPION[0], "sha256": dev_screen.CHAMPION[1]},
        "checkpoint_pool": [{"name": name, "sha256": sha} for name, sha in dev_screen.POOL],
        "checkpoint_snapshots": snapshots,
        "arms": {
            arm: {
                "method": arm_method(arm),
                "lambda": A_LAMBDA if arm in V7_ARMS else B_LAMBDA,
                **({"repeats": CONTROLS[arm]} if arm in CONTROLS else {}),
            }
            for arm in INSTALLERS
        },
        "worlds": {"domain": spec.domain, "namespace": spec.namespace, "seeds": list(seeds)},
        "disjointness": disjoint,
        "roster_parity": parity,
        "shard": args.shard,
        "shards": args.shards,
        "plan": plan_keys(plan),
        "planned_episodes": len(plan),
        "compute_cap": {
            "deadline_utc": args.deadline_utc.isoformat(),
            "max_wall_seconds": MAX_WALL_SECONDS,
        },
        "slot_locks": {
            "root": str(args.slot_lock_root),
            "pool": list(dev_screen.SLOT_POOLS[SLOT_POOL]),
            "held": slot_held,
            "waited_seconds": round(slot_wait_seconds, 3),
            "held_from": "before --out existed until shard_summary.json was written",
        },
        "thermal_guard": {
            **guard.config(),
            "key": sweep.THERMAL_KEY,
            "backoff_seconds": args.thermal_backoff_seconds,
            "max_backoffs": args.thermal_max_backoffs,
        },
        "start_barrier": barrier,
        "package_sha256": package_sha256(),
        "run_params": {key: getattr(args, key) for key in FIXED_RUN_PARAMS},
        "ac_power_checked_at_start": True,
        "threads": {"torch_intraop": 2, "torch_interop": 1},
    }
    dev_screen.write_new_json(out / "intent.json", dev_screen.json_safe(intent))
    crashed = None
    snapshot_changed = False
    try:
        entries, stopped_reason, guard_events = sweep._play(
            args,
            spec,
            plan,
            seeds,
            lookup,
            profile,
            records_dir,
            guard,
            out,
            runner,
            admit_next_episode,
        )
    except (Exception, KeyboardInterrupt) as exc:  # recorded, never relabelled
        crashed = f"{type(exc).__name__}: {exc}"
        stopped_reason = f"error: {crashed}"
        entries, guard_events = _recover_partial(out)
    for sha, path in snapshots.items():
        if dev_screen.sha256_file(Path(path)) != sha:
            crashed = f"checkpoint snapshot {path} changed during the screen"
            stopped_reason = f"error: {crashed}"
            snapshot_changed = True
    check = self_check(entries, seeds, smoke)
    if snapshot_changed:  # an integrity failure, not just a stop
        check["failures"].append(f"shard error: {crashed}")
        check["passes"] = False
    shard_summary: Dict[str, Any] = {
        "schema_version": SHARD_SCHEMA,
        "screen_id": SCREEN_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "smoke": smoke,
        "decision": "SMOKE_NO_DECISION" if smoke else "see merge summary.json",
        "shard": args.shard,
        "shards": args.shards,
        "slot_held": slot_held,
        "slot_wait_seconds": round(slot_wait_seconds, 3),
        "started_utc": intent["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "episodes_run": len(entries),
        "planned_episodes": len(plan),
        "stopped_reason": stopped_reason,
        "snapshot_changed": snapshot_changed,
        "guard": sweep.guard_stats(guard_events),
        "self_check": check,
        "source": {"commit": git.get("commit"), "dirty_paths": git.get("dirty_paths")},
        "intent_sha256": dev_screen.sha256_file(out / "intent.json"),
    }
    if smoke:
        shard_summary["smoke_summary"] = summarize(
            entries,
            seeds,
            args.mixes,
            smoke=True,
            planned_episodes=len(plan),
            control_planned=0,
            source=git,
        )
    dev_screen.write_new_json(out / "shard_summary.json", dev_screen.json_safe(shard_summary))
    print(
        json.dumps({"shard": args.shard, "episodes_run": len(entries), "stopped": stopped_reason})
    )
    if crashed:
        return 4
    return 0 if check["passes"] else 3


def _recover_partial(out: Path) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """After a crash: the records already written and the guard events (for the summary)."""
    entries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((out / "records").glob("*.json"))
    ]
    events = []
    path = out / "events.jsonl"
    if path.is_file():
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            if "event" in event:
                events.append(event)
    return entries, events


# ---------------------------------------------------------------- merge


def merge_refusal(shards: Sequence[Mapping[str, Any]]) -> str | None:
    """Why these shard dirs cannot be merged into one screen (structure, not results)."""
    if len(shards) != SHARDS:
        return f"need exactly {SHARDS} shard dirs"
    intents = [s["intent"] for s in shards]
    if any(i.get("schema_version") != SCHEMA or i.get("screen_id") != SCREEN_ID for i in intents):
        return "a shard is not from this screen"
    if any(i.get("smoke_frames") is not None for i in intents):
        return "a smoke shard cannot be merged"
    if sorted(int(i.get("shard", -1)) for i in intents) != list(range(SHARDS)):
        return f"shard indices are not 0..{SHARDS - 1}"
    if any(int(i.get("shards", 0)) != SHARDS for i in intents):
        return f"a shard was not planned as one of {SHARDS}"
    for key in ("protocol_sha256", "worlds", "mixes"):
        if any(i.get(key) != intents[0].get(key) for i in intents):
            return f"shards differ in {key}"
    if intents[0].get("protocol_sha256") != dev_screen.sha256_file(HERE / "protocol.md"):
        return "the shards' protocol bytes differ from this protocol.md"
    if intents[0]["worlds"].get("domain") != DOMAIN:
        return "the shards did not play this screen's domain"
    if intents[0]["worlds"].get("seeds") != dev_screen.screen_seeds(
        WORLDS_PER_MIX, DOMAIN, NAMESPACE
    ):
        return f"the shards' seeds are not this screen's {WORLDS_PER_MIX} worlds per mix"
    if any(i.get("package_sha256") != package_sha256() for i in intents):
        return "a shard ran another screen.py/protocol.md than this package's"
    for intent in intents:
        if (
            Path(str((intent.get("slot_locks") or {}).get("root"))).resolve()
            != Path(SLOT_LOCK_ROOT).resolve()
        ):
            return f"shard {intent.get('shard')} did not use the default slot lock root"
        if intent.get("run_params") != FIXED_RUN_PARAMS:
            return f"shard {intent.get('shard')} ran with non-default operating parameters"
        if not (intent.get("start_barrier") or {}).get("passed"):
            return f"shard {intent.get('shard')} has no passed start barrier"
    for shard in shards:
        k = int(shard["intent"]["shard"])
        if Path(shard.get("dir", "")).resolve() not in expected_shard_dirs(k):
            return f"shard {k} directory is not <root>/shard-{k} under an allowed run root"
    if len({Path(s.get("dir", "")).resolve().parent for s in shards}) != 1:
        return "the shard directories are not under one run root"
    if any((i.get("git") or {}).get("commit") != intents[0]["git"].get("commit") for i in intents):
        return "shards ran from different commits"
    seeds = intents[0]["worlds"]["seeds"]
    rows = dev_screen._design_rows(seeds)
    for intent in intents:
        expected = plan_keys(
            shard_plan(rows, seeds, intent["mixes"], int(intent["shard"]), SHARDS, False)
        )
        if intent.get("plan") != expected:
            return f"shard {intent['shard']} plan differs from the pre-registered shard plan"
    for shard in shards:
        plan = [tuple(key) for key in shard["intent"]["plan"]]
        got = sorted(
            (e.get("arm"), e.get("mix"), int(e.get("world_seed")))
            for e in shard.get("entries") or []
        )
        if got != sorted(plan[: len(got)]):
            return f"shard {shard['intent']['shard']} records are not a prefix of its plan"
    return None


def _merge_check(entries, seeds, smoke, *, shards) -> Dict[str, Any]:
    """The record self-check plus shard-level integrity failures (changed checkpoints)."""
    check = self_check(entries, seeds, smoke)
    for shard in shards:
        if (shard.get("summary") or {}).get("snapshot_changed"):
            check["failures"].append(f"shard {shard['intent']['shard']}: checkpoint changed")
            check["passes"] = False
    return check


def merge_main(args: argparse.Namespace) -> int:
    """Combine the three shard dirs into ``summary.json`` (the pre-registered decision)."""
    out = Path(args.out).resolve()
    if out.exists() or dev_screen.FORBIDDEN_OUTPUT_ROOT in out.parts:
        print(f"--out {out} exists or is inside a forbidden root", file=sys.stderr)
        return 2
    try:
        shards = [sweep._load_shard(directory) for directory in args.shard_dirs]
    except (OSError, ValueError) as exc:
        print(f"cannot read the shard dirs: {exc}", file=sys.stderr)
        return 2
    refusal = merge_refusal(shards)
    if refusal is None and out != Path(shards[0]["dir"]).resolve().parent / "merged":
        refusal = "merge --out must be <run root>/merged"
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    shards.sort(key=lambda s: int(s["intent"]["shard"]))
    intents = [s["intent"] for s in shards]
    seeds, mixes = intents[0]["worlds"]["seeds"], intents[0]["mixes"]
    entries = [entry for s in shards for entry in s["entries"]]
    planned = sum(len(i["plan"]) for i in intents)
    control_planned = sum(1 for i in intents for key in i["plan"] if key[0] in CONTROLS)
    summary = summarize(
        entries,
        seeds,
        mixes,
        smoke=False,
        planned_episodes=planned,
        control_planned=control_planned,
        source=intents[0]["git"],
        checker=partial(_merge_check, shards=shards),
    )
    summary["merge_source"] = dev_screen._git_state()
    shard_rows = [
        {
            "shard": s["intent"]["shard"],
            "dir": s["dir"],
            "slot_held": s["summary"].get("slot_held"),
            "slot_wait_seconds": s["summary"].get("slot_wait_seconds"),
            "started_utc": s["summary"].get("started_utc"),
            "finished_utc": s["summary"].get("finished_utc"),
            "episodes_run": s["summary"].get("episodes_run"),
            "planned_episodes": len(s["intent"]["plan"]),
            "stopped_reason": s["summary"].get("stopped_reason"),
            "guard": sweep.guard_stats(s["events"]),
            "intent_sha256": dev_screen.sha256_file(Path(s["dir"]) / "intent.json"),
        }
        for s in shards
    ]
    starts = [dev_screen._utc(str(r["started_utc"])) for r in shard_rows if r["started_utc"]]
    ends = [dev_screen._utc(str(r["finished_utc"])) for r in shard_rows if r["finished_utc"]]
    guard_totals = {
        key: sum(int(r["guard"].get(key) or 0) for r in shard_rows)
        for key in (
            "checks",
            "not_ok_checks",
            "thermal_not_ok_checks",
            "slowdown_pauses",
            "thermal_pauses",
            "stops",
        )
    }
    summary.update(
        {
            "shards": shard_rows,
            "episodes_run": len(entries),
            "planned_episodes": planned,
            "stopped_reasons": [row["stopped_reason"] for row in shard_rows],
            "thermal_guard_totals": guard_totals,
            "wall_clock_seconds": (
                (max(ends) - min(starts)).total_seconds() if starts and ends else None
            ),
            "merged_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    out.mkdir(parents=True)
    dev_screen.write_new_json(out / "summary.json", dev_screen.json_safe(summary))
    print(json.dumps({"decision": summary["decision"], "summary": str(out / "summary.json")}))
    return 0 if summary["self_check"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
