#!/usr/bin/env python3
"""Tier-1 DEV lambda sweep (non-authoritative): Apex + v8 (space + head) vs Apex + v7(lambda=4).

Pre-declaration: ``protocol.md`` beside this file. Arms (paired on every world):

  A      champion + ``free-space-veto/v7-space-preference(lambda=4.0)`` (the v7 candidate
         under the strict gate).
  H400   champion + ``free-space-veto/v8-space-and-head(lambda=4.0)`` (= A + head layer).
  H800   champion + v8 at ``lambda = 8.0``.
  H1600  champion + v8 at ``lambda = 16.0``.
  R      H800 repeated on the first world of each mix, in ANOTHER shard (replay control).

Built on ``research/apex_safety_20260926/dev_screen.py`` (seed recipe, disjointness and
roster-parity preflight, strict balanced rosters, checkpoint snapshots, ``run_episode``,
slot locks with ``--slot-pool 3``, the ``--thermal-guard`` admission gate, AC check) and on
the v7 sweep's statistics helpers, unchanged. The sweep runs as THREE concurrent shard
processes (one CPU slot each, slot 3 then 1 then 2), so it doubles as the 3-slot thermal
calibration of ``docs/research/compute_policy_2026-10-02.md``; ``merge`` combines the three
shard directories into one ``summary.json`` with the pre-declared selection and the
pre-declared calibration verdict.

Usage (each shard about 20-30 min; hard cap 3 h; see protocol.md for the full commands):
  ... sweep.py slots-free      # pre-launch: exit 0 only if all 3 pool-3 slots are free now
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v8_lambda_sweep_20261002/sweep.py \\
    run --shard K --out <dir>/shard-K --deadline-utc <now + <= 3 h> \\
    --use-slot-locks --slot-pool 3 --thermal-guard --require-ac-power      # K = 0, 1, 2
  Each non-smoke shard waits (start barrier, <= 120 s) after taking its slot until all 3
  shards hold the 3 pool-3 slots; otherwise it exits 2 before any episode.
  ... sweep.py merge --shard-dirs <dir>/shard-0 <dir>/shard-1 <dir>/shard-2 --out <dir>/merged
Smoke (plumbing only, smoke namespace, arms A and H400 on 1 world, <= 500 frames):
  ... sweep.py run --shard 0 --shards 1 --out /tmp/x --deadline-utc ... --smoke-frames 500 \\
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
from research.apex_veto_v7_screen_20261002 import screen as v7screen  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_ID = "apex-veto-v8-dev-v1"
SCHEMA = "apex-veto-v8-lambda-sweep/v1"
SHARD_SCHEMA = "apex-veto-v8-lambda-sweep-shard/v1"
AUTHORITY = "development sweep (Tier-1 dev, non-authoritative)"
DOMAIN = SWEEP_ID
SMOKE_DOMAIN = "apex-veto-v8-dev-smoke-v1"
NAMESPACE = "worlds"
A_LAMBDA = 4.0
ARM_LAMBDAS: Dict[str, float] = {"A": A_LAMBDA, "H400": 4.0, "H800": 8.0, "H1600": 16.0}
SWEEP_ARMS = tuple(ARM_LAMBDAS)
V8_ARMS = ("H400", "H800", "H1600")
REPLAY_ARM, REPLAY_OF, REPLAY_WORLDS = "R", "H800", 1
SMOKE_ARMS = ("A", "H400")
WORLDS_PER_MIX = 8
SHARDS = 3
# Activity gate (v7's three conditions; per-mix episode minimum 1, see protocol.md: the
# head layer alone vetoes about once per episode, in about 29% of v6-screen episodes).
ACTIVE_MIN_EPISODES_PER_MIX = 1  # episodes with action_differs_from_reference > 0
ACTIVE_MIN_NONZERO_DELTAS_PER_MIX = 1
ACTIVE_MIN_CHANGE_RATE = 1e-4  # pooled action_differs_from_reference / decisions
MAX_WALL_SECONDS = 3 * 3600
EXCLUSION_PREFIX = 1000
SLOT_POOL = 3
THERMAL_KEY = "shard"  # one slowdown key per shard process (protocol.md, calibration)
# The future v8 screen's domains are excluded now so the sweep can never consume them.
SCREEN_DOMAINS = ("apex-veto-v8-screen-v1", "apex-veto-v8-screen-smoke-v1")
V7_STRICT_DOMAINS = (
    "apex-veto-v7-strict-dev-v1",
    "apex-veto-v7-strict-final-v1",
    "apex-veto-v7-strict-serving-v1",
    "apex-veto-v7-strict-smoke-v1",
)
WEB_DOMAINS = (
    "apex-veto-web-serving-v1",
    "apex-veto-web-serving-smoke-v1",
    "apex-veto-v5-web-serving-v1",
    "apex-veto-v5-web-serving-smoke-v1",
)
WEB_PURPOSES = ("watch", "play", "parity", "worlds")


def _merge_domains(*tables: Mapping[str, Sequence[str]]) -> Dict[str, Tuple[str, ...]]:
    out: Dict[str, List[str]] = {}
    for table in tables:
        for name, purposes in table.items():
            row = out.setdefault(name, [])
            row.extend(p for p in purposes if p not in row)
    return {name: tuple(purposes) for name, purposes in out.items()}


EARLIER_DOMAINS: Dict[str, Tuple[str, ...]] = _merge_domains(
    v7screen.EARLIER_DOMAINS,
    {name: WEB_PURPOSES for name in WEB_DOMAINS},
    {name: (NAMESPACE,) for name in (*V7_STRICT_DOMAINS, *SCREEN_DOMAINS)},
    {DOMAIN: (NAMESPACE,), SMOKE_DOMAIN: (NAMESPACE,)},
)
# Calibration baseline (2-slot era, one process): the v7 sweep's arm L400 = v7(lambda=4),
# the same veto as this sweep's arm A. Pinned by sha256 (protocol.md).
BASELINE_EVENTS = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002"
    "/run-v1/events.jsonl"
)
BASELINE_EVENTS_SHA256 = "0ea51914d8c0a8384c04bcfc662fe1a258a73990caed60211fca2ddb333c38ae"
BASELINE_ARM = "L400"
# Arm A's wall ratio vs the baseline must be strictly below this, pooled AND in every mix.
CALIBRATION_MAX_WALL_RATIO = 1.30
# The 3 shards must overlap for at least this fraction of the shortest shard's duration.
CALIBRATION_MIN_OVERLAP_FRACTION = 0.9
# Start barrier (non-smoke runs): after taking its slot, a shard plays nothing until all 3
# shards hold 3 distinct pool-3 slots from the same commit and protocol bytes.
BARRIER_DIRNAME = ".barrier"
BARRIER_TIMEOUT_SECONDS = 120.0
STATUS_SELECTED = "SELECTED"
STATUS_NONE_ACTIVE = "NONE_ACTIVE"
STATUS_NONE_QUALIFIES = "NONE_QUALIFIES"
SELECTION_RULE = (
    "per v8 lambda: paired mass_integral delta (v8_lambda - A) per world and mix, A = v7 "
    "lambda 4. ACTIVE iff in EACH mix >= "
    f"{ACTIVE_MIN_EPISODES_PER_MIX} episode has action_differs_from_reference > 0 and >= "
    f"{ACTIVE_MIN_NONZERO_DELTAS_PER_MIX} world has a nonzero delta, and the pooled "
    f"action_differs_from_reference/decisions >= {ACTIVE_MIN_CHANGE_RATE:g}; a CLEAR LOSER in "
    "a mix iff mean + t(0.90, n-1) * sd / sqrt(n) < 0 there; QUALIFIES iff active and a clear "
    "loser in no mix. SELECTED = the qualifying lambda with the highest pooled mean delta "
    "(ties: smaller lambda); NONE_ACTIVE if no lambda is active, else NONE_QUALIFIES (stop). "
    "Before that: SMOKE_NO_SELECTION, INVALID_SELF_CHECK_FAILED, INCOMPLETE, "
    "INVALID_NONDETERMINISTIC (R vs H800), in order."
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


def install_a(hero: Any) -> Any:
    """Arm A: v7 at lambda 4 (the strict-gated candidate)."""
    from src.evaluation.safety_veto_v7 import install_space_preference_veto

    return install_space_preference_veto(hero, A_LAMBDA)


def install_v8(hero: Any, lam: float) -> Any:
    """v8 arms: space + head at ``lam``, with arm A's lambda as the reference diagnostic."""
    from src.evaluation.safety_veto_v8 import install_space_and_head_veto

    return install_space_and_head_veto(hero, lam, reference_lambda=A_LAMBDA)


INSTALLERS: Dict[str, Any] = {"A": install_a}
INSTALLERS.update({arm: partial(install_v8, lam=ARM_LAMBDAS[arm]) for arm in V8_ARMS})
INSTALLERS[REPLAY_ARM] = INSTALLERS[REPLAY_OF]  # the same object: R repeats H800


def arm_lambda(arm: str) -> float:
    """The arm's lambda (R is H800's)."""
    return ARM_LAMBDAS[REPLAY_OF if arm == REPLAY_ARM else arm]


def arm_method(arm: str) -> str:
    from src.evaluation import safety_veto_v7, safety_veto_v8

    module = safety_veto_v7 if arm == "A" else safety_veto_v8
    return module.method_for(arm_lambda(arm))


def expected_descriptor(arm: str) -> Dict[str, Any]:
    """The probe identity each arm's records must carry (writer: ``veto.record()``)."""
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto
    from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto

    if arm == "A":
        return SpacePreferenceVeto(A_LAMBDA).descriptor()
    return SpaceAndHeadVeto(arm_lambda(arm)).descriptor()


def _spec(domain: str) -> dev_screen.ScreenSpec:
    return dev_screen.ScreenSpec(
        name=SWEEP_ID,
        schema=SCHEMA,
        authority=AUTHORITY,
        domain=domain,
        namespace=NAMESPACE,
        protocol=HERE / "protocol.md",
        arm_vetoes=dict(INSTALLERS),
        extra_namespaces=earlier_namespaces(domain),
        require_slot_locks=True,
        require_ac_power=True,
        owner="Apex safety lane",
        max_wall_seconds=MAX_WALL_SECONDS,
    )


SPEC = _spec(DOMAIN)
SMOKE_SPEC = _spec(SMOKE_DOMAIN)


def v8_identities_hold(diagnostics: Mapping[str, Any], counters: Mapping[str, Any]) -> bool:
    """``SpaceAndHeadCounters`` identities against the probe's v2 counters (reported)."""

    def get(source: Any, key: str) -> int:
        return int(source.get(key) or 0) if isinstance(source, Mapping) else 0

    v7 = diagnostics.get("v7") if isinstance(diagnostics.get("v7"), Mapping) else {}
    over_kept = get(diagnostics, "head_vetoes_of_v7_kept")
    return (
        counters.get("decisions") == get(diagnostics, "decisions") == get(v7, "decisions")
        and counters.get("kept_base") == get(diagnostics, "kept") == get(v7, "kept") - over_kept
        and counters.get("vetoes_applied")
        == get(diagnostics, "vetoes_applied")
        == get(v7, "vetoes_applied") + over_kept
        and counters.get("fallback_no_spacious")
        == get(diagnostics, "no_spacious")
        == get(v7, "no_spacious")
        and get(diagnostics, "head_risky_decisions")
        == get(diagnostics, "head_risky_vetoes")
        + get(diagnostics, "head_risky_kept_no_alternative")
        and get(diagnostics, "action_differs_from_v7") == get(diagnostics, "head_risky_vetoes")
    )


# ---------------------------------------------------------------- self-check (gating)


def check_entry(entry: Mapping[str, Any], smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one sweep entry (dev_screen writer).

    v7's sweep check (identity, probe, seed, finite mass, profile, world identity, H5000
    denominators) with this sweep's arms; on every v8 arm the reference diagnostic the
    activity gate reads must be present, at arm A's lambda, on every decision.
    """
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
        failures.append("entry schema/authority is not this sweep's")
    if entry.get("screen") != SWEEP_ID or entry.get("hero_sha256") != dev_screen.CHAMPION[1]:
        failures.append("entry sweep id or hero sha256 differs")
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
    if arm == "A":
        if not isinstance(diagnostics, Mapping) or "rerank_changes" not in diagnostics:
            warnings.append("arm A v7 counters missing")
    elif (
        not isinstance(diagnostics, Mapping)
        or "action_differs_from_reference" not in diagnostics
        or diagnostics.get("reference_lambda") != A_LAMBDA
        or diagnostics.get("reference_decisions") != counters.get("decisions")
    ):
        failures.append(f"arm {arm} v8 reference diagnostic missing or not at lambda {A_LAMBDA}")
    elif not v8_identities_hold(diagnostics, counters):
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


# ---------------------------------------------------------------- analysis


def _diagnostics(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = entry.get("veto_diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def lambda_table(
    entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], mixes: Sequence[str]
) -> Dict[str, Any]:
    """Per v8 arm: per-mix and pooled paired mass deltas against arm A, plus activity."""
    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e["record"] for e in entries}
    diag = {(e["arm"], e["mix"], int(e["world_seed"])): _diagnostics(e) for e in entries}
    table: Dict[str, Any] = {}
    for arm in V8_ARMS:
        per_mix: Dict[str, Any] = {}
        pooled: List[float] = []
        changes = decisions = 0
        for mix in mixes:
            paired = [s for s in seeds if ("A", mix, s) in by_key and (arm, mix, s) in by_key]
            deltas = [
                by_key[(arm, mix, s)]["mass_integral"] - by_key[("A", mix, s)]["mass_integral"]
                for s in paired
            ]
            pooled.extend(deltas)
            arm_diags = [diag[(arm, mix, s)] for s in seeds if (arm, mix, s) in diag]
            episode_changes = [int(d.get("action_differs_from_reference") or 0) for d in arm_diags]
            changes += sum(episode_changes)
            decisions += sum(int(d.get("decisions") or 0) for d in arm_diags)
            upper = v7sweep._upper_bound_90(deltas)
            per_mix[mix] = {
                "paired_worlds": len(paired),
                "paired_seeds": paired,
                "deltas": deltas,
                "mean_delta": v7sweep._mean(deltas),
                "sd_delta": v7sweep._sd(deltas),
                "mean_A": v7sweep._mean([by_key[("A", mix, s)]["mass_integral"] for s in paired]),
                "mean_arm": v7sweep._mean([by_key[(arm, mix, s)]["mass_integral"] for s in paired]),
                "wins": sum(1 for d in deltas if d > 0),
                "losses": sum(1 for d in deltas if d < 0),
                "ties": sum(1 for d in deltas if d == 0),
                "episodes_with_changes_vs_A": sum(1 for c in episode_changes if c > 0),
                "changes_vs_A": sum(episode_changes),
                "head_risky_vetoes": sum(int(d.get("head_risky_vetoes") or 0) for d in arm_diags),
                "nonzero_deltas": sum(1 for d in deltas if d != 0),
                "upper_bound_90_one_sided": upper,
                "clear_loser": upper is not None and upper < 0,
            }
        rate = changes / decisions if decisions else 0.0
        active = rate >= ACTIVE_MIN_CHANGE_RATE and all(
            row["episodes_with_changes_vs_A"] >= ACTIVE_MIN_EPISODES_PER_MIX
            and row["nonzero_deltas"] >= ACTIVE_MIN_NONZERO_DELTAS_PER_MIX
            for row in per_mix.values()
        )
        clear_loser = any(row["clear_loser"] for row in per_mix.values())
        table[arm] = {
            "lambda": ARM_LAMBDAS[arm],
            "per_mix": per_mix,
            "pooled_pairs": len(pooled),
            "pooled_mean_delta": v7sweep._mean(pooled),
            "pooled_sd_delta": v7sweep._sd(pooled),
            "changes_vs_A": changes,
            "decisions": decisions,
            "change_rate_vs_A_per_decision": rate,
            "active": active,
            "clear_loser_in_some_mix": clear_loser,
            "qualifies": active and not clear_loser and bool(per_mix),
        }
    return table


def select_lambda(table: Mapping[str, Any]) -> Tuple[str, float | None, List[float]]:
    """``(status, lambda, qualifying)`` by the pre-declared rule (ties: smaller lambda)."""
    if not any(table[arm]["active"] for arm in V8_ARMS):
        return STATUS_NONE_ACTIVE, None, []
    qualifying = [arm for arm in V8_ARMS if table[arm]["qualifies"]]
    if not qualifying:
        return STATUS_NONE_QUALIFIES, None, []
    best = max(qualifying, key=lambda arm: (table[arm]["pooled_mean_delta"], -ARM_LAMBDAS[arm]))
    return STATUS_SELECTED, ARM_LAMBDAS[best], [ARM_LAMBDAS[arm] for arm in qualifying]


V8_SUMMED = (
    "decisions",
    "head_checks",
    "head_risky_decisions",
    "head_risky_vetoes",
    "head_risky_kept_no_alternative",
    "head_risk_waived_hero_wins",
    "head_vetoes_speed_switched",
    "head_vetoes_of_v7_kept",
    "head_vetoes_of_v7_rerank",
    "head_vetoes_space_differs_from_highest_q",
    "head_area_evaluations",
    "action_differs_from_reference",
    "apply_seconds_total",
    "head_seconds_total",
    "reference_seconds_total",
)


def counter_report(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reported only: summed v8 (and nested v7) counters and wall time per arm."""
    out: Dict[str, Any] = {}
    for arm in (*SWEEP_ARMS, REPLAY_ARM):
        rows = [e for e in entries if e.get("arm") == arm]
        if not rows:
            continue
        wall = [float(e.get("wall_seconds", 0.0)) for e in rows]
        row: Dict[str, Any] = {
            "episodes": len(rows),
            "mean_episode_wall_seconds": v7sweep._mean(wall),
        }
        v7_rows = [_diagnostics(e) if arm == "A" else _diagnostics(e).get("v7") or {} for e in rows]
        row["v7_rerank_changes"] = sum(int(d.get("rerank_changes") or 0) for d in v7_rows)
        row["v7_decisions"] = sum(int(d.get("decisions") or 0) for d in v7_rows)
        if arm != "A":
            totals = {key: 0.0 for key in V8_SUMMED}
            for entry in rows:
                for key in V8_SUMMED:
                    totals[key] += _diagnostics(entry).get(key) or 0
            decisions = totals["decisions"]
            row.update(totals)
            row["mean_apply_seconds_per_decision"] = (
                totals["apply_seconds_total"] / decisions if decisions else None
            )
            row["head_veto_rate_per_decision"] = (
                totals["head_risky_vetoes"] / decisions if decisions else None
            )
        out[arm] = row
    return {"by_arm": out, "gating": False}


def replay_report(entries: Sequence[Mapping[str, Any]], planned: int) -> Dict[str, Any]:
    """R vs H800: every R record must equal its H800 record (canonical JSON)."""
    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e["record"] for e in entries}
    replays = [e for e in entries if e["arm"] == REPLAY_ARM]
    mismatches = [
        {"mix": e["mix"], "world_seed": e["world_seed"]}
        for e in replays
        if (REPLAY_OF, e["mix"], int(e["world_seed"])) not in by_key
        or dev_screen.canonical_json(by_key[(REPLAY_OF, e["mix"], int(e["world_seed"]))])
        != dev_screen.canonical_json(e["record"])
    ]
    return {
        "arm": f"{REPLAY_ARM} repeats {REPLAY_OF} (in another shard process)",
        "planned": planned,
        "compared": len(replays),
        "mismatches": mismatches,
        "passes": len(replays) == planned and not mismatches,
    }


def summarize(
    entries: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    mixes: Sequence[str],
    *,
    smoke: bool,
    planned_episodes: int,
    replay_planned: int,
    source: Mapping[str, Any],
) -> Dict[str, Any]:
    """The pre-declared selection (protocol.md) plus reported tables. JSON-safe."""
    check = self_check(entries, seeds, smoke)
    table = lambda_table(entries, seeds, mixes)
    replay = replay_report(entries, replay_planned)
    complete = len(entries) == planned_episodes and all(
        table[arm]["per_mix"][mix]["paired_worlds"] == len(seeds)
        for arm in V8_ARMS
        for mix in mixes
    )
    selected, qualifying = None, []
    if smoke:
        status = "SMOKE_NO_SELECTION"
    elif not check["passes"]:
        status = "INVALID_SELF_CHECK_FAILED"
    elif not complete:
        status = "INCOMPLETE"
    elif not replay["passes"]:
        status = "INVALID_NONDETERMINISTIC"
    else:
        status, selected, qualifying = select_lambda(table)
    summary = {
        "schema_version": SCHEMA,
        "sweep_id": SWEEP_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "smoke": smoke,
        "complete": complete,
        "selection": {
            "status": status,
            "passes": status == STATUS_SELECTED,
            "selected_lambda": selected,
            "qualifying_lambdas": qualifying,
            "arm_a": f"v7 lambda {A_LAMBDA}",
            "activity_gate": {
                "min_episodes_with_changes_per_mix": ACTIVE_MIN_EPISODES_PER_MIX,
                "min_nonzero_deltas_per_mix": ACTIVE_MIN_NONZERO_DELTAS_PER_MIX,
                "min_pooled_change_rate": ACTIVE_MIN_CHANGE_RATE,
            },
            "rule": SELECTION_RULE,
        },
        "source": {"commit": source.get("commit"), "dirty_paths": source.get("dirty_paths")},
        "lambdas": table,
        "replay_control": replay,
        "self_check": check,
        "reported": {
            "counters": counter_report(entries),
            "death_causes": v7sweep.death_causes(entries),
        },
    }
    return dev_screen.json_safe(summary)


# ---------------------------------------------------------------- 3-slot calibration


def guard_stats(events: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Thermal-guard tallies from one shard's ``events.jsonl`` lines (pre-declared)."""
    checks = not_ok = thermal_not_ok = slowdown_pauses = thermal_pauses = stops = 0
    max_ratio = None
    reasons: Dict[str, int] = {}
    for event in events:
        kind = event.get("event")
        found = [str(r) for r in event.get("reasons") or []]
        thermal = [r for r in found if not r.startswith("slowdown")]
        if kind == "thermal_guard_check":
            checks += 1
            not_ok += int(not event.get("ok"))
            thermal_not_ok += int(bool(thermal))
            for reason in found:
                key = reason.split(":")[0].split("=")[0]
                reasons[key] = reasons.get(key, 0) + 1
            keys = (((event.get("readings") or {}).get("slowdown") or {}).get("keys")) or {}
            for row in keys.values():
                if isinstance(row, Mapping) and row.get("ratio") is not None:
                    ratio = float(row["ratio"])
                    max_ratio = ratio if max_ratio is None else max(max_ratio, ratio)
        elif kind == "thermal_guard_backoff":
            slowdown_pauses += int(len(thermal) < len(found))
            thermal_pauses += int(bool(thermal))
        elif kind == "thermal_guard_stop":
            stops += 1
    return {
        "checks": checks,
        "not_ok_checks": not_ok,
        "thermal_not_ok_checks": thermal_not_ok,
        "slowdown_pauses": slowdown_pauses,
        "thermal_pauses": thermal_pauses,
        "stops": stops,
        "reason_counts": dict(sorted(reasons.items())),
        "max_slowdown_ratio": max_ratio,
    }


def load_baseline(path: Path) -> Dict[str, Any]:
    """The pinned 2-slot baseline: arm L400 episode wall times per mix from the v7 sweep."""
    path = Path(path)
    if not path.is_file():
        return {"ok": False, "reason": f"missing {path}", "times_by_mix": {}}
    sha = dev_screen.sha256_file(path)
    times: Dict[str, List[float]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        event = json.loads(line) if line.strip() else {}
        if event.get("arm") == BASELINE_ARM and "wall_seconds" in event:
            times.setdefault(str(event["mix"]), []).append(float(event["wall_seconds"]))
    ok = sha == BASELINE_EVENTS_SHA256
    return {
        "ok": ok,
        "reason": None if ok else "baseline events sha256 differs from the pinned one",
        "path": str(path),
        "sha256": sha,
        "times_by_mix": times,
    }


CALIBRATION_RULE = (
    "valid only with 3 shard processes that each passed the start barrier, held the 3 "
    "distinct pool-3 slot files and ran at the same time for at least "
    f"{CALIBRATION_MIN_OVERLAP_FRACTION:.0%} of the shortest shard's duration (starts within "
    "the rest of it), and the pinned baseline; PASS iff zero thermal-guard stops, zero "
    "not-ok guard checks for a non-slowdown reason (warning level, CPU speed or scheduler "
    "limit < 100, pmset unknown, battery), arm "
    f"A's mean episode wall time strictly below {CALIBRATION_MAX_WALL_RATIO:.2f} x the "
    f"baseline's ({BASELINE_ARM} of the v7 sweep run-v1, the same veto, 2-slot era) pooled "
    "AND in each mix, and the sweep complete; slowdown pauses are reported only; FAIL if any "
    "criterion fails (a thermal stop is FAIL even though the sweep is then incomplete); "
    "INVALID otherwise. PASS -> 3 slots may become the Tier-1/dev default (a separate, "
    "explicit edit with its own note that cites the disclosed deviations from the policy: "
    "arm A only, different worlds); FAIL -> keep 2 slots."
)


def _a_times_by_mix(entries: Sequence[Mapping[str, Any]]) -> Dict[str, List[float]]:
    times: Dict[str, List[float]] = {}
    for entry in entries:
        if entry.get("arm") == "A":
            times.setdefault(str(entry["mix"]), []).append(float(entry["wall_seconds"]))
    return times


def _wall_ratios(
    a_times: Mapping[str, List[float]], base_times: Mapping[str, List[float]]
) -> Tuple[float | None, Dict[str, Dict[str, Any]]]:
    """Arm A vs baseline mean wall-time ratio, pooled and per mix."""
    a_all = [t for times in a_times.values() for t in times]
    base_all = [t for times in base_times.values() for t in times]
    pooled = v7sweep._mean(a_all) / v7sweep._mean(base_all) if a_all and base_all else None
    per_mix = {
        mix: {
            "mean_A": v7sweep._mean(a_times.get(mix, [])),
            "mean_baseline": v7sweep._mean(base_times.get(mix, [])),
            "ratio": (
                v7sweep._mean(a_times[mix]) / v7sweep._mean(base_times[mix])
                if a_times.get(mix) and base_times.get(mix)
                else None
            ),
        }
        for mix in sorted(set(a_times) | set(base_times))
    }
    return pooled, per_mix


def _concurrency(shards: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Whether the shards ran together, and arm A episodes finished outside that window.

    ``concurrent`` iff the overlap (latest start to earliest finish) is at least
    :data:`CALIBRATION_MIN_OVERLAP_FRACTION` of the shortest shard's duration AND the starts
    are spread over at most the rest of it (so a short, late shard cannot pass).
    """
    spans = [
        (dev_screen._utc(str(s["started_utc"])), dev_screen._utc(str(s["finished_utc"])))
        for s in shards
        if s.get("started_utc") and s.get("finished_utc")
    ]
    if len(spans) != len(shards) or len(shards) != SHARDS:
        return {
            "concurrent": False,
            "overlap_seconds": None,
            "shortest_seconds": None,
            "start_spread_seconds": None,
            "a_outside": None,
        }
    start, end = max(a for a, _ in spans), min(b for _, b in spans)
    overlap = (end - start).total_seconds()
    shortest = min((b - a).total_seconds() for a, b in spans)
    spread = (start - min(a for a, _ in spans)).total_seconds()
    outside = [
        utc
        for s in shards
        for utc in s.get("arm_a_finished_utc") or []
        if not start <= dev_screen._utc(str(utc)) <= end
    ]
    return {
        "concurrent": 0 < CALIBRATION_MIN_OVERLAP_FRACTION * shortest <= overlap
        and spread <= (1 - CALIBRATION_MIN_OVERLAP_FRACTION) * shortest,
        "overlap_seconds": overlap,
        "shortest_seconds": shortest,
        "start_spread_seconds": spread,
        "a_outside": len(outside),
    }


def calibration_report(
    shards: Sequence[Mapping[str, Any]],
    entries: Sequence[Mapping[str, Any]],
    baseline: Mapping[str, Any],
    complete: bool,
) -> Dict[str, Any]:
    """The pre-declared 3-slot calibration verdict (protocol.md). JSON-safe.

    ``shards`` are the shard rows built by ``merge``: ``guard`` = :func:`guard_stats` of the
    shard's events, ``start_barrier_passed`` from its intent, ``arm_a_finished_utc`` from its
    episode events; ``entries`` are all episode entries.
    """
    problems: List[str] = []
    indices = sorted(int(s.get("shard", -1)) for s in shards)
    if indices != list(range(SHARDS)):
        problems.append(f"shards {indices} are not 0..{SHARDS - 1}")
    held = sorted(str(s.get("slot_held")) for s in shards)
    if held != sorted(dev_screen.SLOT_POOLS[SLOT_POOL]):
        problems.append(f"slot files held {held} are not the 3 distinct pool-3 slots")
    if not all(s.get("start_barrier_passed") is True for s in shards):
        problems.append("a shard did not pass the start barrier")
    overlap = _concurrency(shards)
    if not overlap["concurrent"]:
        problems.append(
            "the shard processes did not run at the same time for at least "
            f"{CALIBRATION_MIN_OVERLAP_FRACTION:.0%} of the shortest shard's duration, with "
            "starts within the rest of it"
        )
    if not baseline.get("ok"):
        problems.append(f"baseline not usable: {baseline.get('reason')}")
    totals = {
        key: sum(int((s.get("guard") or {}).get(key) or 0) for s in shards)
        for key in ("checks", "not_ok_checks", "thermal_not_ok_checks", "slowdown_pauses", "stops")
    }
    ratios = [
        (s.get("guard") or {}).get("max_slowdown_ratio")
        for s in shards
        if (s.get("guard") or {}).get("max_slowdown_ratio") is not None
    ]
    a_times = _a_times_by_mix(entries)
    wall_ratio, per_mix = _wall_ratios(a_times, baseline.get("times_by_mix") or {})
    mix_ratios = [row["ratio"] for row in per_mix.values()]
    criteria = {
        "zero_guard_stops": totals["stops"] == 0,
        "zero_thermal_not_ok_checks": totals["thermal_not_ok_checks"] == 0,
        "wall_ratio_below_max": wall_ratio is not None and wall_ratio < CALIBRATION_MAX_WALL_RATIO,
        "wall_ratio_below_max_in_every_mix": bool(mix_ratios)
        and len(per_mix) == len(dev_screen.MIXES)
        and all(r is not None and r < CALIBRATION_MAX_WALL_RATIO for r in mix_ratios),
    }
    if problems:
        status = "CALIBRATION_INVALID"
    elif not (criteria["zero_guard_stops"] and criteria["zero_thermal_not_ok_checks"]):
        status = "CALIBRATION_FAIL"
    elif not complete:
        status = "CALIBRATION_INVALID"
        problems.append("the sweep is incomplete for a non-thermal reason")
    elif all(criteria.values()):
        status = "CALIBRATION_PASS"
    else:
        status = "CALIBRATION_FAIL"
    recommendation = {
        "CALIBRATION_PASS": (
            "3 slots may become the Tier-1/dev default (separate explicit edit citing the "
            "disclosed deviations: arm A only, different worlds)"
        ),
        "CALIBRATION_FAIL": "keep 2 slots for Tier-1/dev; record why",
        "CALIBRATION_INVALID": "no change; the calibration must be rerun",
    }[status]
    return dev_screen.json_safe(
        {
            "status": status,
            "recommendation": recommendation,
            "rule": CALIBRATION_RULE,
            "problems": problems,
            "criteria": criteria,
            "reported_only": {"slowdown_pauses": totals["slowdown_pauses"]},
            "policy_deviations": [
                "only arm A has a 2-slot baseline (the policy says per arm and mix)",
                "the baseline played different worlds (the policy says the same episodes)",
            ],
            "guard_totals": totals,
            "max_in_run_slowdown_ratio": max(ratios) if ratios else None,
            "shard_overlap_seconds": overlap["overlap_seconds"],
            "shortest_shard_seconds": overlap["shortest_seconds"],
            "shard_start_spread_seconds": overlap["start_spread_seconds"],
            "arm_a_episodes_outside_concurrent_window": overlap["a_outside"],
            "slots_held": held,
            "wall_ratio_A_vs_baseline": wall_ratio,
            "wall_per_mix": per_mix,
            "baseline": {k: v for k, v in baseline.items() if k != "times_by_mix"},
        }
    )


# ---------------------------------------------------------------- plan and arguments


def shard_plan(
    rows: Sequence[Mapping[str, Any]],
    seeds: Sequence[int],
    mixes: Sequence[str],
    shard: int,
    shards: int,
    smoke: bool,
) -> List[Tuple[str, Mapping[str, Any]]]:
    """One shard's episodes (protocol.md, "Shards").

    World-mix rows (mix-major) go to shard ``j % shards``; a shard plays its rows ordered by
    (world index, mix), every arm per row in arm order, then any R episodes it owns. R for a
    mix replays the mix's first world in shard ``(j0 + 1) % shards``, so with 3 shards a
    replay always runs in a different process from its H800 episode. A smoke plays arms A
    and H400 only, never R.
    """
    if smoke:
        return [(arm, row) for row in rows for arm in SMOKE_ARMS]
    index = {int(seed): i for i, seed in enumerate(seeds)}
    order = {mix: i for i, mix in enumerate(mixes)}
    mine = [row for j, row in enumerate(rows) if j % shards == shard]
    mine.sort(key=lambda row: (index[int(row["world_seed"])], order[row["mix"]]))
    plan: List[Tuple[str, Mapping[str, Any]]] = [(arm, row) for row in mine for arm in SWEEP_ARMS]
    for mix in mixes:
        first = [(j, row) for j, row in enumerate(rows) if row["mix"] == mix][:REPLAY_WORLDS]
        plan.extend((REPLAY_ARM, row) for j, row in first if (j + 1) % shards == shard)
    return plan


def plan_keys(plan: Sequence[Tuple[str, Mapping[str, Any]]]) -> List[List[Any]]:
    """``[[arm, mix, world_seed], ...]`` (JSON-safe identity of a plan)."""
    return [[arm, row["mix"], int(row["world_seed"])] for arm, row in plan]


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
    run.add_argument("--slot-timeout-seconds", type=float, default=180.0)
    run.add_argument("--thermal-backoff-seconds", type=float, default=60.0)
    run.add_argument("--thermal-max-backoffs", type=int, default=5)
    run.add_argument("--barrier-timeout-seconds", type=float, default=BARRIER_TIMEOUT_SECONDS)
    free = sub.add_parser("slots-free", help="pre-launch: are all 3 pool-3 slots free now?")
    free.add_argument("--slot-lock-root", type=Path, default=dev_screen.DEFAULT_SLOT_LOCK_ROOT)
    merge = sub.add_parser("merge", help="combine the 3 shard dirs into summary.json")
    merge.add_argument("--shard-dirs", required=True, nargs="+", type=Path)
    merge.add_argument("--out", required=True, type=Path, help="new output dir (must not exist)")
    merge.add_argument("--baseline-events", type=Path, default=BASELINE_EVENTS)
    args = parser.parse_args(argv)
    if args.command == "run":
        if args.thermal_backoff_seconds <= 0 or args.thermal_max_backoffs < 0:
            parser.error("need --thermal-backoff-seconds > 0 and --thermal-max-backoffs >= 0")
        if args.barrier_timeout_seconds <= 0:
            parser.error("need --barrier-timeout-seconds > 0")
        if args.smoke_frames is not None:
            args.mixes = tuple(m for m in args.smoke_mixes.split(",") if m)
            episodes = len(args.mixes) * args.worlds_per_mix * len(SMOKE_ARMS)
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
    if args.command == "slots-free":
        return slots_free_main(args)
    return run_main(args, argv)


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(int(pid), 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _barrier_problems(
    markers: Mapping[int, Mapping[str, Any] | None],
    identity: Mapping[str, Any],
    mine: datetime,
    timeout: float,
    pid_alive: Callable[[int], bool],
) -> List[str]:
    """Why the barrier is not (yet) met; empty when all shards are ready."""
    problems = [f"shard {k} has no readable marker" for k, m in markers.items() if m is None]
    present = {k: m for k, m in markers.items() if m is not None}
    for k, marker in sorted(present.items()):
        if any(marker.get(key) != value for key, value in identity.items()):
            problems.append(f"shard {k} marker is from another commit, protocol or plan")
        written = dev_screen._utc(str(marker.get("written_utc")))
        if abs((written - mine).total_seconds()) > timeout:
            problems.append(f"shard {k} marker is stale")
        if not pid_alive(int(marker.get("pid", -1))):
            problems.append(f"shard {k} process {marker.get('pid')} is not running")
    slots = sorted(str(m.get("slot_held")) for m in present.values())
    if len(present) == len(markers) and slots != sorted(dev_screen.SLOT_POOLS[SLOT_POOL]):
        problems.append(f"slots held {slots} are not the 3 distinct pool-3 slots")
    return problems


def start_barrier(
    barrier_dir: Path,
    shard: int,
    identity: Mapping[str, Any],
    slot_held: str,
    timeout: float,
    *,
    poll: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
    clock: Callable[[], float] = time.monotonic,
    pid_alive: Callable[[int], bool] = _pid_alive,
) -> Dict[str, Any]:
    """Rendezvous of the 3 non-smoke shards, after each took its slot, before any episode.

    Writes ``<barrier_dir>/shard-<shard>.json`` (atomically; commit, protocol sha256,
    shards, slot held, pid, time), then waits at most ``timeout`` s until every shard's
    marker is present, fresh (written within ``timeout`` of this one), from a live process,
    with the same ``identity``, and the markers hold the 3 distinct pool-3 slots. Returns
    ``{"passed": bool, "problems": [...], ...}``; a shard that does not pass plays nothing.
    """
    barrier_dir = Path(barrier_dir)
    barrier_dir.mkdir(parents=True, exist_ok=True)
    mine = datetime.now(timezone.utc)
    own = {**identity, "shard": shard, "slot_held": slot_held, "pid": os.getpid()}
    own["written_utc"] = mine.isoformat()
    tmp = barrier_dir / f".shard-{shard}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps(own, sort_keys=True), encoding="utf-8")
    os.replace(tmp, barrier_dir / f"shard-{shard}.json")
    started = clock()
    while True:
        markers: Dict[int, Mapping[str, Any] | None] = {}
        for k in range(SHARDS):
            try:
                markers[k] = json.loads((barrier_dir / f"shard-{k}.json").read_text("utf-8"))
            except (OSError, ValueError):
                markers[k] = None
        problems = _barrier_problems(markers, identity, mine, timeout, pid_alive)
        waited = clock() - started
        if not problems or waited >= timeout:
            return {
                "passed": not problems,
                "problems": problems,
                "dir": str(barrier_dir),
                "timeout_seconds": timeout,
                "waited_seconds": round(waited, 3),
                "markers": {str(k): m for k, m in markers.items()},
            }
        sleep(poll)


def slots_free_main(args: argparse.Namespace) -> int:
    """Pre-launch check: try-lock all 3 pool-3 slots at once (non-blocking), then release.

    Exit 0 iff all three are free right now (no strict run or other holder), else 2.
    """
    try:
        slots = dev_screen.acquire_cpu_slots(args.slot_lock_root, SHARDS, 0.0, pool=SLOT_POOL)
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"not all {SHARDS} pool-3 slots are free: {exc}", file=sys.stderr)
        return 2
    dev_screen.release_cpu_slots(slots)
    print(json.dumps({"slots_free": list(dev_screen.SLOT_POOLS[SLOT_POOL])}))
    return 0


def run_refusal(args: argparse.Namespace, now: datetime) -> str | None:
    """Why this ``run`` must not start (checked before any lock, output or episode)."""
    smoke = args.smoke_frames is not None
    if not args.use_slot_locks or args.slot_pool != SLOT_POOL or not args.thermal_guard:
        return "this sweep requires --use-slot-locks --slot-pool 3 --thermal-guard"
    if smoke and (args.shards != 1 or args.shard != 0):
        return "a smoke runs as one shard: --shard 0 --shards 1"
    if not smoke and args.shards != SHARDS:
        return f"refusing --shards {args.shards}; the pre-declared run is {SHARDS} shards"
    if not 0 <= args.shard < args.shards:
        return f"--shard must be in 0..{args.shards - 1}"
    if not smoke and args.worlds_per_mix != WORLDS_PER_MIX:
        return (
            f"refusing a non-smoke run with {args.worlds_per_mix} worlds per mix; the "
            f"pre-declared size is {WORLDS_PER_MIX}"
        )
    if args.deadline_utc <= now:
        return "deadline already passed"
    if not smoke and (args.deadline_utc - now).total_seconds() > MAX_WALL_SECONDS:
        return f"--deadline-utc exceeds the {MAX_WALL_SECONDS} s wall-time cap"
    out = Path(args.out).resolve()
    if out.exists() or dev_screen.FORBIDDEN_OUTPUT_ROOT in out.parts:
        return f"--out {out} exists or is inside a forbidden root"
    if not smoke:
        dirty = v7sweep.tracked_modifications(dev_screen._git_state().get("dirty_paths"))
        if dirty:
            return f"refusing a non-smoke sweep with tracked modifications: {dirty}"
    return None


def run_main(args: argparse.Namespace, argv: Sequence[str] | None) -> int:
    """Play one shard: refusals, preflight, AC check, one pool-3 slot, then the loop."""
    refusal = run_refusal(args, datetime.now(timezone.utc))
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    smoke = args.smoke_frames is not None
    spec = SMOKE_SPEC if smoke else SPEC
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
    try:
        slots = dev_screen.acquire_cpu_slots(
            args.slot_lock_root, 1, args.slot_timeout_seconds, pool=SLOT_POOL
        )
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"CPU slot lock not acquired: {exc}", file=sys.stderr)
        return 2
    try:
        held = Path(slots[0].name).name
        barrier = None
        if not smoke:
            identity = {
                "commit": dev_screen._git_state().get("commit"),
                "protocol_sha256": dev_screen.sha256_file(Path(spec.protocol)),
                "shards": args.shards,
            }
            barrier = start_barrier(
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
            args, argv, spec, out, seeds, disjoint, parity, profile, smoke, held, barrier=barrier
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
    runner: Callable[..., Dict[str, Any]] | None = None,
    barrier: Mapping[str, Any] | None = None,
) -> int:
    """The shard body (``run_main`` holds the slot lock around it).

    ``runner`` defaults to ``dev_screen.run_episode`` (looked up at call time).
    """
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
        "sweep_id": SWEEP_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "protocol": str(Path(spec.protocol).resolve()),
        "protocol_sha256": dev_screen.sha256_file(Path(spec.protocol)),
        "selection_rule": SELECTION_RULE,
        "calibration_rule": CALIBRATION_RULE,
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
            arm: {"lambda": arm_lambda(arm), "method": arm_method(arm)}
            for arm in (*SWEEP_ARMS, REPLAY_ARM)
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
            "held_from": "before --out existed until shard_summary.json was written",
        },
        "thermal_guard": {
            **guard.config(),
            "key": THERMAL_KEY,
            "backoff_seconds": args.thermal_backoff_seconds,
            "max_backoffs": args.thermal_max_backoffs,
        },
        "start_barrier": barrier,
        "ac_power_checked_at_start": True,
        "threads": {"torch_intraop": 2, "torch_interop": 1},
    }
    dev_screen.write_new_json(out / "intent.json", intent)
    entries, stopped_reason, guard_events = _play(
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
    for sha, path in snapshots.items():
        if dev_screen.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the sweep")
    check = self_check(entries, seeds, smoke)
    shard_summary = {
        "schema_version": SHARD_SCHEMA,
        "sweep_id": SWEEP_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "smoke": smoke,
        "selection": "SMOKE_NO_SELECTION" if smoke else "see merge summary.json",
        "shard": args.shard,
        "shards": args.shards,
        "slot_held": slot_held,
        "started_utc": intent["started_utc"],
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "episodes_run": len(entries),
        "planned_episodes": len(plan),
        "stopped_reason": stopped_reason,
        "guard": guard_stats(guard_events),
        "self_check": check,
        "source": {"commit": git.get("commit"), "dirty_paths": git.get("dirty_paths")},
        "intent_sha256": dev_screen.sha256_file(out / "intent.json"),
    }
    dev_screen.write_new_json(out / "shard_summary.json", dev_screen.json_safe(shard_summary))
    print(
        json.dumps({"shard": args.shard, "episodes_run": len(entries), "stopped": stopped_reason})
    )
    return 0 if check["passes"] else 3


def _play(
    args: argparse.Namespace,
    spec: dev_screen.ScreenSpec,
    plan: Sequence[Tuple[str, Mapping[str, Any]]],
    seeds: Sequence[int],
    lookup: Mapping[str, tuple],
    profile: Any,
    records_dir: Path,
    guard: Any,
    out: Path,
    runner: Callable[..., Dict[str, Any]] | None,
    admit: Callable[..., Tuple[bool, str | None, Dict[str, int]]],
) -> Tuple[List[Dict[str, Any]], str | None, List[Dict[str, Any]]]:
    """The episode loop: deadline budget, then the guard's admission, then one episode.

    As ``dev_screen --thermal-guard``, except that the slowdown key is the whole shard
    (:data:`THERMAL_KEY`; protocol.md says why). Returns entries, stop reason, guard events.
    """
    entries: List[Dict[str, Any]] = []
    guard_events: List[Dict[str, Any]] = []
    stopped_reason = None
    world_index = {seed: index for index, seed in enumerate(seeds)}

    def remaining() -> float:
        return (args.deadline_utc - datetime.now(timezone.utc)).total_seconds()

    with (out / "events.jsonl").open("x", encoding="utf-8") as events:

        def log_guard(event: Dict[str, Any]) -> None:
            stamped = dev_screen.json_safe(
                {"utc": datetime.now(timezone.utc).isoformat(), "done": len(entries), **event}
            )
            guard_events.append(stamped)
            events.write(dev_screen.canonical_json(stamped) + "\n")
            events.flush()

        for arm, row in plan:
            spent = [e["wall_seconds"] for e in entries]
            budget = max(args.min_episode_budget_seconds, 2 * v7sweep._mean(spent) if spent else 0)
            if remaining() < budget:
                stopped_reason = f"deadline: {remaining():.0f}s left < {budget:.0f}s budget"
                break
            admitted, reason, _ = admit(
                guard,
                backoff_seconds=args.thermal_backoff_seconds,
                max_backoffs=args.thermal_max_backoffs,
                seconds_left=remaining,
                budget_seconds=budget,
                log=log_guard,
                sleep=dev_screen._thermal_sleep,
            )
            if not admitted:
                stopped_reason = reason
                break
            play = runner if runner is not None else dev_screen.run_episode
            entry = play(
                arm,
                row,
                world_index[row["world_seed"]],
                lookup,
                profile,
                records_dir,
                smoke_frames=args.smoke_frames,
                spec=spec,
            )
            entries.append(entry)
            guard.record_episode(THERMAL_KEY, entry["wall_seconds"])
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
                "shard": args.shard,
                "done": len(entries),
                "planned": len(plan),
                "arm": arm,
                "mix": row["mix"],
                "world_seed": row["world_seed"],
                "wall_seconds": round(entry["wall_seconds"], 3),
                "mass_integral": entry["record"]["mass_integral"],
            }
            events.write(dev_screen.canonical_json(event) + "\n")
            events.flush()
    return entries, stopped_reason, guard_events


# ---------------------------------------------------------------- merge


def _load_shard(directory: Path) -> Dict[str, Any]:
    directory = Path(directory)
    intent = json.loads((directory / "intent.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "shard_summary.json").read_text(encoding="utf-8"))
    events = [
        json.loads(line)
        for line in (directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    entries = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((directory / "records").glob("*.json"))
    ]
    return {
        "dir": str(directory),
        "intent": intent,
        "summary": summary,
        "events": events,
        "entries": entries,
    }


def merge_refusal(shards: Sequence[Mapping[str, Any]]) -> str | None:
    """Why these shard dirs cannot be merged into one sweep (structure, not results)."""
    if len(shards) != SHARDS:
        return f"need exactly {SHARDS} shard dirs"
    intents = [s["intent"] for s in shards]
    if any(i.get("schema_version") != SCHEMA or i.get("sweep_id") != SWEEP_ID for i in intents):
        return "a shard is not from this sweep"
    if any(i.get("smoke_frames") is not None for i in intents):
        return "a smoke shard cannot be merged"
    if sorted(int(i.get("shard", -1)) for i in intents) != list(range(SHARDS)):
        return f"shard indices are not 0..{SHARDS - 1}"
    if any(int(i.get("shards", 0)) != SHARDS for i in intents):
        return f"a shard was not planned as one of {SHARDS}"
    for key in ("protocol_sha256", "worlds", "mixes"):
        if any(i.get(key) != intents[0].get(key) for i in intents):
            return f"shards differ in {key}"
    if any((i.get("git") or {}).get("commit") != intents[0]["git"].get("commit") for i in intents):
        return "shards ran from different commits"
    seeds = intents[0]["worlds"]["seeds"]
    rows = dev_screen._design_rows(seeds)
    for intent in intents:
        expected = plan_keys(
            shard_plan(rows, seeds, intent["mixes"], int(intent["shard"]), SHARDS, False)
        )
        if intent.get("plan") != expected:
            return f"shard {intent['shard']} plan differs from the pre-declared shard plan"
    return None


def merge_main(args: argparse.Namespace) -> int:
    """Combine the three shard dirs into ``summary.json`` (selection + calibration)."""
    out = Path(args.out).resolve()
    if out.exists() or dev_screen.FORBIDDEN_OUTPUT_ROOT in out.parts:
        print(f"--out {out} exists or is inside a forbidden root", file=sys.stderr)
        return 2
    try:
        shards = [_load_shard(directory) for directory in args.shard_dirs]
    except (OSError, ValueError) as exc:
        print(f"cannot read the shard dirs: {exc}", file=sys.stderr)
        return 2
    refusal = merge_refusal(shards)
    if refusal:
        print(refusal, file=sys.stderr)
        return 2
    shards.sort(key=lambda s: int(s["intent"]["shard"]))
    intents = [s["intent"] for s in shards]
    seeds, mixes = intents[0]["worlds"]["seeds"], intents[0]["mixes"]
    entries = [entry for s in shards for entry in s["entries"]]
    planned = sum(len(i["plan"]) for i in intents)
    replay_planned = sum(1 for i in intents for key in i["plan"] if key[0] == REPLAY_ARM)
    summary = summarize(
        entries,
        seeds,
        mixes,
        smoke=False,
        planned_episodes=planned,
        replay_planned=replay_planned,
        source=intents[0]["git"],
    )
    shard_rows = [
        {
            "shard": s["intent"]["shard"],
            "dir": s["dir"],
            "slot_held": s["summary"].get("slot_held"),
            "started_utc": s["summary"].get("started_utc"),
            "finished_utc": s["summary"].get("finished_utc"),
            "episodes_run": s["summary"].get("episodes_run"),
            "planned_episodes": len(s["intent"]["plan"]),
            "stopped_reason": s["summary"].get("stopped_reason"),
            "guard": guard_stats(s["events"]),
            "start_barrier_passed": (s["intent"].get("start_barrier") or {}).get("passed"),
            "arm_a_finished_utc": [
                e["utc"] for e in s["events"] if e.get("arm") == "A" and "wall_seconds" in e
            ],
            "intent_sha256": dev_screen.sha256_file(Path(s["dir"]) / "intent.json"),
        }
        for s in shards
    ]
    baseline = load_baseline(args.baseline_events)
    summary["calibration"] = calibration_report(
        shard_rows, entries, baseline, bool(summary["complete"])
    )
    summary.update(
        {
            "shards": shard_rows,
            "episodes_run": len(entries),
            "planned_episodes": planned,
            "stopped_reasons": [row["stopped_reason"] for row in shard_rows],
            "merged_utc": datetime.now(timezone.utc).isoformat(),
        }
    )
    out.mkdir(parents=True)
    dev_screen.write_new_json(out / "summary.json", dev_screen.json_safe(summary))
    selection = summary["selection"]
    print(
        json.dumps(
            {
                "selection": selection["status"],
                "lambda": selection["selected_lambda"],
                "calibration": summary["calibration"]["status"],
            }
        )
    )
    return 0 if summary["self_check"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
