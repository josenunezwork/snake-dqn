#!/usr/bin/env python3
"""Tier-1 DEV lambda sweep (non-authoritative): Apex + v7 space preference vs Apex + v5.

Pre-declaration: ``protocol.md`` beside this file. Arms (paired on every world):

  A     champion + ``free-space-veto/v5-boost-aware`` (released; ``lambda = 0``).
  L100  champion + ``free-space-veto/v7-space-preference(lambda=1.0)``.
  L200  champion + v7 at ``lambda = 2.0``.
  L400  champion + v7 at ``lambda = 4.0``.
  R     L200 repeated on the first world of each mix (replay control).

The sweep reuses ``research/apex_safety_20260926/dev_screen.py`` pieces unchanged and
picks ONE lambda by the pre-declared selection rule (``summary.json`` ``selection``), or
stops. Inert lambdas cannot be selected (activity gate). It always holds a CPU slot lock,
refuses to start on battery and, outside a smoke, refuses a tree with tracked
modifications (the screen could never bind to such a sweep).

Usage (about 1 h; hard cap 3 h):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v7_lambda_sweep_20261002/sweep.py \\
    --out /path/to/new/dir --deadline-utc <now + at most 3 h, with UTC offset>
Smoke (plumbing only, smoke namespace, arms A and L400 on 1 world, <= 500 frames):
  ... sweep.py --out /tmp/x --deadline-utc ... --smoke-frames 500 --worlds-per-mix 1
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
from datetime import datetime, timezone  # noqa: E402
from functools import partial  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence, Tuple  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402
from research.apex_veto_v6_screen_20261002 import screen as v6screen  # noqa: E402

HERE = Path(__file__).resolve().parent
SWEEP_ID = "apex-veto-v7-dev-v1"
SCHEMA = "apex-veto-v7-lambda-sweep/v1"
AUTHORITY = "development sweep (Tier-1 dev, non-authoritative)"
DOMAIN = SWEEP_ID
SMOKE_DOMAIN = "apex-veto-v7-dev-smoke-v1"
NAMESPACE = "worlds"
V5_METHOD = "free-space-veto/v5-boost-aware"
# Amendment 1 (before any sweep episode): the grid was {0.25, 0.5, 1.0} under a log area
# score, which made two-candidate decisions unswitchable; see protocol.md.
LAMBDAS = (1.0, 2.0, 4.0)
ARM_LAMBDAS: Dict[str, float] = {"A": 0.0, "L100": 1.0, "L200": 2.0, "L400": 4.0}
SWEEP_ARMS = tuple(ARM_LAMBDAS)
V7_ARMS = ("L100", "L200", "L400")
REPLAY_ARM, REPLAY_OF, REPLAY_WORLDS = "R", "L200", 1
SMOKE_ARMS = ("A", "L400")
WORLDS_PER_MIX = 8
# Activity gate: a lambda that (almost) never changes a decision is v5 and cannot qualify.
ACTIVE_MIN_EPISODES_PER_MIX = 2  # episodes with rerank_changes > 0, in EACH mix
ACTIVE_MIN_NONZERO_DELTAS_PER_MIX = 1  # worlds with a paired delta != 0, in EACH mix
ACTIVE_MIN_CHANGE_RATE = 1e-4  # pooled rerank_changes / v7 decisions
# Clear-loser rule: excluded in a mix iff the one-sided 90% upper bound of its mean delta
# (mean + t_{0.90, n-1} * sd / sqrt(n)) is below 0. One-sided 0.90 Student t quantiles.
T90_ONE_SIDED = {
    1: 3.0777,
    2: 1.8856,
    3: 1.6377,
    4: 1.5332,
    5: 1.4759,
    6: 1.4398,
    7: 1.4149,
    8: 1.3968,
    9: 1.3830,
    10: 1.3722,
}
MAX_WALL_SECONDS = 3 * 3600
EXCLUSION_PREFIX = 1000
# The v7 screen's domains are excluded so the sweep can never consume screen worlds.
SCREEN_DOMAINS = ("apex-veto-v7-screen-v1", "apex-veto-v7-screen-smoke-v1")
EARLIER_DOMAINS: Dict[str, Sequence[str]] = {
    **{name: tuple(purposes) for name, purposes in v6screen.EARLIER_DOMAINS.items()},
    **{name: (NAMESPACE,) for name in SCREEN_DOMAINS},
    DOMAIN: (NAMESPACE,),
    SMOKE_DOMAIN: (NAMESPACE,),
}
STATUS_SELECTED = "SELECTED"
STATUS_NONE_ACTIVE = "NONE_ACTIVE"
STATUS_NONE_QUALIFIES = "NONE_QUALIFIES"
SELECTION_RULE = (
    "per lambda: paired mass_integral delta (v7_lambda - v5) per world and mix. ACTIVE iff "
    f"in EACH mix >= {ACTIVE_MIN_EPISODES_PER_MIX} episodes have rerank_changes > 0 and >= "
    f"{ACTIVE_MIN_NONZERO_DELTAS_PER_MIX} world has a nonzero delta, and the pooled "
    f"rerank_changes/decisions >= {ACTIVE_MIN_CHANGE_RATE:g}; a CLEAR LOSER in a mix iff "
    "mean + t(0.90, n-1) * sd / sqrt(n) < 0 there; QUALIFIES iff active and a clear loser in "
    "no mix. SELECTED = the qualifying lambda with the highest pooled mean delta (ties: "
    "larger lambda); NONE_ACTIVE if no lambda is active, else NONE_QUALIFIES (stop, no "
    "screen). Before that: SMOKE_NO_SELECTION, INVALID_SELF_CHECK_FAILED, INCOMPLETE, "
    "INVALID_NONDETERMINISTIC (R vs L200), in order."
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


def install_v5(hero: Any) -> Any:
    """Arm A: the released v5 boost-aware veto."""
    from src.evaluation.safety_veto_v5 import install_boost_aware_veto

    return install_boost_aware_veto(hero)


def install_v7(hero: Any, lam: float) -> Any:
    """v7 arms: the space-preference veto at ``lam``."""
    from src.evaluation.safety_veto_v7 import install_space_preference_veto

    return install_space_preference_veto(hero, lam)


INSTALLERS: Dict[str, Any] = {"A": install_v5}
INSTALLERS.update({arm: partial(install_v7, lam=ARM_LAMBDAS[arm]) for arm in V7_ARMS})
INSTALLERS[REPLAY_ARM] = INSTALLERS[REPLAY_OF]  # the same object: R repeats L200


def arm_lambda(arm: str) -> float:
    """The arm's lambda (R is L200's)."""
    return ARM_LAMBDAS[REPLAY_OF if arm == REPLAY_ARM else arm]


def arm_method(arm: str) -> str:
    from src.evaluation.safety_veto_v7 import method_for

    return V5_METHOD if arm == "A" else method_for(arm_lambda(arm))


def expected_descriptor(arm: str) -> Dict[str, Any]:
    """The probe identity each arm's records must carry (writer: ``veto.record()``)."""
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
    from src.evaluation.safety_veto_v7 import SpacePreferenceVeto

    if arm == "A":
        return BoostAwareFreeSpaceVeto().descriptor()
    return SpacePreferenceVeto(arm_lambda(arm)).descriptor()


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


def v7_identities_hold(diagnostics: Mapping[str, Any], counters: Mapping[str, Any]) -> bool:
    """``SpacePreferenceCounters`` identities against the probe's v2 counters (reported)."""

    def get(source: Mapping[str, Any], key: str) -> int:
        return int(source.get(key) or 0)

    v5 = diagnostics.get("v5") if isinstance(diagnostics.get("v5"), Mapping) else {}
    over_kept = get(diagnostics, "rerank_changes_of_v5_kept")
    return (
        counters.get("decisions") == get(diagnostics, "decisions") == get(v5, "decisions")
        and counters.get("kept_base") == get(diagnostics, "kept") == get(v5, "kept") - over_kept
        and counters.get("vetoes_applied")
        == get(diagnostics, "vetoes_applied")
        == get(v5, "vetoes_applied") + over_kept
        and counters.get("fallback_no_spacious")
        == get(diagnostics, "no_spacious")
        == get(v5, "no_spacious")
        and get(diagnostics, "rerank_decisions")
        == get(diagnostics, "rerank_pruned") + get(diagnostics, "rerank_scored")
        and get(diagnostics, "rerank_changes") <= get(diagnostics, "rerank_scored")
        and get(diagnostics, "rerank_changes_tail_release_driven")
        <= get(diagnostics, "rerank_changes")
        and get(diagnostics, "area_cap_hits") <= get(diagnostics, "area_evaluations")
    )


def tracked_modifications(dirty_paths: Any) -> List[str]:
    """``git status --porcelain`` lines other than untracked (``??``) files."""
    lines = [line for line in str(dirty_paths or "").splitlines() if line.strip()]
    return [line for line in lines if not line.startswith("??")]


# ---------------------------------------------------------------- self-check (gating)


def check_entry(entry: Mapping[str, Any], smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one sweep entry (dev_screen writer)."""
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
        if not isinstance(diagnostics, Mapping) or "boost_landing_vetoes" not in diagnostics:
            warnings.append("arm A v5 counters missing")
    elif not isinstance(diagnostics, Mapping) or "rerank_changes" not in diagnostics:
        # Gating since amendment 1: the activity gate reads rerank_changes.
        failures.append(f"arm {arm} v7 counters missing")
    elif not v7_identities_hold(diagnostics, counters):
        warnings.append(f"arm {arm} v7 counters disagree with the probe's v2 counters")
    return {"failures": failures, "warnings": warnings}


def self_check(entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], smoke: bool) -> Dict:
    """Check every entry; ``passes`` iff there are entries and no gating failure."""
    planned = {int(seed) for seed in seeds}
    failures: List[str] = []
    warnings: List[str] = []
    for entry in entries:
        name = f"{entry.get('arm')}-{entry.get('mix')}-{entry.get('world_seed')}"
        result = check_entry(entry, smoke)
        if entry.get("world_seed") not in planned:
            result["failures"].append("world seed is not a planned seed")
        failures.extend(f"{name}: {message}" for message in result["failures"])
        warnings.extend(f"{name}: {message}" for message in result["warnings"])
    return {
        "entries": len(entries),
        "failures": failures,
        "warnings": warnings,
        "passes": not failures and bool(entries),
    }


# ---------------------------------------------------------------- analysis


def _mean(values: Sequence[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _sd(values: Sequence[float]) -> float | None:
    if len(values) < 2:
        return None
    centre = sum(values) / len(values)
    return math.sqrt(sum((v - centre) ** 2 for v in values) / (len(values) - 1))


def lambda_table(
    entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], mixes: Sequence[str]
) -> Dict[str, Any]:
    """Per v7 arm: per-mix and pooled paired mass deltas against arm A."""
    by_key = {(e["arm"], e["mix"], int(e["world_seed"])): e["record"] for e in entries}
    diag = {(e["arm"], e["mix"], int(e["world_seed"])): _diagnostics(e) for e in entries}
    table: Dict[str, Any] = {}
    for arm in V7_ARMS:
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
            episode_changes = [int(d.get("rerank_changes") or 0) for d in arm_diags]
            changes += sum(episode_changes)
            decisions += sum(int(d.get("decisions") or 0) for d in arm_diags)
            upper = _upper_bound_90(deltas)
            per_mix[mix] = {
                "paired_worlds": len(paired),
                "paired_seeds": paired,
                "deltas": deltas,
                "mean_delta": _mean(deltas),
                "sd_delta": _sd(deltas),
                "mean_A": _mean([by_key[("A", mix, s)]["mass_integral"] for s in paired]),
                "mean_arm": _mean([by_key[(arm, mix, s)]["mass_integral"] for s in paired]),
                "wins": sum(1 for d in deltas if d > 0),
                "losses": sum(1 for d in deltas if d < 0),
                "ties": sum(1 for d in deltas if d == 0),
                "episodes_with_rerank_changes": sum(1 for c in episode_changes if c > 0),
                "rerank_changes": sum(episode_changes),
                "nonzero_deltas": sum(1 for d in deltas if d != 0),
                "upper_bound_90_one_sided": upper,
                "clear_loser": upper is not None and upper < 0,
            }
        rate = changes / decisions if decisions else 0.0
        active = rate >= ACTIVE_MIN_CHANGE_RATE and all(
            row["episodes_with_rerank_changes"] >= ACTIVE_MIN_EPISODES_PER_MIX
            and row["nonzero_deltas"] >= ACTIVE_MIN_NONZERO_DELTAS_PER_MIX
            for row in per_mix.values()
        )
        clear_loser = any(row["clear_loser"] for row in per_mix.values())
        table[arm] = {
            "lambda": ARM_LAMBDAS[arm],
            "per_mix": per_mix,
            "pooled_pairs": len(pooled),
            "pooled_mean_delta": _mean(pooled),
            "pooled_sd_delta": _sd(pooled),
            "rerank_changes": changes,
            "decisions": decisions,
            "rerank_change_rate_per_decision": rate,
            "active": active,
            "clear_loser_in_some_mix": clear_loser,
            "qualifies": active and not clear_loser and bool(per_mix),
        }
    return table


def _diagnostics(entry: Mapping[str, Any]) -> Mapping[str, Any]:
    diagnostics = entry.get("veto_diagnostics")
    return diagnostics if isinstance(diagnostics, Mapping) else {}


def _upper_bound_90(deltas: Sequence[float]) -> float | None:
    """One-sided 90% upper bound of the mean (None below two values)."""
    sd = _sd(deltas)
    if sd is None:
        return None
    n = len(deltas)
    t = T90_ONE_SIDED.get(n - 1, 1.2816)  # normal quantile beyond the table
    return sum(deltas) / n + t * sd / math.sqrt(n)


def select_lambda(table: Mapping[str, Any]) -> Tuple[str, float | None, List[float]]:
    """``(status, lambda, qualifying)`` by the pre-declared rule (steps 5-8)."""
    if not any(table[arm]["active"] for arm in V7_ARMS):
        return STATUS_NONE_ACTIVE, None, []
    qualifying = [arm for arm in V7_ARMS if table[arm]["qualifies"]]
    if not qualifying:
        return STATUS_NONE_QUALIFIES, None, []
    best = max(qualifying, key=lambda arm: (table[arm]["pooled_mean_delta"], ARM_LAMBDAS[arm]))
    return STATUS_SELECTED, ARM_LAMBDAS[best], [ARM_LAMBDAS[arm] for arm in qualifying]


V7_SUMMED = (
    "decisions",
    "rerank_decisions",
    "rerank_pruned",
    "rerank_scored",
    "rerank_changes",
    "rerank_changes_of_v5_kept",
    "rerank_changes_boost_mode",
    "rerank_changes_tail_release_driven",
    "area_evaluations",
    "area_cap_hits",
    "static_area_evaluations",
    "extra_landing_checks",
    "area_eval_seconds_total",
    "apply_seconds_total",
)


def v7_counter_report(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reported only: summed v7 counters, cost per decision and wall time per arm."""
    out: Dict[str, Any] = {}
    for arm in (*SWEEP_ARMS, REPLAY_ARM):
        rows = [e for e in entries if e.get("arm") == arm]
        if not rows:
            continue
        wall = [float(e.get("wall_seconds", 0.0)) for e in rows]
        row: Dict[str, Any] = {"episodes": len(rows), "mean_episode_wall_seconds": _mean(wall)}
        if arm != "A":
            totals = {key: 0.0 for key in V7_SUMMED}
            maxima = {"area_eval_seconds_max": 0.0, "apply_seconds_max": 0.0, "area_cap_max": 0}
            for entry in rows:
                diag = entry.get("veto_diagnostics") or {}
                for key in V7_SUMMED:
                    totals[key] += diag.get(key) or 0
                for key in maxima:
                    maxima[key] = max(maxima[key], diag.get(key) or 0)
            decisions = totals["decisions"]
            row.update(totals)
            row.update(maxima)
            row["mean_apply_seconds_per_decision"] = (
                totals["apply_seconds_total"] / decisions if decisions else None
            )
            row["mean_area_eval_seconds_per_decision"] = (
                totals["area_eval_seconds_total"] / decisions if decisions else None
            )
            row["rerank_change_rate_per_decision"] = (
                totals["rerank_changes"] / decisions if decisions else None
            )
        out[arm] = row
    return {"by_arm": out, "gating": False}


def death_causes(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reported only: hero death causes per arm and mix (writer: rollout ``probes``)."""
    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    for entry in entries:
        cause = str(((entry.get("record") or {}).get("probes") or {}).get("death_cause"))
        cell = table.setdefault(str(entry["arm"]), {}).setdefault(str(entry["mix"]), {})
        cause = "survived" if cause == "None" else cause
        cell[cause] = cell.get(cause, 0) + 1
    return {"by_arm_and_mix": table, "gating": False}


def replay_report(
    entries: Sequence[Mapping[str, Any]], seeds: Sequence[int], mixes: Sequence[str], planned: int
) -> Dict[str, Any]:
    """R vs L200: every R record must equal its L200 record (canonical JSON)."""
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
        "arm": f"{REPLAY_ARM} repeats {REPLAY_OF}",
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
    replay = replay_report(entries, seeds, mixes, replay_planned)
    complete = len(entries) == planned_episodes and all(
        table[arm]["per_mix"][mix]["paired_worlds"] == len(seeds)
        for arm in V7_ARMS
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
            "activity_gate": {
                "min_episodes_with_changes_per_mix": ACTIVE_MIN_EPISODES_PER_MIX,
                "min_nonzero_deltas_per_mix": ACTIVE_MIN_NONZERO_DELTAS_PER_MIX,
                "min_pooled_change_rate": ACTIVE_MIN_CHANGE_RATE,
            },
            "rule": SELECTION_RULE,
        },
        "source": {
            "commit": source.get("commit"),
            "dirty_paths": source.get("dirty_paths"),
        },
        "lambdas": table,
        "replay_control": replay,
        "self_check": check,
        "reported": {
            "v7_counters": v7_counter_report(entries),
            "death_causes": death_causes(entries),
        },
    }
    return dev_screen.json_safe(summary)


# ---------------------------------------------------------------- run


def parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--out", required=True, type=Path, help="new output dir (must not exist)")
    parser.add_argument("--worlds-per-mix", type=int, default=WORLDS_PER_MIX)
    parser.add_argument("--deadline-utc", required=True, type=dev_screen._utc)
    parser.add_argument("--config", type=Path, default=dev_screen.DEFAULT_CONFIG)
    parser.add_argument("--checkpoint-dir", type=Path, default=dev_screen.DEFAULT_CHECKPOINT_DIR)
    parser.add_argument("--pilot-output", type=Path, default=dev_screen.DEFAULT_PILOT_OUTPUT)
    parser.add_argument("--min-episode-budget-seconds", type=float, default=45.0)
    parser.add_argument("--smoke-frames", type=int, default=None)
    parser.add_argument("--smoke-mixes", default="scripted")
    parser.add_argument("--slot-lock-root", type=Path, default=dev_screen.DEFAULT_SLOT_LOCK_ROOT)
    parser.add_argument("--slot-timeout-seconds", type=float, default=180.0)
    args = parser.parse_args(argv)
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


def plan_episodes(rows: Sequence[Mapping[str, Any]], mixes: Sequence[str], smoke: bool) -> List:
    """Mix-major, every arm per world in order, then R on the first world of each mix."""
    arms = SMOKE_ARMS if smoke else SWEEP_ARMS
    plan = [(arm, row) for row in rows for arm in arms]
    if not smoke:
        for mix in mixes:
            first = [row for row in rows if row["mix"] == mix][:REPLAY_WORLDS]
            plan.extend((REPLAY_ARM, row) for row in first)
    return plan


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    smoke = args.smoke_frames is not None
    if not smoke and args.worlds_per_mix != WORLDS_PER_MIX:
        print(
            f"refusing a non-smoke run with {args.worlds_per_mix} worlds per mix; the "
            f"pre-declared size is {WORLDS_PER_MIX}",
            file=sys.stderr,
        )
        return 2
    now = datetime.now(timezone.utc)
    if args.deadline_utc <= now:
        print("deadline already passed", file=sys.stderr)
        return 2
    if not smoke and (args.deadline_utc - now).total_seconds() > MAX_WALL_SECONDS:
        print(f"--deadline-utc exceeds the {MAX_WALL_SECONDS} s wall-time cap", file=sys.stderr)
        return 2
    out = Path(args.out).resolve()
    if out.exists() or dev_screen.FORBIDDEN_OUTPUT_ROOT in out.parts:
        print(f"--out {out} exists or is inside a forbidden root", file=sys.stderr)
        return 2
    if not smoke:
        dirty = tracked_modifications(dev_screen._git_state().get("dirty_paths"))
        if dirty:
            print(
                "refusing a non-smoke sweep with tracked modifications (the screen could "
                f"never bind to it): {dirty}",
                file=sys.stderr,
            )
            return 2
    spec = SMOKE_SPEC if smoke else SPEC

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
        slots = dev_screen.acquire_cpu_slots(args.slot_lock_root, 1, args.slot_timeout_seconds)
    except (OSError, TimeoutError, ValueError) as exc:
        print(f"CPU slot lock not acquired: {exc}", file=sys.stderr)
        return 2
    try:
        return _run(args, argv, spec, out, seeds, disjoint, parity, profile, smoke)
    finally:
        dev_screen.release_cpu_slots(slots)


def _run(
    args: argparse.Namespace,
    argv: Sequence[str] | None,
    spec: dev_screen.ScreenSpec,
    out: Path,
    seeds: Sequence[int],
    disjoint: Mapping[str, Any],
    parity: Mapping[str, Any],
    profile: Any,
    smoke: bool,
) -> int:
    """The sweep body (``main`` holds the slot lock around it)."""
    out.mkdir(parents=True)
    records_dir = out / "records"
    records_dir.mkdir()
    snapshots = dev_screen.snapshot_checkpoints(args.checkpoint_dir, out)
    lookup = dev_screen.agent_lookup(snapshots)
    rows = [row for row in dev_screen._design_rows(seeds) if row["mix"] in args.mixes]
    plan = plan_episodes(rows, args.mixes, smoke)
    replay_planned = sum(1 for arm, _ in plan if arm == REPLAY_ARM)
    git = dev_screen._git_state()
    intent = {
        "schema_version": SCHEMA,
        "sweep_id": SWEEP_ID,
        "authority": AUTHORITY,
        "promotion_authorized": False,
        "protocol": str(Path(spec.protocol).resolve()),
        "protocol_sha256": dev_screen.sha256_file(Path(spec.protocol)),
        "selection_rule": SELECTION_RULE,
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
        "planned_episodes": len(plan),
        "compute_cap": {
            "planned_episodes": len(plan),
            "deadline_utc": args.deadline_utc.isoformat(),
            "max_wall_seconds": MAX_WALL_SECONDS,
        },
        "slot_locks": {"root": str(args.slot_lock_root), "held": 1},
        "ac_power_checked_at_start": True,
        "threads": {"torch_intraop": 2, "torch_interop": 1},
    }
    dev_screen.write_new_json(out / "intent.json", intent)

    entries: List[Dict[str, Any]] = []
    stopped_reason = None
    world_index = {seed: index for index, seed in enumerate(seeds)}
    with (out / "events.jsonl").open("x", encoding="utf-8") as events:
        for arm, row in plan:
            spent = [e["wall_seconds"] for e in entries]
            budget = max(args.min_episode_budget_seconds, 2 * _mean(spent) if spent else 0)
            remaining = (args.deadline_utc - datetime.now(timezone.utc)).total_seconds()
            if remaining < budget:
                stopped_reason = f"deadline: {remaining:.0f}s left < {budget:.0f}s budget"
                break
            entry = dev_screen.run_episode(
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
            event = {
                "utc": datetime.now(timezone.utc).isoformat(),
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
    for sha, path in snapshots.items():
        if dev_screen.sha256_file(Path(path)) != sha:
            raise RuntimeError(f"checkpoint snapshot {path} changed during the sweep")
    summary = summarize(
        entries,
        seeds,
        args.mixes,
        smoke=smoke,
        planned_episodes=len(plan),
        replay_planned=replay_planned,
        source=git,
    )
    summary.update(
        {
            "finished_utc": datetime.now(timezone.utc).isoformat(),
            "episodes_run": len(entries),
            "planned_episodes": len(plan),
            "stopped_reason": stopped_reason,
            "intent_sha256": dev_screen.sha256_file(out / "intent.json"),
        }
    )
    dev_screen.write_new_json(out / "summary.json", summary)
    selection = summary["selection"]
    print(json.dumps({"selection": selection["status"], "lambda": selection["selected_lambda"]}))
    return 0 if summary["self_check"]["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
