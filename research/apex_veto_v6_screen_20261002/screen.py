#!/usr/bin/env python3
"""Tier-1 screen (non-authoritative): Apex + v6 head-and-fallback veto vs Apex + v5 veto.

Pre-registration: ``protocol.md`` beside this file. The harness is
``research/apex_safety_20260926/dev_screen.py`` driven by :data:`SPEC` (the pattern of
``research/apex_veto_v5_screen_20261001/screen.py``):

  A  champion + ``free-space-veto/v5-boost-aware`` (src/evaluation/safety_veto_v5.py; the
     released Watch-hero configuration on local main).
  B  champion + ``free-space-veto/v6-head-and-fallback`` (src/evaluation/safety_veto_v6.py).
  C  arm A repeated on the first 8 worlds per mix (determinism control).
  D  arm B repeated on the first 4 worlds per mix (B replay control).

The screen always holds a shared CPU slot lock and refuses to start on battery.
After ``dev_screen.main`` finishes, this wrapper self-checks every record and writes
``receipt.json`` (governance Tier-1 artifact), including v6's per-layer counters.
Nothing here changes a default.

Usage (about 2-2.5 h at the pre-registered size; hard cap 4 h):
  SNAKE_DQN_DEVICE=cpu ./venv/bin/python research/apex_veto_v6_screen_20261002/screen.py \\
    --out /path/to/new/dir --deadline-utc <now + at most 4 h, with UTC offset>
Smoke (plumbing only, separate smoke namespace, <= 2 episodes x 500 frames):
  ... screen.py --out /tmp/x --deadline-utc ... --smoke-frames 500 --worlds-per-mix 1 \\
    --determinism-worlds 0 --smoke-mixes scripted
"""

from __future__ import annotations

import os

os.environ.setdefault("SNAKE_DQN_DEVICE", "cpu")
os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")

import json  # noqa: E402
import sys  # noqa: E402
from datetime import datetime, timezone  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Dict, List, Mapping, Sequence  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from research.apex_safety_20260926 import dev_screen  # noqa: E402
from research.apex_veto_v5_screen_20261001 import screen as v5screen  # noqa: E402

HERE = Path(__file__).resolve().parent
SCREEN_ID = "apex-veto-v6-screen-v1"
SCHEMA = "apex-veto-v6-screen/v1"
AUTHORITY = "screen (non-authoritative)"
DOMAIN = SCREEN_ID
SMOKE_DOMAIN = "apex-veto-v6-screen-smoke-v1"
NAMESPACE = "worlds"
RECEIPT_SCHEMA = "apex-veto-v6-screen-receipt/v1"
V5_METHOD = "free-space-veto/v5-boost-aware"
V6_METHOD = "free-space-veto/v6-head-and-fallback"
ARM_METHODS = {"A": V5_METHOD, "B": V6_METHOD, "C": V5_METHOD, "D": V6_METHOD}
V5_ARMS = ("A", "C")
V6_ARMS = ("B", "D")
REPLAY_WORLDS = 4  # arm D: B replayed on the first 4 worlds per mix (protocol.md)
# Pre-stated wall-time cap (protocol.md "Compute cap and operations").
MAX_WALL_SECONDS = 4 * 3600
EXCLUSION_PREFIX = 1000  # seeds per earlier domain/purpose checked (>= any bank's size)
# Every earlier SHA-prefix domain of this recipe we know of (name -> purposes): the v5
# screen's own exclusion set (the v2 Tier-1 screen, whose worlds the SIMD H5000 parity
# check replayed; every strict v2 bank and smoke; the v2 web-serving lane; the v3/v4
# screens and smokes; the v2 trap-horizon study), plus every apex-veto-v5-* domain, the
# v5 death census and its smoke, and this screen's other domain. dev_screen's
# disjointness_report adds the task-aligned challenger namespaces, the strict pilot's
# observed seeds and seeds 0..999.
EARLIER_DOMAINS: Dict[str, Sequence[str]] = {
    **{name: tuple(purposes) for name, purposes in v5screen.EARLIER_DOMAINS.items()},
    "apex-veto-v5-screen-v1": ("worlds",),
    "apex-veto-v5-screen-smoke-v1": ("worlds",),
    "apex-veto-v5-strict-dev-v1": ("worlds",),
    "apex-veto-v5-strict-final-v1": ("worlds",),
    "apex-veto-v5-strict-serving-v1": ("worlds",),
    "apex-veto-v5-strict-smoke-v1": ("worlds",),
    "apex-veto-v5-web-serving-v1": ("watch", "play", "parity"),
    "apex-veto-v5-web-serving-smoke-v1": ("watch", "play", "parity"),
    # v5 death census (docs/research/death_census_v5_2026-10-02.md), which motivated v6.
    "trap-horizon-v5-dev-v1": ("worlds",),
    "trap-horizon-v5-dev-smoke-v1": ("worlds",),
    DOMAIN: (NAMESPACE,),
    SMOKE_DOMAIN: (NAMESPACE,),
}


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
    """Arms A and C: the released v5 boost-aware veto (it has no options)."""
    from src.evaluation.safety_veto_v5 import install_boost_aware_veto

    return install_boost_aware_veto(hero)


def install_v6(hero: Any) -> Any:
    """Arms B and D: the v6 veto with both layers on (its defaults)."""
    from src.evaluation.safety_veto_v6 import install_head_and_fallback_veto

    return install_head_and_fallback_veto(hero)


def _spec(domain: str) -> dev_screen.ScreenSpec:
    return dev_screen.ScreenSpec(
        name=SCREEN_ID,
        schema=SCHEMA,
        authority=AUTHORITY,
        domain=domain,
        namespace=NAMESPACE,
        protocol=HERE / "protocol.md",
        arm_vetoes={"A": install_v5, "B": install_v6, "C": install_v5, "D": install_v6},
        arm_descriptions={
            "A": f"champion + {V5_METHOD} (src/evaluation/safety_veto_v5.py, released)",
            "B": f"champion + {V6_METHOD} (src/evaluation/safety_veto_v6.py)",
            "C": "arm A repeated on the first determinism worlds per mix",
            "D": f"arm B repeated on the first {REPLAY_WORLDS} worlds per mix",
        },
        extra_namespaces=earlier_namespaces(domain),
        require_slot_locks=True,
        require_ac_power=True,
        owner="Apex safety lane",
        hypothesis=(
            "Adding opponent-head avoidance and a no-spacious fallback (landing switch "
            "plus budgeted escape ranking) on top of the released v5 veto raises the Apex "
            "champion's H5000 mass_integral (B - A > 0)."
        ),
        decision_informs=(
            "RECOMMEND_STRICT_GATE: design a Tier-2 strict gate for v6 against v5; "
            "NOT_ADVANCED: stop (no change to the served v5 configuration)"
        ),
        primary_metric="paired mass_integral B-A per world per mix",
        estimator="one-sided paired t, Holm over 3 mixes, alpha 0.05, >=2 of 3",
        max_wall_seconds=MAX_WALL_SECONDS,
        replay_worlds=REPLAY_WORLDS,
    )


SPEC = _spec(DOMAIN)
SMOKE_SPEC = _spec(SMOKE_DOMAIN)


def expected_descriptor(arm: str) -> Dict[str, Any]:
    """The probe identity each arm's records must carry (writer: ``veto.record()``)."""
    from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto
    from src.evaluation.safety_veto_v6 import HeadAndFallbackVeto

    return (HeadAndFallbackVeto() if arm in V6_ARMS else BoostAwareFreeSpaceVeto()).descriptor()


# ---------------------------------------------------------------- self-check and receipt
# Each rule names its writer. Gating rules cover the decision path (arm identity,
# pairing/world identity, profile digest, denominators); diagnostics are reported only.


def check_record(
    record: Mapping[str, Any],
    arm: str,
    *,
    world_seed: int,
    mix: str,
    roster_hashes: Sequence[str],
    smoke: bool,
) -> List[str]:
    """Gating checks on one rollout record (the shape the strict producer writes)."""
    from src.evaluation.strict_promotion import (
        StrictPromotionArtifactError,
        _expected_world_identity,
        _validate_candidate_wrapper_probe,
    )

    failures: List[str] = []
    probes = record.get("probes") if isinstance(record, Mapping) else None
    probe = probes.get("safety_veto") if isinstance(probes, Mapping) else None
    # Writer: tournament_eval.rollout attaches ``safety_veto.record()``.
    try:
        _validate_candidate_wrapper_probe(probe, expected_descriptor(arm))
    except (StrictPromotionArtifactError, TypeError, KeyError) as exc:
        failures.append(f"safety_veto probe is not arm {arm}'s veto: {exc}")
    if record.get("seed") != world_seed:
        failures.append("record seed differs from the planned world seed")
    mass = record.get("mass_integral")
    if not isinstance(mass, (int, float)) or isinstance(mass, bool) or mass != mass:
        failures.append("mass_integral is not a finite number")
    if smoke:  # legacy (profile-free) smoke path: no profile, identity or denominators
        return failures
    # Writer: rollout(profile=...) via EvaluationMetricsAccumulator and the roster.
    if record.get("evaluation_profile_digest") != dev_screen.PROFILE_DIGEST:
        failures.append("evaluation_profile_digest is not promotion-v2-watch-rect")
    row = {
        "world_seed": world_seed,
        "mix": mix,
        "slots": [{"member_sha256": sha} for sha in roster_hashes],
    }
    if record.get("world_identity") != _expected_world_identity(row):
        failures.append("world_identity does not match the materialized roster")
    denominators = record.get("denominators") or {}
    if denominators.get("scored_frames") != dev_screen.HORIZON:
        failures.append("denominators.scored_frames is not the H5000 horizon")
    counters = (probe or {}).get("counters") or {}
    # One veto decision per frame the hero acted (AISnake.update -> veto.apply).
    if counters.get("decisions") != denominators.get("decision_frames"):
        failures.append("veto decisions differ from denominators.decision_frames")
    return failures


def v6_identities_hold(diagnostics: Mapping[str, Any], counters: Mapping[str, Any]) -> bool:
    """The ``HeadAndFallbackCounters`` identities against the probe's v2 counters (reported).

    Probe mirror (``decisions``, ``kept_base = kept``, ``vetoes_applied``,
    ``fallback_no_spacious = no_spacious = fallback_decisions``), the per-layer splits
    (``head_risky_decisions = vetoes + kept_no_alternative``; ``fallback_searches +
    fallback_skipped_no_search = fallback_decisions``) and the link to the nested v5
    counters (``kept = v5.kept - head_vetoes_of_v5_kept``, ``vetoes_applied =
    v5.vetoes_applied + head_vetoes_of_v5_kept``, ``no_spacious = v5.no_spacious``).
    """

    def get(source: Mapping[str, Any], key: str) -> int:
        return int(source.get(key) or 0)

    v5 = diagnostics.get("v5") if isinstance(diagnostics.get("v5"), Mapping) else {}
    head_over_kept = get(diagnostics, "head_vetoes_of_v5_kept")
    return (
        counters.get("decisions") == get(diagnostics, "decisions") == get(v5, "decisions")
        and counters.get("kept_base") == get(diagnostics, "kept")
        and counters.get("vetoes_applied") == get(diagnostics, "vetoes_applied")
        and counters.get("fallback_no_spacious") == get(diagnostics, "no_spacious")
        and get(diagnostics, "no_spacious") == get(diagnostics, "fallback_decisions")
        and get(diagnostics, "head_risky_decisions")
        == get(diagnostics, "head_risky_vetoes")
        + get(diagnostics, "head_risky_kept_no_alternative")
        and get(diagnostics, "fallback_searches") + get(diagnostics, "fallback_skipped_no_search")
        == get(diagnostics, "fallback_decisions")
        and get(diagnostics, "kept") == get(v5, "kept") - head_over_kept
        and get(diagnostics, "vetoes_applied") == get(v5, "vetoes_applied") + head_over_kept
        and get(diagnostics, "no_spacious") == get(v5, "no_spacious")
    )


def check_entry(entry: Mapping[str, Any], smoke: bool) -> Dict[str, List[str]]:
    """Gating failures and reported warnings for one screen entry (dev_screen writer)."""
    failures: List[str] = []
    warnings: List[str] = []
    arm = entry.get("arm")
    if arm not in ARM_METHODS:
        return {"failures": [f"unknown arm {arm!r}"], "warnings": []}
    if entry.get("schema_version") != SCHEMA or entry.get("authority") != AUTHORITY:
        failures.append("entry schema/authority is not this screen's")
    if entry.get("screen") != SCREEN_ID or entry.get("hero_sha256") != dev_screen.CHAMPION[1]:
        failures.append("entry screen id or hero sha256 differs")
    if entry.get("safety_veto") is not True or entry.get("safety_veto_method") != ARM_METHODS[arm]:
        failures.append(f"entry veto flag/method is not arm {arm}'s")
    failures.extend(
        check_record(
            entry.get("record") or {},
            arm,
            world_seed=entry.get("world_seed"),
            mix=entry.get("mix"),
            roster_hashes=entry.get("roster_member_sha256s") or [],
            smoke=smoke,
        )
    )
    diagnostics = entry.get("veto_diagnostics")
    probe = ((entry.get("record") or {}).get("probes") or {}).get("safety_veto") or {}
    counters = probe.get("counters") or {}
    if arm in V6_ARMS:
        if not isinstance(diagnostics, Mapping) or "head_risky_vetoes" not in diagnostics:
            warnings.append(f"arm {arm} v6 counters missing")
        elif not v6_identities_hold(diagnostics, counters):
            warnings.append(f"arm {arm} v6 counters disagree with the probe's v2 counters")
    elif not isinstance(diagnostics, Mapping) or "boost_landing_vetoes" not in diagnostics:
        warnings.append(f"arm {arm} v5 counters missing")
    elif "head_risky_vetoes" in diagnostics:
        warnings.append(f"arm {arm} carries v6 diagnostics")
    elif not v5screen.probe_identities_hold(diagnostics, counters):
        warnings.append(f"arm {arm} v5 counters disagree with the probe's v2 counters")
    return {"failures": failures, "warnings": warnings}


def self_check(out: Path, smoke: bool) -> Dict[str, Any]:
    """Check every saved record of a finished screen directory."""
    intent = json.loads((out / "intent.json").read_text())
    planned = {int(seed) for seed in intent["worlds"]["seeds"]}
    failures: List[str] = []
    warnings: List[str] = []
    paths = sorted((out / "records").glob("*.json"))
    for path in paths:
        entry = json.loads(path.read_text())
        name = f"{entry.get('arm')}-{entry.get('mix')}-{entry.get('world_seed')}.json"
        result = check_entry(entry, smoke)
        if path.name != name:
            result["failures"].append("file name differs from arm-mix-seed")
        if entry.get("world_seed") not in planned:
            result["failures"].append("world seed is not in the intent's planned seeds")
        failures.extend(f"{path.name}: {message}" for message in result["failures"])
        warnings.extend(f"{path.name}: {message}" for message in result["warnings"])
    return {
        "records": len(paths),
        "failures": failures,
        "warnings": warnings,
        "passes": not failures and bool(paths),
    }


SUMMED_V6_FIELDS = (
    "decisions",
    "kept",
    "vetoes_applied",
    "no_spacious",
    "head_checks",
    "head_risky_decisions",
    "head_risky_vetoes",
    "head_risky_kept_no_alternative",
    "head_risk_waived_hero_wins",
    "head_vetoes_speed_switched",
    "head_vetoes_of_v5_kept",
    "fallback_decisions",
    "fallback_landing_switches",
    "fallback_escape_switches",
    "fallback_budget_unknown",
    "fallback_searches",
    "fallback_skipped_no_search",
    "fallback_search_nodes",
    "action_differs_from_v5",
    "apply_seconds_total",
    "fallback_seconds_total",
)
MAX_V6_FIELDS = ("apply_seconds_max", "fallback_seconds_max")


def _ratio(numerator: float, denominator: float) -> float | None:
    return numerator / denominator if denominator else None


def v6_layer_report(entries: Sequence[Mapping[str, Any]], arm: str = "B") -> Dict[str, Any]:
    """Reported only: v6 per-layer counters and per-decision cost for ``arm`` (B default).

    Writers: ``HeadAndFallbackCounters.to_dict`` via ``veto_diagnostics`` and dev_screen's
    ``wall_seconds``. ``*_seconds_*`` are wall-clock on the screen host; the counts are
    deterministic. Arm A's mean episode wall time is the reference.
    """
    rows: Dict[str, Dict[str, Any]] = {}
    a_seconds: Dict[str, List[float]] = {}

    def fresh() -> Dict[str, Any]:
        row: Dict[str, Any] = {"episodes": 0, "missing_diagnostics": 0, "wall_seconds": []}
        row.update({key: 0.0 for key in MAX_V6_FIELDS})
        row.update({key: 0 for key in SUMMED_V6_FIELDS})
        return row

    def add(row: Dict[str, Any], entry: Mapping[str, Any]) -> None:
        row["episodes"] += 1
        row["wall_seconds"].append(float(entry.get("wall_seconds", 0.0)))
        diag = entry.get("veto_diagnostics")
        if not isinstance(diag, Mapping) or "head_risky_vetoes" not in diag:
            row["missing_diagnostics"] += 1
            return
        for key in SUMMED_V6_FIELDS:
            row[key] += diag.get(key) or 0
        for key in MAX_V6_FIELDS:
            row[key] = max(row[key], float(diag.get(key) or 0.0))

    def finish(row: Dict[str, Any], a_wall: Sequence[float]) -> Dict[str, Any]:
        wall = row.pop("wall_seconds")
        decisions = row["decisions"]
        row["mean_apply_seconds_per_decision"] = _ratio(row["apply_seconds_total"], decisions)
        row["mean_fallback_seconds_per_fallback"] = _ratio(
            row["fallback_seconds_total"], row["fallback_decisions"]
        )
        row["head_veto_rate_per_decision"] = _ratio(row["head_risky_vetoes"], decisions)
        row["fallback_rate_per_decision"] = _ratio(row["fallback_decisions"], decisions)
        row["fallback_switch_rate_per_fallback"] = _ratio(
            row["fallback_landing_switches"] + row["fallback_escape_switches"],
            row["fallback_decisions"],
        )
        row["budget_unknown_rate_per_search"] = _ratio(
            row["fallback_budget_unknown"], row["fallback_searches"]
        )
        row["differs_from_v5_rate_per_decision"] = _ratio(row["action_differs_from_v5"], decisions)
        row["mean_episode_wall_seconds"] = _ratio(sum(wall), len(wall))
        row["mean_episode_wall_seconds_arm_A"] = _ratio(sum(a_wall), len(a_wall))
        return row

    total = fresh()
    for entry in entries:
        mix = str(entry.get("mix"))
        if entry.get("arm") == "A":
            a_seconds.setdefault(mix, []).append(float(entry.get("wall_seconds", 0.0)))
        if entry.get("arm") != arm:
            continue
        add(rows.setdefault(mix, fresh()), entry)
        add(total, entry)
    per_mix = {mix: finish(row, a_seconds.get(mix, [])) for mix, row in sorted(rows.items())}
    all_a = [s for values in a_seconds.values() for s in values]
    return {
        "arm": arm,
        "per_mix": per_mix,
        "total": finish(total, all_a) if per_mix else {},
        "gating": False,
    }


def death_cause_report(entries: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Reported only: hero death causes per arm and mix (writer: rollout ``probes``)."""
    table: Dict[str, Dict[str, Dict[str, int]]] = {}
    for entry in entries:
        probes = (entry.get("record") or {}).get("probes") or {}
        cause = str(probes.get("death_cause") or "survived")
        cell = table.setdefault(str(entry.get("arm")), {}).setdefault(str(entry.get("mix")), {})
        cell[cause] = cell.get(cause, 0) + 1
    return {
        "by_arm_and_mix": {
            arm: {mix: dict(sorted(c.items())) for mix, c in sorted(mixes.items())}
            for arm, mixes in sorted(table.items())
        },
        "gating": False,
    }


def build_receipt(out: Path, check: Mapping[str, Any]) -> Dict[str, Any]:
    """Tier-1 ``receipt.json`` content from a finished screen directory."""
    intent_path, summary_path = out / "intent.json", out / "summary.json"
    intent = json.loads(intent_path.read_text())
    summary = json.loads(summary_path.read_text())
    records = []
    frames = 0
    seconds = 0.0
    by_arm: Dict[str, int] = {}
    entries: List[Mapping[str, Any]] = []
    for path in sorted((out / "records").glob("*.json")):
        entry = json.loads(path.read_text())
        entries.append(entry)
        records.append({"path": f"records/{path.name}", "sha256": dev_screen.sha256_file(path)})
        by_arm[entry["arm"]] = by_arm.get(entry["arm"], 0) + 1
        denominators = entry["record"].get("denominators") or {}
        frames += int(denominators.get("scored_frames", intent.get("smoke_frames") or 0))
        seconds += float(entry["wall_seconds"])
    pooled = summary.get("pooled_informational", {})
    decision = summary["decision"] if check["passes"] else "INVALID_SELF_CHECK_FAILED"
    return {
        "schema_version": RECEIPT_SCHEMA,
        "screen_id": SCREEN_ID,
        "authority": AUTHORITY,
        "label": "screen (non-authoritative)",
        "promotion_authorized": False,
        "intent_sha256": dev_screen.sha256_file(intent_path),
        "summary_sha256": dev_screen.sha256_file(summary_path),
        "records": records,
        "counts": {
            "episodes": len(records),
            "planned_episodes": summary.get("planned_episodes"),
            "by_arm": dict(sorted(by_arm.items())),
            "scored_frames": frames,
            "self_check_failures": len(check["failures"]),
        },
        "wall": {
            "started_utc": intent.get("started_utc"),
            "finished_utc": summary.get("finished_utc"),
            "episode_seconds_total": seconds,
            "stopped_reason": summary.get("stopped_reason"),
            "max_wall_seconds": MAX_WALL_SECONDS,
        },
        "primary": {
            "metric": "paired mass_integral B - A per world (B = v6, A = v5)",
            "holm_one_sided": summary.get("holm_primary"),
            "per_mix_one_sided": {
                mix: row["primary_mass_integral"]["one_sided_test"]
                for mix, row in summary.get("per_mix", {}).items()
            },
            "stratified_mean_of_means_90": pooled.get("stratified_mean_of_means"),
        },
        "decision": decision,
        "screen_decision_before_self_check": summary["decision"],
        "self_check": dict(check),
        "reported": {
            "determinism_control": summary.get("determinism_control"),
            "replay_control": summary.get("replay_control"),
            "v6_layers": v6_layer_report(entries, "B"),
            "v5_landing_arm_A": v5screen.v5_landing_report(entries, "A"),
            "death_causes": death_cause_report(entries),
        },
    }


def main(argv: Sequence[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = dev_screen.parse_args(argv)
    smoke = args.smoke_frames is not None
    design = (args.worlds_per_mix, args.determinism_worlds)
    if not smoke and design != dev_screen.PREREGISTERED_DESIGN:
        # Refuse before any world is played: a smaller look at the real worlds
        # followed by the pre-registered run would reach a decision after a peek.
        print(
            f"refusing a non-smoke run of size {design}; the pre-registered design is "
            f"{dev_screen.PREREGISTERED_DESIGN} (worlds per mix, determinism worlds)",
            file=sys.stderr,
        )
        return 2
    code = dev_screen.main(argv, spec=SMOKE_SPEC if smoke else SPEC)
    if code != 0:
        return code
    out = Path(args.out).resolve()
    check = self_check(out, smoke)
    receipt = build_receipt(out, check)
    receipt["written_utc"] = datetime.now(timezone.utc).isoformat()
    dev_screen.write_new_json(out / "receipt.json", dev_screen.json_safe(receipt))
    print(json.dumps({"decision": receipt["decision"], "receipt": str(out / "receipt.json")}))
    return 0 if check["passes"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
