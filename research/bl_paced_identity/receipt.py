"""Create-only trust-chain receipts and fail-closed L3/L4 identity admission.

Qualification producers attach bindings from their prospectively frozen manifests.
This module only verifies those already-bound values; it never manufactures a
binding for an incomplete corpus.
"""

from __future__ import annotations

import hashlib
import platform
from pathlib import Path
from typing import Any

from .clock_seam import clock_seam_passed
from .events import CandidateIdentity, EventValidationError, sha256
from .record import export_json

SCHEMA = "bl-paced-identity-receipt/v1"
JOURNAL_SCHEMA = "bl-paced-identity-journal/v1"
SEMANTIC_SCHEMA = "bl-paced-identity-semantic/v1"
REPLAY_SCHEMA = "bl-paced-identity-replay/v1"
PACED_SCHEMA = "bl-paced-identity-paced/v1"
_HEX = set("0123456789abcdef")
_E13_KEYS = {
    "engine_identity",
    "policy_first_divergence",
    "exact_resume_real_export",
    "mac_colab_transfer_interval",
}
FAMILY_POLICY = {
    "alpha": 0.05,
    "method": "Holm",
    "freeze": "before_first_strict_bank_open",
    "allowed_slots": ["bare_R", "assisted_B8", "planner_over_R"],
    "reserved_slot": "planner_over_R",
    "include": "only_artifacts_with_frozen_nominees_at_freeze",
    "late_additions": False,
    "ties": "bare_wins_unresolved_ties",
    "composite_p": "max_required_clause_p_values",
}


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and set(value) <= _HEX


def _hash_map(value: Any, name: str, problems: list[str]) -> dict[str, str] | None:
    if not isinstance(value, dict) or not value:
        problems.append(f"{name}_missing")
        return None
    if any(
        not isinstance(key, str) or not key or not _sha(digest) for key, digest in value.items()
    ):
        problems.append(f"{name}_invalid")
        return None
    return dict(value)


def _seal_ok(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    hashes = value.get("chunk_sha256")
    return (
        value.get("schema") == JOURNAL_SCHEMA
        and value.get("status") == "SEALED"
        and type(value.get("event_count")) is int
        and value["event_count"] >= 0
        and type(value.get("chunk_count")) is int
        and value["chunk_count"] >= 0
        and isinstance(hashes, list)
        and len(hashes) == value["chunk_count"]
        and all(_sha(item) for item in hashes)
        and _sha(value.get("header_sha256"))
        and _sha(value.get("terminal_chunk_sha256"))
    )


def freeze_family(nominees: dict[str, str], *, strict_bank_open: bool = False) -> dict[str, Any]:
    """Freeze only present nominees; a reserved absent slot cannot be added later."""
    if strict_bank_open:
        raise ValueError("family must be frozen before any strict bank is opened")
    if not nominees or set(nominees) - set(FAMILY_POLICY["allowed_slots"]):
        raise ValueError("unknown or empty final claim family")
    if any(not _sha(value) for value in nominees.values()):
        raise ValueError("nominees must bind artifact sha256 identities")
    count = len(nominees)
    result = {
        "schema": "bl-paced-identity-family/v1",
        "policy": FAMILY_POLICY,
        "nominees": dict(sorted(nominees.items())),
        "holm_thresholds": [0.05 / (count - index) for index in range(count)],
        "absent_reserved_slot_excluded": "planner_over_R" not in nominees,
    }
    return {**result, "family_sha256": sha256(result)}


def check_paced(
    runs: list[dict[str, Any]],
    *,
    expected_run_ids: list[str],
    candidate_sha256: str,
    frozen_order_sha256: str,
) -> dict[str, Any]:
    """Validate the complete L3 inventory before any efficacy unblinding."""
    result: dict[str, Any] = {
        "schema": PACED_SCHEMA,
        "verdict": "INCOMPLETE",
        "fixture_count": len(runs),
        "expected_run_ids": list(expected_run_ids),
        "candidate_sha256": candidate_sha256,
        "frozen_order_sha256": frozen_order_sha256,
        "first_divergence": None,
    }
    divergence = next(
        (
            row.get("first_divergence")
            for row in runs
            if isinstance(row, dict) and row.get("first_divergence")
        ),
        None,
    )
    contradictory = any(
        isinstance(row, dict)
        and row.get("replay_verdict") == "PASS"
        and row.get("first_divergence")
        for row in runs
    )
    if (
        divergence is not None
        or contradictory
        or any(
            isinstance(row, dict) and row.get("replay_verdict") == "INVALID_IDENTITY"
            for row in runs
        )
    ):
        return {
            **result,
            "verdict": "INVALID_IDENTITY",
            "first_divergence": divergence or {"reason": "pass_with_divergence"},
            "reason": "whole_evaluation_suspended",
        }
    problems: list[str] = []
    if not _sha(candidate_sha256):
        problems.append("candidate_sha256_invalid")
    if (
        not expected_run_ids
        or any(not isinstance(run_id, str) or not run_id for run_id in expected_run_ids)
        or len(set(expected_run_ids)) != len(expected_run_ids)
    ):
        problems.append("expected_run_ids_invalid")
    if sha256(expected_run_ids) != frozen_order_sha256:
        problems.append("frozen_order_digest_invalid")
    if [row.get("run_id") if isinstance(row, dict) else None for row in runs] != expected_run_ids:
        problems.append("missing_or_reordered_attempts")
    world: dict[str, dict[str, set[str]]] = {mix: {} for mix in ("scripted", "frozen", "mixed")}
    sessions: dict[str, set[str]] = {}
    pair_arms: set[tuple[Any, ...]] = set()
    pair_profiles: dict[tuple[str, str, str], tuple[Any, ...]] = {}
    world_identities: dict[str, dict[str, str]] = {mix: {} for mix in world}
    loads: set[str] = set()
    natural = 0
    if len(runs) != 108:
        problems.append("expected_exactly_108_registered_arm_runs")
    for row in runs:
        if not isinstance(row, dict):
            problems.append("run_row_invalid")
            continue
        role = row.get("arm_role")
        if role not in {"candidate", "bare"}:
            problems.append("arm_role_invalid")
        if row.get("synthetic") is not False or row.get("platform") != "Darwin":
            problems.append("not_fresh_darwin_run")
        if row.get("candidate_sha256") != candidate_sha256:
            problems.append("run_candidate_binding_invalid")
        if row.get("chronological_order_sha256") != frozen_order_sha256:
            problems.append("run_chronological_order_mismatch")
        if row.get("replay_verdict") != "PASS" or row.get("recording_enabled") is not True:
            problems.append("trace_replay_not_verified")
        hashes_ok = all(
            _sha(row.get(key))
            for key in (
                "trace_sha256",
                "replay_sha256",
                "trace_identity_sha256",
                "replay_identity_sha256",
                "load_envelope_sha256",
            )
        )
        if not hashes_ok:
            problems.append("run_hash_missing")
        elif (
            row["trace_identity_sha256"] != candidate_sha256
            or row["replay_identity_sha256"] != candidate_sha256
        ):
            problems.append("trace_replay_identity_mismatch")
        else:
            loads.add(row["load_envelope_sha256"])
        if not _seal_ok(row.get("terminal_seal")):
            problems.append("terminal_seal_invalid")
        pair = (
            row.get("world_pair_id")
            if row.get("run_kind") == "world"
            else row.get("session_pair_id")
        )
        pair_key = (row.get("run_kind"), row.get("mix", "watch6"), pair)
        arm_key = (*pair_key, role)
        if arm_key in pair_arms:
            problems.append("duplicate_pair_arm_attempt")
        pair_arms.add(arm_key)
        profile = tuple(row.get(key) for key in ("world_sha256", "roster_sha256", "profile_sha256"))
        if not all(_sha(value) for value in profile):
            problems.append("paired_world_roster_profile_missing")
        if pair_key in pair_profiles and pair_profiles[pair_key] != profile:
            problems.append("paired_world_roster_profile_mismatch")
        pair_profiles[pair_key] = profile
        if row.get("fps") != 24:
            problems.append("paced_fps_must_be_24")
        if row.get("run_kind") == "world":
            if row.get("horizon") != 5000:
                problems.append("world_horizon_must_be_5000")
            mix, pair = row.get("mix"), row.get("world_pair_id")
            if mix not in world or not isinstance(pair, str) or not pair:
                problems.append("world_inventory_invalid")
            elif role in {"candidate", "bare"}:
                world[mix].setdefault(pair, set()).add(role)
                world_identities[mix][pair] = row.get("world_sha256")
        elif row.get("run_kind") == "session":
            pair = row.get("session_pair_id")
            if (
                not isinstance(pair, str)
                or not pair
                or type(row.get("duration_seconds")) is not int
                or row["duration_seconds"] < 600
                or row.get("snakes") != 6
            ):
                problems.append("session_inventory_invalid")
            elif role in {"candidate", "bare"}:
                sessions.setdefault(pair, set()).add(role)
        else:
            problems.append("run_kind_invalid")
        intervention = row.get("natural_interventions", 0)
        if type(intervention) is not int or intervention < 0:
            problems.append("natural_interventions_invalid")
        elif role == "candidate":
            natural += intervention
    if len(loads) != 1:
        problems.append("load_envelope_not_identical")
    if any(len(set(values.values())) != len(values) for values in world_identities.values()):
        problems.append("duplicate_world_identity_under_distinct_pairs")
    if any(
        len(pairs) != 16 or any(arms != {"candidate", "bare"} for arms in pairs.values())
        for pairs in world.values()
    ):
        problems.append("world_inventory_not_16_paired_per_mix")
    if len(sessions) != 6 or any(arms != {"candidate", "bare"} for arms in sessions.values()):
        problems.append("session_inventory_not_6_paired")
    if problems:
        return {**result, "reason": ";".join(sorted(set(problems)))}
    if not natural:
        return {**result, "verdict": "INCONCLUSIVE", "reason": "no_natural_intervention"}
    return {**result, "verdict": "PASS", "natural_interventions": natural}


def evidence_ready(
    *,
    identity: dict[str, Any],
    semantic: dict[str, Any],
    replay: dict[str, Any],
    paced: dict[str, Any],
    source_hashes: dict[str, str],
    input_hashes: dict[str, str],
    reference_hashes: dict[str, str],
    commands: list[str],
    journal_seals: list[dict[str, Any]],
    e13_receipts: dict[str, str] | None,
    planner_over_net: bool,
    clock_seam: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return every admission defect; missing evidence is INCOMPLETE, never a pass."""
    problems: list[str] = []
    try:
        candidate_sha = sha256(CandidateIdentity(**identity).as_dict())
    except (EventValidationError, TypeError, ValueError):
        candidate_sha = None
        problems.append("candidate_identity_invalid")
    source = _hash_map(source_hashes, "source_hashes", problems)
    if source is None or not clock_seam_passed(
        clock_seam, sha256(source), candidate_sha256=candidate_sha
    ):
        problems.append("clock_seam_not_verified")
    inputs = _hash_map(input_hashes, "input_hashes", problems)
    reference = _hash_map(reference_hashes, "reference_hashes", problems)
    if (
        not isinstance(commands, list)
        or not commands
        or any(not isinstance(value, str) or not value for value in commands)
    ):
        problems.append("commands_invalid")
    if (
        not isinstance(journal_seals, list)
        or not journal_seals
        or not all(_seal_ok(value) for value in journal_seals)
    ):
        problems.append("journal_seals_invalid")
    e13 = e13_receipts or {}
    if planner_over_net and (
        set(e13) != _E13_KEYS or any(not _sha(value) for value in e13.values())
    ):
        problems.append("e13_receipts_invalid")
    if (
        semantic.get("schema") != SEMANTIC_SCHEMA
        or semantic.get("qualification") != "SEMANTIC_QUALIFIED"
        or semantic.get("verdict") != "PASS"
    ):
        problems.append("semantic_not_qualified")
    if not _sha(semantic.get("bank114_sha256")):
        problems.append("semantic_bank114_proof_missing")
    if (
        replay.get("schema") != REPLAY_SCHEMA
        or replay.get("qualification") != "REPLAY_QUALIFIED"
        or replay.get("verdict") != "PASS"
    ):
        problems.append("replay_not_qualified")
    if paced.get("schema") != PACED_SCHEMA or paced.get("verdict") != "PASS":
        problems.append("paced_not_qualified")
    if platform.system() != "Darwin":
        problems.append("qualification_requires_darwin")
    binding = None
    if (
        candidate_sha is not None
        and source is not None
        and inputs is not None
        and reference is not None
    ):
        binding = sha256(
            {
                "candidate_sha256": candidate_sha,
                "source_sha256": sha256(source),
                "input_sha256": sha256(inputs),
                "reference_sha256": sha256(reference),
            }
        )
        for name, oracle in (("semantic", semantic), ("replay", replay)):
            if (
                oracle.get("candidate_sha256") != candidate_sha
                or oracle.get("binding_sha256") != binding
            ):
                problems.append(f"{name}_binding_mismatch")
        if paced.get("candidate_sha256") != candidate_sha or paced.get("binding_sha256") != binding:
            problems.append("paced_binding_mismatch")
    base_problems = [
        problem
        for problem in problems
        if problem not in {"paced_not_qualified", "paced_binding_mismatch"}
    ]
    return {
        "ready": not problems,
        "base_ready": not base_problems,
        "full_ready": not problems,
        "problems": sorted(set(problems)),
        "base_problems": sorted(set(base_problems)),
        "candidate_sha256": candidate_sha,
        "binding_sha256": binding,
        "source_binding_sha256": sha256(source) if source is not None else None,
        "input_binding_sha256": sha256(inputs) if inputs is not None else None,
        "reference_binding_sha256": sha256(reference) if reference is not None else None,
    }


def make_receipt(
    *,
    identity: dict[str, Any],
    semantic: dict[str, Any],
    replay: dict[str, Any],
    paced: dict[str, Any],
    source_hashes: dict[str, str],
    input_hashes: dict[str, str],
    reference_hashes: dict[str, str],
    commands: list[str],
    journal_seals: list[dict[str, Any]],
    forecast: dict[str, Any],
    e13_receipts: dict[str, str] | None = None,
    planner_over_net: bool = False,
    clock_seam: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Join three frozen oracle receipts without granting release authority."""
    oracles = {"semantic": semantic, "replay": replay, "paced": paced}
    first = next(
        (
            value.get("first_divergence")
            for value in oracles.values()
            if value.get("first_divergence")
        ),
        None,
    )
    contradictory = any(
        value.get("verdict") == "PASS" and value.get("first_divergence")
        for value in oracles.values()
    )
    readiness = evidence_ready(
        identity=identity,
        semantic=semantic,
        replay=replay,
        paced=paced,
        source_hashes=source_hashes,
        input_hashes=input_hashes,
        reference_hashes=reference_hashes,
        commands=commands,
        journal_seals=journal_seals,
        e13_receipts=e13_receipts,
        planner_over_net=planner_over_net,
        clock_seam=clock_seam,
    )
    invalid = (
        first is not None
        or contradictory
        or any(value.get("verdict") == "INVALID_IDENTITY" for value in oracles.values())
    )
    complete = readiness["full_ready"] and not invalid
    l3_ready = readiness["base_ready"] and not invalid
    return {
        "schema": SCHEMA,
        "verdict": "INVALID_IDENTITY" if invalid else "PASS" if complete else "INCOMPLETE",
        "l3_identity_prerequisites": "PASS" if l3_ready else "BLOCKED",
        "identity": identity,
        "candidate_sha256": readiness["candidate_sha256"],
        "binding_sha256": readiness["binding_sha256"],
        "source_binding_sha256": readiness["source_binding_sha256"],
        "input_binding_sha256": readiness["input_binding_sha256"],
        "reference_binding_sha256": readiness["reference_binding_sha256"],
        "evidence_problems": readiness["problems"],
        "oracles": oracles,
        "clock_seam": clock_seam,
        "oracle_event_counts": {
            name: {
                "expected": value.get("expected_event_count"),
                "observed": value.get(
                    "observed_event_count", value.get("counts", {}).get("events")
                ),
            }
            for name, value in oracles.items()
        },
        "first_divergence": first
        or ({"reason": "pass_with_divergence"} if contradictory else None),
        "source_sha256": source_hashes,
        "input_sha256": input_hashes,
        "reference_sha256": reference_hashes,
        "platform": {"system": platform.system(), "machine": platform.machine()},
        "libraries": {"python": platform.python_version()},
        "journal_seals": journal_seals,
        "forecast": forecast,
        "commands": commands,
        "family_policy": FAMILY_POLICY,
        "planner_over_net": planner_over_net,
        "e13_receipts": e13_receipts or {},
        "efficacy_unblinding": "IDENTITY_CHECKED" if complete else "BLOCKED",
        "serving_authority": "OWNER_ONLY",
        "repair_policy": {
            "verifier_only": "independent_audit_and_prospective_protocol_required",
            "action_affecting": "new_artifact_and_fresh_registered_banks",
            "divergence": "suspend_whole_evaluation_preserve_all_attempts",
        },
    }


def write_receipt(path: Path, **kwargs: Any) -> str:
    """Export a create-only receipt with digest and terminal completion sidecars."""
    return export_json(path, make_receipt(**kwargs))


def file_hashes(paths: list[Path]) -> dict[str, str]:
    """Bind explicitly named source/input files, never recursively scan artifacts."""
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}
