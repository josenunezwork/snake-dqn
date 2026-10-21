"""Semantic oracle for the paced served-search controller.

This module deliberately knows nothing about scheduling, clocks, or the paced
controller.  It compares a candidate runtime with an independently constructed
serial reference runtime at a root and at each of the six possible hero actions.
The production fixture bank is intentionally supplied by the calling package;
the small fixture in :mod:`semantic_fixture` is only a Mac preflight.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Protocol

from research.bl_paced_identity.events import canonical_bytes
from research.bl_paced_identity.events import sha256 as canonical_sha256

SEMANTIC_SCHEMA = "bl-paced-identity-semantic/v1"
NUM_ACTIONS = 6
EXPECTED_PRODUCTION_FIXTURE_COUNT = 114
REQUIRED_PRODUCTION_FIXTURES = {
    "mix_length_fixtures": 18,
    "pinned_roots": 96,
    "horizon": 300,
    "actions_per_root": NUM_ACTIONS,
    "requires_frozen_opponent_carry": True,
}


class SemanticRuntime(Protocol):
    """Opaque runtime boundary used by the semantic oracle.

    ``clone`` must return an isolated branch with complete v3 state, including
    opponent carry and RNG.  ``snapshot`` and ``observation`` must return
    canonical bytes.  The serial reference must live in a separately pinned
    source module and must not import candidate or modified planner helpers;
    its source hash and an explicit attestation are receipt inputs.  The
    attestation records the contract but cannot itself prove non-sharing.
    """

    def clone(self) -> "SemanticRuntime": ...

    def snapshot(self) -> bytes: ...

    def observation(self) -> bytes: ...

    def rank_actions(self) -> Mapping[str, Any]: ...

    def step(self, hero_action: int | None) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class SemanticFixture:
    """One root with separately constructed candidate and reference runtimes."""

    fixture_id: str
    candidate_factory: Callable[[], SemanticRuntime]
    reference_factory: Callable[[], SemanticRuntime]
    horizon: int = 300
    metadata: Mapping[str, Any] | None = None


def _production_attestation(fixtures: list[SemanticFixture]) -> tuple[bool, list[str]]:
    """Check the evidence minimum before a receipt can qualify production semantics."""
    problems: list[str] = []
    mix = [row for row in fixtures if (row.metadata or {}).get("fixture_kind") == "mix_length"]
    roots = [row for row in fixtures if (row.metadata or {}).get("fixture_kind") == "pinned_root"]
    root_pins = [str((row.metadata or {}).get("root_pin", "")) for row in roots]
    fixture_ids = [row.fixture_id for row in fixtures]
    if len(mix) != REQUIRED_PRODUCTION_FIXTURES["mix_length_fixtures"]:
        problems.append("requires exactly 18 mix_length fixtures")
    if len(roots) != REQUIRED_PRODUCTION_FIXTURES["pinned_roots"] or len(set(root_pins)) != 96:
        problems.append("requires 96 unique pinned_root fixtures")
    if len(set(fixture_ids)) != len(fixture_ids):
        problems.append("fixture IDs must be unique")
    mixes: dict[str, set[str]] = {}
    for row in mix:
        metadata = row.metadata or {}
        mix_name = metadata.get("mix")
        stratum = metadata.get("length_stratum")
        if (
            not isinstance(mix_name, str)
            or not mix_name
            or not isinstance(stratum, str)
            or not stratum
        ):
            problems.append(f"{row.fixture_id}: mix_length fixture needs mix and length_stratum")
            continue
        mixes.setdefault(mix_name, set()).add(stratum)
    if len(mixes) != 3 or any(len(strata) != 6 for strata in mixes.values()):
        problems.append("requires exactly 3 mixes x 6 declared unique length strata")
    for row in fixtures:
        metadata = row.metadata or {}
        if row.horizon < REQUIRED_PRODUCTION_FIXTURES["horizon"]:
            problems.append(f"{row.fixture_id}: horizon below H300")
        for key in (
            "candidate_source_sha256",
            "frozen_reference_source_sha256",
            "fixture_input_sha256",
        ):
            value = metadata.get(key)
            if not isinstance(value, str) or len(value) != 64:
                problems.append(f"{row.fixture_id}: missing {key}")
        if metadata.get("frozen_opponent_carry") is not True:
            problems.append(f"{row.fixture_id}: missing frozen opponent carry")
        if metadata.get("unshared_modified_helpers") is not True:
            problems.append(f"{row.fixture_id}: unshared planner-helper attestation missing")
        if metadata.get("ranking_mode") == "test-only-score-free":
            problems.append(f"{row.fixture_id}: test-only ranking cannot qualify production")
        contract = metadata.get("independent_reference_contract")
        if not isinstance(contract, str) or not contract:
            problems.append(f"{row.fixture_id}: independent reference contract missing")
        if metadata.get("fixture_kind") == "pinned_root":
            pin_sha = metadata.get("pinned_root_sha256")
            if not isinstance(pin_sha, str) or len(pin_sha) != 64:
                problems.append(f"{row.fixture_id}: missing pinned_root_sha256")
        if (
            metadata.get("full_v3") is not True
            or metadata.get("digest_schema") != "t34-shadow-full-digest/v3"
        ):
            problems.append(f"{row.fixture_id}: full v3 state digest attestation missing")
    return not problems, sorted(set(problems))


def _canonical(value: Any) -> bytes:
    """Encode report values with stable JSON framing, never Python ``repr``."""
    return canonical_bytes(value)


def _digest(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _optional_sha(value: str | None, field: str) -> str | None:
    """Validate a prospective external binding; it is never derived from outcomes."""
    if value is None:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(char not in "0123456789abcdef" for char in value)
    ):
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _bank114_sha256(fixtures: list[SemanticFixture]) -> str:
    """Hash only frozen fixture identity material, in fixture-ID order."""
    rows = [
        {
            "fixture_id": fixture.fixture_id,
            "horizon": fixture.horizon,
            "root_pin": (fixture.metadata or {}).get("root_pin"),
            "fixture_input_sha256": (fixture.metadata or {}).get("fixture_input_sha256"),
        }
        for fixture in sorted(fixtures, key=lambda row: row.fixture_id)
    ]
    return canonical_sha256({"schema": "bl-paced-identity-bank114/v1", "fixtures": rows})


def _event_bytes(event: Mapping[str, Any]) -> bytes:
    """Normalize runtime event values to a canonical, JSON-safe byte payload."""
    normalized: dict[str, Any] = {}
    for key, value in sorted(event.items()):
        if hasattr(value, "tolist"):
            value = value.tolist()
        elif isinstance(value, bytes):
            value = {"sha256": _digest(value), "bytes": len(value)}
        normalized[str(key)] = value
    return _canonical(normalized)


def _first_difference(candidate: bytes, reference: bytes) -> int | None:
    """Return the byte offset of the first mismatch, if any."""
    if candidate == reference:
        return None
    limit = min(len(candidate), len(reference))
    for index in range(limit):
        if candidate[index] != reference[index]:
            return index
    return limit


def _compare(
    *,
    fixture_id: str,
    phase: str,
    action: int | None,
    field: str,
    candidate: bytes,
    reference: bytes,
) -> dict[str, Any] | None:
    offset = _first_difference(candidate, reference)
    if offset is None:
        return None
    return {
        "fixture_id": fixture_id,
        "phase": phase,
        "action": action,
        "field": field,
        "candidate_sha256": _digest(candidate),
        "reference_sha256": _digest(reference),
        "first_byte_offset": offset,
        "candidate_bytes": len(candidate),
        "reference_bytes": len(reference),
    }


def _branch_impl(
    fixture: SemanticFixture, action: int, candidate: SemanticRuntime, reference: SemanticRuntime
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Run one action branch and compare every post-step semantic output."""
    candidate_before = candidate.snapshot()
    reference_before = reference.snapshot()
    carried_mass = 0
    alive_transitions = 0
    event_hashes: list[dict[str, str]] = []
    for step in range(fixture.horizon):
        selected_action = action if step == 0 else None
        candidate_event = candidate.step(selected_action)
        reference_event = reference.step(selected_action)
        for side, event in (("candidate", candidate_event), ("reference", reference_event)):
            alive = event.get("hero_alive")
            mass = event.get("integer_mass_sum")
            if type(alive) is not bool or type(mass) is not int:
                return (
                    {
                        "fixture_id": fixture.fixture_id,
                        "phase": f"action-step-{step}",
                        "action": action,
                        "field": "invalid_event_types",
                        "side": side,
                        "hero_alive_type": type(alive).__name__,
                        "integer_mass_sum_type": type(mass).__name__,
                    },
                    {
                        "steps": step + 1,
                        "integer_mass_sum": carried_mass,
                        "alive_transitions": alive_transitions,
                    },
                )
        carried_mass += int(candidate_event["integer_mass_sum"])
        alive_transitions += int(candidate_event["hero_alive"])
        for field, left, right in (
            ("event", _event_bytes(candidate_event), _event_bytes(reference_event)),
            ("observation", candidate.observation(), reference.observation()),
            ("snapshot", candidate.snapshot(), reference.snapshot()),
        ):
            divergence = _compare(
                fixture_id=fixture.fixture_id,
                phase=f"action-step-{step}",
                action=action,
                field=field,
                candidate=left,
                reference=right,
            )
            if divergence is not None:
                return divergence, {"steps": step + 1, "integer_mass_sum": carried_mass}
        event_hashes.append({"event": _digest(_event_bytes(candidate_event))})
        if bool(candidate_event.get("terminal", False)):
            break
    return None, {
        "steps": len(event_hashes),
        "integer_mass_sum": carried_mass,
        "alive_transitions": alive_transitions,
        "entry_snapshot_sha256": _digest(candidate_before),
        "reference_entry_snapshot_sha256": _digest(reference_before),
        "terminal_event_sha256": event_hashes[-1]["event"] if event_hashes else None,
        "final_snapshot_sha256": _digest(candidate.snapshot()),
        "survives": bool(not candidate_event.get("terminal", True)) if event_hashes else False,
    }


def _branch(
    fixture: SemanticFixture, action: int, candidate: SemanticRuntime, reference: SemanticRuntime
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Seal runtime failures as a first-divergence receipt instead of raising."""
    try:
        return _branch_impl(fixture, action, candidate, reference)
    except Exception as exc:
        return (
            {
                "fixture_id": fixture.fixture_id,
                "phase": "branch-runtime",
                "action": action,
                "field": "runtime_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            },
            {"steps": 0, "integer_mass_sum": 0},
        )


def _branch_isolation_impl(
    fixture: SemanticFixture, action: int, factory: Callable[[], SemanticRuntime], side: str
) -> dict[str, Any] | None:
    """Prove a branch cannot mutate its parent or untouched sibling on either arm."""
    parent = factory()
    parent_before = parent.snapshot()
    left, sibling = parent.clone(), parent.clone()
    before = sibling.snapshot()
    for step in range(fixture.horizon):
        event = left.step(action if step == 0 else None)
        parent_drift = _compare(
            fixture_id=fixture.fixture_id,
            phase=f"parent-isolation-{side}-step-{step}",
            action=action,
            field="parent_snapshot",
            candidate=parent.snapshot(),
            reference=parent_before,
        )
        if parent_drift is not None:
            return parent_drift
        sibling_drift = _compare(
            fixture_id=fixture.fixture_id,
            phase=f"sibling-isolation-{side}-step-{step}",
            action=action,
            field="sibling_snapshot",
            candidate=sibling.snapshot(),
            reference=before,
        )
        if sibling_drift is not None:
            return sibling_drift
        if bool(event.get("terminal", False)):
            break
    return None


def _branch_isolation(
    fixture: SemanticFixture, action: int, factory: Callable[[], SemanticRuntime], side: str
) -> dict[str, Any] | None:
    """Seal clone/isolation runtime errors as identity failures."""
    try:
        return _branch_isolation_impl(fixture, action, factory, side)
    except Exception as exc:
        return {
            "fixture_id": fixture.fixture_id,
            "phase": f"branch-isolation-{side}",
            "action": action,
            "field": "runtime_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }


def _outcome_ranking(
    actions: list[Mapping[str, Any]], root: Mapping[str, Any]
) -> Mapping[str, Any]:
    """Apply the governed T3 key: alive transitions, B8, root Q, then index."""
    required = {"ranking", "winner", "tie_rule", "bare_action", "q_values"}
    missing = sorted(required.difference(root))
    if missing:
        raise ValueError(f"rank_actions missing required fields: {','.join(missing)}")
    bare = root["bare_action"]
    q_values = root["q_values"]
    if type(bare) is not int or not 0 <= bare < NUM_ACTIONS:
        raise ValueError("rank_actions bare_action must be an action index")
    if not isinstance(q_values, list) or len(q_values) != NUM_ACTIONS:
        raise ValueError("rank_actions q_values must hold six values")
    if any(type(value) not in (int, float) or not math.isfinite(value) for value in q_values):
        raise ValueError("rank_actions q_values must be finite numeric values")
    ranking = sorted(
        actions,
        key=lambda row: (
            -int(row["alive_transitions"]),
            int(row["action"]) != bare,
            -float(q_values[int(row["action"])]),
            int(row["action"]),
        ),
    )
    return {
        "ranking": [int(row["action"]) for row in ranking],
        "winner": int(ranking[0]["action"]),
        "tie_rule": "alive_transitions_then_b8_then_root_q_then_action_index",
        "bare_action": bare,
        "q_values": q_values,
    }


def check_semantics(
    fixtures: Iterable[SemanticFixture],
    candidate: str = "candidate",
    reference: str = "serial-reference",
    *,
    candidate_sha256: str | None = None,
    binding_sha256: str | None = None,
) -> dict[str, Any]:
    """Compare six-action branches for each fixture and return a sealed-style receipt.

    The function stops at the first divergence.  A caller must treat a FAIL as
    ``INVALID_IDENTITY`` for the entire affected evaluation, rather than discard
    the offending root.  This check is semantic only; it does not qualify
    availability replay or paced operation.
    """
    candidate_sha256 = _optional_sha(candidate_sha256, "candidate_sha256")
    binding_sha256 = _optional_sha(binding_sha256, "binding_sha256")
    fixture_rows = list(fixtures)
    production_ready, production_problems = _production_attestation(fixture_rows)
    digest_schemas = {
        str((row.metadata or {}).get("digest_schema", "unknown")) for row in fixture_rows
    }
    state_digest_schema = next(iter(digest_schemas)) if len(digest_schemas) == 1 else "mixed"
    receipt: dict[str, Any] = {
        "schema": SEMANTIC_SCHEMA,
        "state_digest_schema": state_digest_schema,
        "candidate": candidate,
        "reference": reference,
        "required_production_fixtures": REQUIRED_PRODUCTION_FIXTURES,
        "expected_production_fixture_count": EXPECTED_PRODUCTION_FIXTURE_COUNT,
        "fixture_ids": [row.fixture_id for row in fixture_rows],
        "fixture_count": len(fixture_rows),
        "actions_per_fixture": NUM_ACTIONS,
        "verdict": "PASS" if fixture_rows else "INCOMPLETE",
        "qualification": "SEMANTIC_QUALIFIED" if production_ready else "PRECHECK_ONLY",
        "overall_verdict": "PASS" if production_ready else "INCOMPLETE",
        "production_attestation_problems": production_problems,
        "bank114_sha256": _bank114_sha256(fixture_rows) if production_ready else None,
        # These are frozen external producer inputs.  A joined receipt must
        # match both fields across oracle/replay/paced evidence; neither comes
        # from branch actions, scores, or any observed outcome.
        "candidate_sha256": candidate_sha256,
        "binding_sha256": binding_sha256,
        "fixtures": [],
        "first_divergence": None,
    }
    for fixture in fixture_rows:
        if fixture.horizon < 1:
            receipt.update(
                verdict="INCOMPLETE",
                overall_verdict="INCOMPLETE",
                first_divergence={
                    "fixture_id": fixture.fixture_id,
                    "phase": "fixture-validation",
                    "field": "horizon",
                    "error": "horizon must be positive",
                },
            )
            return receipt
        try:
            candidate_root, reference_root = (
                fixture.candidate_factory(),
                fixture.reference_factory(),
            )
        except Exception as exc:
            receipt.update(
                verdict="INVALID_IDENTITY",
                overall_verdict="INCOMPLETE",
                first_divergence={
                    "fixture_id": fixture.fixture_id,
                    "phase": "fixture-construction",
                    "field": "runtime_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            return receipt
        row: dict[str, Any] = {
            "fixture_id": fixture.fixture_id,
            "horizon": fixture.horizon,
            "metadata": dict(fixture.metadata or {}),
            "actions": [],
        }
        try:
            candidate_ranking = candidate_root.rank_actions()
            reference_ranking = reference_root.rank_actions()
            root_pairs = (
                ("observation", candidate_root.observation(), reference_root.observation()),
                ("snapshot", candidate_root.snapshot(), reference_root.snapshot()),
                ("ranking", _canonical(candidate_ranking), _canonical(reference_ranking)),
            )
        except Exception as exc:
            receipt.update(
                verdict="INVALID_IDENTITY",
                overall_verdict="INCOMPLETE",
                first_divergence={
                    "fixture_id": fixture.fixture_id,
                    "phase": "root",
                    "field": "runtime_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                },
            )
            receipt["fixtures"].append(row)
            return receipt
        metadata = fixture.metadata or {}
        candidate_snapshot = root_pairs[1][1]
        bindings = [("fixture_input_sha256", metadata.get("fixture_input_sha256"))]
        if metadata.get("fixture_kind") == "pinned_root":
            bindings.append(("pinned_root_sha256", metadata.get("pinned_root_sha256")))
        for binding_name, declared_sha in bindings:
            if (
                isinstance(declared_sha, str)
                and declared_sha
                and declared_sha != _digest(candidate_snapshot)
            ):
                receipt.update(
                    verdict="INVALID_IDENTITY",
                    overall_verdict="INCOMPLETE",
                    first_divergence={
                        "fixture_id": fixture.fixture_id,
                        "phase": "root-binding",
                        "field": binding_name,
                        "declared_sha256": declared_sha,
                        "observed_sha256": _digest(candidate_snapshot),
                    },
                )
                receipt["fixtures"].append(row)
                return receipt
        root_hashes: dict[str, str] = {}
        for field, left, right in root_pairs:
            divergence = _compare(
                fixture_id=fixture.fixture_id,
                phase="root",
                action=None,
                field=field,
                candidate=left,
                reference=right,
            )
            if divergence is not None:
                receipt.update(
                    verdict="INVALID_IDENTITY",
                    overall_verdict="INCOMPLETE",
                    first_divergence=divergence,
                )
                receipt["fixtures"].append(row)
                return receipt
            root_hashes[f"candidate_{field}_sha256"] = _digest(left)
            root_hashes[f"reference_{field}_sha256"] = _digest(right)
        row["root_hashes"] = root_hashes
        candidate_root_frozen = root_pairs[1][1]
        reference_root_frozen = root_pairs[1][2]
        for action in range(NUM_ACTIONS):
            for factory, side in (
                (fixture.candidate_factory, "candidate"),
                (fixture.reference_factory, "reference"),
            ):
                isolation = _branch_isolation(fixture, action, factory, side)
                if isolation is not None:
                    receipt.update(
                        verdict="INVALID_IDENTITY",
                        overall_verdict="INCOMPLETE",
                        first_divergence=isolation,
                    )
                    receipt["fixtures"].append(row)
                    return receipt
            divergence, action_row = _branch(
                fixture, action, candidate_root.clone(), reference_root.clone()
            )
            action_row["action"] = action
            row["actions"].append(action_row)
            if divergence is not None:
                receipt.update(
                    verdict="INVALID_IDENTITY",
                    overall_verdict="INCOMPLETE",
                    first_divergence=divergence,
                )
                receipt["fixtures"].append(row)
                return receipt
            for runtime, frozen, side in (
                (candidate_root, candidate_root_frozen, "candidate"),
                (reference_root, reference_root_frozen, "reference"),
            ):
                root_drift = _compare(
                    fixture_id=fixture.fixture_id,
                    phase=f"root-isolation-after-action-{action}",
                    action=action,
                    field=f"{side}_root_snapshot",
                    candidate=runtime.snapshot(),
                    reference=frozen,
                )
                if root_drift is not None:
                    receipt.update(
                        verdict="INVALID_IDENTITY",
                        overall_verdict="INCOMPLETE",
                        first_divergence=root_drift,
                    )
                    receipt["fixtures"].append(row)
                    return receipt
        if (fixture.metadata or {}).get("ranking_mode") == "test-only-score-free":
            row["outcome_ranking"] = {"scope": "test-only"}
        else:
            try:
                outcome_ranking = _outcome_ranking(row["actions"], candidate_ranking)
            except Exception as exc:
                receipt.update(
                    verdict="INVALID_IDENTITY",
                    overall_verdict="INCOMPLETE",
                    first_divergence={
                        "fixture_id": fixture.fixture_id,
                        "phase": "outcome-ranking",
                        "field": "invalid_ranking",
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                    },
                )
                receipt["fixtures"].append(row)
                return receipt
            row["outcome_ranking"] = outcome_ranking
            ranking_divergence = _compare(
                fixture_id=fixture.fixture_id,
                phase="outcome-ranking",
                action=None,
                field="candidate_ranking_vs_outcomes",
                candidate=_canonical(candidate_ranking),
                reference=_canonical(outcome_ranking),
            )
            if ranking_divergence is not None:
                receipt.update(
                    verdict="INVALID_IDENTITY",
                    overall_verdict="INCOMPLETE",
                    first_divergence=ranking_divergence,
                )
                receipt["fixtures"].append(row)
                return receipt
        receipt["fixtures"].append(row)
    receipt["receipt_sha256"] = _digest(_canonical(receipt))
    return receipt
