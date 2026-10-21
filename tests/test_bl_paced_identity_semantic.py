"""Focused semantic-oracle checks; these do not qualify a paced study."""

from __future__ import annotations

import pytest

from research.bl_paced_identity.semantic import (
    SemanticFixture,
    _outcome_ranking,
    _production_attestation,
    check_semantics,
)
from research.bl_paced_identity.semantic_fixture import tiny_semantic_fixtures


def test_tiny_grid_fixture_matches_independent_serial_reference() -> None:
    receipt = check_semantics(tiny_semantic_fixtures(), candidate="grid", reference="batch-serial")
    assert receipt["verdict"] == "PASS"
    assert receipt["qualification"] == "PRECHECK_ONLY"
    assert receipt["overall_verdict"] == "INCOMPLETE"
    assert receipt["fixtures"][0]["actions"]
    assert len(receipt["fixtures"][0]["actions"]) == 6


def test_semantic_oracle_reports_first_divergence() -> None:
    fixture = tiny_semantic_fixtures()[0]

    class DivergentReference:
        def __init__(self, inner):
            self.inner = inner

        def clone(self):
            return type(self)(self.inner.clone())

        def snapshot(self):
            return self.inner.snapshot()

        def observation(self):
            return self.inner.observation() + b"drift"

        def rank_actions(self):
            return self.inner.rank_actions()

        def step(self, action):
            return self.inner.step(action)

    receipt = check_semantics(
        [
            SemanticFixture(
                fixture_id="deliberate-observation-drift",
                candidate_factory=fixture.candidate_factory,
                reference_factory=lambda: DivergentReference(fixture.reference_factory()),
                horizon=1,
            )
        ]
    )
    assert receipt["verdict"] == "INVALID_IDENTITY"
    assert receipt["first_divergence"]["field"] == "observation"
    assert receipt["first_divergence"]["phase"] == "root"


def test_fixture_clone_carries_rng_and_keeps_siblings_isolated() -> None:
    runtime = tiny_semantic_fixtures()[0].candidate_factory()
    left, sibling = runtime.clone(), runtime.clone()
    before = sibling.snapshot()
    left.step(1)
    assert sibling.snapshot() == before
    replay = runtime.clone()
    replay.step(1)
    # Identical post-step state includes the per-environment RNG stream.
    assert left.snapshot() == replay.snapshot()


def test_empty_or_short_fixture_set_cannot_qualify_production() -> None:
    empty = check_semantics([])
    assert empty["verdict"] == "INCOMPLETE"
    assert empty["overall_verdict"] == "INCOMPLETE"
    tiny = check_semantics(tiny_semantic_fixtures())
    assert tiny["overall_verdict"] == "INCOMPLETE"
    assert "requires exactly 18 mix_length fixtures" in tiny["production_attestation_problems"]


def test_branch_forces_only_first_action_then_uses_greedy_continuation() -> None:
    fixture = tiny_semantic_fixtures()[0]

    class RecordingRuntime:
        seen: list[int | None] = []

        def __init__(self, inner):
            self.inner = inner

        def clone(self):
            return type(self)(self.inner.clone())

        def snapshot(self):
            return self.inner.snapshot()

        def observation(self):
            return self.inner.observation()

        def rank_actions(self):
            return self.inner.rank_actions()

        def step(self, action):
            type(self).seen.append(action)
            return self.inner.step(action)

    RecordingRuntime.seen = []
    receipt = check_semantics(
        [
            SemanticFixture(
                fixture_id="continuation-probe",
                candidate_factory=lambda: RecordingRuntime(fixture.candidate_factory()),
                reference_factory=lambda: RecordingRuntime(fixture.reference_factory()),
                horizon=2,
                metadata={"ranking_mode": "test-only-score-free"},
            )
        ]
    )
    assert receipt["verdict"] == "PASS"
    assert None in RecordingRuntime.seen
    assert any(action in range(6) for action in RecordingRuntime.seen)


def test_runtime_error_is_sealed_as_identity_failure() -> None:
    fixture = tiny_semantic_fixtures()[0]

    def broken_reference():
        raise RuntimeError("deliberate reference failure")

    receipt = check_semantics(
        [
            SemanticFixture(
                fixture_id="sealed-error",
                candidate_factory=fixture.candidate_factory,
                reference_factory=broken_reference,
                horizon=1,
            )
        ]
    )
    assert receipt["verdict"] == "INVALID_IDENTITY"
    assert receipt["first_divergence"]["field"] == "runtime_error"


def test_t3_ranking_prefers_alive_transitions_over_mass_and_keeps_b8_on_tie() -> None:
    actions = [
        {"action": 0, "alive_transitions": 11, "integer_mass_sum": 1},
        {"action": 1, "alive_transitions": 11, "integer_mass_sum": 999_999},
        {"action": 2, "alive_transitions": 10, "integer_mass_sum": 1_000_000},
        {"action": 3, "alive_transitions": 1, "integer_mass_sum": 0},
        {"action": 4, "alive_transitions": 1, "integer_mass_sum": 0},
        {"action": 5, "alive_transitions": 1, "integer_mass_sum": 0},
    ]
    ranking = _outcome_ranking(
        actions,
        {
            "ranking": [0, 1, 2, 3, 4, 5],
            "winner": 0,
            "tie_rule": "controller-declared",
            "bare_action": 1,
            "q_values": [99.0, -99.0, 1_000_000.0, 0.0, 0.0, 0.0],
        },
    )
    assert ranking["winner"] == 1  # B8 wins the equal-alive tie despite lower root Q.
    assert ranking["ranking"][:3] == [1, 0, 2]  # Richer early death cannot outrank longer alive.


def test_test_only_ranking_cannot_bypass_complete_production_attestation() -> None:
    """A 114-row metadata bank is cheap to build and must still reject the bypass."""
    sha = "a" * 64
    base = {
        "candidate_source_sha256": sha,
        "frozen_reference_source_sha256": sha,
        "fixture_input_sha256": sha,
        "frozen_opponent_carry": True,
        "unshared_modified_helpers": True,
        "independent_reference_contract": "separate pinned serial module",
        "full_v3": True,
        "digest_schema": "t34-shadow-full-digest/v3",
    }
    fixtures = []
    for mix in ("frozen", "mixed", "scripted"):
        for stratum in range(6):
            fixtures.append(
                SemanticFixture(
                    fixture_id=f"{mix}-{stratum}",
                    candidate_factory=lambda: None,
                    reference_factory=lambda: None,
                    metadata={
                        **base,
                        "fixture_kind": "mix_length",
                        "mix": mix,
                        "length_stratum": str(stratum),
                    },
                )
            )
    for root in range(96):
        fixtures.append(
            SemanticFixture(
                fixture_id=f"root-{root}",
                candidate_factory=lambda: None,
                reference_factory=lambda: None,
                metadata={
                    **base,
                    "fixture_kind": "pinned_root",
                    "root_pin": f"root-{root}",
                    "pinned_root_sha256": sha,
                    "ranking_mode": "test-only-score-free" if root == 0 else "production",
                },
            )
        )
    ready, problems = _production_attestation(fixtures)
    assert not ready
    assert any("test-only ranking cannot qualify production" in problem for problem in problems)


def test_delayed_shared_carry_mutation_isolated_through_greedy_continuation() -> None:
    """A clone alias that mutates only on tick two must not escape isolation."""

    class DelayedCarryRuntime:
        def __init__(self, carry=None, *, shared=True):
            self.carry = [] if carry is None else carry
            self.shared = shared
            self.ticks = 0

        def clone(self):
            carry = self.carry if self.shared else list(self.carry)
            return type(self)(carry, shared=self.shared)

        def snapshot(self):
            return repr(self.carry).encode("ascii")

        def observation(self):
            return b"observation"

        def rank_actions(self):
            return {
                "ranking": [1, 0, 2, 3, 4, 5],
                "winner": 1,
                "tie_rule": "test-only-score-free",
                "bare_action": 1,
                "q_values": [0.0] * 6,
            }

        def step(self, action):
            self.ticks += 1
            if self.ticks == 2:
                self.carry.append("delayed")
            return {
                "hero_action": 1 if action is None else action,
                "hero_alive": True,
                "terminal": False,
                "integer_mass_sum": 1,
                "opponent_actions": [],
            }

    receipt = check_semantics(
        [
            SemanticFixture(
                fixture_id="delayed-carry-alias",
                candidate_factory=lambda: DelayedCarryRuntime(shared=True),
                reference_factory=lambda: DelayedCarryRuntime(shared=False),
                horizon=2,
                metadata={"ranking_mode": "test-only-score-free"},
            )
        ]
    )
    assert receipt["verdict"] == "INVALID_IDENTITY"
    assert "isolation-candidate-step-1" in receipt["first_divergence"]["phase"]


def test_external_candidate_and_binding_sha_are_preserved_without_qualification() -> None:
    sha = "b" * 64
    receipt = check_semantics(tiny_semantic_fixtures(), candidate_sha256=sha, binding_sha256=sha)
    assert receipt["candidate_sha256"] == sha
    assert receipt["binding_sha256"] == sha
    assert receipt["bank114_sha256"] is None
    with pytest.raises(ValueError, match="candidate_sha256"):
        check_semantics([], candidate_sha256="not-a-sha")
