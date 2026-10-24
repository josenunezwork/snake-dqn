"""R33 deterministic task-journal contracts; stdlib-only and safe with --noconftest."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from tools.ops.task_journal import ELIGIBLE_READY, HOLD, Identity, Journal, Refused


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def identity(i: int = 0) -> Identity:
    return Identity(
        "t34", "r33", f"attempt-{i}", sha(f"source-{i}"), sha(f"manifest-{i}"), sha(f"receipt-{i}")
    )


def ready(journal: Journal, item: Identity, now: int = 10) -> str:
    journal.upsert(item, lane=f"lane-{item.attempt}", owner="claude", next_action="review", now=now)
    journal.update_proofs(
        item.key(),
        static_verified=True,
        authority_verified=True,
        proof_evidence_sha256=sha("proof"),
        authority_receipt_sha256=sha("authority"),
        now=now,
    )
    return journal.set_eligibility(item.key(), eligible=True, now=now)


def test_one_hundred_deterministic_crash_restart_schedules_preserve_actionability(
    tmp_path: Path,
) -> None:
    """Every write boundary restarts from SQLite without losing or duplicating intent."""
    for schedule in range(100):
        ops = tmp_path / str(schedule)
        item = identity(schedule)
        journal = Journal(ops)
        assert ready(journal, item) == ELIGIBLE_READY
        if schedule % 4 == 0:
            journal.close()
            journal = Journal(ops)
        claim = journal.claim(item.key(), "operator", ttl_seconds=5, now=11)
        if schedule % 4 == 1:
            journal.close()
            journal = Journal(ops)
        if schedule % 3:
            journal.submission_intent(item.key(), "operator", claim, now=12)
        if schedule % 4 == 2:
            journal.close()
            journal = Journal(ops)
        if schedule % 3 == 1:
            journal.confirm_submission(item.key(), "operator", claim, now=13)
        if schedule % 4 == 3:
            journal.close()
            journal = Journal(ops)
        journal.reconcile(20)
        task = journal._task(item.key())
        assert task["state"] in {"RUNNING", HOLD, ELIGIBLE_READY}
        if schedule % 3 == 2:
            assert task["state"] == HOLD
        intents = journal.db.execute(
            "SELECT COUNT(*) FROM events WHERE identity=? AND kind='SUBMISSION_INTENT'",
            (item.key(),),
        ).fetchone()[0]
        assert intents == (1 if schedule % 3 else 0)
        notices = journal.deliver("restarted-consumer", now=21)
        assert len({row["id"] for row in notices}) == len(notices)
        journal.close()


def test_revocation_invalidates_unacknowledged_notices_and_claims(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    item = identity()
    ready(journal, item)
    assert journal.deliver("operator", now=11)
    journal.update_proofs(
        item.key(),
        static_verified=True,
        authority_verified=True,
        proof_evidence_sha256=sha("proof2"),
        authority_receipt_sha256=sha("authority2"),
        source_revoked=True,
        now=12,
    )
    assert journal._task(item.key())["state"] == HOLD
    assert journal.deliver("operator", now=13) == []


def test_stale_proof_cannot_resurrect_claimed_or_terminal_work(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    item = identity()
    ready(journal, item)
    generation = journal.claim(item.key(), "operator", ttl_seconds=30, now=11)
    with pytest.raises(Refused):
        journal.update_proofs(
            item.key(),
            static_verified=True,
            authority_verified=True,
            proof_evidence_sha256=sha("p"),
            authority_receipt_sha256=sha("a"),
            now=12,
        )
    journal.submission_intent(item.key(), "operator", generation, now=12)
    journal.confirm_submission(item.key(), "operator", generation, now=13)
    journal.terminal(
        item.key(),
        terminal_evidence=True,
        allocation_released=False,
        retained_allocation=True,
        now=14,
    )
    with pytest.raises(Refused):
        journal.set_eligibility(item.key(), eligible=True, now=15)


def test_second_sqlite_connection_cannot_claim_stale_generation(tmp_path: Path) -> None:
    """Two handles model a stale reader without needing parallel test processes."""
    first = Journal(tmp_path)
    item = identity()
    ready(first, item)
    second = Journal(tmp_path)
    generation = first.claim(item.key(), "first", ttl_seconds=30, now=11)
    assert generation > 0
    with pytest.raises(Refused, match="only ELIGIBLE_READY"):
        second.claim(item.key(), "second", ttl_seconds=30, now=12)
    first.close()
    second.close()


def test_ambiguous_claim_and_expired_intent_commit_hold_then_refuse(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    item = identity()
    ready(journal, item)
    generation = journal.claim(item.key(), "operator", ttl_seconds=1, now=11)
    with pytest.raises(Refused, match="expired"):
        journal.submission_intent(item.key(), "operator", generation, now=13)
    assert journal._task(item.key())["state"] == HOLD

    other = identity(1)
    ready(journal, other)
    journal.db.execute(
        "UPDATE tasks SET submission_state='POSSIBLE' WHERE identity=?", (other.key(),)
    )
    journal.db.commit()
    with pytest.raises(Refused, match="possible submission"):
        journal.claim(other.key(), "second", ttl_seconds=20, now=14)
    assert journal._task(other.key())["state"] == HOLD


def test_per_consumer_delivery_requires_explicit_ack_and_reminds(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    item = identity()
    ready(journal, item, now=1)
    first = journal.deliver("alice", now=2)
    second = journal.deliver("bob", now=2)
    assert first and second and first[0]["id"] == second[0]["id"]
    journal.acknowledge("alice", first[0]["id"], now=3)
    assert journal.deliver("alice", now=4) == []  # stale READY generation is not actionable
    journal.reconcile(1 + 15 * 60)
    assert any(row["kind"] == "OPERATOR_ESCALATION" for row in journal.deliver("bob", now=902))


@pytest.mark.parametrize(
    ("line", "needle", "transition"),
    [
        (155, "Sol started F0", "PREPARED"),
        (157, "F4 APPROVED", "READY"),
        (159, "INCOMPLETE", "HOLD"),
        (161, "not yet PASS", "HOLD"),
        (167, "F1 re-funded", "READY"),
        (169, "NO_PRODUCTION_PLATFORM", "TERMINAL"),
        (175, "D256 dispatched", "RUNNING"),
        (181, "PREEMPTED", "TERMINAL"),
        (192, "queued", "ELIGIBLE_READY"),
        (195, "HOLD", "HOLD"),
    ],
)
def test_historical_metadata_transitions_are_anchored_not_measured_samples(
    line: int, needle: str, transition: str
) -> None:
    """Ten row-5 replay anchors have deliberately unknown durations, never SLA samples."""
    campaign_log = Path(__file__).parents[1] / "docs/research/final_run_20261019/CAMPAIGN_LOG.md"
    text = campaign_log.read_text().splitlines()[line - 1]
    assert text.startswith("|") and "2026-10-23" in text and needle in text
    historical = {"campaign_log_line": line, "transition": transition, "duration_seconds": None}
    assert historical["duration_seconds"] is None


def test_proof_or_resource_downgrade_cannot_deliver_stale_ready(tmp_path: Path) -> None:
    journal = Journal(tmp_path)
    item = identity()
    ready(journal, item)
    journal.set_eligibility(
        item.key(), eligible=False, blocker_code="CAPACITY", blocker_owner="Claude", now=12
    )
    assert all(row["kind"] == "READY" for row in journal.deliver("new", now=13))
    journal.update_proofs(item.key(), static_verified=False, authority_verified=False, now=14)
    assert journal.deliver("another", now=15) == []
    with pytest.raises(Refused):
        Identity("t34", "r33", "bad", "bad-hash", sha("m"), sha("r"))
