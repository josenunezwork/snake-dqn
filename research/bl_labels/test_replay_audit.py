"""Admission checks for the independent Mac action replay audit."""

from __future__ import annotations

import base64

import pytest

from research.bl_labels import replay_audit


def test_audit_requires_full_64_root_bank_before_any_outcome(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def forbidden(record: dict) -> dict:
        raise AssertionError("partial bank must remain sealed")

    monkeypatch.setattr(replay_audit, "replay_root", forbidden)
    assert replay_audit.audit([{"root_id": "one"}], require_mac=False) == {
        "status": "INCOMPLETE",
        "available_roots": 1,
        "required_roots": 64,
    }


def test_audit_refuses_duplicate_root_ids() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        replay_audit.audit([{"root_id": "same"}] * 64, require_mac=False)


def test_audit_selection_is_order_independent_and_counts_unexplained_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = [{"root_id": f"root-{i}"} for i in range(80)]
    monkeypatch.setattr(
        replay_audit,
        "replay_root",
        lambda row: {
            "decisions": 20,
            "policy_divergences": [],
            "unexplained_policy_divergences": 1,
        },
    )
    first = replay_audit.audit(rows, require_mac=False)
    second = replay_audit.audit(rows[::-1], require_mac=False)
    assert first["roots"] == second["roots"]
    assert first["root_count"] == 64
    assert first["status"] == "INVALID"


def test_snapshot_hash_is_verified_before_unpickling() -> None:
    with pytest.raises(ValueError, match="snapshot digest mismatch"):
        replay_audit.replay_root(
            {
                "snapshot_b64": base64.b64encode(b"never unpickle me").decode(),
                "snapshot_sha256": "0" * 64,
            }
        )
