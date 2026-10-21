"""Fast safety checks for the excluded S1 v4 validation loader."""

from __future__ import annotations

from datetime import timedelta

import pytest

from research.speed_s1_digest_v4 import validate


def test_private_loader_keeps_production_digest_globals_unchanged() -> None:
    from research.t34_exit_b8 import teacher_v2
    from research.t34_shadow import shadow

    original_shadow, original_teacher = shadow.full_digest, teacher_v2.full_digest
    private_shadow, private_teacher = validate.load_private_v4_modules()

    assert shadow.full_digest is original_shadow
    assert teacher_v2.full_digest is original_teacher
    assert private_shadow.full_digest is private_teacher.full_digest
    assert private_shadow.full_digest is not original_shadow
    assert private_shadow.FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v4"
    assert shadow.FULL_DIGEST_SCHEMA == "t34-shadow-full-digest/v3"


def test_partial_or_empty_exports_cannot_qualify_for_byte_identity(tmp_path) -> None:
    partial = validate._one_run(
        lambda _config: {"complete": False, "worlds": 0, "active_row_steps": 1}, {}, tmp_path
    )
    assert partial["exports"] == {}
    assert not validate._identity_eligible(partial)


def test_d2_deadline_closes_s1_and_retains_v3(monkeypatch) -> None:
    class Clock:
        @staticmethod
        def now(_timezone):
            return validate.D2_DEADLINE + timedelta(seconds=1)

    monkeypatch.setattr(validate, "datetime", Clock)
    with pytest.raises(RuntimeError, match="retain v3 and close S1"):
        validate._admit()


def test_payload_without_immutable_sidecars_cannot_claim_identity() -> None:
    assert not validate._identity_eligible(
        {"complete": True, "worlds": 1, "exports": {"world-000000.json": "hash"}}
    )
