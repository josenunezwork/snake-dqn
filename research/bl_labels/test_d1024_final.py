"""Synthetic final-union proofs; no real factory payload, label or scientific look."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_labels import d1024_final as final
from research.bl_labels import d1024_resume as resume
from research.bl_labels.exports import load_ledger, publish_session
from research.bl_labels.test_d256_resume import life
from research.bl_labels.test_d1024_resume import add_attempt, pack_fixture


def _session(output: Path, control: dict, parent: int) -> None:
    item = control["allocation"][parent]
    publish_session(
        output,
        control["namespace"],
        control["allocation"],
        {
            "schema": "bl-labels-f0-session/v1",
            "status": "COMPLETE",
            "namespace": control["namespace"],
            "allocation_id": item["allocation_id"],
            "allocation": item,
            "counters": {"deaths": 1},
        },
    )


def _complete_chain(tmp_path: Path, *, last: dict | None = None) -> tuple[Path, dict, list[Path]]:
    pack, control = pack_fixture(tmp_path)
    first = add_attempt(pack, control, 1)
    second = add_attempt(
        pack,
        control,
        2,
        prior=100,
        terminal_change={"status": "COMPLETE", "pending_parents": [], **(last or {})},
    )
    life(first, control, 0, 0)
    _session(first, control, 0)
    life(second, control, 1, 0)
    life(second, control, 2, 0)
    _session(second, control, 1)
    _session(second, control, 2)
    return pack, control, [first, second]


def test_final_union_accepts_only_last_complete_and_proves_no_pending(
    tmp_path, monkeypatch
) -> None:
    from research.bl_labels import exports

    monkeypatch.setattr(exports, "aggregate_attempts", lambda *_: pytest.fail("new look"))
    pack, control, attempts = _complete_chain(tmp_path)
    # The continuation path keeps refusing a COMPLETE terminal unchanged.
    with pytest.raises(ValueError, match="HOLD"):
        resume.attempt_chain(pack, control)
    out = tmp_path / "final/verified-ledger.json"
    result = final.publish_final(pack, out)
    assert result["attempts"] == 2 and result["pending_parents"] == 0
    assert len(load_ledger(out, control["namespace"], control["allocation"])) == 6
    assert final.verify_final(pack, out)["sha256"] == result["sha256"]
    with pytest.raises(FileExistsError):
        final.publish_final(pack, out)


@pytest.mark.parametrize(
    "last",
    [
        {"status": "PREEMPTED"},
        {"status": "INVALID"},
        {"pending_parents": ["p2"]},
        {"invalid_reason": "scientific failure"},
    ],
)
def test_final_union_holds_without_a_clean_complete_terminal(tmp_path, last) -> None:
    pack, _, _ = _complete_chain(tmp_path, last=last)
    out = tmp_path / "final/verified-ledger.json"
    with pytest.raises(ValueError, match="HOLD"):
        final.publish_final(pack, out)
    assert not out.exists()


def test_final_union_refuses_complete_before_the_last_attempt(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, terminal_change={"status": "COMPLETE", "pending_parents": []})
    add_attempt(pack, control, 2, prior=100)
    with pytest.raises(ValueError, match="HOLD"):
        final.publish_final(pack, tmp_path / "final/verified-ledger.json")


def test_final_union_holds_when_a_parent_session_is_missing(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    output = add_attempt(
        pack, control, 1, terminal_change={"status": "COMPLETE", "pending_parents": []}
    )
    life(output, control, 0, 0)
    _session(output, control, 0)
    out = tmp_path / "final/verified-ledger.json"
    with pytest.raises(ValueError, match="pending; HOLD"):
        final.publish_final(pack, out)
