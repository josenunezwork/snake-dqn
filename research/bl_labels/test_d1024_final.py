"""Synthetic final-union proofs; no real factory payload, label or scientific look."""

from __future__ import annotations

from pathlib import Path

import pytest

from research.bl_labels import d1024_final as final
from research.bl_labels import d1024_resume as resume
from research.bl_labels.exports import load_ledger, publish_session
from research.bl_labels.test_d256_resume import life
from research.bl_labels.test_d1024_resume import add_attempt, pack_fixture

PIN = "c" * 40


@pytest.fixture
def real_pin():
    """Opt out of the pinned-publisher stub; the test exercises the real check."""


@pytest.fixture(autouse=True)
def pinned_publisher(monkeypatch, request):
    if "real_pin" not in request.fixturenames:
        monkeypatch.setattr(final, "require_pinned_source", lambda commit: commit)


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
    result = final.publish_final(pack, out, PIN)
    assert result["attempts"] == 2 and result["pending_parents"] == 0
    assert result["code_commit"] == PIN
    assert len(load_ledger(out, control["namespace"], control["allocation"])) == 6
    assert final.verify_final(pack, out)["sha256"] == result["sha256"]
    with pytest.raises(FileExistsError):
        final.publish_final(pack, out, PIN)


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
        final.publish_final(pack, out, PIN)
    assert not out.exists()


def test_final_union_refuses_complete_before_the_last_attempt(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    add_attempt(pack, control, 1, terminal_change={"status": "COMPLETE", "pending_parents": []})
    add_attempt(pack, control, 2, prior=100)
    with pytest.raises(ValueError, match="HOLD"):
        final.publish_final(pack, tmp_path / "final/verified-ledger.json", PIN)


def test_final_union_holds_when_a_parent_session_is_missing(tmp_path) -> None:
    pack, control = pack_fixture(tmp_path)
    output = add_attempt(
        pack, control, 1, terminal_change={"status": "COMPLETE", "pending_parents": []}
    )
    life(output, control, 0, 0)
    _session(output, control, 0)
    out = tmp_path / "final/verified-ledger.json"
    with pytest.raises(ValueError, match="pending; HOLD"):
        final.publish_final(pack, out, PIN)


class _Repo:
    """Fake repository answers for the publisher pin check."""

    def __init__(self, head: str, dirty: str = "", published: bool = True) -> None:
        self.head, self.dirty, self.published = head, dirty, published

    def __call__(self, *args: str):
        from types import SimpleNamespace

        if args[:2] == ("rev-parse", "--verify"):
            return SimpleNamespace(returncode=0, stdout=PIN + "\n")
        if args[:2] == ("rev-parse", "HEAD"):
            return SimpleNamespace(returncode=0, stdout=self.head + "\n")
        if args[0] == "status":
            return SimpleNamespace(returncode=0, stdout=self.dirty)
        if args[0] == "merge-base":
            return SimpleNamespace(returncode=0 if self.published else 1, stdout="")
        raise AssertionError(args)


@pytest.mark.parametrize(
    "repo,ok",
    [
        (_Repo(PIN), True),
        (_Repo("d" * 40), False),
        (_Repo(PIN, dirty=" M research/x.py\n"), False),
        (_Repo(PIN, published=False), False),
    ],
)
def test_union_publisher_requires_a_clean_published_pinned_commit(
    monkeypatch, real_pin, repo, ok
) -> None:
    monkeypatch.setattr(final, "_git", repo)
    if ok:
        assert final.require_pinned_source(PIN) == PIN
    else:
        with pytest.raises(ValueError, match="HOLD"):
            final.require_pinned_source(PIN)


def test_unpinned_publisher_refuses_before_any_union_io(tmp_path, monkeypatch, real_pin) -> None:
    monkeypatch.setattr(final, "_git", _Repo("d" * 40))
    monkeypatch.setattr(
        final.d1024_resume, "original_control", lambda *_: pytest.fail("chain opened")
    )
    with pytest.raises(ValueError, match="pinned"):
        final.publish_final(tmp_path / "pack", tmp_path / "out.json", PIN)
    assert not (tmp_path / "out.json").exists()
