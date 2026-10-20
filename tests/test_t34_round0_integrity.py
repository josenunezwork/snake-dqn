"""Round 0 immutable-history and one-look integrity guards.

These tests use only a temporary bare Git remote.  They never address the
project remote, launch a supervisor, or allocate a runtime.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.t34_pathb.retention_history_v2 import RegistrationError
from research.t34_pathb.source_binding_v2 import SourceBindingError, verify_frozen_worktree
from research.t34_round0 import dispatch, history


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, text=True, capture_output=True
    ).stdout.strip()


def _authoring_repo(tmp_path: Path) -> Path:
    remote, author = tmp_path / "origin.git", tmp_path / "author"
    subprocess.run(["git", "init", "--bare", str(remote)], check=True, capture_output=True)
    _git(tmp_path, "init", str(author))
    _git(author, "config", "user.email", "round0@example.test")
    _git(author, "config", "user.name", "Round 0 test")
    _git(author, "checkout", "-b", "main")
    _git(author, "remote", "add", "origin", str(remote))
    (author / "README").write_text("fixture\n")
    _git(author, "add", "README")
    _git(author, "commit", "-m", "initial fixture")
    _git(author, "push", "-u", "origin", "main")
    return author


def _registration(tmp_path: Path, author: Path) -> dict[str, object]:
    root = tmp_path / "round0-study"
    return {
        "study_root": str(root),
        "commit": _git(author, "rev-parse", "HEAD"),
        "history_ledger": {
            "repo": str(author.resolve()),
            "path": f"research/t34_round0/ledger/{root.name}.jsonl",
        },
    }


def _publish(
    author: Path, registration: dict[str, object], message: str = "publish ledger"
) -> None:
    _, _, name = history.policy(registration)
    _git(author, "add", name)
    _git(author, "commit", "-m", message)
    _git(author, "push", "origin", "main")


def _reserve_and_finalize(registration: dict[str, object]) -> Path:
    root = Path(str(registration["study_root"]))
    attempt = root / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, registration)
    (attempt / "receipt.json").write_text('{"status":"complete"}\n')
    history.finalize(attempt, registration, "COMPLETE", "test terminal")
    return attempt


def test_fixed_attempt_paths_reject_symlinks_and_remote_claim_blocks_reroll(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    attempt = Path(str(registration["study_root"])) / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, registration)

    claims = _git(author, "ls-remote", "--refs", "origin", "refs/tags/t34-round0/*")
    assert (
        f"t34-round0/{Path(registration['study_root']).name}/chunks/chunk-000/attempt-01" in claims
    )

    (attempt / "admission.json").unlink()
    attempt.rmdir()
    with pytest.raises(RegistrationError, match="history publication|failed"):
        history.reserve(attempt, registration)

    linked = tmp_path / "linked-attempt"
    linked.symlink_to(tmp_path / "not-a-phase")
    with pytest.raises(history.HistoryError, match="unlinked"):
        history.phase_name(linked, registration)


def test_unpublished_changed_and_deleted_history_fails_closed(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    attempt = _reserve_and_finalize(registration)

    with pytest.raises(history.HistoryError, match="committed and pushed"):
        history.require_published_history(registration)

    _publish(author, registration)
    assert history.require_published_history(registration)[0]["status"] == "COMPLETE"

    (attempt / "receipt.json").write_text('{"status":"altered"}\n')
    with pytest.raises(history.HistoryError, match="changed"):
        history.require_published_history(registration)

    (attempt / "receipt.json").write_text('{"status":"complete"}\n')
    _, _, ledger_name = history.policy(registration)
    (author / ledger_name).unlink()
    _git(author, "add", "-u")
    _git(author, "commit", "-m", "delete ledger fixture")
    _git(author, "push", "origin", "main")
    with pytest.raises(RegistrationError, match="append-only|removed"):
        history.require_published_history(registration)


def test_operator_abort_is_sealed_as_kill_with_sole_decision(monkeypatch, tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr(history, "require_published_history", lambda _registration: [])
    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)

    with pytest.raises(KeyboardInterrupt):
        history.run_phase(
            registration, "analysis", lambda _out: (_ for _ in ()).throw(KeyboardInterrupt())
        )

    analysis = Path(str(registration["study_root"])) / "analysis"
    assert json.loads((analysis / "decision.json").read_text())["decision"] == "KILL"
    assert json.loads((analysis / "interruption.json").read_text())["decision"] == "KILL"
    assert json.loads((analysis / "phase_terminal.json").read_text())["status"] == "KILL"


def test_only_one_analysis_reservation_and_kill_close_chunk_selection(
    monkeypatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    monkeypatch.setattr(history, "require_published_history", lambda _registration: [])
    assert (
        history.run_phase(registration, "analysis", lambda _out: {"decision": "INCONCLUSIVE"})[
            "decision"
        ]
        == "INCONCLUSIVE"
    )
    with pytest.raises(history.HistoryError, match="create-only"):
        history.run_phase(registration, "analysis", lambda _out: {"decision": "GO"})

    monkeypatch.setattr(
        history, "require_published_history", lambda _registration: [{"status": "KILL"}]
    )
    with pytest.raises(history.HistoryError, match="KILL"):
        history.require_chunks([], registration)


def test_only_a_measured_mechanical_failure_can_precede_attempt_two(
    monkeypatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    registration = _registration(tmp_path, author)
    root = Path(str(registration["study_root"]))
    attempt_one = root / "chunks/chunk-000/attempt-01"
    attempt_two = root / "chunks/chunk-000/attempt-02"
    attempt_one.mkdir(parents=True)
    attempt_two.mkdir()
    monkeypatch.setattr(history, "require_published_history", lambda _registration: [])
    for path, status in ((attempt_one, "MECHANICAL_FAILURE"), (attempt_two, "COMPLETE")):
        (path / "phase_terminal.json").write_text(json.dumps({"status": status}))

    expected = attempt_two / "job/exports/round0"
    history.require_chunks([expected], registration)

    (attempt_one / "phase_terminal.json").write_text(json.dumps({"status": "COMPLETE"}))
    with pytest.raises(history.HistoryError, match="mechanical rerun"):
        history.require_chunks([expected], registration)

    job = tmp_path / "failed-download-job"
    terminal = {
        "job_id": "round0-001",
        "generation": "generation-001",
        "status": "completed",
        "returncode": 0,
        "reason": None,
        "exports": [{"path": "exports/round0/chunk.json"}],
    }
    (job / "poll-000001.json").parent.mkdir()
    (job / "poll-000001.json").write_text(json.dumps({"terminal": terminal}))
    failed_download = {
        "status": "FAIL",
        "stop_verified": True,
        "cleanup_error": None,
        "error": "TransferError(download immutable export)",
        "job_id": "round0-001",
        "generation": "generation-001",
        "transfers": [
            {
                "status": "failed",
                "direction": "download",
                "remote": "/content/round0/output/exports/round0/chunk.json",
            }
        ],
    }
    assert dispatch._mechanical(failed_download, job)

    refused_download = dict(
        failed_download,
        error="Refused('download hash mismatch or transfer failure: immutable export')",
    )
    assert dispatch._mechanical(refused_download, job)

    for error in ("TransferError(final usage query)", "TransferError(backup check)"):
        no_failed_download = dict(failed_download, error=error, transfers=[])
        assert not dispatch._mechanical(no_failed_download, job)


def test_chunk_sample_is_ceil_ten_percent_per_interleaved_chunk() -> None:
    from research.t34_round0.bank import ordered_rows

    rows = [
        {"world_id": f"{mix}-{index:02d}", "mix": mix}
        for mix, count in (("scripted", 8), ("frozen", 6), ("mixed", 6))
        for index in range(count)
    ]
    ordered = ordered_rows(rows)
    chunks = [
        {
            "slice": {
                "start": start,
                "count": 10,
                "world_ids": [row["world_id"] for row in ordered[start : start + 10]],
            }
        }
        for start in (0, 10)
    ]
    assert [row["world_id"] for row in history.selected_rows(rows, chunks)] == [
        "scripted-00",
        "scripted-04",
    ]


def test_frozen_worktree_rejects_untracked_bytes(tmp_path: Path) -> None:
    repo = tmp_path / "frozen"
    _git(tmp_path, "init", str(repo))
    _git(repo, "config", "user.email", "round0@example.test")
    _git(repo, "config", "user.name", "Round 0 test")
    (repo / "source.py").write_text("value = 1\n")
    _git(repo, "add", "source.py")
    _git(repo, "commit", "-m", "frozen fixture")
    commit = _git(repo, "rev-parse", "HEAD")
    _git(repo, "checkout", "--detach", commit)
    verify_frozen_worktree(repo, commit)
    (repo / "later.txt").write_text("changed\n")
    with pytest.raises(SourceBindingError, match="untracked"):
        verify_frozen_worktree(repo, commit)


def test_real_chunk_sample_retains_all_three_mixes() -> None:
    from collections import Counter

    from research.t34_round0.bank import ordered_rows

    rows = [
        {"world_id": f"{mix}-{i}", "mix": mix}
        for mix, count in (("scripted", 96), ("frozen", 72), ("mixed", 72))
        for i in range(count)
    ]
    ordered = ordered_rows(rows)
    chunks = [
        {
            "slice": {
                "start": start,
                "count": count,
                "world_ids": [r["world_id"] for r in ordered[start : start + count]],
            }
        }
        for start, count in ((0, 140), (140, 100))
    ]
    sample = history.selected_rows(rows, chunks)
    assert len(sample) == 24
    assert Counter(r["mix"] for r in sample) == {"scripted": 10, "frozen": 7, "mixed": 7}
