"""FIX-FIRST #1 history tests; they use a temporary bare Git remote only."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest

from research.t34_round0 import history, registration


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


def test_published_ratification_is_canonical_and_postdates_frozen_source(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    path = author / "research/t34_round0/REGISTRATION_ROUND0.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(value, sort_keys=True) + "\n")
    _git(author, "add", path.relative_to(author).as_posix())
    _git(author, "commit", "-m", "publish ratification")
    _git(author, "push", "origin", "main")

    registration.require_published_registration(value, path)

    path.write_text(json.dumps({**value, "operator": "altered"}, sort_keys=True) + "\n")
    with pytest.raises(ValueError, match="origin/main"):
        registration.require_published_registration(value, path)


def test_remote_claim_for_another_root_blocks_admission(tmp_path: Path) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    _git(
        author,
        "push",
        "origin",
        "HEAD:refs/tags/t34-round0/other-study/chunks/chunk-000/attempt-01",
    )
    attempt = Path(str(value["study_root"])) / "chunks/chunk-000/attempt-01"

    with pytest.raises(history.HistoryError, match="different study_root"):
        history.reserve(attempt, value)


def test_missing_terminal_history_is_kill_after_analysis_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    missing = Path(str(value["study_root"])) / "chunks/chunk-000/attempt-01"
    history.reserve(missing, value)

    with pytest.raises(history.HistoryError, match="missing its terminal"):
        history.run_phase(value, "analysis", lambda _out: history.require_published_history(value))

    decision = json.loads((Path(str(value["study_root"])) / "analysis/decision.json").read_text())
    assert decision["decision"] == "KILL"


def test_ratified_chunk_schedule_rejects_an_absent_terminal_claim(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value["sizing"] = {"chunks": [{"start": 0, "count": 10}]}
    (Path(str(value["study_root"])) / "chunks").mkdir(parents=True)
    monkeypatch.setattr(history, "require_published_history", lambda _registration: [])

    with pytest.raises(history.HistoryError, match="missing a terminal claim"):
        history.require_chunks([], value)


def test_scientific_stop_accepts_all_retained_completed_chunks(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    root = Path(str(value["study_root"]))
    outputs = []
    for number, reason in enumerate(
        ("supervisor PASS", "explained unconditional controller fault")
    ):
        attempt = root / f"chunks/chunk-{number:03d}/attempt-01"
        attempt.mkdir(parents=True)
        (attempt / "phase_terminal.json").write_text(
            json.dumps({"status": "COMPLETE", "reason": reason})
        )
        outputs.append(attempt / "job/exports/round0")
    monkeypatch.setattr(history, "require_published_history", lambda _registration: [])

    history.require_chunks(outputs, value)


def test_pending_schedule_refuses_analysis_without_consuming_sole_reservation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00",
        sizing={"chunks": [{"start": 0, "count": 10}]},
    )

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    monkeypatch.setattr(history.time, "time", lambda: 1.0)

    with pytest.raises(history.HistoryError, match="schedule is still pending"):
        history.run_phase(value, "analysis", lambda _out: {"decision": "GO"})
    assert not (Path(str(value["study_root"])) / "analysis").exists()


def test_partial_completed_schedule_refuses_analysis_before_deadline(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00",
        sizing={"chunks": [{"start": 0, "count": 10}, {"start": 10, "count": 10}]},
    )
    completed = Path(str(value["study_root"])) / "chunks/chunk-000/attempt-01"
    completed.mkdir(parents=True)
    (completed / "phase_terminal.json").write_text("{}\n")

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    monkeypatch.setattr(history.time, "time", lambda: 1.0)

    with pytest.raises(history.HistoryError, match="schedule is still pending"):
        history.run_phase(value, "analysis", lambda _out: {"decision": "GO"})
    assert not (Path(str(value["study_root"])) / "analysis").exists()


def test_erased_local_chunk_claim_still_reserves_analysis_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00",
        sizing={"chunks": [{"start": 0, "count": 10}]},
    )
    _git(
        author,
        "push",
        "origin",
        "HEAD:refs/tags/t34-round0/round0-study/chunks/chunk-000/attempt-01",
    )

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    monkeypatch.setattr(history.time, "time", lambda: 1.0)

    with pytest.raises(history.HistoryError, match="unfinished or erased"):
        history.run_phase(value, "analysis", lambda _out: history.require_published_history(value))
    assert (
        json.loads((Path(str(value["study_root"])) / "analysis/decision.json").read_text())[
            "decision"
        ]
        == "KILL"
    )


@pytest.mark.parametrize(
    "status,reason",
    [
        ("KILL", "operator abort"),
        ("INCONCLUSIVE", "registered first cap"),
        ("COMPLETE", "explained unconditional controller fault; scientific STOP only"),
    ],
)
def test_early_settlement_precheck_never_reads_terminal_entries(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, status: str, reason: str
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00",
        sizing={"chunks": [{"start": 0, "count": 10}, {"start": 10, "count": 10}]},
    )
    attempt = Path(str(value["study_root"])) / "chunks/chunk-000/attempt-01"
    history.reserve(attempt, value)
    history.finalize(attempt, value, status, reason)
    assert (attempt / "study_closed.json").is_file()
    monkeypatch.setattr(history.time, "time", lambda: 1.0)

    def forbidden(_path: Path) -> dict:
        raise AssertionError("terminal entries must not be read before analysis reservation")

    monkeypatch.setattr(history, "read_json", forbidden)
    history._analysis_reservation_allowed(value)


def test_remote_erased_second_attempt_is_not_a_healthy_pending_schedule(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00",
        sizing={"chunks": [{"start": 0, "count": 10}, {"start": 10, "count": 10}]},
    )
    attempt = Path(str(value["study_root"])) / "chunks/chunk-000/attempt-01"
    attempt.mkdir(parents=True)
    (attempt / "phase_terminal.json").write_text("{}")
    _git(
        author,
        "push",
        "origin",
        "HEAD:refs/tags/t34-round0/round0-study/chunks/chunk-000/attempt-02",
    )
    monkeypatch.setattr(history.time, "time", lambda: 1.0)
    history._analysis_reservation_allowed(value)


def test_missing_chunk_after_dispatch_deadline_writes_sole_kill(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    author = _authoring_repo(tmp_path)
    value = _registration(tmp_path, author)
    value.update(
        approved_at="1970-01-01T00:00:00+00:00", sizing={"chunks": [{"start": 0, "count": 10}]}
    )

    class Guard:
        def __init__(self, *_args: object) -> None:
            pass

        def __enter__(self) -> "Guard":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

    monkeypatch.setattr("research.t34_pathb.source_binding_v2.FrozenWorktreeGuard", Guard)
    monkeypatch.setattr(history.time, "time", lambda: 172801.0)
    with pytest.raises(history.HistoryError, match="missing its terminal"):
        history.run_phase(value, "analysis", lambda _out: history.require_chunks([], value))
    decision = Path(str(value["study_root"])) / "analysis/decision.json"
    assert json.loads(decision.read_text())["decision"] == "KILL"
