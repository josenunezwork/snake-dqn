"""Source-bound T34 transport admission without provider calls."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from research.colab_cli import supervisor
from research.t34_exit_b8 import make_manifest as engine
from research.t34_exit_b8 import qualification
from research.t34_exit_b8_fit import make_manifest as fit


def _git(repo: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=repo, check=True, capture_output=True, text=True, timeout=15
    ).stdout.strip()


@pytest.fixture
def accepted_repo(tmp_path: Path) -> tuple[Path, str]:
    repo = tmp_path / "repo"
    transport = repo / "research/colab_cli"
    transport.mkdir(parents=True)
    (transport / "supervisor.py").write_text("# accepted supervisor\n")
    (transport / "launch.sh").write_text("#!/bin/sh\nexit 0\n")
    source_manifest = {
        str(path.relative_to(repo)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in transport.iterdir()
    }
    record = {
        "schema": "snake-dqn-t34-transport-qualification/v1",
        "status": "PASS",
        "profile": "final-only",
        "source_manifest": source_manifest,
    }
    (transport / "QUALIFICATION.md").write_text(
        "# Reviewed record\n\n```json\n" + json.dumps(record) + "\n```\n"
    )
    for relative in ("research/t34_exit_b8/benchmark.sh", "research/t34_exit_b8_fit/fit.sh"):
        script = repo / relative
        script.parent.mkdir(parents=True, exist_ok=True)
        script.write_text("#!/bin/sh\nexit 0\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Accepted fixture",
    )
    return repo, _git(repo, "rev-parse", "HEAD")


def test_matching_local_and_pinned_inventory_is_accepted(
    accepted_repo: tuple[Path, str],
) -> None:
    repo, commit = accepted_repo
    assert qualification.require_qualified_transport(repo, commit)["profile"] == "final-only"
    (repo / "research/colab_cli/ACCEPTANCE.md").write_text("Documentation only\n")
    assert qualification.require_qualified_transport(repo, commit)["status"] == "PASS"


@pytest.mark.parametrize("change", ["modify", "add", "remove", "nested"])
def test_any_executable_inventory_drift_refuses(
    accepted_repo: tuple[Path, str],
    change: str,
) -> None:
    repo, commit = accepted_repo
    transport = repo / "research/colab_cli"
    if change == "modify":
        (transport / "supervisor.py").write_text("# changed\n")
    elif change == "remove":
        (transport / "launch.sh").unlink()
    elif change == "add":
        (transport / "extra.py").write_text("# new\n")
    else:
        (transport / "nested").mkdir()
        (transport / "nested/extra.sh").write_text("exit 0\n")
    with pytest.raises(qualification.QualificationRefused, match="re-run final-only acceptance"):
        qualification.require_qualified_transport(repo, commit)


@pytest.mark.parametrize("change", ["modify", "add", "remove"])
def test_pinned_commit_drift_refuses_even_when_local_files_match(
    accepted_repo: tuple[Path, str],
    change: str,
) -> None:
    repo, accepted = accepted_repo
    transport = repo / "research/colab_cli"
    if change == "modify":
        (transport / "supervisor.py").write_text("# changed pinned code\n")
    elif change == "add":
        (transport / "extra.py").write_text("# changed pinned inventory\n")
    else:
        (transport / "launch.sh").unlink()
    _git(repo, "add", ".")
    _git(
        repo,
        "-c",
        "user.name=Fixture",
        "-c",
        "user.email=fixture@example.test",
        "-c",
        "core.hooksPath=/dev/null",
        "commit",
        "-qm",
        "Unqualified fixture",
    )
    changed = _git(repo, "rev-parse", "HEAD")
    _git(repo, "restore", "--source", accepted, "--", "research/colab_cli")
    if change == "add":
        (transport / "extra.py").unlink(missing_ok=True)
    with pytest.raises(qualification.QualificationRefused, match="pinned colab_cli"):
        qualification.require_qualified_transport(repo, changed)


@pytest.mark.parametrize("document", ["missing", "```json\n{}\n```", "```json\n{\n```"])
def test_absent_or_malformed_record_fails_closed(
    accepted_repo: tuple[Path, str],
    document: str,
) -> None:
    repo, commit = accepted_repo
    path = repo / "research/colab_cli/QUALIFICATION.md"
    if document == "missing":
        path.unlink()
    else:
        path.write_text(document)
    with pytest.raises(qualification.QualificationRefused, match="cannot be verified"):
        qualification.require_qualified_transport(repo, commit)


@pytest.mark.parametrize("builder", [engine, fit])
def test_both_builders_and_validators_refuse_transport_drift(
    accepted_repo: tuple[Path, str],
    monkeypatch: pytest.MonkeyPatch,
    builder: object,
) -> None:
    repo, commit = accepted_repo
    (repo / "research/colab_cli/supervisor.py").write_text("# unqualified accounting fix\n")
    monkeypatch.setattr(builder, "ROOT", repo)
    kwargs = {"kind": "benchmark"} if builder is engine else {}
    with pytest.raises(builder.ManifestRefused, match="hashes differ"):
        builder.make_manifest(commit=commit, output_dir=repo / "output", **kwargs)
    with pytest.raises(builder.ManifestRefused, match="hashes differ"):
        builder.validate_manifest(
            {"commit": commit, "transport": {"dispatch_ready": True}}, repo=repo
        )


def test_record_matches_actual_historical_accepted_source() -> None:
    """Independently resolve the checked-in receipt inventory against its Git commit."""
    repo = Path(__file__).resolve().parents[1]
    document = (repo / "research/colab_cli/QUALIFICATION.md").read_text()
    record = json.loads(document.split("```json\n", 1)[1].split("\n```", 1)[0])
    assert record["qualified_commit"] == "cd766c639c37acfb956d677ab95fdf7359a1bb61"
    assert record["receipt_sha256"] == (
        "54e9f2f84dda243f138ab3f374e2d4b4be13c22b4977939f3c810020bf40e101"
    )
    available = subprocess.run(
        ["git", "cat-file", "-e", record["qualified_commit"]],
        cwd=repo,
        capture_output=True,
        timeout=15,
    )
    if available.returncode:
        pytest.skip("historical acceptance commit unavailable in shallow checkout")
    for name, digest in record["source_manifest"].items():
        result = subprocess.run(
            ["git", "show", f"{record['qualified_commit']}:{name}"],
            cwd=repo,
            check=True,
            capture_output=True,
            timeout=15,
        )
        assert hashlib.sha256(result.stdout).hexdigest() == digest


@pytest.mark.parametrize(
    "script", ["research/t34_exit_b8/benchmark.sh", "research/t34_exit_b8_fit/fit.sh"]
)
def test_saved_manifest_is_rechecked_by_shared_broker_before_allocation(
    accepted_repo: tuple[Path, str],
    script: str,
) -> None:
    repo, commit = accepted_repo
    saved = {
        "job_id": "saved-t34",
        "commit": commit,
        "script": script,
        "requirements": {"gpu": "L4", "min_vcpu": 2, "est_minutes": 1},
        "timeout_minutes": 3,
        "budget": {"rate_units_per_hour": 2, "unit_cap": 1},
        "output_dir": str(repo / "fresh-output"),
    }
    assert supervisor.validate(saved, repo=repo)["job_id"] == "saved-t34"
    (repo / "research/colab_cli/supervisor.py").write_text("# source changed after preparation\n")
    with pytest.raises(supervisor.Refused, match="hashes differ"):
        supervisor.validate(saved, repo=repo)
    assert not (repo / "fresh-output").exists()
