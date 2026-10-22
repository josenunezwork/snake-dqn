"""Contracts for the explicit authoring-checkout publication fence."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from research.published_ledger import (
    PublishedLedgerRefused,
    bind_published_ledger_check,
    require_published_ledger,
)


def _git(repo: Path, *args: str) -> None:
    subprocess.run(["git", "-C", str(repo), *args], check=True, capture_output=True)


def _repo(tmp_path: Path) -> Path:
    remote = tmp_path / "remote.git"
    local = tmp_path / "author"
    _git(tmp_path, "init", "--bare", str(remote))
    _git(tmp_path, "clone", str(remote), str(local))
    _git(local, "config", "user.email", "test@example.invalid")
    _git(local, "config", "user.name", "Test")
    ledger = local / "research" / "example" / "LOOK_LEDGER.md"
    ledger.parent.mkdir(parents=True)
    ledger.write_text("# ledger\n", encoding="utf-8")
    _git(local, "add", "research/example/LOOK_LEDGER.md")
    _git(local, "commit", "-m", "ledger")
    _git(local, "push", "origin", "HEAD:main")
    _git(local, "branch", "-M", "main")
    _git(local, "fetch", "origin", "main")
    return local


def test_published_ledger_proof_binds_exact_local_and_remote_bytes(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    proof = require_published_ledger(repo, "research/example/LOOK_LEDGER.md")
    assert proof["schema"] == "published-ledger-check/v1"
    assert proof["ledger"] == "research/example/LOOK_LEDGER.md"
    assert len(proof["ledger_sha256"]) == len(proof["proof_sha256"]) == 64
    receipt = bind_published_ledger_check({"schema": "decision/v1"}, proof)
    assert receipt["published_ledger_check_sha256"] == proof["proof_sha256"]


def test_published_ledger_refuses_dirty_authoring_checkout(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    (repo / "untracked.txt").write_text("no", encoding="utf-8")
    with pytest.raises(PublishedLedgerRefused, match="tracked or untracked"):
        require_published_ledger(repo, "research/example/LOOK_LEDGER.md")


def test_published_ledger_refuses_a_remote_byte_change(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    writer = tmp_path / "writer"
    _git(tmp_path, "clone", "--branch", "main", str(tmp_path / "remote.git"), str(writer))
    _git(writer, "config", "user.email", "test@example.invalid")
    _git(writer, "config", "user.name", "Test")
    ledger = writer / "research" / "example" / "LOOK_LEDGER.md"
    ledger.write_text("# changed\n", encoding="utf-8")
    _git(writer, "add", str(ledger.relative_to(writer)))
    _git(writer, "commit", "-m", "change ledger")
    _git(writer, "push", "origin", "HEAD:main")
    with pytest.raises(PublishedLedgerRefused, match="differ"):
        require_published_ledger(repo, "research/example/LOOK_LEDGER.md")


def test_published_ledger_refuses_a_worktree_change_during_fetch(
    tmp_path: Path, monkeypatch
) -> None:
    import research.published_ledger as published_ledger

    repo = _repo(tmp_path)
    original = published_ledger._git

    def racing_git(checkout: Path, *args: str) -> bytes:
        value = original(checkout, *args)
        if args and args[0] == "fetch":
            (repo / "raced-untracked.txt").write_text("race", encoding="utf-8")
        return value

    monkeypatch.setattr(published_ledger, "_git", racing_git)
    with pytest.raises(PublishedLedgerRefused, match="changed during"):
        require_published_ledger(repo, "research/example/LOOK_LEDGER.md")


def test_mac_claim_wrappers_bind_the_publication_proof(tmp_path: Path) -> None:
    from research.bl_info_value.decide import publish as publish_e3
    from research.bl_info_value.io import write_json
    from research.bl_ppo.exports import export_json
    from research.bl_prefix.decide import publish as publish_prefix

    repo = _repo(tmp_path)
    proof = require_published_ledger(repo, "research/example/LOOK_LEDGER.md")
    fit_report = tmp_path / "fit_report.json"
    write_json(
        fit_report,
        {
            "schema": "bl-e3-fit-v1",
            "arms": {
                arm: {
                    "decisive_world_regret": {"world-1": 1.0, "world-2": 1.0},
                    "decisive_mean_regret": 1.0,
                    "unnecessary_switch_rate": 0.0,
                }
                for arm in ("A", "B", "C")
            },
        },
    )
    e3 = publish_e3(fit_report, tmp_path / "e3-decision.json", proof)
    assert e3["published_ledger_check_sha256"] == proof["proof_sha256"]
    prefix_result = tmp_path / "e10-results.json"
    export_json(
        prefix_result,
        {"schema": "bl-prefix-e10-results/v1", "preflight": False, "status": "INCONCLUSIVE"},
    )
    prefix = publish_prefix(prefix_result, tmp_path / "prefix-decision.json", proof)
    assert prefix["published_ledger_check_sha256"] == proof["proof_sha256"]


def test_e7_claim_uses_unique_audited_roster_with_three_ancestors(tmp_path: Path) -> None:
    from research.bl_ppo.exports import export_json
    from research.bl_prefix.decide import publish

    repo = _repo(tmp_path)
    proof = require_published_ledger(repo, "research/example/LOOK_LEDGER.md")
    sources = ["world-1", "world-2"]
    confirmations = [
        {
            "source_world": source,
            "ancestor": ancestor,
            "stream": stream,
            "candidate_score": 50.0,
            "b8_score": 0.0,
            "candidate_survival": True,
            "b8_survival": True,
            "fallback_count": 0,
        }
        for source in sources
        for ancestor in (100, 150, 200)
        for stream in (0, 1)
    ]
    raw = {
        "schema": "bl-prefix-e7-results/v1",
        "preflight": False,
        "status": "COMPLETE",
        "expected_sources": sources,
        "selection": [
            {"source_world": source, "ancestor": ancestor}
            for source in sources
            for ancestor in (100, 150, 200)
        ],
        "confirmations": confirmations,
    }
    result = tmp_path / "e7.json"
    export_json(result, raw)
    claim = publish(result, tmp_path / "claim.json", proof)
    assert claim["analysis"]["sources"] == 2
    assert claim["analysis"]["confirmed_fraction"] == 1.0
    assert claim["published_ledger_check_sha256"] == proof["proof_sha256"]
    # A missing world remains in the audited roster and must cause refusal.
    raw["confirmations"] = [row for row in confirmations if row["source_world"] == "world-1"]
    incomplete = tmp_path / "incomplete.json"
    export_json(incomplete, raw)
    with pytest.raises(ValueError, match="two confirmation rows"):
        publish(incomplete, tmp_path / "invalid-claim.json", proof)
    assert not (tmp_path / "invalid-claim.json").exists()
