"""Offline admission tests for the packet checkout discovery boundary."""

from __future__ import annotations

import json
import logging
import os
import subprocess
from pathlib import Path

import pytest

from research.sandbox_a_resident import training_preflight


def _seal(packet: Path, prior: Path) -> None:
    packet.mkdir(parents=True, exist_ok=True)
    (packet / "SEAL.json").write_text(
        json.dumps(
            {
                "schema": "sandbox-a-train01-continuation-seal/v1",
                "prior_stop_evidence_kind": "receipt",
                "prior_packet": prior.name,
                "prior_packet_path": str(prior),
            }
        )
    )


@pytest.mark.parametrize("failed_call", [1, 2])
@pytest.mark.parametrize("error_type", [FileNotFoundError, PermissionError])
def test_git_operating_system_error_refuses_cleanly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed_call: int,
    error_type: type[OSError],
) -> None:
    packet = tmp_path / "packet"
    _seal(packet, tmp_path / "prior")
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _: None)
    calls = 0

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        nonlocal calls
        calls += 1
        if calls == failed_call:
            raise error_type("offline injected Git failure")
        return subprocess.CompletedProcess(command, 1, "", "")

    monkeypatch.setattr(training_preflight.subprocess, "run", run)
    with pytest.raises(ValueError, match="cannot inspect packet Git boundary"):
        training_preflight._validate_continuation_stop(packet, {"completion_attempt": {}})
    assert calls == failed_call


def test_missing_packet_working_directory_refuses_cleanly(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot inspect packet Git boundary"):
        training_preflight._validate_continuation_stop(tmp_path / "absent", {})


def test_git_probe_scrubs_namespace_without_mutating_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    redirects = {
        "GIT_DIR": "/foreign/.git",
        "GIT_WORK_TREE": "/foreign",
        "GIT_INDEX_FILE": "/foreign/index",
        "GIT_COMMON_DIR": "/foreign/common",
        "GIT_OBJECT_DIRECTORY": "/foreign/objects",
        "GIT_ALTERNATE_OBJECT_DIRECTORIES": "/foreign/alternate",
        "GIT_CEILING_DIRECTORIES": "/",
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": "core.worktree",
        "GIT_CONFIG_VALUE_0": "/foreign",
        "GIT_CONFIG_PARAMETERS": "redirect",
    }
    for key, value in redirects.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setenv("TRAIN01_TEST_ENV", "preserved")

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        env = kwargs["env"]
        assert isinstance(env, dict)
        assert not any(key.startswith("GIT_") for key in env)
        assert env["TRAIN01_TEST_ENV"] == "preserved"
        assert kwargs["cwd"] == tmp_path
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(training_preflight.subprocess, "run", run)
    assert training_preflight._packet_git(tmp_path, "rev-parse").returncode == 0
    assert {key: os.environ[key] for key in redirects} == redirects


@pytest.mark.parametrize("external", [False, True])
def test_nonzero_root_probe_logs_narrow_sibling_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    external: bool,
) -> None:
    packet = tmp_path / "packet"
    prior = tmp_path.parent / "foreign-prior" if external else tmp_path / "prior"
    _seal(packet, prior)
    monkeypatch.setattr(training_preflight, "verify_committed_manifest", lambda _: None)

    def run(command: list[str], **kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 128, "", "not a checkout")

    monkeypatch.setattr(training_preflight.subprocess, "run", run)
    original = training_preflight._regular_packet_file

    def check(path: Path) -> None:
        if path.name == "dispatch-manifest.json":
            raise RuntimeError("reached sibling prior manifest")
        original(path)

    monkeypatch.setattr(training_preflight, "_regular_packet_file", check)
    with caplog.at_level(logging.WARNING, logger=training_preflight.__name__):
        if external:
            with pytest.raises(ValueError, match="prior_packet_path differs"):
                training_preflight._validate_continuation_stop(packet, {"completion_attempt": {}})
        else:
            with pytest.raises(RuntimeError, match="reached sibling prior manifest"):
                training_preflight._validate_continuation_stop(packet, {"completion_attempt": {}})
    assert "Git root probe exited 128" in caplog.text
    assert f"restricting prior packet boundary to {tmp_path}" in caplog.text


def test_real_git_probe_ignores_foreign_checkout_redirect(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local = tmp_path / "local"
    foreign = tmp_path / "foreign"
    for repo in (local, foreign):
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
    monkeypatch.setenv("GIT_DIR", str(foreign / ".git"))
    monkeypatch.setenv("GIT_WORK_TREE", str(foreign))
    probe = training_preflight._packet_git(local, "rev-parse", "--show-toplevel")
    assert probe.returncode == 0
    assert Path(probe.stdout.strip()).resolve() == local.resolve()
