"""Argv boundaries and desktop-app exclusions, without a process-name scan."""

from __future__ import annotations

import ctypes
import os
import sys
from pathlib import Path

import pytest

from research.compute.factory_companion_argv import classify_argv, decode_procargs, native_argv


@pytest.mark.parametrize(
    "root",
    [
        "/Applications/ChatGPT.app/",
        str(Path.home() / ".codex/computer-use") + "/",
        str(Path.home() / "Library/Application Support/Codex") + "/",
    ],
)
def test_desktop_helpers_are_excluded_by_executable_argv(root: str) -> None:
    assert classify_argv([root + "Codex (Service)", "codex", "exec"]) is None
    assert classify_argv(["/usr/bin/env", root + "codex", "exec"]) is None
    assert classify_argv(["/bin/bash", root + "codex-handoff.sh", "task"]) is None


@pytest.mark.parametrize(
    "argv",
    [
        ["/opt/homebrew/bin/codex", "exec", "task"],
        ["node", "/opt/homebrew/bin/codex", "exec", "task"],
        ["/bin/bash", "/Users/owner/.claude/bin/codex-handoff.sh", "-C", "/worktree"],
        ["env", "CODEX_MODEL=gpt-6.1-sol", "codex", "exec", "task"],
    ],
)
def test_real_cli_clients_require_registration(argv: list[str]) -> None:
    assert classify_argv(argv) == "unregistered_client"


def test_prompt_or_filename_is_not_an_executable() -> None:
    assert classify_argv(["cat", "codex exec python -m research.train"]) is None
    assert classify_argv(["rg", "codex exec", "research/"]) is None
    assert (
        classify_argv(["codex", "exec", "/Applications/ChatGPT.app/", "prompt"])
        == "unregistered_client"
    )
    assert classify_argv(["python3.12", "-c", "pass"]) == "unmanaged_compute"
    assert (
        classify_argv(["python", "-m", "research.compute.factory_companion", "status"])
        == "supervisor_helper"
    )


def test_procargs_preserves_spaces_and_omits_environment() -> None:
    argv = ["/path with spaces/codex", "exec", "prompt contains codex exec and spaces", ""]
    payload = (
        len(argv).to_bytes(ctypes.sizeof(ctypes.c_int), sys.byteorder, signed=True)
        + b"/actual/executable\0\0\0"
        + b"\0".join(word.encode() for word in argv)
        + b"\0SECRET=never-return\0"
    )
    assert decode_procargs(payload) == tuple(argv)


@pytest.mark.parametrize("payload", [b"", b"\0" * 8, b"\x01\0\0\0no-terminator"])
def test_invalid_native_buffers_fail_closed(payload: bytes) -> None:
    with pytest.raises(ValueError):
        decode_procargs(payload)


@pytest.mark.skipif(sys.platform != "darwin", reason="native Darwin argv")
def test_native_argv_reads_current_process_without_environment() -> None:
    argv = native_argv(os.getpid())
    assert argv and "python" in Path(argv[0]).name.lower()
    assert "pytest" in argv
    assert not any(word.startswith("HOME=") for word in argv)
