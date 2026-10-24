"""Darwin argv vectors and narrow executable recognition for companion guards.

Never search a CLI prompt for a process name. KERN_PROCARGS2 preserves argument
boundaries, including executable paths containing spaces; ps's args rendering
cannot provide those boundaries.
"""

from __future__ import annotations

import ctypes
import re
import sys
from pathlib import Path
from typing import Sequence


def _desktop(executable: str) -> bool:
    return executable.startswith(
        (
            "/Applications/ChatGPT.app/",
            str(Path.home() / ".codex/computer-use") + "/",
            str(Path.home() / "Library/Application Support/Codex") + "/",
        )
    )


def decode_procargs(payload: bytes) -> tuple[str, ...]:
    """Decode only argc argv entries; environment bytes never leave this call."""
    width = ctypes.sizeof(ctypes.c_int)
    if len(payload) <= width:
        raise ValueError("truncated native argv observation")
    argc = int.from_bytes(payload[:width], sys.byteorder, signed=True)
    if argc <= 0 or argc > 65536:
        raise ValueError("invalid native argc")
    end_executable = payload.find(b"\0", width)
    if end_executable < 0:
        raise ValueError("missing native executable terminator")
    offset = end_executable + 1
    while offset < len(payload) and payload[offset] == 0:
        offset += 1
    words: list[str] = []
    for _ in range(argc):
        end = payload.find(b"\0", offset)
        if end < 0:
            raise ValueError("truncated native argv vector")
        words.append(payload[offset:end].decode("utf-8", errors="surrogateescape"))
        offset = end + 1
    return tuple(words)


def native_argv(pid: int) -> tuple[str, ...]:
    """Read one native argv vector; refuse missing/foreign-permission observations."""
    if sys.platform != "darwin" or type(pid) is not int or pid <= 0:
        raise OSError("native argv requires a positive Darwin PID")
    libc = ctypes.CDLL(None, use_errno=True)
    call = libc.sysctl
    call.argtypes = [
        ctypes.POINTER(ctypes.c_int),
        ctypes.c_uint,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_size_t),
        ctypes.c_void_p,
        ctypes.c_size_t,
    ]
    call.restype = ctypes.c_int
    mib = (ctypes.c_int * 3)(1, 49, pid)  # CTL_KERN, KERN_PROCARGS2 (Darwin SDK).
    size = ctypes.c_size_t(0)
    if call(mib, 3, None, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "native argv sizing failed")
    if not 0 < size.value <= 4 * 1024**2:
        raise OSError("native argv observation exceeds bounded buffer")
    buffer = ctypes.create_string_buffer(size.value)
    if call(mib, 3, buffer, ctypes.byref(size), None, 0) != 0:
        raise OSError(ctypes.get_errno(), "native argv read failed")
    return decode_procargs(buffer.raw[: size.value])


def classify_argv(argv: Sequence[str]) -> str | None:
    """Classify executable operands, ignoring all prompt/content arguments."""
    if not argv:
        return None
    words = list(argv)
    if _desktop(words[0]):
        return None
    name = Path(words[0]).name.lower()
    if name == "env":
        index = 1
        while index < len(words):
            token = words[index]
            if token in {"-u", "--unset"}:
                index += 2
            elif token == "--" or "=" in token:
                index += 1
            elif token.startswith("-"):
                return "unmanaged_compute"
            else:
                break
        return classify_argv(words[index:])
    if name in {"codex", "claude", "codex-handoff.sh", "codex-role"}:
        return "unregistered_client"
    if name in {"bash", "sh", "zsh"} and len(words) > 1:
        script = words[1]
        if _desktop(script):
            return None
        if Path(script).name in {"codex-handoff.sh", "codex-role"}:
            return "unregistered_client"
        if not script.startswith("-") and (
            "research" in Path(script).parts or "train" in Path(script).name
        ):
            return "unmanaged_compute"
        return None
    if name == "node" and len(words) > 1:
        script = words[1]
        if _desktop(script):
            return None
        if Path(script).name in {"codex", "codex.js", "claude", "cli.js"}:
            return "unregistered_client"
        return "unmanaged_compute"
    if re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", name):
        if words[1:3] == ["-m", "research.compute.factory_companion"]:
            return "supervisor_helper"
        return "unmanaged_compute"
    if name in {
        "pytest",
        "npm",
        "npx",
        "make",
        "cmake",
        "ninja",
        "ruff",
        "black",
        "flake8",
        "isort",
        "clang",
        "clang++",
        "gcc",
        "g++",
        "cargo",
        "rustc",
    }:
        return "unmanaged_compute"
    return None
