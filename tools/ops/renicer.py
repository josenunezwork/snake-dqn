#!/usr/bin/env python3
"""Operator priority floor; exemptions come solely from the control classifier.

The factory process monitor remains observational. This separate operator tool
never raises priority and never reprioritizes the verified shared tmux server.
Unreadable identity stops this iteration before any priority mutation.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.bl_labels.factory_persistence import _native_identity, process_rows  # noqa: E402


def read(command: list[str]) -> str:
    """Use the same bounded full/native observation as factory admission."""
    return subprocess.check_output(command, text=True, errors="surrogateescape", timeout=10)


def ordinary_candidate(row: dict) -> bool:
    """Retain the operator's ordinary floor targets; never grant an exemption."""
    argv = row.get("argv", "")
    if any(
        token in argv
        for token in (
            "mac_driver",
            "multiprocessing.spawn",
            "RUN_D256",
            "/Applications/ChatGPT.app/",
            "/.codex/computer-use/",
            "Application Support/Codex",
        )
    ):
        return False
    return any(
        token in argv for token in ("colab_cli", "/.local/bin/colab", "codex-handoff", "codex exec")
    )


def targets(rows: list[dict]) -> list[int]:
    """Leave only positively admitted identities untouched, never their subtree."""
    return [
        row["pid"]
        for row in rows
        if row["nice"] < 15
        and not row.get("verified_tmux_server")
        and not row.get("control_identity")
        and (ordinary_candidate(row) or row.get("control_tree"))
    ]


def iteration(known: dict, *, dry_run: bool = False) -> dict:
    """Take a complete successful observation before acting on any target."""
    rows = process_rows(read, known)
    identities = {pid: _native_identity(pid) for pid in targets(rows)}
    for pid in targets(rows):
        if not dry_run:
            # Recheck birth/executable/argv immediately before the operator action.
            # A second complete classification also prevents stale exemption state.
            fresh = process_rows(read, known)
            if pid not in targets(fresh):
                continue
            before = identities[pid]
            after = _native_identity(pid)
            if (before["birth"], before["executable"]) != (after["birth"], after["executable"]):
                continue
            os.setpriority(os.PRIO_PROCESS, pid, 15)
    return {
        (row["pid"], row["birth"]): row
        for row in rows
        if row.get("control_tree") and row.get("birth")
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    known: dict = {}
    while True:
        try:
            known = iteration(known, dry_run=args.dry_run)
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError):
            # No raw argv, credentials or unrestricted RPC payloads in logs.
            print(
                "RENICER_IDENTITY_FAULT: no priority actions for unreadable sample", file=sys.stderr
            )
            return 2
        if args.once:
            return 0
        time.sleep(1)


if __name__ == "__main__":
    raise SystemExit(main())
