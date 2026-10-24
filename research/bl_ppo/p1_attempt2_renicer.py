"""Prepared replacement for the interim renicer; never install or start automatically.

Its P1 exclusion is rooted in a PID and follows PPID ancestry on every scan, so
new descendants of the admitted supervisor remain excluded without argv trust.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from research.bl_ppo.p1_attempt2_window import (
    TIMING_ARMED,
    descendants,
    ps_rows,
    read_registered_control,
)


def eligible(rows: list[dict[str, Any]], p1_root_pids: int | Iterable[int]) -> list[int]:
    """Select legacy-policy candidates while excluding the entire P1 PID tree."""
    roots = {p1_root_pids} if isinstance(p1_root_pids, int) else set(p1_root_pids)
    excluded = {row["pid"] for root in roots for row in descendants(rows, root)}
    selected: list[int] = []
    for row in rows:
        command = str(row["command"])
        if row["pid"] in excluded or row["nice"] >= 15:
            continue
        if any(
            marker in command
            for marker in (
                "mac_driver",
                "multiprocessing.spawn",
                "RUN_D256",
                "/Applications/ChatGPT.app/",
                "/.codex/computer-use/",
                "Application Support/Codex",
            )
        ):
            continue
        if any(
            marker in command
            for marker in ("colab_cli", "/.local/bin/colab", "codex-handoff", "codex exec")
        ):
            selected.append(int(row["pid"]))
    return selected


def pending_roots(
    rows: list[dict[str, Any]], *, root_records: list[Path], arm_path: Path
) -> set[int]:
    """Return validated roots plus tightly scoped roots awaiting their first record.

    Accounting allocation retries intentionally receive a fresh record path; a
    replacement renicer therefore accepts multiple immutable records and keeps
    every still-live control tree excluded until its verified release.
    """
    roots: set[int] = set()
    expected_paths = {str(path.resolve()) for path in root_records}
    if not arm_path.exists():
        return roots  # Restore the ordinary policy once the owned window closes.
    arm = json.loads(arm_path.read_text())
    if arm.get("kind") != "arm" or not arm.get("owner_token"):
        raise RuntimeError("P1 exclusion requires an owned maintenance arm")
    for root_record in root_records:
        if root_record.exists():
            old_pid = json.loads(root_record.read_text()).get("root", {}).get("pid")
            if not any(row["pid"] == old_pid for row in rows):
                continue  # A released prior stage must not break the next stage.
            roots.add(
                read_registered_control(root_record=root_record, arm_path=arm_path, rows=rows)
            )
    if not arm_path.is_file():
        raise RuntimeError("no P1 arm exists for pending control-root exclusion")
    pending = [
        row
        for row in rows
        if "p1_attempt2_control" in str(row["command"])
        and any(expected in str(row["command"]) for expected in expected_paths)
        and row["nice"] == 0
        and row["pid"] not in roots
    ]
    # Multiple pending roots are valid only for the explicitly enumerated
    # original/retry paths; each is still a scoped P1 root.
    roots.update(int(row["pid"]) for row in pending)
    return roots


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-record", type=Path, required=True, action="append")
    parser.add_argument("--arm-path", type=Path, default=TIMING_ARMED)
    parser.add_argument("--check", action="store_true", help="print scope only; never renice")
    args = parser.parse_args()
    while True:
        rows = ps_rows()
        root_pids = pending_roots(rows, root_records=args.root_record, arm_path=args.arm_path)
        targets = eligible(rows, root_pids)
        if args.check:
            print({"excluded_roots": sorted(root_pids), "renice_targets": targets})
            return
        for pid in targets:
            try:
                os.setpriority(os.PRIO_PROCESS, pid, 15)
            except ProcessLookupError:
                pass
        time.sleep(1)


if __name__ == "__main__":
    main()
