"""Explicit, one-time setup of the Tier-1/dev-only third CPU slot lock (``cpu-slot-3.lock``).

``dev_screen --slot-pool 3`` opens slot lock files read-only and never creates them, so the
third slot exists only after this step is run on purpose (docs/research/
compute_policy_2026-10-02.md). Strict Tier-2 packages never read it: they hold
``cpu-slot-1.lock`` and ``cpu-slot-2.lock`` only. This module refuses to create or touch
slots 1 and 2, and creates slot 3 create-only (an existing file is left untouched).

Usage (dry-run by default; ``--create`` writes the empty lock file)::

  ./venv/bin/python research/compute/slot_setup.py \
    --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909 --create
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Sequence

THIRD_SLOT_LOCK_FILE = "cpu-slot-3.lock"
STRICT_SLOT_LOCK_FILES = ("cpu-slot-1.lock", "cpu-slot-2.lock")


def third_slot_status(root: Path) -> Dict[str, Any]:
    """Which slot lock files exist under ``root`` (read-only)."""
    root = Path(root)
    return {
        "root": str(root),
        "strict_slots_present": {name: (root / name).is_file() for name in STRICT_SLOT_LOCK_FILES},
        "third_slot": str(root / THIRD_SLOT_LOCK_FILE),
        "third_slot_present": (root / THIRD_SLOT_LOCK_FILE).is_file(),
    }


def create_third_slot(root: Path) -> Dict[str, Any]:
    """Create ``root/cpu-slot-3.lock`` (empty, create-only); returns what happened.

    The root must already exist and already hold both strict slot files, so this cannot
    mint a slot pool in a wrong directory. An existing slot-3 file is reported, not rewritten.
    """
    status = third_slot_status(root)
    if not Path(root).is_dir():
        raise FileNotFoundError(f"slot lock root {root} does not exist")
    missing = [name for name, ok in status["strict_slots_present"].items() if not ok]
    if missing:
        raise FileNotFoundError(f"refusing: strict slot files missing under {root}: {missing}")
    target = Path(root) / THIRD_SLOT_LOCK_FILE
    try:
        with target.open("x", encoding="utf-8"):
            pass
        status["created"] = True
    except FileExistsError:
        status["created"] = False
    status["third_slot_present"] = target.is_file()
    return status


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--root", type=Path, required=True, help="the shared slot lock root")
    parser.add_argument("--create", action="store_true", help="create cpu-slot-3.lock")
    args = parser.parse_args(argv)
    try:
        result = create_third_slot(args.root) if args.create else third_slot_status(args.root)
    except OSError as exc:
        print(json.dumps({"error": str(exc)}))
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
