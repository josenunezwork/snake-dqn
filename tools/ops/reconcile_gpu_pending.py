#!/usr/bin/env python3
"""Preview or retire a proven unassigned H100 pending intent; never allocate/stop."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import dispatch
else:
    import dispatch


def main() -> int:
    """Require an explicit mode and exact paths; live always repeats the proof."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--pending", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--proof", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--live", action="store_true")
    args = parser.parse_args()
    try:
        if not args.pending.is_absolute() or args.pending.is_symlink():
            raise dispatch.Refused("pending must be an absolute non-symlink account record")
        if args.proof.exists() or args.proof.is_symlink():
            raise dispatch.Refused("proof archive already exists; choose a fresh path")
        ops = Path(os.environ.get("OPS_DIR", dispatch.OPS)).resolve()
        broker = args.broker.resolve()
        # Same lock order as fallback: ops allocation exclusion, then account
        # broker exclusion. Only GET recovery and read-only sessions run here.
        with dispatch.allocation_lock(ops / "alloc.lock", 180) as deadline:
            dispatch.qualified(broker, ops, deadline)
            dispatch.qualified(
                broker, ops, deadline, ref=dispatch.read_json(args.manifest)["commit"]
            )
            evidence = dispatch.reconcile_unavailable(
                broker,
                args.manifest.resolve(),
                args.state_dir.resolve(),
                args.log.resolve(),
                args.proof.resolve(),
                deadline,
                dry_run=args.dry_run,
                expected_pending=args.pending,
            )
        print(json.dumps(evidence, sort_keys=True, indent=2))
    except (
        OSError,
        ValueError,
        TypeError,
        KeyError,
        TimeoutError,
        subprocess.SubprocessError,
    ) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
