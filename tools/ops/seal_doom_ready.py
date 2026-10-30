#!/usr/bin/env python3
"""Seal a DOOM fit using the actual final canary attempts in the ready queue."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

if __package__:
    from . import auto_retry, ready_queue
else:
    import auto_retry
    import ready_queue


def receipt(ops: Path, identity: str, *, h100: bool = False) -> Path:
    """Resolve retries without relabelling or copying a previous canary receipt."""
    queue = auto_retry.read(ops / "ready-queue.json")
    row = ready_queue.latest(queue, identity)
    allowed = {"PASS", "UNAVAILABLE"} if h100 else {"PASS"}
    if row["status"] not in allowed:
        raise ValueError("canary has no completed admissible queue evidence")
    name = row["argv"][1]
    # An exact allocation-accounting retry may have run inside dispatch.sh.
    internal = ops / "manifests" / (name + "-retry-1.json")
    manifest = internal if internal.exists() else Path(row["manifest"])
    return Path(auto_retry.read(manifest)["output_dir"]) / "receipt.json"


def main() -> int:
    """Delegate to the package's unchanged, same-pin admission function."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ops", type=Path, required=True)
    parser.add_argument("--broker", type=Path, required=True)
    parser.add_argument("--packet", type=Path, required=True)
    parser.add_argument("--admission-dir", type=Path, required=True)
    parser.add_argument("--h100-job", required=True)
    parser.add_argument("--a100-job", required=True)
    args = parser.parse_args()
    try:
        h100 = receipt(args.ops, args.h100_job, h100=True)
        a100 = receipt(args.ops, args.a100_job)
        subprocess.run(
            [
                sys.executable,
                "-c",
                "import json,sys; from pathlib import Path; sys.path.insert(0,sys.argv[1]); "
                "from research.doom_auc_smoke.prepare import admit; "
                "p,h,a,d=map(Path,sys.argv[2:]); "
                "admit(p,h,a,a100_only=json.loads(h.read_text())['status']!='PASS',"
                "admission_dir=d,prelook=p/'PRELOOK.json')",
                str(args.broker),
                str(args.packet),
                str(h100),
                str(a100),
                str(args.admission_dir),
            ],
            cwd=args.broker,
            check=True,
            timeout=600,
        )
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        print(f"REFUSE: {exc}", file=sys.stderr)
        return 1
    print("SEALED_NOT_DISPATCHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
