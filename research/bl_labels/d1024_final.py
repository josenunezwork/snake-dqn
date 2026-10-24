"""Publish the final outcome-blind D1024 union after the collector's COMPLETE terminal.

This mirrors the D256 final union (``d256-union-final-01``): it verifies the
whole original attempt chain (every earlier attempt an operational PREEMPTED,
the last the driver's COMPLETE with no pending parents), unions only sealed
immutable triplets by path/size/hash and wrapper identity, and proves that no
allocated parent remains pending. It never opens rows, outcomes or sealed rates.

A chain whose last terminal is not COMPLETE (for example an exhausted wall cap)
is a HOLD here: declaring such a tranche final needs its own operator ruling.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import d1024_resume, mac_driver, mac_gates

FINAL_DIR = d1024_resume.ART / "bl-labels-f0-speed/d1024-union-final-01"
PUBLICATION_PATH = "research/bl_labels/ledger/D1024_UNION_FINAL_20261024.json"


def publish_final(pack: Path, out: Path) -> dict:
    """Create the portable final union; refuse any incomplete or foreign chain."""
    if out.exists() or Path(str(out) + ".complete").exists():
        raise FileExistsError("final D1024 union is create-only; verify the existing one")
    control = d1024_resume.original_control(pack)
    attempts, elapsed = d1024_resume.attempt_chain(pack, control, final=True)
    value = mac_gates.publish_ledger(pack, attempts, out, control=control)
    d1024_resume.verify_union(out, control, attempts)
    _, _, pending = mac_driver._prepare_jobs(control, out)
    if pending:
        raise ValueError("final D1024 union leaves allocated parents pending; HOLD")
    return {
        "ledger": str(out),
        "sha256": mac_gates._verify(out),
        "attempts": len(attempts),
        "units": len(value["units"]),
        "cumulative_elapsed_seconds": elapsed,
        "pending_parents": 0,
    }


def verify_final(pack: Path, ledger: Path) -> dict:
    """Recheck an existing final union against the same immutable chain."""
    control = d1024_resume.original_control(pack)
    attempts, elapsed = d1024_resume.attempt_chain(pack, control, final=True)
    d1024_resume.verify_union(ledger, control, attempts)
    _, _, pending = mac_driver._prepare_jobs(control, ledger)
    if pending:
        raise ValueError("final D1024 union leaves allocated parents pending; HOLD")
    return {"ledger": str(ledger), "sha256": mac_gates._verify(ledger), "attempts": len(attempts)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=d1024_resume.PACK)
    parser.add_argument("--out", type=Path, default=FINAL_DIR / "verified-ledger.json")
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_final(args.pack, args.out)))
    else:
        print(json.dumps(publish_final(args.pack, args.out)))


if __name__ == "__main__":
    main()
