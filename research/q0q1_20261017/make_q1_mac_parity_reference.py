#!/usr/bin/env python3
"""Create the registered Q1 Mac parity reference; this script never queues work."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from research.q0q1_20261017.q1_evaluate import B8_SHA256, sha256


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--b8", type=Path, required=True, help="released B8 checkpoint")
    parser.add_argument("--out", type=Path, required=True, help="new immutable reference path")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error(f"refusing to replace existing reference: {args.out}")
    if not args.b8.is_file() or sha256(args.b8) != B8_SHA256:
        parser.error("--b8 does not match the registered released B8 SHA-256")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "research.q0q1_20261017.q1_parity",
            "--reference",
            "--bank",
            "research/q0q1_20261017/Q1_WORLD_BANK.json",
            "--b8",
            str(args.b8),
            "--out",
            str(args.out),
        ],
        check=False,
    )
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
