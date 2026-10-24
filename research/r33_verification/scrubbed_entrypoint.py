"""Invoke the real tiny lifecycle entrypoint under an explicit, scrubbed source root."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--namespace")
    parser.add_argument("--allocation")
    parser.add_argument("--pin")
    parser.add_argument("--environment-sha256")
    parser.add_argument("--packet")
    parser.add_argument("--source-closure")
    args = parser.parse_args()
    source = args.source.resolve(strict=True)
    if not (source / "research/r33_verification/lifecycle.py").is_file():
        raise SystemExit("refuse: explicit source root lacks R33 lifecycle entrypoint")
    os.environ.clear()
    os.environ.update({"PATH": "/usr/bin:/bin", "PYTHONDONTWRITEBYTECODE": "1"})
    # ``-I`` already removed the current directory and user site. Keep only its
    # interpreter-owned stdlib entries and add the explicit staged source root.
    sys.path[:] = [str(source), *[entry for entry in sys.path if entry]]
    if args.pilot:
        if not all(
            (
                args.ledger,
                args.namespace,
                args.allocation,
                args.pin,
                args.environment_sha256,
                args.packet,
                args.source_closure,
            )
        ):
            raise SystemExit(
                "refuse: pilot preflight needs exact staged ledger and frozen identity"
            )
        os.environ.update(
            {
                "COLAB_INPUT_DIR": str(args.ledger.parent),
                "COLAB_OUTPUT_DIR": str(args.root),
                "R33_NAMESPACE": args.namespace,
                "R33_ALLOCATION": args.allocation,
                "R33_PIN": args.pin,
                "R33_ENVIRONMENT_SHA256": args.environment_sha256,
                "R33_PACKET": args.packet,
                "R33_SOURCE_CLOSURE": args.source_closure,
            }
        )
        from research.r33_verification.pilot_runtime import main as pilot_main

        sys.argv = ["pilot_runtime", "--ledger", str(args.ledger), "--output", str(args.root)]
        pilot_main()
    else:
        from research.r33_verification.lifecycle import rehearse

        print(rehearse(args.root))


if __name__ == "__main__":
    main()
