"""Start C/T status telemetry before importing torch or the C/T producer."""

from __future__ import annotations

import argparse
import runpy
import sys

from research.sandbox_a_resident.ct_liveness import finish_status, start_status


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("runner", "canary"))
    args, producer_args = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    from pathlib import Path

    output_parser.add_argument("--output", type=Path, required=True)
    output, _ = output_parser.parse_known_args(producer_args)
    start_status(output.output)
    try:
        sys.argv = [sys.argv[0], *producer_args]
        module = "ct_run" if args.target == "runner" else "ct_cuda_canary"
        runpy.run_module(f"research.sandbox_a_resident.{module}", run_name="__main__")
    finally:
        finish_status()


if __name__ == "__main__":
    main()
