"""Start C/T status telemetry before importing torch or the C/T producer."""

from __future__ import annotations

import argparse
import runpy
import sys
from pathlib import Path

from research.sandbox_a_resident.ct_liveness import finish_status, start_status
from research.sandbox_a_resident.stall_watchdog import StallWatchdog


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("target", choices=("runner",))
    args, producer_args = parser.parse_known_args()
    output_parser = argparse.ArgumentParser(add_help=False)
    output_parser.add_argument("--output", type=Path, required=True)
    output_parser.add_argument("--stall-timeout-seconds", type=float, default=900.0)
    output_parser.add_argument("--stall-diagnostics", type=Path, default=None)
    output, _ = output_parser.parse_known_args(producer_args)
    # Arm before any torch/numba imports. Handoff closes this timer only after
    # imports finish, before ct_run.main arms its own meaningful-work deadline.
    diagnostics = output.stall_diagnostics or output.output / "stall-watchdog"
    import_guard = StallWatchdog(diagnostics, output.stall_timeout_seconds)
    start_status(output.output)
    try:
        import_guard.start("imports")
        sys.argv = [sys.argv[0], *producer_args]
        module = "ct_stage2.run"
        namespace = runpy.run_module(f"research.sandbox_a_resident.{module}")
        import_guard.close()
        raise SystemExit(namespace["main"]())
    finally:
        import_guard.close()
        finish_status()


if __name__ == "__main__":
    main()
