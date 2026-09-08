"""Tiny subprocess targets used by Ape-X supervisor regressions."""

from __future__ import annotations

import argparse
import signal
import time


def main() -> None:
    """Exit promptly or ignore termination until the test hard deadline."""
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("exit", "ignore-term"))
    args = parser.parse_args()
    if args.mode == "exit":
        raise SystemExit(17)
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
    print("ready", flush=True)
    while True:
        time.sleep(0.05)


if __name__ == "__main__":
    main()
