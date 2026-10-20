"""Hard wall-clock supervisor for one local T34 measurement, without tmux."""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time

from research.t34_shadow.common import ARTIFACTS, ROOT, THREAD_VARS, write_once


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--seconds", type=float, default=1800)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    if not 10 <= args.seconds <= 1800:
        parser.error("hard wall limit must be in [10, 1800]")
    if not args.name or any(c not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for c in args.name):
        parser.error("name must contain only lowercase ASCII letters, digits, hyphens, underscores")
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a measurement command is required after --")
    log_path = ARTIFACTS / "launches" / f"{args.name}.log"
    receipt = log_path.with_suffix(".json")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if receipt.exists() or log_path.exists():
        parser.error("create-only launch name already exists")
    env = {**os.environ, **{key: "1" for key in THREAD_VARS}, "SNAKE_DQN_DEVICE": "cpu"}
    started = time.monotonic()
    with log_path.open("x") as stream:
        child = subprocess.Popen(
            command,
            cwd=ROOT,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        timed_out = False
        try:
            code = child.wait(timeout=args.seconds - 5)
        except subprocess.TimeoutExpired:
            timed_out = True
            # Only the exact child process group created above belongs to this
            # measurement. Never enumerate, signal, or stop any owner's process.
            os.killpg(child.pid, signal.SIGTERM)
            try:
                code = child.wait(timeout=4)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                code = child.wait(timeout=1)
    write_once(
        receipt,
        {
            "schema": "t34-hard-wall-supervisor/v1",
            "argv": sys.argv,
            "command": command,
            "pid": child.pid,
            "status": "TIMEOUT" if timed_out else "EXITED",
            "returncode": code,
            "wall_seconds": time.monotonic() - started,
            "hard_limit_seconds": args.seconds,
            "log": str(log_path),
            "threads": {key: env[key] for key in THREAD_VARS},
        },
    )
    print(f"{args.name}: {'TIMEOUT' if timed_out else code}; log={log_path}", flush=True)
    return 124 if timed_out else (code if code >= 0 else 128 - code)


if __name__ == "__main__":
    raise SystemExit(main())
