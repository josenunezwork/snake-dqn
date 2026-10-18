#!/usr/bin/env python3
"""Mac-side control plane for the Drive-backed Colab queue.

All commands print JSON, making the small tool suitable for a terminal or a
lightweight monitor.  ``--drive-root`` is the *t16_colab* Drive root; queue
metadata is always below ``<drive-root>/queue`` and job outputs are below the
same Drive root.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

try:
    from .core import Queue, QueueError, mac_writer_lock
except ImportError:
    from core import Queue, QueueError, mac_writer_lock


DEFAULT_DRIVE_ROOT = Path(
    "/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab"
)


def queue_from_args(args: argparse.Namespace) -> Queue:
    return Queue(Path(args.drive_root) / "queue")


def emit(value: Any) -> None:
    print(json.dumps(value, sort_keys=True, indent=2, default=str), flush=True)


def read_spec(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise QueueError(f"cannot read job spec {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise QueueError("job spec JSON must be an object")
    return value


def list_status(queue: Queue, stale_seconds: int) -> list[dict[str, Any]]:
    return [queue.job_status(job_id, stale_seconds) for job_id in sorted(queue.load_jobs())]


def command_enqueue(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    with mac_writer_lock(queue):
        job = queue.enqueue(read_spec(args.spec))
    emit({"enqueued": job, "queue_root": str(queue.root)})
    return 0


def command_list(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    emit({"jobs": list_status(queue, args.stale_seconds)})
    return 0


def command_status(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    if args.id:
        emit(queue.job_status(args.id, args.stale_seconds))
    else:
        emit(
            {
                "jobs": list_status(queue, args.stale_seconds),
                "heartbeats": queue.heartbeat_status(args.stale_seconds),
            }
        )
    return 0


def command_cancel(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    with mac_writer_lock(queue):
        control = queue.cancel(args.id, args.reason)
    emit({"control": control})
    return 0


def command_requeue(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    with mac_writer_lock(queue):
        control = queue.requeue(
            args.id, stale_owner_ack=args.stale_owner_stopped, reason=args.reason
        )
    emit({"control": control, "warning": "old receipts are retained but fenced by generation"})
    return 0


def command_watch(args: argparse.Namespace) -> int:
    queue = queue_from_args(args)
    while True:
        emit(
            {
                "observed_at": time.time(),
                "jobs": list_status(queue, args.stale_seconds),
                "heartbeats": queue.heartbeat_status(args.stale_seconds),
            }
        )
        if args.once:
            return 0
        time.sleep(args.interval_seconds)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--drive-root",
        type=Path,
        default=Path(os.environ.get("COLAB_QUEUE_DRIVE_ROOT", DEFAULT_DRIVE_ROOT)),
    )
    sub = result.add_subparsers(dest="command", required=True)

    enqueue = sub.add_parser("enqueue", help="validate and publish a job JSON spec")
    enqueue.add_argument("spec", type=Path)
    enqueue.set_defaults(func=command_enqueue)

    listing = sub.add_parser("list", help="show each job state")
    listing.add_argument("--stale-seconds", type=int, default=300)
    listing.set_defaults(func=command_list)

    status = sub.add_parser("status", help="show a job or whole queue, claims, and heartbeats")
    status.add_argument("id", nargs="?")
    status.add_argument("--stale-seconds", type=int, default=300)
    status.set_defaults(func=command_status)

    cancel = sub.add_parser("cancel", help="fence a job's current owner")
    cancel.add_argument("id")
    cancel.add_argument("--reason", default="cancelled by operator")
    cancel.set_defaults(func=command_cancel)

    requeue = sub.add_parser(
        "requeue", help="start a new fenced generation after failure/cancellation/stale claim"
    )
    requeue.add_argument("id")
    requeue.add_argument(
        "--stale-owner-stopped",
        action="store_true",
        help="required acknowledgement that a stale runtime was stopped",
    )
    requeue.add_argument("--reason", default="")
    requeue.set_defaults(func=command_requeue)

    watch = sub.add_parser("watch", help="emit JSON snapshots until interrupted")
    watch.add_argument("--stale-seconds", type=int, default=300)
    watch.add_argument("--interval-seconds", type=float, default=10)
    watch.add_argument("--once", action="store_true")
    watch.set_defaults(func=command_watch)
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if hasattr(args, "stale_seconds") and args.stale_seconds < 1:
        raise SystemExit("--stale-seconds must be positive")
    if hasattr(args, "interval_seconds") and args.interval_seconds <= 0:
        raise SystemExit("--interval-seconds must be positive")
    try:
        return args.func(args)
    except QueueError as exc:
        print(f"q.py: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
