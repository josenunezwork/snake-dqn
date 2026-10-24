#!/usr/bin/env python3
"""Native-identity lifecycle for the one persistent factory companion monitor."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research.compute.factory_companion_argv import native_argv
from research.compute.factory_companion_native import usage
from research.compute.factory_companion_qualification import require_activation
from research.mac_lease import default_lease_path

POLICY = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "JUDGMENT_FACTORY_CONCURRENCY_20261023.md"
)
POLICY_SHA = "af117cd98d51f428cd83d6e33138fcd13def0b4d7c7cf63a8b2e2b83db99105f"
ROOT = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/factory-companions")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _paths(root: Path) -> tuple[Path, Path]:
    return root / "monitor-lifecycle.lock", root / "monitor-lifecycle.json"


def _read_record(path: Path) -> dict[str, Any]:
    try:
        record = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid retained monitor record: {path}") from error
    required = {"pid", "start_tick", "argv", "policy_sha256", "repo", "root"}
    if not isinstance(record, dict) or not required <= record.keys():
        raise RuntimeError(f"invalid retained monitor record: {path}")
    return record


def _expected_argv(python: str, root: Path, repo: Path) -> tuple[str, ...]:
    return (
        python,
        "-m",
        "research.compute.factory_companion",
        "--root",
        str(root),
        "--policy-sha",
        POLICY_SHA,
        "--lease-identity",
        "pending",
        "--worktree",
        str(repo),
        "monitor",
    )


def _verified(record: dict[str, Any], *, python: str, root: Path, repo: Path) -> bool:
    try:
        native = usage(int(record["pid"]))
        argv = native_argv(native.pid)
    except (OSError, RuntimeError, ValueError, ProcessLookupError):
        return False
    return (
        native.start_tick == int(record["start_tick"])
        and argv == _expected_argv(python, root, repo)
        and tuple(record["argv"]) == argv
        and record["policy_sha256"] == POLICY_SHA
        and record["repo"] == str(repo)
        and record["root"] == str(root)
    )


def _registry_busy(root: Path) -> bool:
    registry = root / "registry.json"
    if not registry.exists():
        return False
    try:
        state = json.loads(registry.read_text())
        records = state.get("registrations", {}).values()
    except (OSError, json.JSONDecodeError, AttributeError) as error:
        raise RuntimeError("cannot determine active companion ownership") from error
    return any(
        item.get("status") == "running" and item.get("class") in {"client", "light", "test"}
        for item in records
        if isinstance(item, dict)
    )


def _core_status(python: str, root: Path, repo: Path) -> int:
    command = list(_expected_argv(python, root, repo)[:-1]) + ["status"]
    return subprocess.call(command, env={**os.environ, "PYTHONPATH": str(repo)})


def start(*, python: str, root: Path, repo: Path) -> int:
    # This is deliberately before every lifecycle side effect, including lock
    # creation and monitor spawn. Status remains the read-only diagnostic path.
    require_activation()
    lock_path, record_path = _paths(root)
    root.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            record = _read_record(record_path)
            if not _verified(record, python=python, root=root, repo=repo):
                raise RuntimeError(
                    "monitor lock is live but its retained identity is not safely reusable"
                )
            print(json.dumps({"status": "MONITOR_REUSED", "pid": record["pid"]}, sort_keys=True))
            return 0
        if record_path.exists():
            raise RuntimeError("retained monitor record requires operator inspection")
        command = list(_expected_argv(python, root, repo))
        child = subprocess.Popen(
            command,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(repo)},
            pass_fds=(lock.fileno(),),
        )
        try:
            native = usage(child.pid)
            argv = native_argv(child.pid)
        except (OSError, RuntimeError, ValueError, ProcessLookupError) as error:
            child.terminate()
            child.wait(timeout=2)
            raise RuntimeError("monitor failed native identity capture") from error
        if argv != tuple(command):
            child.terminate()
            child.wait(timeout=2)
            raise RuntimeError("monitor argv did not match its approved native vector")
        record = {
            "pid": child.pid,
            "start_tick": native.start_tick,
            "argv": list(argv),
            "policy_sha256": POLICY_SHA,
            "repo": str(repo),
            "root": str(root),
        }
        temporary = record_path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, sort_keys=True) + "\n")
        os.replace(temporary, record_path)
        print(json.dumps({"status": "MONITOR_STARTED", "pid": child.pid}, sort_keys=True))
        return 0


def stop(*, python: str, root: Path, repo: Path) -> int:
    lock_path, record_path = _paths(root)
    record = _read_record(record_path)
    if not _verified(record, python=python, root=root, repo=repo):
        raise RuntimeError("refuse stop: monitor identity or argv no longer matches")
    if default_lease_path().exists() or _registry_busy(root):
        raise RuntimeError("refuse stop: Mac lease or companion client/test is still active")
    os.kill(int(record["pid"]), signal.SIGTERM)
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        if not _verified(record, python=python, root=root, repo=repo):
            with lock_path.open("a+") as lock:
                try:
                    fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise RuntimeError("monitor lock remained held after termination") from error
                if _read_record(record_path) != record:
                    raise RuntimeError("retained monitor record changed during stop")
                record_path.unlink()
                print(
                    json.dumps({"status": "MONITOR_STOPPED", "pid": record["pid"]}, sort_keys=True)
                )
                return 0
        time.sleep(0.1)
    raise RuntimeError("monitor did not quiesce after TERM; retained record preserved")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("action", choices=("start", "status", "stop"))
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not (repo / ".git").exists() or _sha256(POLICY) != POLICY_SHA:
        raise SystemExit("refuse: repository or binding policy is unavailable")
    if args.action == "status":
        return _core_status(sys.executable, args.root, repo)
    if args.action == "start":
        return start(python=sys.executable, root=args.root, repo=repo)
    return stop(python=sys.executable, root=args.root, repo=repo)


if __name__ == "__main__":
    raise SystemExit(main())
