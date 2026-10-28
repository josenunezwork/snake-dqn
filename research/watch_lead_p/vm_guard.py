"""Test-only sitecustomize: persist a VM path guard in every spawned interpreter.

Copied into an isolated PYTHONPATH by test_uploaded; never shipped or installed
in production. Every interpreter reports startup and forbidden access by PID.
"""

import json
import os
import sys
from pathlib import Path

_BASE = Path(os.environ["P_VM_SCRATCH"])
_ALLOWED = [os.path.abspath(p) for p in json.loads(os.environ["P_VM_ALLOWED"])]
_REALPATH = os.path.realpath
_BUSY = False


def _within(path: str, root: str) -> bool:
    return path == root or path.startswith(root + os.sep)


def check(path: object, *, ancestors: bool = False) -> None:
    global _BUSY
    if _BUSY or not isinstance(path, (str, bytes, os.PathLike)):
        return
    _BUSY = True
    try:
        name = os.path.abspath(os.fsdecode(path))
        names = (name, _REALPATH(name))
        forbidden = next(
            (
                n
                for n in names
                if _within(n, "/Users")
                and not any(
                    _within(n, root) or (ancestors and _within(root, n)) for root in _ALLOWED
                )
            ),
            None,
        )
        if forbidden:
            with (_BASE / f"violation-{os.getpid()}.jsonl").open("a") as handle:
                handle.write(json.dumps({"path": forbidden}) + "\n")
            raise RuntimeError("VM forbids Mac path: " + forbidden)
    finally:
        _BUSY = False


def audit(event: str, args: tuple) -> None:
    if event in ("open", "os.listdir", "os.scandir", "os.chdir"):
        check(args[0])
    elif event == "subprocess.Popen":
        # Only guarded Python children may execute. Native children could read
        # arbitrary roots without Python audit hooks, even with an innocent cwd.
        if _REALPATH(os.fsdecode(args[0])) != _REALPATH(sys.executable):
            raise RuntimeError("VM guard refuses unguarded subprocess")
        if args[3] is not None and any(
            args[3].get(key) != os.environ.get(key)
            for key in ("PYTHONPATH", "P_VM_SCRATCH", "P_VM_ALLOWED")
        ):
            raise RuntimeError("VM guard refuses child guard environment removal")
        check(args[0])
        check(args[2])  # cwd; other Python children load this same sitecustomize.
        argv = args[1]
        if isinstance(argv, (tuple, list)):
            for arg in argv:
                if isinstance(arg, str) and arg.startswith("/Users/"):
                    check(arg)
    elif event == "os.system":
        raise RuntimeError("VM guard refuses unguarded shell execution")


def wrap(function):
    def guarded(path, *args, **kwargs):
        check(path, ancestors=True)
        return function(path, *args, **kwargs)

    return guarded


sys.addaudithook(audit)
os.stat, os.lstat, os.access = (wrap(os.stat), wrap(os.lstat), wrap(os.access))
(_BASE / f"guard-start-{os.getpid()}").write_text("sitecustomize active\n")
