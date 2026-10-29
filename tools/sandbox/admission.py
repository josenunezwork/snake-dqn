"""Host admission test receipts and detached allocation-lock handoff.

The qualified transport is unchanged. Test reuse never replaces structural
admission, input hashing, or the native broker's accounting/cleanup checks.
"""

from __future__ import annotations

import contextlib
import fcntl
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any, Iterator

TEST_RECEIPT_ENV = "SNAKE_SANDBOX_TEST_RECEIPT"
LEASE_ENV = "SNAKE_SANDBOX_ALLOCATION_LEASE"
REQUIRE_RECEIPT_ENV = "SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT"
VALIDATION_SECONDS = 1200.0  # Base 600s, effective 2x; before allocation.
TEST_SECONDS = 600.0  # Base 300s (262s measured suite), effective 2x.


def environment(receipt: Path, *, require: bool = False) -> dict[str, str]:
    """Give validation and tmux the same explicit test/import environment."""
    result = {
        "HOME": str(Path.home()),
        "PATH": os.environ["PATH"],
        "SNAKE_DQN_PYTHON": "/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python",
        "PYTHONDONTWRITEBYTECODE": "1",
        "SNAKE_DQN_DEVICE": "cpu",
        TEST_RECEIPT_ENV: str(receipt),
    }
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
        "VECLIB_MAXIMUM_THREADS",
        "TORCHINDUCTOR_COMPILE_THREADS",
    ):
        result[key] = "1"
    if require:
        result[REQUIRE_RECEIPT_ENV] = "1"
    return result


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def test_identity(broker: Path, original: dict[str, Any], admission: Path) -> str:
    """Bind tracked code/config bytes and local inputs, plus the test environment.

    Hashes are metadata-only evidence, never a scientific look. Ignored Python
    caches cannot supply source modules; nonignored untracked files or dirty
    tracked bytes refuse reuse. Inputs outside Git are hashed independently.
    """
    from tools.ops.processes import run

    def git(*args: str) -> str:
        return run(
            ["git", *args], cwd=broker, timeout=30, check=True, capture_output=True, text=True
        ).stdout

    if git("status", "--porcelain", "--untracked-files=all").strip():
        raise ValueError("sandbox test reuse requires a clean broker checkout")
    files = git("ls-files", "-z").split("\0")
    suffixes = {".py", ".sh", ".toml", ".cfg", ".ini", ".yaml", ".yml"}
    tracked = [
        (name, digest(broker / name))
        for name in files
        if name
        and (
            Path(name).suffix in suffixes
            or Path(name).parent == admission.parent.relative_to(broker)
        )
    ]
    inputs = [(item["path"], digest(Path(item["path"]))) for item in original.get("inputs", [])]
    value = {
        "schema": "sandbox-admission-test-identity/v1",
        "repo": str(broker.resolve()),
        "head": git("rev-parse", "HEAD").strip(),
        "tracked": tracked,
        "inputs": inputs,
        "admission": str(admission.resolve()),
        "python": str(Path(sys.executable).resolve()),
        "environment": sorted(os.environ.items()),
    }
    # These host routing variables are expected to differ between validation
    # and detached startup. They neither change suite selection nor job inputs.
    for key in (
        TEST_RECEIPT_ENV,
        LEASE_ENV,
        REQUIRE_RECEIPT_ENV,
        "SNAKE_SANDBOX_COLLECTOR_DIR",
        "SHLVL",
        "_",
        "PWD",
        "OLDPWD",
        "TMUX",
        "TMUX_PANE",
        "TERM",
        "TERM_PROGRAM",
    ):
        value["environment"] = [item for item in value["environment"] if item[0] != key]
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def cached_tests(identity: str) -> bool:
    path = os.environ.get(TEST_RECEIPT_ENV)
    if not path or not Path(path).exists():
        if os.environ.get(REQUIRE_RECEIPT_ENV):
            raise ValueError("completed sandbox admission test receipt required before allocation")
        return False
    receipt = json.loads(Path(path).read_text())
    if receipt != {"schema": "sandbox-admission-tests/v1", "identity": identity, "status": "PASS"}:
        raise ValueError(
            "sandbox admission test receipt differs; validate afresh before allocation"
        )
    return True


def publish_tests(identity: str) -> None:
    from tools.ops.dispatch import write_create_only_json

    if path := os.environ.get(TEST_RECEIPT_ENV):
        write_create_only_json(
            Path(path),
            {"schema": "sandbox-admission-tests/v1", "identity": identity, "status": "PASS"},
        )


@contextlib.contextmanager
def parent_lease(lock: Path, lease_path: Path, manifest: Path, seconds: float) -> Iterator[float]:
    """Acquire exclusively, then allow only this runner to share lock ownership.

    Competitors still request LOCK_EX. A child already admitted under LOCK_SH
    keeps them fenced if its parent exits while native cleanup is in progress.
    Revocation precedes parent unlock; a late child can never use a stale grant.
    """
    from tools.ops.dispatch import write_create_only_json

    deadline = time.monotonic() + seconds
    with lock.open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        try:
            write_create_only_json(
                lease_path,
                {
                    "schema": "sandbox-allocation-lease/v1",
                    "parent_pid": os.getpid(),
                    "deadline": deadline,
                    "lock": str(lock),
                    "manifest": str(manifest),
                    "manifest_sha256": digest(manifest),
                },
            )
            fcntl.flock(handle, fcntl.LOCK_SH)
            yield deadline
        finally:
            lease_path.unlink(missing_ok=True)
            fcntl.flock(handle, fcntl.LOCK_UN)


class AllocationLease:
    """Child's shared flock, held from before new until durable calibration."""

    def __init__(self, path: Path, manifest: Path):
        self.path = path
        self.manifest = manifest
        self.handle: Any = None
        self.deadline = 0.0

    def acquire(self) -> None:
        value = json.loads(self.path.read_text())
        self.handle = Path(value["lock"]).open("a")
        try:
            fcntl.flock(self.handle, fcntl.LOCK_SH | fcntl.LOCK_NB)
            # Re-read AFTER locking: revoked/mutated grants refuse. There is no
            # check/unlock/new race because this descriptor remains held.
            if json.loads(self.path.read_text()) != value:
                raise ValueError("allocation lease changed")
            self.deadline = value["deadline"]
            if (
                value["schema"] != "sandbox-allocation-lease/v1"
                or value["manifest"] != str(self.manifest)
                or value["manifest_sha256"] != digest(self.manifest)
                or not math.isfinite(self.deadline)
                or time.monotonic() >= self.deadline
            ):
                raise ValueError("allocation lease expired or manifest differs")
            os.kill(value["parent_pid"], 0)
        except BaseException:
            self.close()
            raise

    def remaining(self, timeout: float) -> float:
        seconds = min(timeout, self.deadline - time.monotonic())
        if seconds <= 0:
            raise ValueError("ops allocation deadline expired before native CLI call")
        return seconds

    def close(self) -> None:
        if self.handle is not None:
            fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
