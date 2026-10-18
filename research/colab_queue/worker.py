"""Dedicated Colab worker; Drive arbitration has a bounded visibility assumption.

Run via worker.sh / the pinned notebook. --local-test never sets up or releases a
runtime and executes only the explicitly provided, commit-matching local repo.
"""

from __future__ import annotations

import argparse
import fcntl
import math
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

try:
    from .core import Queue, QueueError
except ImportError:
    from core import Queue, QueueError


class OwnershipLost(RuntimeError):
    """The claim was cancelled, fenced, or lost arbitration."""


def capabilities() -> dict[str, Any]:
    """Inspect actual assigned GPU and CPU affinity, not notebook metadata."""
    try:
        name = (
            subprocess.check_output(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            .splitlines()[0]
            .strip()
        )
    except (OSError, subprocess.SubprocessError, IndexError):
        name = "cpu"
    gpu = "A100" if "A100" in name else "L4" if re.search(r"\bL4\b", name) else name
    cpus = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count()
    return {"gpu": gpu, "gpu_name": name, "vcpu": cpus or 1}


def stop_group(process: subprocess.Popen, grace: float = 5) -> None:
    """Reap only this worker's owned process group, including orphaned actors."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:
        process.poll()
        try:
            os.killpg(process.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.05)
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait()


def release_runtime(python: str) -> None:
    """Flush Drive before asking Colab to unassign; a flush error prevents release."""
    code = (
        "from google.colab import drive, runtime; "
        "drive.flush_and_unmount(timeout_ms=120000); runtime.unassign()"
    )
    subprocess.run([python, "-c", code], check=True, timeout=150)


class Worker:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.queue = Queue(Path(args.root) / "queue")
        self.worker_id = args.worker_id or f"{socket.gethostname()}-{uuid.uuid4().hex}"
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.worker_id):
            raise ValueError("worker id must contain letters, digits, underscore or hyphen")
        self.caps = capabilities()
        self.stopping = False
        self.last_heartbeat = 0.0
        self.current: dict[str, Any] | None = None
        self.token: str | None = None
        self.prepared_commit: str | None = None
        self.repo = Path(args.repo).resolve()
        self.phase = "starting"

    def heartbeat(self, force: bool = False, **extra: Any) -> None:
        if force or time.monotonic() - self.last_heartbeat >= self.args.heartbeat_seconds:
            self.queue.write_heartbeat(
                self.worker_id,
                {
                    "pid": os.getpid(),
                    "hostname": socket.gethostname(),
                    **self.caps,
                    "phase": self.phase,
                    "job_id": self.current["id"] if self.current else None,
                    "token": self.token,
                    "queue_ref": os.environ.get("QUEUE_REF"),
                    **extra,
                },
            )
            self.last_heartbeat = time.monotonic()

    def assert_owner(self) -> None:
        if self.stopping:
            raise OwnershipLost("worker stopped")
        if self.current is not None:
            control = self.queue.control(self.current["id"])
            if control["cancelled"] or control["generation"] != self.current["queue_generation"]:
                raise OwnershipLost("cancelled or requeued")
            if not self.queue.is_winner(self.current, self.worker_id, self.token):
                raise OwnershipLost("claim lost arbitration")

    def wait(self, seconds: float, check_owner: bool = False) -> None:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            if self.stopping:
                raise OwnershipLost("worker stopped")
            self.heartbeat()
            if check_owner:
                self.assert_owner()
            time.sleep(min(self.args.poll_seconds, max(0, deadline - time.monotonic())))

    def run_process(
        self,
        command: list[str],
        log: Any,
        timeout: float,
        env: dict[str, str] | None = None,
        cwd: Path | None = None,
    ) -> int:
        self.assert_owner()
        process = subprocess.Popen(
            ["nohup", *command],
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=cwd,
            env=env,
            start_new_session=True,
        )
        started = time.monotonic()
        try:
            while process.poll() is None:
                self.assert_owner()
                self.heartbeat()
                if time.monotonic() - started > timeout:
                    raise TimeoutError(f"process exceeded {timeout:g} seconds")
                time.sleep(self.args.poll_seconds)
            self.assert_owner()
            return process.returncode
        finally:
            stop_group(process)

    def prepare(self, job: dict[str, Any], log: Any) -> None:
        commit = job["commit"]
        if not self.args.local_test and self.prepared_commit != commit:
            # Setup replaces checkout/venv/inputs. A failed switch invalidates
            # the prior environment too; never reuse its cached success.
            self.prepared_commit = None
            self.phase = "setup"
            self.heartbeat(force=True)
            # Fetch setup from the job's own immutable source revision. Setup owns only
            # the standard Colab checkout, venv and extracted inputs, never queue code.
            with tempfile.TemporaryDirectory(prefix="colab-queue-setup-") as temp:
                setup = str(Path(temp) / "setup_runtime.sh")
                url = (
                    "https://raw.githubusercontent.com/josenunezwork/snake-dqn/"
                    f"{commit}/research/t16_colab/setup_runtime.sh"
                )
                rc = self.run_process(
                    ["curl", "-fsSL", "--retry", "3", "--max-time", "120", url, "-o", setup],
                    log,
                    400,
                )
                if rc:
                    raise RuntimeError(f"setup download exited {rc}")
                setup_env = os.environ.copy()
                if self.caps["gpu"] == "cpu":
                    setup_env["T16_CPU_ONLY"] = "1"
                rc = self.run_process(
                    ["bash", setup, commit], log, self.args.setup_timeout_seconds, env=setup_env
                )
                if rc:
                    raise RuntimeError(f"setup exited {rc}")
        actual = subprocess.check_output(
            ["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True, timeout=10
        ).strip()
        if actual != commit:
            raise RuntimeError(f"checkout mismatch: requested {commit}, got {actual}")
        if subprocess.check_output(
            ["git", "-C", str(self.repo), "status", "--porcelain", "--untracked-files=no"],
            text=True,
            timeout=10,
        ).strip():
            raise RuntimeError("job checkout has modified tracked files")
        self.prepared_commit = commit

    def execute(self, job: dict[str, Any], token: str) -> None:
        self.current, self.token = job, token
        started = time.time()
        run_started: float | None = None
        rc: int | None = None
        state, error = "failed", None
        output = (Path(self.args.root) / job["output_dir"]).resolve()
        root = Path(self.args.root).resolve()
        log_name = f"g{job['queue_generation']}.{self.worker_id}.{token}.log"
        log_path = output / log_name
        try:
            if root not in output.parents or not output.is_dir():
                raise RuntimeError(
                    "output directory missing or outside Drive root; enqueue on Mac first"
                )
            with log_path.open("ab", buffering=0) as log:
                self.prepare(job, log)
                script = (self.repo / job["script"]).resolve()
                if self.repo not in script.parents or not script.is_file():
                    raise ValueError("job script missing or outside checkout")
                env = os.environ.copy()
                env.update(job["env"])
                env.update(
                    {
                        "QUEUE_JOB_ID": job["id"],
                        "QUEUE_WORKER_ID": self.worker_id,
                        "QUEUE_CLAIM_TOKEN": token,
                        "QUEUE_GENERATION": str(job["queue_generation"]),
                        "QUEUE_OUTPUT_DIR": str(output),
                        "QUEUE_DRIVE_ROOT": str(root),
                        "PYTHONPATH": str(self.repo),
                        "PATH": f"/content/t16-colab-venv/bin:{os.environ.get('PATH', '')}",
                    }
                )
                self.phase = "running"
                self.heartbeat(force=True)
                run_started = time.time()
                timeout = job.get("timeout_minutes", job["requirements"]["est_minutes"] * 2) * 60
                rc = self.run_process(["bash", str(script)], log, timeout, env, self.repo)
                state = "done" if rc == 0 else "failed"
                if rc:
                    error = f"job exited {rc}"
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            print(error, flush=True)
        finally:
            ended = time.time()
            self.phase = "finishing"
            # Fenced receipts are ignored by the current generation; never retry automatically.
            self.queue.write_receipt(
                job["id"],
                job["queue_generation"],
                token,
                state,
                {
                    "worker_id": self.worker_id,
                    **self.caps,
                    "commit": job["commit"],
                    "verified_commit": self.prepared_commit,
                    "started_at": started,
                    "run_started_at": run_started,
                    "finished_at": ended,
                    "elapsed_seconds": ended - started,
                    "run_seconds": ended - run_started if run_started else None,
                    "exit_code": rc,
                    "error": error,
                    "log": str(Path(job["output_dir"]) / log_name),
                },
            )
            self.current, self.token = None, None

    def release(self) -> None:
        """Retry transient flush/unassign failures without declaring release success."""
        for attempt in range(3):
            try:
                release_runtime(self.args.colab_python)
                return
            except (OSError, subprocess.SubprocessError) as exc:
                print(f"release attempt {attempt + 1}/3 failed: {exc}", flush=True)
                if attempt == 2:
                    raise RuntimeError(
                        "automatic release failed; inspect the local worker log and "
                        "disconnect this dedicated runtime manually"
                    ) from exc
                # Drive may already be unmounted; do not recreate paths or write
                # a misleading heartbeat into a local replacement directory.
                time.sleep(15)

    def run(self) -> None:
        idle_since = time.monotonic()
        while not self.stopping:
            self.phase = "idle"
            try:
                self.heartbeat()
                jobs = self.queue.eligible_jobs(self.caps)
                if jobs:
                    job = jobs[0]
                    token = uuid.uuid4().hex
                    self.queue.write_claim(job, self.worker_id, token, self.caps)
                    self.current, self.token = job, token
                    self.phase = "settling"
                    self.heartbeat(force=True)
                    # All contenders wait the full propagation budget; running owners
                    # continue monitoring for a later-visible lower-id claim.
                    self.wait(self.args.settle_seconds)
                    try:
                        self.assert_owner()
                    except OwnershipLost:
                        self.current, self.token = None, None
                        continue
                    self.execute(job, token)
                    idle_since = time.monotonic()
                    continue
            except (OSError, ValueError, QueueError) as exc:
                # Never interpret unreadable/malformed shared state as an empty queue.
                print(f"queue unavailable (no claim made): {exc}", flush=True)
                self.current, self.token = None, None
                self.phase = "queue_error"
            if time.monotonic() - idle_since >= self.args.idle_minutes * 60:
                self.phase = "releasing"
                try:
                    self.heartbeat(force=True)
                except (OSError, QueueError) as exc:
                    print(f"release heartbeat unavailable: {exc}", flush=True)
                if not self.args.local_test:
                    self.release()
                return
            # The next loop writes the heartbeat; an unavailable Drive must not
            # kill the idle timer before it can release this dedicated runtime.
            time.sleep(self.args.poll_seconds)
        self.phase = "stopped"
        self.heartbeat(force=True)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument(
        "--root",
        type=Path,
        default=os.environ.get("DRIVE_ROOT", "/content/drive/MyDrive/t16_colab"),
    )
    result.add_argument("--repo", default="/content/t16-snake-dqn")
    result.add_argument("--worker-id")
    result.add_argument(
        "--idle-minutes", type=float, default=float(os.environ.get("IDLE_MINUTES", 10))
    )
    result.add_argument(
        "--settle-seconds", type=float, default=float(os.environ.get("CLAIM_SETTLE_SECONDS", 120))
    )
    result.add_argument("--poll-seconds", type=float, default=5)
    result.add_argument("--heartbeat-seconds", type=float, default=15)
    result.add_argument("--setup-timeout-seconds", type=float, default=1800)
    result.add_argument("--colab-python", default=os.environ.get("COLAB_PYTHON", sys.executable))
    result.add_argument("--local-test", action="store_true")
    result.add_argument("--lock-path", type=Path, default=Path("/content/colab-queue-worker.lock"))
    return result


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    for key in (
        "idle_minutes",
        "settle_seconds",
        "poll_seconds",
        "heartbeat_seconds",
        "setup_timeout_seconds",
    ):
        if not math.isfinite(getattr(args, key)) or getattr(args, key) <= 0:
            raise SystemExit(f"{key} must be positive")
    if not args.local_test:
        if Path(args.repo) != Path("/content/t16-snake-dqn"):
            raise SystemExit("production setup uses /content/t16-snake-dqn")
        if str(args.root) != "/content/drive/MyDrive/t16_colab":
            raise SystemExit("production setup requires /content/drive/MyDrive/t16_colab")
        if args.lock_path != Path("/content/colab-queue-worker.lock"):
            raise SystemExit("production worker requires its standard local runtime lock")
        if not os.environ.get("TBE_RUNTIME_ADDR"):
            raise SystemExit("not a Colab runtime; local simulations require --local-test")
    # This lock is LOCAL ONLY. It prevents a second notebook cell's worker from
    # resetting the same runtime checkout while the first is running a job.
    with args.lock_path.open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit("a queue worker already owns this runtime")
        worker = Worker(args)

        def stop(signum: int, frame: Any) -> None:
            worker.stopping = True

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        try:
            worker.run()
        except OwnershipLost:
            worker.phase = "stopped"
            worker.heartbeat(force=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
