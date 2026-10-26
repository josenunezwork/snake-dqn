"""Mac broker for pinned jobs. No allocation retries, Drive mounts, or purchases.

All broker processes share one account mutation lock, even across worktrees and
state directories. A lane lock excludes duplicate dispatch; remote generations
and create-only records fence attempts. Live acceptance is deliberately separate.
"""

from __future__ import annotations

import argparse
import base64
import contextlib
import fcntl
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

from research.colab_cli import policy

ACCOUNT_ROOT = Path.home() / ".config/colab-cli/snake"
REPO = Path(__file__).resolve().parents[2]
OAUTH_PYTHON = str(Path.home() / ".venvs/colab-cli/bin/python")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE_CALIBRATION_INTERVAL_SECONDS = 5.0
RATE_CALIBRATION_STABLE_READS = 3
RATE_CALIBRATION_SECONDS = (
    2 * RATE_CALIBRATION_STABLE_READS * policy.READ_SECONDS
    + (RATE_CALIBRATION_STABLE_READS - 1) * RATE_CALIBRATION_INTERVAL_SECONDS
)
RATE_CALIBRATION_STABILITY_UNITS_PER_HOUR = 0.01


class Refused(RuntimeError):
    """Admission, execution or evidence was not proven safe."""


def redact(value: str) -> str:
    """HTTP failures can contain proxy tokens in query parameters."""
    value = re.sub(
        r"(?i)((?:[?&]|%26)(?:[^=\s&]*token|key|code)=)[^&\s\"'<>]+",
        r"\1[REDACTED]",
        value,
    )
    return re.sub(r"(?i)(Bearer\s+)[A-Za-z0-9._~+/=-]+", r"\1[REDACTED]", value)


def sha256(path: Path) -> str:
    """Hash in bounded memory."""
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def write_once(path: Path, value: Any) -> None:
    """Create durable evidence without replacing an earlier attempt."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".receipt-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)  # Atomic create-only publication.
        finally:
            temporary.unlink(missing_ok=True)


@contextlib.contextmanager
def locked(path: Path, blocking: bool = True) -> Iterator[None]:
    """Cross-process account or lane lock; never unlink lock inodes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        except BlockingIOError as exc:
            raise Refused(f"already supervised: {path}") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


@contextlib.contextmanager
def account_locked(cli: Any, *, cleanup: bool = False) -> Iterator[None]:
    """Wait for the allocator without starving another lane's runtime lease."""
    path = ACCOUNT_ROOT / "broker.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    cleanup_deadline = time.monotonic() + 600
    with path.open("a") as handle:
        while True:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if cleanup and time.monotonic() >= cleanup_deadline:
                    print(
                        "\aSTOP_LOCK_UNAVAILABLE: durable recovery record retained",
                        file=sys.stderr,
                        flush=True,
                    )
                    raise Refused("STOP_LOCK_UNAVAILABLE after 600s")
                if not cleanup and cli.deadline is not None and time.monotonic() >= cli.deadline:
                    raise Refused("deadline reached while awaiting account broker")
                if (
                    not cleanup
                    and cli.priced_deadline is not None
                    and time.monotonic() >= cli.priced_deadline
                ):
                    raise Refused("priced session unit cap reached while awaiting account broker")
                if not cleanup and cli.renew is not None:
                    cli.renew()
                time.sleep(2)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def safe_relative(value: str) -> str:
    """Reject traversal and ambiguous remote paths."""
    path = PurePosixPath(value)
    if not value or path.is_absolute() or any(p in {"..", "."} for p in value.split("/")):
        raise Refused(f"unsafe relative path: {value!r}")
    return str(path)


def positive(value: Any, name: str) -> float:
    """Validate caps without accepting booleans, infinities, or NaNs."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise Refused(f"{name} must be numeric")
    if not math.isfinite(value) or value <= 0:
        raise Refused(f"{name} must be finite and positive")
    return float(value)


def validate(manifest: dict[str, Any], repo: Path | None = None) -> dict[str, Any]:
    """Validate inputs and a conservatively priced whole-session envelope."""
    repo = REPO if repo is None else repo
    m = json.loads(json.dumps(manifest))
    from research.colab_cli.schema import SchemaError, validate_manifest

    try:
        validate_manifest(m)
    except SchemaError as exc:
        raise Refused(str(exc)) from exc
    if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", m.get("job_id", "")):
        raise Refused("job_id must be a short lowercase identifier")
    commit = m.get("commit", "")
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise Refused("commit must be a full 40-character SHA")
    kind = subprocess.run(
        ["git", "cat-file", "-t", commit], cwd=repo, capture_output=True, text=True, timeout=10
    )
    if kind.returncode or kind.stdout.strip() != "commit":
        raise Refused("commit SHA must identify a Git commit object")
    script = safe_relative(m.get("script", ""))
    if not script.endswith(".sh"):
        raise Refused("script must be a pinned Bash .sh entrypoint")
    result = subprocess.run(
        ["git", "cat-file", "-t", f"{commit}:{script}"],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=10,
    )
    if result.returncode or result.stdout.strip() != "blob":
        raise Refused("script absent from pinned commit")
    req = m["requirements"]
    if req.get("gpu") not in {"cpu", "L4", "A100", "H100", "G4", "T4"}:
        raise Refused("explicit cpu/L4/A100/H100/G4/T4 required")
    allowed = req.get("allowed_gpus", [req["gpu"]])
    if (
        not isinstance(allowed, list)
        or not allowed
        or any(gpu not in {"cpu", "L4", "A100", "H100", "G4", "T4"} for gpu in allowed)
    ):
        raise Refused("allowed_gpus must name explicitly prepriced hardware")
    req["allowed_gpus"] = allowed
    if type(req.get("min_vcpu")) is not int or req["min_vcpu"] < 1:
        raise Refused("min_vcpu must be a positive integer")
    estimate = positive(req["est_minutes"], "est_minutes")
    timeout = positive(m["timeout_minutes"], "timeout_minutes")
    if timeout < 3:
        raise Refused("timeout_minutes must be >=3 for allocation and guarded cleanup")
    if timeout < 2 * estimate:
        raise Refused("timeout must include at least 2x measured estimate")
    budget = m["budget"]
    rate = positive(budget["rate_units_per_hour"], "prepriced rate ceiling")
    cap = positive(budget["unit_cap"], "unit_cap")
    reserve = budget.get("reserve_units", 0)
    if isinstance(reserve, bool) or not math.isfinite(reserve) or reserve < 0:
        raise Refused("reserve_units must be finite and nonnegative")
    budget["reserve_units"] = reserve
    # Includes allocation, bootstrap and final transfer/stop, not just trainer time.
    if timeout * rate / 60 > cap:
        raise Refused("whole-session timeout priced above unit cap")
    output = Path(m["output_dir"]).expanduser()
    if not output.is_absolute() or output.exists():
        raise Refused("output_dir must be an absolute fresh Mac directory")
    m["output_dir"] = str(output)
    if m.get("backup_dir") and not Path(m["backup_dir"]).expanduser().is_absolute():
        raise Refused("backup_dir must be an absolute Mac/Drive-for-desktop path")
    env = m.get("env", {})
    if not isinstance(env, dict) or any(
        not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", k) or not isinstance(v, str)
        for k, v in env.items()
    ):
        raise Refused("env must map valid variable names to strings")
    if any(
        k in env for k in ("HOME", "PATH", "PYTHONPATH", "COLAB_OUTPUT_DIR", "QUEUE_OUTPUT_DIR")
    ):
        raise Refused("env cannot replace transport paths")
    m["env"] = env
    destinations = set()
    for item in m.get("inputs", []):
        if not isinstance(item, dict):
            raise Refused("input must be an object")
        dest = safe_relative(item["dest"])
        if dest in destinations:
            raise Refused("duplicate input destination")
        destinations.add(dest)
        if "url" in item:
            if "path" in item:
                raise Refused("public input cannot also name a local path")
            from research.colab_cli.inputs import InputError, validate_public_input

            try:
                public = validate_public_input(item)
            except InputError as exc:
                raise Refused(str(exc)) from exc
            if public["size_bytes"] > 2 * 1024**3:
                raise Refused("public input exceeds 2 GiB bounded delivery limit")
            item.update(public)
            continue
        path = Path(item["path"]).expanduser().resolve(strict=True)
        if not path.is_file() or not re.fullmatch(r"[0-9a-f]{64}", item["sha256"]):
            raise Refused("input must be a file with SHA256")
        if sha256(path) != item["sha256"]:
            raise Refused(f"input hash mismatch: {path}")
        item["path"] = str(path)
    lease = positive(m.get("lease_seconds", 180), "lease_seconds")
    if not 180 <= lease <= 600:
        raise Refused("lease_seconds must be 180..600 (bounded transfer margin)")
    m["lease_seconds"] = lease
    m["max_export_bytes"] = int(positive(m.get("max_export_bytes", 1024**3), "max_export_bytes"))
    if m["max_export_bytes"] > 2 * 1024**3:
        raise Refused("maximum artifact size is 2 GiB (transferred in bounded chunks)")
    from research.colab_cli.admission import check_study_manifest

    check_study_manifest(m, repo)
    from research.colab_cli.canary import CanaryError, check

    try:
        check(m, repo)
    except CanaryError as exc:
        raise Refused(str(exc)) from exc
    return m


def input_preflight(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Verify public input bytes on the Mac before reserving or allocating."""
    from research.colab_cli.inputs import InputError, preflight

    rows = [item for item in manifest.get("inputs", []) if "url" in item]
    if not rows:
        return []
    try:
        return preflight(rows)
    except InputError as exc:
        raise Refused(str(exc)) from exc


class CLI:
    """Pinned official CLI as a bounded subprocess, never a shell command."""

    def __init__(self, binary: str, config: Path, session: str = "") -> None:
        self.binary, self.config, self.session = binary, config, session
        self.exec_uncertain = False
        self.cleanup_mode = False
        self.deadline: float | None = None
        self.priced_deadline: float | None = None
        self.notebook_hash = uuid.uuid4().hex
        self.assignment_record = config.with_suffix(".assignment.json")
        self.oauth_python = os.environ.get("SNAKE_COLAB_OAUTH_PYTHON", OAUTH_PYTHON)
        self.renew: Callable[[], Any] | None = None
        self.transfer_records: list[dict[str, Any]] = []
        self.contents_transfers: list[dict[str, Any]] = []
        self.call_records: list[dict[str, Any]] = []
        self.operation_records: list[dict[str, Any]] = []
        self._control_context = threading.local()
        self.verified_exports: dict[str, dict[str, Any]] = {}
        self.warmup_seconds = 0.0
        self._lease_stop = threading.Event()
        self._lease_thread: threading.Thread | None = None

    def remaining_seconds(self, allowance: float, *, staging_control: bool = True) -> float:
        """Clip non-CLI helpers to the same first-cap-wins session clock."""
        staging_deadline = getattr(self._control_context, "deadline", math.inf)
        remaining = min(
            allowance,
            (
                policy.STAGING_CONTROL_SECONDS
                if staging_control and staging_deadline != math.inf
                else math.inf
            ),
            staging_deadline - time.monotonic(),
            (self.deadline if self.deadline is not None else math.inf) - time.monotonic(),
            (self.priced_deadline if self.priced_deadline is not None else math.inf)
            - time.monotonic(),
        )
        if remaining <= 0:
            raise Refused("whole-session deadline reached before helper")
        return remaining

    @contextlib.contextmanager
    def staging_controls(self, deadline: float) -> Iterator[None]:
        """Bound foreground staging without starving the independent lease thread."""
        previous = getattr(self._control_context, "deadline", math.inf)
        self._control_context.deadline = min(previous, deadline)
        try:
            if time.monotonic() >= self._control_context.deadline:
                raise Refused("total upload time cap reached before staging control")
            yield
            if time.monotonic() > self._control_context.deadline:
                raise Refused("total upload time cap reached after staging control")
        finally:
            self._control_context.deadline = previous

    @contextlib.contextmanager
    def operation(self, tag: str) -> Iterator[None]:
        """Record monotonic intervals; overlapping heartbeat work stays identifiable."""
        previous = getattr(self._control_context, "tag", "control")
        self._control_context.tag = tag
        started = time.monotonic()
        status = "failed"
        try:
            yield
            status = "completed"
        finally:
            ended = time.monotonic()
            self.operation_records.append(
                {
                    "operation": tag,
                    "started_monotonic": started,
                    "ended_monotonic": ended,
                    "elapsed_seconds": ended - started,
                    "status": status,
                    "thread": threading.current_thread().name,
                }
            )
            self._control_context.tag = previous

    def start_lease(self, root: str, generation: str, lease_seconds: float) -> None:
        """Renew via Contents while a kernel/control RPC is blocked.

        This single daemon owns only a generation-bound lease PUT. It never
        submits kernel code, allocates, launches, or stops. The watchdog remains
        responsible for actual Mac loss or exhausted wall/unit budget.
        """
        self.stop_lease(wait=True)
        self._lease_stop = threading.Event()
        stop_event = self._lease_stop

        def heartbeat() -> None:
            while not stop_event.is_set():
                try:
                    timeout = self.remaining_seconds(min(policy.READ_SECONDS, lease_seconds / 3))
                    self._put_lease(root, generation, timeout)
                except Exception:
                    # A missing heartbeat is handled by the remote watchdog;
                    # transport errors cannot extend the session or its lease.
                    pass
                if stop_event.wait(30):
                    return

        self._lease_thread = threading.Thread(target=heartbeat, name="colab-lease", daemon=True)
        self._lease_thread.start()

    def _put_lease(self, root: str, generation: str, timeout: float) -> None:
        """Write one complete lease using the Contents atomic file-save path."""
        with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
            json.dump({"generation": generation, "renewal_clock": "contents-mtime"}, handle)
            handle.flush()
            with self.operation("renewal-heartbeat"):
                self.call(
                    "upload",
                    "-s",
                    self.session,
                    handle.name,
                    root + "/lease-heartbeat.json",
                    timeout=timeout,
                )

    def renew_runtime_lease(self, root: str, generation: str, *, force: bool = False) -> Any:
        """Use fenced Contents during staging; obtain kernel state when requested."""
        with self.operation("renewal"):
            if not force and getattr(self._control_context, "deadline", math.inf) != math.inf:
                # The watchdog validates the generation and VM mtime. Staging
                # needs no kernel state read, and must not queue renewal execs
                # behind data PUTs. The caller retains credential/account checks.
                # Contents heartbeats do not execute kernel controls.
                # Match the background heartbeat's 60s allowance;
                # keep the real staging/session caps and single dispatch.
                self._put_lease(
                    root,
                    generation,
                    self.remaining_seconds(policy.STAGING_HEARTBEAT_SECONDS, staging_control=False),
                )
                return None
            operation = (
                f"read_state({root!r},{generation!r},renew=True,compact=True)"
                if force
                else f"renew_lease({root!r},{generation!r})"
            )
            return self.poll_rpc(
                "import importlib.util\n"
                f"runtime_path={root + '/runtime.py'!r}\n"
                "s=importlib.util.spec_from_file_location('snake_runtime',runtime_path)\n"
                "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
                f"result=mod.{operation}"
            )

    def stop_lease(self, *, wait: bool = False) -> None:
        """Fence future lease PUTs before simulated Mac loss or broker cleanup."""
        self._lease_stop.set()
        if wait and self._lease_thread is not None:
            self._lease_thread.join(timeout=policy.READ_SECONDS + 5)
            if self._lease_thread.is_alive():
                raise Refused("lease PUT did not quiesce before Mac-loss observation")
        # Do not wait on an in-flight idempotent PUT before stopping spending.
        # It cannot allocate or undo a verified unassignment.

    @contextlib.contextmanager
    def limit_deadline(self, deadline: float) -> Iterator[None]:
        """Temporarily narrow the session cap; never use for input staging."""
        original = self.deadline
        self.deadline = min(original if original is not None else math.inf, deadline)
        try:
            yield
        finally:
            self.deadline = original

    def call(self, *args: str, timeout: float = policy.READ_SECONDS) -> str:
        """Retry read-only commands within one budget; mutations get one dispatch."""
        deadline = time.monotonic() + timeout
        attempts = policy.ATTEMPTS if args[0] in policy.READ_COMMANDS else 1
        for attempt in range(attempts):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise Refused(f"colab {args[0]} operation budget exhausted")
            started = time.monotonic()
            status = "failed"
            try:
                result = self._call_once(*args, timeout=remaining)
                status = "completed"
                return result
            except (Refused, OSError, subprocess.SubprocessError) as exc:
                if attempt + 1 == attempts or not policy.transient_error(exc):
                    raise
                delay = min(policy.BACKOFF_SECONDS[attempt], max(0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
            finally:
                self.call_records.append(
                    {
                        "command": args[0],
                        "operation": getattr(self._control_context, "tag", "control"),
                        "started_monotonic": started,
                        "ended_monotonic": time.monotonic(),
                        "thread": threading.current_thread().name,
                        "attempt": attempt + 1,
                        "timeout_seconds": remaining,
                        "status": status,
                        "elapsed_seconds": time.monotonic() - started,
                    }
                )
        raise Refused("read retry budget exhausted")

    def _call_once(self, *args: str, timeout: float) -> str:
        if args[0] != "stop" and not (args[0] == "sessions" and self.cleanup_mode):
            deadline = min(
                self.deadline if self.deadline is not None else math.inf,
                self.priced_deadline if self.priced_deadline is not None else math.inf,
                getattr(self._control_context, "deadline", math.inf),
            )
            timeout = min(timeout, deadline - time.monotonic())
            if timeout <= 0:
                if time.monotonic() >= getattr(self._control_context, "deadline", math.inf):
                    raise Refused("total upload time cap reached before CLI call")
                raise Refused("whole-session deadline reached before CLI call")
        env = {
            **os.environ,
            "SNAKE_COLAB_NOTEBOOK_HASH": self.notebook_hash,
            "SNAKE_COLAB_ASSIGNMENT_RECORD": str(self.assignment_record),
            "SNAKE_COLAB_CONFIG": str(self.config),
            "SNAKE_COLAB_SESSION": self.session,
            "NO_COLOR": "1",
            "COLUMNS": "240",
        }
        self.config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        hook = self.config.parent / "cli-hook"
        hook.mkdir(exist_ok=True, mode=0o700)
        guard_path = Path(__file__).with_name("cli_guard.py")
        hook_code = (
            "import os,runpy\ntry:\n    runpy.run_path("
            + repr(str(guard_path))
            + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
        )
        hook_file = hook / "sitecustomize.py"
        if not hook_file.exists() or hook_file.read_text() != hook_code:
            with tempfile.NamedTemporaryFile("w", dir=hook, delete=False) as handle:
                handle.write(hook_code)
                temporary = Path(handle.name)
            try:
                os.replace(temporary, hook_file)
            finally:
                temporary.unlink(missing_ok=True)
        env["PYTHONPATH"] = str(hook) + os.pathsep + env.get("PYTHONPATH", "")
        transfer_started = time.monotonic()
        transfer_status = "failed"
        size = Path(args[-2]).stat().st_size if args[0] == "upload" else 0
        try:
            result = subprocess.run(
                [self.binary, "--auth", "oauth2", "--config", str(self.config), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                umask=0o077,
            )
            if result.returncode == 0:
                transfer_status = "completed"
                if args[0] == "download":
                    size = Path(args[-1]).stat().st_size
        finally:
            if self.config.exists():
                self.config.chmod(0o600)
            if args[0] in {"upload", "download"}:
                elapsed = max(time.monotonic() - transfer_started, 1e-9)
                self.contents_transfers.append(
                    {
                        "direction": args[0],
                        "operation": getattr(self._control_context, "tag", "control"),
                        "started_monotonic": transfer_started,
                        "ended_monotonic": time.monotonic(),
                        "thread": threading.current_thread().name,
                        "remote": args[-1] if args[0] == "upload" else args[-2],
                        "size_bytes": size if transfer_status == "completed" else None,
                        "elapsed_seconds": elapsed,
                        "bytes_per_second": (
                            size / elapsed if transfer_status == "completed" else None
                        ),
                        "status": transfer_status,
                    }
                )
        output = redact(ANSI.sub("", result.stdout + result.stderr))
        if result.returncode:
            raise Refused(f"colab {args[0]} failed ({result.returncode}): {output[-2000:]}")
        return output

    def sessions(self, timeout: float = policy.READ_SECONDS) -> dict[str, str]:
        output = self.call("sessions", timeout=timeout)
        rows = re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\| Hardware:", output, re.M)
        if not rows and "No active sessions found on server." not in output:
            raise Refused("unrecognized account sessions output")
        return {endpoint: name for name, endpoint in rows}

    def usage(self, timeout: float = policy.READ_SECONDS) -> dict[str, float]:
        output = self.call("usage", timeout=timeout)
        result = {}
        for key, label in (
            ("balance", "Current balance"),
            ("rate", "Usage rate"),
            ("assignments", "Active assignments"),
        ):
            match = re.search(rf"{label}:\s*([0-9]+(?:\.[0-9]+)?)", output)
            if not match:
                raise Refused("unrecognized account usage output")
            result[key] = float(match[1])
        return result

    def rpc(
        self, code: str, timeout: float = policy.RPC_SECONDS, *, completion_grace: float = 10
    ) -> Any:
        """Prove completion by nonce, with optional transport acknowledgement grace."""
        if self.exec_uncertain:
            raise Refused("previous exec uncertain/busy; refusing to resubmit to kernel")
        staging_deadline = getattr(self._control_context, "deadline", math.inf)
        if staging_deadline != math.inf:
            timeout = min(
                timeout, policy.STAGING_CONTROL_SECONDS, staging_deadline - time.monotonic()
            )
            completion_grace = 0
            if timeout <= 0:
                raise Refused("total upload time cap reached before control RPC")
        nonce = "SNAKE_RPC_" + uuid.uuid4().hex
        wrapped = (
            "import json, traceback\ntry:\n"
            + "\n".join("    " + line for line in code.splitlines())
            + f"\n    print({nonce!r} + json.dumps({{'ok':True,'value':result}}))\n"
            + "except BaseException as exc:\n"
            + f"    print({nonce!r} + json.dumps({{'ok':False,'error':repr(exc)}}))\n"
        )
        with tempfile.NamedTemporaryFile("w", suffix=".py") as handle:
            compile(wrapped, "<colab-cli-rpc>", "exec")
            handle.write(wrapped)
            handle.flush()
            try:
                output = self.call(
                    "exec",
                    "-s",
                    self.session,
                    "-f",
                    handle.name,
                    "--timeout",
                    str(timeout),
                    timeout=timeout + completion_grace,
                )
                matches = re.findall(re.escape(nonce) + r"(\{[^\n]*\})", output)
                if not matches:
                    raise Refused("kernel completion not acknowledged (busy/error/timeout)")
                reply = json.loads(matches[-1])
            except BaseException:
                self.exec_uncertain = True
                raise
        if not reply["ok"]:
            raise Refused(f"kernel error despite CLI exit zero: {reply.get('error')}")
        return reply["value"]

    def wait_runtime_ready(self, output: Path) -> None:
        """Bound cold-start latency before credentials, uploads or their clock.

        Only read-only probes repeat. After an ambiguous tiny exec, another
        exec requires fresh independent idle proof, just like telemetry.
        Allocation and useful work are never retried here.
        """
        from research.colab_cli.cli_guard import kernel_idle, runtime_ready

        started = time.monotonic()
        evidence: dict[str, Any] = {"status": "INCOMPLETE", "attempts": 0}
        try:
            deadline = started + self.remaining_seconds(policy.WARMUP_SECONDS)
            # A count bound also protects against a stalled monotonic clock.
            for attempt in range(int(policy.WARMUP_SECONDS / policy.WARMUP_INTERVAL_SECONDS) + 1):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                evidence["attempts"] = attempt + 1
                submitted = False
                try:
                    ready = runtime_ready(
                        self.oauth_python,
                        self.config,
                        self.session,
                        timeout=self.remaining_seconds(min(policy.WARMUP_PROBE_SECONDS, remaining)),
                    )
                    if ready and self.exec_uncertain:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        ready = kernel_idle(
                            self.oauth_python,
                            self.config,
                            self.session,
                            timeout=self.remaining_seconds(
                                min(policy.WARMUP_PROBE_SECONDS, remaining)
                            ),
                        )
                    if ready:
                        # A first exec creates the kernel. After uncertainty,
                        # only the independent kernel GET permits another exec.
                        self.exec_uncertain = False
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            break
                        submitted = True
                        result = self.rpc(
                            "# SNAKE_RUNTIME_READINESS\n"
                            "from pathlib import Path\n"
                            "result={'ready': Path('/content').is_dir()}",
                            timeout=self.remaining_seconds(
                                min(policy.WARMUP_PROBE_SECONDS, remaining)
                            ),
                            completion_grace=0,
                        )
                        if result != {"ready": True}:
                            raise Refused("runtime readiness exec returned unknown state")
                        if time.monotonic() >= deadline:
                            break
                        self.remaining_seconds(1)
                        evidence["status"] = "PASS"
                        return
                except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
                    # An unacknowledged exec may still be busy. Poll GETs until
                    # independent idle proof; acknowledged kernel errors fail.
                    if (submitted and not self.exec_uncertain) or not policy.transient_error(exc):
                        raise
                    evidence["last_probe_error"] = redact(repr(exc))
                delay = min(policy.WARMUP_INTERVAL_SECONDS, max(0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
            raise Refused("runtime readiness warmup deadline reached; kernel remains busy/unknown")
        except BaseException as exc:
            evidence["error"] = redact(repr(exc))
            raise
        finally:
            self.warmup_seconds = max(0, time.monotonic() - started)
            evidence["elapsed_seconds"] = self.warmup_seconds
            write_once(output / "warmup-measurement.json", evidence)

    def recoverable_rpc(self, code: str, timeout: float = policy.TELEMETRY_SECONDS) -> Any:
        """Recover an acknowledgement without repeating a committed mutation.

        One durable UUID record fences the operation across verified-idle
        retries. A completed record returns its saved result; a started record
        without a result is ambiguous and refuses. The operation may start only
        when its record does not exist, never merely because the kernel is idle.
        """
        record = "/content/.snake-rpc-" + uuid.uuid4().hex + ".json"
        guarded = (
            "# SNAKE_RECOVERABLE_RPC\n"
            "import json as _snake_json, os as _snake_os\n"
            "from pathlib import Path as _SnakePath\n"
            f"_snake_record=_SnakePath({record!r})\n"
            "if _snake_record.exists():\n"
            "    _snake_saved=_snake_json.loads(_snake_record.read_text())\n"
            "    if _snake_saved.get('status') != 'completed':\n"
            "        raise RuntimeError('operation started without durable completion; '\n"
            "                           'refusing replay')\n"
            "    result=_snake_saved['result']\n"
            "else:\n"
            "    with _snake_record.open('x') as _snake_handle:\n"
            "        _snake_json.dump({'status':'started'},_snake_handle)\n"
            "        _snake_handle.flush()\n"
            "        _snake_os.fsync(_snake_handle.fileno())\n"
            + "\n".join("    " + line for line in code.splitlines())
            + "\n    _snake_temporary=_snake_record.with_suffix('.tmp')\n"
            "    with _snake_temporary.open('x') as _snake_handle:\n"
            "        _snake_json.dump({'status':'completed','result':result},_snake_handle)\n"
            "        _snake_handle.flush()\n"
            "        _snake_os.fsync(_snake_handle.fileno())\n"
            "    _snake_os.replace(_snake_temporary,_snake_record)\n"
        )
        return self.poll_rpc(guarded, timeout=timeout)

    def poll_rpc(self, code: str, timeout: float = policy.TELEMETRY_SECONDS) -> Any:
        """Recover idempotent reads/controls only after independent kernel-idle proof.

        Each exec is bounded by RPC_SECONDS; all attempts, idle probes, and
        backoff share one separate operation budget, clipped to session caps.
        Neither a kernel error nor an exhausted deadline is a retry trigger.
        """
        from research.colab_cli.cli_guard import kernel_idle

        staging_deadline = getattr(self._control_context, "deadline", math.inf)
        deadline = min(
            time.monotonic() + timeout,
            self.deadline if self.deadline is not None else math.inf,
            self.priced_deadline if self.priced_deadline is not None else math.inf,
            staging_deadline,
            (
                time.monotonic() + policy.STAGING_RECOVERY_SECONDS
                if staging_deadline != math.inf
                else math.inf
            ),
        )
        for attempt in range(policy.ATTEMPTS):
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                if time.monotonic() >= staging_deadline:
                    raise Refused("total upload time cap reached during control recovery")
                raise Refused("deadline reached during telemetry recovery")
            try:
                attempt_seconds = min(policy.RPC_SECONDS, remaining)
                # Bound this call directly: narrowing shared self.deadline here
                # would also starve the independent Contents heartbeat thread.
                return self.rpc(code, attempt_seconds, completion_grace=0)
            except (Refused, subprocess.SubprocessError):
                if not self.exec_uncertain or attempt == policy.ATTEMPTS - 1:
                    raise
                # Never enqueue work while the previous exec could still be busy.
                idle = False
                for probe in range(policy.ATTEMPTS):
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if time.monotonic() >= staging_deadline:
                            raise Refused("total upload time cap reached during control recovery")
                        raise Refused("deadline reached during telemetry recovery")
                    try:
                        with self.operation("recovery-idle"):
                            idle = kernel_idle(
                                self.oauth_python,
                                self.config,
                                self.session,
                                timeout=min(
                                    (
                                        policy.STAGING_CONTROL_SECONDS
                                        if staging_deadline != math.inf
                                        else policy.READ_SECONDS
                                    ),
                                    remaining,
                                ),
                            )
                    except Exception as exc:
                        if not policy.transient_error(exc):
                            raise
                    if idle:
                        break
                    delay = min(2, max(0, deadline - time.monotonic()))
                    if delay:
                        time.sleep(delay)
                if not idle:
                    raise Refused("kernel remains busy/unknown after telemetry hiccup")
                self.exec_uncertain = False
                delay = min(policy.BACKOFF_SECONDS[attempt], max(0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
        raise Refused("telemetry recovery exhausted")


def stop_verified(cli: CLI, endpoint: str) -> None:
    """Cleanup reads retain their budget even after the job deadline expires."""
    previous = cli.cleanup_mode
    cli.cleanup_mode = True
    try:
        _stop_verified(cli, endpoint)
    finally:
        cli.cleanup_mode = previous


def _stop_verified(cli: CLI, endpoint: str) -> None:
    """A stop exit code cannot prove release; independently query the account."""
    errors = []
    for attempt in range(3):
        if attempt:
            try:
                if endpoint not in cli.sessions(timeout=60):
                    return
            except Exception as exc:
                errors.append(redact(repr(exc)))
                continue
        try:
            cli.call("stop", "-s", cli.session, timeout=75)
        except Exception as exc:
            errors.append(redact(repr(exc)))
        # Reissue the stop while listed; transient CLI lookup errors prove nothing.
        if attempt > 0:
            try:
                from research.colab_cli.cli_guard import unassign_endpoint

                if endpoint in cli.sessions(timeout=60):
                    unassign_endpoint(cli.oauth_python, endpoint, timeout=75)
            except Exception as exc:
                errors.append(redact(repr(exc)))
        for _ in range(3):
            try:
                if endpoint not in cli.sessions(timeout=60):
                    return
            except Exception as exc:
                errors.append(redact(repr(exc)))
            time.sleep(2)
    message = f"STOP_UNVERIFIED: assignment {endpoint} remains after 3 attempts: {errors!r}"
    print("\a" + message, file=sys.stderr, flush=True)
    raise Refused(message)


def collect_exports(
    cli: CLI, root: str, inventory: list[dict[str, Any]], output: Path, limit: int
) -> None:
    """Collect new immutable exports without polling a cumulative inventory per file."""
    from research.colab_cli.export_batches import export_batch, small_batches
    from research.colab_cli.schema import validate_export_inventory

    validate_export_inventory({}, inventory)
    pending = []
    for entry in inventory:
        if entry["size"] > limit:
            raise Refused("export exceeds bounded whole-file transfer limit")
        previous = cli.verified_exports.get(entry["path"])
        if previous is not None:
            if previous != entry:
                raise Refused("immutable snapshot metadata changed")
            continue
        if (output / "exports" / entry["path"]).exists():
            # Acceptance observers can prefetch into this output before the
            # main loop. Revalidate that exact triple once; unrelated observer
            # downloads (e.g. resume roundtrips) must not enter this job ledger.
            export_file(cli, root, entry, output, limit)
            cli.verified_exports[entry["path"]] = dict(entry)
            continue
        pending.append(entry)

    def verified(entry: dict[str, Any]) -> None:
        cli.verified_exports[entry["path"]] = dict(entry)

    for batch in small_batches(pending):
        if len(batch) > 1:
            export_batch(cli, root, batch, output, limit, on_verified=verified)
        else:
            entry = batch[0]
            export_file(cli, root, entry, output, limit)
            verified(entry)
        # Transfers already call this throttled maintenance callback. It must
        # never force a full state/log/inventory read for each completed file.
        if cli.renew:
            cli.renew()


def export_file(cli: CLI, root: str, entry: dict[str, Any], output: Path, limit: int) -> None:
    """Download immutable data + digest + marker, verify, publish atomically."""
    from research.colab_cli.transfers import TransferError, _contents_call, download_file

    rel = safe_relative(entry["path"])
    size = entry["size"]
    digest = entry["sha256"]
    if type(size) is not int or not 0 <= size <= limit:
        raise Refused("export exceeds bounded whole-file transfer limit")
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise Refused("invalid export digest")
    target = output / "exports" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        if sha256(target) != digest or not Path(str(target) + ".complete").is_file():
            raise Refused("immutable export changed")
        sidecar = Path(str(target) + ".sha256")
        if not sidecar.is_file() or sidecar.read_text().split()[0] != digest:
            raise Refused("local export publication incomplete")
        return
    with tempfile.TemporaryDirectory(dir=target.parent, prefix=".transfer-") as temp:
        data = Path(temp) / "data"
        marker = Path(temp) / "complete"
        sidecar = Path(temp) / "sha256"
        try:
            download_file(
                cli, root + "/output/" + rel, data, size, renew=cli.renew, expected_sha256=digest
            )
        except TransferError as exc:
            raise Refused(f"download hash mismatch or transfer failure: {exc}") from exc
        for suffix, local in ((".sha256", sidecar), (".complete", marker)):
            _contents_call(
                cli,
                ("download", "-s", cli.session, root + "/output/" + rel + suffix, str(local)),
                policy.READ_SECONDS,
                cli.renew,
                deadline=time.monotonic() + policy.READ_SECONDS,
            )
        if data.stat().st_size != size or sha256(data) != digest:
            raise Refused("download hash mismatch; temporary export rejected")
        if sidecar.stat().st_size > 1024 or sidecar.read_text().split()[0] != digest:
            raise Refused("download digest sidecar mismatch")
        if marker.stat().st_size > 4096:
            raise Refused("invalid completion marker")
        # Marker is published last, so interrupted publication is never acknowledged.
        os.replace(data, target)
        os.replace(sidecar, Path(str(target) + ".sha256"))
        os.replace(marker, Path(str(target) + ".complete"))


class Broker:
    """One account allocator, parallel independently supervised lanes."""

    def __init__(
        self,
        state_dir: Path,
        binary: str,
        observer: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.state_dir = state_dir.expanduser().resolve()
        self.binary = binary
        self.observer = observer
        self.state_dir.mkdir(parents=True, exist_ok=True)
        ACCOUNT_ROOT.mkdir(parents=True, exist_ok=True)
        self.state_dir.chmod(0o700)
        ACCOUNT_ROOT.chmod(0o700)

    def run(self, manifest: dict[str, Any], retained: dict[str, Any] | None = None) -> None:
        m = validate(manifest)
        # Public release assets are re-checked for every job immediately before
        # this broker can reserve units or request an assignment.
        public_preflight = input_preflight(m)
        job = m["job_id"]
        lane = self.state_dir / job
        with locked(ACCOUNT_ROOT / f"lane-{job}.lock", blocking=False):
            lane.mkdir(exist_ok=False, mode=0o700)
            output = Path(m["output_dir"])
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / "manifest.json", m)
            if public_preflight:
                write_once(output / "public-input-preflight.json", {"inputs": public_preflight})
            staging = output / "staging"
            staging.mkdir()
            from research.colab_cli.source import prepare

            prepare(REPO, m, staging)
            for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
                data = subprocess.check_output(
                    ["git", "show", f"{m['commit']}:research/colab_cli/{name}"],
                    cwd=REPO,
                    timeout=10,
                )
                (staging / name).write_bytes(data)
            if not (staging / "schema-validation.json").is_file():
                raise Refused("preparation did not publish schema-validation sidecar")
            for path in staging.iterdir():
                if path.stat().st_size > m["max_export_bytes"]:
                    from research.colab_cli.failures import ClassifiedFailure, FailureClass

                    detail = "pinned upload exceeds bounded transfer size before allocation"
                    write_once(
                        output / "receipt.json",
                        {
                            "job_id": job,
                            "generation": None,
                            "commit": m["commit"],
                            "manifest_sha256": sha256(output / "manifest.json"),
                            "session": None,
                            "supervisor_sha256": sha256(Path(__file__)),
                            "endpoint": None,
                            "usage_before": None,
                            "usage_after": None,
                            "accounting": None,
                            "elapsed_seconds": 0.0,
                            "terminal": None,
                            "exports": [],
                            "transfers": [],
                            "contents_transfers": [],
                            "cli_calls": [],
                            "control_operations": [],
                            "warmup_seconds": 0.0,
                            "error": detail,
                            "failure_class": FailureClass.UPLOAD_CAP.value,
                            "cleanup_error": None,
                            "status": "INCOMPLETE",
                            "stop_verified": False,
                        },
                    )
                    raise ClassifiedFailure(FailureClass.UPLOAD_CAP, detail)
            generation = uuid.uuid4().hex
            session = f"snake-{job}-{generation[:12]}"
            cli = CLI(self.binary, lane / "sessions.json", session)
            if retained and retained.get("cli"):
                cli = retained["cli"]
                session = cli.session
                # A retained session starts a new job receipt, not a cumulative
                # transfer ledger from the previous link in the chain.
                cli.transfer_records = []
                cli.contents_transfers = []
                cli.call_records = []
                cli.operation_records = []
                cli.verified_exports = {}
                cli.warmup_seconds = 0.0
            started = time.monotonic()
            cli.deadline = min(
                started + m["timeout_minutes"] * 60 - 45, (retained or {}).get("deadline", math.inf)
            )
            cli.priced_deadline = (
                started + m["budget"]["unit_cap"] * 3600 / m["budget"]["rate_units_per_hour"]
            )
            endpoint = None
            usage_before = None
            accounting = None
            error = None
            failure_class = None
            cleanup = None
            usage_after = None
            terminal = None
            exports = []
            intent = ACCOUNT_ROOT / f"pending-{session}.json"
            active = ACCOUNT_ROOT / f"active-{session}.json"
            holding_admission = [False]
            with contextlib.ExitStack() as admission_lock:
                try:
                    admission_lock.enter_context(account_locked(cli))
                    holding_admission[0] = True
                    admission_lock.callback(lambda: holding_admission.__setitem__(0, False))
                    if list(ACCOUNT_ROOT.glob("pending-*.json")):
                        raise Refused("unresolved allocation/stop: reconcile pending records first")
                    version = cli.call("version")
                    if not re.search(r"\b0\.7\.4\b", version):
                        raise Refused("qualified CLI version must be 0.7.4")
                    from research.colab_cli.release import mint_release_credential

                    cli.release_credential = mint_release_credential(
                        cli.oauth_python,
                        endpoint="",
                        min_valid_seconds=m["lease_seconds"] + 900,
                        timeout=cli.remaining_seconds(150),
                    )
                    before = cli.sessions()
                    usage_before = cli.usage()
                    reusing = bool(retained and retained.get("endpoint"))
                    if reusing:
                        endpoint = retained["endpoint"]
                        if before.get(endpoint) != session:
                            raise Refused("retained assignment vanished; no replacement allocation")
                    if not reusing and max(len(before), usage_before["assignments"]) >= 3:
                        raise Refused("account-wide assignment cap 3 (including notebooks)")
                    reservations = [
                        json.loads(p.read_text())
                        for p in ACCOUNT_ROOT.glob("active-*.json")
                        if p != active
                    ]
                    committed = sum(r["unit_cap"] for r in reservations)
                    reserve = max(
                        [m["budget"]["reserve_units"]] + [r["reserve_units"] for r in reservations]
                    )
                    reserved_cap = (retained or {}).get("unit_cap", m["budget"]["unit_cap"])
                    if usage_before["balance"] <= reserved_cap + committed + reserve:
                        raise Refused("insufficient balance for cap plus study reserve")
                    from research.colab_cli.accounting import admit_account

                    account_before = {e: n for e, n in before.items() if e != endpoint}
                    measured_before = dict(usage_before)
                    if reusing:
                        measured_before["assignments"] -= 1
                        measured_before["rate"] -= retained["observed_rate_units_per_hour"]
                    accounting = admit_account(
                        account_before, measured_before, m["budget"], active_records=reservations
                    )
                    cli.accounting = accounting
                    write_once(
                        output / "accounting-baseline.json", accounting.accounted_assignments()
                    )
                    if not reusing:
                        write_once(
                            intent,
                            {
                                "session": session,
                                "config": str(cli.config),
                                "output": str(output),
                                "before": list(before),
                                "notebook_hash": cli.notebook_hash,
                                "assignment_record": str(cli.assignment_record),
                            },
                        )
                        args = ["new", "-s", session]
                        if m["requirements"]["gpu"] != "cpu":
                            args += ["--gpu", m["requirements"]["gpu"]]
                        cli.call(*args, timeout=policy.ALLOCATION_SECONDS)
                        metadata = json.loads(cli.config.read_text()).get(session, {})
                        endpoint = metadata.get("endpoint")
                        if not endpoint:
                            raise Refused("new allocation identity uncertain; no retry")
                        # First arm the watchdog. Backend identity/concurrency/rate
                        # reconciliation follows immediately in execute().
                        after = {**before, endpoint: session}
                    else:
                        after = before
                    if len(after) > 3:
                        raise Refused("account concurrency changed during allocation")
                    write_once(
                        lane / "assignment.json",
                        {
                            "session": session,
                            "endpoint": endpoint,
                            "generation": generation,
                            "output": str(output),
                        },
                    )
                    if not reusing:
                        write_once(
                            active,
                            {
                                "session": session,
                                "endpoint": endpoint,
                                "config": str(cli.config),
                                "output": str(output),
                                "unit_cap": reserved_cap,
                                "reserve_units": reserve,
                                "rate_units_per_hour": m["budget"]["rate_units_per_hour"],
                                "priced_rate_units_per_hour": m["budget"]["rate_units_per_hour"],
                                "observed_rate_units_per_hour": None,
                            },
                        )
                    terminal, exports = self.execute(
                        cli, m, generation, output, started, usage_before, retained, admission_lock
                    )
                except BaseException as exc:
                    from research.colab_cli.failures import classify_exception

                    error = redact(repr(exc))
                    failure_class = classify_exception(exc).value
                    exports = list(cli.verified_exports.values())
                    terminal_path = output / "terminal.json"
                    if terminal is None and terminal_path.is_file():
                        terminal = json.loads(terminal_path.read_text())
                finally:
                    cli.stop_lease()
                    # Recover identity from CLI metadata if allocation returned uncertain.
                    if endpoint is None and cli.config.exists():
                        try:
                            endpoint = (
                                json.loads(cli.config.read_text()).get(session, {}).get("endpoint")
                            )
                        except (ValueError, OSError):
                            pass
                    if endpoint is None and intent.exists():
                        try:
                            from research.colab_cli.cli_guard import recover_assignment

                            if cli.assignment_record.exists():
                                endpoint = json.loads(cli.assignment_record.read_text()).get(
                                    "endpoint"
                                )
                            if not endpoint:
                                endpoint = recover_assignment(cli.oauth_python, cli.notebook_hash)
                        except Exception as exc:
                            cleanup = f"ALLOCATION_UNRESOLVED: {redact(repr(exc))}"
                        if not endpoint:
                            cleanup = cleanup or "ALLOCATION_UNRESOLVED: retained pending record"
                    keep = bool(retained is not None and not error)
                    if endpoint and not keep:
                        try:
                            with (
                                contextlib.nullcontext()
                                if holding_admission[0]
                                else account_locked(cli, cleanup=True)
                            ):
                                if not intent.exists():
                                    write_once(
                                        intent,
                                        {
                                            "session": session,
                                            "config": str(cli.config),
                                            "endpoint": endpoint,
                                            "output": str(output),
                                        },
                                    )
                                stop_verified(cli, endpoint)
                                cli.deadline = (
                                    None  # Only read account data after verified release.
                                )
                                cli.priced_deadline = None
                                intent.unlink()
                                active.unlink(missing_ok=True)
                        except BaseException as exc:
                            cleanup = redact(repr(exc))
                        if cleanup is None:
                            try:
                                usage_after = cli.usage()
                            except BaseException as exc:
                                error = error or f"final usage unavailable: {redact(repr(exc))}"
                    if not keep and not cleanup and m.get("backup_dir"):
                        try:
                            self.backup(output, m["backup_dir"], job, generation)
                        except BaseException as exc:
                            error = error or redact(repr(exc))
                    if (error or cleanup) and failure_class is None:
                        from research.colab_cli.failures import classify_exception

                        failure_class = classify_exception(RuntimeError(error or cleanup)).value
                    receipt = {
                        "job_id": job,
                        "generation": generation,
                        "commit": m["commit"],
                        "manifest_sha256": sha256(output / "manifest.json"),
                        "session": session,
                        "supervisor_sha256": sha256(Path(__file__)),
                        "endpoint": endpoint,
                        "usage_before": usage_before,
                        "usage_after": usage_after,
                        "accounting": accounting.receipt() if accounting else None,
                        "elapsed_seconds": time.monotonic() - started,
                        "warmup_seconds": cli.warmup_seconds,
                        "terminal": terminal,
                        "exports": exports,
                        "transfers": cli.transfer_records,
                        "contents_transfers": cli.contents_transfers,
                        "cli_calls": cli.call_records,
                        "control_operations": cli.operation_records,
                        "error": error,
                        "failure_class": failure_class,
                        "cleanup_error": cleanup,
                        "status": (
                            ("COMPLETED_SESSION_RETAINED" if keep else "PASS")
                            if not error and not cleanup
                            else "INCOMPLETE"
                        ),
                        "stop_verified": bool(endpoint and not cleanup and not keep),
                    }
                    from research.colab_cli.canary import evidence, is_canary

                    if is_canary(m) and receipt["status"] == "PASS":
                        try:
                            receipt["canary_evidence"] = evidence(m, receipt, output, REPO)
                        except Exception as exc:
                            error = redact(repr(exc))
                            receipt.update(
                                error=error, status="INCOMPLETE", failure_class="SUPERVISOR"
                            )
                    write_once(output / "receipt.json", receipt)
                    if keep:
                        retained.update(
                            cli=cli,
                            endpoint=endpoint,
                            active=active,
                            intent=intent,
                            observed_rate_units_per_hour=accounting.observed_rate_units_per_hour,
                        )
                        retained.setdefault("completed", []).append(
                            ("/content/snake-cli/" + generation, generation)
                        )
                        retained.setdefault("outputs", []).append(str(output))
                        if m.get("backup_dir"):
                            retained.setdefault("backups", []).append(
                                (output, m["backup_dir"], job, generation)
                            )
                if error or cleanup:
                    raise Refused(
                        f"INCOMPLETE: {error or cleanup}; receipt: {output / 'receipt.json'}"
                    )

    def calibrate_rate(
        self,
        cli: CLI,
        m: dict[str, Any],
        output: Path,
        started: float,
        retained: dict[str, Any] | None,
        *,
        account_lock_held: bool = False,
    ) -> None:
        """Bound provider lag before admitting useful work; ceilings still cap spend.

        The pending record blocks new broker allocations while this assignment
        is uncalibrated. The armed watchdog is renewed between reads. Existing
        endpoints must remain unchanged: a changed baseline cannot calibrate
        an individual account-rate delta safely.
        """
        tracker = cli.accounting
        original_deadline = cli.deadline
        calibration_started = time.monotonic()
        deadline = min(
            original_deadline,
            calibration_started + RATE_CALIBRATION_SECONDS,
            started + tracker.unit_cap * 3600 / tracker.own_rate_units_per_hour,
        )
        cli.deadline = deadline
        evidence = {
            "status": "INCOMPLETE",
            "priced_rate_units_per_hour": tracker.own_rate_units_per_hour,
            "observed_rate_units_per_hour": None,
            "timeout_seconds": RATE_CALIBRATION_SECONDS,
            "interval_seconds": RATE_CALIBRATION_INTERVAL_SECONDS,
            "required_stable_reads": RATE_CALIBRATION_STABLE_READS,
            "stability_units_per_hour": RATE_CALIBRATION_STABILITY_UNITS_PER_HOUR,
            "samples": [],
        }
        tracker.calibration = evidence
        stable: list[float] = []
        endpoint = json.loads(cli.config.read_text())[cli.session]["endpoint"]

        def remaining() -> float:
            seconds = deadline - time.monotonic()
            if seconds <= 0:
                raise Refused("provider rate calibration did not stabilize within bounded window")
            return min(policy.READ_SECONDS, seconds)

        try:
            # Count bound also protects against a stalled/test monotonic clock.
            for sample in range(
                int(RATE_CALIBRATION_SECONDS / RATE_CALIBRATION_INTERVAL_SECONDS) + 1
            ):
                remaining()
                if (
                    time.monotonic() - started
                ) * tracker.own_rate_units_per_hour / 3600 >= tracker.unit_cap:
                    raise Refused("priced session unit cap reached during calibration")
                if cli.renew:
                    cli.renew()
                with contextlib.nullcontext() if account_lock_held else account_locked(cli):
                    assignments = cli.sessions(timeout=remaining())
                    usage = cli.usage(timeout=remaining())
                    remaining()
                    reading = {
                        "elapsed_seconds": time.monotonic() - calibration_started,
                        "sessions": assignments,
                        "usage": usage,
                    }
                    evidence["samples"].append(reading)
                    write_once(output / f"rate-calibration-{sample:03d}.json", reading)
                    if max(len(assignments), usage["assignments"]) > 3:
                        raise Refused("account concurrency changed during allocation")
                    if usage["balance"] <= m["budget"]["reserve_units"]:
                        raise Refused("study reserve reached during calibration")
                    if endpoint in assignments and assignments[endpoint] != cli.session:
                        raise Refused("allocated assignment identity not confirmed by backend")
                    observed = tracker.calibration_read(assignments, usage, endpoint)
                    if observed is None:
                        stable = []
                    else:
                        stable.append(observed)
                        stable = stable[-RATE_CALIBRATION_STABLE_READS:]
                        if (
                            max(stable) - min(stable)
                            > RATE_CALIBRATION_STABILITY_UNITS_PER_HOUR + 1e-9
                        ):
                            stable = [observed]
                    if len(stable) == RATE_CALIBRATION_STABLE_READS:
                        previous = (retained or {}).get("observed_rate_units_per_hour")
                        if (
                            previous is not None
                            and abs(observed - previous) > tracker.rate_tolerance_units_per_hour
                        ):
                            raise Refused("provider usage rate drift in retained assignment")
                        tracker.confirm_own_assignment(assignments, usage, endpoint)
                        active = ACCOUNT_ROOT / f"active-{cli.session}.json"
                        record = json.loads(active.read_text())
                        record["observed_rate_units_per_hour"] = observed
                        replacement = active.with_name(f".calibrated-{uuid.uuid4().hex}.json")
                        try:
                            write_once(replacement, record)
                            os.replace(replacement, active)
                        finally:
                            replacement.unlink(missing_ok=True)
                        write_once(output / "usage-allocated.json", usage)
                        (ACCOUNT_ROOT / f"pending-{cli.session}.json").unlink(missing_ok=True)
                        evidence.update(status="PASS", observed_rate_units_per_hour=observed)
                        return
                delay = min(RATE_CALIBRATION_INTERVAL_SECONDS, max(0, deadline - time.monotonic()))
                if delay:
                    time.sleep(delay)
            raise Refused("provider rate calibration did not stabilize within bounded window")
        except BaseException as exc:
            evidence["error"] = redact(repr(exc))
            raise
        finally:
            cli.deadline = original_deadline
            write_once(output / "rate-calibration.json", evidence)

    def execute(
        self,
        cli: CLI,
        m: dict[str, Any],
        generation: str,
        output: Path,
        started: float,
        baseline: dict[str, float],
        retained: dict[str, Any] | None = None,
        admission_lock: contextlib.ExitStack | None = None,
    ) -> tuple[Any, list[Any]]:
        """Warm the transport, then arm the watchdog before dependency work."""
        root = "/content/snake-cli/" + generation
        cli.wait_runtime_ready(output)
        startup_context = {"cli": cli, "manifest": m, "output": output}
        if self.observer:
            self.observer("runtime_ready", startup_context)
        with contextlib.nullcontext():
            staging = output / "staging"
            # Runtime code comes from the same pinned commit as the job, never
            # the working tree. Embed its small watchdog in the very first RPC.
            pinned = {}
            for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
                path = staging / name
                pinned[name] = path
            wait_code = (
                "import os,sys,time; from pathlib import Path; "
                "root=Path(sys.argv[1]); "
                "exec(\"while not (root/'ready.json').exists(): time.sleep(0.2)\"); "
                "os.execvp('bash',['bash',str(root/'bootstrap.sh'),str(root),sys.argv[2]])"
            )
            spec = {
                "job_id": m["job_id"],
                "generation": generation,
                "root": root,
                "command": ["python3", "-c", wait_code, root, m["script"]],
                "env": {**m["env"], "COLAB_INPUT_DIR": root + "/inputs", "COLAB_CLI_ROOT": root},
                "timeout_seconds": max(1, cli.deadline - time.monotonic() - 25),
                "lease_seconds": m["lease_seconds"],
                "checkpoint_grace_seconds": 15,
                "max_export_bytes": m["max_export_bytes"],
                "public_inputs": [
                    {key: item[key] for key in ("url", "sha256", "size_bytes", "dest")}
                    for item in m.get("inputs", [])
                    if "url" in item
                ],
                "release_endpoint": json.loads(cli.config.read_text())[cli.session]["endpoint"],
            }
            if retained is not None:
                spec["post_terminal_timeout_seconds"] = max(
                    spec["timeout_seconds"], retained["deadline"] - time.monotonic() - 25
                )
            encoded_runtime = base64.b64encode(pinned["runtime.py"].read_bytes()).decode()
            encoded_release = base64.b64encode(pinned["release.py"].read_bytes()).decode()
            credential_remote = f"/content/snake-cli-{generation}-release.json"

            def upload_credential(deadline: float | None = None) -> None:
                with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
                    value = {**cli.release_credential, "endpoint": spec["release_endpoint"]}
                    json.dump(value, handle)
                    handle.flush()
                    from research.colab_cli.transfers import upload_contents

                    upload_contents(cli, Path(handle.name), credential_remote, deadline=deadline)

            upload_credential()
            if self.observer:
                self.observer("first_upload", startup_context)
            lease = {"generation": generation, "renewed_at": time.time()}
            initial = (
                "import base64,json,subprocess,sys,time\nfrom pathlib import Path\n"
                f"root=Path({root!r})\nroot.mkdir(parents=True,exist_ok=False)\n"
                f"(root/'runtime.py').write_bytes(base64.b64decode({encoded_runtime!r}))\n"
                f"(root/'release.py').write_bytes(base64.b64decode({encoded_release!r}))\n"
                f"secret=Path({credential_remote!r})\nsecret.chmod(0o600)\n"
                "secret.replace(root/'release-credential.json')\n"
                f"(root/'spec.json').write_text(json.dumps({spec!r}))\n"
                f"lease={lease!r}\nlease['renewed_at']=time.time()\n"
                "(root/'lease.json').write_text(json.dumps(lease))\n"
                "log=(root/'supervisor.log').open('xb')\n"
                "args=[sys.executable,str(root/'runtime.py'),'--spec',str(root/'spec.json')]\n"
                "p=subprocess.Popen(args,"
                "stdout=log,stderr=subprocess.STDOUT,start_new_session=True)\n"
                "result={'pid':p.pid}"
            )
            # Contents is independent of the kernel: arm renewal before the
            # launch RPC so a slow acknowledgement cannot consume the lease.
            # A PUT before root creation safely returns 404 and is retried at
            # the next heartbeat; it never creates a runtime or kernel job.
            cli.start_lease(root, generation, m["lease_seconds"])
            cli.recoverable_rpc(initial)
            cli.current_root = root
            # Runtime writes state before Popen; wait through file downloads,
            # never by queuing another exec behind an uncertain/busy kernel.
            time.sleep(1)

            renewal = {"at": -math.inf, "state": None}

            def renew(force: bool = False) -> Any:
                # A subprocess per renewal is expensive. Keep network chunks bounded
                # while renewing at most every 30s (well below the 180s minimum lease).
                if not force and time.monotonic() - renewal["at"] < 30:
                    return renewal["state"]
                from research.colab_cli.release import (
                    credential_renewal_due,
                    mint_release_credential,
                )

                # The independent Contents heartbeat covers slow credential
                # work; rotation borrows no time beyond the first real cap.
                result = cli.renew_runtime_lease(root, generation, force=force)
                if credential_renewal_due(cli.release_credential, "", safety_window_seconds=900):
                    cli.release_credential = mint_release_credential(
                        cli.oauth_python,
                        endpoint="",
                        min_valid_seconds=m["lease_seconds"] + 900,
                        timeout=cli.remaining_seconds(150),
                    )
                    credential_deadline = time.monotonic() + cli.remaining_seconds(150)
                    upload_credential(credential_deadline)
                    cli.recoverable_rpc(
                        "from pathlib import Path\n"
                        f"p=Path({credential_remote!r})\np.chmod(0o600)\n"
                        f"p.replace(Path({root!r})/'release-credential.json')\n"
                        "result=True",
                    )
                self.renew_retained(cli, retained)
                renewal["at"] = time.monotonic()
                if force:
                    renewal["state"] = result
                return renewal["state"]

            cli.renew = renew
            renew()
            self.calibrate_rate(
                cli, m, output, started, retained, account_lock_held=admission_lock is not None
            )
            # Until calibration is durable, no other lane can observe an active
            # record with an unknown rate. Waiting lanes renew their watchdogs.
            if admission_lock is not None:
                admission_lock.close()
            if self.observer:
                self.observer(
                    "runtime_started",
                    {
                        "cli": cli,
                        "manifest": m,
                        "generation": generation,
                        "root": root,
                        "output": output,
                    },
                )
            from research.colab_cli.transfers import UPLOAD_TOTAL_TIMEOUT_SECONDS, upload_file

            upload_started = time.monotonic()
            upload_deadline = upload_started + UPLOAD_TOTAL_TIMEOUT_SECONDS
            transfer_seconds = 0.0
            source_delivery_seconds = 0.0

            def stage_upload(path: Path, dest: str) -> None:
                nonlocal transfer_seconds
                started = time.monotonic()
                upload_file(cli, path, dest, renew=renew, deadline=upload_deadline)
                # Include verified-upload hashing, retries and lease renewals,
                # while excluding the runtime's source fetch/install work.
                transfer_seconds += time.monotonic() - started

            local_files = [
                (pinned["bootstrap.sh"], "bootstrap.sh", ""),
                (pinned["source.py"], "source.py", ""),
            ]
            local_files += [
                (Path(i["path"]), "inputs/" + i["dest"], i["sha256"])
                for i in m.get("inputs", [])
                if "path" in i
            ]
            with cli.staging_controls(upload_deadline):
                checks = {}
                setup_started = time.monotonic()
                parents = sorted(
                    {str(Path(root + "/" + dest).parent) for _, dest, _ in local_files}
                )
                with cli.operation("mkdir"):
                    cli.poll_rpc(
                        "# SNAKE_STAGING_PARENTS\nfrom pathlib import Path\n"
                        f"parents={parents!r}\n"
                        "for parent in parents: Path(parent).mkdir(parents=True,exist_ok=True)\n"
                        "result=True"
                    )
                setup_seconds = time.monotonic() - setup_started
                for path, dest, digest in local_files:
                    if path.stat().st_size > m["max_export_bytes"]:
                        raise Refused("upload exceeds bounded transfer size")
                    digest = digest or sha256(path)
                    stage_upload(path, root + "/" + dest)
                    checks[dest] = digest
                checks["runtime.py"] = sha256(pinned["runtime.py"])
                checks["release.py"] = sha256(pinned["release.py"])
                # The runtime downloads source directly from GitHub, rather than
                # passing every committed evidence blob through ContentsClient.
                plan = json.loads((staging / "source-plan.json").read_text())

                def deliver_source(operation: str) -> Any:
                    nonlocal source_delivery_seconds
                    started = time.monotonic()
                    renew()
                    result = cli.recoverable_rpc(
                        "# SNAKE_SOURCE_DELIVERY\n"
                        "import importlib.util\nfrom pathlib import Path\n"
                        f"root=Path({root!r})\nplan={plan!r}\n"
                        "s=importlib.util.spec_from_file_location("
                        "'snake_source',root/'source.py')\n"
                        "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
                        f"result=mod.{operation}(root,plan)",
                    )
                    renew()
                    source_delivery_seconds += time.monotonic() - started
                    return result

                delivery = deliver_source("fetch")
                if not isinstance(delivery, dict):
                    raise Refused("source delivery returned no identity evidence")
                if delivery.get("transport") == "fallback_required":
                    path = staging / "source.tar.gz"
                    stage_upload(path, root + "/source.tar.gz")
                    checks["source.tar.gz"] = sha256(path)
                    delivery = deliver_source("install_fallback")
                if (
                    not isinstance(delivery, dict)
                    or delivery.get("transport") not in {"github", "fallback"}
                    or delivery.get("commit") != plan["commit"]
                    or delivery.get("tree") != plan["tree"]
                ):
                    raise Refused("source delivery commit/tree identity mismatch")
            measurement = {
                "elapsed_seconds": time.monotonic() - upload_started,
                "transfer_seconds": transfer_seconds,
                "source_delivery_seconds": source_delivery_seconds,
                "setup_seconds": setup_seconds,
            }
            write_once(output / "upload-measurement.json", measurement)
            write_once(output / "source-delivery.json", delivery)
            write_once(output / "input-hashes.json", checks)
            # A probe's upload/staging bars are admission controls, not a
            # post-success qualification check. Preserve all measurements for
            # diagnosis, then refuse before ready.json releases the child.
            from research.colab_cli.canary import StagingCapError, is_canary, staging_measurement
            from research.colab_cli.failures import ClassifiedFailure, FailureClass

            if is_canary(m):
                try:
                    staging_measurement(m, measurement)
                except StagingCapError as exc:
                    raise ClassifiedFailure(FailureClass.UPLOAD_CAP, str(exc)) from exc
            # Hardware admission and process launch are short RPCs. Bootstrap
            # runs in the lease-fenced child, never in the notebook kernel.
            code = (
                "import hashlib, json, os, subprocess, sys, time\n"
                "from pathlib import Path\n"
                f"root=Path({root!r})\nchecks={checks!r}\n"
                "for rel, expected in checks.items():\n"
                "    with (root/rel).open('rb') as f:\n"
                "        assert hashlib.file_digest(f, 'sha256').hexdigest()==expected, rel\n"
                f"assert (os.cpu_count() or 1)>={m['requirements']['min_vcpu']}\n"
                "repo=root/'repo'\nassert repo.is_dir()\n"
                "(root/'ready.json').write_text(json.dumps({'ready':True}))\n"
                "result={'ready':True}"
            )
            # Price and admit substitutions explicitly, before bootstrap or training.
            code = code.replace(
                "(root/'ready.json').write_text(json.dumps({'ready':True}))\n"
                "result={'ready':True}",
                "import shutil\n"
                "gpu={'name':'cpu','kind':'cpu','compute_cap':None}\n"
                "if shutil.which('nvidia-smi'):\n"
                "    q=subprocess.run(['nvidia-smi','--query-gpu=name,compute_cap',"
                "'--format=csv,noheader'],capture_output=True,text=True,timeout=5)\n"
                "    if q.returncode==0 and q.stdout.strip():\n"
                "        name,cap=q.stdout.splitlines()[0].rsplit(',',1)\n"
                "        kind=next((g for g in ['H100','A100','L4','T4'] if g in name),"
                "'G4' if float(cap)>=10 else 'unknown')\n"
                "        gpu={'name':name,'kind':kind,'compute_cap':cap.strip()}\n"
                f"assert gpu['kind'] in {m['requirements']['allowed_gpus']!r},gpu\n"
                "(root/'ready.json').write_text(json.dumps({'ready':True,'hardware':gpu}))\n"
                "result={'ready':True,'hardware':gpu}",
            )
            admitted = cli.recoverable_rpc(code)
            write_once(output / "hardware.json", admitted)
            if self.observer:
                self.observer(
                    "ready",
                    {
                        "cli": cli,
                        "manifest": m,
                        "generation": generation,
                        "root": root,
                        "output": output,
                    },
                )
        exported = cli.verified_exports
        poll = 0
        while True:
            elapsed = time.monotonic() - started
            if elapsed >= m["timeout_minutes"] * 60 - 45:
                raise Refused("whole-session wall cap reached (including setup/export)")
            if elapsed * m["budget"]["rate_units_per_hour"] / 3600 >= m["budget"]["unit_cap"]:
                raise Refused("priced session unit cap reached")
            if poll % 4 == 0:
                with account_locked(cli):
                    current = cli.usage()
                    spend = cli.accounting.check(
                        current,
                        sessions=cli.sessions(),
                        active_records=[
                            json.loads(p.read_text())
                            for p in ACCOUNT_ROOT.glob("active-*.json")
                            if p.name != f"active-{cli.session}.json"
                        ],
                        budget_other_assignments=m["budget"].get("other_assignments"),
                    )
                write_once(output / f"usage-{poll:06d}.json", current)
                write_once(output / f"spend-{poll:06d}.json", spend)
                if spend.get("provider_transition_status") != "OK":
                    print(
                        json.dumps(
                            {
                                "WARNING": "BROKER_TRANSITION_PROVIDER_LAG",
                                "job": m["job_id"],
                                "spend": spend,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if spend.get("balance_crosscheck_status") != "OK":
                    print(
                        json.dumps(
                            {
                                "ALERT": "UNEXPLAINED_ACCOUNT_DEBIT",
                                "job": m["job_id"],
                                "spend": spend,
                            }
                        ),
                        file=sys.stderr,
                        flush=True,
                    )
                if current["assignments"] > 3:
                    raise Refused("account-wide assignment cap exceeded during job")
                if current["balance"] <= m["budget"]["reserve_units"]:
                    raise Refused("study reserve reached")
            state = renew(force=True)
            write_once(output / f"poll-{poll:06d}.json", state)
            print(
                json.dumps(
                    {
                        "job": m["job_id"],
                        "elapsed": round(elapsed, 1),
                        "status": state.get("state", {}).get("status"),
                        "terminal": bool(state.get("terminal")),
                        "exports": len(
                            (state.get("terminal") or {}).get("exports", state.get("exports", []))
                        ),
                        "job_log_tail": state.get("job_log_tail", "")[-4096:],
                    }
                ),
                flush=True,
            )
            terminal = state.get("terminal")
            if terminal:
                from research.colab_cli.failures import FailureClass, terminal_failure

                failure = terminal_failure(terminal, generation)
                if failure and failure.failure_class == FailureClass.FOREIGN_GENERATION:
                    raise failure
                if not terminal.get("last_traceback_line") and terminal.get("returncode") != 0:
                    terminal = {
                        **terminal,
                        "last_traceback_line": next(
                            (
                                line
                                for line in reversed(state.get("job_log_tail", "").splitlines())
                                if line.strip()
                            ),
                            None,
                        ),
                    }
                # Persist observed completion before any export RPC can fail.
                write_once(output / "terminal.json", terminal)
            inventory = (
                state.get("terminal", {}).get("exports", [])
                if state.get("terminal")
                else state.get("exports", [])
            )
            collect_exports(cli, root, inventory, output, m["max_export_bytes"])
            if terminal:
                from research.colab_cli.schema import SchemaError, validate_export_inventory

                failure = terminal_failure(terminal, generation)
                if failure is not None:
                    raise failure
                try:
                    validate_export_inventory(m, list(exported.values()))
                except SchemaError as exc:
                    raise Refused(str(exc)) from exc
                if self.observer:
                    self.observer(
                        "terminal",
                        {
                            "cli": cli,
                            "manifest": m,
                            "generation": generation,
                            "root": root,
                            "output": output,
                            "terminal": terminal,
                            "exports": list(exported.values()),
                        },
                    )
                return terminal, list(exported.values())
            poll += 1
            time.sleep(15)

    @staticmethod
    def backup(output: Path, backup_dir: str, job: str, generation: str) -> None:
        """Copy only after remote release; hung Drive I/O has a 60-second bound."""
        destination = Path(backup_dir).expanduser() / job / generation
        subprocess.run(
            [
                sys.executable,
                "-m",
                "research.colab_cli.backup",
                str(output / "exports"),
                str(destination),
            ],
            cwd=REPO,
            check=True,
            timeout=60,
        )
        write_once(output / "backup.json", {"destination": str(destination), "hash_verified": True})

    @staticmethod
    def renew_retained(cli: CLI, retained: dict[str, Any] | None) -> None:
        """Renew completed-job watchdogs only while approved useful work follows."""
        for old_root, generation in (retained or {}).get("completed", []):
            cli.poll_rpc(
                "import json,time\nfrom pathlib import Path\n"
                f"p=Path({old_root!r})/'lease.json'\nlease=json.loads(p.read_text())\n"
                f"assert lease['generation']=={generation!r}\n"
                f"credential=Path({getattr(cli, 'current_root', old_root)!r})\n"
                "credential=credential/'release-credential.json'\n"
                "target=p.parent/'release-credential.json'\n"
                "if credential != target:\n"
                "    new=target.with_suffix('.rotation.tmp')\n"
                "    new.write_bytes(credential.read_bytes())\n    new.chmod(0o600)\n"
                "    new.replace(target)\n"
                "lease['renewed_at']=time.time()\n"
                "temp=p.with_suffix('.renew.tmp')\n"
                "temp.write_text(json.dumps(lease))\ntemp.replace(p)\nresult=True"
            )

    def chain(
        self, manifests: list[dict[str, Any]], retention_minutes: float, retention_units: float
    ) -> None:
        """Keep an obtained H100 only for a finite already-approved job list."""
        jobs = [validate(m) for m in manifests]
        if len(jobs) < 2 or any(m["requirements"]["gpu"] != "H100" for m in jobs):
            raise Refused("chain requires >=2 approved H100 jobs; no hardware fallback")
        if len({m["job_id"] for m in jobs}) != len(jobs):
            raise Refused("duplicate chain job identifiers")
        rate = max(m["budget"]["rate_units_per_hour"] for m in jobs)
        minutes = positive(retention_minutes, "retention_minutes")
        units = positive(retention_units, "retention_units")
        if sum(m["timeout_minutes"] for m in jobs) > minutes or minutes * rate / 60 > units:
            raise Refused("approved chain does not fit priced retention lease")
        started = time.monotonic()
        retained: dict[str, Any] = {"deadline": started + minutes * 60 - 45, "unit_cap": units}
        failure = None
        try:
            for m in jobs:
                if time.monotonic() - started >= minutes * 60:
                    raise Refused("retention lease elapsed; no further job")
                self.run(m, retained)
        except BaseException as exc:
            failure = exc
            raise
        finally:
            if retained.get("cli"):
                cli = retained["cli"]
                release_error = None
                try:
                    with account_locked(cli, cleanup=True):
                        stop_verified(cli, retained["endpoint"])
                        retained["active"].unlink(missing_ok=True)
                        retained["intent"].unlink(missing_ok=True)
                except BaseException as exc:
                    release_error = repr(exc)
                for out in retained["outputs"]:
                    write_once(
                        Path(out) / "session-release.json",
                        {
                            "endpoint": retained["endpoint"],
                            "stop_verified": release_error is None,
                            "error": release_error,
                            "chain_error": repr(failure) if failure else None,
                            "priced_retention_minutes": minutes,
                            "priced_retention_units": units,
                            "elapsed_seconds": time.monotonic() - started,
                        },
                    )
                if release_error and failure is None:
                    raise Refused(release_error)
                if release_error is None:
                    for spec in retained.get("backups", []):
                        self.backup(*spec)

    def recover(self, pending: Path) -> None:
        """Stop a known broker-owned orphan; never infer identity from hardware."""
        if pending.parent.resolve() != ACCOUNT_ROOT.resolve():
            raise Refused("recovery requires an account broker pending/active record")
        record = json.loads(pending.read_text())
        cli = CLI(self.binary, Path(record["config"]), record["session"])
        with account_locked(cli, cleanup=True):
            metadata = json.loads(cli.config.read_text()) if cli.config.exists() else {}
            recorded_endpoint = metadata.get(cli.session, {}).get("endpoint")
            sessions = cli.sessions()
            endpoint = (
                record.get("endpoint")
                or recorded_endpoint
                or next((e for e, name in sessions.items() if name == record["session"]), None)
            )
            if not endpoint and record.get("assignment_record"):
                assignment_record = Path(record["assignment_record"])
                if assignment_record.exists():
                    endpoint = json.loads(assignment_record.read_text()).get("endpoint")
            if not endpoint and record.get("notebook_hash"):
                from research.colab_cli.cli_guard import recover_assignment

                endpoint = recover_assignment(cli.oauth_python, record["notebook_hash"])
            if not endpoint:
                candidates = sorted(set(sessions) - set(record.get("before", sessions)))
                raise Refused(
                    f"allocation identity uncertain (new endpoints={candidates!r}); "
                    "preserve record for independent reconciliation"
                )
            stop_verified(cli, endpoint)
            write_once(
                Path(record["output"]) / f"recovery-{uuid.uuid4().hex}.json",
                {"endpoint": endpoint, "stop_verified": True},
            )
            pending.unlink()
            (ACCOUNT_ROOT / f"active-{cli.session}.json").unlink(missing_ok=True)
            (ACCOUNT_ROOT / f"pending-{cli.session}.json").unlink(missing_ok=True)

    def status(self) -> dict[str, Any]:
        """Read account and local jobs without allocating or executing kernels."""
        cli = CLI(self.binary, self.state_dir / "status-sessions.json")
        jobs = []
        for lane in sorted(self.state_dir.iterdir()):
            if lane.is_dir():
                record = lane / "assignment.json"
                assignment = json.loads(record.read_text()) if record.exists() else None
                receipt = (
                    Path(assignment["output"]) / "receipt.json"
                    if assignment and assignment.get("output")
                    else None
                )
                jobs.append(
                    {
                        "lane": lane.name,
                        "assignment": assignment,
                        "receipt": (
                            json.loads(receipt.read_text())
                            if receipt and receipt.exists()
                            else None
                        ),
                    }
                )
        return {
            "usage": cli.usage(),
            "sessions": cli.sessions(),
            "jobs": jobs,
            "unresolved": [str(p) for p in ACCOUNT_ROOT.glob("pending-*.json")],
            "active": [json.loads(p.read_text()) for p in ACCOUNT_ROOT.glob("active-*.json")],
        }


def main() -> None:
    """Dispatch explicit manifests; default behavior never spends units."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--state-dir", type=Path, default=ACCOUNT_ROOT / "lanes")
    parser.add_argument("--colab", default=str(Path.home() / ".local/bin/colab"))
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("validate", "run"):
        sub.add_parser(name).add_argument("manifest", type=Path)
    sub.add_parser("status")
    sub.add_parser("recover").add_argument("record", type=Path)
    chain = sub.add_parser("chain")
    chain.add_argument("manifests", nargs="+", type=Path)
    chain.add_argument("--retention-minutes", type=float, required=True)
    chain.add_argument("--retention-units", type=float, required=True)
    args = parser.parse_args()
    try:
        if args.command == "validate":
            print(json.dumps(validate(json.loads(args.manifest.read_text())), indent=2))
        elif args.command == "status":
            print(json.dumps(Broker(args.state_dir, args.colab).status(), indent=2))
        elif args.command == "recover":
            Broker(args.state_dir, args.colab).recover(args.record)
        elif args.command == "chain":
            Broker(args.state_dir, args.colab).chain(
                [json.loads(p.read_text()) for p in args.manifests],
                args.retention_minutes,
                args.retention_units,
            )
        else:
            Broker(args.state_dir, args.colab).run(json.loads(args.manifest.read_text()))
    except (Refused, ValueError, KeyError, OSError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
