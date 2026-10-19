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
import tarfile
import tempfile
import time
import uuid
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Iterator

ACCOUNT_ROOT = Path.home() / ".config/colab-cli/snake"
REPO = Path(__file__).resolve().parents[2]
OAUTH_PYTHON = str(Path.home() / ".venvs/colab-cli/bin/python")
ANSI = re.compile(r"\x1b\[[0-9;]*m")
RATE_CALIBRATION_SECONDS = 120.0
RATE_CALIBRATION_INTERVAL_SECONDS = 5.0
RATE_CALIBRATION_STABLE_READS = 3
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


def validate(manifest: dict[str, Any], repo: Path = REPO) -> dict[str, Any]:
    """Validate inputs and a conservatively priced whole-session envelope."""
    m = json.loads(json.dumps(manifest))
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
        dest = safe_relative(item["dest"])
        if dest in destinations:
            raise Refused("duplicate input destination")
        destinations.add(dest)
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
    return m


class CLI:
    """Pinned official CLI as a bounded subprocess, never a shell command."""

    def __init__(self, binary: str, config: Path, session: str = "") -> None:
        self.binary, self.config, self.session = binary, config, session
        self.exec_uncertain = False
        self.deadline: float | None = None
        self.priced_deadline: float | None = None
        self.notebook_hash = uuid.uuid4().hex
        self.assignment_record = config.with_suffix(".assignment.json")
        self.oauth_python = os.environ.get("SNAKE_COLAB_OAUTH_PYTHON", OAUTH_PYTHON)
        self.renew: Callable[[], Any] | None = None

    def call(self, *args: str, timeout: float = 45) -> str:
        if self.deadline is not None and args[0] not in {"stop", "sessions"}:
            timeout = min(timeout, self.deadline - time.monotonic())
            if timeout <= 0:
                raise Refused("whole-session deadline reached before CLI call")
        env = {
            **os.environ,
            "SNAKE_COLAB_NOTEBOOK_HASH": self.notebook_hash,
            "SNAKE_COLAB_ASSIGNMENT_RECORD": str(self.assignment_record),
            "NO_COLOR": "1",
            "COLUMNS": "240",
        }
        self.config.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        hook = self.config.parent / "cli-hook"
        hook.mkdir(exist_ok=True, mode=0o700)
        guard_path = Path(__file__).with_name("cli_guard.py")
        (hook / "sitecustomize.py").write_text(
            "import os,runpy\ntry:\n    runpy.run_path("
            + repr(str(guard_path))
            + ", run_name='snake_cli_guard')\nexcept BaseException:\n    os._exit(70)\n"
        )
        env["PYTHONPATH"] = str(hook) + os.pathsep + env.get("PYTHONPATH", "")
        try:
            result = subprocess.run(
                [self.binary, "--auth", "oauth2", "--config", str(self.config), *args],
                capture_output=True,
                text=True,
                timeout=timeout,
                env=env,
                umask=0o077,
            )
        finally:
            if self.config.exists():
                self.config.chmod(0o600)
        output = redact(ANSI.sub("", result.stdout + result.stderr))
        if result.returncode:
            raise Refused(f"colab {args[0]} failed ({result.returncode}): {output[-2000:]}")
        return output

    def sessions(self, timeout: float = 45) -> dict[str, str]:
        output = self.call("sessions", timeout=timeout)
        rows = re.findall(r"^\[([^\]]+)\]\s+(\S+)\s+\| Hardware:", output, re.M)
        if not rows and "No active sessions found on server." not in output:
            raise Refused("unrecognized account sessions output")
        return {endpoint: name for name, endpoint in rows}

    def usage(self, timeout: float = 45) -> dict[str, float]:
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

    def rpc(self, code: str, timeout: float = 20) -> Any:
        """A nonce response proves kernel completion even when CLI exits zero."""
        if self.exec_uncertain:
            raise Refused("previous exec uncertain/busy; refusing to resubmit to kernel")
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
                    timeout=timeout + 10,
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

    def poll_rpc(self, code: str, timeout: float = 20) -> Any:
        """Retry only idempotent telemetry after independent proof of kernel idle."""
        from research.colab_cli.cli_guard import kernel_idle

        for attempt in range(3):
            try:
                return self.rpc(code, timeout)
            except (Refused, subprocess.SubprocessError):
                if not self.exec_uncertain or attempt == 2:
                    raise
                # Never enqueue work while the previous exec could still be busy.
                # The bounded REST probe executes no kernel code.
                idle = False
                for _ in range(3):
                    remaining = (self.deadline or (time.monotonic() + 60)) - time.monotonic()
                    if remaining <= 1:
                        raise Refused("deadline reached during telemetry recovery")
                    if kernel_idle(
                        self.oauth_python, self.config, self.session, timeout=min(45, remaining)
                    ):
                        idle = True
                        break
                    time.sleep(2)
                if not idle:
                    raise Refused("kernel remains busy/unknown after telemetry hiccup")
                self.exec_uncertain = False
        raise Refused("telemetry recovery exhausted")


def stop_verified(cli: CLI, endpoint: str) -> None:
    """A stop exit code cannot prove release; independently query the account."""
    errors = []
    for attempt in range(3):
        try:
            cli.call("stop", "-s", cli.session, timeout=75)
        except Exception as exc:
            errors.append(redact(repr(exc)))
        # Reissue the stop while listed; transient CLI lookup errors prove nothing.
        if attempt > 0:
            try:
                from research.colab_cli.cli_guard import unassign_endpoint

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


def export_file(cli: CLI, root: str, entry: dict[str, Any], output: Path, limit: int) -> None:
    """Download immutable data + digest + marker, verify, publish atomically."""
    from research.colab_cli.transfers import TransferError, download_file

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
            cli.call(
                "download",
                "-s",
                cli.session,
                root + "/output/" + rel + suffix,
                str(local),
                timeout=60,
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


def build_bundle(commit: str, directory: Path) -> Path:
    """Archive only committed source; reject links and traversal before upload."""
    bundle = directory / "source.tar"
    subprocess.run(
        ["git", "archive", "--format=tar", "-o", str(bundle), commit],
        cwd=REPO,
        check=True,
        timeout=60,
    )
    with tarfile.open(bundle) as archive:
        for item in archive:
            safe_relative(item.name.rstrip("/"))
            if not (item.isfile() or item.isdir()):
                raise Refused("source bundle contains links/special files")
    return bundle


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
        job = m["job_id"]
        lane = self.state_dir / job
        with locked(ACCOUNT_ROOT / f"lane-{job}.lock", blocking=False):
            lane.mkdir(exist_ok=False, mode=0o700)
            output = Path(m["output_dir"])
            output.mkdir(parents=True, exist_ok=False)
            write_once(output / "manifest.json", m)
            staging = output / "staging"
            staging.mkdir()
            build_bundle(m["commit"], staging)
            for name in ("runtime.py", "bootstrap.sh", "release.py"):
                data = subprocess.check_output(
                    ["git", "show", f"{m['commit']}:research/colab_cli/{name}"],
                    cwd=REPO,
                    timeout=10,
                )
                (staging / name).write_bytes(data)
            for path in staging.iterdir():
                if path.stat().st_size > m["max_export_bytes"]:
                    raise Refused("pinned upload exceeds bounded transfer size before allocation")
            generation = uuid.uuid4().hex
            session = f"snake-{job}-{generation[:12]}"
            cli = CLI(self.binary, lane / "sessions.json", session)
            if retained and retained.get("cli"):
                cli = retained["cli"]
                session = cli.session
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
                        cli.oauth_python, endpoint="", min_valid_seconds=m["lease_seconds"] + 300
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
                        cli.call(*args, timeout=60)
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
                    error = redact(repr(exc))
                finally:
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
                        "terminal": terminal,
                        "exports": exports,
                        "error": error,
                        "cleanup_error": cleanup,
                        "status": (
                            ("COMPLETED_SESSION_RETAINED" if keep else "PASS")
                            if not error and not cleanup
                            else "INCOMPLETE"
                        ),
                        "stop_verified": bool(endpoint and not cleanup and not keep),
                    }
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
            return min(20.0, seconds)

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
        """Arm the runtime watchdog before starting any slow dependency work."""
        root = "/content/snake-cli/" + generation
        with contextlib.nullcontext():
            staging = output / "staging"
            source = staging / "source.tar"
            # Runtime code comes from the same pinned commit as the job, never
            # the working tree. Embed its small watchdog in the very first RPC.
            pinned = {}
            for name in ("runtime.py", "bootstrap.sh", "release.py"):
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
                "release_endpoint": json.loads(cli.config.read_text())[cli.session]["endpoint"],
            }
            if retained is not None:
                spec["post_terminal_timeout_seconds"] = max(
                    spec["timeout_seconds"], retained["deadline"] - time.monotonic() - 25
                )
            encoded_runtime = base64.b64encode(pinned["runtime.py"].read_bytes()).decode()
            encoded_release = base64.b64encode(pinned["release.py"].read_bytes()).decode()
            credential_remote = f"/content/snake-cli-{generation}-release.json"

            def upload_credential() -> None:
                with tempfile.NamedTemporaryFile("w", suffix=".json") as handle:
                    value = {**cli.release_credential, "endpoint": spec["release_endpoint"]}
                    json.dump(value, handle)
                    handle.flush()
                    cli.call(
                        "upload", "-s", cli.session, handle.name, credential_remote, timeout=60
                    )

            upload_credential()
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
            cli.rpc(initial, timeout=30)
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

                if credential_renewal_due(cli.release_credential, "", safety_window_seconds=900):
                    cli.release_credential = mint_release_credential(
                        cli.oauth_python, endpoint="", min_valid_seconds=m["lease_seconds"] + 300
                    )
                    upload_credential()
                    cli.rpc(
                        "from pathlib import Path\n"
                        f"p=Path({credential_remote!r})\np.chmod(0o600)\n"
                        f"p.replace(Path({root!r})/'release-credential.json')\nresult=True"
                    )
                result = cli.poll_rpc(
                    "import importlib.util\n"
                    f"runtime_path={root + '/runtime.py'!r}\n"
                    "s=importlib.util.spec_from_file_location('snake_runtime',runtime_path)\n"
                    "mod=importlib.util.module_from_spec(s)\ns.loader.exec_module(mod)\n"
                    f"result=mod.read_state({root!r},{generation!r},renew=True)"
                )
                self.renew_retained(cli, retained)
                renewal.update(at=time.monotonic(), state=result)
                return result

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
            local_files = [
                (source, "source.tar", sha256(source)),
                (pinned["bootstrap.sh"], "bootstrap.sh", ""),
            ]
            local_files += [
                (Path(i["path"]), "inputs/" + i["dest"], i["sha256"]) for i in m.get("inputs", [])
            ]
            checks = {}
            for path, dest, digest in local_files:
                if path.stat().st_size > m["max_export_bytes"]:
                    raise Refused("upload exceeds bounded transfer size")
                digest = digest or sha256(path)
                renew()
                cli.rpc(
                    "from pathlib import Path\n"
                    f"parent=Path({(root + '/' + dest)!r}).parent\n"
                    "parent.mkdir(parents=True,exist_ok=True)\nresult=True"
                )
                from research.colab_cli.transfers import upload_file

                upload_file(cli, path, root + "/" + dest, renew=renew)
                checks[dest] = digest
            checks["runtime.py"] = sha256(pinned["runtime.py"])
            checks["release.py"] = sha256(pinned["release.py"])
            write_once(output / "input-hashes.json", checks)
            # Extraction and process launch are short RPCs. Never bootstrap in the kernel.
            code = (
                "import hashlib, json, os, subprocess, sys, tarfile, time\n"
                "from pathlib import Path\n"
                f"root=Path({root!r})\nchecks={checks!r}\n"
                "for rel, expected in checks.items():\n"
                "    with (root/rel).open('rb') as f:\n"
                "        assert hashlib.file_digest(f, 'sha256').hexdigest()==expected, rel\n"
                f"assert (os.cpu_count() or 1)>={m['requirements']['min_vcpu']}\n"
                "repo=root/'repo'\nrepo.mkdir(exist_ok=False)\n"
                "with tarfile.open(root/'source.tar') as archive:\n"
                "    archive.extractall(repo, filter='data')\n"
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
            admitted = cli.rpc(code, timeout=30)
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
        exported: dict[str, Any] = {}
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
                json.dumps({"job": m["job_id"], "elapsed": round(elapsed, 1), "state": state}),
                flush=True,
            )
            inventory = (
                state.get("terminal", {}).get("exports", [])
                if state.get("terminal")
                else state.get("exports", [])
            )
            for entry in inventory:
                if entry["path"] in exported and entry != exported[entry["path"]]:
                    raise Refused("immutable snapshot metadata changed")
                export_file(cli, root, entry, output, m["max_export_bytes"])
                exported[entry["path"]] = entry
                cli.rpc(f"result=mod.read_state({root!r},{generation!r},renew=True)")
                self.renew_retained(cli, retained)
            if state.get("terminal"):
                terminal = state["terminal"]
                write_once(output / "terminal.json", terminal)
                if terminal.get("generation") != generation or terminal.get("returncode") != 0:
                    raise Refused("remote terminal failed or foreign generation")
                if terminal.get("status") not in {"completed", "success", "succeeded", "PASS"}:
                    raise Refused(f"remote terminal not successful: {terminal.get('status')}")
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
            cli.rpc(
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
