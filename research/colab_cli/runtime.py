#!/usr/bin/env python3
"""Runtime-side, lease-fenced subprocess supervisor for Colab CLI jobs.

This module deliberately has no Colab CLI dependency.  A short ``colab exec``
cell imports :func:`read_state` to renew the lease and read immutable exports;
the long-lived supervisor owns the child process and the runtime unassignment.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import URLError
from urllib.request import Request, urlopen

STATE_INTERVAL_SECONDS = 15
# Three release attempts each perform GET and POST requests with 20-second
# bounds plus one second of request overhead, then two retry gaps.  A SIGTERM
# can wait a further 10 seconds after checkpoint grace.  The release module
# carries the matching timeout arithmetic; keep this Popen guard explicit.
RELEASE_RETRY_WINDOW_SECONDS = 130
STOP_CHILD_TAIL_SECONDS = 10
PUBLIC_INPUT_FETCH_ATTEMPTS = 3
PUBLIC_INPUT_FETCH_TIMEOUT_SECONDS = 30
PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS = 300
PUBLIC_INPUT_READ_BYTES = 1024 * 1024


class SpecError(ValueError):
    """A malformed or unsafe runtime specification."""


def _release_module() -> Any:
    """Load the co-shipped release client when runtime.py runs as a script."""
    path = Path(__file__).with_name("release.py")
    module_spec = importlib.util.spec_from_file_location("snake_colab_release", path)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError("runtime release module is unavailable")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module


def _now() -> float:
    return time.time()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _create_json(path: Path, value: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        "w", dir=path.parent, prefix=".receipt-", delete=False
    ) as handle:
        temporary = Path(handle.name)
        try:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
            os.link(temporary, path)
        except FileExistsError as exc:
            raise RuntimeError(f"create-only receipt already exists: {path}") from exc
        finally:
            temporary.unlink(missing_ok=True)


def _load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise RuntimeError(f"expected object in {path}")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_spec(
    raw: Mapping[str, Any], *, allow_existing_output: bool = False
) -> dict[str, Any]:
    required = {
        "job_id": str,
        "generation": (str, int),
        "root": str,
        "command": list,
        "env": dict,
        "timeout_seconds": (int, float),
        "lease_seconds": (int, float),
        "checkpoint_grace_seconds": (int, float),
        "max_export_bytes": int,
        "release_endpoint": str,
    }
    for key, kinds in required.items():
        if key not in raw or not isinstance(raw[key], kinds):
            raise SpecError(f"spec.{key} has invalid type")
    if (
        not raw["job_id"]
        or not raw["command"]
        or not all(isinstance(x, str) and x for x in raw["command"])
    ):
        raise SpecError("job_id and every command item must be non-empty strings")
    if any(
        isinstance(raw[key], bool) or raw[key] <= 0
        for key in (
            "timeout_seconds",
            "lease_seconds",
            "checkpoint_grace_seconds",
            "max_export_bytes",
        )
    ):
        raise SpecError("timeout, lease, grace, and export cap must be positive")
    if raw["checkpoint_grace_seconds"] > 30:
        raise SpecError("checkpoint_grace_seconds must be <= 30")
    try:
        _release_module()._unassign_url(raw["release_endpoint"])
    except Exception as exc:
        raise SpecError(f"release_endpoint is invalid: {exc}") from exc
    if "post_terminal_timeout_seconds" in raw and (
        isinstance(raw["post_terminal_timeout_seconds"], bool)
        or not isinstance(raw["post_terminal_timeout_seconds"], (int, float))
        or raw["post_terminal_timeout_seconds"] <= 0
    ):
        raise SpecError("post_terminal_timeout_seconds must be positive when provided")
    root = Path(raw["root"]).resolve()
    if root == Path("/") or not str(root).startswith("/content/snake-cli/"):
        raise SpecError("root must be a pre-created /content/snake-cli/<uuid> directory")
    if not root.is_dir():
        raise SpecError(f"root was not pre-created: {root}")
    output = root / "output"
    if output.exists() and not allow_existing_output:
        raise SpecError(f"fresh output directory required; already exists: {output}")
    for key, value in raw["env"].items():
        if not isinstance(key, str) or not isinstance(value, str):
            raise SpecError("env must map strings to strings")
    public_inputs = raw.get("public_inputs", [])
    if not isinstance(public_inputs, list):
        raise SpecError("spec.public_inputs must be a list")
    destinations: set[str] = set()
    for item in public_inputs:
        if not isinstance(item, dict):
            raise SpecError("public input must be an object")
        url, digest, size, dest = (
            item.get("url"),
            item.get("sha256"),
            item.get("size_bytes"),
            item.get("dest"),
        )
        if (
            not isinstance(url, str)
            or not url.startswith("https://")
            or any(c.isspace() for c in url)
        ):
            raise SpecError("public input URL must be HTTPS")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(c not in "0123456789abcdef" for c in digest)
        ):
            raise SpecError("public input sha256 must be lowercase SHA-256")
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise SpecError("public input size_bytes must be nonnegative")
        if (
            not isinstance(dest, str)
            or not dest
            or dest.startswith("/")
            or any(part in {"", ".", ".."} for part in dest.split("/"))
            or dest in destinations
        ):
            raise SpecError("public input destination is unsafe or duplicated")
        destinations.add(dest)
    return {**raw, "root": str(root)}


def _fetch_public_inputs(root: Path, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Fetch pinned release objects into a temporary path then atomically publish."""
    deadline = time.monotonic() + PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS
    verified: list[dict[str, Any]] = []
    inputs = root / "inputs"
    for row in rows:
        target = inputs / row["dest"]
        target.parent.mkdir(parents=True, exist_ok=True)
        last_error: BaseException | None = None
        for attempt in range(1, PUBLIC_INPUT_FETCH_ATTEMPTS + 1):
            temporary = target.with_name(f".{target.name}.download-{os.getpid()}-{attempt}.part")
            try:
                if time.monotonic() >= deadline:
                    raise RuntimeError("public input fetch total timeout")
                digest = hashlib.sha256()
                count = 0
                request = Request(row["url"], headers={"Accept-Encoding": "identity"})
                timeout = min(
                    PUBLIC_INPUT_FETCH_TIMEOUT_SECONDS, max(0.1, deadline - time.monotonic())
                )
                with urlopen(request, timeout=timeout) as reply, temporary.open("xb") as handle:
                    while True:
                        if time.monotonic() >= deadline:
                            raise RuntimeError("public input fetch total timeout")
                        chunk = reply.read(PUBLIC_INPUT_READ_BYTES)
                        if not chunk:
                            break
                        count += len(chunk)
                        if count > row["size_bytes"]:
                            raise RuntimeError("public input exceeded pinned size")
                        digest.update(chunk)
                        handle.write(chunk)
                    handle.flush()
                    os.fsync(handle.fileno())
                if count != row["size_bytes"] or digest.hexdigest() != row["sha256"]:
                    raise RuntimeError("public input size or SHA-256 mismatch")
                os.replace(temporary, target)
                verified.append({"dest": row["dest"], "sha256": row["sha256"], "size_bytes": count})
                break
            except (OSError, URLError, RuntimeError) as exc:
                last_error = exc
                temporary.unlink(missing_ok=True)
                # A byte-identity mismatch is deterministic; retrying cannot
                # make that URL safe and only delays cleanup.
                if "mismatch" in str(exc) or "exceeded pinned size" in str(exc):
                    break
        else:
            raise RuntimeError(f"public input fetch failed: {last_error}") from last_error
        if len(verified) == 0 or verified[-1]["dest"] != row["dest"]:
            raise RuntimeError(f"public input fetch failed: {last_error}") from last_error
    return verified


def _fetch_inputs_then_exec(spec_path: Path) -> int:
    """Run in the owned child group; never start the trainer before fetching."""
    # The parent supervisor created output before spawning this owned fetch
    # child. All other specification checks are deliberately repeated here.
    spec = _validate_spec(_load_json(spec_path), allow_existing_output=True)
    root = Path(spec["root"])
    try:
        verified = _fetch_public_inputs(root, spec.get("public_inputs", []))
        _atomic_json(root / "public-inputs-verified.json", {"inputs": verified})
    except BaseException as exc:
        _atomic_json(root / "public-input-fetch-failed.json", {"error": repr(exc)})
        return 74
    environment = os.environ.copy()
    environment.update(spec["env"])
    environment["COLAB_CLI_ROOT"] = str(root)
    environment.setdefault("COLAB_INPUT_DIR", str(root / "inputs"))
    environment["COLAB_OUTPUT_DIR"] = str(root / "output")
    environment["QUEUE_OUTPUT_DIR"] = str(root / "output")
    os.execvpe(spec["command"][0], spec["command"], environment)
    raise AssertionError("os.execvpe unexpectedly returned")


def _runtime_child_command(spec: Mapping[str, Any], spec_path: Path) -> list[str]:
    """Make URL retrieval a killable member of the supervisor's process group."""
    if spec.get("public_inputs"):
        return [sys.executable, str(Path(__file__).resolve()), "--fetch-inputs", str(spec_path)]
    return list(spec["command"])


def _signature(path: Path) -> tuple[int, int, int, int, int]:
    stat = path.stat()
    return (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)


def _exports(
    root: Path,
    max_export_bytes: int,
    cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] | None = None,
) -> list[dict[str, Any]]:
    output = root / "output"
    if not output.is_dir():
        if cache:
            raise RuntimeError("immutable completed export directory was removed")
        return []
    records: list[dict[str, Any]] = []
    for marker in sorted(output.rglob("*.complete")):
        candidate = marker.with_name(marker.name[: -len(".complete")])
        checksum = candidate.with_name(candidate.name + ".sha256")
        if (
            not candidate.is_file()
            or not checksum.is_file()
            or marker.is_symlink()
            or candidate.is_symlink()
            or checksum.is_symlink()
        ):
            raise RuntimeError(f"invalid completed export marker: {marker}")
        resolved = candidate.resolve()
        if (
            output.resolve() not in resolved.parents
            or output.resolve() not in checksum.resolve().parents
            or candidate.stat().st_size > max_export_bytes
            or checksum.stat().st_size > 1024
            or marker.stat().st_size > 4096
        ):
            raise RuntimeError(f"unsafe or oversized completed export: {candidate}")
        relative = str(candidate.relative_to(output))
        signature = (_signature(candidate), _signature(checksum), _signature(marker))
        if cache is not None and relative in cache:
            old_signature, record = cache[relative]
            if old_signature != signature:
                raise RuntimeError(f"immutable completed export changed: {candidate}")
            records.append(record)
            continue
        with checksum.open(encoding="utf-8") as handle:
            words = handle.read(1025).split()
        declared = words[0].lower() if words else ""
        observed = _sha256(candidate)
        if len(declared) != 64 or declared != observed:
            raise RuntimeError(f"completed export hash mismatch: {candidate}")
        if signature != (_signature(candidate), _signature(checksum), _signature(marker)):
            raise RuntimeError(f"completed export changed during hash: {candidate}")
        record = {"path": relative, "sha256": observed, "size": candidate.stat().st_size}
        if cache is not None:
            cache[relative] = (signature, record)
        records.append(record)
    if cache is not None and set(cache) != {entry["path"] for entry in records}:
        raise RuntimeError("immutable completed export was removed")
    return records


def read_state(
    root: str | Path, generation: str | int, renew: bool = True, *, disable_contents: bool = False
) -> dict[str, Any]:
    """Read a fenced state and optionally renew its lease.

    A wrong generation never refreshes the lease.  Callers receive the state,
    terminal receipt when present, and only completed hash-verified exports.
    """
    root_path = Path(root).resolve()
    state = _load_json(root_path / "state.json")
    if str(state.get("generation")) != str(generation):
        raise RuntimeError("generation fence mismatch; lease was not renewed")
    if renew:
        lease = _load_json(root_path / "lease.json")
        if str(lease.get("generation")) != str(generation):
            raise RuntimeError("lease generation fence mismatch; lease was not renewed")
        renewed = {"generation": generation, "renewed_at": _now()}
        if disable_contents or lease.get("contents_disabled") is True:
            renewed["contents_disabled"] = True
        _atomic_json(root_path / "lease.json", renewed)
    terminal_path = root_path / "terminal.json"
    terminal = _load_json(terminal_path) if terminal_path.exists() else None
    return {
        "state": state,
        "terminal": terminal,
        # Inventory is computed by the supervisor, not a short polling kernel.
        "exports": terminal.get("exports", []) if terminal else state.get("exports", []),
        "job_log_tail": _tail(root_path / "job.log"),
        "supervisor_log_tail": _tail(root_path / "supervisor.log"),
    }


def _tail(path: Path, limit: int = 64 * 1024) -> str:
    """Return bounded diagnostic text without making polling RPCs large."""
    if not path.is_file():
        return ""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        handle.seek(max(0, handle.tell() - limit))
        return handle.read(limit).decode("utf-8", errors="replace")


def _unassign(root: Path, endpoint: str) -> None:
    """Release through the tunnel endpoint, independent of a notebook frontend."""
    _release_module().release_from_file(
        root, endpoint, min_valid_seconds=RELEASE_RETRY_WINDOW_SECONDS
    )


def _release_credential_expiring(spec: Mapping[str, Any]) -> bool:
    """Fail closed before a missing Mac rotation leaves no usable release token."""
    required = (
        float(spec["lease_seconds"])
        + float(spec["checkpoint_grace_seconds"])
        + STOP_CHILD_TAIL_SECONDS
        + RELEASE_RETRY_WINDOW_SECONDS
    )
    return _release_module().credential_renewal_due(
        _release_module().load_release_credential(
            spec["root"], spec["release_endpoint"], min_valid_seconds=0
        ),
        spec["release_endpoint"],
        safety_window_seconds=required,
    )


def _require_release_safety(spec: Mapping[str, Any]) -> None:
    """Refuse to start a paid child without time for lease-loss release."""
    required = (
        float(spec["lease_seconds"])
        + float(spec["checkpoint_grace_seconds"])
        + STOP_CHILD_TAIL_SECONDS
        + RELEASE_RETRY_WINDOW_SECONDS
    )
    _release_module().load_release_credential(
        spec["root"], spec["release_endpoint"], min_valid_seconds=required
    )


_CONTENTS_LEASES: dict[tuple[str, str], float] = {}


def _lease_expired(root: Path, generation: str | int, lease_seconds: float) -> str | None:
    lease = _load_json(root / "lease.json")
    if str(lease.get("generation")) != str(generation):
        return "lease_generation_changed"
    renewed_at = float(lease.get("renewed_at", 0))
    # Contents chunk=1 writes may expose an incomplete file. Keep them separate
    # from the VM-owned atomic lease and retain the last validated heartbeat
    # in this watchdog process. A partial/failed PUT cannot erase a good lease
    # or extend it. The client sends no timestamp metadata: stat uses VM time.
    if lease.get("contents_disabled") is not True:
        key = (str(root), str(generation))
        heartbeat = root / "lease-heartbeat.json"
        try:
            before = heartbeat.stat()
            control = _load_json(heartbeat)
            after = heartbeat.stat()
            if (before.st_mtime_ns, before.st_size) != (after.st_mtime_ns, after.st_size):
                raise ValueError("heartbeat changed during read")
            if str(control.get("generation")) != str(generation):
                return "lease_generation_changed"
            if control.get("renewal_clock") == "contents-mtime":
                stamp = before.st_mtime
                if stamp > _now() + 5:
                    return "lease_clock_invalid"
                _CONTENTS_LEASES[key] = max(_CONTENTS_LEASES.get(key, 0), stamp)
        except (OSError, ValueError, TypeError, RuntimeError):
            pass  # Normal clock-based expiry still applies to the last good lease.
        renewed_at = max(renewed_at, _CONTENTS_LEASES.get(key, 0))
    if renewed_at > _now() + 5:
        return "lease_clock_invalid"
    if _now() - renewed_at > lease_seconds:
        return "lease_expired"
    return None


def _unassign_or_raise(
    state_path: Path,
    state: Mapping[str, Any],
    spec: Mapping[str, Any],
    cleanup: dict[str, Any] | None = None,
) -> None:
    try:
        if cleanup is not None:
            cleanup["unassign_attempted"] = True
        _unassign(Path(spec["root"]), str(spec["release_endpoint"]))
    except Exception as exc:  # unassignment is an explicit evidence failure
        message = f"{type(exc).__name__}: {exc}"
        _best_effort_state(state_path, {**state, "unassign_error": message})
        raise RuntimeError(f"runtime unassign failed: {message}") from exc


def _best_effort_state(path: Path, state: Mapping[str, Any]) -> None:
    """Evidence storage failure must never prevent owned-process cleanup."""
    try:
        _atomic_json(path, state)
    except OSError:
        pass


def _stop_child(process: subprocess.Popen[bytes], grace_seconds: float) -> None:
    """Ask the owned process group to checkpoint, then bound its lifetime."""
    try:
        os.killpg(process.pid, signal.SIGUSR1)
    except (ProcessLookupError, PermissionError):
        return
    deadline = _now() + grace_seconds
    # The shell leader can exit after forwarding SIGUSR1 while its trainer is
    # checkpointing; the owned process group still receives the full grace.
    while _now() < deadline:
        time.sleep(0.2)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError):
        return
    if process.poll() is None:
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


def supervise(spec: Mapping[str, Any]) -> int:
    cleanup: dict[str, Any] = {"process": None, "unassign_attempted": False}
    try:
        return _supervise(spec, cleanup)
    except BaseException:
        process = cleanup["process"]
        if process is not None:
            try:
                _stop_child(process, float(spec.get("checkpoint_grace_seconds", 1)))
            except BaseException:
                pass
        if not cleanup["unassign_attempted"]:
            cleanup["unassign_attempted"] = True
            try:
                _unassign(Path(spec["root"]), str(spec["release_endpoint"]))
            except BaseException:
                pass
        raise


def _supervise(spec: Mapping[str, Any], cleanup: dict[str, Any]) -> int:
    spec = _validate_spec(spec)
    _require_release_safety(spec)
    root = Path(spec["root"])
    output = root / "output"
    output.mkdir(mode=0o700)
    started = _now()
    _atomic_json(root / "lease.json", {"generation": spec["generation"], "renewed_at": started})
    # This state write happens before Popen: the lease watchdog is armed before
    # any dependency bootstrap that may be represented by the command.
    state: dict[str, Any] = {
        "job_id": spec["job_id"],
        "generation": spec["generation"],
        "status": "starting",
        "started_at": started,
        "updated_at": started,
        "max_export_bytes": spec["max_export_bytes"],
    }
    try:
        _atomic_json(root / "state.json", state)
    except BaseException:
        _unassign_or_raise(root / "state.json", state, spec, cleanup)
        raise
    environment = os.environ.copy()
    environment.update(spec["env"])
    environment["COLAB_CLI_ROOT"] = str(root)
    environment.setdefault("COLAB_INPUT_DIR", str(root / "inputs"))
    environment["COLAB_OUTPUT_DIR"] = str(output)
    environment["QUEUE_OUTPUT_DIR"] = str(output)
    log = (root / "job.log").open("ab", buffering=0)
    try:
        process = subprocess.Popen(
            _runtime_child_command(spec, root / "spec.json"),
            cwd=root,
            env=environment,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        cleanup["process"] = process
    except BaseException as exc:
        log.close()
        terminal = {
            "job_id": spec["job_id"],
            "generation": spec["generation"],
            "status": "failed",
            "returncode": None,
            "reason": "popen_failed",
            "error": repr(exc),
            "finished_at": _now(),
            "exports": [],
        }
        try:
            _create_json(root / "terminal.json", terminal)
        except OSError:
            pass
        state.update(
            status="failed", updated_at=_now(), reason="popen_failed", error=repr(exc), exports=[]
        )
        _best_effort_state(root / "state.json", state)
        _unassign_or_raise(root / "state.json", state, spec, cleanup)
        raise
    reason: str | None = None
    last_state = 0.0
    failure: BaseException | None = None
    export_cache: dict[str, tuple[tuple[Any, ...], dict[str, Any]]] = {}
    try:
        while process.poll() is None:
            now = _now()
            reason = _lease_expired(root, spec["generation"], float(spec["lease_seconds"]))
            if (
                reason is None
                and spec.get("public_inputs")
                and not (root / "public-inputs-verified.json").is_file()
                and now - started > PUBLIC_INPUT_FETCH_TOTAL_TIMEOUT_SECONDS
            ):
                reason = "public_input_fetch_timeout"
            if reason is None and _release_credential_expiring(spec):
                reason = "release_credential_expiring"
            if reason is None and now - started > float(spec["timeout_seconds"]):
                reason = "timeout"
            if reason:
                _stop_child(process, float(spec["checkpoint_grace_seconds"]))
                break
            if now - last_state >= STATE_INTERVAL_SECONDS:
                state.update(
                    status="running",
                    updated_at=now,
                    pid=process.pid,
                    exports=_exports(root, spec["max_export_bytes"], export_cache),
                )
                _atomic_json(root / "state.json", state)
                last_state = now
            time.sleep(0.2)
    except BaseException as exc:
        failure = exc
        reason = "runtime_error"
        state.update(status="failed", updated_at=_now(), reason=reason, error=repr(exc))
        _best_effort_state(root / "state.json", state)
        _stop_child(process, float(spec["checkpoint_grace_seconds"]))
    finally:
        returncode = process.wait()
        log.close()
    fetch_failure = root / "public-input-fetch-failed.json"
    if fetch_failure.is_file():
        reason = "public_input_fetch_failed"
    try:
        completed_exports = _exports(root, spec["max_export_bytes"], export_cache)
    except BaseException as exc:
        completed_exports = []
        failure = failure or exc
        reason = "runtime_error"
    final_status = (
        "succeeded" if returncode == 0 and reason is None and failure is None else "failed"
    )
    terminal = {
        "job_id": spec["job_id"],
        "generation": spec["generation"],
        "status": final_status,
        "returncode": returncode,
        "reason": reason,
        "started_at": started,
        "finished_at": _now(),
        "exports": completed_exports,
    }
    try:
        _create_json(root / "terminal.json", terminal)
    except OSError as exc:
        failure = failure or exc
        reason = "runtime_error"
    state.update(
        status=final_status,
        updated_at=_now(),
        returncode=returncode,
        reason=reason,
        exports=terminal["exports"],
    )
    _best_effort_state(root / "state.json", state)
    # A normal completion deliberately keeps the assignment for the Mac's final
    # verified download (and, if explicitly approved, a chained useful job).  It
    # remains lease-fenced: client loss after completion still unassigns it.
    if reason is not None:
        _unassign_or_raise(root / "state.json", state, spec, cleanup)
        if failure is not None:
            raise RuntimeError(f"runtime supervisor failure: {failure!r}") from failure
        return returncode
    post_terminal_timeout = float(
        spec.get("post_terminal_timeout_seconds", spec["timeout_seconds"])
    )
    while True:
        try:
            reason = _lease_expired(root, spec["generation"], float(spec["lease_seconds"]))
            if reason is None and _release_credential_expiring(spec):
                reason = "release_credential_expiring_after_terminal"
        except BaseException as exc:
            reason = "runtime_error_after_terminal"
            state["error"] = repr(exc)
        if reason is None and _now() - started > post_terminal_timeout:
            reason = "timeout_after_terminal"
        if reason:
            state.update(status="lease_expired_after_terminal", updated_at=_now(), reason=reason)
            _atomic_json(root / "state.json", state)
            _unassign_or_raise(root / "state.json", state, spec, cleanup)
            return returncode
        time.sleep(0.2)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--spec", type=Path)
    group.add_argument("--fetch-inputs", type=Path)
    args = parser.parse_args(argv)
    try:
        if args.fetch_inputs is not None:
            return _fetch_inputs_then_exec(args.fetch_inputs)
        with args.spec.open(encoding="utf-8") as handle:
            return supervise(json.load(handle))
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"colab-cli runtime: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
