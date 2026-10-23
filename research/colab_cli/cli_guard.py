"""Safety interposition for the pinned ``colab-cli`` 0.7.4 client.

This module is deliberately stdlib-only until a helper process imports the
installed CLI.  ``CLI.call`` places a tiny ``sitecustomize.py`` ahead of the
installed package, so :func:`install_hooks` runs before ``colab_cli`` creates a
client.  The hook gives each allocation a supervisor-owned notebook hash and
writes the non-secret assignment identity durably before the CLI can update its
own local session store.

The helper entry points are intentionally narrow.  They load an already saved
OAuth credential but never start an OAuth flow, and they emit only endpoint or
kernel state -- never proxy tokens or refresh credentials.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

# Keep these literals here: this module is executed directly by the pinned CLI
# hook and must not depend on package imports being available in that process.
# They mirror the Mac-side read policy.  Runs 3--5 showed tunnel/kernel and
# OAuth latency can exceed the former 10/45/75-second limits.
CONNECT_TIMEOUT_SECONDS = 30
READ_TIMEOUT_SECONDS = 120
HELPER_TIMEOUT_SECONDS = 150
READ_HELPER_ATTEMPTS = 3
READ_HELPER_BACKOFF_SECONDS = (2.0, 5.0)


class GuardError(RuntimeError):
    """A bounded helper could not establish the requested safe fact."""


def _timeout_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Supply the real requests timeout the upstream CLI otherwise omits."""
    patched = dict(kwargs)
    # ``requests`` treats ``timeout=None`` as unbounded, so setdefault alone
    # would leave a caller able to bypass the supervisor's transport limit.
    if patched.get("timeout") is None:
        patched["timeout"] = (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS)
    return patched


def _write_record(path: Path, notebook_hash: str, endpoint: str) -> None:
    """Atomically persist only non-secret allocation identity with fsync."""
    record = {"notebook_hash": notebook_hash, "endpoint": endpoint}
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, temporary_name = tempfile.mkstemp(prefix=".assignment-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(fd, 0o600)
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def install_hooks() -> None:
    """Patch the installed CLI once when supervisor allocation env is present."""
    notebook_hash = os.environ.get("SNAKE_COLAB_NOTEBOOK_HASH")
    record_name = os.environ.get("SNAKE_COLAB_ASSIGNMENT_RECORD")
    if not notebook_hash or not record_name:
        return
    try:
        desired_hash = uuid.UUID(notebook_hash)
    except ValueError as exc:
        raise GuardError("SNAKE_COLAB_NOTEBOOK_HASH must be a UUID") from exc

    # Imported only in the process that is already executing the trusted,
    # pinned CLI.  A detached runtime watchdog never imports this module.
    from colab_cli.client import Client

    if getattr(Client, "_snake_guard_installed", False):
        return
    import requests

    original_assign = Client.assign
    original_issue_request = Client._issue_request
    original_session_request = requests.sessions.Session.request
    record_path = Path(record_name)
    kernel_creation_attempted = False

    def guarded_issue_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_issue_request(self, *args, **_timeout_kwargs(kwargs))

    def guarded_session_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        nonlocal kernel_creation_attempted
        method = args[0] if args else kwargs.get("method")
        url = args[1] if len(args) > 1 else kwargs.get("url")
        # Persist creation identity before websocket startup (the upstream
        # callback runs too late). Never infer ownership from a kernel list.
        capture = None
        if str(method).upper() == "POST" and str(url).endswith("/api/kernels"):
            capture = _kernel_creation_target(url, kwargs, record_path)
            if kernel_creation_attempted:
                raise GuardError("kernel creation already attempted; refusing uncertain retry")
            kernel_creation_attempted = True
        response = original_session_request(self, *args, **_timeout_kwargs(kwargs))
        if capture is not None:
            store, state = capture
            response.raise_for_status()
            body = response.json()
            kernel_id = body.get("id") if isinstance(body, dict) else None
            if not isinstance(kernel_id, str) or not kernel_id.strip():
                raise GuardError("kernel creation returned malformed kernel identity")
            state.kernel_id = kernel_id
            store.add(state)
        return response

    def guarded_assign(
        self: Any,
        notebook_hash: uuid.UUID,
        *args: Any,
        **kwargs: Any,
    ) -> Any:
        result = original_assign(self, desired_hash, *args, **kwargs)
        endpoint = getattr(result, "endpoint", None)
        if not isinstance(endpoint, str) or not endpoint:
            raise GuardError("Colab assignment response omitted endpoint")
        _write_record(record_path, str(desired_hash), endpoint)
        return result

    Client._issue_request = guarded_issue_request
    Client.assign = guarded_assign
    # ContentsClient bypasses Client._issue_request and calls requests.request;
    # the latter creates a Session, so patch the class-level request path too.
    requests.sessions.Session.request = guarded_session_request
    Client._snake_guard_installed = True


def _kernel_creation_target(url: str, kwargs: dict[str, Any], record_path: Path) -> Any:
    """Bind a creation response to this process's exact tracked assignment.

    A second POST after creation is forbidden even if websocket setup failed.
    The next supervisor probe must independently prove the saved kernel idle.
    """
    from colab_cli.state import StateStore

    config = os.environ.get("SNAKE_COLAB_CONFIG")
    session = os.environ.get("SNAKE_COLAB_SESSION")
    if not config or not session:
        raise GuardError("kernel creation lacks supervisor session identity")
    store = StateStore(config)
    state = store.get(session)
    record = json.loads(record_path.read_text())
    headers = kwargs.get("headers", {})
    if (
        state is None
        or not state.url
        or not state.token
        or url != state.url.rstrip("/") + "/api/kernels"
        or not state.endpoint
        or state.endpoint != record.get("endpoint")
        or record.get("notebook_hash") != os.environ.get("SNAKE_COLAB_NOTEBOOK_HASH")
        or headers.get("X-Colab-Runtime-Proxy-Token") != state.token
    ):
        raise GuardError("kernel creation does not match tracked runtime identity")
    if state.kernel_id:
        raise GuardError("tracked kernel reconnect required; kernel already exists")
    return store, state


def _run_helper(
    oauth_python: str, *arguments: str, timeout: float = HELPER_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Run a noninteractive helper with a hard outer deadline and JSON result."""
    if not oauth_python:
        raise GuardError("oauth_python is required")
    try:
        completed = subprocess.run(
            [oauth_python, str(Path(__file__).resolve()), *arguments],
            capture_output=True,
            text=True,
            check=False,
            timeout=min(timeout, HELPER_TIMEOUT_SECONDS),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise GuardError(f"guard helper failed: {type(exc).__name__}") from exc
    if completed.returncode:
        raise GuardError(
            f"guard helper refused ({completed.returncode}): {completed.stderr[-500:]}"
        )
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise GuardError("guard helper returned invalid JSON") from exc
    if not isinstance(value, dict):
        raise GuardError("guard helper result must be an object")
    return value


def _transient_helper_error(exc: BaseException) -> bool:
    """Recognize a transport failure without turning auth/state refusals into retries."""
    if isinstance(exc, (OSError, subprocess.TimeoutExpired, TimeoutError)):
        return True
    text = str(exc).lower()
    return any(
        marker in text
        for marker in (
            "500",
            "502",
            "503",
            "504",
            "connection reset",
            "connection aborted",
            "connection refused",
            "connection error",
            "timed out",
            "timeout",
            "temporarily unavailable",
        )
    )


def _run_read_helper(
    oauth_python: str, *arguments: str, timeout: float = HELPER_TIMEOUT_SECONDS
) -> dict[str, Any]:
    """Retry only idempotent helper reads inside one caller-owned deadline.

    Recovery and kernel state checks issue GET/read RPCs only.  An unassign is
    deliberately excluded: its caller must first establish endpoint state
    before choosing another release attempt.
    """
    deadline = time.monotonic() + min(timeout, HELPER_TIMEOUT_SECONDS)
    last_error: GuardError | None = None
    for attempt in range(READ_HELPER_ATTEMPTS):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            return _run_helper(oauth_python, *arguments, timeout=remaining)
        except GuardError as exc:
            last_error = exc
            if attempt + 1 == READ_HELPER_ATTEMPTS or not _transient_helper_error(exc):
                raise
            delay = min(READ_HELPER_BACKOFF_SECONDS[attempt], max(0.0, deadline - time.monotonic()))
            if delay:
                time.sleep(delay)
    raise GuardError("read helper operation budget exhausted") from last_error


def recover_assignment(
    oauth_python: str, notebook_hash: str, *, timeout: float = HELPER_TIMEOUT_SECONDS
) -> str | None:
    """GET the backend's identity for a pre-recorded notebook hash, never POST."""
    try:
        uuid.UUID(notebook_hash)
    except ValueError as exc:
        raise GuardError("notebook_hash must be a UUID") from exc
    result = _run_read_helper(oauth_python, "--recover-assignment", notebook_hash, timeout=timeout)
    endpoint = result.get("endpoint")
    if endpoint is None:
        return None
    if not isinstance(endpoint, str) or not endpoint:
        raise GuardError("recovery helper returned malformed endpoint")
    return endpoint


def unassign_endpoint(
    oauth_python: str, endpoint: str, *, timeout: float = HELPER_TIMEOUT_SECONDS
) -> None:
    """Release one already proven endpoint through the installed tunnel client."""
    if not endpoint:
        raise GuardError("endpoint is required")
    result = _run_helper(oauth_python, "--unassign-endpoint", endpoint, timeout=timeout)
    if result != {"unassigned": True}:
        raise GuardError("unassign helper did not confirm release")


def kernel_idle(
    oauth_python: str,
    config: Path | str,
    session: str,
    *,
    timeout: float = HELPER_TIMEOUT_SECONDS,
) -> bool:
    """Return ``True`` only for an explicitly idle tracked kernel.

    A busy or not-yet-visible kernel returns ``False``. A missing cold-start
    kernel ID can be recovered from a unique idle kernel on the tracked runtime.
    Missing runtime metadata, transport errors, and malformed state raise
    :class:`GuardError`.
    """
    result = _run_read_helper(oauth_python, "--kernel-idle", str(config), session, timeout=timeout)
    if set(result) != {"idle"} or not isinstance(result["idle"], bool):
        raise GuardError("kernel helper returned unknown state")
    return result["idle"]


def _noninteractive_client() -> Any:
    """Build an installed CLI client from an existing refreshable token only."""
    import requests
    from colab_cli.auth import PUBLIC_SCOPES, TOKEN_CONFIG_PATH
    from colab_cli.client import Client, Prod
    from google.auth.transport.requests import AuthorizedSession, Request
    from google.oauth2.credentials import Credentials

    token_path = Path(TOKEN_CONFIG_PATH)
    if not token_path.is_file():
        raise GuardError("saved OAuth token is required; interactive auth is forbidden")
    credentials = Credentials.from_authorized_user_file(str(token_path), PUBLIC_SCOPES)
    # This short-lived helper owns its interpreter.  Patch the base Session so
    # both explicit backend calls and google-auth's internal refresh Session
    # receive an actual connect/read deadline.
    original_session_request = requests.sessions.Session.request

    def bounded_session_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_session_request(self, *args, **_timeout_kwargs(kwargs))

    requests.sessions.Session.request = bounded_session_request
    if not credentials.valid:
        if not credentials.expired or not credentials.refresh_token:
            raise GuardError("saved OAuth token is not refreshable")
        # ``Request`` takes a Session rather than a timeout.  The base Session
        # patch above also binds google-auth's own internal refresh request.
        credentials.refresh(Request())
    session = AuthorizedSession(credentials)
    original_request = session.request

    def bounded_request(*args: Any, **kwargs: Any) -> Any:
        return original_request(*args, **_timeout_kwargs(kwargs))

    session.request = bounded_request
    return Client(Prod(), session)


def runtime_ready(oauth_python: str, config: Path | str, session: str, *, timeout: float) -> bool:
    """Read Contents readiness without creating a kernel or submitting work."""
    # One attempt only: the supervisor owns the shared warmup deadline/backoff.
    result = _run_helper(oauth_python, "--runtime-ready", str(config), session, timeout=timeout)
    if set(result) != {"ready"} or not isinstance(result["ready"], bool):
        raise GuardError("runtime readiness helper returned unknown state")
    return result["ready"]


def _runtime_ready_once(config: str, session: str) -> dict[str, bool]:
    """Probe the same Contents route as PUTs without requiring an existing kernel."""
    import requests
    from colab_cli.contents import ContentsClient
    from colab_cli.state import StateStore

    state = StateStore(config).get(session)
    if state is None or not state.url or not state.token:
        raise GuardError("tracked session lacks runtime probe metadata")
    try:
        # Metadata only; never list/download runtime files or upload a dummy.
        # Bound requests here as well as by the helper subprocess timeout.
        original_request = requests.sessions.Session.request

        def bounded_request(self: Any, *args: Any, **kwargs: Any) -> Any:
            return original_request(self, *args, **_timeout_kwargs(kwargs))

        requests.sessions.Session.request = bounded_request
        try:
            contents = ContentsClient(state)._request("GET", "/content", params={"content": "0"})
        finally:
            requests.sessions.Session.request = original_request
        if not isinstance(contents, dict) or contents.get("type") != "directory":
            raise GuardError("runtime Contents probe returned unknown state")
        return {"ready": True}
    except (FileNotFoundError, requests.HTTPError) as exc:
        # A fresh server may not yet expose the Contents route. Auth,
        # quota and other permanent failures retain their normal refusal path.
        if isinstance(exc, FileNotFoundError) or (
            exc.response is not None and exc.response.status_code == 404
        ):
            return {"ready": False}
        raise


def _recover_assignment_once(notebook_hash: str) -> dict[str, Any]:
    client = _noninteractive_client()
    result = client._get_assignment(uuid.UUID(notebook_hash))
    endpoint = getattr(result, "endpoint", None)
    return {"endpoint": endpoint} if isinstance(endpoint, str) and endpoint else {"endpoint": None}


def _unassign_endpoint_once(endpoint: str) -> dict[str, bool]:
    _noninteractive_client().unassign(endpoint)
    return {"unassigned": True}


def _kernel_idle_once(config: str, session: str) -> dict[str, bool]:
    """Prove idle, recovering a cold-start ID only from the tracked runtime.

    CLI 0.7.4 saves a new kernel ID after websocket startup. An outer exec
    timeout can kill that client before its callback saves the ID. Contents
    readiness alone cannot clear this ambiguity: discover the sole kernel by
    GET, require explicit idle, and save its ID before any subsequent exec so
    the CLI reconnects instead of creating another kernel.
    """
    import requests
    from colab_cli.state import StateStore

    store = StateStore(config)
    state = store.get(session)
    if state is None or not state.url or not state.token:
        raise GuardError("tracked session lacks kernel probe metadata")
    recovering = not state.kernel_id
    url = state.url.rstrip("/") + "/api/kernels"
    if not recovering:
        url += "/" + quote(state.kernel_id, safe="")
    response = requests.get(
        url,
        headers={
            "X-Colab-Client-Agent": "colab-cli",
            "X-Colab-Runtime-Proxy-Token": state.token,
        },
        params={"authuser": "0", "colab-runtime-proxy-token": state.token},
        timeout=(CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS),
    )
    response.raise_for_status()
    body = response.json()
    if recovering:
        if not isinstance(body, list):
            raise GuardError("kernel discovery returned no recognized kernel inventory")
        if not body:
            # A timed-out creation may still be in flight. An empty list does
            # not prove that submitting another exec is safe.
            return {"idle": False}
        if len(body) != 1:
            raise GuardError("kernel discovery is ambiguous; expected one tracked runtime kernel")
        body = body[0]
        if (
            not isinstance(body, dict)
            or not isinstance(body.get("id"), str)
            or not body["id"].strip()
        ):
            raise GuardError("kernel discovery returned malformed kernel identity")
    execution_state = body.get("execution_state") if isinstance(body, dict) else None
    if execution_state == "idle":
        if recovering:
            state.kernel_id = body["id"]
            store.add(state)
        return {"idle": True}
    if execution_state in {"busy", "starting"}:
        return {"idle": False}
    raise GuardError("kernel probe returned no recognized execution_state")


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_mutually_exclusive_group(required=True)
    commands.add_argument("--recover-assignment", metavar="NOTEBOOK_HASH")
    commands.add_argument("--unassign-endpoint", metavar="ENDPOINT")
    commands.add_argument("--kernel-idle", nargs=2, metavar=("CONFIG", "SESSION"))
    commands.add_argument("--runtime-ready", nargs=2, metavar=("CONFIG", "SESSION"))
    args = parser.parse_args(argv)
    try:
        if args.recover_assignment:
            result: dict[str, Any] = _recover_assignment_once(args.recover_assignment)
        elif args.unassign_endpoint:
            result = _unassign_endpoint_once(args.unassign_endpoint)
        elif args.runtime_ready:
            result = _runtime_ready_once(*args.runtime_ready)
        else:
            result = _kernel_idle_once(*args.kernel_idle)
    except Exception as exc:
        print(f"cli_guard: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ != "__main__":
    install_hooks()

if __name__ == "__main__":
    raise SystemExit(_main())
