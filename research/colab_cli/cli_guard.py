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
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

CONNECT_TIMEOUT_SECONDS = 10
READ_TIMEOUT_SECONDS = 45
HELPER_TIMEOUT_SECONDS = 75


class GuardError(RuntimeError):
    """A bounded helper could not establish the requested safe fact."""


def _timeout_kwargs(kwargs: dict[str, Any]) -> dict[str, Any]:
    """Supply the real requests timeout the upstream CLI otherwise omits."""
    patched = dict(kwargs)
    patched.setdefault("timeout", (CONNECT_TIMEOUT_SECONDS, READ_TIMEOUT_SECONDS))
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

    def guarded_issue_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_issue_request(self, *args, **_timeout_kwargs(kwargs))

    def guarded_session_request(self: Any, *args: Any, **kwargs: Any) -> Any:
        return original_session_request(self, *args, **_timeout_kwargs(kwargs))

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


def recover_assignment(
    oauth_python: str, notebook_hash: str, *, timeout: float = HELPER_TIMEOUT_SECONDS
) -> str | None:
    """GET the backend's identity for a pre-recorded notebook hash, never POST."""
    try:
        uuid.UUID(notebook_hash)
    except ValueError as exc:
        raise GuardError("notebook_hash must be a UUID") from exc
    result = _run_helper(oauth_python, "--recover-assignment", notebook_hash, timeout=timeout)
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

    A busy kernel returns ``False``.  Missing metadata, transport errors, and
    malformed state are unknown facts and therefore raise :class:`GuardError`.
    """
    result = _run_helper(oauth_python, "--kernel-idle", str(config), session, timeout=timeout)
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


def _recover_assignment_once(notebook_hash: str) -> dict[str, Any]:
    client = _noninteractive_client()
    result = client._get_assignment(uuid.UUID(notebook_hash))
    endpoint = getattr(result, "endpoint", None)
    return {"endpoint": endpoint} if isinstance(endpoint, str) and endpoint else {"endpoint": None}


def _unassign_endpoint_once(endpoint: str) -> dict[str, bool]:
    _noninteractive_client().unassign(endpoint)
    return {"unassigned": True}


def _kernel_idle_once(config: str, session: str) -> dict[str, bool]:
    import requests
    from colab_cli.state import StateStore

    state = StateStore(config).get(session)
    if state is None or not state.kernel_id or not state.url or not state.token:
        raise GuardError("tracked session lacks kernel probe metadata")
    url = state.url.rstrip("/") + "/api/kernels/" + quote(state.kernel_id, safe="")
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
    execution_state = body.get("execution_state") if isinstance(body, dict) else None
    if execution_state == "idle":
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
    args = parser.parse_args(argv)
    try:
        if args.recover_assignment:
            result: dict[str, Any] = _recover_assignment_once(args.recover_assignment)
        elif args.unassign_endpoint:
            result = _unassign_endpoint_once(args.unassign_endpoint)
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
