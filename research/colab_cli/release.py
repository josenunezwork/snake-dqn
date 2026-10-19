"""Narrow, short-lived credentials for a runtime-side Colab release.

The Colab CLI releases an assignment with an authenticated GET/POST exchange
against ``/tun/m/unassign/<endpoint>``.  A watchdog running inside the VM has
no notebook frontend, so it cannot call ``google.colab.runtime.unassign()``.
This module deliberately sends only a short-lived OAuth *access* token to the
VM.  The account refresh token stays on the Mac.

The token is requested with the two scopes the Colab tunnel requires for this
operation.  Google may decline scope reduction for a refresh grant; that is a
hard refusal, not a reason to send the CLI's normal broad credential.
"""

from __future__ import annotations

import json
import math
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.parse import quote
from urllib.request import HTTPRedirectHandler, Request, build_opener

COLAB_ORIGIN = "https://colab.research.google.com"
RELEASE_SCOPES = frozenset(
    {
        "https://www.googleapis.com/auth/colaboratory",
        "https://www.googleapis.com/auth/userinfo.email",
    }
)
RELEASE_FILE = "release-credential.json"
REQUEST_TIMEOUT_SECONDS = 20
REQUEST_OVERHEAD_SECONDS = 1
RELEASE_ATTEMPTS = 3
RELEASE_RETRY_SECONDS = 2
RELEASE_MAX_SECONDS = (
    RELEASE_ATTEMPTS * 2 * (REQUEST_TIMEOUT_SECONDS + REQUEST_OVERHEAD_SECONDS)
    + (RELEASE_ATTEMPTS - 1) * RELEASE_RETRY_SECONDS
)
_ENDPOINT_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class ReleaseError(RuntimeError):
    """A release credential cannot safely unassign its intended runtime."""


def _now() -> float:
    return time.time()


def _unassign_url(endpoint: str) -> str:
    if not isinstance(endpoint, str) or not _ENDPOINT_RE.fullmatch(endpoint):
        raise ReleaseError("release endpoint is malformed")
    return f"{COLAB_ORIGIN}/tun/m/unassign/{quote(endpoint, safe='')}"


def _validate_endpoint_if_bound(endpoint: str | None) -> None:
    """Allow pre-allocation scope checks before a backend endpoint exists."""
    if endpoint:
        _unassign_url(endpoint)


def _validate_credential(
    raw: Mapping[str, Any], endpoint: str, *, min_valid_seconds: float = 0
) -> dict[str, Any]:
    """Validate a refresh-token-free credential before use or persistence."""
    if not isinstance(raw, Mapping):
        raise ReleaseError("release credential must be an object")
    forbidden = {"refresh_token", "client_secret", "token_uri"} & set(raw)
    if forbidden:
        raise ReleaseError("release credential contains forbidden long-lived material")
    token = raw.get("access_token")
    expires_at = raw.get("expires_at")
    scopes = raw.get("scopes")
    if not isinstance(token, str) or len(token) < 16:
        raise ReleaseError("release credential has no usable access token")
    if (
        isinstance(expires_at, bool)
        or not isinstance(expires_at, (int, float))
        or not math.isfinite(float(expires_at))
    ):
        raise ReleaseError("release credential has invalid expiry")
    if not isinstance(scopes, list) or not all(isinstance(scope, str) for scope in scopes):
        raise ReleaseError("release credential has invalid granted scopes")
    granted = frozenset(scopes)
    if granted != RELEASE_SCOPES:
        raise ReleaseError(
            "release credential scopes are not exactly the restricted release scopes"
        )
    _validate_endpoint_if_bound(endpoint)
    if float(expires_at) - _now() < min_valid_seconds:
        raise ReleaseError("release credential expires before the required safety window")
    return {
        "access_token": token,
        "expires_at": float(expires_at),
        "scopes": sorted(granted),
    }


def credential_renewal_due(
    credential: Mapping[str, Any], endpoint: str, *, safety_window_seconds: float
) -> bool:
    """Return whether the Mac must rotate the VM release credential now."""
    try:
        _validate_credential(credential, endpoint, min_valid_seconds=safety_window_seconds)
    except ReleaseError:
        return True
    return False


def _atomic_private_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".release-", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, separators=(",", ":"))
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        os.chmod(path, 0o600)
    finally:
        temporary.unlink(missing_ok=True)


def write_release_credential(
    root: str | Path, endpoint: str, credential: Mapping[str, Any], *, min_valid_seconds: float
) -> None:
    """Atomically rotate the runtime's access token without writing a refresh token."""
    checked = _validate_credential(credential, endpoint, min_valid_seconds=min_valid_seconds)
    _atomic_private_json(Path(root) / RELEASE_FILE, checked)


def load_release_credential(
    root: str | Path, endpoint: str, *, min_valid_seconds: float
) -> dict[str, Any]:
    """Load the current runtime release credential and enforce its safety margin."""
    path = Path(root) / RELEASE_FILE
    try:
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:
            raise ReleaseError("release credential permissions are not private")
        with path.open(encoding="utf-8") as handle:
            raw = json.load(handle)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot load release credential: {exc}") from exc
    return _validate_credential(raw, endpoint, min_valid_seconds=min_valid_seconds)


_MINT_PROGRAM = r"""
import json
import sys
import time
from datetime import timezone
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials

scopes = json.loads(sys.argv[1])
credentials = Credentials.from_authorized_user_file(
    __import__("os").path.expanduser("~/.config/colab-cli/token.json"), scopes
)
credentials.refresh(Request())
granted = set(credentials.granted_scopes or ())
if granted != set(scopes):
    raise RuntimeError("provider did not grant exactly the requested release scopes")
if not credentials.token or not credentials.expiry:
    raise RuntimeError("provider returned no bounded access token")
expiry = credentials.expiry
if expiry.tzinfo is None:
    # google-auth commonly returns a naive UTC datetime.  Never let the VM
    # credential lifetime depend on the Mac's local timezone.
    expiry = expiry.replace(tzinfo=timezone.utc)
print(json.dumps({
    "access_token": credentials.token,
    "expires_at": expiry.timestamp(),
    "scopes": sorted(granted),
}, separators=(",", ":")))
"""


def mint_release_credential(
    oauth_python: str | Path, endpoint: str, min_valid_seconds: float
) -> dict[str, Any]:
    """Mint a scope-checked, short-lived release access token on the Mac.

    The subprocess reads the owner's existing token store locally.  Its sole
    stdout value is the access-token JSON returned to the supervisor; neither
    its command line nor the runtime contains a refresh token.
    """
    _validate_endpoint_if_bound(endpoint)
    if min_valid_seconds <= 0:
        raise ReleaseError("release credential minimum lifetime must be positive")
    try:
        completed = subprocess.run(
            [str(oauth_python), "-c", _MINT_PROGRAM, json.dumps(sorted(RELEASE_SCOPES))],
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        raw = json.loads(completed.stdout)
    except (OSError, subprocess.SubprocessError, ValueError, json.JSONDecodeError) as exc:
        raise ReleaseError("could not mint restricted runtime release credential") from exc
    return _validate_credential(raw, endpoint, min_valid_seconds=min_valid_seconds)


class _RejectRedirect(HTTPRedirectHandler):
    """Do not forward an account bearer token beyond the fixed Colab origin."""

    def redirect_request(
        self, req: Request, fp: Any, code: int, msg: str, headers: Any, newurl: str
    ) -> Request:
        raise ReleaseError(f"refusing release credential redirect ({code})")


def _open(request: Request, timeout: int) -> Any:
    return build_opener(_RejectRedirect()).open(request, timeout=timeout)


def _request(
    url: str,
    token: str,
    *,
    method: str = "GET",
    xsrf: str | None = None,
    timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> bytes:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {token}",
        "X-Colab-Client-Agent": "snake-dqn-colab-release",
    }
    if xsrf:
        headers["X-Goog-Colab-Token"] = xsrf
    request = Request(url + "?authuser=0", headers=headers, method=method)
    with _open(request, timeout=timeout) as response:
        return response.read()


def unassign_with_credential(
    credential: Mapping[str, Any],
    endpoint: str,
    *,
    request_timeout: float = REQUEST_TIMEOUT_SECONDS,
) -> None:
    """Use the installed CLI's documented GET/XSRF-POST unassign sequence."""
    checked = _validate_credential(credential, endpoint)
    url = _unassign_url(endpoint)
    try:
        body = _request(url, checked["access_token"], timeout=request_timeout)
        text = body.decode("utf-8")
        if text.startswith(")]}'\n"):
            text = text[5:]
        payload = json.loads(text)
        if not isinstance(payload, dict):
            raise ReleaseError("release GET returned a malformed envelope")
        xsrf = payload.get("token")
        if not isinstance(xsrf, str) or not xsrf:
            raise ReleaseError("release GET returned no XSRF token")
        _request(url, checked["access_token"], method="POST", xsrf=xsrf, timeout=request_timeout)
    except (HTTPError, URLError, OSError, ValueError) as exc:
        raise ReleaseError(f"runtime release request failed: {type(exc).__name__}: {exc}") from exc


def release_from_file(root: str | Path, endpoint: str, *, min_valid_seconds: float = 0) -> None:
    """Release with bounded retries; leaves no credential in diagnostics."""
    credential = load_release_credential(
        root, endpoint, min_valid_seconds=max(min_valid_seconds, RELEASE_MAX_SECONDS)
    )
    expiry = float(credential["expires_at"])
    last_error: BaseException | None = None
    for attempt in range(RELEASE_ATTEMPTS):
        try:
            attempts_after = RELEASE_ATTEMPTS - attempt - 1
            future_seconds = attempts_after * (
                2 * (REQUEST_TIMEOUT_SECONDS + REQUEST_OVERHEAD_SECONDS) + RELEASE_RETRY_SECONDS
            )
            available = expiry - _now() - future_seconds
            request_timeout = min(REQUEST_TIMEOUT_SECONDS, available / 2 - REQUEST_OVERHEAD_SECONDS)
            if request_timeout <= 0:
                raise ReleaseError("release credential expires before remaining bounded attempts")
            unassign_with_credential(credential, endpoint, request_timeout=request_timeout)
            return
        except ReleaseError as exc:
            last_error = exc
            if attempt + 1 < RELEASE_ATTEMPTS:
                time.sleep(RELEASE_RETRY_SECONDS)
    raise ReleaseError(f"runtime release failed after {RELEASE_ATTEMPTS} attempts: {last_error}")
