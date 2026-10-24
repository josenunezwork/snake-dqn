"""Read-only GitHub publication and HTTPS time authority for I2 admission.

No Git dates authorize publication. All requests use a fixed HTTPS API host,
GET, no proxies or redirects, and bounded execution. Credentials never enter receipts/logs.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import http.client
import json
import math
import os
import signal
import ssl
import subprocess
import threading
import time
from contextlib import contextmanager
from datetime import datetime
from email.utils import parsedate_to_datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator
from urllib.parse import parse_qs, urlparse

import yaml

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_residual import f2

REPOSITORY = "josenunezwork/snake-dqn"
API = f"repos/{REPOSITORY}"
REF = "refs/heads/main"
TIME_POLICY = {
    "authority": "https://api.github.com",
    "repository": REPOSITORY,
    "ref": REF,
    "max_clock_skew_seconds": 120,
    "max_request_seconds": 30,
    "publication_schema": "bl-residual-github-publication/v1",
}


class AuthorityUnavailable(ValueError):
    """No verifiable server evidence; never substitute local or Git time."""


HOST = "api.github.com"
PORT = 443
MAX_RESPONSE_BYTES = 8_000_000
_OVERRIDE_ENV = {
    "GH_CONFIG_DIR",
    "GH_HOST",
    "GH_API_URL",
    "GITHUB_API_URL",
    "GITHUB_GRAPHQL_URL",
    "SSL_CERT_FILE",
    "SSL_CERT_DIR",
    "REQUESTS_CA_BUNDLE",
    "CURL_CA_BUNDLE",
}


def _reject_transport_overrides() -> None:
    """Fail closed on caller routing/trust overrides; never log their values."""
    if any(key.upper() in _OVERRIDE_ENV or key.upper().endswith("_PROXY") for key in os.environ):
        raise AuthorityUnavailable("GitHub transport environment override detected")
    config_dir = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "gh"

    seen: set[int] = set()

    def check(value: Any, depth: int = 0) -> None:
        if depth > 32 or id(value) in seen:
            raise AuthorityUnavailable("GitHub configuration cannot be verified")
        if isinstance(value, (dict, list)):
            seen.add(id(value))
        if isinstance(value, dict):
            for key, item in value.items():
                if str(key).lower() == "http_unix_socket" and item:
                    raise AuthorityUnavailable("GitHub Unix socket override detected")
                check(item, depth + 1)
        elif isinstance(value, list):
            for item in value:
                check(item, depth + 1)

    # These files are inspected only for forbidden transport settings. They are
    # never used by the HTTPS client. hosts.yml may contain secrets: no logging.
    for name in ("config.yml", "hosts.yml"):
        path = config_dir / name
        if path.exists():
            if path.stat().st_size > 1_000_000:
                raise AuthorityUnavailable("GitHub configuration cannot be verified")
            try:
                seen.clear()
                check(yaml.safe_load(path.read_bytes()))
            except (yaml.YAMLError, RecursionError):
                raise AuthorityUnavailable("GitHub configuration cannot be verified") from None


@lru_cache(maxsize=1)
def _token() -> str:
    """Read credentials once per process, without any API call or Git helper."""
    token = os.environ.get("GITHUB_TOKEN")
    if token is None:
        # Only credential discovery uses gh; no inherited GH_*, GIT_*, proxy,
        # trust, debug or alternate-host settings enter this subprocess.
        env = {
            key: os.environ[key]
            for key in ("PATH", "HOME", "USER", "LOGNAME", "TMPDIR", "XDG_CONFIG_HOME")
            if key in os.environ
        }
        env["GH_PROMPT_DISABLED"] = "1"
        result = subprocess.run(
            ["gh", "auth", "token", "--hostname", "github.com"],
            env=env,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode:
            raise AuthorityUnavailable("GitHub credential unavailable")
        token = result.stdout.strip()
    if not token or any(ord(char) < 33 or ord(char) > 126 for char in token):
        raise AuthorityUnavailable("GitHub credential unavailable")
    return token


def _system_tls_context() -> ssl.SSLContext:
    """Load OpenSSL's system roots explicitly, bypassing all CA environment vars."""
    paths = ssl.get_default_verify_paths()
    cafile = paths.openssl_cafile if Path(paths.openssl_cafile).is_file() else None
    capath = paths.openssl_capath if Path(paths.openssl_capath).is_dir() else None
    # python.org macOS builds may have no compiled CA path. Use the OS bundle,
    # never certifi or a caller-supplied bundle, as the explicit fallback.
    if cafile is None and Path("/etc/ssl/cert.pem").is_file():
        cafile = "/etc/ssl/cert.pem"
    if cafile is None and capath is None:
        raise AuthorityUnavailable("system TLS trust roots unavailable")
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    context.verify_mode = ssl.CERT_REQUIRED
    context.check_hostname = True
    context.load_verify_locations(cafile=cafile, capath=capath)
    return context


@contextmanager
def _deadline() -> Iterator[None]:
    """Bound DNS, handshake, headers and body without workers or extra threads."""
    if threading.current_thread() is not threading.main_thread() or any(
        signal.getitimer(signal.ITIMER_REAL)
    ):
        raise AuthorityUnavailable("GitHub wall-clock deadline cannot be installed")
    previous = signal.getsignal(signal.SIGALRM)

    def expired(_signum: int, _frame: Any) -> None:
        raise AuthorityUnavailable("GitHub response deadline exceeded")

    signal.signal(signal.SIGALRM, expired)
    try:
        signal.setitimer(signal.ITIMER_REAL, 30)
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def _request(endpoint: str) -> tuple[Any, dict[str, str], dict[str, Any]]:
    with _deadline():
        return _request_https(endpoint)


def _request_https(endpoint: str) -> tuple[Any, dict[str, str], dict[str, Any]]:
    """GET directly from the pinned TLS host; no gh API, proxies or redirects."""
    if (
        not endpoint.startswith(API + "/")
        or "://" in endpoint
        or "#" in endpoint
        or any(ord(char) <= 32 or ord(char) >= 127 for char in endpoint)
    ):
        raise AuthorityUnavailable("foreign GitHub authority endpoint")
    connection = None
    try:
        _reject_transport_overrides()
        token = _token()
        context = _system_tls_context()
        if context.verify_mode != ssl.CERT_REQUIRED or not context.check_hostname:
            raise AuthorityUnavailable("TLS verification is required")
        local_before, tick = time.time(), time.monotonic()
        # http.client has no proxy/environment discovery and never follows 3xx.
        connection = http.client.HTTPSConnection(HOST, PORT, timeout=30, context=context)
        connection.connect()
        peer = connection.sock
        if (
            connection.host != HOST
            or connection.port != PORT
            or peer is None
            or peer.context is not context
            or peer.server_hostname != HOST
            or not peer.getpeercert()
            or not peer.version()
        ):
            raise AuthorityUnavailable("GitHub TLS peer cannot be verified")
        tls_version = peer.version()
        connection.request(
            "GET",
            "/" + endpoint,
            headers={
                "Host": HOST,
                "Authorization": "Bearer " + token,
                "User-Agent": "snake-dqn-i2-authority",
                "Accept": "application/vnd.github+json",
                "X-GitHub-Api-Version": "2022-11-28",
                "Cache-Control": "no-cache",
                "Connection": "close",
            },
        )
        response = connection.getresponse()
        if response.status != 200:
            raise AuthorityUnavailable("GitHub authority requires HTTP 200; redirects refused")
        headers: dict[str, str] = {}
        for name, value in response.getheaders():
            key = name.lower()
            if key in headers:
                raise AuthorityUnavailable("ambiguous GitHub response header")
            headers[key] = value.strip()
        body = bytearray()
        while True:
            remaining = 30 - (time.monotonic() - tick)
            if remaining <= 0:
                raise AuthorityUnavailable("GitHub response deadline exceeded")
            peer.settimeout(remaining)
            chunk = response.read1(min(65536, MAX_RESPONSE_BYTES + 1 - len(body)))
            body.extend(chunk)
            if len(body) > MAX_RESPONSE_BYTES:
                raise AuthorityUnavailable("GitHub response exceeds size limit")
            if not chunk:
                break
        elapsed, local_after = time.monotonic() - tick, time.time()
        date = parsedate_to_datetime(headers["date"])
        epoch = date.timestamp()
        if (
            date.utcoffset() is None
            or date.utcoffset().total_seconds() != 0
            or not all(math.isfinite(value) for value in (epoch, local_before, local_after))
            or not 0 <= elapsed <= 30
            or abs((local_after - local_before) - elapsed) > 2
            or max(abs(epoch - local_before), abs(epoch - local_after)) > 120
            or int(headers.get("age", "0")) > 0
            or not headers.get("x-github-request-id")
        ):
            raise AuthorityUnavailable("unverifiable GitHub time or clock skew > 120 seconds")
        observation = {
            "endpoint": f"https://{HOST}/{endpoint}",
            "verified_peer_host": HOST,
            "peer_port": PORT,
            "tls_verified": True,
            "tls_version": tls_version,
            "date": headers["date"],
            "epoch": epoch,
            "request_id": headers["x-github-request-id"],
            "local_before_epoch": local_before,
            "local_after_epoch": local_after,
            "elapsed_seconds": elapsed,
        }
        return json.loads(body), headers, observation
    except AuthorityUnavailable:
        raise
    except (OSError, ValueError, KeyError, http.client.HTTPException, subprocess.SubprocessError):
        # Neither token subprocess output nor server payloads enter exceptions.
        raise AuthorityUnavailable("unverifiable GitHub API/time response") from None
    finally:
        if connection is not None:
            connection.close()


def trusted_time() -> dict[str, Any]:
    """Fresh server Date observation, required even for the strikes branch."""
    value, _, observation = _request(f"{API}/git/ref/heads/main")
    if (
        not isinstance(value, dict)
        or value.get("ref") != REF
        or not isinstance(value.get("object"), dict)
        or value["object"].get("type") != "commit"
    ):
        raise AuthorityUnavailable("GitHub main ref is unverifiable")
    return observation


def _valid_commit(sha: Any) -> bool:
    return isinstance(sha, str) and len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def _contains(commit: str, tip: str) -> bool:
    """Ask the server for ancestry, even when both objects exist locally."""
    if not _valid_commit(commit) or commit == "0" * 40 or not _valid_commit(tip):
        raise AuthorityUnavailable("invalid activity commit")
    if tip == "0" * 40:
        return False
    value, _, _ = _request(f"{API}/compare/{commit}...{tip}")
    if (
        not isinstance(value, dict)
        or not isinstance(value.get("base_commit"), dict)
        or not isinstance(value.get("merge_base_commit"), dict)
        or value["base_commit"].get("sha") != commit
        or not _valid_commit(value["merge_base_commit"].get("sha"))
        or value.get("status") not in ("ahead", "identical", "behind", "diverged")
    ):
        raise AuthorityUnavailable("unverifiable GitHub comparison")
    contains = value["merge_base_commit"]["sha"] == commit
    if contains != (value["status"] in ("ahead", "identical")):
        raise AuthorityUnavailable("inconsistent GitHub comparison")
    return contains


def _server_export(commit: str, relative: str, root: Path) -> bytes:
    """Bind the ledger and BOTH sidecars to server bytes at an exact commit.

    Contents requests use an immutable full SHA, never a branch name. Validate
    the returned Git blob ID as well as SHA256 equality with each local file.
    A missing file, truncated response or API error is unavailable evidence.
    """
    if not _valid_commit(commit) or relative not in (f2.INPUT_LEDGER, f2.RESULT_LEDGER):
        raise AuthorityUnavailable("invalid committed export identity")
    ledger = b""
    for suffix in ("", ".sha256", ".complete"):
        name = relative + suffix
        value, _, _ = _request(f"{API}/contents/{name}?ref={commit}")
        if (
            not isinstance(value, dict)
            or value.get("type") != "file"
            or value.get("path") != name
            or value.get("encoding") != "base64"
            or not isinstance(value.get("content"), str)
            or type(value.get("size")) is not int
            or not 0 <= value["size"] <= 2_000_000
            or not _valid_commit(value.get("sha"))
        ):
            raise AuthorityUnavailable("unverifiable GitHub ledger blob")
        try:
            payload = base64.b64decode("".join(value["content"].splitlines()), validate=True)
        except ValueError as exc:
            raise AuthorityUnavailable("invalid GitHub blob encoding") from exc
        blob = b"blob " + str(len(payload)).encode("ascii") + b"\0" + payload
        if (
            len(payload) != value["size"]
            or hashlib.sha1(blob).hexdigest() != value["sha"]
            or hashlib.sha256(payload).digest()
            != hashlib.sha256((root / name).read_bytes()).digest()
        ):
            raise AuthorityUnavailable("GitHub committed ledger/sidecar bytes differ")
        if not suffix:
            ledger = payload
    return ledger


def first_publication(commit: str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Find the first server push transition that made this exact commit reachable.

    Scan all available activity pages, refusing truncated pagination, broken
    ancestry or absent transitions. An exact rebased commit needs its own event.
    An older unobservable introduction can only make this conservative (later),
    never grant eligibility from a preserved author/committer timestamp.
    """
    # Cache only within this fresh scan; every revalidation contacts the server.
    ancestry: dict[str, bool] = {}

    def contains(tip: str) -> bool:
        if not _valid_commit(tip):
            raise AuthorityUnavailable("invalid activity commit")
        if tip not in ancestry:
            ancestry[tip] = _contains(commit, tip)
        return ancestry[tip]

    head, _, observation = _request(f"{API}/git/ref/heads/main")
    if (
        not isinstance(head, dict)
        or head.get("ref") != REF
        or not isinstance(head.get("object"), dict)
        or head["object"].get("type") != "commit"
        or not contains(head["object"].get("sha"))
    ):
        raise AuthorityUnavailable("result commit is not on authoritative GitHub main")
    endpoint = f"{API}/activity?ref={REF}&per_page=100"
    candidates = []
    seen: set[str] = set()
    for _ in range(100):
        rows, headers, _ = _request(endpoint)
        if not isinstance(rows, list):
            raise AuthorityUnavailable("invalid repository activity")
        for row in rows:
            if not isinstance(row, dict):
                raise AuthorityUnavailable("invalid repository activity row")
            if row.get("ref") != REF or row.get("activity_type") not in ("push", "force_push"):
                continue
            before, after = row.get("before"), row.get("after")
            if contains(after) and not contains(before):
                timestamp = row.get("timestamp")
                try:
                    if not isinstance(timestamp, str) or not timestamp.endswith("Z"):
                        raise ValueError("activity time must be UTC")
                    date = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
                except ValueError as exc:
                    raise AuthorityUnavailable("invalid server activity timestamp") from exc
                epoch = date.timestamp()
                if (
                    date.utcoffset() is None
                    or not math.isfinite(epoch)
                    or epoch > observation["epoch"]
                    or not row.get("id")
                ):
                    raise AuthorityUnavailable("invalid server activity timestamp")
                candidates.append(
                    {
                        "id": row["id"],
                        "timestamp": timestamp,
                        "epoch": epoch,
                        "before": before,
                        "after": after,
                        "ref": REF,
                        "activity_type": row["activity_type"],
                    }
                )
        next_links = [
            part.split(";", 1)[0].strip()[1:-1]
            for part in headers.get("link", "").split(",")
            if 'rel="next"' in part
        ]
        if not next_links:
            break
        if len(next_links) != 1 or next_links[0] in seen:
            raise AuthorityUnavailable("ambiguous/cyclic activity pagination")
        seen.add(next_links[0])
        url = urlparse(next_links[0])
        if (
            url.scheme != "https"
            or url.netloc != "api.github.com"
            or url.path
            not in (f"/repos/{REPOSITORY}/activity", "/repositories/1271845215/activity")
            or parse_qs(url.query).get("ref") != [REF]
        ):
            raise AuthorityUnavailable("foreign GitHub activity pagination")
        # GitHub emits its numeric repository route; keep requests on the pinned named repo.
        endpoint = f"{API}/activity?{url.query}"
    else:
        raise AuthorityUnavailable("GitHub activity scan truncated")
    if not candidates:
        raise AuthorityUnavailable("no independently timestamped publication transition")
    return min(candidates, key=lambda row: (row["epoch"], str(row["id"]))), observation


def published_receipt_binding(receipt: Path, *, root: Path | None = None) -> dict[str, str]:
    """Verify publication metadata without semantically reading the sealed result."""
    from research.bl_residual.f2_inputs import _plain_path

    root = f2.ROOT if root is None else Path(root)
    receipt = _plain_path(receipt)
    if receipt != f2.RECEIPT.resolve():
        raise ValueError("F2 requires the canonical one-look receipt")
    f2._published(root / f2.INPUT_LEDGER, f2.INPUT_LEDGER, root)
    plan = json.loads((root / f2.INPUT_LEDGER).read_text())
    if not isinstance(plan, dict) or not isinstance(plan.get("sources", {}), dict):
        raise ValueError("F2 input ledger must contain a metadata object")
    f2._validate_plan(plan, root)
    publication_ref = f2._published(root / f2.RESULT_LEDGER, f2.RESULT_LEDGER, root)
    publication = json.loads((root / f2.RESULT_LEDGER).read_text())
    if (
        not isinstance(publication, dict)
        or publication.get("schema") != "bl-residual-f2-publication/v1"
        or publication.get("receipt_path") != str(receipt)
        or publication.get("receipt_sha256") != verify_export(receipt)
        or publication.get("receipt_bytes") != receipt.stat().st_size
        or publication.get("source_sha256") != digest(root / f2.INPUT_LEDGER)
        or publication.get("judgment_sha256") != f2.JUDGMENT_SHA
    ):
        raise ValueError("F2 receipt differs from its published ledger")
    f2._commit_contains(
        root,
        publication["input_published_ref"],
        f2.INPUT_LEDGER,
        (root / f2.INPUT_LEDGER).read_bytes(),
    )
    return {
        "path": str(receipt),
        "sha256": publication["receipt_sha256"],
        "source_sha256": publication["source_sha256"],
        "publication_ref": publication_ref,
    }


def evidence_path(binding: dict[str, Any]) -> Path:
    return Path(binding["path"]).with_name(
        f"publication-evidence-{binding['publication_ref']}.json"
    )


def _identity(binding: dict[str, Any]) -> dict[str, Any]:
    commit = binding["publication_ref"]
    ledger = f2.ROOT / f2.RESULT_LEDGER
    if f2._published(ledger, f2.RESULT_LEDGER, f2.ROOT) != commit:
        raise AuthorityUnavailable("F2 publication commit changed after verification")
    f2._commit_contains(f2.ROOT, commit, f2.RESULT_LEDGER, ledger.read_bytes())
    payload = _server_export(commit, f2.RESULT_LEDGER, f2.ROOT)
    # Also bind the input ledger/sidecars used by F2's identity checks. Only
    # publication metadata is parsed here; the sealed statistic stays closed.
    publication = json.loads(payload)
    if not isinstance(publication, dict):
        raise AuthorityUnavailable("invalid GitHub publication metadata")
    input_commit = publication["input_published_ref"]
    if not _contains(input_commit, commit):
        raise AuthorityUnavailable("input publication is not an ancestor of the result")
    inputs = _server_export(input_commit, f2.INPUT_LEDGER, f2.ROOT)
    if (
        publication.get("receipt_sha256") != binding["sha256"]
        or publication.get("source_sha256") != hashlib.sha256(inputs).hexdigest()
    ):
        raise AuthorityUnavailable("GitHub publication metadata differs from the F2 binding")
    return {
        "schema": TIME_POLICY["publication_schema"],
        "repository": REPOSITORY,
        "ref": REF,
        "commit": commit,
        "ledger": f2.RESULT_LEDGER,
        "ledger_sha256": hashlib.sha256(payload).hexdigest(),
        "receipt_sha256": binding["sha256"],
    }


def capture_publication(binding: dict[str, Any]) -> dict[str, Any]:
    """Create-only metadata receipt after real publication; no semantic outcome read."""
    identity = _identity(binding)
    event, observation = first_publication(identity["commit"])
    return export_json(
        evidence_path(binding),
        {
            **identity,
            "activity": event,
            "captured_time_observation": observation,
        },
    )


def publication_evidence(binding: dict[str, Any]) -> dict[str, Any] | None:
    """Revalidate the sealed receipt against live GitHub; absent proof means strikes."""
    try:
        path = evidence_path(binding)
        sha = verify_export(path)
        receipt = json.loads(path.read_text())
        identity = _identity(binding)
        event, _ = first_publication(identity["commit"])
        if (
            not isinstance(receipt, dict)
            or {key: receipt.get(key) for key in identity} != identity
            or receipt.get("activity") != event
        ):
            raise AuthorityUnavailable("publication receipt differs from GitHub authority")
        return {
            **identity,
            "activity": event,
            "published_epoch": event["epoch"],
            "evidence_path": str(path),
            "evidence_sha256": sha,
        }
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("check-time", "capture-publication"))
    args = parser.parse_args()
    if args.action == "check-time":
        result = trusted_time()
    else:
        result = capture_publication(published_receipt_binding(f2.RECEIPT))
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
