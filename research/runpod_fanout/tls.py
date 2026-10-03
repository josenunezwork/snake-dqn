"""TLS for every HTTPS call the runner makes (verification is never disabled).

The project venv's python.org interpreter ships without a system CA bundle, so the
default context fails with CERTIFICATE_VERIFY_FAILED. Use certifi's bundle when it is
importable (it is in the venv), else the interpreter default.
"""

from __future__ import annotations

import ssl
import urllib.error
import urllib.request
from typing import List, Optional, Sequence

USER_AGENT = "rpf-runner/1.0 (snake-dqn runpod fan-out)"
# TLS-only probes: any HTTP status (401, 404, ...) proves the certificate chain verified.
PREFLIGHT_URLS = (
    "https://rest.runpod.io/v1/pods",
    "https://api.runpod.io/graphql",
    "https://tls-preflight-8000.proxy.runpod.net/",
)


def ca_file() -> Optional[str]:
    try:
        import certifi
    except ImportError:
        return None
    return certifi.where()


def ssl_context() -> ssl.SSLContext:
    """A verifying context (CERT_REQUIRED + hostname check) with certifi's CAs if present."""
    ctx = ssl.create_default_context(cafile=ca_file())
    assert ctx.verify_mode == ssl.CERT_REQUIRED and ctx.check_hostname
    return ctx


def preflight(urls: Sequence[str] = PREFLIGHT_URLS, timeout: float = 20.0) -> List[str]:
    """Problems reaching ``urls`` over verified TLS (empty = OK). Sends no credentials."""
    problems: List[str] = []
    ctx = ssl_context()
    for url in urls:
        req = urllib.request.Request(url, method="GET")
        req.add_header("User-Agent", USER_AGENT)
        try:
            urllib.request.urlopen(req, timeout=timeout, context=ctx).read(64)
        except urllib.error.HTTPError:
            continue  # TLS handshake and certificate verification succeeded
        except urllib.error.URLError as exc:
            problems.append(f"{url}: {exc.reason}")
        except (OSError, ValueError) as exc:
            problems.append(f"{url}: {exc}")
    return problems
