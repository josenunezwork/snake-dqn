"""Version fence for exact uv warning grammar; never modify qualified transport."""

from __future__ import annotations

import argparse
import re
import subprocess
from typing import Any, Mapping

# Captured locally for packet 11; smoke08/09 did not retain uv version metadata.
# The matching --system warning was reproduced with this exact release offline.
QUALIFIED_UV_VERSION = "0.13.1"
UV_VERSION_ENV = "CT_QUALIFIED_UV_VERSION"


def require_uv_version(expected: str, command: list[str]) -> str:
    """Refuse unavailable, malformed, or changed uv before any numerical work."""
    if expected != QUALIFIED_UV_VERSION:
        raise ValueError("uv version differs from qualified capture")
    result = subprocess.run(command, text=True, capture_output=True, timeout=10, check=True)
    date = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
    build = rf"(?:[0-9a-f]{{7,40}} {date}|Homebrew {date} aarch64-apple-darwin)"
    match = re.fullmatch(rf"uv ([0-9]+\.[0-9]+\.[0-9]+)(?: \({build}\))?\n?", result.stdout)
    if not match or result.stderr or match[1] != expected:
        raise ValueError("uv version differs from qualified capture")
    return match[1]


def require_manifest_uv(manifest: Mapping[str, Any]) -> None:
    """Check the sealed expected version at host dispatch admission."""
    if manifest.get("env", {}).get(UV_VERSION_ENV) != QUALIFIED_UV_VERSION:
        raise ValueError("manifest lacks qualified uv version")
    require_uv_version(QUALIFIED_UV_VERSION, ["uv", "--version"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("expected")
    args = parser.parse_args()
    require_uv_version(args.expected, ["uv", "--version"])


if __name__ == "__main__":
    main()
