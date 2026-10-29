"""Closed uv version-line grammar; versions are logged, never startup-gated."""

import re

# Official pip-wheel Display includes a target, optionally preceded by hash/date.
# Keep older version-only/hash-date and the captured local Homebrew form too.
# https://github.com/astral-sh/uv/blob/0.13.6/crates/uv-cli/src/version.rs
_DATE = r"[0-9]{4}-[0-9]{2}-[0-9]{2}"
_TARGET = r"(?:x86_64-unknown-linux-(?:gnu|musl)|aarch64-apple-darwin)"
_BUILD = (
    rf"(?:[0-9a-f]{{7,40}} {_DATE}(?: {_TARGET})?|"
    rf"{_TARGET}|Homebrew {_DATE} aarch64-apple-darwin)"
)
UV_VERSION_PATTERN = re.compile(rf"uv [0-9]+\.[0-9]+\.[0-9]+(?: \({_BUILD}\))?")
