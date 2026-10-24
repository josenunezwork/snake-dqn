"""Hash-verified immutable environment snapshots for retained assignments."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any


def inventory(root: Path) -> dict[str, Any]:
    rows = {}
    for path in sorted(root.rglob("*")):
        relative = str(path.relative_to(root))
        if path.is_symlink():
            # uv's interpreter links refer to the shared managed interpreter.
            # Other links must stay within the environment snapshot.
            if root.resolve() not in path.resolve().parents and not relative.startswith(
                "bin/python"
            ):
                raise RuntimeError("environment snapshot contains an external symlink")
            rows[relative] = {"link": os.readlink(path)}
        elif path.is_file():
            with path.open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            rows[relative] = {"sha256": digest, "size": path.stat().st_size}
    return rows


def publish(cache: Path, environment: Path, contract: str) -> None:
    """Publish an environment once; no job outputs or credentials enter it."""
    if cache.exists():
        raise RuntimeError("partial or existing immutable environment cache")
    cache.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".environment-", dir=cache.parent))
    try:
        shutil.copytree(environment, temporary / ".venv", symlinks=True)
        (temporary / "cache-contract.json").write_text(contract)
        (temporary / "environment-inventory.json").write_text(
            json.dumps(
                {"prefix": str(environment), "files": inventory(temporary / ".venv")},
                sort_keys=True,
            )
        )
        (temporary / "environment.complete").write_text(
            hashlib.sha256((temporary / "environment-inventory.json").read_bytes()).hexdigest()
        )
        # No replacement: the broker admits only one serial chain per cache.
        os.rename(temporary, cache)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def restore(cache: Path, environment: Path, contract: str) -> None:
    """Verify before copying, then relocate console-script interpreter headers."""
    if environment.exists() or (cache / "cache-contract.json").read_text() != contract:
        raise RuntimeError("environment cache contract or fresh destination differs")
    raw = (cache / "environment-inventory.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != (cache / "environment.complete").read_text():
        raise RuntimeError("environment cache inventory publication differs")
    saved = json.loads(raw)
    if saved["files"] != inventory(cache / ".venv"):
        raise RuntimeError("immutable environment cache bytes changed")
    shutil.copytree(cache / ".venv", environment, symlinks=True)
    original = saved["prefix"]
    for path in (environment / "bin").iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        try:
            text = path.read_text()
        except UnicodeDecodeError:
            continue
        if original in text:
            path.write_text(text.replace(original, str(environment)))


if __name__ == "__main__":
    action, cache, environment, contract = sys.argv[1:]
    {"publish": publish, "restore": restore}[action](Path(cache), Path(environment), contract)
