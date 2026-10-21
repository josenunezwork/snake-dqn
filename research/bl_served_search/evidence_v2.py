"""One bounded upload for immutable bank, authority, ruler and parity evidence."""

from __future__ import annotations

import argparse
import base64
import gzip
import json
from pathlib import Path

from research.bl_served_search.exports import digest, export_bytes, verify_export

NAME = "l2-evidence.json.gz"
MAX_BYTES = 6 * 1024**2
MAX_UNPACKED = 16 * 1024**2
ALLOWED = {
    "bank.json",
    "audit.json",
    "registration.md",
    "amendment-08.md",
    "amendment-v2.md",
    "ruler-fixture-spec-08.md",
    "ruler-fixtures.tar.gz",
    "ruler-mac.json",
    "ruler-l4.json",
    "parity-mac.json",
}


def pack(files: list[tuple[str, Path]], out: Path) -> Path:
    """Bind payload bytes and sidecars without editing original evidence."""
    members = {}
    for name, path in files:
        if name not in ALLOWED or name in members:
            raise ValueError("unexpected or duplicate evidence name")
        if Path(str(path) + ".complete").exists():
            verify_export(path)
        for suffix in ("", ".sha256", ".complete"):
            source = Path(str(path) + suffix)
            if suffix and not source.exists():
                continue  # committed Markdown is bound by the envelope digest
            payload = (
                f"{digest(path)}  {name}\n".encode() if suffix == ".sha256" else source.read_bytes()
            )
            members[name + suffix] = base64.b64encode(payload).decode()
    raw = json.dumps(members, sort_keys=True, separators=(",", ":")).encode()
    payload = gzip.compress(raw, mtime=0)
    if len(payload) > MAX_BYTES or len(raw) > MAX_UNPACKED:
        raise ValueError("evidence exceeds bounded transport")
    export_bytes(out, payload)
    return out


def unpack(path: Path, out: Path, checksum: str) -> Path:
    """Restore a fresh evidence directory; reject traversal and decompression bombs."""
    if out.exists():
        raise FileExistsError("fresh evidence directory required")
    if path.stat().st_size > MAX_BYTES or digest(path) != checksum:
        raise ValueError("evidence size/hash mismatch")
    with gzip.open(path, "rb") as stream:
        raw = stream.read(MAX_UNPACKED + 1)
    if len(raw) > MAX_UNPACKED:
        raise ValueError("unpacked evidence too large")
    value = json.loads(raw)
    allowed = {name + suffix for name in ALLOWED for suffix in ("", ".sha256", ".complete")}
    if not isinstance(value, dict) or not set(value) <= allowed:
        raise ValueError("unsafe evidence members")
    decoded = {name: base64.b64decode(data, validate=True) for name, data in value.items()}
    if sum(map(len, decoded.values())) > MAX_UNPACKED:
        raise ValueError("decoded evidence too large")
    out.mkdir(parents=True)
    for name, data in decoded.items():
        with (out / name).open("xb") as handle:
            handle.write(data)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(unpack(args.bundle, args.out, args.sha256))


if __name__ == "__main__":
    main()
