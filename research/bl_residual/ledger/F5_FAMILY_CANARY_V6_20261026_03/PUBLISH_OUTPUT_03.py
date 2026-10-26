"""Preserve every saved canary-03 output byte without opening scientific payloads."""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "f5-timing-recovery-offline-20261026/canary-03"
)
DEST = Path(__file__).resolve().parent
PART_SIZE = 1_500_000


def digest(raw: bytes) -> str:
    """Hash opaque exact bytes."""
    return hashlib.sha256(raw).hexdigest()


def encode(obj: object) -> bytes:
    """Encode publication metadata."""
    return (json.dumps(obj, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def create(path: Path, raw: bytes) -> None:
    """Create once, or verify identical bytes after interrupted publication."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.is_symlink() or not path.is_file() or path.read_bytes() != raw:
            raise ValueError(f"existing publication differs: {path}")
        return
    with path.open("xb") as stream:
        stream.write(raw)


def publish(path: Path, raw: bytes) -> None:
    """Write payload, SHA sidecar, and completion marker last."""
    create(path, raw)
    create(Path(f"{path}.sha256"), f"{digest(raw)}  {path.name}\n".encode())
    create(Path(f"{path}.complete"), b"complete\n")


def preserve(path: Path, raw: bytes) -> dict:
    """Publish exact bytes, compressing and splitting large opaque files."""
    result = {"size": len(raw), "sha256": digest(raw)}
    if len(raw) <= PART_SIZE:
        publish(path, raw)
        result.update(encoding="identity", parts=[str(path.relative_to(DEST))])
    else:
        packed = gzip.compress(raw, mtime=0)
        parts = []
        for index, offset in enumerate(range(0, len(packed), PART_SIZE)):
            part = Path(f"{path}.gz.part-{index:03d}")
            publish(part, packed[offset : offset + PART_SIZE])
            parts.append(str(part.relative_to(DEST)))
        assert gzip.decompress(b"".join((DEST / part).read_bytes() for part in parts)) == raw
        result.update(encoding="gzip-concatenated-parts", parts=parts)
        publish(Path(f"{path}.COMPRESSED_INDEX.json"), encode(result))
        create(Path(f"{path}.sha256"), f"{digest(raw)}  {path.name}\n".encode())
        create(Path(f"{path}.complete"), b"complete\n")
    return result


def main() -> None:
    """Publish the complete saved output set before any verification report."""
    source = ROOT / "output-03"
    if (DEST / "OUTPUT_PUBLICATION_03.json").exists():
        raise FileExistsError("publication already sealed")
    paths = sorted(source.rglob("*"))
    if source.is_symlink() or any(p.is_symlink() for p in paths):
        raise ValueError("symlink refused")
    entries = []
    for path in paths:
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("nonregular output refused")
        relative = path.relative_to(source)
        metadata = preserve(DEST / "output-03" / relative, path.read_bytes())
        entries.append({"source_relative_path": str(relative), **metadata})
    acceptance = ROOT.parent.parent / "astra-20261024/ACCEPT_F5_FAMILY_CANARY_03.md"
    preserve(DEST / acceptance.name, acceptance.read_bytes())
    script = Path(__file__)
    publish(script, script.read_bytes())
    publish(
        DEST / "OUTPUT_PUBLICATION_03.json",
        encode(
            {
                "schema": "bl-f5-family-canary-output-publication/v2",
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "source_root": str(source),
                "scope": "COMPLETE_SAVED_OUTPUT_03_OPAQUE_BYTES",
                "entries": entries,
                "original_file_count": len(entries),
                "original_total_bytes": sum(e["size"] for e in entries),
                "excluded": ["state-03", "accounts", "ops records outside output-03"],
                "scientific_payloads_deserialized": False,
                "provider_calls": 0,
                "dispatches": 0,
                "verification_report_created": False,
            }
        ),
    )
    print(json.dumps({"files": len(entries), "bytes": sum(e["size"] for e in entries)}))


if __name__ == "__main__":
    main()
