"""Transport the exact frozen ruler exports in one bounded compressed upload."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import tarfile
import tempfile
from pathlib import Path

from research.bl_served_search.exports import digest, export_bytes, verify_export
from research.bl_served_search.ruler import source_identity, validate_fixtures

BUNDLE_NAME = "ruler-fixtures.tar.gz"
MAX_COMPRESSED_BYTES = 4 * 1024**2
MAX_UNPACKED_BYTES = 16 * 1024**2


def validate_directory(directory: Path) -> list[Path]:
    """Verify all nine original roots, source identity and immutable sidecars."""
    manifest = directory / "ruler-fixtures.json"
    verify_export(manifest)
    value = json.loads(manifest.read_text())
    rows = validate_fixtures(value, full=True)
    if value.get("mode") != "full" or value.get("source_identity") != source_identity():
        raise ValueError("full fixtures with unchanged ruler sources required")
    files = [manifest]
    for row in rows:
        name = row["root_file"]
        if name != f"fixture-{row['fixture_id']}.root.json" or Path(name).name != name:
            raise ValueError("fixture root must be a portable basename")
        path = directory / name
        verify_export(path)
        root = json.loads(path.read_text())
        if any(
            root.get(key) != row[key] for key in ("fixture_id", "root_sha256", "root_bytes_b64")
        ):
            raise ValueError("root sidecar payload disagrees with frozen fixture bank")
        files.append(path)
    return sorted(
        Path(str(path) + suffix) for path in files for suffix in ("", ".sha256", ".complete")
    )


def pack(directory: Path, target: Path) -> Path:
    """Create a deterministic archive without changing any fixture bytes."""
    files = validate_directory(directory)
    if sum(path.stat().st_size for path in files) > MAX_UNPACKED_BYTES:
        raise ValueError("fixture bundle exceeds unpacked byte limit")
    compressed = io.BytesIO()
    with gzip.GzipFile(fileobj=compressed, mode="wb", mtime=0, filename="") as stream:
        with tarfile.open(fileobj=stream, mode="w|", format=tarfile.USTAR_FORMAT) as archive:
            for path in files:
                payload = path.read_bytes()
                info = tarfile.TarInfo(path.name)
                info.size = len(payload)
                info.mode = 0o600
                archive.addfile(info, io.BytesIO(payload))
    payload = compressed.getvalue()
    if len(payload) > MAX_COMPRESSED_BYTES:
        raise ValueError("fixture bundle exceeds compressed byte limit")
    export_bytes(target, payload)
    return target


def unpack(bundle: Path, target: Path, checksum: str) -> Path:
    """Restore into a fresh directory only after all hashes and roots agree."""
    if target.exists():
        raise FileExistsError("fresh fixture destination required")
    if bundle.stat().st_size > MAX_COMPRESSED_BYTES or digest(bundle) != checksum:
        raise ValueError("fixture bundle size/hash mismatch")
    # Bound decompression before parsing tar headers; never use extractall.
    with gzip.open(bundle, "rb") as stream:
        payload = stream.read(MAX_UNPACKED_BYTES + 1)
    if len(payload) > MAX_UNPACKED_BYTES:
        raise ValueError("fixture bundle exceeds unpacked byte limit")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".ruler-fixtures-", dir=target.parent) as temporary:
        stage = Path(temporary) / "verified"
        stage.mkdir()
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            members = archive.getmembers()
            if len(members) != 30:
                raise ValueError("fixture bundle must contain exactly 30 export components")
            if sum(member.size for member in members) > MAX_UNPACKED_BYTES:
                raise ValueError("fixture bundle exceeds extracted byte limit")
            for member in members:
                if (
                    not member.isfile()
                    or member.sparse is not None
                    or member.pax_headers
                    or not 0 <= member.size <= MAX_UNPACKED_BYTES
                    or member.name != Path(member.name).name
                    or member.name in {"", ".", ".."}
                ):
                    raise ValueError("unsafe fixture bundle member")
                source = archive.extractfile(member)
                if source is None:
                    raise ValueError("missing fixture bundle member")
                with (stage / member.name).open("xb") as handle:
                    handle.write(source.read())
        expected = {path.name for path in validate_directory(stage)}
        if {member.name for member in members} != expected:
            raise ValueError("fixture bundle members disagree with frozen fixture bank")
        stage.rename(target)
    return target / "ruler-fixtures.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(unpack(args.bundle, args.out, args.sha256))


if __name__ == "__main__":
    main()
