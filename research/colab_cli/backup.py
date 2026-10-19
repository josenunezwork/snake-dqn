"""Bounded subprocess target for optional Mac Drive-for-desktop backups."""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
import uuid
from pathlib import Path


def copy_verified(source: Path, destination: Path) -> None:
    """Publish only after copying and hashing every acknowledged snapshot."""
    if not destination.is_absolute() or destination.exists():
        raise ValueError("backup destination must be absolute and fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp = destination.with_name(".backup-" + uuid.uuid4().hex)
    try:
        shutil.copytree(source, temp)
        for marker in temp.rglob("*.complete"):
            payload = marker.with_name(marker.name[: -len(".complete")])
            sidecar = Path(str(payload) + ".sha256")
            with payload.open("rb") as handle:
                digest = hashlib.file_digest(handle, "sha256").hexdigest()
            if digest != sidecar.read_text().split()[0]:
                raise ValueError("backup hash mismatch")
        os.rename(temp, destination)
    finally:
        if temp.exists():
            shutil.rmtree(temp)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    copy_verified(args.source, args.destination)


if __name__ == "__main__":
    main()
