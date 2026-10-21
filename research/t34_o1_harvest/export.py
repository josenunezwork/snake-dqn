"""Create the bounded immutable O1 runtime export bundle."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path

MAX_EXPORT_BYTES = 64 * 1024**2


class ExportRefused(ValueError):
    """The completed O1 output tree cannot become a safe transport export."""


def _members(run_dir: Path) -> list[Path]:
    if not run_dir.is_absolute() or not run_dir.is_dir() or run_dir.is_symlink():
        raise ExportRefused("run directory must be an absolute non-symlink directory")
    members = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if not members:
        raise ExportRefused("run directory has no output files")
    if any(path.is_symlink() for path in members):
        raise ExportRefused("run directory may not contain symlinks")
    return members


def _tar_info(path: Path, name: str) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.size = path.stat().st_size
    info.mode = 0o644
    info.mtime = 0
    info.uid = info.gid = 0
    info.uname = info.gname = ""
    return info


def bundle(run_dir: Path, output: Path) -> str:
    """Write a reproducible tar.gz and runtime-compatible immutable sidecars."""
    run_dir, output = run_dir.resolve(), output.resolve()
    members = _members(run_dir)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ExportRefused("export output must be fresh, absolute, and non-symlink")
    if output.parent == run_dir or run_dir in output.parents:
        raise ExportRefused("export output must be outside the run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with (
            output.open("xb") as raw,
            gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as compressed,
        ):
            with tarfile.open(fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT) as archive:
                for path in members:
                    relative = path.relative_to(run_dir).as_posix()
                    with path.open("rb") as stream:
                        archive.addfile(_tar_info(path, relative), stream)
        if output.stat().st_size > MAX_EXPORT_BYTES:
            raise ExportRefused("export exceeds the registered 64 MiB cap")
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        checksum = output.with_name(output.name + ".sha256")
        marker = output.with_name(output.name + ".complete")
        with checksum.open("x", encoding="utf-8") as stream:
            stream.write(f"{digest}  {output.name}\n")
        with marker.open("x", encoding="utf-8") as stream:
            stream.write("complete\n")
        return digest
    except Exception:
        # The runtime accepts only completed markers.  A failed partial bundle
        # is deliberately left unmarked for terminal receipt diagnosis.
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(bundle(args.run_dir, args.out), flush=True)


if __name__ == "__main__":
    main()
