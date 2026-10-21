"""Create a deterministic, bounded O2 export with supervisor sidecars."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import tarfile
from pathlib import Path

MAX_EXPORT_BYTES = 64 * 1024**2


class ExportRefused(ValueError):
    """The proposed O2 evidence export is incomplete or unsafe."""


def _members(run_dir: Path) -> list[Path]:
    if not run_dir.is_absolute() or not run_dir.is_dir() or run_dir.is_symlink():
        raise ExportRefused("run directory must be an absolute non-symlink directory")
    files = sorted(path for path in run_dir.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ExportRefused("run directory needs non-symlink evidence files")
    return files


def bundle(run_dir: Path, output: Path) -> str:
    """Write a reproducible create-only tar.gz plus verified runtime markers."""
    run_dir, output = run_dir.resolve(), output.resolve()
    files = _members(run_dir)
    if not output.is_absolute() or output.exists() or output.is_symlink():
        raise ExportRefused("export output must be fresh, absolute, and non-symlink")
    if output.parent == run_dir or run_dir in output.parents:
        raise ExportRefused("export output must be outside the run directory")
    output.parent.mkdir(parents=True, exist_ok=True)
    with (
        output.open("xb") as raw,
        gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped,
    ):
        with tarfile.open(fileobj=zipped, mode="w", format=tarfile.PAX_FORMAT) as archive:
            for path in files:
                info = tarfile.TarInfo(path.relative_to(run_dir).as_posix())
                info.size = path.stat().st_size
                info.mode, info.mtime, info.uid, info.gid = 0o644, 0, 0, 0
                info.uname = info.gname = ""
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    if output.stat().st_size > MAX_EXPORT_BYTES:
        raise ExportRefused("export exceeds the registered 64 MiB cap")
    digest = hashlib.sha256(output.read_bytes()).hexdigest()
    with output.with_name(output.name + ".sha256").open("x", encoding="utf-8") as stream:
        stream.write(f"{digest}  {output.name}\n")
    with output.with_name(output.name + ".complete").open("x", encoding="utf-8") as stream:
        stream.write("complete\n")
    return digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(bundle(args.run_dir, args.out), flush=True)


if __name__ == "__main__":
    main()
