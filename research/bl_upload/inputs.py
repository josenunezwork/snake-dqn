"""Materialize the six registered public checkpoints into a fresh runtime tree."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path, PurePosixPath

from research.t34_exit_b8.public_inputs import materialize


def materialize_public_checkpoints(
    input_dir: Path, *, aliases: dict[str, str] | None = None
) -> Path:
    """Download and verify the public checkpoint catalog below *input_dir*.

    ``public_inputs.materialize`` deliberately refuses a pre-existing output
    directory.  That makes a retry unable to reuse unverified partial bytes.
    """
    root = input_dir.resolve()
    # The runtime's uv environment has certifi.  Use it when available so the
    # public release verifier does not depend on a notebook image CA bundle.
    try:
        import certifi
    except ImportError:
        pass
    else:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())
    catalog = root / "public-inputs.json"
    if not catalog.is_file():
        raise ValueError(f"missing uploaded public input catalog: {catalog}")
    target = root / "public-checkpoints"
    materialize(catalog, target)
    for source, dest in (aliases or {}).items():
        source_rel, dest_rel = PurePosixPath(source), PurePosixPath(dest)
        if (
            not source
            or not dest
            or source_rel.is_absolute()
            or dest_rel.is_absolute()
            or ".." in source_rel.parts
            or ".." in dest_rel.parts
        ):
            raise ValueError(f"unsafe public checkpoint alias: {source!r} -> {dest!r}")
        source_path, dest_path = target / source_rel, root / dest_rel
        if not source_path.is_file() or dest_path.exists() or dest_path.is_symlink():
            raise ValueError(f"unsafe public checkpoint alias: {source!r} -> {dest!r}")
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        os.link(source_path, dest_path)
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    args = parser.parse_args()
    aliases = json.loads(os.environ.get("BL_PUBLIC_INPUT_ALIASES", "{}"))
    if not isinstance(aliases, dict) or not all(
        isinstance(source, str) and isinstance(dest, str) for source, dest in aliases.items()
    ):
        raise ValueError("BL_PUBLIC_INPUT_ALIASES must map strings to strings")
    root = materialize_public_checkpoints(args.input_dir, aliases=aliases)
    print(json.dumps({"input_root": str(root)}, sort_keys=True))


if __name__ == "__main__":
    main()
