"""Materialize T33's six public checkpoint inputs with exact byte bindings.

The catalog is validated in full before opening a network connection.  This
module is intentionally a runtime helper, not a publisher or URL discoverer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

CHUNK_BYTES = 1024 * 1024
EXPECTED = {
    "B8_s3_final.pth": (
        "B8_s3_final.pth",
        6562005,
        "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    ),
    "champion_a5_freespace_20260621.pth": (
        "checkpoints/champion_a5_freespace_20260621.pth",
        6898955,
        "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
    ),
    "best_apex_fs.pth": (
        "checkpoints/best_apex_fs.pth",
        3451633,
        "768306b182b98e9175e6d90de1470b26f6d99a6ccfb19ecdedffae29027ea195",
    ),
    "best_apex_stage1_fs.pth": (
        "checkpoints/best_apex_stage1_fs.pth",
        3451651,
        "0eb5c121711ecb81c491bb319faeed232249f1548711f40c21c2e997fccc1c8d",
    ),
    "best_apex_pre_fs.pth": (
        "checkpoints/best_apex_pre_fs.pth",
        3449513,
        "fd96cd00e1000d44e6adfa28c733c4cd86e39caf38bfb5afee16051f4764da6e",
    ),
    "frp3_m3_s12_u60000_20261005.pth": (
        "checkpoints/frp3_m3_s12_u60000_20261005.pth",
        6906247,
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
    ),
}


class PublicInputError(ValueError):
    """A public input is absent, unsafe, or differs from its immutable pin."""


def load_catalog(path: Path) -> list[dict[str, Any]]:
    """Validate catalog identities and every URL before any download begins."""
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise PublicInputError(f"invalid public input catalog: {path}") from exc
    rows = value.get("inputs") if isinstance(value, dict) else None
    if not isinstance(rows, list) or len(rows) != len(EXPECTED):
        raise PublicInputError("catalog must contain exactly the six required public inputs")
    by_name = {row.get("name"): row for row in rows if isinstance(row, dict)}
    if set(by_name) != set(EXPECTED):
        raise PublicInputError("catalog names do not match the fixed T33 input identities")
    for name, (dest, size, digest) in EXPECTED.items():
        row = by_name[name]
        if (row.get("dest"), row.get("size_bytes"), row.get("sha256")) != (dest, size, digest):
            raise PublicInputError(f"catalog identity differs for {name}")
        url = row.get("public_url")
        if (
            not isinstance(url, str)
            or not url.startswith("https://")
            or len(url) <= len("https://")
        ):
            raise PublicInputError(f"publication is pending for {name}; no download was attempted")
    return [by_name[name] for name in EXPECTED]


def _download(
    row: dict[str, Any], out_dir: Path, opener: Callable[..., Any], timeout: float
) -> Path:
    name = str(row["name"])
    dest, size, digest = EXPECTED[name]
    target = out_dir / dest
    target.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    hasher = hashlib.sha256()
    try:
        with (
            opener(str(row["public_url"]), timeout=timeout) as response,
            target.open("xb") as stream,
        ):
            while True:
                block = response.read(min(CHUNK_BYTES, size - written + 1))
                if not block:
                    break
                written += len(block)
                if written > size:
                    raise PublicInputError(f"public input exceeds exact size: {name}")
                hasher.update(block)
                stream.write(block)
        if written != size or hasher.hexdigest() != digest:
            raise PublicInputError(f"public input size or hash mismatch: {name}")
    except Exception:
        target.unlink(missing_ok=True)
        raise
    return target


def materialize(
    catalog_path: Path,
    out_dir: Path,
    *,
    opener: Callable[..., Any] = urlopen,
    timeout: float = 30.0,
) -> list[Path]:
    """Download a complete validated catalog into a fresh directory."""
    rows = load_catalog(catalog_path)
    if out_dir.exists():
        raise PublicInputError(f"output directory must be fresh: {out_dir}")
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir()
    return [_download(row, out_dir, opener, timeout) for row in rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()
    paths = materialize(args.catalog, args.out_dir, timeout=args.timeout)
    print(json.dumps({"inputs": [str(path) for path in paths]}, sort_keys=True))


if __name__ == "__main__":
    main()
