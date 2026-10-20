"""Download the two public S1 assets and verify all six T34 checkpoint bytes."""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
from pathlib import Path, PurePosixPath
from typing import Any, Callable
from urllib.request import urlopen

CHUNK = 1024 * 1024
EXPECTED_ASSETS = {
    "B8_s3_final.pth": (
        6562005,
        "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
    ),
    "league.tar.gz": (21362869, "e2c691063bd2d6f4c3ec396b4d4520b1eb367c4f0814517b7b74d6e4db0f4186"),
}
EXPECTED_MEMBERS = {
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
    """The pinned public release cannot provide exactly the registered inputs."""


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _load(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    value = json.loads(path.read_text())
    rows, assets = value.get("inputs"), value.get("assets")
    if (
        not isinstance(rows, list)
        or len(rows) != 6
        or not isinstance(assets, list)
        or len(assets) != 2
    ):
        raise PublicInputError("catalog must bind six checkpoint members and two release assets")
    by_asset = {str(item.get("name")): item for item in assets if isinstance(item, dict)}
    if set(by_asset) != {"B8_s3_final.pth", "league.tar.gz"}:
        raise PublicInputError("catalog asset names differ from the registered release")
    for asset in by_asset.values():
        if not isinstance(asset.get("url"), str) or not asset["url"].startswith("https://"):
            raise PublicInputError("public release URL is absent")
        if not isinstance(asset.get("size_bytes"), int) or not isinstance(asset.get("sha256"), str):
            raise PublicInputError("public release asset lacks immutable byte binding")
    by_name = {str(item.get("name")): item for item in rows if isinstance(item, dict)}
    if set(by_name) != set(EXPECTED_MEMBERS):
        raise PublicInputError(
            "catalog checkpoint members differ from the registered six identities"
        )
    for name, (dest, size, digest) in EXPECTED_MEMBERS.items():
        row = by_name[name]
        if (row.get("dest"), row.get("size_bytes"), row.get("sha256")) != (dest, size, digest):
            raise PublicInputError(f"catalog checkpoint member identity differs: {name}")
    for name, (size, digest) in EXPECTED_ASSETS.items():
        asset = by_asset[name]
        if (asset["size_bytes"], asset["sha256"]) != (size, digest):
            raise PublicInputError(f"public release asset identity differs: {name}")
    return rows, by_asset


def _download(asset: dict[str, Any], target: Path, opener: Callable[..., Any]) -> None:
    written, digest = 0, hashlib.sha256()
    with opener(str(asset["url"]), timeout=60) as response, target.open("xb") as stream:
        while block := response.read(CHUNK):
            written += len(block)
            if written > int(asset["size_bytes"]):
                raise PublicInputError(f"release asset exceeds declared size: {asset['name']}")
            digest.update(block)
            stream.write(block)
    if written != int(asset["size_bytes"]) or digest.hexdigest() != asset["sha256"]:
        raise PublicInputError(f"release asset bytes differ: {asset['name']}")


def materialize(
    catalog: Path, out_dir: Path, *, opener: Callable[..., Any] = urlopen
) -> list[Path]:
    """Create a fresh input tree after archive and per-member verification."""
    rows, assets = _load(catalog)
    if out_dir.exists():
        raise PublicInputError(f"input directory must be fresh: {out_dir}")
    out_dir.mkdir(parents=True)
    archive = out_dir / "league.tar.gz"
    try:
        _download(assets["B8_s3_final.pth"], out_dir / "B8_s3_final.pth", opener)
        _download(assets["league.tar.gz"], archive, opener)
        with tarfile.open(archive, "r:gz") as tar:
            members = {
                Path(member.name).name: member for member in tar.getmembers() if member.isfile()
            }
            for row in rows:
                name, dest = str(row["name"]), PurePosixPath(str(row["dest"]))
                if dest.is_absolute() or ".." in dest.parts:
                    raise PublicInputError(f"unsafe member destination: {dest}")
                target = out_dir / str(dest)
                if name == "B8_s3_final.pth":
                    continue
                member = members.get(name)
                if member is None:
                    raise PublicInputError(f"league archive lacks {name}")
                target.parent.mkdir(parents=True, exist_ok=True)
                source = tar.extractfile(member)
                if source is None:
                    raise PublicInputError(f"unreadable league member {name}")
                with source, target.open("xb") as stream:
                    while block := source.read(CHUNK):
                        stream.write(block)
        paths = []
        for row in rows:
            target = out_dir / str(row["dest"])
            if (
                not target.is_file()
                or target.stat().st_size != row["size_bytes"]
                or _sha(target) != row["sha256"]
            ):
                raise PublicInputError(f"checkpoint member bytes differ: {row['name']}")
            paths.append(target)
        archive.unlink()
        return paths
    except Exception:
        for path in sorted(out_dir.rglob("*"), reverse=True):
            if path.is_file():
                path.unlink()
            elif path.is_dir():
                path.rmdir()
        out_dir.rmdir()
        raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            {"inputs": [str(path) for path in materialize(args.catalog, args.out_dir)]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
