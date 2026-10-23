"""Lossless A2 observation projection; no replay, fitting, or outcome scoring."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
import tempfile
from pathlib import Path

import numpy as np

from research.a100_a2.data import _safe_extract, _verify_bundle
from research.bl_ppo.exports import digest

COLUMNS = ("source_decision", "state_packed", "state_scalars")
SCHEMA = "a2-observation-projection/v1"


def project_shard(payload: bytes) -> tuple[bytes, dict]:
    """Retain original dtypes and first-seen order; refuse conflicting duplicates."""
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in COLUMNS}
    decisions = arrays["source_decision"]
    seen: dict[int, int] = {}
    keep = []
    for index, decision in enumerate(decisions):
        value = int(decision)
        if value in seen:
            prior = seen[value]
            if any(arrays[k][prior].tobytes() != arrays[k][index].tobytes() for k in COLUMNS):
                raise ValueError("conflicting duplicate observation")
        else:
            seen[value] = index
            keep.append(index)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **{key: arrays[key][keep] for key in COLUMNS})
    return buffer.getvalue(), {"original_rows": len(decisions), "unique_rows": len(keep)}


def compact_bundle(source: Path, output: Path) -> dict:
    """Project only T4e observations; copy all other source files byte-for-byte."""
    if output.exists():
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    source_sha = digest(source)
    with tempfile.TemporaryDirectory(prefix="a2-project-") as temporary:
        root = _safe_extract(source, Path(temporary))
        original = _verify_bundle(root)
        original_receipt_sha = digest(root / "bundle.json")
        (root / "bundle.json").rename(root / "source-bundle.json")
        shards = {}
        for item in original["files"]:
            name = item["path"]
            if name.startswith("t4e/") and name.endswith(".npz"):
                path = root / name
                projected, counts = project_shard(path.read_bytes())
                path.write_bytes(projected)
                shards[name] = {
                    "source_sha256": item["sha256"],
                    "sha256": digest(path),
                    **counts,
                }
        projection = {
            "schema": SCHEMA,
            "source_archive_sha256": source_sha,
            "source_bundle_sha256": original_receipt_sha,
            "columns": list(COLUMNS),
            "shards": shards,
        }
        (root / "projection.json").write_text(json.dumps(projection, sort_keys=True) + "\n")
        files = [
            {"path": str(p.relative_to(root)), "sha256": digest(p), "bytes": p.stat().st_size}
            for p in sorted(root.rglob("*"))
            if p.is_file()
        ]
        receipt = {
            **original,
            "source_archive_sha256": source_sha,
            "source_bundle_sha256": original_receipt_sha,
            "files": files,
        }
        (root / "bundle.json").write_text(json.dumps(receipt, sort_keys=True) + "\n")
        # XZ benefits the verbose A5 JSON without changing any JSON bytes.
        with tarfile.open(output, "x:xz", preset=9) as archive:
            archive.add(root, arcname="labels", recursive=True)
    if digest(source) != source_sha:
        raise ValueError("source archive changed during projection")
    return {
        "source_archive_sha256": source_sha,
        "archive_sha256": digest(output),
        "archive_bytes": output.stat().st_size,
        "shards": len(shards),
        "original_rows": sum(s["original_rows"] for s in shards.values()),
        "unique_rows": sum(s["unique_rows"] for s in shards.values()),
    }


def verify_projection(bundle: Path, source_archive_sha: str | None = None) -> dict:
    """Bind projected observations to the original source inventory and manifests."""
    compact = _verify_bundle(bundle)
    projection = json.loads((bundle / "projection.json").read_text())
    source = json.loads((bundle / "source-bundle.json").read_text())
    if (
        projection.get("schema") != SCHEMA
        or projection.get("columns") != list(COLUMNS)
        or projection.get("source_archive_sha256") != compact.get("source_archive_sha256")
        or (
            source_archive_sha is not None
            and compact.get("source_archive_sha256") != source_archive_sha
        )
        or digest(bundle / "source-bundle.json") != projection.get("source_bundle_sha256")
        or compact.get("source_bundle_sha256") != projection.get("source_bundle_sha256")
    ):
        raise ValueError("projection source identity differs")
    projected = projection["shards"]
    expected_shards = {
        i["path"]
        for i in source["files"]
        if i["path"].startswith("t4e/") and i["path"].endswith(".npz")
    }
    if set(projected) != expected_shards:
        raise ValueError("projection shard inventory differs")
    for item in source["files"]:
        name = item["path"]
        if name in projected:
            row = projected[name]
            if row["source_sha256"] != item["sha256"] or digest(bundle / name) != row["sha256"]:
                raise ValueError("projection shard identity differs")
            manifest = json.loads((bundle / name).parent.joinpath("manifest.json").read_text())
            if manifest["shards"].get(Path(name).name) != item["sha256"]:
                raise ValueError("projection original manifest differs")
        elif digest(bundle / name) != item["sha256"]:
            raise ValueError("projection changed an unprojected source")
    return projection


def verify_compact_archive(archive: Path, source_archive_sha: str) -> dict:
    """Exercise the exact archive/lineage path in the excluded family canary."""
    with tempfile.TemporaryDirectory(prefix="a2-check-projection-") as temporary:
        root = _safe_extract(archive, Path(temporary))
        projection = verify_projection(root, source_archive_sha)
    return {"source_archive_sha256": source_archive_sha, "shards": len(projection["shards"])}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(compact_bundle(args.source, args.out), sort_keys=True))


if __name__ == "__main__":
    main()
