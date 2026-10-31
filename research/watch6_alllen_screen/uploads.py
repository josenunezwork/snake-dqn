"""Bounded 2 MiB checkpoint parts, packed metadata archive, and offline upload budget gate."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from research.colab_cli import source, transfer_policy
from research.watch6_alllen_screen import protocol as p
from tools.ops.package_spec import Input, digest, verify_immutable

PART_BYTES = 2 * 1024**2
MAX_FILE_BYTES = 4 * 1024**2
MAX_METADATA_BYTES = MAX_FILE_BYTES
METADATA = "input-metadata.tar.gz"
INDEX = "input-inventory.json"

# Measured packet-01 upload model (~5 s/RPC + ~1 MB/s) and target budget (<=12 transfers, <=90 s).
MAX_TRANSFERS = 12
SECONDS_PER_TRANSFER = 5.0
BYTES_PER_SECOND = 1_000_000
MAX_ESTIMATED_TRANSFER_SECONDS = 90.0

CHECKPOINTS: dict[str, tuple[str, int]] = {
    "b8_s3_t7_20261016.pth": ("b8-s3", 6_562_005),
    "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12", 6_906_247),
}


def part_names() -> set[str]:
    """Exact 2 MiB checkpoint part destinations uploaded individually."""
    return {
        f"checkpoint-parts/{name}.{index:03d}.part"
        for name, (_, size) in CHECKPOINTS.items()
        for index in range((size + PART_BYTES - 1) // PART_BYTES)
    }


def metadata_names() -> set[str]:
    """Exact small payloads and all .sha256/.complete sidecars bundled into METADATA."""
    names = {
        f"{name}{suffix}"
        for name in (
            "main-contract.json",
            "main-ledger.json",
            "canary-contract.json",
            "canary-ledger.json",
            "release-evidence.tar.gz",
        )
        for suffix in ("", ".sha256", ".complete")
    }
    names.update(f"{part}{suffix}" for part in part_names() for suffix in (".sha256", ".complete"))
    return names


def pack_inputs(rows: list[dict[str, Any]], archive_path: Path) -> list[dict[str, Any]]:
    """Bundle small immutable files and sidecars; keep each 2 MiB checkpoint part separate."""
    expected_small = metadata_names()
    expected_parts = part_names()
    small = [row for row in rows if row["dest"] in expected_small]
    parts = [row for row in rows if row["dest"] not in expected_small]
    if {row["dest"] for row in small} != expected_small or len(small) != len(expected_small):
        raise ValueError("metadata inventory differs from frozen schema")
    if {row["dest"] for row in parts} != expected_parts or len(parts) != len(expected_parts):
        raise ValueError("checkpoint part inventory differs from frozen schema")
    for row in parts:
        if int(row["size_bytes"]) > PART_BYTES:
            raise ValueError(f"checkpoint part exceeds 2 MiB single-chunk bound: {row['dest']}")

    inventory = {
        row["dest"]: {"sha256": row["sha256"], "size": int(row["size_bytes"])} for row in small
    }
    payloads = {row["dest"]: Path(row["path"]).read_bytes() for row in small}
    payloads[INDEX] = json.dumps(inventory, sort_keys=True).encode("utf-8")
    if sum(map(len, payloads.values())) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds 4 MiB unpacked size bound")

    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(payloads.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))

    archive_size = archive_path.stat().st_size
    if archive_size > MAX_FILE_BYTES:
        raise ValueError("compressed metadata archive exceeds 4 MiB per-file bound")
    return parts + [
        {
            "path": str(archive_path),
            "dest": METADATA,
            "sha256": digest(archive_path),
            "size_bytes": archive_size,
        }
    ]


def stage_metadata(root: Path, expected_sha256: str) -> None:
    """Verify the manifest-bound archive and every member before create-only publication."""
    root = root.resolve(strict=True)
    path = root / METADATA
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > MAX_METADATA_BYTES
        or digest(path) != expected_sha256
    ):
        raise ValueError("metadata archive SHA256 or size mismatch")

    expected_small = metadata_names()
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or set(names) != expected_small | {INDEX}
            or any(not member.isfile() or member.size < 0 for member in members)
            or sum(member.size for member in members) > MAX_METADATA_BYTES
        ):
            raise ValueError("unsafe or unexpected metadata archive inventory")
        payloads: dict[str, bytes] = {}
        for member in members:
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"missing tar member stream: {member.name}")
            payloads[member.name] = extracted.read()

    inventory = json.loads(payloads.pop(INDEX).decode("utf-8"))
    if set(inventory) != expected_small:
        raise ValueError("metadata index inventory differs")

    for name, data in payloads.items():
        if inventory[name] != {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}:
            raise ValueError(f"metadata member digest/size mismatch: {name}")
        target = root / name
        if target.exists() or target.is_symlink():
            raise ValueError(f"metadata destination already exists or is unsafe: {name}")
        cur = target.parent
        while cur != root:
            if cur.is_symlink():
                raise ValueError(f"metadata parent directory is a symlink: {cur}")
            cur = cur.parent

    # Publish all payloads and .sha256 sidecars first, then .complete markers last.
    with tempfile.TemporaryDirectory(prefix=".metadata-", dir=root) as temporary:
        ordered = sorted(payloads, key=lambda item: (item.endswith(".complete"), item))
        for index, name in enumerate(ordered):
            staged = Path(temporary) / str(index)
            staged.write_bytes(payloads[name])
            staged.chmod(0o444)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, target)


def checkpoint_inputs(saved: Path, directory: Path) -> list[Input]:
    """Split exact released checkpoint bytes into 2 MiB parts without reserializing."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text("utf-8"))
    directory.mkdir(parents=True, exist_ok=False)
    inputs: list[Input] = []
    for name, (pin, size) in CHECKPOINTS.items():
        path = saved / name
        if (
            path.stat().st_size != size
            or digest(path) != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"released checkpoint identity differs: {name}")
        with path.open("rb") as stream:
            for index in range((size + PART_BYTES - 1) // PART_BYTES):
                part = directory / f"{name}.{index:03d}.part"
                with part.open("xb") as output:
                    output.write(stream.read(PART_BYTES))
                inputs.append(Input(part, "checkpoint-parts/" + part.name))
    return inputs


def stage_checkpoints(input_root: Path) -> None:
    """Verify every immutable checkpoint part, then atomically publish full checkpoints."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text("utf-8"))
    output = input_root / "checkpoints"
    output.mkdir(exist_ok=False)
    for name, (pin, size) in CHECKPOINTS.items():
        with tempfile.NamedTemporaryFile(dir=output, delete=False) as stream:
            temporary = Path(stream.name)
            try:
                for index in range((size + PART_BYTES - 1) // PART_BYTES):
                    part = input_root / "checkpoint-parts" / f"{name}.{index:03d}.part"
                    verified = verify_immutable(part)
                    if verified["size"] != min(PART_BYTES, size - index * PART_BYTES):
                        raise ValueError(f"checkpoint part size differs: {part.name}")
                    stream.write(part.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
                if (
                    temporary.stat().st_size != size
                    or digest(temporary) != pins["checkpoints"][pin]["checkpoint_sha256"]
                ):
                    raise ValueError(f"reassembled checkpoint digest differs from pin: {name}")
                temporary.chmod(0o444)
                os.link(temporary, output / name)
            finally:
                temporary.unlink(missing_ok=True)


def check_uploads(manifest: dict[str, Any]) -> dict[str, Any]:
    """Enforce <= 4 MiB per file, <= 12 transfers, and <= 90 s at 5 s/RPC + 1 MB/s."""
    limit = int(manifest["max_export_bytes"])
    if limit <= 0 or limit > MAX_FILE_BYTES:
        raise ValueError(f"manifest max_export_bytes must be in 1..{MAX_FILE_BYTES}, got {limit}")
    rows: list[dict[str, Any]] = []

    def check(name: str, size: int) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest["inputs"]:
        if "path" in item:
            check("inputs/" + item["dest"], Path(item["path"]).stat().st_size)

    with tempfile.TemporaryDirectory(prefix="watch6-alllen-upload-preflight-") as temporary:
        staging = Path(temporary)
        source.prepare(p.ROOT, manifest, staging)
        for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
            data = subprocess.check_output(
                ["git", "show", f"{manifest['commit']}:research/colab_cli/{name}"],
                cwd=p.ROOT,
                timeout=10,
            )
            (staging / name).write_bytes(data)
        for path in sorted(staging.iterdir()):
            check(path.name, path.stat().st_size)

    normal_transfers = [
        row
        for row in rows
        if row["dest"].startswith("inputs/") or row["dest"] in ("bootstrap.sh", "source.py")
    ]
    fallback_transfers = [
        row
        for row in rows
        if row["dest"].startswith("inputs/")
        or row["dest"] in ("bootstrap.sh", "source.py", "source.tar.gz")
    ]

    normal_count = len(normal_transfers)
    normal_bytes = sum(int(row["size_bytes"]) for row in normal_transfers)
    normal_estimated = normal_count * SECONDS_PER_TRANSFER + normal_bytes / BYTES_PER_SECOND

    count = len(fallback_transfers)
    total_bytes = sum(int(row["size_bytes"]) for row in fallback_transfers)
    estimated = count * SECONDS_PER_TRANSFER + total_bytes / BYTES_PER_SECOND

    transfer_cap = transfer_policy.transfer_limit(manifest)
    staging_estimated = estimated + source.FETCH_TIMEOUT_SECONDS
    if (
        count > MAX_TRANSFERS
        or estimated > MAX_ESTIMATED_TRANSFER_SECONDS
        or estimated >= transfer_cap
        or staging_estimated >= transfer_policy.MAX_STAGING_SECONDS
    ):
        raise ValueError(
            f"upload transfer forecast refuses offline: {count} transfers (max {MAX_TRANSFERS}), "
            f"{estimated:.3f}s (budget {MAX_ESTIMATED_TRANSFER_SECONDS:g}s, "
            f"cap {transfer_cap:g}s); {staging_estimated:.3f}s staging forecast"
        )

    return {
        "status": "PASS",
        "max_upload_bytes": limit,
        "files": rows,
        "normal_transfer_count": normal_count,
        "normal_transfer_bytes": normal_bytes,
        "normal_estimated_transfer_seconds": normal_estimated,
        "transfer_count": count,
        "transfer_bytes": total_bytes,
        "seconds_per_transfer": SECONDS_PER_TRANSFER,
        "bytes_per_second": BYTES_PER_SECOND,
        "max_transfers_budget": MAX_TRANSFERS,
        "max_estimated_transfer_seconds_budget": MAX_ESTIMATED_TRANSFER_SECONDS,
        "estimated_transfer_seconds": estimated,
        "transfer_cap_seconds": transfer_cap,
        "estimated_staging_seconds": staging_estimated,
        "staging_cap_seconds": transfer_policy.MAX_STAGING_SECONDS,
        "scope": "offline forecast (normal + sparse fallback) under 5 s/RPC + 1 MB/s model",
    }
