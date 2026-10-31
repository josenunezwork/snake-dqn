"""Pack small immutable inputs/sidecars into one deterministic archive (<=12 transfers, <<150 s)."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any

from research.watch_replay.run import ROOT
from tools.ops.package_spec import digest, verify_immutable

CHUNK_BYTES = 2 * 1024**2
MAX_METADATA_BYTES = 4 * 1024**2
METADATA = "input-metadata.tar.gz"
INDEX = "input-inventory.json"

MAX_TRANSFERS = 12
SECONDS_PER_TRANSFER = 5.0
BYTES_PER_SECOND = 1_000_000
MAX_ESTIMATED_TRANSFER_SECONDS = 90.0

CHECKPOINTS: dict[str, tuple[str, int]] = {
    "b8_s3_t7_20261016.pth": ("b8-s3", 6_562_005),
    "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12", 6_906_247),
}

SMALL_PAYLOADS: tuple[str, ...] = (
    "main-contract.json",
    "main-ledger.json",
    "canary-contract.json",
    "canary-ledger.json",
    "release-evidence.tar.gz",
)


def checkpoint_names() -> set[str]:
    """Direct checkpoint input destinations uploaded in parallel 2 MiB chunks by transfers.py."""
    return {f"checkpoints/{name}" for name in CHECKPOINTS}


def metadata_names() -> set[str]:
    """Exact small payloads and all .sha256/.complete sidecars bundled into METADATA."""
    names = {
        f"{name}{suffix}" for name in SMALL_PAYLOADS for suffix in ("", ".sha256", ".complete")
    }
    names.update(
        f"checkpoints/{name}{suffix}" for name in CHECKPOINTS for suffix in (".sha256", ".complete")
    )
    return names


def _verify_checkpoint_pins(rows_or_root: list[dict[str, Any]] | Path) -> None:
    pins = json.loads(
        (ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    if isinstance(rows_or_root, Path):
        for name, (pin, size) in CHECKPOINTS.items():
            path = rows_or_root / "checkpoints" / name
            verified = verify_immutable(path)
            if (
                verified["size"] != size
                or verified["sha256"] != pins["checkpoints"][pin]["checkpoint_sha256"]
            ):
                raise ValueError(f"staged checkpoint differs from released pin: {name}")
        return
    by_dest = {row["dest"]: row for row in rows_or_root}
    for name, (pin, size) in CHECKPOINTS.items():
        row = by_dest[f"checkpoints/{name}"]
        if (
            int(row["size_bytes"]) != size
            or row["sha256"] != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"released checkpoint identity differs: {name}")


def pack_inputs(
    rows: list[dict[str, Any]], archive_path: Path, *, verify_pins: bool = True
) -> list[dict[str, Any]]:
    """Bundle 19 small immutable files/sidecars into METADATA; keep 2 checkpoints direct."""
    expected_small = metadata_names()
    expected_direct = checkpoint_names()
    small = [row for row in rows if row["dest"] in expected_small]
    direct = [row for row in rows if row["dest"] not in expected_small]
    if {row["dest"] for row in small} != expected_small or len(small) != len(expected_small):
        raise ValueError("metadata inventory differs from frozen schema")
    if {row["dest"] for row in direct} != expected_direct or len(direct) != len(expected_direct):
        raise ValueError("checkpoint inventory differs from frozen schema")
    if verify_pins:
        _verify_checkpoint_pins(direct)

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
    if archive_size > MAX_METADATA_BYTES:
        raise ValueError("compressed metadata archive exceeds 4 MiB bound")
    ordered_direct = sorted(direct, key=lambda row: str(row["dest"]))
    return [
        *(
            {
                "path": str(row["path"]),
                "dest": str(row["dest"]),
                "sha256": str(row["sha256"]),
                "size_bytes": int(row["size_bytes"]),
            }
            for row in ordered_direct
        ),
        {
            "path": str(archive_path),
            "dest": METADATA,
            "sha256": digest(archive_path),
            "size_bytes": archive_size,
        },
    ]


def stage_metadata(root: Path, expected_sha256: str, *, verify_pins: bool = True) -> None:
    """Verify METADATA archive, unpack create-only with .complete last, verify all triplets."""
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

    for payload_name in SMALL_PAYLOADS:
        verify_immutable(root / payload_name)
    if verify_pins:
        _verify_checkpoint_pins(root)
    else:
        for ckpt_name in CHECKPOINTS:
            verify_immutable(root / "checkpoints" / ckpt_name)


def _serial_rpcs_for_size(size: int) -> int:
    """Count effective serial RPC waves (2 parallel chunk workers + 1 assemble RPC if >2 MiB)."""
    if size <= CHUNK_BYTES:
        return 1
    chunks = math.ceil(size / CHUNK_BYTES)
    return math.ceil(chunks / 2) + 1


def check_uploads(manifest: dict[str, Any]) -> dict[str, Any]:
    """Enforce <= 12 transfers and <= 90 s offline forecast (well under 150 s transport cap)."""
    from research.colab_cli import source, transfer_policy

    limit = int(manifest["max_export_bytes"])
    if limit <= 0:
        raise ValueError(f"manifest max_export_bytes must be positive, got {limit}")
    rows: list[dict[str, Any]] = []

    def check(name: str, size: int) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest["inputs"]:
        if "path" in item:
            check("inputs/" + item["dest"], Path(item["path"]).stat().st_size)

    with tempfile.TemporaryDirectory(prefix="watch-replay-upload-preflight-") as temporary:
        staging = Path(temporary)
        source.prepare(ROOT, manifest, staging)
        for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
            data = subprocess.check_output(
                ["git", "show", f"{manifest['commit']}:research/colab_cli/{name}"],
                cwd=ROOT,
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
    normal_waves = sum(_serial_rpcs_for_size(int(row["size_bytes"])) for row in normal_transfers)
    normal_estimated = normal_waves * SECONDS_PER_TRANSFER + normal_bytes / BYTES_PER_SECOND

    count = len(fallback_transfers)
    total_bytes = sum(int(row["size_bytes"]) for row in fallback_transfers)
    fallback_waves = sum(
        _serial_rpcs_for_size(int(row["size_bytes"])) for row in fallback_transfers
    )
    chunk_puts = sum(
        max(1, math.ceil(int(row["size_bytes"]) / CHUNK_BYTES)) for row in fallback_transfers
    )
    estimated = fallback_waves * SECONDS_PER_TRANSFER + total_bytes / BYTES_PER_SECOND

    transfer_cap = transfer_policy.transfer_limit(manifest)
    staging_estimated = estimated + source.FETCH_TIMEOUT_SECONDS
    if (
        count > MAX_TRANSFERS
        or chunk_puts > MAX_TRANSFERS
        or estimated > MAX_ESTIMATED_TRANSFER_SECONDS
        or estimated >= transfer_cap
        or staging_estimated >= transfer_policy.MAX_STAGING_SECONDS
    ):
        raise ValueError(
            f"upload transfer forecast refuses offline: {count} transfers / {chunk_puts} chunk "
            f"PUTs (max {MAX_TRANSFERS}), {estimated:.3f}s (budget "
            f"{MAX_ESTIMATED_TRANSFER_SECONDS:g}s, cap {transfer_cap:g}s); "
            f"{staging_estimated:.3f}s staging forecast"
        )

    return {
        "status": "PASS",
        "max_upload_bytes": limit,
        "files": rows,
        "normal_transfer_count": normal_count,
        "normal_transfer_bytes": normal_bytes,
        "normal_estimated_transfer_seconds": normal_estimated,
        "transfer_count": count,
        "chunk_put_count": chunk_puts,
        "transfer_bytes": total_bytes,
        "seconds_per_transfer": SECONDS_PER_TRANSFER,
        "bytes_per_second": BYTES_PER_SECOND,
        "max_transfers_budget": MAX_TRANSFERS,
        "max_estimated_transfer_seconds_budget": MAX_ESTIMATED_TRANSFER_SECONDS,
        "estimated_transfer_seconds": estimated,
        "transfer_cap_seconds": transfer_cap,
        "estimated_staging_seconds": staging_estimated,
        "staging_cap_seconds": transfer_policy.MAX_STAGING_SECONDS,
        "scope": "offline forecast (normal + sparse fallback) under 5 s/RPC-wave + 1 MB/s model",
    }
