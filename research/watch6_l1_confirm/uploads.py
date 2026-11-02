"""Single-archive (<2 MiB, <=8 transfers, <=60 s) input packing and VM staging for Watch6-L1-V1."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable
from urllib.request import urlopen

from research.t34_exit_b8 import public_inputs
from research.watch6_l1_confirm import protocol as p
from research.watch_replay.uploads import (
    BYTES_PER_SECOND,
    CHECKPOINTS,
    CHUNK_BYTES,
    INDEX,
    MAX_ESTIMATED_TRANSFER_SECONDS,
    MAX_METADATA_BYTES,
    MAX_TRANSFERS,
    METADATA,
    SECONDS_PER_TRANSFER,
    _assert_safe_new_target,
    _serial_rpcs_for_size,
    _verify_checkpoint_pins,
    materialize_watch_checkpoints,
)
from tools.ops.package_spec import digest, verify_immutable

SMALL_PAYLOADS: tuple[str, ...] = (
    "runtime-contract.json",
    "verified-ledger.json",
    "judge-approval.json",
    "mac-mirror.json",
    "cost-evidence.json",
    "release-evidence.tar.gz",
    "public-inputs.json",
)


def metadata_names() -> set[str]:
    """Exact small immutable payloads and .sha256/.complete sidecars bundled into METADATA."""
    return {f"{name}{suffix}" for name in SMALL_PAYLOADS for suffix in ("", ".sha256", ".complete")}


def pack_inputs(
    rows: list[dict[str, Any]],
    archive_path: Path,
    *,
    verify_pins: bool = True,
) -> list[dict[str, Any]]:
    """Bundle all small immutable payloads and sidecars into one <=2 MiB METADATA archive."""
    expected_small = metadata_names()
    if {row["dest"] for row in rows} != expected_small or len(rows) != len(expected_small):
        raise ValueError(
            "Watch6-L1-V1 metadata inventory differs from frozen SMALL_PAYLOADS schema"
        )
    by_dest = {row["dest"]: row for row in rows}
    for name in SMALL_PAYLOADS:
        verified = verify_immutable(Path(by_dest[name]["path"]))
        if verified["size"] != int(by_dest[name]["size_bytes"]) or verified["sha256"] != str(
            by_dest[name]["sha256"]
        ):
            raise ValueError(f"staged payload metadata mismatch: {name}")
    if verify_pins:
        _verify_checkpoint_pins(Path(by_dest["public-inputs.json"]["path"]))

    inventory = {
        row["dest"]: {"sha256": row["sha256"], "size": int(row["size_bytes"])} for row in rows
    }
    payloads = {row["dest"]: Path(row["path"]).read_bytes() for row in rows}
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
            member.mtime = 0
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, io.BytesIO(data))

    archive_size = archive_path.stat().st_size
    if archive_size > CHUNK_BYTES:
        raise ValueError("compressed metadata archive exceeds 2 MiB single-chunk bound")
    return [
        {
            "path": str(archive_path),
            "dest": METADATA,
            "sha256": digest(archive_path),
            "size_bytes": archive_size,
        }
    ]


def stage_metadata(
    root: Path,
    expected_sha256: str,
    *,
    verify_pins: bool = True,
    materialize_checkpoints: bool = False,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Verify METADATA archive, unpack create-only (.complete last), optionally fetch models."""
    root = root.resolve(strict=True)
    path = root / METADATA
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > CHUNK_BYTES
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
        _assert_safe_new_target(root, name)

    with tempfile.TemporaryDirectory(prefix=".l1-metadata-", dir=root) as temporary:
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
        _verify_checkpoint_pins(root / "public-inputs.json")
    if materialize_checkpoints:
        materialize_watch_checkpoints(root, verify_pins=verify_pins, opener=opener)


def check_uploads(manifest: dict[str, Any]) -> dict[str, Any]:
    """Enforce <= 8 transfers, <= 2 MiB per local input, and <= 60 s offline transfer forecast."""
    import math

    from research.colab_cli import source, transfer_policy

    limit = int(manifest["max_export_bytes"])
    if limit <= 0:
        raise ValueError(f"manifest max_export_bytes must be positive, got {limit}")
    rows: list[dict[str, Any]] = []

    def check(name: str, size: int, *, local_input: bool = False) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        if local_input and size > CHUNK_BYTES:
            raise ValueError(
                f"unsplit local input exceeds 2 MiB single-chunk bound: "
                f"{name}: {size} > {CHUNK_BYTES}"
            )
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest["inputs"]:
        if "path" in item:
            in_path = Path(item["path"])
            if in_path.is_symlink() or not in_path.is_file():
                raise ValueError(f"missing or symlinked input path: {in_path}")
            check("inputs/" + item["dest"], in_path.stat().st_size, local_input=True)

    with tempfile.TemporaryDirectory(prefix="watch6-l1-upload-preflight-") as temporary:
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
        normal_count > MAX_TRANSFERS
        or count > MAX_TRANSFERS
        or chunk_puts > MAX_TRANSFERS
        or normal_estimated > MAX_ESTIMATED_TRANSFER_SECONDS
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
        "max_input_part_bytes": CHUNK_BYTES,
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
        "checkpoints": list(CHECKPOINTS),
        "scope": "offline forecast (normal + sparse fallback) under 5 s/RPC-wave + 1 MB/s model",
    }


def stage_vm_inputs() -> None:
    """VM entrypoint helper: unpack input-metadata.tar.gz and fetch verified checkpoints."""
    input_root = Path(os.environ["COLAB_INPUT_DIR"])
    expected_sha = os.environ["WATCH6_METADATA_SHA256"]
    try:
        stage_metadata(
            input_root,
            expected_sha,
            verify_pins=True,
            materialize_checkpoints=True,
        )
    except public_inputs.PublicInputError as exc:
        print(
            f"WATCH6_L1_PRECOMPUTE_PUBLIC_INPUT_FETCH_FAILED: {exc}",
            file=sys.stderr,
            flush=True,
        )
        sys.exit(74)


if __name__ == "__main__":
    stage_vm_inputs()
