"""Bounded checkpoint inputs and offline parity with broker upload size guards."""

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

from research.colab_cli import source, transfer_policy
from research.watch6_search_efficacy import protocol as p
from tools.ops.package_spec import Input, digest, verify_immutable

PART_BYTES = 2 * 1024**2
METADATA = "input-metadata.tar.gz"
INDEX = "input-inventory.json"
MAX_METADATA_BYTES = 4 * 1024**2
MAX_TRANSFERS = 20
SECONDS_PER_TRANSFER = 3.0
BYTES_PER_SECOND = 512 * 1024
CONTROL_MARGIN_SECONDS = 30.0
CHECKPOINTS = {
    "b8_s3_t7_20261016.pth": ("b8-s3", 6_562_005),
    "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12", 6_906_247),
}


def metadata_names() -> set[str]:
    """Keep the archive restricted to the existing small payloads and sidecars."""
    names = {
        name + suffix
        for name in (
            "runtime-contract.json",
            "verified-ledger.json",
            "judge-approval.json",
            "mac-mirror.json",
            "release-evidence.tar.gz",
        )
        for suffix in ("", ".sha256", ".complete")
    }
    names.update(
        f"checkpoint-parts/{name}.{index:03d}.part{suffix}"
        for name, (_, size) in CHECKPOINTS.items()
        for index in range((size + PART_BYTES - 1) // PART_BYTES)
        for suffix in (".sha256", ".complete")
    )
    return names


def pack_inputs(rows: list[dict], archive_path: Path) -> list[dict]:
    """Bundle small immutable files; retain each 2 MiB part as one input upload."""
    small = [row for row in rows if row["dest"] in metadata_names()]
    parts = [row for row in rows if row not in small]
    if {row["dest"] for row in small} != metadata_names() or len(small) != len(metadata_names()):
        raise ValueError("metadata inventory differs")
    inventory = {row["dest"]: {"sha256": row["sha256"], "size": row["size_bytes"]} for row in small}
    payloads = {row["dest"]: Path(row["path"]).read_bytes() for row in small}
    payloads[INDEX] = json.dumps(inventory, sort_keys=True).encode()
    if sum(map(len, payloads.values())) > MAX_METADATA_BYTES:
        raise ValueError("metadata exceeds unpacked size bound")
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for name, data in sorted(payloads.items()):
            member = tarfile.TarInfo(name)
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    return parts + [{"path": str(archive_path), "dest": METADATA, "sha256": digest(archive_path)}]


def stage_metadata(root: Path, expected_sha256: str) -> None:
    """Verify the manifest-bound archive and every member before create-only publication."""
    root = root.resolve(strict=True)
    path = root / METADATA
    if (
        path.is_symlink()
        or path.stat().st_size > MAX_METADATA_BYTES
        or digest(path) != expected_sha256
    ):
        raise ValueError("metadata archive SHA256 mismatch")
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or set(names) != metadata_names() | {INDEX}
            or any(not member.isfile() or member.size < 0 for member in members)
            or sum(member.size for member in members) > MAX_METADATA_BYTES
        ):
            raise ValueError("unsafe or different metadata archive inventory")
        payloads = {member.name: archive.extractfile(member).read() for member in members}
    inventory = json.loads(payloads.pop(INDEX))
    if set(inventory) != metadata_names():
        raise ValueError("metadata index inventory differs")
    for name, data in payloads.items():
        if inventory[name] != {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}:
            raise ValueError(f"metadata member digest/size mismatch: {name}")
        target = root / name
        if (
            target.exists()
            or target.is_symlink()
            or any(parent.is_symlink() for parent in target.parents)
        ):
            raise ValueError(f"metadata destination exists or is unsafe: {name}")
    # All bytes are checked first. Each payload/sidecar is atomic, with markers last.
    with tempfile.TemporaryDirectory(prefix=".metadata-", dir=root) as temporary:
        for index, name in enumerate(sorted(payloads, key=lambda name: name.endswith(".complete"))):
            staged = Path(temporary) / str(index)
            staged.write_bytes(payloads[name])
            staged.chmod(0o444)
            target = root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, target)


def checkpoint_inputs(saved: Path, directory: Path) -> list[Input]:
    """Split exact released bytes without reserializing or changing checkpoint pins."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    directory.mkdir(parents=True, exist_ok=False)
    inputs = []
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
    """Verify each immutable part, then publish only the full pinned checkpoint."""
    pins = json.loads((p.ROOT / "web/backend/served_checkpoint_pins.json").read_text())
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
                        raise ValueError("checkpoint part size differs")
                    stream.write(part.read_bytes())
                stream.flush()
                os.fsync(stream.fileno())
                if (
                    temporary.stat().st_size != size
                    or digest(temporary) != pins["checkpoints"][pin]["checkpoint_sha256"]
                ):
                    raise ValueError("reassembled checkpoint digest differs from released pin")
                temporary.chmod(0o444)
                os.link(temporary, output / name)  # atomic, create-only publication
            finally:
                temporary.unlink(missing_ok=True)


def check_uploads(manifest: dict) -> dict:
    """Check every possible upload using the broker's unchanged per-file bound.

    The broker uses max_export_bytes for private uploads and staging files too.
    Build its actual sparse source plan/archive, including fallback, offline;
    include all four commit-pinned runtime controls and every input sidecar.
    """
    limit = manifest["max_export_bytes"]
    rows = []

    def check(name: str, size: int) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest["inputs"]:
        if "path" in item:
            check("inputs/" + item["dest"], Path(item["path"]).stat().st_size)
    with tempfile.TemporaryDirectory(prefix="watch6-upload-preflight-") as temporary:
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
    # Match supervisor._start's stage_upload calls, not all checked staging files:
    # runtime.py/release.py and source-plan travel embedded in control RPCs.
    # Include the optional sparse fallback, even on the normal GitHub path.
    transfers = [
        row
        for row in rows
        if row["dest"].startswith("inputs/")
        or row["dest"] in ("bootstrap.sh", "source.py", "source.tar.gz")
    ]
    count = len(transfers)
    total_bytes = sum(row["size_bytes"] for row in transfers)
    estimated = count * SECONDS_PER_TRANSFER + total_bytes / BYTES_PER_SECOND
    estimated += CONTROL_MARGIN_SECONDS
    transfer_cap = transfer_policy.transfer_limit(manifest)
    staging_estimated = estimated + source.FETCH_TIMEOUT_SECONDS
    if (
        count > MAX_TRANSFERS
        or estimated >= transfer_cap
        or staging_estimated >= transfer_policy.MAX_STAGING_SECONDS
    ):
        raise ValueError(
            f"upload transfer forecast refuses offline: {count} transfers (max {MAX_TRANSFERS}), "
            f"{estimated:.3f}s vs {transfer_cap:g}s transfer cap; "
            f"{staging_estimated:.3f}s staging forecast"
        )
    return {
        "status": "PASS",
        "max_upload_bytes": limit,
        "files": rows,
        "transfer_count": count,
        "normal_transfer_count": count - 1,
        "transfer_bytes": total_bytes,
        "seconds_per_transfer": SECONDS_PER_TRANSFER,
        "bytes_per_second": BYTES_PER_SECOND,
        "control_margin_seconds": CONTROL_MARGIN_SECONDS,
        "estimated_transfer_seconds": estimated,
        "transfer_cap_seconds": transfer_cap,
        "estimated_staging_seconds": staging_estimated,
        "staging_cap_seconds": transfer_policy.MAX_STAGING_SECONDS,
        "scope": "offline forecast including fallback; live staging qualification still required",
        "credential_uploads_before_staging_clock": 1,
    }
