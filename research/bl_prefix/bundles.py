"""Bounded, deterministic bank archives for the supervisor's final-only exports."""

from __future__ import annotations

import hashlib
import io
import json
import tarfile
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import export_bytes, export_json, verify_export

CHUNK_BYTES = 60 * 1024**2
MAX_TOTAL_BYTES = 2 * 1024**3


def pack_bank(bank_dir: Path, out: Path, chunk_bytes: int = CHUNK_BYTES) -> dict[str, Any]:
    """Keep working snapshots outside output and publish a few complete tar chunks."""
    if not 10240 <= chunk_bytes <= CHUNK_BYTES:
        raise ValueError("bank chunk bound must be 10KiB through 60MiB")
    files = sorted(path for path in bank_dir.rglob("*") if path.is_file())
    if not files or any(path.is_symlink() for path in files):
        raise ValueError("bank bundle needs regular files without symlinks")
    total = sum(path.stat().st_size for path in files)
    if total > MAX_TOTAL_BYTES:
        raise ValueError("bank bundle exceeds 2GiB total bound")
    # Every payload's own sidecars are preserved inside the tar for later load_root.
    for path in files:
        if not str(path).endswith((".complete", ".sha256")):
            verify_export(path)
    chunks, inventory, pending, size = [], [], [], 0

    def publish() -> None:
        nonlocal pending, size
        if not pending:
            return
        buffer = io.BytesIO()
        with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as archive:
            for path in pending:
                name = path.relative_to(bank_dir).as_posix()
                payload = path.read_bytes()
                item = tarfile.TarInfo(name)
                item.size, item.mode, item.mtime = len(payload), 0o600, 0
                archive.addfile(item, io.BytesIO(payload))
                inventory.append(
                    {
                        "path": name,
                        "size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        raw = buffer.getvalue()
        if len(raw) > chunk_bytes:
            raise ValueError("tar overhead exceeded the registered chunk bound")
        target = out / f"bank-{len(chunks):03d}.tar"
        exported = export_bytes(target, raw)
        chunks.append({"path": target.name, "size": exported["size"], "sha256": exported["sha256"]})
        pending, size = [], 0

    for path in files:
        padded = 512 + ((path.stat().st_size + 511) // 512) * 512
        if padded + 10240 > chunk_bytes:
            raise ValueError("single bank member exceeds chunk bound")
        if size + padded + 10240 > chunk_bytes:
            publish()
        pending.append(path)
        size += padded
    publish()
    result = {
        "schema": "bl-prefix-bank-bundle/v1",
        "chunks": chunks,
        "members": inventory,
        "payload_bytes": total,
    }
    export_json(out / "bank-bundle.json", result)
    return result


def unpack_bank(index: Path, out: Path) -> Path:
    """Verify all bytes and extract regular relative files into a fresh directory."""
    verify_export(index)
    record = json.loads(index.read_text())
    if record.get("schema") != "bl-prefix-bank-bundle/v1":
        raise ValueError("invalid bank bundle schema")
    if out.exists():
        raise FileExistsError(out)
    expected = {entry["path"]: entry for entry in record["members"]}
    if len(expected) != len(record["members"]):
        raise ValueError("duplicate bundle members")
    seen, total = set(), 0
    # Validate every archive before writing any file.
    contents = []
    for chunk in record["chunks"]:
        relative = Path(chunk["path"])
        if relative.is_absolute() or len(relative.parts) != 1:
            raise ValueError("unsafe archive chunk path")
        path = index.parent / relative
        raw = path.read_bytes()
        if len(raw) != chunk["size"] or len(raw) > CHUNK_BYTES:
            raise ValueError("bundle chunk size mismatch")
        if hashlib.sha256(raw).hexdigest() != chunk["sha256"]:
            raise ValueError("bundle chunk digest mismatch")
        with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
            for member in archive:
                name = Path(member.name)
                if not member.isfile() or name.is_absolute() or ".." in name.parts:
                    raise ValueError("unsafe bank archive member")
                if member.name not in expected or member.name in seen:
                    raise ValueError("unexpected or duplicate bank member")
                bound = expected[member.name]
                if member.size != bound["size"]:
                    raise ValueError("bank member size mismatch")
                total += member.size
                if total > MAX_TOTAL_BYTES:
                    raise ValueError("bank bundle exceeds total bound")
                handle = archive.extractfile(member)
                payload = handle.read() if handle is not None else b""
                if hashlib.sha256(payload).hexdigest() != bound["sha256"]:
                    raise ValueError("bank member digest mismatch")
                contents.append((name, payload))
                seen.add(member.name)
    if seen != set(expected) or total != record["payload_bytes"]:
        raise ValueError("bank bundle member inventory mismatch")
    out.mkdir(parents=True)
    for relative, payload in contents:
        target = out / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as handle:
            handle.write(payload)
    bank = out / "bank.json"
    verify_export(bank)
    return bank
