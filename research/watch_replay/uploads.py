"""Pack small immutable inputs/sidecars into one deterministic archive (<=8 transfers, <=60 s)."""

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
from typing import Any, Callable
from urllib.error import URLError
from urllib.request import urlopen

from research.t34_exit_b8 import public_inputs
from research.watch_replay.run import ROOT
from tools.ops.package_spec import _relative, digest, publish_immutable, verify_immutable

CHUNK_BYTES = 2 * 1024**2
MAX_METADATA_BYTES = 4 * 1024**2
METADATA = "input-metadata.tar.gz"
INDEX = "input-inventory.json"
CHECKPOINT_RESUME_ARCHIVE = "checkpoints-resume.tar.gz"
CHECKPOINT_RESUME_INDEX = "checkpoints-inventory.json"
MAX_CHECKPOINT_RESUME_PARTS = 4

MAX_TRANSFERS = 8
SECONDS_PER_TRANSFER = 5.0
BYTES_PER_SECOND = 1_000_000
MAX_ESTIMATED_TRANSFER_SECONDS = 60.0
DOWNLOAD_ATTEMPTS = 3

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
    "public-inputs.json",
)


def metadata_names() -> set[str]:
    """Exact small payloads and all .sha256/.complete sidecars bundled into METADATA."""
    return {f"{name}{suffix}" for name in SMALL_PAYLOADS for suffix in ("", ".sha256", ".complete")}


def _verify_checkpoint_pins(root_or_catalog: Path) -> None:
    pins = json.loads(
        (ROOT / "web/backend/served_checkpoint_pins.json").read_text(encoding="utf-8")
    )
    if root_or_catalog.is_file():
        rows, _assets = public_inputs._load(root_or_catalog)
        by_name = {str(item["name"]): item for item in rows}
        catalog_map = {
            "b8_s3_t7_20261016.pth": "B8_s3_final.pth",
            "frp3_m3_s12_u60000_20261005.pth": "frp3_m3_s12_u60000_20261005.pth",
        }
        for name, (pin, size) in CHECKPOINTS.items():
            row = by_name[catalog_map[name]]
            if (
                int(row["size_bytes"]) != size
                or row["sha256"] != pins["checkpoints"][pin]["checkpoint_sha256"]
            ):
                raise ValueError(f"public catalog checkpoint differs from released pin: {name}")
        return
    for name, (pin, size) in CHECKPOINTS.items():
        path = root_or_catalog / "checkpoints" / name
        verified = verify_immutable(path)
        if (
            verified["size"] != size
            or verified["sha256"] != pins["checkpoints"][pin]["checkpoint_sha256"]
        ):
            raise ValueError(f"staged checkpoint differs from released pin: {name}")


def _assert_safe_new_target(root: Path, rel_name: str) -> Path:
    _relative(rel_name, "staged member")
    target = root / rel_name
    if target.exists() or target.is_symlink():
        raise ValueError(f"metadata destination already exists or is unsafe: {rel_name}")
    cur = target.parent
    while cur != root:
        if cur.is_symlink():
            raise ValueError(f"metadata parent directory is a symlink: {cur}")
        cur = cur.parent
    return target


def pack_inputs(
    rows: list[dict[str, Any]], archive_path: Path, *, verify_pins: bool = True
) -> list[dict[str, Any]]:
    """Bundle all small immutable payloads and sidecars into one <=2 MiB METADATA archive."""
    expected_small = metadata_names()
    if {row["dest"] for row in rows} != expected_small or len(rows) != len(expected_small):
        raise ValueError("metadata inventory differs from frozen schema")
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


def _download_verified_asset(
    asset: dict[str, Any],
    target: Path,
    opener: Callable[..., Any],
) -> None:
    last_error: BaseException | None = None
    for _attempt in range(1, DOWNLOAD_ATTEMPTS + 1):
        target.unlink(missing_ok=True)
        written = 0
        hasher = hashlib.sha256()
        try:
            with opener(str(asset["url"]), timeout=60) as response, target.open("xb") as stream:
                while block := response.read(public_inputs.CHUNK):
                    written += len(block)
                    if written > int(asset["size_bytes"]):
                        raise public_inputs.PublicInputError(
                            f"release asset exceeds declared size: {asset['name']}"
                        )
                    hasher.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if written != int(asset["size_bytes"]) or hasher.hexdigest() != str(asset["sha256"]):
                raise public_inputs.PublicInputError(f"release asset bytes differ: {asset['name']}")
            return
        except public_inputs.PublicInputError:
            target.unlink(missing_ok=True)
            raise
        except (OSError, URLError) as exc:
            target.unlink(missing_ok=True)
            last_error = exc
    raise public_inputs.PublicInputError(
        f"failed to fetch public asset {asset.get('name')}: {last_error}"
    ) from last_error


def materialize_watch_checkpoints(
    root: Path,
    *,
    verify_pins: bool = True,
    opener: Callable[..., Any] = urlopen,
) -> dict[str, str]:
    """Download public release assets and publish ONLY the 2 Watch6 checkpoints atomically."""
    root = root.resolve(strict=True)
    try:
        import certifi
    except ImportError:
        pass
    else:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    catalog = root / "public-inputs.json"
    verify_immutable(catalog)
    rows, assets = public_inputs._load(catalog)
    by_name = {str(item["name"]): item for item in rows}

    for ckpt_name in CHECKPOINTS:
        for suffix in ("", ".sha256", ".complete"):
            _assert_safe_new_target(root, f"checkpoints/{ckpt_name}{suffix}")

    published: dict[str, str] = {}
    with tempfile.TemporaryDirectory(prefix=".watch-ckpts-", dir=root) as tmp_str:
        tmp = Path(tmp_str)
        b8_row = by_name["B8_s3_final.pth"]
        b8_tmp = tmp / "B8_s3_final.pth"
        _download_verified_asset(assets["B8_s3_final.pth"], b8_tmp, opener)
        if b8_tmp.stat().st_size != int(b8_row["size_bytes"]) or digest(b8_tmp) != str(
            b8_row["sha256"]
        ):
            raise public_inputs.PublicInputError("B8 checkpoint bytes differ from catalog")
        pub_b8 = publish_immutable(
            root / "checkpoints/b8_s3_t7_20261016.pth",
            b8_tmp.read_bytes(),
        )
        published["checkpoints/b8_s3_t7_20261016.pth"] = str(pub_b8["sha256"])

        frp3_name = "frp3_m3_s12_u60000_20261005.pth"
        frp3_row = by_name[frp3_name]
        league_tmp = tmp / "league.tar.gz"
        _download_verified_asset(assets["league.tar.gz"], league_tmp, opener)
        with tarfile.open(league_tmp, "r:gz") as tar:
            members = {
                Path(member.name).name: member for member in tar.getmembers() if member.isfile()
            }
            member = members.get(frp3_name)
            if member is None:
                raise public_inputs.PublicInputError(
                    f"league archive lacks required checkpoint {frp3_name}"
                )
            extracted = tar.extractfile(member)
            if extracted is None:
                raise public_inputs.PublicInputError(f"unreadable league member {frp3_name}")
            ckpt_tmp = tmp / frp3_name
            written = 0
            hasher = hashlib.sha256()
            with extracted, ckpt_tmp.open("xb") as stream:
                while block := extracted.read(public_inputs.CHUNK):
                    written += len(block)
                    if written > int(frp3_row["size_bytes"]):
                        raise public_inputs.PublicInputError(
                            f"league member exceeds declared size: {frp3_name}"
                        )
                    hasher.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if written != int(frp3_row["size_bytes"]) or hasher.hexdigest() != str(
                frp3_row["sha256"]
            ):
                raise public_inputs.PublicInputError(f"checkpoint member bytes differ: {frp3_name}")
            pub_frp3 = publish_immutable(
                root / f"checkpoints/{frp3_name}",
                ckpt_tmp.read_bytes(),
            )
            published[f"checkpoints/{frp3_name}"] = str(pub_frp3["sha256"])

    if verify_pins:
        _verify_checkpoint_pins(root)
    else:
        for ckpt_name in CHECKPOINTS:
            verify_immutable(root / "checkpoints" / ckpt_name)
    return published


def stage_metadata(
    root: Path,
    expected_sha256: str,
    *,
    verify_pins: bool = True,
    materialize_checkpoints: bool = True,
    opener: Callable[..., Any] = urlopen,
) -> None:
    """Verify METADATA archive, unpack create-only with .complete last, materialize checkpoints."""
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
    if materialize_checkpoints:
        materialize_watch_checkpoints(root, verify_pins=verify_pins, opener=opener)


def _encode_checkpoint_tar_gz(
    groups: list[tuple[str, dict[str, bytes], dict[str, dict[str, Any]]]],
) -> bytes:
    payloads: dict[str, bytes] = {}
    inventory: dict[str, dict[str, Any]] = {}
    for _unit_name, unit_payloads, unit_inv in groups:
        payloads.update(unit_payloads)
        inventory.update(unit_inv)
    payloads[CHECKPOINT_RESUME_INDEX] = json.dumps(inventory, sort_keys=True).encode("utf-8")
    buf = io.BytesIO()
    with (
        gzip.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0, compresslevel=9) as compressed,
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
    return buf.getvalue()


def pack_checkpoint_resume(checkpoint_dir: Path, out_dir: Path) -> list[dict[str, Any]]:
    """Bundle verified staged session checkpoints into <=2 MiB deterministic tar.gz part(s)."""
    from research.world_checkpoint.checkpoint import find_latest_checkpoint, load_checkpoint

    checkpoint_dir = checkpoint_dir.resolve(strict=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    unit_dirs = sorted(
        p
        for p in checkpoint_dir.iterdir()
        if p.is_dir() and not p.is_symlink() and len(p.name) == 6 and p.name.isdigit()
    )
    groups: list[tuple[str, dict[str, bytes], dict[str, dict[str, Any]]]] = []
    for unit_dir in unit_dirs:
        latest = find_latest_checkpoint(unit_dir, verify_identity=False)
        if latest is None:
            continue
        verify_immutable(latest)
        load_checkpoint(latest, verify_identity=False)
        unit_payloads: dict[str, bytes] = {}
        unit_inv: dict[str, dict[str, Any]] = {}
        for suffix in ("", ".sha256", ".complete"):
            src = Path(str(latest) + suffix)
            if src.is_symlink() or not src.is_file():
                raise ValueError(f"missing checkpoint sidecar: {src}")
            rel = f"{unit_dir.name}/{src.name}"
            data = src.read_bytes()
            unit_payloads[rel] = data
            unit_inv[rel] = {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        groups.append((unit_dir.name, unit_payloads, unit_inv))

    if not groups:
        return []

    single_bytes = _encode_checkpoint_tar_gz(groups)
    if len(single_bytes) <= CHUNK_BYTES:
        archive_path = out_dir / CHECKPOINT_RESUME_ARCHIVE
        meta = publish_immutable(archive_path, single_bytes)
        return [
            {
                "path": str(archive_path),
                "dest": CHECKPOINT_RESUME_ARCHIVE,
                "sha256": str(meta["sha256"]),
                "size_bytes": int(meta["size"]),
            }
        ]

    parts: list[list[tuple[str, dict[str, bytes], dict[str, dict[str, Any]]]]] = []
    current: list[tuple[str, dict[str, bytes], dict[str, dict[str, Any]]]] = []
    for group in groups:
        candidate = [*current, group]
        if len(_encode_checkpoint_tar_gz(candidate)) <= CHUNK_BYTES:
            current = candidate
        else:
            if not current:
                raise ValueError(f"single session checkpoint exceeds 2 MiB chunk bound: {group[0]}")
            parts.append(current)
            current = [group]
            if len(_encode_checkpoint_tar_gz(current)) > CHUNK_BYTES:
                raise ValueError(f"single session checkpoint exceeds 2 MiB chunk bound: {group[0]}")
    if current:
        parts.append(current)
    if len(parts) > MAX_CHECKPOINT_RESUME_PARTS:
        raise ValueError(
            f"checkpoint resume requires {len(parts)} parts (max {MAX_CHECKPOINT_RESUME_PARTS})"
        )

    rows: list[dict[str, Any]] = []
    for idx, part_groups in enumerate(parts):
        dest_name = f"checkpoints-resume-part{idx:02d}.tar.gz"
        archive_path = out_dir / dest_name
        part_bytes = _encode_checkpoint_tar_gz(part_groups)
        meta = publish_immutable(archive_path, part_bytes)
        rows.append(
            {
                "path": str(archive_path),
                "dest": dest_name,
                "sha256": str(meta["sha256"]),
                "size_bytes": int(meta["size"]),
            }
        )
    return rows


def stage_checkpoint_resume(root: Path, expected_shas_json: str = "") -> Path | None:
    """Verify and unpack uploaded checkpoint-resume archive(s) into root/checkpoints-resume."""
    root = root.resolve(strict=True)
    expected_shas: dict[str, str] = (
        json.loads(expected_shas_json) if expected_shas_json.strip() else {}
    )
    if expected_shas:
        archive_names = sorted(expected_shas)
    else:
        archive_names = sorted(
            p.name
            for p in root.glob("checkpoints-resume*.tar.gz")
            if p.is_file() and not p.is_symlink()
        )
    if not archive_names:
        return None

    dest_root = root / "checkpoints-resume"
    if dest_root.exists() or dest_root.is_symlink():
        raise ValueError("checkpoints-resume destination already exists or is unsafe")
    dest_root.mkdir(parents=True, exist_ok=False)

    all_payloads: dict[str, bytes] = {}
    for archive_name in archive_names:
        if archive_name != CHECKPOINT_RESUME_ARCHIVE and not (
            archive_name.startswith("checkpoints-resume-part") and archive_name.endswith(".tar.gz")
        ):
            raise ValueError(f"unexpected checkpoint resume archive name: {archive_name}")
        path = root / archive_name
        if (
            path.is_symlink()
            or not path.is_file()
            or path.stat().st_size > CHUNK_BYTES
            or (archive_name in expected_shas and digest(path) != expected_shas[archive_name])
        ):
            raise ValueError(f"checkpoint resume archive SHA256 or size mismatch: {archive_name}")
        with tarfile.open(path, "r:gz") as archive:
            members = archive.getmembers()
            names = [member.name for member in members]
            if (
                len(names) != len(set(names))
                or CHECKPOINT_RESUME_INDEX not in names
                or any(not member.isfile() or member.size < 0 for member in members)
                or sum(member.size for member in members) > MAX_METADATA_BYTES * 4
            ):
                raise ValueError(f"unsafe or malformed checkpoint resume archive: {archive_name}")
            part_payloads: dict[str, bytes] = {}
            for member in members:
                extracted = archive.extractfile(member)
                if extracted is None:
                    raise ValueError(f"missing checkpoint tar member stream: {member.name}")
                part_payloads[member.name] = extracted.read()
        inventory = json.loads(part_payloads.pop(CHECKPOINT_RESUME_INDEX).decode("utf-8"))
        if set(inventory) != set(part_payloads):
            raise ValueError(f"checkpoint resume inventory mismatch: {archive_name}")
        for name, data in part_payloads.items():
            if name in all_payloads:
                raise ValueError(f"duplicate checkpoint resume member across parts: {name}")
            if inventory[name] != {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}:
                raise ValueError(f"checkpoint resume member digest/size mismatch: {name}")
            _assert_safe_new_target(dest_root, name)
            all_payloads[name] = data

    with tempfile.TemporaryDirectory(prefix=".ckpt-resume-", dir=dest_root) as temporary:
        ordered = sorted(all_payloads, key=lambda item: (item.endswith(".complete"), item))
        for index, name in enumerate(ordered):
            staged = Path(temporary) / str(index)
            staged.write_bytes(all_payloads[name])
            staged.chmod(0o444)
            target = dest_root / name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, target)

    ckpt_files = sorted(dest_root.glob("*/*.ckpt"))
    if not ckpt_files:
        raise ValueError("checkpoint resume archive contained zero .ckpt payloads")
    for ckpt_file in ckpt_files:
        verify_immutable(ckpt_file)
    return dest_root


def _serial_rpcs_for_size(size: int) -> int:
    """Count effective serial RPC waves (2 parallel chunk workers + 1 assemble RPC if >2 MiB)."""
    if size <= CHUNK_BYTES:
        return 1
    chunks = math.ceil(size / CHUNK_BYTES)
    return math.ceil(chunks / 2) + 1


def check_uploads(manifest: dict[str, Any]) -> dict[str, Any]:
    """Enforce <= 8 transfers, <= 2 MiB per local input, and <= 60 s offline forecast."""
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
            p = Path(item["path"])
            if p.is_symlink() or not p.is_file():
                raise ValueError(f"missing or symlinked input path: {p}")
            check("inputs/" + item["dest"], p.stat().st_size, local_input=True)

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
        "scope": "offline forecast (normal + sparse fallback) under 5 s/RPC-wave + 1 MB/s model",
    }
