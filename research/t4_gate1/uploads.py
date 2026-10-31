"""Packed metadata archive, slim 2-checkpoint staging, and <=8 transfer / <=60 s upload budget gate.

Packs all Gate-1 small files (``canary-config.json``, ``main-config.json``,
``preflight-config.json``, ``seed-manifest.json``, ``verified-ledger.json``,
``public-inputs.json``) plus all ``.sha256`` / ``.complete`` sidecars into one deterministic
``input-metadata.tar.gz`` (< 30 KB), keeping total ContentsClient transfers at:
  - 3 transfers on normal path (``bootstrap.sh``, ``source.py``, ``inputs/input-metadata.tar.gz``),
  - 4 transfers on sparse-source fallback (``+ source.tar.gz``),
well under the ``<= 8`` transfer and ``<= 60.0 s`` estimated transfer time bounds.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import os
import subprocess
import tarfile
import tempfile
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence
from urllib.error import URLError
from urllib.request import urlopen

from research.colab_cli import source, transfer_policy
from research.t4_gate1 import B8_S3_SHA256, FRP3_S12_SHA256
from research.t34_exit_b8.public_inputs import CHUNK as DOWNLOAD_CHUNK
from research.t34_exit_b8.public_inputs import PublicInputError
from research.t34_exit_b8.public_inputs import _load as load_public_catalog
from tools.ops.package_spec import (
    _relative,
    digest,
    publish_immutable,
    verify_immutable,
)

ROOT = Path(__file__).resolve().parents[2]

PART_BYTES = 2 * 1024**2
MAX_FILE_BYTES = 4 * 1024**2
MAX_METADATA_BYTES = MAX_FILE_BYTES
METADATA = "input-metadata.tar.gz"
INDEX = "input-inventory.json"
INVENTORY_SCHEMA = "t4-gate1-packed-input-inventory/v1"

MAX_TRANSFERS = 8
SECONDS_PER_TRANSFER = 6.0
BYTES_PER_SECOND = 800_000
MAX_ESTIMATED_TRANSFER_SECONDS = 60.0

BASE_PAYLOADS: frozenset[str] = frozenset(
    {
        "canary-config.json",
        "canary-ledger.json",
        "main-config.json",
        "preflight-config.json",
        "preflight-ledger.json",
        "seed-manifest.json",
        "verified-ledger.json",
        "public-inputs.json",
    }
)

# Only the 2 checkpoints used by Watch6-v2 harvest/branching (B8 s3 hero + FRP3-s12 opponents).
REQUIRED_B8_CHECKPOINT: tuple[str, int, str] = ("b8-s3.pth", 6_562_005, B8_S3_SHA256)
REQUIRED_LEAGUE_CHECKPOINTS: dict[str, tuple[str, int, str]] = {
    "frp3_m3_s12_u60000_20261005.pth": ("frp3-s12.pth", 6_906_247, FRP3_S12_SHA256),
}
LOCAL_B8_FILENAME = "b8_s3_t7_20261016.pth"
DOWNLOAD_ATTEMPTS = 3


def pack_inputs(
    rows: Sequence[Mapping[str, Any]],
    archive_path: Path,
    *,
    parts_dir: Path | None = None,
    expected_payloads: set[str] | frozenset[str] | None = None,
) -> list[dict[str, Any]]:
    """Bundle ``<= 2 MiB`` inputs + sidecars into one deterministic ``input-metadata.tar.gz``."""
    expected = set(BASE_PAYLOADS if expected_payloads is None else expected_payloads)
    if not expected:
        raise ValueError("expected_payloads must be nonempty")
    for name in expected:
        _relative(name, "payload destination")
        if name.endswith((".sha256", ".complete")) or name in {METADATA, INDEX}:
            raise ValueError(f"reserved or sidecar payload name: {name}")

    by_dest: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        dest = _relative(str(row["dest"]), "staged destination")
        if dest in by_dest:
            raise ValueError(f"duplicate staged input destination: {dest}")
        by_dest[dest] = row

    expected_all = {
        f"{name}{suffix}" for name in expected for suffix in ("", ".sha256", ".complete")
    }
    if set(by_dest) != expected_all:
        raise ValueError("staged input inventory differs from expected Gate-1 payload schema")

    tar_payloads: dict[str, bytes] = {}
    direct_part_rows: list[dict[str, Any]] = []
    splits: dict[str, dict[str, Any]] = {}
    resolved_parts_dir = (
        parts_dir if parts_dir is not None else archive_path.parent / f"{archive_path.name}.parts"
    )

    for name in sorted(expected):
        main_row = by_dest[name]
        main_path = Path(str(main_row["path"]))
        verified = verify_immutable(main_path)
        size = int(verified["size"])
        sha = str(verified["sha256"])
        if size != int(main_row["size_bytes"]) or sha != str(main_row["sha256"]):
            raise ValueError(f"staged payload metadata mismatch: {name}")

        for suffix in (".sha256", ".complete"):
            sc_dest = f"{name}{suffix}"
            sc_bytes = Path(str(by_dest[sc_dest]["path"])).read_bytes()
            if len(sc_bytes) != int(by_dest[sc_dest]["size_bytes"]) or hashlib.sha256(
                sc_bytes
            ).hexdigest() != str(by_dest[sc_dest]["sha256"]):
                raise ValueError(f"staged sidecar digest/size mismatch: {sc_dest}")
            tar_payloads[sc_dest] = sc_bytes

        if size <= PART_BYTES:
            tar_payloads[name] = main_path.read_bytes()
        else:
            part_specs: list[dict[str, Any]] = []
            num_parts = (size + PART_BYTES - 1) // PART_BYTES
            with main_path.open("rb") as stream:
                for idx in range(num_parts):
                    chunk = stream.read(PART_BYTES)
                    part_rel = _relative(f"parts/{name}.{idx:03d}.part", "part destination")
                    part_path = resolved_parts_dir / part_rel
                    pub = publish_immutable(part_path, chunk)
                    direct_part_rows.append(
                        {
                            "path": str(part_path),
                            "dest": part_rel,
                            "sha256": pub["sha256"],
                            "size_bytes": int(pub["size"]),
                        }
                    )
                    part_specs.append(
                        {
                            "dest": part_rel,
                            "sha256": pub["sha256"],
                            "size": int(pub["size"]),
                        }
                    )
                    for suffix in (".sha256", ".complete"):
                        sc_file = Path(f"{part_path}{suffix}")
                        tar_payloads[f"{part_rel}{suffix}"] = sc_file.read_bytes()
            splits[name] = {
                "sha256": sha,
                "size": size,
                "parts": part_specs,
            }

    members_meta = {
        dest: {"sha256": hashlib.sha256(data).hexdigest(), "size": len(data)}
        for dest, data in sorted(tar_payloads.items())
    }
    inventory = {
        "schema": INVENTORY_SCHEMA,
        "expected_payloads": sorted(expected),
        "members": members_meta,
        "splits": splits,
    }
    tar_payloads[INDEX] = (json.dumps(inventory, sort_keys=True) + "\n").encode("utf-8")
    if sum(map(len, tar_payloads.values())) > MAX_METADATA_BYTES:
        raise ValueError("metadata archive exceeds 4 MiB unpacked size bound")

    archive_path.parent.mkdir(parents=True, exist_ok=True)
    with (
        archive_path.open("xb") as raw,
        gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
        tarfile.open(fileobj=compressed, mode="w") as archive,
    ):
        for dest, data in sorted(tar_payloads.items()):
            member = tarfile.TarInfo(dest)
            member.size = len(data)
            member.mtime = 0
            member.mode = 0o444
            member.uid = 0
            member.gid = 0
            member.uname = ""
            member.gname = ""
            archive.addfile(member, io.BytesIO(data))

    archive_size = archive_path.stat().st_size
    if archive_size > PART_BYTES:
        raise ValueError(
            f"compressed metadata archive exceeds 2 MiB single-transfer bound: {archive_size}"
        )
    return direct_part_rows + [
        {
            "path": str(archive_path),
            "dest": METADATA,
            "sha256": digest(archive_path),
            "size_bytes": archive_size,
        }
    ]


def _assert_safe_new_target(root: Path, rel_name: str) -> Path:
    _relative(rel_name, "staged member")
    target = root / rel_name
    if target.exists() or target.is_symlink():
        raise ValueError(f"metadata destination already exists or is unsafe: {rel_name}")
    cur = target.parent
    while cur != root:
        if cur.is_symlink():
            raise ValueError(f"metadata parent directory is a symlink: {cur}")
        if not cur.resolve().is_relative_to(root):
            raise ValueError(f"metadata parent directory escapes root: {cur}")
        cur = cur.parent
    return target


def stage_metadata(
    root: Path,
    expected_sha256: str,
    *,
    expected_payloads: set[str] | frozenset[str] | None = None,
) -> dict[str, Any]:
    """Verify manifest-bound archive, reassemble any >2 MiB parts, publish .complete last."""
    root = root.resolve(strict=True)
    path = root / METADATA
    if (
        path.is_symlink()
        or not path.is_file()
        or path.stat().st_size > PART_BYTES
        or digest(path) != expected_sha256
    ):
        raise ValueError("metadata archive SHA256 or size mismatch")

    expected = set(BASE_PAYLOADS if expected_payloads is None else expected_payloads)
    with tarfile.open(path, "r:gz") as archive:
        members = archive.getmembers()
        names = [member.name for member in members]
        if (
            len(names) != len(set(names))
            or INDEX not in names
            or any(not member.isfile() or member.size < 0 for member in members)
            or sum(member.size for member in members) > MAX_METADATA_BYTES
        ):
            raise ValueError("unsafe or unexpected metadata archive inventory")
        payloads: dict[str, bytes] = {}
        for member in members:
            if member.name != INDEX:
                _relative(member.name, "tar member")
            extracted = archive.extractfile(member)
            if extracted is None:
                raise ValueError(f"missing tar member stream: {member.name}")
            data = extracted.read()
            if len(data) != member.size:
                raise ValueError(f"tar member size mismatch: {member.name}")
            payloads[member.name] = data

    inventory = json.loads(payloads.pop(INDEX).decode("utf-8"))
    if (
        not isinstance(inventory, dict)
        or inventory.get("schema") != INVENTORY_SCHEMA
        or set(inventory.get("expected_payloads", [])) != expected
    ):
        raise ValueError("metadata index schema or expected payload set differs")

    declared_members = inventory.get("members")
    splits = inventory.get("splits")
    if (
        not isinstance(declared_members, dict)
        or not isinstance(splits, dict)
        or set(payloads) != set(declared_members)
        or not set(splits).issubset(expected)
    ):
        raise ValueError("metadata index inventory differs")

    expected_tar_names: set[str] = set()
    for name in expected:
        expected_tar_names.add(f"{name}.sha256")
        expected_tar_names.add(f"{name}.complete")
        if name not in splits:
            expected_tar_names.add(name)
        else:
            spec = splits[name]
            if (
                not isinstance(spec, dict)
                or not isinstance(spec.get("parts"), list)
                or not spec["parts"]
                or type(spec.get("size")) is not int
                or spec["size"] <= PART_BYTES
            ):
                raise ValueError(f"invalid split descriptor for {name}")
            for part_spec in spec["parts"]:
                part_dest = _relative(str(part_spec.get("dest", "")), "part destination")
                expected_tar_names.add(f"{part_dest}.sha256")
                expected_tar_names.add(f"{part_dest}.complete")

    if set(payloads) != expected_tar_names:
        raise ValueError("metadata member set differs from frozen payload/split schema")

    for name, data in payloads.items():
        if declared_members[name] != {
            "sha256": hashlib.sha256(data).hexdigest(),
            "size": len(data),
        }:
            raise ValueError(f"metadata member digest/size mismatch: {name}")
        _assert_safe_new_target(root, name)

    for split_name in splits:
        _assert_safe_new_target(root, split_name)

    part_complete_names = {
        f"{part_spec['dest']}.complete" for spec in splits.values() for part_spec in spec["parts"]
    }
    first_wave = sorted(n for n in payloads if not n.endswith(".complete"))
    part_completes = sorted(n for n in payloads if n in part_complete_names)
    final_completes = sorted(
        n for n in payloads if n.endswith(".complete") and n not in part_complete_names
    )

    with tempfile.TemporaryDirectory(prefix=".t4-metadata-", dir=root) as temporary:
        tmp_dir = Path(temporary)
        counter = 0

        def _link_bytes(rel_dest: str, raw_bytes: bytes) -> None:
            nonlocal counter
            staged = tmp_dir / f"m-{counter}"
            counter += 1
            staged.write_bytes(raw_bytes)
            staged.chmod(0o444)
            target = root / rel_dest
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(staged, target)

        for name in first_wave:
            _link_bytes(name, payloads[name])
        for name in part_completes:
            _link_bytes(name, payloads[name])

        for split_name in sorted(splits):
            spec = splits[split_name]
            assembled = tmp_dir / f"assembled-{counter}"
            counter += 1
            with assembled.open("xb") as out_stream:
                for idx, part_spec in enumerate(spec["parts"]):
                    part_dest = str(part_spec["dest"])
                    part_path = root / part_dest
                    if not part_path.resolve().is_relative_to(root):
                        raise ValueError(f"split part escapes root: {part_dest}")
                    verified_part = verify_immutable(part_path)
                    expected_part_size = min(PART_BYTES, int(spec["size"]) - idx * PART_BYTES)
                    if (
                        verified_part["size"] != expected_part_size
                        or verified_part["size"] != int(part_spec["size"])
                        or verified_part["sha256"] != str(part_spec["sha256"])
                    ):
                        raise ValueError(f"split part digest/size mismatch: {part_dest}")
                    out_stream.write(part_path.read_bytes())
                out_stream.flush()
                os.fsync(out_stream.fileno())
            if assembled.stat().st_size != int(spec["size"]) or digest(assembled) != str(
                spec["sha256"]
            ):
                raise ValueError(f"reassembled payload digest/size mismatch: {split_name}")
            assembled.chmod(0o444)
            target = root / split_name
            target.parent.mkdir(parents=True, exist_ok=True)
            os.link(assembled, target)

        for name in final_completes:
            _link_bytes(name, payloads[name])

    for name in sorted(expected):
        verify_immutable(root / name)

    return {
        "staged_payloads": sorted(expected),
        "split_payloads": sorted(splits),
    }


def _download_verified_asset(
    asset: Mapping[str, Any],
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
                while block := response.read(DOWNLOAD_CHUNK):
                    written += len(block)
                    if written > int(asset["size_bytes"]):
                        raise PublicInputError(
                            f"release asset exceeds declared size: {asset['name']}"
                        )
                    hasher.update(block)
                    stream.write(block)
                stream.flush()
                os.fsync(stream.fileno())
            if written != int(asset["size_bytes"]) or hasher.hexdigest() != str(asset["sha256"]):
                raise PublicInputError(f"release asset bytes differ: {asset['name']}")
            return
        except PublicInputError:
            target.unlink(missing_ok=True)
            raise
        except (OSError, URLError) as exc:
            target.unlink(missing_ok=True)
            last_error = exc
    raise PublicInputError(
        f"failed to fetch public asset {asset.get('name')}: {last_error}"
    ) from last_error


def materialize_gate1_checkpoints(
    input_dir: Path,
    *,
    opener: Callable[..., Any] = urlopen,
    local_checkpoint_dir: Path | None = None,
) -> dict[str, str]:
    """Materialize ONLY the 2 checkpoints Gate 1 uses (``b8-s3.pth`` and ``frp3-s12.pth``)."""
    root = input_dir.resolve(strict=True)
    catalog = root / "public-inputs.json"
    verify_immutable(catalog)
    _rows, assets = load_public_catalog(catalog)

    b8_alias, b8_size, b8_sha = REQUIRED_B8_CHECKPOINT
    all_aliases = [b8_alias, *(alias for alias, _, _ in REQUIRED_LEAGUE_CHECKPOINTS.values())]
    for alias in all_aliases:
        _assert_safe_new_target(root, alias)

    if local_checkpoint_dir is not None:
        src_root = local_checkpoint_dir.resolve(strict=True)
        published: dict[str, str] = {}
        with tempfile.TemporaryDirectory(prefix=".t4-local-ckpts-", dir=root) as tmp_s:
            tmp = Path(tmp_s)
            local_pairs = [
                (src_root / LOCAL_B8_FILENAME, b8_alias, b8_size, b8_sha),
                *(
                    (src_root / member_name, alias, exp_size, exp_sha)
                    for member_name, (alias, exp_size, exp_sha) in sorted(
                        REQUIRED_LEAGUE_CHECKPOINTS.items()
                    )
                ),
            ]
            for src_path, alias, exp_size, exp_sha in local_pairs:
                if not src_path.is_file():
                    raise FileNotFoundError(f"Missing local checkpoint {src_path}")
                data = src_path.read_bytes()
                if len(data) != exp_size or hashlib.sha256(data).hexdigest() != exp_sha:
                    raise PublicInputError(f"Local checkpoint digest/size mismatch: {src_path}")
                staged_tmp = tmp / alias
                staged_tmp.write_bytes(data)
                staged_tmp.chmod(0o444)
                os.link(staged_tmp, root / alias)
                published[alias] = exp_sha
        return published

    try:
        import certifi
    except ImportError:
        pass
    else:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    published = {}
    with tempfile.TemporaryDirectory(prefix=".t4-ckpts-", dir=root) as tmp_str:
        tmp = Path(tmp_str)
        b8_tmp = tmp / "B8_s3_final.pth"
        _download_verified_asset(assets["B8_s3_final.pth"], b8_tmp, opener)
        if b8_tmp.stat().st_size != b8_size or digest(b8_tmp) != b8_sha:
            raise PublicInputError("B8 checkpoint bytes differ from Gate-1 pin")
        b8_tmp.chmod(0o444)
        os.link(b8_tmp, root / b8_alias)
        published[b8_alias] = b8_sha

        league_tmp = tmp / "league.tar.gz"
        _download_verified_asset(assets["league.tar.gz"], league_tmp, opener)
        with tarfile.open(league_tmp, "r:gz") as tar:
            members = {
                Path(member.name).name: member for member in tar.getmembers() if member.isfile()
            }
            for member_name, (alias, expected_size, expected_sha) in sorted(
                REQUIRED_LEAGUE_CHECKPOINTS.items()
            ):
                member = members.get(member_name)
                if member is None:
                    raise PublicInputError(
                        f"league archive lacks required checkpoint {member_name}"
                    )
                extracted = tar.extractfile(member)
                if extracted is None:
                    raise PublicInputError(f"unreadable league member {member_name}")
                ckpt_tmp = tmp / alias
                written = 0
                hasher = hashlib.sha256()
                with extracted, ckpt_tmp.open("xb") as stream:
                    while block := extracted.read(DOWNLOAD_CHUNK):
                        written += len(block)
                        if written > expected_size:
                            raise PublicInputError(
                                f"league member exceeds declared size: {member_name}"
                            )
                        hasher.update(block)
                        stream.write(block)
                    stream.flush()
                    os.fsync(stream.fileno())
                if written != expected_size or hasher.hexdigest() != expected_sha:
                    raise PublicInputError(f"checkpoint member bytes differ: {member_name}")
                ckpt_tmp.chmod(0o444)
                os.link(ckpt_tmp, root / alias)
                published[alias] = expected_sha

    return published


def check_uploads(manifest: Mapping[str, Any], *, repo: Path = ROOT) -> dict[str, Any]:
    """Enforce <= 2 MiB per local input, <= 8 transfers, and <= 60 s at 6 s/RPC + 0.8 MB/s."""
    manifest_dict = dict(manifest)
    limit = int(manifest_dict["max_export_bytes"])
    if limit <= 0 or limit > MAX_FILE_BYTES:
        raise ValueError(f"manifest max_export_bytes must be in 1..{MAX_FILE_BYTES}, got {limit}")
    rows: list[dict[str, Any]] = []

    def check(name: str, size: int, *, local_input: bool = False) -> None:
        if size > limit:
            raise ValueError(f"upload exceeds bounded transfer size: {name}: {size} > {limit}")
        if local_input and size > PART_BYTES:
            raise ValueError(
                f"unsplit local input exceeds 2 MiB single-chunk bound: "
                f"{name}: {size} > {PART_BYTES}"
            )
        rows.append({"dest": name, "size_bytes": size})

    for item in manifest_dict["inputs"]:
        if "path" in item:
            p = Path(str(item["path"]))
            if p.is_symlink() or not p.is_file():
                raise ValueError(f"missing or symlinked input path: {p}")
            check("inputs/" + str(item["dest"]), p.stat().st_size, local_input=True)

    with tempfile.TemporaryDirectory(prefix="t4-gate1-upload-preflight-") as temporary:
        staging = Path(temporary)
        source.prepare(repo, manifest_dict, staging)
        for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
            data = subprocess.check_output(
                ["git", "show", f"{manifest_dict['commit']}:research/colab_cli/{name}"],
                cwd=repo,
                timeout=10,
            )
            (staging / name).write_bytes(data)
        for path in sorted(staging.iterdir()):
            check(path.name, path.stat().st_size, local_input=False)

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

    transfer_cap = transfer_policy.transfer_limit(manifest_dict)
    staging_estimated = estimated + source.FETCH_TIMEOUT_SECONDS
    if (
        normal_count > MAX_TRANSFERS
        or count > MAX_TRANSFERS
        or normal_estimated > MAX_ESTIMATED_TRANSFER_SECONDS
        or estimated > MAX_ESTIMATED_TRANSFER_SECONDS
        or estimated >= transfer_cap
        or staging_estimated >= transfer_policy.MAX_STAGING_SECONDS
    ):
        raise ValueError(
            f"Gate-1 upload transfer forecast refuses offline: {count} fallback transfers "
            f"(normal {normal_count}, max {MAX_TRANSFERS}), "
            f"{estimated:.3f}s fallback / {normal_estimated:.3f}s normal "
            f"(budget {MAX_ESTIMATED_TRANSFER_SECONDS:g}s, cap {transfer_cap:g}s); "
            f"{staging_estimated:.3f}s staging forecast"
        )

    return {
        "status": "PASS",
        "max_upload_bytes": limit,
        "max_input_part_bytes": PART_BYTES,
        "files": rows,
        "normal_transfer_count": normal_count,
        "normal_transfer_bytes": normal_bytes,
        "normal_estimated_transfer_seconds": round(normal_estimated, 4),
        "transfer_count": count,
        "transfer_bytes": total_bytes,
        "estimated_transfer_seconds": round(estimated, 4),
        "seconds_per_transfer": SECONDS_PER_TRANSFER,
        "bytes_per_second": BYTES_PER_SECOND,
        "max_transfers_budget": MAX_TRANSFERS,
        "max_estimated_transfer_seconds_budget": MAX_ESTIMATED_TRANSFER_SECONDS,
        "transfer_cap_seconds": transfer_cap,
        "estimated_staging_seconds": round(staging_estimated, 4),
        "staging_cap_seconds": transfer_policy.MAX_STAGING_SECONDS,
        "scope": "offline forecast (normal + sparse fallback) under 6 s/RPC + 0.8 MB/s model",
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_stage = sub.add_parser("stage")
    p_stage.add_argument("--input-dir", type=Path, required=True)
    p_stage.add_argument("--metadata-sha256", type=str, required=True)
    p_stage.add_argument("--local-checkpoint-dir", type=Path, default=None)
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.cmd == "stage":
        staged = stage_metadata(args.input_dir, args.metadata_sha256)
        ckpts = materialize_gate1_checkpoints(
            args.input_dir,
            local_checkpoint_dir=args.local_checkpoint_dir,
        )
        print(
            json.dumps(
                {
                    "staged_payloads": staged["staged_payloads"],
                    "split_payloads": staged["split_payloads"],
                    "checkpoints": ckpts,
                },
                sort_keys=True,
            ),
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
