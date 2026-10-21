"""Select E3 roots from the unsealed, hash-bound T3 provenance before targets."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import tarfile
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from research.bl_info_value.io import export_bytes, sha256, write_json

BANK_DIGEST = "0b216da882b634e79ac640a9970a58157828da96e93ec4dbe68e6924c8141391"
PROTOCOL = Path(__file__).with_name("PROTOCOL.md")


def check_path(path: Path) -> Path:
    """Reject the embargoed tree without enumerating or opening it."""
    resolved = path.resolve()
    if "t34-round0-study-01" in resolved.parts:
        raise ValueError("sealed study is prohibited")
    return resolved


def world_order(key: str) -> tuple[str, str]:
    """Return the prospectively fixed outcome-independent world ordering."""
    return hashlib.sha256(("bl-e3/v1|" + key).encode()).hexdigest(), key


def choose_roots(
    provenance: list[dict[str, Any]],
    sizes: tuple[int, int, int] = (192, 32, 32),
    per_class: int = 3,
) -> list[dict[str, Any]]:
    """Allocate world-disjoint splits and balanced quotas using source classes."""
    grouped: dict[str, dict[bool, list[dict]]] = defaultdict(lambda: {True: [], False: []})
    seen = set()
    for row in provenance:
        identity = (row["source_key"], int(row["decision"]))
        if identity in seen:
            raise ValueError("duplicate source root")
        seen.add(identity)
        grouped[row["source_key"]][bool(row["decisive"])].append(row)
    keys = sorted((k for k, v in grouped.items() if v[True] and v[False]), key=world_order)
    if len(keys) < sum(sizes):
        raise ValueError("INCOMPLETE: insufficient worlds with both classes")
    result, offset = [], 0
    for split, size in enumerate(sizes):
        part = keys[offset : offset + size]
        selected: dict[str, list[dict]] = {key: [] for key in part}
        for decisive in (True, False):
            pool, count = [], 0
            for key in part:
                rows = sorted(grouped[key][decisive], key=lambda row: row["decision"])
                selected[key].extend(rows[:per_class])
                count += len(rows[:per_class])
                pool.extend((key, row) for row in rows[per_class:])
            missing = per_class * size - count
            if len(pool) < missing:
                raise ValueError("INCOMPLETE: split class quota cannot be filled")
            for key, row in pool[:missing]:
                selected[key].append(row)
        for world, key in enumerate(part, start=offset):
            result.append(
                dict(
                    key=key,
                    world=world,
                    split=split,
                    roots=sorted(selected[key], key=lambda r: r["decision"]),
                )
            )
        offset += size
    return result


def prepare(bank: Path, out: Path, *, smoke: bool = False) -> dict[str, Any]:
    """Bind the selected original rows/checkpoints in a portable input archive."""
    bank, out = check_path(bank), check_path(out)
    if out.exists():
        raise FileExistsError(out)
    if sha256(bank / "manifest.json") != BANK_DIGEST:
        raise ValueError("unsealed T3 manifest identity mismatch")
    manifest = json.loads((bank / "manifest.json").read_text())
    provenance_name = "bank-00000.provenance.jsonl"
    for name in (provenance_name, "bank-00000.npz"):
        if sha256(bank / name) != manifest["files"][name]:
            raise ValueError(f"source digest mismatch: {name}")
    provenance = [json.loads(line) for line in (bank / provenance_name).read_text().splitlines()]
    selected = choose_roots(provenance, (1, 1, 1) if smoke else (192, 32, 32), 1 if smoke else 3)
    # Tiny input exercises three real units, retaining one root/class/world.
    if smoke:
        for item in selected:
            item["roots"] = [
                next(r for r in item["roots"] if r["decisive"] == cls) for cls in (True, False)
            ]
    payloads: dict[str, Path] = {}
    checkpoints: dict[str, str] = {}
    tasks = []
    from research.b8_death_census_20261014.census import b8_path
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    for item in selected:
        key = item["key"]
        receipt_path, source = bank / "worlds" / f"{key}.json", bank / "worlds" / f"{key}.npz"
        receipt = json.loads(receipt_path.read_text())
        receipt_hashes = manifest["world_receipts"]
        expected = receipt_hashes.get(str(receipt_path.relative_to(bank)))
        if expected is None:
            expected = receipt_hashes.get(key + ".json")
        if expected != sha256(receipt_path) or receipt["status"] != "PASS":
            raise ValueError(f"untrusted source receipt: {key}")
        if sha256(source) != receipt["rows_sha256"]:
            raise ValueError(f"source shard mismatch: {key}")
        with np.load(source, allow_pickle=False) as data:
            by_decision = {int(d): i for i, d in enumerate(data["decision"])}
            indices = [by_decision[r["decision"]] for r in item["roots"]]
            for row, index in zip(item["roots"], indices):
                if (
                    bool(data["decisive"][index]) != row["decisive"]
                    or int(data["frame"][index]) != row["frame"]
                ):
                    raise ValueError("provenance/source mismatch")
        hero = b8_path(receipt["arm"])
        hero_dest = f"checkpoints/{receipt['arm']}.pth"
        hero_digest = receipt["run_identity"]["b8_final_sha256"][receipt["arm"]]
        if sha256(hero) != hero_digest:
            raise ValueError("hero checkpoint changed")
        payloads[hero_dest], checkpoints[hero_dest] = hero, hero_digest
        plan = json.loads(json.dumps(receipt["roster"]))
        for spec, digest in zip(plan["specs"], plan["member_sha256s"]):
            if spec[0] == "checkpoint":
                path = check_path(Path(spec[1]))
                dest = f"checkpoints/{path.name}"
                if sha256(path) != digest:
                    raise ValueError("opponent checkpoint changed")
                payloads[dest], checkpoints[dest] = path, digest
                spec[1] = dest
        source_dest = f"worlds/{key}.npz"
        payloads[source_dest] = source
        payloads[f"worlds/{key}.json"] = receipt_path
        tasks.append(
            {k: item[k] for k in ("key", "world", "split")}
            | dict(
                arm=receipt["arm"],
                plan=plan,
                checkpoint=hero_dest,
                source_npz=source_dest,
                indices=indices,
                death_decision=receipt.get("death_decision"),
                depth=300,
                source_sha256=receipt["rows_sha256"],
                receipt_sha256=sha256(receipt_path),
            )
        )
    # build_world's pinned context verifies its whole pool, even for scripted rows.
    for name, digest in [*gid._context(1)["ds"].POOL, gid.FRP3_S12]:
        dest, path = f"checkpoints/{name}", gid.CHECKPOINT_DIR / name
        if sha256(path) != digest:
            raise ValueError("context checkpoint mismatch")
        payloads[dest], checkpoints[dest] = path, digest
    catalog = dict(
        schema="bl-e3-input/v1",
        smoke=smoke,
        tasks=tasks,
        source_manifest_sha256=BANK_DIGEST,
        protocol_sha256=sha256(PROTOCOL),
        files={name: sha256(path) for name, path in sorted(payloads.items())},
        checkpoints=checkpoints,
    )
    out.mkdir(parents=True)
    write_json(out / "catalog.json", catalog)
    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", mtime=0) as gz:
        with tarfile.open(fileobj=gz, mode="w") as archive:
            content = (json.dumps(catalog, sort_keys=True) + "\n").encode()
            info = tarfile.TarInfo("catalog.json")
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))
            for name, path in sorted(payloads.items()):
                info = tarfile.TarInfo(name)
                info.size = path.stat().st_size
                with path.open("rb") as stream:
                    archive.addfile(info, stream)
    export_bytes(out / "inputs.tar.gz", buffer.getvalue())
    return catalog


def unpack(archive: Path, out: Path, expected: str) -> dict[str, Any]:
    """Unpack only bounded regular portable inputs and verify every file."""
    if sha256(archive) != expected or out.exists():
        raise ValueError("input archive mismatch or reused input directory")
    out.mkdir(parents=True)
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        if sum(m.size for m in members) > 512 * 1024**2:
            raise ValueError("input archive exceeds bound")
        for member in members:
            if (
                not member.isfile()
                or Path(member.name).is_absolute()
                or ".." in Path(member.name).parts
            ):
                raise ValueError("unsafe input archive member")
        bundle.extractall(out, members=members, filter="data")
    catalog = json.loads((out / "catalog.json").read_text())
    if catalog["protocol_sha256"] != sha256(PROTOCOL):
        raise ValueError("frozen protocol changed")
    for name, digest in catalog["files"].items():
        path = out / name
        if not path.resolve().is_relative_to(out.resolve()) or sha256(path) != digest:
            raise ValueError("input member mismatch")
    return catalog
