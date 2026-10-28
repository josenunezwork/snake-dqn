"""Exact closed-F0 allowlist and domain-separated development-world identities.

Only positive membership grants admission. Protected banks are denied by default;
this module never inventories or opens their payloads to discover exclusions.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = Path(__file__).with_name("SANDBOX_INPUT_REGISTRY.json")
TRANSPORT_TREE = "5833e54e9c5396121cbbf99a5cd4e52006e564fd"
B8_SHA = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
# Positive, training-only extension: released fixed opponents, never label banks.
# The original hash-bound input registry and B8-only checkpoint kind stay intact.
TRAINING_OPPONENTS = frozenset(
    {
        "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723",
        "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93",
    }
)
DENIED = re.compile(r"(?:^|[/_.-])(?:f5|d4096|t34-round0-study-01|colab_cli)(?=$|[/_.-])")
B1_UNIT_NAME = re.compile(
    r"[0-9]{2}-(linear|gated_mlp|cosine)-kl(1|2|4|8|16)-survival(1|2|4|8)\.tar"
)
B2_UNIT_NAME = re.compile(r"(?P<index>[0-9]{2})-(?P<config>[a-z0-9][a-z0-9_-]*)\.tar")


def safe_path(path: str | Path) -> Path:
    """Reject forbidden identities before filesystem reads, including aliases."""
    p = Path(path)
    for candidate in (str(p), str(p.resolve())):
        if DENIED.search(candidate.lower().replace("\\", "/")):
            raise ValueError("sandbox refuses protected/transport path")
    return p


def digest(path: str | Path) -> str:
    """Hash a permitted file using bounded memory."""
    p = safe_path(path)
    h = hashlib.sha256()
    with p.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def f4_fold(parent: str) -> int:
    """Preserve the original ultimate-parent exclusion rule."""
    return (
        int.from_bytes(
            hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
        )
        % 5
    )


def world_seed(study: str, idx: int) -> int:
    """Use the first eight SHA256 bytes, big endian; never narrow to uint32."""
    if not isinstance(study, str) or not re.fullmatch(r"[a-z0-9][a-z0-9_-]*", study):
        raise ValueError("canonical nonempty study slug required")
    if type(idx) is not int or idx < 0:
        raise ValueError("nonnegative integer canonical world index required")
    return int.from_bytes(hashlib.sha256(f"a100-dev/{study}/{idx}".encode()).digest()[:8], "big")


def load_registry(path: str | Path | None = None) -> dict:
    """Load a registry and reject inconsistent or broadened parent membership."""
    value = json.loads(safe_path(path or DEFAULT_REGISTRY).read_text())
    if value.get("schema") != "a100-sandbox-input-registry/v1":
        raise ValueError("foreign registry schema")
    excluded = set(value["excluded_parents"])
    ids = set()
    for row in value["lives"]:
        parent = row["ultimate_parent_id"]
        life = row["source_life_id"]
        if life in ids or parent in excluded or f4_fold(parent) == 0:
            raise ValueError("duplicate/protected ancestry in registry")
        if life.split("/")[1] != parent:
            raise ValueError("registry ancestry alias")
        safe_path(row["payload_path"])
        ids.add(life)
    for dose in ("D256", "D1024"):
        if not set(value["datasets"][dose]["life_ids"]) <= ids:
            raise ValueError("unlisted dataset life")
    return value


def verify_sources(registry: dict) -> None:
    """Check frozen source bytes without opening any protected payloads."""
    for source in registry["sources"]:
        if digest(source["path"]) != source["sha256"]:
            raise ValueError(f"registry source drift: {source['path']}")


def validate_rows(records: list[dict], registry: dict | None = None, dose: str = "D1024") -> None:
    """Require each row's exact life and ultimate parent, with unique row IDs."""
    registry = registry or load_registry()
    if dose not in registry["datasets"]:
        raise ValueError("foreign dose")
    allowed = set(registry["datasets"][dose]["life_ids"])
    parents = {r["source_life_id"]: r["ultimate_parent_id"] for r in registry["lives"]}
    seen = set()
    if not records:
        raise ValueError("empty training rows")
    for row in records:
        life, parent, row_id = (row.get(k) for k in ("life_id", "parent_world", "row_id"))
        if life not in allowed or parents.get(life) != parent:
            raise ValueError("foreign/protected life or parent")
        suffix = row_id[len(life) + 1 :] if isinstance(row_id, str) else ""
        if (
            not isinstance(row_id, str)
            or not row_id.startswith(life + "/")
            or not suffix.isdigit()
            or row_id in seen
        ):
            raise ValueError("duplicate/foreign row identity")
        seen.add(row_id)


def _valid_compact_at_source_quadruple(original_pt: Path) -> bool:
    """Verify a complete compact-at-source snapshot quadruple when raw .pt is absent."""
    xz_path = Path(f"{original_pt}.xz")
    manifest_path = Path(f"{original_pt}.xz.manifest")
    meta_path = original_pt.with_suffix(".json")
    actor_path = original_pt.with_name(f"{original_pt.stem}-actor.pth")
    digests: dict[Path, str] = {}
    for member in (xz_path, manifest_path, meta_path, actor_path):
        sha_sidecar = Path(f"{member}.sha256")
        complete_sidecar = Path(f"{member}.complete")
        if not member.is_file() or not sha_sidecar.is_file() or not complete_sidecar.is_file():
            return False
        member_sha = digest(member)
        if sha_sidecar.read_text().strip() != member_sha:
            return False
        digests[member] = member_sha
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(meta, dict):
        return False
    compact_meta = meta.get("compact_snapshot")
    if not isinstance(compact_meta, dict):
        return False
    deriv = compact_meta.get("derivative")
    man = compact_meta.get("manifest")
    if not isinstance(deriv, dict) or not isinstance(man, dict):
        return False
    return (
        compact_meta.get("schema") == "sandbox-a-train-compact-export/v1"
        and deriv.get("name") == xz_path.name
        and deriv.get("sha256") == digests[xz_path]
        and man.get("name") == manifest_path.name
        and man.get("sha256") == digests[manifest_path]
    )


def validate_input(record: dict, registry: dict | None = None) -> None:
    """Validate content identity before a consumer opens numerical arrays."""
    registry = registry or load_registry()
    kind = record.get("kind")
    if kind == "worlds":
        indices = record.get("indices")
        if not isinstance(indices, list) or not indices or len(set(indices)) != len(indices):
            raise ValueError("world indices must be unique")
        expected = [world_seed(record["study"], i) for i in indices]
        if record.get("seeds") != expected:
            raise ValueError("world seed or uint64 normalization differs")
        if set(expected) & set(registry["excluded_seeds"]):
            raise ValueError("world exclusion intersection")
        return
    path = safe_path(record.get("path", ""))
    claimed = record.get("sha256")
    if not isinstance(claimed, str) or digest(path) != claimed:
        raise ValueError("input hash mismatch")
    if kind == "closed-f0":
        dataset = registry["datasets"].get(record.get("dose"), {})
        if claimed != dataset.get("sha256"):
            raise ValueError("unregistered training payload")
    elif kind == "frozen-checkpoint":
        if claimed != B8_SHA:
            raise ValueError("only frozen released B8 input admitted")
    elif kind == "frozen-opponent":
        if claimed not in TRAINING_OPPONENTS:
            raise ValueError("only canonical released TRAIN-01 opponents admitted")
    elif kind == "training-export":
        if not re.fullmatch(r"iteration-[0-9]{4}(?:\.pt|\.json|-actor\.pth)", path.name):
            raise ValueError("canonical TRAIN-01 exported snapshot path required")
    elif kind == "training-export-sidecar":
        match = re.fullmatch(
            r"(iteration-[0-9]{4}(?:\.pt|\.json|-actor\.pth))\.(sha256|complete)", path.name
        )
        if match is None:
            raise ValueError("canonical TRAIN-01 snapshot sidecar path required")
        payload = path.with_name(match.group(1))
        payload_sha = Path(str(payload) + ".sha256")
        payload_complete = Path(str(payload) + ".complete")
        if (
            not payload.is_file()
            or not payload_sha.is_file()
            or not payload_complete.is_file()
            or payload_sha.read_text().strip() != digest(payload)
        ):
            raise ValueError("training-export sidecar lacks a complete canonical payload")
    elif kind == "training-export-derivative":
        # A lossless transport derivative (XZ stream) of a canonical full-state
        # snapshot, or its manifest of the original decoded tensors.  It must be
        # a completion-last artifact next to either its complete original raw
        # .pt or a complete compact-at-source quadruple (.pt.xz, .pt.xz.manifest,
        # .json with compact_snapshot binding, and -actor.pth); which derivative
        # may resume is frozen by the TRAIN-01 runtime allowlist.
        match = re.fullmatch(r"(iteration-[0-9]{4}\.pt)\.xz(?:\.manifest)?", path.name)
        if match is None:
            raise ValueError("canonical TRAIN-01 snapshot derivative path required")
        sidecar = Path(str(path) + ".sha256")
        complete = Path(str(path) + ".complete")
        original = path.with_name(match.group(1))
        orig_sha = Path(str(original) + ".sha256")
        orig_complete = Path(str(original) + ".complete")
        if (
            not sidecar.is_file()
            or not complete.is_file()
            or sidecar.read_text().strip() != claimed
        ):
            raise ValueError("snapshot derivative lacks completion sidecars or its original")
        any_raw = original.exists() or orig_sha.exists() or orig_complete.exists()
        raw_complete = original.is_file() and orig_sha.is_file() and orig_complete.is_file()
        if any_raw:
            if not raw_complete:
                raise ValueError("snapshot derivative lacks completion sidecars or its original")
        elif not _valid_compact_at_source_quadruple(original):
            raise ValueError("snapshot derivative lacks completion sidecars or its original")
    elif kind == "verified-ledger":
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("units"), list):
            raise ValueError("verified ledger object/units required")
    elif kind == "verified-unit":
        if B1_UNIT_NAME.fullmatch(path.name):
            return
        match = B2_UNIT_NAME.fullmatch(path.name)
        if match is None:
            raise ValueError("verified unit must have canonical unit-NNN.tar name")
        # B2 unit filenames bind both the frozen configuration identity and
        # its canonical allocation index. Import only in this B2-specific
        # branch so the registry remains independent of the B2 package at
        # ordinary sandbox admission/import time.
        from research.sandbox_b_heads.core_b2 import CONFIGS

        index = int(match["index"])
        if index >= len(CONFIGS) or match["config"] != CONFIGS[index].id:
            raise ValueError("verified B2 unit index/config identity differs from frozen inventory")
    elif kind == "registry":
        if claimed != digest(DEFAULT_REGISTRY):
            raise ValueError("registry identity differs")
    else:
        raise ValueError("unregistered input kind")


def validate_npz(path: str | Path, dose: str = "D1024", registry: dict | None = None) -> None:
    """Check registered bytes then read only lineage columns for row admission."""
    import numpy as np

    registry = registry or load_registry()
    validate_input(
        {"kind": "closed-f0", "dose": dose, "path": str(path), "sha256": digest(path)}, registry
    )
    with np.load(path, allow_pickle=False) as data:
        records = [
            dict(zip(("row_id", "life_id", "parent_world"), row))
            for row in zip(
                data["row_id"].tolist(), data["life_id"].tolist(), data["parent_world"].tolist()
            )
        ]
    validate_rows(records, registry, dose)
