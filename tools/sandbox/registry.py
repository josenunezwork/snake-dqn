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
DENIED = ("f5", "d4096", "t34-round0-study-01", "colab_cli")


def safe_path(path: str | Path) -> Path:
    """Reject forbidden identities before filesystem reads, including aliases."""
    p = Path(path)
    for candidate in (str(p), str(p.resolve())):
        if any(token in candidate.lower() for token in DENIED):
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
    elif kind == "verified-ledger":
        value = json.loads(path.read_text())
        if not isinstance(value, dict) or not isinstance(value.get("units"), list):
            raise ValueError("verified ledger object/units required")
    elif kind == "verified-unit":
        if not re.fullmatch(
            r"[0-9]{2}-(linear|gated_mlp|cosine)-kl(1|2|4|8|16)-survival(1|2|4|8)\.tar", path.name
        ):
            raise ValueError("verified unit must have canonical unit-NNN.tar name")
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
