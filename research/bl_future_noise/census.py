"""Metadata-only A0 gate. Never discover banks, reconstruct worlds, or dispatch.

Inputs are explicit clearance metadata, not a directory to search. Every member
is checked against that clearance before reading. Passing this tool prepares a
census for independent Codex-V review; it does not admit A1 or A2 by itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

from research.bl_served_search.exports import export_json

RULING_SHA256 = "f17130be17b446c4eb0ee6abcbc668068c684b41f194fcc769580d3280cbb84e"
NAMESPACE = "bl-r31-future-noise-v1"
B8_SHA256 = "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8"
PROTECTED_CLASSES = (
    "f4_parents_and_descendants",
    "reserve_archive_probe_gameplay",
    "strict_product",
    "sealed_timing_labels",
    "active_p1",
    "round28",
    "current_factory",
)
FORBIDDEN_COMPONENTS = frozenset({"t34-round0-study-01", "bl-labels-f0", "gpu_sim_spike", "bl_ppo"})
MAX_METADATA_BYTES = 16 * 1024 * 1024


class Refused(ValueError):
    """An input cannot establish the ruled archive boundary."""


def _text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise Refused(f"nonempty {name} required")
    return value


def _hash(value: Any, name: str, length: int = 64) -> str:
    if not isinstance(value, str) or re.fullmatch(f"[0-9a-f]{{{length}}}", value) is None:
        raise Refused(f"{name} must be a full lowercase digest")
    return value


def safe_path(path: Path) -> Path:
    """Reject protected lexical paths before resolving links, then check aliases."""
    if FORBIDDEN_COMPONENTS.intersection(part.casefold() for part in path.parts):
        raise Refused("protected input path")
    resolved = path.resolve()
    if FORBIDDEN_COMPONENTS.intersection(part.casefold() for part in resolved.parts):
        raise Refused("protected input alias")
    return resolved


def _json(raw: bytes) -> dict[str, Any]:
    def unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        value: dict[str, Any] = {}
        for key, item in pairs:
            if key in value:
                raise Refused("duplicate JSON key")
            value[key] = item
        return value

    def nonfinite(value: str) -> None:
        raise Refused(f"nonfinite JSON: {value}")

    value = json.loads(raw, object_pairs_hook=unique, parse_constant=nonfinite)
    if not isinstance(value, dict):
        raise Refused("metadata must be an object")
    return value


def _read(path: Path) -> bytes:
    path = safe_path(path)
    if not path.is_file() or path.stat().st_size > MAX_METADATA_BYTES:
        raise Refused("missing or oversized metadata")
    return path.read_bytes()


def _member(root: Path, binding: dict[str, Any]) -> tuple[Path, bytes]:
    relative = Path(_text(binding.get("path"), "member path"))
    if relative.is_absolute() or ".." in relative.parts:
        raise Refused("member must be relative and contained")
    target = safe_path(root / relative)
    if not target.is_relative_to(root):
        raise Refused("member escaped cleared archive")
    raw = _read(target)
    if (
        type(binding.get("size")) is not int
        or binding["size"] != len(raw)
        or _hash(binding.get("sha256"), "member sha256") != hashlib.sha256(raw).hexdigest()
    ):
        raise Refused("member size/hash differs")
    return target, raw


def _order(kind: str, *identities: str) -> str:
    # JSON tuple encoding avoids ambiguous delimiter concatenations.
    return hashlib.sha256(json.dumps([NAMESPACE, kind, *identities]).encode()).hexdigest()


def _actions(value: Any) -> set[int]:
    if (
        not isinstance(value, list)
        or any(type(a) is not int or not 0 <= a < 6 for a in value)
        or len(set(value)) != len(value)
    ):
        raise Refused("unique actions in [0,5] required")
    return set(value)


def _outcomes(root: dict[str, Any]) -> list[list[int]]:
    legal = _actions(root.get("legal"))
    selectable = _actions(root.get("served_selectable"))
    if not selectable <= legal:
        raise Refused("served mask exceeds legal mask")
    bare, rescue = root.get("bare_action"), root.get("a_star")
    if any(type(a) is not int or a not in selectable for a in (bare, rescue)) or bare == rescue:
        raise Refused("frozen actions must be distinct legal served-selectable actions")
    outcomes = root.get("original_outcomes")
    if not isinstance(outcomes, dict) or set(outcomes) != {str(a) for a in range(6)}:
        raise Refused("all six original action statuses required, including unknowns")
    survivors, diers = [], []
    for key, result in outcomes.items():
        if not isinstance(result, dict) or result.get("status") not in (
            "EXACT",
            "UNKNOWN",
            "NOT_EVALUATED",
        ):
            raise Refused("original outcome status missing")
        if result["status"] != "EXACT":
            continue
        mass = result.get("mass_sum")
        if (
            result.get("horizon") != 300
            or type(result.get("horizon")) is not int
            or type(result.get("survived")) is not bool
            or isinstance(mass, bool)
            or not isinstance(mass, (int, float))
            or not math.isfinite(mass)
            or mass < 0
        ):
            raise Refused("exact H300 original mass/survival required")
        if int(key) in selectable:
            (survivors if result["survived"] else diers).append(int(key))
    if rescue not in survivors or bare not in diers or root.get("decisive_doom") is not True:
        raise Refused("root is not an original decisive doom rescue")
    return [[s, d] for s in sorted(survivors) for d in sorted(diers)]


def prepare(clearance_path: Path | None) -> dict[str, Any]:
    """Freeze all cleared metadata before hash selection; never open raw outcomes.

    Original outcomes here must already be unsealed archive metadata. No external
    reconstruction payload, protected bank or checkpoint is opened by A0.
    """
    common = {
        "schema": "bl-r31-a0/v1",
        "ruling_sha256": RULING_SHA256,
        "namespace": NAMESPACE,
        "dispatch_allowed": False,
        "colab_units": 0,
        "rollout_decisions": 0,
        "implementation_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "intent_sha256": hashlib.sha256(
            Path(__file__).with_name("INTENT.md").read_bytes()
        ).hexdigest(),
    }
    if clearance_path is None:
        return {
            **common,
            "status": "BLOCKED_MISSING_CLEARANCE",
            "eligible_parent_count": None,
            "admitted_parent_count": 0,
            "next_stage": "STOP_AT_A0",
            "reason": "No separately cleared archive supplied; physical support is unknown.",
        }
    clearance_raw = _read(clearance_path)
    clearance = _json(clearance_raw)
    if (
        clearance.get("schema") != "bl-r31-clearance/v1"
        or clearance.get("ruling_sha256") != RULING_SHA256
        or clearance.get("source_kind") != "historical-development"
        or clearance.get("already_unsealed") is not True
        or clearance.get("current_factory_output") is not False
    ):
        raise Refused("missing/foreign clearance or prohibited source")
    _text(clearance.get("cleared_by"), "clearance author")
    _text(clearance.get("clearance_basis"), "clearance basis")
    archive = Path(_text(clearance.get("archive_root"), "archive root"))
    if not archive.is_absolute():
        raise Refused("cleared archive root must be absolute")
    archive = safe_path(archive)
    _, census_raw = _member(archive, clearance["census"])
    _, audit_raw = _member(archive, clearance["ancestry_audit"])
    census, audit = _json(census_raw), _json(audit_raw)
    namespace = _text(census.get("source_namespace"), "source namespace")
    source_commit = _hash(census.get("source_commit"), "source commit", 40)
    checkpoints = census.get("checkpoint_sha256")
    if not isinstance(checkpoints, dict) or not checkpoints:
        raise Refused("checkpoint identity map required")
    for name, checksum in checkpoints.items():
        _text(name, "checkpoint name")
        _hash(checksum, "checkpoint sha256")
    if checkpoints.get("b8") != B8_SHA256:
        raise Refused("comparator is not pinned B8 s3")
    if (
        census.get("schema") != "bl-r31-source-census/v1"
        or audit.get("schema") != "bl-r31-ancestry-audit/v1"
        or audit.get("census_sha256") != hashlib.sha256(census_raw).hexdigest()
        or set(audit.get("protected", {})) != set(PROTECTED_CLASSES)
    ):
        raise Refused("whole census ancestry coverage missing")
    protected_ids, protected_namespaces = set(), set()
    for category in PROTECTED_CLASSES:
        item = audit["protected"][category]
        _text(item.get("inventory_basis"), "protected inventory evidence basis")
        _hash(item.get("inventory_sha256"), "protected inventory digest")
        for key, destination in (
            ("ancestry_ids", protected_ids),
            ("namespaces", protected_namespaces),
        ):
            values = item.get(key)
            if not isinstance(values, list):
                raise Refused("protected identity/namespace list required")
            destination.update(_text(v, "protected identity") for v in values)
    if namespace in protected_namespaces:
        raise Refused("protected source namespace collision")
    parents = census.get("parents")
    if not isinstance(parents, list):
        raise Refused("whole source parent census required")
    seen_parents, seen_roots, seen_ancestors = set(), set(), set()
    eligible = []
    # Validate EVERY candidate, including parents beyond the first 64. A bad
    # reconstruction never becomes an outcome-driven replacement after sampling.
    for parent in parents:
        parent_id = _text(parent.get("parent_id"), "parent ID")
        ancestry = parent.get("ancestry_ids")
        if not isinstance(ancestry, list) or not ancestry:
            raise Refused("complete parent ancestry required")
        ancestry = {_text(v, "ancestor ID") for v in ancestry}
        if parent_id not in ancestry or parent_id in seen_parents:
            raise Refused("missing ultimate parent or duplicate parent")
        if ancestry & protected_ids:
            raise Refused("protected parent/descendant collision")
        if ancestry & seen_ancestors:
            raise Refused("parents share ancestry")
        seen_ancestors.update(ancestry)
        seen_parents.add(parent_id)
        _text(parent.get("stream"), "stream")
        _text(parent.get("roster"), "ordered roster identity")
        roots = parent.get("roots")
        if not isinstance(roots, list) or not roots:
            raise Refused("eligible roots required for every census parent")
        checked = []
        for root in roots:
            root_id = _text(root.get("root_id"), "root ID")
            if root_id in seen_roots or root.get("prepared") is not True:
                raise Refused("duplicate or unprepared root")
            seen_roots.add(root_id)
            pairs = _outcomes(root)
            # These are small, already-unsealed reconstruction receipts. Their
            # input hashes bind payloads without opening or unpickling them here.
            _, proof_raw = _member(archive, root["reconstruction_receipt"])
            proof = _json(proof_raw)
            if (
                proof.get("schema") != "bl-r31-reconstruction-evidence/v1"
                or proof.get("parent_id") != parent_id
                or proof.get("root_id") != root_id
                or proof.get("source_commit") != source_commit
                or proof.get("checkpoint_sha256") != checkpoints
                or proof.get("original_outcomes") != root["original_outcomes"]
                or proof.get("original_stream_identity") is not True
                or proof.get("prepared_once") is not True
            ):
                raise Refused("whole census reconstruction/original identity not established")
            _hash(proof.get("root_state_sha256"), "prepared root digest")
            inputs = proof.get("inputs")
            if not isinstance(inputs, list) or not inputs:
                raise Refused("reconstruction input bindings missing")
            for binding in inputs:
                relative = Path(_text(binding.get("path"), "reconstruction input"))
                if relative.is_absolute() or ".." in relative.parts:
                    raise Refused("reconstruction input is not contained")
                target = safe_path(archive / relative)
                if not target.is_relative_to(archive):
                    raise Refused("reconstruction input escaped archive")
                _hash(binding.get("sha256"), "reconstruction input digest")
                if type(binding.get("size")) is not int or binding["size"] <= 0:
                    raise Refused("positive reconstruction input size required")
            checked.append({**root, "original_pairs": pairs})
        chosen = min(checked, key=lambda r: (_order("root", parent_id, r["root_id"]), r["root_id"]))
        eligible.append({**parent, "roots": checked, "selected_root_id": chosen["root_id"]})
    frozen = {
        **common,
        "clearance_sha256": hashlib.sha256(clearance_raw).hexdigest(),
        "source_census_sha256": hashlib.sha256(census_raw).hexdigest(),
        "ancestry_audit_sha256": hashlib.sha256(audit_raw).hexdigest(),
        "source_namespace": namespace,
        "source_commit": source_commit,
        "checkpoint_sha256": checkpoints,
        "eligible_parent_count": len(eligible),
        "source_census": census,
        "ancestry_audit": audit,
        "clearance": clearance,
        "admitted_parent_count": 0,
    }
    if len(eligible) < 64:
        return {**frozen, "status": "INSUFFICIENT_SUPPORT", "next_stage": "STOP_AT_A0"}
    selected = sorted(eligible, key=lambda p: (_order("parent", p["parent_id"]), p["parent_id"]))[
        :64
    ]
    pair_parents = {
        p["parent_id"]
        for p in sorted(
            selected, key=lambda p: (_order("pair-parent", p["parent_id"]), p["parent_id"])
        )[:16]
    }
    allocation = []
    for index, parent in enumerate(selected):
        root = next(r for r in parent["roots"] if r["root_id"] == parent["selected_root_id"])
        pair = None
        if parent["parent_id"] in pair_parents:
            pair = min(
                root["original_pairs"],
                key=lambda pair: _order(
                    "original-pair", parent["parent_id"], root["root_id"], *map(str, pair)
                ),
            )
        allocation.append(
            {
                "unit_id": f"{NAMESPACE}:{index}",
                "allocation_index": index,
                "parent_id": parent["parent_id"],
                "stream": parent["stream"],
                "roster": parent["roster"],
                "root": root,
                "pair": pair,
            }
        )
    return {
        **frozen,
        "status": "CENSUS_PREPARED_PENDING_CODEX_V",
        "next_stage": "INDEPENDENT_CLEARANCE_REVIEW",
        "allocation": allocation,
        "disclosure": (
            "Outcome-selected original rescue stratum; " "not natural occupancy or efficacy."
        ),
    }


def main() -> int:
    """Publish one immutable A0 receipt; absence/shortage exits without admission."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clearance", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    output = safe_path(args.out)
    try:
        result = prepare(args.clearance)
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as exc:
        result = {
            "schema": "bl-r31-a0/v1",
            "ruling_sha256": RULING_SHA256,
            "status": "INVALID",
            "dispatch_allowed": False,
            "admitted_parent_count": 0,
            "next_stage": "STOP_AT_A0",
            "reason": str(exc),
        }
    exported = export_json(output, result)
    print(json.dumps({"status": result["status"], **exported}, sort_keys=True))
    return 0 if result["status"] == "CENSUS_PREPARED_PENDING_CODEX_V" else 2


if __name__ == "__main__":
    raise SystemExit(main())
