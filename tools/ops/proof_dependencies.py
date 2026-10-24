#!/usr/bin/env python3
"""Fail-closed proof dependency inventory and independent delta-acceptance input.

This is deliberately an input to, not a replacement for, an independent
Codex-V review.  It does not read outcomes, approve dispatch, or cache current
host/account/native admission checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterable

SCHEMA = "r33-proof-dependencies/v1"
RULING_SHA256 = "7db57b262f1bfa4ae70597772d79e2904f8d6a885ea530c54178d869911cf07a"
REQUIRED_KINDS = {
    "source",
    "input",
    "environment",
    "platform",
    "command",
    "fixture",
    "allocation",
    "contract",
}
ALLOWED_KINDS = REQUIRED_KINDS | {"location"}
REQUIRED_LIVE = {"native_canary", "live_admission"}
REUSABLE_NAMES = ("destination:", "transport_location:")


class Refused(ValueError):
    """The dependency closure is incomplete or no longer matches a proof."""


def canonical(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value)).hexdigest()


def load_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise Refused(f"JSON object required: {path}")
    return value


def dependency_inventory(package: str, dependencies: list[dict[str, Any]]) -> dict[str, Any]:
    """Normalize a complete, named closure without silently accepting duplicates."""
    if not package:
        raise Refused("package is required")
    seen: set[str] = set()
    values: list[dict[str, str]] = []
    kinds: set[str] = set()
    for item in dependencies:
        if set(item) != {"name", "kind", "sha256"}:
            raise Refused("each dependency has exactly name, kind and sha256")
        name, kind, sha = item["name"], item["kind"], item["sha256"]
        if (
            not isinstance(name, str)
            or not name
            or not isinstance(kind, str)
            or kind not in ALLOWED_KINDS
        ):
            raise Refused("dependency name/kind is invalid")
        if (
            not isinstance(sha, str)
            or len(sha) != 64
            or any(c not in "0123456789abcdef" for c in sha)
        ):
            raise Refused("dependency sha256 is invalid")
        if name in seen:
            raise Refused(f"duplicate dependency name: {name}")
        seen.add(name)
        kinds.add(kind)
        values.append({"name": name, "kind": kind, "sha256": sha})
    missing = REQUIRED_KINDS - kinds
    if missing:
        raise Refused("dependency closure missing kinds: " + ", ".join(sorted(missing)))
    values.sort(key=lambda item: item["name"])
    body = {"schema": SCHEMA, "package": package, "dependencies": values}
    return {**body, "inventory_sha256": digest(body)}


def compare(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    """Return the exact complete delta; unknown/add/remove changes fail closed."""
    for value in (previous, current):
        if value.get("schema") != SCHEMA or not isinstance(value.get("dependencies"), list):
            raise Refused("inventory has unknown schema")
        # Re-validate rather than trusting an asserted digest.
        rebuilt = dependency_inventory(value.get("package", ""), value["dependencies"])
        if rebuilt["inventory_sha256"] != value.get("inventory_sha256"):
            raise Refused("inventory digest mismatch")
    old = {item["name"]: item for item in previous["dependencies"]}
    new = {item["name"]: item for item in current["dependencies"]}
    if previous.get("package") != current.get("package"):
        raise Refused("foreign package inventory requires full review")
    removed, added = sorted(set(old) - set(new)), sorted(set(new) - set(old))
    if removed or added:
        raise Refused("dependency closure changed (added/removed dependency)")
    changed = [name for name in sorted(old) if old[name] != new[name]]
    return {"changed": changed, "unchanged": sorted(set(old) - set(changed))}


def delta_input(
    *,
    previous: dict[str, Any],
    current: dict[str, Any],
    reused_proofs: list[dict[str, str]],
    reviewer: str,
    live_obligations: list[str],
    charter_opt_in: bool,
) -> dict[str, Any]:
    """Create reviewer input; it is never itself an acceptance or self-certification."""
    if not charter_opt_in:
        raise Refused("package did not prospectively opt into the R33 charter")
    if not reviewer or reviewer.lower() in {"sol", "builder", "sol-o", "sol-p"}:
        raise Refused("an independent reviewer identity is required")
    if set(live_obligations) != REQUIRED_LIVE:
        raise Refused("fresh native_canary and live_admission obligations are required")
    delta = compare(previous, current)
    if not delta["changed"]:
        raise Refused("no delta: use the original proof, do not manufacture acceptance")
    current_by_name = {item["name"]: item for item in current["dependencies"]}
    reusable = all(
        name.startswith(REUSABLE_NAMES) and current_by_name[name]["kind"] == "location"
        for name in delta["changed"]
    )
    proof_ids: set[str] = set()
    normalized: list[dict[str, str]] = []
    for proof in reused_proofs:
        required = {"path", "sha256", "inventory_sha256", "contract_sha256", "reuse_permitted"}
        if set(proof) != required or not all(
            isinstance(proof[k], str) and proof[k]
            for k in ("path", "sha256", "inventory_sha256", "contract_sha256")
        ):
            raise Refused(
                "reused proof requires path, sha256, closure inventory, contract and permission"
            )
        if any(
            len(proof[key]) != 64 for key in ("sha256", "inventory_sha256", "contract_sha256")
        ) or not isinstance(proof["reuse_permitted"], bool):
            raise Refused("reused proof sha256 is invalid")
        if proof["inventory_sha256"] != previous["inventory_sha256"]:
            raise Refused("reused proof closure does not bind the prior complete inventory")
        if not proof["reuse_permitted"]:
            reusable = False
        if proof["path"] in proof_ids:
            raise Refused("duplicate reused proof")
        proof_ids.add(proof["path"])
        normalized.append(dict(proof))
    if not normalized:
        raise Refused("at least one prior proof is required")
    body = {
        "schema": "r33-independent-delta-input/v1",
        "ruling_sha256": RULING_SHA256,
        "reviewer": reviewer,
        "previous_inventory_sha256": previous["inventory_sha256"],
        "current_inventory_sha256": current["inventory_sha256"],
        "delta": delta,
        "reused_proofs": sorted(normalized, key=lambda item: item["path"]),
        "fresh_live_obligations": sorted(live_obligations),
        "status": "PENDING_INDEPENDENT_ACCEPTANCE" if reusable else "FULL_REVIEW_REQUIRED",
        "reuse_class": "destination_or_transport_location" if reusable else None,
        "reviewer_time_target": "UNMEASURED_TARGET_50_PERCENT_REDUCTION",
    }
    return {**body, "input_sha256": digest(body)}


def validate_independent_acceptance(value: dict[str, Any], delta: dict[str, Any]) -> None:
    """Validate a Codex-V-created acceptance record without creating one for it."""
    required = {"schema", "input_sha256", "reviewer", "decision", "checked", "created_at"}
    if set(value) != required or value.get("schema") != "r33-independent-delta-acceptance/v1":
        raise Refused("unknown independent acceptance schema")
    if value.get("input_sha256") != delta.get("input_sha256") or value.get("reviewer") != delta.get(
        "reviewer"
    ):
        raise Refused("acceptance does not bind this independent delta input")
    if value.get("decision") != "ACCEPT" or delta.get("status") != "PENDING_INDEPENDENT_ACCEPTANCE":
        raise Refused("only a narrow pending delta may be accepted")
    checked = value.get("checked")
    required_checks = {
        "downloaded_bytes",
        "destination_semantics",
        "access_integrity",
        "runtime_environment",
        "consumer_behavior",
    }
    if not isinstance(checked, list) or set(checked) != required_checks:
        raise Refused("independent acceptance lacks a required narrow-charter check")


def create_only(path: Path, value: dict[str, Any]) -> None:
    """Atomically publish a reviewer input once, without replacing prior evidence."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(canonical(value))
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    except FileExistsError as exc:
        raise Refused(f"create-only acceptance input already exists: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inventory = sub.add_parser("inventory")
    inventory.add_argument("--package", required=True)
    inventory.add_argument("--dependencies", type=Path, required=True)
    inventory.add_argument("--out", type=Path, required=True)
    delta = sub.add_parser("delta-input")
    delta.add_argument("--previous", type=Path, required=True)
    delta.add_argument("--current", type=Path, required=True)
    delta.add_argument("--proofs", type=Path, required=True)
    delta.add_argument("--reviewer", required=True)
    delta.add_argument("--live-obligation", action="append", required=True)
    delta.add_argument("--out", type=Path, required=True)
    delta.add_argument("--charter-opt-in", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)
    try:
        if args.command == "inventory":
            dependencies = json.loads(args.dependencies.read_text())
            if not isinstance(dependencies, list):
                raise Refused("dependencies file must be a JSON list")
            value = dependency_inventory(args.package, dependencies)
        else:
            proofs = json.loads(args.proofs.read_text())
            if not isinstance(proofs, list):
                raise Refused("proofs file must be a JSON list")
            value = delta_input(
                previous=load_object(args.previous),
                current=load_object(args.current),
                reused_proofs=proofs,
                reviewer=args.reviewer,
                live_obligations=args.live_obligation,
                charter_opt_in=args.charter_opt_in,
            )
        create_only(args.out, value)
        print(json.dumps(value, sort_keys=True))
        return 0
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(f"REFUSED: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
