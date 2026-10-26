"""Build a provisional F5 catalog from metadata mirrors and repository audits.

Protected directories are pruned before bounded repository discovery. The
resulting exact source register is hash-bound, but filename discovery never
proves completeness. Repository seed-only projections preserve unknown ancestry
and block acceptance. Explicit lawful mirrors may carry resolved identities;
complete coverage and a bound custodian attestation remain mandatory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

from research.bl_residual import f5_r10a_annex as annex

SCHEMA = "f5-r10a-collision-catalog/v1"
MIRROR_SCHEMA = "f5-r10a-lawful-collision-mirror/v1"
MAX_FILES = 256
MAX_BYTES_PER_FILE = 8 * 1024 * 1024
MAX_ROWS_PER_FILE = 200_000
REQUIRED_CLASSES = (
    "training_development",
    "validation",
    "probe_canary_timing",
    "strict",
    "serving",
    "f4_folds_census",
    "r30_1",
    "r30_2_registered",
    "reserved_failed_aborted_retried_superseded",
    "protected_allocations",
)
PROTECTED_TOKENS = ("d4096", "t34-round0-study-01")
ANNEX_SCHEMA = "bl-f5-r10a-metadata-annex/v1"
AUDIT_NAME = re.compile(
    r"(?:seed[_-]?audit|seed[_-]?bank[_-]?audit|allocation.*rejected|world[_-]?bank)", re.I
)
FORBIDDEN_KEY = re.compile(
    rb'"(?:mass|return|reward|score|survival|death|outcome|metric|loss|q_value|qmax|'
    rb'alive_frame)"\s*:'
)
SEED_ARRAY = re.compile(rb'"([^" ]*seed[^" ]*)"\s*:\s*\[([^\]]*)\]', re.I)
SEED_SCALAR = re.compile(rb'"([^" ]*seed[^" ]*)"\s*:\s*(\d+)', re.I)
INTEGER = re.compile(rb"\d+")


def canonical(value: Any) -> bytes:
    """Return finite deterministic JSON bytes."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def sha(raw: bytes) -> str:
    """Hash bytes that were explicitly admitted by the input plan."""
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed instead of inferring omitted collision metadata."""
    if not condition:
        raise ValueError(message)


def forbidden(path: Path) -> bool:
    """Identify sources this ruling bars from opening, including metadata."""
    lowered = str(path).lower()
    return any(token in lowered for token in PROTECTED_TOKENS)


def _safe_source(path: Path) -> None:
    require(path.is_absolute(), "mirror path must be absolute")
    require(not forbidden(path), "protected source must be represented by an external mirror")
    require(
        not path.is_symlink() and not any(parent.is_symlink() for parent in path.parents),
        "symlink mirror refused",
    )
    require(path.is_file(), "missing collision mirror")
    require(path.stat().st_size <= MAX_BYTES_PER_FILE, "collision mirror exceeds byte bound")


def _row(row: Any, source_id: str) -> dict[str, Any]:
    """Validate one metadata-only collision record; reject foreign fields."""
    require(isinstance(row, dict), "mirror record must be an object")
    require(
        set(row) == {"parent_id", "seed", "aliases", "ancestry", "scopes", "state"},
        "foreign mirror field",
    )
    require(isinstance(row["parent_id"], str) and row["parent_id"], "missing parent identity")
    require(type(row["seed"]) is int and 0 <= row["seed"] < 2**64, "invalid physical seed")
    require(
        isinstance(row["aliases"], list) and all(isinstance(x, str) and x for x in row["aliases"]),
        "invalid aliases",
    )
    require(
        isinstance(row["ancestry"], list)
        and all(isinstance(x, str) and x for x in row["ancestry"]),
        "invalid ancestry",
    )
    require(
        isinstance(row["scopes"], list) and set(row["scopes"]) <= set(REQUIRED_CLASSES),
        "invalid scopes",
    )
    require(row["scopes"], "missing exclusion scope")
    require(
        row["state"]
        in {"reserved", "failed", "aborted", "retried", "superseded", "completed", "planned"},
        "invalid allocation state",
    )
    identities = {row["parent_id"], *row["aliases"], *row["ancestry"]}
    require(
        len(identities) == len(row["aliases"]) + len(row["ancestry"]) + 1,
        "duplicate identity alias",
    )
    return {"source_id": source_id, **row, "identities": sorted(identities)}


def read_mirror(ref: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Read one sealed lawful mirror, never a protected source or payload."""
    require(set(ref) == {"id", "path", "sha256", "classes", "seal"}, "invalid mirror reference")
    require(isinstance(ref["id"], str) and ref["id"], "missing mirror id")
    require(
        isinstance(ref["classes"], list) and set(ref["classes"]) <= set(REQUIRED_CLASSES),
        "bad mirror classes",
    )
    require(ref["classes"], "mirror has no coverage class")
    require(ref["seal"] == "sidecar", "mirror must have sha256 sidecar")
    path = Path(ref["path"])
    _safe_source(path)
    raw = path.read_bytes()
    require(sha(raw) == ref["sha256"], "mirror hash mismatch")
    sidecar = Path(f"{path}.sha256")
    _safe_source(sidecar)
    require(sidecar.read_text().split()[0] == ref["sha256"], "mirror sidecar mismatch")
    payload = _MirrorScan(raw).parse()
    require(
        set(payload) == {"schema", "records"} and payload["schema"] == MIRROR_SCHEMA,
        "wrong mirror schema",
    )
    require(
        isinstance(payload["records"], list) and len(payload["records"]) <= MAX_ROWS_PER_FILE,
        "mirror row bound",
    )
    rows = [_row(row, ref["id"]) for row in payload["records"]]
    return rows, {
        "id": ref["id"],
        "path": str(path),
        "sha256": ref["sha256"],
        "classes": sorted(ref["classes"]),
        "row_count": len(rows),
    }


def read_prior_annex(ref: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Project the prior annex's admitted 3,024 collision rows only.

    This is an exact-source exception, not a generic receipt parser.  The
    annex's own access record says that label and outcome payloads were not
    opened; this function projects only its historical collision metadata.
    """
    require(set(ref) == {"path", "sha256"}, "invalid prior-annex reference")
    path = Path(ref["path"])
    _safe_source(path)
    raw = path.read_bytes()
    require(sha(raw) == ref["sha256"], "prior annex hash mismatch")
    payload = json.loads(raw)
    require(payload.get("schema") == ANNEX_SCHEMA, "wrong prior annex schema")
    prior = payload.get("allocation", {}).get("inspected_historical_allocations")
    require(isinstance(prior, list) and len(prior) == 3024, "prior annex historical count changed")
    rows = []
    for item in prior:
        require(
            set(item) == {"parent_id", "seed", "aliases", "source"},
            "foreign prior-annex record",
        )
        # The immutable prior metadata may repeat parent_id in its alias list.
        # Preserve collision coverage while canonicalizing that representation.
        aliases = [alias for alias in item["aliases"] if alias != item["parent_id"]]
        require(len(aliases) == len(set(aliases)), "duplicate prior-annex alias")
        rows.append(
            _row(
                {
                    "parent_id": item["parent_id"],
                    "seed": item["seed"],
                    "aliases": aliases,
                    "ancestry": [],
                    "scopes": ["training_development"],
                    "state": "completed",
                },
                "f5-r10a-prior-annex-3024",
            )
        )
    return rows, {
        "id": "f5-r10a-prior-annex-3024",
        "path": str(path),
        "sha256": ref["sha256"],
        "classes": ["training_development"],
        "row_count": len(rows),
        "projection": "allocation.inspected_historical_allocations only",
    }


def _scope(path: Path) -> str:
    """Classify an audit's intended exclusion scope from its registered location."""
    lowered = str(path).lower()
    if "serving" in lowered:
        return "serving"
    if "strict" in lowered:
        return "strict"
    if "round30" in lowered or "r30" in lowered:
        return "r30_1"
    if "f4" in lowered or "census" in lowered:
        return "f4_folds_census"
    if any(part in lowered for part in ("probe", "timing", "canary")):
        return "probe_canary_timing"
    if any(part in lowered for part in ("rejected", "reserved", "failed", "retry")):
        return "reserved_failed_aborted_retried_superseded"
    return "validation"


class _SeedScan:
    """Structural seed projection; strings and foreign scientific values stay opaque.

    Unlike a regexp over raw bytes, this cannot mistake a quoted outcome string
    containing ``\"seed\":123`` for an actual seed field. Keys are metadata;
    only unsigned integer values under seed fields are decoded. A forbidden
    subtree is traversed only for delimiters, without decoding its keys/values.
    """

    def __init__(self, raw: bytes, mirror_keys: set[str] | None = None) -> None:
        self.raw, self.index, self.values, self.nodes = raw, 0, [], 0
        self.mirror_keys = mirror_keys

    def _ws(self) -> None:
        while self.index < len(self.raw) and self.raw[self.index] in b" \n\r\t":
            self.index += 1

    def _take(self, char: int) -> None:
        self._ws()
        require(self.index < len(self.raw) and self.raw[self.index] == char, "metadata structure")
        self.index += 1

    def _string(self, key: bool = False) -> str | None:
        self._ws()
        start = self.index
        self._take(34)
        escaped = False
        while self.index < len(self.raw):
            char = self.raw[self.index]
            self.index += 1
            if escaped:
                escaped = False
            elif char == 92:
                escaped = True
            elif char == 34:
                if key:
                    try:
                        return json.loads(self.raw[start : self.index])
                    except Exception:
                        raise ValueError("metadata key encoding") from None
                return None
        raise ValueError("metadata unterminated string")

    def _value(self, seed: bool = False, inspect: bool = True, depth: int = 0) -> None:
        self.nodes += 1
        require(depth <= 64 and self.nodes <= 2_000_000, "metadata traversal bound")
        self._ws()
        require(self.index < len(self.raw), "metadata truncated")
        char = self.raw[self.index]
        if char == 34:
            self._string()
        elif char in (123, 91):
            closing = 125 if char == 123 else 93
            self.index += 1
            self._ws()
            if self.index < len(self.raw) and self.raw[self.index] == closing:
                self.index += 1
                return
            seen = set()
            while True:
                child_seed, child_inspect = seed, inspect
                if char == 123:
                    key = self._string(key=inspect)
                    self._take(58)
                    if inspect:
                        require(key not in seen, "duplicate metadata field")
                        seen.add(key)
                        if self.mirror_keys is not None:
                            require(key in self.mirror_keys, "foreign mirror field")
                        lowered = key.lower()
                        child_seed = "seed" in lowered and not any(
                            term in lowered for term in ("count", "sha", "namespace")
                        )
                        child_inspect = (
                            not any(
                                term in lowered
                                for term in (
                                    "outcome",
                                    "mass",
                                    "return",
                                    "reward",
                                    "score",
                                    "surviv",
                                    "death",
                                    "died",
                                    "margin",
                                    "diagnostic",
                                    "snapshot",
                                    "q_value",
                                )
                            )
                            and lowered != "q"
                        )
                self._value(child_seed, child_inspect, depth + 1)
                self._ws()
                if self.index < len(self.raw) and self.raw[self.index] == closing:
                    self.index += 1
                    break
                self._take(44)
        else:
            start = self.index
            while self.index < len(self.raw) and self.raw[self.index] not in b",]} \n\r\t":
                self.index += 1
            token = self.raw[start : self.index]
            require(bool(token), "metadata scalar")
            if inspect and seed and re.fullmatch(rb"0|[1-9][0-9]{0,19}", token):
                value = int(token)
                if value < 2**64:
                    self.values.append(value)

    def run(self) -> list[int]:
        self._value()
        self._ws()
        require(self.index == len(self.raw), "metadata trailing bytes")
        return self.values


def _seed_values(raw: bytes) -> list[int]:
    """Project raw uint64 seeds without decoding any other scalar values."""
    return _SeedScan(raw).run()


class _MirrorScan(_SeedScan):
    """Position/type-bound mirror parser; foreign values are never decoded."""

    def _array(self, reader: Any) -> list[Any]:
        self._take(91)
        result = []
        self._ws()
        if self.index < len(self.raw) and self.raw[self.index] == 93:
            self.index += 1
            return result
        while True:
            require(len(result) < MAX_ROWS_PER_FILE, "mirror array bound")
            result.append(reader())
            self._ws()
            if self.index < len(self.raw) and self.raw[self.index] == 93:
                self.index += 1
                return result
            self._take(44)

    def _object(self, readers: dict[str, Any]) -> dict[str, Any]:
        self._take(123)
        result = {}
        while True:
            key = self._string(key=True)
            require(key in readers and key not in result, "foreign or duplicate mirror field")
            self._take(58)
            result[key] = readers[key]()
            self._ws()
            if self.index < len(self.raw) and self.raw[self.index] == 125:
                self.index += 1
                require(set(result) == set(readers), "missing mirror field")
                return result
            self._take(44)

    def _uint(self) -> int:
        self._ws()
        start = self.index
        while self.index < len(self.raw) and self.raw[self.index] in b"0123456789":
            self.index += 1
        token = self.raw[start : self.index]
        require(bool(re.fullmatch(rb"0|[1-9][0-9]{0,19}", token)), "mirror seed syntax")
        value = int(token)
        require(value < 2**64, "mirror seed bound")
        return value

    def parse(self) -> dict[str, Any]:
        def string() -> str | None:
            return self._string(key=True)

        def string_list() -> list[Any]:
            return self._array(string)

        fields = {
            "parent_id": string,
            "seed": self._uint,
            "aliases": string_list,
            "ancestry": string_list,
            "scopes": string_list,
            "state": string,
        }
        result = self._object(
            {"schema": string, "records": lambda: self._array(lambda: self._object(fields))}
        )
        self._ws()
        require(self.index == len(self.raw), "mirror trailing bytes")
        return result


def discover_repository(roots: list[str]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Boundedly catalog named audit files, decoding only preflight-safe metadata."""
    paths: list[Path] = []
    for root_name in roots:
        root = Path(root_name)
        require(
            root.is_absolute()
            and not forbidden(root)
            and not any(part.is_symlink() for part in (root, *root.parents))
            and root.is_dir(),
            "invalid repository audit root",
        )
        for directory, dirnames, filenames in os.walk(root, topdown=True, followlinks=False):
            dirnames[:] = [
                name
                for name in dirnames
                if not forbidden(Path(directory, name)) and not Path(directory, name).is_symlink()
            ]
            paths.extend(
                Path(directory, name)
                for name in filenames
                if name.endswith(".json")
                and AUDIT_NAME.search(name)
                and not Path(directory, name).is_symlink()
            )
    paths = sorted({path for path in paths if not forbidden(path)}, key=str)
    require(len(paths) <= MAX_FILES, "repository audit file bound")
    register, rows = [], []
    for path in paths:
        if path.is_symlink():
            register.append(
                {
                    "path": str(path),
                    "sha256": None,
                    "size": None,
                    "scope": _scope(path),
                    "status": "UNRESOLVED_SYMLINK",
                    "seed_count": None,
                }
            )
            continue
        if path.stat().st_size > MAX_BYTES_PER_FILE:
            register.append(
                {
                    "path": str(path),
                    "sha256": None,
                    "size": path.stat().st_size,
                    "scope": _scope(path),
                    "status": "UNRESOLVED_OVERSIZE",
                    "seed_count": None,
                }
            )
            continue
        _safe_source(path)
        raw = path.read_bytes()
        source = {"path": str(path), "sha256": sha(raw), "size": len(raw), "scope": _scope(path)}
        if FORBIDDEN_KEY.search(raw):
            register.append(
                {**source, "status": "UNRESOLVED_MIXED_OR_OUTCOME_FIELDS", "seed_count": None}
            )
            continue
        seeds = sorted(set(_seed_values(raw)))
        lifecycle = "rejected" if "rejected" in str(path).lower() else "registered"
        register.append(
            {
                **source,
                "status": "ADMITTED_LEXICAL_SEED_METADATA",
                "seed_count": len(seeds),
                "lifecycle": lifecycle,
            }
        )
        for index, seed in enumerate(seeds):
            parent = f"repo-audit:{path.relative_to(path.anchor)}#{index:06d}"
            rows.append(
                {
                    "source_id": str(path),
                    "parent_id": parent,
                    "seed": seed,
                    "aliases": [],
                    "ancestry": [],
                    "ancestry_status": "UNKNOWN",
                    "scopes": [source["scope"]],
                    "state": lifecycle,
                    "identities": [parent],
                }
            )
    return register, rows


def _attestation(attestation: Any, coverage_hash: str) -> tuple[bool, str | None]:
    """Accept only an existing, bound custodian attestation; never fabricate one."""
    if attestation is None:
        return False, "custodian completeness attestation absent"
    if not isinstance(attestation, dict):
        return False, "custodian completeness attestation malformed"
    expected = {"attestor", "statement", "catalog_sha256", "covered_classes"}
    if set(attestation) != expected:
        return False, "custodian completeness attestation fields invalid"
    if not isinstance(attestation["attestor"], str) or not attestation["attestor"]:
        return False, "custodian identity absent"
    if attestation["statement"] != "COMPLETE_F5_HISTORICAL_COLLISION_REGISTER":
        return False, "custodian did not attest completeness"
    if attestation["catalog_sha256"] != coverage_hash:
        return False, "custodian attestation not bound to source catalog"
    if set(attestation["covered_classes"]) != set(REQUIRED_CLASSES):
        return False, "custodian attestation omits required class"
    return True, None


def build(plan: dict[str, Any]) -> dict[str, Any]:
    """Build a bounded catalog and run all 684 only after completeness is proven."""
    require(
        set(plan)
        == {"schema", "mirrors", "prior_annex", "repository_roots", "custodian_attestation"},
        "invalid catalog plan",
    )
    require(plan["schema"] == SCHEMA, "wrong catalog plan schema")
    require(
        isinstance(plan["mirrors"], list) and len(plan["mirrors"]) <= MAX_FILES, "mirror file bound"
    )
    records: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    if plan["prior_annex"] is not None:
        rows, binding = read_prior_annex(plan["prior_annex"])
        records.extend(rows)
        bindings.append(binding)
    source_register, repository_rows = discover_repository(plan["repository_roots"])
    records.extend(repository_rows)
    for ref in plan["mirrors"]:
        rows, binding = read_mirror(ref)
        records.extend(rows)
        bindings.append(binding)
    require(len({binding["id"] for binding in bindings}) == len(bindings), "duplicate mirror id")
    by_parent: dict[str, dict[str, Any]] = {}
    identities: set[str] = set()
    for row in records:
        require(row["parent_id"] not in by_parent, "duplicate parent record")
        require(
            not (identities & set(row["identities"])),
            "alias or ancestry collision in historical catalog",
        )
        by_parent[row["parent_id"]] = row
        identities.update(row["identities"])
    observed_classes = {scope for row in records for scope in row["scopes"]}
    missing_classes = sorted(set(REQUIRED_CLASSES) - observed_classes)
    unresolved_sources = [
        source for source in source_register if not source["status"].startswith("ADMITTED")
    ]
    unknown_ancestry = any(row.get("ancestry_status") == "UNKNOWN" for row in records)
    coverage_hash = sha(
        canonical({"bindings": bindings, "source_register": source_register, "records": records})
    )
    attested, reason = _attestation(plan["custodian_attestation"], coverage_hash)
    complete = not missing_classes and not unresolved_sources and not unknown_ancestry and attested
    audit: dict[str, Any] = {
        "status": "HOLD_INCOMPLETE" if not complete else "READY_TO_AUDIT",
        "reason": reason
        or ("missing coverage classes: " + ", ".join(missing_classes) if missing_classes else None),
        "all_684_executed": False,
        "accepted_round": None,
        "accepted_allocation_sha256": None,
        "provisional_round": None,
        "provisional_allocation_sha256": None,
        "rejected_rounds": [],
    }
    # A partial union is still useful for a clearly-labelled provisional whole-bank
    # collision result.  It can never accept a round; completeness governs that.
    if records:
        historical = [
            {"parent_id": r["parent_id"], "seed": r["seed"], "aliases": r["identities"]}
            for r in records
        ]
        for round_index in range(1000):
            candidates = annex.allocate(round_index)
            reasons = annex.collision_audit(candidates, historical)
            if not reasons:
                audit.update(
                    {
                        "all_684_executed": True,
                        "provisional_round": round_index,
                        "provisional_allocation_sha256": sha(canonical(candidates)),
                    }
                )
                if complete:
                    audit.update(
                        {
                            "status": "CLEAR",
                            "accepted_round": round_index,
                            "accepted_allocation_sha256": sha(canonical(candidates)),
                        }
                    )
                else:
                    audit["status"] = "PROVISIONAL_HOLD_INCOMPLETE"
                break
            audit["rejected_rounds"].append({"round": round_index, "reasons": reasons})
        else:
            audit.update({"status": "HOLD_ALL_ROUNDS_COLLIDE", "all_684_executed": True})
    return {
        "schema": SCHEMA,
        "disposition": "HOLD" if not complete else audit["status"],
        "bounds": {
            "max_files": MAX_FILES,
            "max_bytes_per_file": MAX_BYTES_PER_FILE,
            "max_rows_per_file": MAX_ROWS_PER_FILE,
        },
        "required_classes": list(REQUIRED_CLASSES),
        "protected_allocations": {
            "D4096": "external lawful mirror required; unopened",
            "t34-round0-study-01": "external lawful mirror required; unopened",
        },
        "source_bindings": bindings,
        "repository_source_register": source_register,
        "coverage_hash": coverage_hash,
        "observed_classes": sorted(observed_classes),
        "unresolved_source_count": len(unresolved_sources),
        "unknown_ancestry": unknown_ancestry,
        "catalog_source_bindings_sha256": sha(canonical(bindings)),
        "records": sorted(records, key=lambda r: (r["parent_id"], r["seed"])),
        "record_count": len(records),
        "unique_seed_count": len({row["seed"] for row in records}),
        "missing_classes": missing_classes,
        "custodian_completeness_attestation": plan["custodian_attestation"],
        "complete": complete,
        "audit": audit,
    }


def write(path: Path, report: dict[str, Any]) -> None:
    """Preflight all outputs, publish create-only, and seal completion last."""
    require(not forbidden(path), "protected publication path")
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink publication")
    report = dict(report)
    records = report.pop("records")
    shards, files = [], {}
    for number, start in enumerate(range(0, len(records), 2_000), start=1):
        shard_path = path.with_name(f"collision-records-{number:03d}.json")
        raw_shard = canonical(
            {"schema": "f5-r10a-collision-records/v1", "records": records[start : start + 2_000]}
        )
        require(len(raw_shard) <= 2_000_000, "collision record shard exceeds byte bound")
        files[shard_path] = raw_shard
        digest = sha(raw_shard)
        files[Path(f"{shard_path}.sha256")] = f"{digest}  {shard_path.name}\n".encode()
        shards.append(
            {
                "path": shard_path.name,
                "sha256": digest,
                "record_count": len(records[start : start + 2_000]),
            }
        )
    report["record_shards"] = shards
    raw = canonical(report)
    require(len(raw) <= 2_000_000, "catalog output exceeds byte bound")
    files[path] = raw
    files[Path(f"{path}.sha256")] = f"{sha(raw)}  {path.name}\n".encode()
    marker = Path(f"{path}.complete.json")
    files[marker] = canonical(
        {
            "schema": "f5-collision-publication/v1",
            "files": {str(p.name): sha(value) for p, value in files.items()},
        }
    )
    require(
        not any(p.exists() or p.is_symlink() for p in files), "create-only collision output exists"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    for target, value in files.items():
        descriptor, temporary = tempfile.mkstemp(prefix=".f5-collision-", dir=target.parent)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(value)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(temporary, target)
        finally:
            os.unlink(temporary)


def main() -> None:
    """Freeze bounded repository discovery plus the explicit metadata inputs."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    plan_path = Path(args.plan)
    _safe_source(plan_path)
    write(Path(args.out), build(json.loads(plan_path.read_bytes())))


if __name__ == "__main__":
    main()
