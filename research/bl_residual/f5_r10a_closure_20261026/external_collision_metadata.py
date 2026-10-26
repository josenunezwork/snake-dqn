"""Audit explicit historical FRP exclusion metadata; never follow payload references.

The seven admitted bare integer arrays are contemporaneous collision metadata,
not labels. Signed seeds are preserved exactly. This adds evidence, not a
completeness certificate: protected mirrors and ancestry remain mandatory HOLDs.
"""

from __future__ import annotations

import argparse
import hashlib
import heapq
import json
import os
import re
import tempfile
import time
from array import array
from pathlib import Path
from typing import Any, Iterator

from research.bl_residual import f5_r10a_annex as original

MAX_INPUT_BYTES = 64 * 1024 * 1024
MAX_SOURCE_ROWS = 1_200_000
MAX_TOTAL_ROWS = 8_000_000
MAX_SECONDS = 240
SHARD_ROWS = 75_000
MAX_OUTPUT_BYTES = 2_000_000
PROTECTED = ("d4096", "t34-round0-study-01")
REGISTER_SHA256 = "8cd2d43f3c19ed51c74b9ee0a2bf328eaad463095c8032dc0e8b5653e4eeee08"
TOKEN = re.compile(rb"\s*(-?(?:0|[1-9][0-9]*))\s*([,\]])")
FROZEN = (
    (
        "frp-v2-20261004/namespaces",
        425901,
        "09e4cb22e569ceea6b8af7d3fb335f2e59e791a749b07d0151108c28548d3b72",
    ),
    (
        "frp-v3-20261005/namespaces",
        586062,
        "4b0f33775544e36e2ccbe6a5d8ab35b81de40565c4328dc8abddd3cea4bc1edf",
    ),
    (
        "frp-v4-20261007/namespaces",
        714256,
        "d6223bf5c0bd03cb30bdbc2c455208800e30e880ba395bf0cb5d2f6c433f6d2a",
    ),
    (
        "frp-v5h-20261008/namespaces",
        872512,
        "8a06d278ae51a71c3f71eff8afa120b1b9fc57f22ff2cac138302be29cf68cd4",
    ),
    (
        "frp-v5h-20261008/namespaces-r2",
        872513,
        "04dc76fc95458e212a595772de86c8a8addfa0499ec480b83a0f8ac80e5ecf96",
    ),
    (
        "frp-v5s-20261009/namespaces",
        1002787,
        "76b3b1fbb601115056d42e7198048b650ed7350ff8042a266cc830d97b9ddd88",
    ),
    (
        "frp-v5s2-20261009/namespaces",
        1135019,
        "30726df2c26da9eb26a37659c3f525ec63bf3618495de5b380bc8e426edbd885",
    ),
)


def canonical(value: Any) -> bytes:
    """Canonical JSON, shared with the unchanged allocator's audit hash."""
    return original.encoded(value)


def digest(raw: bytes) -> str:
    """SHA256 of admitted bytes."""
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, message: str) -> None:
    """Fail closed with a non-payload error."""
    if not condition:
        raise ValueError(message)


def check_path(path: Path, bound: int) -> None:
    """Reject protected paths and escapes before any file access."""
    require(path.is_absolute(), "absolute metadata path required")
    require(not any(x in str(path).lower() for x in PROTECTED), "protected source refused")
    require(not any(x.is_symlink() for x in (path, *path.parents)), "symlink refused")
    require(path.is_file() and path.stat().st_size <= bound, "metadata size/file guard")


def safe_read(path: Path, expected: str, bound: int = MAX_INPUT_BYTES) -> bytes:
    """Read an explicit metadata path with hash, size and no-symlink guards."""
    check_path(path, bound)
    with path.open("rb") as handle:
        raw = handle.read(bound + 1)
    require(len(raw) <= bound and digest(raw) == expected, "metadata seal/size mismatch")
    return raw


class SignedSeeds:
    """Compact exact integers spanning negative and full uint64 seed domains."""

    def __init__(self) -> None:
        self.negative = array("Q")
        self.positive = array("Q")

    def __len__(self) -> int:
        return len(self.negative) + len(self.positive)

    def __iter__(self) -> Iterator[int]:
        yield from (-value for value in self.negative)
        yield from self.positive

    def append(self, number: int) -> None:
        (self.negative if number < 0 else self.positive).append(abs(number))


def integer_array(raw: bytes, expected_count: int, deadline: float) -> SignedSeeds:
    """Project a sorted, unique signed integer-only array without JSON decoding."""
    require(len(raw) <= MAX_INPUT_BYTES, "integer source byte guard")
    require(0 <= expected_count <= MAX_SOURCE_ROWS, "source row guard")
    start = len(raw) - len(raw.lstrip())
    require(raw[start : start + 1] == b"[", "integer array schema refused")
    position = start + 1
    values = SignedSeeds()
    previous = None
    if raw[position:].strip() == b"]":
        require(expected_count == 0, "integer count mismatch")
        return values
    while True:
        match = TOKEN.match(raw, position)
        require(match is not None, "integer array syntax refused")
        token = match.group(1)
        require(len(token) <= 21, "integer range guard")
        number = int(token)
        require(-(2**64) < number < 2**64, "integer range guard")
        require(previous is None or number > previous, "duplicate/unsorted integer seed")
        previous = number
        values.append(number)
        require(len(values) <= expected_count, "integer row guard")
        if len(values) % 4096 == 0:
            require(time.monotonic() <= deadline, "metadata time guard")
        position = match.end()
        if match.group(2) == b"]":
            require(not raw[position:].strip(), "trailing metadata bytes")
            break
    require(len(values) == expected_count, "integer count mismatch")
    return values


def union_values(sources: list[Any]) -> Iterator[int]:
    """Merge strict sorted signed arrays with deterministic deduplication."""
    previous = None
    for number in heapq.merge(*sources):
        if number != previous:
            yield number
            previous = number


def write_sealed(path: Path, value: Any) -> dict[str, Any]:
    """Publish bounded immutable metadata and sidecar, with no replacement."""
    check_output_path(path)
    raw = canonical(value)
    require(len(raw) < MAX_OUTPUT_BYTES, "output byte guard")
    sidecar = Path(f"{path}.sha256")
    check_output_path(sidecar)
    require(not path.exists() and not sidecar.exists(), "existing output refused")
    sidecar_raw = f"{digest(raw)}  {path.name}\n".encode("ascii")
    atomic_create(path, raw)
    atomic_create(sidecar, sidecar_raw)
    return {
        "path": path.name,
        "sha256": digest(raw),
        "size": len(raw),
        "sidecar": {"path": sidecar.name, "sha256": digest(sidecar_raw)},
    }


def check_output_path(path: Path) -> None:
    """Reject excluded destinations and symlink parents before creating anything."""
    require(path.is_absolute(), "absolute output path required")
    require(not any(x in str(path).lower() for x in PROTECTED), "protected output refused")
    require(not any(x.is_symlink() for x in (path, *path.parents)), "output symlink refused")


def atomic_create(path: Path, raw: bytes) -> None:
    """Fsync then atomically link a fresh file; never overwrite an existing inode."""
    check_output_path(path)
    descriptor, temporary = tempfile.mkstemp(prefix=".metadata-", dir=path.parent)
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.link(temporary_path, path)
        directory_descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    finally:
        temporary_path.unlink(missing_ok=True)


def load_repository(catalog_path: Path) -> tuple[list[dict], dict]:
    """Load frozen metadata catalog shards, without consulting scientific inputs."""
    check_path(catalog_path, MAX_OUTPUT_BYTES)
    sidecar = Path(f"{catalog_path}.sha256")
    check_path(sidecar, 512)
    expected = sidecar.read_text().split()[0]
    catalog_raw = safe_read(catalog_path, expected, MAX_OUTPUT_BYTES)
    catalog = json.loads(catalog_raw)
    rows = []
    bindings = []
    for shard in catalog["record_shards"]:
        name = shard["path"]
        require(Path(name).name == name, "repository shard escape")
        path = catalog_path.parent / name
        raw = safe_read(path, shard["sha256"], MAX_OUTPUT_BYTES)
        payload = json.loads(raw)
        require(payload["schema"] == "f5-r10a-collision-records/v1", "shard schema refused")
        require(len(payload["records"]) == shard["record_count"], "shard count mismatch")
        rows.extend(payload["records"])
        bindings.append({"path": str(path), "sha256": shard["sha256"]})
    require(len(rows) == catalog["record_count"], "repository count mismatch")
    return rows, {
        "path": str(catalog_path),
        "sha256": digest(catalog_raw),
        "shards": bindings,
        "record_count": len(rows),
        "complete": catalog["complete"],
        "unknown_ancestry": catalog["unknown_ancestry"],
        "missing_classes": catalog["missing_classes"],
    }


def audit_union(repository: list[dict], external_hits: list[dict], rounds: int = 1000) -> dict:
    """Call the unchanged whole-bank allocator/auditor, pruning only impossible hits.

    Exact equivalence follows from collision_audit's only two comparisons: raw
    seed equality OR exact parent membership in old aliases. Historical records
    with neither comparison possible cannot produce a reason. No scope is
    silently treated as fresh-bank permission: all observed seeds are excluded.
    """
    require(1 <= rounds <= 1000, "round bound")
    rejected = []
    for round_index in range(rounds):
        bank = original.allocate(round_index)
        require(len(bank) == 684, "whole-bank size changed")
        seeds = {row["seed"] for row in bank}
        parents = {row["parent_id"] for row in bank}
        possible = [
            row
            for row in (*repository, *external_hits)
            if row["seed"] in seeds or parents.intersection(row["aliases"])
        ]
        reasons = original.collision_audit(bank, possible)
        if not reasons:
            return {
                "status": "PROVISIONAL_HOLD_INCOMPLETE",
                "all_684_executed": True,
                "provisional_round": round_index,
                "provisional_allocation_sha256": digest(canonical(bank)),
                "accepted_round": None,
                "accepted_allocation_sha256": None,
                "rejected_rounds": rejected,
                "reason": "missing protected mirrors, ancestry and custodian completeness",
            }
        rejected.append({"round": round_index, "reasons": reasons})
    return {
        "status": "HOLD_NO_CLEAR_ROUND",
        "accepted_round": None,
        "accepted_allocation_sha256": None,
        "rejected_rounds": rejected,
    }


def build(repo: Path, artifacts: Path, output: Path, catalog: Path) -> dict:
    """Read only the seven frozen external exclusions and named repository shards."""
    deadline = time.monotonic() + MAX_SECONDS
    tool_source_pin = digest(Path(__file__).read_bytes())
    allocator_source_pin = digest(Path(original.__file__).read_bytes())
    check_output_path(output)
    require(not output.exists(), "existing output directory refused")
    output.mkdir(parents=True, exist_ok=False)
    register = repo / "research/t34_round0/SEED_AUDIT_ROUND0.json"
    register_raw = safe_read(register, REGISTER_SHA256, MAX_OUTPUT_BYTES)
    registered = json.loads(register_raw)["historical_frp_exclusions"]
    refs = []
    for suffix, count, pin in FROZEN:
        path = artifacts / suffix / "training_exclusions.json"
        ref = {"path": str(path), "sha256": pin, "count": count}
        require(ref in registered, "frozen external reference absent from original register")
        refs.append(ref)
    require(sum(ref["count"] for ref in refs) <= MAX_TOTAL_ROWS, "total row guard")
    plan = {
        "schema": "f5-external-collision-access-plan/v1",
        "sources": refs,
        "register": {"path": str(register), "sha256": digest(register_raw)},
        "tool_source_sha256": tool_source_pin,
        "unchanged_allocator_source_sha256": allocator_source_pin,
        "bounds": {
            "files": 7,
            "input_bytes_each": MAX_INPUT_BYTES,
            "rows_each": MAX_SOURCE_ROWS,
            "rows_total": MAX_TOTAL_ROWS,
            "seconds": MAX_SECONDS,
            "output_bytes_each": MAX_OUTPUT_BYTES,
            "memory_design_limit_mib": 512,
        },
        "allowed_schema": "bare sorted unique signed integer JSON array only",
        "fields": ["raw_physical_seed"],
        "automatic_reference_traversal": False,
        "protected_sources": "NEVER OPENED, including metadata",
    }
    plan_binding = write_sealed(output / "access-plan.json", plan)
    repository, repository_binding = load_repository(catalog)
    # Freeze the entire 0..999 candidate domain to make pruning exact and auditable.
    candidate_seeds = set()
    for round_index in range(1000):
        candidate_seeds.update(row["seed"] for row in original.allocate(round_index))
    sources = []
    source_bindings = []
    hits = []
    for source_index, ref in enumerate(refs):
        path = Path(ref["path"])
        raw = safe_read(path, ref["sha256"])
        values = integer_array(raw, ref["count"], deadline)
        del raw
        sources.append(values)
        source_bindings.append(
            {
                **ref,
                "source_id": FROZEN[source_index][0],
                "scope": "training_exclusion",
                "ancestry": "UNKNOWN",
                "negative_seed_count": sum(x < 0 for x in values),
            }
        )
        for index, seed in enumerate(values):
            if seed in candidate_seeds:
                hits.append(
                    {
                        "parent_id": f"frp-exclusion:{source_index}:{index}",
                        "seed": seed,
                        "aliases": [],
                        "source_index": source_index,
                        "source_allocation_index": index,
                    }
                )
    repository_seeds = sorted({row["seed"] for row in repository})
    union_digest = hashlib.sha256()
    shards, pending, count, negatives = [], [], 0, 0
    for seed in union_values([*sources, repository_seeds]):
        # Canonical digest preserves full signed/uint64 domain as decimal ASCII lines.
        union_digest.update(f"{seed}\n".encode("ascii"))
        count += 1
        negatives += seed < 0
        pending.append(seed)
        if len(pending) == SHARD_ROWS:
            shards.append(
                write_sealed(output / f"seed-union-{len(shards) + 1:03d}.json", {"seeds": pending})
            )
            pending = []
    if pending:
        shards.append(
            write_sealed(output / f"seed-union-{len(shards) + 1:03d}.json", {"seeds": pending})
        )
    # Preserve all repository identity aliases; raw FRP exclusion arrays have none.
    history = [
        {"parent_id": row["parent_id"], "seed": row["seed"], "aliases": row["identities"]}
        for row in repository
    ]
    audit = audit_union(history, hits)
    report = {
        "schema": "f5-external-collision-evidence/v1",
        "disposition": "HOLD",
        "complete": False,
        "source_bindings": source_bindings,
        "access_plan_sha256": digest(canonical(plan)),
        "repository_binding": repository_binding,
        "source_rows_total": sum(len(values) for values in sources),
        "union_unique_seed_count": count,
        "union_negative_seed_count": negatives,
        "canonical_union_encoding": "sorted unique raw integers, decimal ASCII plus LF per seed",
        "canonical_union_sha256": union_digest.hexdigest(),
        "seed_shards": shards,
        "candidate_domain": {
            "rounds": 1000,
            "rows_per_round": 684,
            "unique_seed_count": len(candidate_seeds),
        },
        "external_candidate_domain_hits": hits,
        "audit": audit,
        "ancestry_aliases": "UNKNOWN for raw exclusion arrays; no inferred ancestry",
        "scope_rule": "raw seed union conservatively excludes both scopes; training and "
        "fresh-bank provenance remain distinct and unresolved",
        "missing": [
            "custodian campaign/package allocation-register completeness attestation",
            "ultimate ancestry and cross-namespace aliases",
            "lawful external protected-allocation mirrors or custodian certificate",
            "unresolved allocation classes from repository catalog",
        ],
        "access": {
            "outcomes_opened": False,
            "protected_sources_opened": False,
            "generic_dependency_traversal": False,
            "colab_dispatch": False,
        },
    }
    require(time.monotonic() <= deadline, "metadata time guard")
    require(digest(Path(__file__).read_bytes()) == tool_source_pin, "tool source changed")
    require(
        digest(Path(original.__file__).read_bytes()) == allocator_source_pin,
        "allocator source changed",
    )
    report_binding = write_sealed(output / "external-collision-report.json", report)
    write_sealed(
        output / "COMPLETED.json",
        {
            "schema": "f5-external-collision-completion/v1",
            "tool_source_sha256": tool_source_pin,
            "unchanged_allocator_source_sha256": allocator_source_pin,
            "artifacts": [plan_binding, *shards, report_binding],
            "publication": "atomic no-replace; file and directory fsync; marker published last",
            "disposition": "HOLD",
        },
    )
    return report


def main() -> None:
    """Run the explicit metadata-only preparation; no execution or release authority."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, type=Path)
    parser.add_argument("--artifacts", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--repository-catalog", required=True, type=Path)
    args = parser.parse_args()
    report = build(args.repo, args.artifacts, args.output, args.repository_catalog)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "disposition",
                    "union_unique_seed_count",
                    "canonical_union_sha256",
                    "audit",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
