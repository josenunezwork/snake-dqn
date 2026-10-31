"""Idempotent shard units, path/size/hash verified ledger, and Mac aggregator."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

SCHEMA_UNIT = "devicesim-v2-parity-unit/v1"
SCHEMA_LEDGER = "devicesim-v2-parity-ledger/v1"
SCHEMA_SUMMARY = "devicesim-v2-parity-summary/v1"
UNIT_KIND = "shard"


@dataclass(frozen=True)
class ShardSpec:
    unit_id: str
    shard_index: int
    num_envs: int
    ticks: int
    ego_check_stride: int
    max_capacity: int
    mature_fraction: float
    check_ct_sim: bool


LANE_SPECS: dict[str, tuple[ShardSpec, ...]] = {
    "preflight": (
        ShardSpec(
            unit_id="shard-preflight-00",
            shard_index=0,
            num_envs=4,
            ticks=8,
            ego_check_stride=4,
            max_capacity=256,
            mature_fraction=0.5,
            check_ct_sim=True,
        ),
    ),
    "canary": (
        ShardSpec(
            unit_id="shard-canary-00",
            shard_index=0,
            num_envs=32,
            ticks=64,
            ego_check_stride=16,
            max_capacity=1600,
            mature_fraction=0.5,
            check_ct_sim=True,
        ),
    ),
    "main": tuple(
        ShardSpec(
            unit_id=f"shard-{idx:02d}",
            shard_index=idx,
            num_envs=64,
            ticks=1000,
            ego_check_stride=25,
            max_capacity=1600,
            mature_fraction=0.5,
            check_ct_sim=True,
        )
        for idx in range(4)
    ),
}


def specs_for_lane(lane: str) -> tuple[ShardSpec, ...]:
    if lane not in LANE_SPECS:
        raise ValueError(f"unknown lane {lane!r}; expected one of {sorted(LANE_SPECS)}")
    return LANE_SPECS[lane]


def allocation_for_lane(lane: str) -> tuple[str, ...]:
    return tuple(spec.unit_id for spec in specs_for_lane(lane))


def namespace_for(commit: str, lane: str) -> str:
    clean = commit.strip()
    if not clean:
        raise ValueError("commit must be nonempty")
    specs_for_lane(lane)
    return f"devicesim-v2-parity/v1/{clean}/{lane}"


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _fsync_dir(directory: Path) -> None:
    fd = os.open(str(directory), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def write_bytes_create_only(target: Path, payload: bytes) -> None:
    """Write bytes atomically and create-only via temp + fsync + hardlink."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        raise FileExistsError(f"refusing to overwrite immutable artifact {target}")
    nonce = f"{os.getpid()}-{hashlib.sha256(os.urandom(16)).hexdigest()[:12]}"
    tmp = target.parent / f".{target.name}.tmp-{nonce}"
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.link(tmp, target)
    finally:
        tmp.unlink(missing_ok=True)
    _fsync_dir(target.parent)


def publish_immutable_json(
    target: Path,
    document: dict[str, Any],
    *,
    unit_id: str | None = None,
    namespace: str | None = None,
    relpath: str | None = None,
) -> dict[str, Any]:
    """Publish JSON payload, then .sha256 sidecar, then .complete marker last."""
    encoded = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
    digest = sha256_bytes(encoded)
    size = len(encoded)
    sha_path = target.with_name(target.name + ".sha256")
    complete_path = target.with_name(target.name + ".complete")
    for candidate in (target, sha_path, complete_path):
        if candidate.exists():
            raise FileExistsError(f"refusing to overwrite immutable artifact {candidate}")

    write_bytes_create_only(target, encoded)
    write_bytes_create_only(sha_path, f"{digest}  {target.name}\n".encode("utf-8"))
    marker: dict[str, Any] = {
        "payload_relpath": relpath or target.name,
        "sha256": digest,
        "size": size,
    }
    if unit_id is not None:
        marker["unit_id"] = unit_id
    if namespace is not None:
        marker["namespace"] = namespace
    marker_bytes = (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode("utf-8")
    write_bytes_create_only(complete_path, marker_bytes)
    return {
        "unit_id": unit_id or target.stem,
        "payload_relpath": relpath or target.name,
        "sha256": digest,
        "size": size,
    }


def verify_published_json(
    target: Path,
    *,
    expected_unit_id: str | None = None,
    expected_namespace: str | None = None,
    expected_relpath: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Verify completion marker, SHA sidecar, and JSON payload; return (ledger_entry, doc)."""
    sha_path = target.with_name(target.name + ".sha256")
    complete_path = target.with_name(target.name + ".complete")
    if not complete_path.is_file():
        raise FileNotFoundError(f"missing completion marker {complete_path}")
    if not sha_path.is_file():
        raise FileNotFoundError(f"missing sha256 sidecar {sha_path}")
    if not target.is_file():
        raise FileNotFoundError(f"missing payload {target}")

    marker = json.loads(complete_path.read_text(encoding="utf-8"))
    sha_tokens = sha_path.read_text(encoding="utf-8").strip().split()
    if len(sha_tokens) < 1:
        raise ValueError(f"malformed sha256 sidecar {sha_path}")
    sidecar_sha = sha_tokens[0]
    raw = target.read_bytes()
    actual_sha = sha256_bytes(raw)
    actual_size = len(raw)

    if actual_sha != sidecar_sha or actual_sha != marker.get("sha256"):
        raise ValueError(
            f"sha256 mismatch for {target}: payload={actual_sha}, "
            f"sidecar={sidecar_sha}, marker={marker.get('sha256')}"
        )
    if actual_size != int(marker.get("size", -1)):
        raise ValueError(
            f"size mismatch for {target}: payload={actual_size}, marker={marker.get('size')}"
        )
    if expected_unit_id is not None and marker.get("unit_id") != expected_unit_id:
        raise ValueError(
            f"unit_id mismatch for {target}: "
            f"marker={marker.get('unit_id')!r} != {expected_unit_id!r}"
        )
    if expected_namespace is not None and marker.get("namespace") != expected_namespace:
        raise ValueError(
            f"foreign namespace in marker {complete_path}: "
            f"{marker.get('namespace')!r} != {expected_namespace!r}"
        )
    if expected_relpath is not None and marker.get("payload_relpath") != expected_relpath:
        raise ValueError(
            f"payload_relpath mismatch in marker {complete_path}: "
            f"{marker.get('payload_relpath')!r} != {expected_relpath!r}"
        )
    doc = json.loads(raw.decode("utf-8"))
    if expected_namespace is not None and doc.get("namespace") != expected_namespace:
        raise ValueError(
            f"foreign namespace in payload {target}: "
            f"{doc.get('namespace')!r} != {expected_namespace!r}"
        )
    if expected_unit_id is not None and doc.get("unit_id") != expected_unit_id:
        raise ValueError(
            f"unit_id mismatch in payload {target}: "
            f"{doc.get('unit_id')!r} != {expected_unit_id!r}"
        )
    entry = {
        "unit_id": expected_unit_id or str(marker.get("unit_id") or target.stem),
        "payload_relpath": str(marker["payload_relpath"]),
        "sha256": actual_sha,
        "size": actual_size,
    }
    return entry, doc


def empty_ledger(*, namespace: str, lane: str) -> dict[str, Any]:
    return {
        "schema": SCHEMA_LEDGER,
        "namespace": namespace,
        "lane": lane,
        "unit_kind": UNIT_KIND,
        "allocation": list(allocation_for_lane(lane)),
        "skip_verified": True,
        "verified_units": [],
    }


def validate_ledger_document(
    doc: dict[str, Any],
    *,
    expected_namespace: str,
    expected_lane: str,
) -> dict[str, dict[str, Any]]:
    """Validate a path/size/hash-only verified ledger and return unit_id -> entry."""
    if doc.get("schema") != SCHEMA_LEDGER:
        raise ValueError(f"unexpected ledger schema {doc.get('schema')!r}")
    if doc.get("namespace") != expected_namespace:
        raise ValueError(
            f"refusing foreign ledger namespace {doc.get('namespace')!r}; "
            f"expected {expected_namespace!r}"
        )
    if doc.get("lane") != expected_lane:
        raise ValueError(
            f"refusing foreign ledger lane {doc.get('lane')!r}; expected {expected_lane!r}"
        )
    if doc.get("unit_kind") != UNIT_KIND:
        raise ValueError(f"unexpected ledger unit_kind {doc.get('unit_kind')!r}")
    if doc.get("skip_verified") is not True:
        raise ValueError("ledger skip_verified must be boolean True")
    expected_alloc = list(allocation_for_lane(expected_lane))
    if doc.get("allocation") != expected_alloc:
        raise ValueError(
            f"refusing foreign allocation {doc.get('allocation')!r}; expected {expected_alloc!r}"
        )
    raw_units = doc.get("verified_units")
    if not isinstance(raw_units, list):
        raise ValueError("ledger verified_units must be a list")

    allowed_keys = {"unit_id", "payload_relpath", "sha256", "size"}
    by_id: dict[str, dict[str, Any]] = {}
    allowed_ids = set(expected_alloc)
    for item in raw_units:
        if not isinstance(item, dict):
            raise ValueError("ledger verified_units entries must be objects")
        extra_keys = set(item) - allowed_keys
        missing_keys = allowed_keys - set(item)
        if extra_keys or missing_keys:
            raise ValueError(
                f"ledger entries must contain path/size/hash keys only; "
                f"missing={sorted(missing_keys)}, extra={sorted(extra_keys)}"
            )
        uid = item["unit_id"]
        relpath = item["payload_relpath"]
        digest = item["sha256"]
        size = item["size"]
        if uid not in allowed_ids:
            raise ValueError(f"unknown unit_id {uid!r} outside frozen allocation")
        if uid in by_id:
            raise ValueError(f"duplicate unit_id {uid!r} in verified ledger")
        expected_rel = f"shards/{uid}.json"
        if relpath != expected_rel:
            raise ValueError(f"unexpected payload_relpath {relpath!r}; expected {expected_rel!r}")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(ch not in "0123456789abcdef" for ch in digest)
        ):
            raise ValueError(f"invalid sha256 {digest!r} for unit {uid!r}")
        if not isinstance(size, int) or isinstance(size, bool) or size <= 0:
            raise ValueError(f"invalid byte size {size!r} for unit {uid!r}")
        by_id[uid] = {
            "unit_id": uid,
            "payload_relpath": relpath,
            "sha256": digest,
            "size": size,
        }
    return by_id


def load_verified_ledger(
    path: Path,
    *,
    expected_namespace: str,
    expected_lane: str,
) -> dict[str, dict[str, Any]]:
    doc = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError(f"ledger {path} must decode to a JSON object")
    return validate_ledger_document(
        doc,
        expected_namespace=expected_namespace,
        expected_lane=expected_lane,
    )


def build_ledger_from_attempt_dirs(
    attempt_dirs: Sequence[Path],
    *,
    namespace: str,
    lane: str,
) -> dict[str, Any]:
    """Mac-side path/size/hash-only ledger builder across immutable attempt directories."""
    expected_alloc = allocation_for_lane(lane)
    verified: dict[str, dict[str, Any]] = {}
    for attempt_dir in attempt_dirs:
        shards_dir = attempt_dir / "shards"
        if not shards_dir.is_dir():
            continue
        for complete_file in sorted(shards_dir.glob("*.json.complete")):
            payload_name = complete_file.name[: -len(".complete")]
            payload_path = shards_dir / payload_name
            uid = payload_path.stem
            if uid not in expected_alloc:
                raise ValueError(
                    f"foreign unit {uid!r} in {shards_dir} outside allocation {expected_alloc}"
                )
            relpath = f"shards/{uid}.json"
            entry, _doc = verify_published_json(
                payload_path,
                expected_unit_id=uid,
                expected_namespace=namespace,
                expected_relpath=relpath,
            )
            if uid in verified:
                raise ValueError(f"duplicate verified unit {uid!r} across attempt directories")
            verified[uid] = entry

    ordered = [verified[uid] for uid in expected_alloc if uid in verified]
    ledger_doc = {
        "schema": SCHEMA_LEDGER,
        "namespace": namespace,
        "lane": lane,
        "unit_kind": UNIT_KIND,
        "allocation": list(expected_alloc),
        "skip_verified": True,
        "verified_units": ordered,
    }
    validate_ledger_document(ledger_doc, expected_namespace=namespace, expected_lane=lane)
    return ledger_doc


def aggregate_verified_units(
    unit_docs: Sequence[dict[str, Any]],
    *,
    namespace: str,
    lane: str,
    commit: str,
    skipped_unit_ids: Sequence[str] = (),
    require_complete_allocation: bool = True,
) -> dict[str, Any]:
    """Reduce verified shard documents into a deterministic parity summary."""
    expected_alloc = allocation_for_lane(lane)
    seen: dict[str, dict[str, Any]] = {}
    for doc in unit_docs:
        if doc.get("schema") != SCHEMA_UNIT:
            raise ValueError(f"unexpected unit schema {doc.get('schema')!r}")
        if doc.get("namespace") != namespace:
            raise ValueError(
                f"foreign unit namespace {doc.get('namespace')!r}; expected {namespace!r}"
            )
        uid = str(doc.get("unit_id", ""))
        if uid not in expected_alloc:
            raise ValueError(f"unknown unit_id {uid!r} outside allocation {expected_alloc}")
        if uid in seen:
            raise ValueError(f"duplicate unit_id {uid!r} in aggregation")
        seen[uid] = doc

    missing = [
        uid for uid in expected_alloc if uid not in seen and uid not in set(skipped_unit_ids)
    ]
    if require_complete_allocation and len(seen) != len(expected_alloc):
        missing_all = [uid for uid in expected_alloc if uid not in seen]
        raise ValueError(f"incomplete allocation for {namespace}: missing {missing_all}")

    ordered_docs = [seen[uid] for uid in expected_alloc if uid in seen]
    total_envs = 0
    ticks_values: set[int] = set()
    world_ticks = 0
    snake_ticks = 0
    discrete_mismatches = 0
    ct_discrete_mismatches = 0
    ego_obs_mismatches = 0
    total_live_decisions = 0
    tie_explained_decisions = 0
    unexplained_mismatches = 0
    policy_unexplained_mismatches = 0
    events_sum: dict[str, int] = {}
    tie_samples: list[dict[str, Any]] = []
    mismatch_samples: list[dict[str, Any]] = []
    devices: set[str] = set()
    elapsed_seconds = 0.0

    for doc in ordered_docs:
        rep = doc["report"]
        total_envs += int(rep["num_envs"])
        ticks_values.add(int(rep["ticks"]))
        world_ticks += int(rep["world_ticks"])
        snake_ticks += int(rep["snake_ticks"])
        discrete_mismatches += int(rep["discrete_mismatches"])
        ct_discrete_mismatches += int(rep["ct_discrete_mismatches"])
        ego_obs_mismatches += int(rep["ego_obs_mismatches"])
        total_live_decisions += int(rep["total_live_decisions"])
        tie_explained_decisions += int(rep["tie_explained_decisions"])
        unexplained_mismatches += int(rep["unexplained_mismatches"])
        policy_unexplained_mismatches += int(rep["policy_unexplained_mismatches"])
        devices.add(str(rep["device"]))
        elapsed_seconds += float(doc.get("elapsed_seconds", 0.0))
        for k, v in rep["events"].items():
            events_sum[k] = events_sum.get(k, 0) + int(v)
        for sample in rep.get("tie_samples", []):
            if len(tie_samples) < 32:
                tie_samples.append({"unit_id": doc["unit_id"], **sample})
        for sample in rep.get("mismatch_samples", []):
            if len(mismatch_samples) < 32:
                mismatch_samples.append({"unit_id": doc["unit_id"], **sample})

    all_accounted = len(seen) + len(set(skipped_unit_ids)) == len(expected_alloc) and not missing
    if require_complete_allocation:
        passed = len(seen) == len(expected_alloc) and unexplained_mismatches == 0
    else:
        passed = all_accounted and unexplained_mismatches == 0

    return {
        "schema": SCHEMA_SUMMARY,
        "namespace": namespace,
        "commit": commit,
        "lane": lane,
        "allocation": list(expected_alloc),
        "completed_in_view_unit_ids": [doc["unit_id"] for doc in ordered_docs],
        "skipped_verified_unit_ids": sorted(set(skipped_unit_ids)),
        "missing_unit_ids": missing,
        "complete_allocation_verified": len(seen) == len(expected_alloc),
        "devices": sorted(devices),
        "num_envs_evaluated": total_envs,
        "ticks_per_shard": sorted(ticks_values),
        "world_ticks": world_ticks,
        "snake_ticks": snake_ticks,
        "discrete_mismatches": discrete_mismatches,
        "ct_discrete_mismatches": ct_discrete_mismatches,
        "ego_obs_mismatches": ego_obs_mismatches,
        "total_live_decisions": total_live_decisions,
        "tie_explained_decisions": tie_explained_decisions,
        "policy_unexplained_mismatches": policy_unexplained_mismatches,
        "unexplained_mismatches": unexplained_mismatches,
        "events": events_sum,
        "tie_samples": tie_samples,
        "mismatch_samples": mismatch_samples,
        "shard_elapsed_seconds_sum": round(elapsed_seconds, 4),
        "status": "PASS" if passed else "FAIL",
    }


def aggregate_attempt_dirs(
    attempt_dirs: Sequence[Path],
    *,
    commit: str,
    lane: str,
) -> dict[str, Any]:
    """Mac-side authoritative single look across all verified units from attempt dirs."""
    namespace = namespace_for(commit, lane)
    expected_alloc = allocation_for_lane(lane)
    unit_docs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for attempt_dir in attempt_dirs:
        shards_dir = attempt_dir / "shards"
        if not shards_dir.is_dir():
            continue
        for complete_file in sorted(shards_dir.glob("*.json.complete")):
            payload_name = complete_file.name[: -len(".complete")]
            payload_path = shards_dir / payload_name
            uid = payload_path.stem
            if uid not in expected_alloc:
                raise ValueError(f"foreign unit {uid!r} in {shards_dir}")
            if uid in seen_ids:
                raise ValueError(f"duplicate verified unit {uid!r} across attempts")
            _entry, doc = verify_published_json(
                payload_path,
                expected_unit_id=uid,
                expected_namespace=namespace,
                expected_relpath=f"shards/{uid}.json",
            )
            seen_ids.add(uid)
            unit_docs.append(doc)
    return aggregate_verified_units(
        unit_docs,
        namespace=namespace,
        lane=lane,
        commit=commit,
        require_complete_allocation=True,
    )


def _cli() -> int:
    parser = argparse.ArgumentParser(description="Ledger builder and Mac aggregator for v2 parity.")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_ledger = sub.add_parser("build-ledger", help="Build path/size/hash verified ledger on Mac.")
    p_ledger.add_argument("--commit", required=True)
    p_ledger.add_argument("--lane", choices=sorted(LANE_SPECS), required=True)
    p_ledger.add_argument("--output", type=Path, required=True)
    p_ledger.add_argument("attempt_dirs", nargs="+", type=Path)

    p_agg = sub.add_parser("aggregate", help="Authoritative Mac-side aggregation across attempts.")
    p_agg.add_argument("--commit", required=True)
    p_agg.add_argument("--lane", choices=sorted(LANE_SPECS), required=True)
    p_agg.add_argument("--output", type=Path, required=True)
    p_agg.add_argument("attempt_dirs", nargs="+", type=Path)

    args = parser.parse_args()
    namespace = namespace_for(args.commit, args.lane)
    if args.cmd == "build-ledger":
        doc = build_ledger_from_attempt_dirs(
            args.attempt_dirs,
            namespace=namespace,
            lane=args.lane,
        )
        encoded = (json.dumps(doc, indent=2, sort_keys=True) + "\n").encode("utf-8")
        write_bytes_create_only(args.output, encoded)
        return 0

    summary = aggregate_attempt_dirs(
        args.attempt_dirs,
        commit=args.commit,
        lane=args.lane,
    )
    publish_immutable_json(args.output, summary)
    if summary["status"] != "PASS":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli())
