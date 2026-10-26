"""Create-only F5 register reconciliation and custodian-gated allocation sealing.

Reads an explicit hash-pinned metadata plan, never follows payload references,
and never promotes observed-source clearance to campaign completeness. The
original allocator and collision auditor remain byte-for-byte unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

from research.bl_residual import f5_r10a_annex as allocator

SCHEMA = "f5-adoption-register-reconciliation/v1"
ALLOCATOR_SHA256 = "055cb66b5b7e270328c4ba67b83ff133a5ce9cb289d1857d2bfa9c361e0cc9ee"
RULE = "raw-integer-seed+source-attested-transitive-parent-alias-ancestry/v1"
CLASSES = (
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
LIFECYCLES = ("reserved", "rejected", "failed", "aborted", "retried", "superseded", "completed")
ASSERTIONS = (
    "existing_custodian_authority_verified",
    "complete_campaign_package_register_coverage",
    "all_lifecycle_states_covered",
    "all_r30_1_splits_and_registered_r30_2_covered",
    "protected_metadata_lawfully_covered",
    "physical_seed_rules_resolved",
    "ultimate_ancestry_and_namespace_aliases_resolved",
    "both_exclusion_scopes_preserved",
    "all_684_candidates_compared_together",
    "all_prior_round_rejections_verified",
    "no_unresolved_scope_ancestry_alias_omissions",
)
MAX_BYTES = 8 * 1024 * 1024
BLOCKED = ("d4096", "t34-round0-study-01")
PAYLOAD_DIRS = {"lives", "payloads", "outcomes", "runtime-output", "exports"}


def canonical(value: Any) -> bytes:
    """Use the unchanged allocator's exact allocation serialization."""
    return allocator.encoded(value)


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def require(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(reason)


def safe_path(path: Path) -> None:
    """Reject protected directories and symlink escapes before stat/open."""
    require(
        path.is_absolute() and ".." not in path.parts, "absolute canonical metadata path required"
    )
    require(not any(token in str(path).lower() for token in BLOCKED), "protected path refused")
    require(not PAYLOAD_DIRS.intersection(path.parts), "payload directory refused")
    require(not any(p.is_symlink() for p in (*reversed(path.parents), path)), "symlink refused")


def read(ref: dict) -> tuple[Any, dict]:
    """Read exactly one planned metadata file, with no reference traversal."""
    path = Path(ref["path"])
    safe_path(path)
    require(path.is_file() and path.stat().st_size <= MAX_BYTES, "metadata size bound")
    raw = path.read_bytes()
    require(digest(raw) == ref["sha256"], "metadata hash mismatch")
    return json.loads(raw), {"path": str(path), "sha256": digest(raw), "size": len(raw)}


def transitive_aliases(rows: list[dict]) -> list[dict]:
    """Resolve source-attested equivalence only; do not invent lexical aliases."""
    links: dict[str, set[str]] = {}
    for row in rows:
        ids = {row["parent_id"], *row.get("aliases", []), *row.get("ancestry", [])}
        for name in ids:
            links.setdefault(name, set()).update(ids)
    components: dict[str, list[str]] = {}
    for name in links:
        if name in components:
            continue
        seen, todo = set(), [name]
        while todo:
            current = todo.pop()
            if current in seen:
                continue
            seen.add(current)
            todo.extend(links[current] - seen)
        names = sorted(seen)
        for item in names:
            components[item] = names
    return [
        {"parent_id": r["parent_id"], "seed": r["seed"], "aliases": components[r["parent_id"]]}
        for r in rows
    ]


def collector_roots(value: dict) -> list[int]:
    """Recover metadata-only finite roots and verify the contemporaneous digest.

    Earlier rejected pair audits publish the exact root-list hash and complete
    seed family instead of the list. No simulator or collector is instantiated.
    NumPy's SeedSequence is used exactly as in collector._world_seed; failure to
    match the published original hash refuses the recovery.
    """
    import numpy as np

    def root_digest(roots: list[int]) -> str:
        return digest(",".join(str(x) for x in sorted(set(roots))).encode())

    require(
        value["schema"]
        in {"bl-p1-collection-allocation-audit/v1", "bl-p1-collection-allocation-audit/v3"},
        "collector audit schema",
    )
    burned = value["burned_roots"]
    require(
        isinstance(burned, list) and all(type(x) is int and 0 <= x < 2**64 for x in burned),
        "invalid burned roots",
    )
    require(
        len(burned) == value["burned_root_count"]
        and root_digest(burned) == value["burned_roots_sha256"],
        "burned roots identity",
    )
    allocated = value.get("allocated_roots")
    if allocated is None:
        allocated, blocked = [], set(burned)
        worlds = value["worlds"]
        start, end = (value["episodes_inclusive"][key] for key in ("start", "end"))
        require(worlds == 128 and start == 0 and end == 16, "unregistered metadata reconstruction")
        offset = value["natural_collection_world_id_offset"]
        for world in range(offset, offset + worlds):
            for episode in range(start, end + 1):
                for attempt in range(1024):
                    seed = int(
                        np.random.SeedSequence(
                            [value["seed"], 0xE2, world, episode, attempt]
                        ).generate_state(1)[0]
                    )
                    if seed not in blocked:
                        allocated.append(seed)
                        break
                else:
                    raise ValueError("metadata root rejection bound")
    require(
        isinstance(allocated, list) and all(type(x) is int and 0 <= x < 2**64 for x in allocated),
        "invalid allocated roots",
    )
    require(
        len(allocated) == value["allocated_root_count"]
        and root_digest(allocated) == value["allocated_roots_sha256"],
        "allocated roots identity",
    )
    return sorted(set(burned) | set(allocated))


def audit(rows: list[dict], seeds: set[int]) -> dict:
    """Enumerate rounds in order and call the unchanged whole-bank auditor.

    Pruning is exact: its only historical predicates are seed equality or
    candidate parent membership in historical aliases. All matching records and
    reasons are retained. Raw negative/full-width seeds are never reduced.
    """
    history = transitive_aliases(rows)
    known = {r["seed"] for r in history}
    by_seed: dict[int, list[tuple[int, dict]]] = {}
    by_alias: dict[str, list[tuple[int, dict]]] = {}
    for index, row in enumerate(history):
        by_seed.setdefault(row["seed"], []).append((index, row))
        for alias in row["aliases"]:
            by_alias.setdefault(alias, []).append((index, row))
    rejected = []
    for round_index in range(1000):
        candidates = allocator.allocate(round_index)
        require(len(candidates) == 684, "whole allocation changed")
        possible: dict[int, dict] = {}
        for row in candidates:
            seed, parent = row["seed"], row["parent_id"]
            for index, old in (*by_seed.get(seed, []), *by_alias.get(parent, [])):
                possible[index] = old
            if seed in seeds and seed not in known:
                possible[len(history) + seed] = {
                    "parent_id": f"registered-raw-seed/{seed}",
                    "seed": seed,
                    "aliases": [],
                }
        reasons = allocator.collision_audit(candidates, [possible[i] for i in sorted(possible)])
        if not reasons:
            return {
                "status": "OBSERVED_METADATA_CLEAR_PENDING_CUSTODIAN",
                "candidate_round": round_index,
                "candidate_allocation_sha256": digest(canonical(candidates)),
                "accepted_round": None,
                "accepted_allocation_sha256": None,
                "all_684_executed": True,
                "rejected_rounds": rejected,
                "candidate_allocation": candidates,
            }
        rejected.append({"round": round_index, "reasons": reasons})
    return {
        "status": "HOLD_NO_CLEAR_ROUND",
        "candidate_round": None,
        "candidate_allocation_sha256": None,
        "accepted_round": None,
        "accepted_allocation_sha256": None,
        "all_684_executed": True,
        "rejected_rounds": rejected,
        "candidate_allocation": [],
    }


def reconcile(plan: dict) -> tuple[dict, list[dict], set[int]]:
    """Crosswalk explicit registers to the sealed historical catalog."""
    require(plan["schema"] == SCHEMA, "wrong plan schema")
    require(
        digest(Path(allocator.__file__).read_bytes()) == ALLOCATOR_SHA256,
        "allocator source changed",
    )
    sources, rows, seeds = [], [], set()
    for ref in [*plan["authority"], *plan["source_rules"]]:
        path = Path(ref["path"])
        safe_path(path)
        require(path.is_file() and path.stat().st_size <= MAX_BYTES, "authority size bound")
        raw = path.read_bytes()
        require(digest(raw) == ref["sha256"], "authority hash mismatch")
        sources.append({**ref, "size": len(raw), "kind": "binding_authority"})
    catalog, binding = read(plan["catalog"])
    sources.append({**binding, "kind": "prior_catalog", "complete": catalog["complete"]})
    for shard in catalog["record_shards"]:
        path = Path(plan["catalog"]["path"]).parent / shard["path"]
        require(Path(shard["path"]).name == shard["path"], "catalog shard escape")
        value, binding = read({"path": str(path), "sha256": shard["sha256"]})
        require(len(value["records"]) == shard["record_count"], "catalog row count")
        rows.extend(value["records"])
        sources.append({**binding, "kind": "prior_catalog_shard"})
    require(len(rows) == catalog["record_count"], "catalog total count")
    seeds.update(r["seed"] for r in rows)
    prior_seeds = set(seeds)
    external, binding = read(plan["external_report"])
    sources.append({**binding, "kind": "seven_frp_ledgers_report"})
    union_hash = hashlib.sha256()
    external_values = []
    for shard in external["seed_shards"]:
        path = Path(plan["external_report"]["path"]).parent / shard["path"]
        require(Path(shard["path"]).name == shard["path"], "external shard escape")
        value, binding = read({"path": str(path), "sha256": shard["sha256"]})
        external_values.extend(value["seeds"])
        sources.append({**binding, "kind": "seven_frp_ledgers_union_shard"})
    require(external_values == sorted(set(external_values)), "union ordering or duplicates")
    for seed in external_values:
        union_hash.update(f"{seed}\n".encode("ascii"))
    require(union_hash.hexdigest() == external["canonical_union_sha256"], "old union digest")
    require(len(external_values) == external["union_unique_seed_count"], "old union count")
    seeds.update(external_values)
    prior_union = set(seeds)
    annex, binding = read(plan["annex"])
    originals = annex["allocation"]["inspected_historical_allocations"]
    original_map = {}
    for row in originals:
        require(
            row["parent_id"] not in original_map
            or original_map[row["parent_id"]]["seed"] == row["seed"],
            "original parent seed conflict",
        )
        original_map[row["parent_id"]] = row
        rows.append({"parent_id": row["parent_id"], "seed": row["seed"], "aliases": row["aliases"]})
    seeds.update(row["seed"] for row in originals)
    # Source life IDs, namespaces and original allocation IDs are admitted aliases.
    for life in annex["corpus"]["selected_lives"]:
        parent = life["ultimate_parent_id"]
        require(
            original_map[parent]["seed"] == life["original_world"]["seed"],
            "life ancestry seed conflict",
        )
        rows.append(
            {
                "parent_id": parent,
                "seed": original_map[parent]["seed"],
                "aliases": [life["life_id"]],
            }
        )
    exclusions = annex["exclusions"]
    require(len(exclusions["training_excluded_parent_union"]) == 167, "training union changed")
    require(len(exclusions["permanent_development_parents"]) == 29, "census changed")
    require(
        set(exclusions["permanent_development_parents"])
        <= set(exclusions["training_excluded_parent_union"]),
        "lost census exclusion",
    )
    sources.append(
        {
            **binding,
            "kind": "original_parent_ancestry_and_distinct_exclusions",
            "training_union": 167,
            "census_subset": 29,
        }
    )
    inventory, binding = read(plan["r30_inventory"])
    r30parents = {p["ultimate_parent_id"] for p in inventory["parents"]}
    require(
        len(r30parents) == 372 and r30parents == set(exclusions["r30_1_parents_all_splits"]),
        "R30 all-split mismatch",
    )
    split_counts = Counter()
    for row in inventory["parents"]:
        parent = row["ultimate_parent_id"]
        require(parent in original_map, "unresolved R30 ultimate ancestor")
        expected = (
            "validation"
            if int.from_bytes(
                hashlib.sha256((inventory["split_salt"] + "\0" + parent).encode()).digest(), "big"
            )
            % 10
            < 2
            else "train"
        )
        require(
            row["split"] == expected == inventory["frozen_split"][parent],
            "R30 frozen split conflict",
        )
        split_counts[expected] += 1
    sources.append(
        {
            **binding,
            "kind": "r30_1_all_splits",
            "parent_count": 372,
            "split_parent_counts": dict(split_counts),
            "new_seeds_vs_prior_union": 0,
        }
    )
    crosswalk = []
    for ref in plan["registers"]:
        value, binding = read(ref)
        projected = []
        if ref["kind"] == "collector_root_audit":
            projected = collector_roots(value)
        elif ref["kind"] == "r30_control":
            require(
                value["schema"] == "r30-control/v1" and value["synthetic"] is True,
                "unclassified R30 control",
            )
        elif ref["kind"] != "register_metadata":
            raise ValueError("unrecognized metadata projection")
        unique = set(projected)
        crosswalk.append(
            {
                **binding,
                "kind": ref["kind"],
                "classes": ref["classes"],
                "lifecycle": ref["lifecycle"],
                "seed_count": len(unique),
                "new_seeds_vs_prior_union": len(unique - prior_union),
                "prior_catalog_seed_count": len(unique & prior_seeds),
                "prior_union_seed_count": len(unique & prior_union),
                "ancestry_status": ref["ancestry_status"],
                "projection": ref["projection"],
            }
        )
        seeds.update(unique)
        sources.append(binding)
    source_manifest = {
        "schema": SCHEMA,
        "register_snapshot_cutoff_utc": plan["register_snapshot_cutoff_utc"],
        "input_plan_sha256": digest(canonical(plan)),
        "reconciler_sha256": digest(Path(__file__).read_bytes()),
        "source_bindings": sources,
        "prior_repository_register": catalog["repository_source_register"],
        "seven_frp_ledger_bindings": external["source_bindings"],
        "register_to_catalog_crosswalk": crosswalk,
        "r30_1_all_splits": {
            "parent_count": 372,
            "split_parent_counts": dict(split_counts),
            "parents": sorted(r30parents),
        },
        "training_exclusion": {
            "parents": exclusions["training_excluded_parent_union"],
            "census_parents": exclusions["permanent_development_parents"],
            "descendants": "all transitive continuations, lives, rows, snapshots "
            "and namespace aliases",
        },
        "fresh_bank_exclusion": {
            "classes": list(CLASSES),
            "lifecycles": list(LIFECYCLES),
            "scope": "all historical ancestry, all R30-1 splits and registered R30-2; "
            "broader than training union",
        },
        "comparison_rule": RULE,
        "seed_normalization": (
            "Preserve raw signed/full-width integers. No universal modulo, abs or uint32 "
            "cast is supported by source evidence. Source-specific missing normalization "
            "is an explicit custodian obligation."
        ),
        "registered_r30_2": (
            "four synthetic prepared package controls included; fixture has four synthetic "
            "parents, no physical world seeds; real registered allocation completeness "
            "requires custodian verification"
        ),
        "unresolved": plan["unresolved"],
        "complete": False,
        "unique_raw_seed_count": len(seeds),
        "unique_raw_seed_sha256": digest("".join(f"{x}\n" for x in sorted(seeds)).encode("ascii")),
        "unchanged_allocator_sha256": digest(Path(allocator.__file__).read_bytes()),
    }
    return source_manifest, rows, seeds


def certificate_template(manifest: dict, evaluation: dict) -> dict:
    """Bind all reviewable evidence without impersonating the bank custodian."""
    return {
        "schema": "f5-custodian-collision-certificate/v1",
        "status": "UNATTESTED_TEMPLATE",
        "issuer": None,
        "existing_authority": None,
        "attested_at_utc": None,
        "register_snapshot_cutoff_utc": manifest.get("register_snapshot_cutoff_utc"),
        "approval_record": None,
        "register_catalog_manifest_sha256": digest(canonical(manifest)),
        "candidate_round": evaluation["candidate_round"],
        "candidate_allocation_sha256": evaluation["candidate_allocation_sha256"],
        "allocation_evaluation_sha256": digest(canonical(evaluation)),
        "comparison_rule": RULE,
        "covered_classes": list(CLASSES),
        "covered_lifecycles": list(LIFECYCLES),
        "protected_register_catalog_bindings": [],
        "additional_metadata_bindings": [],
        "assertions": {name: None for name in ASSERTIONS},
        "collision_result_all_684": None,
        "prior_round_rejections": evaluation["rejected_rounds"],
        "omissions": None,
        "instructions": (
            "Claude as existing campaign operator/custodian must verify complete ops records "
            "and lawful protected coverage, supply attributable dated approval record and "
            "exact protected catalog/register hashes, and publish a NEW completed certificate. "
            "Do not edit this template. Any collision or omission requires new reconciliation "
            "and unchanged whole-round allocation; never attest an unknown clear."
        ),
    }


def validate_certificate_structure(cert: dict, manifest: dict, evaluation: dict) -> None:
    """Check certificate structure only; this cannot authenticate its issuer.

    This helper never authorizes acceptance or sealing. An independent reviewer
    must establish actual custodian authorship/authority outside this packet.
    """
    require(cert.get("schema") == "f5-custodian-collision-certificate/v1", "certificate schema")
    require(cert.get("status") == "ATTESTED", "unattested certificate")
    for field in (
        "issuer",
        "existing_authority",
        "attested_at_utc",
        "register_snapshot_cutoff_utc",
    ):
        require(
            isinstance(cert.get(field), str) and bool(cert[field].strip()),
            "missing custodian identity/date/authority",
        )
    try:
        attested = datetime.fromisoformat(cert["attested_at_utc"].replace("Z", "+00:00"))
        cutoff = datetime.fromisoformat(cert["register_snapshot_cutoff_utc"].replace("Z", "+00:00"))
        require(
            attested.utcoffset() is not None
            and cutoff.utcoffset() is not None
            and attested >= cutoff,
            "invalid dated custodian attestation",
        )
    except (ValueError, TypeError):
        raise ValueError("invalid dated custodian attestation") from None
    if "register_snapshot_cutoff_utc" in manifest:
        require(
            cert["register_snapshot_cutoff_utc"] == manifest["register_snapshot_cutoff_utc"],
            "stale register cutoff",
        )
    require(
        cert.get("register_catalog_manifest_sha256") == digest(canonical(manifest)),
        "stale register binding",
    )
    require(
        cert.get("allocation_evaluation_sha256") == digest(canonical(evaluation)),
        "stale evaluation binding",
    )
    require(
        cert.get("candidate_allocation_sha256") == evaluation["candidate_allocation_sha256"]
        and cert.get("candidate_round") == evaluation["candidate_round"]
        and evaluation["candidate_round"] is not None,
        "stale allocation binding",
    )
    require(cert.get("comparison_rule") == RULE, "foreign comparison rule")
    require(set(cert.get("covered_classes", [])) == set(CLASSES), "incomplete classes")
    require(
        set(cert.get("covered_lifecycles", [])) == set(LIFECYCLES), "incomplete lifecycle coverage"
    )
    require(
        set(cert.get("assertions", {})) == set(ASSERTIONS)
        and all(cert["assertions"][key] is True for key in ASSERTIONS),
        "incomplete custodian assertions",
    )
    require(cert.get("omissions") == [], "unresolved omissions")
    require(
        cert.get("collision_result_all_684") == "NO_COLLISION", "candidate collision or unknown"
    )
    require(
        cert.get("prior_round_rejections") == evaluation["rejected_rounds"],
        "unbound prior rejection reasons",
    )
    protected = cert.get("protected_register_catalog_bindings", [])
    require(
        {item.get("bank") for item in protected} == {"D4096", "t34-round0-study-01"},
        "missing protected bank bindings",
    )
    for item in protected:
        for key in ("register_sha256", "catalog_sha256"):
            require(
                isinstance(item.get(key), str)
                and len(item[key]) == 64
                and all(c in "0123456789abcdef" for c in item[key]),
                "invalid protected digest",
            )
    approval = cert.get("approval_record")
    require(isinstance(approval, dict), "missing attributable approved record")
    # This reads only a lawful external approval JSON, never the protected bank.
    record, _ = read(approval)
    require(
        record.get("issuer") == cert["issuer"] and record.get("approved") is True,
        "unapproved or foreign issuer",
    )
    require(
        record.get("certificate_content_sha256")
        == digest(canonical({k: v for k, v in cert.items() if k != "approval_record"})),
        "approval not bound to certificate content",
    )


def publish(output: Path, files: dict[str, Any]) -> None:
    """Publish immutable metadata atomically and seal the file inventory last."""
    safe_path(output)
    require(not output.exists(), "create-only output exists")
    output.mkdir(parents=True, exist_ok=False)
    all_files = {}
    for name, value in files.items():
        require(Path(name).name == name, "output filename escape")
        raw = canonical(value)
        require(len(raw) < 2_000_000, "publication file exceeds 2 MB")
        all_files[name] = raw
        all_files[name + ".sha256"] = f"{digest(raw)}  {name}\n".encode()
    all_files["COMPLETED.json"] = canonical(
        {
            "schema": "f5-register-packet-completion/v1",
            "files": {name: digest(raw) for name, raw in all_files.items()},
        }
    )
    for name, raw in all_files.items():
        fd, tmp = tempfile.mkstemp(prefix=".f5-", dir=output)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(tmp, output / name)
        finally:
            os.unlink(tmp)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    safe_path(args.plan)
    require(args.plan.stat().st_size < MAX_BYTES, "plan bound")
    plan = json.loads(args.plan.read_bytes())
    manifest, rows, seeds = reconcile(plan)
    evaluation = audit(rows, seeds)
    files = {
        "register-catalog-manifest.json": manifest,
        "allocation-evaluation.json": evaluation,
        "custodian-certificate-template.json": certificate_template(manifest, evaluation),
    }
    publish(args.out, files)
    print(
        json.dumps(
            {
                "candidate_round": evaluation["candidate_round"],
                "candidate_allocation_sha256": evaluation["candidate_allocation_sha256"],
                "accepted_round": None,
                "unresolved": manifest["unresolved"],
            }
        )
    )


if __name__ == "__main__":
    main()
