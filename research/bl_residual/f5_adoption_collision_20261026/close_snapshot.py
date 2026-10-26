"""Reconcile the finite 135-source snapshot without opening scientific outputs.

This is a new, create-only supplement. It preserves the original reconciliation
and allocator. An evidence seal is not an authenticated custodian certificate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from research.bl_residual.f5_adoption_collision_20261026 import reconcile as prior

PACKAGE = Path(__file__).absolute().parent
REPO = PACKAGE.parents[2]
SNAPSHOT_SHA = "58fec697539a32d0aa9f9cc8f5fabfb03c3336d989baba396c76b852eebc8496"
DOMAIN = "t34-exit-b8-20261019/v1"
SEED_KEYS = {"seed", "world_seed", "seeds", "world_seeds", "episode_seeds", "burned_roots"}
OPAQUE_INDICES = {182, 184, 185, 186, *range(203, 216)}
RULE_SOURCES = (
    "research/bl_served_search/mac_identity.py",
    "research/bl_ppo/p1_eval.py",
    "research/t34_pathb/bank.py",
    "research/t34_round0/bank.py",
    "research/gpu_sim_spike/identity_tape.py",
    "research/gpu_sim_spike/identity_pack.py",
    "research/t34_claim_a/seed_audit.py",
    "research/t34_exit_b8/trajectory.py",
    "research/apex_safety_20260926/dev_screen.py",
    "research/speed_s0_baseline/harness.py",
    "research/gpu_sim_spike/reference.py",
    "research/gpu_sim_spike/realistic_benchmark.py",
    "research/gpu_sim_spike/a100_ruler.py",
    "research/gpu_sim_spike/integration_parity.py",
    "research/gpu_sim_spike/move2_parity.py",
    "research/gpu_sim_spike/v8_split_bench.py",
    "research/gpu_sim_spike/v8_split_discrepancies.py",
    "research/gpu_sim_spike/sparse_bench.py",
    "research/gpu_sim_spike/integration_bench.py",
    "research/gpu_sim_spike/move2_bench.py",
    "research/a3_p1_ceiling/benchmark.py",
    "research/a3_p1_ceiling/runtime.py",
    "research/bl_ppo/collector.py",
    "research/bl_ppo/run.py",
    "research/bl_ppo/bank.py",
    "research/bl_ppo/late_body.py",
    "research/a100_a2/job.py",
    "research/a100_a2/protocol.py",
    "research/bl_info_value/catalog.py",
    "research/bl_info_value/reconstruct.py",
    "research/bl_residual/job.py",
    "research/bl_residual/data.py",
    "research/bl_round30/fixture.py",
    "research/bl_round30/job.py",
    "research/bl_round30/prepare.py",
    "research/bl_round30/timing.py",
    "research/bl_round30/admission.py",
    "research/r32_rehearsal/job.py",
    "research/r32_rehearsal/matrix.py",
    "src/simd_env/rng.py",
    "src/simd_env/batch_sim.py",
    "src/core/seeding.py",
)


def binding(path: Path) -> dict:
    """Hash one explicit lawful metadata/source file, never a referenced payload."""
    prior.safe_path(path)
    prior.require(path.is_file() and path.stat().st_size <= prior.MAX_BYTES, "source bound")
    raw = path.read_bytes()
    return {"path": str(path), "sha256": prior.digest(raw), "size": len(raw)}


def seed_digest(seeds: list[int]) -> str:
    """Match identity_pack's ordered comma-separated seed digest."""
    return prior.digest(",".join(map(str, seeds)).encode())


def sha_seed(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:4], "big")


def require_clear(evaluation: dict) -> None:
    """Refuse a CLEAR evidence seal if all one thousand rounds were rejected."""
    prior.require(
        evaluation["candidate_round"] is not None
        and evaluation["status"] == "OBSERVED_METADATA_CLEAR_PENDING_CUSTODIAN",
        "HOLD: no clear whole allocation",
    )


def prep_seeds(namespace: str, count: int) -> list[int]:
    return [sha_seed(f"{DOMAIN}|{namespace}|{i}") for i in range(count)]


def p3_rows(round_index: int) -> list[dict]:
    namespace = "bl-served-search-mac-identity-20261022/v1"
    return [
        {"parent_id": name, "seed": sha_seed(name), "aliases": []}
        for mix, index in (
            ("frozen", 0),
            ("frozen", 1),
            ("mixed", 0),
            ("mixed", 1),
            ("scripted", 0),
        )
        for name in [f"{namespace}/round-{round_index}/{mix}/{index}"]
    ]


def p1_seeds() -> list[int]:
    prefix = "bl-ppo-p1-eval-20261021/v1/round-0/endpoint"
    return [
        sha_seed(f"{prefix}/{mix}/world/{i}")
        for mix in ("scripted", "frozen", "mixed")
        for i in range(128)
    ]


def literal_seeds(value: Any) -> set[int]:
    """Project exact seed fields only from admitted metadata, not assay/fit seeds."""
    result: set[int] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            if key in SEED_KEYS:
                if type(item) is int:
                    result.add(item)
                elif isinstance(item, list) and all(type(x) is int for x in item):
                    result.update(item)
            elif isinstance(item, (dict, list)):
                result.update(literal_seeds(item))
    elif isinstance(value, list):
        for item in value:
            result.update(literal_seeds(item))
    return result


def life_rows(value: dict, aliases: dict[str, dict]) -> list[dict]:
    """Join allocation life/parent edges to source-attested original physical roots."""
    result = []
    for life, parent in value["parents"].items():
        prior.require(parent in aliases, f"unresolved life ancestor: {parent}")
        original = aliases[parent]
        result.append(
            {
                "parent_id": original["parent_id"],
                "seed": original["seed"],
                "aliases": [parent, life],
            }
        )
    # Do not decode catalog rescue flags, row labels, or any outcome payload.
    prior.require(
        {r["life_id"] for r in value["catalog"]} == set(value["parents"]),
        "life/parent catalog mismatch",
    )
    return result


def a3_envelope(assay_seed: int, repeats: int, worlds: int) -> set[int]:
    """Exclude a pre-data superset of all possible reset roots, without gameplay.

    There are two initial rollouts and at most repeats further 128-step
    rollouts. Even death on every step cannot exceed this episode bound.
    Calibration peeks once at episode zero using assay_seed+104729.
    Repeated independent units reuse these roots. This is an exclusion
    envelope, not a claim about observed deaths or actual visited episodes.
    """
    import numpy as np

    counts = {"scripted": 128, "frozen": 48, "mixed": 48}
    prefix = "bl-ppo-e2-screen-20261021/v2/round-0"
    excluded = {
        sha_seed(f"{prefix}/{section}/{mix}/world/{i}")
        for section in ("screen", "preflight", "parity")
        for mix, count in counts.items()
        for i in range(count if section == "screen" else 1)
    }

    def root(root_seed: int, world: int, episode: int, blocked: set[int]) -> int:
        for attempt in range(1024):
            seed = int(
                np.random.SeedSequence([root_seed, 0xE2, world, episode, attempt]).generate_state(
                    1
                )[0]
            )
            if seed not in blocked:
                return seed
        raise ValueError("A3 seed attempt bound")

    holdout = {root(2026102111, 2_000_000 + world, 0, excluded) for world in range(12)}
    result = set(holdout)
    for root_seed, episodes in ((assay_seed, (2 + repeats) * 128 + 1), (assay_seed + 104729, 1)):
        for world in range(worlds):
            for episode in range(episodes):
                result.add(
                    root(
                        root_seed,
                        world,
                        episode,
                        excluded | holdout if root_seed == assay_seed + 104729 else excluded,
                    )
                )
    return result


def collection_roots(value: dict) -> set[int]:
    """Recover older finite audit families and verify both original list hashes."""
    import numpy as np

    burned = value["burned_roots"]
    blocked = set(burned)
    allocated = value.get("allocated_roots")
    if allocated is None:
        allocated = []
        start, end = (value["episodes_inclusive"][key] for key in ("start", "end"))
        prior.require(
            value["worlds"] == 128 and start == 0 and end <= 127, "collector family bound"
        )
        offset = value["natural_collection_world_id_offset"]
        for world in range(offset, offset + 128):
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
                    raise ValueError("collector root attempt bound")
    for kind, values in (("burned", burned), ("allocated", allocated)):
        prior.require(len(values) == value[f"{kind}_root_count"], "collector count mismatch")
        prior.require(
            seed_digest(sorted(set(values))) == value[f"{kind}_roots_sha256"],
            "collector root hash mismatch",
        )
    return set(burned) | set(allocated)


def disposition(index: int) -> str:
    """Every snapshot item has an explicit closed set of metadata interpretations."""
    groups = {
        "source_review_no_allocation": {0},
        "a2_cached_labels_no_new_worlds": set(range(1, 7)),
        "a3_collector_potential_root_envelope": set(range(7, 11)),
        "literal_seed_audit": {35, 36, 38, 78, 187, 189, 191, 193, 195, 197, 199, 201},
        "e3_original_t3_ancestry_replay": set(range(72, 78)),
        "p3_both_allocated_rounds": {82, 83, 84, 85},
        "p1_endpoint_384": {86},
        "hermetic_pytest_fixture": {*range(87, 94)},
        "collector_burned_reserved_and_allocated_roots": {97, 98, 100, 101},
        "collector_rejection_supersession_chain": {99},
        "transport_control_no_world_allocation": set(range(152, 156)),
        "f3_f4_synthetic_or_existing_factory_ancestry": {
            156,
            157,
            158,
            159,
            160,
            161,
            163,
            165,
            167,
            169,
            171,
            173,
            175,
            177,
            179,
            181,
        },
        "f4_all_folds_life_parent_alias_join": {162, 164, 166, 168, 170, 172, 174, 176, 178, 180},
        "opaque_discrepancy_source_fixture_envelope": OPAQUE_INDICES,
        "gpu_identity_reconstructed_and_digest_verified": {183},
        "gpu_label_allocation_and_retry_aliases": {188, 190, 192, 194, 196, 198, 200, 202},
        "failed_and_aborted_compute_allocation_accounting": {216},
        "preservation_record_no_new_worlds": {217, 218, 219, 220, 229, 232, 235},
        "r30_existing_all_split_parent_ancestry": {221, 222, 223},
        "r30_reservation_and_retry_chain_same_parent_corpus": {225, 227, 230, 233, 236},
        "r30_synthetic_no_physical_worlds": {
            238,
            239,
            242,
            243,
            244,
            245,
            248,
            249,
            250,
            251,
            254,
            255,
        },
        "r32_synthetic_statistics_no_physical_worlds": set(range(258, 263)),
    }
    matches = [name for name, indices in groups.items() if index in indices]
    prior.require(len(matches) == 1, f"unclassified or multiply classified snapshot index {index}")
    return matches[0]


def crosswalk(
    snapshot: dict, aliases: dict[str, dict], inherited: dict | None = None
) -> tuple[list[dict], list[dict], set[int]]:
    """Project every additional metadata source; never open discrepancy outputs."""
    entries, rows, seeds, envelopes = [], [], set(), {}
    inherited = inherited or {}
    for index, ref in enumerate(snapshot["sources"]):
        if ref["prior_catalog_crosswalk"] == "HASH_BOUND_SOURCE":
            continue
        kind = disposition(index)
        projected, added_rows = set(), []
        entry = {
            **ref,
            "snapshot_index": index,
            "disposition": kind,
            "lifecycle_policy": (
                "exclude allocated/reserved identities irrespective of success, "
                "failure, retry or supersession"
            ),
        }
        if index in OPAQUE_INDICES:
            # A snapshot hash is not a new verification of forbidden output bytes.
            entry["source_read"] = False
            entry["hash_verification"] = (
                "inherited immutable filename inventory only; output not reopened"
            )
            projected.update(prep_seeds("t34-speed-s0-baseline/v1", 6))
            entry["evidence"] = (
                "source-only six-root S0 fixture envelope; I0 delegates MOVE2; MOVE2/V8 "
                "select the three long roots; sparse benchmark uses ruler fixtures; "
                "historical source identity and ad hoc sparse-cap160 register remain attestable"
            )
        else:
            value, fresh = prior.read(ref)
            prior.require(fresh["size"] == ref["size"], "snapshot size changed")
            entry["source_read"] = True
            entry["hash_verification"] = "reverified against immutable snapshot"
            entry["schema"] = value.get("schema")
            entry["namespace"] = value.get("namespace")
            if kind == "a3_collector_potential_root_envelope":
                key = (value["assay_seed"], value.get("repeats", 3), value.get("shape", [128])[0])
                prior.require(key[1] <= 3 and key[2] <= 128, "unregistered A3 envelope")
                if key not in envelopes:
                    envelopes[key] = a3_envelope(*key)
                projected = envelopes[key]
                entry["root_envelope"] = {
                    "assay_seed": key[0],
                    "repeats": key[1],
                    "worlds": key[2],
                    "episodes_inclusive": [0, (2 + key[1]) * 128],
                    "calibration_episode": 0,
                    "calibration_seed_offset": 104729,
                    "actual_episode_outcomes_inspected": False,
                }
            elif kind == "collector_burned_reserved_and_allocated_roots":
                projected = collection_roots(value)
            elif kind == "p3_both_allocated_rounds":
                round_index = value["round_index"]
                added_rows = p3_rows(round_index)
                if "rows" in value:
                    prior.require(
                        [r["world_seed"] for r in value["rows"]] == [r["seed"] for r in added_rows],
                        "P3 allocation reconstruction changed",
                    )
            elif kind == "p1_endpoint_384":
                projected = set(p1_seeds())
                prior.require(
                    value["candidate_count"] == len(projected) == 384, "P1 count mismatch"
                )
                entry["ordered_seed_digest"] = seed_digest(p1_seeds())
            elif kind == "f4_all_folds_life_parent_alias_join":
                added_rows = life_rows(value, aliases)
                entry["life_count"] = len(added_rows)
                entry["all_folds"] = dict(Counter(value["folds"].values()))
                entry["parent_count"] = len({r["parent_id"] for r in added_rows})
            elif kind == "gpu_identity_reconstructed_and_digest_verified":
                entry["reconstructed_namespaces"] = []
                for namespace, spec in value["namespaces"].items():
                    allocation = prep_seeds(namespace, spec["count"])
                    prior.require(
                        seed_digest(allocation) == spec["sha256"],
                        "GPU identity seed digest mismatch",
                    )
                    projected.update(allocation)
                    entry["reconstructed_namespaces"].append(
                        {
                            "namespace": namespace,
                            "count": len(allocation),
                            "sha256": seed_digest(allocation),
                        }
                    )
            elif kind == "r30_existing_all_split_parent_ancestry":
                inventory_path = Path(ref["path"]).parent / "inventory.json"
                inventory, inventory_binding = prior.read(
                    {"path": str(inventory_path), "sha256": value["inventory_sha256"]}
                )
                expected_parents = set(inherited["r30_1_all_splits"]["parents"])
                prior.require(
                    {r["ultimate_parent_id"] for r in inventory["parents"]} == expected_parents,
                    "R30 staged parent universe differs",
                )
                prior.require(
                    value["salt"] == inventory["split_salt"], "R30 staged split salt differs"
                )
                for item in inventory["parents"]:
                    parent = item["ultimate_parent_id"]
                    prior.require(parent in aliases, "R30 unresolved staged ancestor")
                    expected_split = (
                        "validation"
                        if int.from_bytes(
                            hashlib.sha256(
                                (inventory["split_salt"] + "\0" + parent).encode()
                            ).digest(),
                            "big",
                        )
                        % 10
                        < 2
                        else "train"
                    )
                    prior.require(
                        item["split"] == expected_split == inventory["frozen_split"][parent],
                        "R30 staged split changed",
                    )
                    old = aliases[parent]
                    added_rows.append(
                        {
                            "parent_id": old["parent_id"],
                            "seed": old["seed"],
                            "aliases": [f"{value['namespace']}/{parent}"],
                        }
                    )
                for life in inventory.get("lives", []):
                    parent = life["ultimate_parent_id"]
                    prior.require(parent in expected_parents, "R30 staged life ancestor differs")
                    old = aliases[parent]
                    added_rows.append(
                        {
                            "parent_id": old["parent_id"],
                            "seed": old["seed"],
                            "aliases": [life["source_life_id"]],
                        }
                    )
                entry["verified_staged_inventory"] = inventory_binding
                entry["parent_count"] = len(expected_parents)
                entry["split_parent_counts"] = inherited["r30_1_all_splits"]["split_parent_counts"]
            elif kind == "e3_original_t3_ancestry_replay":
                inherited_rows = inherited["t3_catalog_rows"]
                prior.require(inherited_rows, "missing inherited T3 root corpus")
                projected = {r["seed"] for r in inherited_rows}
                added_rows = [dict(r) for r in inherited_rows]
                entry["inherited_source_bindings"] = inherited["t3_source_bindings"]
                entry["source_manifest_sha256_from_frozen_catalog_rule"] = (
                    "0b216da882b634e79ac640a9970a58157828da96e93ec4dbe68e6924c8141391"
                )
                entry["opaque_input_archive_bindings"] = value["inputs"]
                entry["scope"] = (
                    "conservative entire prior T3/T4 root corpus, not selected label rows; "
                    "source catalog/reconstruct reuse original receipt rosters; input "
                    "archives and outcomes unopened"
                )
            elif kind == "r30_reservation_and_retry_chain_same_parent_corpus":
                projected = {
                    aliases[parent]["seed"] for parent in inherited["r30_1_all_splits"]["parents"]
                }
                entry["inherited_parent_count"] = 372
                entry["scope"] = (
                    "compute reservation only; timing.py consumes the existing admitted "
                    "corpus, creates no new physical world seeds"
                )
            else:
                projected = (
                    literal_seeds(value)
                    if kind in {"literal_seed_audit", "gpu_label_allocation_and_retry_aliases"}
                    else set()
                )
            if kind == "gpu_label_allocation_and_retry_aliases":
                for item in value["allocation"]:
                    added_rows.append(
                        {
                            "parent_id": item["allocation_id"],
                            "seed": item["seed"],
                            "aliases": [f"{value['namespace']}/{item['allocation_id']}"],
                        }
                    )
            # Preserve manifest/control bindings, but never follow archive/label refs.
            entry["allocation_binding"] = (
                value.get("allocation") if isinstance(value.get("allocation"), str) else None
            )
            entry["synthetic"] = value.get("synthetic")
            if "prior_allocations" in value:
                entry["retained_compute_attempts"] = [
                    {
                        k: r.get(k)
                        for k in (
                            "job_id",
                            "status",
                            "failure_class",
                            "sha256",
                            "stop_verified",
                            "not_replaced_by_successor_reservation",
                        )
                    }
                    for r in value["prior_allocations"]
                ]
        projected.update(r["seed"] for r in added_rows)
        entry["seed_count"] = len(projected)
        entry["seed_set_sha256"] = prior.digest(
            "".join(f"{x}\n" for x in sorted(projected)).encode()
        )
        entry["ancestry_edge_count"] = len(added_rows)
        seeds.update(projected)
        rows.extend(added_rows)
        entries.append(entry)
    prior.require(len(entries) == 135, "finite snapshot coverage changed")
    return entries, rows, seeds


def build() -> dict[str, Any]:
    """Build reviewable closure evidence while keeping protected gaps explicit."""
    snapshot_ref = {
        "path": str(PACKAGE / "campaign-package-register-filename-inventory.json"),
        "sha256": SNAPSHOT_SHA,
    }
    snapshot, _ = prior.read(snapshot_ref)
    prior.require(snapshot["source_count"] == 301, "snapshot scope changed")
    plan, _ = prior.read(
        {
            "path": str(PACKAGE / "input-plan-v2.json"),
            "sha256": "dcc4c1806b672449b3feb562282844c60ab5385f462a0377b058746645b422ea",
        }
    )
    manifest, rows, seeds = prior.reconcile(plan)
    aliases = {}
    for row in rows:
        for name in (row["parent_id"], *row.get("aliases", [])):
            if name in aliases:
                prior.require(aliases[name]["seed"] == row["seed"], "ambiguous historical alias")
            aliases[name] = row
    t3_bindings = [
        r
        for r in manifest["prior_repository_register"]
        if "/t3_screen_20261015/" in r["path"] or "/t4_curve_20261015/" in r["path"]
    ]
    t3_sources = {r["path"] for r in t3_bindings}
    t3_rows = [r for r in rows if r.get("source_id") in t3_sources]
    # The original catalog identifies source rows by the register path.
    entries, added_rows, additional_seeds = crosswalk(
        snapshot,
        aliases,
        {**manifest, "t3_source_bindings": t3_bindings, "t3_catalog_rows": t3_rows},
    )
    rows.extend(added_rows)
    seeds.update(additional_seeds)
    files: dict[str, Any] = {
        "additional-135-crosswalk.json": {
            "schema": "f5-finite-register-crosswalk/v3",
            "snapshot": snapshot_ref,
            "source_count": 135,
            "sources": entries,
        }
    }
    sources = [binding(REPO / name) for name in RULE_SOURCES]
    sources.append(binding(Path(__file__).absolute()))
    # These are lawful mirrors outside the protected trees. No path references
    # in these files grant permission to open the protected source.
    mirrors = []
    for name in (
        "research/t34_round0/BANK_ROUND0.json",
        "research/t34_round0/SEED_AUDIT_ROUND0.json",
    ):
        ref = binding(REPO / name)
        value, _ = prior.read(ref)
        projected = literal_seeds(value)
        seeds.update(projected)
        mirrors.append(
            {
                **ref,
                "physical_seed_count": len(projected),
                "scope": (
                    "public intended Round 0 bank only; protected complete-history "
                    "equivalence not attested"
                ),
            }
        )
    omissions = [
        {
            "id": "protected-D4096",
            "bank": "D4096",
            "attestable": True,
            "scope": (
                "all reserved, allocated, rejected, failed, aborted, retried, "
                "superseded and completed D4096 physical parent seeds, ultimate "
                "ancestry, continuation/life descendants and namespace aliases; public "
                "D256/D1024 life allocations do not identify any later D4096 collector "
                "parents"
            ),
            "lawful_mirror": None,
            "required_evidence": (
                "existing authorized custodian must supply external metadata "
                "catalog/register hashes and attest all 684 candidates together under "
                "the frozen rule, including the rejection chain; no new protected "
                "access authorized"
            ),
        },
        {
            "id": "protected-round0-history",
            "bank": "t34-round0-study-01",
            "attestable": True,
            "scope": (
                "complete protected study allocation/register lifecycle, including "
                "intended 240 bank worlds, 2 public preflight worlds, additional "
                "reservations/retries and ultimate ancestry/aliases if any"
            ),
            "lawful_mirror": mirrors,
            "required_evidence": (
                "existing custodian must verify public bank covers the protected study "
                "or supply the missing external seed/register mirror, with complete "
                "lifecycle and ancestry/alias comparison"
            ),
        },
        {
            "id": "opaque-discrepancy-source-binding",
            "attestable": True,
            "scope": (
                "17 snapshot discrepancy output copies were deliberately not opened; "
                "source-only fixture envelope is included but historical code/input "
                "identity and complete component-fixture ancestry must be verified"
            ),
            "snapshot_indices": sorted(OPAQUE_INDICES),
            "known_source_roots": prep_seeds("t34-speed-s0-baseline/v1", 6),
            "ad_hoc_sparse_cap160_unverified_index": 182,
            "required_evidence": (
                "custodian compares the predeclared fixture seed catalog and historical"
                " source pins; if any root is missing supply lawful metadata and rerun "
                "unchanged whole allocator"
            ),
        },
        {
            "id": "campaign-cutoff-completeness",
            "attestable": True,
            "scope": (
                "authoritative campaign/package registers at the snapshot cutoff, "
                "including source-specific physical seed normalization and ultimate "
                "ancestry/namespace aliases absent from seed-only historical "
                "inventories; each of the 135 files now has an explicit crosswalk"
            ),
            "required_evidence": (
                "existing custodian verifies complete allocation lifecycle/register "
                "coverage, source-specific seed interpretation and source-bound "
                "superset envelopes; no unsupported Sol assertion of universal coverage"
            ),
            "e3_historical_input_equivalence": {
                "snapshot_indices": list(range(72, 78)),
                "requirement": (
                    "verify each bound opaque input package inherits only the excluded "
                    "original T3/T4 root corpus and source-key ancestry/aliases; archives "
                    "were not opened"
                ),
            },
            "seed_only_registers_requiring_ancestry_attestation": [
                r["path"]
                for r in manifest["prior_repository_register"]
                if r.get("status") == "ADMITTED_LEXICAL_SEED_METADATA"
            ],
        },
    ]
    evaluation = prior.audit(rows, seeds)
    require_clear(evaluation)
    evaluation.update(
        {
            "seal_kind": "whole-684-observed-metadata-first-clear",
            "sealed_round": evaluation["candidate_round"],
            "sealed_allocation_sha256": evaluation["candidate_allocation_sha256"],
            "complete_coverage_attested": False,
            "omissions": omissions,
        }
    )
    manifest.update(
        {
            "schema": "f5-register-catalog-closure/v3",
            "prior_manifest_sha256": prior.digest(prior.canonical(manifest)),
            "additional_crosswalk_sha256": prior.digest(
                prior.canonical(files["additional-135-crosswalk.json"])
            ),
            "additional_source_count": 135,
            "comparison_rule": prior.RULE,
            "rule_source_bindings": sources,
            "lawful_protected_metadata_mirrors": mirrors,
            "unresolved": omissions,
            "unique_raw_seed_count": len(seeds),
            "unique_raw_seed_sha256": prior.digest(
                "".join(f"{x}\n" for x in sorted(seeds)).encode()
            ),
            "complete": False,
            "coverage_status": "FINITE_135_RECONCILED_CUSTODIAN_ATTESTATION_PENDING",
        }
    )
    # Publish the exact union, not only its count, so the operator can rerun the
    # allocator against hash-bound metadata without opening any outcomes.
    ordered = sorted(seeds)
    shards = []
    for offset in range(0, len(ordered), 75000):
        name = f"seed-union-{offset // 75000:03d}.json"
        value = {
            "schema": "f5-physical-seed-exclusion-union/v3",
            "seeds": ordered[offset : offset + 75000],
        }
        files[name] = value
        shards.append(
            {
                "path": name,
                "sha256": prior.digest(prior.canonical(value)),
                "count": len(value["seeds"]),
            }
        )
    manifest["seed_union_shards"] = shards
    manifest["alias_catalog_sha256"] = prior.digest(prior.canonical(rows))
    # Large alias catalogs remain at the artifact destination; bounded shards
    # allow selective publication without changing any source evidence.
    alias_shards = []
    for offset in range(0, len(rows), 2000):
        name = f"alias-records-{offset // 2000:03d}.json"
        value = {"records": rows[offset : offset + 2000]}
        files[name] = value
        alias_shards.append(
            {
                "path": name,
                "sha256": prior.digest(prior.canonical(value)),
                "count": len(value["records"]),
            }
        )
    manifest["alias_record_shards"] = alias_shards
    files["register-catalog-manifest.json"] = manifest
    files["allocation-evaluation.json"] = evaluation
    files["omissions.json"] = {
        "schema": "f5-attestable-omissions/v3",
        "omissions": omissions,
        "unattestable_omissions": [],
    }
    access_record = {
        "schema": "f5-metadata-access-record/v3",
        "root_operations": (
            "explicit pinned metadata and repository source; no gameplay, "
            "post-mortem, fit, inference, dispatch, payload archive or checkpoint "
            "reads"
        ),
        "opaque_discrepancy_outputs_opened": False,
        "protected_artifact_roots_opened": False,
        "delegated_search_incident": {
            "lane": "projection_rules",
            "command": (
                "pwd && git status --short && rg -n "
                '"bl-served-search-mac-identity|candidate_count|bank-round00|'
                'SEED_AUDIT|bl-ppo-p1-eval"'
                " research/bl_served_search research/bl_ppo research -g '*.py' -g "
                "'*.json' -g '*.md' -g '!**/D4096/**' -g '!**/t34-round0-study-01/**' "
                "-g '!**/outcome*' -g '!**/discrepancy*'"
            ),
            "error": "generic research directory accidentally included in source lookup",
            "visible_exposure": (
                "unrelated historical Q0 training telemetry; not used for allocation or"
                " scientific decisions"
            ),
            "omitted_output_auditable": False,
            "protected_artifact_root_passed": False,
            "f4_f5_or_protected_outcomes_visible": False,
            "remediation": (
                "lane stopped immediately; follow-up reads used exact source files; no "
                "wholly outcome-free delegated-search assertion"
            ),
        },
    }
    files["ACCESS_RECORD.json"] = access_record
    manifest["access_record_sha256"] = prior.digest(prior.canonical(access_record))
    certificate = prior.certificate_template(manifest, evaluation)
    certificate.update(
        {
            "seed_catalog_manifest_sha256": prior.digest(prior.canonical(manifest)),
            "campaign_package_register_filename_inventory_sha256": SNAPSHOT_SHA,
            "additional_135_crosswalk_sha256": manifest["additional_crosswalk_sha256"],
            "omissions_to_resolve_before_signing": omissions,
            "operator_checklist": [
                (
                    "Identify actual issuer and existing protected-bank authority; date and"
                    " attribute approval; do not grant new protected access."
                ),
                (
                    "Verify COMPLETED.json and every listed SHA256/size, input-plan-v2 and "
                    "source pins; preserve earlier immutable packets."
                ),
                (
                    "Reconcile all 301 snapshot sources: prior 166 source bindings plus "
                    "each of the 135 crosswalk entries; verify campaign cutoff and every "
                    "lifecycle state, including unused and superseded reservations."
                ),
                (
                    "Verify original 3024 allocation records and all F4 life/parent/fold "
                    "joins; training exclusion is exactly original 167 parents including 29"
                    " census parents and all descendants; broader freshness excludes all "
                    "historical uses."
                ),
                (
                    "Verify 372 R30-1 parents across 291 train and 81 validation, all "
                    "registered R30-2 real/reserved/synthetic records and every "
                    "timing/canary/retry package."
                ),
                (
                    "Resolve every item in omissions_to_resolve_before_signing: protected "
                    "D4096, protected Round0 history, discrepancy source binding and "
                    "campaign/physical-seed/ancestry/alias completeness. Supply separate "
                    "protected register and catalog SHA256s."
                ),
                (
                    "Confirm raw physical seed semantics per source; verify source-specific"
                    " signed/full-width handling, root-envelope derivations and "
                    "source-attested transitive ancestry/namespace aliases; never interpret"
                    " fit/assay seeds as physical roots."
                ),
                (
                    "Reproduce unchanged allocator source SHA256 "
                    "055cb66b5b7e270328c4ba67b83ff133a5ce9cb289d1857d2bfa9c361e0cc9ee; "
                    "iterate rounds 0..999 in order; reject the whole 684 on any collision "
                    "and retain all reasons."
                ),
                (
                    "Verify exact full candidate allocation SHA256 and all 672 science plus"
                    " 12 timing parents together against COMPLETE attested metadata; any "
                    "additional seed/alias requires a new packet and rerun, never "
                    "individual replacement or salt change."
                ),
                (
                    "Only after every coverage item is resolved, publish a NEW certificate "
                    "with assertions true, omissions=[], "
                    "collision_result_all_684=NO_COLLISION, exact round/hash and every "
                    "prior rejection; bind its content hash to an attributable approval "
                    "record. Template nulls are not assertions."
                ),
                (
                    "Verify common-environment amendment and ordering erratum hashes in the"
                    " closure packet. Independent F5 adoption publication still precedes "
                    "any diagnostic exposure; this metadata seal grants no execution or "
                    "release authority."
                ),
            ],
        }
    )
    files["CUSTODIAN_CERTIFICATE_TEMPLATE.json"] = certificate
    old_packet, old_binding = prior.read(binding(PACKAGE / "CLOSURE_PACKET_v2.json"))
    environment = {}
    for key in ("environment_amendment", "batch_ordering_clarification", "environment_json"):
        ref = old_packet[key]
        current = binding(REPO / ref["path"])
        prior.require(current["sha256"] == ref["sha256"], "environment binding changed")
        environment[key] = current
    files["CLOSURE_PACKET_v3.json"] = {
        "schema": "f5-finite-collision-closure-packet/v3",
        "status": "SEALED_METADATA_CLEAR_CUSTODIAN_ATTESTATION_PENDING",
        "accepted_round": None,
        "sealed_round": evaluation["sealed_round"],
        "sealed_allocation_sha256": evaluation["sealed_allocation_sha256"],
        "rejected_rounds": evaluation["rejected_rounds"],
        "register_catalog_manifest_sha256": prior.digest(prior.canonical(manifest)),
        "allocation_evaluation_sha256": prior.digest(prior.canonical(evaluation)),
        "certificate_template_sha256": prior.digest(prior.canonical(certificate)),
        "omissions_sha256": prior.digest(prior.canonical(files["omissions.json"])),
        "access_record_sha256": prior.digest(prior.canonical(access_record)),
        "prior_packet": old_binding,
        "environment": environment,
        "selected_960_scan_bindings_from_ruling_only": old_packet[
            "selected_960_scan_bindings_from_ruling_only"
        ],
        "diagnostic_exposure_authorized": False,
    }
    return files


def verify_packet(output: Path) -> dict:
    """Verify sealed metadata and reproduce the whole-round collision decision."""
    completion_ref = binding(output / "COMPLETED.json")
    completion, _ = prior.read(completion_ref)
    for name, expected in completion["files"].items():
        prior.require(Path(name).name == name, "completion filename escape")
        prior.require(binding(output / name)["sha256"] == expected, "sealed packet hash mismatch")
    manifest, _ = prior.read(binding(output / "register-catalog-manifest.json"))
    rows, values = [], []
    for ref in manifest["alias_record_shards"]:
        prior.require(Path(ref["path"]).name == ref["path"], "alias shard escape")
        shard, _ = prior.read({"path": str(output / ref["path"]), "sha256": ref["sha256"]})
        prior.require(len(shard["records"]) == ref["count"], "alias shard count")
        rows.extend(shard["records"])
    for ref in manifest["seed_union_shards"]:
        prior.require(Path(ref["path"]).name == ref["path"], "seed shard escape")
        shard, _ = prior.read({"path": str(output / ref["path"]), "sha256": ref["sha256"]})
        prior.require(len(shard["seeds"]) == ref["count"], "seed shard count")
        values.extend(shard["seeds"])
    prior.require(values == sorted(set(values)), "seed union order/duplicates")
    prior.require(len(values) == manifest["unique_raw_seed_count"], "seed union total count")
    prior.require(
        prior.digest("".join(f"{v}\n" for v in values).encode())
        == manifest["unique_raw_seed_sha256"],
        "seed union digest",
    )
    prior.require(
        prior.digest(prior.canonical(rows)) == manifest["alias_catalog_sha256"],
        "alias catalog digest",
    )
    prior.require(
        prior.digest(Path(prior.allocator.__file__).read_bytes()) == prior.ALLOCATOR_SHA256,
        "allocator changed",
    )
    expected, _ = prior.read(binding(output / "allocation-evaluation.json"))
    actual = prior.audit(rows, set(values))
    require_clear(actual)
    for key in actual:
        prior.require(actual[key] == expected[key], f"allocation reproduction mismatch: {key}")
    prior.require(
        expected["accepted_round"] is None and not expected["complete_coverage_attested"],
        "unattested acceptance",
    )
    return {
        "status": "VERIFIED",
        "completion_sha256": completion_ref["sha256"],
        "sealed_round": actual["candidate_round"],
        "allocation_sha256": actual["candidate_allocation_sha256"],
        "unique_seed_count": len(values),
        "omission_count": len(expected["omissions"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if args.verify:
        print(json.dumps(verify_packet(args.out)))
        return
    files = build()
    prior.publish(args.out, files)
    evaluation = files["allocation-evaluation.json"]
    print(
        json.dumps(
            {
                "sealed_round": evaluation["sealed_round"],
                "allocation_sha256": evaluation["sealed_allocation_sha256"],
                "additional_sources": 135,
                "attestable_omissions": len(evaluation["omissions"]),
                "accepted_round": None,
            }
        )
    )


if __name__ == "__main__":
    main()
