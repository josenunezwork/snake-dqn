"""Reproduce a metadata-only F5 R10a annex, refusing unresolved adoption claims.

Only the explicit, SHA-pinned metadata allowlist is read. References inside those
files are never followed: in particular no factory life, tensor, snapshot, label,
checkpoint, result, post-mortem runner or protected study is opened or hashed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

DRAFT_SHA = "6aeda9e66fbe67da41e0abf3549880a16f33bd52b448964b933234bbfb94f34d"
SEMANTIC_COMMIT = "124dd22cd560a74a210a61f0ba473461328e82f8"
NAMESPACE = "bl-f5-predat-v2/20261025"
NAME = "F5_R10A_METADATA_ANNEX_20261025"
GROUPS = (
    ("archive/watch6", 64),
    ("archive/scripted", 22),
    ("archive/frozen", 21),
    ("archive/mixed", 21),
    ("probe/scripted", 11),
    ("probe/frozen", 11),
    ("probe/mixed", 10),
    ("gameplay/scripted", 128),
    ("gameplay/frozen", 128),
    ("gameplay/mixed", 128),
    ("product/watch6", 128),
    ("timing/watch6", 3),
    ("timing/scripted", 3),
    ("timing/frozen", 3),
    ("timing/mixed", 3),
)
FILES = {
    "draft": "F5_DRAFT_PREDATA_REGISTRATION_20261025_v2.md",
    "judgment": "JUDGMENT_F5_V2_REVISE_20261026T002045162238Z.md",
    "development": "development-parents.json",
    "d256_allocation": "life-allocation.json",
    "d1024_allocation": "life-allocation.json",
    "d256_union": "D256_UNION_FINAL_20261023.json",
    "d1024_union": "D1024_UNION_FINAL_20261024.json",
    "d256_control": "control.json",
    "d1024_control": "control.json",
    "d256_collection": "d256-control.json",
    "d1024_collection": "d1024-control.json",
    "r30_inventory": "corpus-inventory.json",
}
SOURCE_PINS = {
    "research/bl_residual/fit.py": (
        "85c7e77d34e14a3b5036984cc1af92df" "4e3724f4c9fe453d813a56c1963822a7"
    ),
    "research/bl_residual/factory_masks.py": (
        "971358be6e80c515c2f4a223ec6e85ef" "892382fb629fe0396696b622823baf6f"
    ),
    "research/bl_labels/factory.py": (
        "06712668c6cadf292763d5f88d499dab" "5df6d03aaf7db85c9bae06ae1de2d45c"
    ),
    "research/bl_served_search/b2_engine.py": (
        "1baaaca7694bc7f03795c817b262dabc" "764bb65ff8803ac54e9fd46f5dc63323"
    ),
    "research/t34_shadow/shadow.py": (
        "9346b8ff447dd7322c78b0f87aa01e76" "03870e95b6a394e92104889880019058"
    ),
    "research/t3_rollout_labels_20261015/bank_engine.py": (
        "6f120c443b23ae7358ebe5b87a064bfc" "2c67d5198c9ff825e1ca9ec4c59909e1"
    ),
    "research/bl_ppo/collector.py": (
        "d1b3a1d34e00be4b473da9fd8fadf67b" "be7b1ff7e36687ab5721dd25d8f299e8"
    ),
    "research/apex_safety_20260926/dev_screen.py": (
        "50374f4f236ecac208f250eaacfc94d5" "142cce99b96989dae231744dd30211fe"
    ),
}


def sha(raw: bytes) -> str:
    """Hash bytes already permitted to be read."""
    return hashlib.sha256(raw).hexdigest()


def encoded(value: Any) -> bytes:
    """Canonical, finite JSON, including a final newline."""
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode()


def require(condition: bool, reason: str) -> None:
    """Refuse inconsistent identities instead of repairing them."""
    if not condition:
        raise ValueError(reason)


def safe_path(path: Path) -> None:
    """Reject protected paths and symlink escapes before any content read."""
    lowered = str(path.absolute()).lower()
    require("4096" not in lowered and "t34-round0-study-01" not in lowered, "forbidden path")
    require(
        not {"lives", "exports", "payloads", "outcomes", "results", "runtime-output"}
        & set(path.parts),
        "payload directory refused",
    )
    require(
        "postmortem" not in lowered
        or any(
            str(path).endswith("f4_postmortem_20261025/development-parents.json" + suffix)
            for suffix in ("", ".sha256", ".complete")
        ),
        "only the explicitly authorized post-mortem exclusion ledger is allowed",
    )
    require(not any(p.is_symlink() for p in (path, *path.parents)), "symlink metadata refused")


def load_sources(plan: dict) -> tuple[dict, list[dict]]:
    """Read only exact named metadata, verifying pins and publication sidecars."""
    require(set(plan) == {"schema", "sources"}, "unexpected input plan fields")
    require(plan["schema"] == "bl-f5-r10a-inputs/v1", "wrong input plan schema")
    require(set(plan["sources"]) == set(FILES), "incomplete or foreign source roles")
    values, bindings = {}, []
    for role, basename in FILES.items():
        ref = plan["sources"][role]
        require(set(ref) == {"path", "sha256", "seal"}, "unexpected source reference fields")
        path = Path(ref["path"])
        safe_path(path)
        require(path.is_absolute() and path.name == basename, "source outside metadata allowlist")
        require(ref["seal"] in ("sidecar", "complete"), "missing seal policy")
        require(path.is_file() and path.stat().st_size <= 16 * 1024**2, "invalid metadata size")
        raw = path.read_bytes()
        require(sha(raw) == ref["sha256"], f"changed metadata: {role}")
        sidecar = Path(str(path) + ".sha256")
        safe_path(sidecar)
        require(sidecar.read_text().split()[0] == ref["sha256"], f"bad sidecar: {role}")
        if ref["seal"] == "complete":
            marker = Path(str(path) + ".complete")
            # The exclusion ledger is the only post-mortem metadata exception.
            require(not marker.is_symlink(), "symlink completion marker")
            require(marker.read_bytes() == b"complete\n", f"unpublished metadata: {role}")
        values[role] = raw if role in ("draft", "judgment") else json.loads(raw)
        bindings.append({"role": role, **ref, "size": len(raw)})
    require(plan["sources"]["draft"]["sha256"] == DRAFT_SHA, "wrong v2 draft")
    return values, bindings


def phase(parent: str) -> dict:
    """Implement R6a's full-width rejection sampler without RNG/library imports."""
    limit = 2**256 - (2**256 % 200)
    for counter in range(1_000_000):
        digest = hashlib.sha256(f"{NAMESPACE}|ordinary-phase|{parent}|{counter}".encode())
        value = int.from_bytes(digest.digest(), "big")
        if value < limit:
            return {"q": value % 200, "counter": counter, "sha256": digest.hexdigest()}
    raise ValueError("HOLD: phase rejection guard")


def allocate(round_index: int) -> list[dict]:
    """Materialize the whole combined bank; never replace individual worlds."""
    require(type(round_index) is int and 0 <= round_index <= 999, "invalid round")
    rows = []
    for group, count in GROUPS:
        for index in range(count):
            parent = f"{NAMESPACE}/r={round_index}/{group}/{index:04d}"
            raw = f"{NAMESPACE}|r={round_index}/{group}|{index}".encode()
            row = {
                "allocation_index": len(rows),
                "parent_id": parent,
                "group": group,
                "group_index": index,
                "seed": int.from_bytes(hashlib.sha256(raw).digest()[:4], "big"),
                "timing_only": group.startswith("timing/"),
            }
            if group.startswith("archive/"):
                row["ordinary_phase"] = phase(parent)
            rows.append(row)
    return rows


def collision_audit(rows: list[dict], historical: list[dict]) -> list[dict]:
    """Audit explicit parent aliases and physical seeds, retaining all reasons."""
    reasons, seen_seeds, seen_parents = [], set(), set()
    for row in rows:
        seed, parent = row["seed"], row["parent_id"]
        if seed in seen_seeds or parent in seen_parents:
            reasons.append({"kind": "internal_collision", "parent_id": parent, "seed": seed})
        seen_seeds.add(seed)
        seen_parents.add(parent)
        for old in historical:
            if seed == old["seed"] or parent in old["aliases"]:
                reasons.append({"kind": "historical_collision", "candidate": parent, "old": old})
    return reasons


def corpus_metadata(values: dict) -> tuple[dict, dict, list[dict]]:
    """Reconcile life selection and lineage from metadata, never payload refs."""
    prior, current = values["d256_allocation"], values["d1024_allocation"]
    selected = current["doses"]["1024"]["train_lives"]
    prefix = prior["doses"]["256"]["train_lives"]
    require(len(selected) == len(set(selected)) == 960, "wrong D1024 selection")
    require(len(prefix) == 192 and selected[:192] == prefix, "D256 prefix mismatch")
    require(current["doses"]["256"]["train_lives"] == prefix, "changed D256 prefix")
    catalog = {}
    for row in current["catalog"]:
        life = row["life_id"]
        require(life not in catalog or catalog[life] == row, "conflicting catalog duplicate")
        require(current["parents"][life] == row["parent_world"], "conflicting parent map")
        catalog[life] = row
    prior_catalog = {}
    for row in prior["catalog"]:
        life = row["life_id"]
        require(
            life not in prior_catalog or prior_catalog[life] == row,
            "conflicting prior catalog duplicate",
        )
        prior_catalog[life] = row
    require(all(catalog[life] == prior_catalog[life] for life in prefix), "changed prefix metadata")
    development = values["development"]
    dev = set(development["parent_worlds"])
    require(len(dev) == development["parent_count"] == 29, "wrong permanent exclusion census")
    require(development["designation"] == "DEVELOPMENT_FOREVER", "wrong exclusion designation")
    inventories = values["r30_inventory"]["lives"]
    require([r["source_life_id"] for r in inventories] == selected, "R30 source set mismatch")
    units, original_worlds = {}, {}
    historical = []
    for role in ("d256_union", "d1024_union"):
        union = values[role]
        require(union["schema"] == "bl-labels-f0-verified-ledger/v1", "wrong union schema")
        collection = values[role.replace("union", "collection")]
        require(
            collection["allocation"] == union["allocation"]
            and collection["namespace"] == union["namespace"],
            "changed collection allocation",
        )
        for unit in union["units"]:
            key = (union["namespace"], unit["unit_id"])
            require(key not in units or units[key] == unit, "conflicting immutable unit")
            units[key] = unit
        for world in union["allocation"]:
            parent = world["allocation_id"]
            require(type(world["seed"]) is int, "unknown physical seed")
            require(
                parent not in original_worlds or original_worlds[parent] == world,
                "conflicting original allocation",
            )
            original_worlds[parent] = world
            historical.append(
                {
                    "parent_id": parent,
                    "seed": world["seed"],
                    "aliases": [parent, f"{union['namespace']}/{parent}"],
                    "source": role,
                }
            )
    fold0 = {
        parent
        for parent in original_worlds
        if int.from_bytes(
            hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
        )
        % 5
        == 0
    }
    blocked = dev | fold0
    lives, excluded = [], []
    for index, record in enumerate(inventories):
        life, row = selected[index], catalog[selected[index]]
        require(row["rescue_bearing"] is True, "selected life lacks original rescue stratum")
        namespace, parent, number = life.split("/")
        fold = (
            int.from_bytes(
                hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
            )
            % 5
        )
        unit = units[(namespace, f"{parent}--life-{int(number):06d}")]
        require(row["parent_world"] == record["ultimate_parent_id"] == parent, "unknown ancestry")
        require(current["folds"][life] == record["original_fold"] == fold, "changed fold")
        require(record["original_allocation"] == original_worlds[parent], "changed original world")
        require(
            record["payload_sha256"] == unit["sha256"] and record["payload_size"] == unit["size"],
            "conflicting payload identity metadata",
        )
        require(
            row["allocation_index"] == record["original_allocation_index"], "changed life index"
        )
        require(row["stream"] == record["stream"], "changed stream")
        item = {
            "selection_index": index,
            "life_id": life,
            "ultimate_parent_id": parent,
            "original_fold": fold,
            "original_allocation_index": row["allocation_index"],
            "stream": row["stream"],
            "original_world": original_worlds[parent],
            "payload_identity_from_ledger_only": unit,
            "namespace_embedded_commit_from_metadata": record["collection_source_commit"],
            "collector_identity_status": (
                "Namespace commit is not proof of the original collector implementation; "
                "see per-origin identity gaps."
            ),
        }
        (excluded if parent in blocked else lives).append(item)
    training_parents = {r["ultimate_parent_id"] for r in lives}
    siblings = []
    for row in sorted(catalog.values(), key=lambda r: r["allocation_index"]):
        if row["rescue_bearing"] or row["parent_world"] not in training_parents:
            continue
        require(row["rescue_bearing"] is False, "invalid ordinary sibling stratum flag")
        require(row["life_id"] not in selected, "selected life duplicated as ordinary sibling")
        namespace, parent, number = row["life_id"].split("/")
        require(parent == row["parent_world"], "unknown ordinary sibling ancestry")
        derived_fold = (
            int.from_bytes(
                hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
            )
            % 5
        )
        require(
            current["folds"][row["life_id"]] == derived_fold != 0, "changed ordinary sibling fold"
        )
        unit = units[(namespace, f"{parent}--life-{int(number):06d}")]
        siblings.append(
            {
                **row,
                "original_fold": current["folds"][row["life_id"]],
                "payload_identity_from_ledger_only": unit,
                "role": "original ordinary rows only",
            }
        )
    require(all(parent in original_worlds for parent in dev), "unresolved census world identity")
    exclusions = {
        "permanent_development_parents": sorted(dev),
        "permanent_development_count": len(dev),
        "original_fold0_parents": sorted(fold0),
        "original_fold0_count": len(fold0),
        "training_excluded_parent_union": sorted(blocked),
        "training_excluded_parent_count": len(blocked),
        "training_descendant_rule": (
            "Exclude all lives, rows, clones, continuations, snapshots and descendants "
            "of this parent union, including unexposed siblings."
        ),
        "fresh_scope": (
            "Additionally exclude every prior "
            "training/development/validation/probe/strict/serving allocation, all R30-1 "
            "parents and any R30-2 parents. Namespace changes never restore "
            "independence."
        ),
        "r30_1_parents_all_splits": sorted({r["ultimate_parent_id"] for r in inventories}),
        "r30_1_parent_count": len({r["ultimate_parent_id"] for r in inventories}),
        "d4096": "Categorically excluded; no directory/content/metadata inspection or hashing.",
        "protected_study": "t34-round0-study-01/ unopened; historical metadata mirror unresolved.",
        "descendant_expansion": (
            "Parent-level exclusion is transitive; no claim of an enumerated "
            "alias/descendant universe."
        ),
        "alias_audit_coverage": (
            "Canonical D256/D1024 allocation IDs, namespace-qualified parent IDs, "
            "source life IDs and original seeds reconciled. Other aliases/ultimate "
            "ancestry require external metadata coverage."
        ),
    }
    corpus = {
        "source_life_count": 960,
        "nested_d256_prefix_count": 192,
        "prefix_verified_by_metadata": True,
        "eligible_selected_life_count": len(lives),
        "eligible_selected_parent_count": len({r["ultimate_parent_id"] for r in lives}),
        "stream_life_counts": dict(Counter(r["stream"] for r in lives)),
        "excluded_selected_life_count": len(excluded),
        "excluded_selected_lives": excluded,
        "selected_lives": lives,
        "selected_lives_sha256": sha(encoded(lives)),
        "ordinary_only_sibling_lives": siblings,
        "ordinary_only_sibling_life_count": len(siblings),
        "total_eligible_life_metadata_count": len(lives) + len(siblings),
        "row_inventory": None,
        "row_count": None,
        "ordinary_row_count": None,
        "doom_row_count": None,
        "both_row_strata_verified": False,
        "ordinary_sibling_scope": (
            "Retain original ordinary rows from eligible training parents under the "
            "original F4 conversion. The 960 selected rescue lives alone do not "
            "enumerate those rows or sibling lives."
        ),
        "row_metadata_evidence": values["r30_inventory"]["row_totals"],
        "minimum_parent_support_met": len({r["ultimate_parent_id"] for r in lives}) >= 32,
    }
    return corpus, exclusions, historical


def source_closure(repo: Path) -> dict:
    """Verify selected declared source hashes at the draft's immutable commit."""
    files = {}
    for name, expected in SOURCE_PINS.items():
        raw = subprocess.run(
            ["git", "show", f"{SEMANTIC_COMMIT}:{name}"], cwd=repo, check=True, capture_output=True
        ).stdout
        require(sha(raw) == expected, f"draft source pin mismatch: {name}")
        files[name] = expected
    return {
        "semantic_commit": SEMANTIC_COMMIT,
        "declared_files_verified": files,
        "transitive_semantics_bound_by_commit": True,
        "constructors_executed": False,
    }


def build(values: dict, bindings: list[dict], closure: dict) -> dict:
    """Build an honest HOLD annex when the available metadata is incomplete."""
    corpus, exclusions, historical = corpus_metadata(values)
    rejected = []
    candidate, clear_round = None, None
    for round_index in range(1000):
        rows = allocate(round_index)
        reasons = collision_audit(rows, historical)
        if not reasons:
            candidate, clear_round = rows, round_index
            break
        rejected.append({"round": round_index, "reasons": reasons})
    origins = {}
    for dose in ("d256", "d1024"):
        control = values[f"{dose}_control"]
        collection = values[f"{dose}_collection"]
        origins[dose] = {
            "collection_control_fields": {
                key: value for key, value in collection.items() if key != "allocation"
            },
            "collection_allocation_sha256": sha(encoded(collection["allocation"])),
            "fit_input_hashes_metadata_only": control["input_hashes"],
            "original_environment": None,
            "numeric_library_versions": None,
            "rng_implementation_versions": None,
            "full_config_identity": None,
            "ordered_rosters": None,
            "applicable_cross_platform_tie_rule": None,
        }
    blockers = [
        (
            "Exact original row/ordinary-sibling inventory and both-stratum counts are "
            "unavailable in the admitted metadata; R30 explicitly defers row totals."
        ),
        (
            "Exact COV missing-action inventory, count U and inventory hash cannot be "
            "derived from life IDs or payload hashes. U is unknown, not zero; the "
            "46,080 ceiling cannot be assessed."
        ),
        (
            "Original per-origin environment/numeric-library/RNG/config/ordered-roster "
            "identities and the applicable frozen cross-platform tie-classification "
            "rule are unresolved; source pins alone do not instantiate these."
        ),
        (
            "Complete external seed/ultimate-ancestry/alias audit coverage is "
            "unresolved, including a lawful metadata mirror for prohibited study "
            "allocations and all prior development/validation/probe/strict/serving/R30 "
            "allocations. No accepted bank is asserted."
        ),
    ]
    if candidate is None:
        blockers.append("All 1,000 candidate rounds collide with the inspected metadata.")
    return {
        "schema": "bl-f5-r10a-metadata-annex/v1",
        "disposition": "HOLD_BEFORE_EXPOSURE",
        "adoption_ready": False,
        "adopted": False,
        "selection_receipt": None,
        "draft_sha256": DRAFT_SHA,
        "generator_sha256": sha(Path(__file__).read_bytes()),
        "input_bindings": bindings,
        "source_closure": closure,
        "corpus": corpus,
        "exclusions": exclusions,
        "exclusion_inventory_sha256": sha(encoded(exclusions)),
        "original_continuation_identities": origins,
        "fresh_continuation": {
            "hero_slot": 0,
            "hero_sha256": "3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8",
            "watch_opponent_sha256": (
                "eec144bf92509a42664e2d650b8741f7" "d7774b86d7b6601d012bdd9425dd3723"
            ),
            "watch_slots": [1, 2, 3, 4, 5],
            "horizon": 300,
            "backend": "rowwise fp32 CPU",
            "action_rule": "advisory resolved mask; normal-only empty fallback; first-index ties",
            "gate_ordered_rosters": None,
            "full_config_identity": None,
        },
        "cov": {
            "inventory": None,
            "count_U": None,
            "inventory_sha256": None,
            "ceiling_U": 46080,
            "maximum_branch_frames": 13824000,
            "ceiling_assessment": "UNRESOLVED",
            "prefix_selection_allowed": False,
        },
        "allocation": {
            "namespace": NAMESPACE,
            "groups": dict(GROUPS),
            "parent_count": 684,
            "science_parent_count": 672,
            "timing_parent_count": 12,
            "accepted_round": None,
            "accepted_allocation_sha256": None,
            "first_clear_round_against_inspected_metadata_only": clear_round,
            "provisional_parents": candidate,
            "provisional_allocation_sha256": sha(encoded(candidate)) if candidate else None,
            "rejected_rounds_against_inspected_metadata_only": rejected,
            "audit_complete": False,
            "inspected_historical_allocations": historical,
            "inspected_inventory_sha256": sha(encoded(historical)),
            "inspected_parent_count": len({r["parent_id"] for r in historical}),
            "inspected_seed_count": len({r["seed"] for r in historical}),
            "timing_exclusion": (
                "Timing parents excluded from every science bank, fit and checkpoint "
                "selection forever."
            ),
        },
        "execution": {
            "science_authorized_by_annex": False,
            "metadata_tool_processes": 1,
            "test_nice": 15,
            "test_threads": 1,
            "test_max_processes": 2,
            "registered_settings_and_forecasts": (
                "Exact v2 R8/R8b bytes bound by draft_sha256; no timing qualification " "performed."
            ),
            "registered_effective_hours_with_cov": 84,
            "registered_effective_hours_without_cov": 76,
        },
        "blockers": blockers,
        "access_attestation": {
            "label_or_outcome_payloads_opened_or_hashed": False,
            "referenced_payload_paths_followed": False,
            "d4096_accessed": False,
            "protected_study_accessed": False,
            "postmortem_run_or_modified": False,
            "colab_or_dispatch": False,
            "permitted_postmortem_read": (
                "Only the authorized development-parents exclusion ledger and publication "
                "sidecars."
            ),
        },
    }


def summary(annex: dict, json_sha: str) -> bytes:
    """Render a human summary bound to the exact machine-readable annex."""
    corpus, exclusions, bank = annex["corpus"], annex["exclusions"], annex["allocation"]
    lines = [
        "# F5 R10a metadata annex — HOLD before exposure",
        "",
        "This is a metadata-only HOLD record, not an adoption or selection receipt. The available",
        (
            "lawful metadata cannot satisfy all R10a(A) prerequisites. Diagnostic "
            "exposure remains blocked."
        ),
        "",
        f"v2 SHA256: `{DRAFT_SHA}`",
        f"Annex JSON SHA256: `{json_sha}`",
        f"Generator SHA256: `{annex['generator_sha256']}`",
        "",
        (
            "- Source selection: 960 unique selected lives, with the same 192-life D256 "
            "prefix (verified)."
        ),
        (
            f"- Eligible selected lives: {corpus['eligible_selected_life_count']}; "
            f"parents: {corpus['eligible_selected_parent_count']}; "
            f"excluded selected lives: {corpus['excluded_selected_life_count']}."
        ),
        (
            "- Original ordinary-only sibling lives: "
            f"{corpus['ordinary_only_sibling_life_count']}; "
            f"total life metadata records: {corpus['total_eligible_life_metadata_count']}. "
            "These add ordinary rows under the original F4 conversion, "
            "not new selected rescue lives."
        ),
        (
            f"- Permanent census exclusions: {exclusions['permanent_development_count']} parents "
            "plus every descendant; "
            f"original fold-0: {exclusions['original_fold0_count']} parents; "
            f"training exclusion union: {exclusions['training_excluded_parent_count']} parents."
        ),
        (
            "- D4096 categorically excluded and unopened. All "
            f"{exclusions['r30_1_parent_count']} R30-1 source parents "
            "are barred from fresh banks regardless of R30 split."
        ),
        (
            "- Row/ordinary-sibling counts and exact COV count U: **UNRESOLVED**; "
            "inventory/hash null. Ceiling: 46,080 units / 13,824,000 branch frames; no "
            "prefix selection."
        ),
        (
            "- 684 provisional parent records: 672 science and 12 timing. Accepted "
            "round/hash: **null**."
        ),
        (
            "- First clear round against inspected metadata only: "
            f"{bank['first_clear_round_against_inspected_metadata_only']}; "
            f"inspected historical parents: {bank['inspected_parent_count']}; "
            f"unique seeds: {bank['inspected_seed_count']}. "
            "This is an incomplete audit, not an accepted allocation."
        ),
        "",
        "## Unresolved adoption prerequisites",
        "",
        *[f"{i}. {reason}" for i, reason in enumerate(annex["blockers"], 1)],
        "",
        "## Evidence boundaries and reproduction",
        "",
        (
            "The JSON binds all input paths, byte hashes, source closure, original "
            "collection controls,"
        ),
        "per-life original allocation/seed/fold/ledger identities, both exclusion scopes, and all",
        (
            "provisional allocation records. Hashes of payloads are copied from "
            "metadata only. No payload"
        ),
        "reference was followed. No outcome, diagnostic aggregate, checkpoint, NPZ or snapshot was",
        (
            "opened/hashed; no protected study or D4096 access, post-mortem execution, "
            "Colab or dispatch."
        ),
        (
            "Original environment/config/RNG/roster/tie-rule gaps remain explicit, not "
            "inferred from the current Mac."
        ),
        "",
        "Reproduce with the command in README.md using the pinned input manifest. Publication is",
        (
            "create-only, deterministic and completion-last. A later metadata "
            "resolution requires a new"
        ),
        "sibling annex; these bytes must remain immutable. No adoption or exposure is authorized.",
        "",
    ]
    return "\n".join(lines).encode()


def publish(directory: Path, annex: dict) -> dict[str, str]:
    """Atomically create each file; publish the binding completion record last."""
    raw = encoded(annex)
    human = summary(annex, sha(raw))
    outputs = {f"{NAME}.json": raw, f"{NAME}.md": human}
    hashes = {name: sha(payload) for name, payload in outputs.items()}
    for name, digest in hashes.items():
        outputs[f"{name}.sha256"] = f"{digest}  {name}\n".encode()
    outputs[f"{NAME}.complete.json"] = encoded(
        {
            "schema": "bl-f5-r10a-publication/v1",
            "files": hashes,
            "disposition": annex["disposition"],
        }
    )
    safe_path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    require(
        not any((directory / name).exists() or (directory / name).is_symlink() for name in outputs),
        "create-only publication already exists",
    )
    for name, payload in outputs.items():
        descriptor, temporary = tempfile.mkstemp(prefix=".r10a-", dir=directory)
        try:
            with os.fdopen(descriptor, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.link(
                temporary, directory / name
            )  # atomic no-replace, even with competing publishers
        finally:
            os.unlink(temporary)
    return hashes


def main() -> None:
    """Run the explicitly pinned metadata report without any scientific work."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--publish", type=Path, action="append", required=True)
    args = parser.parse_args()
    safe_path(args.inputs)
    require(args.inputs.name == "R10A_INPUTS_20261025.json", "wrong input plan filename")
    input_raw = args.inputs.read_bytes()
    values, bindings = load_sources(json.loads(input_raw))
    annex = build(values, bindings, source_closure(Path(__file__).resolve().parents[2]))
    annex["input_manifest_sha256"] = sha(input_raw)
    for directory in args.publish:
        print(
            json.dumps(
                {
                    "directory": str(directory),
                    "files": publish(directory, annex),
                    "disposition": annex["disposition"],
                },
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
