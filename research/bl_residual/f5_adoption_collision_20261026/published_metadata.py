"""Create-only F5 successor using expressly published D4096 allocation metadata.

No referenced output, state, life, ledger, archive or protected tree is followed.
The original allocator, reconciler, v3 seal and access incident remain unchanged.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_residual.f5_adoption_collision_20261026 import close_snapshot as old

prior = old.prior
PACKAGE = old.PACKAGE
REPO = old.REPO
ARTIFACTS = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PARENT = ARTIFACTS / "big-lever/exp/astra-20261024/F5_ADOPTION_COLLISION_SEALED_20261026"
PARENT_SHA = "0ec121159dad466080df49c44c2f9980acb97ff62b151ffdb9f5d2afec0ce258"
ALLOCATION_SHA = "74b1c5db8cd0c082311cf2c02666418847855298762174e0b9f721762cbff229"
ORIGINAL_COMMIT = "bf8ac97dc66d1b4515e4c5510db3ad5886fa695d"
REBOUND_COMMIT = "111c47818a097c57451386ef4f8a7009616f8f77"
COLLECTOR_COMMIT = "61900af28932e47e22e332840fc4406559adb062"
PACK = ARTIFACTS / "bl-labels-f0-speed/d1024-d4096-pack-01"
ALLOWLIST = {
    PACK / "d4096-control.json": "c3fe9861cd177de57644175d487746f028024d1596fbeda1fdc57250a0a28207",
    PACK
    / "d4096-mac-hold.json": "87e738da7bebd2b0d979a2757e6e229f06df4bf0875fcfeb9b7797c1e1f27cc3",
    PACK
    / "d4096-seed-audit.json": "2e5a82961f9e3d246ecaf3f7ec9ebf5f48c524cc8e099fa18c22ae3841498a15",
    REPO
    / "research/bl_labels/ledger/D4096_OLD_EQUIV02_BINDING_20261024.json": (
        "32b4ecc988da7cdfe954bb5a50eb832711a162e02141af643ed98a090b743df4"
    ),
    REPO
    / "research/bl_labels/ledger/D4096_OLD_EQUIV02_ERRATUM_20261024.md": (
        "04460430a0b6e2bee440bcce0d00310c11b3b18be8485e0042a2270b654c951d"
    ),
}


def published_read(path: Path) -> tuple[bytes, dict]:
    """Permit exactly five published files without weakening the old path guard."""
    prior.require(path in ALLOWLIST, "not an expressly allowed published metadata file")
    prior.require(not any(p.is_symlink() for p in (*reversed(path.parents), path)), "symlink")
    prior.require(path.is_file() and path.stat().st_size <= prior.MAX_BYTES, "metadata bound")
    raw = path.read_bytes()
    prior.require(prior.digest(raw) == ALLOWLIST[path], "published metadata hash mismatch")
    if path.suffix == ".json":
        sidecar = Path(str(path) + ".sha256")
        prior.require(not sidecar.is_symlink(), "sidecar symlink")
        prior.require(sidecar.stat().st_size <= 1024, "sidecar bound")
        prior.require(sidecar.read_text().split()[0] == ALLOWLIST[path], "sidecar mismatch")
    return raw, {"path": str(path), "sha256": prior.digest(raw), "size": len(raw)}


def validate_allocation(control: dict, hold: dict, audit: dict) -> list[dict]:
    """Check the frozen array and source-derived identities, never regenerate it."""
    allocation = control["allocation"]
    raw = json.dumps(allocation, sort_keys=True, separators=(",", ":")).encode()
    prior.require(prior.digest(raw) == ALLOCATION_SHA, "allocation array digest")
    seeds = [r["seed"] for r in allocation]
    ids = [r["allocation_id"] for r in allocation]
    indices = [r["allocation_index"] for r in allocation]
    prior.require(len(allocation) == len(set(seeds)) == len(set(ids)) == 7020, "selected count")
    prior.require(indices == sorted(set(indices)), "allocation index order")
    prior.require((min(indices), max(indices)) == (3553, 12344), "index span")
    prior.require(audit["seeds"] == seeds, "ordered audit seeds")
    prior.require(audit["allocation_sha256"] == ALLOCATION_SHA, "audit allocation pin")
    selection = hold["selection"]
    prior.require(selection["allocation_sha256"] == ALLOCATION_SHA, "hold allocation pin")
    prior.require(
        selection["world_folds"] == audit["world_fold_filter"] == [1, 2, 3, 4], "fold set"
    )
    prior.require(selection["allocation_index_start"] == 3553, "hold start")
    prior.require(selection["allocation_index_end"] == 12344, "hold end")
    namespace = control["namespace"]
    prior.require(namespace == hold["namespace"], "namespace")
    prior.require(
        control["source_commit"] == hold["source_commit"] == ORIGINAL_COMMIT, "source pin"
    )
    rows = []
    for row in allocation:
        parent, seed = row["allocation_id"], row["seed"]
        fold = (
            int.from_bytes(
                hashlib.sha256(f"bl-residual-f3-split-v1/{parent}".encode()).digest()[:8], "big"
            )
            % 5
        )
        candidate = int(parent.rsplit("-", 1)[1])
        token = f"{namespace}/d4096/{row['stream']}/{candidate}"
        expected_seed = int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], "big")
        prior.require(
            type(seed) is int and 0 <= seed < 2**32 and seed == expected_seed, "physical seed"
        )
        prior.require(row["world_fold"] == fold and fold in (1, 2, 3, 4), "selected fold")
        if "plan" in row:
            prior.require(row["plan"]["seed"] == seed, "gate physical seed")
        rows.append(
            {
                "parent_id": parent,
                "seed": seed,
                "aliases": [f"{namespace}/{parent}"],
                "allocation_index": row["allocation_index"],
                "world_fold": fold,
                "source_id": str(PACK / "d4096-control.json"),
                "ancestry_scope": "published allocated parent only; external life joins unresolved",
            }
        )
    return rows


def source_bindings() -> list[dict]:
    """Bind explicit Git source objects, without importing or running collectors."""
    refs = [
        (ORIGINAL_COMMIT, "research/bl_labels/mac_package.py"),
        (ORIGINAL_COMMIT, "research/bl_residual/data.py"),
        (COLLECTOR_COMMIT, "research/bl_labels/factory.py"),
        (COLLECTOR_COMMIT, "research/bl_labels/mac_driver.py"),
        (COLLECTOR_COMMIT, "research/bl_labels/runtime.py"),
        (REBOUND_COMMIT, "research/bl_labels/mac_package.py"),
    ]
    result = []
    for commit, path in refs:
        raw = subprocess.run(
            ["git", "show", f"{commit}:{path}"], cwd=REPO, check=True, capture_output=True
        ).stdout
        result.append(
            {"commit": commit, "path": path, "sha256": prior.digest(raw), "size": len(raw)}
        )
    return result


def load_catalog(root: Path, manifest: dict) -> tuple[list[dict], set[int]]:
    rows, seeds = [], set()
    for key, field, target in (
        ("alias_record_shards", "records", rows),
        ("seed_union_shards", "seeds", seeds),
    ):
        for ref in manifest[key]:
            prior.require(Path(ref["path"]).name == ref["path"], "shard escape")
            value, _ = prior.read({"path": str(root / ref["path"]), "sha256": ref["sha256"]})
            prior.require(len(value[field]) == ref["count"], "shard count")
            target.extend(value[field]) if isinstance(target, list) else target.update(value[field])
    return rows, seeds


def build() -> dict[str, Any]:
    prior.require(old.binding(PARENT / "COMPLETED.json")["sha256"] == PARENT_SHA, "parent seal pin")
    old.verify_packet(PARENT)
    manifest, _ = prior.read(old.binding(PARENT / "register-catalog-manifest.json"))
    rows, seeds = load_catalog(PARENT, manifest)
    admitted, bindings = {}, []
    for path in ALLOWLIST:
        raw, ref = published_read(path)
        bindings.append(ref)
        if path.suffix == ".json":
            admitted[path.name] = json.loads(raw)
    control, hold, audit = (
        admitted[n] for n in ("d4096-control.json", "d4096-mac-hold.json", "d4096-seed-audit.json")
    )
    added_rows = validate_allocation(control, hold, audit)
    rebinding = admitted["D4096_OLD_EQUIV02_BINDING_20261024.json"]
    prior.require(rebinding["allocation_sha256"] == ALLOCATION_SHA, "rebinding allocation")
    for name in ("d4096-control.json", "d4096-mac-hold.json", "d4096-seed-audit.json"):
        prior.require(
            rebinding["original_files"][name]["sha256"] == ALLOWLIST[PACK / name],
            "rebinding original",
        )
    selected = {r["seed"] for r in added_rows}
    added, overlap = selected - seeds, selected & seeds
    prior.require(len(added) == 7019 and len(overlap) == 1, "published omission count changed")
    rows.extend(added_rows)
    seeds.update(selected)
    evaluation = prior.audit(rows, seeds)
    old.require_clear(evaluation)
    direct = sorted(selected & {r["seed"] for r in evaluation["candidate_allocation"]})
    checklist = [
        "Verify all three original pack file hashes and sidecars, plus the binding/erratum "
        "hashes; never follow state/output/life references.",
        "Recompute sorted-key compact JSON allocation SHA256 with NO trailing newline: "
        + ALLOCATION_SHA
        + "; require exactly 7020 distinct ordered seeds matching the seed audit. "
        "The 3553..12344 interval has 8792 positions, not 8792 selected parents.",
        "Verify all 684 F5 parents together, including 672 science and 12 timing, against all "
        "7020 published seeds and the entire successor catalog using the unchanged allocator; "
        "preserve whole-round rejection reasons.",
        "Check pinned mac_package.future_allocation and data.world_fold: allocation IDs, "
        "per-stream candidate counters, append-only index gaps, folds 1--4 and "
        "first-four-SHA256-byte physical seeds; gate plan seeds must match.",
        "Distinguish retained bf8ac97d namespace identity, rebound source 111c4781 and OLD "
        "collector closure 61900af2. Bind actual external invocation/source/config/input pins "
        "to these source objects; source availability does not prove historical execution.",
        "Reconcile existing external dispatch, reservation, rejection, failure, abort, retry, "
        "supersession and completion records through cutoff 2026-10-26T02:13:30.697703+00:00, "
        "including unused and fold-excluded candidate identities outside the selected rows. "
        "Supply record paths, SHA256s and dispositions; do not infer exhaustion from selected N.",
        "Join every external continuation/life ID and namespace alias to its original "
        "allocation ID and physical root using pre-existing metadata and pinned "
        "factory/mac_driver/runtime rules; preserve transitive ancestry and source-specific "
        "normalization. Do not open outcomes/state/lives to resolve missing joins.",
        "Issue only scoped findings with issuer, existing authority, timestamp, source hashes, "
        "comparison rule and residual omissions. Leave unsupported lifecycle/ancestry/"
        "completeness assertions null; supply missing lawful metadata and reseal/rerun if "
        "coverage changes.",
    ]
    report = {
        "schema": "f5-published-allocation-reconciliation/v4",
        "authority": old.binding(
            ARTIFACTS / "big-lever/exp/astra-20261024/JUDGMENT_F5_RESIDUAL_RISK_20261025.md"
        ),
        "allocation_sha256": ALLOCATION_SHA,
        "selected_rows": 7020,
        "selected_distinct_seeds": len(selected),
        "index_start": 3553,
        "index_end": 12344,
        "inclusive_index_span": 8792,
        "seeds_added": len(added),
        "already_excluded_seeds": sorted(overlap),
        "published_metadata_bindings": bindings,
        "collector_source_bindings": source_bindings(),
        "direct_all_684_seed_collisions": direct,
        "lifecycle_ancestry_complete": False,
        "operator_checklist": checklist,
        "source_identity": {
            "namespace_commit": ORIGINAL_COMMIT,
            "rebound_source_commit": REBOUND_COMMIT,
            "old_collector_closure": COLLECTOR_COMMIT,
        },
    }
    omissions, _ = prior.read(old.binding(PARENT / "omissions.json"))
    omissions = copy.deepcopy(omissions)
    omissions["schema"] = "f5-scoped-omissions/v4"
    item = next(x for x in omissions["omissions"] if x["id"] == "protected-D4096")
    item.update(
        {
            "scope": (
                "D4096 historical lifecycle, rejected/reserved/fold-excluded identities, "
                "transitive ancestry, continuation/life joins and namespace aliases beyond "
                "the checked published selected allocation"
            ),
            "lawful_mirror": bindings,
            "published_selected_allocation_verified": True,
            "allocation_sha256": ALLOCATION_SHA,
            "selected_seed_count": 7020,
            "remaining_coverage_verified": False,
            "operator_checklist": checklist,
            "required_evidence": (
                "Existing external lifecycle/source/ancestry records with attributable paths "
                "and hashes through the declared cutoff; published selected allocation alone "
                "cannot establish complete history. No new protected access authorized."
            ),
        }
    )
    for item in omissions["omissions"]:
        item["attestable"] = False
        item["attestation_status"] = (
            "UNRESOLVED_UNLESS_EXISTING_EXTERNAL_EVIDENCE_ESTABLISHES_CLAIM"
        )
    omissions["unattestable_omissions"] = copy.deepcopy(omissions["omissions"])
    omissions["note"] = (
        "No claim is certified unknowable forever; current packet does not establish these "
        "four claims. No custodian title supplies missing evidence."
    )
    evaluation.update(
        seal_kind="whole-684-published-metadata-successor",
        sealed_round=evaluation["candidate_round"],
        sealed_allocation_sha256=evaluation["candidate_allocation_sha256"],
        complete_coverage_attested=False,
        omissions=omissions["omissions"],
    )
    access = {
        "schema": "f5-metadata-access-record/v4",
        "parent_access_record": old.binding(PARENT / "ACCESS_RECORD.json"),
        "reads": (
            "Parent exported catalog/aliases, Astra judgment and explicit five-file published "
            "allowlist plus pinned Git source only; references never followed."
        ),
        "d4096_outputs_state_lives_opened": False,
        "protected_round0_opened": False,
        "parent_delegated_search_incident_preserved": True,
        "diagnostic_exposure_authorized": False,
    }
    manifest = copy.deepcopy(manifest)
    manifest.update(
        schema="f5-register-catalog-closure/v4",
        parent_completion_sha256=PARENT_SHA,
        parent_catalog_manifest=old.binding(PARENT / "register-catalog-manifest.json"),
        access_record_sha256=prior.digest(prior.canonical(access)),
        unique_raw_seed_count=len(seeds),
        complete=False,
        unique_raw_seed_sha256=prior.digest("".join(f"{x}\n" for x in sorted(seeds)).encode()),
        alias_catalog_sha256=prior.digest(prior.canonical(rows)),
        unresolved=omissions["omissions"],
        coverage_status="PUBLISHED_D4096_SELECTED_ALLOCATION_ADDED_LIFECYCLE_PENDING",
        published_allocation_sha256=prior.digest(prior.canonical(report)),
        successor_source_binding=old.binding(Path(__file__).absolute()),
    )
    files: dict[str, Any] = {}
    for key, field, values, width, prefix in (
        ("seed_union_shards", "seeds", sorted(seeds), 75000, "seed-union"),
        ("alias_record_shards", "records", rows, 2000, "alias-records"),
    ):
        refs = []
        for offset in range(0, len(values), width):
            name = f"{prefix}-{offset // width:03d}.json"
            value = {field: values[offset : offset + width]}
            files[name] = value
            refs.append(
                {
                    "path": name,
                    "sha256": prior.digest(prior.canonical(value)),
                    "count": len(value[field]),
                }
            )
        manifest[key] = refs
    certificate, _ = prior.read(old.binding(PARENT / "CUSTODIAN_CERTIFICATE_TEMPLATE.json"))
    certificate = copy.deepcopy(certificate)
    certificate.update(
        register_catalog_manifest_sha256=prior.digest(prior.canonical(manifest)),
        seed_catalog_manifest_sha256=prior.digest(prior.canonical(manifest)),
        allocation_evaluation_sha256=prior.digest(prior.canonical(evaluation)),
        omissions_to_resolve_before_signing=omissions["omissions"],
        published_allocation_findings=report,
        d4096_operator_checklist=checklist,
        instructions=(
            "Publish a NEW scoped certificate only for claims supported by existing lawful "
            "external evidence. Null completeness assertions and residual omissions remain "
            "unresolved; no new protected access, dispatch, adoption or release authorized."
        ),
        candidate_round=evaluation["candidate_round"],
        candidate_allocation_sha256=evaluation["candidate_allocation_sha256"],
        prior_round_rejections=evaluation["rejected_rounds"],
    )
    certificate["operator_checklist"][5] = (
        "Resolve residual D4096 lifecycle/ancestry plus Round0, discrepancy and campaign "
        "claims only where pre-existing external evidence establishes them. Checked public "
        "banks do not prove complete history."
    )
    files.update(
        {
            "register-catalog-manifest.json": manifest,
            "allocation-evaluation.json": evaluation,
            "omissions.json": omissions,
            "published-allocation.json": report,
            "CUSTODIAN_CERTIFICATE_TEMPLATE.json": certificate,
            "ACCESS_RECORD.json": access,
        }
    )
    parent_packet, _ = prior.read(old.binding(PARENT / "CLOSURE_PACKET_v3.json"))
    files["CLOSURE_PACKET_v4.json"] = {
        "schema": "f5-published-metadata-closure-packet/v4",
        "accepted_round": None,
        "status": "SEALED_METADATA_CLEAR_RESIDUAL_COVERAGE_UNRESOLVED",
        "parent_completion_sha256": PARENT_SHA,
        "sealed_round": evaluation["sealed_round"],
        "sealed_allocation_sha256": evaluation["sealed_allocation_sha256"],
        "rejected_rounds": evaluation["rejected_rounds"],
        "environment": parent_packet["environment"],
        "selected_960_scan_bindings_from_ruling_only": parent_packet[
            "selected_960_scan_bindings_from_ruling_only"
        ],
        "file_hashes": {
            name: prior.digest(prior.canonical(value))
            for name, value in files.items()
            if not name.startswith(("seed-union-", "alias-records-"))
        },
        "diagnostic_exposure_authorized": False,
    }
    return files


def verify(output: Path) -> dict:
    """Replay allocation and verify the exact published projection and union delta."""
    result = old.verify_packet(output)
    manifest, _ = prior.read(old.binding(output / "register-catalog-manifest.json"))
    rows, seeds = load_catalog(output, manifest)
    report, _ = prior.read(old.binding(output / "published-allocation.json"))
    prior.require(old.binding(PARENT / "COMPLETED.json")["sha256"] == PARENT_SHA, "parent pin")
    old.verify_packet(PARENT)
    parent_manifest, _ = prior.read(old.binding(PARENT / "register-catalog-manifest.json"))
    parent_rows, parent_seeds = load_catalog(PARENT, parent_manifest)
    admitted, bindings = {}, []
    for path in ALLOWLIST:
        raw, ref = published_read(path)
        bindings.append(ref)
        if path.suffix == ".json":
            admitted[path.name] = json.loads(raw)
    expected_rows = validate_allocation(
        *(
            admitted[n]
            for n in ("d4096-control.json", "d4096-mac-hold.json", "d4096-seed-audit.json")
        )
    )
    verify_delta(parent_rows, parent_seeds, rows, seeds, expected_rows, report)
    prior.require(report["published_metadata_bindings"] == bindings, "metadata bindings")
    prior.require(report["collector_source_bindings"] == source_bindings(), "collector source pins")
    prior.require(
        manifest["successor_source_binding"] == old.binding(Path(__file__).absolute()),
        "successor source pin",
    )
    prior.require(report["allocation_sha256"] == ALLOCATION_SHA, "published allocation pin")
    prior.require(
        manifest["published_allocation_sha256"] == prior.digest(prior.canonical(report)),
        "report binding",
    )
    certificate, _ = prior.read(old.binding(output / "CUSTODIAN_CERTIFICATE_TEMPLATE.json"))
    evaluation, _ = prior.read(old.binding(output / "allocation-evaluation.json"))
    omissions, _ = prior.read(old.binding(output / "omissions.json"))
    access, _ = prior.read(old.binding(output / "ACCESS_RECORD.json"))
    prior.require(
        certificate["register_catalog_manifest_sha256"]
        == certificate["seed_catalog_manifest_sha256"]
        == prior.digest(prior.canonical(manifest)),
        "certificate manifest binding",
    )
    prior.require(
        certificate["allocation_evaluation_sha256"] == prior.digest(prior.canonical(evaluation)),
        "certificate allocation binding",
    )
    prior.require(
        manifest["unresolved"]
        == evaluation["omissions"]
        == certificate["omissions_to_resolve_before_signing"]
        == omissions["omissions"],
        "omission bindings",
    )
    prior.require(
        manifest["access_record_sha256"] == prior.digest(prior.canonical(access)),
        "access record binding",
    )
    prior.require(certificate["published_allocation_findings"] == report, "certificate findings")
    prior.require(
        report["direct_all_684_seed_collisions"]
        == sorted(
            {r["seed"] for r in expected_rows}
            & {r["seed"] for r in evaluation["candidate_allocation"]}
        ),
        "direct comparison",
    )
    prior.require(certificate["status"] == "UNATTESTED_TEMPLATE", "unattested template required")
    prior.require(
        all(v is None for v in certificate["assertions"].values()), "unsupported assertion"
    )
    prior.require(
        certificate["issuer"] is None and certificate["omissions"] is None, "custody claim"
    )
    result.update(
        seeds_added=report["seeds_added"], published_selected_seed_count=len(expected_rows)
    )
    return result


def verify_delta(
    parent_rows: list[dict],
    parent_seeds: set[int],
    rows: list[dict],
    seeds: set[int],
    expected_rows: list[dict],
    report: dict,
) -> None:
    """Refuse substituted parents, removed history or a dishonest addition count."""
    prior.require(
        rows == parent_rows + expected_rows, "published rows or inherited ancestry changed"
    )
    selected = {r["seed"] for r in expected_rows}
    prior.require(seeds == parent_seeds | selected, "successor union delta")
    prior.require(report["selected_rows"] == report["selected_distinct_seeds"] == 7020, "report N")
    prior.require(report["seeds_added"] == len(selected - parent_seeds) == 7019, "added count")
    prior.require(report["already_excluded_seeds"] == sorted(selected & parent_seeds), "overlap")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        prior.publish(args.out, build())
    print(json.dumps(verify(args.out)))


if __name__ == "__main__":
    main()
