"""Prepare, but never dispatch, the Amendment-04 P1 canary/source pair.

The pair deliberately ships one identical input archive.  It contains distinct,
identity-bound source and canary controls plus empty ledgers.  ``p1_colab.sh``
selects the canary control only when the supervisor's family-canary environment
is present.  Therefore the qualified family-proof still covers every uploaded
byte while the two natural collections cannot consume the same world roots.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_inputs import unpack_bundle

ROOT = Path(__file__).resolve().parents[2]
CANARY_OFFSET = 3_000_000
SOURCE_OFFSET = 3_001_000
CANARY_CONFIG = {"pause_at": 32, "processes": 1, "rollout_steps": 16, "worlds": 2}
SOURCE_EXPORTS = ["receipt.json"]
CANARY_EXPORTS = SOURCE_EXPORTS + ["canary/pause.json", "canary/pause-snapshot.pt"]
QUALIFIED_CLI_TREE = "553af3f692515b2bf66173356fe4d099acc96942"
REVIEW_HOLD = "HOLD: C2 first meaningful fixture failed; judgment kill rule forbids Colab spend"


def _ledger(control: dict[str, Any]) -> dict[str, Any]:
    resume = control.get("resume")
    if not isinstance(resume, dict):
        raise ValueError("P1 source control lacks resume declaration")
    return {
        "schema": "bl-p1-verified-ledger/v1",
        "namespace": resume["namespace"],
        "allocation": resume["allocation"],
        "units": [],
    }


def _require_source_control(control: dict[str, Any], *, offset: int) -> None:
    if control.get("phase") != "y3-source" or control.get("seed") != 20261021:
        raise ValueError("Amendment-04 packs require the fixed Y3 source control")
    if control.get("e8_critic_convergence_amendment_04") is not True:
        raise ValueError("source control must enable Amendment-04")
    if control.get("e8_critic_convergence_amendment_03") is not False:
        raise ValueError("Amendment-03 must be explicitly disabled for Amendment-04")
    if control.get("natural_collection_world_id_offset") != offset:
        raise ValueError("source control offset differs from the registered source allocation")
    if not isinstance(control.get("e8_critic_convergence_amendment_04_sha256"), str):
        raise ValueError("source control lacks the Amendment-04 digest")


def _require_audit(path: Path, *, offset: int, control: dict[str, Any]) -> None:
    if verify_export(path) != control.get("collection_allocation_audit_sha256"):
        raise ValueError("collection allocation audit digest differs from its control")
    audit = json.loads(path.read_text())
    if (
        audit.get("schema") != "bl-p1-collection-allocation-audit/v1"
        or audit.get("natural_collection_world_id_offset") != offset
        or audit.get("collision_count") != 0
        or audit.get("burned_roots") != control.get("natural_collection_burned_roots")
    ):
        raise ValueError("collection allocation audit does not prove the source control")


def _canary_control(source: dict[str, Any], audit: Path) -> dict[str, Any]:
    value = copy.deepcopy(source)
    value["natural_collection_world_id_offset"] = CANARY_OFFSET
    # The operational canary must not draw from the source's root stream even
    # if a future allocation mistake were to reuse an offset.
    value["natural_collection_seed"] = 2026102203
    value["collection_allocation_audit_sha256"] = verify_export(audit)
    value["collection_allocation_audit_path"] = "canary-collection-allocation-audit.json"
    value["resume"] = {
        **value["resume"],
        "namespace": f"{value['resume']['namespace']}_canary",
        "allocation": f"{value['resume']['allocation']}_canary",
        "ledger_path": "inputs/canary-verified-ledger.json",
    }
    return value


def bootstrap_source_template(
    *, source_manifest: Path, root: Path, commit: str, amendment: Path, source_audit: Path
) -> Path:
    """Rebind the permitted source-05 template into a review-only source-06 staging input.

    This does not admit an experiment.  It preserves the prior public/evaluation
    inputs while replacing only the amendment-bound control, empty ledger and
    source identity.  The C2 kill rule is recorded directly in the control so
    neither this staging template nor any pack derived from it can allocate.
    """
    from research.bl_ppo.manifests import _bounded_source_paths, input_row
    from research.bl_ppo.p1_manifests import bundle_inputs

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=False)
    template = json.loads(source_manifest.read_text())
    old_bundle = next(
        (row for row in template.get("inputs", []) if row.get("dest") == "p1-inputs.tar.gz"), None
    )
    if not isinstance(old_bundle, dict) or digest(Path(old_bundle["path"])) != old_bundle.get(
        "sha256"
    ):
        raise ValueError("source-05 input archive is missing or changed")
    unpacked = root / "unpacked-source05"
    unpack_bundle(Path(old_bundle["path"]), unpacked)
    old_control = unpacked / "p1-control.json"
    verify_export(old_control)
    control = json.loads(old_control.read_text())
    control.update(
        source_commit=commit,
        e8_critic_convergence_amendment_03=False,
        e8_critic_convergence_amendment_04=True,
        e8_critic_convergence_amendment_04_sha256=digest(amendment),
        natural_collection_world_id_offset=SOURCE_OFFSET,
        natural_collection_seed=20261021,
        collection_allocation_audit_sha256=verify_export(source_audit),
        collection_allocation_audit_path="collection-allocation-audit.json",
        scientific_admission=False,
        dispatch_blockers=[REVIEW_HOLD],
    )
    source_audit_value = json.loads(source_audit.read_text())
    control["natural_collection_burned_roots"] = source_audit_value["burned_roots"]
    control["resume"] = {
        **control["resume"],
        "namespace": "bl_p1_y3_source_latebody_06_e8_04",
        "allocation": "source_latebody_06_e8_04",
        "ledger_path": "inputs/verified-ledger.json",
    }
    _require_source_control(control, offset=SOURCE_OFFSET)
    _require_audit(source_audit, offset=SOURCE_OFFSET, control=control)
    new_control = root / "p1-control.json"
    new_ledger = root / "verified-ledger.json"
    export_json(new_control, control)
    export_json(new_ledger, _ledger(control))
    omitted = {
        "p1-control.json",
        "p1-control.json.sha256",
        "p1-control.json.complete",
        "verified-ledger.json",
        "verified-ledger.json.sha256",
        "verified-ledger.json.complete",
    }
    rows = [
        input_row(item, item.name)
        for item in sorted(unpacked.iterdir())
        if item.name not in omitted
    ]
    for path in (new_control, new_ledger):
        for suffix in ("", ".sha256", ".complete"):
            item = Path(str(path) + suffix)
            rows.append(input_row(item, item.name))
    inputs = [row for row in template["inputs"] if row["dest"] != "p1-inputs.tar.gz"]
    inputs += bundle_inputs(root, rows)
    template.update(
        commit=commit,
        job_id=f"bl-p1-y3-source-e8-04-review-{commit[:8]}",
        output_dir=str(root / "source-output"),
        inputs=inputs,
        resume=control["resume"],
    )
    template["env"]["BL_PPO_COMMIT"] = commit
    template.pop("canary", None)
    template["bl_p1"] = {**template.get("bl_p1", {}), "control": control, "blockers": [REVIEW_HOLD]}
    template["source_paths"] = sorted(
        set(template.get("source_paths", [])) | set(_bounded_source_paths(template))
    )
    destination = root / "source-template-bootstrap.json"
    export_json(destination, template)
    return destination


def _input_rows(
    bundle_dir: Path,
    source_control: Path,
    canary_control: Path,
    source_ledger: Path,
    canary_ledger: Path,
    source_audit: Path,
    canary_audit: Path,
) -> list[dict[str, str]]:
    from research.bl_ppo.manifests import input_row

    omitted = {
        "p1-control.json",
        "p1-control.json.sha256",
        "p1-control.json.complete",
        "verified-ledger.json",
        "verified-ledger.json.sha256",
        "verified-ledger.json.complete",
    }
    rows = [
        input_row(path, path.name)
        for path in sorted(bundle_dir.iterdir())
        if path.name not in omitted
    ]
    for path in (
        source_control,
        canary_control,
        source_ledger,
        canary_ledger,
        source_audit,
        canary_audit,
    ):
        for suffix in ("", ".sha256", ".complete"):
            member = Path(str(path) + suffix)
            rows.append(input_row(member, member.name))
    return rows


def prepare(
    *, source_manifest: Path, root: Path, commit: str, source_audit: Path, canary_audit: Path
) -> dict[str, Any]:
    """Create a dual-control, shared-upload preparation directory.

    ``source_manifest`` is a freshly prepared, unadmitted Amendment-04 source
    template.  This function is local-only and refuses to create an admitted
    source manifest, so a genuine terminal canary receipt remains necessary.
    """
    from research.bl_ppo.manifests import _bounded_source_paths
    from research.bl_ppo.p1_manifests import bundle_inputs
    from research.bl_upload.qualify import qualify
    from research.colab_cli.canary import input_inventory
    from research.colab_cli.canary import prepare as probe_manifest
    from research.colab_cli.supervisor import validate

    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if any(item.name != "bootstrap" for item in root.iterdir()):
        raise FileExistsError("Amendment-04 preparation root must be fresh")
    template = json.loads(source_manifest.read_text())
    if template.get("commit") != commit or template.get("env", {}).get("BL_P1_MODE") != "y3-source":
        raise ValueError("requires a fresh Amendment-04 Y3 source template at the requested commit")
    bundled = next(
        (row for row in template.get("inputs", []) if row.get("dest") == "p1-inputs.tar.gz"), None
    )
    if not isinstance(bundled, dict) or digest(Path(bundled["path"])) != bundled.get("sha256"):
        raise ValueError("source template input archive is missing or changed")
    unpacked = root / "unpacked-source-inputs"
    unpack_bundle(Path(bundled["path"]), unpacked)
    original_control = unpacked / "p1-control.json"
    verify_export(original_control)
    source_control_value = json.loads(original_control.read_text())
    _require_source_control(source_control_value, offset=SOURCE_OFFSET)
    _require_audit(source_audit, offset=SOURCE_OFFSET, control=source_control_value)
    canary_control_value = _canary_control(source_control_value, canary_audit)
    _require_audit(canary_audit, offset=CANARY_OFFSET, control=canary_control_value)

    source_control = root / "p1-control.json"
    canary_control = root / "canary-control.json"
    source_ledger = root / "verified-ledger.json"
    canary_ledger = root / "canary-verified-ledger.json"
    export_json(source_control, source_control_value)
    export_json(canary_control, canary_control_value)
    export_json(source_ledger, _ledger(source_control_value))
    export_json(canary_ledger, _ledger(canary_control_value))
    rows = [row for row in template["inputs"] if row["dest"] != "p1-inputs.tar.gz"]
    rows += _input_rows(
        unpacked,
        source_control,
        canary_control,
        source_ledger,
        canary_ledger,
        source_audit,
        canary_audit,
    )
    shared_inputs = bundle_inputs(root, rows)

    source = copy.deepcopy(template)
    source.update(
        job_id=f"bl-p1-y3-source-e8-04-{commit[:8]}",
        output_dir=str(root / "source-output"),
        required_exports=SOURCE_EXPORTS,
        inputs=shared_inputs,
    )
    source["env"]["BL_PPO_COMMIT"] = commit
    source["resume"] = source_control_value["resume"]
    # Dispatch's wrapper reads this top-level fuse before any supervisor side
    # effect; the nested control independently refuses the runtime.
    source["scientific_admission"] = False
    source["bl_p1"] = {
        **source.get("bl_p1", {}),
        "control": source_control_value,
        "blockers": [REVIEW_HOLD],
    }
    source["source_paths"] = sorted(
        set(source.get("source_paths", [])) | set(_bounded_source_paths(source))
    )
    source.pop("canary", None)
    export_json(root / "source-template.json", source)
    export_json(root / "source-upload-preflight.json", qualify(source, repo=ROOT))

    canary = probe_manifest(
        source,
        package="bl_ppo",
        output=root / "canary-output",
        job_id=f"bl-p1-e8-04-canary-{commit[:8]}",
        env={
            "BL_P1_CANARY_CONFIG": json.dumps(CANARY_CONFIG, sort_keys=True, separators=(",", ":"))
        },
        timeout_minutes=30,
    )
    canary.update(
        required_exports=CANARY_EXPORTS,
        device="cuda:0",
        lease_seconds=180,
        claim="TRANSPORT_ONLY_EXCLUDED_FROM_P1_Y3_AND_EVERY_SCIENTIFIC_VERDICT",
        resume=canary_control_value["resume"],
    )
    canary["budget"].update(
        unit_cap=2.0,
        reserve_units=0.4,
        rate_units_per_hour=1.54,
        rate_tolerance_units_per_hour=0.0,
        other_assignments={},
    )
    canary["requirements"]["est_minutes"] = 15
    canary["bl_p1"] = {**canary.get("bl_p1", {}), "control": canary_control_value}
    canary["scientific_admission"] = False
    canary = validate(canary, ROOT)
    export_json(root / "canary-manifest.json", canary)
    export_json(root / "canary-upload-preflight.json", qualify(canary, repo=ROOT))
    if source["inputs"] != canary["inputs"]:
        raise ValueError("family canary and source must upload exactly the same input inventory")
    result = {
        "schema": "bl-p1-e8-04-preparation/v1",
        "status": "REVIEW_ONLY_NOT_DISPATCHABLE",
        "commit": commit,
        "qualified_colab_cli_tree": QUALIFIED_CLI_TREE,
        "source_template_sha256": digest(root / "source-template.json"),
        "canary_manifest_sha256": digest(root / "canary-manifest.json"),
        "shared_input_inventory": input_inventory(canary),
        "offsets": {"canary": CANARY_OFFSET, "source": SOURCE_OFFSET},
        "budgets": {
            "rate_units_per_hour": 1.54,
            "rate_tolerance": 0.0,
            "charged_before_round18_units": 7.26,
            "historical_source05_debit_status": "STALE_NOT_ADMISSION_EVIDENCE",
            "canary": {
                "estimate_minutes": 15,
                "timeout_minutes": 30,
                "unit_cap": 2.0,
                "reserve_units": 0.4,
            },
            "source": {
                "estimate_minutes": 75,
                "timeout_minutes": 300,
                "unit_cap": 9.62,
                "reserve_units": 1.92,
            },
        },
        "blockers": [
            REVIEW_HOLD,
            "independent Codex-V receipt records C2 failure",
            "no dispatch or allocation is authorized",
        ],
        "colab_calls": 0,
    }
    export_json(root / "preparation.json", result)
    from research.bl_ppo.exports import export_bytes

    export_bytes(
        root / "DISPATCH.txt",
        (
            "REVIEW_ONLY_NOT_DISPATCHABLE\n" + REVIEW_HOLD + "\n"
            "No dispatch command is emitted; do not invoke tools/ops/dispatch.sh.\n"
        ).encode(),
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--source-audit", type=Path, required=True)
    parser.add_argument("--canary-audit", type=Path, required=True)
    parser.add_argument("--bootstrap-source05", action="store_true")
    parser.add_argument("--amendment", type=Path)
    args = parser.parse_args()
    source_manifest = args.source_manifest
    if args.bootstrap_source05:
        if args.amendment is None:
            parser.error("--bootstrap-source05 requires --amendment")
        source_manifest = bootstrap_source_template(
            source_manifest=args.source_manifest,
            root=args.root / "bootstrap",
            commit=args.commit,
            amendment=args.amendment,
            source_audit=args.source_audit,
        )
    print(
        json.dumps(
            prepare(
                source_manifest=source_manifest,
                root=args.root,
                commit=args.commit,
                source_audit=args.source_audit,
                canary_audit=args.canary_audit,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
