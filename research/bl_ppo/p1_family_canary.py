"""Prepare P1's real-input canary and admit a sibling only from genuine evidence.

No provider is imported or called. Existing source03 artifacts remain immutable.
Y3-05 binds its two-unit canary to the shared owner-authorized envelope;
preparation does not dispatch or establish a live scientific result.
"""

from __future__ import annotations

import argparse
import copy
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.p1_inputs import unpack_bundle

TINY = {"pause_at": 32, "processes": 1, "rollout_steps": 16, "worlds": 2}
ROOT = Path(__file__).resolve().parents[2]
EXPORTS = ["receipt.json"]
CANARY_EXPORTS = EXPORTS + ["canary/pause.json", "canary/pause-snapshot.pt"]


def acceptance_status(path: Path) -> dict[str, Any]:
    """Retain the supplied PASS while checking its actual current-source scope."""
    from research.colab_cli.admission import require_final_only
    from research.colab_cli.supervisor import Refused

    value = json.loads(path.read_text())
    try:
        require_final_only(value, ROOT)
        status, reason = "DISCHARGED", None
    except Refused as exc:
        status, reason = "HOLD", str(exc)
    return {
        "path": str(path.resolve()),
        "sha256": digest(path),
        "reported_status": value.get("status"),
        "accounting_item": status,
        "reason": reason,
    }


def prepare(source: Path, output: Path, commit: str, acceptance: Path) -> dict[str, Any]:
    """Rebind control at a published pin and share identical input bytes in both jobs."""
    from research.bl_ppo.manifests import _bounded_source_paths, input_row
    from research.bl_ppo.p1_manifests import bundle_inputs
    from research.bl_upload.qualify import qualify
    from research.colab_cli.canary import input_inventory
    from research.colab_cli.canary import prepare as probe_manifest
    from research.colab_cli.supervisor import validate

    original = json.loads(source.read_text())
    if (
        original["script"] != "research/bl_ppo/p1_colab.sh"
        or original["env"].get("BL_P1_MODE") != "y3-source"
    ):
        raise ValueError("requires the P1 Y3 source entrypoint")
    subprocess.run(["git", "cat-file", "-e", f"{commit}^{{commit}}"], cwd=ROOT, check=True)
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    bundled = next(row for row in original["inputs"] if row["dest"] == "p1-inputs.tar.gz")
    if digest(Path(bundled["path"])) != bundled["sha256"]:
        raise ValueError("source03 input bytes changed")
    unpacked = output / "inputs"
    unpack_bundle(Path(bundled["path"]), unpacked)
    control_path = unpacked / "p1-control.json"
    verify_export(control_path)
    control = json.loads(control_path.read_text())
    control["source_commit"] = commit
    from research.bl_ppo.export_budget import freeze_control_budget, validate_manifest_budget

    byte_budget = freeze_control_budget(
        control,
        timeout_minutes=original["timeout_minutes"],
        max_export_bytes=original["max_export_bytes"],
    )
    # New control bytes get new sidecars in a sibling; nothing frozen is edited.
    new_control = output / "p1-control.json"
    export_json(new_control, control)
    rows = [row for row in original["inputs"] if row["dest"] != "p1-inputs.tar.gz"]
    for path in sorted(unpacked.iterdir()):
        if path.name.startswith("p1-control.json"):
            continue
        rows.append(input_row(path, path.name))
    for suffix in ("", ".sha256", ".complete"):
        rows.append(input_row(Path(str(new_control) + suffix), "p1-control.json" + suffix))
    template = copy.deepcopy(original)
    attempt_tag = "source05" if control.get("y3_05_envelope") else "source03"
    template.update(
        commit=commit,
        job_id=f"bl-p1-y3-source-admitted-{commit[:8]}",
        output_dir=str(output / "source-output"),
        required_exports=EXPORTS,
        inputs=bundle_inputs(output, rows),
        max_total_export_bytes=byte_budget["max_total_export_bytes"],
    )
    template["env"]["BL_PPO_COMMIT"] = commit
    template["bl_p1"]["control"] = control
    # Source closure includes the new excluded runtime and current broker modules.
    template["source_paths"] = sorted(
        set(template["source_paths"])
        | set(
            _bounded_source_paths(
                {"commit": commit, "source_paths": ["research/bl_ppo", "research/colab_cli"]}
            )
        )
    )
    template.pop("canary", None)
    validate_manifest_budget(template)
    export_json(output / "source-template.json", template)
    canary = probe_manifest(
        template,
        package="bl_ppo",
        output=output / "canary-output",
        job_id=f"bl-p1-{attempt_tag}-canary-{commit[:8]}",
        env={"BL_P1_CANARY_CONFIG": json.dumps(TINY, sort_keys=True, separators=(",", ":"))},
        timeout_minutes=30,
    )
    canary["claim"] = "TRANSPORT_ONLY_EXCLUDED_FROM_P1_Y3_AND_EVERY_SCIENTIFIC_VERDICT"
    canary["required_exports"] = CANARY_EXPORTS
    canary["lease_seconds"] = 180
    # The prospective Y3-05 canary is a separate operational unit.  Its
    # 15-minute forecast is unchanged; the owner rule doubles cap/reserve
    # while the qualified transport still imposes its 30-minute hard maximum.
    canary["budget"].update(unit_cap=2.0, reserve_units=0.4)
    canary["requirements"]["est_minutes"] = 15
    canary["device"] = "cuda:0"
    canary["resume"] = copy.deepcopy(template.get("resume", {}))
    if 30 * canary["budget"]["rate_units_per_hour"] / 60 + 0.4 > 2:
        raise ValueError("canary worst-case liability and reserve exceed two units")
    validated = validate(canary, ROOT)
    export_json(output / "canary-manifest.json", validated)
    upload = qualify(validated, repo=ROOT)
    export_json(output / "upload-preflight.json", upload)
    status = {
        "schema": "bl-p1-family-canary-preparation/v1",
        "status": "VALIDATED_NOT_DISPATCHED",
        "commit": commit,
        "original_manifest": {"path": str(source.resolve()), "sha256": digest(source)},
        "canary_manifest_sha256": digest(output / "canary-manifest.json"),
        "source_template_sha256": digest(output / "source-template.json"),
        "inputs": input_inventory(validated),
        "required_exports": CANARY_EXPORTS,
        "acceptance": acceptance_status(acceptance),
        "budget": {
            "base_cap": 1.0,
            "base_reserve": 0.2,
            "canary_cap": 2.0,
            "reserve": 0.4,
            "estimate_minutes": 15,
            "base_timeout_minutes": 15,
            "effective_timeout_minutes": 30,
            "separate_funding_required": not bool(control.get("y3_05_envelope")),
            "funding_scope": (
                "within Y3-05 25.5u total"
                if control.get("y3_05_envelope")
                else "operator authorization"
            ),
        },
        "blockers": [
            "current-source final-only transport qualification",
            "independent review of the additive canary runtime and rebound source inventory",
            "separate <=2u canary funding within the frozen Y3-05 envelope",
            "fresh empty account inventory before each dispatch",
            "genuine source-bound canary receipt before source admission",
        ],
        "colab_calls": 0,
    }
    export_json(output / "preparation.json", status)
    return status


def preflight(root: Path, acceptance: Path, review: Path) -> dict[str, Any]:
    """Fence canary spending on source qualification, review and separate funding."""
    from research.colab_cli.admission import require_final_only

    prepared = json.loads((root / "preparation.json").read_text())
    if digest(root / "source-template.json") != prepared["source_template_sha256"]:
        raise ValueError("source template differs from the frozen preparation")
    if digest(root / "canary-manifest.json") != prepared["canary_manifest_sha256"]:
        raise ValueError("canary manifest differs from the frozen preparation")
    require_final_only(json.loads(acceptance.read_text()), ROOT)
    verify_export(review)
    audit = json.loads(review.read_text())
    from research.bl_ppo.p1_manifests import _source_identity

    template = json.loads((root / "source-template.json").read_text())
    from research.bl_ppo.export_budget import validate_manifest_budget

    validate_manifest_budget(template)
    identity = _source_identity(template["commit"])
    from research.colab_cli.canary import executable_identity

    for name in ("p1_canary_runtime.py", "p1_family_canary.py"):
        path = "research/bl_ppo/" + name
        body = subprocess.check_output(["git", "show", f"{template['commit']}:{path}"], cwd=ROOT)
        import hashlib

        identity[path] = hashlib.sha256(body).hexdigest()
        if digest(ROOT / path) != identity[path]:
            raise ValueError("new canary source differs from executable pin")
    if (
        audit.get("status") != "PASS"
        or audit.get("unresolved_findings") != []
        or audit.get("source_sha256") != identity
        or audit.get("transport_identity") != executable_identity(ROOT)
        or audit.get("source_template_sha256") != prepared["source_template_sha256"]
        or audit.get("separate_canary_funding_units") != prepared["budget"]["canary_cap"]
    ):
        raise ValueError("independent source review/funding evidence is incomplete or differs")
    return template


def admit(root: Path, receipt: Path, acceptance: Path, review: Path) -> Path:
    """Create the new source manifest only after every local evidence gate passes."""
    from research.colab_cli.supervisor import validate

    template = preflight(root, acceptance, review)
    if receipt.resolve() != (root / "canary-output/receipt.json").resolve():
        raise ValueError("requires this preparation's real broker terminal receipt")
    template["canary"] = {"receipt": {"path": str(receipt.resolve()), "sha256": digest(receipt)}}
    validated = validate(template, ROOT)
    if template.get("bl_p1", {}).get("control", {}).get("y3_05_envelope"):
        import math

        from research.bl_ppo.p1_admission import reconcile_y3_05
        from tools.ops.spend import verify_receipts

        debit_path = root.parent / "y3-debit.json"
        verify_export(debit_path)
        frozen_debit = json.loads(debit_path.read_text())
        actual = reconcile_y3_05([Path(row["path"]) for row in frozen_debit["attempts"]])
        if actual != frozen_debit:
            raise ValueError("prior Y3 debit changed before source admission")
        charged = verify_receipts([receipt])[0]
        terminal = json.loads(receipt.read_text())
        before, after = terminal.get("usage_before", {}).get("balance"), terminal.get(
            "usage_after", {}
        ).get("balance")
        if any(
            type(value) not in (int, float) or not math.isfinite(value) for value in (before, after)
        ):
            raise ValueError("canary debit lacks finite provider balance evidence")
        canary_charge = math.ceil((max(charged.units, before - after, 0) - 1e-9) * 100) / 100
        liability = actual["legacy_conservative_spend_units"] + canary_charge + 9.62 + 3.86
        if liability > 25.5 + 1e-9:
            raise ValueError("actual canary debit leaves insufficient shared Y3 envelope")
        export_json(
            root / "source-admission-debit.json",
            {
                "status": "RECONCILED",
                "canary_receipt_sha256": digest(receipt),
                "prior_debit_sha256": digest(debit_path),
                "canary_own_units": charged.units,
                "canary_charged_units": canary_charge,
                "reserved_total_units": liability,
                "envelope_units": 25.5,
                "replacement_reserved_units": 3.86,
            },
        )
    target = root / "source-admitted.json"
    export_json(target, validated)
    export_json(
        root / "admission.json",
        {
            "status": "VALIDATED_NOT_DISPATCHED",
            "manifest_sha256": digest(target),
            "canary_receipt_sha256": digest(receipt),
            "acceptance_sha256": digest(acceptance),
            "independent_review_sha256": digest(review),
            "remaining_blocker": "fresh empty inventory and calibrated rate <=1.54u/h",
        },
    )
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--root", type=Path, required=True)
    build.add_argument("--commit", required=True)
    build.add_argument("--acceptance", type=Path, required=True)
    bind = commands.add_parser("admit")
    bind.add_argument("--root", type=Path, required=True)
    bind.add_argument("--receipt", type=Path, required=True)
    bind.add_argument("--acceptance", type=Path, required=True)
    bind.add_argument("--review", type=Path, required=True)
    check = commands.add_parser("preflight")
    check.add_argument("--root", type=Path, required=True)
    check.add_argument("--acceptance", type=Path, required=True)
    check.add_argument("--review", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        print(json.dumps(prepare(args.source, args.root, args.commit, args.acceptance)))
    elif args.command == "admit":
        print(admit(args.root, args.receipt, args.acceptance, args.review))
    else:
        preflight(args.root, args.acceptance, args.review)
        print("PREFLIGHT_PASS_NOT_DISPATCHED; fresh empty inventory remains required")


if __name__ == "__main__":
    main()
