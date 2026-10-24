"""Create-only R30 qualified-path packages. Preparation never dispatches."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30.protocol import B8_SHA, CAPS, JUDGMENT_SHA, NAMESPACE, require_series
from research.metadata_archive import bundle_inputs
from tools.ops.qualification import OPS, check

ROOT = Path(__file__).resolve().parents[2]
RESUME_POLICY = {
    "unit_kind": "complete-matched-fit-packet",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": True,
    "max_outcome_blind_retries": 2,
}
CANARY_EXEMPTION = {
    "kind": "short-canary",
    "reason": "Bounded consumer proof; canary units cannot enter scientific resume ledgers",
}
SOURCE_PATHS = [
    "src",
    "research/bl_round30",
    "research/bl_residual",
    "research/a100_a2/__init__.py",
    "research/a100_a2/model.py",
    "research/bl_ppo/__init__.py",
    "research/bl_ppo/exports.py",
    "research/colab_cli",
    "research/metadata_archive.py",
    "research/t34_exit_b8/PUBLIC_INPUTS.json",
]
MAX_LOCAL_INPUT_BYTES = 1_000_000


def transport_inputs(destination: Path, archive: Path) -> list[dict]:
    """Use registered HTTPS assets for large files and one verified metadata tar.

    Preserve destinations and every original sidecar. Private/unknown large
    inputs refuse preparation; this helper never publishes scientific inputs.
    The unchanged supervisor verifies URL sizes/hashes on Mac and runtime.
    """
    catalog = json.loads((ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_text())
    assets = {(row["sha256"], row["size_bytes"]): row for row in catalog["assets"]}
    inventory = []
    for path in sorted(destination.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe R30 input type")
        checksum, size = digest(path), path.stat().st_size
        row = {"dest": path.name, "sha256": checksum}
        if size > MAX_LOCAL_INPUT_BYTES:
            asset = assets.get((checksum, size))
            if asset is None:
                raise ValueError(f"input over 1 MB needs a registered public URL: {path.name}")
            row.update(url=asset["url"], size_bytes=size)
        else:
            row["path"] = str(path.resolve())
        inventory.append(row)
    result = bundle_inputs(inventory, archive)
    if any(
        "path" in row and Path(row["path"]).stat().st_size > MAX_LOCAL_INPUT_BYTES for row in result
    ):
        raise ValueError("R30 metadata archive exceeds 1 MB local upload bound")
    return result


def make_pack(
    output: Path,
    commit: str,
    b8: Path,
    *,
    stage: str = "r30-1",
    inputs: Path | None = None,
    series: Path | None = None,
    canary_receipt: Path | None = None,
    preflight_receipt: Path | None = None,
) -> dict:
    """Bind the published pin, allowed input destinations and immutable envelope.

    Without admitted inputs produce a synthetic consumer canary and a deliberately
    blocked main template. A canary does not manufacture scientific admission.
    """
    commit = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True, timeout=10
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", commit, "origin/main"],
        cwd=ROOT,
        check=True,
        timeout=10,
    )
    check(ROOT, OPS / "colab-cli-qualified.json", commit)
    check(ROOT, OPS / "colab-cli-qualified.json")
    changed = subprocess.check_output(
        ["git", "diff", commit, "--", *SOURCE_PATHS], cwd=ROOT, timeout=10
    )
    if changed:
        raise ValueError("package source differs from published pin")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS],
        cwd=ROOT,
        timeout=10,
    )
    if untracked:
        raise ValueError("untracked runtime source cannot enter a pinned package")
    if digest(b8) != B8_SHA:
        raise ValueError("wrong B8 initializer")
    if output.exists():
        raise FileExistsError(output)
    preflight_passed = False
    if preflight_receipt is not None:
        from research.bl_round30.local_check import source_identity

        verify_export(preflight_receipt)
        preflight = json.loads(preflight_receipt.read_text())
        if inputs is None and preflight.get("schema") == "r30-packaging-preflight/v1":
            from research.bl_round30.packaging_check import check_receipt

            check_receipt(preflight, commit)
        elif (
            preflight.get("schema") != "r30-mac-preflight/v1"
            or preflight.get("status") != "PASS"
            or preflight.get("source_commit") != commit
            or preflight.get("source_identity") != source_identity()
        ):
            raise ValueError("current-source Mac preflight has not passed")
        preflight_passed = True
    synthetic = inputs is None
    if not synthetic:
        from research.bl_round30.job import preflight_inputs

        control = preflight_inputs(inputs, mode=stage, commit=commit)
        if series is None:
            raise ValueError("main needs cumulative accounting")
        accounting = json.loads(series.read_text())
        if (
            accounting.get("canary_allowance_hours") != 1 / 3
            or accounting.get("canary_allowance_units") != 2 / 3
        ):
            raise ValueError("freeze the exact-input canary liability before preparation")
        remaining = require_series(accounting, stage)
        cap_hours = remaining["remaining_hours"] - accounting["canary_allowance_hours"]
        cap_units = remaining["remaining_units"] - accounting["canary_allowance_units"]
        if cap_hours <= 0 or cap_units <= 0:
            raise ValueError("INCOMPLETE: no budget after canary liability")
    else:
        control = {
            "schema": "r30-control/v1",
            "synthetic": True,
            "stage": stage,
            "namespace": NAMESPACE + "-synthetic",
            "commit": commit,
            "ruling_sha256": JUDGMENT_SHA,
            "b8_sha256": B8_SHA,
        }
        # Canary ceiling is a subset of the stage envelope, doubled once.
        cap_hours, cap_units = 20 / 60, 2 / 3
    output.mkdir(parents=True)
    destination = output / "inputs"
    if synthetic:
        destination.mkdir()
        shutil.copyfile(b8, destination / "b8.pth")
        export_json(destination / "control.json", control)
        export_json(
            destination / "verified-ledger.json",
            {
                "schema": "r30-job-ledger/v1",
                "namespace": control["namespace"],
                "allocation": digest(destination / "control.json"),
                "units": [],
            },
        )
    else:
        destination.mkdir()
        names = {
            "control.json",
            "inventory.json",
            "approval.json",
            "verified-ledger.json",
            *control["input_hashes"],
        }
        ledger = json.loads((inputs / "verified-ledger.json").read_text())
        if ledger.get("units"):
            if len(ledger["units"]) != 1:
                raise ValueError("duplicate completed units")
            previous = inputs / "unit-0000.pth"
            saved = ledger["units"][0]
            if (
                saved.get("name") != previous.name
                or verify_export(previous) != saved.get("sha256")
                or previous.stat().st_size != saved.get("size")
            ):
                raise ValueError("unverified resumed payload")
            names.add(previous.name)
            names.add("allocation.json")
        if stage == "r30-2":
            names.add("successor-gate.json")
        for name in names:
            for suffix in ("", ".sha256", ".complete"):
                source = inputs / (name + suffix)
                if source.is_file():
                    shutil.copyfile(source, destination / source.name)
        shutil.copyfile(series, destination / "series.json")
    inventory = transport_inputs(destination, output / "metadata.tar")
    metadata = next(row for row in inventory if row["dest"] == "metadata.tar")
    manifest = {
        "job_id": f"r30-{stage}-{'canary' if synthetic else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": "research/bl_round30/colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 1,
            "est_minutes": 5 if synthetic else (60 if stage == "r30-1" else 120),
        },
        "timeout_minutes": cap_hours * 60,
        "output_dir": str((output / "runtime-output").resolve()),
        "inputs": inventory,
        "env": {
            "R30_DEVICE": "cuda",
            "R30_MODE": "canary" if synthetic else stage,
            "COLAB_CLI_REQUIREMENTS": "research/bl_round30/requirements.txt",
            "METADATA_ARCHIVE_SHA256": metadata["sha256"],
        },
        "required_exports": ["receipt.json", "allocation.json", "unit-0000.pth"],
        "budget": {
            "unit_cap": cap_units,
            "rate_units_per_hour": 1.54,
            "reserve_units": cap_units / 6,
            "other_assignments": {},
        },
        "resume": dict(RESUME_POLICY),
        "max_export_bytes": 128 * 1024**2,
        "r30": {
            "ruling_sha256": JUDGMENT_SHA,
            "stage": stage,
            "caps": CAPS[stage],
            "combined_effective_hours": 7.2,
            "combined_effective_units": 14.4,
            "canary_base_minutes": 10,
            "canary_effective_minutes": 20,
            "synthetic": synthetic,
            "main_admission": "HOLD" if synthetic else "ADMITTED",
        },
    }
    if preflight_passed:
        manifest["r30"]["mac_preflight"] = {
            "path": str(preflight_receipt.resolve()),
            "sha256": digest(preflight_receipt),
            "source_commit": commit,
        }
    if synthetic:
        manifest.pop("resume")
        manifest["resume_exemption"] = dict(CANARY_EXEMPTION)
        manifest["canary"] = {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": manifest["script"],
        }
        manifest["env"].update(
            SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/bl_round30"
        )
    elif canary_receipt is not None:
        manifest["canary"] = {
            "receipt": {"path": str(canary_receipt.resolve()), "sha256": digest(canary_receipt)}
        }
    else:
        manifest["scientific_admission"] = False
        manifest["blocked_reason"] = "HOLD: exact-input current-pin native canary pending"
    if not synthetic:
        # Exact inventory (including unchanged accounting bytes) is necessary to
        # qualify a main job. The synthetic-only rehearsal above cannot do so.
        native_canary = {
            **manifest,
            "job_id": manifest["job_id"].replace("main", "canary"),
            "timeout_minutes": 20,
            "output_dir": str((output / "native-canary-output").resolve()),
            "canary": {
                "schema": "colab-family-canary/v1",
                "mode": "probe",
                "family": manifest["script"],
            },
            "budget": {**manifest["budget"], "unit_cap": 2 / 3, "reserve_units": 1 / 9},
            "env": {
                **manifest["env"],
                "R30_MODE": "canary",
                "SNAKE_COLAB_CANARY": "1",
                "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_round30",
            },
        }
        native_canary.pop("resume")
        native_canary["resume_exemption"] = dict(CANARY_EXEMPTION)
        native_canary.pop("scientific_admission", None)
        native_canary.pop("blocked_reason", None)
        native_canary["requirements"] = {**manifest["requirements"], "est_minutes": 5}
        if not preflight_passed:
            # Preserve native-canary HOLD too; a transport proof is not a Mac preflight.
            native_canary["scientific_admission"] = False
            native_canary["blocked_reason"] = "HOLD: current-source Mac preflight pending"
        export_json(output / "canary-job.json", native_canary)
    if not preflight_passed:
        manifest["scientific_admission"] = False
        manifest["blocked_reason"] = "HOLD: current-source Mac preflight pending"
    export_json(output / "job.json", manifest)
    if synthetic:
        template = {
            **manifest,
            "scientific_admission": False,
            "blocked_reason": "HOLD: Claude corpus reuse/account reservation and Codex-V admission",
        }
        template.pop("resume_exemption")
        template["resume"] = dict(RESUME_POLICY)
        template["timeout_minutes"] = CAPS[stage]["hours"] * 60
        template["budget"] = {
            **manifest["budget"],
            "unit_cap": CAPS[stage]["units"],
            "reserve_units": CAPS[stage]["units"] / 6,
        }
        export_json(output / "main-template.json", template)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(CAPS), default="r30-1")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--series", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    args = parser.parse_args()
    value = make_pack(
        args.output.resolve(),
        args.commit,
        args.b8,
        stage=args.stage,
        inputs=args.inputs,
        series=args.series,
        canary_receipt=args.canary_receipt,
        preflight_receipt=args.preflight_receipt,
    )
    print(json.dumps({"manifest": str(args.output / "job.json"), "job_id": value["job_id"]}))


if __name__ == "__main__":
    main()
