"""Create-only R30 qualified-path packages. Preparation never dispatches."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import subprocess
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_round30.private_bundle import MAX_BUNDLE_BYTES, bundle
from research.bl_round30.protocol import (
    AMENDMENT_SHA,
    B8_SHA,
    JUDGMENT_SHA,
    NAMESPACE,
    require_series,
)
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
# Standard R30 bars.  No D1024 transfer exception belongs to this family.
TRANSFER_BAR_SECONDS = 150.0
STAGING_BAR_SECONDS = 300.0
# Conservative metadata-only L4 transport model. The 47.9842 s transfer /
# 57.4917 s staging values are a recent qualified sparse requalification. The
# byte slope comes from the separate 45,397,643-byte D1024 canary upload: it
# supplies only an observed transport rate; its 300 s exception is never
# attached to R30. The exact-input native canary remains the required proof.
FIXED_TRANSFER_SECONDS = 47.98416666581761
FIXED_STAGING_SECONDS = 57.491733915987425
LARGE_PRIVATE_BYTES = 45_397_643
LARGE_PRIVATE_TRANSFER_SECONDS = 136.40460358303972
LARGE_PRIVATE_BYTE_SECONDS = (
    LARGE_PRIVATE_TRANSFER_SECONDS - FIXED_TRANSFER_SECONDS
) / LARGE_PRIVATE_BYTES
HISTORICAL_TRANSFER_UPPER_SECONDS = 136.40460358303972
HISTORICAL_STAGING_UPPER_SECONDS = 183.58192350005265
STAGE_WALL_CAP_MINUTES = {"r30-1": 144.0, "r30-2": 288.0}
COMBINED_WALL_CAP_MINUTES = 432.0
CANARY_WALL_MINUTES = 20.0
TECHNICAL_RATE_RESERVE = 0.20


def transport_inputs(destination: Path, archive: Path) -> list[dict]:
    """Keep B8 public and ship every other admitted byte in a private bundle."""
    catalog = json.loads((ROOT / "research/t34_exit_b8/PUBLIC_INPUTS.json").read_text())
    assets = {(row["sha256"], row["size_bytes"]): row for row in catalog["assets"]}
    public: dict | None = None
    for path in sorted(destination.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe R30 input type")
        checksum, size = digest(path), path.stat().st_size
        if path.name == "b8.pth":
            asset = assets.get((checksum, size))
            if asset is None:
                raise ValueError("B8 must use the registered public URL")
            public = {"dest": "b8.pth", "url": asset["url"], "size_bytes": size, "sha256": checksum}
    if public is None:
        raise ValueError("private R30 packet requires the public B8 initializer")
    packed = bundle(destination, archive, exclude={"b8.pth"})
    projected = FIXED_TRANSFER_SECONDS + packed["size_bytes"] * LARGE_PRIVATE_BYTE_SECONDS
    staging = FIXED_STAGING_SECONDS + projected
    if (
        packed["size_bytes"] > MAX_BUNDLE_BYTES
        or projected >= TRANSFER_BAR_SECONDS
        or staging >= STAGING_BAR_SECONDS
    ):
        raise ValueError("private R30 bundle cannot meet the standard 150-second transfer bar")
    return [
        public,
        {
            "dest": "private-inputs.tar.gz",
            "path": packed["path"],
            "sha256": packed["sha256"],
            "size_bytes": packed["size_bytes"],
            "r30_private_bundle": {
                "schema": packed["schema"],
                "unpacked_bytes": packed["unpacked_bytes"],
                "files": packed["files"],
                "forecast": {
                    "fixed_transfer_seconds": FIXED_TRANSFER_SECONDS,
                    "fixed_staging_seconds": FIXED_STAGING_SECONDS,
                    "large_private_bytes": LARGE_PRIVATE_BYTES,
                    "large_private_transfer_seconds": LARGE_PRIVATE_TRANSFER_SECONDS,
                    "historical_transfer_upper_seconds": HISTORICAL_TRANSFER_UPPER_SECONDS,
                    "historical_staging_upper_seconds": HISTORICAL_STAGING_UPPER_SECONDS,
                    "projected_transfer_seconds": projected,
                    "projected_staging_seconds": staging,
                    "transfer_bar_seconds": TRANSFER_BAR_SECONDS,
                    "staging_bar_seconds": STAGING_BAR_SECONDS,
                },
            },
        },
    ]


def _positive_number(value: object, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"finite nonnegative {name} required")
    return float(value)


def _main_operational_guard(
    accounting: dict, stage: str, remaining_before_canary: object, measured_estimate: object
) -> tuple[float, float, float]:
    """Derive the only unit guard from remaining wall time and an observed rate."""
    rate = _positive_number(accounting.get("verified_rate_units_per_hour"), "verified L4 rate")
    if rate <= 0:
        raise ValueError("verified positive L4 rate required for operational guard")
    remaining = _positive_number(remaining_before_canary, "remaining series minutes")
    measured_estimate = _positive_number(measured_estimate, "measured runtime estimate")
    remaining -= CANARY_WALL_MINUTES
    if remaining <= 0:
        raise ValueError("INCOMPLETE: no wall allowance remains after exact-input canary liability")
    if remaining < 2 * measured_estimate:
        raise ValueError(
            "INCOMPLETE: remaining wall allowance is below twice the measured estimate"
        )
    work_units = remaining / 60 * rate
    return remaining, rate, work_units


def _runtime_forecast(
    path: Path | None, *, commit: str, canary_receipt: Path | None, input_control: Path
) -> tuple[float, str]:
    """Bind a Mac-only post-canary estimate without mutating uploaded series bytes."""
    if path is None or canary_receipt is None:
        raise ValueError("main requires a Mac-only runtime forecast and exact-input canary receipt")
    if (
        path.is_symlink()
        or not path.is_file()
        or canary_receipt.is_symlink()
        or not canary_receipt.is_file()
    ):
        raise ValueError("runtime forecast/canary receipt must be regular files")
    value = json.loads(path.read_text())
    if (
        not isinstance(value, dict)
        or value.get("schema") != "r30-runtime-forecast/v1"
        or value.get("source_commit") != commit
        or value.get("canary_receipt_sha256") != digest(canary_receipt)
        or value.get("input_control_sha256") != digest(input_control)
    ):
        raise ValueError("runtime forecast source/canary/control binding differs")
    estimate = _positive_number(value.get("measured_est_minutes"), "measured runtime estimate")
    if estimate <= 0:
        raise ValueError("positive measured runtime estimate required")
    return estimate, digest(path)


def _canary_operational_guard(accounting: dict, stage: str) -> tuple[float, float, float]:
    """Reserve the exact-input canary from verified wall/rate evidence only."""
    rate = _positive_number(accounting.get("verified_rate_units_per_hour"), "verified L4 rate")
    if rate <= 0:
        raise ValueError("verified positive L4 rate required for operational guard")
    remaining = _positive_number(
        require_series(accounting, stage)["remaining_minutes"], "remaining series minutes"
    )
    if remaining < CANARY_WALL_MINUTES:
        raise ValueError("INCOMPLETE: no wall allowance remains for exact-input canary")
    canary_units = CANARY_WALL_MINUTES / 60 * rate
    return remaining - CANARY_WALL_MINUTES, rate, canary_units


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
    canary_only: bool = False,
    runtime_forecast: Path | None = None,
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
    if canary_only and synthetic:
        raise ValueError("exact-input canary requires admitted real inputs")
    if not synthetic:
        from research.bl_round30.job import preflight_inputs

        control = preflight_inputs(inputs, mode=stage, commit=commit)
        if series is None:
            raise ValueError("main needs cumulative accounting")
        accounting = json.loads(series.read_text())
        if canary_only:
            remaining_after_canary, rate, cap_units = _canary_operational_guard(accounting, stage)
            cap_minutes, cap_hours, estimate_minutes = (
                CANARY_WALL_MINUTES,
                CANARY_WALL_MINUTES / 60,
                5,
            )
        else:
            measured_estimate, runtime_forecast_sha256 = _runtime_forecast(
                runtime_forecast,
                commit=commit,
                canary_receipt=canary_receipt,
                input_control=inputs / "control.json",
            )
            accounting_remaining = require_series(accounting, stage)["remaining_minutes"]
            cap_minutes, rate, cap_units = _main_operational_guard(
                accounting, stage, accounting_remaining, measured_estimate
            )
            cap_hours = cap_minutes / 60
            estimate_minutes = int(math.ceil(measured_estimate))
    else:
        control = {
            "schema": "r30-control/v1",
            "synthetic": True,
            "stage": stage,
            "namespace": NAMESPACE + "-synthetic",
            "commit": commit,
            "ruling_sha256": JUDGMENT_SHA,
            "amendment_sha256": AMENDMENT_SHA,
            "b8_sha256": B8_SHA,
        }
        # Canary ceiling is a subset of the stage envelope, doubled once.
        cap_hours, cap_units = 20 / 60, 2 / 3
        estimate_minutes = 5
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
    inventory = transport_inputs(destination, output / "private-inputs.tar.gz")
    private_bundle = next(row for row in inventory if row["dest"] == "private-inputs.tar.gz")
    manifest = {
        "job_id": f"r30-{stage}-{'canary' if synthetic or canary_only else 'main'}-{commit[:8]}",
        "commit": commit,
        "script": "research/bl_round30/colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 1,
            "est_minutes": estimate_minutes,
        },
        "timeout_minutes": cap_hours * 60,
        "output_dir": str((output / "runtime-output").resolve()),
        "inputs": inventory,
        "env": {
            "R30_DEVICE": "cuda",
            "R30_MODE": "canary" if synthetic or canary_only else stage,
            "COLAB_CLI_REQUIREMENTS": "research/bl_round30/requirements.txt",
            "R30_PRIVATE_BUNDLE_SHA256": private_bundle["sha256"],
        },
        "required_exports": ["receipt.json", "allocation.json", "unit-0000.pth"],
        "budget": {
            "unit_cap": cap_units,
            "rate_units_per_hour": 1.54 if synthetic else rate,
            "reserve_units": cap_units * TECHNICAL_RATE_RESERVE,
            "other_assignments": {},
        },
        "resume": dict(RESUME_POLICY),
        "max_export_bytes": 128 * 1024**2,
        "r30": {
            "ruling_sha256": JUDGMENT_SHA,
            "amendment_sha256": AMENDMENT_SHA,
            "stage": stage,
            "operational_wall_cap_minutes": cap_hours * 60,
            "combined_wall_cap_minutes": COMBINED_WALL_CAP_MINUTES,
            "technical_rate_reserve": TECHNICAL_RATE_RESERVE,
            "transport": private_bundle["r30_private_bundle"],
            "canary_base_minutes": 10,
            "canary_effective_minutes": 20,
            "synthetic": synthetic,
            "main_admission": "HOLD" if synthetic or canary_only else "ADMITTED",
        },
    }
    if not synthetic and not canary_only:
        manifest["r30"]["runtime_forecast_sha256"] = runtime_forecast_sha256
    if preflight_passed:
        manifest["r30"]["mac_preflight"] = {
            "path": str(preflight_receipt.resolve()),
            "sha256": digest(preflight_receipt),
            "source_commit": commit,
        }
    if synthetic or canary_only:
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
    if not synthetic and not canary_only:
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
    if synthetic or canary_only:
        template = {
            **manifest,
            "scientific_admission": False,
            "blocked_reason": (
                "HOLD: measured runtime forecast and exact-input native canary pending"
                if canary_only
                else "HOLD: Claude corpus reuse/account reservation and Codex-V admission"
            ),
        }
        template.pop("resume_exemption")
        template.pop("canary")
        template["resume"] = dict(RESUME_POLICY)
        template["job_id"] = template["job_id"].replace("canary", "main")
        template["env"] = {
            key: value
            for key, value in {**template["env"], "R30_MODE": stage}.items()
            if key not in {"SNAKE_COLAB_CANARY", "SNAKE_COLAB_CANARY_PACKAGE"}
        }
        if canary_only:
            template["timeout_minutes"] = remaining_after_canary
            template["requirements"] = {
                key: value
                for key, value in template["requirements"].items()
                if key != "est_minutes"
            }
            template["budget"] = {
                **template["budget"],
                "unit_cap": remaining_after_canary / 60 * rate,
                "reserve_units": remaining_after_canary / 60 * rate * TECHNICAL_RATE_RESERVE,
            }
        else:
            template["timeout_minutes"] = STAGE_WALL_CAP_MINUTES[stage] - CANARY_WALL_MINUTES
        template["budget"] = {
            **template["budget"],
            **(
                {}
                if canary_only
                else {
                    "unit_cap": template["timeout_minutes"] / 60 * 1.54,
                    "reserve_units": template["timeout_minutes"]
                    / 60
                    * 1.54
                    * TECHNICAL_RATE_RESERVE,
                }
            ),
        }
        export_json(output / "main-template.json", template)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--stage", choices=tuple(STAGE_WALL_CAP_MINUTES), default="r30-1")
    parser.add_argument("--inputs", type=Path)
    parser.add_argument("--series", type=Path)
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--preflight-receipt", type=Path)
    parser.add_argument("--canary-only", action="store_true")
    parser.add_argument("--runtime-forecast", type=Path)
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
        canary_only=args.canary_only,
        runtime_forecast=args.runtime_forecast,
    )
    print(json.dumps({"manifest": str(args.output / "job.json"), "job_id": value["job_id"]}))


if __name__ == "__main__":
    main()
