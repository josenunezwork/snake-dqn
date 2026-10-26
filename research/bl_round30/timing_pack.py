"""Create the isolated, metadata-only R30-1 L4 timing-calibration packet.

This module never unpacks the frozen private bundle and never dispatches.  It
only freezes the operator-facing manifest after checking that its timing config
is tied to the admitted native-canary control metadata and the current,
published runtime source closure.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

from research.bl_ppo.exports import digest, export_bytes, export_json
from research.bl_round30.pack import SOURCE_PATHS as R30_SOURCE_PATHS
from research.bl_round30.protocol import B8_SHA
from research.bl_round30.timing import (
    AMENDMENT_SHA,
    CERTIFICATE_AMENDMENT_SHA,
    DEADLINES,
    SCHEDULE,
    TIMING_PAYLOAD_BYTES,
)
from research.bl_round30.timing_telemetry import OBSERVABILITY_SHA
from tools.ops.qualification import OPS, check

ROOT = Path(__file__).resolve().parents[2]
RULING = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/"
    "JUDGMENT_R30_G5_TIMING_CALIBRATION_20261025.md"
)
RULING_SHA256 = "9b07b66663d8a7ac168f80dbfc35889b3551df8797f347335c10d83a8d81801f"
FROZEN_NATIVE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-30/"
    "bl-r30-postf4-20261025-v1/native-canary-01"
)
FROZEN_BUNDLE = FROZEN_NATIVE / "private-inputs.tar.gz"
FROZEN_BUNDLE_SHA256 = "af82651cec44d3878b91b7410c3b844bf890d779e0bceb305ac3d93a70ef9455"
FROZEN_BUNDLE_BYTES = 36_189_672
TRANSPORT_BYTES = 67_108_864
UPLOAD_RULING = RULING.with_name("JUDGMENT_R30_TIMING_UPLOAD_CAP_NONATTEMPT_20261025.md")
UPLOAD_RULING_SHA256 = "de404ed2ec595ed60f506ca403e142ae0ee7e90666b2a6c49b64c457c103a51b"
OBSERVABILITY_RULING = RULING.with_name("JUDGMENT_R30_PROBE02_INCOMPLETE_OBSERVABILITY_20261025.md")
TRANSFER_BAR_SECONDS = 150
STAGING_BAR_SECONDS = 300
RATE_UNITS_PER_HOUR = 1.54
AMENDMENT = RULING.with_name("JUDGMENT_R30_TIMING_185D8395_FIX_FIRST_20261025.md")
OLD_RESERVATION = FROZEN_NATIVE.parent / "timing-calibration-01/accounting-reservation.json"
OLD_RESERVATION_SHA256 = "621fc26028b93fc5be0fe933f642ebaac8ef79ab48277129e03091e041aad6c6"
CERTIFICATE_AMENDMENT = RULING.with_name("JUDGMENT_R30_PROBE03_CERTIFICATE_AMENDMENT_20261025.md")
PRIOR_PROBE_RESERVATION = (
    FROZEN_NATIVE.parent / "timing-calibration-probe-03/accounting-reservation.json"
)
PRIOR_PROBE_RESERVATION_SHA256 = "765347652319cac2f377f35691ec2192165a70fd67c6705ec1a1ad30c7b1223a"
BASE_MINUTES = 20
EFFECTIVE_MINUTES = 40
RESERVE_FRACTION = 0.20
SOURCE_PATHS = [*R30_SOURCE_PATHS, "tools/ops/qualification.py", "tools/ops/dispatch.py"]
EXPECTED_SCHEDULE = SCHEDULE.copy()


def identity_paths() -> list[str]:
    """Return the deterministic source closure required by timing reports/configs."""
    fixed = {"research/t34_exit_b8/PUBLIC_INPUTS.json", "tools/ops/qualification.py"}
    for relative in SOURCE_PATHS:
        path = ROOT / relative
        if path.is_file():
            fixed.add(relative)
        elif path.is_dir():
            fixed.update(
                str(candidate.relative_to(ROOT))
                for candidate in path.rglob("*")
                if candidate.is_file()
                and candidate.suffix in {".py", ".sh", ".txt"}
                and "__pycache__" not in candidate.parts
            )
    return sorted(fixed)


def _sha(value: object, name: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(c not in "0123456789abcdef" for c in value)
    ):
        raise ValueError(f"invalid {name} SHA-256")
    return value


def _read_json(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"unsafe or missing {path.name}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise ValueError(f"{path.name} must be a JSON object")
    return value


def load_timing_config(path: Path, *, commit: str, control: dict[str, Any]) -> dict[str, Any]:
    """Refuse anything except the ruling's exact timing-only schedule/bindings."""
    value = _read_json(path)
    if (
        value.get("schema") != "r30-timing-config/v2"
        or value.get("observability_ruling_sha256") != OBSERVABILITY_SHA
        or value.get("ruling_sha256") != RULING_SHA256
        or value.get("amendment_sha256") != AMENDMENT_SHA
        or value.get("certificate_amendment_sha256") != CERTIFICATE_AMENDMENT_SHA
        or value.get("deadlines") != DEADLINES
        or value.get("source_commit") != commit
        or value.get("bundle_sha256") != FROZEN_BUNDLE_SHA256
        or value.get("synthetic") is not False
        or value.get("schedule") != EXPECTED_SCHEDULE
    ):
        raise ValueError("timing config differs from the admitted calibration contract")
    if value.get("control_sha256") != digest(FROZEN_NATIVE / "inputs/control.json"):
        raise ValueError("timing config control binding differs")
    frozen_hashes = control.get("input_hashes")
    if not isinstance(frozen_hashes, dict) or frozen_hashes.get("b8.pth") != B8_SHA:
        raise ValueError("frozen control does not bind the B8 metadata")
    if value.get("input_hashes") != frozen_hashes:
        raise ValueError("timing config input hashes differ from frozen control")
    source_hashes = value.get("source_hashes")
    if not isinstance(source_hashes, dict) or set(source_hashes) != set(identity_paths()):
        raise ValueError("timing config must bind a source hash closure")
    for relative, checksum in source_hashes.items():
        if (
            not isinstance(relative, str)
            or relative.startswith("/")
            or ".." in Path(relative).parts
        ):
            raise ValueError("timing source closure contains an unsafe path")
        if digest(ROOT / relative) != _sha(checksum, f"source {relative}"):
            raise ValueError(f"timing source closure differs: {relative}")
    return value


def _published_clean(commit: str) -> str:
    pin = subprocess.check_output(
        ["git", "rev-parse", f"{commit}^{{commit}}"], cwd=ROOT, text=True, timeout=10
    ).strip()
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", pin, "origin/main"],
        cwd=ROOT,
        check=True,
        timeout=10,
    )
    check(ROOT, OPS / "colab-cli-qualified.json", pin)
    check(ROOT, OPS / "colab-cli-qualified.json")
    if subprocess.run(
        ["git", "diff", "--quiet", pin, "--", *SOURCE_PATHS], cwd=ROOT, timeout=10
    ).returncode:
        raise ValueError("timing runtime source differs from the published reviewed pin")
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "--", *SOURCE_PATHS],
        cwd=ROOT,
        text=True,
        timeout=10,
    )
    if untracked.strip():
        raise ValueError("untracked timing runtime source cannot enter a reviewed packet")
    return pin


def make_timing_pack(output: Path, *, commit: str, timing_config: Path) -> dict[str, Any]:
    """Create one immutable timing-only packet; no provider operation occurs."""
    if output.name != "timing-calibration-probe-04":
        raise ValueError("only timing-calibration-probe-04 is authorized")
    if output.exists():
        raise FileExistsError(output)
    if digest(RULING) != RULING_SHA256 or digest(AMENDMENT) != AMENDMENT_SHA:
        raise ValueError("timing ruling hash differs")
    if digest(OBSERVABILITY_RULING) != OBSERVABILITY_SHA:
        raise ValueError("timing observability ruling hash differs")
    if digest(UPLOAD_RULING) != UPLOAD_RULING_SHA256:
        raise ValueError("timing upload ruling hash differs")
    if (
        digest(FROZEN_BUNDLE) != FROZEN_BUNDLE_SHA256
        or FROZEN_BUNDLE.stat().st_size != FROZEN_BUNDLE_BYTES
    ):
        raise ValueError("frozen private bundle identity differs")
    if digest(OLD_RESERVATION) != OLD_RESERVATION_SHA256:
        raise ValueError("original calibration reservation differs")
    if digest(CERTIFICATE_AMENDMENT) != CERTIFICATE_AMENDMENT_SHA:
        raise ValueError("certificate amendment hash differs")
    if digest(PRIOR_PROBE_RESERVATION) != PRIOR_PROBE_RESERVATION_SHA256:
        raise ValueError("prior probe reservation differs")
    pin = _published_clean(commit)
    control = _read_json(FROZEN_NATIVE / "inputs/control.json")
    load_timing_config(timing_config, commit=pin, control=control)
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    config_copy = export_bytes(inputs / "timing-config.json", timing_config.read_bytes())
    ruling_copy = export_bytes(inputs / "timing-ruling.md", RULING.read_bytes())
    amendment_copy = export_bytes(inputs / "timing-amendment.md", AMENDMENT.read_bytes())
    upload_ruling_copy = export_bytes(
        inputs / "timing-upload-ruling.md", UPLOAD_RULING.read_bytes()
    )
    observability_copy = export_bytes(
        inputs / "timing-observability-ruling.md", OBSERVABILITY_RULING.read_bytes()
    )
    certificate_copy = export_bytes(
        inputs / "timing-certificate-amendment.md", CERTIFICATE_AMENDMENT.read_bytes()
    )
    reservation = export_json(
        output / "accounting-reservation.json",
        {
            "schema": "r30-g5-timing-calibration-accounting/v1",
            "entry": f"r30-1-g5-{output.name}",
            "upload_ruling_sha256": UPLOAD_RULING_SHA256,
            "prior_probe_disposition": "INCOMPLETE_CERTIFICATE_DEADLINE",
            "observability_ruling_sha256": OBSERVABILITY_SHA,
            "additional_attempt_allowance": 1,
            "amendment_sha256": AMENDMENT_SHA,
            "certificate_amendment_sha256": CERTIFICATE_AMENDMENT_SHA,
            "certificate_cap_seconds": {"base": 600, "effective": 1200},
            "original_reservation": {
                "path": str(OLD_RESERVATION),
                "sha256": OLD_RESERVATION_SHA256,
            },
            "retained_liability_minutes": 70,
            "prior_probe_reservation": {
                "path": str(PRIOR_PROBE_RESERVATION),
                "sha256": PRIOR_PROBE_RESERVATION_SHA256,
            },
            "reconciliation": (
                "retain all prior liabilities without reclaiming any; "
                "add separate nontransferable 40-minute calibration reservation"
            ),
            "ruling_sha256": RULING_SHA256,
            "source_commit": pin,
            "base_minutes": BASE_MINUTES,
            "effective_minutes": EFFECTIVE_MINUTES,
            "combined_effective_minutes_before": 502,
            "combined_effective_minutes_after": 542,
            "r30_1_allowance_minutes": 144,
            "r30_2_allowance_minutes": 288,
            "nontransferable": True,
            "no_automatic_retry": True,
            "technical_rate": {
                "verified_rate_units_per_hour": RATE_UNITS_PER_HOUR,
                "reserve_fraction": RESERVE_FRACTION,
            },
        },
    )
    bundle = {
        "dest": "private-inputs.tar.gz",
        "path": str(FROZEN_BUNDLE),
        "sha256": FROZEN_BUNDLE_SHA256,
        "size_bytes": FROZEN_BUNDLE_BYTES,
        "r30_private_bundle": {
            "opaque_reuse": True,
            "transfer_bar_seconds": TRANSFER_BAR_SECONDS,
            "staging_bar_seconds": STAGING_BAR_SECONDS,
        },
    }
    timeout_units = EFFECTIVE_MINUTES / 60 * RATE_UNITS_PER_HOUR
    public_b8 = {
        "dest": "b8.pth",
        "url": (
            "https://github.com/josenunezwork/snake-dqn/releases/download/"
            "s1-inputs-20261019/B8_s3_final.pth"
        ),
        "size_bytes": 6_562_005,
        "sha256": B8_SHA,
    }
    manifest = {
        "job_id": f"r30-1-{output.name}-{pin[:8]}",
        "commit": pin,
        "script": "research/bl_round30/timing_colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "canary": {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": "research/bl_round30/timing_colab.sh",
        },
        "requirements": {
            "gpu": "L4",
            "allowed_gpus": ["L4"],
            "min_vcpu": 1,
            "est_minutes": BASE_MINUTES,
        },
        "timeout_minutes": EFFECTIVE_MINUTES,
        "output_dir": str((output / "runtime-output").resolve()),
        "inputs": [
            public_b8,
            bundle,
            {
                "dest": "timing-certificate-amendment.md",
                "path": certificate_copy["path"],
                "sha256": certificate_copy["sha256"],
                "size_bytes": certificate_copy["size"],
            },
            {
                "dest": "timing-config.json",
                "path": config_copy["path"],
                "sha256": config_copy["sha256"],
                "size_bytes": config_copy["size"],
            },
            {
                "dest": "timing-ruling.md",
                "path": ruling_copy["path"],
                "sha256": ruling_copy["sha256"],
                "size_bytes": ruling_copy["size"],
            },
            {
                "dest": "timing-amendment.md",
                "path": amendment_copy["path"],
                "sha256": amendment_copy["sha256"],
                "size_bytes": amendment_copy["size"],
            },
            {
                "dest": "timing-observability-ruling.md",
                "path": observability_copy["path"],
                "sha256": observability_copy["sha256"],
                "size_bytes": observability_copy["size"],
            },
            {
                "dest": "timing-upload-ruling.md",
                "path": upload_ruling_copy["path"],
                "sha256": upload_ruling_copy["sha256"],
                "size_bytes": upload_ruling_copy["size"],
            },
        ],
        "env": {
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/bl_round30",
            "R30_DEVICE": "cuda",
            "R30_TIMING_CONFIG_SHA256": config_copy["sha256"],
            "R30_PRIVATE_BUNDLE_SHA256": FROZEN_BUNDLE_SHA256,
            "COLAB_CLI_REQUIREMENTS": "research/bl_round30/requirements.txt",
        },
        "required_exports": ["timing.json"],
        "max_export_bytes": TRANSPORT_BYTES,
        "budget": {
            "unit_cap": timeout_units,
            "rate_units_per_hour": RATE_UNITS_PER_HOUR,
            "reserve_units": timeout_units * RESERVE_FRACTION,
            "other_assignments": {},
        },
        "resume_exemption": {
            "kind": "short-canary",
            "reason": (
                "One isolated timing-only calibration; no scientific unit, ledger, retry, or "
                "outcome export."
            ),
            "timeout_minutes": EFFECTIVE_MINUTES,
        },
        "timing_calibration": {
            "schema": "r30-g5-timing-calibration/v2",
            "observability_ruling_sha256": OBSERVABILITY_SHA,
            "upload_ruling_sha256": UPLOAD_RULING_SHA256,
            "timing_payload_max_bytes": TIMING_PAYLOAD_BYTES,
            "amendment_sha256": AMENDMENT_SHA,
            "certificate_amendment_sha256": CERTIFICATE_AMENDMENT_SHA,
            "certificate_cap_seconds": {"base": 600, "effective": 1200},
            "deadlines": DEADLINES,
            "tested_dispatcher_sha256": digest(ROOT / "tools/ops/dispatch.py"),
            "ruling_sha256": RULING_SHA256,
            "config_sha256": config_copy["sha256"],
            "config_source_sha256": digest(timing_config),
            "frozen_control_sha256": digest(FROZEN_NATIVE / "inputs/control.json"),
            "frozen_bundle_sha256": FROZEN_BUNDLE_SHA256,
            "accounting_reservation_sha256": reservation["sha256"],
            "base_minutes": BASE_MINUTES,
            "effective_minutes": EFFECTIVE_MINUTES,
            "combined_effective_minutes_before": 502,
            "combined_effective_minutes_after": 542,
            "r30_1_allowance_minutes": 144,
            "r30_2_allowance_minutes": 288,
            "nontransferable": True,
            "no_automatic_retry": True,
            "main_remaining_minutes": 106,
            "main_estimate_ceiling_minutes": 53,
            "transport_bars": {
                "transfer_seconds": TRANSFER_BAR_SECONDS,
                "staging_seconds": STAGING_BAR_SECONDS,
            },
            "technical_rate_guard": {
                "verified_rate_units_per_hour": RATE_UNITS_PER_HOUR,
                "reserve_fraction": RESERVE_FRACTION,
                "unit_cap_role": "wall-clock reconciliation technical guard; not unit rationing",
            },
            "separate_accounting": reservation,
            "independent_predispatch_review_required": True,
            "qualified_transport_status": "SOURCE_BOUND_PROBE_PENDING_ACCEPT",
            "scientific_admission": "not-applicable-timing-only",
            "export_policy": (
                "timing.json plus telemetry/[000000..000063].json and immutable sidecars; "
                "all output including sidecars <=524288 bytes; "
                "incomplete telemetry is diagnostic only"
            ),
        },
    }
    export_json(output / "job.json", manifest)
    export_json(
        output / "SOURCE_BINDINGS.json",
        {
            "schema": "r30-timing-probe-source-bindings/v1",
            "source_commit": pin,
            "qualified_transport_tree": check(ROOT, OPS / "colab-cli-qualified.json", pin),
            "qualification_registry_sha256": digest(OPS / "colab-cli-qualified.json"),
            "dispatcher_sha256": digest(ROOT / "tools/ops/dispatch.py"),
            "manifest_sha256": digest(output / "job.json"),
            "config_sha256": config_copy["sha256"],
            "amendment_sha256": AMENDMENT_SHA,
            "certificate_amendment_sha256": CERTIFICATE_AMENDMENT_SHA,
            "certificate_cap_seconds": {"base": 600, "effective": 1200},
            "live_account_state": "NOT_CHECKED",
            "upload_ruling_sha256": UPLOAD_RULING_SHA256,
        },
    )
    export_json(
        output / "PRE_DISPATCH_REVIEW.json",
        {
            "schema": "r30-timing-predispatch-review-requirement/v1",
            "status": "REQUIRED",
            "job_sha256": digest(output / "job.json"),
            "requirements": [
                "independent implementation/redaction review",
                "timing schedule and source closure review",
                "qualified supervisor and current account review",
                "source-bound 40-minute family probe; no live account state established",
            ],
        },
    )
    return manifest


def check_upload_sizes(manifest: dict[str, Any], staging: Path) -> dict[str, Any]:
    """Check every prepared/control file and declared private input, without unpacking."""
    bound = manifest["max_export_bytes"]
    if type(bound) is not int or bound <= 0:
        raise ValueError("invalid upload transport bound")
    records = []
    for path in sorted(staging.iterdir()):
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe staged artifact")
        size = path.stat().st_size
        if size > bound:
            raise ValueError(f"UPLOAD_CAP: staged {path.name}")
        records.append(
            {"kind": "staged", "path": str(path), "size_bytes": size, "sha256": digest(path)}
        )
    for item in manifest["inputs"]:
        if "path" not in item:
            continue  # Public URL input is not privately uploaded.
        path = Path(item["path"])
        if path.is_symlink() or not path.is_file():
            raise ValueError("unsafe private input")
        size = path.stat().st_size
        if size != item["size_bytes"] or digest(path) != item["sha256"]:
            raise ValueError("private input size/hash differs")
        if size > bound:
            raise ValueError(f"UPLOAD_CAP: private {item['dest']}")
        records.append(
            {
                "kind": "private-input",
                "path": str(path),
                "size_bytes": size,
                "sha256": item["sha256"],
            }
        )
    return {
        "schema": "r30-timing-upload-preflight/v1",
        "status": "PASS",
        "transport_max_bytes": bound,
        "files": records,
    }


def prepare_upload_preflight(output: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Mirror qualified local preparation only; never construct a CLI or broker."""
    from research.colab_cli.source import prepare
    from research.colab_cli.supervisor import validate

    manifest = validate(manifest, ROOT)
    staging = output / "offline-staging"
    staging.mkdir(exist_ok=False)
    prepare(ROOT, manifest, staging)
    for name in ("runtime.py", "bootstrap.sh", "release.py", "source.py"):
        data = subprocess.check_output(
            ["git", "show", f"{manifest['commit']}:research/colab_cli/{name}"],
            cwd=ROOT,
            timeout=10,
        )
        (staging / name).write_bytes(data)
    result = check_upload_sizes(manifest, staging)
    export_json(output / "UPLOAD_PREFLIGHT.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timing-config", type=Path, required=True)
    args = parser.parse_args()
    manifest = make_timing_pack(
        args.output.resolve(), commit=args.commit, timing_config=args.timing_config
    )
    prepare_upload_preflight(args.output.resolve(), manifest)
    print(json.dumps({"manifest": str(args.output / "job.json"), "job_id": manifest["job_id"]}))


if __name__ == "__main__":
    main()
