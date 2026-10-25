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
TRANSFER_BAR_SECONDS = 150
STAGING_BAR_SECONDS = 300
RATE_UNITS_PER_HOUR = 1.54
BASE_MINUTES = 20
EFFECTIVE_MINUTES = 40
RESERVE_FRACTION = 0.20
SOURCE_PATHS = [*R30_SOURCE_PATHS, "tools/ops/qualification.py"]
EXPECTED_SCHEDULE = {
    "adam_blocks": [[0, 10], [10, 20], [20, 30]],
    "closure_updates": [0, 10, 30],
    "fit_batches": 2000,
    "fit_seed": 3403,
    "audit_seed": 3404,
    "certificate_update": 30,
    "certificate_cap_seconds": 600,
}


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
        value.get("schema") != "r30-timing-config/v1"
        or value.get("ruling_sha256") != RULING_SHA256
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
    if output.exists():
        raise FileExistsError(output)
    if digest(RULING) != RULING_SHA256:
        raise ValueError("timing ruling hash differs")
    if (
        digest(FROZEN_BUNDLE) != FROZEN_BUNDLE_SHA256
        or FROZEN_BUNDLE.stat().st_size != FROZEN_BUNDLE_BYTES
    ):
        raise ValueError("frozen private bundle identity differs")
    pin = _published_clean(commit)
    control = _read_json(FROZEN_NATIVE / "inputs/control.json")
    load_timing_config(timing_config, commit=pin, control=control)
    output.mkdir(parents=True)
    inputs = output / "inputs"
    inputs.mkdir()
    config_copy = export_bytes(inputs / "timing-config.json", timing_config.read_bytes())
    ruling_copy = export_bytes(inputs / "timing-ruling.md", RULING.read_bytes())
    reservation = export_json(
        output / "accounting-reservation.json",
        {
            "schema": "r30-g5-timing-calibration-accounting/v1",
            "entry": "r30-1-g5-timing-calibration-01",
            "ruling_sha256": RULING_SHA256,
            "source_commit": pin,
            "base_minutes": BASE_MINUTES,
            "effective_minutes": EFFECTIVE_MINUTES,
            "combined_effective_minutes_before": 432,
            "combined_effective_minutes_after": 472,
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
        "job_id": f"r30-1-timing-calibration-{pin[:8]}",
        "commit": pin,
        "script": "research/bl_round30/timing_colab.sh",
        "source_paths": SOURCE_PATHS,
        "device": "cuda",
        "requirements": {"gpu": "L4", "allowed_gpus": ["L4"], "min_vcpu": 1, "est_minutes": 20},
        "timeout_minutes": EFFECTIVE_MINUTES,
        "output_dir": str((output / "runtime-output").resolve()),
        "inputs": [
            public_b8,
            bundle,
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
        ],
        "env": {
            "R30_DEVICE": "cuda",
            "R30_TIMING_CONFIG_SHA256": config_copy["sha256"],
            "R30_PRIVATE_BUNDLE_SHA256": FROZEN_BUNDLE_SHA256,
            "COLAB_CLI_REQUIREMENTS": "research/bl_round30/requirements.txt",
        },
        "required_exports": ["timing.json"],
        "max_export_bytes": 512 * 1024,
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
            "schema": "r30-g5-timing-calibration/v1",
            "ruling_sha256": RULING_SHA256,
            "config_sha256": config_copy["sha256"],
            "config_source_sha256": digest(timing_config),
            "frozen_control_sha256": digest(FROZEN_NATIVE / "inputs/control.json"),
            "frozen_bundle_sha256": FROZEN_BUNDLE_SHA256,
            "accounting_reservation_sha256": reservation["sha256"],
            "base_minutes": BASE_MINUTES,
            "effective_minutes": EFFECTIVE_MINUTES,
            "combined_effective_minutes_before": 432,
            "combined_effective_minutes_after": 472,
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
            "qualified_transport_status": "HOLD",
            "dispatch_blocker": (
                "The qualified supervisor currently requires a <=30-minute family canary receipt; "
                "this ruling admits no G4 probe and fixes the isolated calibration at 40 minutes. "
                "A narrowly reviewed transport admission is required before dispatch."
            ),
            "scientific_admission": "not-applicable-timing-only",
            "export_policy": "timing.json and its supervisor sidecars only",
        },
    }
    export_json(output / "job.json", manifest)
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
                "narrowly reviewed qualified-transport admission for the 40-minute calibration",
            ],
        },
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--timing-config", type=Path, required=True)
    args = parser.parse_args()
    manifest = make_timing_pack(
        args.output.resolve(), commit=args.commit, timing_config=args.timing_config
    )
    print(json.dumps({"manifest": str(args.output / "job.json"), "job_id": manifest["job_id"]}))


if __name__ == "__main__":
    main()
