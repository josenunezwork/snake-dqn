"""Prepare an isolated D256 operational sibling, then run calibrated W12 or W6.

The original Mac package and its ongoing M-E run are read-only. There is one
fixed sibling and no reset/migration of an existing D256 attempt. F1/F0/F4
admission and the original namespace/allocation/cap remain binding.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from . import mac_driver, mac_gates, w12_calibration
from .exports import _materialize_triplet, _verify

OPERATIONAL_NAME = "mac-d256-w12-01"
COPY_NAMES = (
    "control.json",
    "canary-b-manifest.json",
    "d256-control.json",
    "d256-seed-audit.json",
    "d256-verified-ledger.json",
    "f0-look-claim.json",
)


def select_workers(receipt: Path, owner: Path) -> tuple[int, list[str]]:
    """Fallback only on a sealed measured calibration failure, never a HOLD."""
    approval_sha = w12_calibration.owner_approval(owner)
    value = mac_gates._read(receipt)
    if (
        value.get("schema") != w12_calibration.SCHEMA
        or value.get("code_sha") != os.environ.get("F0_COMMIT")
        or value.get("platform") != "darwin-arm64"
        or value.get("owner_approval_sha256") != approval_sha
        or value.get("bank_sha256") != w12_calibration.F1_BANK_SHA256
        or value.get("runner_sha256") != w12_calibration.sha(Path(w12_calibration.__file__))
    ):
        raise ValueError("calibration provenance differs; HOLD")
    samples = receipt.parent / "host-samples.jsonl"
    if samples.stat().st_size != value.get("host_samples_bytes") or w12_calibration.sha(
        samples
    ) != value.get("host_samples_sha256"):
        raise ValueError("calibration host samples differ; HOLD")
    if value.get("status") == "PASS":
        mac_driver._verify_w12_calibration(receipt)
        return 12, ["--owner-w12-ok", "--w12-calibration-receipt", str(receipt.resolve())]
    ratio = value.get("w12_w6_wall_ratio")
    ratio_failed = (
        isinstance(ratio, (float, int))
        and not isinstance(ratio, bool)
        and math.isfinite(ratio)
        and ratio > 1.30
        and value.get("duration_seconds", 0) >= 1800
    )
    if value.get("status") == "FAIL" and (ratio_failed or value.get("slowdown_trips") == 1):
        return 6, []
    raise ValueError("calibration missing, incomplete or mechanically unsafe; HOLD")


def require_no_prior_d256(pack: Path) -> None:
    """Prevent relocation from discarding charged attempts, even empty starts."""
    for pattern in ("d256-state*", "d256-output*", "d256-admitted*"):
        if any(pack.glob(pattern)):
            raise ValueError("original D256 attempt/admission already exists; HOLD, no relocation")
    ledger = mac_gates._read(pack / "d256-verified-ledger.json")
    if ledger.get("units") != []:
        raise ValueError("original D256 ledger is not empty; HOLD")


def prepare_sibling(pack: Path) -> Path:
    """Copy byte-identical sealed prerequisites; preserve original look bindings."""
    require_no_prior_d256(pack)
    target = pack.parent / OPERATIONAL_NAME
    if target.exists():
        # A lease/check refusal may leave a prepared but never-started package.
        # Reuse only those exact bytes: any output/state forbids this path.
        allowed = {
            name + suffix
            for name in (*COPY_NAMES, "d256-admitted.json")
            for suffix in ("", ".sha256", ".complete")
        }
        if any(p.name not in allowed or not p.is_file() for p in target.iterdir()):
            raise ValueError("D256 sibling has attempt/foreign evidence; HOLD, no reset")
        for name in COPY_NAMES:
            if _verify(pack / name) != _verify(target / name):
                raise ValueError("existing D256 sibling prerequisite differs; HOLD")
        return target
    target.mkdir(exist_ok=False)
    for name in COPY_NAMES:
        _materialize_triplet(pack / name, target / name)
        if _verify(pack / name) != _verify(target / name):
            raise ValueError("D256 sibling prerequisite copy differs")
    return target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=w12_calibration.PACK)
    parser.add_argument("--owner-approval", type=Path, default=w12_calibration.OWNER_FILE)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--f4-approval", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    if args.pack.resolve() != w12_calibration.PACK.resolve():
        raise ValueError(
            "D256 launcher requires canonical mac-pack-02; no alternate accounting root"
        )
    os.environ.update(mac_driver.THREAD_ENV)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    pin = mac_gates._source_pin(args.pack)
    if os.environ.get("F0_COMMIT", pin) != pin:
        raise ValueError("F0_COMMIT differs from frozen package source")
    os.environ.update(F0_COMMIT=pin, F1_COMMIT=pin)
    workers, flags = select_workers(args.calibration_receipt, args.owner_approval)
    calibration = mac_gates._read(args.calibration_receipt)
    for key, path in (("b8_sha256", args.b8), ("frp3_sha256", args.frp3)):
        if calibration.get(key) != w12_calibration.sha(path):
            raise ValueError("D256 checkpoints differ from calibrated inputs; HOLD")
    require_no_prior_d256(args.pack)
    # Run all scientific gate checks before creating any sibling. No new look.
    f1, look = args.pack / "f1-output", args.pack / "f0-look/decision.json"
    mac_gates._f1(args.pack, f1, pin)
    mac_gates._require_f0(args.pack, look, pin)
    mac_gates._approval(args.f4_approval)
    mac_gates._seed_audit(args.pack)
    if mac_driver._timing_armed(mac_driver.DEFAULT_TIMING_ARMED):
        raise ValueError("timing armed; HOLD")
    mac_driver._guard_snapshot(args.pack.parent)
    target = args.pack.parent / OPERATIONAL_NAME
    if args.check:
        if target.exists():
            prepare_sibling(args.pack)  # Verification-only for an existing sibling.
        print(json.dumps({"status": "CHECKED", "workers": workers, "package": str(target)}))
        return
    target = prepare_sibling(args.pack)
    control = target / "d256-admitted.json"
    if any(target.glob("d256-admitted.json*")):
        admitted = mac_gates._read(control)
        mac_gates.require_admission(admitted)
        if (
            Path(admitted["mac_gate"]["bindings"]["f4"]["path"]).resolve()
            != args.f4_approval.resolve()
        ):
            raise ValueError("prepared D256 approval path differs; HOLD")
    else:
        mac_gates.admit_d256(target, f1, look, args.f4_approval, control)
    command = [
        sys.executable,
        "-m",
        "research.bl_labels.mac_driver",
        "--control",
        str(control),
        "--ledger",
        str(target / "d256-verified-ledger.json"),
        "--b8",
        str(args.b8),
        "--frp3",
        str(args.frp3),
        "--output",
        str(target / "d256-output-mac-01"),
        "--state-dir",
        str(target / "d256-state-mac-01"),
        "--workers",
        str(workers),
        "--deadline-seconds",
        "57600",
        *flags,
    ]
    print(
        json.dumps({"event": "d256_selected", "workers": workers, "package": str(target)}),
        flush=True,
    )
    subprocess.run([*command, "--check"], check=True)
    # The frozen driver owns the only lease and all cumulative accounting.
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
