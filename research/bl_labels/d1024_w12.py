"""Admit and launch a sealed D1024 control at W12 or measured-failure W6.

This launcher makes no judgment from labels.  It only binds the new speed pin
to a PASS T-EQ receipt and a separately frozen W12 calibration receipt.  A
calibration HOLD never becomes a W6 fallback.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from . import mac_driver, mac_gates, speed_equiv, w12_calibration
from .exports import load_ledger


def _calibration_workers(receipt: Path, legacy_pin: str) -> tuple[int, list[str]]:
    """Return W12 only for PASS; use W6 only for a sealed measured failure."""
    status = mac_gates._future_calibration(receipt, legacy_pin, mac_gates.CALIBRATION_RUNNER_COMMIT)
    if status == "PASS":
        return 12, ["--owner-w12-ok", "--w12-calibration-receipt", str(receipt.resolve())]
    if status == "FAIL":
        return 6, []
    raise AssertionError("future calibration verifier returned an unknown state")


def _validate_inputs(
    control: Path,
    speed_receipt: Path,
    calibration_receipt: Path,
    b8: Path,
    frp3: Path,
    old_factory_root: Path | None,
    f4_control: Path,
    f4_inputs: Path,
) -> tuple[dict, int, list[str], bool]:
    value = mac_gates._read(control)
    gate = value.get("mac_gate")
    if not isinstance(gate, dict) or gate.get("stage") != "d1024" or gate.get("status") != "HOLD":
        raise ValueError("D1024 control must be an unmodified sealed HOLD")
    pin = mac_gates._future_source_pin(value)
    legacy_pin = gate.get("w12_calibration_source_commit")
    if not isinstance(legacy_pin, str):
        raise ValueError("D1024 control lacks a frozen calibration source pin")
    if os.environ.get("F0_COMMIT", pin) != pin:
        raise ValueError("F0_COMMIT differs from the sealed D1024 speed pin")
    os.environ.update(F0_COMMIT=pin, F1_COMMIT=pin)
    mac_gates._future_readiness("d1024", f4_control, f4_inputs)
    speed = mac_gates._read(speed_receipt)
    old_fallback = speed.get("status") == "FAIL"
    if old_fallback:
        if old_factory_root is None:
            raise ValueError("T-EQ FAIL requires explicit --old-factory-root; HOLD")
        if (
            speed.get("schema") != speed_equiv.SCHEMA
            or speed.get("old", {}).get("commit") != legacy_pin
            or speed.get("new", {}).get("commit") != pin
            or speed.get("f1_bank_sha256") != mac_gates.F1_BANK_SHA256
        ):
            raise ValueError("T-EQ FAIL receipt does not bind the OLD fallback; HOLD")
    else:
        if old_factory_root is not None:
            raise ValueError("--old-factory-root is only legal after T-EQ FAIL")
        speed_equiv.verify_receipt(
            speed_receipt, legacy_pin, pin, expected_bank_sha=mac_gates.F1_BANK_SHA256
        )
    workers, flags = _calibration_workers(calibration_receipt, legacy_pin)
    if old_fallback:
        # The speed candidate is rejected; execution returns to the historical
        # factory at W6 even if the independent width calibration passed.
        workers, flags = 6, []
    calibration = mac_gates._read(calibration_receipt)
    for key, checkpoint in (("b8_sha256", b8), ("frp3_sha256", frp3)):
        if not checkpoint.is_file() or calibration.get(key) != w12_calibration.sha(checkpoint):
            raise ValueError("D1024 checkpoint differs from frozen calibration; HOLD")
    return value, workers, flags, old_fallback


def _check_paths(
    control: dict, ledger: Path, output: Path, state_dir: Path, old_root: Path | None
) -> None:
    """Fail before CHECKED on a foreign ledger, path, evidence or OLD tree."""
    load_ledger(ledger, control["namespace"], control["allocation"])
    expected_state = (
        Path(control["mac_gate"].get("package_root", control.get("_control_parent", "")))
        / control["mac_gate"]["state_dir_name"]
    ).resolve()
    if state_dir.resolve() != expected_state:
        raise ValueError("D1024 state directory differs from sealed canonical path")
    if output.exists() and any(output.iterdir()):
        raise ValueError("D1024 output already contains evidence; HOLD")
    if old_root is not None:
        if old_root.resolve() == Path(__file__).resolve().parents[2]:
            raise ValueError("OLD factory root must differ from the speed worktree")
        if (old_root / ".IN_USE_LOCK").exists():
            raise ValueError("OLD factory root has a live lock")
        head = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=old_root, text=True
        ).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=old_root, text=True)
        detached = (
            subprocess.run(
                ["git", "symbolic-ref", "-q", "HEAD"], cwd=old_root, check=False
            ).returncode
            != 0
        )
        if head != mac_gates.LEGACY_FACTORY_COMMIT or dirty.strip() or not detached:
            raise ValueError("OLD factory root is not clean detached frozen 61900af2")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--admitted-control", type=Path, required=True)
    parser.add_argument("--speed-equivalence-receipt", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--old-factory-root", type=Path)
    parser.add_argument("--f4-control", type=Path, required=True)
    parser.add_argument("--f4-inputs", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    os.environ.update(mac_driver.THREAD_ENV)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    control, workers, flags, old_fallback = _validate_inputs(
        args.control,
        args.speed_equivalence_receipt,
        args.calibration_receipt,
        args.b8,
        args.frp3,
        args.old_factory_root,
        args.f4_control,
        args.f4_inputs,
    )
    control["_control_parent"] = str(args.control.parent.resolve())
    _check_paths(control, args.ledger, args.output, args.state_dir, args.old_factory_root)
    deadline = control["mac_gate"]["deadline_seconds"]
    if args.check:
        if args.admitted_control.exists():
            mac_gates.require_admission(mac_gates._read(args.admitted_control))
        print(json.dumps({"status": "CHECKED", "workers": workers, "deadline_seconds": deadline}))
        return
    if args.admitted_control.exists():
        admitted = mac_gates._read(args.admitted_control)
        mac_gates.require_admission(admitted)
    else:
        mac_gates.admit_future(
            args.control,
            speed_equivalence=args.speed_equivalence_receipt,
            calibration=args.calibration_receipt,
            out=args.admitted_control,
            f4_control=args.f4_control,
            f4_inputs=args.f4_inputs,
            old_factory_root=args.old_factory_root if old_fallback else None,
        )
    command = [
        sys.executable,
        "-m",
        "research.bl_labels.mac_driver",
        "--control",
        str(args.admitted_control),
        "--ledger",
        str(args.ledger),
        "--b8",
        str(args.b8),
        "--frp3",
        str(args.frp3),
        "--output",
        str(args.output),
        "--state-dir",
        str(args.state_dir),
        "--workers",
        str(workers),
        "--deadline-seconds",
        str(deadline),
        *flags,
    ]
    subprocess.run([*command, "--check"], check=True)
    print(
        json.dumps({"event": "d1024_selected", "workers": workers, "old_fallback": old_fallback}),
        flush=True,
    )
    os.execv(sys.executable, command)


if __name__ == "__main__":
    main()
