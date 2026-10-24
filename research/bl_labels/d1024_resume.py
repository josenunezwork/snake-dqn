"""Outcome-blind, accounting-preserving continuation of D1024 OLD W6.

Reuse the original admission and state, union only verified immutable exports,
and charge every terminal attempt against the original cumulative wall cap.
No admission, scientific aggregation, or outcome reporting is performed here.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from . import d1024_w12, frozen_factory, mac_driver, mac_gates
from .d256_resume import REGISTERED_RATE_STOP, canonical_sha
from .exports import _verify, load_ledger

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PACK = ART / "bl-labels-f0-speed/d1024-old-w6-equiv-02-pack-01"
SCIENCE_PIN = "111c47818a097c57451386ef4f8a7009616f8f77"
ORIGINAL_OPERATIONAL_PIN = "587ff73ecdb9ebb25a104c3e45450ceb851c0358"
CAP = 144000
WORKERS = 6


def original_control(pack: Path) -> dict:
    """Read the existing sealed admission; never create or replace it."""
    value = mac_gates._read(pack / "d1024-admitted.json")
    original = mac_gates._read(pack / "d1024-control.json")
    gate = value.get("mac_gate", {})
    if (
        value.get("source_commit") != SCIENCE_PIN
        or gate.get("source_commit") != SCIENCE_PIN
        or gate.get("stage") != "d1024"
        or gate.get("status") != "APPROVED"
        or gate.get("deadline_seconds") != CAP
        or gate.get("package_root") != str(pack.resolve())
        or gate.get("state_dir") != str((pack / "d1024-state-mac-01").resolve())
        or gate.get("original_control_path") != str((pack / "d1024-control.json").resolve())
        or gate.get("original_control_sha256") != _verify(pack / "d1024-control.json")
        or gate.get("collector_source_commit") != frozen_factory.OLD_PIN
        or gate.get("bindings", {}).get("speed_equivalence", {}).get("status")
        != "FAIL_OLD_FALLBACK"
        or value.get("namespace") != original.get("namespace")
        or value.get("allocation") != original.get("allocation")
    ):
        raise ValueError("original D1024 OLD W6 admission identity differs; HOLD")
    return value


def attempt_chain(pack: Path, control: dict, *, final: bool = False) -> tuple[list[Path], float]:
    """Verify all starts, terminals and exports, including zero-unit attempts.

    ``final`` is used only by the post-collection union publisher: the last
    attempt must then be the driver's COMPLETE terminal and every earlier one
    an operational PREEMPTED terminal. The default continuation path is unchanged.
    """
    state = pack / "d1024-state-mac-01"
    starts = sorted(state.glob("attempt-*.start.json"), key=lambda p: p.stat().st_mtime_ns)
    if not starts:
        raise ValueError("original D1024 start receipt missing; HOLD")
    names: list[str] = []
    attempts: list[Path] = []
    elapsed_total = 0.0
    worker_total = 0.0
    for index, path in enumerate(starts, 1):
        start = mac_gates._read(path)
        terminal_path = path.with_name(path.name.replace(".start.json", ".terminal.json"))
        if not terminal_path.is_file():
            raise ValueError("prior D1024 attempt lacks a verified terminal receipt; HOLD")
        terminal = mac_gates._read(terminal_path)
        if terminal.get("registered_stop_reason") is not None or any(
            terminal.get(key) == REGISTERED_RATE_STOP
            for key in ("stop_reason", "post_slowdown_stop_reason")
        ):
            raise ValueError("registered D1024 stopping rule already fired; HOLD")
        output = pack / f"d1024-output-mac-{index:02d}"
        if (
            start.get("schema") != "bl-labels-f0-mac-attempt-start/v1"
            or start.get("namespace") != control["namespace"]
            or start.get("source_commit") != SCIENCE_PIN
            or start.get("allocation_sha256") != canonical_sha(control["allocation"])
            or start.get("control_sha256") != canonical_sha(control)
            or start.get("deadline_seconds") != CAP
            or start.get("original_wall_cap_seconds") != CAP
            or start.get("state_dir") != str(state.resolve())
            or start.get("output") != str(output.resolve())
            or type(start.get("workers")) is not int
            or start["workers"] != WORKERS
            or start.get("owner_w12_ok") is not False
            or start.get("w12_calibration_receipt_sha256") is not None
            or start.get("w12_calibration_source_commit") != frozen_factory.OLD_PIN
            or (index == 1 and start.get("operational_source_commit") != ORIGINAL_OPERATIONAL_PIN)
        ):
            raise ValueError("prior D1024 start identity differs; HOLD")
        exported = mac_gates._read(output / "runtime-result.json")
        if {k: v for k, v in exported.items() if k != "mac_lease"} != terminal:
            raise ValueError("D1024 terminal/export receipt differs; HOLD")
        names.append(terminal_path.name)
        if (
            terminal.get("schema") != "bl-labels-f0-mac-driver/v1"
            or terminal.get("status")
            != ("COMPLETE" if final and index == len(starts) else "PREEMPTED")
            or (final and index == len(starts) and terminal.get("pending_parents") != [])
            or terminal.get("invalid_reason") is not None
            or list(output.glob("INVALID*.json*"))
            or terminal.get("namespace") != control["namespace"]
            or terminal.get("workers") != WORKERS
            or terminal.get("owner_w12_ok") is not False
            or terminal.get("collector_source_commit") != frozen_factory.OLD_PIN
            or terminal.get("operational_source_commit") != start.get("operational_source_commit")
            or terminal.get("w12_calibration_source_commit") != frozen_factory.OLD_PIN
            or terminal.get("w12_calibration_receipt_sha256") is not None
            or terminal.get("output") != str(output.resolve())
            or terminal.get("state_dir") != str(state.resolve())
            or terminal.get("start_sha256") != _verify(path)
            or terminal.get("start_receipt") != path.name
            or terminal.get("attempts_ledger") != names
            or terminal.get("prior_elapsed_seconds") != elapsed_total
            or terminal.get("prior_worker_seconds_denominator") != worker_total
        ):
            raise ValueError("non-preempted, INVALID or mismatched D1024 terminal chain; HOLD")
        elapsed = terminal.get("elapsed_seconds")
        if type(elapsed) not in (int, float) or not math.isfinite(elapsed) or elapsed <= 0:
            raise ValueError("invalid prior D1024 wall charge; HOLD")
        worker_charge = WORKERS * elapsed
        if terminal.get("worker_seconds_denominator") != worker_charge:
            raise ValueError("prior D1024 worker-second accounting differs; HOLD")
        elapsed_total += elapsed
        worker_total += worker_charge
        if (
            terminal.get("cumulative_elapsed_seconds") != elapsed_total
            or terminal.get("cumulative_worker_seconds_denominator") != worker_total
        ):
            raise ValueError("cumulative D1024 accounting differs; HOLD")
        sealed = path.with_name(path.name.replace(".start.json", ".rate-sealed.json"))
        if terminal.get("rate_sealed_path") != str(sealed.resolve()):
            raise ValueError("D1024 sealed rate path differs; HOLD")
        _verify(sealed)  # Hash only: no sealed-rate outcome is read.
        attempts.append(output)
    if {p.name for p in state.glob("attempt-*.terminal.json")} != set(names):
        raise ValueError("orphan D1024 terminal receipt; HOLD")
    if set(pack.glob("d1024-output-mac-*")) != set(attempts):
        raise ValueError("uncharged or foreign D1024 output directory; HOLD")
    if elapsed_total > CAP or (not final and elapsed_total >= CAP):
        raise ValueError("prior D1024 attempts exhausted the frozen wall cap; HOLD")
    return attempts, elapsed_total


def verify_union(ledger: Path, control: dict, attempts: list[Path]) -> None:
    """Reuse only the identical verified union after a check or lease refusal."""
    value = mac_gates._read(ledger)
    if value.get("attempt_exports") != [str(p.resolve()) for p in attempts]:
        raise ValueError("prepared D1024 union attempt set differs; HOLD")
    if value.get("units") != mac_gates.ledger_units(attempts):
        raise ValueError("prepared D1024 union units differ; HOLD")
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    mac_driver._verify_resume_metadata(ledger, control["namespace"], control["allocation"], names)


def publish_ledger(pack: Path, attempts: list[Path], out: Path) -> dict:
    """Publish path/size/hash and wrapper identities without any new look."""
    control = original_control(pack)
    expected, _ = attempt_chain(pack, control)
    if [p.resolve() for p in attempts] != [p.resolve() for p in expected]:
        raise ValueError("omitted, duplicate or foreign D1024 attempts; HOLD")
    return mac_gates.publish_ledger(pack, attempts, out, control=control)


def launch(
    pack: Path,
    speed: Path,
    calibration: Path,
    old_root: Path,
    f4_control: Path,
    f4_inputs: Path,
    b8: Path,
    frp3: Path,
    *,
    check: bool = False,
) -> None:
    """Revalidate the original admission, then check or execute the coordinator."""
    if pack.resolve() != PACK.resolve():
        raise ValueError("D1024 resume requires canonical OLD W6 equiv-02 package; HOLD")
    control = original_control(pack)
    attempts, elapsed = attempt_chain(pack, control)
    os.environ.update(mac_driver.THREAD_ENV)
    os.environ["SNAKE_DQN_DEVICE"] = "cpu"
    d1024_w12._prepare_science_root(pack / "d1024-control.json", "d1024")
    _, workers, flags, old_fallback = d1024_w12._validate_inputs(
        "d1024",
        pack / "d1024-control.json",
        speed,
        calibration,
        b8,
        frp3,
        old_root,
        f4_control,
        f4_inputs,
    )
    gate = control["mac_gate"]
    bindings = gate["bindings"]
    if (
        workers != WORKERS
        or flags
        or not old_fallback
        or str(old_root.resolve()) != gate["collector_root"]
        or str(speed.resolve()) != bindings["speed_equivalence"]["path"]
        or str(calibration.resolve()) != bindings["calibration"]["path"]
        or str(f4_control.resolve()) != bindings["f4_readiness"]["control_path"]
        or str(f4_inputs.resolve()) != bindings["f4_readiness"]["inputs_path"]
    ):
        raise ValueError("D1024 resume environment differs from original admission; HOLD")
    mac_gates.require_admission(control)
    frozen_factory._validate(old_root, frozen_factory.OLD_PIN)
    if mac_driver._timing_armed(mac_driver.DEFAULT_TIMING_ARMED):
        raise ValueError("timing armed; HOLD")
    mac_driver._guard_snapshot(pack.parent)
    index = len(attempts) + 1
    output = pack / f"d1024-output-mac-{index:02d}"
    ledger = pack / f"d1024-resume-{index:02d}/verified-ledger.json"
    if ledger.exists():
        verify_union(ledger, control, attempts)
    else:
        publish_ledger(pack, attempts, ledger)
    _, _, pending = mac_driver._prepare_jobs(control, ledger)
    if not pending:
        raise ValueError("D1024 has no missing parents; HOLD")
    command = [
        sys.executable,
        "-m",
        "research.bl_labels.mac_driver",
        "--control",
        str(pack / "d1024-admitted.json"),
        "--ledger",
        str(ledger),
        "--b8",
        str(b8),
        "--frp3",
        str(frp3),
        "--output",
        str(output),
        "--state-dir",
        str(pack / "d1024-state-mac-01"),
        "--workers",
        str(WORKERS),
        "--deadline-seconds",
        str(CAP),
    ]
    print(
        json.dumps(
            {
                "event": "d1024_resume",
                "workers": WORKERS,
                "prior_attempts": len(attempts),
                "prior_elapsed_seconds": elapsed,
                "deadline_seconds": CAP,
                "remaining_cap_seconds": CAP - elapsed,
                "pending_parents": len(pending),
                "output": str(output),
                "ledger": str(ledger),
            }
        ),
        flush=True,
    )
    subprocess.run([*command, "--check"], check=True)
    if not check:
        os.execv(sys.executable, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--speed-equivalence-receipt", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--old-factory-root", type=Path, required=True)
    parser.add_argument("--f4-control", type=Path, required=True)
    parser.add_argument("--f4-inputs", type=Path, required=True)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    launch(
        args.pack,
        args.speed_equivalence_receipt,
        args.calibration_receipt,
        args.old_factory_root,
        args.f4_control,
        args.f4_inputs,
        args.b8,
        args.frp3,
        check=args.check,
    )


if __name__ == "__main__":
    main()
