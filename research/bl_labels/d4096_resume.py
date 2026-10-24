"""Outcome-blind, accounting-preserving continuation of D4096 OLD collection.

This is deliberately separate from the live D1024 W6 continuation.  It accepts
only the sealed D4096 OLD rebinding, charges every terminal Mac attempt against
the original 130-hour wall cap, and unions only sidecar-verified immutable
exports.  W6/W12 may change only between attempts; the driver owns the
per-width operational baselines and W12 sanity checks.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from . import d1024_w12, d4096_pack, d4096_sealed, frozen_factory, mac_driver, mac_gates
from .d256_resume import REGISTERED_RATE_STOP, canonical_sha
from .exports import _verify, load_ledger

ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts")
PACK = ART / "bl-labels-f0-speed/d4096-old-equiv-02-pack-01"
SCIENCE_PIN = d4096_pack.SCIENCE
# The rebinding's original science string is ``bf8…``.  The admitted OLD
# collector and calibration provenance are instead the frozen 61900af2 tree.
OLD_PIN = frozen_factory.OLD_PIN
CAP = 468000
WIDTHS = frozenset((6, 12))


def _read(path: Path) -> dict[str, Any]:
    return mac_gates._read(path)


def _finite_positive(value: object, what: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{what} is not finite; HOLD")
    if value <= 0:
        raise ValueError(f"{what} must be positive; HOLD")
    return float(value)


def original_control(pack: Path) -> dict[str, Any]:
    """Read and revalidate the sealed D4096 admission without replacing it."""
    control_path = pack / "d4096-control.json"
    admitted_path = pack / "d4096-admitted.json"
    control, admitted = _read(control_path), _read(admitted_path)
    gate = admitted.get("mac_gate", {})
    if (
        control.get("source_commit") != SCIENCE_PIN
        or control.get("mac_gate", {}).get("source_commit") != SCIENCE_PIN
        or control.get("mac_gate", {}).get("stage") != "d4096"
        or control.get("mac_gate", {}).get("status") != "HOLD"
        or gate.get("stage") != "d4096"
        or gate.get("status") != "APPROVED"
        or admitted.get("source_commit") != SCIENCE_PIN
        or gate.get("source_commit") != SCIENCE_PIN
        or gate.get("deadline_seconds") != CAP
        or gate.get("package_root") != str(pack.resolve())
        or gate.get("state_dir") != str((pack / "d4096-state-mac-01").resolve())
        or gate.get("original_control_path") != str(control_path.resolve())
        or gate.get("original_control_sha256") != _verify(control_path)
        or gate.get("collector_source_commit") != OLD_PIN
        or gate.get("bindings", {}).get("speed_equivalence", {}).get("status")
        != "FAIL_OLD_FALLBACK"
        or admitted.get("namespace") != control.get("namespace")
        or admitted.get("allocation") != control.get("allocation")
    ):
        raise ValueError("original D4096 sealed OLD admission identity differs; HOLD")
    mac_gates.require_admission(admitted)
    return admitted


def _verify_start(start: dict[str, Any], control: dict[str, Any], state: Path, output: Path) -> int:
    workers = start.get("workers")
    if type(workers) is not int or workers not in WIDTHS:
        raise ValueError("prior D4096 start worker width differs; HOLD")
    w12 = workers == 12
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
        or start.get("funding_state") != "PENDING_SEALED"
        or start.get("judgment_sha256") != d4096_pack.JUDGMENT_SHA
        or not isinstance(start.get("operational_source_commit"), str)
        or len(start["operational_source_commit"]) != 40
        or start.get("owner_w12_ok") is not w12
        or (w12 and not isinstance(start.get("w12_calibration_receipt_sha256"), str))
        or (not w12 and start.get("w12_calibration_receipt_sha256") is not None)
        or start.get("w12_calibration_source_commit") != OLD_PIN
    ):
        raise ValueError("prior D4096 start identity differs; HOLD")
    return workers


def attempt_chain(
    pack: Path, control: dict[str, Any], *, final: bool = False
) -> tuple[list[Path], float]:
    """Verify every terminal attempt and charge its full elapsed wall time.

    No labels, terminal rate payloads, or result metrics are inspected.  The
    only export data later used is the immutable sidecar-verified triplet union.
    """
    state = pack / "d4096-state-mac-01"
    starts = sorted(state.glob("attempt-*.start.json"), key=lambda path: path.stat().st_mtime_ns)
    terminals = {path.name for path in state.glob("attempt-*.terminal.json")}
    names: list[str] = []
    attempts: list[Path] = []
    elapsed_total = 0.0
    worker_total = 0.0
    for index, path in enumerate(starts, 1):
        terminal_path = path.with_name(path.name.replace(".start.json", ".terminal.json"))
        if not terminal_path.is_file():
            raise ValueError("prior D4096 attempt lacks a verified terminal receipt; HOLD")
        start, terminal = _read(path), _read(terminal_path)
        output = pack / f"d4096-output-mac-{index:02d}"
        workers = _verify_start(start, control, state, output)
        exported = _read(output / "runtime-result.json")
        if {key: value for key, value in exported.items() if key != "mac_lease"} != terminal:
            raise ValueError("D4096 terminal/export receipt differs; HOLD")
        names.append(terminal_path.name)
        complete = final and index == len(starts)
        # Charge the terminal before deciding whether it is reusable.  An
        # INVALID or registered-stop attempt is operationally unusable, but
        # its measured lifetime may never disappear from the frozen cap.
        elapsed = _finite_positive(terminal.get("elapsed_seconds"), "prior D4096 wall charge")
        charge = workers * elapsed
        if terminal.get("worker_seconds_denominator") != charge:
            raise ValueError("prior D4096 worker-second accounting differs; HOLD")
        prior_elapsed, prior_worker_total = elapsed_total, worker_total
        next_elapsed = prior_elapsed + elapsed
        next_worker_total = prior_worker_total + charge
        if (
            terminal.get("cumulative_elapsed_seconds") != next_elapsed
            or terminal.get("cumulative_worker_seconds_denominator") != next_worker_total
        ):
            raise ValueError("cumulative D4096 width accounting differs; HOLD")
        elapsed_total, worker_total = next_elapsed, next_worker_total
        if terminal.get("registered_stop_reason") is not None or any(
            terminal.get(key) == REGISTERED_RATE_STOP
            for key in ("stop_reason", "post_slowdown_stop_reason")
        ):
            raise ValueError("registered D4096 stopping rule already fired; HOLD")
        if (
            terminal.get("schema") != "bl-labels-f0-mac-driver/v1"
            or terminal.get("status") != ("COMPLETE" if complete else "PREEMPTED")
            or (complete and terminal.get("pending_parents") != [])
            or terminal.get("invalid_reason") is not None
            or list(output.glob("INVALID*.json*"))
            or terminal.get("namespace") != control["namespace"]
            or terminal.get("workers") != workers
            or terminal.get("owner_w12_ok") is not (workers == 12)
            or terminal.get("collector_source_commit") != OLD_PIN
            or terminal.get("operational_source_commit") != start["operational_source_commit"]
            or terminal.get("w12_calibration_source_commit") != OLD_PIN
            or terminal.get("w12_calibration_receipt_sha256")
            != start.get("w12_calibration_receipt_sha256")
            or terminal.get("output") != str(output.resolve())
            or terminal.get("state_dir") != str(state.resolve())
            or terminal.get("start_sha256") != _verify(path)
            or terminal.get("start_receipt") != path.name
            or terminal.get("attempts_ledger") != names
            or terminal.get("prior_elapsed_seconds") != prior_elapsed
            or terminal.get("prior_worker_seconds_denominator") != prior_worker_total
        ):
            raise ValueError("non-preempted, INVALID or mismatched D4096 terminal chain; HOLD")
        sealed = path.with_name(path.name.replace(".start.json", ".rate-sealed.json"))
        if terminal.get("rate_sealed_path") != str(sealed.resolve()):
            raise ValueError("D4096 sealed rate path differs; HOLD")
        _verify(sealed)
        attempts.append(output)
    if terminals != set(names):
        raise ValueError("orphan D4096 terminal receipt; HOLD")
    if set(pack.glob("d4096-output-mac-*")) != set(attempts):
        raise ValueError("uncharged or foreign D4096 output directory; HOLD")
    if elapsed_total > CAP or (starts and not final and elapsed_total >= CAP):
        raise ValueError("prior D4096 attempts exhausted the frozen wall cap; HOLD")
    return attempts, elapsed_total


def verify_union(ledger: Path, control: dict[str, Any], attempts: list[Path]) -> None:
    """Verify a prepared continuation union without opening any payload."""
    value = _read(ledger)
    if value.get("attempt_exports") != [str(path.resolve()) for path in attempts]:
        raise ValueError("prepared D4096 union attempt set differs; HOLD")
    if value.get("units") != mac_gates.ledger_units(attempts):
        raise ValueError("prepared D4096 union units differ; HOLD")
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    mac_driver._verify_resume_metadata(ledger, control["namespace"], control["allocation"], names)


def initial_ledger(pack: Path, control: dict[str, Any]) -> Path:
    """Return the immutable empty pack ledger for the first attempted life."""
    ledger = pack / "d4096-verified-ledger.json"
    value = _read(ledger)
    if value.get("units") != [] or value.get("allocation") != control["allocation"]:
        raise ValueError("D4096 initial verified ledger differs; HOLD")
    load_ledger(ledger, control["namespace"], control["allocation"])
    return ledger


def publish_ledger(pack: Path, attempts: list[Path], out: Path) -> dict[str, Any]:
    """Create an internal continuation union from exactly all verified attempts."""
    control = original_control(pack)
    expected, _ = attempt_chain(pack, control)
    if [path.resolve() for path in attempts] != [path.resolve() for path in expected]:
        raise ValueError("omitted, duplicate or foreign D4096 attempts; HOLD")
    return mac_gates.publish_ledger(pack, attempts, out, control=control, sealed_resume=True)


def _require_delegation_funding(delegation_dir: Path | None, funding_receipt: Path | None) -> None:
    if delegation_dir is None:
        return
    # The pre-data L4 path is authorized only after funding is PASS.  The
    # validator intentionally owns the receipt schema and create-only lookup.
    d4096_sealed.require_funding_receipt(funding_receipt)


def launch(
    pack: Path,
    speed: Path,
    calibration: Path,
    old_root: Path,
    f4_control: Path | None,
    f4_inputs: Path | None,
    b8: Path,
    frp3: Path,
    *,
    workers: int = 6,
    owner_w12_ok: bool = False,
    w12_calibration_receipt: Path | None = None,
    sealed_readiness: Path,
    projection_receipt: Path,
    delegation_dir: Path | None = None,
    funding_receipt: Path | None = None,
    check: bool = False,
) -> None:
    """Revalidate the sealed admission, then check or execute one fresh attempt."""
    if pack.resolve() != PACK.resolve():
        raise ValueError("D4096 resume requires canonical OLD equiv-02 package; HOLD")
    if workers not in WIDTHS:
        raise ValueError("D4096 workers must be W6 or W12; HOLD")
    d4096_sealed.validate_readiness(sealed_readiness)
    d4096_sealed.validate_projection(projection_receipt, sealed_readiness)
    d4096_sealed.require_not_discarded(pack / "d4096-not-run-by-rule.json")
    _require_delegation_funding(delegation_dir, funding_receipt)
    control_path = pack / "d4096-control.json"
    d1024_w12._prepare_science_root(control_path, "d4096", sealed_path=True)
    control, selected, _flags, old_fallback = d1024_w12._validate_inputs(
        "d4096",
        control_path,
        speed,
        calibration,
        b8,
        frp3,
        old_root,
        f4_control,
        f4_inputs,
        sealed_readiness=sealed_readiness,
        projection_receipt=projection_receipt,
        owner_w12_ok=owner_w12_ok if workers == 12 else False,
    )
    if selected != workers or not old_fallback:
        raise ValueError("D4096 requested width differs from sealed OLD admission; HOLD")
    if workers == 12 and w12_calibration_receipt != calibration:
        raise ValueError("D4096 W12 must use its bound calibration receipt; HOLD")
    if workers == 6 and (owner_w12_ok or w12_calibration_receipt is not None):
        raise ValueError("D4096 W6 cannot carry W12 authorization; HOLD")
    from .w12_runtime import resolve_resume_workers

    effective_workers, width_reasons = resolve_resume_workers(workers, pack / "d4096-state-mac-01")
    if effective_workers not in WIDTHS:
        raise ValueError("D4096 runtime width policy returned an unsupported width; HOLD")
    if effective_workers != workers:
        workers, owner_w12_ok, w12_calibration_receipt = effective_workers, False, None
    admitted_path = pack / "d4096-admitted.json"
    if admitted_path.exists():
        admitted = original_control(pack)
    elif check:
        # A dry-run must not manufacture an admission receipt.  D4096's sealed
        # path does not consume the historical F4 readiness inputs, and the
        # plan check below needs only its frozen allocation and empty ledger.
        admitted = control
    else:
        admitted = mac_gates.admit_future(
            control_path,
            speed_equivalence=speed,
            calibration=calibration,
            out=admitted_path,
            f4_control=f4_control,
            f4_inputs=f4_inputs,
            old_factory_root=old_root,
            sealed_readiness=sealed_readiness,
            projection_receipt=projection_receipt,
        )
    attempts, elapsed = attempt_chain(pack, admitted)
    ledger = (
        initial_ledger(pack, admitted)
        if not attempts
        else pack / f"d4096-resume-{len(attempts)+1:02d}/verified-ledger.json"
    )
    if attempts:
        if ledger.exists():
            verify_union(ledger, admitted, attempts)
        else:
            publish_ledger(pack, attempts, ledger)
    _, _, pending = mac_driver._prepare_jobs(admitted, ledger)
    if not pending:
        raise ValueError("D4096 has no missing parents; HOLD")
    output = pack / f"d4096-output-mac-{len(attempts)+1:02d}"
    if output.exists() and any(output.iterdir()):
        raise ValueError("D4096 fresh attempt output already contains evidence; HOLD")
    command = [
        sys.executable,
        "-m",
        "research.bl_labels.mac_driver",
        "--control",
        str(admitted_path),
        "--ledger",
        str(ledger),
        "--b8",
        str(b8),
        "--frp3",
        str(frp3),
        "--output",
        str(output),
        "--state-dir",
        str(pack / "d4096-state-mac-01"),
        "--workers",
        str(workers),
        "--deadline-seconds",
        str(CAP),
    ]
    if workers == 12:
        command.extend(["--owner-w12-ok", "--w12-calibration-receipt", str(calibration)])
    if delegation_dir is not None:
        command.extend(["--delegation-dir", str(delegation_dir)])
        command.extend(["--funding-receipt", str(funding_receipt)])
    print(
        json.dumps(
            {
                "event": "d4096_resume",
                "workers": workers,
                "width_reasons": width_reasons,
                "prior_attempts": len(attempts),
                "prior_elapsed_seconds": elapsed,
                "deadline_seconds": CAP,
                "remaining_cap_seconds": CAP - elapsed,
                "pending_parents": len(pending),
                "output": command[command.index("--output") + 1],
                "ledger": str(ledger),
            }
        ),
        flush=True,
    )
    if check and not admitted_path.exists():
        return
    subprocess.run([*command, "--check"], check=True)
    if not check:
        os.execv(sys.executable, command)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, default=PACK)
    parser.add_argument("--speed-equivalence-receipt", type=Path, required=True)
    parser.add_argument("--calibration-receipt", type=Path, required=True)
    parser.add_argument("--old-factory-root", type=Path, required=True)
    parser.add_argument("--f4-control", type=Path)
    parser.add_argument("--f4-inputs", type=Path)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--sealed-readiness", type=Path, required=True)
    parser.add_argument("--projection-receipt", type=Path, required=True)
    parser.add_argument("--workers", choices=(6, 12), type=int, default=6)
    parser.add_argument("--owner-w12-ok", action="store_true")
    parser.add_argument("--w12-calibration-receipt", type=Path)
    parser.add_argument("--delegation-dir", type=Path)
    parser.add_argument("--funding-receipt", type=Path)
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
        workers=args.workers,
        owner_w12_ok=args.owner_w12_ok,
        w12_calibration_receipt=args.w12_calibration_receipt,
        sealed_readiness=args.sealed_readiness,
        projection_receipt=args.projection_receipt,
        delegation_dir=args.delegation_dir,
        funding_receipt=args.funding_receipt,
        check=args.check,
    )


if __name__ == "__main__":
    main()
