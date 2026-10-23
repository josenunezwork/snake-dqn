"""Outcome-blind continuation of the original, immutable D256 Mac series.

Only operational receipt fields and export wrapper identities are inspected.
The original admission and scientific import closure remain at OLD_PIN; the
current coordinator supplies cumulative wall/worker accounting and W6/W12 seams.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

from . import d256_w12, frozen_factory, mac_driver, mac_gates, w12_admission, w12_calibration
from .exports import _verify, load_ledger

OLD_PIN = frozen_factory.OLD_PIN
OLD_DRIVER_SHA256 = "0694dcfb2dffd5229acb51b3b3164a7949a36a140e64023c99fc719cb0bfefc5"
CAP = 57600
REGISTERED_RATE_STOP = "sealed rolling six-hour rescue rate below 50 percent Canary-B baseline"
PACK = w12_calibration.PACK
ROOT = Path(__file__).resolve().parents[2]
FROZEN_ROOT = ROOT.parent / "snake-dqn-d256-resume-science.noindex"


def canonical_sha(value: object) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def original_control(pack: Path) -> dict:
    """Bind to the existing admission, never create or re-admit it."""
    value = mac_gates._read(pack / "d256-admitted.json")
    original = mac_gates._read(pack / "d256-control.json")
    gate = value.get("mac_gate", {})
    if (
        value.get("source_commit") != OLD_PIN
        or gate.get("stage") != "d256"
        or gate.get("status") != "APPROVED"
        or gate.get("package_root") != str(pack.resolve())
        or gate.get("original_control_sha256") != _verify(pack / "d256-control.json")
        or value.get("namespace") != original.get("namespace")
        or value.get("allocation") != original.get("allocation")
    ):
        raise ValueError("original D256 admission identity differs; HOLD")
    return value


def verify_frozen_admission(control: dict, root: Path) -> None:
    """Use an isolated frozen process to recheck all original F1/F0/F4 gates."""
    pack = Path(control["mac_gate"]["package_root"])
    if pack.resolve() != PACK.resolve():
        raise ValueError("D256 resume requires canonical mac-pack-02; HOLD")
    if original_control(pack) != control:
        raise ValueError("original D256 control changed; HOLD")
    frozen_factory._validate(root, OLD_PIN)
    # No aggregator or new look: require_admission only rechecks the sealed look.
    subprocess.run(
        [
            sys.executable,
            "-c",
            "import json,sys; from research.bl_labels.mac_gates import require_admission; "
            "require_admission(json.load(sys.stdin))",
        ],
        input=json.dumps(control),
        text=True,
        cwd=root,
        env={**os.environ, "PYTHONPATH": str(root), "PYTHONDONTWRITEBYTECODE": "1"},
        check=True,
    )


def verify_worker_charge(start: dict, terminal: dict, elapsed: float) -> None:
    """Accept the old driver's separate clock reads only for a legacy start.

    The old formula samples time independently twice. Its recorded charge is
    retained; the canonical continuation denominator is sum(width * elapsed).
    One millisecond is a fail-closed compatibility bound, not a cap allowance.
    """
    expected = start["workers"] * elapsed
    actual = terminal.get("worker_seconds_denominator")
    legacy = legacy_start(start)
    if (
        isinstance(actual, bool)
        or not isinstance(actual, (int, float))
        or not math.isfinite(actual)
    ):
        raise ValueError("prior D256 worker-second accounting differs; HOLD")
    if actual != expected and not (legacy and abs(actual - expected) <= start["workers"] * 0.001):
        raise ValueError("prior D256 worker-second accounting differs; HOLD")


def legacy_start(start: dict) -> bool:
    """Recognize receipts from the exact original driver, not arbitrary old shapes."""
    return (
        "owner_w12_ok" not in start
        and start.get("source_commit") == OLD_PIN
        and start.get("workers") == 6
        and start.get("platform_attestation", {}).get("driver_sha256") == OLD_DRIVER_SHA256
    )


def attempt_chain(pack: Path, control: dict) -> tuple[list[Path], float]:
    """Verify every durable start/terminal/export, including zero-unit attempts."""
    state = pack / "d256-state-mac-01"
    starts = sorted(state.glob("attempt-*.start.json"), key=lambda p: p.stat().st_mtime_ns)
    if not starts:
        raise ValueError("original D256 start receipt missing; HOLD")
    terminal_names = []
    attempts = []
    elapsed_total = 0.0
    worker_total = 0.0
    for index, path in enumerate(starts, 1):
        start = mac_gates._read(path)
        terminal_path = path.with_name(path.name.replace(".start.json", ".terminal.json"))
        if not terminal_path.exists():
            raise ValueError("prior D256 attempt lacks a verified terminal receipt; HOLD")
        terminal = mac_gates._read(terminal_path)
        if (
            terminal.get("stop_reason") == REGISTERED_RATE_STOP
            or terminal.get("registered_stop_reason") == REGISTERED_RATE_STOP
        ):
            raise ValueError("registered D256 stopping rule already fired; HOLD")
        output = pack / f"d256-output-mac-{index:02d}"
        if (
            start.get("namespace") != control["namespace"]
            or start.get("source_commit") != OLD_PIN
            or start.get("allocation_sha256") != canonical_sha(control["allocation"])
            or start.get("control_sha256") != canonical_sha(control)
            or start.get("deadline_seconds") != CAP
            or start.get("original_wall_cap_seconds") != CAP
            or start.get("state_dir") != str(state.resolve())
            or start.get("output") != str(output.resolve())
            or type(start.get("workers")) is not int
            or start["workers"] not in (6, 12)
        ):
            raise ValueError("prior D256 start identity differs; HOLD")
        if index == 1 and start["workers"] != 6:
            raise ValueError("original D256 must start at W6; HOLD")
        if start["workers"] == 12 and not (
            start.get("owner_w12_ok") is True
            and isinstance(start.get("w12_calibration_receipt_sha256"), str)
            and terminal.get("w12_calibration_receipt_sha256")
            == start["w12_calibration_receipt_sha256"]
        ):
            raise ValueError("prior W12 authorization differs; HOLD")
        exported = mac_gates._read(output / "runtime-result.json")
        # The CLI adds only its lease receipt after the driver seals the terminal.
        if {k: v for k, v in exported.items() if k != "mac_lease"} != terminal:
            raise ValueError("D256 terminal/export receipt differs; HOLD")
        terminal_names.append(terminal_path.name)
        if (
            terminal.get("status") not in {"PREEMPTED", "COMPLETE"}
            or terminal.get("invalid_reason") is not None
            or list(output.glob("INVALID*.json*"))
            or terminal.get("namespace") != control["namespace"]
            or terminal.get("workers") != start["workers"]
            or terminal.get("output") != str(output.resolve())
            or terminal.get("state_dir") != str(state.resolve())
            or terminal.get("start_sha256") != _verify(path)
            or terminal.get("start_receipt") != path.name
            or terminal.get("attempts_ledger") != terminal_names
            or terminal.get("prior_elapsed_seconds") != elapsed_total
        ):
            raise ValueError("INVALID or mismatched D256 terminal chain; HOLD")
        elapsed = terminal.get("elapsed_seconds")
        if (
            isinstance(elapsed, bool)
            or not isinstance(elapsed, (int, float))
            or not math.isfinite(elapsed)
            or elapsed <= 0
        ):
            raise ValueError("invalid prior D256 wall charge; HOLD")
        verify_worker_charge(start, terminal, elapsed)
        if "owner_w12_ok" in start and (
            terminal.get("prior_worker_seconds_denominator") != worker_total
            or terminal.get("cumulative_worker_seconds_denominator")
            != worker_total + start["workers"] * elapsed
        ):
            raise ValueError("cumulative D256 worker-second accounting differs; HOLD")
        elapsed_total += elapsed
        worker_total += start["workers"] * elapsed
        cumulative = terminal.get("cumulative_elapsed_seconds")
        if cumulative != elapsed_total and not (
            legacy_start(start)
            and type(cumulative) in (int, float)
            and math.isfinite(cumulative)
            and abs(cumulative - elapsed_total) <= 0.001
        ):
            raise ValueError("cumulative D256 wall accounting differs; HOLD")
        sealed = terminal.get("rate_sealed_path")
        if sealed != str(
            path.with_name(path.name.replace(".start.json", ".rate-sealed.json")).resolve()
        ):
            raise ValueError("D256 sealed rate path differs; HOLD")
        _verify(Path(sealed))  # Hash only; never inspect the sealed rate outcomes.
        attempts.append(output)
    if {p.name for p in state.glob("attempt-*.terminal.json")} != set(terminal_names):
        raise ValueError("orphan D256 terminal receipt; HOLD")
    if set(pack.glob("d256-output-mac-*")) != set(attempts):
        raise ValueError("uncharged or foreign D256 output directory; HOLD")
    if elapsed_total >= CAP:
        raise ValueError("prior D256 attempts exhausted the frozen wall cap; HOLD")
    return attempts, elapsed_total


def verify_resume_request(
    control: dict, root: Path, ledger: Path, output: Path, state: Path | None, cap: float | None
) -> None:
    """Validate the explicit driver route before creating any output or start."""
    verify_frozen_admission(control, root)
    pack = Path(control["mac_gate"]["package_root"])
    attempts, _ = attempt_chain(pack, control)
    if (
        state is None
        or state.resolve() != (pack / "d256-state-mac-01").resolve()
        or cap != CAP
        or output.resolve() != (pack / f"d256-output-mac-{len(attempts) + 1:02d}").resolve()
    ):
        raise ValueError("original D256 resume cap/state/output differs; HOLD")
    verify_union(ledger, control, attempts)


def verify_union(ledger: Path, control: dict, attempts: list[Path]) -> None:
    """Recheck a reusable union after a check or lease refusal."""
    value = mac_gates._read(ledger)
    if value.get("attempt_exports") != [str(p.resolve()) for p in attempts]:
        raise ValueError("prepared D256 union attempt set differs; HOLD")
    if value.get("units") != mac_gates.ledger_units(attempts):
        raise ValueError("prepared D256 union units differ; HOLD")
    names = load_ledger(ledger, control["namespace"], control["allocation"])
    mac_driver._verify_resume_metadata(ledger, control["namespace"], control["allocation"], names)


def publish_ledger(pack: Path, attempts: list[Path], out: Path) -> dict:
    """Publish a portable union without invoking any scientific aggregator."""
    control = original_control(pack)
    expected, _ = attempt_chain(pack, control)
    if [p.resolve() for p in attempts] != [p.resolve() for p in expected]:
        raise ValueError("omitted, duplicate or foreign D256 attempts; HOLD")
    return mac_gates.publish_ledger(pack, attempts, out, control=control)


def select_workers(receipt: Path, owner: Path, b8: Path, frp3: Path) -> tuple[int, list[str]]:
    """Select W12 only from an original PASS or the ruled admission record."""
    if not any(Path(str(receipt) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        return 6, []
    record = mac_gates._read(receipt)
    if record.get("schema") == w12_admission.SCHEMA:
        w12_admission.verify(receipt, expected_code_sha=OLD_PIN)
        value = mac_gates._read(w12_admission.CALIBRATION_RECEIPT)
        workers = 12
        flags = ["--owner-w12-ok", "--w12-calibration-receipt", str(receipt.resolve())]
    else:
        # Preserve a genuine v1 PASS route.  The legacy FAIL receipt remains a
        # W6 choice and cannot be silently reinterpreted here.
        workers, flags = d256_w12.select_workers(receipt, owner)
        value = record
    if value.get("b8_sha256") != w12_calibration.sha(b8) or value.get(
        "frp3_sha256"
    ) != w12_calibration.sha(frp3):
        raise ValueError("D256 checkpoints differ from calibrated inputs; HOLD")
    return workers, flags


def require_w12(receipt: Path | None, b8: Path, frp3: Path) -> None:
    """The direct driver route must also enforce the launcher's full sealed rule."""
    if receipt is None or select_workers(receipt, w12_calibration.OWNER_FILE, b8, frp3)[0] != 12:
        raise ValueError("D256 W12 requires a v1 PASS or ruled sealed admission; HOLD")


def ensure_frozen_tree(root: Path) -> None:
    """Create a separate frozen reference, never reuse the original run tree."""
    if not root.exists():
        subprocess.run(
            ["git", "worktree", "add", "--detach", str(root), OLD_PIN], cwd=ROOT, check=True
        )
    frozen_factory._validate(root, OLD_PIN)


def launch(
    pack: Path,
    frozen_root: Path,
    receipt: Path,
    owner: Path,
    b8: Path,
    frp3: Path,
    *,
    check: bool = False,
) -> None:
    if pack.resolve() != PACK.resolve():
        raise ValueError("D256 resume requires canonical mac-pack-02; HOLD")
    control = original_control(pack)
    os.environ.update(mac_driver.THREAD_ENV)
    # A future-stage operational wrapper must never bleed into the historical
    # D256 science closure inherited by this process.
    os.environ.pop("F0_OPERATIONAL_COMMIT", None)
    os.environ.pop("F0_SCIENCE_ROOT", None)
    os.environ.update(F0_COMMIT=OLD_PIN, F1_COMMIT=OLD_PIN, SNAKE_DQN_DEVICE="cpu")
    attempts, elapsed = attempt_chain(pack, control)
    requested_workers, flags = select_workers(receipt, owner, b8, frp3)
    # The runtime policy reads only sealed terminal operational state.  It can
    # force the next continuation to W6 after a ruled W12 fallback; clear the
    # W12 authorization flags so accounting and admission width agree.
    from .w12_runtime import resolve_resume_workers

    workers, width_reasons = resolve_resume_workers(requested_workers, pack / "d256-state-mac-01")
    if workers != requested_workers:
        if workers != 6:
            raise ValueError("D256 runtime width policy returned an unsupported worker count")
        flags = []
    ensure_frozen_tree(frozen_root)
    verify_frozen_admission(control, frozen_root)
    if mac_driver._timing_armed(mac_driver.DEFAULT_TIMING_ARMED):
        raise ValueError("timing armed; HOLD")
    mac_driver._guard_snapshot(pack.parent)
    index = len(attempts) + 1
    output = pack / f"d256-output-mac-{index:02d}"
    ledger = pack / f"d256-resume-{index:02d}/verified-ledger.json"
    if ledger.exists():
        # A prior driver --check/lease refusal may leave this exact ledger.
        verify_union(ledger, control, attempts)
    else:
        publish_ledger(pack, attempts, ledger)
    command = [
        sys.executable,
        "-m",
        "research.bl_labels.mac_driver",
        "--control",
        str(pack / "d256-admitted.json"),
        "--ledger",
        str(ledger),
        "--b8",
        str(b8),
        "--frp3",
        str(frp3),
        "--output",
        str(output),
        "--state-dir",
        str(pack / "d256-state-mac-01"),
        "--workers",
        str(workers),
        "--deadline-seconds",
        str(CAP),
        "--d256-frozen-root",
        str(frozen_root),
        *flags,
    ]
    print(
        json.dumps(
            {
                "event": "d256_resume",
                "workers": workers,
                "width_reasons": width_reasons,
                "prior_elapsed_seconds": elapsed,
                "output": str(output),
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
    parser.add_argument("--frozen-root", type=Path, default=FROZEN_ROOT)
    parser.add_argument(
        "--calibration-receipt",
        type=Path,
        default=PACK.parent / "w12-admission-01/receipt.json",
    )
    parser.add_argument("--owner-approval", type=Path, default=w12_calibration.OWNER_FILE)
    parser.add_argument("--b8", type=Path, required=True)
    parser.add_argument("--frp3", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--ledger-out", type=Path)
    parser.add_argument(
        "--host-check",
        action="store_true",
        help="log the required host check before this operator-requested resume",
    )
    args = parser.parse_args()
    if args.ledger_out:
        control = original_control(args.pack)
        attempts, _ = attempt_chain(args.pack, control)
        publish_ledger(args.pack, attempts, args.ledger_out)
        print(json.dumps({"status": "PUBLISHED", "out": str(args.ledger_out)}))
    else:
        if args.host_check:
            from .slowdown_resume import record_host_check

            if args.pack.resolve() != PACK.resolve():
                raise ValueError("D256 host check requires canonical mac-pack-02; HOLD")
            original_control(args.pack)
            record_host_check(args.pack / "d256-state-mac-01")
        launch(
            args.pack,
            args.frozen_root,
            args.calibration_receipt,
            args.owner_approval,
            args.b8,
            args.frp3,
            check=args.check,
        )


if __name__ == "__main__":
    main()
