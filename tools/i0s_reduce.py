"""I0S Mac reducer (JUDGMENT_I0R_PATH_20261024 §2 decision rule, §3 item 3).

Two modes:

* ``integrity`` -- allowed at any time. Path, size, SHA256 and sidecar checks of each
  attempt's exported files against the broker receipt. It never opens a unit payload
  and never computes a rate (I-arm rate blinding).
* ``final`` -- once, create-only, only after every presented attempt is terminal,
  released and settled in the sprint register. It rebuilds the main ledger from the
  scored attempt's exports with the frozen ``integration_units.build_ledger`` (the
  producer publishes no ``verified-ledger.json`` on a TimeoutError), validates cells
  with the frozen ``integration_analyze`` validators and applies the ruling's
  precedence INVALID > STOP_SPEED > INCOMPLETE > PASS_SPEED.

Gating is the three E1024 cells only: per roster, the I-arm complete-tick median
must be <= 0.512 s AND every I-arm continuous block >= 2,000 complete world-steps/s.
E256/E64/E6 are descriptive (``DESCRIPTIVE_NOT_RUN`` when absent) and never decide.
Only the first attempt that reached a non-mechanical terminal status is scored.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Callable

from tools import i0s_attempt as attempt_module
from tools import small_bench_operator as operator
from tools import small_bench_register as register

PIN = attempt_module.PIN
GATING = ("cell-frozen-e1024", "cell-mixed-e1024", "cell-scripted-e1024")
DESCRIPTIVE = tuple(
    f"cell-{mix}-e{world}" for world in (256, 64, 6) for mix in ("frozen", "mixed", "scripted")
)
MEDIAN_BAR_SECONDS = 0.512
BLOCK_BAR_STEPS_PER_SECOND = 2000.0


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------- integrity


def integrity(outputs: list[Path]) -> dict[str, Any]:
    """Path/size/hash/sidecar checks only; outcome-blind and rate-blind."""
    rows, errors = [], []
    for output in outputs:
        receipt_path = Path(output) / "receipt.json"
        if not receipt_path.is_file():
            errors.append(f"receipt absent: {receipt_path}")
            continue
        receipt = json.loads(receipt_path.read_text())
        listed = receipt.get("exports") or []
        expected = {row.get("path"): row for row in listed if isinstance(row, dict)}
        root = Path(output) / "exports"
        actual = {}
        if root.is_dir():
            for path in sorted(root.rglob("*")):
                if path.is_symlink():
                    errors.append(f"symlink export: {path}")
                    continue
                if not path.is_file() or path.name.endswith((".sha256", ".complete")):
                    continue
                name = path.relative_to(root).as_posix()
                actual[name] = {"path": name, "sha256": sha(path), "size": path.stat().st_size}
                side, done = Path(f"{path}.sha256"), Path(f"{path}.complete")
                if side.is_file() and side.read_text().split()[:1] != [actual[name]["sha256"]]:
                    errors.append(f"sidecar hash mismatch: {name}")
                if side.is_file() != done.is_file():
                    errors.append(f"incomplete sidecar pair: {name}")
                elif done.is_file() and done.read_text() != "complete\n":
                    errors.append(f"malformed completion marker: {name}")
        if len(expected) != len(listed):
            errors.append(f"duplicate receipt export rows: {receipt_path}")
        for name in sorted(set(expected) | set(actual)):
            if expected.get(name) != actual.get(name):
                errors.append(f"export differs from receipt inventory: {name}")
        rows.append(
            {
                "output": str(output),
                "receipt": {"path": str(receipt_path), "sha256": sha(receipt_path)},
                "files": len(actual),
                "unit_files": sum(1 for name in actual if name.startswith("integration-units/")),
            }
        )
    return {
        "schema": "i0s-export-integrity/v1",
        "status": "INTEGRITY_PASS" if not errors else "INTEGRITY_FAIL",
        "errors": errors,
        "outputs": rows,
        "rates_computed": False,
        "unit_payloads_opened": False,
    }


# ----------------------------------------------------------------------- scoring


def frozen_validators() -> dict[str, Callable[..., Any]]:
    """The qualified checkout's own validators; refuse any other source."""
    from research.gpu_sim_spike import integration_analyze, integration_units

    for module in (integration_analyze, integration_units):
        if not Path(module.__file__).resolve().is_relative_to(attempt_module.RUN):
            raise ValueError(f"{module.__name__} did not resolve from the qualified checkout")
    return {
        "valid_cell": integration_analyze.valid_cell,
        "native_valid": integration_analyze.native_valid,
        "units": integration_units,
    }


def gate(row: dict[str, Any]) -> dict[str, Any]:
    """Ruling bar for one verified E1024 cell (computed only in final mode)."""
    median = statistics.median(row["arms"]["I"]["plain_seconds"])
    blocks = [
        item["complete_world_steps_per_second"]
        for item in row["continuous"]["arms"]["I"]["repeats"]
    ]
    median_ok = median <= MEDIAN_BAR_SECONDS
    blocks_ok = bool(blocks) and all(rate >= BLOCK_BAR_STEPS_PER_SECOND for rate in blocks)
    return {
        "i_arm_complete_tick_median_seconds": median,
        "i_arm_continuous_block_steps_per_second": blocks,
        "median_bar_pass": median_ok,
        "blocks_bar_pass": blocks_ok,
        "pass": median_ok and blocks_ok,
    }


def describe(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "i_arm_complete_tick_median_seconds": statistics.median(row["arms"]["I"]["plain_seconds"])
    }


def evaluate_exports(
    exports: Path, work: Path, validators: dict[str, Callable[..., Any]]
) -> dict[str, Any]:
    """Rebuild the ledger from one attempt's exports and classify every cell."""
    units = validators["units"]
    invalid, incomplete = [], []
    ns, cells = units.namespace(PIN), units.allocation()
    failure = exports / "integration-failure.json"
    if failure.exists():
        try:
            units._verify(failure)
            if json.loads(failure.read_text()).get("status") == "INVALID":
                invalid.append("producer published integration-failure.json INVALID")
        except (OSError, ValueError) as exc:
            invalid.append(f"failure marker unverifiable: {exc}")
    native_ok = False
    native = exports / "native-parity.json"
    if native.exists():
        try:
            units._verify(native)
            report = json.loads(native.read_text())
            if report.get("status") == "INVALID":
                invalid.append("native parity INVALID")
            native_ok = bool(validators["native_valid"](report, PIN))
            if not native_ok:
                incomplete.append("native report lacks executed A100 parity proof at this pin")
        except (OSError, ValueError) as exc:
            invalid.append(f"native report unverifiable: {exc}")
    else:
        incomplete.append("native-parity.json not exported")
    register_ok = False
    discrepancy = exports / "discrepancy-register.json"
    if discrepancy.exists():
        try:
            units._verify(discrepancy)
            body = json.loads(discrepancy.read_text())
            register_ok = (
                body.get("commit") == PIN
                and body.get("cpu_component_register", {}).get("status") == "DIAGNOSTIC"
            )
            if not register_ok:
                incomplete.append("discrepancy register classification missing")
        except (OSError, ValueError) as exc:
            invalid.append(f"discrepancy register unverifiable: {exc}")
    else:
        incomplete.append("discrepancy-register.json not exported")
    rebuilt = work / "rebuilt-ledger" / "verified-ledger.json"
    rows: dict[str, dict[str, Any]] = {}
    try:
        built = units.build_ledger(exports, rebuilt, commit=PIN)
        rows = units.load_ledger(rebuilt, ns, cells)
        exported = exports / "verified-ledger.json"
        if exported.exists():
            producer = units.load_ledger(exported, ns, cells, exports=exports)
            if set(producer) != set(rows):
                invalid.append("producer ledger and Mac-rebuilt ledger disagree")
        for unit, row in rows.items():
            trace = row.get("trace", {})
            if trace.get("status") == "PASS":
                relative = Path(trace.get("path", ""))
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError("foreign trace path")
                target = exports / relative
                if units._verify(target) != trace.get("sha256") or (
                    target.stat().st_size != trace.get("size")
                ):
                    raise ValueError(f"trace bytes differ: {unit}")
        ledger = {"path": str(rebuilt), "sha256": sha(rebuilt), "units": len(built["units"])}
    except (OSError, ValueError, KeyError, TypeError) as exc:
        invalid.append(f"export/ledger mismatch: {exc}")
        ledger = None
    proof_ok = native_ok and register_ok
    statuses: dict[str, dict[str, Any]] = {}
    for unit in cells:
        row = rows.get(unit)
        if row is None:
            statuses[unit] = {"status": "MISSING" if unit in GATING else "DESCRIPTIVE_NOT_RUN"}
        elif not validators["valid_cell"](unit, row):
            invalid.append(f"published cell fails frozen validation: {unit}")
            statuses[unit] = {"status": "MALFORMED"}
        elif not proof_ok:
            statuses[unit] = {"status": "UNVERIFIED_PROOF_INCOMPLETE"}
        else:
            statuses[unit] = {"status": "VERIFIED"}
    return {
        "invalid": invalid,
        "incomplete": incomplete,
        "proof_ok": proof_ok,
        "rebuilt_ledger": ledger,
        "rows": rows,
        "cells": statuses,
    }


def verdict(cells: dict[str, dict[str, Any]], invalid: list[str]) -> dict[str, Any]:
    """Ruling §2 precedence: INVALID > STOP_SPEED > INCOMPLETE > PASS_SPEED."""
    verified = [unit for unit in GATING if cells[unit]["status"] == "VERIFIED"]
    failing = [unit for unit in verified if not cells[unit]["pass"]]
    if invalid:
        name = "INVALID"
    elif failing:
        name = "STOP_SPEED"
    elif len(verified) < len(GATING):
        name = "INCOMPLETE"
    else:
        name = "PASS_SPEED"
    label = f"STOP_SPEED (gating verified {len(verified)}/3)" if name == "STOP_SPEED" else name
    return {
        "verdict": name,
        "label": label,
        "gating_verified": len(verified),
        "gating_failing": failing,
    }


def select_scored(attempts: dict[int, list[Path]]) -> dict[str, Any]:
    """Only the first attempt with a non-mechanical terminal status is scored."""
    if set(attempts) - {1, 2} or 1 not in attempts:
        raise ValueError("I0S reduces attempt 1 and at most one retry")
    log = {}
    scored = None
    for number in sorted(attempts):
        classification = attempt_module.classify_attempt(attempts[number])
        if not classification["terminal"] or not classification.get("released"):
            raise ValueError(f"attempt {number} is not terminal and released")
        if scored is not None:
            log[number] = {"scored": False, "reason": "a prior attempt was already scored"}
        elif classification["mechanical_zero_unit"]:
            log[number] = {"scored": False, "reason": "mechanical zero-unit failure"}
        else:
            scored = number
            log[number] = {"scored": True}
        log[number]["classification"] = classification
    if attempts.get(2) and log[1]["scored"]:
        log[2]["reason"] = "retry outside the ruling predicate; never scored"
    return {"scored": scored, "attempts": log}


def require_settled(rows: list[dict[str, Any]], attempts: dict[int, list[Path]]) -> None:
    """Rate blinding: rates only after every presented attempt is settled."""
    current = register.state(rows)
    for number in attempts:
        entry = current["reservations"].get(attempt_module.reservation_id(number))
        if entry is None or not entry["settled"]:
            raise ValueError(f"attempt {number} is not settled; rates stay blinded")


def final(
    attempts: dict[int, list[Path]],
    rows: list[dict[str, Any]],
    work: Path,
    validators: dict[str, Callable[..., Any]] | None = None,
) -> dict[str, Any]:
    require_settled(rows, attempts)
    validators = validators or frozen_validators()
    checks = {number: integrity(outputs) for number, outputs in attempts.items()}
    invalid = [
        f"attempt {number}: {error}"
        for number, check in checks.items()
        for error in check["errors"]
    ]
    selection = select_scored(attempts)
    cells = {unit: {"status": "MISSING"} for unit in GATING}
    cells.update({unit: {"status": "DESCRIPTIVE_NOT_RUN"} for unit in DESCRIPTIVE})
    incomplete: list[str] = []
    ledger = None
    if selection["scored"] is None:
        incomplete.append("no attempt reached a non-mechanical terminal status")
    else:
        output = attempts[selection["scored"]][-1]
        result = evaluate_exports(output / "exports", work, validators)
        invalid += result["invalid"]
        incomplete += result["incomplete"]
        ledger = result["rebuilt_ledger"]
        cells = result["cells"]
        for unit, status in cells.items():
            if status["status"] != "VERIFIED":
                continue
            row = result["rows"][unit]
            status.update(gate(row) if unit in GATING else describe(row))
            if unit in DESCRIPTIVE:
                status["status"] = "VERIFIED_DESCRIPTIVE"
    decision = verdict(cells, invalid)
    return {
        "schema": "i0s-reduction/v1",
        "ruling_sha256": attempt_module.RULING_SHA,
        "runtime_pin": PIN,
        **decision,
        "scored_attempt": selection["scored"],
        "attempts": selection["attempts"],
        "integrity": checks,
        "invalid_reasons": invalid,
        "incomplete_reasons": incomplete,
        "rebuilt_ledger": ledger,
        "cells": cells,
        "bars": {
            "i_arm_complete_tick_median_seconds_max": MEDIAN_BAR_SECONDS,
            "i_arm_continuous_block_steps_per_second_min": BLOCK_BAR_STEPS_PER_SECOND,
            "gating": list(GATING),
        },
        "engineering_only": "no training, label, strict-gate or serving consequence",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("integrity", "final"))
    parser.add_argument("--packet", type=Path, default=attempt_module.PACKET)
    parser.add_argument("--register", type=Path, default=attempt_module.REGISTER)
    parser.add_argument("--output", type=Path, help="create-only record (+ sidecars)")
    args = parser.parse_args()
    attempts = {
        number: attempt_module.attempt_outputs(number, args.packet)
        for number in (1, 2)
        if attempt_module.attempt_dir(number, args.packet).exists()
    }
    attempts = {number: outputs for number, outputs in attempts.items() if outputs}
    if args.mode == "integrity":
        result = {number: integrity(outputs) for number, outputs in attempts.items()}
    else:
        if args.output is None:
            parser.error("final reduction requires --output (create-only, runs once)")
        existing = sorted(args.packet.glob("REDUCTION*.json"))
        if args.output.exists() or existing:
            raise SystemExit(f"reduction already published; the reducer runs once: {existing}")
        work = args.output.parent / f"{args.output.stem}-work"
        if work.exists():
            raise SystemExit(f"reduction work directory already exists: {work}")
        result = final(attempts, register.read(args.register), work)
    if args.output:
        operator.publish_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
