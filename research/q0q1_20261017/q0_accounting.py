#!/usr/bin/env python3
"""Create immutable, receipt-backed Q0/Q0b debit accounting.

The queue's terminal receipts include all worker time from claim handling through
cleanup.  This tool charges every Q0/Q0b claim from the requested UTC day.  A
claim without a terminal receipt is an error unless an explicit idle heartbeat or
operator control record proves a conservative end bound.  It never silently
turns an unknown attempt into zero cost.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

SCHEMA = "q0q1-observed-q0-accounting/v1"
PRICING_SCHEMA = "q0q1-observed-pricing/v1"
OVERHEAD_SCHEMA = "q0q1-cpu-eval-overhead/v1"
DEFAULT_DATE = "2026-10-18"
BASELINE_Q0_UNITS_CAP = 40.0
BASELINE_COMBINED_UNITS_CAP = 175.0
Q1_ORIGINAL_UNITS_CAP = 135.0
RESERVE_FRACTION = 0.20
BASELINE_RESERVE_UNITS = BASELINE_COMBINED_UNITS_CAP * RESERVE_FRACTION
DEFAULT_DRIVE_ROOT = (
    "/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/"
    "My Drive/t16_colab"
)
DEFAULT_OVERHEAD = Path(__file__).with_name("CPU_EVAL_OVERHEAD_MAC_20261018.json")


class AccountingError(ValueError):
    """Accounting cannot be completed from the immutable operational evidence."""


def _same_number(actual: Any, expected: float) -> bool:
    """Compare derived JSON numbers without admitting a material rounding change."""
    return (
        not isinstance(actual, bool)
        and isinstance(actual, (int, float))
        and math.isfinite(actual)
        and math.isclose(float(actual), expected, rel_tol=0.0, abs_tol=1e-9)
    )


def validate_a3_envelope(accounting: Mapping[str, Any]) -> dict[str, Any]:
    """Return the exact A3 budget derivation bound into observed accounting.

    Amendment A3 raises the combined cap by the *observed* Q0 overrun.  It does
    not recompute the reserve as 20 percent of that larger cap: doing so would
    consume part of Q1's original usable headroom.  The 35-unit reserve remains
    the 20 percent reservation of the original 175-unit envelope.
    """
    # The generator copies this pair into the hash-bound study plan.  Keep the
    # algebra validator usable for that derived plan as well as the original
    # accounting receipt; callers that consume raw accounting must separately
    # require its complete terminal-attempt status.
    schema = accounting.get("schema")
    if schema not in (None, SCHEMA, "q0q1-study-budget/v1"):
        raise AccountingError("A3 envelope requires observed Q0 accounting or study-plan schema")
    q0_units = _finite(accounting.get("q0_units"), "Q0 accounting q0_units", positive=True)
    envelope = accounting.get("a3_envelope")
    if not isinstance(envelope, Mapping):
        raise AccountingError("Q0 accounting lacks an A3 envelope")
    overrun = q0_units - BASELINE_Q0_UNITS_CAP
    expected = {
        "amendment": "A3",
        "baseline_q0_units_cap": BASELINE_Q0_UNITS_CAP,
        "observed_q0_units": q0_units,
        "q0_units_cap": q0_units,
        "q1_original_units_cap": Q1_ORIGINAL_UNITS_CAP,
        "baseline_combined_units_cap": BASELINE_COMBINED_UNITS_CAP,
        "reserve_fraction": RESERVE_FRACTION,
        "reserve_basis_combined_units": BASELINE_COMBINED_UNITS_CAP,
        "reserve_units": BASELINE_RESERVE_UNITS,
        "q1_dispatch_headroom_units": Q1_ORIGINAL_UNITS_CAP - BASELINE_RESERVE_UNITS,
        "q0_overrun_units": overrun,
        "combined_units_cap": BASELINE_COMBINED_UNITS_CAP + overrun,
        "combined_dispatch_limit_units": q0_units
        + (Q1_ORIGINAL_UNITS_CAP - BASELINE_RESERVE_UNITS),
    }
    if envelope.get("amendment") != "A3":
        raise AccountingError("Q0 accounting A3 envelope has the wrong amendment")
    for name, value in expected.items():
        if name == "amendment":
            continue
        if not _same_number(envelope.get(name), value):
            raise AccountingError(f"Q0 accounting A3 envelope has invalid {name}")
    if overrun <= 0:
        raise AccountingError("A3 envelope requires observed Q0 units above the 40-unit cap")
    if not isinstance(envelope.get("derivation"), str) or not envelope["derivation"]:
        raise AccountingError("Q0 accounting A3 envelope lacks its derivation")
    return dict(envelope)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise AccountingError(f"{label} is not readable JSON: {path}") from exc
    if not isinstance(value, dict):
        raise AccountingError(f"{label} must be a JSON object: {path}")
    return value


def _timestamp(value: Any, label: str) -> dt.datetime:
    if not isinstance(value, str):
        raise AccountingError(f"{label} must be an RFC3339 UTC timestamp")
    try:
        parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise AccountingError(f"{label} is not an RFC3339 timestamp") from exc
    if parsed.tzinfo is None:
        raise AccountingError(f"{label} must include a timezone")
    return parsed.astimezone(dt.timezone.utc)


def _finite(value: Any, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise AccountingError(f"{label} must be a finite number")
    result = float(value)
    if positive and result <= 0:
        raise AccountingError(f"{label} must be positive")
    if not positive and result < 0:
        raise AccountingError(f"{label} must be nonnegative")
    return result


def _load_pricing(path: Path) -> tuple[dict[str, Any], dict[str, float]]:
    value = _read_object(path, "observed pricing")
    if value.get("schema") != PRICING_SCHEMA or not isinstance(value.get("evidence"), str):
        raise AccountingError("pricing must be a documented q0q1 observed-pricing observation")
    rates = value.get("rates")
    if not isinstance(rates, Mapping):
        raise AccountingError("pricing.rates must be an object")
    normalized = {
        kind: _finite(rates.get(kind), f"pricing.rates.{kind}", positive=True)
        for kind in ("A100", "L4", "cpu")
    }
    return value, normalized


def _load_overhead(path: Path) -> dict[str, Any]:
    value = _read_object(path, "CPU evaluation overhead receipt")
    if value.get("schema") != OVERHEAD_SCHEMA:
        raise AccountingError("CPU evaluation overhead receipt has the wrong schema")
    root = Path(__file__).resolve().parents[2]
    required = {
        "status": "MEASURED",
        "measurement": "real_q1_evaluate_empty_shard_with_staging_publish_and_cleanup",
        "no_selection_worlds": True,
        "cpu_threads": 1,
        "q1_evaluate_sha256": sha256_file(root / "research/q0q1_20261017/q1_evaluate.py"),
    }
    if any(value.get(key) != expected for key, expected in required.items()):
        raise AccountingError(
            "CPU evaluation overhead receipt is not bound to the current real evaluator"
        )
    if not isinstance(value.get("system"), str) or not value["system"]:
        raise AccountingError("CPU evaluation overhead receipt lacks its Mac system identity")
    _finite(value.get("max_seconds"), "CPU evaluation overhead receipt.max_seconds", positive=True)
    runs = value.get("runs")
    if not isinstance(runs, list) or len(runs) < 3:
        raise AccountingError("CPU evaluation overhead receipt requires at least three real runs")
    if any(
        _finite(row.get("elapsed_seconds"), "CPU overhead run elapsed_seconds", positive=True) <= 0
        for row in runs
        if isinstance(row, Mapping)
    ) or not all(isinstance(row, Mapping) for row in runs):
        raise AccountingError("CPU evaluation overhead receipt has invalid run timings")
    if value["max_seconds"] != max(float(row["elapsed_seconds"]) for row in runs):
        raise AccountingError("CPU evaluation overhead max_seconds must equal the slowest real run")
    return value


def _iter_claims(queue_root: Path, since: dt.date) -> list[tuple[Path, dict[str, Any]]]:
    result = []
    for path in sorted((queue_root / "claimed").glob("q0*/*.json")):
        claim = _read_object(path, "queue claim")
        job_id = claim.get("id")
        if not isinstance(job_id, str) or not job_id.startswith(("q0-", "q0b-")):
            continue
        if _timestamp(claim.get("created_at"), f"claim {path}.created_at").date() >= since:
            result.append((path, claim))
    if not result:
        raise AccountingError(f"no Q0/Q0b queue claims found since {since.isoformat()}")
    return result


def _terminal_receipts(queue_root: Path) -> dict[tuple[str, int, str], tuple[Path, dict[str, Any]]]:
    result: dict[tuple[str, int, str], tuple[Path, dict[str, Any]]] = {}
    for state in ("done", "failed"):
        for path in sorted((queue_root / state).glob("q0*/*.json")):
            receipt = _read_object(path, "queue terminal receipt")
            key = (receipt.get("id"), receipt.get("generation"), receipt.get("token"))
            if (
                not isinstance(key[0], str)
                or not isinstance(key[1], int)
                or not isinstance(key[2], str)
            ):
                raise AccountingError(f"terminal receipt lacks id/generation/token: {path}")
            if key in result:
                raise AccountingError(f"duplicate terminal evidence for queue attempt {key}")
            result[key] = (path, receipt)
    return result


def _conservative_end(
    queue_root: Path, claim_path: Path, claim: Mapping[str, Any]
) -> tuple[dt.datetime, Path, dict[str, Any], str] | None:
    """Return an explicit post-claim closure, never an inferred heartbeat timeout."""
    worker_id, job_id, token = claim.get("worker_id"), claim.get("id"), claim.get("token")
    if not all(isinstance(value, str) for value in (worker_id, job_id, token)):
        raise AccountingError(f"claim lacks worker/id/token: {claim_path}")
    heartbeat_path = queue_root / "heartbeats" / f"{worker_id}.json"
    if heartbeat_path.is_file():
        heartbeat = _read_object(heartbeat_path, "queue heartbeat")
        if (
            heartbeat.get("worker_id") == worker_id
            and heartbeat.get("phase") == "idle"
            and heartbeat.get("job_id") is None
            and heartbeat.get("token") is None
        ):
            end = _timestamp(heartbeat.get("updated_at"), f"heartbeat {heartbeat_path}.updated_at")
            return end, heartbeat_path, heartbeat, "explicit_idle_heartbeat_upper_bound"
    control_path = queue_root / "control" / f"{job_id}.json"
    if control_path.is_file():
        control = _read_object(control_path, "queue control")
        reason = control.get("reason")
        explicit_runtime_loss = (
            isinstance(reason, str) and "runtime verified disconnected" in reason.lower()
        )
        if control.get("id") == job_id and explicit_runtime_loss:
            end = _timestamp(control.get("updated_at"), f"control {control_path}.updated_at")
            return end, control_path, control, "explicit_verified_runtime_disconnect_upper_bound"
    return None


def build_accounting(
    *, drive_root: Path, pricing_path: Path, overhead_path: Path, day: str = DEFAULT_DATE
) -> dict[str, Any]:
    try:
        since_date = dt.date.fromisoformat(day)
    except ValueError as exc:
        raise AccountingError("--date must use YYYY-MM-DD") from exc
    pricing, rates = _load_pricing(pricing_path)
    overhead = _load_overhead(overhead_path)
    queue_root = drive_root / "queue"
    claims = _iter_claims(queue_root, since_date)
    terminals = _terminal_receipts(queue_root)
    claim_keys: set[tuple[str, int, str]] = set()
    for claim_path, claim in claims:
        key = (claim.get("id"), claim.get("generation"), claim.get("token"))
        if (
            not isinstance(key[0], str)
            or not isinstance(key[1], int)
            or not isinstance(key[2], str)
        ):
            raise AccountingError(f"claim lacks id/generation/token: {claim_path}")
        if key in claim_keys:
            raise AccountingError(f"duplicate queue claim evidence for attempt {key}")
        claim_keys.add(key)
    for key, (receipt_path, receipt) in terminals.items():
        recorded = receipt.get("recorded_at")
        if (
            isinstance(recorded, str)
            and _timestamp(recorded, f"receipt {receipt_path}.recorded_at").date() >= since_date
            and key not in claim_keys
        ):
            raise AccountingError(
                f"terminal receipt since lower bound has no matching queue claim: {receipt_path}"
            )
    attempts = []
    gpu_seconds = 0.0
    cpu_seconds = 0.0
    units = 0.0
    for claim_path, claim in claims:
        generation = claim.get("generation")
        token = claim.get("token")
        job_id = claim.get("id")
        if (
            not isinstance(generation, int)
            or not isinstance(token, str)
            or not isinstance(job_id, str)
        ):
            raise AccountingError(f"claim lacks id/generation/token: {claim_path}")
        hardware = (
            claim.get("capabilities", {}).get("gpu")
            if isinstance(claim.get("capabilities"), Mapping)
            else None
        )
        if hardware not in rates:
            raise AccountingError(
                f"claim has unknown hardware debit rate {hardware!r}: {claim_path}"
            )
        started = _timestamp(claim["created_at"], f"claim {claim_path}.created_at")
        terminal = terminals.get((job_id, generation, token))
        conservative = False
        if terminal is not None:
            receipt_path, receipt = terminal
            receipt_elapsed = _finite(
                receipt.get("elapsed_seconds"),
                f"receipt {receipt_path}.elapsed_seconds",
                positive=True,
            )
            if receipt.get("gpu") != hardware:
                raise AccountingError(f"receipt hardware differs from claim: {receipt_path}")
            if receipt.get("worker_id") is not None and receipt.get("worker_id") != claim.get(
                "worker_id"
            ):
                raise AccountingError(f"receipt worker differs from claim: {receipt_path}")
            claim_to_finished = None
            queue_setup = None
            if receipt.get("started_at") is not None or receipt.get("finished_at") is not None:
                raw_started, raw_finished = receipt.get("started_at"), receipt.get("finished_at")
                if not all(
                    isinstance(value, (int, float)) and not isinstance(value, bool)
                    for value in (raw_started, raw_finished)
                ):
                    raise AccountingError(f"receipt timestamps are incomplete: {receipt_path}")
                if abs((float(raw_finished) - float(raw_started)) - receipt_elapsed) > 5.0:
                    raise AccountingError(
                        f"receipt elapsed_seconds does not reconcile: {receipt_path}"
                    )
                claim_to_finished = float(raw_finished) - started.timestamp()
                if claim_to_finished <= 0:
                    raise AccountingError(f"receipt finished before queue claim: {receipt_path}")
                raw_run_started = receipt.get("run_started_at")
                if raw_run_started is not None:
                    if not isinstance(raw_run_started, (int, float)) or isinstance(
                        raw_run_started, bool
                    ):
                        raise AccountingError(f"receipt run_started_at is invalid: {receipt_path}")
                    queue_setup = float(raw_run_started) - float(raw_started)
                    if queue_setup < 0:
                        raise AccountingError(
                            f"receipt run_started_at predates started_at: {receipt_path}"
                        )
            elapsed = max(receipt_elapsed, claim_to_finished or 0.0)
            end_text = receipt.get("recorded_at")
            provenance = "terminal_receipt_elapsed_seconds"
            closure_path = receipt_path
            closure_sha = sha256_file(receipt_path)
        else:
            closure = _conservative_end(queue_root, claim_path, claim)
            if closure is None:
                raise AccountingError(
                    f"claim has no terminal receipt or explicit closure evidence: {claim_path}"
                )
            end, closure_path, _, provenance = closure
            elapsed = (end - started).total_seconds()
            if elapsed <= 0:
                raise AccountingError(f"closure does not follow claim: {closure_path}")
            end_text = end.isoformat().replace("+00:00", "Z")
            closure_sha = sha256_file(closure_path)
            conservative = True
        charged_units = elapsed / 3600.0 * rates[hardware]
        if hardware == "cpu":
            cpu_seconds += elapsed
        else:
            gpu_seconds += elapsed
        units += charged_units
        attempts.append(
            {
                "id": job_id,
                "generation": generation,
                "token": token,
                "hardware": hardware,
                "rate_units_per_hour": rates[hardware],
                "claim_path": str(claim_path),
                "claim_sha256": sha256_file(claim_path),
                "claim_started_at": claim["created_at"],
                "closure_path": str(closure_path),
                "closure_sha256": closure_sha,
                "closure_at": end_text,
                "elapsed_seconds": elapsed,
                "terminal_elapsed_seconds": receipt_elapsed if terminal is not None else None,
                "claim_to_finished_seconds": claim_to_finished if terminal is not None else None,
                "queue_prelude_seconds": (
                    max(0.0, (claim_to_finished or 0.0) - receipt_elapsed)
                    if terminal is not None
                    else None
                ),
                "queue_setup_seconds": queue_setup if terminal is not None else None,
                "charged_units": charged_units,
                "provenance": provenance,
                "conservative_upper_bound": conservative,
                "terminal_status": terminal[1]["status"] if terminal is not None else None,
            }
        )
    observed_setups = [
        attempt["queue_setup_seconds"] + attempt["queue_prelude_seconds"]
        for attempt in attempts
        if attempt["queue_setup_seconds"] is not None
    ]
    if not observed_setups:
        raise AccountingError("no terminal receipt exposes observed queue setup timing")
    observed_queue_setup = max(observed_setups)
    # The queue receipts are Colab observations.  The Colab checklist records
    # 2--3x slower per-core CPU than this Mac, so use the conservative endpoint
    # for a CPU evaluation-job forecast; export reserve is registered at 60 s.
    cpu_overhead = 3.0 * overhead["max_seconds"] + observed_queue_setup + 60.0
    result = {
        "schema": SCHEMA,
        "since_date_utc": since_date.isoformat(),
        "status": (
            "COMPLETE_WITH_CONSERVATIVE_BOUNDS"
            if any(a["conservative_upper_bound"] for a in attempts)
            else "COMPLETE"
        ),
        "pricing": {
            "path": str(pricing_path),
            "sha256": sha256_file(pricing_path),
            "evidence": pricing["evidence"],
            "rates_units_per_hour": rates,
        },
        "cpu_eval_overhead": {
            "path": str(overhead_path),
            "sha256": sha256_file(overhead_path),
            "seconds": overhead["max_seconds"],
            "measurement_schema": overhead["schema"],
            "observed_colab_queue_setup_and_prelude_seconds": observed_queue_setup,
            "mac_to_colab_cpu_projection_multiplier": 3.0,
            "registered_export_reserve_seconds": 60.0,
            "forecast_seconds": cpu_overhead,
        },
        "attempts": attempts,
        "q0_units": units,
        "q0_gpu_hours": gpu_seconds / 3600.0,
        "q0_cpu_hours": cpu_seconds / 3600.0,
        "cpu_unit_rate_per_hour": rates["cpu"],
        "cpu_eval_job_overhead_seconds": cpu_overhead,
        "derivation": (
            "sum(attempt elapsed_seconds / 3600 * observed-pricing rate); "
            "terminal charge includes claim arbitration through cleanup; "
            "CPU overhead forecast is 3x measured Mac staging/publish/cleanup "
            "plus max observed Colab queue setup and prelude "
            "plus 60s export/analysis/cleanup reserve"
        ),
    }
    q0_overrun = units - BASELINE_Q0_UNITS_CAP
    result["a3_envelope"] = {
        "amendment": "A3",
        "baseline_q0_units_cap": BASELINE_Q0_UNITS_CAP,
        "observed_q0_units": units,
        "q0_units_cap": units,
        "q1_original_units_cap": Q1_ORIGINAL_UNITS_CAP,
        "baseline_combined_units_cap": BASELINE_COMBINED_UNITS_CAP,
        "reserve_fraction": RESERVE_FRACTION,
        "reserve_basis_combined_units": BASELINE_COMBINED_UNITS_CAP,
        "reserve_units": BASELINE_RESERVE_UNITS,
        "q1_dispatch_headroom_units": Q1_ORIGINAL_UNITS_CAP - BASELINE_RESERVE_UNITS,
        "q0_overrun_units": q0_overrun,
        "combined_units_cap": BASELINE_COMBINED_UNITS_CAP + q0_overrun,
        "combined_dispatch_limit_units": units + (Q1_ORIGINAL_UNITS_CAP - BASELINE_RESERVE_UNITS),
        "derivation": (
            "A3: q0_units_cap=observed_q0_units; combined_units_cap="
            "175+(observed_q0_units-40); reserve_units=0.20*175=35; "
            "combined_dispatch_limit_units=observed_q0_units+(135-35)."
        ),
    }
    # A3 is a raised-cap amendment.  Preserve ordinary accounting for fixtures
    # and diagnostics below the historical cap, but never let them become a
    # dispatch budget: consumers must call this validator before admission.
    if units > BASELINE_Q0_UNITS_CAP:
        validate_a3_envelope(result)
    return result


def measure_cpu_eval_overhead(
    *, bank: Path, b8: Path, parity_receipt: Path, out: Path, runs: int = 3
) -> dict[str, Any]:
    """Measure real evaluator startup/copy/export without scoring a bank world.

    Shard ``128/129`` is deliberately outside Q1's 128 scored worlds.  The
    evaluator still executes its real CLI, imports, B8/hash/bank/parity checks,
    context construction, and receipt export.  Copying its inputs before each
    run preserves the queue-local staging cost while avoiding a selection look.
    """
    if out.exists():
        raise AccountingError(f"refusing to overwrite CPU evaluation overhead receipt: {out}")
    if runs < 3:
        raise AccountingError("CPU evaluation overhead requires at least three runs")
    if platform.system() != "Darwin":
        raise AccountingError("CPU evaluation overhead measurement must run on the Mac")
    audit = bank.with_name("Q1_SEED_AUDIT.json")
    for path, label in (
        (bank, "bank"),
        (audit, "Q1 seed audit"),
        (b8, "B8 checkpoint"),
        (parity_receipt, "parity receipt"),
    ):
        if not path.is_file():
            raise AccountingError(f"CPU evaluation overhead {label} is missing: {path}")
    root = Path(__file__).resolve().parents[2]
    values = []
    with tempfile.TemporaryDirectory(prefix="q1-cpu-eval-overhead-") as temporary:
        temporary_root = Path(temporary)
        for index in range(runs):
            run_root = temporary_root / f"run-{index}"
            run_root.mkdir()
            local_bank, local_audit, local_b8, local_parity = (
                run_root / "Q1_WORLD_BANK.json",
                run_root / "Q1_SEED_AUDIT.json",
                run_root / "b8.pth",
                run_root / "runtime-parity.json",
            )
            started = time.monotonic()
            for source, target in (
                (bank, local_bank),
                (audit, local_audit),
                (b8, local_b8),
                (parity_receipt, local_parity),
            ):
                shutil.copy2(source, target)
            # Exercise the same archive extraction boundary as a candidate
            # evaluation without creating a scientific candidate or selecting
            # a registered world.  B8-only bytes are an explicit timing fixture.
            training_archive = run_root / "b8-only-training.tar.gz"
            with tarfile.open(training_archive, "w:gz") as archive:
                archive.add(local_b8, arcname="final.pth")
            with tarfile.open(training_archive, "r:gz") as archive:
                archive.extractall(run_root / "staged-training", filter="data")
            staged_b8 = run_root / "staged-training" / "final.pth"
            command = [
                sys.executable,
                "-m",
                "research.q0q1_20261017.q1_evaluate",
                "--root",
                str(run_root / "export"),
                "--bank",
                str(local_bank),
                "--b8-checkpoint",
                str(staged_b8),
                "--checkpoint",
                str(local_b8),
                "--arm",
                "B8",
                "--seed",
                "261701",
                "--shard",
                "128/129",
                "--workers",
                "1",
                "--parity-receipt",
                str(local_parity),
            ]
            environment = dict(os.environ)
            environment.update(
                {
                    "SNAKE_DQN_DEVICE": "cpu",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "NUMBA_NUM_THREADS": "1",
                    "VECLIB_MAXIMUM_THREADS": "1",
                }
            )
            completed = subprocess.run(
                command, cwd=root, env=environment, text=True, capture_output=True, check=False
            )
            elapsed = time.monotonic() - started
            if completed.returncode:
                raise AccountingError(
                    f"real Q1 evaluator overhead run {index + 1} failed ({completed.returncode}): "
                    f"{completed.stderr.strip() or completed.stdout.strip()}"
                )
            exports = list((run_root / "export").rglob("*.json"))
            if len(exports) != 1:
                raise AccountingError(
                    "real Q1 evaluator overhead run did not export exactly one receipt"
                )
            from research.q0q1_20261017.run_queue_job import _publish_attempt

            exchange = run_root / "queue-output"
            exchange.mkdir()
            _publish_attempt(
                exchange,
                run_root / "export",
                job_id="q1-eval-overhead",
                status="COMPLETED",
                error=None,
            )
            archive = exchange / "result.tar.gz"
            sidecar = exchange / "result.tar.gz.sha256"
            if (
                not archive.is_file()
                or not sidecar.is_file()
                or sha256_file(archive) not in sidecar.read_text()
            ):
                raise AccountingError(
                    "real Q1 evaluator overhead run did not publish a verified archive"
                )
            shutil.rmtree(run_root / "staged-training")
            elapsed = time.monotonic() - started
            values.append(
                {
                    "elapsed_seconds": elapsed,
                    "command": command,
                    "export": str(exports[0]),
                    "published_archive_sha256": sha256_file(archive),
                }
            )
    value = {
        "schema": OVERHEAD_SCHEMA,
        "status": "MEASURED",
        "measurement": "real_q1_evaluate_empty_shard_with_staging_publish_and_cleanup",
        "no_selection_worlds": True,
        "bank_sha256": sha256_file(bank),
        "q1_seed_audit_sha256": sha256_file(audit),
        "b8_sha256": sha256_file(b8),
        "parity_receipt_sha256": sha256_file(parity_receipt),
        "q1_evaluate_sha256": sha256_file(root / "research/q0q1_20261017/q1_evaluate.py"),
        "system": platform.platform(),
        "cpu_threads": 1,
        "runs": values,
        "max_seconds": max(row["elapsed_seconds"] for row in values),
        "limitation": (
            "Measures per-job staging/startup/validation/export only; " "it excludes scored worlds."
        ),
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return value


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--drive-root", type=Path, default=Path(DEFAULT_DRIVE_ROOT))
    parser.add_argument("--pricing", type=Path, required=True)
    parser.add_argument("--cpu-eval-overhead-receipt", type=Path, default=DEFAULT_OVERHEAD)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--since-date",
        "--date",
        dest="date",
        default=DEFAULT_DATE,
        help="UTC lower bound; includes later attempts after midnight",
    )
    parser.add_argument("--measure-cpu-eval-overhead", action="store_true")
    parser.add_argument("--q1-bank", type=Path)
    parser.add_argument("--b8-checkpoint", type=Path)
    parser.add_argument("--runtime-parity-receipt", type=Path)
    args = parser.parse_args(argv)
    if args.measure_cpu_eval_overhead:
        if any(
            value is None
            for value in (args.q1_bank, args.b8_checkpoint, args.runtime_parity_receipt)
        ):
            parser.error(
                "--measure-cpu-eval-overhead needs --q1-bank --b8-checkpoint "
                "--runtime-parity-receipt"
            )
        try:
            receipt = measure_cpu_eval_overhead(
                bank=args.q1_bank,
                b8=args.b8_checkpoint,
                parity_receipt=args.runtime_parity_receipt,
                out=args.cpu_eval_overhead_receipt,
            )
        except AccountingError as exc:
            parser.error(str(exc))
        print(
            json.dumps(
                {"out": str(args.cpu_eval_overhead_receipt), "max_seconds": receipt["max_seconds"]}
            )
        )
        return 0
    if args.out.exists():
        parser.error(f"refusing to overwrite accounting: {args.out}")
    try:
        result = build_accounting(
            drive_root=args.drive_root,
            pricing_path=args.pricing,
            overhead_path=args.cpu_eval_overhead_receipt,
            day=args.date,
        )
    except AccountingError as exc:
        parser.error(str(exc))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("x") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(result, sort_keys=True, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
