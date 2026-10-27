"""Preserve the measured forecast boundary and real prepare-target refusal."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from research.bl_ppo.exports import digest, export_bytes, export_json, verify_export
from research.bl_residual import f5_step3_package

DEST = Path(__file__).resolve().parent
REPO = DEST.parents[3]
ART = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "sol-20261026/f5-attempt03-target-preparation"
)
PACKAGE = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/"
    "sol-20261026/f5-hold-corrections-v5/step3-package"
)
PUBLICATION = REPO / "research/bl_residual/ledger/F5_STEP1_ATTEMPT03_20261026"
PUBLICATION_COMMIT = "bc22e072a70941fab45b37fc624276bcd7453e08"


def binding(path: Path) -> dict:
    """Bind existing immutable evidence without rewriting it."""
    return {"path": str(path), "sha256": digest(path), "size": path.stat().st_size}


def publish(name: str, value: dict) -> None:
    """Publish identical metadata to artifact and Git ledgers create-only."""
    export_json(ART / name, value)
    export_bytes(DEST / name, (ART / name).read_bytes())


def main() -> None:
    """Use a truthful HOLD forecast; never fabricate a qualified target."""
    ART.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "merge-base", "--is-ancestor", PUBLICATION_COMMIT, "origin/main"],
        cwd=REPO,
        check=True,
    )
    index = json.loads((PUBLICATION / "PUBLICATION.json").read_bytes())
    verify_export(PUBLICATION / "PUBLICATION.json")
    for entry in index["entries"]:
        parts = []
        for relative in entry["parts"]:
            path = PUBLICATION / relative
            verify_export(path)
            raw = path.read_bytes()
            assert raw == subprocess.check_output(
                ["git", "show", f"{PUBLICATION_COMMIT}:{path.relative_to(REPO)}"], cwd=REPO
            )
            parts.append(raw)
        raw = b"".join(parts)
        raw = gzip.decompress(raw) if entry["encoding"] == "gzip" else raw
        assert len(raw) == entry["size"]
        assert hashlib.sha256(raw).hexdigest() == entry["sha256"]
        assert raw == Path(entry["original_path"]).read_bytes()
    f5_step3_package.verify(PACKAGE)
    output = PACKAGE / "qualification-output-attempt03"
    receipt = json.loads((output / "receipt.json").read_bytes())
    assert receipt["status"] == "PASS" and receipt["stop_verified"] is True
    for row in receipt["exports"]:
        path = output / "exports" / row["path"]
        assert verify_export(path) == row["sha256"]
        assert path.stat().st_size == row["size"]
    publish(
        "PUBLICATION_VERIFICATION.json",
        {
            "status": "PASS",
            "publication_commit": PUBLICATION_COMMIT,
            "publication": binding(PUBLICATION / "PUBLICATION.json"),
            "reconstructed_files": len(index["entries"]),
            "reconstructed_bytes": sum(e["size"] for e in index["entries"]),
            "all_parts_match_published_git_objects_and_originals": True,
            "original_output_file_count": sum(
                e["source_relative_path"].startswith("output/") for e in index["entries"]
            ),
            "original_output_bytes": sum(
                e["size"]
                for e in index["entries"]
                if e["source_relative_path"].startswith("output/")
            ),
            "package_verification": "PASS_UNCHANGED_EXECUTABLE_SEAL_AND_INPUTS",
        },
    )
    # This is one inclusive lifecycle charge. Nested setup/transfer/child/export
    # clocks are diagnostic subdivisions and are never added again to it.
    attempts = []
    base = PACKAGE.parents[1]
    for number, path in (
        (
            1,
            base
            / "f5-hold-corrections-v4/step3-package/qualification-output-attempt01/receipt.json",
        ),
        (
            2,
            base
            / "f5-hold-corrections-v4/step3-package/qualification-output-attempt02/receipt.json",
        ),
        (3, output / "receipt.json"),
    ):
        old = json.loads(path.read_bytes())
        attempts.append(
            {
                "attempt": number,
                "receipt": binding(path),
                "status": old["status"],
                "elapsed_seconds": old["elapsed_seconds"],
                "stop_verified": old["stop_verified"],
                "account_debit_observed_units": (
                    old["usage_before"]["balance"] - old["usage_after"]["balance"]
                ),
                "own_integrated_units_at_last_crosscheck": (
                    (old["accounting"].get("last_crosscheck") or {}).get("own_elapsed_units")
                ),
                "inclusive_conservative_charge_units_at_declared_2_per_hour": (
                    old["elapsed_seconds"] * 2.0 / 3600
                ),
                "debit_scope": "Saved lagged account observations; not exact final billed units",
            }
        )
    publish(
        "ACCOUNTING_APPEND.json",
        {
            "scope": "Preliminary qualification attempts 01-03 only; original history retained",
            "attempts": attempts,
            "elapsed_seconds_inclusive_sum": sum(a["elapsed_seconds"] for a in attempts),
            "no_nested_clock_double_charge": True,
            "historical_timing_seconds_separate": 10902.09342033323,
            "historical_canary_seconds_separate": 3849.601884124102,
            "source": ("V4/V5 envelope historical attribution; not a global campaign spend audit"),
            "prospective_effective_480_minutes_unchanged": True,
        },
    )
    envelope = f5_step3_package.envelope()
    scan = envelope["scan_forecast_seconds"]
    branches = envelope["branch_diagnostic_seconds"]
    envelope.update(
        {
            "forecast_status": "HOLD_MISSING_INDEPENDENTLY_ACCEPTED_OPERATIONAL_COSTS",
            "preliminary_qualification_verified": True,
            "qualification_receipt": binding(output / "receipt.json"),
            "qualification": binding(output / "exports/qualification.json"),
            "environment": binding(output / "exports/environment.json"),
            "known_measured_lifecycle_seconds": receipt["elapsed_seconds"],
            "diagnostic_historical_scan_seconds": scan,
            "diagnostic_historical_bundle_and_six_individuals_seconds": branches,
            "diagnostic_partial_seconds": receipt["elapsed_seconds"] + scan + branches,
            "diagnostic_scope": (
                "Preliminary lifecycle plus inherited short-body expansion; not a complete "
                "measured "
                "forecast, upper bound, production ETA, long-body measurement, or admission"
            ),
            "measured_total_seconds": None,
            "unknown_costs": [
                "Current long-body crossing snapshot/restore/digest, features and legal setup",
                "Actual long-body all-actions bundle and clone+individual H300 path bounds",
                "Current instrumented crossing-scan incremental cost",
                "Six spawned workers' imports, model/cache/JIT preparation and pool start/stop",
                "Per-unit immutable exports, rolling ledgers, summary, transfer and I/O",
                "Maximum worker imbalance/idle and complete cleanup attribution",
            ],
            "missing_evidence_is_not_zero": True,
            "timing_units_collected_by_this_task": 0,
            "mac_companion_timing_collected": False,
            "mac_load_observation": {
                "local_time": "2026-10-26 17:30:46 America/Los_Angeles",
                "load_average": [6.54, 6.97, 8.15],
                "logical_cpus": 18,
                "cpu_user_percent": 16.23,
                "cpu_sys_percent": 10.53,
                "cpu_idle_percent": 73.23,
                "power": "AC attached; 80 percent",
                "thermal": "No warning recorded; no CPU power status recorded",
                "uncontended_timing_certified": False,
                "scope": "Read-only machine snapshot; no benchmark or Mac companion measurement",
            },
            "authority": [
                binding(REPO / path)
                for path in (
                    "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_V4.md",
                    "research/bl_residual/f5_hold_correction_20261026/ENGINEERING_CORRECTIONS.md",
                    "research/bl_residual/f5_step3_driver.py",
                )
            ],
            "independent_acceptance": "NOT_GRANTED",
            "science_admission": False,
        }
    )
    publish("MEASURED_FORECAST_HOLD.json", envelope)
    target = ART / "prepare-target-attempt01"
    command = [
        sys.executable,
        "-m",
        "research.bl_residual.f5_step3_package",
        "prepare-target",
        str(PACKAGE),
        "--output",
        str(target),
        "--qualification",
        str(output / "exports/qualification.json"),
        "--forecast",
        str(ART / "MEASURED_FORECAST_HOLD.json"),
        "--ledger",
        str(PACKAGE / "inputs/verified-ledger.json"),
    ]
    completed = subprocess.run(command, cwd=REPO, env=os.environ.copy(), capture_output=True)
    export_bytes(ART / "prepare-target.stdout.log", completed.stdout)
    export_bytes(ART / "prepare-target.stderr.log", completed.stderr)
    assert completed.returncode != 0
    assert b"measure requires finite qualified 60/120-minute envelope" in completed.stderr
    assert not (target / "TARGET.json").exists()
    publish(
        "PREPARE_TARGET_RESULT.json",
        {
            "status": "REFUSED_EXPECTED_INCOMPLETE_FORECAST",
            "command": command,
            "process_nice": os.getpriority(os.PRIO_PROCESS, 0),
            "numeric_threads": {
                k: os.environ[k]
                for k in (
                    "OMP_NUM_THREADS",
                    "OMP_THREAD_LIMIT",
                    "MKL_NUM_THREADS",
                    "OPENBLAS_NUM_THREADS",
                    "NUMBA_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS",
                )
            },
            "returncode": completed.returncode,
            "stdout": binding(ART / "prepare-target.stdout.log"),
            "stderr": binding(ART / "prepare-target.stderr.log"),
            "partial_staging_path": str(target),
            "target_json_produced": False,
            "staged_files": [binding(p) for p in sorted(target.rglob("*")) if p.is_file()],
            "seal_final_run": False,
            "dispatches": 0,
            "provider_calls": 0,
            "source_or_guard_changed": False,
        },
    )
    print(
        json.dumps(
            {
                "forecast": binding(ART / "MEASURED_FORECAST_HOLD.json"),
                "prepare_target": "REFUSED",
                "target_json": False,
            }
        )
    )


if __name__ == "__main__":
    main()
