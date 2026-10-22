"""Prepare/check or execute the leased Mac V0 entrypoint (Claude timing only)."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import re
import signal
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from research.bl_mac_latency.host import check_guard, process_screen, snapshot
from research.bl_served_search.exports import export_json, verify_export
from research.bl_v0 import contract
from research.bl_v0.lease import DEFAULT_LEASE, acquire, require


def protected_jobs(raw: str | None = None) -> None:
    """Refuse real K-1/X5 processes, including rescheduled or legacy launches."""
    raw = (
        raw
        if raw is not None
        else subprocess.check_output(["ps", "-Ao", "pid,comm,args"], text=True, timeout=10)
    )
    for line in raw.splitlines()[1:]:
        fields = line.strip().split(None, 2)
        if len(fields) != 3:
            continue
        # macOS truncates COMM to 16 columns (e.g. /Library/Framewo).
        # Inspect the executable at the start of ARGS, never arbitrary prompt text.
        executable = fields[2].split(None, 1)[0] if fields[2].strip() else ""
        if not any("python" in Path(name).name.lower() for name in (fields[1], executable)):
            continue
        if re.search(
            r"(?:-m\s+research\.(?:bl_k1\.(?:audit|timing)|bl_x5(?:\.timing)?)"
            r"|research/(?:bl_k1/(?:audit|timing)|bl_x5/timing)\.py)(?:\s|$)",
            fields[2],
        ):
            raise RuntimeError(f"protected K-1/X5 process active: pid {fields[0]}")


def thread_guard() -> None:
    """Require one-thread CPU settings even when bypassing the shell launcher."""
    for key in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "NUMBA_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
    ):
        if os.environ.get(key) != "1":
            raise RuntimeError(f"require {key}=1")
    if os.environ.get("SNAKE_DQN_DEVICE") != "cpu" or os.getpriority(os.PRIO_PROCESS, 0) < 10:
        raise RuntimeError("require CPU and nice >=10")


def verify_canary(path: Path, manifest: dict[str, Any]) -> str:
    """Bind real target-platform entrypoint proof to current family/source."""
    if (path.parent / "INVALIDATION.json").exists():
        raise ValueError("Mac canary was invalidated; fresh operator-run canary required")
    checksum = verify_export(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "bl-v0-mac-receipt/v1"
        or value.get("status") != "PRECHECK_ONLY"
        or value.get("sources") != manifest["sources"]
        or value.get("completed_batches") != 4
        or value.get("canary_ok") is not True
        or value.get("platform", {}).get("system") != "Darwin"
        or value.get("host_identity") != host_identity()
        or len(value.get("batch_exports", [])) != 4
    ):
        raise ValueError("missing or different-source Mac canary")
    rows = []
    for item in value["batch_exports"]:
        if verify_export(Path(item["path"])) != item["sha256"]:
            raise ValueError("canary batch drift")
        row = json.loads(Path(item["path"]).read_text())
        if contract.attestation(row) or row.get("identity_ok") is not True:
            raise ValueError("canary counters/identity invalid")
        fixture = next(r for r in manifest["fixtures"] if r["fixture_id"] == row["fixture_id"])
        contract.validate_execution(row, fixture)
        rows.append(row)
    if (
        {r["cell"] for r in rows} != set(contract.CELLS)
        or len({(r["fixture_id"], r["root_sha256"], r["repeat"]) for r in rows}) != 1
        or len({r["semantic"]["chosen"] for r in rows}) != 1
        or not any(
            r["fixture_id"] == rows[0]["fixture_id"] and r["root_sha256"] == rows[0]["root_sha256"]
            for r in manifest["fixtures"]
        )
    ):
        raise ValueError("canary cell/root/action coverage differs")
    return checksum


def host_identity() -> dict[str, str]:
    """Bind the canary to this Mac and Python ABI, not just an OS label."""
    return {
        "node": platform.node(),
        "machine": platform.machine(),
        "python": platform.python_version(),
    }


def verify_k1(path: Path, fixtures: list[dict[str, Any]], manifest_path: Path | None = None) -> str:
    """Require the completed measured K-1 baseline without reusing its timing."""
    checksum = verify_export(path)
    value = json.loads(path.read_text())
    if (
        value.get("schema") != "bl-k1-mac-timing/v1"
        or value.get("completed_batches") != 810
        or value.get("status") not in ("K1_PASS", "EXACT_SERVED_SEARCH_CLOSED_WEEK2")
        or not math.isfinite(value.get("speedup", float("nan")))
        or value.get("speedup", 0) <= 0
        or not re.fullmatch(r"[0-9a-f]{64}", value.get("identity_sha256", ""))
        or manifest_path is None
    ):
        raise ValueError("K-1 timing baseline is not complete/measured")
    manifest_hash = verify_export(manifest_path)
    baseline = json.loads(manifest_path.read_text())
    if (
        value.get("manifest_sha256") != manifest_hash
        or value.get("source_closure") != baseline["sources"]
    ):
        raise ValueError("K-1 source/manifest link differs")
    if {r["fixture_id"]: r["root_sha256"] for r in baseline["fixtures"]} != {
        r["fixture_id"]: r["root_sha256"] for r in fixtures
    }:
        raise ValueError("K-1 fixture coverage differs")
    from research.bl_k1.audit import verify_identity

    if (
        verify_identity(path.parent / "identity.json", baseline, manifest_hash)
        != value["identity_sha256"]
    ):
        raise ValueError("K-1 identity digest differs")
    samples: dict[str, list[float]] = {"stock": [], "kernel": []}
    for repeat in range(30):
        for mode in samples:
            raw = path.parent / "batches" / f"frozen-1500-{repeat:02d}-{mode}.json"
            verify_export(raw)
            row = json.loads(raw.read_text())
            if row["fixture_id"] != "frozen-1500" or row["repeat"] != repeat or row["mode"] != mode:
                raise ValueError("K-1 raw timing identity differs")
            if mode == "kernel" and contract.attestation(row):
                raise ValueError("K-1 native counters invalid")
            seconds = row["stage_seconds"]["opponent_v8"]
            if not math.isfinite(seconds) or seconds <= 0:
                raise ValueError("K-1 raw timer invalid")
            samples[mode].append(seconds)
    ratio = statistics.median(samples["stock"][1:]) / statistics.median(samples["kernel"][1:])
    expected_status = "K1_PASS" if ratio >= 5 else "EXACT_SERVED_SEARCH_CLOSED_WEEK2"
    if value["speedup"] != ratio or value["status"] != expected_status:
        raise ValueError("K-1 result does not match raw warm29 samples")
    return checksum


def run(args: argparse.Namespace) -> None:
    """Run one fresh attempt; preserve every completed batch on any stop."""
    manifest = contract.verify_manifest(args.manifest)
    thread_guard()
    protected_jobs()
    seconds = min(180 if args.smoke else 3600, contract.window(smoke=args.smoke))
    admission = {}
    if not args.smoke:
        if args.operator != "Claude" or not args.canary or not args.k1:
            raise ValueError("real timing requires --operator Claude --canary --k1")
        admission = {
            "canary_sha256": verify_canary(args.canary, manifest),
            "k1_sha256": verify_k1(args.k1, manifest["fixtures"], args.k1_manifest),
        }
    args.out.mkdir(parents=True, exist_ok=False)
    rows: list[dict[str, Any]] = []
    exports = []
    started = time.monotonic()
    lease_value = None
    status = "INCOMPLETE"

    def expired(*_: Any) -> None:
        raise TimeoutError("V0 first-cap deadline reached")

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        seconds = min(seconds, contract.window(smoke=args.smoke))
        with acquire(
            DEFAULT_LEASE, args.operator, "V0 smoke" if args.smoke else "V0 ladder", seconds
        ) as lease:
            lease_value = lease
            signal.setitimer(signal.ITIMER_REAL, seconds)
            before = snapshot()
            export_json(args.out / "idle.json", {**before, "lease": lease})
            if not args.smoke and not before["quiet"]:
                raise RuntimeError("initial host not idle; no daemon/process killing")
            from research.bl_v0.operator import V0Operator

            operator = V0Operator()
            fixtures = manifest["fixtures"]
            if args.smoke:
                fixtures = [row for row in fixtures if row["fixture_id"] == args.smoke_fixture]
                if len(fixtures) != 1:
                    raise ValueError("unknown smoke fixture")
            repeats = 1 if args.smoke else 30
            for fixture in fixtures:
                for repeat in range(repeats):
                    expected = None
                    for cell in contract.CELLS:
                        contract.window(smoke=args.smoke)
                        require(DEFAULT_LEASE, lease)
                        protected_jobs()
                        if time.monotonic() - started >= seconds:
                            raise TimeoutError("V0 budget exhausted")
                        guard = {
                            **check_guard(),
                            "process_screen": process_screen(),
                            "load_average": list(os.getloadavg()),
                        }
                        if not args.smoke and guard["process_screen"]["other_heavy_processes"]:
                            raise RuntimeError("foreign heavy CPU process; host no longer quiet")
                        result = operator.batch(fixture, cell)
                        chosen = int(result["semantic"]["chosen"])
                        if cell == "C1":
                            expected = chosen
                        row = {
                            **result,
                            "fixture_id": fixture["fixture_id"],
                            "root_sha256": fixture["root_sha256"],
                            "mix": fixture["mix"],
                            "cell": cell,
                            "repeat": repeat,
                            "lease_id": lease["id"],
                            "guard": guard,
                            "identity_ok": chosen == expected,
                        }
                        row["attestation_failures"] = contract.attestation(row)
                        try:
                            protected_jobs()
                        except RuntimeError as exc:
                            row["execution_failure"] = str(exc)
                            status = "INVALID"
                        try:
                            contract.validate_execution(row, fixture)
                        except (ValueError, KeyError) as exc:
                            row["execution_failure"] = str(exc)
                            status = "INVALID"
                        if row["attestation_failures"]:
                            status = "INVALID"
                        timers = {
                            "operator_wall_seconds": row["operator_wall_seconds"],
                            "leaf_call_wall_seconds": row["leaf_call_wall_seconds"],
                            **{
                                f"stage_seconds/{key}": value
                                for key, value in row["stage_seconds"].items()
                            },
                        }
                        invalid_timers = {
                            key: repr(value)
                            for key, value in timers.items()
                            if not math.isfinite(value) or value < 0
                        }
                        if row["operator_wall_seconds"] == 0:
                            invalid_timers["operator_wall_seconds"] = "0"
                        if invalid_timers:
                            status = "INVALID"
                            export_json(
                                args.out / "invalid-batch.json",
                                {
                                    "fixture_id": fixture["fixture_id"],
                                    "cell": cell,
                                    "repeat": repeat,
                                    "invalid_timers": invalid_timers,
                                    "lease_id": lease["id"],
                                },
                            )
                            raise ValueError("invalid batch/stage timer")
                        record = export_json(
                            args.out
                            / "batches"
                            / f"{fixture['fixture_id']}-{repeat:02d}-{cell}.json",
                            row,
                        )
                        rows.append(row)
                        exports.append(record)
                        print(
                            json.dumps(
                                {
                                    "completed": len(rows),
                                    "fixture": fixture["fixture_id"],
                                    "cell": cell,
                                }
                            ),
                            flush=True,
                        )
                        if not row["identity_ok"]:
                            status = "INVALID"
                            raise RuntimeError(f"{cell} action mismatch vs C1")
                        if "execution_failure" in row:
                            raise RuntimeError(row["execution_failure"])
                        # Zero-eligible scripted rows are preserved as INVALID,
                        # never fabricated and never silently exempted.
                        if row["attestation_failures"] and fixture["mix"] != "scripted":
                            status = "INVALID"
                            raise RuntimeError("kernel execution attestation failed")
                        contract.window(smoke=args.smoke)
                        require(DEFAULT_LEASE, lease)
            try:
                contract.verify_manifest(args.manifest)
            except (ValueError, RuntimeError, subprocess.CalledProcessError):
                status = "INVALID"
                raise
            outcome = (
                {"status": "PRECHECK_ONLY", "admits": "NONE"}
                if args.smoke
                else contract.decision(rows)
            )
            canary_ok = len(rows) == 4 and all(
                r["identity_ok"] and not r["attestation_failures"] for r in rows
            )
            receipt = {
                "schema": "bl-v0-mac-receipt/v1",
                **outcome,
                "canary_ok": canary_ok if args.smoke else None,
                "argv": sys.argv,
                "platform": contract.ruler.platform_identity("mac"),
                "sources": manifest["sources"],
                "host_identity": host_identity(),
                "manifest_sha256": verify_export(args.manifest),
                "ledger": contract.published(),
                "lease": lease,
                "workers": 1,
                "threads": 1,
                "elapsed_seconds": time.monotonic() - started,
                "completed_batches": len(rows),
                "batch_exports": exports,
                "metrics": contract.summarize(rows),
                "attestation_invalid_batches": sum(bool(r["attestation_failures"]) for r in rows),
                "baseline_policy": "C1 re-timed; no K-1 sample reuse",
                "admission": admission,
                "premises": {"L2_prime": "NOT_READY(L2′)"},
                "host_after": snapshot(),
                "serving_or_release": "NOT_GRANTED",
            }
            export_json(args.out / ("smoke.json" if args.smoke else "timing.json"), receipt)
    except BaseException as exc:
        # Disable the deadline while writing the stopped artifact.
        signal.setitimer(signal.ITIMER_REAL, 0)
        export_json(
            args.out / "stopped.json",
            {
                "status": status,
                "error": str(exc),
                "failure_class": type(exc).__name__,
                "completed_batches": len(rows),
                "batch_exports": exports,
                "lease": lease_value,
                "elapsed_seconds": time.monotonic() - started,
                "failed_observation": getattr(exc, "observation", None),
                "admits": "NONE",
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--fixtures", type=Path, default=contract.DEFAULT_FIXTURES)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--smoke-fixture", default="frozen-800")
    parser.add_argument("--operator", default="Sol-K")
    parser.add_argument("--canary", type=Path)
    parser.add_argument("--k1", type=Path)
    parser.add_argument("--k1-manifest", type=Path)
    args = parser.parse_args()
    if args.prepare:
        if args.out is None:
            parser.error("--prepare needs --out")
        print(contract.prepare(args.fixtures, args.out))
    elif args.check:
        value = contract.verify_manifest(args.manifest)
        print(
            json.dumps(
                {
                    "status": "PREPARED_NOT_TIMED",
                    "forecast_seconds": value["forecast_seconds"],
                    "units": 0,
                }
            )
        )
    else:
        if args.manifest is None or args.out is None:
            parser.error("run requires --manifest and --out")
        run(args)


if __name__ == "__main__":
    main()
