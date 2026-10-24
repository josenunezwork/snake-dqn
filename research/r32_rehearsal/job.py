"""Portable excluded R32 units. Never launches workloads or accesses study banks."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import subprocess
import tarfile
import threading
import time
from pathlib import Path

from research.bl_ppo.exports import digest, export_json, verify_export
from research.r32_rehearsal.lifecycle import (
    allocation,
    check_sources,
    disposition,
    publish_unit,
    read_ledger,
    start_attempt,
)

ROOT = Path(__file__).resolve().parents[2]


def run_portable(inputs: Path, output: Path, *, mode: str) -> dict:
    """Run each frozen unit once, skip uploaded verified IDs, publish receipt last."""
    from research.r32_rehearsal import decisions, matrix, statistics

    if mode not in ("canary", "main", "tiny"):
        raise ValueError("unknown R32 mode")
    spec_path = inputs / ("canary-spec.json" if mode == "canary" else "spec.json")
    spec = json.loads(spec_path.read_text())
    allocation(spec)
    check_sources(spec, ROOT)
    ledger_path = inputs / ("canary-ledger.json" if mode == "canary" else "verified-ledger.json")
    ledger = json.loads(ledger_path.read_text())
    verified = read_ledger(ledger, spec)
    if digest(ledger_path) != spec["ledger_sha256"]:
        raise ValueError("unbound Mac-verified ledger")
    start_attempt(output, spec)
    started = time.monotonic()
    counters = {"done": len(verified), "total": len(spec["units"])}
    stopped = threading.Event()

    def pulse() -> None:
        while not stopped.wait(110):
            print(
                json.dumps({**counters, "elapsed_seconds": time.monotonic() - started}), flush=True
            )

    heartbeat = threading.Thread(target=pulse, daemon=True)
    heartbeat.start()
    durations = []
    try:
        for unit in spec["units"]:
            index = unit["index"]
            if index in verified:
                continue
            tick = time.monotonic()
            if unit["kind"] == "decision":
                value = decisions.run_case(unit["case"])
                if value.get("pass") is not True:
                    raise ValueError("false-ready: decision disagrees with independent oracle")
            elif unit["kind"] == "lifecycle":
                value = matrix.run_matrix(output / "lifecycle-fixtures")
            elif unit["kind"] == "statistics":
                value = statistics.run_frozen_unit(unit["scenario"], unit["trials"])
            else:
                raise ValueError("unknown frozen unit kind")
            publish_unit(output, spec, index, value)
            counters["done"] += 1
            durations.append({"index": index, "elapsed_seconds": time.monotonic() - tick})
        receipt = {
            "schema": "r32-attempt/v1",
            "synthetic": True,
            "scientific_result": False,
            "namespace": spec["namespace"],
            "allocation": allocation(spec),
            "spec_sha256": digest(spec_path),
            "ledger_sha256": digest(ledger_path),
            "completed": [row["index"] for row in durations],
            "skipped": sorted(verified),
            "done": counters["done"],
            "expected": counters["total"],
            "durations": durations,
            "elapsed_seconds": time.monotonic() - started,
            "runtime_release": "SUPERVISOR_MUST_VERIFY",
        }
        export_json(output / "receipt.json", receipt)
        return receipt
    finally:
        stopped.set()
        heartbeat.join(timeout=2)


def reduce_verified(inputs: Path, ledger_path: Path, output: Path, receipts: list[Path]) -> dict:
    """Mac-only explicit terminal look; filename changes cannot create a second look."""
    import platform

    from tools.ops.spend import verify_receipts

    if platform.system() != "Darwin":
        raise ValueError("Mac-only reducer")
    spec = json.loads((inputs / "spec.json").read_text())
    check_sources(spec, ROOT)
    ledger = json.loads(ledger_path.read_text())
    rows = read_ledger(ledger, spec)
    if set(rows) != set(range(len(spec["units"]))):
        raise ValueError("INCOMPLETE: full frozen allocation required")
    if not receipts or len(set(path.resolve() for path in receipts)) != len(receipts):
        raise ValueError("distinct supervisor receipts required")
    charges = verify_receipts(receipts)
    if any(row.own_rate_units_per_hour <= 0 for row in charges):
        raise ValueError("positive verified own debit required")
    if (
        sum(row.elapsed_seconds for row in charges) > 288 * 60
        or sum(row.units for row in charges) > 0.384
    ):
        raise ValueError("shared R32 CPU envelope exceeded")
    bound_receipts = []
    recovered = {}
    for path in receipts:
        value = json.loads(path.read_text())
        if value.get("stop_verified") is not True:
            raise ValueError("STOP_UNVERIFIED: preserve units; reconcile runtime before reduction")
        saved_manifest = path.parent / "manifest.json"
        manifest = json.loads(saved_manifest.read_text())
        input_rows = {item["dest"]: item for item in manifest.get("inputs", [])}
        spec_input = input_rows.get("spec.json", {})
        input_spec_path = Path(spec_input.get("path", "/missing-r32-spec"))
        if not input_spec_path.is_file() or digest(input_spec_path) != spec_input.get("sha256"):
            raise ValueError("unbound supervisor spec input")
        attempt_spec = json.loads(input_spec_path.read_text())
        ledger_input = input_rows.get("verified-ledger.json", {})
        if (
            digest(saved_manifest) != value.get("manifest_sha256")
            or manifest.get("commit") != value.get("commit")
            or manifest.get("script") != "research/r32_rehearsal/colab.sh"
            or manifest.get("env", {}).get("R32_MODE") != "main"
            or allocation(attempt_spec) != allocation(spec)
            or ledger_input.get("sha256") != attempt_spec.get("ledger_sha256")
        ):
            raise ValueError("foreign/unbound supervisor receipt")
        ledger_input_path = Path(ledger_input.get("path", "/missing-r32-ledger"))
        if not ledger_input_path.is_file() or digest(ledger_input_path) != ledger_input["sha256"]:
            raise ValueError("unbound supervisor verified-ledger input")
        read_ledger(json.loads(ledger_input_path.read_text()), spec)
        if value.get("status") != "PASS" and value.get("failure_class") not in {
            "RPC_EXEC_POLL_TIMEOUT",
            "UPLOAD_CAP",
            "ACCOUNTING",
            "SUPERVISOR",
        }:
            raise ValueError("unclassified failed supervisor attempt")
        if value.get("status") == "PASS" and any(
            value.get(key) for key in ("error", "cleanup_error", "failure_class")
        ):
            raise ValueError("inconsistent supervisor success receipt")
        if value.get("cleanup_error"):
            raise ValueError("unresolved supervisor cleanup")
        for row in value.get("exports", []):
            name = row.get("path")
            if name and name.startswith("unit-"):
                actual_path = (path.parent / "exports" / name).resolve()
                if actual_path in recovered:
                    raise ValueError("duplicate recovered immutable unit")
                recovered[actual_path] = row
        bound_receipts.append({"path": str(path.resolve()), "sha256": digest(path)})
    for row in rows.values():
        proof = recovered.get(Path(row["path"]).resolve())
        if not proof or proof.get("sha256") != row["sha256"] or proof.get("size") != row["size"]:
            raise ValueError("unit lacks matching supervisor export/release evidence")
    from research.colab_cli.canary import check as check_canary

    canary_paths = set()
    for path in receipts:
        manifest = json.loads((path.parent / "manifest.json").read_text())
        check_canary(manifest, ROOT)
        canary_paths.add(Path(manifest["canary"]["receipt"]["path"]).resolve())
    series_charges = verify_receipts([*receipts, *sorted(canary_paths)])
    if (
        sum(row.elapsed_seconds for row in series_charges) > 288 * 60
        or sum(row.units for row in series_charges) > 0.384
        or any(row.own_rate_units_per_hour <= 0 for row in series_charges)
    ):
        raise ValueError("shared R32 CPU envelope exceeded including canaries")
    # Verify the producer's archived source closure, not just the current checkout.
    for path in receipts:
        receipt = json.loads(path.read_text())
        try:
            archive = subprocess.check_output(
                ["git", "archive", receipt["commit"], *spec["sources"]],
                cwd=ROOT,
                stderr=subprocess.PIPE,
                timeout=30,
            )
        except (subprocess.SubprocessError, KeyError) as exc:
            raise ValueError("supervisor producer source proof unavailable") from exc
        with tarfile.open(fileobj=io.BytesIO(archive)) as bundle:
            for name, expected_sha in spec["sources"].items():
                stream = bundle.extractfile(name)
                if stream is None or hashlib.sha256(stream.read()).hexdigest() != expected_sha:
                    raise ValueError("supervisor producer source closure differs")
    # Fixed adjacent create-only claim is independent of the caller's output filename.
    claim = inputs / "terminal-look.json"
    if output.exists() or claim.exists():
        raise FileExistsError("terminal look already claimed")
    for row in rows.values():
        path = Path(row["path"])
        if path.stat().st_size != row["size"] or verify_export(path) != row["sha256"]:
            raise ValueError("verified immutable unit changed")
    export_json(claim, {"allocation": allocation(spec), "ledger_sha256": digest(ledger_path)})
    values = {str(index): json.loads(Path(row["path"]).read_text()) for index, row in rows.items()}
    value = {
        "schema": "r32-terminal/v1",
        "synthetic": True,
        "scientific_result": False,
        "consumer_readiness": "BLOCKED_PENDING_MAC_CONSUMER_EDGES",
        "statistics_adoption": "RETAIN_FIXED_DESIGN",
        "units": values,
        "claim_sha256": digest(claim),
        "supervisor_receipts": bound_receipts,
        "execution": disposition(spec, ledger, stop_verified=True, outcome="clean"),
        "runtime_release": "CHECK_SUPERVISOR_RECEIPTS_SEPARATELY",
    }
    export_json(output, value)
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("tiny", "canary", "main"), default="tiny")
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    args = parser.parse_args()
    # Nice is inherited; no call can raise this task's priority.
    if hasattr(os, "getpriority") and os.getpriority(os.PRIO_PROCESS, 0) < 15:
        os.nice(15 - os.getpriority(os.PRIO_PROCESS, 0))
    if args.ledger:
        result = reduce_verified(args.inputs, args.ledger, args.output, args.receipt)
    else:
        result = run_portable(args.inputs, args.output, mode=args.mode)
    print(json.dumps({k: v for k, v in result.items() if k in ("done", "expected", "synthetic")}))


if __name__ == "__main__":
    main()
