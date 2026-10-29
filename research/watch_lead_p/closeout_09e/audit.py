"""Offline, outcome-blind custody and runtime audit; never a scientific reducer."""

from __future__ import annotations

import argparse
import csv
import hashlib
import heapq
import json
import re
import statistics
import subprocess
import tarfile
from pathlib import Path
from typing import Any

PIN = "5028a996bb94ec47823627cc113ab6dad6073157"
ARMS = ("P0", "P48", "bare")
SUFFIXES = (
    "",
    ".sha256",
    ".complete",
    ".metadata.json",
    ".metadata.json.sha256",
    ".metadata.json.complete",
)


def sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def read_metadata(path: Path) -> Any:
    """Deserialize operational metadata only, never a session outcome payload."""
    require(not re.fullmatch(r"session-\d+\.json", path.name), "outcome read prohibited")
    return json.loads(path.read_text())


def triplet(path: Path) -> dict[str, Any]:
    for suffix in ("", ".sha256", ".complete"):
        part = Path(str(path) + suffix)
        require(part.is_file() and not part.is_symlink(), f"unsafe/missing {part}")
    digest = sha(path)
    require(Path(str(path) + ".complete").read_bytes() == b"complete\n", "bad marker")
    require(
        Path(str(path) + ".sha256").read_text().split() == [digest, path.name], "bad digest/name"
    )
    return {"sha256": digest, "size": path.stat().st_size}


def verify_unit(path: Path, index: int, ledger: dict) -> dict:
    payload = triplet(path)
    metadata_path = Path(str(path) + ".metadata.json")
    triplet(metadata_path)
    metadata = read_metadata(metadata_path)
    expected = {
        "unit_id": f"{ledger['namespace']}{ledger['allocation']}/{index:02d}",
        "construction": ledger["construction"],
        **payload,
    }
    require(metadata == expected, "construction/namespace/index metadata mismatch")
    return {
        "allocation_index": index,
        "unit_id": expected["unit_id"],
        "path": f"units/{path.name}",
        **payload,
    }


def write_json(path: Path, value: Any) -> None:
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def project(costs: dict[str, float], workers: int) -> float:
    loads = [0.0] * workers
    for start in range(0, 32, workers):
        for arm in ARMS:
            for _ in range(start, min(start + workers, 32)):
                heapq.heappush(loads, heapq.heappop(loads) + costs[arm])
    return max(loads) + 1200


def audit(repo: Path, artifacts: Path, output: Path) -> None:
    """Import verified opaque units and report only integrity/progress/timing."""
    require(not output.exists(), "fresh closeout directory required")
    packet = repo / "research/watch_lead_p/packet-09/reseal-05"
    ledger = read_metadata(packet / "verified-ledger.json")
    accounting = read_metadata(packet / "retry-accounting.json")
    require(
        accounting["completion_attempts_remaining_after_execution"] == 0,
        "unexpected attempt accounting",
    )
    for name, digest in read_metadata(packet / "seal.json").items():
        require(sha(packet / name) == digest, f"packet seal mismatch: {name}")

    run_root = artifacts / "t34-watch-lead-1028"
    run_output = run_root / "attempt-09e/output"
    receipt = read_metadata(run_output / "receipt.json")
    require(receipt["commit"] == PIN, "runtime pin mismatch")
    manifest = read_metadata(run_output / "manifest.json")
    require(
        sha(run_output / "manifest.json") == receipt["manifest_sha256"],
        "actual dispatch manifest mismatch",
    )
    require(
        manifest["commit"] == PIN and manifest["job_id"] == receipt["job_id"],
        "manifest identity mismatch",
    )
    require(
        receipt["status"] == "INCOMPLETE"
        and receipt["stop_verified"] is True
        and receipt["cleanup_error"] is None,
        "unexpected terminal custody",
    )
    for item in manifest["inputs"]:
        require(sha(Path(item["path"])) == item["sha256"], "input byte drift")
    require(
        read_metadata(Path(manifest["inputs"][-1]["path"])) == ledger, "dispatched ledger differs"
    )

    plan = read_metadata(run_output / "staging/source-plan.json")
    archive = run_output / "staging/source.tar.gz"
    require(
        plan["commit"] == PIN and sha(archive) == plan["fallback_sha256"],
        "source archive identity mismatch",
    )
    with tarfile.open(archive) as tar:
        members = tar.getmembers()
        require(len({m.name for m in members}) == len(members), "duplicate source path")
        require(set(tar.getnames()) == set(plan["files"]), "source inventory mismatch")
        for member in members:
            require(member.isfile(), "non-file source entry")
            data = tar.extractfile(member).read()
            pinned = subprocess.run(
                ["git", "show", f"{PIN}:{member.name}"], cwd=repo, check=True, capture_output=True
            ).stdout
            require(
                data == pinned == (repo / member.name).read_bytes(),
                f"runtime/current/pinned source mismatch: {member.name}",
            )
            info = plan["files"][member.name]
            require(
                len(data) == info["size"] and hashlib.sha256(data).hexdigest() == info["sha256"],
                "source-plan content mismatch",
            )
    sources = read_metadata(repo / "research/watch_lead_p/science-sources.json")
    identities = [(name, sha(repo / name)) for name in sources]
    recipe = hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode()).hexdigest()
    require(recipe == ledger["recipe_sha256"], "scientific identity mismatch")

    attempts = {}
    evidence = []
    for attempt in ("06b", "09c", "09e"):
        root = run_root / f"attempt-{attempt}/output"
        r = read_metadata(root / "receipt.json")
        records = r["exports"]
        expected_files = set()
        for record in records:
            relative = Path(record["path"])
            require(not relative.is_absolute() and ".." not in relative.parts, "unsafe export")
            p = root / "exports" / relative
            require(
                triplet(p) == {k: record[k] for k in ("size", "sha256")}, "receipt/export mismatch"
            )
            expected_files.update(str(p) + suffix for suffix in ("", ".sha256", ".complete"))
        export_root = root / "exports"
        actual_files = {str(p) for p in export_root.rglob("*") if p.is_file()}
        require(actual_files == expected_files, "unaccounted/partial retained export")
        attempts[attempt] = {
            k: r[k]
            for k in (
                "status",
                "failure_class",
                "stop_verified",
                "cleanup_error",
                "elapsed_seconds",
            )
        }
        attempts[attempt]["export_records"] = len(records)
        evidence.append({"path": str(root / "receipt.json"), "sha256": sha(root / "receipt.json")})

    units_root = run_output / "exports/watch-lead/units"
    paths = sorted(
        p for p in units_root.glob("session-*.json") if re.fullmatch(r"session-\d{2}\.json", p.name)
    )
    indices = [int(p.stem[-2:]) for p in paths]
    require(
        len(indices) == len(set(indices)) and set(indices) <= set(range(32)),
        "duplicate/foreign allocation",
    )
    units = [verify_unit(path, index, ledger) for path, index in zip(paths, indices)]
    require(len(units) == 21, "handoff/export count mismatch")

    log = artifacts / "ops/logs/sandbox-p-watch-lead-20261028-09e.log"
    text = log.read_text()
    sandbox = read_metadata(packet / "sandbox.json")
    seeds = next(i["seeds"] for i in sandbox["inputs"] if i["kind"] == "worlds")
    seed_index = {seed: i for i, seed in enumerate(seeds)}
    pulses = {}
    unique = set()
    pattern = r"P_PROGRESS arm=(bare|P0|P48) frame=(\d+)/(\d+) " r"seed=(\d+) elapsed=([\d.]+)s"
    for match in re.finditer(pattern, text):
        arm, frame, horizon, seed, elapsed = match.groups()
        require(int(horizon) == 6000 and int(seed) in seed_index, "foreign progress")
        key = (seed_index[int(seed)], arm)
        row = (int(frame), float(elapsed))
        unique.add((*key, *row))
        if key not in pulses or row[1] > pulses[key][1]:
            pulses[key] = row
    runtime = [
        (int(m[1]), float(m[2]))
        for m in re.finditer(
            r"P_RUNTIME completed=(\d+) skipped=0 total=32 elapsed=([\d.]+)s", text
        )
    ]
    require(max(n for n, _ in runtime) == len(units), "progress/export mismatch")
    forecast = read_metadata(packet / "forecast.json")
    times = []
    for index in range(32):
        for arm in ARMS:
            pulse = pulses.get((index, arm))
            times.append(
                {
                    "session": index,
                    "arm": arm,
                    "retained_complete": index in indices,
                    "last_observed_frame": pulse[0] if pulse else None,
                    "last_elapsed_seconds": pulse[1] if pulse else None,
                    "exact_completion_seconds": None,
                    "forecast_seconds": forecast["arm_seconds"][arm],
                }
            )
    summary = {}
    for arm in ARMS:
        elapsed = [
            r["last_elapsed_seconds"] for r in times if r["arm"] == arm and r["retained_complete"]
        ]
        require(all(v is not None for v in elapsed), "completed arm lacks timing prefix")
        summary[arm] = {
            "min_last_elapsed_seconds": min(elapsed),
            "median_last_elapsed_seconds": statistics.median(elapsed),
            "max_last_elapsed_seconds": max(elapsed),
            "forecast_seconds": forecast["arm_seconds"][arm],
            "median_lower_bound_ratio": statistics.median(elapsed) / forecast["arm_seconds"][arm],
        }

    output.mkdir(parents=True)
    imported = output / "imported/units"
    imported.mkdir(parents=True)
    files = []
    for path in paths:
        for suffix in SUFFIXES:
            source = Path(str(path) + suffix)
            dest = imported / source.name
            with dest.open("xb") as handle:
                handle.write(source.read_bytes())
            require(sha(dest) == sha(source), "import altered bytes")
            files.append(
                {
                    "source": str(source),
                    "imported": str(dest),
                    "size": source.stat().st_size,
                    "sha256": sha(source),
                }
            )
    write_json(output / "imported/verified-ledger-audit.json", {**ledger, "units": units})
    with (output / "arm-timings.csv").open("x") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(times[0]))
        writer.writeheader()
        writer.writerows(times)
    calls = receipt["cli_calls"]
    cli = {
        op: {
            "count": sum(c["operation"] == op for c in calls),
            "failed": sum(c["operation"] == op and c["status"] == "failed" for c in calls),
            "sum_call_seconds_overlaps_worker_execution": sum(
                c["elapsed_seconds"] for c in calls if c["operation"] == op
            ),
        }
        for op in sorted({c["operation"] for c in calls})
    }
    upper_observed_costs = {arm: summary[arm]["max_last_elapsed_seconds"] + 60 for arm in ARMS}
    evidence.extend(
        {"path": str(p), "sha256": sha(p)}
        for p in [
            log,
            archive,
            packet / "forecast.json",
            packet / "retry-accounting.json",
            run_output / "manifest.json",
            repo / "research/watch_lead_p/README.md",
            repo / "research/watch_lead_p/resume.py",
            repo / "research/watch_lead_p/protocol.py",
        ]
    )
    write_json(
        output / "audit.json",
        {
            "schema": "watch-lead-p-outcome-blind-closeout/v1",
            "runtime_pin": PIN,
            "decision": "INVALID/INCOMPLETE",
            "scientific_look_taken": False,
            "outcome_payloads_deserialized": 0,
            "complete_sessions": indices,
            "missing_sessions": sorted(set(range(32)) - set(indices)),
            "retained_complete_arm_count": 3 * len(indices),
            "missing_whole_session_arm_slots": 3 * (32 - len(indices)),
            "unaccounted_or_partial_retained_files": [],
            "files": files,
            "evidence": evidence,
            "source_archive_verified_files": len(plan["files"]),
            "recipe_sha256": recipe,
            "attempts": attempts,
            "manual_completions_remaining": 0,
            "arm_timing_lower_bounds": summary,
            "unique_progress_pulses": len(unique),
            "publication_milestones_seconds": {
                str(n): min(t for count, t in runtime if count >= n) for n in (1, 11, 12, 20, 21)
            },
            "last_runtime_elapsed_seconds": max(t for _, t in runtime),
            "receipt_elapsed_to_forecast_ratio": receipt["elapsed_seconds"]
            / forecast["projected_seconds"],
            "cli_calls": cli,
            "illustrative_observed_max_plus_60_schedule_seconds": project(upper_observed_costs, 11),
            "illustrative_one_session_three_arm_seconds": sum(upper_observed_costs.values()),
            "limits": [
                "Payloads hashed/copied without JSON parsing; scientific validity not adjudicated.",
                "Three-arm completion follows pinned publisher/markers, not a scientific look.",
                "Logs lack arm completion events; times are last observed lower bounds.",
                "Absent pulse proves no retained log observation, not that an arm never started.",
                "Missing sessions have no unit; individual finished arms are not recoverable.",
                "Max-plus-60 illustrates gameplay cost, not a tail bound or full-run forecast.",
            ],
        },
    )
    print(
        f"Verified/imported {len(units)} sessions, {len(files)} components; "
        "INVALID/INCOMPLETE; no look"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[3])
    parser.add_argument("--artifacts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit(args.repo.resolve(), args.artifacts.resolve(), args.output.resolve())


if __name__ == "__main__":
    main()
