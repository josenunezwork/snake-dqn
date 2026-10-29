"""Offline custody and timing assessment of excluded P engineering profiles only."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import statistics
import subprocess
import tarfile
from pathlib import Path

from research.bl_served_search.exports import digest, verify_export
from research.watch_lead_p_successor.analyse import analyse
from research.watch_lead_p_successor.profile import concurrency

ROOT = Path(__file__).resolve().parents[2]
PHASES = ("contention", "bare", "tail")
IDENTITY = ("seed", "arm", "frames", "initial_state", "final_state", "trace_sha256")


def read(path: Path) -> dict:
    """Read public engineering metadata, never scientific session payloads."""
    return json.loads(path.read_text())


def inventory(output: Path) -> list[dict]:
    """Require exact equality with receipt exports and complete safe triples."""
    profile = output / "exports/profile"
    files = sorted(profile.glob("*.json"))
    expected = {p.name + suffix for p in files for suffix in ("", ".sha256", ".complete")}
    if {p.name for p in profile.iterdir()} != expected:
        raise ValueError("foreign or orphan profile component")
    rows = [
        {"path": f"profile/{p.name}", "size": p.stat().st_size, "sha256": verify_export(p)}
        for p in files
    ]
    receipt_rows = read(output / "receipt.json")["exports"]
    if sorted(receipt_rows, key=lambda r: r["path"]) != rows:
        raise ValueError("receipt/export inventory mismatch or duplicate")
    return rows


def cell(profile: Path, name: str) -> dict:
    """Recheck individual rows, overlap arithmetic and single-thread barriers."""
    value = read(profile / f"{name}.json")
    rows = value["rows"]
    if [r["task_index"] for r in rows] != list(range(len(rows))):
        raise ValueError("duplicate or missing task index")
    if len(list(profile.glob(f"{name}-task*.json"))) != len(rows):
        raise ValueError("unexpected task exports")
    for row in rows:
        task = read(profile / f"{name}-task{row['task_index']}.json")
        if {k: v for k, v in row.items() if k != "mean_concurrent_tasks"} != task:
            raise ValueError("sweep/task mismatch")
    checked = [dict(row) for row in rows]
    concurrency(checked)
    if any(
        abs(a["mean_concurrent_tasks"] - b["mean_concurrent_tasks"]) > 1e-10
        for a, b in zip(rows, checked)
    ):
        raise ValueError("incorrect overlap")
    if any(
        r["torch_threads"] != 1 or r["torch_interop_threads"] != 1 for r in value["worker_boot"]
    ):
        raise ValueError("worker math-thread drift")
    samples = value["samples"]
    busy = (samples[-1]["child_cpu_seconds"] - samples[0]["child_cpu_seconds"]) / (
        samples[-1]["elapsed"] - samples[0]["elapsed"]
    )
    return {
        "wall_seconds": value["wall_seconds"],
        "pool_spawn_seconds": value["pool_spawn_seconds"],
        "active_cpu_cores": busy,
        "overlap_mean": statistics.mean(r["mean_concurrent_tasks"] for r in rows),
        "rows": rows,
    }


def identities(rows: list[dict]) -> list[tuple]:
    return [tuple(row[k] for k in IDENTITY) for row in rows]


def source_archive(output: Path, pin: str) -> dict:
    """Compare archived files to published Git blobs without extracting them."""
    archive = output / "staging/source.tar.gz"
    if digest(archive) != read(output / "input-hashes.json")["source.tar.gz"]:
        raise ValueError("archive upload hash mismatch")
    count = size = 0
    names = set()
    with tarfile.open(archive, "r:gz") as handle:
        members = [m for m in handle.getmembers() if not m.isdir()]
        batch = subprocess.check_output(
            ["git", "cat-file", "--batch"],
            input="".join(f"{pin}:{m.name}\n" for m in members).encode(),
            cwd=ROOT,
        )
        stream = io.BytesIO(batch)
        for member in members:
            if not member.isfile() or member.name in names:
                raise ValueError("unsafe or duplicate archive member")
            names.add(member.name)
            payload = handle.extractfile(member).read()
            header = stream.readline().split()
            if len(header) != 3 or header[1] != b"blob":
                raise ValueError("missing published archive source")
            blob = stream.read(int(header[2]))
            if stream.read(1) != b"\n":
                raise ValueError("invalid Git batch output")
            if payload != blob or hashlib.sha256(payload).hexdigest() != digest(ROOT / member.name):
                raise ValueError(f"source mismatch: {member.name}")
            count += 1
            size += len(payload)
    return {"sha256": digest(archive), "files": count, "unpacked_bytes": size}


def packet_record(packet: Path, phase: str) -> dict:
    """Bind successful engineering exports to sealed packet, inputs and source."""
    for name, checksum in read(packet / "seal.json").items():
        if digest(packet / name) != checksum:
            raise ValueError(f"packet seal mismatch: {name}")
    output = packet / "l4/output"
    receipt = read(output / "receipt.json")
    if not (
        receipt["status"] == "PASS"
        and receipt["stop_verified"] is True
        and receipt["cleanup_error"] is None
        and receipt["error"] is None
        and receipt["terminal"]["status"] == "succeeded"
        and receipt["terminal"]["returncode"] == 0
    ):
        raise ValueError("unsuccessful or unreleased probe")
    manifest = read(output / "manifest.json")
    if digest(output / "manifest.json") != receipt["manifest_sha256"]:
        raise ValueError("dispatched manifest hash mismatch")
    sealed = read(packet / "l4-manifest.json")
    for key in ("commit", "script", "env", "inputs", "requirements", "required_exports"):
        if manifest[key] != sealed[key]:
            raise ValueError(f"sealed/dispatched identity mismatch: {key}")
    if receipt["commit"] != manifest["commit"]:
        raise ValueError("receipt source mismatch")
    uploaded = read(output / "input-hashes.json")
    for item in manifest["inputs"]:
        if uploaded[f"inputs/{item['dest']}"] != item["sha256"]:
            raise ValueError("uploaded input identity mismatch")
        if digest(Path(item["path"])) != item["sha256"]:
            raise ValueError("local input changed")
    files = inventory(output)
    if not set(manifest["required_exports"]).issubset(r["path"] for r in files):
        raise ValueError("missing required export")
    summary = read(output / "exports/profile/followup.json")
    if not (summary["complete"] and summary["engineering_only"] and not summary["smoke"]):
        raise ValueError("incomplete/nonengineering summary")
    if summary["phase"] != phase:
        raise ValueError("foreign phase")
    evidence = receipt["canary_evidence"]
    if evidence["hardware"] != "L4" or evidence["device"] != "cpu":
        raise ValueError("wrong host/device")
    assignments = receipt["accounting"]["calibration"]["samples"]
    if any(s["usage"]["assignments"] > 3 for s in assignments):
        raise ValueError("account assignment cap exceeded")
    upload = read(output / "upload-measurement.json")
    session_files = list((packet / "l4/state").glob("*/sessions.json"))
    if len(session_files) != 1 or receipt["endpoint"] in read(session_files[0]):
        raise ValueError("retained assignment disappearance not verified")
    return {
        "output": str(output),
        "receipt_sha256": digest(output / "receipt.json"),
        "seal_sha256": digest(packet / "seal.json"),
        "manifest_sha256": receipt["manifest_sha256"],
        "status": receipt["status"],
        "terminal_returncode": receipt["terminal"]["returncode"],
        "stop_verified": receipt["stop_verified"],
        "retained_sessions_sha256": digest(session_files[0]),
        "account_assignments_at_calibration": max(s["usage"]["assignments"] for s in assignments),
        "elapsed_seconds": receipt["elapsed_seconds"],
        "rate_units_per_hour": receipt["accounting"]["own_rate_units_per_hour"],
        "upload": upload,
        "transfer_margin_seconds": 150 - upload["transfer_seconds"],
        "archive": source_archive(output, receipt["commit"]),
        "inventory": files,
        "summary": summary,
        "host": read(output / "exports/profile/host.json"),
    }


def assess(artifacts: Path) -> dict:
    """Combine only the five named engineering outputs; closed 09e is inaccessible here."""
    prior = {
        "A100": analyse(artifacts / "astra-p-successor/packet-v2/a100/output"),
        "L4": analyse(artifacts / "astra-p-successor3/packet-l4c/l4/output"),
    }
    for value in prior.values():
        inventory(Path(value["output"]))
    base = artifacts / "astra-p-successor4/packet-v4"
    probes = {phase: packet_record(base / phase, phase) for phase in PHASES}
    profile = base / "contention/l4/output/exports/profile"
    cells = {w: cell(profile, f"missing-contention-w{w}") for w in (8, 11)}
    if identities(cells[8]["rows"]) != identities(cells[11]["rows"]):
        raise ValueError("contention cross-worker trajectory mismatch")
    # Record cross-allocation disagreement; never silently claim a stitched identity grid.
    old = Path(prior["L4"]["output"]) / "exports/profile"
    cross_allocation = {}
    for w in (1, 4):
        before = read(old / f"contention-w{w}.json")["rows"]
        after = cells[8]["rows"]
        cross_allocation[w] = {
            "full_identity_equal": identities(before) == identities(after),
            "trace_identity_equal": [r["trace_sha256"] for r in before]
            == [r["trace_sha256"] for r in after],
            "different_fields": sorted(
                {key for a, b in zip(before, after) for key in IDENTITY if a[key] != b[key]}
            ),
        }
    tail_profile = base / "tail/l4/output/exports/profile"
    tails = {label: cell(tail_profile, f"tail-{label}-w4") for label in ("peak", "late")}
    for value in tails.values():
        if identities(value["rows"][2:3]) != identities(value["rows"][3:4]):
            raise ValueError("repeated bare trajectory mismatch")
    seed = read(base / "tail/profile-bank.json")["seeds"][0]
    bare = read(base / f"bare/l4/output/exports/profile/scan-{seed}.json")
    tail = read(tail_profile / f"scan-{seed}.json")
    if bare["frames"] != 6000 or tail["frames"] != 5000:
        raise ValueError("scan horizon mismatch")
    return {
        "status": "INSUFFICIENT_FOR_SUCCESSOR_ENVELOPES",
        "no_scientific_payload_opened": True,
        "prior": prior,
        "probes": probes,
        "contention_cells": cells,
        "tail_cells": tails,
        "full_bare": bare,
        "tail_scan": tail,
        "contention_identity_within_new_w8_w11": True,
        "cross_allocation_contention_identity": cross_allocation,
        "tail_roots_identical": tail["selected"]["peak"] == tail["selected"]["late"],
        "long_body_support": tail["selected"]["peak"]["length"] >= 1000,
        "full_searched_sessions_measured": 0,
        "registration_ready_for_ratification": False,
        "scientific_shard_prepared": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifacts", type=Path, required=True, help="t34-sol artifact root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = assess(args.artifacts)
    with args.output.open("x") as handle:
        json.dump(result, handle, sort_keys=True, indent=2, allow_nan=False)
        handle.write("\n")
    print(
        json.dumps(
            {
                "status": result["status"],
                "verified_new_payloads": sum(
                    len(p["inventory"]) for p in result["probes"].values()
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
