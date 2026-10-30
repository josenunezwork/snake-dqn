"""Prepare branch-pinned Colab packets; never allocate, dispatch, or alter serving."""

from __future__ import annotations

import argparse
import gzip
import io
import json
import math
import subprocess
import tarfile
from pathlib import Path

from research.colab_cli import canary
from research.watch_replay.run import ROOT, allocation, completed, identity
from research.watch_replay.source import source_paths
from tools.ops.package_spec import Input, digest, empty_ledger, publish_json, stage_inputs


def release_bundle(bundle: Path, pins: dict) -> None:
    """Keep upload bytes identical when binding a canary in a fresh main packet."""
    with (
        bundle.open("xb") as stream,
        gzip.GzipFile(fileobj=stream, filename="", mode="wb", mtime=0) as compressed,
    ):
        with tarfile.open(fileobj=compressed, mode="w") as archive:
            for name, pin in sorted(pins["checkpoints"].items()):
                for ref in pin["strict_receipt"].values():
                    if isinstance(ref, dict):
                        path = Path(pin["strict_receipt"]["root"]) / ref["path"]
                        if digest(path) != ref["sha256"]:
                            raise ValueError("release anchor hash mismatch")
                        member = tarfile.TarInfo(name + "/" + ref["path"])
                        member.size = path.stat().st_size
                        archive.addfile(member, io.BytesIO(path.read_bytes()))


def prepare(
    out: Path,
    commit: str,
    benchmark: Path,
    namespace: str,
    count: int,
    controller: str,
    canary_receipt: Path | None = None,
    ledger: Path | None = None,
    previous_receipts: list[Path] | None = None,
) -> dict:
    """Ship the identical inventory for a full-length canary and N sessions."""
    if not 1 <= count <= 12:
        raise ValueError("one packet contains 1..12 sessions within bounded exports")
    refs = subprocess.check_output(
        ["git", "branch", "-r", "--contains", commit], cwd=ROOT, text=True
    )
    if len(commit) != 40 or "origin/astra-watch-replay" not in refs:
        raise ValueError("requires full published branch pin")
    if subprocess.check_output(["git", "diff", commit, "--", *source_paths()], cwd=ROOT):
        raise ValueError("working executable bytes differ from source pin")
    measured = json.loads(benchmark.read_text())
    if measured["commit"] != commit:
        raise ValueError("Mac preflight must match current commit")
    out.mkdir(parents=True, exist_ok=False)
    contracts = {
        "main": {**allocation(namespace, count, controller), "commit": commit},
        "canary": {**allocation(namespace + "-canary", 1, controller), "commit": commit},
    }
    previous_receipts = previous_receipts or []
    if len(previous_receipts) > 2 or bool(previous_receipts) != bool(ledger):
        raise ValueError(
            "one original plus at most two completion attempts; ledger requires receipts"
        )
    verified = []
    if ledger:
        verified = completed(ledger, contracts["main"])
        if len(verified) == count:
            raise ValueError("completion requires missing units")
        hashes = set()
        exports = {}
        for attempt, receipt_path in enumerate(previous_receipts, 1):
            receipt = json.loads(receipt_path.read_text())
            former_path = receipt_path.parent / "manifest.json"
            former = json.loads(former_path.read_text())
            error = str(receipt.get("error", "")).upper()
            if (
                digest(receipt_path) in hashes
                or digest(former_path) != receipt.get("manifest_sha256")
                or former.get("commit") != commit
                or former.get("watch_replay", {}).get("allocation") != identity(contracts["main"])
                or former.get("watch_replay", {}).get("attempt") != attempt
                or receipt.get("stop_verified") is not True
                or receipt.get("status") == "PASS"
                or not any(
                    word in error
                    for word in (
                        "TIMEOUT",
                        "LEASE",
                        "TRANSFER",
                        "UPLOAD",
                        "STAGING",
                        "INTERRUPT",
                        "RUNTIME_LOST",
                    )
                )
            ):
                raise ValueError("completion requires exact stopped mechanically failed lineage")
            hashes.add(digest(receipt_path))
            for row in receipt.get("exports", []):
                name = row.get("path", "")
                if name.startswith("sessions/") and name.endswith(".watch.gz"):
                    index = int(Path(name).name.split(".")[0])
                    if index in exports:
                        raise ValueError("duplicate completed units across attempts")
                    exports[index] = {"sha256": row["sha256"], "size": row["size"]}
        if {r["index"]: {k: r[k] for k in ("sha256", "size")} for r in verified} != exports:
            raise ValueError("ledger must contain exactly all verified prior receipt units")
    inputs = []
    for label, contract in contracts.items():
        publish_json(out / f"{label}-contract.json", contract)
        publish_json(
            out / f"{label}-ledger.json",
            (
                json.loads(ledger.read_text())
                if label == "main" and ledger
                else empty_ledger(contract["namespace"], "sessions", identity(contract))
            ),
        )
        inputs.extend(
            [
                Input(out / f"{label}-contract.json", f"{label}-contract.json"),
                Input(out / f"{label}-ledger.json", f"{label}-ledger.json"),
            ]
        )
    saved = ROOT.parent / "snake-dqn/saved_snakes"
    for name in ("b8_s3_t7_20261016.pth", "frp3_m3_s12_u60000_20261005.pth"):
        inputs.append(Input(saved / name, "checkpoints/" + name))
    pins = json.loads((ROOT / "web/backend/served_checkpoint_pins.json").read_text())
    bundle = out / "release-evidence.tar.gz"
    release_bundle(bundle, pins)
    inputs.append(Input(bundle, "release-evidence.tar.gz"))
    staged = stage_inputs(out / "staged", inputs)
    singleton = max(t["seconds"] / t["branches"] for t in measured["branch_timings"])
    # Planning assumptions, explicitly not measured Colab throughput or a hard bound.
    session_seconds = 1200 * 1.2 * singleton * 3 * 1.5
    minutes = math.ceil(session_seconds * math.ceil(count / 4) / 60 + 20)
    if minutes > 600:
        raise ValueError("forecast exceeds bounded runtime; use separate frozen packets")
    forecast = {
        "mac_h300_singleton_seconds": singleton,
        "roots_per_session_max": 1200,
        "branches_per_root_forecast": 1.2,
        "branches_per_root_max": 6,
        "cloud_slowdown_projection": 3,
        "contention_projection": 1.5,
        "workers": 4,
        "base_minutes": minutes,
        "effective_minutes": 2 * minutes,
        "expected_sessions_per_colab_hour": 4 * 3600 / session_seconds,
        "scope": "projection, not measured Colab throughput; excludes setup/export",
    }
    publish_json(out / "forecast.json", forecast)
    for label in ("canary", "main"):
        base = 15 if label == "canary" else minutes
        rate = 1.54
        unit = base / 60 * rate
        manifest = {
            "job_id": f"watch-replay-{label}-{out.name}",
            "commit": commit,
            "script": "research/watch_replay/job.sh",
            "source_paths": source_paths(),
            "device": "cpu",
            "lease_seconds": 300,
            "timeout_minutes": 2 * base,
            "output_dir": str(out / f"output-{label}"),
            "max_export_bytes": min(
                2 * 1024**3, max(4 * 1024**2, contracts[label]["count"] * 128 * 1024**2 + 1024**2)
            ),
            "requirements": {
                "gpu": "L4",
                "allowed_gpus": ["L4"],
                "min_vcpu": 12,
                "est_minutes": base,
            },
            "budget": {
                "unit_cap": 2 * unit,
                "reserve_units": 2 * unit * 0.2,
                "rate_units_per_hour": rate,
                "other_assignments": {},
            },
            "watch_replay": {
                "allocation": identity(contracts[label]),
                "attempt": len(previous_receipts) + 1,
                "base_minutes": base,
                "base_unit_cap": unit,
                "base_reserve_units": unit * 0.2,
                "forecast": forecast,
            },
            "resume": {
                "unit_kind": "watch-session",
                "ledger_path": f"inputs/{label}-ledger.json",
                "skip_verified": True,
                "max_outcome_blind_retries": 2,
            },
            "required_exports": ["sessions/operational.json", "sessions/contract.json"],
            "inputs": [{k: row[k] for k in ("path", "dest", "sha256")} for row in staged],
            "env": {
                "COLAB_CLI_REQUIREMENTS": "research/bl_served_search/requirements.txt",
                "PYTHONUNBUFFERED": "1",
                "SNAKE_DQN_DEVICE": "cpu",
                "WATCH_REPLAY_CONTRACT": f"{label}-contract.json",
                "WATCH_REPLAY_LEDGER": f"{label}-ledger.json",
                "WATCH_REPLAY_WORKERS": "1" if label == "canary" else "4",
            },
        }
        skipped = {row["index"] for row in verified} if label == "main" else set()
        if label == "canary":
            manifest["required_exports"].extend(
                f"sessions/{row['index']:06d}.watch.gz"
                for row in contracts[label]["rows"]
                if row["index"] not in skipped
            )
        if label == "canary":
            manifest["env"].update(
                SNAKE_COLAB_CANARY="1", SNAKE_COLAB_CANARY_PACKAGE="research/watch_replay"
            )
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "mode": "probe",
                "family": canary.family(manifest),
            }
            manifest["required_exports"].append("branches.json")
        elif canary_receipt is not None:
            manifest["canary"] = {
                "schema": canary.SCHEMA,
                "receipt": {
                    "path": str(canary_receipt.resolve()),
                    "sha256": digest(canary_receipt),
                },
            }
        (out / f"{label}.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return forecast


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--benchmark", type=Path, required=True)
    parser.add_argument("--namespace", default="t34-watch-replay-fix-20261030-bank01")
    parser.add_argument("--count", type=int, default=12)
    parser.add_argument(
        "--controller", choices=["exact-stop-l2-prime", "bare-b8"], default="exact-stop-l2-prime"
    )
    parser.add_argument("--canary-receipt", type=Path)
    parser.add_argument("--ledger", type=Path)
    parser.add_argument("--previous-receipts", type=Path, nargs="*", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(
                args.out.resolve(),
                args.commit,
                args.benchmark,
                args.namespace,
                args.count,
                args.controller,
                args.canary_receipt,
                args.ledger,
                args.previous_receipts,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
