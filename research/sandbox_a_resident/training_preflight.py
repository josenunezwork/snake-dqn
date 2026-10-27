"""Offline TRAIN-01 schema, source, registry and executable-test admission."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

from research.sandbox_a_resident.training_prepare import STUDY, seed_allocations
from research.sandbox_a_resident.training_units import load_ledger
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[2]


def validate_uploaded_ledger(packet: Path, broker: dict, control: dict) -> None:
    """Verify relative runtime ledger paths against the actual Mac upload set."""
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    uploads = {row["dest"]: row for row in broker["inputs"]}
    if len(uploads) != len(broker["inputs"]):
        raise ValueError("duplicate runtime input destination")
    referenced = set()
    for unit in ledger.get("units", []):
        expected = [
            f"resume/iteration-{unit['iteration']:04d}{suffix}"
            for suffix in (".pt", ".json", "-actor.pth")
        ]
        if [row["path"] for row in unit["exports"]] != expected:
            raise ValueError("complete canonical snapshot triple required")
        for row in unit["exports"]:
            destination = row["path"]
            upload = uploads.get(destination)
            if upload is None or row["sha256"] != upload["sha256"]:
                raise ValueError("resume payload is not a hash-bound uploaded input")
            source = Path(upload["path"])
            for suffix in ("", ".sha256", ".complete"):
                name = destination + suffix
                item = uploads.get(name)
                if item is None or Path(item["path"]) != Path(str(source) + suffix):
                    raise ValueError("resume sidecars must be exact uploaded siblings")
                referenced.add(name)
            row["path"] = str(source)
    if referenced != {name for name in uploads if name.startswith("resume/")}:
        raise ValueError("unreferenced resume upload")
    # Only metadata/path/size/hash are examined; Torch payloads stay opaque.
    with tempfile.TemporaryDirectory(prefix="train01-verify-ledger-") as directory:
        path = Path(directory) / "ledger.json"
        path.write_text(json.dumps(ledger))
        load_ledger(path, control)


def validate_packet(packet: Path) -> dict[str, Any]:
    """Bind the actual runtime controls to the exhaustive committed seed audit."""
    from research.sandbox_a_resident.train import TrainControl

    payload = preflight(packet / "sandbox.json", run_tests=False)
    broker = json.loads((packet / "dispatch-manifest.json").read_text())
    control = json.loads((packet / "control.json").read_text())
    default = asdict(TrainControl(study=STUDY, source_commit=broker["commit"], allocation_index=0))
    if control != default or control != json.loads(broker["env"]["SANDBOX_A_CONTROL_JSON"]):
        raise ValueError("TRAIN-01 control differs from frozen production recipe")
    if payload["study"] != STUDY or broker["script"] != "research/sandbox_a_resident/train_job.sh":
        raise ValueError("wrong training study/entrypoint")
    if broker["env"]["SANDBOX_A_SOURCE_COMMIT"] != broker["commit"]:
        raise ValueError("runtime source pin differs")
    if payload["endpoint"] != {"kind": "steps", "value": 100_663_296}:
        raise ValueError("wrong finite actor endpoint")
    for name in ("control.json", "registry-check.json", "verified-ledger.json"):
        verify_committed_manifest(packet / name)
    check = json.loads((packet / "registry-check.json").read_text())
    expected = seed_allocations(control)
    registry = next(r for r in payload["inputs"] if r["kind"] == "registry")
    if (
        check.get("schema") != "sandbox-a-train-seed-audit/v1"
        or check.get("status") != "PASS"
        or check.get("study") != STUDY
        or check.get("allocations") != expected
        or check.get("count") != sum(row["count"] for row in expected)
        or check.get("registry_sha256") != registry["sha256"]
        or check.get("effective_seed_bits") != 63
        or check.get("aliases") != 0
        or check.get("excluded_intersection") != 0
        or broker["sandbox_a"].get("seed_allocations") != expected
        or broker["sandbox_a"].get("seed_audit_sha256") != digest(packet / "registry-check.json")
    ):
        raise ValueError("exhaustive seed audit does not bind this allocation/registry")
    validate_uploaded_ledger(packet, broker, control)
    return check


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    check = validate_packet(args.packet.resolve())
    env = dict(os.environ)
    for key in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMBA_NUM_THREADS",
        "OMP_THREAD_LIMIT",
    ):
        env[key] = "1"
    env["SNAKE_DQN_DEVICE"] = "cpu"
    subprocess.run(
        [
            "nice",
            "-n",
            "15",
            str(ROOT / "venv/bin/python"),
            "-m",
            "pytest",
            "-q",
            "-o",
            "addopts=",
            "-p",
            "no:xdist",
            "research/sandbox_a_resident",
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    print(
        f"PASS_TRAIN01_PREFLIGHT seeds={check['count']} schema=PASS registry=PASS tests=PASS "
        "allocations=0"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
