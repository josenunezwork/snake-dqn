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
from research.sandbox_a_resident.training_units import (
    COMPACT_DERIVATIVE_SUFFIX,
    COMPACT_MANIFEST_SUFFIX,
    COMPACT_RESUME_FIELD,
    load_ledger,
    validate_compact_resume,
)
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[2]


def validate_uploaded_ledger(packet: Path, broker: dict, control: dict) -> None:
    """Verify relative runtime ledger paths against the actual Mac upload set.

    With an allowlisted ``compact_resume`` binding, the full-state ``.pt`` is not
    uploaded: its derivative and manifest are, as exact hash-bound siblings of
    the Mac original, together with the original's completion sidecars.  The
    Mac original itself is still verified by path/size/hash below.
    """
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    compact = validate_compact_resume(ledger)
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
            if compact is not None and destination == compact["original"]["path"]:
                if destination in uploads:
                    raise ValueError("compact resume must not also upload the full-state payload")
                sources = []
                for role, suffix in (
                    ("derivative", COMPACT_DERIVATIVE_SUFFIX),
                    ("manifest", COMPACT_MANIFEST_SUFFIX),
                ):
                    item = uploads.get(destination + suffix)
                    bound = compact[role]
                    if (
                        item is None
                        or bound["path"] != destination + suffix
                        or item["sha256"] != bound["sha256"]
                        or not item["path"].endswith(suffix)
                        or digest(Path(item["path"])) != bound["sha256"]
                        or Path(item["path"]).stat().st_size != bound["size"]
                    ):
                        raise ValueError(f"compact resume {role} is not a hash-bound upload")
                    sources.append(Path(item["path"][: -len(suffix)]))
                    referenced.add(destination + suffix)
                if sources[0] != sources[1]:
                    raise ValueError("compact derivative and manifest must be original siblings")
                source = sources[0]
                for suffix in (".sha256", ".complete"):
                    item = uploads.get(destination + suffix)
                    if item is None or Path(item["path"]) != Path(str(source) + suffix):
                        raise ValueError("resume sidecars must be exact uploaded siblings")
                    referenced.add(destination + suffix)
                row["path"] = str(source)
                continue
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
    # The compact binding was validated against relative paths above; the Mac
    # copy below verifies the local original by path/size/hash only.
    ledger.pop(COMPACT_RESUME_FIELD, None)
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
    parser.add_argument(
        "--cuda-semantics",
        action="store_true",
        help="run strict CUDA API-contract mocks without a GPU",
    )
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
    cuda_contract_tests = [
        "research/sandbox_a_resident/test_training_cuda_semantics.py",
        "research/sandbox_a_resident/test_training_learner.py"
        "::test_critic_quality_baseline_has_cuda_like_same_device_semantics",
        "research/sandbox_a_resident/test_training_cpu_reference.py"
        "::test_cpu_reference_can_route_frozen_inference_to_cuda_semantics",
        "research/sandbox_a_resident/test_training_rosters.py"
        "::test_runtime_resume_normalizes_relocated_row_identity_to_cpu",
    ]
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
            *(
                cuda_contract_tests
                if args.cuda_semantics
                else ["research/sandbox_a_resident", "tools/sandbox/tests/test_source_delivery.py"]
            ),
        ],
        cwd=ROOT,
        env=env,
        check=True,
    )
    label = "PASS_TRAIN01_CUDA_SEMANTICS" if args.cuda_semantics else "PASS_TRAIN01_PREFLIGHT"
    print(f"{label} seeds={check['count']} schema=PASS registry=PASS tests=PASS allocations=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
