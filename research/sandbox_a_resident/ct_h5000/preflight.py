"""Offline, committed packet admission and actual tiny entrypoint qualification."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from dataclasses import replace
from pathlib import Path
from typing import Any

from research.compute.gpu_preference import variants
from research.sandbox_a_resident.ct_h5000.prepare import STUDY, ct_endpoint_steps, source_paths
from research.sandbox_a_resident.ct_h5000.task import CTControl, ct_seed_allocations
from research.sandbox_a_resident.ct_h5000.units import empty_ledger, load_ledger
from research.sandbox_a_resident.ct_topology import TOPOLOGY_SPEC
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

ROOT = Path(__file__).resolve().parents[3]
TESTS = (
    "research/sandbox_a_resident/ct_h5000/test_successor.py",
    "research/sandbox_a_resident/test_ct_mechanics_and_snapshots.py",
    "research/sandbox_a_resident/test_ct_branching_fit_and_stats.py",
    "research/sandbox_a_resident/test_ct_cuda_semantics.py",
    "research/sandbox_a_resident/test_ct_runner_and_resume.py",
    "research/sandbox_a_resident/test_ct_gpu_preference.py",
    "research/sandbox_a_resident/test_stall_watchdog.py",
)


def validate_packet(packet: Path, *, require_committed: bool = True) -> dict[str, Any]:
    """Fail closed on control, namespace, inputs, seals, source or derived hardware drift."""
    payload = preflight(
        packet / "sandbox.json", require_committed=require_committed, run_tests=False
    )
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    seal = json.loads((packet / "SEAL.json").read_text())
    names = {
        "export-projection.json",
        "control.json",
        "registry-check.json",
        "topology-manifest.json",
        "verified-ledger.json",
        "dispatch-manifest.json",
        "sandbox.json",
        "a100-fallback-preview.json",
    }
    if (
        seal.get("schema") != "t34-ct-h5000-prep-seal/v1"
        or set(seal.get("components", {})) != names
    ):
        raise ValueError("mandatory successor seal components missing")
    for name in names | {"SEAL.json"}:
        path = packet / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing or symlinked packet component")
        verify_committed_manifest(path, require_committed=require_committed)
        if name != "SEAL.json" and seal["components"][name] != digest(path):
            raise ValueError("successor seal component hash mismatch")
    pin = manifest["commit"]
    expected = CTControl(source_commit=pin).to_dict()
    control = json.loads((packet / "control.json").read_text())
    if control != expected or json.loads(manifest["env"]["SANDBOX_CT_CONTROL_JSON"]) != expected:
        raise ValueError("successor control differs from frozen recipe")
    if (
        payload["study"] != STUDY
        or manifest["script"] != "research/sandbox_a_resident/ct_h5000/job.sh"
    ):
        raise ValueError("wrong successor study or entrypoint")
    if manifest["env"]["SANDBOX_A_SOURCE_COMMIT"] != pin or seal["source_commit"] != pin:
        raise ValueError("successor source pin differs")
    from research.sandbox_a_resident.ct_h5000.export_projection import project_stage_bytes

    if json.loads((packet / "export-projection.json").read_text()) != project_stage_bytes():
        raise ValueError("mature-body export projection differs")
    if manifest["max_export_bytes"] != project_stage_bytes()["transfer_cap_bytes"]:
        raise ValueError("transfer export bound differs")
    paths = source_paths(pin)
    if manifest["source_paths"] != paths or set(seal["source_sha256"]) != set(paths):
        raise ValueError("runtime closure differs from sealed closure")
    for name in paths:
        if digest(ROOT / name) != seal["source_sha256"][name]:
            raise ValueError("source hash mismatch: " + name)
    if payload["endpoint"] != {"kind": "steps", "value": ct_endpoint_steps(expected)}:
        raise ValueError("collection-only endpoint differs")
    if manifest["requirements"]["est_minutes"] != 180 or manifest["timeout_minutes"] != 360:
        raise ValueError("frozen session envelope differs")
    if json.loads((packet / "topology-manifest.json").read_text()) != dict(TOPOLOGY_SPEC):
        raise ValueError("topology contract differs")
    audit = json.loads((packet / "registry-check.json").read_text())
    registry = next(row for row in payload["inputs"] if row["kind"] == "registry")
    allocs = ct_seed_allocations(expected)
    if any(
        audit.get(k) != v
        for k, v in {
            "schema": "t34-cf-seed-audit/v1",
            "status": "PASS",
            "study": STUDY,
            "namespace": f"a100-dev/{STUDY}/",
            "allocations": allocs,
            "count": sum(row["count"] for row in allocs),
            "registry_sha256": registry["sha256"],
            "effective_seed_bits": 63,
            "aliases": 0,
            "excluded_intersection": 0,
            "train01_protected_count": 1056,
            "train01_excluded_intersection": 0,
            "predecessor_count": 2623488,
            "predecessor_intersection": 0,
        }.items()
    ):
        raise ValueError("successor seed audit does not bind allocation/exclusions")
    if manifest["sandbox_a"] != {
        "seed_allocations": allocs,
        "seed_audit_sha256": digest(packet / "registry-check.json"),
        "topology_manifest_sha256": digest(packet / "topology-manifest.json"),
    }:
        raise ValueError("manifest audit hashes differ")
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    if ledger != empty_ledger(expected):
        raise ValueError("initial successor packet requires empty verified ledger")
    load_ledger(packet / "verified-ledger.json", expected)
    if manifest.get("resume") != {
        "unit_kind": "ct-stage-snapshot",
        "ledger_path": "verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("resume declaration differs")
    if manifest.get("completion_attempt") is not None:
        raise ValueError("initial support packet cannot extend a completed collection")
    primary, fallback = variants(manifest)
    if primary["requirements"].get("prefer") != ["H100", "A100"]:
        raise ValueError("H100-first/A100-fallback required")
    if json.loads((packet / "a100-fallback-preview.json").read_text()) != fallback:
        raise ValueError("A100 fallback differs from exact derived sibling")
    return audit


def tiny_entrypoint(
    output: Path, *, capped: bool = False, checkpoint_dir: Path | None = None
) -> dict[str, Any]:
    """Run production early-telemetry entrypoint on a seeded synthetic tiny bank."""
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    torch.manual_seed(31)
    save_ego2s_checkpoint(output / "b8.pth", Ego2sNet(local_channels=7, n_scalars=60).eval())
    for name in ("frp3", "champion"):
        net = torch.nn.Linear(61, 6).eval()
        torch.save(
            {"tiny_linear_vector61": True, "state_dict": net.state_dict()}, output / f"{name}.pth"
        )
    checkpoints = {
        "b8": output / "b8.pth",
        "frp3": output / "frp3.pth",
        "champion": output / "champion.pth",
    }
    if checkpoint_dir is not None:
        from research.sandbox_a_resident.ct_run import B8_S3_SHA256
        from research.sandbox_a_resident.training_rosters import CHAMPION_SHA256, FRP3_S12_SHA256

        checkpoints = {
            "b8": checkpoint_dir / "b8_s3_t7_20261016.pth",
            "frp3": checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth",
            "champion": checkpoint_dir / "champion_a5_freespace_20260621.pth",
        }
        for name, expected in (
            ("b8", B8_S3_SHA256),
            ("frp3", FRP3_S12_SHA256),
            ("champion", CHAMPION_SHA256),
        ):
            if digest(checkpoints[name]) != expected:
                raise ValueError("tiny production checkpoint pin differs: " + name)
    control = replace(
        CTControl(),
        worlds=6,
        collection_ticks=24,
        max_body_capacity=128,
        max_deaths=4,
        max_ordinary_roots=12,
        anchor_rows=24,
        tiny=True,
        collection_base_seconds=0.0000005 if capped else 30,
        collection_cap_seconds=0.000001 if capped else 60,
    )
    (output / "control.json").write_text(json.dumps(control.to_dict()))
    (output / "verified-ledger.json").write_text(json.dumps(empty_ledger(control.to_dict())))
    command = [
        sys.executable,
        "-m",
        "research.sandbox_a_resident.ct_h5000.entrypoint",
        "runner",
        "--checkpoint",
        str(checkpoints["b8"]),
        "--frp3",
        str(checkpoints["frp3"]),
        "--champion",
        str(checkpoints["champion"]),
        "--control",
        str(output / "control.json"),
        "--ledger",
        str(output / "verified-ledger.json"),
        "--output",
        str(output / "exports"),
        "--device",
        "cpu",
        "--progress-interval-seconds",
        "0",
    ]
    start = time.perf_counter()
    with (output / "entrypoint.log").open("x") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "SNAKE_DQN_DEVICE": "cpu"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=True,
        )
    summary = json.loads((output / "exports/summary.json").read_text())
    expected = "INCOMPLETE_COLLECTION" if capped else "INCOMPLETE_SUPPORT"
    if summary["verdict"] != expected:
        raise ValueError(f"tiny real entrypoint: {summary['verdict']} != {expected}")
    if not capped and summary["collection"]["completed_ticks"] != 24:
        raise ValueError("tiny collection did not complete all ticks")
    report = {
        "status": "PASS",
        "checkpoint_sha256": {name: digest(path) for name, path in checkpoints.items()},
        "production_checkpoints": checkpoint_dir is not None,
        "command": command,
        "wall_sec": time.perf_counter() - start,
        "verdict": summary["verdict"],
        "completed_ticks": summary["collection"]["completed_ticks"],
        "summary_sha256": digest(output / "exports/summary.json"),
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path, nargs="?")
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--tiny-output", type=Path)
    parser.add_argument("--capped", action="store_true")
    parser.add_argument("--checkpoint-dir", type=Path)
    args = parser.parse_args()
    if args.tiny_output:
        print(
            json.dumps(
                tiny_entrypoint(
                    args.tiny_output.resolve(),
                    capped=args.capped,
                    checkpoint_dir=args.checkpoint_dir,
                )
            )
        )
        return 0
    if args.packet is None:
        parser.error("packet or --tiny-output required")
    audit = validate_packet(args.packet.resolve())
    if not args.checks_only:
        test_env = dict(os.environ)
        for key in (
            "SNAKE_SANDBOX_TEST_RECEIPT",
            "SNAKE_SANDBOX_ALLOCATION_LEASE",
            "SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT",
        ):
            test_env.pop(key, None)
        test_env["CT_PREFLIGHT_SUBPROCESS"] = "1"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-o",
                "addopts=",
                "-n",
                "2",
                "--dist",
                "loadfile",
                *TESTS,
            ],
            cwd=ROOT,
            env=test_env,
            check=True,
            timeout=600,
        )
        with tempfile.TemporaryDirectory(prefix="ct-h5000-admission-") as temp:
            tiny_entrypoint(Path(temp) / "tiny")
            tiny_entrypoint(Path(temp) / "cap", capped=True)
    print(f"PASS_CT_H5000 seeds={audit['count']} allocations=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
