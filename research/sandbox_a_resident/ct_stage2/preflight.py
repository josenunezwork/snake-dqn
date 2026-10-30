"""Offline sealed admission, tests and a tiny real-entrypoint Mac bank slice."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from research.compute.gpu_preference import variants
from research.sandbox_a_resident.ct_stage2.prepare import (
    ARTIFACT_ROOT,
    BANK_ROOT,
    BASE_MINUTES,
    EFFECTIVE_MINUTES,
    ENDPOINT,
    JOB_ID,
    ROOT,
    STUDY,
    expected_science,
    source_paths,
    staging_projection,
)
from research.sandbox_a_resident.ct_stage2.units import empty_ledger
from tools.sandbox.preflight import preflight, verify_committed_manifest
from tools.sandbox.registry import digest

TESTS = (
    "research/sandbox_a_resident/ct_stage2/test_stage2.py",
    "research/sandbox_a_resident/test_ct_branching_fit_and_stats.py",
    "research/sandbox_a_resident/test_ct_cuda_semantics.py",
    "research/sandbox_a_resident/test_ct_mechanics_and_snapshots.py",
    "research/sandbox_a_resident/test_ct_gpu_preference.py",
    "tools/sandbox/tests/test_source_delivery_admission.py",
)


def validate_packet(packet: Path, *, require_committed: bool = True) -> dict[str, Any]:
    """Check whole immutable packet; no fixture, completion or arbitrary input admission."""
    sandbox = preflight(
        packet / "sandbox.json", require_committed=require_committed, run_tests=False
    )
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    config = json.loads((packet / "control.json").read_text())
    seal = json.loads((packet / "SEAL.json").read_text())
    names = {
        "control.json",
        "bank-verification.json",
        "verified-ledger.json",
        "dispatch-manifest.json",
        "sandbox.json",
        "a100-fallback-preview.json",
        "transfer-projection.json",
    }
    if seal.get("schema") != "ct-stage2-seal/v1" or set(seal.get("components", {})) != names:
        raise ValueError("downstream seal component set differs")
    for name in names | {"SEAL.json"}:
        path = packet / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("missing/symlinked sealed packet component")
        verify_committed_manifest(path, require_committed=require_committed)
        if name != "SEAL.json" and digest(path) != seal["components"][name]:
            raise ValueError("downstream packet component changed")
    pin = manifest["commit"]
    expected_config = {
        "schema": "ct-stage2-control/v1",
        "study": STUDY,
        "source_commit": pin,
        "fixture": False,
        "bank": json.loads((packet / "bank-verification.json").read_text()),
        "science": expected_science(pin),
    }
    if (
        config != expected_config
        or json.loads(manifest["env"]["SANDBOX_CT_CONTROL_JSON"]) != config
    ):
        raise ValueError("downstream recipe, bank or fixture flag differs")
    if seal["source_commit"] != pin or manifest["env"]["SANDBOX_A_SOURCE_COMMIT"] != pin:
        raise ValueError("source pin differs")
    paths = source_paths(pin)
    if manifest["source_paths"] != paths or set(seal["source_sha256"]) != set(paths):
        raise ValueError("downstream source closure differs")
    for name in paths:
        if digest(ROOT / name) != seal["source_sha256"][name]:
            raise ValueError("downstream source changed: " + name)
    if (
        sandbox["study"] != STUDY
        or sandbox["salt"] != f"a100-dev/{STUDY}/"
        or sandbox["endpoint"] != {"kind": "steps", "value": ENDPOINT}
        or manifest["script"] != "research/sandbox_a_resident/ct_stage2/job.sh"
        or manifest["job_id"] != JOB_ID
        or manifest["output_dir"] != str(ARTIFACT_ROOT / "output")
        or manifest["requirements"]["est_minutes"] != BASE_MINUTES
        or manifest["timeout_minutes"] != EFFECTIVE_MINUTES
        or manifest["max_export_bytes"] != 768 * 1024**2
        or manifest.get("completion_attempt") is not None
    ):
        raise ValueError("downstream endpoint/identity/envelope differs")
    ledger = json.loads((packet / "verified-ledger.json").read_text())
    if ledger != empty_ledger(config) or manifest["resume"] != {
        "unit_kind": "ct-stage2-snapshot",
        "ledger_path": "verified-ledger.json",
        "skip_verified": True,
        "max_outcome_blind_retries": 2,
    }:
        raise ValueError("initial downstream packet requires empty exact ledger")
    if config["bank"]["payload"] != {
        "sha256": "112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0",
        "size": 126341023,
    }:
        raise ValueError("foreign collection bank")
    uploads = {r["dest"]: r for r in manifest["inputs"]}
    expected_dest = {
        "b8-s3.pth",
        "frp3-s12.pth",
        "champion.pth",
        "registry.json",
        "verified-ledger.json",
    } | {
        f"bank/stage-0001.{ext}{extra}"
        for ext in ("pt", "json")
        for extra in ("", ".sha256", ".complete")
    }
    if len(uploads) != len(manifest["inputs"]) or set(uploads) != expected_dest:
        raise ValueError("arbitrary or missing downstream input")
    for ext, role in (("pt", "payload"), ("json", "header")):
        row = uploads[f"bank/stage-0001.{ext}"]
        if (
            Path(row["path"]) != BANK_ROOT / f"stage-0001.{ext}"
            or row["sha256"] != config["bank"][role]["sha256"]
        ):
            raise ValueError("bank upload differs from frozen descriptor")
        from research.sandbox_a_resident.ct_stage2.units import complete_file

        complete_file(Path(row["path"]), config["bank"][role])
    projection = staging_projection(manifest["inputs"])
    if projection != json.loads((packet / "transfer-projection.json").read_text()):
        raise ValueError("real-bank transfer projection differs")
    primary, fallback = variants(manifest)
    if primary["requirements"].get("prefer") != ["H100", "A100"] or fallback != json.loads(
        (packet / "a100-fallback-preview.json").read_text()
    ):
        raise ValueError("hardware variants differ")
    return config


def tiny_entrypoint(config: dict[str, Any], output: Path) -> dict[str, Any]:
    """Run actual status entrypoint with released models and a deterministic real-bank slice."""
    output.mkdir(parents=True, exist_ok=False)
    fixture = {
        **config,
        "fixture": True,
        "science": expected_science(config["source_commit"], fixture=True),
    }
    config_path, ledger_path = output / "control.json", output / "verified-ledger.json"
    config_path.write_text(json.dumps(fixture, sort_keys=True) + "\n")
    ledger_path.write_text(json.dumps(empty_ledger(fixture), sort_keys=True) + "\n")
    checkpoint_dir = ROOT.parent / "snake-dqn/saved_snakes"
    command = [
        sys.executable,
        "-m",
        "research.sandbox_a_resident.ct_stage2.entrypoint",
        "runner",
        "--config",
        str(config_path),
        "--bank-root",
        str(BANK_ROOT),
        "--ledger",
        str(ledger_path),
        "--input-root",
        str(output),
        "--checkpoint",
        str(checkpoint_dir / "b8_s3_t7_20261016.pth"),
        "--frp3",
        str(checkpoint_dir / "frp3_m3_s12_u60000_20261005.pth"),
        "--champion",
        str(checkpoint_dir / "champion_a5_freespace_20260621.pth"),
        "--output",
        str(output / "exports"),
        "--device",
        "cpu",
    ]
    start = time.perf_counter()
    with (output / "entrypoint.log").open("x") as stream:
        subprocess.run(
            command,
            cwd=ROOT,
            env={**os.environ, "SNAKE_DQN_DEVICE": "cpu"},
            stdout=stream,
            stderr=subprocess.STDOUT,
            timeout=180,
            check=True,
        )
    summary = json.loads((output / "exports/summary.json").read_text())
    if summary["verdict"] != "PREFLIGHT_ONLY" or summary["completed_stages"] != list(range(1, 11)):
        raise ValueError("real-bank tiny downstream entrypoint failed")
    report = {
        "status": "PASS",
        "fixture_only": True,
        "command": command,
        "roots": 4,
        "anchors": 16,
        "branch_horizon": 2,
        "minibatches_per_arm": 1,
        "wall_sec": time.perf_counter() - start,
        "completed_stages": list(range(1, 11)),
        "summary_sha256": digest(output / "exports/summary.json"),
    }
    (output / "preflight.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    parser.add_argument("--checks-only", action="store_true")
    parser.add_argument("--tiny-output", type=Path)
    args = parser.parse_args()
    config = validate_packet(args.packet.resolve())
    if args.tiny_output:
        print(json.dumps(tiny_entrypoint(config, args.tiny_output.resolve())))
    elif not args.checks_only:
        test_env = dict(os.environ)
        for key in (
            "SNAKE_SANDBOX_TEST_RECEIPT",
            "SNAKE_SANDBOX_ALLOCATION_LEASE",
            "SNAKE_SANDBOX_REQUIRE_TEST_RECEIPT",
        ):
            test_env.pop(key, None)
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
        with tempfile.TemporaryDirectory(prefix="ct-stage2-preflight-") as temp:
            tiny_entrypoint(config, Path(temp) / "tiny")
    print("PASS_CT_STAGE2 allocations=0")


if __name__ == "__main__":
    main()
