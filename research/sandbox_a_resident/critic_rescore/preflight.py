"""Offline admission for the frozen critic re-score package, never dispatch."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from research.sandbox_a_resident.critic_rescore.prepare import audit
from research.sandbox_a_resident.critic_rescore.run import load_ledger
from tools.sandbox.preflight import preflight, verify_committed_manifest


def check(packet: Path, *, run_tests: bool = True) -> dict:
    """Bind runtime control, all uploaded sources, exact bank and empty/resume ledger."""
    preflight(packet / "sandbox.json", run_tests=run_tests)
    for name in (
        "control.json",
        "seed-audit.json",
        "dispatch-manifest.json",
        "verified-ledger.json",
    ):
        verify_committed_manifest(packet / name)
    manifest = json.loads((packet / "dispatch-manifest.json").read_text())
    control = json.loads((packet / "control.json").read_text())
    if json.loads(manifest["env"]["SANDBOX_A_CONTROL_JSON"]) != control:
        raise ValueError("runtime control differs from committed recipe")
    if control["source_commit"] != manifest["commit"] or control["test_only"]:
        raise ValueError("production source identity/test-only mismatch")
    if (
        control["worlds"],
        control["horizon"],
        control["late_length"],
        control["late_lives_per_mix"],
        control["max_waves"],
        control["holdout_offset"],
        control["baseline_offset"],
    ) != (96, 5000, 800, 8, 32, 9_000_000_000, 9_100_000_000):
        raise ValueError("collection recipe differs from frozen development endpoint")
    upload_hashes = {r["dest"]: r["sha256"] for r in manifest["inputs"]}
    if {k: upload_hashes[k] for k in control["input_hashes"]} != control["input_hashes"]:
        raise ValueError("runtime input identities differ")
    sandbox = json.loads((packet / "sandbox.json").read_text())
    registry = next(Path(r["path"]) for r in sandbox["inputs"] if r["kind"] == "registry")
    if audit(control, registry) != json.loads((packet / "seed-audit.json").read_text()):
        raise ValueError("seed audit mismatch")
    verified = load_ledger(packet / "verified-ledger.json", control)
    priors = [item for item in manifest["inputs"] if item["dest"].startswith("prior/")]
    expected = {
        (Path(item["path"]).name, item["size"], item["sha256"]) for item in verified.values()
    }
    actual = {
        (Path(item["dest"]).name, Path(item["path"]).stat().st_size, item["sha256"])
        for item in priors
    }
    if expected != actual or len(priors) != len(verified):
        raise ValueError("verified ledger and prior uploaded unit set differ")
    if run_tests:
        env = os.environ.copy()
        env.update(
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            SNAKE_DQN_DEVICE="cpu",
        )
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "-q",
                "-p",
                "no:xdist",
                "research/sandbox_a_resident/critic_rescore/test_rescore.py",
            ],
            check=True,
            env=env,
        )
    return control


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("packet", type=Path)
    args = parser.parse_args()
    check(args.packet.resolve())
    print("CRITIC_RESCORE_PREFLIGHT PASS; allocations=0")
