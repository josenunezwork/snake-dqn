"""Prepare a <=15-minute, <=3-unit combined H100/A100 acceptance packet only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from research.compute.gpu_preference import prepare_preference, variants

SCRIPT = "research/gpu_preference/acceptance.sh"


def manifest(commit: str, output: Path) -> dict:
    """Freeze two 7.5-minute allocation envelopes; estimates are forecasts."""
    packet = {
        "job_id": "gpu-pref-acceptance-20261028",
        "commit": commit,
        "script": SCRIPT,
        "device": "cuda",
        "source_paths": ["research/gpu_preference", "research/colab_cli", "pyproject.toml"],
        "requirements": {
            "gpu": "H100",
            "allowed_gpus": ["H100"],
            "min_vcpu": 1,
            "est_minutes": 3.75,
        },
        "timeout_minutes": 7.5,
        "output_dir": str(output.resolve()),
        "lease_seconds": 180,
        "max_export_bytes": 1024 * 1024,
        "inputs": [],
        "required_exports": ["gpu-acceptance.json"],
        "env": {
            "COLAB_CLI_REQUIREMENTS": "research/gpu_preference/requirements.txt",
            "SNAKE_COLAB_CANARY": "1",
            "SNAKE_COLAB_CANARY_PACKAGE": "research/gpu_preference",
        },
        "budget": {
            "rate_units_per_hour": 12,
            "unit_cap": 1.5,
            "reserve_units": 0,
            "other_assignments": {},
        },
        "resume_exemption": {
            "kind": "acceptance",
            "reason": "Synthetic hardware smoke only; no science or scoring look",
        },
        "canary": {"schema": "colab-family-canary/v1", "mode": "probe", "family": SCRIPT},
    }
    prepare_preference(packet)
    return packet


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--packet", type=Path, required=True)
    args = parser.parse_args()
    args.packet.mkdir(parents=True, exist_ok=False)
    packet = manifest(args.commit, args.packet / "output")
    with (args.packet / "manifest.json").open("x") as stream:
        json.dump(packet, stream, indent=2, sort_keys=True)
        stream.write("\n")
    for item in variants(packet):
        print(item["job_id"], item["requirements"]["gpu"], item["timeout_minutes"], item["budget"])


if __name__ == "__main__":
    main()
