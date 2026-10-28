"""Narrow host routing guard for the excluded C/T CUDA smoke canary."""

from __future__ import annotations

from typing import Any

SCRIPT = "research/sandbox_a_resident/ct_cuda_canary_job.sh"


def require_smoke(manifest: dict[str, Any]) -> None:
    """Permit background sparse delivery only for this finite native probe."""
    if (
        not isinstance(manifest, dict)
        or any(
            not isinstance(manifest.get(k), dict)
            for k in ("canary", "resume_exemption", "budget", "env")
        )
        or not isinstance(manifest.get("inputs"), list)
        or len(manifest["inputs"]) != 4
        or any(not isinstance(row, dict) for row in manifest["inputs"])
    ):
        raise ValueError("malformed C/T CUDA smoke manifest")
    if (
        manifest.get("script") != SCRIPT
        or manifest.get("canary")
        != {
            "schema": "colab-family-canary/v1",
            "mode": "probe",
            "family": SCRIPT,
        }
        or manifest.get("sandbox_source_delivery") != "background-sparse-v1"
        or "sandbox_admission" in manifest
        or "resume" in manifest
        or manifest.get("resume_exemption", {}).get("kind") != "short-canary"
        or manifest.get("requirements")
        != {
            "gpu": "A100",
            "allowed_gpus": ["A100"],
            "min_vcpu": 2,
            "est_minutes": 15,
        }
        or manifest.get("timeout_minutes") != 30
        or manifest.get("budget", {}).get("unit_cap") != 4.1
        or manifest.get("budget", {}).get("rate_units_per_hour") != 8.0
        or manifest.get("budget", {}).get("reserve_units") != 0.1
        or manifest.get("device") != "cuda"
        or manifest.get("env", {}).get("CT_CANARY_SOURCE_COMMIT") != manifest.get("commit")
        or manifest.get("env", {}).get("SNAKE_COLAB_CANARY") != "1"
        or manifest.get("env", {}).get("SNAKE_COLAB_CANARY_PACKAGE")
        != "research/sandbox_a_resident"
        or manifest.get("required_exports") != ["ct-cuda-canary/capture.json"]
        or {row.get("dest") for row in manifest.get("inputs", [])}
        != {
            "b8-fixture.pth",
            "frp3-fixture.pth",
            "champion-fixture.pth",
            "fixture-audit.json",
        }
    ):
        raise ValueError("C/T CUDA smoke manifest differs from its finite fixture contract")
