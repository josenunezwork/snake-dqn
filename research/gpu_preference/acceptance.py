"""Synthetic H100/A100 CUDA acceptance; no game, checkpoint, or science inputs."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import torch

from research.colab_cli.acceptance_probe import publish_json


def run(output: Path) -> dict:
    """Execute an actual CUDA operation and preserve device/capability evidence."""
    smi = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=name,compute_cap", "--format=csv,noheader"],
        text=True,
        timeout=10,
    ).strip()
    assert torch.cuda.is_available()
    name, capability = torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0)
    assert any(kind in name for kind in ("H100", "A100")), name
    assert torch.version.cuda and torch.version.cuda.startswith("12.6"), torch.version.cuda
    tensor = torch.arange(6, dtype=torch.float32, device="cuda").reshape(1, 6)
    # A synthetic six-choice decision lets the existing canary observer attest
    # the pinned entrypoint's CUDA execution, without any game or scoring look.
    action = int(tensor.argmax(dim=-1).item())
    product = torch.ones((32, 32), device="cuda") @ torch.ones((32, 32), device="cuda")
    torch.cuda.synchronize()
    assert action == 5 and float(product.sum().item()) == 32768
    report = {
        "scope": "synthetic hardware/transport acceptance only",
        "status": "PASS",
        "nvidia_smi": smi,
        "torch_device_name": name,
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "action": action,
        "matmul_sum": 32768,
        "rates": "host supervisor receipt usage_before/usage_after and rate-calibration.json",
    }
    publish_json(output / "gpu-acceptance.json", report)
    return report


if __name__ == "__main__":
    print(json.dumps(run(Path(os.environ["COLAB_OUTPUT_DIR"])), sort_keys=True), flush=True)
