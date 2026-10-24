"""Excluded synthetic L4 fit probe with the production input/export inventory."""

from __future__ import annotations

import dataclasses
import os
import shutil
from pathlib import Path

import torch

from research.bl_ppo.exports import digest, export_json
from research.bl_residual.fit import FitRecipe
from research.bl_residual.job import run
from research.bl_residual.protocol import allocation_hash, canonical_sha, empty_ledger
from src.model.ego2s_network import load_ego2s_checkpoint

CANARY_DOSES = {"256": 256, "1024": 1024}


def canary_dose() -> int:
    """Select the producer filename of the paired main; never a real partition.

    ``F4_DOSE`` is absent from the frozen D256 manifests, so D256 keeps dose 256.
    """
    value = os.environ.get("F4_DOSE", "256")
    if value not in CANARY_DOSES:
        raise ValueError("F4_DOSE must be 256 or 1024 for an excluded canary")
    return CANARY_DOSES[value]


def main() -> None:
    """Use synthetic labels only; the registered training NPZ stays unopened."""
    dose = canary_dose()
    root = Path(os.environ["COLAB_CLI_ROOT"])
    uploaded = Path(os.environ["COLAB_INPUT_DIR"])
    inputs = root / "excluded-canary-inputs"
    inputs.mkdir(exist_ok=False)
    for name in ("b8.pth", "F4_REGISTRATION.md"):
        shutil.copyfile(uploaded / name, inputs / name)
    control = {
        "schema": "bl-residual-control/v1",
        "synthetic": True,
        "namespace": f"bl-residual-f4-d{dose}-excluded-canary",
        "doses": [dose],
        "recipe_sha256": canonical_sha(dataclasses.asdict(FitRecipe())),
        "input_hashes": {name: digest(inputs / name) for name in ("b8.pth", "F4_REGISTRATION.md")},
    }
    control["allocation"] = allocation_hash(control)
    export_json(
        inputs / "verified-ledger.json", empty_ledger(control["namespace"], control["allocation"])
    )
    control["ledger_sha256"] = digest(inputs / "verified-ledger.json")
    export_json(inputs / "control.json", control)
    # A real six-action CUDA decision is attributed by the qualified observer.
    base = load_ego2s_checkpoint(str(inputs / "b8.pth")).cuda().eval()
    with torch.no_grad():
        base(
            torch.zeros(1, 7, 31, 31, device="cuda", dtype=torch.uint8),
            torch.zeros(1, 4, 37, 37, device="cuda", dtype=torch.uint8),
            torch.zeros(1, 60, device="cuda"),
        ).argmax(dim=1)
    del base
    # Existing canary mode performs three synthetic updates, never the real fit.
    run(inputs, root / "output/fit", mode="canary", device="cuda")


if __name__ == "__main__":
    main()
