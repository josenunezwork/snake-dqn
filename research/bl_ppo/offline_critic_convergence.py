"""Replay amendment 03's production head installation on verified Y3 exports only."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from research.bl_ppo.exports import digest, export_json, verify_export
from research.bl_ppo.learner import PPOLearner
from research.bl_ppo.model import PopArtValueHead
from research.bl_ppo.offline_critic_diagnostic import (
    _fit_predict,
    _metrics,
    load_sidecar,
)


class _HeadReplayPolicy(nn.Module):
    """Existing fp32 PopArt head; no actor forward, encoder or world collection."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.core = nn.Identity()
        self.residual_policy = nn.Identity()
        self.critic = PopArtValueHead(257)

    def value_from_critic_features(self, features: torch.Tensor) -> torch.Tensor:
        """Use the same existing head-only forward as PPOPolicy."""
        return self.critic(features)


def replay(sidecar: Path, reference: Path) -> dict[str, Any]:
    """Install using FIT only, then compare frozen held-out metrics with D1."""
    started = time.monotonic()
    reference_sha = verify_export(reference)
    old = json.loads(reference.read_text())
    fit, heldout, binding = load_sidecar(sidecar)
    if np.any(fit["world_ids"] % 5 == 0) or np.any(heldout["world_ids"] % 5 != 0):
        raise ValueError("Y3 replay requires the registered world_id % 5 split")
    if old.get("input_binding") != binding:
        raise ValueError("D1 diagnostic input binding differs from replay exports")
    torch.set_num_threads(1)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        policy = _HeadReplayPolicy()
    learner = PPOLearner(policy)
    solve_started = time.monotonic()
    convergence = learner.install_exact_return_critic_solution(
        torch.from_numpy(fit["features"]), torch.from_numpy(fit["returns"])
    )
    solve_seconds = time.monotonic() - solve_started
    # Match qualification's forward shape: it selects the late reservoir rows
    # before the fp32 linear forward, rather than slicing an all-heldout GEMM.
    late = heldout["lengths"] >= 800.0
    with torch.no_grad():
        prediction = policy.critic(torch.from_numpy(heldout["features"][late]).float()).numpy()
    actual = _metrics(heldout["returns"][late], prediction)
    baseline = _metrics(
        heldout["returns"][late], _fit_predict(fit, heldout, "baseline_1_h_L")[late]
    )
    expected = old["heldout_natural_late"]["registered_z_h"]
    i0 = actual["count"] == 162 and abs(baseline["mse"] / 561561.60 - 1.0) <= 0.005
    ev_delta = abs(actual["explained_variance"] - expected["explained_variance"])
    mse_relative_delta = abs(actual["mse"] / expected["mse"] - 1.0)
    reproduced = i0 and ev_delta <= 1e-4 and mse_relative_delta <= 1e-3
    return {
        "schema": "bl-p1-e8-03-offline-replay/v1",
        "status": "REPRODUCED_D1" if reproduced else "REPLAY_MISMATCH",
        "purpose": "offline numerical reproduction only; historical gate unchanged",
        "forward_selection": "natural heldout L>=800 before fp32 forward; same as qualification",
        "input_binding": binding,
        "reference": {"path": str(reference), "sha256": reference_sha},
        "source_sha256": {
            str(path.relative_to(Path(__file__).resolve().parents[2])): digest(path)
            for path in (
                Path(__file__).resolve(),
                Path(__file__).with_name("learner.py"),
                Path(__file__).with_name("model.py"),
                Path(__file__).with_name("refit.py"),
                Path(__file__).with_name("E8_LATE_BODY_AMENDMENT_03.md"),
            )
        },
        "fit_rows": len(fit["returns"]),
        "heldout_rows": len(heldout["returns"]),
        "convergence": convergence,
        "heldout_natural_late_installed_fp32_head": actual,
        "baseline_1_h_L": baseline,
        "reference_registered_z_h": expected,
        "identity_i0_passed": i0,
        "reproduction": {
            "ev_absolute_delta": ev_delta,
            "ev_absolute_tolerance": 1e-4,
            "mse_relative_delta": mse_relative_delta,
            "mse_relative_tolerance": 1e-3,
            "passed": reproduced,
        },
        "timing": {
            "solve_and_install_seconds": solve_seconds,
            "total_seconds": time.monotonic() - started,
            "platform": "Mac CPU; not an L4 timing or full source forecast measurement",
        },
        "thread_environment": {
            name: os.environ.get(name)
            for name in (
                "OMP_NUM_THREADS",
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
            )
        },
        "torch_threads": torch.get_num_threads(),
        "actor_transitions": 0,
        "provider_calls": 0,
        "historical_qualification_changed": False,
        "dispatch_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sidecar", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(Path(str(args.out) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError(f"refusing to overwrite offline replay: {args.out}")
    result = replay(args.sidecar, args.reference)
    export_json(args.out, result)
    print(json.dumps({"out": str(args.out), "status": result["status"]}))
    return 0 if result["reproduction"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
