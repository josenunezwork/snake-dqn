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


class _HeadReplayCore(nn.Module):
    """Minimal F-core structural witness for the amendment precondition."""

    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(256)


class _HeadReplayPolicy(nn.Module):
    """Existing fp32 PopArt head; no actor forward, encoder or world collection."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Identity()
        self.core = _HeadReplayCore()
        self.residual_policy = nn.Identity()
        self.critic = PopArtValueHead(257)

    def value_from_critic_features(self, features: torch.Tensor) -> torch.Tensor:
        """Use the same existing head-only forward as PPOPolicy."""
        return self.critic(features)


def replay(sidecar: Path, reference: Path) -> dict[str, Any]:
    """Install using FIT only; held-out output is ledger-only evidence."""
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
    features = torch.from_numpy(fit["features"])
    returns = torch.from_numpy(fit["returns"])
    # Keep an exact disposable fp32 Amendment-03 installation for the C1
    # prediction-distance acceptance check; it is never a selection metric.
    old_policy = _HeadReplayPolicy()
    old_learner = PPOLearner(old_policy)
    old_convergence = old_learner.install_exact_return_critic_solution(features, returns)
    with torch.no_grad():
        old_installed_fit = old_policy.critic(features.float()).numpy()
    convergence = learner.install_exact_return_critic_solution(
        features, returns, zero_sum_basis=True
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
    with torch.no_grad():
        installed_fit = policy.critic(torch.from_numpy(fit["features"]).float()).numpy()
    old_fit_mse = float(old_convergence["raw_lstsq_fit_mse"])
    fit_prediction_delta = float(np.max(np.abs(installed_fit - old_installed_fit)))
    raw_fit_mse = float(convergence["raw_lstsq_fit_mse"])
    raw_fit_reference = 896454.557
    fit_mse_passed = abs(
        raw_fit_mse / raw_fit_reference - 1.0
    ) <= 1e-4 or raw_fit_mse <= old_fit_mse * (1.0 + 1e-4)
    fit_acceptance = {
        "rank": convergence["rank"] == 257,
        "raw_fit_mse": fit_mse_passed,
        "installed_prediction_delta": fit_prediction_delta <= 20.0,
        "common_mode_sensitivity": abs(float(convergence["common_mode_sensitivity_per_1e-4_bias"]))
        <= 1.0,
    }
    reproduced = all(fit_acceptance.values())
    return {
        "schema": "bl-p1-e8-04-offline-replay/v1",
        "status": "FIT_ACCEPTANCE_PASS" if reproduced else "FIT_ACCEPTANCE_FAIL",
        "purpose": "FIT-side C1 acceptance only; held-out output is non-selection ledger evidence",
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
                Path(__file__).with_name("E8_LATE_BODY_AMENDMENT_04.md"),
            )
        },
        "fit_rows": len(fit["returns"]),
        "heldout_rows": len(heldout["returns"]),
        "convergence": convergence,
        "heldout_natural_late_installed_fp32_head": actual,
        "baseline_1_h_L": baseline,
        "fit_acceptance": {
            "rank": convergence["rank"],
            "required_rank": 257,
            "raw_fit_mse": raw_fit_mse,
            "reference_raw_fit_mse": raw_fit_reference,
            "old_z_h_fit_mse": old_fit_mse,
            "max_installed_prediction_delta_from_old_z_h": fit_prediction_delta,
            "max_allowed_prediction_delta": 20.0,
            "common_mode_sensitivity_per_1e-4_bias": convergence[
                "common_mode_sensitivity_per_1e-4_bias"
            ],
            "max_allowed_common_mode_sensitivity": 1.0,
            "checks": fit_acceptance,
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
    return 0 if result["fit_acceptance"]["passed"] else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
