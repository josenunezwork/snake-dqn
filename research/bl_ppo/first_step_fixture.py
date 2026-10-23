"""Create-only C2 Mac-CPU first-actor-step stability evidence."""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path

import torch
from torch.distributions import Categorical

from research.bl_ppo.exports import export_json
from research.bl_ppo.learner import PPOLearner, PPORollout, compute_gae
from research.bl_ppo.model import PPOPolicy
from src.model.ego2s_network import Ego2sNet


def _zero_sum_features(rows: int = 300) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(20261022)
    core = torch.randn((rows, 256), generator=generator, dtype=torch.float64)
    core.sub_(core.mean(dim=1, keepdim=True))
    horizon = torch.linspace(0.01, 1.0, rows, dtype=torch.float64).unsqueeze(1)
    features = torch.cat((core, horizon), dim=1)
    coefficients = torch.linspace(-0.25, 0.25, 256, dtype=torch.float64)
    returns = core @ coefficients + 17.0 * horizon.squeeze(1) + 5.0
    return features, returns


def _legacy_diagnostic() -> dict[str, object]:
    """Run three real actor-enabled PPO updates after the C1 installation."""
    torch.set_num_threads(1)
    torch.manual_seed(20261022)
    policy = PPOPolicy("F", None, 0.7, b8_model=Ego2sNet(7, 60))
    learner = PPOLearner(policy, critic_refit_updates=0)
    features, returns = _zero_sum_features()
    convergence = learner.install_exact_return_critic_solution(
        features, returns, zero_sum_basis=True
    )
    local = torch.randint(0, 256, (1, 2, 7, 31, 31), dtype=torch.uint8)
    global_ = torch.randint(0, 256, (1, 2, 4, 37, 37), dtype=torch.uint8)
    scalars = torch.randn((1, 2, 60))
    with torch.no_grad():
        logits, values, _ = policy(local[0], global_[0], scalars[0])
        actions = Categorical(logits=logits).sample().unsqueeze(0)
        logp = Categorical(logits=logits).log_prob(actions[0]).unsqueeze(0)
    rollout = PPORollout(
        local=local,
        global_=global_,
        scalars=scalars,
        actions=actions,
        logp=logp,
        rewards=torch.tensor([[1.0, -0.5]]),
        dones=torch.zeros((1, 2), dtype=torch.bool),
        values=values.unsqueeze(0),
        last_value=torch.zeros(2),
    )
    held_local = torch.randint(0, 256, (2, 7, 31, 31), dtype=torch.uint8)
    held_global = torch.randint(0, 256, (2, 4, 37, 37), dtype=torch.uint8)
    held_scalars = torch.randn((2, 60))
    with torch.no_grad():
        _, before, _ = policy(held_local, held_global, held_scalars)
        before = before.clone()
    steps: list[dict[str, float]] = []
    failures: list[str] = []
    for step in range(3):
        metrics = learner.update(rollout, actor_enabled=True)
        with torch.no_grad():
            _, after, _ = policy(held_local, held_global, held_scalars)
        gradients_finite = all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in policy.parameters()
        )
        if (
            not torch.isfinite(after).all()
            or not all(torch.isfinite(torch.tensor(v)) for v in metrics.values())
            or not gradients_finite
        ):
            failures.append(f"step_{step + 1}: non-finite value, metric, or gradient")
        delta = float((after - before).abs().max())
        if delta > float(policy.critic.scale):
            failures.append(
                f"step_{step + 1}: fixed-feature value displacement exceeded PopArt scale"
            )
        steps.append(
            {
                "step": float(step + 1),
                "max_abs_value_delta": delta,
                "gradients_finite": gradients_finite,
                **metrics,
            }
        )
    with torch.no_grad():
        original = policy.core.norm.bias.clone()
        _, base, _ = policy(held_local, held_global, held_scalars)
        policy.core.norm.bias.add_(1e-4)
        _, shifted, _ = policy(held_local, held_global, held_scalars)
        policy.core.norm.bias.copy_(original)
    actual = float((shifted - base).abs().max())
    analytic = abs(float(convergence["common_mode_sensitivity_per_1e-4_bias"]))
    if actual > 1.0 or analytic > 1.0:
        failures.append("uniform 1e-4 core.norm.bias shift exceeded one return unit")
    return {
        "schema": "bl-p1-e8-04-first-step-fixture/v1",
        "status": "PASS" if not failures else "FAIL",
        "platform": "Mac CPU",
        "colab_units": 0,
        "ppo_updates": 3,
        "actor_enabled": True,
        "optimizer_lrs": [float(group["lr"]) for group in learner.optimizer.param_groups],
        "convergence": convergence,
        "installed_popart_scale": float(policy.critic.scale),
        "steps": steps,
        "analytic_common_mode_bias_shift": analytic,
        "actual_uniform_1e-4_core_norm_bias_shift": actual,
        "acceptance_failures": failures,
        "dispatch_ready": False,
    }


def _observations(rows: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.randint(0, 256, (rows, 7, 31, 31), dtype=torch.uint8),
        torch.randint(0, 256, (rows, 4, 37, 37), dtype=torch.uint8),
        torch.randn((rows, 60)),
    )


def _steps(
    learner: PPOLearner,
    rollout: PPORollout,
    held: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    *,
    actor_enabled: bool,
) -> list[dict[str, object]]:
    with torch.no_grad():
        _, before, _ = learner.policy(*held)
        before = before.clone()
    result = []
    for step in range(3):
        metrics = learner.update(rollout, actor_enabled=actor_enabled)
        with torch.no_grad():
            _, after, _ = learner.policy(*held)
        finite = bool(torch.isfinite(before).all() and torch.isfinite(after).all())
        finite = finite and all(math.isfinite(v) for v in metrics.values())
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in learner.policy.parameters()
        )
        result.append(
            {
                "step": step + 1,
                "max_abs_value_delta": float((after - before).abs().max()),
                "values_losses_finite": finite,
                "gradients_finite": gradients_finite,
                **metrics,
            }
        )
    return result


def _arm(*, registered_control: bool) -> dict[str, object]:
    # Reinitialisation and all observations/RNG draws are identical across arms.
    torch.manual_seed(20261022)
    policy = PPOPolicy("F", None, 0.7, b8_model=Ego2sNet(7, 60))
    learner = PPOLearner(policy, critic_refit_updates=0)
    fit_observations = _observations(1024)
    horizon = torch.rand(1024)
    features = []
    with torch.no_grad():
        for start in range(0, 1024, 128):
            policy(
                *(v[start : start + 128] for v in fit_observations),
                remaining_horizon=horizon[start : start + 128],
            )
            features.append(policy.last_critic_features.clone())
    features = torch.cat(features).double()
    coefficients = torch.linspace(-0.25, 0.25, 256, dtype=torch.float64)
    coefficients -= coefficients.mean()
    returns = features[:, :256] @ coefficients + 17.0 * features[:, 256] + 5.0
    returns += 0.3 * torch.randn(1024, dtype=torch.float64)
    if registered_control:
        # Judgment calls this fit_complete_targets; the frozen registered method
        # at learner.py:345 is named fit_completed_critic. No learner alias/edit.
        convergence = {
            "registered_fit_loss": learner.fit_completed_critic(features.float(), returns.float())
        }
    else:
        convergence = learner.install_exact_return_critic_solution(
            features, returns, zero_sum_basis=True
        )
    sigma = float(policy.critic.scale)
    observations = _observations(16)
    held = _observations(16)
    reward_unit = torch.rand((1, 16)) * 0.2 - 0.1
    with torch.no_grad():
        logits, values, _ = policy(*observations)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        logp = distribution.log_prob(actions)
    rollout = PPORollout(
        local=observations[0].unsqueeze(0),
        global_=observations[1].unsqueeze(0),
        scalars=observations[2].unsqueeze(0),
        actions=actions.unsqueeze(0),
        logp=logp.unsqueeze(0),
        rewards=reward_unit * sigma,
        dones=torch.zeros((1, 16), dtype=torch.bool),
        values=values.unsqueeze(0),
        last_value=values.clone(),
    )
    _, targets = compute_gae(rollout.rewards, rollout.values, rollout.dones, rollout.last_value)
    gap = targets - rollout.values
    # D2 uses the same installed arm/rollout, before actor updates, on a fresh copy.
    head_only = copy.deepcopy(learner) if not registered_control else None
    steps = _steps(learner, rollout, held, actor_enabled=True)
    with torch.no_grad():
        original = policy.core.norm.bias.clone()
        _, base, _ = policy(*held)
        policy.core.norm.bias.add_(1e-4)
        _, shifted, _ = policy(*held)
        policy.core.norm.bias.copy_(original)
    return {
        "convergence": convergence,
        "fit_rows": 1024,
        "fit_feature_source": "policy.last_critic_features; same policy and observations",
        "fit_noise_sd": 0.3,
        "control_method": (
            "PPOLearner.fit_completed_critic (judgment fit_complete_targets)"
            if registered_control
            else None
        ),
        "installed_popart_scale": sigma,
        "max_abs_reward": float(rollout.rewards.abs().max()),
        "max_abs_target_gap": float(gap.abs().max()),
        "rms_target_gap": float(gap.square().mean().sqrt()),
        "bootstrap_to_own_value": bool(torch.equal(rollout.last_value, rollout.values[0])),
        "initial_values_targets_finite": bool(
            torch.isfinite(values).all()
            and torch.isfinite(targets).all()
            and torch.isfinite(returns).all()
        ),
        "steps": steps,
        "actual_uniform_1e-4_core_norm_bias_shift": float((shifted - base).abs().max()),
        "D2_head_only_raw_displacements": (
            _steps(head_only, rollout, held, actor_enabled=False) if head_only is not None else None
        ),
        "optimizer_lrs": [float(group["lr"]) for group in learner.optimizer.param_groups],
    }


def run_fixture() -> dict[str, object]:
    """Run exactly the judge's C2-prime fixture and frozen diagnostics once."""
    torch.set_num_threads(1)
    amended = _arm(registered_control=False)
    control = _arm(registered_control=True)
    failures = []
    for name, arm in (("amendment_04", amended), ("registered_control", control)):
        if not arm["initial_values_targets_finite"] or not all(
            step["values_losses_finite"] and step["gradients_finite"] for step in arm["steps"]
        ):
            failures.append(f"{name}: non-finite values, losses or gradients")
        if arm["max_abs_reward"] > 0.1 * arm["installed_popart_scale"] + 1e-6:
            failures.append(f"{name}: reward outside frozen self-consistent bound")
    analytic = abs(float(amended["convergence"]["common_mode_sensitivity_per_1e-4_bias"]))
    if analytic > 1.0 or amended["actual_uniform_1e-4_core_norm_bias_shift"] > 1.0:
        failures.append("uniform core.norm.bias shift exceeds one return unit")
    bounds = []
    for step, reference in zip(amended["steps"], control["steps"]):
        bound = max(amended["installed_popart_scale"], 1.25 * reference["max_abs_value_delta"])
        bounds.append(bound)
        if step["max_abs_value_delta"] > bound:
            failures.append(f"step_{step['step']}: drift exceeds max(sigma, 1.25x control)")
    return {
        "schema": "bl-p1-e8-04-first-step-fixture/v2",
        "authority": "JUDGMENT_P1_E8_04.md section 4 C2-prime",
        "status": "PASS" if not failures else "FAIL",
        "platform": "Mac CPU",
        "colab_units": 0,
        "ppo_updates": 3,
        "actor_enabled": True,
        "amendment_04": amended,
        "control_registered_head": control,
        "per_step_drift_bounds": bounds,
        "analytic_common_mode_bias_shift": analytic,
        "D1_original_fixture_03_unchanged": _legacy_diagnostic(),
        "diagnostics_are_acceptance_inputs": False,
        "acceptance_failures": failures,
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
        "dispatch_ready": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    if any(Path(str(args.out) + suffix).exists() for suffix in ("", ".sha256", ".complete")):
        raise FileExistsError(f"refusing to overwrite C2 fixture: {args.out}")
    result = run_fixture()
    export_json(args.out, result)
    print(json.dumps({"out": str(args.out), "status": result["status"]}))
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
