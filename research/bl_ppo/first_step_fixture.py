"""Create-only C2 Mac-CPU first-actor-step stability evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.distributions import Categorical

from research.bl_ppo.exports import export_json
from research.bl_ppo.learner import PPOLearner, PPORollout
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


def run_fixture() -> dict[str, object]:
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
