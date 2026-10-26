"""GPU-resident stochastic actor, GAE, PPO and exact full-state resume.

This is a new simulator-D engineering recipe, not a P1 reproduction. Only seat
zero trains; five greedy-food opponents use the existing DevicePolicy boundary.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.distributions import Categorical

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.ego_device import build_observation
from research.gpu_sim_spike.v8_device import v8_decide
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


@dataclass(frozen=True)
class Recipe:
    """Frozen engineering workload; wider world count is a separate recipe."""

    worlds: int = 128
    horizon: int = 128
    minibatch: int = 256
    epochs: int = 1
    episode_limit: int = 5000
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    learning_rate: float = 0.0003
    seed_namespace: str = "a100-dev/sandbox-a"
    seed_offset: int = 0
    max_episodes: int = 1024
    tiny: bool = False

    def __post_init__(self) -> None:
        if min(self.worlds, self.horizon, self.minibatch, self.epochs, self.episode_limit) < 1:
            raise ValueError("positive finite workload required")
        if self.seed_namespace != "a100-dev/sandbox-a":
            raise ValueError("foreign namespace")


def seed64(namespace: str, index: int) -> int:
    """Canonical uint64 SHA seed; DeviceSim uses its explicitly recorded low 63 bits."""
    if namespace != "a100-dev/sandbox-a" or index < 0:
        raise ValueError("foreign seed allocation")
    return int.from_bytes(hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:8], "big")


def allowed_mask(mask: torch.Tensor) -> torch.Tensor:
    """Match DevicePolicy's normal-speed fallback when no advisory action survives."""
    fallback = torch.arange(6, device=mask.device)[None] < 3
    return torch.where(mask.any(-1, keepdim=True), mask, fallback)


def distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Stable masked categorical; illegal actions have exactly zero probability."""
    return Categorical(logits=logits.masked_fill(~allowed_mask(mask), -torch.inf))


class ActorCritic(nn.Module):
    """B8 encoder and advantage initialized from publication; fresh zero value head."""

    def __init__(self, b8: Ego2sNet) -> None:
        super().__init__()
        self.local = copy.deepcopy(b8.local)
        self.glob = copy.deepcopy(b8.glob)
        self.fuse = copy.deepcopy(b8.fuse)
        self.actor = copy.deepcopy(b8.adv)
        self.critic = nn.Linear(256, 1, device=next(b8.parameters()).device)
        nn.init.zeros_(self.critic.weight)
        nn.init.zeros_(self.critic.bias)

    def forward(self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor) -> tuple:
        h = self.fuse(
            torch.cat(
                (self.local(local.float() / 255), self.glob(glob.float() / 255), scalars), dim=1
            )
        )
        return self.actor(h), self.critic(h).squeeze(-1)


def gae(
    rewards: torch.Tensor,
    values: torch.Tensor,
    next_values: torch.Tensor,
    terminal: torch.Tensor,
    truncated: torch.Tensor,
    gamma: float,
    lam: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Death never bootstraps; time limits bootstrap but cut trace across reset."""
    advantage = torch.zeros_like(rewards)
    carry = torch.zeros_like(rewards[0])
    for t in range(rewards.shape[0] - 1, -1, -1):
        delta = rewards[t] + gamma * next_values[t] * ~terminal[t] - values[t]
        carry = delta + gamma * lam * ~(terminal[t] | truncated[t]) * carry
        advantage[t] = carry
    return advantage, advantage + values


class ScriptedRuntime:
    """DevicePolicy adapter: no vector-policy carry exists for scripted opponents."""

    def prepare(self, sim: DeviceSim) -> None:
        pass

    def observe_step(self, sim: DeviceSim) -> None:
        pass


class ResidentAssay:
    """Own the simulator, optimizer, behavior RNG, horizon state and reset RNG keys."""

    def __init__(self, recipe: Recipe, b8: Ego2sNet, device: str = "cpu") -> None:
        self.recipe = recipe
        self.device = torch.device("cuda:0" if device == "cuda" else device)
        self.actor = ActorCritic(b8).to(self.device)
        self.optimizer = torch.optim.Adam(self.actor.parameters(), lr=recipe.learning_rate)
        self.generator = torch.Generator(device=self.device).manual_seed(3403)
        cfg = BatchSimConfig(
            num_envs=recipe.worlds,
            num_snakes=6,
            **(
                dict(game_width=120, game_height=120, initial_food=2, max_food=3, max_capacity=32)
                if recipe.tiny
                else dict(max_capacity=10400, body_storage_capacity=1024)
            ),
        )
        self.sim = DeviceSim(
            cfg,
            device=self.device,
            train_mode=False,
            allow_respawn=True,
            move_backend="indexed" if self.device.type == "cuda" else "reference",
        )
        self.sim.capacity_overflow = torch.zeros(
            (self.sim.E, self.sim.S), dtype=torch.bool, device=self.device
        )
        r2_cache.enable(self.sim)
        # A fixed finite allocation: even one death per tick cannot exhaust the table.
        self.keys = torch.tensor(
            [
                (seed64(recipe.seed_namespace, recipe.seed_offset + i) + 2**63) % 2**64 - 2**63
                for i in range(recipe.worlds * recipe.max_episodes)
            ],
            dtype=torch.long,
            device=self.device,
        ).reshape(recipe.max_episodes, recipe.worlds)
        if not bool(((self.keys & 0x7FFF_FFFF_FFFF_FFFF) != 0).all()):
            raise ValueError("zero effective seed refused")
        self.episode = torch.zeros(recipe.worlds, dtype=torch.long, device=self.device)
        self.age = torch.zeros_like(self.episode)
        self.reset_count = torch.zeros((), dtype=torch.long, device=self.device)
        self.cycle = 0
        self.policy = DevicePolicy(
            self.device,
            b8,
            opponent_networks=(None,) * 5,
            anchor_kinds=("greedy_food",) * 5,
            anchor_tables=(None,) * 5,
            vector_runtime=ScriptedRuntime(),
        )
        # Empty template and reset scratch are device-only; they never read historical roots.
        self.empty = self.sim.clone()
        self.scratch = self.sim.clone()
        self._reset(torch.ones_like(self.episode, dtype=torch.bool), initial=True)

    @torch.no_grad()
    def _reset(self, mask: torch.Tensor, *, initial: bool = False) -> None:
        """Generate fresh device roots, then merge only reset rows (no host world copy)."""
        if not initial:
            self.episode += mask.long()
            self.reset_count += mask.sum()
        if bool((self.episode >= self.recipe.max_episodes).any()):
            raise RuntimeError("frozen reset allocation exhausted")
        scratch = self.scratch
        for name, value in self.empty.__dict__.items():
            if isinstance(value, torch.Tensor):
                getattr(scratch, name).copy_(value)
        scratch._prepared = False
        scratch.alive.zero_()
        scratch.terminal_slots.zero_()
        scratch.rng_key.copy_(self.keys[self.episode, torch.arange(self.sim.E, device=self.device)])
        scratch._respawn()
        scratch.terminal_slots[:, 0] = True
        scratch.prepare()
        if not bool(scratch.alive.all()):
            raise RuntimeError("device spawn rejection budget exhausted")
        for name, value in scratch.__dict__.items():
            if isinstance(value, torch.Tensor) and value.ndim and value.shape[0] == self.sim.E:
                target = getattr(self.sim, name)
                target.copy_(
                    torch.where(mask.reshape((-1,) + (1,) * (value.ndim - 1)), value, target)
                )
        self.sim._prepared = True
        self.age.masked_fill_(mask, 0)
        r2_cache.bump(self.sim)

    @torch.no_grad()
    def observe(self) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        self.sim.prepare()
        rows = torch.stack(
            (torch.arange(self.sim.E, device=self.device), torch.zeros_like(self.age)), dim=1
        )
        legal = self.sim.advisory_mask()[:, 0]
        features = (
            v8_decide(
                self.sim,
                rows,
                torch.zeros((self.sim.E, 6), device=self.device),
                legal,
                8.0,
                features=True,
            )
            .feature_planes()
            .reshape(self.sim.E, 30)
        )
        return build_observation(self.sim, rows, features), legal

    @torch.no_grad()
    def collect(self) -> dict[str, Any]:
        """Retain all PPO inputs on device, one valid hero row/world/tick."""
        fields: dict[str, list] = {
            key: []
            for key in ("mask", "action", "logp", "value", "reward", "terminal", "truncated", "age")
        }
        observations: list[tuple] = []
        trunc_values: list[torch.Tensor] = []
        for _ in range(self.recipe.horizon):
            obs, legal = self.observe()
            logits, value = self.actor(*obs)
            dist = distribution(logits, legal)
            action = torch.multinomial(dist.probs, 1, generator=self.generator).squeeze(1)
            observations.append(tuple(x.clone() for x in obs))
            for key, tensor in (
                ("mask", legal),
                ("action", action),
                ("logp", dist.log_prob(action)),
                ("value", value),
                ("age", self.age),
            ):
                fields[key].append(tensor.clone())
            # DevicePolicy inference tensors are consumed by dynamics only, never by autograd.
            self.sim.step(self.policy.actions(self.sim, action))
            self.policy.after_step(self.sim)
            if bool(self.sim.capacity_overflow.any()):
                raise RuntimeError("physical body capacity exceeded: assay INVALID")
            self.age += 1
            terminal = self.sim._last_done[:, 0].clone()
            truncated = (self.age >= self.recipe.episode_limit) & ~terminal
            fields["reward"].append(self.sim._last_reward[:, 0].clone())
            fields["terminal"].append(terminal)
            fields["truncated"].append(truncated)
            # Prepare before reset, so one global prepared flag remains valid for every row.
            self.sim.prepare()
            bootstrap = torch.zeros_like(value)
            if bool(truncated.any()):
                next_obs, _ = self.observe()
                bootstrap.copy_(self.actor(*next_obs)[1])
            trunc_values.append(bootstrap)
            reset = terminal | truncated
            if bool(reset.any()):
                self._reset(reset)
        final_obs, _ = self.observe()
        final_value = self.actor(*final_obs)[1]
        result = {name: torch.stack(items) for name, items in fields.items()}
        result["obs"] = tuple(torch.stack([obs[i] for obs in observations]) for i in range(3))
        shifted = torch.cat((result["value"][1:], final_value[None]))
        next_values = torch.where(result["truncated"], torch.stack(trunc_values), shifted)
        result["advantage"], result["returns"] = gae(
            result["reward"],
            result["value"],
            next_values,
            result["terminal"],
            result["truncated"],
            self.recipe.gamma,
            self.recipe.gae_lambda,
        )
        result["next_value"] = next_values
        return result

    def update(self, rollout: dict[str, Any]) -> dict[str, torch.Tensor]:
        """One minimal clipped PPO update with bounded activation memory."""
        rows = self.recipe.horizon * self.recipe.worlds
        obs = tuple(x.flatten(0, 1) for x in rollout["obs"])
        flat = {k: v.flatten(0, 1) for k, v in rollout.items() if k != "obs"}
        adv = flat["advantage"]
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        last: dict[str, torch.Tensor] = {}
        for _ in range(self.recipe.epochs):
            order = torch.randperm(rows, device=self.device, generator=self.generator)
            for start in range(0, rows, self.recipe.minibatch):
                ix = order[start : start + self.recipe.minibatch]
                logits, value = self.actor(*(x[ix] for x in obs))
                dist = distribution(logits, flat["mask"][ix])
                ratio = (dist.log_prob(flat["action"][ix]) - flat["logp"][ix]).exp()
                policy_loss = -torch.minimum(
                    ratio * adv[ix],
                    ratio.clamp(1 - self.recipe.clip, 1 + self.recipe.clip) * adv[ix],
                ).mean()
                value_loss = (value - flat["returns"][ix]).square().mean()
                loss = policy_loss + 0.5 * value_loss - 0.01 * dist.entropy().mean()
                if not bool(torch.isfinite(loss)):
                    raise RuntimeError("nonfinite PPO loss")
                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5, error_if_nonfinite=True
                )
                self.optimizer.step()
                last = {"loss": loss.detach(), "grad_norm": norm.detach()}
        self.cycle += 1
        return last

    def snapshot(self) -> dict[str, Any]:
        """Full boundary snapshot; no mid-rollout checkpoint is advertised."""
        return copy.deepcopy(
            {
                "schema": "sandbox-a-full-state/v1",
                "device": str(self.device),
                "recipe": asdict(self.recipe),
                "actor": self.actor.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "sim": {k: v for k, v in self.sim.__dict__.items() if k != "_timing"},
                "keys": self.keys,
                "episode": self.episode,
                "age": self.age,
                "reset_count": self.reset_count,
                "cycle": self.cycle,
                "generator": self.generator.get_state(),
                "torch_rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state_all() if self.device.type == "cuda" else [],
                "vector_prepared": self.policy.vector_prepared,
            }
        )

    def restore(self, state: dict[str, Any]) -> None:
        """Restore all state and invalidate derived R2 cache; refuse recipe drift."""
        if state["schema"] != "sandbox-a-full-state/v1" or state["recipe"] != asdict(self.recipe):
            raise ValueError("checkpoint recipe/namespace differs")
        if state.get("device") != str(self.device):
            raise ValueError("same-platform resume requires the same device")
        state = copy.deepcopy(state)
        self.actor.load_state_dict(state["actor"])
        self.optimizer.load_state_dict(state["optimizer"])
        for name, value in state["sim"].items():
            setattr(self.sim, name, value)
        for name in ("keys", "episode", "age", "reset_count", "cycle"):
            setattr(self, name, state[name])
        self.generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        self.policy.vector_prepared = state["vector_prepared"]
        r2_cache.enable(self.sim)

    def save(self, path: Path) -> None:
        """Serialize a complete boundary checkpoint; immutable envelope is owned by runner."""
        torch.save(self.snapshot(), path)
