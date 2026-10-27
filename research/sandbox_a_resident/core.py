"""GPU-resident stochastic actor, GAE, PPO and exact full-state resume.

This is a new simulator-D engineering recipe, not a P1 reproduction. Only seat
zero trains; five greedy-food opponents use the existing DevicePolicy boundary.
"""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, dataclass, replace
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
from research.sandbox_a_resident.device_sim import ResidentDeviceSim
from research.sandbox_a_resident.runtime_tools import CompiledBlock, PhaseProfiler
from research.sandbox_a_resident.tensor_blocks import TensorBlocks
from src.model.ego2s_network import Ego2sNet
from src.simd_env.batch_sim import BatchSimConfig


@dataclass(frozen=True)
class Recipe:
    """Frozen engineering workload; wider world count is a separate recipe."""

    worlds: int = 128
    horizon: int = 128
    minibatch: int = 256
    epochs: int = 1
    episode_limit: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip: float = 0.2
    learning_rate: float = 0.0003
    seed_namespace: str = "a100-dev/sandbox-a"
    seed_offset: int = 0
    max_episodes: int = 1024
    tiny: bool = False
    stagger_initial_age: bool = False

    def __post_init__(self) -> None:
        if min(self.worlds, self.horizon, self.minibatch, self.epochs, self.episode_limit) < 1:
            raise ValueError("positive finite workload required")
        if self.seed_namespace != "a100-dev/sandbox-a":
            raise ValueError("foreign namespace")
        if type(self.stagger_initial_age) is not bool:
            raise ValueError("stagger_initial_age must be boolean")


def seed64(namespace: str, index: int) -> int:
    """Canonical uint64 SHA seed; DeviceSim uses its explicitly recorded low 63 bits."""
    if namespace != "a100-dev/sandbox-a" or index < 0:
        raise ValueError("foreign seed allocation")
    return int.from_bytes(hashlib.sha256(f"{namespace}/{index}".encode()).digest()[:8], "big")


def allowed_mask(mask: torch.Tensor) -> torch.Tensor:
    """Match DevicePolicy's normal-speed fallback when no advisory action survives."""
    fallback = r2_cache.const((True, True, True, False, False, False), torch.bool, mask.device)
    return torch.where(mask.any(-1, keepdim=True), mask, fallback)


def distribution(logits: torch.Tensor, mask: torch.Tensor) -> Categorical:
    """Stable masked categorical; illegal actions have exactly zero probability."""
    return Categorical(
        logits=logits.masked_fill(~allowed_mask(mask), -torch.inf), validate_args=False
    )


def categorical_terms(
    logits: torch.Tensor, mask: torch.Tensor, action: torch.Tensor | None = None
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """Masked categorical probabilities, entropy, and optional selected log probability.

    This is algebraically the same distribution as :func:`distribution`, but avoids
    allocating a ``Categorical`` object in the hot collection/minibatch paths.
    """
    legal = allowed_mask(mask)
    log_probs = torch.log_softmax(logits.masked_fill(~legal, -torch.inf), dim=-1)
    probs = log_probs.exp()
    # ``0 * -inf`` is NaN even though its entropy contribution is mathematically zero.
    finite_log_probs = torch.where(legal, log_probs, torch.zeros_like(log_probs))
    entropy = -(probs * finite_log_probs).sum(-1)
    selected = None if action is None else log_probs.gather(-1, action[..., None]).squeeze(-1)
    return probs, entropy, selected


# Explicit simulator state contract. Never infer an environment axis from E == S.
ENV_FIELDS = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "_last_reward",
    "_last_done",
    "_last_death_cause",
    "_last_kills",
    "_last_food_ate",
    "_last_ambient_food_ate",
    "_last_corpse_food_ate",
    "_last_transition_valid",
    "_boosted_this_step",
    "terminal_slots",
    "capacity_overflow",
)

# ``observe``/``prepare`` need only this state.  A bounded bank of these fields
# holds only captured pre-reset time-limit worlds.
BOOTSTRAP_FIELDS = (
    "bodies",
    "head_ptr",
    "seg_count",
    "length",
    "alive",
    "direction",
    "boost_frames",
    "frames_since_food",
    "respawn_timer",
    "_reward_prev_length",
    "frame",
    "rng_counter",
    "rng_key",
    "food_cells",
    "food_present",
    "corpse_food",
    "food_count",
    "terminal_slots",
    "_boosted_this_step",
)


def world_subset(sim: DeviceSim, indices: torch.Tensor) -> DeviceSim:
    """Compact only declared environment state; keep constants and metadata intact."""
    out = object.__new__(DeviceSim)
    out.__dict__ = sim.__dict__.copy()
    out.E = indices.numel()
    out.cfg = replace(sim.cfg, num_envs=out.E)
    for name in ENV_FIELDS:
        setattr(out, name, getattr(sim, name).index_select(0, indices))
    r2_cache.inherit(out, sim)
    return out


def bootstrap_subset(
    sim: DeviceSim, bank: dict[str, torch.Tensor], rows: torch.Tensor
) -> DeviceSim:
    """Materialize the one compact bootstrap batch after the fixed horizon."""
    out = object.__new__(DeviceSim)
    out.__dict__ = sim.__dict__.copy()
    out.E = rows.numel()
    out.cfg = replace(sim.cfg, num_envs=out.E)
    for name in BOOTSTRAP_FIELDS:
        setattr(
            out,
            name,
            bank[name]
            .flatten(0, 1)
            .index_select(0, bank["storage"].flatten().index_select(0, rows)),
        )
    out._prepared = False
    r2_cache.inherit(out, sim)
    return out


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
        h = self.encode(local, glob, scalars)
        return self.actor(h), self.critic(h).squeeze(-1)

    def encode(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Shared encoder used by policy/value and post-horizon bootstrap paths."""
        return self.fuse(
            torch.cat(
                (self.local(local.float() / 255), self.glob(glob.float() / 255), scalars), dim=1
            )
        )

    def critic_only(
        self, local: torch.Tensor, glob: torch.Tensor, scalars: torch.Tensor
    ) -> torch.Tensor:
        """Evaluate only the value head for deferred time-limit bootstraps."""
        return self.critic(self.encode(local, glob, scalars)).squeeze(-1)


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
        self.sim = ResidentDeviceSim(
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
        self.collection_faults = torch.zeros(3, dtype=torch.bool, device=self.device)
        self._world_indices = torch.arange(recipe.worlds, device=self.device)
        self._slot_indices = torch.arange(self.sim.S, device=self.device)
        self._first_counter = torch.ones(recipe.worlds, dtype=torch.long, device=self.device)
        self._rows = torch.stack((self._world_indices, torch.zeros_like(self.age)), dim=1)
        self._v8_q = torch.zeros((recipe.worlds, 6), device=self.device)
        self._action_buffer = torch.ones(
            (recipe.worlds, self.sim.S), dtype=torch.long, device=self.device
        )
        self._rollout: dict[str, Any] | None = None
        self._bootstrap_bank: dict[str, torch.Tensor] | None = None
        # Collection inference and learner autograd have different Dynamo/AOT
        # specializations.  Keep their validation and fallback lifetime apart.
        self._forward_collect: Any = self.actor
        self._forward_train: Any = self.actor
        self._gae: Any = gae
        self.compile_mode: str | None = None
        self.compile_report: dict[str, Any] = {}
        self.phase_profile: dict[str, Any] = {}
        self._v8: Any = v8_decide
        self._observation: Any = build_observation
        self._anchors: Any = self._anchor_tensor
        self.tensor_blocks = TensorBlocks(self.sim)
        if self.sim.cfg.arena_type != "rectangular" or self.sim.grid_w < self.sim.S:
            raise ValueError("sandbox in-place reset requires six distinct rectangular spawn cells")
        self._reset(torch.ones_like(self.episode, dtype=torch.bool), initial=True)
        if recipe.stagger_initial_age:
            # Stagger cohorts, retaining scheduled truncations in every H128 cycle
            # without forcing a scheduled reset on every single tick.
            cohorts = max(1, recipe.episode_limit // min(recipe.horizon, recipe.episode_limit))
            self.age.copy_((self._world_indices % cohorts) * (recipe.episode_limit // cohorts))

    def _allocate_rollout(self) -> dict[str, Any]:
        """Allocate the fixed-shape PPO ownership buffers once per resident assay."""
        h, e, device = self.recipe.horizon, self.recipe.worlds, self.device
        return {
            "obs": (
                torch.empty((h, e, 7, 31, 31), dtype=torch.uint8, device=device),
                torch.empty((h, e, 4, 37, 37), dtype=torch.uint8, device=device),
                torch.empty((h, e, 60), device=device),
            ),
            "mask": torch.empty((h, e, 6), dtype=torch.bool, device=device),
            "action": torch.empty((h, e), dtype=torch.long, device=device),
            "logp": torch.empty((h, e), device=device),
            "value": torch.empty((h, e), device=device),
            "reward": torch.empty((h, e), device=device),
            "terminal": torch.empty((h, e), dtype=torch.bool, device=device),
            "truncated": torch.empty((h, e), dtype=torch.bool, device=device),
            "age": torch.empty((h, e), dtype=torch.long, device=device),
            "next_value": torch.empty((h, e), device=device),
            "advantage": torch.empty((h, e), device=device),
            "returns": torch.empty((h, e), device=device),
        }

    def _bootstrap_buffers(self) -> dict[str, torch.Tensor]:
        """Keep bounded addressing metadata; payload contains only captured rows."""
        slots = (self.recipe.horizon + self.recipe.episode_limit - 1) // self.recipe.episode_limit
        bank = {
            "valid": torch.zeros((slots, self.recipe.worlds), dtype=torch.bool, device=self.device),
            "tick": torch.zeros((slots, self.recipe.worlds), dtype=torch.long, device=self.device),
            "storage": torch.full(
                (slots, self.recipe.worlds), -1, dtype=torch.long, device=self.device
            ),
        }
        self._bootstrap_bank = bank
        return bank

    def enable_compile(self, mode: str, *, backend: str | None = None) -> None:
        """Compile static full-world blocks; record eager fallback for each block.

        Dynamic bootstrap subsets stay eager. ActorCritic retains parameter and
        optimizer ownership; compilation is timing evidence, not bit identity.
        """
        if mode not in ("default", "reduce-overhead"):
            raise ValueError("unsupported compile mode")
        for name, attribute, eager in (
            ("ActorCritic_collect", "_forward_collect", self.actor),
            ("ActorCritic_train", "_forward_train", self.actor),
            ("v8_decide", "_v8", self.tensor_blocks.v8_features),
            ("build_observation", "_observation", self.tensor_blocks.observation),
            ("anchors", "_anchors", self._anchor_tensor),
        ):
            setattr(
                self,
                attribute,
                CompiledBlock(eager, mode, self.compile_report, name, backend=backend),
            )
        self.compile_mode = mode

    @torch.no_grad()
    def _reset(self, mask: torch.Tensor, *, initial: bool = False) -> None:
        """Reset selected worlds in place; live rows and their RNG counters never move."""
        indices = torch.where(mask)[0]
        if not indices.numel():
            return
        if not initial:
            self.episode.index_add_(0, indices, torch.ones_like(indices))
            self.reset_count.add_(indices.numel())
        self.collection_faults[1] |= (self.episode >= self.recipe.max_episodes).any()
        # Fresh sandbox worlds are empty and rectangular.  The first of DeviceSim's
        # 32 deterministic respawn trials always places all six distinct slots; it
        # still advances the counter by all 32 trials to preserve the frozen RNG
        # contract.  This replaces 56 compact index-select/index-copy launches.
        episode = self.episode.index_select(0, indices).clamp_max(self.recipe.max_episodes - 1)
        key = self.keys[episode, indices]
        mixed = (key * 6364136223846793005 + 1442695040888963407) & 0x7FFF_FFFF_FFFF_FFFF
        base_x = mixed.remainder(self.sim.grid_w)
        base_y = (mixed // self.sim.grid_w).remainder(self.sim.grid_h)
        cells = torch.stack(
            (
                (base_x[:, None] + self._slot_indices).remainder(self.sim.grid_w),
                (base_y[:, None] + self._slot_indices // self.sim.grid_w).remainder(
                    self.sim.grid_h
                ),
            ),
            dim=-1,
        )

        def fill(name: str, value: int | float | bool) -> None:
            target = getattr(self.sim, name)
            target.index_fill_(0, indices, value)

        fill("bodies", 0)
        self.sim.bodies[indices, :, 0] = cells
        for name in (
            "head_ptr",
            "boost_frames",
            "frames_since_food",
            "respawn_timer",
            "frame",
            "food_count",
            "_last_death_cause",
            "_last_kills",
        ):
            fill(name, 0)
        for name in ("seg_count", "length", "direction", "_reward_prev_length"):
            fill(name, 1)
        fill("alive", True)
        for name in (
            "food_cells",
            "food_present",
            "corpse_food",
            "_last_reward",
            "_last_done",
            "_last_food_ate",
            "_last_ambient_food_ate",
            "_last_corpse_food_ate",
            "_last_transition_valid",
            "_boosted_this_step",
            "capacity_overflow",
        ):
            fill(name, 0)
        self.sim.rng_key.index_copy_(0, indices, key)
        fill("rng_counter", 32)
        fill("terminal_slots", False)
        self.sim.terminal_slots[indices, 0] = True
        self.collection_faults[2] |= (mask & ~self.sim.alive.all(dim=1)).any()
        self.sim._prepared = False
        self.age.index_fill_(0, indices, 0)
        r2_cache.bump(self.sim)
        if initial:
            self._check_collection_faults()

    def _check_collection_faults(self) -> None:
        """One scalar transfer per horizon, never one check per tick/reset."""
        faults = self.collection_faults.cpu().tolist()
        for failed, reason in zip(
            faults,
            (
                "physical body capacity exceeded: assay INVALID",
                "frozen reset allocation exhausted",
                "device spawn rejection budget exhausted",
            ),
        ):
            if failed:
                raise RuntimeError(reason)

    @torch.no_grad()
    def observe(
        self, sim: DeviceSim | None = None, advisory_mask: torch.Tensor | None = None
    ) -> tuple[tuple[torch.Tensor, ...], torch.Tensor]:
        sim = self.sim if sim is None else sim
        sim.prepare()
        rows = (
            self._rows
            if sim is self.sim
            else torch.stack(
                (
                    torch.arange(sim.E, device=self.device),
                    torch.zeros(sim.E, dtype=torch.long, device=self.device),
                ),
                dim=1,
            )
        )
        full_mask = sim.advisory_mask() if advisory_mask is None else advisory_mask
        legal = full_mask[:, 0]
        if sim is self.sim and isinstance(self._v8, CompiledBlock):
            all_occ, other_occ, own_cells, own_valid = self.tensor_blocks.v8_inputs(sim, rows)
            features = self._v8(
                *self.tensor_blocks.state(sim),
                rows=rows,
                mask=legal,
                all_occ=all_occ,
                other_occ=other_occ,
                own_cells=own_cells,
                own_valid=own_valid,
            )
            ttl, region, food_grid, heads = self.tensor_blocks.observation_inputs(sim, rows)
            return (
                self._observation(
                    *self.tensor_blocks.state(sim),
                    ttl=ttl,
                    region=region,
                    food_grid=food_grid,
                    heads=heads,
                    rows=rows,
                    features=features,
                ),
                legal,
            )
        features = (
            v8_decide(
                sim,
                rows,
                self._v8_q if sim is self.sim else torch.zeros((sim.E, 6), device=self.device),
                legal,
                8.0,
                features=True,
            )
            .feature_planes()
            .reshape(sim.E, 30)
        )
        return build_observation(sim, rows, features), legal

    @torch.no_grad()
    def _anchor_actions(self, full_mask: torch.Tensor, hero_action: torch.Tensor) -> torch.Tensor:
        """Run the fixed greedy-food anchors from the observation's one advisory mask."""
        out = self._action_buffer
        out[:, 0] = hero_action
        turns = r2_cache.const((-1, 0, 1), torch.long, self.device)
        cardinal = r2_cache.const(((0, -1), (1, 0), (0, 1), (-1, 0)), torch.long, self.device)
        priority = r2_cache.const((1, 0, 2), torch.long, self.device)
        out[:, 1:] = self._anchors(
            full_mask[:, 1:, :3],
            self.sim.direction[:, 1:],
            self.sim.heads()[:, 1:],
            self.sim.food_cells,
            self.sim.food_present,
            self.sim.alive[:, 1:],
            turns,
            cardinal,
            priority,
        )
        return out

    @staticmethod
    def _anchor_tensor(
        safe: torch.Tensor,
        heading: torch.Tensor,
        heads: torch.Tensor,
        food: torch.Tensor,
        present: torch.Tensor,
        alive: torch.Tensor,
        turns: torch.Tensor,
        cardinal: torch.Tensor,
        priority: torch.Tensor,
    ) -> torch.Tensor:
        """Five greedy-food seats in one tensor pass with the frozen tie order."""
        safe = torch.where(safe.any(2, keepdim=True), safe, torch.ones_like(safe))
        direction = (heading[:, :, None] + turns) % 4
        candidate = heads[:, :, None] + cardinal[direction]
        # Separate coordinates avoid an E*5*3*F*2 int64 temporary in eager
        # fallback at E4096; the compiled block can fuse these reductions.
        distance = (candidate[..., 0, None] - food[:, None, None, :, 0]).square()
        distance.add_((candidate[..., 1, None] - food[:, None, None, :, 1]).square())
        distance = distance.masked_fill(~present[:, None, None], torch.iinfo(torch.long).max)
        nearest = distance.amin(3)
        nearest = torch.where(present.any(1)[:, None, None], nearest, torch.zeros_like(nearest))
        score = nearest.clamp_max(torch.iinfo(torch.long).max // 8) * 4 + priority
        chosen = score.masked_fill(~safe, torch.iinfo(torch.long).max).argmin(2)
        return torch.where(alive, chosen, torch.ones_like(chosen))

    @torch.no_grad()
    def collect(self) -> dict[str, Any]:
        """Return borrowed rollout storage, valid until the next collect call.

        Retain one valid hero row/world/tick on device; proof callers must freeze
        signatures before a later collection overwrites these buffers.
        """
        if self._rollout is None:
            self._rollout = self._allocate_rollout()
        result = self._rollout
        bank = self._bootstrap_buffers()
        profiler = PhaseProfiler(self.device, self.recipe.horizon)
        captures: dict[str, list[torch.Tensor]] = {name: [] for name in BOOTSTRAP_FIELDS}
        captured_rows = 0
        trunc_ordinal = torch.zeros_like(self.age)
        for tick in range(self.recipe.horizon):
            with profiler.phase("t_obs"):
                self.sim.prepare()
                full_mask = self.sim.advisory_mask()
                obs, legal = self.observe(advisory_mask=full_mask)
            with profiler.phase("t_actor"):
                logits, value = self._forward_collect(*obs)
                log_probs = torch.log_softmax(
                    logits.masked_fill(~allowed_mask(legal), -torch.inf), dim=-1
                )
                action = torch.multinomial(log_probs.exp(), 1, generator=self.generator).squeeze(1)
                logp = log_probs.gather(-1, action[:, None]).squeeze(-1)
                for target, value_at_tick in zip(result["obs"], obs):
                    target[tick].copy_(value_at_tick)
                result["mask"][tick].copy_(legal)
                result["action"][tick].copy_(action)
                result["logp"][tick].copy_(logp)
                result["value"][tick].copy_(value)
                result["age"][tick].copy_(self.age)
            # DevicePolicy inference tensors are consumed by dynamics only, never by autograd.
            with profiler.phase("t_anchor"):
                actions = self._anchor_actions(full_mask, action)
            with profiler.phase("t_step"):
                self.sim.step(actions)
                self.policy.after_step(self.sim)
            with profiler.phase("t_bank_reset"):
                self.collection_faults[0] |= self.sim.capacity_overflow.any()
                self.age += 1
                terminal = result["terminal"][tick]
                terminal.copy_(self.sim._last_done[:, 0])
                truncated = (self.age >= self.recipe.episode_limit) & ~terminal
                result["reward"][tick].copy_(self.sim._last_reward[:, 0])
                result["terminal"][tick].copy_(terminal)
                result["truncated"][tick].copy_(truncated)
                indices = torch.where(truncated)[0]
                if indices.numel():
                    ordinal = trunc_ordinal.index_select(0, indices)
                    bank["valid"][ordinal, indices] = True
                    bank["tick"][ordinal, indices] = tick
                    bank["storage"][ordinal, indices] = torch.arange(
                        captured_rows, captured_rows + indices.numel(), device=self.device
                    )
                    for name in BOOTSTRAP_FIELDS:
                        captures[name].append(getattr(self.sim, name).index_select(0, indices))
                    captured_rows += indices.numel()
                trunc_ordinal.add_(truncated.long())
                self._reset(terminal | truncated)
        with profiler.phase("t_post"):
            # Only actual truncation rows own heavy state; no E*slots payload allocation.
            for name, chunks in captures.items():
                source = getattr(self.sim, name)
                bank[name] = (torch.cat(chunks, dim=0) if chunks else source[:0]).unsqueeze(0)
                chunks.clear()
            self._check_collection_faults()
            final_obs, _ = self.observe()
            final_value = self.actor.critic_only(*final_obs)
            result["next_value"][:-1].copy_(result["value"][1:])
            result["next_value"][-1].copy_(final_value)
            # One combined observation/critic invocation across recorded time limits;
            # ordinal metadata preserves the original slot/world evaluation order.
            rows = torch.where(bank["valid"].flatten())[0]
            if rows.numel():
                subset = bootstrap_subset(self.sim, bank, rows)
                next_obs, _ = self.observe(subset)
                values = self.actor.critic_only(*next_obs)
                ticks = bank["tick"].flatten().index_select(0, rows)
                destination = ticks * self.recipe.worlds + rows.remainder(self.recipe.worlds)
                result["next_value"].flatten().index_copy_(0, destination, values)
            result["advantage"], result["returns"] = self._gae(
                result["reward"],
                result["value"],
                result["next_value"],
                result["terminal"],
                result["truncated"],
                self.recipe.gamma,
                self.recipe.gae_lambda,
            )
        self.phase_profile = profiler.finish()
        return result

    def update(self, rollout: dict[str, Any]) -> dict[str, torch.Tensor]:
        """One minimal clipped PPO update with bounded activation memory."""
        rows = self.recipe.horizon * self.recipe.worlds
        obs = tuple(x.flatten(0, 1) for x in rollout["obs"])
        flat = {k: v.flatten(0, 1) for k, v in rollout.items() if k != "obs"}
        adv = flat["advantage"]
        adv = (adv - adv.mean()) / adv.std(unbiased=False).clamp_min(1e-8)
        last: dict[str, torch.Tensor] = {}
        nonfinite = torch.zeros((), dtype=torch.bool, device=self.device)
        for _ in range(self.recipe.epochs):
            order = torch.randperm(rows, device=self.device, generator=self.generator)
            for start in range(0, rows, self.recipe.minibatch):
                ix = order[start : start + self.recipe.minibatch]

                def compute_loss() -> torch.Tensor:
                    logits, value = self._forward_train(*(x[ix] for x in obs))
                    _, entropy, logp = categorical_terms(
                        logits, flat["mask"][ix], flat["action"][ix]
                    )
                    ratio = (logp - flat["logp"][ix]).exp()
                    policy_loss = -torch.minimum(
                        ratio * adv[ix],
                        ratio.clamp(1 - self.recipe.clip, 1 + self.recipe.clip) * adv[ix],
                    ).mean()
                    value_loss = (value - flat["returns"][ix]).square().mean()
                    return policy_loss + 0.5 * value_loss - 0.01 * entropy.mean()

                loss = compute_loss()
                self.optimizer.zero_grad(set_to_none=True)
                try:
                    loss.backward()
                except Exception as exc:
                    # AOTAutograd can defer compiler failures until backward.
                    # No optimizer step has occurred; clear partial gradients and
                    # recompute this same minibatch eagerly before proceeding.
                    if (
                        not isinstance(self._forward_train, CompiledBlock)
                        or self._forward_train.compiled is None
                    ):
                        raise
                    self._forward_train._fallback(exc)
                    self.optimizer.zero_grad(set_to_none=True)
                    loss = compute_loss()
                    loss.backward()
                nonfinite |= ~torch.isfinite(loss)
                norm = nn.utils.clip_grad_norm_(
                    self.actor.parameters(), 0.5, error_if_nonfinite=False
                )
                nonfinite |= ~torch.isfinite(norm)
                self.optimizer.step()
                last = {"loss": loss.detach(), "grad_norm": norm.detach()}
        if bool(nonfinite):
            raise RuntimeError("nonfinite PPO loss/gradient: assay INVALID")
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
                "collection_faults": self.collection_faults,
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
            setattr(
                self.sim, name, value.to(self.device) if isinstance(value, torch.Tensor) else value
            )
        for name in ("keys", "episode", "age", "reset_count", "collection_faults", "cycle"):
            value = state[name]
            setattr(self, name, value.to(self.device) if isinstance(value, torch.Tensor) else value)
        self.generator.set_state(state["generator"].cpu())
        torch.set_rng_state(state["torch_rng"].cpu())
        if self.device.type == "cuda":
            torch.cuda.set_rng_state_all([x.cpu() for x in state["cuda_rng"]])
        self.policy.vector_prepared = state["vector_prepared"]
        # These are derived device-only launch scratch, never checkpoint state.
        self._world_indices = torch.arange(self.recipe.worlds, device=self.device)
        self._slot_indices = torch.arange(self.sim.S, device=self.device)
        self._first_counter = torch.ones(self.recipe.worlds, dtype=torch.long, device=self.device)
        self._rows = torch.stack((self._world_indices, torch.zeros_like(self.age)), dim=1)
        r2_cache.enable(self.sim)

    def save(self, path: Path) -> None:
        """Serialize a complete boundary checkpoint; immutable envelope is owned by runner."""
        torch.save(self.snapshot(), path)
