"""Fresh-seed, sandbox-local opponent rosters for resident PPO training.

The canonical gate construction remains the authority for its member order,
checkpoint identities and v8 wrapper.  This module contains a small local
projection of that construction for new ``a100-dev`` roots; it never reads a
saved gate bank or trajectory.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike.anchors_device import random_table
from research.gpu_sim_spike.device_policy import DevicePolicy
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime

FRP3_S12_SHA256 = "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
CHAMPION_SHA256 = "43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93"
MIXES = ("scripted", "frozen", "mixed")
MIX_IDS = {name: index for index, name in enumerate(MIXES)}


def vector61_params(cfg: Any) -> Any:
    """Freeze the Mac gate's network normalizers independently of global config.

    Source: apex_safety_20260926/deployment.yaml plus default.yaml's vector
    knobs. World mechanics follow TRAIN-01's explicit mechanics-v1 recipe.
    """
    from src.simd_env.vector61_featurizer import Vector61Params

    return Vector61Params(
        game_width=int(cfg.game_width),
        game_height=int(cfg.game_height),
        segment_size=int(cfg.segment_size),
        max_length=150,
        num_sectors=16,
        danger_max_distance=30,
        use_boundary_as_danger=True,
        min_boost_length=int(cfg.min_boost_length),
        food_capacity=int(cfg.max_food),
        use_free_space=True,
        arena_type="rectangular",
    )


@dataclass(frozen=True)
class TrainingRoster:
    """One ordered five-opponent gate-shaped roster."""

    mix: str
    slots: tuple[tuple[str, str], ...]
    veto: str = "v8"
    veto_lambda: float = 8.0

    def __post_init__(self) -> None:
        if self.mix not in MIX_IDS or len(self.slots) != 5:
            raise ValueError("a roster must name a known mix and exactly five slots")
        if self.veto != "v8" or self.veto_lambda != 8.0:
            raise ValueError("training rosters use the gate's v8 wrapper at lambda 8")
        for kind, reference in self.slots:
            if kind not in {"checkpoint", "scripted"} or not reference:
                raise ValueError("unsupported gate roster member")

    @property
    def identity(self) -> tuple[Any, ...]:
        """Immutable dispatch identity; do not collapse slot order."""
        return (self.mix, self.slots, self.veto, self.veto_lambda)


def _balanced_counts(worlds: int) -> dict[str, int]:
    if worlds < 3:
        raise ValueError("balanced three-mix collection needs at least three worlds")
    base, extra = divmod(worlds, len(MIXES))
    return {mix: base + (index < extra) for index, mix in enumerate(MIXES)}


def training_rosters(
    worlds: int,
    *,
    namespace: str = "a100-dev/gpu-train-01/rosters/v1",
    horizon: int = 5000,
) -> tuple[TrainingRoster, ...]:
    """Construct balanced fresh rosters using the copied T34 descriptor.

    This is the small, source-closure-safe projection of
    ``t34_round0.bank._rows``: the gate replaces every learned strict member
    with FRP3/champion in the index/slot rotation below.  Tests compare this
    local descriptor to that CPU source oracle, but runtime imports no T34
    package and ships no protected research dependency closure.
    """
    if horizon != 5000:
        raise ValueError("TRAIN-01 rosters are defined for H5000")
    counts = _balanced_counts(worlds)
    ordered: list[TrainingRoster] = []
    emitted = {mix: 0 for mix in MIXES}
    while len(ordered) != worlds:
        for mix in MIXES:
            index = emitted[mix]
            if index >= counts[mix]:
                continue
            if mix == "scripted":
                slots = (("scripted", "greedy_food"),) * 5
            elif mix == "frozen":
                slots = tuple(
                    ("checkpoint", (FRP3_S12_SHA256, CHAMPION_SHA256)[(index + slot) % 2])
                    for slot in range(5)
                )
            else:
                slots = tuple(
                    (
                        ("checkpoint", (FRP3_S12_SHA256, CHAMPION_SHA256)[(index + slot) % 2])
                        if slot % 2 == 0
                        else ("scripted", "random_safe")
                    )
                    for slot in range(5)
                )
            ordered.append(
                TrainingRoster(
                    mix=mix,
                    slots=slots,
                )
            )
            emitted[mix] += 1
    return tuple(ordered)


def roster_ids(rosters: Sequence[TrainingRoster], device: torch.device | str) -> torch.Tensor:
    """Return stable equal-mix loss labels (scripted=0, frozen=1, mixed=2)."""
    return torch.tensor([MIX_IDS[row.mix] for row in rosters], device=device, dtype=torch.long)


def proxy_rosters(mix: str, worlds: int) -> tuple[TrainingRoster, ...]:
    """Return a deterministic fixed-mix roster prefix for the paired GPU proxy."""
    if mix not in MIX_IDS or worlds < 1:
        raise ValueError("proxy requires a known mix and at least one world")
    rows = training_rosters(max(3, worlds * 3))
    return tuple(row for row in rows if row.mix == mix)[:worlds]


_ENV_FIELDS = (
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


def _view(sim: Any, rows: torch.Tensor) -> Any:
    """Private read-only environment view used only during opponent inference."""
    out = object.__new__(type(sim))
    out.__dict__ = sim.__dict__.copy()
    out.E = int(rows.numel())
    for name in _ENV_FIELDS:
        value = getattr(sim, name)
        setattr(out, name, value.index_select(0, rows))
    r2_cache.inherit(out, sim)
    return out


class DeviceRosterRuntime:
    """Batch opponent decisions by complete immutable roster identity.

    The vector carry belongs to each identity group and is reset only for
    completed worlds in that group.  There is no CPU policy dispatch in this
    adapter.  Frozen checkpoint modules are supplied explicitly by the runner
    after hash verification; this module never guesses a checkpoint path.
    """

    def __init__(
        self,
        sim: Any,
        rosters: Sequence[TrainingRoster],
        *,
        b8_network: nn.Module,
        checkpoint_networks: Mapping[str, nn.Module],
        seeds: Sequence[int] | None = None,
    ) -> None:
        if len(rosters) != int(sim.E):
            raise ValueError("every resident world needs one immutable roster")
        self.device = torch.device(sim.device)
        self.rosters = tuple(rosters)
        self._groups: list[tuple[torch.Tensor, DevicePolicy, dict[int, torch.Tensor]]] = []
        seed_values = (
            tuple(range(int(sim.E))) if seeds is None else tuple(int(value) for value in seeds)
        )
        if len(seed_values) != int(sim.E):
            raise ValueError("seed count must equal resident world count")
        grouped: dict[tuple[Any, ...], list[int]] = {}
        for index, roster in enumerate(self.rosters):
            grouped.setdefault(roster.identity, []).append(index)
        for identity, indexes in grouped.items():
            rows = torch.tensor(indexes, dtype=torch.long, device=self.device)
            roster = self.rosters[indexes[0]]
            nets: list[nn.Module | None] = []
            kinds: list[str | None] = []
            tables: list[torch.Tensor | None] = []
            random_slots: dict[int, torch.Tensor] = {}
            for slot, (kind, reference) in enumerate(roster.slots, start=1):
                if kind == "checkpoint":
                    if reference not in checkpoint_networks:
                        raise KeyError(f"missing verified frozen checkpoint {reference}")
                    nets.append(checkpoint_networks[reference].to(self.device).eval())
                    kinds.append(None)
                    tables.append(None)
                else:
                    nets.append(None)
                    kinds.append("greedy_food" if reference == "random_safe" else reference)
                    tables.append(None)
                    if reference == "random_safe":
                        random_slots[slot] = torch.stack(
                            [
                                random_table(seed_values[index], slot, self.device)
                                for index in indexes
                            ]
                        )
            runtime = DeviceVectorRuntime(vector61_params(sim.cfg))
            controlled = torch.tensor(
                [[False] + [kind == "checkpoint" for kind, _ in roster.slots]],
                device=self.device,
                dtype=torch.bool,
            ).expand(len(indexes), -1)
            runtime.bind(_view(sim, rows), controlled)
            self._groups.append(
                (
                    rows,
                    DevicePolicy(
                        self.device,
                        b8_network,
                        tuple(nets),
                        tuple(kinds),
                        tuple(tables),
                        runtime,
                        vector_prepared=False,
                        v8_lambda=roster.veto_lambda,
                    ),
                    random_slots,
                )
            )

    @torch.inference_mode()
    def actions(self, sim: Any, hero_actions: torch.Tensor) -> torch.Tensor:
        if hero_actions.shape != (sim.E,):
            raise ValueError("hero actions must have shape (E,)")
        out = torch.empty((sim.E, sim.S), dtype=torch.long, device=self.device)
        for rows, policy, random_slots in self._groups:
            sub = _view(sim, rows)
            action = policy.actions(sub, hero_actions.index_select(0, rows))
            for slot, table in random_slots.items():
                safe = sub.advisory_mask()[:, slot, :3]
                safe = torch.where(safe.any(1, keepdim=True), safe, torch.ones_like(safe))
                frame = (sub.frame - 1).clamp(0, table.shape[1] - 1)
                index = torch.arange(sub.E, device=self.device)
                rank = table[index, frame, safe.sum(1) - 1]
                action[:, slot] = ((safe.cumsum(1) - 1 == rank[:, None]) & safe).long().argmax(1)
            out.index_copy_(0, rows, action)
        return out

    @torch.inference_mode()
    def after_step(self, sim: Any) -> None:
        for rows, policy, random_slots in self._groups:
            policy.after_step(_view(sim, rows))

    @torch.inference_mode()
    def reset(self, mask: torch.Tensor, seeds: Sequence[int] | None = None) -> None:
        """Clear only reset worlds' vector carry; roster identity never changes."""
        mask = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
        for rows, policy, random_slots in self._groups:
            local = mask.index_select(0, rows)
            runtime = policy.vector_runtime
            if not bool(local.any()) or runtime is None:
                continue
            for name in ("_carry_valid", "sel_ready", "sel_fresh"):
                value = getattr(runtime, name)
                if value is not None:
                    value[local] = False
            for name in ("prev_dist",):
                value = getattr(runtime, name)
                if value is not None:
                    value[local] = torch.inf
            for name in ("prev_id",):
                value = getattr(runtime, name)
                if value is not None:
                    value[local] = -1
            policy.vector_prepared = False
            if seeds is not None:
                for slot, table in random_slots.items():
                    for local_index, global_index in enumerate(rows.cpu().tolist()):
                        if bool(mask[global_index]):
                            table[local_index].copy_(
                                random_table(int(seeds[global_index]), slot, self.device)
                            )

    def state_dict(self) -> dict[str, Any]:
        """Snapshot only mutable runtime carry; networks are immutable inputs."""
        groups = []
        for rows, policy, random_slots in self._groups:
            runtime = policy.vector_runtime
            state = {
                key: value.detach().clone()
                for key, value in runtime.__dict__.items()
                if isinstance(value, torch.Tensor)
            }
            groups.append(
                {
                    "rows": rows.detach().cpu(),
                    "runtime": state,
                    "prepared": policy.vector_prepared,
                    "random_slots": {
                        int(slot): table.detach().clone() for slot, table in random_slots.items()
                    },
                }
            )
        return {"schema": "sandbox-a-training-rosters/v1", "groups": groups}

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "sandbox-a-training-rosters/v1":
            raise ValueError("foreign roster runtime state")
        groups = state.get("groups")
        if not isinstance(groups, list) or len(groups) != len(self._groups):
            raise ValueError("roster runtime group mismatch")
        for saved, (rows, policy, random_slots) in zip(groups, self._groups):
            if not torch.equal(torch.as_tensor(saved["rows"]), rows.cpu()):
                raise ValueError("roster runtime row identity mismatch")
            for name, value in saved["runtime"].items():
                current = getattr(policy.vector_runtime, name)
                if tuple(current.shape) != tuple(value.shape):
                    raise ValueError(f"roster runtime state shape mismatch: {name}")
                current.copy_(value.to(self.device))
            saved_tables = saved.get("random_slots", {})
            if {int(slot) for slot in saved_tables} != set(random_slots):
                raise ValueError("roster random-safe slot identity mismatch")
            for slot, value in saved_tables.items():
                current = random_slots[int(slot)]
                if tuple(current.shape) != tuple(value.shape):
                    raise ValueError("roster random-safe table shape mismatch")
                current.copy_(value.to(self.device))
            policy.vector_prepared = bool(saved["prepared"])
