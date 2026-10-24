"""Frozen fixtures and fixed-shape forward instrumentation for T4A-DEV-01.

The module deliberately owns no R2 engine code.  It only creates the three
pre-registered dev roots, advances them to their frozen history seams, and
wraps existing policy modules without changing their real-row semantics.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import torch
from torch import nn

from research.gpu_sim_spike import a100_ruler as ruler
from research.gpu_sim_spike import integration_engine as engine
from research.gpu_sim_spike import r2_cache

ROSTERS = ("frozen", "mixed", "scripted")
HISTORY_OFFSETS = (0, 3, 11, 23)
ROOT_SEEDS = {
    "main": (9402401, 9402402, 9402403),
    "canary": (9402411, 9402412, 9402413),
    "smoke": (9402421, 9402422, 9402423),
}
LAYOUTS = ("canonical", "reversed", "interspersed")
FORWARD_ROWS = 1024
_SEAM_BANK: dict[tuple[str, str, str, bool, bool], tuple[tuple[Any, Any], ...]] = {}


class T4ANotReady(RuntimeError):
    """The pinned policy cannot receive a semantics-preserving padded call."""


@dataclass(frozen=True)
class ForwardCall:
    """A compact, typed fixed-shape proof for one physical inner forward."""

    name: str
    real_rows: int
    input_shapes: tuple[tuple[int, ...], ...]
    padded_shapes: tuple[tuple[int, ...], ...]
    output_shape: tuple[int, ...]


@dataclass(frozen=True)
class RowMapProof:
    """Independent insertion/extraction maps for the four frozen seams.

    ``insertion_rows`` is the map used while building the physical batch.
    ``expected_rows`` and ``extraction_rows`` are derived independently from
    the frozen layout declaration.  Keeping all three in the sealed payload
    makes a shared-map implementation error visible to the reducer/audit.
    """

    layout: str
    worlds: int
    logical_ids: tuple[int, int, int, int]
    history_offsets: tuple[int, int, int, int]
    insertion_rows: tuple[int, int, int, int]
    expected_rows: tuple[int, int, int, int]
    extraction_rows: tuple[int, int, int, int]
    placement_checks: tuple[bool, ...] = ()

    def exported(self) -> dict[str, Any]:
        """Return only JSON primitives for the immutable producer payload."""
        return {
            "layout": self.layout,
            "worlds": self.worlds,
            "logical_ids": list(self.logical_ids),
            "history_offsets": list(self.history_offsets),
            "insertion_rows": list(self.insertion_rows),
            "expected_rows": list(self.expected_rows),
            "extraction_rows": list(self.extraction_rows),
            "placement_valid": self.insertion_rows == self.expected_rows
            and (not self.placement_checks or all(self.placement_checks)),
            "extraction_valid": self.extraction_rows == self.expected_rows,
            "placement_checks": list(self.placement_checks),
        }


class Fixed1024Forward(nn.Module):
    """Pad a native policy call to 1,024 rows and return only its real rows.

    This accepts B8's raster tuple and vector61's single tensor equally.  It
    rejects optional/non-row arguments instead of guessing how to pad them.
    """

    def __init__(self, inner: nn.Module, name: str) -> None:
        super().__init__()
        self.inner = inner
        self.name = name
        self.calls: list[ForwardCall] = []
        self.outputs: list[torch.Tensor] = []

    def forward(self, *args: torch.Tensor) -> torch.Tensor:
        if not args or any(not isinstance(arg, torch.Tensor) or arg.ndim < 1 for arg in args):
            raise T4ANotReady(f"{self.name}: non-row tensor input")
        rows = int(args[0].shape[0])
        if not 0 < rows <= FORWARD_ROWS or any(int(arg.shape[0]) != rows for arg in args):
            raise T4ANotReady(f"{self.name}: incompatible leading dimensions")
        padded = tuple(
            torch.cat(
                (
                    arg,
                    torch.zeros(
                        (FORWARD_ROWS - rows, *arg.shape[1:]), dtype=arg.dtype, device=arg.device
                    ),
                ),
                dim=0,
            )
            for arg in args
        )
        out = self.inner(*padded)
        if not isinstance(out, torch.Tensor) or out.ndim < 1 or int(out.shape[0]) != FORWARD_ROWS:
            raise T4ANotReady(f"{self.name}: inner forward was not fixed shape")
        self.calls.append(
            ForwardCall(
                self.name,
                rows,
                tuple(tuple(x.shape) for x in args),
                tuple(tuple(x.shape) for x in padded),
                tuple(out.shape),
            )
        )
        self.outputs.append(out[:rows].detach().clone())
        return out[:rows]


def mode(*, smoke: bool, canary: bool) -> str:
    if smoke and canary:
        raise ValueError("smoke and canary are exclusive")
    return "smoke" if smoke else "canary" if canary else "main"


def seed_for(roster: str, *, smoke: bool, canary: bool) -> int:
    if roster not in ROSTERS:
        raise ValueError("unknown frozen T4A roster")
    return ROOT_SEEDS[mode(smoke=smoke, canary=canary)][ROSTERS.index(roster)]


def layout_rows(worlds: int, layout: str) -> tuple[int, int, int, int]:
    if worlds < 4 or layout not in LAYOUTS:
        raise ValueError("invalid T4A layout")
    if layout == "canonical":
        rows = (0, 1, 2, 3)
    elif layout == "reversed":
        rows = tuple(worlds - 1 - item for item in range(4))
    else:
        rows = tuple((item * (worlds - 1)) // 3 for item in range(4))
    if len(set(rows)) != 4:
        raise T4ANotReady("layout failed to retain four unique target rows")
    return rows  # type: ignore[return-value]


def _frozen_expected_rows(worlds: int, layout: str) -> tuple[int, int, int, int]:
    """Return the protocol-declared positions without consulting insertion code.

    This deliberately duplicates the small frozen table calculation.  It is
    an audit oracle, so sharing ``layout_rows`` here would remove the check it
    is intended to provide.
    """
    if worlds < 4 or layout not in LAYOUTS:
        raise ValueError("invalid T4A layout")
    if layout == "canonical":
        return (0, 1, 2, 3)
    if layout == "reversed":
        return (worlds - 1, worlds - 2, worlds - 3, worlds - 4)
    return (0, (worlds - 1) // 3, 2 * (worlds - 1) // 3, worlds - 1)


def row_map_proof(worlds: int, layout: str) -> RowMapProof:
    """Build separately checked row maps for insertion and logical extraction."""
    insertion_rows = layout_rows(worlds, layout)
    expected_rows = _frozen_expected_rows(worlds, layout)
    # Do not use insertion_rows as an extraction source: this independent map
    # preserves the declared logical order even if insertion is defective.
    extraction_rows = _frozen_expected_rows(worlds, layout)
    proof = RowMapProof(
        layout=layout,
        worlds=worlds,
        logical_ids=(0, 1, 2, 3),
        history_offsets=HISTORY_OFFSETS,
        insertion_rows=insertion_rows,
        expected_rows=expected_rows,
        extraction_rows=extraction_rows,
    )
    if proof.insertion_rows != proof.expected_rows:
        raise T4ANotReady("declared physical placement differs from frozen layout")
    if proof.extraction_rows != proof.expected_rows:
        raise T4ANotReady("logical extraction map differs from frozen layout")
    return proof


def pin_fixed_forwards(policy: Any) -> list[Fixed1024Forward]:
    """Install one shared wrapper per actual module identity."""
    wrappers: dict[int, Fixed1024Forward] = {}

    def wrap(value: Any, name: str) -> Any:
        if value is None:
            return None
        if not isinstance(value, nn.Module):
            raise T4ANotReady(f"{name}: learned policy is not a torch module")
        return wrappers.setdefault(id(value), Fixed1024Forward(value, name))

    policy.b8_network = wrap(policy.b8_network, "hero_b8")
    policy.opponent_networks = tuple(
        wrap(net, f"opponent_{slot}") for slot, net in enumerate(policy.opponent_networks, 1)
    )
    return list(wrappers.values())


def build_case(
    checkpoint: Path, roster: str, worlds: int, device: str, *, smoke: bool, canary: bool
) -> tuple[tuple, list[Fixed1024Forward]]:
    """Build the admitted I-prime case and bind the T4A fixed forward wrapper.

    R2's factory is the sole construction path.  Its resident root is then
    assigned the frozen dev key, avoiding any production allocation input.
    """
    # R2's convenient benchmark factory hardcodes its own source root.  T4A
    # must instead construct its frozen synthetic root *before* device policy
    # tables are made, otherwise random_safe and initial food retain that old
    # root even if ``rng_key`` is overwritten later.
    from research.gpu_sim_spike.device_policy import DevicePolicy
    from research.gpu_sim_spike.device_sim import DeviceSim
    from research.t34_exit_b8 import trajectory

    key = seed_for(roster, smoke=smoke, canary=canary)
    row = trajectory.make_world_rows(
        {
            "namespace": f"t4a-dev-01/{mode(smoke=smoke, canary=canary)}/{roster}",
            "mixes": (roster,),
        },
        1,
    )[0]
    row["world_seed"] = key
    row["seed"] = key
    world = trajectory.build_world(row, checkpoint)
    one_sim = DeviceSim.from_reference(world.sim, device=device)
    one_policy = DevicePolicy.from_world(world, device=device, sim=one_sim)
    sim, policy = one_sim.repeat_interleave(worlds), one_policy.repeat_interleave(worlds)
    for target in (sim,):
        engine._set_move_backend(target, "indexed")
        r2_cache.enable(target, debug=False)
    engine._install_geometry(policy, "native")
    template, policy_template = sim.clone(), policy.repeat_interleave(1)
    engine._set_move_backend(template, "indexed")
    r2_cache.enable(template, debug=False)
    engine._install_geometry(policy_template, "native")
    # DevicePolicy defaults to vector_prepared=True even though this fresh
    # device snapshot is an unprepared selector seam.  Freeze that lifecycle
    # boundary explicitly for both live/template arms.
    for target_sim, target_policy in ((sim, policy), (template, policy_template)):
        target_sim._prepared = False
        target_policy.vector_prepared = False
        if not hasattr(target_sim, "capacity_overflow"):
            target_sim.capacity_overflow = torch.zeros(
                (target_sim.E, target_sim.S), dtype=torch.bool, device=target_sim.device
            )
    collection = ruler._collection(1, worlds, sim.device)
    case = sim, policy, template, policy_template, collection, "I_prime"
    wrappers = pin_fixed_forwards(policy)
    pin_fixed_forwards(policy_template)
    return case, wrappers


def advance_to_offset(case: tuple, offset: int) -> None:
    """Advance a real policy/simulator lifecycle to a post-step unprepared seam."""
    sim, policy, _template, _policy_template, _collection, _arm = case
    if offset not in HISTORY_OFFSETS:
        raise ValueError("history offset is outside the frozen schedule")
    with engine._arm("I"), torch.inference_mode():
        for index in range(offset):
            collection = ruler._collection(1, sim.E, sim.device)
            ruler._full_tick(sim, policy, collection, index=0)
    if sim._prepared or policy.vector_prepared:
        raise T4ANotReady("seam is not post-step/unprepared")


def clone_case(case: tuple) -> tuple:
    """Clone the mutable state used by a replay without aliasing simulator state."""
    sim, policy, template, policy_template, collection, arm = case
    # Policy repetition preserves its configured runtime class and deep-copies
    # per-world runtime tensors.  One world is enough because fixture state is
    # copied from the prepared seam by the harness before each layout run.
    return (
        sim.clone(),
        copy.deepcopy(policy),
        template.clone(),
        copy.deepcopy(policy_template),
        {name: value.clone() for name, value in collection.items()},
        arm,
    )


def build_mapped_case(
    checkpoint: Path,
    roster: str,
    worlds: int,
    layout: str,
    device: str,
    *,
    smoke: bool,
    canary: bool,
) -> tuple[tuple, list[Fixed1024Forward], RowMapProof]:
    """Build a physical batch and place four independently progressed seams.

    The transfer is a device-to-device copy of every world-indexed simulator
    and vector-runtime tensor.  This preserves a logical history when layouts
    permute physical rows, including RNG/carry state rather than regenerating
    either from the new row number.
    """
    destination, wrappers = build_case(
        checkpoint, roster, worlds, device, smoke=smoke, canary=canary
    )
    row_proof = row_map_proof(worlds, layout)
    rows = row_proof.insertion_rows
    dest_sim, dest_policy = destination[:2]
    bank_key = (str(checkpoint.resolve()), roster, device, smoke, canary)
    if bank_key not in _SEAM_BANK:
        seams: list[tuple[Any, Any]] = []
        source, _source_wrappers = build_case(
            checkpoint, roster, 1, device, smoke=smoke, canary=canary
        )
        previous = 0
        for offset in HISTORY_OFFSETS:
            for _ in range(offset - previous):
                collection = ruler._collection(1, source[0].E, source[0].device)
                with engine._arm("I"), torch.inference_mode():
                    ruler._full_tick(source[0], source[1], collection, 0)
            previous = offset
            seams.append((source[0].clone(), copy.deepcopy(source[1])))
        _SEAM_BANK[bank_key] = tuple(seams)
    placement_checks: list[bool] = []
    for target_row, (src_sim, src_policy) in zip(rows, _SEAM_BANK[bank_key]):
        placement_ok = True
        for owner_dst, owner_src in (
            (dest_sim, src_sim),
            (dest_policy.vector_runtime, src_policy.vector_runtime),
        ):
            for name, value in vars(owner_dst).items():
                source_value = getattr(owner_src, name, None)
                if (
                    isinstance(value, torch.Tensor)
                    and isinstance(source_value, torch.Tensor)
                    and value.ndim
                    and source_value.ndim
                    and int(value.shape[0]) == worlds
                    and int(source_value.shape[0]) == 1
                ):
                    value[target_row].copy_(source_value[0])
                    placement_ok = placement_ok and torch.equal(value[target_row], source_value[0])
        # ``vector_prepared`` is a scalar lifecycle flag and all four frozen
        # seams are explicitly post-step/unprepared.
        dest_policy.vector_prepared = False
        dest_sim._prepared = False
        placement_checks.append(placement_ok)
    _fill_sacrificial(
        dest_sim,
        dest_policy.vector_runtime,
        set(rows),
        seed_for(roster, smoke=smoke, canary=canary),
    )
    # This is intentionally checked after the real device copies.  A correct
    # declaration is insufficient if a future insertion loop changes its row
    # selection after the maps were constructed.
    row_proof = replace(row_proof, placement_checks=tuple(placement_checks))
    if rows != row_proof.expected_rows or not all(row_proof.placement_checks):
        raise T4ANotReady("physical seam insertion diverged from frozen layout")
    return destination, wrappers, row_proof


def _fill_sacrificial(sim: Any, runtime: Any, targets: set[int], root_seed: int) -> None:
    """Make non-target lanes valid heterogeneous distractors before inference.

    They retain the same roster tables, while per-lane bodies/carry/RNG are
    deterministic synthetic dev state.  The long and short lengths force the
    R2 live-window boundary without modifying the candidate mechanics.
    """
    rows_list = [r for r in range(sim.E) if r not in targets]
    if not rows_list:
        return
    lengths = (8, 15, 31, 63, 127, 255, 256, 257, 511)
    chosen = [lengths[r % len(lengths)] for r in rows_list]
    width = sim.grid_w - 4
    band_height = (sim.grid_h - 4) // sim.S
    if sim.cap <= max(chosen) + 1 or width * band_height < max(chosen):
        raise T4ANotReady("arena/capacity cannot hold the frozen distractor schedule")
    rows = torch.tensor(rows_list, dtype=torch.long, device=sim.device)
    n = torch.tensor(chosen, dtype=torch.long, device=sim.device)
    slots = torch.arange(sim.S, device=sim.device)
    ptr = (rows[:, None] + slots[None] * 17) % sim.cap
    offsets = torch.arange(max(chosen), device=sim.device)
    # A contiguous self-avoiding sweep, head first.  The physical ring rotation
    # is independent of its logical geometry.  No Python scalar CUDA writes.
    sweep = (n[:, None] - 1 - offsets[None]).clamp_min(0)
    line, column = sweep // width, sweep % width
    x = 2 + torch.where(line % 2 == 0, column, width - 1 - column)
    y = 2 + slots[None, :, None] * band_height + line[:, None, :]
    cells = torch.stack((x[:, None, :].expand(-1, sim.S, -1), y), dim=-1)
    rings = (ptr[..., None] - offsets) % sim.cap
    old = sim.bodies[rows[:, None, None], slots[None, :, None], rings]
    valid = offsets[None, None, :] < n[:, None, None]
    sim.bodies[rows[:, None, None], slots[None, :, None], rings] = torch.where(
        valid[..., None], cells, old
    )
    sim.head_ptr[rows] = ptr
    sim.seg_count[rows] = n[:, None]
    sim.length[rows] = n[:, None]
    sim._reward_prev_length[rows] = n[:, None]
    delta = cells[:, :, 0] - cells[:, :, 1]
    sim.direction[rows] = torch.where(delta[..., 0] == 1, 1, torch.where(delta[..., 0] == -1, 3, 2))
    sim.frames_since_food[rows] = rows[:, None] % 5
    sim.rng_key[rows] = root_seed + 1009 * (rows + 1)
    sim.rng_counter[rows] = rows
    sim.alive[rows] = True
    sim.respawn_timer[rows, 1] = torch.where(rows % 7 == 0, 1, 0)
    sim.alive[rows, 1] = rows % 7 != 0
    sim.respawn_timer[rows, 2] = torch.where(rows % 11 == 0, 3, 0)
    sim.alive[rows, 2] = rows % 11 != 0
    # Remove food covered by the new live bodies, retaining ordered list and
    # corpse flags.  This uses geometric membership rather than a huge
    # E×food×all-segments temporary.
    food = sim.food_cells.index_select(0, rows)
    fx = food[..., 0] - 2
    fy = food[..., 1, None] - (2 + slots * band_height)
    flat = fy * width + torch.where(fy % 2 == 0, fx[..., None], width - 1 - fx[..., None])
    occupied = (
        (fx[..., None] >= 0)
        & (fx[..., None] < width)
        & (fy >= 0)
        & (flat < n[:, None, None])
        & sim.alive[rows, None, :]
    ).any(-1)
    present = sim.food_present.index_select(0, rows) & ~occupied
    order = torch.where(
        present,
        torch.arange(sim.food_capacity, device=sim.device),
        torch.arange(sim.food_capacity, device=sim.device) + sim.food_capacity,
    )
    order = order.argsort(dim=1, stable=True)
    sim.food_cells[rows] = food.gather(1, order[..., None].expand(-1, -1, 2))
    sim.corpse_food[rows] = sim.corpse_food.index_select(0, rows).gather(1, order)
    sim.food_present[rows] = present.gather(1, order)
    sim.corpse_food[rows] &= sim.food_present[rows]
    sim.food_count[rows] = present.sum(1)
    for name in ("_carry_valid", "sel_ready", "sel_fresh"):
        getattr(runtime, name)[rows] = False
    runtime._alive_post[rows] = sim.alive[rows]
    if not any(v <= 255 for v in chosen) or not any(v >= 257 for v in chosen):
        raise T4ANotReady("sacrificial lane schedule lacks the R2 live-window crossing")


def history_schedule() -> tuple[dict[str, int | str], ...]:
    return tuple(
        {
            "logical_id": index,
            "history_offset": offset,
            "name": ("root", "short", "medium", "long")[index],
        }
        for index, offset in enumerate(HISTORY_OFFSETS)
    )
