"""R2 (L1 live windows, L2 per-state memo, L3 device constants) exactness tests.

Every comparison is R2 versus a *pinned-mode twin*: a clone of the same state
that was never registered with :mod:`r2_cache`, so it executes the unmodified
I0 runtime code.  The engine-level comparison against the actual pinned source
tree (``968f0276``) is ``test_r2_engine_identity.py``.  CUDA variants are run by
the R2 A100 canary/main native parity step (``-k cuda``).
"""

from __future__ import annotations

from collections import Counter
from unittest.mock import patch

import pytest
import torch

from research.gpu_sim_spike import r2_cache
from research.gpu_sim_spike import realistic_benchmark as realistic
from research.gpu_sim_spike.anchors_device import choose_anchor
from research.gpu_sim_spike.device_sim import DeviceSim
from research.gpu_sim_spike.ego_device import _ttl_by_owner, build_observation
from research.gpu_sim_spike.integration_engine import inventory
from research.gpu_sim_spike.masks_device import advisory_mask
from research.gpu_sim_spike.v8_device import v8_decide
from research.gpu_sim_spike.vector_device import DeviceVectorRuntime
from src.simd_env.batch_sim import BatchSimConfig

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only"),
    ),
]
WIDTH = 141  # serpentine row width inside the production 145-cell board


def _config(envs: int, cap: int) -> BatchSimConfig:
    return BatchSimConfig(
        num_envs=envs,
        num_snakes=6,
        game_width=1450,
        game_height=830,
        initial_food=0,
        max_food=20,
        max_capacity=cap,
        body_storage_capacity=cap,
        mechanics_version=2,
    )


def _serpentine(slot: int, count: int) -> torch.Tensor:
    """Head-first cells of a band-confined serpentine (12 rows per slot)."""
    k = torch.arange(count)
    row = k // WIDTH
    x = torch.where(row % 2 == 0, 2 + k % WIDTH, 2 + WIDTH - 1 - k % WIDTH)
    y = 1 + slot * 13 + row
    return torch.stack((x, y), dim=1).flip(0)  # tail..head -> head first


def world(
    device: str,
    counts: list[list[int]],
    *,
    cap: int = 10400,
    grow: int = 0,
    seed: int = 7,
) -> DeviceSim:
    """Long, wrapped, padded rings with stale in-board garbage beyond the tail.

    ``counts[e][s]`` is a slot's live segment count (0 = dead row).  Heads are
    placed so that half of the live prefixes cross storage index zero.  Padding is
    a mix of off-board ``-7`` sentinels and *in-board* stale cells, which would
    corrupt any painter that read past ``seg_count``.
    """
    gen = torch.Generator().manual_seed(seed)
    envs = len(counts)
    sim = DeviceSim(_config(envs, cap), device="cpu", cap=cap, move_backend="indexed")
    sim.bodies.fill_(-7)
    stale = torch.randint(0, 80, (envs, 6, cap, 2), generator=gen)
    keep = torch.rand((envs, 6, cap), generator=gen) < 0.5
    sim.bodies = torch.where(keep[..., None], stale, sim.bodies)
    for e, row in enumerate(counts):
        for s, count in enumerate(row):
            head = int(torch.randint(0, cap, (1,), generator=gen))
            if (e + s) % 2 == 0 and count > 2:
                head = (count // 3 + e) % cap  # live prefix crosses index zero
            sim.head_ptr[e, s] = head
            if count == 0:
                sim.alive[e, s] = False
                sim.seg_count[e, s] = int(torch.randint(1, 40, (1,), generator=gen))
                sim.length[e, s] = sim.seg_count[e, s]
                continue
            cells = _serpentine(s, count)
            ring = (head - torch.arange(count)) % cap
            sim.bodies[e, s, ring] = cells
            sim.seg_count[e, s] = count
            sim.length[e, s] = count + grow
            sim._reward_prev_length[e, s] = count + grow
            sim.direction[e, s] = 1 if ((count - 1) // WIDTH) % 2 == 0 else 3
            sim.boost_frames[e, s] = int(torch.randint(0, 3, (1,), generator=gen))
    sim.food_cells[:, :3] = torch.tensor(((20, 90), (40, 2), (100, 70)))
    sim.food_present[:, :3] = True
    sim.food_count[:] = 3
    sim._prepared = True
    sim.train_mode = False
    return _to(sim, device)


def _to(sim: DeviceSim, device: str) -> DeviceSim:
    if device == "cpu":
        return sim
    out = sim.clone()
    for key, value in tuple(vars(out).items()):
        if isinstance(value, torch.Tensor):
            setattr(out, key, value.to(device))
    out.device = torch.device("cuda:0")
    out._timing = type(out._timing)(out.device)
    return out


def twins(sim: DeviceSim, *, debug: bool = True) -> tuple[DeviceSim, DeviceSim]:
    """``(r2, pinned)``: identical states, only the first registered with R2."""
    pinned = sim.clone()
    r2_cache.disable(pinned)
    r2_cache.enable(sim, debug=debug)
    return sim, pinned


def _same(left, right) -> None:
    if isinstance(left, torch.Tensor):
        assert left.dtype == right.dtype and left.shape == right.shape
        assert torch.equal(left, right)
    elif hasattr(left, "__dataclass_fields__"):
        for name in left.__dataclass_fields__:
            _same(getattr(left, name), getattr(right, name))
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _same(a, b)
    else:
        assert left == right


def _q(sim: DeviceSim, rows: int, salt: int) -> torch.Tensor:
    gen = torch.Generator().manual_seed(1000 + salt)
    return torch.randn((rows, 6), generator=gen).to(sim.device)


def surface(sim: DeviceSim, runtime: DeviceVectorRuntime | None = None) -> list:
    """Every R2-affected observable output for one state, in a fixed order."""
    out = [advisory_mask(sim), _ttl_by_owner(sim)]
    envs = torch.arange(sim.E, device=sim.device)
    mask = advisory_mask(sim)
    for slot in range(sim.S):
        rows = torch.stack((envs, torch.full_like(envs, slot)), 1)
        for features in (True, False):
            out.append(v8_decide(sim, rows, _q(sim, sim.E, slot), mask[:, slot], features=features))
    # Arbitrary caller rows: duplicates, mixed slots, dead rows.
    rows = torch.zeros((5, 2), dtype=torch.long, device=sim.device)
    for index, (env, slot) in enumerate(((0, 1), (sim.E - 1, 3), (0, 1), (0, 0), (sim.E - 1, 5))):
        rows[index, 0], rows[index, 1] = env, slot
    out.append(v8_decide(sim, rows, _q(sim, len(rows), 99), mask[rows[:, 0], rows[:, 1]]))
    out.append(build_observation(sim))
    for slot in range(1, sim.S):
        out.append(choose_anchor(sim, slot, "greedy_food"))
    if runtime is not None:
        out.append(runtime._free_space_all(sim))
        out.append(runtime.features(sim, update_memory=False))
    return out


def _runtime(sim: DeviceSim) -> DeviceVectorRuntime:
    runtime = DeviceVectorRuntime()
    runtime.bind(sim)
    return runtime


# --------------------------------------------------------------------- L1 window


def test_window_rule_rounds_live_length_plus_growth_and_clamps_to_cap():
    sim = world("cpu", [[1, 2, 3, 4, 5, 6]], cap=600)
    r2_cache.enable(sim)
    for longest, expected in ((6, 256), (253, 256), (254, 256), (255, 512), (510, 512)):
        sim.seg_count[0, 2] = longest
        r2_cache.bump(sim)
        assert r2_cache.window(sim) == expected
    sim.seg_count[0, 2] = 511  # 513 > 512: window would be 768 > cap -> cap
    r2_cache.bump(sim)
    assert r2_cache.window(sim) == 600
    r2_cache.disable(sim)
    assert r2_cache.window(sim) == 600  # pinned mode always scans the full ring


def test_window_guard_rejects_a_stale_short_window():
    sim = world("cpu", [[100, 2, 3, 4, 5, 6]], cap=600)
    r2_cache.enable(sim)
    assert r2_cache.window(sim) == 256
    sim.seg_count[0, 0] = 300  # mutated without a generation bump
    with pytest.raises(RuntimeError, match="live window"):
        r2_cache.window(sim)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize(
    "counts",
    [
        # One snake exactly at the window edge (seg_count + growth == B), one
        # just past it (forces the next window), wrapped rings, dead stale rows.
        [[254, 30, 0, 255, 6, 1], [1, 0, 256, 2, 254, 9]],
        [[800, 900, 1000, 1100, 1300, 1500], [1534, 0, 1535, 3, 1536, 1]],
    ],
)
def test_r2_surface_is_byte_identical_to_pinned(device, counts):
    sim, pinned = twins(world(device, counts, grow=3))
    assert bool((sim.head_ptr < sim.seg_count - 1).any())  # some rings wrap
    _same(surface(sim, _runtime(sim)), surface(pinned, _runtime(pinned)))
    assert r2_cache.stats(sim)["debug_checks"] > 0


@pytest.mark.parametrize("device", DEVICES)
def test_small_capacity_window_equal_to_cap(device):
    # cap 64 < 256: the window clamps to cap and must equal the pinned scan.
    counts = [[60, 64, 2, 0, 33, 63], [64, 1, 0, 5, 62, 64]]
    sim, pinned = twins(world(device, counts, cap=64))
    assert r2_cache.window(sim) == 64
    _same(surface(sim), surface(pinned))


@pytest.mark.parametrize("device", DEVICES)
def test_growth_across_window_boundary_over_steps(device):
    """Live prefixes grow from 250 past 254 (window 256 -> 512); boosts burn."""
    counts = [[250, 251, 100, 0, 30, 252], [251, 249, 0, 40, 252, 248]]
    sim, pinned = twins(world(device, counts, grow=12))
    windows = []
    for tick in range(6):
        boost = torch.full((sim.E, sim.S), 4 if tick % 2 else 1, device=sim.device)
        for target in (sim, pinned):
            target.step(boost)
        windows.append(r2_cache.window(sim))
        assert inventory(sim) == inventory(pinned)
        _same(surface(sim), surface(pinned))
    assert windows[0] == 256 and windows[-1] == 512


@pytest.mark.parametrize("device", DEVICES)
def test_dead_alive_and_boost_cases_in_advisory_mask_and_post_body(device):
    counts = [[300, 0, 6, 5, 4, 0], [0, 299, 5, 0, 1, 7]]
    base = world(device, counts, grow=0)
    base.boost_frames[:, :] = int(base.cfg.boost_length_cost_frames) - 1  # burn-ready
    base.length[0, 3] = base.seg_count[0, 3] + 1  # growing, boost-eligible
    base.length[1, 4] = 1  # below min boost length
    sim, pinned = twins(base)
    _same(advisory_mask(sim), advisory_mask(pinned))
    envs = torch.arange(sim.E, device=sim.device)
    for slot in range(sim.S):
        rows = torch.stack((envs, torch.full_like(envs, slot)), 1)
        q = _q(sim, sim.E, slot)
        q[:, 3:] += 10.0  # prefer boosts so landing/release paths are exercised
        _same(
            v8_decide(sim, rows, q, advisory_mask(sim)[:, slot], features=True),
            v8_decide(pinned, rows, q, advisory_mask(pinned)[:, slot], features=True),
        )


# ------------------------------------------------------------- L2 invalidation


def _warm(sim: DeviceSim) -> None:
    advisory_mask(sim)
    r2_cache.slot_occupancy(sim)
    r2_cache.all_occupancy(sim)


MUTATORS = ("step", "prepare_respawn", "respawn", "maintain_food", "reset", "clone", "repeat")


@pytest.mark.parametrize("mutator", MUTATORS)
def test_every_mutator_invalidates_the_state_cache(mutator):
    counts = [[300, 0, 6, 40, 4, 0], [0, 299, 5, 0, 1, 7]]
    base = world("cpu", counts, grow=2)
    base.allow_respawn = True
    base.terminal_slots[:] = False
    template = base.clone()
    template.seg_count[0, 0] = 299  # a different state to restore from
    sim, pinned = twins(base, debug=False)
    _warm(sim)
    before = r2_cache.generation(sim)
    actions = torch.ones((sim.E, sim.S), dtype=torch.long)
    if mutator == "step":
        sim.step(actions)
        pinned.step(actions)
    elif mutator == "prepare_respawn":
        for target in (sim, pinned):
            target._prepared = False
            target.respawn_timer[:] = 0
            target.prepare()
        assert bool(sim.alive.all())
    elif mutator == "respawn":
        for target in (sim, pinned):
            target.respawn_timer[:] = 0
        r2_cache.bump(sim)  # the timer write above is a test-side mutation
        _warm(sim)
        before = r2_cache.generation(sim)
        sim._respawn()
        pinned._respawn()
    elif mutator == "maintain_food":
        for target in (sim, pinned):
            target.train_mode = True
            target._maintain_food()
    elif mutator == "reset":
        realistic._resident_reset(sim, template)
        realistic._resident_reset(pinned, template)
    elif mutator == "clone":
        old = sim
        sim, pinned = sim.clone(), pinned.clone()
        assert r2_cache.enabled(sim) and not r2_cache.enabled(pinned)
        assert r2_cache.generation(sim) != r2_cache.generation(old)
        sim.step(actions)
        pinned.step(actions)
    else:
        sim, pinned = sim.repeat_interleave(2), pinned.repeat_interleave(2)
        assert r2_cache.enabled(sim) and not r2_cache.enabled(pinned)
    assert r2_cache.generation(sim) != before
    _same(advisory_mask(sim), advisory_mask(pinned))
    _same(r2_cache.all_occupancy(sim), r2_cache._reference_planes(pinned).any(1))
    _same(surface(sim), surface(pinned))


def test_debug_mode_catches_a_mutation_that_skipped_its_bump():
    sim, _ = twins(world("cpu", [[300, 2, 3, 4, 5, 6]], cap=600), debug=True)
    _warm(sim)
    sim.alive[0, 0] = False  # a state write that skipped its generation bump
    with pytest.raises(r2_cache.R2CacheError):
        advisory_mask(sim)


def test_debug_mode_catches_a_consumer_mutating_a_shared_value():
    sim, _ = twins(world("cpu", [[300, 2, 3, 4, 5, 6]], cap=600), debug=True)
    mask = advisory_mask(sim)
    mask.logical_not_()
    with pytest.raises(r2_cache.R2CacheError):
        advisory_mask(sim)


def test_r2_keeps_persistent_inventory_and_shares_one_state_evaluation():
    base = world("cpu", [[300, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]], cap=600)
    keys = set(vars(base))
    reference = inventory(base.clone())
    sim, _ = twins(base, debug=False)
    surface(sim)
    assert set(vars(sim)) == keys and inventory(sim) == reference
    stats = r2_cache.stats(sim)
    # One state: advisory, occupancy, window and TTL each computed once.
    assert stats["misses"] <= 6 and stats["hits"] >= 2 * sim.S


# ------------------------------------------------------------- L3 constants


def test_r2_hot_paths_make_no_per_call_host_constant_copies():
    sim, pinned = twins(world("cpu", [[300, 2, 3, 4, 5, 6]], cap=600), debug=False)
    runtime = _runtime(sim)
    surface(sim, runtime)  # warm the device constant cache
    counts: Counter = Counter()
    real_tensor, real_as_tensor = torch.tensor, torch.as_tensor

    def counted(name, real):
        def call(*args, **kwargs):
            counts[name] += 1
            return real(*args, **kwargs)

        return call

    with (
        patch.object(torch, "tensor", counted("tensor", real_tensor)),
        patch.object(torch, "as_tensor", counted("as_tensor", real_as_tensor)),
    ):
        r2_cache.bump(sim)
        surface(sim)
        r2_counts = sum(counts.values())
        counts.clear()
        surface(pinned)
        pinned_counts = sum(counts.values())
    assert r2_counts == 0 and pinned_counts > 0
