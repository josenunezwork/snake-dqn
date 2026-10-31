"""Unit tests for R6 Watch6-v2 L<600 safe-harvest separability census."""

from __future__ import annotations

import numpy as np

from research.r6_l600_census import R6_KILL_THRESHOLDS
from research.r6_l600_census.features import (
    extract_fifo_corpse_global_planes_and_scalars,
)
from research.r6_l600_census.probes import (
    assign_world_folds,
    run_world_disjoint_oof_probes,
)
from research.r6_l600_census.seeds import (
    audit_seeds,
    derive_world_seeds,
)
from research.t34_o1_harvest.provenance import (
    AMBIENT,
    BOOST_PELLET,
    DEATH_DROP,
    ProvenanceObserver,
)
from src.simd_env.batch_sim import BatchSimConfig
from src.simd_env.grid_sim import GridBatchSim


def test_r6_seeds_deterministic_and_disjoint() -> None:
    seeds_a = derive_world_seeds(count=64)
    seeds_b = derive_world_seeds(count=64)
    assert seeds_a == seeds_b
    assert len(set(seeds_a)) == 64
    assert all(10_000_000 <= s <= 0x7FFF_FFFF for s in seeds_a)

    audit = audit_seeds(seeds_a)
    assert audit["status"] == "CLEAR"
    assert audit["num_seeds"] == 64
    assert audit["excluded_total"] >= 25_000
    assert audit["collisions"] == 0


def test_r6_corpse_fifo_features_rotation_and_fifo_decay() -> None:
    cfg = BatchSimConfig(
        num_envs=1,
        num_snakes=6,
        mechanics_version=2,
        max_capacity=256,
        body_storage_capacity=256,
    )
    peak_drop_coords: list[tuple[int, int]] = []
    peak_trail_coords: list[tuple[int, int]] = []

    cardinal = ((0, -1), (1, 0), (0, 1), (-1, 0))

    for direction in range(4):
        sim = GridBatchSim(cfg)
        sim.reset()
        fx, fy = cardinal[direction]
        rx, ry = -fy, fx

        # Center hero in an 8x8 block center so rotation maps coarse blocks symmetrically.
        hx, hy = 68, 44
        sim.alive[0, :] = False
        sim.alive[0, 0] = True
        sim.direction[0, 0] = direction
        sim.length[0, 0] = 20
        sim.seg_count[0, 0] = 3
        sim.head_ptr[0, 0] = 0
        sim.bodies[0, 0, 0] = (hx, hy)
        sim.bodies[0, 0, 1] = (hx - fx, hy - fy)
        sim.bodies[0, 0, 2] = (hx - 2 * fx, hy - 2 * fy)

        # Place items in distinct 8x8 blocks relative to hero:
        # 1) Old DEATH_DROP (earlier in FIFO queue -> smaller remaining FIFO slots)
        # 2) Fresh DEATH_DROP (later in FIFO queue -> larger remaining FIFO slots)
        # 3) Fresh BOOST_PELLET
        # 4) AMBIENT food (not in corpse_cells -> 0 in both planes)
        dd_old = (hx + 24 * fx, hy + 24 * fy)
        dd_fresh = (hx + 16 * fx + 8 * rx, hy + 16 * fy + 8 * ry)
        bp_fresh = (hx + 16 * fx - 16 * rx, hy + 16 * fy - 16 * ry)
        amb = (hx + 8 * fx, hy + 8 * fy)

        prov = ProvenanceObserver(sim)
        sim.food_cells[0] = [amb, dd_old, bp_fresh, dd_fresh]
        sim.corpse_cells[0] = {dd_old, bp_fresh, dd_fresh}
        prov._sources = {
            (0, amb): AMBIENT,
            (0, dd_old): DEATH_DROP,
            (0, bp_fresh): BOOST_PELLET,
            (0, dd_fresh): DEATH_DROP,
        }

        planes, pooled, scalars4, aux4 = extract_fifo_corpse_global_planes_and_scalars(
            sim, prov, env=0
        )
        assert planes.shape == (2, 37, 37)
        assert pooled.shape == (28,)
        assert scalars4.shape == (4,)
        assert aux4.shape == (4,)
        assert np.all(np.isfinite(planes))
        assert np.all(np.isfinite(pooled))
        assert np.all(np.isfinite(scalars4))

        nonzero_drop = planes[0][planes[0] > 0.0]
        assert nonzero_drop.size == 2
        # Fresh FIFO entry (later in food_cells) strictly outweighs older FIFO entry.
        assert float(nonzero_drop.max()) > float(nonzero_drop.min())

        # Trail plane responds strictly to BOOST_PELLET (1 block), ambient contributes 0.
        nonzero_trail = planes[1][planes[1] > 0.0]
        assert nonzero_trail.size == 1

        dd_peak = tuple(int(x) for x in np.unravel_index(np.argmax(planes[0]), (37, 37)))
        bp_peak = tuple(int(x) for x in np.unravel_index(np.argmax(planes[1]), (37, 37)))
        peak_drop_coords.append(dd_peak)
        peak_trail_coords.append(bp_peak)

    assert len(set(peak_drop_coords)) == 1
    assert len(set(peak_trail_coords)) == 1


def test_r6_world_disjoint_oof_probes_and_kill_thresholds() -> None:
    assert R6_KILL_THRESHOLDS["min_qualifying_delta_A500"] == 5.0
    assert R6_KILL_THRESHOLDS["min_oof_qualifying_root_recall"] == 0.15
    assert R6_KILL_THRESHOLDS["min_mean_selected_delta_A500"] == 8.0
    assert R6_KILL_THRESHOLDS["max_H500_lethal_flip_rate"] == 0.005

    rng = np.random.default_rng(20261031)
    n_worlds = 20
    roots_per_world = 6
    world_ids = [w for w in range(n_worlds) for _ in range(roots_per_world)]
    folds = assign_world_folds(world_ids, n_folds=5)
    for w in range(n_worlds):
        w_folds = {int(folds[i]) for i, wid in enumerate(world_ids) if wid == w}
        assert len(w_folds) == 1

    def make_roots(*, informative_corpse: bool) -> list[dict]:
        roots: list[dict] = []
        for idx, wid in enumerate(world_ids):
            mod = idx % 5
            is_safe_harvest = mod in (0, 1)
            is_lethal_trap = mod in (2, 3)

            pooled = rng.normal(0.0, 0.02, size=28).astype(np.float32)
            s4 = rng.normal(0.0, 0.02, size=4).astype(np.float32)
            planes = rng.normal(0.0, 0.01, size=(2, 37, 37)).astype(np.float32)

            if informative_corpse:
                if is_safe_harvest:
                    pooled[12] = 5.0  # action 1 cone drop signal
                    s4[0] = 0.8
                    s4[1] = 4.0
                    s4[3] = 2.0
                elif is_lethal_trap:
                    pooled[12] = -5.0
                    s4[0] = 0.8
                    s4[1] = -4.0
                    s4[3] = -2.0

            branches = {
                "0": {"survives": True, "delta_A500": 0.0, "death_delta_A500": 0.0},
                "1": {
                    "survives": not is_lethal_trap,
                    "delta_A500": 20.0 if is_safe_harvest else (-35.0 if is_lethal_trap else 0.0),
                    "death_delta_A500": 21.0 if is_safe_harvest else 0.0,
                },
                "2": {"survives": True, "delta_A500": 0.0, "death_delta_A500": 0.0},
            }
            roots.append(
                {
                    "world_idx": wid,
                    "world_seed": 20_000_000 + wid,
                    "bare_action": 0,
                    "hero_length": 180,
                    "phi": rng.normal(0.0, 0.1, size=256).astype(np.float32),
                    "q_b8": rng.normal(0.0, 0.1, size=6).astype(np.float32),
                    "scalars_bv8": rng.normal(0.0, 0.1, size=60).astype(np.float32),
                    "corpse_trail_planes": planes,
                    "corpse_trail_pooled": pooled,
                    "corpse_fifo_scalars": s4,
                    "corpse_fifo_aux": np.zeros(4, dtype=np.float32),
                    "branches": branches,
                }
            )
        return roots

    rep_pass = run_world_disjoint_oof_probes(make_roots(informative_corpse=True), n_folds=5)
    assert rep_pass["world_disjoint_verified"] is True
    assert rep_pass["verdict"] == "PASS"

    rep_kill = run_world_disjoint_oof_probes(make_roots(informative_corpse=False), n_folds=5)
    assert rep_kill["verdict"] == "KILL"
    assert rep_kill["verdict_all_root_denom_sensitivity"] == "KILL"
