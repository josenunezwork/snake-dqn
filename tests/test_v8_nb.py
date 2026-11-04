"""Parity: compiled v8 veto (src.simd_env.v8_nb) == the Python v8 path, row for row.

The reference is the training-time shield (``train_m3c.make_shield``: the live
``SpaceAndHeadVeto`` at lambda 8 run through ``Vector61SimdPolicy`` on ``SimRowSnake``
views). Worlds are M3-C training segments (``build_segment``) with scripted, frozen and
mixed rosters; the hero plays the Python v8 action. Q per frame cycles through random
normal Q, coarse integer Q (many exact float ties), the frp3-s12 teacher's own Q and
random normal Q under a random hard mask.
Every hero decision must match (``action``; also the base argmax and the v2 counts), and
on feature frames every per-action diagnostic must equal the live helper's value.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest

CKPT = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
HAVE_CKPT = (CKPT / "frp3_m3_s12_u60000_20261005.pth").exists()


def build_world(worlds: int, seed: int, segment: int = 0):
    """An M3-C training segment (1/3 scripted, 1/3 frozen, 1/3 mixed) plus the shield."""
    from research.redesign_m2_20261008 import ni_spec
    from research.redesign_m3c_20261008 import train_m3c as T
    from research.redesign_scope_20261007 import grid_h5000_identity as gid
    from src.simd_env.vector61_policy import Vector61SimdPolicy

    ctx = gid._context(1)
    args = T.default_args()
    args.worlds = worlds
    args.seed = seed
    args.frozen_frac = 1.0 / 3.0
    args.mixed_frac = 1.0 / 3.0
    sim, runtime, policies, groups = T.build_segment(args, ctx, segment, 0.99, hero_v61=True)
    carrier = ctx["lookup"][ni_spec.FRP3_S12[1]][1]
    shield = T.make_shield(carrier, runtime, backend="python")
    teacher = Vector61SimdPolicy(carrier, runtime, forward="batched")
    return sim, runtime, policies, groups, shield, teacher


def numba_shield(runtime):
    """The trainer's opt-in compiled shield (``make_shield(..., backend="numba")``)."""
    from research.redesign_m2_20261008 import ni_spec
    from research.redesign_m3c_20261008 import train_m3c as T
    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    carrier = str(gid.CHECKPOINT_DIR / ni_spec.FRP3_S12[0])
    return T.make_shield(carrier, runtime, backend="numba")


def reference_features(sim, rows, counts, caps) -> Dict[str, np.ndarray]:
    """Per-action diagnostics from the live helpers on SimRowSnake views."""
    from src.evaluation.safety_veto_v3 import grid_for, static_blocked
    from src.evaluation.safety_veto_v5 import landing_count
    from src.evaluation.safety_veto_v6 import head_risky_actions, opponent_head_reach
    from src.evaluation.safety_veto_v7 import area_cap, landing_area
    from src.simd_env.vector61_policy import sim_row_roster

    grid_shape = (int(sim.grid_w), int(sim.grid_h))
    n = len(rows)
    out = {
        "area": np.zeros((n, 6), dtype=np.int64),
        "area_cap": np.zeros(n, dtype=np.int64),
        "landing": np.zeros((n, 3), dtype=np.int64),
        "risky": np.zeros((n, 6), dtype=bool),
        "waived": np.zeros((n, 6), dtype=bool),
    }
    for i, (env, slot) in enumerate(rows.tolist()):
        hero, roster = sim_row_roster(sim, env, slot, grid_shape, counts[i], int(caps[i]))
        grid = grid_for(hero)
        blocked = static_blocked(hero, roster, grid)
        acap = area_cap(hero.length, grid, blocked)
        out["area_cap"][i] = acap
        for a in range(6):
            out["area"][i, a] = landing_area(hero, a, blocked, grid, acap)
        for d in range(3):
            c = landing_count(hero, roster, d, int(caps[i]))
            out["landing"][i, d] = -2 if c is None else c
        risky, waived = head_risky_actions(hero, opponent_head_reach(hero, roster, grid), grid)
        out["risky"][i] = risky
        out["waived"][i] = waived
    return out


def run_parity(
    worlds: int,
    frames: int,
    seed: int = 0,
    feature_every: int = 4,
    teacher_only: bool = False,
    grow_to: int = 0,
) -> Dict[str, Any]:
    """Drive the worlds for ``frames`` frames; return parity statistics (asserts inline)."""
    import torch

    from src.simd_env.v8_nb import v8_decide

    sim, runtime, policies, groups, shield, teacher = build_world(worlds, seed)
    shield_nb = numba_shield(runtime)
    rng = np.random.default_rng(seed)
    if grow_to:
        # Long bodies (the census lengths) without thousands of warm-up frames: raise every
        # snake's target length; the sim grows the bodies one cell per frame from here.
        sim.length[:, :] = rng.integers(grow_to // 4, grow_to + 1, size=sim.length.shape)
    stats = {
        "rows": 0,
        "mismatches": 0,
        "overrides": 0,
        "overrides_teacher_q": 0,
        "rows_teacher_q": 0,
        "v5_changes": 0,
        "v7_changes": 0,
        "head_changes": 0,
        "no_spacious": 0,
        "feature_rows": 0,
        "feature_mismatches": 0,
        "tie_rows": 0,
        "v5_reasons": np.zeros(5, dtype=np.int64),
        "random_mask_rows": 0,
    }
    for frame in range(frames):

        def choose(prepared, frame=frame):
            alive = prepared.get_alive()
            masks = prepared.get_resolved_action_mask()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                kind = 2 if teacher_only else frame % 4
                if kind == 3:
                    # Random hard masks (both paths read the runtime's selection mask):
                    # reaches v5's landing-veto branches the advisory mask rarely allows.
                    rnd = rng.random((len(hero), 6)) < 0.5
                    runtime.sel_mask[hero[:, 0], hero[:, 1]] = rnd
                states, sel_mask = runtime.selection(hero)
                if kind in (0, 3):
                    q = rng.normal(size=(len(hero), 6)).astype(np.float32)
                elif kind == 1:
                    q = rng.integers(0, 3, size=(len(hero), 6)).astype(np.float32)
                else:
                    with torch.no_grad():
                        q = teacher.q_values(states).float().cpu().numpy()
                a_py = shield.act(q, masks[hero[:, 0], 0], prepared, hero)
                feats = frame % feature_every == 0
                res = v8_decide(prepared, hero, q, sel_mask, 8.0, features=feats)
                lv = shield.last_veto
                if kind != 3:  # the trainer's switch reads the sim's (real) advisory mask
                    a_nb = shield_nb.act(q, masks[hero[:, 0], 0], prepared, hero)
                    assert np.array_equal(a_nb, a_py), frame
                assert np.array_equal(res.base, lv["base"]), frame
                assert np.array_equal(res.counts, lv["counts"]), frame
                assert np.array_equal(res.need, lv["need"]), frame
                bad = res.action != a_py
                stats["mismatches"] += int(bad.sum())
                assert not bad.any(), (frame, hero[bad], res.action[bad], a_py[bad])
                stats["rows"] += len(hero)
                over = res.action != res.base
                stats["overrides"] += int(over.sum())
                if kind == 2:
                    stats["rows_teacher_q"] += len(hero)
                    stats["overrides_teacher_q"] += int(over.sum())
                if kind == 1:
                    stats["tie_rows"] += len(hero)
                if kind == 3:
                    stats["random_mask_rows"] += len(hero)
                stats["v5_changes"] += int((res.v5_action != res.base).sum())
                stats["v7_changes"] += int((res.v7_action != res.v5_action).sum())
                stats["head_changes"] += int((res.action != res.v7_action).sum())
                stats["no_spacious"] += int((res.outcome == 2).sum())
                stats["v5_reasons"] += np.bincount(res.v5_reason, minlength=5)
                if feats:
                    ref = reference_features(prepared, hero, lv["counts"], res.cap)
                    for key, got in (
                        ("area", res.area),
                        ("area_cap", res.area_cap),
                        ("landing", res.landing),
                        ("risky", res.head_risky),
                        ("waived", res.head_waived),
                    ):
                        diff = np.asarray(got) != ref[key]
                        while diff.ndim > 1:
                            diff = diff.any(axis=-1)
                        stats["feature_mismatches"] += int(diff.sum())
                        assert not diff.any(), (frame, key, hero[diff])
                    stats["feature_rows"] += len(hero)
                actions[hero[:, 0], 0] = a_py
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    return stats


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_v8_nb_matches_python_v8_on_training_worlds():
    """Fresh M3-C training worlds (short heroes): mostly v5 vetoes."""
    stats = run_parity(worlds=6, frames=360, seed=0, feature_every=4)
    print("v8_nb parity (fresh worlds):", stats)
    assert stats["mismatches"] == 0 and stats["feature_mismatches"] == 0
    assert stats["rows"] >= 2000 and stats["feature_rows"] >= 500
    assert stats["overrides"] > 0 and stats["v5_changes"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_v8_nb_matches_python_v8_with_long_bodies():
    """Long bodies (up to 1500): v7 re-ranks, head vetoes and no_spacious decisions occur."""
    stats = run_parity(worlds=8, frames=900, seed=3, feature_every=5, grow_to=1500)
    print("v8_nb parity (long bodies):", stats)
    assert stats["mismatches"] == 0 and stats["feature_mismatches"] == 0
    assert stats["rows"] >= 7000 and stats["feature_rows"] >= 1400
    assert stats["v7_changes"] > 0 and stats["head_changes"] > 0 and stats["no_spacious"] > 0
    assert np.all(stats["v5_reasons"] > 0)  # every v5 branch, incl. the landing vetoes


def test_v8_nb_rejects_bad_lambda():
    from src.simd_env.v8_nb import v8_decide

    with pytest.raises(ValueError):
        v8_decide(object(), np.zeros((0, 2)), np.zeros((0, 6)), np.zeros((0, 6)), lam=-1.0)


def test_make_shield_backend_switch(monkeypatch):
    from research.redesign_m3c_20261008 import train_m3c as T

    with pytest.raises(ValueError):
        T.make_shield("unused", None, backend="cuda")
    monkeypatch.setenv(T.V8_BACKEND_ENV, "fortran")
    with pytest.raises(ValueError):
        T.make_shield("unused", None)


def _assert_v8_result_identical(res_ref, res_fast) -> int:
    mismatches = 0
    for field in res_ref.__dataclass_fields__:
        a = np.asarray(getattr(res_ref, field))
        b = np.asarray(getattr(res_fast, field))
        diff = int(np.sum(a != b))
        mismatches += diff
        assert diff == 0, (field, a, b)
    return mismatches


def test_v6_fast_kernel_default_disabled_and_scope():
    from src.simd_env.v8_nb import (
        set_v6_fast_kernel,
        v6_fast_kernel_enabled,
        v6_fast_kernel_scope,
    )

    prev = set_v6_fast_kernel(False)
    try:
        assert not v6_fast_kernel_enabled()
        with v6_fast_kernel_scope(True):
            assert v6_fast_kernel_enabled()
        assert not v6_fast_kernel_enabled()
    finally:
        set_v6_fast_kernel(prev)


def test_v6_fast_kernel_bitwise_identity_serpentines_and_rollout():
    """Bitwise identity of v6_fast_kernel vs default across L=150..1500 coils and rollouts."""
    from research.bl_ppo.collector import _prepare_sim
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.ego_raster_bv8 import v8_scalars
    from src.simd_env.ego_raster_nb import region_sizes_numba
    from src.simd_env.grid_scenarios import inject_serpentines
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.v8_nb import (
        V8Result,
        _get_result_buffer,
        v6_fast_kernel_scope,
        v8_decide,
        v8_feature_planes,
    )
    from src.simd_env.vector61_policy import Vector61Runtime

    rng = np.random.default_rng(34910101)
    mismatches = 0
    for hero_len in (150, 450, 900, 1200, 1500):
        sim = GridBatchSim(
            BatchSimConfig(
                num_envs=1,
                num_snakes=6,
                mechanics_version=2,
                max_capacity=2048,
                body_storage_capacity=2048,
            ),
            seeds=[34910101 + hero_len],
        )
        inject_serpentines([sim], (hero_len, 900, 700, 500, 400, 300), box_height=11)
        _prepare_sim(sim)
        rt = Vector61Runtime(backend="numba")
        rt.bind(sim, np.ones((sim.E, sim.S), dtype=bool))
        rt.prepare(sim)
        owner_before = sim._owner_pad.copy()

        rows_all = np.argwhere(sim.alive)
        with v6_fast_kernel_scope(False):
            dfs_ref = rt.decision_free_space(sim, rows_all).copy()
            reg_ref = region_sizes_numba(sim.ego_view(), rows_all).copy()
            vs_ref, vres_ref = v8_scalars(sim, rows_all, reuse_buffer=False)
            vs_ref = vs_ref.copy()
            vres_ref_copy = V8Result(
                **{k: getattr(vres_ref, k).copy() for k in V8Result.__dataclass_fields__}
            )
            ego_ref = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows_all[:1]
            )
            ego_ref = {k: v.copy() for k, v in ego_ref.items()}

        with v6_fast_kernel_scope(True):
            dfs_fast = rt.decision_free_space(sim, rows_all).copy()
            reg_fast = region_sizes_numba(sim.ego_view(), rows_all).copy()
            vs_fast, vres_fast = v8_scalars(sim, rows_all, reuse_buffer=True)
            vs_fast = vs_fast.copy()
            vres_fast_copy = V8Result(
                **{k: getattr(vres_fast, k).copy() for k in V8Result.__dataclass_fields__}
            )
            ego_fast = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows_all[:1]
            )

        assert np.array_equal(sim._owner_pad, owner_before)
        assert np.array_equal(dfs_ref, dfs_fast)
        assert np.array_equal(reg_ref, reg_fast)
        assert np.array_equal(vs_ref, vs_fast)
        mismatches += _assert_v8_result_identical(vres_ref_copy, vres_fast_copy)
        for k in ego_ref:
            assert np.array_equal(ego_ref[k], ego_fast[k]), k

        _, sel_mask = rt.selection(rows_all)
        pre_counts = np.asarray(dfs_fast, dtype=np.int64)
        for trial in range(4):
            if trial == 0:
                q = rng.normal(size=(len(rows_all), 6)).astype(np.float32)
                mk = sel_mask
            elif trial == 1:
                q = rng.integers(0, 3, size=(len(rows_all), 6)).astype(np.float32)
                mk = sel_mask
            else:
                q = rng.normal(size=(len(rows_all), 6)).astype(np.float32)
                mk = rng.random((len(rows_all), 6)) < 0.5
            for feats in (False, True):
                with v6_fast_kernel_scope(False):
                    r0 = v8_decide(sim, rows_all, q, mk, 8.0, features=feats)
                    r0 = V8Result(
                        **{k: getattr(r0, k).copy() for k in V8Result.__dataclass_fields__}
                    )
                with v6_fast_kernel_scope(True):
                    r1 = v8_decide(sim, rows_all, q, mk, 8.0, features=feats)
                    r1 = V8Result(
                        **{k: getattr(r1, k).copy() for k in V8Result.__dataclass_fields__}
                    )
                    r2 = v8_decide(
                        sim,
                        rows_all,
                        q,
                        mk,
                        8.0,
                        features=feats,
                        precomputed_counts=pre_counts,
                        out_buffer=_get_result_buffer(len(rows_all)),
                    )
                    r2 = V8Result(
                        **{k: getattr(r2, k).copy() for k in V8Result.__dataclass_fields__}
                    )
                mismatches += _assert_v8_result_identical(r0, r1)
                mismatches += _assert_v8_result_identical(r0, r2)
                if feats:
                    assert np.array_equal(v8_feature_planes(r0), v8_feature_planes(r2))
    assert mismatches == 0


_R7_SNAP_PATH = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
    "t34-sol/argon-r7-repack/r7-preflight/mac-timing-l800-snapshots.pt"
)


@pytest.mark.skipif(
    not _R7_SNAP_PATH.exists(),
    reason="needs cached R7 L>=800 root snapshots",
)
def test_v6_fast_kernel_cached_32_roots_bitwise_identity():
    """Bitwise identity across the 16 L>=800 R7 roots and 16 regression roots from ROADMAP §3.2."""
    import json

    import torch

    from research.bl_ppo.collector import _prepare_sim
    from src.simd_env.batch_sim import BatchSimConfig
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.ego_raster_nb import region_sizes_numba
    from src.simd_env.grid_scenarios import inject_serpentines
    from src.simd_env.grid_sim import GridBatchSim
    from src.simd_env.v8_nb import (
        V8Result,
        _get_result_buffer,
        v6_fast_kernel_scope,
        v8_decide,
    )
    from src.simd_env.vector61_policy import Vector61Runtime

    def _sim_from_r7_snap(sdict):
        bcap = max(2048, int(sdict["body_cap"]))
        cfg = BatchSimConfig(
            num_envs=1,
            num_snakes=int(sdict["num_snakes"]),
            mechanics_version=2,
            max_capacity=bcap,
            body_storage_capacity=bcap,
        )
        sim = GridBatchSim(cfg, seeds=[1], train_mode=False, allow_respawn=True)
        st = sdict["small_tensors"]
        boff = np.asarray(sdict["body_offsets"], dtype=np.int64)
        bxy = np.asarray(sdict["body_segments_xy"], dtype=np.int64)
        for s in range(sim.S):
            hp = int(st["head_ptr"][s])
            n = int(st["seg_count"][s])
            sim.head_ptr[0, s] = hp
            sim.seg_count[0, s] = n
            sim.length[0, s] = int(st["length"][s])
            sim.alive[0, s] = bool(st["alive"][s])
            sim.direction[0, s] = int(st["direction"][s])
            sim.boost_frames[0, s] = int(st["boost_frames"][s])
            sim.frames_since_food[0, s] = int(st["frames_since_food"][s])
            sim._boosted_this_step[0, s] = bool(st["_boosted_this_step"][s])
            if n > 0:
                rings = (hp - np.arange(n)) % sim.cap
                sim.bodies[0, s, rings] = bxy[boff[s] : boff[s + 1]]
        fxy = np.asarray(sdict["food_xy"], dtype=np.int64)
        fcorp = np.asarray(sdict["food_corpse"], dtype=np.uint8)
        sim.food_cells[0] = [(int(xy[0]), int(xy[1])) for xy in fxy]
        sim.corpse_cells[0] = {(int(xy[0]), int(xy[1])) for xy, c in zip(fxy, fcorp) if c}
        sim._sync_food_array(0)
        sim._rebuild_env_grids(0, food=True)
        sim._last_mask = sim._compute_action_masks()
        return sim

    raw_snaps = torch.load(_R7_SNAP_PATH, map_location="cpu", weights_only=False)["snapshots"]
    snap_list = list(raw_snaps)[:16]
    rng = np.random.default_rng(20261103)
    mismatches = 0

    for item in snap_list:
        sim = _sim_from_r7_snap(item)
        _prepare_sim(sim)
        rt = Vector61Runtime(backend="numba")
        rt.bind(sim, np.ones((sim.E, sim.S), dtype=bool))
        rt.prepare(sim)
        rows = np.argwhere(sim.alive)
        _, sel_mask = rt.selection(rows)
        with v6_fast_kernel_scope(False):
            dfs0 = rt.decision_free_space(sim, rows).copy()
            reg0 = region_sizes_numba(sim.ego_view(), rows).copy()
            ego0 = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows[:1]
            )
            ego0 = {k: v.copy() for k, v in ego0.items()}
        with v6_fast_kernel_scope(True):
            dfs1 = rt.decision_free_space(sim, rows).copy()
            reg1 = region_sizes_numba(sim.ego_view(), rows).copy()
            ego1 = build_ego_raster(
                sim, EgoRasterConfig(version="bv8"), backend="numba", rows=rows[:1]
            )
        assert np.array_equal(dfs0, dfs1)
        assert np.array_equal(reg0, reg1)
        for k in ego0:
            assert np.array_equal(ego0[k], ego1[k]), k
        pre = np.asarray(dfs1, dtype=np.int64)
        q = rng.normal(size=(len(rows), 6)).astype(np.float32)
        for feats in (False, True):
            with v6_fast_kernel_scope(False):
                r0 = v8_decide(sim, rows, q, sel_mask, 8.0, features=feats)
                r0 = V8Result(**{k: getattr(r0, k).copy() for k in V8Result.__dataclass_fields__})
            with v6_fast_kernel_scope(True):
                r1 = v8_decide(
                    sim,
                    rows,
                    q,
                    sel_mask,
                    8.0,
                    features=feats,
                    precomputed_counts=pre,
                    out_buffer=_get_result_buffer(len(rows)),
                )
                r1 = V8Result(**{k: getattr(r1, k).copy() for k in V8Result.__dataclass_fields__})
            mismatches += _assert_v8_result_identical(r0, r1)

    analysis_path = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-alllen-a100/analysis.json"
    )
    if analysis_path.exists():
        per_session = json.loads(analysis_path.read_text())["per_session"]
        worst16 = sorted(
            per_session,
            key=lambda r: float(r["exact_stop_alllen_mass"]) - float(r["l2_prime_600_mass"]),
        )[:16]
        for row_info in worst16:
            w_seed = int(row_info["seed"])
            sim = GridBatchSim(
                BatchSimConfig(
                    num_envs=1,
                    num_snakes=6,
                    mechanics_version=2,
                    max_capacity=2048,
                    body_storage_capacity=2048,
                ),
                seeds=[w_seed],
            )
            inject_serpentines([sim], (900, 700, 500, 400, 300, 200), box_height=11)
            _prepare_sim(sim)
            rt = Vector61Runtime(backend="numba")
            rt.bind(sim, np.ones((sim.E, sim.S), dtype=bool))
            rt.prepare(sim)
            rows = np.argwhere(sim.alive)
            _, sel_mask = rt.selection(rows)
            with v6_fast_kernel_scope(False):
                dfs0 = rt.decision_free_space(sim, rows).copy()
                reg0 = region_sizes_numba(sim.ego_view(), rows).copy()
            with v6_fast_kernel_scope(True):
                dfs1 = rt.decision_free_space(sim, rows).copy()
                reg1 = region_sizes_numba(sim.ego_view(), rows).copy()
            assert np.array_equal(dfs0, dfs1)
            assert np.array_equal(reg0, reg1)
            pre = np.asarray(dfs1, dtype=np.int64)
            q = rng.normal(size=(len(rows), 6)).astype(np.float32)
            for feats in (False, True):
                with v6_fast_kernel_scope(False):
                    r0 = v8_decide(sim, rows, q, sel_mask, 8.0, features=feats)
                    r0 = V8Result(
                        **{k: getattr(r0, k).copy() for k in V8Result.__dataclass_fields__}
                    )
                with v6_fast_kernel_scope(True):
                    r1 = v8_decide(
                        sim,
                        rows,
                        q,
                        sel_mask,
                        8.0,
                        features=feats,
                        precomputed_counts=pre,
                        out_buffer=_get_result_buffer(len(rows)),
                    )
                    r1 = V8Result(
                        **{k: getattr(r1, k).copy() for k in V8Result.__dataclass_fields__}
                    )
                mismatches += _assert_v8_result_identical(r0, r1)
    assert mismatches == 0
