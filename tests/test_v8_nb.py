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
                a_nb = shield_nb.act(q, masks[hero[:, 0], 0], prepared, hero)
                assert np.array_equal(a_nb, a_py), frame  # the trainer's opt-in switch
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
