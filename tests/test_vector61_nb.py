"""Parity: compiled vector61 featurizer (src.simd_env.vector61_nb) == the NumPy featurizer.

Worlds are M3-C training segments (``tests/test_v8_nb.build_world``: scripted, frozen and
mixed rosters, frp3-s12 opponents) driven by the real opponent policies and the Python v8
hero. Two extra :class:`Vector61Runtime` instances (``backend="python"`` and ``"numba"``)
control EVERY slot of every world and are stepped through the same ``prepare`` /
``observe_step`` sequence; their selection states, selection masks, carried states and
trend memory must be bit-identical every frame. Direct ``featurize`` calls (all rows incl.
dead ones, no memory) and the decision-time free-space counts are compared too, plus
58-D and no-boundary-danger parameter variants.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT, build_world  # noqa: E402


def _bits_equal(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row bitwise equality of two float32 arrays."""
    a = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    b = np.ascontiguousarray(b, dtype=np.float32).view(np.uint32)
    eq = a == b
    return eq.all(axis=-1) if eq.ndim > 1 else eq


def run_parity(worlds: int, frames: int, seed: int, grow_to: int = 0, direct_every: int = 3):
    from src.simd_env.vector61_featurizer import Vector61Featurizer
    from src.simd_env.vector61_nb import Vector61FeaturizerNB
    from src.simd_env.vector61_policy import Vector61Runtime

    sim, runtime, policies, groups, shield, _ = build_world(worlds, seed)
    rng = np.random.default_rng(seed)
    if grow_to:
        sim.length[:, :] = rng.integers(grow_to // 4, grow_to + 1, size=sim.length.shape)
    every = np.ones((sim.E, sim.S), dtype=bool)
    ref = Vector61Runtime(backend="python")
    nb = Vector61Runtime(backend="numba")
    ref.bind(sim, every)
    nb.bind(sim, every)
    assert isinstance(nb.featurizer, Vector61FeaturizerNB)
    assert type(ref.featurizer) is Vector61Featurizer
    variants = []
    for kw in ({"use_free_space": False}, {"use_boundary_as_danger": False}):
        params = replace(ref.featurizer.p, **kw)
        variants.append((Vector61Featurizer(params), Vector61FeaturizerNB(params)))
    stats = {"rows": 0, "mismatches": 0, "direct_rows": 0, "max_len": 0, "trend_nonzero": 0}

    def compare_runtime(tag, frame):
        sel = ref.sel_ready
        assert np.array_equal(sel, nb.sel_ready), (tag, frame)
        assert np.array_equal(ref.sel_fresh, nb.sel_fresh), (tag, frame)
        bad = ~_bits_equal(ref.sel_state, nb.sel_state)
        assert np.array_equal(ref.sel_mask, nb.sel_mask), (tag, frame)
        assert np.array_equal(ref._carry_valid, nb._carry_valid), (tag, frame)
        bad |= ~_bits_equal(ref._carry_state, nb._carry_state)
        assert np.array_equal(ref._carry_mask, nb._carry_mask), (tag, frame)
        assert np.array_equal(ref.memory.prev_id, nb.memory.prev_id), (tag, frame)
        assert np.array_equal(
            ref.memory.prev_dist.view(np.uint64), nb.memory.prev_dist.view(np.uint64)
        ), (tag, frame)
        stats["mismatches"] += int(bad.sum())
        assert not bad.any(), (tag, frame, np.argwhere(bad)[:5])

    for frame in range(frames):

        def choose(prepared, frame=frame):
            alive = prepared.get_alive()
            masks = prepared.get_resolved_action_mask()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            ref.prepare(prepared)
            nb.prepare(prepared)
            compare_runtime("prepare", frame)
            ready = np.argwhere(ref.sel_ready)
            stats["rows"] += len(ready)
            stats["trend_nonzero"] += int((ref.sel_state[ref.sel_ready][:, 49] != 0).sum())
            if frame % direct_every == 0:
                allrows = np.argwhere(np.ones((sim.E, sim.S), dtype=bool))  # dead rows too
                a = ref.featurizer.featurize(prepared, allrows, None, update_memory=False)
                b = nb.featurizer.featurize(prepared, allrows, None, update_memory=False)
                ok = _bits_equal(a, b)
                stats["mismatches"] += int((~ok).sum())
                assert ok.all(), (frame, allrows[~ok][:5])
                stats["direct_rows"] += len(allrows)
                live = np.argwhere(prepared.get_alive())
                if len(live):
                    assert np.array_equal(
                        ref.decision_free_space(prepared, live),
                        nb.decision_free_space(prepared, live),
                    ), frame
                for fp, fn in variants:
                    a = fp.featurize(prepared, allrows, None, update_memory=False)
                    b = fn.featurize(prepared, allrows, None, update_memory=False)
                    ok = _bits_equal(a, b)
                    stats["mismatches"] += int((~ok).sum())
                    assert ok.all(), (frame, fp.p, allrows[~ok][:5])
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                states, _ = runtime.selection(hero)
                q = rng.normal(size=(len(hero), 6)).astype(np.float32)
                actions[hero[:, 0], 0] = shield.act(q, masks[hero[:, 0], 0], prepared, hero)
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
        ref.observe_step(sim)
        nb.observe_step(sim)
        compare_runtime("observe", frame)
        stats["max_len"] = max(stats["max_len"], int(sim.seg_count[sim.alive].max(initial=0)))
    assert nb.featurizer.fallbacks == 0
    return stats


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_vector61_nb_matches_numpy_on_training_worlds():
    """Fresh training worlds (short bodies), every slot of every world featurized."""
    stats = run_parity(worlds=6, frames=240, seed=0)
    print("vector61_nb parity (fresh):", stats)
    assert stats["mismatches"] == 0
    assert stats["rows"] >= 6000 and stats["direct_rows"] >= 2500
    assert stats["trend_nonzero"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_vector61_nb_matches_numpy_with_long_bodies():
    """Long bodies (targets up to 1500; bodies grow a cell per frame)."""
    stats = run_parity(worlds=8, frames=700, seed=3, grow_to=1500, direct_every=5)
    print("vector61_nb parity (long bodies):", stats)
    assert stats["mismatches"] == 0
    assert stats["rows"] >= 20000 and stats["max_len"] >= 400


def test_runtime_backend_switch(monkeypatch):
    from src.simd_env.vector61_policy import V61_BACKEND_ENV, Vector61Runtime

    assert Vector61Runtime().backend == "python"
    assert Vector61Runtime(backend="numba").backend == "numba"
    monkeypatch.setenv(V61_BACKEND_ENV, "numba")
    assert Vector61Runtime().backend == "numba"
    with pytest.raises(ValueError):
        Vector61Runtime(backend="cuda")
