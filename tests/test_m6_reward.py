"""M6 reward levers of ``research/redesign_m3c_20261008/train_m3c.py`` (2026-10-14).

* Off == old, bitwise: with every lever off (the defaults, or ``--tight-cost 0
  --death-mass-mult 0`` passed explicitly) ``actor_main`` emits exactly the frame messages
  and ring rows of the pre-M6 actor, replayed from git (``BASE_REV``), on the B8 recipe
  (bv8, mass reward, on-policy labels) and on the plain M3 actor, with bodies grown to
  L 200..1399 so deaths and large-L decisions occur (a few seconds: not marked slow).
* On: only the rewards change, by exactly the lever terms, recomputed here independently
  from the recorded v8 result, decision-time length and death flags; actions, n-step
  windows, discounts, labels and ring rows are unchanged; the message counters match.
* Unit: the tight predicate (the census's length-aware ``best safe area < L``, incl. the
  boundary ``area == L`` and "no safe action"), the reward arithmetic (incl. ``L == Lmin``
  and deaths), the log fields, CLI refusals and ``m3c_flags``.
* Real states: the predicate equals the uncapped ``area < L`` from the live v7 flood, and
  the bv8 obs-scalar form ``max safe area_g < 0.5`` wherever ``area_cap == 2L``.
"""

from __future__ import annotations

import importlib.util
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
BASE_REV = "836646c"  # perf-integrate before M6: the reference ("old") trainer
TRAINER = "research/redesign_m3c_20261008/train_m3c.py"

B8_ACTOR = (
    "--reward",
    "mass",
    "--kill-scale",
    "0",
    "--gamma-end",
    "0.998",
    "--anchor-mode",
    "v8margin",
    "--onpolicy-labels",
    "--obs-version",
    "bv8",
    "--label-frac",
    "1.0",
)


def _argv(*extra):
    return ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x", *extra]


# ----------------------------------------------------------------------------- predicate
def test_tight_predicate_census_definition():
    adv = np.array(
        [
            [1, 1, 0, 0, 0, 0],  # best safe 99 < 100: tight
            [1, 0, 1, 0, 0, 0],  # best safe 100 == L: not tight (boundary)
            [0, 0, 0, 0, 0, 0],  # no safe action: best 0 -> tight
            [0, 1, 0, 0, 0, 0],  # the unsafe 500 does not count: tight
            [0, 0, 0, 0, 0, 1],  # a safe boost action counts: not tight
        ],
        dtype=bool,
    )
    area = np.array(
        [
            [99, 50, 900, 900, 900, 900],
            [100, 900, 7, 0, 0, 0],
            [900, 900, 900, 900, 900, 900],
            [500, 3, -1, -1, -1, -1],  # unevaluated (-1) areas are fine where unsafe
            [0, 0, 0, 0, 0, 150],
        ]
    )
    L = np.array([100, 100, 1, 100, 150])
    assert T.tight_predicate(area, adv, L).tolist() == [True, False, True, True, False]
    # census.py, verbatim: best = max(where(adv, area, 0)); enclosed-like test best < L
    best = np.where(adv, area, 0).max(axis=1)
    assert np.array_equal(T.tight_predicate(area, adv, L), best < L)
    assert T.tight_predicate(np.zeros((0, 6)), np.zeros((0, 6), bool), np.zeros(0)).shape == (0,)
    with pytest.raises(ValueError, match="not evaluated"):
        T.tight_predicate(np.full((1, 6), -1), np.ones((1, 6), bool), [5])
    with pytest.raises(ValueError):
        T.tight_predicate(np.zeros((2, 3)), np.zeros((2, 3), bool), [1, 1])


# ----------------------------------------------------------------------------- arithmetic
def test_tight_cost_term_boundary_and_value():
    g, c, lmin = 0.998, 2.0, 600
    L = np.array([599, 600, 601, 1000, 1000])
    tight = np.array([True, True, True, True, False])
    out = T.tight_cost_term(L, tight, g, c, lmin)
    assert out.dtype == np.float64
    assert out[0] == 0.0  # L < Lmin
    assert out[1] == -c * (1.0 - g) * (600 / 10.0)  # L == Lmin is charged
    assert out[2] == -c * (1.0 - g) * (601 / 10.0)
    assert out[3] == pytest.approx(-0.4)  # the design note: c = 2, L = 1000 -> 0.4
    assert out[4] == 0.0  # not tight
    assert np.all(T.tight_cost_term(L, tight, g, 0.0, lmin) <= 0.0)


def test_death_mass_term():
    out = T.death_mass_term(np.array([974, 974, 10]), np.array([True, False, True]), 2.0)
    assert out.dtype == np.float64
    assert out.tolist() == [-2.0 * 97.4, 0.0, -2.0 * 1.0]


def test_levers_frame_combines_terms_and_counts():
    assert T.M6Levers.from_args(T.default_args()) is None
    lv = T.M6Levers(tight_cost=2.0, tight_min_len=600, death_mass_mult=3.0)
    L = np.array([599, 600, 1000, 1200, 50])
    tight = np.array([True, True, False, True, True])
    died = np.array([False, False, False, True, True])
    delta, st = lv.frame(L, tight, died, 0.998)
    want = T.tight_cost_term(L, tight, 0.998, 2.0, 600) + T.death_mass_term(L, died, 3.0)
    assert np.array_equal(delta, want)
    # the fatal, tight row at 1200 pays both: the tight cost of its decision and the death
    assert delta[3] == pytest.approx(-2.0 * 0.002 * 120.0 - 3.0 * 120.0)
    assert delta[4] == -3.0 * 5.0  # tight but below Lmin: the death term only
    assert st == {
        "m6_death_len": 1250,
        "m6_big": 3,
        "m6_tight": 2,
        "m6_big1k": 2,
        "m6_tight1k": 1,
    }
    # telemetry only: no reward change, same counters
    tel = T.M6Levers(tight_telemetry=True)
    d2, st2 = tel.frame(L, tight, died, 0.998)
    assert d2 is None and st2 == st
    # death lever alone: no tightness needed, no tight counters
    dm = T.M6Levers(death_mass_mult=2.0)
    assert not dm.tight
    d3, st3 = dm.frame(L, None, died, 0.998)
    assert np.array_equal(d3, T.death_mass_term(L, died, 2.0)) and st3 == {"m6_death_len": 1250}
    assert lv.empty_stats() == {k: 0 for k in st}


def test_m6_log_fields():
    acc = {"m6_death_len": 3000, "m6_big": 200, "m6_tight": 10, "m6_big1k": 50, "m6_tight1k": 5}
    out = T.m6_log_fields(acc, decisions=1000, deaths=3)
    assert out == {
        "tight_frac": 0.05,
        "tight_big_share": 0.2,
        "tight_frac_1000": 0.1,
        "death_len_mean": 1000.0,
    }
    zero = {"m6_death_len": 0, "m6_big": 0, "m6_tight": 0, "m6_big1k": 0, "m6_tight1k": 0}
    empty = T.m6_log_fields(zero, 0, 0)
    assert empty["tight_frac"] is None and empty["death_len_mean"] is None
    assert empty["tight_frac_1000"] is None and empty["tight_big_share"] == 0.0
    assert T.m6_log_fields({"m6_death_len": 5}, 10, 1) == {"death_len_mean": 5.0}


# ----------------------------------------------------------------------------- CLI
@pytest.mark.parametrize(
    "extra",
    [
        ["--tight-cost", "2", "--obs-version", "bv8"],
        ["--death-mass-mult", "2"],
        ["--tight-telemetry", "--obs-version", "bv8"],
    ],
)
def test_m6_levers_refused_on_the_sync_trainer(extra):
    with pytest.raises(SystemExit, match="multi-actor"):
        T.main(_argv(*extra))


@pytest.mark.parametrize(
    "extra, match",
    [
        (["--actors", "1", "--tight-cost", "2"], "bv8"),
        (["--actors", "1", "--tight-telemetry"], "bv8"),
        (["--actors", "1", "--tight-min-len", "800"], "needs --tight-cost"),
        (["--actors", "1", "--death-mass-mult", "-1"], ">= 0"),
        (["--actors", "1", "--tight-cost", "nan"], ">= 0"),
        (["--actors", "1", "--tight-cost", "inf"], ">= 0"),
        (["--actors", "1", "--tight-telemetry", "--tight-min-len", "-1"], ">= 0"),
    ],
)
def test_m6_bad_settings_refused(extra, match):
    with pytest.raises(SystemExit, match=match):
        T.main(_argv(*extra))


def test_m6_flags_recorded_only_when_set():
    d = T.default_args()
    assert T.m3c_flags(d) == {}
    assert d.tight_cost == 0.0 and d.death_mass_mult == 0.0 and not d.tight_telemetry
    assert d.tight_min_len == T.TIGHT_MIN_LEN == 600
    p = T.build_parser()
    # explicit zeros are the defaults: nothing recorded
    assert T.m3c_flags(p.parse_args(_argv("--tight-cost", "0", "--death-mass-mult", "0"))) == {}
    f = T.m3c_flags(p.parse_args(_argv("--tight-cost", "2")))
    assert f == {"tight_cost": 2.0, "tight_min_len": 600}  # Lmin recorded with its lever
    f = T.m3c_flags(p.parse_args(_argv("--tight-cost", "2", "--tight-min-len", "800")))
    assert f == {"tight_cost": 2.0, "tight_min_len": 800}
    assert T.m3c_flags(p.parse_args(_argv("--death-mass-mult", "2"))) == {"death_mass_mult": 2.0}
    f = T.m3c_flags(p.parse_args(_argv("--tight-telemetry")))
    assert f == {"tight_telemetry": True, "tight_min_len": 600}


# ----------------------------------------------------------------------------- real states
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_tight_equals_uncapped_area_and_obs_scalar_form():
    """Training worlds with injected long serpentines (hero body 840, target L 600..2400 or
    5000), hero = random Q under the v8 veto or a random safe move. Per hero decision:

    * :func:`tight_predicate` on the actor's ``features=True`` v8 result == the uncapped
      ``max safe landing area < L`` from the live v7 flood (cap ``L``: ``min(area, L) < L``
      iff ``area < L``), except the documented ``L > 4096`` rows (always tight);
    * == the bv8 obs-scalar form ``max safe area_g < 0.5`` on every row with ``area_cap == 2L``.
    """
    from test_v8_nb import build_world

    from src.evaluation.safety_veto_v3 import grid_for, static_blocked
    from src.evaluation.safety_veto_v7 import landing_area
    from src.simd_env.ego_raster_bv8 import v8_scalars
    from src.simd_env.grid_scenarios import inject_serpentines
    from src.simd_env.v8_nb import v8_redecide
    from src.simd_env.vector61_policy import sim_row_roster

    rng = np.random.default_rng(11)
    st = {"rows": 0, "tight": 0, "flood": 0, "flood_tight": 0, "acap2L": 0, "huge": 0}
    sim, runtime, policies, groups, _, _ = build_world(8, 1)
    inject_serpentines([sim], [840, 500, 500, 500, 500, 500])
    sim.length[:, 0] = [600, 900, 1200, 1500, 1800, 2100, 2400, 5000]
    grid_shape = (int(sim.grid_w), int(sim.grid_h))

    def check(prepared, hero, sel, frame):
        feats, res = v8_scalars(prepared, hero, sel)
        L = prepared.length[hero[:, 0], 0].astype(np.int64)
        tight = T.tight_predicate(res.area, sel, L)
        st["rows"] += len(hero)
        st["tight"] += int(tight.sum())
        two = res.area_cap == 2 * L  # obs-scalar form where v7's cap is exactly 2L
        g = np.where(sel, feats[:, 0::5], 0.0).max(axis=1)
        assert np.array_equal(tight[two], g[two] < 0.5)
        st["acap2L"] += int(two.sum())
        huge = L > 4096  # area capped at 4096 < L: tight by definition
        assert tight[huge].all()
        st["huge"] += int(huge.sum())
        for i, (e, s) in enumerate(hero.tolist()):
            if huge[i] or not (tight[i] or frame % 10 == 0):
                continue
            hs, roster = sim_row_roster(prepared, e, s, grid_shape, res.counts[i], int(res.cap[i]))
            grid = grid_for(hs)
            blocked = static_blocked(hs, roster, grid)
            safe = [a for a in range(6) if sel[i, a]]
            best = max([landing_area(hs, a, blocked, grid, int(L[i])) for a in safe], default=0)
            assert tight[i] == (best < L[i]), (frame, e, int(L[i]), best)
            st["flood"] += 1
            st["flood_tight"] += int(tight[i])
        q = rng.normal(size=(len(hero), 6)).astype(np.float32)
        act = v8_redecide(res, q, sel).action
        mm = sel.copy()
        mm[~mm.any(axis=1), :3] = True
        walk = rng.random(len(hero)) < 0.05  # a few random safe moves: coils, pockets
        return np.where(walk, T.sample_legal(rng, mm), act)

    for frame in range(300):

        def choose(prepared, frame=frame):
            alive = prepared.get_alive()
            masks = prepared.get_resolved_action_mask()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                _, sel = runtime.selection(hero)
                actions[hero[:, 0], 0] = check(prepared, hero, sel, frame)
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    print("m6 tight on real states:", st)
    assert st["rows"] >= 2000 and st["acap2L"] >= 500 and st["huge"] >= 100
    assert st["flood_tight"] >= 10 and st["flood"] - st["flood_tight"] >= 100
    assert 0 < st["tight"] - st["huge"] < st["rows"] // 4


# ----------------------------------------------------------------------------- end to end
class _Stop:
    def __init__(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True


class _Collect:
    def __init__(self, frames: int, stop: _Stop) -> None:
        self.frames, self.stop, self.msgs = frames, stop, []

    def put(self, msg, timeout: float = 0.0) -> None:
        self.msgs.append(msg)
        if len(self.msgs) >= self.frames:
            self.stop.set()


def _run_actor(mod, tmp_path, init, frames, extra=()):
    """``mod.actor_main`` in-process on 6 worlds (as tests/test_perf_actor.py)."""
    argv = [
        *("--seed", "5", "--init", str(init), "--demo-data", ".", "--out", str(tmp_path / "x")),
        *("--actors", "1", "--worlds", "6", "--frozen-frac", "0.34", "--mixed-frac", "0.33"),
        *("--segment-frames", "45", "--device", "cpu", "--v61-backend", "numba"),
        *extra,
    ]
    args = mod.build_parser().parse_args(argv)
    nscal = mod.nscal_for(args)
    margin = args.worlds * 16
    ring_cap = args.worlds * (frames + 4) + margin
    net = mod.load_init(args, "cpu")
    spec = mod._weights_spec(net)
    n_weights = int(sum(n for _, _, n in spec))
    blocks = [
        mod.Block.create(mod.ring_nbytes(ring_cap, nscal), str(tmp_path), "ring0"),
        mod.Block.create(8 + 8 * (n_weights + 1), str(tmp_path), "weights"),
        mod.Block.create(32, str(tmp_path), "progress"),
    ]
    names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
    wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
    wver[0] = 0
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
    prog = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
    prog[:] = 0
    prog[0] = 9_000_000  # gamma at its end value (0.998 on the B8 recipe)
    mod._publish_weights(net, wbuf, wver)
    cfg = {
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "K": 1,
        "ring_cap": ring_cap,
        "ring_margin": margin,
        "n_weights": n_weights,
        "actor_device": "cpu",
    }
    stop = _Stop()
    sink = _Collect(frames, stop)
    mod.actor_main(0, cfg, names, sink, stop)
    rows = mod.SharedObsRing(blocks[0], ring_cap, margin, nscal)
    ring = {k: v.copy() for k, v in rows.arrays.items()}
    del wver, wbuf, prog, rows
    for b in blocks:
        b.unlink()
        b.close()
    return sink.msgs, ring


def _base_trainer(tmp_path, monkeypatch):
    """The pre-M6 ``train_m3c`` module, loaded from git at ``BASE_REV`` (registered in
    ``sys.modules`` for the test's duration: its dataclasses resolve their module there)."""
    r = subprocess.run(
        ["git", "show", f"{BASE_REV}:{TRAINER}"], cwd=REPO, capture_output=True, text=True
    )
    if r.returncode:
        pytest.skip(f"git show {BASE_REV} failed: {r.stderr.strip()[:200]}")
    assert "tight_cost" not in r.stdout  # really the pre-M6 trainer
    name = "train_m3c_pre_m6"
    path = tmp_path / "base" / "a" / "b" / f"{name}.py"
    path.parent.mkdir(parents=True)
    path.write_text(r.stdout)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, mod)
    monkeypatch.setattr(sys, "path", list(sys.path))  # its REPO insert is undone afterwards
    spec.loader.exec_module(mod)
    return mod


def _init(tmp_path):
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))  # bv8 runs widen it (Q unchanged)
    return init


def _grow_bodies(monkeypatch, mod):
    """Every segment's snakes get lengths 200..1399 (as tests/test_ego_raster_bv8.drive), the
    same per segment in every run: large-L decisions, tight states and deaths in 60 frames."""
    real_build = mod.build_segment

    def grown_segment(args, ctx, segment, gamma, hero_v61=False):
        out = real_build(args, ctx, segment, gamma, hero_v61=hero_v61)
        g = np.random.default_rng([99, segment])
        out[0].length[:, :] = g.integers(200, 1400, size=out[0].length.shape)
        return out

    monkeypatch.setattr(mod, "build_segment", grown_segment)


def _assert_msgs_equal(ref, got, ctx, skip=("env_s", "timing", "j")):
    assert len(ref) == len(got), ctx
    for f, (a, b) in enumerate(zip(ref, got)):
        assert a.keys() == b.keys(), (ctx, f, sorted(a.keys() ^ b.keys()))
        for k in a.keys() - set(skip):
            x, y = a[k], b[k]
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype, (ctx, f, k)
                if x.dtype == np.float32:  # bitwise (signed zeros included)
                    assert np.array_equal(x.view(np.uint32), y.view(np.uint32)), (ctx, f, k)
                else:
                    assert np.array_equal(x, y), (ctx, f, k)
            else:
                assert x == y, (ctx, f, k)


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
@pytest.mark.parametrize("recipe", ["b8", "plain"])
def test_levers_off_is_the_pre_m6_actor_bitwise(tmp_path, monkeypatch, recipe):
    base = _base_trainer(tmp_path, monkeypatch)
    for mod in (base, T):  # eps 0.3: pin the actor RNG stream through many draws
        monkeypatch.setattr(mod.Schedules, "eps", lambda self, t: 0.3)
        _grow_bodies(monkeypatch, mod)
    init = _init(tmp_path)
    extra = B8_ACTOR if recipe == "b8" else ()
    frames = 60  # crosses a segment end (45 frames)
    runs = {}
    for name, mod, more in (
        ("old", base, ()),
        ("new-default", T, ()),
        ("new-explicit-zero", T, ("--tight-cost", "0", "--death-mass-mult", "0")),
    ):
        d = tmp_path / name
        d.mkdir()
        runs[name] = _run_actor(mod, d, init, frames, (*extra, *more))
    ref_msgs, ref_ring = runs["old"]
    assert len(ref_msgs) == frames and sum(m["decisions"] for m in ref_msgs) > 300
    assert sum(m["deaths"] for m in ref_msgs) > 0  # the death path is exercised
    for name in ("new-default", "new-explicit-zero"):
        msgs, ring = runs[name]
        _assert_msgs_equal(ref_msgs, msgs, name)
        for k in ref_ring:
            assert np.array_equal(ref_ring[k], ring[k]), (name, k)


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_levers_on_change_only_the_rewards_by_the_lever_terms(tmp_path, monkeypatch):
    import src.simd_env.ego_raster_bv8 as bv8mod

    monkeypatch.setattr(T.Schedules, "eps", lambda self, t: 0.3)
    init = _init(tmp_path)
    lmin, c, w = 500, 2.0, 2.0
    _grow_bodies(monkeypatch, T)
    rec = {}
    real_scalars = bv8mod.v8_scalars

    def scalars(sim, rows, mask=None, lam=8.0):
        out = real_scalars(sim, rows, mask, lam)
        L = sim.length[rows[:, 0], rows[:, 1]].astype(np.int64)
        rec["v8"].append((np.array(rows), np.array(mask), out[1].area.copy(), L))
        return out

    class Recorder(T.NStepBuilder):
        def step(self, worlds, ids, acts, reward, done, gamma_now):
            w = np.asarray(worlds)
            rec["r"].append((w.copy(), np.asarray(reward, np.float64)[w], np.asarray(done)[w]))
            return super().step(worlds, ids, acts, reward, done, gamma_now)

    monkeypatch.setattr(bv8mod, "v8_scalars", scalars)
    monkeypatch.setattr(T, "NStepBuilder", Recorder)
    frames = 60
    runs = {}
    for name, more in (
        ("off", ()),
        ("on", ("--tight-cost", str(c), "--tight-min-len", str(lmin), "--death-mass-mult", str(w))),
    ):
        rec[name] = {"v8": [], "r": []}
        rec.update(rec[name])
        d = tmp_path / name
        d.mkdir()
        runs[name] = _run_actor(T, d, init, frames, (*B8_ACTOR, *more))
    (off_msgs, off_ring), (on_msgs, on_ring) = runs["off"], runs["on"]
    # everything but R and the M6 counters is unchanged
    m6_keys = {"m6_death_len", "m6_big", "m6_tight", "m6_big1k", "m6_tight1k"}
    for f, (a, b) in enumerate(zip(off_msgs, on_msgs)):
        assert b.keys() - a.keys() == m6_keys and not a.keys() - b.keys(), f
    _assert_msgs_equal(
        [{k: v for k, v in m.items() if k != "R"} for m in off_msgs],
        [{k: v for k, v in m.items() if k not in m6_keys | {"R"}} for m in on_msgs],
        "on-vs-off",
    )
    for k in off_ring:
        assert np.array_equal(off_ring[k], on_ring[k]), k
    # the rewards differ by exactly the lever terms, recomputed here from the recorded inputs
    gamma = T.schedules_from_args(T.build_parser().parse_args(_argv(*B8_ACTOR))).gamma(9_000_000)
    assert gamma == 0.998
    off, on = rec["off"], rec["on"]
    acted = [m for m in on_msgs if m["decisions"]]  # frames with hero rows: one record each
    assert len(off["r"]) == len(on["r"]) == len(on["v8"]) == len(off["v8"]) == len(acted)
    assert len(acted) >= frames - 5
    n = {"tight": 0, "deaths": 0, "rows": 0}
    for f, ((w0, r0, d0), (w1, r1, d1), (rows, mask, area, L), m) in enumerate(
        zip(off["r"], on["r"], on["v8"], acted)
    ):
        assert np.array_equal(w0, w1) and np.array_equal(d0, d1)
        assert np.array_equal(rows[:, 0], w1), f  # the v8 rows are the hero rows, in order
        tight = np.where(mask, area, 0).max(axis=1) < L  # census.py's best_area < L
        charged = tight & (L >= lmin)
        want = np.where(charged, -c * (1.0 - gamma) * (L / 10.0), 0.0)
        want = want + np.where(d1, -w * (L / 10.0), 0.0)
        np.testing.assert_allclose(r1 - r0, want, rtol=0, atol=1e-9, err_msg=str(f))
        assert m["m6_big"] == int((L >= lmin).sum()) and m["m6_tight"] == int(charged.sum())
        assert m["m6_big1k"] == int((L >= 1000).sum())
        assert m["m6_tight1k"] == int((tight & (L >= 1000)).sum())
        assert m["m6_death_len"] == int(L[d1].sum())
        n["tight"] += int(charged.sum())
        n["deaths"] += int(d1.sum())
        n["rows"] += len(L)
    print("m6 on-run:", n)
    assert n["tight"] >= 8 and n["deaths"] >= 1 and n["rows"] > 300  # not vacuous
    # R moved where (and only where) a window holds a lever term
    assert any(not np.array_equal(a["R"], b["R"]) for a, b in zip(off_msgs, on_msgs))


def test_lever_terms_are_finite_for_extreme_lengths():
    L = np.array([0, 1, 4096, 100_000])
    assert np.all(np.isfinite(T.tight_cost_term(L, np.ones(4, bool), 0.998, 5.0, 0)))
    assert np.all(np.isfinite(T.death_mass_term(L, np.ones(4, bool), 5.0)))
    assert math.isclose(T.death_mass_term([100_000], [True], 1.0)[0], -10_000.0)
