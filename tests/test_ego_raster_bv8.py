"""M4-B ``ego2s-bv8``: featurizer, cached v8 re-decision, warm-start widening, trainer, eval.

Real-state tests drive M3-C training worlds (``tests/test_v8_nb.build_world``: scripted,
frozen and mixed rosters, the frp3-s12 checkpoints) with bodies grown long enough that the
v7 re-rank and the v8 head layer fire.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008 import train_m3c as T
from research.redesign_m4b_20261012.widen_bv8 import widen_net
from src.model.ego2s_network import OBS_SHAPES, Ego2sNet, obs_tensors, spec_for
from src.model.obs_spec import EGO2S_BV8, EGO2S_OBS_SPECS, KNOWN_OBS_SPECS

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT, build_world  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008")
STUDENT = ART / "student_m2b_final.pth"
DEMO = ART / "data"
HAVE_DATA = STUDENT.exists() and (DEMO / "round-12").is_dir()


def drive(frames: int, seed: int, visit, worlds: int = 8, grow_to: int = 1200):
    """Run training worlds; ``visit(prepared, hero, sel_mask, frame)`` -> hero actions."""
    sim, runtime, policies, groups, _, _ = build_world(worlds, seed)
    rng = np.random.default_rng(seed)
    sim.length[:, :] = rng.integers(grow_to // 4, grow_to + 1, size=sim.length.shape)
    for frame in range(frames):

        def choose(prepared, frame=frame):
            alive = prepared.get_alive()
            masks = prepared.get_resolved_action_mask()
            actions = np.ones((sim.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero = np.argwhere(alive[:, :1])
            if len(hero):
                _, sel = runtime.selection(hero)
                actions[hero[:, 0], 0] = visit(prepared, hero, sel, frame)
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    mk = masks[live[:, 0], live[:, 1]]
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(mk, prepared, live)
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)


# ----------------------------------------------------------------------------- contract
def test_obs_spec_registered_but_not_served():
    assert OBS_SHAPES[EGO2S_BV8] == (7, 60)
    assert EGO2S_BV8 not in EGO2S_OBS_SPECS and EGO2S_BV8 not in KNOWN_OBS_SPECS
    from src.simd_env.ego2s_policy import ego_config_for

    assert spec_for(Ego2sNet(7, 60)) == EGO2S_BV8
    assert ego_config_for(Ego2sNet(7, 60)).version == "bv8"
    assert ego_config_for(Ego2sNet(7, 30)).version == "b"


def test_bv8_refuses_an_ego_grid_view():
    from src.simd_env.ego_raster_bv8 import v8_scalars

    with pytest.raises(TypeError):
        v8_scalars(object(), np.zeros((1, 2), dtype=np.int64))


# ----------------------------------------------------------------------------- real states
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_bv8_featurizer_q_independent_and_redecide_matches_v8_decide():
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster
    from src.simd_env.ego_raster_bv8 import v8_scalars
    from src.simd_env.v8_nb import v8_decide, v8_feature_planes, v8_redecide

    rng = np.random.default_rng(5)
    st = {"rows": 0, "v7": 0, "head": 0, "veto": 0, "reasons": np.zeros(5, dtype=np.int64)}

    def visit(sim, hero, sel, frame):
        n = len(hero)
        # features: shape, range, Q-independence, advisory mask == runtime selection mask
        adv = sim.get_advisory_action_mask()[hero[:, 0], hero[:, 1]]
        assert np.array_equal(adv, sel)
        q = rng.normal(size=(n, 6)).astype(np.float32)
        if frame % 2:  # coarse Q: many exact ties
            q = rng.integers(0, 3, size=(n, 6)).astype(np.float32)
        # Random hard masks on some frames reach the v5 landing / v8 branches the advisory
        # mask rarely allows (the re-decision must match for any mask).
        m = rng.random((n, 6)) < 0.5 if frame % 4 == 3 else sel
        feats, res = v8_scalars(sim, hero, m)
        assert feats.shape == (n, 30) and feats.dtype == np.float32
        assert feats.min() >= 0.0 and feats.max() <= 1.0
        full = v8_decide(sim, hero, q, m, 8.0, features=True)
        assert np.array_equal(v8_feature_planes(full).reshape(n, 30), feats)
        # the cached re-decision equals both v8_decide modes, every decision field
        again = v8_redecide(res, q, m)
        lazy = v8_decide(sim, hero, q, m, 8.0)
        for k in ("action", "base", "v5_action", "v7_action", "outcome", "v5_reason"):
            assert np.array_equal(getattr(again, k), getattr(full, k)), k
            assert np.array_equal(getattr(again, k), getattr(lazy, k)), k
        st["rows"] += n
        st["v7"] += int((again.v7_action != again.v5_action).sum())
        st["head"] += int((again.action != again.v7_action).sum())
        st["veto"] += int((again.action != again.base).sum())
        st["reasons"] += np.bincount(again.v5_reason, minlength=5)
        if m is not sel:
            return again.action
        feats, _ = v8_scalars(sim, hero)  # default mask: the advisory mask
        # the featurizer: b planes / scalars unchanged, v8 scalars appended
        ob = build_ego_raster(sim, EgoRasterConfig(version="b"), backend="numba", rows=hero)
        ov = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=hero)
        assert ov["scalars"].shape == (n, 60)
        assert np.array_equal(ov["scalars"][:, :30], ob["scalars"])
        assert np.array_equal(ov["scalars"][:, 30:], feats)
        assert np.array_equal(ov["local"], ob["local"])
        assert np.array_equal(ov["global"], ob["global"])
        if frame % 10 == 0:  # whole-batch form: (E, S, 60), dead rows zero
            whole = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba")
            assert whole["scalars"].shape == (sim.E, sim.S, 60)
            assert np.array_equal(whole["scalars"][hero[:, 0], hero[:, 1]], ov["scalars"])
            assert not whole["scalars"][~sim.alive].any()
        return again.action

    for seed in (1, 3):
        drive(900, seed=seed, visit=visit, grow_to=1500)
    print("bv8 featurizer:", st)
    assert st["rows"] >= 14000 and st["veto"] > 0 and st["v7"] > 0 and st["head"] > 0
    assert np.all(st["reasons"] > 0)  # every v5 branch, incl. the landing vetoes


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_widening_is_exact_on_real_states():
    from src.model.ego2s_network import load_ego2s_checkpoint
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    torch.manual_seed(0)
    nets = [Ego2sNet(7, 30).eval()]
    if HAVE_DATA:
        nets.append(load_ego2s_checkpoint(str(STUDENT)))
    pairs = [(n, widen_net(n)) for n in nets]
    assert all(spec_for(w) == EGO2S_BV8 for _, w in pairs)
    st = {"rows": 0}

    def visit(sim, hero, sel, frame):
        ob = build_ego_raster(sim, EgoRasterConfig(version="b"), backend="numba", rows=hero)
        ov = build_ego_raster(sim, EgoRasterConfig(version="bv8"), backend="numba", rows=hero)
        assert ov["scalars"][:, 30:].any()  # the new inputs are really non-zero
        with torch.no_grad():
            for nb, nw in pairs:
                assert torch.equal(nb(*obs_tensors(ob, "cpu")), nw(*obs_tensors(ov, "cpu")))
        st["rows"] += len(hero)
        return np.ones(len(hero), dtype=np.int64)

    drive(60, seed=2, visit=visit)
    assert st["rows"] >= 300


def test_widen_tool_writes_bv8_checkpoint(tmp_path):
    from research.redesign_m4b_20261012 import widen_bv8
    from src.model.ego2s_network import load_ego2s_checkpoint, save_ego2s_checkpoint

    src = tmp_path / "b.pth"
    save_ego2s_checkpoint(src, Ego2sNet(7, 30))
    out = tmp_path / "bv8.pth"
    assert widen_bv8.main([str(src), str(out)]) == 0
    wide = load_ego2s_checkpoint(str(out))
    assert spec_for(wide) == EGO2S_BV8
    w = wide.fuse[0].weight.detach()
    assert w.shape[1] == 384 + 60 and not w[:, 384 + 30 :].any()
    assert widen_bv8.main([str(src), str(out)]) == 2  # create-only
    with pytest.raises(ValueError):
        widen_net(wide)


# ----------------------------------------------------------------------------- trainer
def _args(*extra):
    return ["--seed", "0", "--init", "x", "--demo-data", ".", "--out", "x", *extra]


@pytest.mark.parametrize(
    "extra",
    [
        ["--actors", "1"],
        ["--actors", "1", "--onpolicy-labels", "--anchor-mode", "v8margin"],  # label-frac 0.5
        ["--actors", "1", "--onpolicy-labels", "--label-frac", "1.0"],  # m3 anchor
    ],
)
def test_bv8_trainer_requires_label_only_margin_anchor(extra):
    with pytest.raises(SystemExit):
        T.main(_args("--obs-version", "bv8", *extra))


def test_bv8_trainer_requires_actors():
    with pytest.raises(SystemExit, match="multi-actor"):
        T.main(_args("--obs-version", "bv8"))


def test_obs_version_flag_and_ring_width():
    assert T.nscal_for(T.default_args()) == 30
    args = T.build_parser().parse_args(_args("--obs-version", "bv8"))
    assert T.nscal_for(args) == 60 and T.m3c_flags(args) == {"obs_version": "bv8"}
    cap, n = 64, 5
    shm = shared_memory.SharedMemory(create=True, size=T.ring_nbytes(cap, 60))
    try:
        ring = T.SharedObsRing(shm, cap, 8, 60)
        obs = {
            "local": np.zeros((n,) + T.LOCAL, np.uint8),
            "global": np.zeros((n,) + T.GLOBAL, np.uint8),
            "scalars": np.arange(n * 60, dtype=np.float32).reshape(n, 60),
        }
        ids = ring.add(obs, np.ones((n, 6), bool))
        multi = T.MultiRing([ring])
        _, _, sc, _ = multi.get(multi.encode(0, ids))
        assert np.array_equal(sc, obs["scalars"])
        del ring, multi, sc
    finally:
        shm.close()
        shm.unlink()


def _env():
    return dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")


@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_DATA and HAVE_CKPT), reason="needs the M2b student and data")
def test_bv8_multi_actor_smoke(tmp_path):
    init = tmp_path / "bv8.pth"
    from research.redesign_m4b_20261012 import widen_bv8

    assert widen_bv8.main([str(STUDENT), str(init)]) == 0
    out = tmp_path / "run"
    argv = [
        sys.executable,
        str(REPO / "research/redesign_m3c_20261008/train_m3c.py"),
        *("--seed", "7", "--init", str(init), "--demo-data", str(DEMO), "--out", str(out)),
        *("--transitions", "14000", "--worlds", "16", "--segment-frames", "300"),
        *("--replay", "8000", "--warmup", "3000", "--checkpoint-every", "7000"),
        *("--device", "cpu", "--actors", "1", "--no-guard", "--ring-dir", str(tmp_path)),
        *("--obs-version", "bv8", "--onpolicy-labels", "--label-frac", "1.0"),
        *("--anchor-mode", "v8margin", "--shield", "v8", "--reward", "mass"),
    ]
    r = subprocess.run(argv, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=1500)
    assert r.returncode == 0, r.stderr[-3000:]
    s = json.loads((out / "summary.json").read_text())
    assert s["halt"] is None and s["actor_exitcodes"] == [0]
    assert s["obs_version"] == "bv8" and s["m3c"]["obs_version"] == "bv8"
    assert s["transitions"] >= 14000 and s["updates"] >= 60
    # TD only until 10k labels, then label-only anchor updates
    assert s["label_buffer"] >= T.LABEL_ANCHOR_MIN_ROWS
    assert 0 < s["anchor_updates"] < s["updates"]
    blob = torch.load(out / "final.pth", map_location="cpu", weights_only=False)
    assert blob["obs_spec"] == EGO2S_BV8
    assert not list(tmp_path.glob("m3c-ring*.bin"))


@pytest.mark.slow
@pytest.mark.skipif(not (HAVE_DATA and HAVE_CKPT), reason="needs the M2b student and data")
def test_dev_probe_no_veto_widened_equals_original(tmp_path):
    """End to end: a widened bv8 student plays exactly the original b student's games."""
    init = tmp_path / "bv8.pth"
    from research.redesign_m4b_20261012 import widen_bv8

    assert widen_bv8.main([str(STUDENT), str(init)]) == 0
    rows = []
    for name, ckpt in (("b", STUDENT), ("bv8", init)):
        out = tmp_path / f"{name}.jsonl"
        argv = [
            sys.executable,
            str(REPO / "research/redesign_m2_20261008/dev_probe.py"),
            *("--student", str(ckpt), "--arm", "student", "--mix", "mixed"),
            *("--worlds", "2", "--frames", "300", "--ego2s-mask", "advisory", "--out", str(out)),
        ]
        r = subprocess.run(argv, cwd=REPO, env=_env(), capture_output=True, text=True, timeout=900)
        assert r.returncode == 0, r.stderr[-3000:]
        rows.append(json.loads(out.read_text().splitlines()[-1]))
    assert rows[0]["mass"] == rows[1]["mass"] and rows[0]["deaths"] == rows[1]["deaths"]
