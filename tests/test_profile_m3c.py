"""perf-profile (2026-10-13): the opt-in ``M3C_PROFILE`` hook of ``train_m3c``.

* Off by default: ``ProcessProfile.from_env`` returns ``None`` unless ``M3C_PROFILE`` names a
  mode (an unknown mode is an error, not a silent no-op).
* Warm-up, dump files and the compact pyinstrument session (identical stacks merged; the
  merged session still loads and keeps the total time).
* End to end: ``actor_main`` with ``M3C_PROFILE=cprofile|pyinstrument`` emits exactly the
  frame messages and ring rows it emits with profiling off (B8 actor flags), and writes its
  dump. (The learner hook only reads counters; the multi-process learner is not
  bit-reproducible, so it has no end-to-end parity test.)
* ``analyze_profiles.classify``: innermost-rule attribution and the generic-leaf refinement.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

from research.perf_profile_20261013 import analyze_profiles as A
from research.redesign_m3c_20261008 import profile_m3c as P
from research.redesign_m3c_20261008 import train_m3c as T

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_perf_actor import _run_actor  # noqa: E402
from test_v8_nb import HAVE_CKPT  # noqa: E402

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


def test_profile_off_by_default(monkeypatch):
    monkeypatch.delenv(P.PROFILE_ENV, raising=False)
    assert P.profile_mode() is None
    assert P.ProcessProfile.from_env("actor0") is None
    for off in ("", "off", "0", "none", "OFF"):
        monkeypatch.setenv(P.PROFILE_ENV, off)
        assert P.ProcessProfile.from_env("actor0") is None, off
    monkeypatch.setenv(P.PROFILE_ENV, "pyspy")
    with pytest.raises(ValueError):
        P.ProcessProfile.from_env("actor0")


def test_profile_warm_and_cprofile_dump(tmp_path, monkeypatch):
    monkeypatch.setenv(P.PROFILE_ENV, "cprofile")
    monkeypatch.setenv(P.PROFILE_DIR_ENV, str(tmp_path / "p"))
    monkeypatch.setenv(P.PROFILE_WARM_ENV, "3600")
    idle = P.ProcessProfile.from_env("learner")
    assert idle is not None and not idle.tick() and not idle.running
    assert idle.dump() is None  # never started: nothing written
    assert not (tmp_path / "p").exists()

    monkeypatch.setenv(P.PROFILE_WARM_ENV, "0")
    prof = P.ProcessProfile.from_env("actor3", tmp_path / "unused")
    assert prof.out_dir == tmp_path / "p"
    assert prof.tick() and prof.running
    for _ in range(4):
        sum(i * i for i in range(1000))
        prof.tick()
    meta_path = prof.dump({"worlds": 7})
    meta = json.loads(meta_path.read_text())
    assert meta["role"] == "actor3" and meta["mode"] == "cprofile"
    assert meta["ticks"] == 4 and meta["worlds"] == 7 and meta["profiled_s"] > 0
    assert (tmp_path / "p" / "actor3.pstats").stat().st_size > 0
    assert "function calls" in (tmp_path / "p" / "actor3.txt").read_text()
    assert not prof.tick()  # stopped for good


def test_profile_default_dir_is_out_profile(tmp_path, monkeypatch):
    monkeypatch.setenv(P.PROFILE_ENV, "cprofile")
    monkeypatch.delenv(P.PROFILE_DIR_ENV, raising=False)
    prof = P.ProcessProfile.from_env("learner", tmp_path / "run")
    assert prof.out_dir == tmp_path / "run" / "profile"


def test_profile_duration_starts_after_first_tick_warmup(tmp_path, monkeypatch):
    now = [10.0]
    monkeypatch.setattr(P.time, "perf_counter", lambda: now[0])
    monkeypatch.setenv(P.PROFILE_ENV, "cprofile")
    monkeypatch.setenv(P.PROFILE_WARM_ENV, "60")
    monkeypatch.setenv(P.PROFILE_FIRST_TICK_ENV, "1")
    monkeypatch.setenv(P.PROFILE_DURATION_ENV, "60")
    prof = P.ProcessProfile.from_env("learner", tmp_path)
    now[0] = 210.0  # expensive setup must not consume the warm-up
    assert not prof.tick() and not prof.running
    now[0] = 269.0
    assert not prof.tick()
    now[0] = 270.0
    assert prof.tick() and prof.running
    now[0] = 329.0
    assert not prof.tick() and prof.running
    now[0] = 330.25  # stop on the first loop tick past the duration
    assert not prof.tick() and not prof.running
    now[0] = 400.0
    assert not prof.tick()
    meta = json.loads(prof.dump().read_text())
    assert meta["warm_s"] == 60
    assert meta["profiled_s"] == 60.25
    assert meta["requested_duration_s"] == 60
    assert meta["warm_from_first_tick"] is True


@pytest.mark.parametrize("duration", ["-1", "nan", "inf"])
def test_profile_duration_rejects_invalid_values(tmp_path, monkeypatch, duration):
    monkeypatch.setenv(P.PROFILE_ENV, "cprofile")
    monkeypatch.setenv(P.PROFILE_DURATION_ENV, duration)
    with pytest.raises(ValueError, match="DURATION_S"):
        P.ProcessProfile.from_env("learner", tmp_path)


def test_compact_session_merges_stacks(tmp_path):
    pytest.importorskip("pyinstrument")
    from pyinstrument.session import Session

    a, b = ["m\x00x.py\x001", "f\x00x.py\x003"], ["m\x00x.py\x001", "g\x00x.py\x009"]
    recs = [(list(a), 0.25), (list(b), 0.5), (list(a), 0.125), (list(a), 0.0625)]
    s = Session(recs, 0.0, 1.0, 0.001, 0.002, 4, [], "t", 0.5, [], [])
    n = P.save_session_compact(s, tmp_path / "x.pyisession")
    back = Session.load(str(tmp_path / "x.pyisession"))
    assert n == 2 and len(back.frame_records) == 2
    got = {tuple(st): t for st, t in back.frame_records}
    assert got == {tuple(a): 0.4375, tuple(b): 0.5}
    assert back.sample_count == 4


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
@pytest.mark.parametrize("mode", ["cprofile", "pyinstrument"])
def test_actor_messages_unchanged_by_profiling(tmp_path, monkeypatch, mode):
    if mode == "pyinstrument":
        pytest.importorskip("pyinstrument")
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    monkeypatch.setattr(T.Schedules, "eps", lambda self, t: 0.3)  # exercise the actor RNG
    torch.manual_seed(0)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    frames = 60  # crosses a segment end (45 frames)
    monkeypatch.delenv(P.PROFILE_ENV, raising=False)
    (tmp_path / "off").mkdir()
    off_msgs, off_ring = _run_actor(tmp_path / "off", init, "vectorized", "numba", frames, B8_ACTOR)
    monkeypatch.setenv(P.PROFILE_ENV, mode)
    monkeypatch.setenv(P.PROFILE_DIR_ENV, str(tmp_path / "prof"))
    monkeypatch.setenv(P.PROFILE_WARM_ENV, "0")
    (tmp_path / "on").mkdir()
    on_msgs, on_ring = _run_actor(tmp_path / "on", init, "vectorized", "numba", frames, B8_ACTOR)
    assert len(off_msgs) == len(on_msgs) == frames
    assert sum(m["decisions"] for m in off_msgs) > 200
    skip = {"env_s", "timing", "j"}
    for f, (a, b) in enumerate(zip(off_msgs, on_msgs)):
        assert a.keys() == b.keys(), f
        for k in a.keys() - skip:
            x, y = a[k], b[k]
            if isinstance(x, np.ndarray):
                assert x.dtype == y.dtype and np.array_equal(x, y), (f, k)
            else:
                assert x == y, (f, k)
    for k in off_ring:
        assert np.array_equal(off_ring[k], on_ring[k]), k
    meta = json.loads((tmp_path / "prof" / "actor0.meta.json").read_text())
    assert meta["mode"] == mode and meta["frames"] >= frames - 1
    assert meta["decisions"] == sum(m["decisions"] for m in on_msgs[-meta["frames"] :])
    ext = ".pstats" if mode == "cprofile" else ".pyisession"
    assert (tmp_path / "prof" / f"actor0{ext}").stat().st_size > 0


# ----------------------------------------------------------------------------- analyzer
def _id(func, path, cls=None):
    return f"{func}\x00{path}\x001" + (f"\x01c{cls}" if cls else "") + "\x01l2"


def test_classify_innermost_rule_and_refinement():
    idx = A._index(A.ACTOR_RULES)
    root = [_id("actor_main", "/r/research/redesign_m3c_20261008/train_m3c.py")]
    step = root + [_id("step_with_policy", "/r/src/simd_env/batch_sim.py", "GridBatchSim")]
    choose = step + [_id("choose", "/r/research/redesign_m3c_20261008/train_m3c.py")]
    to = _id("Tensor.to", "<built-in>")
    c: dict = {}
    assert A.classify(choose + [to], idx, c).startswith("hero H2D")
    assert A.classify(root + [to], idx, c) == "weight sync"
    food = step + [
        _id("step", "/r/src/simd_env/batch_sim.py", "GridBatchSim"),
        _id("_consume_food", "/r/src/simd_env/grid_sim.py", "GridBatchSim"),
        _id("_spawn", "/r/src/simd_env/grid_sim.py", "GridBatchSim"),
        _id("ufunc.at", "<built-in>"),
    ]
    assert A.classify(food, idx, c) == A.SIM_FOOD
    opp = choose + [
        _id("opponents", "/r/research/redesign_m3c_20261008/train_m3c.py"),
        _id("actions", "/r/src/simd_env/vector61_policy.py", "Vector61SimdPolicy"),
        _id("Tensor.cpu", "<built-in>"),
    ]
    # a generic leaf outside the policy glue keeps its enclosing component
    assert A.classify(opp, idx, c) == "opponents: vector61 mask/argmax"
    assert A.classify(choose + [_id("Tensor.cpu", "<built-in>")], idx, c) == A.HERO_D2H
    assert A.classify([_id("<module>", "x.py")], idx, c) == "other (outside the rules)"
    assert A.category(A.SIM_FOOD) == "grid sim internals"
    assert A.category("opponents: scripted") == "scripted opponents (fast_anchors)"
    # the food-list conversion the Python food lists cost the vector61 featurizer
    v61 = root + [
        _id("observe_step", "/r/src/simd_env/vector61_policy.py", "Vector61Runtime"),
        _id("featurize", "/r/src/simd_env/vector61_nb.py", "Vector61FeaturizerNB"),
        _id("_food_arrays", "/r/src/simd_env/vector61_nb.py", "Vector61FeaturizerNB"),
        _id("fromiter", "<built-in>"),
    ]
    assert A.classify(v61, idx, c) == A.V61_FOOD
    assert A.classify(v61[:-2], idx, c) == "vector61 featurize (post-step)"
    assert A.category(A.V61_FOOD) == "vector61 featurization + opponent MLPs"


def test_classify_learner_sync_points():
    idx = A._index(A.LEARNER_RULES)
    lm_py = "/r/research/redesign_m3c_20261008/learner_m3c.py"
    upd = [
        _id("main_multi", "/r/research/redesign_m3c_20261008/train_m3c.py"),
        _id("update", lm_py, "Learner"),
        _id("_update_fast", lm_py, "Learner"),
    ]
    c: dict = {}
    fin = upd + [_id("_finalize", lm_py, "Learner")]
    assert A.classify(fin, idx, c) == A.L_SYNC_FINAL  # self time = the finite-flag wait
    adam = fin + [_id("wrapper", "/t/optim/optimizer.py"), _id("_fused_adam", "/t/adam.py")]
    assert A.classify(adam, idx, c) == "learner/optim: Adam step (launch)"
    assert A.classify(upd + [_id("_masked_argmax", lm_py)], idx, c) == A.L_SYNC_ARGMAX
    up = upd + [_id("upload", lm_py, "_Staging"), _id("Tensor.to", "<built-in>")]
    assert A.classify(up, idx, c) == A.L_H2D
    assert A.classify(upd + [_id("Tensor.to", "<built-in>")], idx, c).startswith("learner/h2d")
    assert A.category(A.L_SYNC_FINAL) == A.category(A.L_SYNC_ARGMAX)
    assert A.category(A.L_SYNC_FINAL) == "device sync (host waits for GPU compute)"
    assert A.category(A.L_H2D) == "transfer: H2D batch upload"


def test_amdahl_bound():
    comps = {
        A.SIM_FOOD: 2.0,
        "sim: movement + grid writes": 1.0,
        "opponents: scripted": 1.0,
        A.V61_FOOD: 2.0,
        A.HERO_H2D: 4.0,
        "queue put (back-pressure: learner not keeping up)": 10.0,
    }
    am = A.amdahl(comps)
    assert am["profiled_s"] == 20.0 and am["busy_s"] == 10.0 and am["utilization"] == 0.5
    a, b = am["sim+scripted"], am["sim+scripted+food lists"]
    assert a["p"] == pytest.approx(0.4) and a["S_inf"] == pytest.approx(1 / 0.6)
    assert a["S_3x"] == pytest.approx(1 / (0.6 + 0.4 / 3))
    assert b["p"] == pytest.approx(0.6) and b["S_inf"] == pytest.approx(2.5)


# ----------------------------------------------------------------------------- learner bench
@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_masked_argmax_nosync_is_the_shipped_argmax(device):
    """``bench_learner_sync.masked_argmax_nosync`` (element-wise fallback, no host sync)
    picks exactly the shipped ``_masked_argmax`` actions, incl. all-false and tied rows."""
    import torch

    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("no MPS")
    from research.perf_profile_20261013 import bench_learner_sync as B
    from research.redesign_m3c_20261008 import learner_m3c as lm

    g = torch.Generator().manual_seed(3)
    n = 4096
    q = torch.randn(n, 6, generator=g)
    q[::7] = torch.round(q[::7])  # exact ties: argmax keeps the first maximum
    m = torch.rand(n, 6, generator=g) < 0.5
    m[::5] = False  # all-false rows: the shipped fallback enables the three normal actions
    m[1::5, 3:] = False
    q, m = q.to(device), m.to(device)
    m0 = m.clone()
    want = lm._masked_argmax(torch, q, m)
    got = B.masked_argmax_nosync(torch, q, m)
    assert torch.equal(m, m0)  # neither variant mutates the caller's mask
    assert int((~m.any(dim=1)).sum()) >= n // 5
    assert torch.equal(want, got)
