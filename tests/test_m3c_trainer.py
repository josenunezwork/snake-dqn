"""M3-C trainer: flags default to M3 exactly; new pieces (mass reward, n-step builder, shared
rings, weighted anchor, multi-actor) are correct. Slow tests run real tiny trainings."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from collections import deque
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3_20261008 import train_m3 as m3
from research.redesign_m3c_20261008 import train_m3c as m3c

REPO = Path(__file__).resolve().parents[1]
ART = Path("/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008")
STUDENT = ART / "student_m2b_final.pth"
DEMO = ART / "data"
HAVE_DATA = STUDENT.exists() and (DEMO / "round-12").is_dir()


def test_defaults_equal_m3():
    from dataclasses import asdict

    assert asdict(m3c.Schedules()) == asdict(m3.Schedules())
    d = m3c.default_args()
    assert d.reward == "v2" and d.actors == 0 and d.anchor_boost_weight == 1.0
    assert d.mixed_frac == 0.0 and d.namespace == m3.TRAIN_NAMESPACE
    assert asdict(m3c.schedules_from_args(d)) == asdict(m3.Schedules())
    assert m3c.m3c_flags(d) == {}
    flags = m3c.m3c_flags(
        m3c.build_parser().parse_args(
            [
                "--seed",
                "1",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "y",
                "--reward",
                "mass",
                "--mass-kappa",
                "0.5",
                "--actors",
                "3",
                "--mixed-frac",
                "0.33",
            ]
        )
    )
    assert flags == {"reward": "mass", "mass_kappa": 0.5, "actors": 3, "mixed_frac": 0.33}


def test_kappa_needs_mass_reward():
    with pytest.raises(SystemExit):
        m3c.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "/nonexistent/x",
                "--mass-kappa",
                "2",
            ]
        )


def test_mass_term():
    out = m3c.mass_term(np.array([100, 300, 50]), np.array([True, True, False]), 0.995, 1.0)
    np.testing.assert_allclose(out, [0.005 * 10.0, 0.005 * 30.0, 0.0])
    # a snake that keeps L forever: sum_k gamma^k (1-gamma) Phi = Phi
    g = 0.995
    r = m3c.mass_term(np.array([400]), np.array([True]), g, 1.0)[0]
    assert abs(sum(r * g**k for k in range(20000)) - 40.0) < 1e-6


def _sync_reference(frames, n_worlds, n_step=5):
    """The M3 sync loop's n-step bookkeeping, transcribed (train_m3.main)."""
    out = []
    pend = [deque() for _ in range(n_worlds)]
    last_obs = np.full(n_worlds, -1, dtype=np.int64)

    def flush(e, bootstrap, gamma_now):
        while pend[e]:
            R, g = 0.0, 1.0
            for _, _, r in pend[e]:
                R += g * r
                g *= gamma_now
            s0, a0, _ = pend[e].popleft()
            out.append((s0, a0, R, 0.0 if bootstrap < 0 else g, bootstrap))

    for hero, ids, acts, reward, done, gamma_now in frames:
        for j, e in enumerate(hero):
            e = int(e)
            if len(pend[e]) == n_step:
                R, g = 0.0, 1.0
                for _, _, r in pend[e]:
                    R += g * r
                    g *= gamma_now
                s0, a0, _ = pend[e].popleft()
                out.append((s0, a0, R, g, int(ids[j])))
            pend[e].append((int(ids[j]), int(acts[j]), float(reward[e])))
            last_obs[e] = ids[j]
            if done[e]:
                flush(e, -1, gamma_now)
    gamma_end = frames[-1][-1]
    for e in range(n_worlds):
        if pend[e]:
            pend[e].pop()
            flush(e, int(last_obs[e]), gamma_end)
    return out


def test_nstep_builder_matches_sync_loop():
    rng = np.random.default_rng(3)
    E, frames, nid = 6, [], 0
    for f in range(80):
        alive = rng.random(E) < 0.85
        hero = np.flatnonzero(alive)
        ids = np.arange(nid, nid + len(hero))
        nid += len(hero)
        frames.append(
            (
                hero,
                ids,
                rng.integers(0, 6, len(hero)),
                rng.normal(size=E),
                rng.random(E) < 0.05,
                0.99 + 0.00005 * f,
            )
        )
    ref = _sync_reference(frames, E)
    b = m3c.NStepBuilder(E)
    got = []
    for hero, ids, acts, reward, done, g in frames:
        b.step(hero, ids, acts, reward, done, g)
        got += b.out
        b.out = []
    b.end_segment(frames[-1][-1])
    got += b.out
    assert got == ref and len(ref) > 300
    b2 = m3c.NStepBuilder(E)
    for hero, ids, acts, reward, done, g in frames[:10]:
        b2.step(hero, ids, acts, reward, done, g)
    s, a, R, d, nx = b2.take()
    assert len(s) == len(a) == len(R) and s.dtype == np.int64 and R.dtype == np.float32
    assert b2.take()[0].size == 0


def test_shared_ring_and_multiring():
    cap, margin = 50, 5
    shms = [shared_memory.SharedMemory(create=True, size=m3c.ring_nbytes(cap)) for _ in range(2)]
    try:
        rings = [m3c.SharedObsRing(s, cap, margin) for s in shms]
        rings[0]._count[0] = 0
        rings[1]._count[0] = 0
        multi = m3c.MultiRing(rings)

        def obs(n, v):
            return {
                "local": np.full((n,) + m3c.LOCAL, v, np.uint8),
                "global": np.full((n,) + m3c.GLOBAL, v, np.uint8),
                "scalars": np.full((n, m3c.NSCAL), v, np.float32),
            }

        ids0 = rings[0].add(obs(30, 7), np.ones((30, 6), bool))
        ids1 = rings[1].add(obs(10, 9), np.zeros((10, 6), bool))
        g0, g1 = multi.encode(0, ids0), multi.encode(1, ids1)
        assert set(g0 % 2) == {0} and set(g1 % 2) == {1}
        assert multi.encode(1, np.array([-1]))[0] == -1
        l_, gl, sc, mk = multi.get(np.concatenate([g0[:3], g1[:2]]))
        assert (l_[:3] == 7).all() and (l_[3:] == 9).all() and mk[:3].all() and not mk[3:].any()
        assert multi.valid(g0).all() and multi.valid(g1).all()
        rings[0].add(obs(40, 1), np.ones((40, 6), bool))  # count 70: ids < 25 are stale
        v = multi.valid(g0)
        assert not v[:25].any() and v[25:].all()
        assert not multi.valid(np.array([-1]))[0]
    finally:
        for s in shms:
            s.close()
            s.unlink()


def test_weighted_anchor_reduces_to_anchor():
    torch = pytest.importorskip("torch")
    g = torch.Generator().manual_seed(0)
    q = torch.randn(64, 6, generator=g)
    b = {
        "q_teacher": torch.randn(64, 6, generator=g),
        "a_v8": torch.randint(0, 6, (64,), generator=g),
        "mask_resolved": torch.rand(64, 6, generator=g) > 0.2,
    }
    a = m3c.anchor_loss(torch, q, b, 0.3)
    w1 = m3c.anchor_loss_weighted(torch, q, b, 0.3, 1.0)
    assert torch.allclose(a, w1, rtol=1e-5, atol=1e-6)
    w10 = m3c.anchor_loss_weighted(torch, q, b, 0.3, 10.0)
    assert not torch.allclose(a, w10)
    assert m3c.anchor_loss is not m3.anchor_loss  # copied, and identical in value:
    assert torch.equal(m3.anchor_loss(torch, q, b, 0.3), a)


def test_weights_seqlock_roundtrip():
    torch = pytest.importorskip("torch")
    net = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    spec = m3c._weights_spec(net)
    n = sum(k for _, _, k in spec)
    wver = np.zeros(1, np.int64)
    wbuf = np.zeros(n + 1, np.float64)
    m3c._publish_weights(net, wbuf, wver)
    assert wver[0] == 2
    other = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
    assert m3c._read_weights(other, spec, wbuf, wver, -1) == 2
    for a, b in zip(net.state_dict().values(), other.state_dict().values()):
        assert torch.equal(a, b)
    assert m3c._read_weights(other, spec, wbuf, wver, 2) == 2  # nothing newer
    wver[0] = 3  # a writer in progress: never read
    assert m3c._read_weights(other, spec, wbuf, wver, 2) == 2
    wver[0] = 4
    wbuf[0] += 1.0  # torn copy: checksum mismatch -> rejected
    assert m3c._read_weights(other, spec, wbuf, wver, 2) == 2


# ----------------------------------------------------------------------------- slow: real runs
TINY = [
    "--transitions",
    "12000",
    "--worlds",
    "16",
    "--segment-frames",
    "300",
    "--replay",
    "8000",
    "--demo-capacity",
    "20000",
    "--demo-initial-shards",
    "2",
    "--warmup",
    "3000",
    "--checkpoint-every",
    "6000",
    "--device",
    "cpu",
]


def _run(script, out, extra=()):
    env = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", SNAKE_DQN_DEVICE="cpu")
    argv = [
        sys.executable,
        str(REPO / script),
        "--seed",
        "7",
        "--init",
        str(STUDENT),
        "--demo-data",
        str(DEMO),
        "--out",
        str(out),
        *TINY,
        *extra,
    ]
    return subprocess.run(argv, cwd=REPO, env=env, capture_output=True, text=True, timeout=1500)


def _weights(path):
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _state(blob):
    for key in ("state_dict", "model_state_dict", "model"):
        if isinstance(blob, dict) and key in blob:
            return blob[key]
    return blob


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="needs the M2b student and demonstrations")
def test_default_flags_reproduce_m3_bitwise(tmp_path):
    import torch

    r3 = _run("research/redesign_m3_20261008/train_m3.py", tmp_path / "m3")
    if r3.returncode == 3:
        pytest.skip("Mac guard refused (AC / lid / thermal)")
    assert r3.returncode == 0, r3.stderr[-2000:]
    rc = _run("research/redesign_m3c_20261008/train_m3c.py", tmp_path / "m3c")
    assert rc.returncode == 0, rc.stderr[-2000:]
    for name in ("ckpt_6k.pth", "final.pth"):
        a, b = _state(_weights(tmp_path / "m3" / name)), _state(_weights(tmp_path / "m3c" / name))
        assert a.keys() == b.keys()
        for k in a:
            assert torch.equal(a[k], b[k]), (name, k)
    s3 = json.loads((tmp_path / "m3" / "summary.json").read_text())
    sc = json.loads((tmp_path / "m3c" / "summary.json").read_text())
    for k in (
        "transitions",
        "updates",
        "halt",
        "flags",
        "demo_qmax",
        "demo_distinct",
        "holdout_agreement_final",
    ):
        assert s3[k] == sc[k], k
    assert sc["m3c"] == {} and sc["schema"] == s3["schema"]
    rows = [json.loads(x) for x in (tmp_path / "m3c" / "manifest.jsonl").read_text().splitlines()]
    assert [r["file"] for r in rows] == ["ckpt_6k.pth", "final.pth"]
    assert all(r["sha256"] == m3c.sha256_file(tmp_path / "m3c" / r["file"]) for r in rows)


@pytest.mark.slow
@pytest.mark.skipif(not HAVE_DATA, reason="needs the M2b student and demonstrations")
def test_mass_reward_and_multi_actor_smoke(tmp_path):
    r = _run(
        "research/redesign_m3c_20261008/train_m3c.py",
        tmp_path / "multi",
        [
            "--reward",
            "mass",
            "--actors",
            "2",
            "--mixed-frac",
            "0.25",
            "--no-guard",
            "--ring-dir",
            str(tmp_path),
        ],
    )
    assert r.returncode == 0, r.stderr[-3000:]
    out = tmp_path / "multi"
    s = json.loads((out / "summary.json").read_text())
    assert s["halt"] is None and s["actors"] == 2 and s["actor_exitcodes"] == [0, 0]
    assert s["transitions"] >= 12000 and s["updates"] >= 60
    assert s["m3c"] == {"reward": "mass", "actors": 2, "mixed_frac": 0.25}
    rows = [json.loads(x) for x in (out / "manifest.jsonl").read_text().splitlines()]
    assert rows[-1]["file"] == "final.pth"
    assert not list(tmp_path.glob("m3c-ring*.bin"))  # file-backed rings removed at exit
    blob = _weights(out / "final.pth")
    meta = blob.get("meta", blob) if isinstance(blob, dict) else {}
    assert "redesign-m3c-train/v1" in json.dumps(meta, default=str)


def test_ego2s_simd_policy_advisory_mask(monkeypatch):
    """advisory mode = the live AISnake rule (advisory mask, normal-only fallback); the
    default resolved mode is unchanged."""
    torch = pytest.importorskip("torch")
    from src.simd_env import ego2s_policy as ep

    q = torch.tensor([[1.0, 2.0, 3.0, 9.0, 0.0, 0.0], [5.0, 1.0, 1.0, 9.0, 9.0, 9.0]])
    monkeypatch.setattr(ep, "student_q", lambda net, sim, slots, forward: q)

    class Sim:
        def get_advisory_action_mask(self):
            m = np.zeros((2, 6, 6), dtype=bool)
            m[0, 0, :3] = [True, True, False]  # row 0: boost unsafe, normal 0/1 safe
            return m  # row 1 (env 1): nothing safe -> normal-only fallback

    pol = object.__new__(ep.Ego2sSimdPolicy)
    pol.net, pol.forward = None, "batched"
    resolved = np.array([[1, 1, 1, 1, 1, 1], [0, 0, 0, 0, 0, 0]], dtype=bool)
    slots = np.array([[0, 0], [1, 0]])
    pol.mask_mode = "resolved"
    assert pol.actions(resolved, Sim(), slots).tolist() == [3, 0]
    pol.mask_mode = "advisory"
    assert pol.actions(resolved, Sim(), slots).tolist() == [1, 0]
    with pytest.raises(ValueError):
        ep.Ego2sSimdPolicy.__init__(pol, "x", mask_mode="legal")
