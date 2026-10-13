"""Eval records are identical with the compiled vector61 featurizer (perf-eval, item 8).

``run_simd_eval(vector61_backend="numba")`` against the default Python featurizer, through
``dev_probe.probe`` (the engine, profile, distillation-namespace worlds and strict rosters of
the probes ``probe_watch.py`` launches: grid engine, vector61 rowwise forwards, ego2s rowwise
hero forward, advisory mask for the no-veto student). Each run has the per-frame whole-world
digest recorder of ``grid_h5000_identity`` installed (every snake's body, heading, masks,
events, food, corpses and env RNG state after every frame), so equal digests mean equal
per-frame actions. Arms: the no-veto advisory student (M3-C / M4 primary probe), the student
inside the v8 veto and the v8-vetoed incumbent frp3-s12; every mix.

Negative controls: the numba run must really call the compiled featurizer (and never fall
back to the NumPy one), and a deliberately perturbed compiled featurizer must be caught.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
STUDENT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/redesign-m2b-20261008/student_m2b_final.pth"
)
HAVE_ALL = HAVE_CKPT and STUDENT.exists()
needs_artifacts = pytest.mark.skipif(
    not HAVE_ALL, reason="needs the pinned frp3-s12 pool and the M2b student"
)


@pytest.fixture(scope="module")
def ctx():
    import torch

    from research.redesign_scope_20261007 import grid_h5000_identity as gid

    torch.set_num_threads(1)
    return gid._context(1)


class Spy:
    """Counts compiled-featurizer calls and NumPy fallbacks in a run."""

    def __init__(self, monkeypatch, perturb=None):
        from src.simd_env.vector61_nb import Vector61FeaturizerNB

        self.calls = 0
        self.rows = 0
        self.fallbacks = 0
        orig = Vector61FeaturizerNB.featurize
        spy = self

        def featurize(featurizer, sim, rows=None, memory=None, update_memory=True):
            before = featurizer.fallbacks
            out = orig(featurizer, sim, rows, memory, update_memory)
            spy.calls += 1
            spy.rows += len(out)
            spy.fallbacks += featurizer.fallbacks - before
            return perturb(out) if perturb is not None else out

        monkeypatch.setattr(Vector61FeaturizerNB, "featurize", featurize)


def _pair(ctx, arm, mix, worlds, frames, monkeypatch, perturb=None):
    from research.perf_eval_20261013 import eval_speed as es

    ns = argparse.Namespace(
        mix=mix, worlds=worlds, frames=frames, round_index=52, student=str(STUDENT)
    )
    spy = Spy(monkeypatch, perturb)
    ref = es.run_recorded(es.probe_args(arm, "default", ns), ctx)
    assert spy.calls == 0  # the reference never touches the compiled featurizer
    var = es.run_recorded(es.probe_args(arm, "numba", ns), ctx)
    seeds = es.dev_probe.probe_seeds(worlds, 52)
    return es.compare(ref, var, seeds), ref, var, spy


def _assert_identical(rows, ref, var, spy, expect_vector61_rows=True):
    bad = [r for r in rows if not r["identical"]]
    assert not bad, bad[:2]
    assert all(r["frames_compared"] == len(ref["frame_digests"][0]) for r in rows)
    assert ref["row"]["mass"] == var["row"]["mass"]
    assert ref["row"]["deaths"] == var["row"]["deaths"]
    assert ref["row"]["causes"] == var["row"]["causes"]
    assert "v61_backend" not in ref["row"] and var["row"]["v61_backend"] == "numba"
    assert spy.fallbacks == 0
    if expect_vector61_rows:
        assert spy.calls > 0 and spy.rows > 0


@needs_artifacts
def test_numba_backend_records_identical_quick(ctx, monkeypatch):
    rows, ref, var, spy = _pair(ctx, "student-nv", "frozen", 2, 300, monkeypatch)
    _assert_identical(rows, ref, var, spy)


@needs_artifacts
def test_identity_check_catches_a_broken_featurizer(ctx, monkeypatch):
    """Negative control: a compiled featurizer that perturbs its rows is detected."""

    def perturb(out):
        out = out.copy()
        out[:, :8] = out[:, :8][:, ::-1]  # swap a few leading features: actions must change
        return out

    rows, ref, var, spy = _pair(ctx, "student-nv", "frozen", 2, 300, monkeypatch, perturb)
    assert spy.calls > 0
    assert any(not r["identical"] for r in rows)
    assert any(r["first_differing_frame"] is not None for r in rows)


@pytest.mark.slow
@needs_artifacts
@pytest.mark.parametrize("arm", ["student-nv", "student-v8", "incumbent-v8"])
@pytest.mark.parametrize("mix", ["frozen", "scripted", "mixed"])
def test_numba_backend_records_identical_every_arm_and_mix(ctx, monkeypatch, arm, mix):
    rows, ref, var, spy = _pair(ctx, arm, mix, 3, 600, monkeypatch)
    # scripted mix + no-veto student: no vector61 row exists, the backend is never called
    _assert_identical(
        rows, ref, var, spy, expect_vector61_rows=(arm, mix) != ("student-nv", "scripted")
    )


def test_run_simd_eval_validates_backend():
    from src.simd_env import eval_engine as ee

    with pytest.raises(ValueError, match="vector61_backend needs vector61=True"):
        ee.run_simd_eval(("scripted", "random_safe"), [], 1, [0], vector61_backend="numba")


def test_dev_probe_backend_resolution(monkeypatch):
    from research.redesign_m2_20261008 import dev_probe

    monkeypatch.delenv("SNAKE_V61_BACKEND", raising=False)
    assert dev_probe.effective_v61_backend(None) == "python"
    assert dev_probe.effective_v61_backend("numba") == "numba"
    monkeypatch.setenv("SNAKE_V61_BACKEND", "numba")
    assert dev_probe.effective_v61_backend(None) == "numba"
    assert dev_probe.effective_v61_backend("python") == "python"
    args = dev_probe.build_parser().parse_args(
        ["--arm", "student", "--mix", "frozen", "--out", "x", "--student", "s.pth"]
    )
    assert args.v61_backend is None and args.ego2s_forward == "batched"


def test_probe_watch_passes_the_backend_only_when_asked(tmp_path):
    from research.redesign_m3c_20261008 import probe_watch as pw

    job = ("A_s0", "final", "/x/final.pth", "none", "frozen")
    default = pw.probe_argv(job, tmp_path, 16, 52, slots=False)
    assert "--v61-backend" not in default
    numba = pw.probe_argv(job, tmp_path, 16, 52, slots=False, v61_backend="numba")
    assert numba[: len(default)] == default and numba[len(default) :] == ["--v61-backend", "numba"]
    slots = pw.probe_argv(job, tmp_path, 16, 52, slots=True, v61_backend="numba")
    assert slots[1].endswith("slotrun.py") and slots[-2:] == ["--v61-backend", "numba"]


def test_probe_watch_resumes_rows_of_either_backend(tmp_path):
    """Rows with ``v61_backend`` count as done (record-identical backends)."""
    from research.redesign_m3c_20261008 import probe_watch as pw

    sha = "a" * 64
    out = tmp_path / "A_s0__final__none.jsonl"
    base = {
        "student_sha256": sha,
        "veto": "none",
        "round_index": 52,
        "ego2s_forward": "rowwise",
        "ego2s_mask": "advisory",
    }
    out.write_text(
        json.dumps({**base, "mix": "frozen"})
        + "\n"
        + json.dumps({**base, "mix": "mixed", "v61_backend": "numba"})
        + "\n"
    )
    assert pw.done_mixes(out, sha, "none", 52) == {"frozen", "mixed"}


@needs_artifacts
def test_dev_probe_cli_records_the_backend(tmp_path):
    out = tmp_path / "p.jsonl"
    argv = [
        sys.executable,
        str(REPO / "research/redesign_m2_20261008/dev_probe.py"),
        *("--student", str(STUDENT), "--arm", "student", "--mix", "mixed", "--worlds", "1"),
        *("--frames", "60", "--ego2s-forward", "rowwise", "--ego2s-mask", "advisory"),
        *("--round-index", "52", "--v61-backend", "numba", "--out", str(out)),
    ]
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "SNAKE_DQN_DEVICE": "cpu"}
    r = subprocess.run(
        argv, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr[-2000:]
    row = json.loads(out.read_text().splitlines()[-1])
    assert row["v61_backend"] == "numba" and row["ego2s_forward"] == "rowwise"
    assert len(row["mass"]) == 1 and np.isfinite(list(row["mass"].values())[0])
