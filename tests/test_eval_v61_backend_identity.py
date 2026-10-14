"""Eval records are identical with the compiled vector61 featurizer and the compiled scripted
anchors (perf-eval, item 8).

``run_simd_eval(vector61_backend="numba")`` (and ``scripted_anchors="fast"``) against the
default Python featurizer (and Python anchors), through
``dev_probe.probe`` (the engine, profile, distillation-namespace worlds and strict rosters of
the probes ``probe_watch.py`` launches: grid engine, vector61 rowwise forwards, ego2s rowwise
hero forward, advisory mask for the no-veto student). Each run has the per-frame whole-world
digest recorder of ``grid_h5000_identity`` (every snake's body, heading, masks, events, food,
corpses and env RNG state after every frame) and a per-frame action recorder (every slot's
action) installed. Arms: the no-veto advisory student (M3-C / M4 primary probe), the student
inside the v8 veto and the v8-vetoed incumbent frp3-s12; every mix.

Negative controls: the variant must really call the compiled featurizer (never falling back to
the NumPy one) / the compiled anchors (grouped rows), and a deliberately perturbed compiled
featurizer or anchor must be caught.
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


class AnchorSpy:
    """Counts compiled scripted-anchor calls / rows in a run (optionally perturbing them)."""

    def __init__(self, monkeypatch, perturb=None):
        from src.simd_env.fast_anchors import FastProfileAnchorSimdPolicy

        self.calls = 0
        self.rows = 0
        orig = FastProfileAnchorSimdPolicy.actions
        spy = self

        def actions(policy, masks, sim, slots):
            out = orig(policy, masks, sim, slots)
            spy.calls += 1
            spy.rows += len(out)
            return perturb(out) if perturb is not None else out

        monkeypatch.setattr(FastProfileAnchorSimdPolicy, "actions", actions)


def _pair(
    ctx, arm, mix, worlds, frames, monkeypatch, perturb=None, variant="numba", anchor_perturb=None
):
    from research.perf_eval_20261013 import eval_speed as es

    ns = argparse.Namespace(
        mix=mix, worlds=worlds, frames=frames, round_index=52, student=str(STUDENT)
    )
    spy = Spy(monkeypatch, perturb)
    spy.anchors = AnchorSpy(monkeypatch, anchor_perturb)
    ref = es.run_recorded(es.probe_args(arm, "default", ns), ctx)
    # the reference never touches the compiled featurizer or the compiled anchors
    assert spy.calls == 0 and spy.anchors.calls == 0
    var = es.run_recorded(es.probe_args(arm, variant, ns), ctx)
    seeds = es.dev_probe.probe_seeds(worlds, 52)
    return es.compare(ref, var, seeds), ref, var, spy


def _assert_identical(rows, ref, var, spy, expect_vector61_rows=True, variant="numba"):
    from research.perf_eval_20261013 import eval_speed as es

    bad = [r for r in rows if not r["identical"]]
    assert not bad, bad[:2]
    frames = len(ref["frame_digests"][0])
    assert frames > 0 and all(r["frames_compared"] == frames for r in rows)
    assert all(r["actions_compared"] == frames for r in rows)  # every frame's action row
    assert all(r["first_differing_action_frame"] is None for r in rows)
    assert ref["row"]["mass"] == var["row"]["mass"]
    assert ref["row"]["deaths"] == var["row"]["deaths"]
    assert ref["row"]["causes"] == var["row"]["causes"]
    v = es.VARIANTS[variant]
    assert "v61_backend" not in ref["row"] and "anchors" not in ref["row"]
    assert var["row"].get("v61_backend", "python") == v["v61_backend"]
    assert var["row"].get("anchors", "python") == v["anchors"]
    assert spy.fallbacks == 0
    if expect_vector61_rows and v["v61_backend"] == "numba":
        assert spy.calls > 0 and spy.rows > 0
    if v["anchors"] == "fast" and ref["row"]["mix"] != "frozen":  # frozen has no anchor slot
        assert spy.anchors.calls > 0 and spy.anchors.rows > spy.anchors.calls  # grouped rows


@needs_artifacts
def test_numba_backend_records_identical_quick(ctx, monkeypatch):
    from src.simd_env import eval_engine as ee

    sim_class = ee._terminal_hero_sim_class("grid")
    own_before = "step_with_policy" in vars(sim_class)
    rows, ref, var, spy = _pair(ctx, "student-nv", "frozen", 2, 300, monkeypatch)
    _assert_identical(rows, ref, var, spy)
    assert ("step_with_policy" in vars(sim_class)) == own_before  # recorders removed


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
    assert any(r["first_differing_action_frame"] is not None for r in rows)


def _fake_run(records, digests, actions):
    return {"records": records, "frame_digests": digests, "frame_actions": actions}


def test_compare_checks_actions_and_ignores_only_provenance():
    """``compare``: an action-only difference fails; only the named provenance path is
    ignored. (Synthetic runs: a deliberate break of either check fails this test.)"""
    from research.perf_eval_20261013 import eval_speed as es

    rec = {"seed": 1, "mass_integral": 10.0, "ego2s_hero": {"forward": "rowwise"}}
    acts = [b"\x01\x01", b"\x02\x01", b"\x00\x01"]
    ref = _fake_run([rec], [["d0", "d1", "d2"]], [acts])

    same = es.compare(ref, _fake_run([dict(rec)], [["d0", "d1", "d2"]], [list(acts)]), [1])
    assert same[0]["identical"] and same[0]["actions_compared"] == 3

    flipped = [acts[0], b"\x05\x01", acts[2]]  # same world digests, one hero action differs
    rows = es.compare(ref, _fake_run([dict(rec)], [["d0", "d1", "d2"]], [flipped]), [1])
    assert not rows[0]["identical"] and rows[0]["record_identical"]
    assert rows[0]["first_differing_action_frame"] == 1
    assert rows[0]["differing_action_frames"] == 1

    batched = dict(rec, ego2s_hero={"forward": "batched"})
    var = _fake_run([batched], [["d0", "d1", "d2"]], [list(acts)])
    assert not es.compare(ref, var, [1])[0]["identical"]  # without the provenance rule
    rows = es.compare(ref, var, [1], es.PROVENANCE_PATHS["batched"])
    assert rows[0]["identical"] and rows[0]["ignored_provenance"][0]["variant"] == "batched"
    worse = dict(batched, mass_integral=10.5)  # a real difference is never ignored
    rows = es.compare(ref, _fake_run([worse], [["d0", "d1", "d2"]], [acts]), [1], ("x",))
    assert not rows[0]["identical"] and not rows[0]["record_identical"]
    rows = es.compare(ref, _fake_run([dict(rec)], [["d0", "dX", "d2"]], [acts]), [1])
    assert not rows[0]["identical"] and rows[0]["first_differing_frame"] == 1


def test_profile_split_is_exclusive_and_sums_to_wall():
    from research.perf_eval_20261013 import eval_speed as es

    nv = {
        "step_with_policy": 8.0,
        "v61_prepare": 0.5,
        "v61_observe": 3.0,
        "dispatch": 7.0,
        "v61_actions": 2.0,
        "v61_q_apex": 1.5,
        "ego2s_actions": 4.5,
        "ego2s_raster": 1.0,
        "ego2s_forward": 3.0,
        "scripted_actions": 0.2,
    }
    v8 = {**nv, "ego2s_actions": 0.0, "v61_actions": 6.5, "veto": 0.6, "v8_free_space": 0.4}
    for seconds in (nv, v8):
        split = es.profile_split(seconds, 12.0)
        assert sum(split.values()) == pytest.approx(12.0)
        assert all(v >= -1e-9 for v in split.values()), split
    assert es.profile_split(v8, 12.0)["v61_policy_other"] == pytest.approx(0.4)


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


@needs_artifacts
@pytest.mark.parametrize("mix", ["scripted", "mixed"])  # greedy_food / random_safe anchors
def test_fast_anchors_records_identical_quick(ctx, monkeypatch, mix):
    variant = "numba+fast-anchors"
    rows, ref, var, spy = _pair(ctx, "student-nv", mix, 2, 300, monkeypatch, variant=variant)
    _assert_identical(rows, ref, var, spy, expect_vector61_rows=mix != "scripted", variant=variant)


@needs_artifacts
def test_identity_check_catches_a_broken_anchor(ctx, monkeypatch):
    """Negative control: compiled anchors that turn instead of going straight are detected."""

    def perturb(out):
        return np.where(out == 1, 0, out)

    rows, ref, var, spy = _pair(
        ctx, "student-nv", "scripted", 2, 300, monkeypatch, None, "fast-anchors", perturb
    )
    assert spy.anchors.calls > 0
    assert any(not r["identical"] for r in rows)
    assert any(r["first_differing_action_frame"] is not None for r in rows)


@pytest.mark.slow
@needs_artifacts
@pytest.mark.parametrize("arm", ["student-nv", "student-v8", "incumbent-v8"])
@pytest.mark.parametrize("mix", ["scripted", "mixed"])
def test_fast_anchors_records_identical_every_arm_and_mix(ctx, monkeypatch, arm, mix):
    variant = "numba+fast-anchors"
    rows, ref, var, spy = _pair(ctx, arm, mix, 3, 600, monkeypatch, variant=variant)
    _assert_identical(
        rows,
        ref,
        var,
        spy,
        expect_vector61_rows=(arm, mix) != ("student-nv", "scripted"),
        variant=variant,
    )


def test_run_simd_eval_validates_backend():
    from src.simd_env import eval_engine as ee

    with pytest.raises(ValueError, match="vector61_backend needs vector61=True"):
        ee.run_simd_eval(("scripted", "random_safe"), [], 1, [0], vector61_backend="numba")
    with pytest.raises(ValueError, match="scripted_anchors must be"):
        ee.run_simd_eval(("scripted", "random_safe"), [], 1, [0], scripted_anchors="numba")
    with pytest.raises(ValueError, match="needs a profile and sim_engine='grid'"):
        ee.run_simd_eval(("scripted", "random_safe"), [], 1, [0], scripted_anchors="fast")


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
    assert args.anchors == "python"


def test_probe_watch_passes_the_backend_only_when_asked(tmp_path):
    from research.redesign_m3c_20261008 import probe_watch as pw

    job = ("A_s0", "final", "/x/final.pth", "none", "frozen")
    default = pw.probe_argv(job, tmp_path, 16, 52, slots=False)
    assert "--v61-backend" not in default
    numba = pw.probe_argv(job, tmp_path, 16, 52, slots=False, v61_backend="numba")
    assert numba[: len(default)] == default and numba[len(default) :] == ["--v61-backend", "numba"]
    slots = pw.probe_argv(job, tmp_path, 16, 52, slots=True, v61_backend="numba")
    assert slots[1].endswith("slotrun.py") and slots[-2:] == ["--v61-backend", "numba"]
    assert "--anchors" not in default
    both = pw.probe_argv(job, tmp_path, 16, 52, False, v61_backend="numba", anchors="fast")
    assert both[len(default) :] == ["--v61-backend", "numba", "--anchors", "fast"]


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
        *("--round-index", "52", "--v61-backend", "numba", "--anchors", "fast"),
        *("--out", str(out)),
    ]
    env = {"OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "SNAKE_DQN_DEVICE": "cpu"}
    r = subprocess.run(
        argv, cwd=REPO, env={**os.environ, **env}, capture_output=True, text=True, timeout=600
    )
    assert r.returncode == 0, r.stderr[-2000:]
    row = json.loads(out.read_text().splitlines()[-1])
    assert row["v61_backend"] == "numba" and row["ego2s_forward"] == "rowwise"
    assert row["anchors"] == "fast"
    assert len(row["mass"]) == 1 and np.isfinite(list(row["mass"].values())[0])


def test_compare_records_ignores_only_run_fields(tmp_path):
    """Phase R record dirs: run fields (binding, commit, wall, ...) are not compared; any
    episode field is; a record present on one side only fails the comparison."""
    from research.perf_eval_20261013 import eval_speed as es

    entry = {
        "binding": {"plan_file_sha256": "a"},
        "commit": "c1",
        "unit_wall_seconds": 10.0,
        "hero": "incumbent",
        "record": {"mass_integral": 12.5, "deaths": 0},
        "prefix_h5000": {"mass_integral": 6.0},
    }
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir()
    b.mkdir()
    (a / "r1.json").write_text(json.dumps(entry))
    other_run = dict(entry, binding={"plan_file_sha256": "b"}, commit="c2", unit_wall_seconds=5.0)
    (b / "r1.json").write_text(json.dumps(other_run))
    assert es.compare_record_dirs(a, b)["identical"] == 1
    assert es.main(["compare-records", "--a", str(a), "--b", str(b)]) == 0
    changed = dict(other_run, record=dict(entry["record"], mass_integral=12.75))
    (b / "r1.json").write_text(json.dumps(changed))
    result = es.compare_record_dirs(a, b)
    assert result["identical"] == 0 and result["rows"][0]["differences"]
    assert es.main(["compare-records", "--a", str(a), "--b", str(b)]) == 1
    (b / "r1.json").write_text(json.dumps(other_run))
    (a / "r2.json").write_text(json.dumps(entry))
    assert es.compare_record_dirs(a, b)["only_a"] == ["r2.json"]
    assert es.main(["compare-records", "--a", str(a), "--b", str(b)]) == 1
