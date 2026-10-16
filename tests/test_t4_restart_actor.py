"""Focused no-learning checks for T4 actor restart bookkeeping."""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T
from research.t4_snapshot_restart_20261016 import snapshots as S
from research.t4_snapshot_restart_20261016.actor import RestartActorTelemetry

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_v8_nb import HAVE_CKPT  # noqa: E402


def _init(tmp_path):
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    torch.manual_seed(0)
    path = tmp_path / "init.pth"
    save_ego2s_checkpoint(path, Ego2sNet(7, 30))
    return path


class _Stop:
    def __init__(self):
        self.flag = False

    def is_set(self):
        return self.flag

    def set(self):
        self.flag = True


class _Sink:
    def __init__(self, count, stop):
        self.count, self.stop, self.msgs = count, stop, []

    def put(self, msg, timeout=0.0):
        self.msgs.append(msg)
        if len(self.msgs) >= self.count:
            self.stop.set()


def _run_actor(mod, tmp_path, init, frames, extra=(), worlds=5):
    tmp_path.mkdir(parents=True, exist_ok=True)
    args = mod.build_parser().parse_args(
        [
            "--seed",
            "5",
            "--init",
            str(init),
            "--demo-data",
            ".",
            "--out",
            str(tmp_path / "x"),
            "--actors",
            "1",
            "--worlds",
            str(worlds),
            "--frozen-frac",
            "0.2",
            "--segment-frames",
            "45",
            "--device",
            "cpu",
            "--v61-backend",
            "numba",
            *extra,
        ]
    )
    nscal, margin = mod.nscal_for(args), worlds * 16
    cap = worlds * (frames + 4) + margin
    net = mod.load_init(args, "cpu")
    n_weights = sum(n for _, _, n in mod._weights_spec(net))
    blocks = [
        mod.Block.create(mod.ring_nbytes(cap, nscal), str(tmp_path), "ring"),
        mod.Block.create(8 + 8 * (n_weights + 1), str(tmp_path), "weights"),
        mod.Block.create(32, str(tmp_path), "progress"),
    ]
    try:
        names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
        wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
        wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
        progress = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
        progress[:] = 0
        mod._publish_weights(net, wbuf, wver)
        cfg = {
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "K": 1,
            "ring_cap": cap,
            "ring_margin": margin,
            "n_weights": n_weights,
            "actor_device": "cpu",
        }
        stop, sink = _Stop(), _Sink(frames, None)
        sink.stop = stop
        mod.actor_main(0, cfg, names, sink, stop)
        ring = mod.SharedObsRing(blocks[0], cap, margin, nscal)
        out = {key: value.copy() for key, value in ring.arrays.items()}
        del ring
        return sink.msgs, out
    finally:
        for block in blocks:
            block.unlink()
            block.close()


def _baseline(tmp_path, monkeypatch):
    # T4f's length-stratified boost telemetry is part of the frozen pre-T6
    # contract, so 865ae05 is no longer a valid default-path reference.
    rev = "f3375b26d410c4de0feb2cd049c6021171902e53"
    source = subprocess.check_output(
        ["git", "show", f"{rev}:research/redesign_m3c_20261008/train_m3c.py"], text=True
    )
    path = tmp_path / "baseline.py"
    path.write_text(source)
    name = "t4_pre_restart_baseline"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


def test_restart_actor_telemetry_splits_decisions_deaths_and_lengths():
    telemetry = RestartActorTelemetry()
    telemetry.record(
        worlds=np.array([0, 2, 3]),
        lengths=np.array([10, 20, 30]),
        died=np.array([False, True, True]),
        restarted=np.array([True, False, True, False]),
    )
    assert telemetry.take() == {
        "restart_decisions": 2,
        "fresh_decisions": 1,
        "restart_decisions_ge800": 0,
        "fresh_decisions_ge800": 0,
        "restart_deaths": 1,
        "fresh_deaths": 1,
        "restart_deaths_ge800": 0,
        "fresh_deaths_ge800": 0,
        "restart_length_sum": 30,
        "fresh_length_sum": 30,
        "restart_death_length_sum": 20,
        "fresh_death_length_sum": 30,
    }
    assert all(value == 0 for value in telemetry.take().values())


def test_restart_actor_telemetry_counts_ge800_at_decision_time():
    telemetry = RestartActorTelemetry()
    telemetry.record(
        worlds=np.array([0, 1, 2, 3]),
        lengths=np.array([799, 800, 801, 900]),
        died=np.array([True, True, False, True]),
        restarted=np.array([True, True, False, False]),
    )
    values = telemetry.take()
    assert values["restart_decisions_ge800"] == 1
    assert values["restart_deaths_ge800"] == 1
    assert values["fresh_decisions_ge800"] == 2
    assert values["fresh_deaths_ge800"] == 1


def test_ge800_death_rate_is_null_without_decisions():
    assert T.per_1k_or_none(0, 0) is None
    assert T.per_1k_or_none(2, 800) == 2.5


def test_nstep_boundary_flushes_only_expired_world():
    b = T.NStepBuilder(2, n_step=3)
    b.step(np.array([0, 1]), [10, 20], [1, 2], np.array([1.0, 2.0]), np.zeros(2, bool), 0.9)
    b.step(np.array([0, 1]), [11, 21], [1, 2], np.array([3.0, 4.0]), np.zeros(2, bool), 0.9)
    b.end_worlds(np.array([0]), 0.9)
    s, a, reward, discount, nxt = b.take()
    assert s.tolist() == [10]
    assert a.tolist() == [1]
    assert reward.tolist() == [1.0]
    assert np.array_equal(discount.view(np.uint32), np.array([0.9], np.float32).view(np.uint32))
    assert nxt.tolist() == [11]
    assert b.cnt.tolist() == [0, 2]
    # A final lone decision has no successor and therefore emits no row, but
    # its world must still be cleared before a later segment can use it.
    b.end_worlds(np.array([1]), 0.9)
    assert b.cnt.tolist() == [0, 0]


def test_restart_flags_are_default_off_and_fraction_is_parsed():
    base = T.build_parser().parse_args(
        ["--seed", "1", "--init", "init", "--demo-data", ".", "--out", "x"]
    )
    assert base.restart_bank is None
    assert base.restart_frac == 0.0
    enabled = T.build_parser().parse_args(
        [
            "--seed",
            "1",
            "--init",
            "init",
            "--demo-data",
            ".",
            "--out",
            "x",
            "--actors",
            "1",
            "--restart-bank",
            "bank",
            "--restart-frac",
            "0.2",
        ]
    )
    assert enabled.restart_frac == 0.2
    assert str(enabled.restart_bank) == "bank"


def _snapshot_bank(tmp_path, args):
    """Make one late, living boundary using the trainer's ordinary world builder."""
    ctx = T.gid._context(1)
    source_args = T.build_parser().parse_args(
        ["--seed", "5", "--init", "x", "--demo-data", ".", "--out", "x", "--worlds", "1"]
    )
    sim, runtime, policies, groups = T.build_segment(source_args, ctx, 0, 0.99)
    specs = [None] * (sim.S - 1)
    for spec, cells in groups.items():
        for _world, slot in cells:
            specs[int(slot) - 1] = spec
    for _ in range(30):

        def choose(prepared):
            alive, masks = prepared.get_alive(), prepared.get_resolved_action_mask()
            actions = np.ones((prepared.E, prepared.S), dtype=np.int64)
            runtime.prepare(prepared)
            hero_mask = masks[0, 0]
            actions[0, 0] = np.flatnonzero(hero_mask)[0] if hero_mask.any() else 1
            for spec, cells in groups.items():
                live = cells[alive[cells[:, 0], cells[:, 1]]]
                if len(live):
                    actions[live[:, 0], live[:, 1]] = policies[spec].actions(
                        masks[live[:, 0], live[:, 1]], prepared, live
                    )
            return actions

        sim.step_with_policy(choose)
        runtime.observe_step(sim)
    assert sim.alive[0, 0]
    snap = S.capture(sim, runtime, specs)
    entry = S.save_snapshot(tmp_path / "bank" / "late.npz", snap)
    (tmp_path / "bank" / "manifest.json").write_text(
        json.dumps({"schema": S.SCHEMA, "snapshots": [entry]})
    )
    return tmp_path / "bank"


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
@pytest.mark.parametrize("fraction", [0.2, 1.0])
def test_restart_actor_cpu_smoke_retires_snapshot_worlds_at_their_clock_boundary(
    tmp_path, fraction
):
    init = _init(tmp_path)
    bank = _snapshot_bank(tmp_path, None)
    messages, _ring = _run_actor(
        T,
        tmp_path / "run",
        init,
        60,
        ("--restart-bank", str(bank), "--restart-frac", str(fraction)),
    )
    assert len(messages) == 60
    assert all(msg["restart_frac_actual"] == fraction for msg in messages)
    if fraction == 1.0:
        # All worlds retire after 15 decisions; four whole segments fit in 60 messages.
        assert [msg["restart_decisions"] for msg in messages] == [5] * 60
        assert not any(msg["fresh_decisions"] for msg in messages)
        return
    # The lone restored world has source frame 30 and acts only through frame 45;
    # fresh worlds run the whole first segment and the second begins at message 45.
    assert [msg["restart_decisions"] for msg in messages[:15]] == [1] * 15
    assert [msg["restart_decisions"] for msg in messages[15:45]] == [0] * 30
    assert [msg["restart_decisions"] for msg in messages[45:]] == [1] * 15
    assert sum(msg["restart_decisions"] for msg in messages) == 30
    assert sum(msg["fresh_decisions"] for msg in messages) == 240
    assert all("restart_length_sum" in msg and "fresh_length_sum" in msg for msg in messages)
    assert all(
        "restart_decisions_ge800" in msg and "fresh_decisions_ge800" in msg for msg in messages
    )
    assert all("restart_deaths_ge800" in msg and "fresh_deaths_ge800" in msg for msg in messages)


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
def test_restart_default_actor_trace_matches_frozen_pre_t6_baseline(tmp_path, monkeypatch):
    baseline = _baseline(tmp_path, monkeypatch)
    init = _init(tmp_path)
    ref_messages, ref_ring = _run_actor(baseline, tmp_path / "base", init, 50)
    got_messages, got_ring = _run_actor(T, tmp_path / "current", init, 50)
    assert len(ref_messages) == len(got_messages)
    for index, (ref, got) in enumerate(zip(ref_messages, got_messages)):
        assert ref.keys() == got.keys(), index
        for key in ref.keys() - {"env_s", "timing", "j"}:
            if isinstance(ref[key], np.ndarray):
                assert ref[key].dtype == got[key].dtype
                assert ref[key].shape == got[key].shape
                assert ref[key].tobytes() == got[key].tobytes(), (index, key)
            else:
                assert ref[key] == got[key], (index, key)
    for key in ref_ring:
        assert ref_ring[key].dtype == got_ring[key].dtype
        assert ref_ring[key].tobytes() == got_ring[key].tobytes(), key


@pytest.mark.skipif(not HAVE_CKPT, reason="needs the pinned frp3-s12 checkpoints")
@pytest.mark.parametrize("mix", ["scripted", "frozen", "mixed"])
def test_real_decisive_bank_snapshot_enters_actor_td_pipeline(tmp_path, mix):
    source = Path(
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts/"
        "t4b-snapshot-restart-20261016/benchmark-1000-v2"
    )
    if not (source / "manifest.json").exists():
        pytest.skip("needs the measured T4b decisive snapshot store")
    manifest = json.loads((source / "manifest.json").read_text())
    entry = next(
        entry
        for entry in manifest["snapshots"]
        if S.load_snapshot(source / entry["file"])["provenance"]["roster"]["mix"] == mix
    )
    bank = tmp_path / "bank"
    bank.mkdir()
    (bank / entry["file"]).write_bytes((source / entry["file"]).read_bytes())
    (bank / "manifest.json").write_text(
        json.dumps(
            {"schema": S.SCHEMA, "snapshots": [entry], "checkpoints": manifest["checkpoints"]}
        )
    )
    messages, _ = _run_actor(
        T,
        tmp_path / "run",
        _init(tmp_path),
        8,
        ("--restart-bank", str(bank), "--restart-frac", "0.2", "--segment-frames", "5000"),
    )
    assert messages[0]["restart_decisions"] == 1
    assert messages[0]["fresh_decisions"] == 4
    assert all(m["decisions"] == m["restart_decisions"] + m["fresh_decisions"] for m in messages)
    assert any(len(m["s"]) for m in messages)
