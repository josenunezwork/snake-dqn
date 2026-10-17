"""Bounded main_sync full-state integration tests with a synthetic environment."""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as trainer

torch = pytest.importorskip("torch")


class _ToyNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = torch.nn.Parameter(torch.arange(6, dtype=torch.float32).reshape(1, 6))
        self.scale = torch.nn.Parameter(torch.tensor(0.25))

    def forward(self, local, global_, scalars):
        del local, global_
        return self.bias.expand(scalars.shape[0], -1) + self.scale * scalars[:, :1]


class _ToyRuntime:
    def prepare(self, sim):
        del sim

    def observe_step(self, sim):
        del sim


class _ToySim:
    def __init__(self, envs=2):
        self.E = envs
        self.S = 1
        self.alive = np.ones((envs, 1), dtype=bool)
        self.length = np.full((envs, 1), 5, dtype=np.int64)
        self.done = np.zeros((envs, 1), dtype=bool)
        self.frame = 0

    def get_resolved_action_mask(self):
        return np.ones((self.E, self.S, 6), dtype=bool)

    def get_alive(self):
        return self.alive

    def get_reward(self):
        return np.zeros((self.E, self.S), dtype=np.float64)

    def get_done(self):
        return self.done

    def get_lengths(self):
        return self.length

    def step_with_policy(self, choose):
        choose(self)
        self.frame += 1


def _raster(_sim, _config, *, rows=None, **_kwargs):
    count = len(rows) if rows is not None else 2
    return {
        "local": np.zeros((count, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((count, 4, 37, 37), dtype=np.uint8),
        "scalars": np.ones((count, 30), dtype=np.float32),
    }


def _write_demo_fixture(root: Path) -> None:
    data = root / "round-12"
    data.mkdir(parents=True)
    n = 512
    world_seed = np.concatenate([np.zeros(n // 2, dtype=np.int64), np.ones(n // 2, dtype=np.int64)])
    np.savez(
        data / "chunk-000-part-000.npz",
        world_seed=world_seed,
        local=np.zeros((n, 7, 31, 31), dtype=np.uint8),
        global_=np.zeros((n, 4, 37, 37), dtype=np.uint8),
        scalars=np.zeros((n, 30), dtype=np.float32),
        q_teacher=np.zeros((n, 6), dtype=np.float32),
        a_v8=np.zeros(n, dtype=np.int64),
        mask_resolved=np.ones((n, 6), dtype=bool),
    )
    # DemoStream expects the field name ``global``; np.savez cannot use it as a
    # keyword argument, so rewrite the archive with the exact key.
    with np.load(data / "chunk-000-part-000.npz") as archive:
        fields = {key: archive[key] for key in archive.files if key != "global_"}
        fields["global"] = archive["global_"]
    np.savez(data / "chunk-000-part-000.npz", **fields)
    (root / "margin.json").write_text(json.dumps({"margin": 0.3}))


def _args(tmp_path, data, *, transitions=512, replay_mode="full", resume=None):
    args = trainer.default_args()
    args.seed = 7
    args.init = str(tmp_path / "init.pth")
    args.demo_data = data
    args.demo_rounds = "12"
    args.transitions = transitions
    args.worlds = 2
    args.segment_frames = 300
    args.replay = 512
    args.demo_capacity = 256
    args.demo_initial_shards = 1
    args.warmup = 1
    args.checkpoint_every = 1_000_000
    args.device = "cpu"
    args.no_guard = True
    args.full_state_every = 256
    args.full_state_replay = replay_mode
    args.ckpt_local_dir = tmp_path / "snapshots"
    args.ckpt_durable_dir = None
    args.resume_full_state = resume
    args.actors = 0
    args.frozen_frac = 0.0
    args.mixed_frac = 0.0
    args.obs_version = "b"
    args.demo_initial_shards = 1
    args.out = tmp_path / "out"
    args.init_path = None
    return args


def _install_toy_runtime(monkeypatch):
    def fake_load(_path, _device):
        return _ToyNet()

    def fake_build_segment(args, _ctx, _segment, _gamma):
        return _ToySim(args.worlds), _ToyRuntime(), {}, {}

    def fake_save(path, net, meta, _manifest):
        torch.save({"state_dict": copy.deepcopy(net.state_dict()), "meta": meta}, path)

    monkeypatch.setattr("src.model.ego2s_network.load_ego2s_checkpoint", fake_load)
    monkeypatch.setattr(trainer, "build_segment", fake_build_segment)
    monkeypatch.setattr("src.simd_env.ego_raster.build_ego_raster", _raster)
    monkeypatch.setattr(trainer, "save_checkpoint_atomic", fake_save)


def _snapshot_at(root: Path, transitions: int) -> Path:
    from research.redesign_m3c_20261008.async_checkpoint import discover_snapshots

    rows = [row for row in discover_snapshots(root) if row["transitions"] == transitions]
    assert len(rows) == 1
    return root / rows[0]["snapshot"]


def _state_equal(left, right):
    if isinstance(left, torch.Tensor):
        assert isinstance(right, torch.Tensor)
        assert torch.equal(left, right)
    elif isinstance(left, np.ndarray):
        np.testing.assert_array_equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _state_equal(left[key], right[key])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _state_equal(a, b)
    elif isinstance(left, tuple):
        assert isinstance(right, tuple) and len(left) == len(right)
        for a, b in zip(left, right):
            _state_equal(a, b)
    elif hasattr(left, "__class__") and left.__class__.__name__ == "deque":
        assert list(left) == list(right)
    else:
        assert left == right


def test_main_sync_full_state_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    _install_toy_runtime(monkeypatch)
    data = tmp_path / "demo"
    _write_demo_fixture(data)
    init = tmp_path / "init.pth"
    init.write_bytes(b"synthetic-init")

    baseline_root = tmp_path / "baseline"
    baseline_args = _args(baseline_root, data)
    baseline_args.init = str(init)
    assert trainer.main_sync(baseline_args) == 0
    summary = json.loads((baseline_args.out / "summary.json").read_text())
    assert summary["transitions"] == 512
    assert summary["updates"] > 0
    assert summary["full_state_complete"] is True

    resume_source = tmp_path / "resume-source"
    resume_source.mkdir()
    shutil.copytree(
        _snapshot_at(baseline_args.ckpt_local_dir, 256),
        resume_source / _snapshot_at(baseline_args.ckpt_local_dir, 256).name,
    )
    resumed_root = tmp_path / "resumed"
    resumed_args = _args(resumed_root, data, resume=resume_source)
    resumed_args.init = str(init)
    assert trainer.main_sync(resumed_args) == 0
    resumed_summary = json.loads((resumed_args.out / "summary.json").read_text())
    assert resumed_summary["transitions"] == summary["transitions"]
    assert resumed_summary["updates"] == summary["updates"]
    assert resumed_summary["resumed_with_full_replay"] is True
    assert any(
        event.get("event") == "resume-full-state" for event in resumed_summary["full_state_events"]
    )

    baseline_final = torch.load(baseline_args.out / "final.pth", weights_only=False)
    resumed_final = torch.load(resumed_args.out / "final.pth", weights_only=False)
    _state_equal(baseline_final["state_dict"], resumed_final["state_dict"])

    from research.redesign_m3c_20261008.async_checkpoint import load_latest_snapshot

    baseline_state, _ = load_latest_snapshot(baseline_args.ckpt_local_dir)
    resumed_state, _ = load_latest_snapshot(resumed_args.ckpt_local_dir)
    for key in ("online", "target", "optimizer", "replay", "rings", "rng"):
        _state_equal(baseline_state[key], resumed_state[key])
    _state_equal(
        {
            key: value
            for key, value in baseline_state["counters"].items()
            if key not in {"learn_s", "env_s"}
        },
        {
            key: value
            for key, value in resumed_state["counters"].items()
            if key not in {"learn_s", "env_s"}
        },
    )


def test_main_sync_resume_without_replay_warms_up_and_marks_summary(tmp_path, monkeypatch):
    _install_toy_runtime(monkeypatch)
    data = tmp_path / "demo"
    _write_demo_fixture(data)
    init = tmp_path / "init.pth"
    init.write_bytes(b"synthetic-init")

    source_root = tmp_path / "source"
    source_args = _args(source_root, data, transitions=256, replay_mode="none")
    source_args.init = str(init)
    assert trainer.main_sync(source_args) == 0
    source_snapshot = _snapshot_at(source_args.ckpt_local_dir, 256)
    resume_root = tmp_path / "resume"
    resume_source = tmp_path / "resume-source"
    resume_source.mkdir()
    shutil.copytree(source_snapshot, resume_source / source_snapshot.name)

    resumed_args = _args(
        resume_root, data, transitions=512, replay_mode="none", resume=resume_source
    )
    resumed_args.init = str(init)
    assert trainer.main_sync(resumed_args) == 0
    summary = json.loads((resumed_args.out / "summary.json").read_text())
    assert summary["resumed_without_replay"] is True
    assert summary["resumed_with_full_replay"] is False
    assert summary["transitions"] == 512
    assert any(
        event.get("mode") == "resumed-without-replay"
        for event in summary["full_state_events"]
        if event.get("event") == "resume-full-state"
    )
