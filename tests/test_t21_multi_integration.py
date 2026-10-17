"""Bounded orchestration tests for the T21 multi-actor terminal checkpoint path."""

from __future__ import annotations

import json
import queue
import threading
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T

torch = pytest.importorskip("torch")


def _write_demo(root: Path) -> None:
    round_ = root / "round-12"
    round_.mkdir(parents=True)
    n = 512
    fields = {
        "world_seed": np.arange(n, dtype=np.int64) % 10 + 1,
        "local": np.zeros((n, 7, 31, 31), dtype=np.uint8),
        "global": np.zeros((n, 4, 37, 37), dtype=np.uint8),
        "scalars": np.zeros((n, 30), dtype=np.float32),
        "q_teacher": np.zeros((n, 6), dtype=np.float32),
        "a_v8": np.zeros(n, dtype=np.int64),
        "mask_resolved": np.ones((n, 6), dtype=bool),
    }
    np.savez(round_ / "chunk-000-part-000.npz", **fields)
    (root / "margin.json").write_text(json.dumps({"margin": 0.3}))


class _FakeProcess:
    """Thread-backed actor that speaks just the learner's frame/marker protocol."""

    def __init__(self, *, target, args, daemon=False):
        self.target, self.args, self.daemon = target, args, daemon
        self.thread = None
        self.exitcode = None

    def start(self):
        def run():
            j, _cfg, _names, tx, stop, control = self.args
            zeros_i = np.zeros(0, dtype=np.int64)
            zeros_f = np.zeros(0, dtype=np.float32)
            while not stop.is_set():
                try:
                    request = control.get_nowait() if control is not None else None
                except queue.Empty:
                    request = None
                if request and request.get("op") == "snapshot":
                    tx.put(
                        {"kind": "snapshot", "j": j, "id": request["id"], "state": {"segment": j}}
                    )
                    while not stop.is_set():
                        try:
                            release = control.get(timeout=0.05)
                        except queue.Empty:
                            continue
                        if release.get("op") == "release" and release.get("id") == request["id"]:
                            break
                    continue
                tx.put(
                    {
                        "j": j,
                        "s": zeros_i,
                        "a": zeros_i,
                        "R": zeros_f,
                        "disc": zeros_f,
                        "nxt": zeros_i,
                        "acts": np.array([1], dtype=np.int8),
                        "lab_ids": zeros_i,
                        "lab_a": np.zeros(0, dtype=np.int8),
                        "deaths": 0,
                        "env_s": 0.0,
                        "timing": {
                            "featurizer_s": 0.0,
                            "forward_s": 0.0,
                            "opponents_s": 0.0,
                            "v8_s": 0.0,
                        },
                        "weights_version": 2,
                        "decisions": 1,
                        "boost": 0,
                        "greedy_boost": 0,
                        "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
                        "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
                        "v8_override": 0,
                    },
                    timeout=0.1,
                )
            self.exitcode = 0

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def is_alive(self):
        return self.thread is not None and self.thread.is_alive()

    def join(self, timeout=None):
        self.thread.join(timeout)

    def terminate(self):
        self.args[4].set()
        self.join(1)


class _Context:
    def Event(self):
        return threading.Event()

    def Queue(self, maxsize=0):
        return queue.Queue(maxsize=maxsize)

    def Process(self, **kwargs):
        return _FakeProcess(**kwargs)


def _args(tmp_path: Path, data: Path, init: Path, *, wall: float):
    args = T.default_args()
    args.seed = 17
    args.init = str(init)
    args.demo_data = data
    args.demo_rounds = "12"
    args.out = tmp_path / "out"
    args.actors = 1
    args.worlds = 1
    args.transitions = 1
    args.warmup = 1_000_000
    args.replay = 32
    args.demo_capacity = 16
    args.demo_initial_shards = 1
    args.device = args.actor_device = "cpu"
    args.no_guard = True
    args.full_state_every = 1
    args.full_state_replay = "none"
    args.ckpt_local_dir = tmp_path / "snapshots"
    args.ckpt_durable_dir = None
    args.max_wall_seconds = wall
    args.actor_queue_frames = 1
    args.checkpoint_every = 1_000_000
    return args


@pytest.mark.parametrize("wall", [0.0, 1e-9], ids=["terminal-cadence", "max-wall-terminal"])
def test_main_multi_terminal_snapshot_barrier_finishes_with_fake_actor(tmp_path, monkeypatch, wall):
    """Terminal cadence=1 and wall stop both drain a marker then release its actor."""
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    data = tmp_path / "demo"
    _write_demo(data)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    args = _args(tmp_path, data, init, wall=wall)

    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_context", lambda _method: _Context())
    assert T.main_multi(args) == 0
    summary = json.loads((args.out / "summary.json").read_text())
    assert summary["full_state_complete"] is True
    assert any(
        event.get("event") == "full-state-snapshot" for event in summary["full_state_events"]
    )
    if wall:
        assert summary["stopped_for_time"] is True
