"""S1 full-state barrier packet accounting without a Colab runtime.

The trainer stops learner ingestion after requesting actor snapshot markers.  These
tests exercise the persisted FIFO packet boundary independently of multiprocessing:
the full no-barrier learner trajectory oracle is in ``test_s1_no_barrier_oracle``.
"""

from __future__ import annotations

import importlib.util
import json
import queue
import threading
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008.train_m3c import deferred_actor_packets


def _t21_module():
    path = Path(__file__).with_name("test_t21_multi_integration.py")
    spec = importlib.util.spec_from_file_location("s1_t21_fake", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _packet(actor: int, decisions: int, marker: str) -> dict:
    return {"actor": actor, "message": {"decisions": decisions, "marker": marker}}


def test_full_replay_restores_each_actor_fifo_across_consecutive_barriers() -> None:
    # The first two entries model an earlier capture which was restored but had
    # not drained before a later barrier staged the trailing two packets.
    saved = [
        _packet(0, 5, "old-a"),
        _packet(1, 7, "old-b"),
        _packet(0, 11, "new-a"),
        _packet(1, 13, "new-b"),
    ]

    restored, dropped = deferred_actor_packets(saved, actors=2, replay_is_present=True)

    assert dropped == 0
    assert [item["marker"] for item in restored[0]] == ["old-a", "new-a"]
    assert [item["marker"] for item in restored[1]] == ["old-b", "new-b"]
    assert sum(item["decisions"] for queue in restored.values() for item in queue) == 36


def test_replay_none_discards_ring_addressed_packets_with_accounted_decisions() -> None:
    saved = [_packet(0, 17, "a"), _packet(1, 19, "b")]

    restored, dropped = deferred_actor_packets(saved, actors=2, replay_is_present=False)

    assert all(not queue for queue in restored.values())
    assert dropped == 36


@pytest.mark.parametrize("target", [1, 2, 12_000, 8_000_000])
def test_main_multi_final_barrier_holds_fifo_packet_in_snapshot(
    tmp_path, monkeypatch, target: int
) -> None:
    """A real main loop freezes its terminal counter before the actor marker.

    The fake actor deliberately queues a second ordinary packet after the learner
    has consumed the packet that crosses the terminal target.  The old path
    ingested it while waiting for the marker; the barrier must retain it in the
    saved full-state payload instead.
    """
    t21 = _t21_module()
    from research.redesign_m3c_20261008 import async_checkpoint
    from research.redesign_m3c_20261008 import train_m3c as trainer
    from research.redesign_m3c_20261008.learner_m3c import Learner
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    original_init = Learner.__init__

    def seeded_counter(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        state = kwargs["st"]
        # The 12k/8M rows model the already-reconciled prefix without claiming
        # to execute that many frames.  Target two intentionally starts at zero
        # to exercise a nonterminal release and local held-packet refill.
        prefix = 0 if target == 2 else target - 1
        state["transitions"] = prefix
        state["decisions"] = prefix
        state["replay_rows"] = prefix

    monkeypatch.setattr(Learner, "__init__", seeded_counter)

    def packet(j: int, marker: int) -> dict:
        zeros_i = np.zeros(0, dtype=np.int64)
        zeros_f = np.zeros(0, dtype=np.float32)
        return {
            "j": j,
            "barrier_test_marker": marker,
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
            "timing": {"featurizer_s": 0.0, "forward_s": 0.0, "opponents_s": 0.0, "v8_s": 0.0},
            "weights_version": 2,
            "decisions": 1,
            "boost": 0,
            "greedy_boost": 0,
            "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
            "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
            "v8_override": 0,
        }

    def start(self):
        def run():
            j, _cfg, _names, tx, stop, control = self.args
            try:
                tx.put(packet(j, 1), timeout=1)
                # This blocks until the learner consumes the target-crossing
                # packet and requests the barrier, then deliberately sits ahead
                # of the snapshot marker in the same actor FIFO.
                tx.put(packet(j, 2), timeout=1)
                request = control.get(timeout=1)
                assert request["op"] == "snapshot"
                tx.put({"kind": "snapshot", "j": j, "id": request["id"], "state": {"segment": j}})
                while not stop.is_set():
                    try:
                        if control.get(timeout=0.05).get("op") == "release":
                            break
                    except queue.Empty:
                        pass
                if stop.is_set() or target != 2:
                    return
                # On the nonterminal cadence test this is fresh traffic after
                # release.  The learner must consume the locally-held second
                # packet first, then preserve this third packet at the terminal
                # barrier instead of dropping it.
                tx.put(packet(j, 3), timeout=1)
                request = control.get(timeout=1)
                assert request["op"] == "snapshot"
                tx.put({"kind": "snapshot", "j": j, "id": request["id"], "state": {"segment": j}})
                while not stop.is_set():
                    try:
                        if control.get(timeout=0.05).get("op") == "release":
                            break
                    except queue.Empty:
                        pass
            finally:
                self.exitcode = 0

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    monkeypatch.setattr(t21._FakeProcess, "start", start)
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_context", lambda _method: t21._Context())
    data = tmp_path / "demo"
    t21._write_demo(data)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 30))
    args = t21._args(tmp_path, data, init, wall=0.0)
    args.transitions = target
    assert trainer.main_multi(args) == 0

    summary = json.loads((args.out / "summary.json").read_text())
    events = [
        item for item in summary["full_state_events"] if item["event"] == "full-state-snapshot"
    ]
    event = events[-1]
    assert summary["transitions"] == target
    assert summary["transition_budget_overshoot"] == 0
    assert event["transitions"] == target and event["deferred_decisions"] == 1
    payload, _ = async_checkpoint.load_latest_snapshot(args.ckpt_local_dir)
    held = payload["extra"]["deferred_actor_messages"]
    assert [(item["actor"], item["message"]["decisions"]) for item in held] == [(0, 1)]
    if target == 2:
        assert [item["transitions"] for item in events] == [1, 2]
        assert held[0]["message"]["barrier_test_marker"] == 3


@pytest.mark.parametrize(
    "saved",
    [
        [{"actor": 2, "message": {}}],
        [{"actor": -1, "message": {}}],
        [{"actor": 0, "message": None}],
    ],
)
def test_malformed_deferred_packets_are_rejected_before_actor_start(saved: list[dict]) -> None:
    with pytest.raises(ValueError, match="deferred actor message is invalid"):
        deferred_actor_packets(saved, actors=2, replay_is_present=True)
