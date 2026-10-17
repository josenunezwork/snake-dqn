"""Actor FIFO checkpoint-marker integration coverage for T21.

This deliberately invokes ``actor_main`` in-process.  File-backed blocks make
the shared observation ring boundary visible to the test without spawning an
actor or running learner training.
"""

from __future__ import annotations

import queue
from pathlib import Path

import numpy as np
import pytest

from research.redesign_m3c_20261008 import train_m3c as T


class _Stop:
    def __init__(self) -> None:
        self.flag = False

    def is_set(self) -> bool:
        return self.flag

    def set(self) -> None:
        self.flag = True


class _Control:
    """The small subset of a multiprocessing Queue used by ``actor_main``."""

    def __init__(self) -> None:
        self.items = []

    def put(self, value, **_kwargs) -> None:
        self.items.append(value)

    def get_nowait(self):
        if not self.items:
            raise queue.Empty
        return self.items.pop(0)

    def get(self, timeout=None):
        del timeout
        return self.get_nowait()


class _Sink:
    def __init__(
        self, stop: _Stop, control: _Control, marker_after: int | None, stop_after: int, ring
    ):
        self.stop = stop
        self.control = control
        self.marker_after = marker_after
        self.stop_after = stop_after
        self.ring = ring
        self.frames = []
        self.snapshot = None
        self.marker_ring = None

    def put(self, msg, timeout: float = 0.0) -> None:
        del timeout
        if msg.get("kind") == "snapshot":
            self.snapshot = msg
            # The marker is emitted only after the actor has finished the frame;
            # this is exactly the ring state the learner copies before release.
            self.marker_ring = _ring_state(self.ring)
            self.control.put({"op": "release", "id": msg["id"]})
            return
        self.frames.append(msg)
        if self.marker_after is not None and len(self.frames) == self.marker_after:
            self.control.put({"op": "snapshot", "id": 71})
        if len(self.frames) >= self.stop_after:
            self.stop.set()


class _MockStaticPool:
    """Static mixed-seat dispatch with no pinned checkpoint dependency."""

    policy_ids = ("mock-a", "mock-b")

    def __init__(self) -> None:
        from research.redesign_m3c_20261008.league_pool import PoolIdentity

        # The actor snapshots wrapper execution state by identity even when the
        # policy has no mutable wrapper counters.
        self.policies = {policy_id: object() for policy_id in self.policy_ids}
        self.identities = tuple(
            PoolIdentity(policy_id, "scripted", None, None, "scripted", "resolved")
            for policy_id in self.policy_ids
        )

    def identity_records(self):
        return [{"policy_id": item, "kind": "test"} for item in self.policy_ids]

    def prime_b8_with_learner_rows(self, _prepared, _hero, _alive, _seats):
        return None  # This fixture contains only scripted policies.

    def v8_with_learner_rows(self, prepared, hero, mask, _alive, _seats):
        from src.simd_env.ego_raster_bv8 import v8_scalars

        return v8_scalars(prepared, hero, mask)  # No frozen B8 group to share in this fixture.

    def dispatch(self, _prepared, masks, alive, actions, seats):
        rows = np.argwhere(alive & np.asarray(seats != None))  # noqa: E711 - elementwise None
        if len(rows):
            legal = np.asarray(masks)[rows[:, 0], rows[:, 1]]
            actions[rows[:, 0], rows[:, 1]] = np.argmax(legal, axis=1)
        return {"frozen_live_rows": {"mock": int(len(rows))}, "frozen_forward_gpu_s": 0.0}


def _ring_state(ring):
    return {
        "count": ring.count,
        "arrays": {name: value.copy() for name, value in ring.arrays.items()},
    }


def _restore_ring(ring, state) -> None:
    for name, value in state["arrays"].items():
        ring.arrays[name][...] = value
    ring._count[0] = state["count"]


def _new_blocks(tmp_path, args):
    nscal = T.nscal_for(args)
    margin = args.worlds * 16
    capacity = args.worlds * (args.segment_frames + 4) + margin
    net = T.load_init(args, "cpu")
    n_weights = int(sum(count for _, _, count in T._weights_spec(net)))
    blocks = [
        T.Block.create(T.ring_nbytes(capacity, nscal), str(tmp_path), "ring0"),
        T.Block.create(8 + 8 * (n_weights + 1), str(tmp_path), "weights"),
        T.Block.create(32, str(tmp_path), "progress"),
    ]
    names = {"ring0": blocks[0].desc, "weights": blocks[1].desc, "progress": blocks[2].desc}
    wver = np.ndarray((1,), dtype=np.int64, buffer=blocks[1].buf, offset=0)
    wbuf = np.ndarray((n_weights + 1,), dtype=np.float64, buffer=blocks[1].buf, offset=8)
    progress = np.ndarray((4,), dtype=np.int64, buffer=blocks[2].buf, offset=0)
    wver[0] = 0
    progress[:] = 0
    T._publish_weights(net, wbuf, wver)
    ring = T.SharedObsRing(blocks[0], capacity, margin, nscal)
    return blocks, names, ring, wver, wbuf, progress, n_weights


def _close_blocks(blocks) -> None:
    for block in blocks:
        block.unlink()
        block.close()


def _assert_messages_equal(expected, actual) -> None:
    assert len(expected) == len(actual)
    for frame, (left, right) in enumerate(zip(expected, actual)):
        assert left.keys() == right.keys(), frame
        for key, value in left.items():
            if key in {"env_s", "timing"}:
                continue
            other = right[key]
            if isinstance(value, np.ndarray):
                assert value.dtype == other.dtype, (frame, key)
                assert np.array_equal(value, other), (frame, key)
            else:
                assert value == other, (frame, key)


@pytest.mark.parametrize("mixed", [False, True], ids=["hero", "learner-seats-3"])
def test_actor_fifo_marker_resume_preserves_ring_and_nstep_continuation(
    tmp_path, monkeypatch, mixed
):
    """A marker snapshot resumes the actual actor bit-for-bit from its ring boundary."""
    import torch

    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    torch.manual_seed(701)
    init = tmp_path / "init.pth"
    save_ego2s_checkpoint(init, Ego2sNet(7, 60 if mixed else 30))
    extra = []
    if mixed:
        import research.redesign_m3c_20261008.league_pool as league_pool

        monkeypatch.setattr(
            league_pool, "build_default_static_pool", lambda *args: _MockStaticPool()
        )
        extra = [
            "--learner-seats",
            "3",
            "--obs-version",
            "bv8",
            "--onpolicy-labels",
            "--label-frac",
            "1.0",
            "--anchor-mode",
            "v8margin",
        ]
    args = T.build_parser().parse_args(
        [
            "--seed",
            "41",
            "--init",
            str(init),
            "--demo-data",
            ".",
            "--out",
            str(tmp_path / "out"),
            "--actors",
            "1",
            "--worlds",
            "2",
            "--segment-frames",
            "16",
            "--device",
            "cpu",
            "--actor-device",
            "cpu",
            *extra,
        ]
    )
    marker_after, continuation = 3, 5
    blocks, names, ring, wver, wbuf, progress, n_weights = _new_blocks(tmp_path, args)
    try:
        control, stop = _Control(), _Stop()
        baseline = _Sink(stop, control, marker_after, marker_after + continuation, ring)
        cfg = {
            "args": {
                key: (str(value) if isinstance(value, Path) else value)
                for key, value in vars(args).items()
            },
            "K": 1,
            "ring_cap": ring.cap,
            "ring_margin": ring.margin,
            "n_weights": n_weights,
            "actor_device": "cpu",
        }
        T.actor_main(0, cfg, names, baseline, stop, control)
        assert baseline.snapshot is not None
        assert len(baseline.frames) == marker_after + continuation
        assert baseline.snapshot["state"]["frame"] == marker_after
        assert baseline.marker_ring["count"] > 0
        full_ring = _ring_state(ring)
        saved_wver, saved_wbuf, saved_progress = int(wver[0]), wbuf.copy(), progress.copy()
    finally:
        _close_blocks(blocks)

    # This is a new runtime allocation, exactly like a process reconnect.  Copy
    # the marker ring and shared learner state before starting the resumed actor.
    resume_dir = tmp_path / "resume"
    resume_dir.mkdir()
    blocks, names, ring, wver, wbuf, progress, n_weights = _new_blocks(resume_dir, args)
    try:
        _restore_ring(ring, baseline.marker_ring)
        wver[0], wbuf[:], progress[:] = saved_wver, saved_wbuf, saved_progress
        cfg["resume_actor_states"] = {"0": baseline.snapshot["state"]}
        cfg["resume_with_full_replay"] = True
        resumed_stop, resumed_control = _Stop(), _Control()
        resumed = _Sink(resumed_stop, resumed_control, None, continuation, ring)
        T.actor_main(0, cfg, names, resumed, resumed_stop, resumed_control)
        _assert_messages_equal(baseline.frames[marker_after:], resumed.frames)
        assert _ring_state(ring)["count"] == full_ring["count"]
        for name, value in full_ring["arrays"].items():
            assert np.array_equal(ring.arrays[name], value), name
    finally:
        _close_blocks(blocks)

    # Replay-none deliberately cannot keep n-step IDs: their source observation
    # rows were intentionally not saved.  The actor must begin a fresh window.
    no_replay_dir = tmp_path / "resume-no-replay"
    no_replay_dir.mkdir()
    blocks, names, ring, wver, wbuf, progress, n_weights = _new_blocks(no_replay_dir, args)
    try:
        _restore_ring(ring, baseline.marker_ring)
        wver[0], wbuf[:], progress[:] = saved_wver, saved_wbuf, saved_progress
        cfg["resume_with_full_replay"] = False
        no_replay_stop, no_replay_control = _Stop(), _Control()
        no_replay = _Sink(no_replay_stop, no_replay_control, None, 1, ring)
        T.actor_main(0, cfg, names, no_replay, no_replay_stop, no_replay_control)
        assert len(no_replay.frames) == 1
        assert len(no_replay.frames[0]["s"]) == 0
        assert len(no_replay.frames[0]["nxt"]) == 0
    finally:
        _close_blocks(blocks)
