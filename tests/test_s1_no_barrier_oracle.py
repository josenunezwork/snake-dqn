"""S1 no-barrier whole-trainer oracle against the frozen 474b3c2e trainer.

The actor transport below is deliberately deterministic: each fake actor writes a fixed
set of rows into the real shared observation ring and puts the same fixed packets into an
unbounded FIFO.  This proves the no-barrier learner trajectory without pretending to prove
an arbitrary asynchronous actor schedule.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import queue
import subprocess
import sys
import threading
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.redesign_m3c_20261008 import learner_m3c as LM  # noqa: E402
from research.redesign_m3c_20261008 import train_m3c as CANDIDATE  # noqa: E402

_BASE_LEARNER_UPDATE = LM.Learner.update
_BASE_LEARNER_CLOSE = LM.Learner.close

REPO = Path(__file__).resolve().parents[1]
BASE_REV = "474b3c2e233b167f071f9c5b9ea729ca39222371"
TRAINER = "research/redesign_m3c_20261008/train_m3c.py"
FROZEN_SHA256 = "4304a1127eccd6a1abe5af789e0908116168ba4c5c274a01388978f1338b646d"
ACTORS = 2
FRAMES = 20
ROWS_PER_FRAME = 24  # 8 worlds x 3 learner seats
TOTAL_ROWS = FRAMES * ROWS_PER_FRAME


def _digest(value) -> str:
    """Deterministic recursive digest for tensors, arrays, mappings, and scalars."""
    h = hashlib.sha256()

    def visit(obj) -> None:
        if isinstance(obj, torch.Tensor):
            obj = obj.detach().cpu().contiguous()
            h.update(b"tensor\0")
            h.update(str(obj.dtype).encode())
            h.update(repr(tuple(obj.shape)).encode())
            h.update(obj.numpy().tobytes())
            return
        if isinstance(obj, np.ndarray):
            h.update(b"array\0")
            h.update(str(obj.dtype).encode())
            h.update(repr(obj.shape).encode())
            h.update(obj.tobytes(order="C"))
            return
        if isinstance(obj, dict):
            h.update(b"dict\0")
            for key in sorted(obj, key=lambda item: repr(item)):
                visit(key)
                visit(obj[key])
            return
        if isinstance(obj, (list, tuple)):
            h.update((b"list\0" if isinstance(obj, list) else b"tuple\0"))
            for item in obj:
                visit(item)
            return
        if isinstance(obj, (bytes, bytearray, memoryview)):
            h.update(b"bytes\0")
            h.update(bytes(obj))
            return
        h.update(type(obj).__name__.encode())
        h.update(repr(obj).encode())

    visit(value)
    return h.hexdigest()


def _semantic_counters(st: dict) -> dict:
    """Counters excluding wall-clock fields and other intentionally volatile telemetry."""
    keys = (
        "transitions",
        "replay_rows",
        "updates",
        "since_update",
        "ring_overrun_rows",
        "truncation_rows_dropped",
        "pending_nstep_rows_by_actor",
        "decisions",
        "boost",
        "greedy_boost",
        "deaths",
        "v8_override",
        "label_rows",
        "anchor_updates",
        "td",
        "anc",
        "qmax",
        "lag",
        "live_rows_by_policy",
    )
    return {key: st.get(key) for key in keys}


def _learner_digest(learner) -> dict:
    """Capture the state that determines the next learner trajectory."""
    return {
        "online": _digest(learner.online.state_dict()),
        "target": _digest(learner.target.state_dict()),
        "optimizer": _digest(learner.opt.state_dict()),
        "replay": _digest(
            {
                "s": learner.replay.s,
                "a": learner.replay.a,
                "R": learner.replay.R,
                "disc": learner.replay.disc,
                "nxt": learner.replay.nxt,
                "size": learner.replay.size,
                "ptr": learner.replay.ptr,
                "added": learner.replay.added,
                "stale_rows_filtered": learner.replay.stale_rows_filtered,
            }
        ),
        "rng": _digest(learner.rng.bit_generator.state),
        "pending": _digest(getattr(learner, "_pending", None)),
    }


def _load_frozen(tmp_path: Path) -> ModuleType:
    source = subprocess.check_output(["git", "show", f"{BASE_REV}:{TRAINER}"], cwd=REPO)
    assert hashlib.sha256(source).hexdigest() == FROZEN_SHA256
    path = tmp_path / "train_m3c_pre_474b3c2e.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("s1_frozen_train_m3c", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.REPO = REPO
    return module


def _write_demo(root: Path) -> None:
    round_ = root / "round-12"
    round_.mkdir(parents=True)
    n = 16
    np.savez(
        round_ / "chunk-000-part-000.npz",
        world_seed=np.arange(n, dtype=np.int64) + 1,
        local=np.zeros((n, 7, 31, 31), dtype=np.uint8),
        **{"global": np.zeros((n, 4, 37, 37), dtype=np.uint8)},
        scalars=np.zeros((n, 30), dtype=np.float32),
        q_teacher=np.zeros((n, 6), dtype=np.float32),
        a_v8=np.zeros(n, dtype=np.int64),
        mask_resolved=np.ones((n, 6), dtype=bool),
    )
    (root / "margin.json").write_text(json.dumps({"margin": 0.3}))


def _packet(actor: int, frame: int) -> dict:
    ids = np.arange(frame * ROWS_PER_FRAME, (frame + 1) * ROWS_PER_FRAME, dtype=np.int64)
    return {
        "j": actor,
        "s": ids,
        "a": (ids + actor) % 3,
        "R": (ids.astype(np.float32) * 0.001 + actor).astype(np.float32),
        "disc": np.full(len(ids), 0.99, dtype=np.float32),
        "nxt": (ids + 1) % TOTAL_ROWS,
        "acts": ((ids + actor) % 3).astype(np.int8),
        "lab_ids": np.zeros(0, dtype=np.int64),
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
        "decisions": ROWS_PER_FRAME,
        "boost": 0,
        "greedy_boost": 0,
        "greedy_boost_length_decisions": np.zeros(4, dtype=np.int64),
        "greedy_boost_length_boosts": np.zeros(4, dtype=np.int64),
        "v8_override": 0,
    }


class _FakeProcess:
    """Synchronous actor replacement with the real ring and learner transport objects."""

    def __init__(self, *, target, args, daemon=False):
        self.target, self.args, self.daemon = target, args, daemon
        self._alive = False
        self.exitcode = None

    def start(self):
        actor, cfg, names, tx, _stop, _control = self.args
        block = self.target_module.Block.attach(names[f"ring{actor}"])
        ring = self.target_module.SharedObsRing(
            block,
            cfg["ring_cap"],
            cfg["ring_margin"],
            60,
        )
        rng = np.random.default_rng([17, actor, 901])
        n = TOTAL_ROWS
        local = rng.integers(0, 256, (n,) + self.target_module.LOCAL, dtype=np.uint8)
        global_ = rng.integers(0, 256, (n,) + self.target_module.GLOBAL, dtype=np.uint8)
        scalars = rng.standard_normal((n, 60), dtype=np.float32)
        mask = np.zeros((n, 6), dtype=bool)
        mask[:, :3] = True
        ring.add({"local": local, "global": global_, "scalars": scalars}, mask)
        for frame in range(FRAMES):
            tx.put(_packet(actor, frame))
        self._alive = True

    def is_alive(self):
        return self._alive

    def join(self, timeout=None):
        del timeout
        self._alive = False
        self.exitcode = 0

    def terminate(self):
        self.join()


class _Context:
    def __init__(self, module: ModuleType):
        self.module = module

    def Event(self):
        return threading.Event()

    def Queue(self, maxsize=0):
        del maxsize
        return queue.Queue()

    def Process(self, **kwargs):
        process = _FakeProcess(**kwargs)
        process.target_module = self.module
        return process


def _args(module: ModuleType, root: Path, demo: Path, init: Path):
    args = module.default_args()
    args.seed = 17
    args.init = str(init)
    args.demo_data = demo
    args.demo_rounds = "12"
    args.out = root / "out"
    args.actors = ACTORS
    args.worlds = 8
    args.learner_seats = 3
    args.transitions = ACTORS * TOTAL_ROWS
    args.warmup = 256
    args.replay = 2048
    args.device = args.actor_device = "cpu"
    args.no_guard = True
    args.full_state_every = 0
    args.schedule_package = "mature"
    module.apply_schedule_package(args)
    args.obs_version = "bv8"
    args.onpolicy_labels = True
    args.label_frac = 1.0
    args.learner_fast = True
    args.batch_scale = 2
    args.actor_queue_frames = 2
    args.checkpoint_every = 10**9
    return args


def _run(module: ModuleType, root: Path, demo: Path, init: Path, monkeypatch, captures):
    import multiprocessing as mp

    monkeypatch.setattr(mp, "get_context", lambda _method: _Context(module))
    original_sample = module.Replay.sample
    samples = []

    def sample_spy(self, *args, **kwargs):
        value = original_sample(self, *args, **kwargs)
        samples.append(value.copy())
        return value

    monkeypatch.setattr(module.Replay, "sample", sample_spy)
    active = {"value": captures}
    original_update = _BASE_LEARNER_UPDATE
    original_close = _BASE_LEARNER_CLOSE

    def update_spy(self, lam):
        original_update(self, lam)
        active["value"].append(
            {
                "state": _learner_digest(self),
                "counters": _digest(_semantic_counters(self.st)),
            }
        )

    def close_spy(self):
        if not getattr(self, "_s1_close_seen", False):
            captures.append(
                {"final": _learner_digest(self), "counters": _digest(_semantic_counters(self.st))}
            )
            self._s1_close_seen = True
        original_close(self)

    monkeypatch.setattr(LM.Learner, "update", update_spy)
    monkeypatch.setattr(LM.Learner, "close", close_spy)
    args = _args(module, root, demo, init)
    assert module.main_multi(args) == 0
    summary = json.loads((args.out / "summary.json").read_text())
    return samples, captures, summary


def test_s1_no_barrier_whole_learner_trajectory_matches_frozen(tmp_path, monkeypatch):
    """No full-state barrier means current and frozen trainers are byte-identical."""
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint

    demo = tmp_path / "demo"
    _write_demo(demo)
    init = tmp_path / "init.pth"
    torch.manual_seed(17)
    save_ego2s_checkpoint(init, Ego2sNet(7, 60))
    frozen = _load_frozen(tmp_path)
    candidate_samples, candidate_trace, candidate_summary = _run(
        CANDIDATE, tmp_path / "candidate", demo, init, monkeypatch, []
    )
    frozen_samples, frozen_trace, frozen_summary = _run(
        frozen, tmp_path / "frozen", demo, init, monkeypatch, []
    )

    assert len(candidate_samples) == len(frozen_samples) == 3
    assert len(candidate_trace) == len(frozen_trace) == 4  # 3 updates plus the final close
    assert all(np.array_equal(a, b) for a, b in zip(candidate_samples, frozen_samples))
    assert candidate_trace == frozen_trace
    for key in (
        "transitions",
        "learner_decisions",
        "replay_rows",
        "emitted_replay_rows",
        "learner_decisions_accounted",
        "pending_update_debt",
        "decision_replay_delta",
        "transition_budget_overshoot",
        "updates",
        "ring_overrun_rows",
        "anchor_updates",
        "actor_exitcodes",
        "full_state_events",
        "full_state_complete",
    ):
        assert candidate_summary[key] == frozen_summary[key], key
    assert candidate_summary["transitions"] == 960
    assert candidate_summary["transition_budget_overshoot"] == 0
    assert candidate_summary["full_state_events"] == []
    assert candidate_summary["full_state_complete"] is None
