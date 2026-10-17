"""Focused tests for the T21 full-state snapshot boundary.

These tests deliberately exercise the public, format-independent checkpoint API.  The
trainer owns the contents of ``state``; the checkpointer owns detached copies, atomic
publication, checksums, retention, and best-effort durable publication.
"""

from __future__ import annotations

import json
import random
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest


def _checkpoint_api():
    """Load the implementation without coupling tests to the trainer's import site."""
    candidates = (
        "research.redesign_m3c_20261008.async_checkpoint",
        "research.async_checkpoint",
    )
    errors = []
    for name in candidates:
        try:
            module = __import__(name, fromlist=["AsyncCheckpointer"])
            return module
        except ModuleNotFoundError as exc:
            errors.append(f"{name}: {exc}")
    raise AssertionError("T21 async checkpoint module is missing: " + "; ".join(errors))


def _capture_state(module, state):
    capture = getattr(module, "capture_state", None)
    if capture is None:
        capture = getattr(module.AsyncCheckpointer, "capture_state", None)
    if capture is None:
        raise AssertionError("AsyncCheckpointer.capture_state or capture_state is required")
    return capture(state)


def _manifest_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file() and "manifest" in p.name.lower() and ".tmp" not in p.name
    )


def _payload_files(root: Path) -> list[Path]:
    return sorted(
        p
        for p in root.rglob("*")
        if p.is_file()
        and ".tmp" not in p.name
        and "manifest" not in p.name.lower()
        and p.suffix not in {".lock", ".jsonl"}
    )


def _wait_close(checkpointer) -> None:
    """Close is the public drain point; tolerate a false result for durable-copy failure."""
    result = checkpointer.close(timeout=10)
    assert result in (True, False, None)
    return result


def test_capture_state_detaches_nested_tensors_and_arrays():
    torch = pytest.importorskip("torch")
    module = _checkpoint_api()
    tensor = torch.arange(6, dtype=torch.float32, requires_grad=True).reshape(2, 3)
    array = np.arange(4, dtype=np.int64)
    state = {"tensor": tensor, "nested": {"array": array}, "tuple": (tensor, array)}

    captured = _capture_state(module, state)
    tensor.data.add_(100)
    array[:] = -1

    assert captured["tensor"].device.type == "cpu"
    assert not captured["tensor"].requires_grad
    assert torch.equal(captured["tensor"], torch.arange(6, dtype=torch.float32).reshape(2, 3))
    np.testing.assert_array_equal(captured["nested"]["array"], np.arange(4, dtype=np.int64))
    assert captured["tuple"][0] is not tensor
    assert captured["tuple"][1] is not array


def test_async_snapshot_roundtrip_and_atomic_manifest(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    durable = tmp_path / "durable"
    local.mkdir()
    durable.mkdir()
    state = {
        "online": np.arange(12, dtype=np.float32).reshape(3, 4),
        "schedule": {"gamma": 0.997, "eps": 0.21},
        "rng": np.random.default_rng(17).bit_generator.state,
    }
    checkpointer = module.AsyncCheckpointer(local, durable, keep=2, retries=2, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, state), transitions=123, metadata={"seed": 4})
    _wait_close(checkpointer)

    loaded, manifest = module.load_latest_snapshot(local)
    np.testing.assert_array_equal(loaded["online"], state["online"])
    assert loaded["schedule"] == state["schedule"]
    assert manifest["transitions"] == 123
    assert manifest["files"][0]["sha256"]
    assert _manifest_files(local)
    assert _manifest_files(durable)
    assert not any(p.name.endswith(".tmp") for p in tmp_path.rglob("*"))


def test_retention_keeps_last_two_complete_snapshots(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    durable = tmp_path / "durable"
    local.mkdir()
    durable.mkdir()
    checkpointer = module.AsyncCheckpointer(local, durable, keep=2, retries=1, retry_delay=0)
    for transitions in (1, 2, 3):
        state = {"transitions": transitions, "values": np.array([transitions], dtype=np.int64)}
        assert checkpointer.submit(_capture_state(module, state), transitions=transitions)
        assert checkpointer.wait_local(timeout=10)
    _wait_close(checkpointer)

    loaded, manifest = module.load_latest_snapshot(local)
    assert loaded["transitions"] == 3
    assert manifest["transitions"] == 3
    assert len(_manifest_files(local)) <= 2
    assert len(_payload_files(local)) <= 2
    assert len(_manifest_files(durable)) <= 2


def test_corrupt_newest_snapshot_falls_back_to_older_valid_snapshot(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    for transitions in (1, 2):
        assert checkpointer.submit(
            _capture_state(module, {"transitions": transitions}), transitions=transitions
        )
        assert checkpointer.wait_local(timeout=10)
    _wait_close(checkpointer)

    newest = module.discover_snapshots(local)[0]
    (local / newest["snapshot"] / "state.pt").write_bytes(b"corrupt newest")
    loaded, manifest = module.load_latest_snapshot(local)
    assert loaded["transitions"] == 1
    assert manifest["transitions"] == 1


def test_corrupt_snapshot_is_rejected(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    _wait_close(checkpointer)
    payloads = _payload_files(local)
    assert payloads, "the complete snapshot must contain a payload file"
    payloads[-1].write_bytes(b"corrupt snapshot")
    with pytest.raises(Exception):
        module.load_latest_snapshot(local)


def test_partial_snapshot_is_ignored_or_rejected_without_affecting_training(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    _wait_close(checkpointer)
    (local / "snapshot-partial.tmp").write_bytes(b"incomplete")

    loaded, manifest = module.load_latest_snapshot(local)
    np.testing.assert_array_equal(loaded["value"], np.arange(3))
    assert manifest["transitions"] == 1


def test_durable_copy_failure_is_nonfatal_and_local_snapshot_remains_loadable(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    durable_file = tmp_path / "durable-file"
    durable_file.write_text("destination is intentionally not a directory")
    checkpointer = module.AsyncCheckpointer(local, durable_file, keep=2, retries=2, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    _wait_close(checkpointer)

    loaded, manifest = module.load_latest_snapshot(local)
    np.testing.assert_array_equal(loaded["value"], np.arange(3))
    assert manifest["transitions"] == 1
    events = checkpointer.poll_events()
    assert events is not None


def test_local_write_failure_returns_false_from_close_and_later_recovery_is_valid(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local-file"
    local.write_text("this path is intentionally a regular file")
    failed = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert failed.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    assert failed.close(timeout=10) is False
    events = failed.poll_events()
    assert any(event.get("event") == "checkpoint_local_failed" for event in events)

    local.unlink()
    local.mkdir()
    recovered = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert recovered.submit(_capture_state(module, {"value": np.arange(4)}), transitions=2)
    assert recovered.close(timeout=10) is True
    state, manifest = module.load_latest_snapshot(local)
    np.testing.assert_array_equal(state["value"], np.arange(4))
    assert manifest["transitions"] == 2


def test_malformed_manifest_inventory_and_timestamp_are_not_resumable(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    assert checkpointer.close(timeout=10) is True
    manifest_path = _manifest_files(local)[0]
    original = json.loads(manifest_path.read_text())
    for malformed in (
        {**original, "files": {"name": "state.pt"}},
        {**original, "files": "state.pt"},
        {key: value for key, value in original.items() if key != "created_ns"},
    ):
        manifest_path.write_text(json.dumps(malformed))
        assert module.discover_snapshots(local) == []
        with pytest.raises(ValueError, match="no complete"):
            module.load_latest_snapshot(local)
        manifest_path.write_text(json.dumps(original))


def test_submit_does_not_wait_for_blocked_durable_copy(tmp_path, monkeypatch):
    module = _checkpoint_api()
    local = tmp_path / "local"
    durable = tmp_path / "durable"
    local.mkdir()
    durable.mkdir()
    checkpointer = module.AsyncCheckpointer(local, durable, keep=2, retries=1, retry_delay=0)
    entered = threading.Event()
    release = threading.Event()
    original = checkpointer._copy_snapshot

    def blocked(name):
        entered.set()
        assert release.wait(timeout=10)
        return original(name)

    monkeypatch.setattr(checkpointer, "_copy_snapshot", blocked)
    started = time.perf_counter()
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    submit_ms = (time.perf_counter() - started) * 1000
    assert submit_ms < 50, f"submit blocked for {submit_ms:.2f} ms"
    assert checkpointer.wait_local(timeout=10)
    assert entered.wait(timeout=10)

    started = time.perf_counter()
    accepted = checkpointer.submit(_capture_state(module, {"value": np.arange(4)}), transitions=2)
    second_submit_ms = (time.perf_counter() - started) * 1000
    assert (
        second_submit_ms < 50
    ), f"submit blocked behind durable copy for {second_submit_ms:.2f} ms"
    assert accepted
    release.set()
    _wait_close(checkpointer)


def test_failed_durable_copies_are_bounded_and_newer_success_recovers(tmp_path, monkeypatch):
    module = _checkpoint_api()
    local = tmp_path / "local"
    durable = tmp_path / "durable"
    local.mkdir()
    durable.mkdir()
    checkpointer = module.AsyncCheckpointer(local, durable, keep=2, retries=1, retry_delay=0)
    original = checkpointer._copy_snapshot
    failed = {"count": 0}

    def fail_first_six(name):
        if failed["count"] < 6:
            failed["count"] += 1
            raise OSError("simulated Drive outage")
        return original(name)

    monkeypatch.setattr(checkpointer, "_copy_snapshot", fail_first_six)
    for transitions in range(1, 8):
        assert checkpointer.submit(
            _capture_state(module, {"transitions": transitions}), transitions=transitions
        )
        assert checkpointer.wait_local(timeout=10)
    assert _wait_close(checkpointer) is True

    assert len(_manifest_files(local)) <= 3  # keep plus at most one active upload
    assert len(_manifest_files(durable)) <= 2
    loaded, manifest = module.load_latest_snapshot(durable)
    assert loaded["transitions"] == 7
    assert manifest["transitions"] == 7
    assert failed["count"] == 6


def test_small_snapshot_capture_stall_is_under_fifty_ms():
    module = _checkpoint_api()
    state = {
        "online": np.zeros((32, 32), dtype=np.float32),
        "target": np.ones((32, 32), dtype=np.float32),
    }
    _capture_state(module, state)  # warm up imports and allocator
    started = time.perf_counter()
    _capture_state(module, state)
    elapsed_ms = (time.perf_counter() - started) * 1000
    assert elapsed_ms < 50, f"small-model capture stalled learner for {elapsed_ms:.2f} ms"


def test_stale_partial_snapshot_is_reclaimed_by_next_writer(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    stale = local / ".tmp-snapshot-stale-after-crash"
    stale.mkdir()
    (stale / "partial.state").write_bytes(b"interrupted write")

    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=1)
    assert checkpointer.close(timeout=10) is True

    assert not stale.exists(), "the lease holder should reclaim stale temporary snapshots"
    state, manifest = module.load_latest_snapshot(local)
    np.testing.assert_array_equal(state["value"], np.arange(3))
    assert manifest["transitions"] == 1


def test_active_writer_lease_preserves_live_temp_for_second_writer(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()

    first = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert first.submit(_capture_state(module, {"owner": "first"}), transitions=1)
    assert first.wait_local(timeout=10)
    state, manifest = module.load_latest_snapshot(local)
    assert state["owner"] == "first"
    assert manifest["transitions"] == 1

    # This represents a first writer still in the middle of a process-local save.  The
    # second writer must fail its nonblocking lease acquisition and leave this directory.
    live_temp = local / ".tmp-snapshot-live-first-writer"
    live_temp.mkdir()
    (live_temp / "partial.state").write_bytes(b"active writer marker")

    second = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert second.submit(_capture_state(module, {"owner": "second"}), transitions=2)
    assert second.close(timeout=10) is False
    assert any(event.get("event") == "checkpoint_local_failed" for event in second.poll_events())
    assert live_temp.exists(), "a live writer's temporary directory must not be reclaimed"

    # The first writer still owns the lease and its complete snapshot remains resumable.
    state, manifest = module.load_latest_snapshot(local)
    assert state["owner"] == "first"
    assert manifest["transitions"] == 1
    assert first.close(timeout=10) is True
    shutil.rmtree(live_temp)


def _toy_update(torch, net, optimizer, rng, replay, counters):
    """One deterministic learner-like update with replay and RNG consumption."""
    x = torch.from_numpy(rng.normal(size=(8, 8)).astype(np.float32))
    target = torch.from_numpy(rng.normal(size=(8, 6)).astype(np.float32))
    # Consume the torch stream too; restore_full_state must put it back exactly.
    target = target + torch.rand_like(target) * 0.01
    loss = (net(x) - target).square().mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    replay.add(counters["updates"], counters["updates"] % 6, float(loss.detach()), 0.99, -1)
    counters["updates"] += 1
    counters["transitions"] += 8


def test_full_state_payload_restores_cpu_optimizer_replay_and_rng_bitwise(tmp_path):
    """K updates + persisted full state + M updates equals uninterrupted K+M updates."""
    torch = pytest.importorskip("torch")
    from research.redesign_m3c_20261008 import train_m3c as trainer

    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    torch.set_num_threads(1)
    torch.manual_seed(90210)
    random.seed(90210)
    np.random.seed(90210)
    initial = torch.nn.Linear(8, 6).state_dict()

    def fresh():
        net = torch.nn.Linear(8, 6)
        net.load_state_dict(initial)
        return net, torch.optim.Adam(net.parameters(), lr=1e-3), trainer.Replay(32)

    net_a, opt_a, replay_a = fresh()
    ring_a = trainer.ObsRing(32)
    labels_a = trainer.LabelBuffer(16)
    rng_a = np.random.default_rng(44)
    counters_a = {"updates": 0, "transitions": 0}
    schedules = trainer.Schedules()
    for _ in range(2):
        _toy_update(torch, net_a, opt_a, rng_a, replay_a, counters_a)
    payload = trainer.full_state_payload(
        torch,
        net_a,
        net_a,
        opt_a,
        rng_a,
        counters_a,
        schedules,
        replay_a,
        labels_a,
        [ring_a],
        "full",
        extra={
            "actor_rngs": [np.random.default_rng(901).bit_generator.state],
            "league_seat_state": {"world": 0, "seat": 2},
        },
    )
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, payload), transitions=16)
    _wait_close(checkpointer)
    saved, manifest = module.load_latest_snapshot(local)
    assert manifest["transitions"] == 16 and saved["replay_mode"] == "full"

    for _ in range(3):
        _toy_update(torch, net_a, opt_a, rng_a, replay_a, counters_a)

    net_b, opt_b, replay_b = fresh()
    ring_b = trainer.ObsRing(32)
    labels_b = trainer.LabelBuffer(16)
    rng_b = np.random.default_rng(44)
    assert trainer.restore_full_state(
        torch,
        saved,
        net_b,
        net_b,
        opt_b,
        rng_b,
        replay_b,
        labels_b,
        [ring_b],
    )
    counters_b = dict(saved["counters"])
    for _ in range(3):
        _toy_update(torch, net_b, opt_b, rng_b, replay_b, counters_b)

    for left, right in zip(net_a.state_dict().values(), net_b.state_dict().values()):
        assert torch.equal(left, right)
    assert counters_a == counters_b
    assert replay_a.ptr == replay_b.ptr and replay_a.size == replay_b.size
    np.testing.assert_array_equal(replay_a.s, replay_b.s)
    np.testing.assert_array_equal(replay_a.R, replay_b.R)


class _SharedBuffer:
    def __init__(self, size):
        self.buf = bytearray(size)


def _real_learner_fixture(seed=7):
    """Construct the actual M3-C Learner with a deterministic one-actor CPU world."""
    import torch

    from research.redesign_m3c_20261008 import learner_m3c as lm
    from research.redesign_m3c_20261008 import train_m3c as trainer
    from src.model.ego2s_network import Ego2sNet

    cap, nscal = 512, 60
    rng_data = np.random.default_rng(seed)
    ring = trainer.SharedObsRing(_SharedBuffer(trainer.ring_nbytes(cap, nscal)), cap, 16, nscal)
    ring.arrays["local"][:] = rng_data.integers(0, 256, ring.arrays["local"].shape, dtype=np.uint8)
    ring.arrays["global"][:] = rng_data.integers(
        0, 256, ring.arrays["global"].shape, dtype=np.uint8
    )
    ring.arrays["scalars"][:] = rng_data.standard_normal(ring.arrays["scalars"].shape)
    ring.arrays["mask"][:] = rng_data.random(ring.arrays["mask"].shape) < 0.7
    ring._count[0] = cap
    multi = trainer.MultiRing([ring])
    replay = trainer.Replay(2048)
    ids = rng_data.integers(16, cap, 1200)
    replay.add_many(
        ids,
        rng_data.integers(0, 6, len(ids)),
        rng_data.standard_normal(len(ids)),
        np.full(len(ids), 0.97, dtype=np.float32),
        np.full(len(ids), -1),
    )
    labels = trainer.LabelBuffer(trainer.LABEL_ANCHOR_MIN_ROWS + 100)
    labels.add(rng_data.integers(16, cap, labels.cap), rng_data.integers(0, 6, labels.cap))
    args = trainer.default_args()
    args.obs_version = "bv8"
    args.onpolicy_labels = True
    args.label_frac = 1.0
    args.anchor_mode = "v8margin"
    args.actor_sync_updates = 25
    net = Ego2sNet(7, nscal).train()
    target = Ego2sNet(7, nscal).eval()
    target.load_state_dict(net.state_dict())
    optimizer = lm.make_optimizer(torch, net, lm.LearnerOpts(), "cpu")
    st = {
        "updates": 0,
        "halt": None,
        "td": [],
        "anc": [],
        "qmax": 0.0,
        "learn_s": 0.0,
        "ring_overrun_rows": 0,
        "label_rows": [],
        "anchor_updates": 0,
    }
    learner = lm.Learner(
        torch,
        online=net,
        target=target,
        opt=optimizer,
        replay=replay,
        multi=multi,
        demos=None,
        labels=labels,
        rng=np.random.default_rng([seed, 17]),
        st=st,
        args=args,
        nscal=nscal,
        margin=0.3,
        device="cpu",
        opts=lm.LearnerOpts(),
        agent_rows=24,
        anchor_rows=8,
        min_rows=4,
    )
    learner.target_every = 3
    return torch, trainer, learner, ring, replay, labels, st


def test_actual_m3c_learner_full_state_resume_is_bitwise(tmp_path):
    """The production Learner, optimizer, replay, ring, labels, and RNG resume exactly."""
    module = _checkpoint_api()
    torch, trainer, learner_a, ring_a, replay_a, labels_a, st_a = _real_learner_fixture()
    for _ in range(2):
        learner_a.update(0.5)
    payload = trainer.full_state_payload(
        torch,
        learner_a.online,
        learner_a.target,
        learner_a.opt,
        learner_a.rng,
        st_a,
        trainer.Schedules(),
        replay_a,
        labels_a,
        [ring_a],
        "full",
        extra={
            "actor_rngs": [np.random.default_rng(901).bit_generator.state],
            "league_seat_state": {"world": 0, "seat": 2},
        },
    )
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, payload), transitions=256)
    _wait_close(checkpointer)
    saved, _ = module.load_latest_snapshot(local)
    assert saved["extra"]["league_seat_state"] == {"world": 0, "seat": 2}
    assert saved["extra"]["actor_rngs"][0]["state"]

    for _ in range(3):
        learner_a.update(0.5)
    torch_b, trainer_b, learner_b, ring_b, replay_b, labels_b, st_b = _real_learner_fixture()
    assert torch_b is torch and trainer_b is trainer
    assert trainer.restore_full_state(
        torch,
        saved,
        learner_b.online,
        learner_b.target,
        learner_b.opt,
        learner_b.rng,
        replay_b,
        labels_b,
        [ring_b],
    )
    st_b.update(saved["counters"])
    learner_b.st = st_b
    for _ in range(3):
        learner_b.update(0.5)
    for left, right in zip(
        learner_a.online.state_dict().values(), learner_b.online.state_dict().values()
    ):
        assert torch.equal(left, right)
    for left, right in zip(
        learner_a.target.state_dict().values(), learner_b.target.state_dict().values()
    ):
        assert torch.equal(left, right)
    assert {key: value for key, value in st_a.items() if key != "learn_s"} == {
        key: value for key, value in st_b.items() if key != "learn_s"
    }
    assert learner_a.rng.bit_generator.state == learner_b.rng.bit_generator.state
    np.testing.assert_array_equal(replay_a.s, replay_b.s)
    np.testing.assert_array_equal(replay_a.a, replay_b.a)
    np.testing.assert_array_equal(labels_a.ids, labels_b.ids)
    np.testing.assert_array_equal(ring_a.arrays["scalars"], ring_b.arrays["scalars"])


def test_snapshot_manifest_is_json_and_records_digest(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    local.mkdir()
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=7)
    _wait_close(checkpointer)
    manifests = _manifest_files(local)
    assert manifests
    decoded = json.loads(manifests[-1].read_text())
    digests = [entry.get("sha256") for entry in decoded["files"]]
    assert all(isinstance(digest, str) and len(digest) == 64 for digest in digests)
