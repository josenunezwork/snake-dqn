"""Focused tests for the T21 full-state snapshot boundary.

These tests deliberately exercise the public, format-independent checkpoint API.  The
trainer owns the contents of ``state``; the checkpointer owns detached copies, atomic
publication, checksums, retention, and best-effort durable publication.
"""

from __future__ import annotations

import json
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
    assert manifest["sha256"]
    assert _manifest_files(local)
    assert _manifest_files(durable)
    assert not any(p.name.endswith(".tmp") for p in tmp_path.rglob("*"))


def test_retention_keeps_last_two_complete_snapshots(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    for transitions in (1, 2, 3):
        state = {"transitions": transitions, "values": np.array([transitions], dtype=np.int64)}
        assert checkpointer.submit(_capture_state(module, state), transitions=transitions)
    _wait_close(checkpointer)

    loaded, manifest = module.load_latest_snapshot(local)
    assert loaded["transitions"] == 3
    assert manifest["transitions"] == 3
    assert len(_manifest_files(local)) <= 2
    assert len(_payload_files(local)) <= 2


def test_corrupt_snapshot_is_rejected(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
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


def test_snapshot_manifest_is_json_and_records_digest(tmp_path):
    module = _checkpoint_api()
    local = tmp_path / "local"
    checkpointer = module.AsyncCheckpointer(local, keep=2, retries=1, retry_delay=0)
    assert checkpointer.submit(_capture_state(module, {"value": np.arange(3)}), transitions=7)
    _wait_close(checkpointer)
    manifests = _manifest_files(local)
    assert manifests
    decoded = json.loads(manifests[-1].read_text())
    assert isinstance(decoded.get("sha256"), str) and len(decoded["sha256"]) == 64
