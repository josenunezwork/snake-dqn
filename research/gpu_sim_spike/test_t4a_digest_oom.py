"""Frozen-c9fe digest identity and device-independent scratch/lifetime proofs."""

import hashlib
import importlib.util
import weakref
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_flatten

from research.gpu_sim_spike import t4a_bench as bench
from research.gpu_sim_spike import t4a_compare as candidate
from research.gpu_sim_spike import t4a_events as events

REFERENCE = Path(__file__).resolve().parents[2] / "tests/fixtures/t4a_oom/t4a_compare_c9fe.py"
REFERENCE_SHA256 = "95a0485e16f2ea58928962849ab88f8525cde60233bf5f4474537d62d8d2f5e2"


@pytest.fixture(scope="module")
def frozen():
    assert hashlib.sha256(REFERENCE.read_bytes()).hexdigest() == REFERENCE_SHA256
    spec = importlib.util.spec_from_file_location("frozen_c9fe_compare", REFERENCE)
    module = importlib.util.module_from_spec(spec)
    # Dataclass resolves annotations through sys.modules.
    import sys

    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("size", [0, 1, 7, 8, 31, 63, 64, 65, 257, 2047, 1024**2 + 17])
def test_frozen_bytes_boundaries_tails_and_overflow(frozen, monkeypatch, size):
    values = (torch.arange(size, dtype=torch.int64) * 197 + 255).to(torch.uint8)
    # Small windows force many boundaries; the large case uses the real bound.
    monkeypatch.setattr(candidate, "BODY_DIGEST_CHUNK_BYTES", 64 if size < 2048 else 1024**2)
    assert candidate.body_digest(values) == frozen.body_digest(values)
    assert candidate.BODY_DIGEST_ALGORITHM == frozen.BODY_DIGEST_ALGORITHM


@pytest.mark.parametrize("kind", ["signed", "noncontiguous", "float_bits", "all_bytes"])
def test_frozen_representations(frozen, monkeypatch, kind):
    values = {
        "signed": torch.tensor([-(2**63), 2**63 - 1, -1, 0, 1], dtype=torch.int64),
        "noncontiguous": torch.arange(120, dtype=torch.int64).reshape(4, 30).T,
        "float_bits": torch.tensor([0.0, -0.0, float("nan"), float("inf"), -float("inf")]),
        "all_bytes": torch.arange(256).to(torch.uint8),
    }[kind]
    monkeypatch.setattr(candidate, "BODY_DIGEST_CHUNK_BYTES", 13)
    assert candidate.body_digest(values) == frozen.body_digest(values)


@pytest.mark.parametrize("offset", [1, 7, 127, 65537])
def test_global_offset_chunk_is_frozen_prefix_difference(frozen, offset):
    prefix = torch.full((offset,), 255, dtype=torch.uint8)
    raw = torch.arange(259).to(torch.uint8)
    before = bytes.fromhex(frozen.body_digest(prefix))
    after = bytes.fromhex(frozen.body_digest(torch.cat((prefix, raw))))
    expected = b"".join(
        (
            (
                int.from_bytes(after[i : i + 8], "little")
                - int.from_bytes(before[i : i + 8], "little")
            )
            % 2**64
        ).to_bytes(8, "little")
        for i in range(0, 32, 8)
    )
    actual = candidate._body_digest_chunk(raw, offset).view(torch.uint8).numpy().tobytes()
    assert actual == expected


class AllocationAudit(TorchDispatchMode):
    """Measure explicit live tensor payloads; backend workspace is separate."""

    def __init__(self):
        super().__init__()
        self.live = {}
        self.peak = self.largest = 0

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        output = func(*args, **(kwargs or {}))
        for tensor in tree_flatten(output)[0]:
            if isinstance(tensor, torch.Tensor) and id(tensor) not in self.live:
                key = id(tensor)
                size = tensor.numel() * tensor.element_size()
                self.live[key] = (
                    weakref.ref(tensor, lambda _, k=key: self.live.pop(k, None)),
                    size,
                )
                self.largest = max(self.largest, size)
        self.peak = max(self.peak, sum(size for _, size in self.live.values()))
        return output


def test_real_chunk_peak_tensor_scratch_bound():
    raw = torch.arange(candidate.BODY_DIGEST_CHUNK_BYTES).to(torch.uint8)
    with AllocationAudit() as audit:
        lanes = candidate._body_digest_chunk(raw, 1022361600 - raw.numel())
    assert lanes.shape == (4,)
    assert audit.largest <= 8 * candidate.BODY_DIGEST_CHUNK_BYTES
    assert audit.peak <= candidate.BODY_DIGEST_SCRATCH_BYTES


def test_full_e1024_ring_walk_has_fixed_scratch_and_no_skipped_bytes(monkeypatch):
    # Full production shape with no payload allocation. The real chunk kernel's
    # peak is measured above; this verifies its call schedule at the OOM shape.
    ring = torch.empty((1024, 6, 10400, 2), dtype=torch.int64, device="meta")
    cursor = 0
    maximum = 0

    def chunk(raw, offset):
        nonlocal cursor, maximum
        assert offset == cursor
        assert raw.dtype == torch.uint8
        assert 0 < raw.numel() <= candidate.BODY_DIGEST_CHUNK_BYTES
        cursor += raw.numel()
        maximum = max(maximum, raw.numel())
        return torch.zeros(4, dtype=torch.int64, device="meta")

    monkeypatch.setattr(candidate, "_body_digest_chunk", chunk)
    assert candidate._body_digest_lanes(ring).shape == (4,)
    assert cursor == 1022361600
    assert maximum == candidate.BODY_DIGEST_CHUNK_BYTES


@pytest.mark.parametrize("fail", [False, True])
def test_event_owned_tapes_cleaned_and_witness_immutable(monkeypatch, fail):
    original = dict(events._TAPES)
    tensor = torch.tensor([17], dtype=torch.uint8)
    witness = {"_raw_mapped_tape": [{"bytes": tensor.clone()}]}

    def recorded(*args):
        events._TAPES[-1001] = [{"bytes": tensor}]
        events._TAPES[-1002] = [{"bytes": tensor.clone()}]
        if fail:
            raise RuntimeError("synthetic diagnostic OOM")
        return True, witness

    monkeypatch.setattr(events, "_run_one_recorded", recorded)
    try:
        if fail:
            with pytest.raises(RuntimeError, match="synthetic diagnostic OOM"):
                events._run_one("clone_restore", Path("unused"), "frozen", 4, "cpu", "smoke")
        else:
            assert (
                events._run_one("clone_restore", Path("unused"), "frozen", 4, "cpu", "smoke")[1]
                is witness
            )
            tensor[0] = 99
            assert witness["_raw_mapped_tape"][0]["bytes"].item() == 17
        assert set(events._TAPES) == set(original)
    finally:
        events._TAPES.clear()
        events._TAPES.update(original)


@pytest.mark.parametrize("telemetry_failure", [False, True])
def test_memory_telemetry_cannot_replace_scientific_result(
    tmp_path, monkeypatch, telemetry_failure
):
    result = {"status": "INVALID_SELF_CONSISTENCY", "first_witness": {"byte": 1}}
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "reset_peak_memory_stats", lambda *a: None)
    monkeypatch.setattr(bench, "_run", lambda args: result)

    def peak(*args):
        if telemetry_failure:
            raise RuntimeError("synthetic stats failure")
        return 123

    monkeypatch.setattr(torch.cuda, "max_memory_allocated", peak)
    monkeypatch.setattr(torch.cuda, "max_memory_reserved", lambda *a: 256)
    monkeypatch.setattr(
        torch.cuda, "get_device_properties", lambda *a: SimpleNamespace(total_memory=512)
    )
    assert bench.run(SimpleNamespace(device="cuda", output_dir=tmp_path)) is result
    import json

    memory = json.loads((tmp_path / "device-memory.json").read_text())
    assert memory["status"] == ("UNAVAILABLE" if telemetry_failure else "MEASURED")
    # A create-only export failure is also observational only.
    assert bench.run(SimpleNamespace(device="cuda", output_dir=tmp_path)) is result
