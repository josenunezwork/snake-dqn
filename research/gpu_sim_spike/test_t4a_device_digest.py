"""The diagnostic ring digest transfers compact integers; raw comparison is authoritative."""

import pytest
import torch

from research.gpu_sim_spike import t4a_compare as comparison


def test_body_digest_is_position_sensitive_and_transfers_only_32_bytes(monkeypatch):
    original = torch.Tensor.cpu
    transfers = []

    def counted(value, *args, **kwargs):
        transfers.append(value.numel() * value.element_size())
        return original(value, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "cpu", counted)
    body = torch.arange(4 * 6 * 513 * 2, dtype=torch.int64).reshape(4, 6, 513, 2)
    desc = comparison.compact_inventory({"sim.bodies": body})["sim.bodies"]
    assert transfers == [32]
    assert desc["digest_algorithm"] == comparison.BODY_DIGEST_ALGORITHM
    assert "sha256" not in desc
    assert len(desc["digest"]) == 64
    assert comparison.body_digest(body.clone()) == desc["digest"]
    swapped = body.clone()
    swapped.flatten()[0], swapped.flatten()[1] = body.flatten()[1], body.flatten()[0]
    assert comparison.body_digest(swapped) != desc["digest"]
    assert comparison.compare(body, swapped, "sim.bodies") is not None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA host required")
def test_body_digest_cpu_cuda_integer_identity():
    values = torch.arange(1024, dtype=torch.int64).reshape(4, 256)
    assert comparison.body_digest(values) == comparison.body_digest(values.cuda())
