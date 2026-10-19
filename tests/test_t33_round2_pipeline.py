"""Inference ownership, fp32 head and masked-only qualification contracts."""

import numpy as np
import pytest
import torch

from research.t33_batched_teacher.inference import FixedInference, _HalfRasterForward, agreement
from research.t33_batched_teacher.observations import ObservationBuffer
from src.model.ego2s_network import Ego2sNet


def test_submit_collect_fences_slab_ownership_cpu() -> None:
    torch.set_num_threads(1)
    adapter = FixedInference(Ego2sNet(local_channels=7, n_scalars=60), "cpu")
    buffer = ObservationBuffer(2)
    ticket = adapter.submit(buffer)
    with pytest.raises(RuntimeError, match="pending"):
        adapter.submit(buffer)
    with pytest.raises(ValueError, match="ticket"):
        adapter.collect((1, 2))
    assert adapter.collect(ticket).shape == (2, 6)
    with pytest.raises(ValueError, match="ticket"):
        adapter.collect(ticket)


def test_fp32_dueling_head_under_cpu_autocast() -> None:
    torch.set_num_threads(1)
    net = Ego2sNet(local_channels=7, n_scalars=60).eval()
    seen = []
    hook = net.adv.register_forward_pre_hook(lambda m, args: seen.append(args[0].dtype))
    with torch.inference_mode(), torch.autocast("cpu", dtype=torch.bfloat16):
        q = _HalfRasterForward(net)(
            torch.zeros(2, 7, 31, 31, dtype=torch.uint8),
            torch.zeros(2, 4, 37, 37, dtype=torch.uint8),
            torch.zeros(2, 60),
        )
    hook.remove()
    assert seen == [torch.float32]
    assert q.dtype == torch.float32


def test_illegal_raw_argmax_flips_do_not_fail_qualification() -> None:
    reference = np.tile(np.arange(6, dtype=np.float32), (1000, 1))
    candidate = reference.copy()
    candidate[:, 5] = 100
    masks = np.ones_like(reference, dtype=bool)
    masks[:, 5] = False
    candidate[:, 0] = 200  # also illegal
    masks[:, 0] = False
    report = agreement(reference, candidate, masks)
    assert report["raw_agreement"] == 0.0
    assert report["masked_agreement"] == 1.0
    assert report["threshold_pass"]
