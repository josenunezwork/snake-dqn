"""Tests for the dual-scale raster Dueling network (blueprint P3)."""

from __future__ import annotations

import numpy as np
import torch

from src.model.raster_network import (SCALARS_SHAPE, STRATEGIC_SHAPE,
                                      TACTICAL_SHAPE, RasterDuelingNetwork,
                                      raster_tensors_from_obs)
from src.simd_env.featurizer import SCALARS_DIM


def _rand_inputs(n: int, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    tac = torch.randn(n, *TACTICAL_SHAPE, generator=g)
    strat = torch.randn(n, *STRATEGIC_SHAPE, generator=g)
    scal = torch.randn(n, *SCALARS_SHAPE, generator=g)
    return tac, strat, scal


def test_forward_shape():
    net = RasterDuelingNetwork()
    tac, strat, scal = _rand_inputs(5)
    q = net(tac, strat, scal)
    assert q.shape == (5, 6)


def test_forward_mapping_and_positional_agree():
    net = RasterDuelingNetwork().eval()
    tac, strat, scal = _rand_inputs(3)
    with torch.no_grad():
        q_pos = net(tac, strat, scal)
        q_map = net({"tactical": tac, "strategic": strat, "scalars": scal})
    assert torch.allclose(q_pos, q_map)


def test_param_count_near_target():
    net = RasterDuelingNetwork()
    total = net.get_num_parameters()["total"]
    # Blueprint target ~1.3M; assert within a sane band.
    assert 1_000_000 < total < 2_000_000, total


def test_activations_contract_keys():
    net = RasterDuelingNetwork().eval()
    tac, strat, scal = _rand_inputs(4)
    with torch.no_grad():
        q, acts = net.forward_with_activations(tac, strat, scal)
    # Exact key set the web UI depends on.
    assert set(acts.keys()) == {"input", "hidden", "value", "advantages", "output"}
    assert acts["value"].shape == (4, 1)
    assert acts["advantages"].shape == (4, 6)
    assert acts["output"].shape == (4, 6)
    assert acts["input"].shape == (4, SCALARS_DIM)


def test_dueling_identity():
    """Q == V + (A - mean A) exactly (the dueling contract)."""
    net = RasterDuelingNetwork().eval()
    tac, strat, scal = _rand_inputs(6)
    with torch.no_grad():
        q, acts = net.forward_with_activations(tac, strat, scal)
    v = acts["value"]
    a = acts["advantages"]
    recon = v + (a - a.mean(dim=-1, keepdim=True))
    assert torch.allclose(recon, acts["output"], atol=1e-6)
    # And the returned q matches the output activation.
    assert torch.allclose(q.cpu(), acts["output"], atol=1e-6)


def test_forward_with_activations_qvalues_match_forward():
    net = RasterDuelingNetwork().eval()
    tac, strat, scal = _rand_inputs(3)
    with torch.no_grad():
        q1 = net(tac, strat, scal)
        q2, _ = net.forward_with_activations(tac, strat, scal)
    assert torch.allclose(q1, q2, atol=1e-6)


def test_raster_tensors_from_obs_shapes():
    """The uint8->float helper flattens (E,S) and expands the tactical planes."""
    E, S = 2, 3
    obs = {
        "tactical_uint8": np.zeros((E, S, 2, 31, 31), dtype=np.uint8),
        "strategic_uint8": np.zeros((E, S, 3, 25, 25), dtype=np.uint8),
        "scalars": np.zeros((E, S, SCALARS_DIM), dtype=np.float32),
    }
    out = raster_tensors_from_obs(obs)
    assert out["tactical"].shape == (E * S, *TACTICAL_SHAPE)
    assert out["strategic"].shape == (E * S, *STRATEGIC_SHAPE)
    assert out["scalars"].shape == (E * S, SCALARS_DIM)
    assert out["tactical"].dtype == torch.float32


def test_gradients_flow():
    net = RasterDuelingNetwork()
    tac, strat, scal = _rand_inputs(4)
    q = net(tac, strat, scal)
    q.sum().backward()
    grads = [p.grad for p in net.parameters() if p.grad is not None]
    assert len(grads) > 0
    assert all(torch.isfinite(g).all() for g in grads)
