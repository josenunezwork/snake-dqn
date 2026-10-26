import argparse
import json

import numpy as np
import torch

from research.sandbox_b_heads.job import _served, metrics, parent_split, run
from research.sandbox_b_heads.model import make_head


def test_parent_split_is_stable_and_parent_scoped():
    parents = np.array(["a", "a", "b", "c"], dtype="U1")
    first, second = parent_split(parents), parent_split(parents)
    assert np.array_equal(first, second)
    assert first[0] == first[1]


def test_all_declared_heads_are_six_action_finite_residuals():
    features = torch.randn(4, 256)
    length, margin = torch.rand(4), torch.rand(4)
    for name in ("linear", "gated_mlp", "cosine"):
        result = make_head(name)(features, length, margin)
        assert result.shape == (4, 6)
        assert torch.isfinite(result).all()


def test_metrics_reports_rescue_and_ordinary_harm_proxy():
    table = {
        "q": torch.zeros(2, 6),
        "action_mask": torch.ones(2, 6, dtype=torch.bool),
        "bare": torch.tensor([0, 0]),
        "legal": torch.ones(2, 6, dtype=torch.bool),
        "died": torch.tensor(
            [[True, False, False, False, False, False], [False, True, False, False, False, False]]
        ),
        "ordinary": torch.tensor([False, True]),
        "status": torch.ones(2, 6, dtype=torch.bool),
    }
    corrected = torch.tensor([[0.0, 2.0, 0, 0, 0, 0], [0.0, 2.0, 0, 0, 0, 0]])
    report = metrics(corrected, table)
    assert report["rescue_retention"] == 1.0
    assert report["ordinary_harm_proxy"] == 1.0
    assert _served(corrected, table["action_mask"]).tolist() == [1, 1]


def test_tiny_cpu_entrypoint_exports_all_config_hashes(tmp_path):
    """Exercise the real finite sweep with synthetic, non-registry fixture data."""
    from src.model.ego2s_network import Ego2sNet

    candidate_parents = np.array([f"synthetic-{i}" for i in range(50)], dtype="U16")
    split = parent_split(candidate_parents)
    parents = np.array(
        [candidate_parents[np.where(~split)[0][0]], candidate_parents[np.where(split)[0][0]]],
        dtype="U16",
    )
    n = 2
    table = {
        "local": np.zeros((n, 7, 31, 31), np.uint8),
        "global": np.zeros((n, 4, 37, 37), np.uint8),
        "scalars": np.zeros((n, 60), np.float32),
        "q": np.zeros((n, 6), np.float32),
        "legal": np.ones((n, 6), bool),
        "action_mask": np.ones((n, 6), bool),
        "bare": np.zeros(n, np.int64),
        "died": np.array(
            [[True, False, False, False, False, False], [False, True, False, False, False, False]]
        ),
        "status": np.full((n, 6), "EXACT", dtype="U13"),
        "mass": np.ones((n, 6), np.float32),
        "ordinary": np.array([False, True]),
        "natural_stream": np.array([False, True]),
        "sampling_weight": np.ones(n, np.float32),
        "parent_world": parents,
        "life_id": np.array(["life-a", "life-b"]),
        "row_id": np.array(["row-a", "row-b"]),
    }
    data, b8, output = tmp_path / "synthetic.npz", tmp_path / "b8.pt", tmp_path / "out"
    np.savez(data, **table)
    torch.save(Ego2sNet(7, 60).state_dict(), b8)
    result = run(
        argparse.Namespace(
            input=str(data),
            b8_checkpoint=str(b8),
            output=str(output),
            namespace="synthetic",
            salt="a100-dev/sandbox-b/split-v1",
            device="cpu",
            updates=1,
            batch_size=2,
            learning_rate=1e-3,
            seed=3,
        )
    )
    payload = json.loads(result.read_text())
    assert len(payload["rows"]) == 60
    assert len(list(output.glob("*.pt.sha256"))) == 60
