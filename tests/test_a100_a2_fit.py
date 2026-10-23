"""Fast synthetic contract tests for the registered A2 fit core."""

from __future__ import annotations

import numpy as np
import torch

from research.a100_a2.model import HORIZONS, A2Predictor, m_parameter_count
from research.a100_a2.run import _choose_action, _hazard_loss, _ordered_training_rows, fit_unit
from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint


def _data(n: int = 20) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(3)
    world = np.asarray([f"world-{i // 2}" for i in range(n)])
    fold = np.asarray([(i // 2) % 5 for i in range(n)], dtype=np.int64)
    died = rng.integers(0, 2, (n, 6), dtype=np.uint8)
    alive = rng.integers(0, 300, (n, 6), dtype=np.int16)
    alive[~died.astype(bool)] = 300
    return {
        "local": rng.integers(0, 256, (n, 7, 31, 31), dtype=np.uint8),
        "global": rng.integers(0, 256, (n, 4, 37, 37), dtype=np.uint8),
        "scalars": rng.normal(size=(n, 60)).astype(np.float32),
        "world": world,
        "fold": fold,
        "legal": np.ones((n, 6), dtype=bool),
        "died": died,
        "alive_steps": alive,
        "observed": np.ones((n, 6), dtype=bool),
        "decisive": np.zeros(n, dtype=bool),
        "a_star": np.zeros(n, dtype=np.int64),
        "q": rng.normal(size=(n, 6)).astype(np.float32),
    }


def test_nested_world_subsets_exclude_heldout_fold():
    data = _data()
    a, b, c = (_ordered_training_rows(data, 0, dose) for dose in (25, 50, 100))
    assert set(a).issubset(b) and set(b).issubset(c)
    assert not np.any(data["fold"][c] == 0)


def test_m_shape_and_registered_scale():
    net = A2Predictor("m_random", "hazard")
    out = net(
        torch.zeros(2, 7, 31, 31, dtype=torch.uint8),
        torch.zeros(2, 4, 37, 37, dtype=torch.uint8),
        torch.zeros(2, 60),
    )
    assert out.shape == (2, 6, len(HORIZONS))
    assert 6_000_000 <= m_parameter_count() <= 7_000_000


def test_tiny_hazard_fit_is_deterministic_and_heldout_only(tmp_path):
    checkpoint = tmp_path / "b8.pth"
    save_ego2s_checkpoint(checkpoint, Ego2sNet(7, 60))
    data = _data()
    first = fit_unit(
        data, str(checkpoint), "c0", "hazard", 25, 11, 0, "cpu", epochs=1, batch_size=2
    )
    second = fit_unit(
        data, str(checkpoint), "c0", "hazard", 25, 11, 0, "cpu", epochs=1, batch_size=2
    )
    assert np.array_equal(first["row_index"], np.flatnonzero(data["fold"] == 0))
    assert first["p_hazards"].shape[1:] == (6, 5)
    assert np.all(np.diff(first["p_hazards"], axis=-1) >= -1e-6)
    assert np.array_equal(first["p_d300"], first["p_hazards"][:, :, -1])
    assert np.array_equal(first["p_d300"], second["p_d300"])


def test_hazard_death_at_30_observes_only_its_first_interval():
    logits = torch.zeros(1, 6, 5)
    legal = torch.zeros(1, 6, dtype=torch.bool)
    legal[0, 0] = True
    died = torch.zeros(1, 6, dtype=torch.bool)
    died[0, 0] = True
    alive = torch.zeros(1, 6, dtype=torch.int64)
    alive[0, 0] = 29  # registered event interval = alive_steps + 1 = 30
    observed = legal.clone()
    expected = torch.log(torch.tensor(2.0)).item()
    assert _hazard_loss(logits, legal, died, alive, observed).item() == expected


def test_choice_masks_illegal_unsearched_and_falls_back_when_q_is_unknown():
    death = torch.tensor(
        [[0.1, 0.4, 0.2, 0.6, 0.1, 0.3], [0.2] * 6, [0.8, 0.2, 0.3, 0.9, 0.9, 0.9]]
    )
    legal = torch.tensor([[True, True, True, True, False, True], [True] * 6, [True] * 6])
    observed = torch.tensor([[True, False, True, True, True, True], [True] * 6, [True] * 6])
    q = torch.tensor(
        [[99.0, 100.0, 3.0, 3.0, 101.0, 2.0], [float("nan")] * 6, [0.0, 7.0, 7.0, 0.0, 0.0, 0.0]]
    )
    # Row 0: action 0's Q wins among thresholded, legal, observed survivors.
    # Row 1: all Q values are unavailable (T5); highest survival then lowest tie wins.
    # Row 2: Q tie among survivors resolves to the lowest action index.
    assert _choose_action(death, legal, observed, q).tolist() == [0, 0, 1]
