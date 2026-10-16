"""Focused format and sampling tests for the T4d branch-return bank."""

import json

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008.learner_m3c import return_regression_loss
from research.redesign_m3c_20261008.obs_pack import PackLayout
from research.t4d_return_regression_20261016.bank import ReturnBank, realized_returns


def _write_bank(tmp_path, *, died=False, terminal_mask=True):
    layout = PackLayout(rect_cells=1)
    n = 2
    manifest = {
        "schema_version": 1,
        "gamma": 0.998,
        "kappa": 1.0,
        "phi": "length/10,dead=0",
        "layout": layout.to_dict(),
        "packing_version": "ego2s-bv8/obs_pack-v1",
        "reward_semantics": "mass+pbrs+death_no_overlays_or_kill",
    }
    bank_path = tmp_path / "rows.npz"
    np.savez(
        bank_path,
        manifest_json=np.array(json.dumps(manifest)),
        state_packed=np.zeros((n, layout.nbytes), np.uint8),
        state_scalars=np.zeros((n, 60), np.float32),
        action=np.array([0, 5], np.int64),
        T=np.array([2, 3], np.int32),
        died=np.full(n, died, np.bool_),
        S=np.array([11.0, 12.0], np.float64),
        L0=np.array([100, 120], np.int32),
        L_T=np.zeros(n, np.int32) if died else np.array([110, 130], np.int32),
        terminal_packed=np.zeros((n, layout.nbytes), np.uint8),
        terminal_scalars=np.zeros((n, 60), np.float32),
        terminal_mask=np.full((n, 6), terminal_mask, np.bool_),
        row_weight=np.array([1.0, 0.5], np.float32),
        family=np.array([0, 1], np.uint8),
    )
    import hashlib

    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                **manifest,
                "shards": {bank_path.name: hashlib.sha256(bank_path.read_bytes()).hexdigest()},
            }
        )
    )
    return bank_path


def test_return_bank_loads_flattened_rows_and_samples(tmp_path):
    bank = ReturnBank(_write_bank(tmp_path), 60)
    rows = bank.sample(np.random.default_rng(3), 4, kappa=1.0)
    assert bank.size == 2
    assert rows["local"].shape == (4, 7, 31, 31)
    assert rows["terminal_mask"].shape == (4, 6)
    assert rows["weight"].dtype == np.float32
    assert rows["realised"].dtype == np.float32


def test_return_bank_allows_empty_survivor_advisory_mask_for_trainer_fallback(tmp_path):
    bank = ReturnBank(_write_bank(tmp_path, terminal_mask=False), 60)
    assert not bank.terminal_mask.any()


def test_return_bank_rejects_nonempty_dead_terminal_mask(tmp_path):
    with pytest.raises(ValueError, match="dead return-bank rows"):
        ReturnBank(_write_bank(tmp_path, died=True, terminal_mask=True), 60)


def test_realized_returns_matches_closed_form():
    got = realized_returns(
        np.array([2, 3]),
        np.array([False, True]),
        np.array([21.0, 7.0]),
        np.array([100, 100]),
        np.array([120, 0]),
        0.9,
        2.0,
    )
    expected = np.array([2 * 0.1 * 21 - 10 + 0.9**2 * 12, 2 * 0.1 * 7 - 10 - 3 * 0.9**2])
    np.testing.assert_allclose(got, expected)


class _ConstantQ(torch.nn.Module):
    def __init__(self, values):
        super().__init__()
        self.values = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))

    def forward(self, local, glob, scalars):
        return self.values.expand(len(local), -1)


def test_return_loss_is_weighted_masked_double_dqn_and_detaches_terminal_value():
    online = _ConstantQ([0, 1, 2, 9, 3, 4])
    # Target action 2 exceeds target action 5; selecting 5 proves Double DQN
    # selected with online values, then gathered the target value.
    target = _ConstantQ([10, 11, 99, 13, 14, 15])
    rows = {
        "local": torch.zeros((3, 7, 31, 31), dtype=torch.uint8),
        "global": torch.zeros((3, 4, 37, 37), dtype=torch.uint8),
        "scalars": torch.zeros((3, 60)),
        "action": torch.tensor([0, 0, 0]),
        "T": torch.tensor([2, 3, 1]),
        "died": torch.tensor([False, True, False]),
        "realised": torch.tensor([1.0, 2.0, 3.0]),
        "terminal_local": torch.zeros((3, 7, 31, 31), dtype=torch.uint8),
        "terminal_global": torch.zeros((3, 4, 37, 37), dtype=torch.uint8),
        "terminal_scalars": torch.zeros((3, 60)),
        "terminal_mask": torch.tensor(
            [[False, False, True, False, True, True], [False] * 6, [False] * 6]
        ),
        "weight": torch.tensor([2.0, 0.5, 1.0]),
    }
    loss, y = return_regression_loss(torch, online, target, rows, gamma=0.9, delta=100.0)
    expected_y = torch.tensor([1.0 + 0.9**2 * 15.0, 2.0, 3.0 + 0.9 * 99.0])
    torch.testing.assert_close(y, expected_y)
    expected = (
        ((0.0 - expected_y[0]) ** 2 / 2 * 2.0) + (2.0**2 / 2 * 0.5) + expected_y[2] ** 2 / 2
    ) / 3
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert target.values.grad is None
    torch.testing.assert_close(
        online.values.grad,
        torch.tensor([-expected_y.mul(rows["weight"]).sum() / 3, 0, 0, 0, 0, 0]),
    )


def test_return_bank_hash_binding_rejects_mutation(tmp_path):
    path = _write_bank(tmp_path)
    with path.open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        ReturnBank(path, 60)


def test_return_sample_uses_runtime_kappa_and_no_float64_device_fields(tmp_path):
    bank = ReturnBank(_write_bank(tmp_path), 60)
    left = bank.sample(np.random.default_rng(4), 6, kappa=0.0)
    right = bank.sample(np.random.default_rng(4), 6, kappa=2.0)
    assert np.all(right["realised"] > left["realised"])
    assert not any(value.dtype == np.float64 for value in right.values())


def test_cli_default_return_options_are_inert():
    from research.redesign_m3c_20261008.train_m3c import build_parser, m3c_flags

    args = build_parser().parse_args(
        ["--seed", "9", "--init", "unused", "--demo-data", "unused", "--out", "unused"]
    )
    assert args.return_bank is None and args.return_frac == 0
    assert not any(key.startswith("return_") for key in m3c_flags(args))
