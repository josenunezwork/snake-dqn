"""Leakage, branch weighting and raw-input contracts for the diagnostic CNN."""

import numpy as np
import pytest
import torch

pytest.importorskip("sklearn")
from research.t8_probes_20261017.cnn import DeathCNN  # noqa: E402
from research.t8_probes_20261017.cnn import inner_split, masked_loss


def test_nested_split_keeps_entire_worlds_outside_training_and_validation() -> None:
    world = np.repeat(np.arange(30).astype(str), 3)
    fold = np.repeat(np.arange(30) % 5, 3)
    for current in range(5):
        train, valid, test = inner_split(world, fold, current)
        sets = [set(world[rows]) for rows in (train, valid, test)]
        assert not sets[0] & sets[1] and not sets[0] & sets[2] and not sets[1] & sets[2]
        assert len(train) + len(valid) + len(test) == len(world)
        assert np.all(fold[test] == current)
        again = inner_split(world, fold, current)
        for a, b in zip((train, valid, test), again):
            np.testing.assert_array_equal(a, b)


def test_loss_ignores_illegal_branches_and_weights_each_legal_branch_equally() -> None:
    logits = torch.tensor([[0.0, 1.0, -5.0], [2.0, 3.0, 8.0]], requires_grad=True)
    legal = torch.tensor([[True, False, False], [True, True, False]])
    labels = torch.tensor([[0.0, 1.0, 1.0], [1.0, 0.0, 0.0]])
    loss = masked_loss(logits, labels, legal)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        torch.tensor([0.0, 2.0, 3.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert torch.count_nonzero(logits.grad[~legal]) == 0


def test_small_cnn_uses_both_rasters_scalars_and_explicit_action() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(1)
    net = DeathCNN().eval()
    assert 200_000 < sum(p.numel() for p in net.parameters()) < 500_000
    local = torch.rand(2, 7, 31, 31, requires_grad=True)
    glob = torch.rand(2, 4, 37, 37, requires_grad=True)
    scalars = torch.rand(2, 60, requires_grad=True)
    logits = net(local, glob, scalars)
    assert logits.shape == (2, 6)
    assert not torch.allclose(logits[:, 0], logits[:, 1])
    logits.sum().backward()
    assert local.grad.abs().sum() > 0
    assert glob.grad.abs().sum() > 0
    assert scalars.grad.abs().sum() > 0
