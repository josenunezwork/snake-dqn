"""Focused E8 complete-life critic-refit evidence tests."""

from __future__ import annotations

import torch

from research.bl_ppo.refit import CompleteLifeRefitBank


def test_bank_uses_complete_gamma_one_lives_and_world_disjoint_baseline() -> None:
    bank = CompleteLifeRefitBank()
    # World 1 is fit; its terminal suffixes are 6, 5, 3.  World 0 is held out.
    bank.add_batch(
        torch.tensor([1, 0, 1, 0, 1, 0]),
        torch.zeros((6, 257)),
        torch.tensor([800, 800, 900, 900, 1000, 1000]),
        torch.tensor([1.0, 1.0, 0.8, 0.8, 0.6, 0.6]),
        torch.tensor([1.0, 1.0, 2.0, 2.0, 3.0, 3.0]),
        torch.tensor([False, False, False, False, True, True]),
    )
    assert bank.counts() == {"fit": 3, "holdout": 3}
    assert sorted(value for _, _, value in bank._reservoir["fit"]) == [3.0, 5.0, 6.0]
    assert sorted(value for _, _, value in bank._reservoir["holdout"]) == [3.0, 5.0, 6.0]


def test_bank_excludes_open_life_from_counts() -> None:
    bank = CompleteLifeRefitBank()
    bank.add_batch(
        torch.tensor([0, 0]),
        torch.zeros((2, 257)),
        torch.tensor([800, 900]),
        torch.tensor([1.0, 0.8]),
        torch.tensor([1.0, 1.0]),
        torch.tensor([False, False]),
    )
    assert bank.counts() == {"fit": 0, "holdout": 0}
    assert bank.support_counts() == {"fit": 0, "holdout": 0}
