"""T4a two-sided bank routing: raw provenance, ratio sampling, and both losses."""

from pathlib import Path

import numpy as np
import pytest
import torch

from research.redesign_m3c_20261008 import learner_m3c as lm
from research.redesign_m3c_20261008 import train_m3c as m3c
from research.t3_rollout_labels_20261015.bank import LabelBank
from src.model.ego2s_network import Ego2sNet
from tests.test_perf_learner import _args, _assert_same, _ingest, _st, _world


def _write_v2_bank(path: Path) -> LabelBank:
    n = 4
    # The first two rows emulate v2's preferred labels: B8 action 3 survived in
    # the raw outcome, but preference removed it from the derived surviving set.
    original = np.array(
        [[1, 0, 0, 1, 0, 0], [0, 1, 0, 1, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
        dtype=bool,
    )
    np.savez(
        path / "bank-v2.npz",
        local=np.zeros((n, 7, 31, 31), np.uint8),
        **{"global": np.zeros((n, 4, 37, 37), np.uint8)},
        scalars=np.zeros((n, 60), np.float32),
        mask=np.ones((n, 6), bool),
        surviving=np.array(
            [[1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0], [1, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0]],
            bool,
        ),
        surviving_original=original,
        action=np.array([3, 3, 3, 3], np.int64),
        weight=np.array([2.0, 3.0, 1.0, 1.0], np.float32),
    )
    return LabelBank(path, 60)


def test_v2_counter_identity_uses_original_outcomes_and_ratio(tmp_path: Path) -> None:
    bank = _write_v2_bank(tmp_path)
    assert bank.counter.tolist() == [True, True, False, False]
    rows = bank.sample(np.random.default_rng(3), 8, counter_ratio=1.0)
    assert rows["counter"].sum() == 4
    chosen = rows["surviving_original"][rows["counter"], rows["action"][rows["counter"]]]
    assert chosen.all()


def test_two_sided_losses_use_all_actions_and_bank_weights() -> None:
    q = torch.tensor(
        [[0.0, 2.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]],
        requires_grad=True,
    )
    db = {
        "surviving": torch.tensor(
            [[True, False, False, False, False, False], [True, False, False, False, False, False]]
        ),
        "weight": torch.tensor([1.0, 2.0]),
        "action": torch.tensor([0, 3]),
        "counter": torch.tensor([False, True]),
        "a_v8": torch.tensor([], dtype=torch.long),
        "mask_resolved": torch.empty((0, 6), dtype=torch.bool),
    }
    anchor, bank_loss, agree = lm._bank_anchor_terms(
        torch, q, db, n_bank=2, margin=0.3, scale=1.0, counter_mode="single-action"
    )
    # Decisive: 2.3. Counter: (1.0 + .3 - 0.0) * weight 2 = 2.6.  Mean = 2.45.
    assert float(anchor.detach()) == pytest.approx(2.45)
    assert float(bank_loss.detach()) == pytest.approx(2.45)
    bank_loss.backward()
    torch.testing.assert_close(q.grad, torch.tensor([[-0.5, 0.5, 0, 0, 0, 0], [0, 0, 0, -1, 1, 0]]))
    assert float(agree) == pytest.approx(0.0)
    _, decisive, counter = lm._bank_side_agreement(q, db["surviving"], db["action"], db["counter"])
    assert float(decisive) == 0.0 and float(counter) == 0.0

    saturated = torch.tensor([[0.0, 0.0, 0.0, 2.0, 0.0, 0.0]], requires_grad=True)
    saturated_db = {
        **db,
        "surviving": db["surviving"][1:],
        "weight": db["weight"][1:],
        "action": db["action"][1:],
        "counter": db["counter"][1:],
    }
    _, saturated_loss, _ = lm._bank_anchor_terms(
        torch, saturated, saturated_db, 1, 0.3, 1.0, "single-action"
    )
    saturated_loss.backward()
    assert float(saturated_loss.detach()) == 0.0 and not saturated.grad.any()


def test_side_metrics_preserve_preferred_agreement_and_missing_side_is_none() -> None:
    # v2 preference removed B8's surviving boost action 3 from the selected set.
    q = torch.tensor([[1.0, 0.0, 0.0, 2.0, 0.0, 0.0]])
    surviving = torch.tensor([[True, False, False, False, False, False]])
    overall, decisive, counter = lm._bank_side_agreement(
        q, surviving, torch.tensor([3]), torch.tensor([True])
    )
    assert float(overall) == 0.0 and decisive is None and float(counter) == 1.0


def test_ratio_validation_and_empty_sample_are_stable(tmp_path: Path) -> None:
    bank = _write_v2_bank(tmp_path)
    assert bank.sample(np.random.default_rng(1), 0, counter_ratio=1.0)["counter"].shape == (0,)
    with pytest.raises(ValueError, match="counter ratio"):
        bank.validate_counter_ratio(float("nan"))
    with pytest.raises(SystemExit, match="counter-ratio needs"):
        m3c.main(
            [
                "--seed",
                "0",
                "--init",
                "x",
                "--demo-data",
                ".",
                "--out",
                "x",
                "--label-bank-counter-ratio",
                "1",
            ]
        )


@pytest.mark.parametrize("fraction", [1.0, 0.5])
def test_two_sided_cpu_reference_and_fast_match(tmp_path: Path, fraction: float) -> None:
    """Exercise bank-only and mixed label-bank updates with both learner paths."""
    torch.set_num_threads(1)
    bank = _write_v2_bank(tmp_path)

    def run(fast: bool):
        torch.manual_seed(29)
        nscal, multi, replay, labels, demos = _world("bv8", seed=41)
        if fraction == 1.0:
            labels = None
        online, target = Ego2sNet(7, nscal), Ego2sNet(7, nscal)
        target.load_state_dict(online.state_dict())
        opts = lm.LearnerOpts(fast=fast, fused_adam=False)
        stats, rng = _st(), np.random.default_rng([19, 23])
        learner = lm.Learner(
            torch,
            online=online,
            target=target,
            opt=lm.make_optimizer(torch, online, opts, "cpu"),
            replay=replay,
            multi=multi,
            demos=demos,
            labels=labels,
            rng=rng,
            st=stats,
            args=_args(
                "bv8",
                label_bank_frac=fraction,
                label_frac=1.0 - fraction,
                label_bank_counter_mode="single-action",
                label_bank_counter_ratio=1.0,
                onpolicy_labels=fraction != 1.0,
            ),
            nscal=nscal,
            margin=0.3,
            device="cpu",
            opts=opts,
            agent_rows=24,
            anchor_rows=8,
            min_rows=4,
            bank=bank,
        )
        feed = np.random.default_rng(71)
        for _ in range(4):
            learner.update(0.5)
            _ingest(replay, feed, 40)
        learner.flush()
        assert stats["halt"] is None and stats["updates"] == 4
        assert stats["bank_agree_decisive"] and stats["bank_agree_counter"]
        return online, target, stats, rng, multi

    reference, fast = run(False), run(True)
    _assert_same(reference, fast)
    for key in ("bank_margin_loss", "bank_agree", "bank_agree_decisive", "bank_agree_counter"):
        assert reference[2][key] == fast[2][key]
