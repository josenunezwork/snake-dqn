"""T4a two-sided bank routing: raw provenance, ratio sampling, and both losses."""

import json
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


def _write_outside_bank(path: Path) -> LabelBank:
    n = 2
    np.savez(
        path / "outside.npz",
        local=np.zeros((n, 7, 31, 31), np.uint8),
        **{"global": np.zeros((n, 4, 37, 37), np.uint8)},
        scalars=np.zeros((n, 60), np.float32),
        mask=np.ones((n, 6), bool),
        surviving=np.tile([True, False, False, False, False, False], (n, 1)),
        surviving_original=np.tile([True, False, False, False, False, False], (n, 1)),
        action=np.zeros(n, np.int64),
        weight=np.ones(n, np.float32),
        row_weight=np.full(n, 0.5, np.float32),
        row_family=np.full(n, 2, np.int8),
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


def test_t4c_outside_hinge_keeps_primary_anchor_strength() -> None:
    """A zero-weight outside row cannot dilute the pre-existing anchor mixture."""
    q_primary = torch.tensor(
        [
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],
        ]
    )
    db = {
        "surviving": torch.tensor([[True, False, False, False, False, False]] * 2),
        "weight": torch.tensor([1.0, 1.0]),
        "action": torch.tensor([0, 0]),
        "counter": torch.tensor([False, True]),
        "a_v8": torch.tensor([0]),
        "mask_resolved": torch.ones((1, 6), dtype=torch.bool),
    }
    primary_anchor, primary_loss, _ = lm._bank_anchor_terms(
        torch, q_primary, db, 2, 0.3, 1.0, "single-action"
    )
    q_outside = torch.cat([q_primary, torch.tensor([[0.0, 0.0, 0.0, 0.0, 2.0, 0.0]])])
    outside_db = {
        **db,
        "surviving": torch.cat([db["surviving"], db["surviving"][:1]]),
        "weight": torch.tensor([1.0, 1.0, 0.0]),
        "action": torch.tensor([0, 0, 0]),
    }
    outside_anchor, outside_loss, _ = lm._bank_anchor_terms(
        torch,
        q_outside,
        outside_db,
        3,
        0.3,
        1.0,
        "single-action",
        torch.tensor([0, 1, 2]),
        0.0632,
        2,
    )
    assert float(outside_loss) == pytest.approx(float(primary_loss))
    assert float(outside_anchor) == pytest.approx(float(primary_anchor))


def test_t4c_outside_hinge_uses_its_own_scale_and_half_weight() -> None:
    q = torch.tensor([[0.0, 2.0, 0.0, 0.0, 0.0, 0.0]], requires_grad=True)
    db = {
        "surviving": torch.tensor([[True, False, False, False, False, False]]),
        "weight": torch.tensor([0.5]),
        "action": torch.tensor([0]),
        "counter": torch.tensor([True]),
        "a_v8": torch.empty(0, dtype=torch.long),
        "mask_resolved": torch.empty((0, 6), dtype=torch.bool),
    }
    _, loss, _ = lm._bank_anchor_terms(
        torch, q, db, 1, 0.3, 1.0, "single-action", torch.tensor([2]), 0.0632, 1
    )
    assert float(loss.detach()) == pytest.approx((2.0 + 0.3 * 0.0632) * 0.5)
    loss.backward()
    torch.testing.assert_close(q.grad, torch.tensor([[-0.5, 0.5, 0, 0, 0, 0]]))


def test_t4c_family_agreement_uses_separate_denominators_and_missing_is_none() -> None:
    q = torch.tensor(
        [
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # decisive: preferred action 0
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],  # decisive: misses preferred action 0
            [0.0, 0.0, 2.0, 0.0, 0.0, 0.0],  # in-window: matches action 2
            [0.0, 0.0, 0.0, 2.0, 0.0, 0.0],  # outside: misses action 4
        ]
    )
    surviving = torch.tensor([[True, False, False, False, False, False]] * 4)
    decisive, inwindow, outside = lm._bank_family_agreement(
        q, surviving, torch.tensor([0, 0, 2, 4]), torch.tensor([0, 0, 1, 2])
    )
    assert float(decisive) == pytest.approx(0.5)
    assert float(inwindow) == pytest.approx(1.0)
    assert float(outside) == pytest.approx(0.0)
    missing = lm._bank_family_agreement(q[:1], surviving[:1], torch.tensor([0]), torch.tensor([0]))
    assert float(missing[0]) == 1.0 and missing[1:] == (None, None)


def test_t4c_outside_flags_require_explicit_two_sided_mode(tmp_path: Path) -> None:
    args = [
        "--seed",
        "0",
        "--init",
        "x",
        "--demo-data",
        ".",
        "--out",
        str(tmp_path / "out"),
        "--label-bank-outside",
        str(tmp_path / "outside"),
    ]
    with pytest.raises(SystemExit, match="outside needs --label-bank-frac"):
        m3c.main(args)


def test_t4c_outside_path_is_json_safe_in_summary_flags(tmp_path: Path) -> None:
    args = m3c.default_args()
    args.label_bank_outside = tmp_path / "outside"
    flags = m3c.m3c_flags(args)
    assert flags["label_bank_outside"] == str(args.label_bank_outside)
    json.dumps({"m3c": flags})


@pytest.mark.parametrize("manifest_names", [{"bank-v2.npz"}, {"missing.npz"}])
def test_t4c_strict_bank_manifest_rejects_extra_or_missing_shards(
    tmp_path: Path, manifest_names: set[str]
) -> None:
    bank = _write_v2_bank(tmp_path)
    (tmp_path / "manifest.json").write_text(
        json.dumps({"shards": {name: "registered" for name in manifest_names}})
    )
    if manifest_names == {"bank-v2.npz"}:
        np.savez(tmp_path / "extra.npz", x=np.zeros(1))
    with pytest.raises(ValueError, match="shard set"):
        LabelBank(tmp_path, 60, strict_shards=True)
    assert bank.size == 4


def test_t4c_outside_cpu_reference_and_fast_match(tmp_path: Path) -> None:
    """The separate outside part adds telemetry and has identical CPU learner paths."""
    torch.set_num_threads(1)
    primary_dir, outside_dir = tmp_path / "primary", tmp_path / "outside"
    primary_dir.mkdir()
    outside_dir.mkdir()
    bank, outside = _write_v2_bank(primary_dir), _write_outside_bank(outside_dir)

    def run(fast: bool):
        torch.manual_seed(47)
        nscal, multi, replay, labels, demos = _world("bv8", seed=41)
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
                label_bank_frac=0.5,
                label_frac=0.5,
                label_bank_counter_mode="single-action",
                label_bank_counter_ratio=1.0,
                label_bank_outside_margin_scale=0.0632,
            ),
            nscal=nscal,
            margin=0.3,
            device="cpu",
            opts=opts,
            agent_rows=24,
            anchor_rows=8,
            min_rows=4,
            bank=bank,
            outside_bank=outside,
        )
        feed = np.random.default_rng(71)
        for _ in range(2):
            learner.update(0.5)
            _ingest(replay, feed, 40)
        learner.flush()
        assert stats["bank_agree_decisive"] and stats["bank_agree_counter"]
        assert stats["bank_agree_outside"] and stats["bank_rows"] == [6, 6]
        return online, target, stats, rng, multi

    reference, fast = run(False), run(True)
    _assert_same(reference, fast)
    for key in (
        "bank_margin_loss",
        "bank_agree_decisive",
        "bank_agree_counter",
        "bank_agree_outside",
    ):
        assert reference[2][key] == fast[2][key]


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
