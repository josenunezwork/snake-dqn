"""Focused, CPU-only unit coverage for P4 arithmetic and replay safeguards."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from research.t34_exit_b8_fit.data import (
    ReplayDivergenceError,
    StreamingReplayLoader,
    sample_weights,
)
from research.t34_exit_b8_fit.diagnostics import (
    FitTripwire,
    enforce_tripwires,
    trajectory_diagnostics,
)
from research.t34_exit_b8_fit.interface import FakeReplaySource, ReplayWorld
from research.t34_exit_b8_fit.loss import LossConfig, exit_fit_loss
from research.t34_exit_b8_fit.train import optimizer_for


def _batch(weight: float) -> dict[str, torch.Tensor]:
    return {
        "q_prev": torch.tensor([[0.3, 0.1, 0.0, -0.2, -0.4, -0.1]]),
        "action_star": torch.tensor([0]),
        "forbidden_mask": torch.tensor([[False, True, False, False, False, False]]),
        "sample_weight": torch.tensor([weight]),
    }


def test_hinge_is_clamped_and_empty_forbidden_is_zero() -> None:
    q = torch.tensor([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
    loss, parts = exit_fit_loss(q, _batch(1.0), LossConfig(advantage_weight=0.0))
    assert float(parts["hinge"]) == 0.0
    batch = _batch(1.0)
    batch["forbidden_mask"][:] = False
    _, empty = exit_fit_loss(q, batch, LossConfig(advantage_weight=0.0))
    assert float(empty["hinge"]) == 0.0


def test_older_half_weight_halves_gradient_not_normalized_away() -> None:
    q1 = torch.tensor([[0.0] * 6], requires_grad=True)
    l1, _ = exit_fit_loss(q1, _batch(1.0))
    l1.backward()
    q2 = torch.tensor([[0.0] * 6], requires_grad=True)
    l2, _ = exit_fit_loss(q2, _batch(0.5))
    l2.backward()
    assert torch.allclose(q2.grad, q1.grad * 0.5, atol=1e-6)


def test_death_window_weight_is_ten_current_five_old_including_decisive() -> None:
    rows = [
        {"world_id": "w", "death_window": True, "decisive": True, "round_age": 0},
        {"world_id": "w", "death_window": True, "decisive": False, "round_age": 0},
        {"world_id": "old", "death_window": True, "decisive": True, "round_age": 1},
        {"world_id": "old", "death_window": True, "decisive": False, "round_age": 1},
    ]
    weights = sample_weights(rows)
    assert weights[:2].sum() == pytest.approx(10.0)
    assert weights[2:].sum() == pytest.approx(5.0)
    assert weights[0] == pytest.approx(weights[1] * 20)


def test_two_windows_in_one_world_are_independently_normalized() -> None:
    rows = [
        {
            "world_id": "w",
            "death_window": True,
            "death_window_id": wid,
            "decisive": False,
            "round_age": 0,
        }
        for wid in (0, 0, 1, 1)
    ]
    weights = sample_weights(rows)
    assert weights[:2].sum() == pytest.approx(10.0)
    assert weights[2:].sum() == pytest.approx(10.0)


def test_tripwire_uses_global_empirical_deciles_and_exact_boundary() -> None:
    lengths = np.arange(100)
    old = np.arange(100, dtype=np.int64) % 3
    new = old.copy()
    new[-1] = 1
    probe = trajectory_diagnostics(previous_actions=old, actions=new, lengths=lengths)
    assert probe["max_decile_disagreement"] == pytest.approx(0.1)
    with pytest.raises(FitTripwire, match="disagreement"):
        enforce_tripwires(loss=0.0, heldout_decisive_agreement=0.7, probe=probe)


def test_new_tower_parameters_get_new_rate() -> None:
    class Model(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.base = torch.nn.Linear(1, 1)
            self.tower_fuse = torch.nn.Linear(1, 1)
            self.free_region_head = torch.nn.Linear(1, 1)

    opt = optimizer_for(Model(), pretrained=True)
    assert sorted(group["lr"] for group in opt.param_groups) == [2.5e-5, 1e-4]


def test_fake_replay_is_stable_across_epochs() -> None:
    source = FakeReplaySource(worlds=1, rows_per_world=4)
    first, second = list(source.replay_epoch(1))[0], list(source.replay_epoch(2))[0]
    assert np.array_equal(first.rows[0]["local"], second.rows[0]["local"])


def test_loader_refuses_bad_complete_chunk_only_after_full_count() -> None:
    class Source:
        is_fake = True

        def replay_epoch(self, epoch):
            good = list(FakeReplaySource(worlds=1, rows_per_world=4).replay_epoch(epoch))[0]
            yield good
            yield ReplayWorld("bad", (), False, chunk_id="unpartitioned")

    loader = StreamingReplayLoader(Source(), batch_size=2)
    with pytest.raises(ReplayDivergenceError):
        list(loader.epoch(1))


@pytest.mark.parametrize(
    "field,value,reason",
    [
        ("loss", float("nan"), "non-finite"),
        ("agreement", 0.699, "70%"),
        ("max_action_share", 0.900001, "one-action"),
        ("max_decile_disagreement", 0.050001, "disagreement"),
        ("boost_share", 0.150001, "boost"),
    ],
)
def test_registered_tripwire_boundaries(field, value, reason):
    probe = {
        "max_action_share": 0.90,
        "max_decile_disagreement": 0.05,
        "boost_share": 0.15,
        "previous_boost_share": 0.10,
    }
    loss, agreement = 0.0, 0.70
    enforce_tripwires(loss=loss, heldout_decisive_agreement=agreement, probe=probe)
    if field == "loss":
        loss = value
    elif field == "agreement":
        agreement = value
    else:
        probe[field] = value
    with pytest.raises(FitTripwire, match=reason):
        enforce_tripwires(loss=loss, heldout_decisive_agreement=agreement, probe=probe)


class _TinyAuxModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = torch.nn.Linear(60, 6)
        self.aux = torch.nn.Parameter(torch.tensor(0.0))

    def forward(self, local, glob, scalars):
        return self.projection(scalars.float())

    def aux_free_region(self, local, glob, scalars, full_sight):
        return self.aux.expand(len(scalars))


def test_masked_missing_aux_target_does_not_contaminate_fit():
    import copy

    from research.t34_exit_b8_fit.train import fit

    class Source(FakeReplaySource):
        def replay_epoch(self, epoch):
            for world in super().replay_epoch(epoch):
                for index, row in enumerate(world.rows):
                    row["hindsight_log_free_region"] = 1.0 if index == 0 else float("nan")
                    row["hindsight_log_free_region_mask"] = index == 0
                yield world

    model = _TinyAuxModel()
    result = fit(
        source=Source(rows_per_world=4),
        model=model,
        previous_model=copy.deepcopy(model),
        allow_fake=True,
        max_updates=1,
        batch_size=4,
        epochs=4,
    )
    assert result["completed"] is True
    assert torch.isfinite(model.aux)
    assert model.aux.item() != 0


def test_preflight_update_limit_still_checks_last_world():
    import copy

    from research.t34_exit_b8_fit.train import run_fit_preflight

    class Source(FakeReplaySource):
        def replay_epoch(self, epoch):
            yield next(iter(super().replay_epoch(epoch)))
            yield ReplayWorld("bad-last-world", (), False, chunk_id="unpartitioned")

    model = _TinyAuxModel()
    result = run_fit_preflight(
        source=Source(rows_per_world=4),
        model=model,
        previous_model=copy.deepcopy(model),
        device="cpu",
        updates=1,
        batch_size=2,
        allow_fake=True,
    )
    assert not result["completed"]
    assert "2%" in result["tripwire"]


def test_production_fit_requires_probe_before_any_replay():
    import copy

    from research.t34_exit_b8_fit.train import fit

    class Source:
        is_fake = False

        def replay_epoch(self, epoch):
            raise AssertionError("must refuse before replay")

    model = _TinyAuxModel()
    with pytest.raises(ValueError, match="probe hook"):
        fit(source=Source(), model=model, previous_model=copy.deepcopy(model))


def test_heldout_does_not_accept_unbound_verified_boolean():
    from research.t34_exit_b8_fit.train import _heldout_metrics

    class Source:
        is_fake = False

        def replay_heldout(self, epoch):
            world = next(iter(FakeReplaySource(rows_per_world=4).replay_epoch(epoch)))
            yield world

    with pytest.raises(ValueError, match="sha256 proof"):
        _heldout_metrics(_TinyAuxModel(), Source(), torch.device("cpu"), 1)
