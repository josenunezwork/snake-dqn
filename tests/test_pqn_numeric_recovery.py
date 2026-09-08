"""Regression tests for PQN numeric rollback and incident checkpoint policy."""

from __future__ import annotations

import copy
import json

import pytest
import torch

from src.model.raster_network import SCALARS_DIM, STRATEGIC_SHAPE, TACTICAL_SHAPE
from src.scripts import train_pqn
from src.scripts.train_pqn import train_loop
from src.training.pqn_selfplay import HERO_POLICY_ID
from src.training.pqn_trainer import PQNConfig, PQNTelemetry, PQNTrainer, TripwireError


def _trainer(*, minibatches: int = 2) -> PQNTrainer:
    return PQNTrainer(
        PQNConfig(
            num_envs=1,
            num_snakes=1,
            rollout_len=1,
            minibatches=minibatches,
            minibatch_size=1,
            flip_augment=False,
        )
    )


def _roll() -> dict[str, object]:
    return {
        "policy_ids": torch.tensor([[HERO_POLICY_ID]]).numpy(),
        "valid": torch.ones((1, 1, 1), dtype=torch.bool).numpy(),
        "tactical": torch.zeros((1, 1, 1, *TACTICAL_SHAPE)),
        "strategic": torch.zeros((1, 1, 1, *STRATEGIC_SHAPE)),
        "scalars": torch.zeros((1, 1, 1, SCALARS_DIM)),
        "actions": torch.zeros((1, 1, 1), dtype=torch.long).numpy(),
    }


def _assert_state_equal(left, right) -> None:
    if isinstance(left, torch.Tensor):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    elif isinstance(left, dict):
        assert set(left) == set(right)
        for key in left:
            _assert_state_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            _assert_state_equal(a, b)
    else:
        assert left == right


def _pre_failure_state(trainer: PQNTrainer) -> dict[str, object]:
    return {
        "network": copy.deepcopy(trainer.network.state_dict()),
        "optimizer": copy.deepcopy(trainer.optimizer.state_dict()),
    }


@pytest.mark.parametrize(
    ("stage", "expected_steps"),
    [
        ("target", 0),
        ("loss", 0),
        ("gradient", 0),
        ("post_optimizer_parameters", 1),
        ("post_optimizer_state", 1),
        ("post_optimizer_learning_rate", 1),
    ],
)
def test_nonfinite_sgd_paths_restore_weights_and_optimizer(monkeypatch, stage, expected_steps):
    """Each checked numeric boundary rolls back and prevents later minibatches."""
    trainer = _trainer()
    roll = _roll()
    targets = torch.zeros((1, 1, 1))
    if stage == "target":
        targets.fill_(float("nan"))
    elif stage.startswith("post_optimizer"):
        # Populate Adam moments first.  Recovery must restore those moments,
        # not merely return to the uninitialized-optimizer special case.
        trainer._sgd(roll, targets)

    if stage == "loss":
        monkeypatch.setattr(
            "src.training.pqn_trainer.F.smooth_l1_loss",
            lambda *_args, **_kwargs: torch.tensor(float("nan"), requires_grad=True),
        )
    elif stage == "gradient":
        next(trainer.network.parameters()).register_hook(
            lambda grad: torch.full_like(grad, float("nan"))
        )

    original_step = trainer.optimizer.step
    calls = []

    def poisoned_step(*args, **kwargs):
        result = original_step(*args, **kwargs)
        calls.append(1)
        if stage == "post_optimizer_parameters":
            with torch.no_grad():
                next(trainer.network.parameters()).fill_(float("nan"))
        elif stage == "post_optimizer_state":
            first_state = next(iter(trainer.optimizer.state.values()))
            tensor = next(
                value for value in first_state.values() if isinstance(value, torch.Tensor)
            )
            tensor.fill_(float("nan"))
        elif stage == "post_optimizer_learning_rate":
            trainer.optimizer.param_groups[0]["lr"] = float("nan")
        return result

    monkeypatch.setattr(trainer.optimizer, "step", poisoned_step)
    before = _pre_failure_state(trainer)

    with pytest.raises(TripwireError) as caught:
        trainer._sgd(roll, targets)

    expected_incident_stage = (
        "post_optimizer_state" if stage == "post_optimizer_learning_rate" else stage
    )
    assert caught.value.incident_class == f"nonfinite_{expected_incident_stage}"
    assert caught.value.incident["recovered"] is True
    assert calls == [1] * expected_steps
    _assert_state_equal(before["network"], trainer.network.state_dict())
    _assert_state_equal(before["optimizer"], trainer.optimizer.state_dict())


def test_finite_sgd_matches_a_trainer_without_recovery_snapshot():
    """Recovery copying does not change successful optimizer arithmetic."""
    torch.manual_seed(123)
    recovered = _trainer(minibatches=1)
    baseline = _trainer(minibatches=1)
    baseline.network.load_state_dict(recovered.network.state_dict())
    baseline.optimizer.load_state_dict(recovered.optimizer.state_dict())
    baseline.refresh_numeric_recovery_state = lambda: None

    left = recovered._sgd(_roll(), torch.zeros((1, 1, 1)))
    right = baseline._sgd(_roll(), torch.zeros((1, 1, 1)))

    assert left == pytest.approx(right)
    _assert_state_equal(recovered.network.state_dict(), baseline.network.state_dict())
    _assert_state_equal(recovered.optimizer.state_dict(), baseline.optimizer.state_dict())
    assert recovered.numeric_recovery_metadata()["snapshot_bytes"] > 0


def _first_tensor(value):
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, dict):
        for nested in value.values():
            found = _first_tensor(nested)
            if found is not None:
                return found
    if isinstance(value, (list, tuple)):
        for nested in value:
            found = _first_tensor(nested)
            if found is not None:
                return found
    return None


@pytest.mark.parametrize(
    "poisoned_key", ["dqn_state_dict", "optimizer_state_dict", "optimizer_learning_rate"]
)
def test_poisoned_resume_is_rejected_before_mutating_the_fresh_trainer(poisoned_key):
    """A bad resume never becomes a newly initialized trainer's rollback point."""
    source = _trainer()
    if poisoned_key == "optimizer_state_dict":
        source._sgd(_roll(), torch.zeros((1, 1, 1)))
    checkpoint = source.checkpoint_state()
    if poisoned_key == "optimizer_learning_rate":
        checkpoint["optimizer_state_dict"] = copy.deepcopy(checkpoint["optimizer_state_dict"])
        checkpoint["optimizer_state_dict"]["param_groups"][0]["lr"] = float("nan")
        expected_key = "optimizer_state_dict"
    else:
        checkpoint[poisoned_key] = copy.deepcopy(checkpoint[poisoned_key])
        tensor = _first_tensor(checkpoint[poisoned_key])
        assert tensor is not None
        tensor.fill_(float("nan"))
        expected_key = poisoned_key
    fresh = _trainer()
    before = _pre_failure_state(fresh)

    with pytest.raises(ValueError, match=f"{expected_key}.*non-finite"):
        train_pqn.apply_resume_checkpoint(fresh, checkpoint)

    _assert_state_equal(before["network"], fresh.network.state_dict())
    _assert_state_equal(before["optimizer"], fresh.optimizer.state_dict())


class _PreSgdNumericTrainer:
    """Train-loop fixture whose old telemetry must not be attributed to a new fault."""

    def __init__(self) -> None:
        self.agent_steps = 37
        self.update_idx = 9
        self.last_telemetry = PQNTelemetry(
            update=8,
            agent_steps=31,
            loss=0.5,
            grad_norm=1.0,
            mean_abs_q=1.0,
            max_abs_q=1.0,
            epsilon=0.2,
            mean_reward=0.0,
            action_entropy=1.0,
            legacy_kills_per_rollout=0.0,
            boost_fraction=0.0,
            pool_size=0,
        )

    def update(self):
        raise TripwireError(
            "non-finite target during SGD at update 9",
            incident_class="nonfinite_target",
            incident={"stage": "target", "nonfinite_counts": {"target": 3}},
        )

    def save_checkpoint(self, _path):
        raise AssertionError("numeric failure must not replace latest")

    def numeric_recovery_metadata(self):
        return {"strategy": "fixture", "snapshot_bytes": 42}


def test_pre_sgd_incident_does_not_misatrribute_previous_telemetry(tmp_path):
    """The incident has current counters and counts, but no invented telemetry row."""
    train_loop(_PreSgdNumericTrainer(), 38, 1, 0, tmp_path)

    incident = json.loads(next((tmp_path / "incidents").glob("*.json")).read_text())
    assert incident["telemetry"] is None
    assert incident["trainer_state"] == {"update_idx": 9, "agent_steps": 37}
    assert incident["details"]["nonfinite_counts"] == {"target": 3}


class _FiniteAlarmTrainer:
    """Minimal train-loop fixture with an accepted latest and finite flagged state."""

    def __init__(self, incident_class: str) -> None:
        self.agent_steps = 0
        self.update_idx = 7
        self.last_telemetry = None
        self.saved_incidents = []
        self.incident_class = incident_class

    def update(self):
        message = (
            "max|Q|=1001 exceeded alarm 1000"
            if self.incident_class == "max_abs_q"
            else "action collapse (entropy=0, streak=1, samples=1)"
        )
        raise TripwireError(
            message,
            telemetry=PQNTelemetry(
                update=7,
                agent_steps=0,
                loss=1.0,
                grad_norm=1.0,
                mean_abs_q=1001.0,
                max_abs_q=1001.0,
                epsilon=0.1,
                mean_reward=0.0,
                action_entropy=1.0,
                legacy_kills_per_rollout=0.0,
                boost_fraction=0.0,
                pool_size=0,
            ),
            incident_class=self.incident_class,
        )

    def save_checkpoint(self, _path):
        raise AssertionError("latest must not be replaced on a finite alarm")

    def save_incident_checkpoint(self, path, incident):
        self.saved_incidents.append((path, incident))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("finite-incident")

    def numeric_recovery_metadata(self):
        return {"strategy": "fixture", "snapshot_bytes": 12}


@pytest.mark.parametrize("incident_class", ["max_abs_q", "action_collapse"])
def test_finite_alarm_preserves_latest_and_writes_labeled_incident(tmp_path, incident_class):
    latest = tmp_path / "latest_pqn.pth"
    latest.write_bytes(b"accepted-checkpoint")

    history, tripped = train_loop(
        _FiniteAlarmTrainer(incident_class),
        total_steps=1,
        log_every=1,
        ckpt_every=0,
        out_dir=tmp_path,
    )

    assert history == []
    assert tripped is not None
    assert latest.read_bytes() == b"accepted-checkpoint"
    incident_json = next((tmp_path / "incidents").glob("*.json"))
    payload = json.loads(incident_json.read_text())
    assert payload["class"] == incident_class
    assert payload["telemetry"]["max_abs_q"] == 1001.0
    assert payload["numeric_recovery"]["snapshot_bytes"] == 12
    incident_checkpoint = incident_json.with_suffix(".pth")
    assert incident_checkpoint.read_text() == "finite-incident"
