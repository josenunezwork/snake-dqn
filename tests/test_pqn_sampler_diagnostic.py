"""Focused tests for the reproducible PQN sampler diagnostic runner."""

import json
from argparse import Namespace
from pathlib import Path

import pytest

from src.scripts import pqn_sampler_diagnostic as diagnostic
from src.training.pqn_trainer import PQNTelemetry


def _args(**overrides: object) -> Namespace:
    values = {
        "sampler": "epoch",
        "seed": 7,
        "total_steps": 100,
        "max_seconds": 5.0,
        "threads": 2,
        "device": "cpu",
        "envs": 3,
        "rollout_len": 4,
        "out_dir": Path("unused"),
        "sgd_epochs": None,
        "eps_decay_steps": None,
        "profile": False,
        "pad_sgd_batches": False,
        "action_collapse_patience": 1,
        "action_collapse_min_samples": 0,
        "action_collapse_raw_actions": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_build_config_epoch_uses_exactly_one_epoch_by_default() -> None:
    config = diagnostic.build_config(_args())

    assert config.sgd_epochs == 1
    assert config.num_envs == 3
    assert config.num_snakes == 6
    assert config.pool_capacity == 0
    assert config.hero_frac == 1.0
    assert config.flip_augment is True
    assert config.mechanics_version == 2
    assert config.reward_version == 2
    assert config.eps_decay_steps == 60
    assert config.sgd_seed == 10_000_007


def test_build_config_legacy_preserves_legacy_sampler() -> None:
    config = diagnostic.build_config(_args(sampler="legacy"))

    assert config.sgd_epochs is None


def test_legacy_rejects_epoch_count() -> None:
    with pytest.raises(ValueError, match="only to --sampler epoch"):
        diagnostic.build_config(_args(sampler="legacy", sgd_epochs=1))


def test_prepare_output_refuses_nonempty_directory(tmp_path: Path) -> None:
    destination = tmp_path / "existing"
    destination.mkdir()
    (destination / "old.json").write_text("old")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        diagnostic.prepare_output(destination)


def test_provenance_records_experimental_config_and_source_hashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args()
    config = diagnostic.build_config(args)
    monkeypatch.setattr(diagnostic, "_source_hashes", lambda: {"runner.py": "abc"})

    provenance = diagnostic._provenance(
        args, config, "initial-hash", diagnostic.torch.device("cpu")
    )

    assert provenance["evaluation_status"] == diagnostic.EXPERIMENTAL_STATUS
    assert provenance["config"]["sgd_epochs"] == 1
    assert provenance["initial_model_tensor_sha256"] == "initial-hash"
    assert provenance["source_sha256"] == {"runner.py": "abc"}
    assert provenance["device"] == "cpu"


def test_resolve_device_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(diagnostic.torch.backends.mps, "is_available", lambda: False)

    with pytest.raises(ValueError, match="MPS is unavailable"):
        diagnostic.resolve_device("mps")


def test_padding_and_tiny_budget_are_recorded_in_config() -> None:
    config = diagnostic.build_config(_args(total_steps=1, pad_sgd_batches=True))
    assert config.pad_sgd_batches is True
    assert config.eps_decay_steps == 1


@pytest.mark.parametrize("seconds", [float("nan"), float("inf"), 0.0, -1.0])
def test_wall_budget_must_be_finite_and_positive(seconds: float) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        diagnostic.validate_args(_args(max_seconds=seconds))


def test_raw_collapse_guard_is_recorded_identically_for_both_samplers() -> None:
    for sampler in ("legacy", "epoch"):
        config = diagnostic.build_config(
            _args(
                sampler=sampler,
                action_collapse_patience=3,
                action_collapse_min_samples=2048,
                action_collapse_raw_actions=True,
            )
        )
        assert config.action_collapse_patience == 3
        assert config.action_collapse_min_samples == 2048
        assert config.action_collapse_raw_actions is True


@pytest.mark.parametrize(
    "overrides", [{"action_collapse_patience": 0}, {"action_collapse_min_samples": -1}]
)
def test_invalid_collapse_guard_budget_is_rejected(overrides: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="collapse patience"):
        diagnostic.validate_args(_args(**overrides))


def test_tripwire_keeps_applied_update_in_history(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The failed update counts in both the saved odometer and its audit trail."""
    telemetry = PQNTelemetry(
        update=0,
        agent_steps=64,
        loss=0.1,
        grad_norm=0.2,
        mean_abs_q=2.0,
        max_abs_q=2.5,
        epsilon=0.02,
        mean_reward=0.0,
        action_entropy=0.0,
        kills_per_ep=0.0,
        boost_fraction=0.0,
        pool_size=0,
        eligible_hero_transitions=64,
        action_collapse_streak=32,
        action_collapse_evidence_samples=2048,
    )

    class StoppedTrainer:
        def __init__(self, config: object, device: object) -> None:
            self.agent_steps = 0
            self.network = diagnostic.torch.nn.Linear(1, 1)

        def update(self) -> PQNTelemetry:
            self.agent_steps = 64
            raise diagnostic.TripwireError("test tripwire", telemetry=telemetry)

        def checkpoint_state(self) -> dict[str, int]:
            return {"agent_steps": self.agent_steps}

    monkeypatch.setattr(diagnostic, "PQNTrainer", StoppedTrainer)
    monkeypatch.setattr(diagnostic, "_provenance", lambda *args: {})
    monkeypatch.setattr(diagnostic.torch, "set_num_threads", lambda _: None)
    monkeypatch.setattr(diagnostic.torch, "set_num_interop_threads", lambda _: None)
    output = tmp_path / "failed"
    result = diagnostic.main(
        ["--sampler", "legacy", "--seed", "7", "--total-steps", "100", "--out-dir", str(output)]
    )

    assert result == 2
    history = [json.loads(row) for row in (output / "history.jsonl").read_text().splitlines()]
    summary = json.loads((output / "summary.json").read_text())
    assert len(history) == 1
    assert history[0]["agent_step_delta"] == summary["agent_steps"] == 64
    assert history[0]["action_collapse_evidence_samples"] == 2048
    assert summary["status"] == "tripwire"
