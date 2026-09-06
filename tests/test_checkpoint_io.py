"""Tests for durable checkpoint writes shared by both training stacks."""

from __future__ import annotations

import pytest
import torch

from src.model.checkpoint_io import atomic_torch_save
from src.model.checkpoint_manager import CheckpointManager
from src.training.pqn_trainer import PQNConfig, PQNTrainer


class _PoisonPickle:
    """Object whose serialization fails after the temporary file is created."""

    def __init__(self, error):
        self.error = error

    def __reduce__(self):
        raise self.error


def _tmp_files(directory):
    return sorted(path.name for path in directory.iterdir() if path.suffix == ".tmp")


def test_atomic_torch_save_replaces_existing_checkpoint(tmp_path):
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save({"generation": 1}, destination)
    atomic_torch_save({"generation": 2}, destination)

    assert torch.load(destination, weights_only=False) == {"generation": 2}
    assert _tmp_files(tmp_path) == []


@pytest.mark.parametrize("failure", [RuntimeError("serialization failed"), KeyboardInterrupt()])
def test_atomic_torch_save_preserves_destination_and_cleans_temp_on_save_failure(tmp_path, failure):
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save({"generation": 1}, destination)

    payload = {"generation": 2, "poison": _PoisonPickle(failure)}
    expected_exception = (
        KeyboardInterrupt if isinstance(failure, KeyboardInterrupt) else RuntimeError
    )
    with pytest.raises(expected_exception):
        atomic_torch_save(payload, destination)

    assert torch.load(destination, weights_only=False) == {"generation": 1}
    assert _tmp_files(tmp_path) == []


def test_atomic_torch_save_preserves_destination_and_cleans_temp_on_replace_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save({"generation": 1}, destination)

    def fail_replace(source, target):
        raise OSError("replace failed")

    monkeypatch.setattr("src.model.checkpoint_io.os.replace", fail_replace)
    with pytest.raises(OSError, match="replace failed"):
        atomic_torch_save({"generation": 2}, destination)

    assert torch.load(destination, weights_only=False) == {"generation": 1}
    assert _tmp_files(tmp_path) == []


def test_atomic_torch_save_preserves_destination_and_cleans_temp_on_fsync_failure(
    tmp_path, monkeypatch
):
    destination = tmp_path / "checkpoint.pth"
    atomic_torch_save({"generation": 1}, destination)

    def fail_fsync(file_descriptor):
        raise OSError("fsync failed")

    monkeypatch.setattr("src.model.checkpoint_io.os.fsync", fail_fsync)
    with pytest.raises(OSError, match="fsync failed"):
        atomic_torch_save({"generation": 2}, destination)

    assert torch.load(destination, weights_only=False) == {"generation": 1}
    assert _tmp_files(tmp_path) == []


def test_apex_checkpoint_round_trip_preserves_metadata_override_and_caller_dict(tmp_path):
    manager = CheckpointManager(str(tmp_path), verbose=False)
    source = {"policy_type": "other", "iteration": 7, "marker": "apex"}

    path = manager.save_checkpoint_dict(source, "apex.pth")
    loaded = manager.load_checkpoint(torch.device("cpu"), "apex.pth")

    assert path == str(tmp_path / "apex.pth")
    assert loaded == {"policy_type": "apex", "iteration": 7, "marker": "apex"}
    assert source == {"policy_type": "other", "iteration": 7, "marker": "apex"}


def test_apex_model_checkpoint_round_trip_preserves_metadata_override(tmp_path):
    manager = CheckpointManager(str(tmp_path), verbose=False)
    dqn = torch.nn.Linear(2, 2)
    target = torch.nn.Linear(2, 2)
    optimizer = torch.optim.Adam(dqn.parameters())

    manager.save_checkpoint(
        dqn,
        target,
        optimizer,
        {"policy_type": "custom-apex", "iteration": 7},
        "model.pth",
    )
    loaded = manager.load_checkpoint(torch.device("cpu"), "model.pth")

    assert loaded["policy_type"] == "custom-apex"
    assert loaded["iteration"] == 7
    assert set(("dqn_state_dict", "target_dqn_state_dict", "optimizer_state_dict")) <= set(loaded)


def test_pqn_checkpoint_round_trip_uses_shared_writer(tmp_path):
    trainer = PQNTrainer(PQNConfig(num_envs=1, num_snakes=1, rollout_len=2))
    destination = tmp_path / "nested" / "latest_pqn.pth"

    trainer.save_checkpoint(str(destination))
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)

    assert checkpoint["algo"] == "pqn"
    assert checkpoint["obs_spec"] == "raster31v2"
    assert "dqn_state_dict" in checkpoint
