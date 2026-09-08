"""P1 resume modes keep run identity and mutable training state honest."""

import pytest
import torch

from src.scripts import train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def _config():
    return PQNConfig(num_envs=1, num_snakes=2, rollout_len=2, max_frames=20)


def test_legacy_checkpoint_requires_explicit_weights_only(tmp_path):
    path = tmp_path / "legacy.pth"
    legacy = PQNTrainer(_config()).checkpoint_state()
    del legacy["recipe"]
    torch.save(legacy, path)

    with pytest.raises(RuntimeError, match="continuation provenance"):
        train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="continuation")

    blob = train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="weights-only")
    assert blob is not None and blob["_p1_parent_hash"]


def test_exact_resume_is_rejected_and_weights_only_starts_fresh_world(tmp_path):
    original = PQNTrainer(_config())
    path = tmp_path / "checkpoint.pth"
    original.save_checkpoint(str(path))

    with pytest.raises(RuntimeError, match="exact PQN resume is unsupported"):
        train_pqn.load_pqn_resume_checkpoint(str(path), _config(), mode="exact")

    fresh = PQNTrainer(_config())
    blob = train_pqn.load_pqn_resume_checkpoint(str(path), fresh.cfg, mode="weights-only")
    train_pqn.apply_resume_checkpoint(fresh, blob, mode="weights-only")

    assert fresh.agent_steps == 0
    assert fresh.update_idx == 0
    assert fresh.checkpoint_state()["resume_mode"] == "weights-only"
    assert fresh.checkpoint_state()["resume_state"]["environment"] == "fresh"
