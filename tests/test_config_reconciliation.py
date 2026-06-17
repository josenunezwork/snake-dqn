"""Tests for YAML training-to-Apex configuration reconciliation."""

import pytest

from src.core.config_loader import load_config
from src.core.game_config import (
    ApexSettings,
    AppConfig,
    GameConfig,
    TrainingSettings,
    get_config,
    initialize_config,
)
from src.main import apply_training_batch_size_override


@pytest.fixture(autouse=True)
def restore_global_config():
    """Restore the process-global config after each test."""
    original_config = get_config()
    try:
        yield
    finally:
        initialize_config(original_config)


def test_training_learner_overrides_populate_apex_when_apex_omits_them(tmp_path):
    """Documented training learner knobs should reach the live Apex learner."""
    config_path = tmp_path / "training_only.yaml"
    config_path.write_text(
        """
training:
  learning_rate: 0.001
  gamma: 0.97
  batch_size: 256
  memory_size: 200000
  target_update_frequency: 1000
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.apex.learning_rate == pytest.approx(0.001)
    assert config.apex.gamma == pytest.approx(0.97)
    assert config.apex.batch_size == 256
    assert config.apex.buffer_size == 200000
    assert config.apex.target_update_freq == 1000


def test_explicit_apex_value_wins_over_training_reconciliation(tmp_path):
    """Apex-specific YAML fields remain authoritative when explicitly set."""
    config_path = tmp_path / "explicit_apex.yaml"
    config_path.write_text(
        """
training:
  learning_rate: 0.001
apex:
  learning_rate: 0.0005
""",
        encoding="utf-8",
    )

    config = load_config(str(config_path))

    assert config.training.learning_rate == pytest.approx(0.001)
    assert config.apex.learning_rate == pytest.approx(0.0005)


def test_production_training_overrides_reach_apex_config():
    """Production tuning values should affect the Apex learner config."""
    config = load_config("configs/production.yaml")

    assert config.apex.learning_rate == pytest.approx(0.001)
    assert config.apex.batch_size == 256
    assert config.apex.buffer_size == 200000
    assert config.apex.target_update_freq == 1000


def test_main_batch_size_override_updates_apex_batch_size():
    """The main.py batch-size override should report and apply the Apex batch size."""
    initialize_config(
        AppConfig(
            training=TrainingSettings(batch_size=16, memory_size=1000),
            apex=ApexSettings(batch_size=32, buffer_size=1000, min_buffer_size=32),
        )
    )

    applied_batch_size = apply_training_batch_size_override(64)

    assert applied_batch_size == 64
    assert GameConfig.BATCH_SIZE == 64
    assert GameConfig.APEX_BATCH_SIZE == 64
    assert get_config().training.batch_size == 64
    assert get_config().apex.batch_size == 64
