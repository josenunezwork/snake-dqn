"""Lock the shared-config contract for the ``pqn:`` YAML section.

``src/scripts/train_pqn.py`` documents that one mechanics-v2 config drives both
the vector pipeline and PQN. The vector loader validates with ``extra='forbid'``,
so ``pqn:`` must be a known (and typed) section there or every ``load_config()``
caller — tournament_eval, main.py, apex_train, evaluate_checkpoints, and sweep's
own gate command — hard-fails on the file the moment the block is added.
"""

import dataclasses as dc

import pytest
from pydantic import ValidationError

from src.core.config_loader import ConfigSchema, PQNSettingsSchema, load_config
from src.training.pqn_trainer import PQNConfig

PQN_BLOCK = {
    "num_envs": 64,
    "num_snakes": 6,
    "rollout_len": 24,
    "gamma": 0.997,
    "lambda_": 0.65,
    "eps_decay_steps": 120000,
    "mechanics_version": 2,
}


def test_pqn_schema_field_parity_with_pqn_config():
    """Every PQNConfig knob is settable from YAML, and nothing else is."""
    config_fields = {f.name for f in dc.fields(PQNConfig)}
    schema_fields = set(PQNSettingsSchema.model_fields.keys())
    assert config_fields == schema_fields, (
        "pqn: dataclass/schema field drift — "
        f"only in PQNConfig: {sorted(config_fields - schema_fields)}, "
        f"only in schema: {sorted(schema_fields - config_fields)}"
    )


def test_config_schema_accepts_pqn_block():
    schema = ConfigSchema(**{"pqn": PQN_BLOCK})
    assert schema.pqn.num_envs == 64
    assert schema.pqn.lambda_ == 0.65


def test_pqn_block_is_optional():
    assert ConfigSchema().pqn.num_envs is None


def test_pqn_typo_is_still_rejected():
    with pytest.raises(ValidationError, match="rollout_lenn"):
        ConfigSchema(**{"pqn": {"rollout_lenn": 24}})


def test_pqn_wrong_type_is_rejected():
    with pytest.raises(ValidationError):
        ConfigSchema(**{"pqn": {"num_envs": "sixty-four"}})


def test_pqn_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        ConfigSchema(**{"pqn": {"hero_frac": 1.5}})


def test_pqn_epsilon_order_is_rejected():
    with pytest.raises(ValidationError, match="eps_end must not exceed"):
        ConfigSchema(**{"pqn": {"eps_start": 0.1, "eps_end": 0.5}})


def test_unknown_top_level_section_still_rejected():
    with pytest.raises(ValidationError, match="pqnn"):
        ConfigSchema(**{"pqnn": PQN_BLOCK})


def test_load_config_with_pqn_block_yields_appconfig(tmp_path):
    """The vector loader accepts a config carrying a pqn block and drops it."""
    import yaml

    path = tmp_path / "mechanics_v2_with_pqn.yaml"
    path.write_text(
        yaml.safe_dump(
            {"game": {"mechanics_version": 2}, "rewards": {"version": 2}, "pqn": PQN_BLOCK}
        )
    )

    config = load_config(str(path))

    assert config.game.mechanics_version == 2
    assert config.rewards.version == 2
    assert not hasattr(config, "pqn")


def test_train_pqn_reads_the_same_block(tmp_path):
    """The same file the vector loader accepts resolves to a PQNConfig."""
    import yaml

    from src.scripts.train_pqn import _load_config_overrides

    path = tmp_path / "mechanics_v2_with_pqn.yaml"
    path.write_text(yaml.safe_dump({"game": {"mechanics_version": 2}, "pqn": PQN_BLOCK}))

    overrides = _load_config_overrides(str(path))

    assert PQNConfig(**overrides).num_envs == 64
    assert PQNConfig(**overrides).mechanics_version == 2
