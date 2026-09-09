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
    "sgd_epochs": 1,
    "pad_sgd_batches": True,
    "sgd_seed": 101,
    "action_collapse_patience": 3,
    "action_collapse_min_samples": 256,
    "action_collapse_raw_actions": True,
    "eps_decay_steps": 120000,
    "mechanics_version": 2,
}


def test_pqn_schema_field_parity_with_pqn_config_and_recipe_selector():
    """YAML exposes trainer knobs, while world/runtime provenance stays internal."""
    runtime_or_shared = {
        "game_width",
        "game_height",
        "segment_size",
        "wall_thickness",
        "initial_food",
        "max_food",
        "min_boost_length",
        "boost_length_cost_frames",
        "frame_rate",
        "starvation_max",
        "max_length",
        "obs_spec",
        "field_sources",
        "source_revision",
        "requested_device",
        "effective_device",
        # Fixed-policy mode requires a concrete constructor-injected policy;
        # its identity is runtime provenance, not an executable YAML setting.
        "rollout_policy_mode",
        "fixed_policy_identity",
        # B5 injects this only after checkpoint-byte preflight; accepting it
        # from generic YAML would not establish the claimed source identity.
        "initial_opponent_checkpoint_sha256",
    }
    config_fields = {f.name for f in dc.fields(PQNConfig)} - runtime_or_shared
    schema_fields = set(PQNSettingsSchema.model_fields.keys())
    assert config_fields | {"recipe"} == schema_fields, (
        "pqn: dataclass/schema field drift — "
        f"only in PQNConfig: {sorted(config_fields - schema_fields)}, "
        f"only in schema: {sorted(schema_fields - config_fields)}"
    )


def test_pqn_schema_accepts_the_two_explicit_lifecycle_knobs() -> None:
    """Lifecycle choices are YAML-visible while checkpoint identity remains runtime-only."""
    parsed = ConfigSchema(
        pqn={
            "episode_reset_mode": "per_env_autoreset_v1",
            "episode_seed_mode": "derived_env_episode_v1",
            "pool_admission_mode": "disabled_v1",
        }
    )
    assert parsed.pqn.episode_reset_mode == "per_env_autoreset_v1"
    assert parsed.pqn.episode_seed_mode == "derived_env_episode_v1"
    assert parsed.pqn.pool_admission_mode == "disabled_v1"


def test_config_schema_accepts_pqn_block():
    schema = ConfigSchema(**{"pqn": PQN_BLOCK})
    assert schema.pqn.num_envs == 64
    assert schema.pqn.lambda_ == 0.65
    assert schema.pqn.sgd_epochs == 1
    assert schema.pqn.pad_sgd_batches is True
    assert schema.pqn.sgd_seed == 101
    assert schema.pqn.action_collapse_patience == 3
    assert schema.pqn.action_collapse_min_samples == 256
    assert schema.pqn.action_collapse_raw_actions is True


def test_pqn_block_is_optional():
    assert ConfigSchema().pqn.num_envs is None
    assert ConfigSchema().pqn.recipe is None


def test_pqn_recipe_selector_is_closed_and_preserved_by_appconfig(tmp_path):
    schema = ConfigSchema(**{"pqn": {"recipe": "corrected-v3"}})
    assert schema.pqn.recipe == "corrected-v3"
    with pytest.raises(ValidationError, match="recipe"):
        ConfigSchema(**{"pqn": {"recipe": "experimental"}})
    path = tmp_path / "corrected_recipe.yaml"
    path.write_text("pqn:\n  recipe: corrected-v3\n", encoding="utf-8")
    config = load_config(str(path))
    assert config.pqn.recipe == "corrected-v3"
    assert "pqn.recipe" in config.provided_fields


@pytest.mark.parametrize(
    ("key", "value"),
    [("seed", True), ("sgd_seed", True), ("seed", -1), ("sgd_seed", 2**64)],
)
def test_pqn_seed_fields_are_strict_unsigned_64_bit_integers(key, value):
    with pytest.raises(ValidationError, match=key):
        ConfigSchema(**{"pqn": {key: value}})


def test_pqn_typo_is_still_rejected():
    with pytest.raises(ValidationError, match="rollout_lenn"):
        ConfigSchema(**{"pqn": {"rollout_lenn": 24}})


def test_pqn_wrong_type_is_rejected():
    with pytest.raises(ValidationError):
        ConfigSchema(**{"pqn": {"num_envs": "sixty-four"}})


def test_fixed_vector_observation_rejects_non_sixteen_sector_game():
    with pytest.raises(ValidationError, match="num_sectors"):
        ConfigSchema(**{"game": {"num_sectors": 8}})


def test_pqn_out_of_range_is_rejected():
    with pytest.raises(ValidationError):
        ConfigSchema(**{"pqn": {"hero_frac": 1.5}})


@pytest.mark.parametrize("epochs", [0, -1])
def test_pqn_nonpositive_sgd_epochs_is_rejected(epochs):
    with pytest.raises(ValidationError, match="sgd_epochs"):
        ConfigSchema(**{"pqn": {"sgd_epochs": epochs}})


def test_pqn_epsilon_order_is_rejected():
    with pytest.raises(ValidationError, match="eps_end must not exceed"):
        ConfigSchema(**{"pqn": {"eps_start": 0.1, "eps_end": 0.5}})


def test_unknown_top_level_section_still_rejected():
    with pytest.raises(ValidationError, match="pqnn"):
        ConfigSchema(**{"pqnn": PQN_BLOCK})


def test_load_config_with_pqn_block_yields_frozen_appconfig_overrides(tmp_path):
    """The vector loader exposes validated PQN overrides without reparsing YAML."""
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
    assert config.pqn.num_envs == 64
    assert config.pqn.lambda_ == 0.65
    assert config.pqn.lr is None
    assert "pqn.num_envs" in config.provided_fields


def test_provided_fields_distinguishes_omitted_and_explicit_legacy_values(tmp_path):
    path = tmp_path / "sources.yaml"
    path.write_text(
        "game:\n  mechanics_version: 1\n  arena_type: rectangular\n  num_snakes: 4\n"
        "  max_frames: 5000\nrewards:\n  version: 1\npqn:\n  lr: 0.001\n",
        encoding="utf-8",
    )
    config = load_config(str(path))
    assert config.provided_fields >= {
        "game.mechanics_version",
        "game.arena_type",
        "game.num_snakes",
        "game.max_frames",
        "rewards.version",
        "pqn.lr",
    }
    assert "game.width" not in config.provided_fields
    assert config.pqn.lr == pytest.approx(0.001)


def test_no_config_has_no_provided_yaml_paths():
    assert load_config().provided_fields == frozenset()


def test_train_pqn_reads_the_same_block(tmp_path):
    """The same file the vector loader accepts resolves to a PQNConfig."""
    import yaml

    from src.scripts.train_pqn import _load_config_overrides

    path = tmp_path / "mechanics_v2_with_pqn.yaml"
    path.write_text(yaml.safe_dump({"game": {"mechanics_version": 2}, "pqn": PQN_BLOCK}))

    overrides = _load_config_overrides(str(path))

    assert PQNConfig(**overrides).num_envs == 64
    assert PQNConfig(**overrides).mechanics_version == 2
