"""P1 effective-world and corrected-recipe checkpoints."""

from src.model.obs_spec import RASTER31V3
from src.scripts import train_pqn
from src.training.pqn_trainer import PQNConfig, PQNTrainer


def test_config_file_has_complete_resolved_world_and_sources(tmp_path):
    path = tmp_path / "world.yaml"
    path.write_text(
        "game:\n  width: 300\n  height: 220\n  segment_size: 10\n  wall_thickness: 10\n"
        "  max_frames: 101\n  max_length: 91\nrewards:\n  starvation_max_frames: 63\n"
        "pqn:\n  recipe: corrected-v3\n  num_envs: 1\n  num_snakes: 2\n"
    )
    fields = train_pqn._load_config_overrides(str(path))
    fields["num_envs"] = 3  # explicit CLI-equivalent override
    fields["obs_spec"] = RASTER31V3  # corrected recipe resolver sets this in build_config
    fields["flip_augment"] = False
    fields["field_sources"] = {key: "config" for key in fields}
    fields["field_sources"]["num_envs"] = "cli"
    config = PQNConfig(**fields)

    assert config.recipe == "corrected-v3"
    assert config.obs_spec == RASTER31V3
    assert config.num_envs == 3
    assert config.field_sources["num_envs"] == "cli"
    assert (config.game_width, config.game_height, config.max_frames) == (300, 220, 101)
    assert (config.starvation_max, config.max_length) == (63, 91)


def test_corrected_checkpoint_records_explicit_world_and_contract_digests():
    config = PQNConfig(
        num_envs=1,
        num_snakes=2,
        game_width=300,
        game_height=220,
        max_frames=71,
        starvation_max=63,
        max_length=91,
        recipe="corrected-v3",
        obs_spec=RASTER31V3,
        flip_augment=False,
    )
    trainer = PQNTrainer(config)
    state = trainer.checkpoint_state()

    assert state["effective_world"]["normalization"] == {
        "max_frames": 71.0,
        "starvation_max": 63.0,
        "max_length": 91.0,
    }
    assert state["effective_world_digest"]
    assert state["runtime_contract_digest"]
    assert state["action_mask_contract"] == {
        "version": "legal-advisory-resolved-v1",
        "action_count": 6,
        "resolution": "row_local_intersection_else_legal",
        "dead_rows": "all_false",
    }
    assert state["obs_spec"] == RASTER31V3
