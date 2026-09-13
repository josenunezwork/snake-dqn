"""History serialization regression for PQN food-contact telemetry."""

from src.scripts.train_pqn import _telemetry_record
from src.training.pqn_trainer import PQNTelemetry


def test_history_record_retains_exact_hero_food_contact_events():
    telemetry = PQNTelemetry(
        update=3,
        agent_steps=12,
        loss=0.0,
        grad_norm=0.0,
        mean_abs_q=0.0,
        max_abs_q=0.0,
        epsilon=0.2,
        mean_reward=0.0,
        action_entropy=0.0,
        legacy_kills_per_rollout=None,
        boost_fraction=0.0,
        pool_size=0,
        hero_food_contact_events=7,
    )
    assert _telemetry_record(telemetry)["hero_food_contact_events"] == 7
