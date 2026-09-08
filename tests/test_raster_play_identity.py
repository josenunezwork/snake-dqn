"""Regression coverage for raster serving in mixed human/AI Play rosters."""

import pytest
import torch

pytest.importorskip("fastapi")

from src.model.obs_spec import (OBS_SPEC_KEY, RASTER31V2,  # noqa: E402
                                RASTER31V2_SHAPES)
from src.model.raster_network import RasterDuelingNetwork  # noqa: E402
from web.backend.session import MODE_PLAY, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_config():
    """Keep GameSession's process-global configuration local to each test."""
    from src.core import game_config

    previous = game_config._current_config
    yield
    game_config._current_config = previous


@pytest.fixture()
def raster_checkpoint(tmp_path):
    """Save a minimal raster checkpoint with ordinary v2 metadata."""
    path = tmp_path / "raster.pth"
    network = RasterDuelingNetwork(output_size=6)
    torch.save(
        {
            "dqn_state_dict": network.state_dict(),
            OBS_SPEC_KEY: RASTER31V2,
            "output_size": 6,
            **RASTER31V2_SHAPES.to_metadata(),
        },
        path,
    )
    return str(path)


class _RosterRowLogits:
    """Return a different legal normal-speed action for each raster row."""

    def __call__(self, tensors):
        rows = int(tensors["tactical"].shape[0])
        logits = torch.full((rows, 6), -100.0, device=tensors["tactical"].device)
        for row in range(rows):
            logits[row, row % 3] = 100.0
        return logits


class _ForeignPolicy:
    """Minimal forward-only policy for a live mixed-policy roster."""

    training = False
    epsilon = 0.0
    memory = None
    device = torch.device("cpu")

    @staticmethod
    def dqn(state):
        return torch.tensor([[100.0, -100.0, -100.0, -100.0, -100.0, -100.0]])


def _play_session_with_two_ai(raster_checkpoint):
    session = GameSession(checkpoint=raster_checkpoint)
    session.set_play_opponents(2)
    session.set_mode(MODE_PLAY)
    assert session.mode == MODE_PLAY
    assert len(session.game.snakes) == 3
    return session


def test_human_row_does_not_shift_raster_ai_actions(raster_checkpoint):
    """A real mixed GameState.update dispatches rows 1 and 2 to the two AIs.

    The human occupies featurizer row 0 but never calls the raster DQN shim.
    Q outputs deliberately choose action 0 for row 0, action 1 for row 1, and
    action 2 for row 2. If the human enters the dispatch queue, the first AI
    takes action 0 and this assertion fails.
    """
    session = _play_session_with_two_ai(raster_checkpoint)
    session.policy.agent.network = _RosterRowLogits()

    session.game.update(train_mode=False, learn=False, allow_respawn=True)

    human, first_ai, second_ai = session.game.snakes
    assert human.is_alive
    assert first_ai.last_action == 1
    assert second_ai.last_action == 2


def test_raster_policy_primes_once_before_human_moves(raster_checkpoint):
    """Play snapshots the whole raster roster before the human changes row zero."""
    session = _play_session_with_two_ai(raster_checkpoint)
    human, first_ai, _ = session.game.snakes
    initial_head = tuple(human.segments[0])
    calls = []
    original = session.policy.prepare_frame

    def prepare():
        calls.append(tuple(human.segments[0]))
        original()

    session.policy.prepare_frame = prepare
    session.game.update(train_mode=False, learn=False, allow_respawn=True)
    assert calls == [initial_head]
    assert session.policy.hero_observation(first_ai.id) is not None


def test_dead_and_foreign_policy_rows_do_not_shift_dispatch(raster_checkpoint):
    """Only live AIs owned by the policy consume raster rows on a real update."""
    session = _play_session_with_two_ai(raster_checkpoint)
    session.policy.agent.network = _RosterRowLogits()
    _, first_ai, second_ai = session.game.snakes

    first_ai.is_alive = False
    first_ai.respawn_timer = 99  # Keep it dead through this update.
    second_ai.policy = _ForeignPolicy()  # A mixed-policy roster must not consume a row either.

    session.game.update(train_mode=False, learn=False, allow_respawn=True)

    assert session.policy._dispatch_queue == []
    # The policy-owned caller set was empty; no foreign or dead snake consumed
    # a raster dispatch row.
    assert first_ai.last_action is None
    assert second_ai.last_action == 0

    # Once the dead policy-owned AI respawns, it again owns its featurizer row
    # (row 1). The foreign AI still takes its own policy's action and never
    # shifts the raster dispatch queue.
    first_ai.respawn_timer = 0
    session.game.update(train_mode=False, learn=False, allow_respawn=True)
    assert first_ai.is_alive
    assert first_ai.last_action == 1
    assert second_ai.last_action == 0
