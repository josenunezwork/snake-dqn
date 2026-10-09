"""Web serving of an ego2s-b student (Watch / Play only; the default is unchanged)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")

from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint  # noqa: E402
from src.model.inference_agent import InferenceAgent  # noqa: E402
from src.model.obs_spec import EGO2S_B, KNOWN_OBS_SPECS  # noqa: E402
from web.backend.ego2s_policy import Ego2sServingPolicy  # noqa: E402
from web.backend.session import MODE_PLAY, MODE_TRAIN, MODE_WATCH, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_config():
    from src.core import game_config

    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture()
def ego_ckpt(tmp_path):
    torch.manual_seed(0)
    path = tmp_path / "student_b.pth"
    save_ego2s_checkpoint(path, Ego2sNet(local_channels=7, n_scalars=30), {"note": "test"})
    return str(path)


def test_spec_registered_and_inference_agent_refuses(ego_ckpt):
    assert EGO2S_B in KNOWN_OBS_SPECS
    with pytest.raises(ValueError, match="ego2s"):
        InferenceAgent.from_checkpoint(ego_ckpt)


def test_watch_session_serves_the_student(ego_ckpt):
    sess = GameSession(checkpoint=ego_ckpt)
    assert isinstance(sess.policy, Ego2sServingPolicy)
    assert sess.obs_spec == EGO2S_B and sess.mode == MODE_WATCH
    heads0 = [tuple(s.segments[0]) for s in sess.game.snakes]
    for _ in range(30):
        sess.step()
    assert [tuple(s.segments[0]) for s in sess.game.snakes] != heads0
    frame = sess.snapshot()
    assert frame["obs_spec"] == EGO2S_B
    assert frame["architecture"] == "Ego2s Dueling CNN (ego2s-b)"
    assert frame["hero_raster"] is None
    insp, nv = frame["inspector"], frame["netviz"]
    if insp is not None:  # the hero may be dead on this frame
        assert insp["input_size"] == 30 and len(insp["q_values"]) == 6
        assert nv["input_label"] == "Scalars (30 of ego2s-b input)"
        assert len(nv["advantages"]) == 6


def test_served_q_rows_match_a_direct_forward(ego_ckpt):
    """Each AI snake receives its own row (dispatch order), equal to a fresh featurize."""
    from src.simd_env.ego_live_adapter import game_state_to_ego_view
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    sess = GameSession(checkpoint=ego_ckpt)
    for _ in range(5):
        sess.step()
    pol = sess.policy
    # The serving cache is built at the decision point of each update; rebuild it on the
    # current (post-step) world to compare with a direct featurize of the same world.
    pol._invalidate()
    pol.prepare_frame()
    view = game_state_to_ego_view(sess.game)
    rows = np.array([[0, i] for i in range(view.S) if view.alive[0, i]])
    obs = build_ego_raster(view, EgoRasterConfig(version="b"), backend="numpy", rows=rows)
    with torch.no_grad():
        q = pol.net(
            *[
                torch.from_numpy(np.ascontiguousarray(obs[k]))
                for k in ("local", "global", "scalars")
            ]
        )
    for i, (_, s) in enumerate(rows):
        np.testing.assert_allclose(pol.q_for(sess.game.snakes[s].id), q[i].numpy(), rtol=1e-5)


def test_train_refused_and_play_works(ego_ckpt):
    sess = GameSession(checkpoint=ego_ckpt)
    sess.set_mode(MODE_TRAIN)
    assert sess.mode == MODE_WATCH and isinstance(sess.policy, Ego2sServingPolicy)
    sess.set_mode(MODE_PLAY)
    assert sess.mode == MODE_PLAY and isinstance(sess.policy, Ego2sServingPolicy)
    for _ in range(20):
        sess.step()


def test_default_session_is_unchanged():
    sess = GameSession()
    assert not isinstance(sess.policy, Ego2sServingPolicy)
    assert sess.obs_spec == "vector61"


def test_reset_clears_the_served_cache(ego_ckpt):
    """A reset at frame 1 must not serve the previous game's frame-1 Q rows."""
    sess = GameSession(checkpoint=ego_ckpt)
    sess.step()
    old = sess.policy._q.copy() if sess.policy._q is not None else None
    sess.reset_game()
    assert sess.policy._q is None and sess.policy._frame is None
    sess.step()
    assert sess.policy._q is not None
    if old is not None and old.shape == sess.policy._q.shape:
        assert not np.array_equal(old, sess.policy._q) or sess.game.frame != 1
