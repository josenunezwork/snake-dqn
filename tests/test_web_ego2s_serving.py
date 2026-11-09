"""Web serving of an ego2s-b student (Watch / Play only)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")

from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint  # noqa: E402
from src.model.inference_agent import InferenceAgent  # noqa: E402
from src.model.obs_spec import EGO2S_B, EGO2S_BV8, KNOWN_OBS_SPECS  # noqa: E402
from web.backend.ego2s_policy import Ego2sServingPolicy  # noqa: E402
from web.backend.serving_speed import speed_ceiling  # noqa: E402
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
    assert sess.control_state()["input_size"] == 30
    assert frame["session"]["input_size"] == 30
    assert frame["session"]["can_save_weights"] is False
    assert frame["architecture"] == "Ego2s Dueling CNN (ego2s-b)"
    assert frame["hero_raster"] is None
    insp, nv = frame["inspector"], frame["netviz"]
    if insp is not None:  # the hero may be dead on this frame
        assert insp["input_size"] == 30 and len(insp["q_values"]) == 6
        assert insp["free_space"] is None
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
    sess.set_mode(MODE_TRAIN)
    assert sess.mode == MODE_WATCH
    assert sess.policy.training is False and sess.policy.epsilon == 0.0
    assert "ego2s students" in sess.last_error and "dropped to watch" in sess.last_error
    watch_game = sess.game
    sess.set_mode(MODE_TRAIN)
    assert sess.game is watch_game


def test_default_session_serves_released_b8_with_mixed_watch_opponents(monkeypatch):
    from tests.checkpoint_paths import has_saved_checkpoints
    from web.backend import served_checkpoint as registry

    required = (registry.B8_S3_FILENAME, registry.FRP3_S12_FILENAME)
    if not has_saved_checkpoints(*required):
        pytest.skip("released b8-s3 and frp3-s12 checkpoint bytes are not available")

    sess = GameSession()
    assert isinstance(sess.policy, Ego2sServingPolicy)
    assert sess.served_checkpoint["name"] == registry.NAME_B8_S3
    assert sess.obs_spec == EGO2S_BV8
    assert sess.control_state()["speed_ceiling"] == 24.0
    assert sess.watch_roster == "mixed"
    assert sess.game.snakes[0].policy is sess.policy
    assert sess.game.snakes[0].safety_veto is None
    assert sess.safety_veto_state()["active"] is True
    assert sess.safety_veto_state()["scope"] == "watch_opponents"
    assert sess.safety_veto_state()["wrapped_snake_ids"] == [1, 2, 3, 4, 5]

    monkeypatch.setenv(registry.ENV_CHECKPOINT, registry.NAME_FRP3_S12)
    rollback = GameSession()
    assert not isinstance(rollback.policy, Ego2sServingPolicy)
    assert rollback.served_checkpoint["name"] == registry.NAME_FRP3_S12
    assert rollback.obs_spec == "vector61"
    assert rollback.control_state()["can_save_weights"] is True
    assert sess.control_state()["can_save_weights"] is False


def test_ego2s_speed_ceiling_counts_the_driven_roster_and_clamps(ego_ckpt):
    """The cap follows respawnable AI roster size, never current liveness."""
    sess = GameSession(checkpoint=ego_ckpt)
    assert sess.control_state()["speed_ceiling"] == 24.0  # Watch's six AI roster
    sess.set_speed(120)
    assert sess.speed == 24.0

    # A dead AI remains a future driven/respawnable roster member.
    sess.game.snakes[0].is_alive = False
    sess._refresh_speed_ceiling()
    assert sess.speed_ceiling == 24.0

    # Play replaces one snake with a human. Six opponents means six driven AI,
    # while seven crosses into the Watch-12 ceiling band after the rebuild.
    sess.set_mode(MODE_PLAY)
    sess.set_play_opponents(6)
    assert sess.human_id is not None
    assert sess.speed_ceiling == 24.0
    sess.set_play_opponents(7)
    assert sess.speed_ceiling == 12.0
    assert sess.speed == 12.0


def test_ego2s_speed_ceiling_bands_and_non_ego_default():
    from src.model.obs_spec import EGO2S_B

    assert speed_ceiling(EGO2S_B, 6) == 24.0
    assert speed_ceiling(EGO2S_B, 7) == 12.0
    assert speed_ceiling(EGO2S_B, 12) == 12.0
    assert speed_ceiling(EGO2S_B, 13) == 11.0
    assert speed_ceiling(EGO2S_B, 1440) == 1.0
    assert speed_ceiling("vector61", 999) == 120.0


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


# ----------------------------------------------------------------------------- ego2s-bv8
@pytest.fixture()
def bv8_ckpt(tmp_path):
    torch.manual_seed(1)
    path = tmp_path / "student_bv8.pth"
    save_ego2s_checkpoint(path, Ego2sNet(local_channels=7, n_scalars=60), {"note": "test"})
    return str(path)


def test_bv8_watch_and_play_serve_the_student(bv8_ckpt):
    from src.model.obs_spec import EGO2S_BV8

    sess = GameSession(checkpoint=bv8_ckpt)
    assert isinstance(sess.policy, Ego2sServingPolicy)
    assert sess.obs_spec == EGO2S_BV8 and sess.mode == MODE_WATCH
    assert sess.policy.ego_cfg.version == "bv8"
    for _ in range(30):
        sess.step()
    frame = sess.snapshot()
    assert frame["obs_spec"] == EGO2S_BV8
    assert sess.control_state()["input_size"] == 60
    assert frame["session"]["input_size"] == 60
    assert frame["session"]["can_save_weights"] is False
    with pytest.raises(RuntimeError, match="forward-only"):
        sess.save_weights()
    if frame["inspector"] is not None:  # the hero may be dead on this frame
        assert frame["inspector"]["input_size"] == 60
        state_vals = frame["inspector"]["state"]
        assert frame["inspector"]["free_space"] == [
            state_vals[30],
            state_vals[35],
            state_vals[40],
        ]
        assert frame["netviz"]["input_label"] == "Scalars (60 of ego2s-bv8 input)"
    sess.set_mode(MODE_PLAY)
    assert sess.mode == MODE_PLAY and isinstance(sess.policy, Ego2sServingPolicy)
    for _ in range(20):
        sess.step()


def test_bv8_served_rows_are_the_driven_snakes_with_live_v8_features(bv8_ckpt):
    """Only the snakes the policy drives are featurized; their rows equal a direct featurize
    with the live v8 source (the 30 v8 scalars included)."""
    from src.game.ai_snake import AISnake
    from src.simd_env.ego_live_adapter import game_state_to_ego_view
    from src.simd_env.ego_live_bv8 import game_state_to_v8_source
    from src.simd_env.ego_raster import EgoRasterConfig, build_ego_raster

    sess = GameSession(checkpoint=bv8_ckpt)
    for _ in range(5):
        sess.step()
    pol = sess.policy
    pol._invalidate()
    pol.prepare_frame()
    snakes = sess.game.snakes
    driven = [
        i
        for i, s in enumerate(snakes)
        if isinstance(s, AISnake) and s.is_alive and getattr(s, "policy", None) is pol
    ]
    assert driven and list(pol._obs["rows"]) == driven
    view = game_state_to_ego_view(sess.game)
    rows = np.array([[0, i] for i in driven])
    obs = build_ego_raster(
        view,
        EgoRasterConfig(version="bv8"),
        backend="numpy",
        rows=rows,
        v8_source=game_state_to_v8_source(sess.game),
    )
    assert obs["scalars"].shape == (len(driven), 60)
    assert np.array_equal(obs["scalars"], pol._obs["scalars"])
    with torch.no_grad():
        q = pol.net(
            *[
                torch.from_numpy(np.ascontiguousarray(obs[k]))
                for k in ("local", "global", "scalars")
            ]
        )
    for i, s in enumerate(driven):
        np.testing.assert_allclose(pol.q_for(snakes[s].id), q[i].numpy(), rtol=1e-5)


def test_ego2s_hero_raster_identity_and_subscription_gate(bv8_ckpt):
    """Subscribed live Watch exports exact cached uint8 planes and 60 scalars."""
    import json

    from web.backend.serialize import (
        EGO2S_GLOBAL_CHANNELS,
        EGO2S_LOCAL_CHANNELS,
        decode_ego2s_raster,
    )

    sess = GameSession(checkpoint=bv8_ckpt)
    sess.step()

    # Unsubscribed by default -> hero_raster is None
    frame_unsub = sess.snapshot()
    assert frame_unsub["hero_raster"] is None

    # Direct policy.hero_observation returns the exact cached slices for hero_id
    pol = sess.policy
    row_slot = pol._id_to_row[int(sess.hero_id)]
    hero_row = int(np.flatnonzero(pol._obs["rows"] == row_slot)[0])
    raw_obs = pol.hero_observation(sess.hero_id)
    assert raw_obs is not None
    assert np.array_equal(raw_obs["local"], pol._obs["local"][hero_row])
    assert np.array_equal(raw_obs["global"], pol._obs["global"][hero_row])
    assert np.array_equal(raw_obs["scalars"], pol._obs["scalars"][hero_row])

    # Subscribe a client -> hero_raster is populated and matches policy._obs bit-for-bit
    sess.set_raster_stream("conn-1", True)
    frame_sub = sess.snapshot()
    hr = frame_sub["hero_raster"]
    assert hr is not None
    assert hr["encoding"] == "u8_b64"
    assert hr["local_shape"] == [7, 31, 31]
    assert hr["local_channels"] == list(EGO2S_LOCAL_CHANNELS)
    assert hr["global_shape"] == [4, 37, 37]
    assert hr["global_channels"] == list(EGO2S_GLOBAL_CHANNELS)
    assert hr["scalar_groups"] == [
        {"name": "ego2s-b scalars", "start": 0, "end": 30},
        {"name": "v8 veto features (5 per action)", "start": 30, "end": 60},
    ]

    decoded = decode_ego2s_raster(hr)
    assert decoded["local"].dtype == np.uint8
    assert decoded["global"].dtype == np.uint8
    assert np.array_equal(decoded["local"], pol._obs["local"][hero_row])
    assert np.array_equal(decoded["global"], pol._obs["global"][hero_row])
    np.testing.assert_allclose(decoded["scalars"], pol._obs["scalars"][hero_row], rtol=0, atol=1e-6)

    # Compact payload target: full frame <= 32 KiB
    encoded_bytes = len(json.dumps(frame_sub, separators=(",", ":")).encode("utf-8"))
    assert encoded_bytes <= 32 * 1024

    # Unsubscribing restores hero_raster=None
    sess.set_raster_stream("conn-1", False)
    assert sess.snapshot()["hero_raster"] is None


def test_ego2s_hero_raster_zero_extra_forwards_or_flood_calls(bv8_ckpt, monkeypatch):
    """Exporting hero_raster uses cached obs with zero extra forwards or flood calls."""
    import src.simd_env.ego_live_bv8 as ego_live_bv8
    import src.simd_env.ego_raster_bv8 as ego_bv8
    import src.simd_env.ego_raster_nb as ego_nb
    import web.backend.ego2s_policy as pol_mod

    sess = GameSession(checkpoint=bv8_ckpt)
    sess.step()
    pol = sess.policy

    counts = {"forward": 0, "raster": 0, "v8": 0, "planes_nb": 0, "region_nb": 0, "v8_kernel": 0}

    orig_forward = pol.net.forward
    orig_raster = pol_mod.build_ego_raster
    orig_v8 = ego_live_bv8.game_state_to_v8_source
    orig_planes_nb = ego_nb.build_planes_numba
    orig_region_nb = ego_nb.region_sizes_numba
    orig_v8_kernel = ego_bv8.v8_scalars

    def wrapped_forward(*args, **kwargs):
        counts["forward"] += 1
        return orig_forward(*args, **kwargs)

    def wrapped_raster(*args, **kwargs):
        counts["raster"] += 1
        return orig_raster(*args, **kwargs)

    def wrapped_v8(*args, **kwargs):
        counts["v8"] += 1
        return orig_v8(*args, **kwargs)

    def wrapped_planes_nb(*args, **kwargs):
        counts["planes_nb"] += 1
        return orig_planes_nb(*args, **kwargs)

    def wrapped_region_nb(*args, **kwargs):
        counts["region_nb"] += 1
        return orig_region_nb(*args, **kwargs)

    def wrapped_v8_kernel(*args, **kwargs):
        counts["v8_kernel"] += 1
        return orig_v8_kernel(*args, **kwargs)

    monkeypatch.setattr(pol.net, "forward", wrapped_forward)
    monkeypatch.setattr(pol_mod, "build_ego_raster", wrapped_raster)
    monkeypatch.setattr(ego_live_bv8, "game_state_to_v8_source", wrapped_v8)
    monkeypatch.setattr(ego_nb, "build_planes_numba", wrapped_planes_nb)
    monkeypatch.setattr(ego_nb, "region_sizes_numba", wrapped_region_nb)
    monkeypatch.setattr(ego_bv8, "v8_scalars", wrapped_v8_kernel)

    # Calling hero_observation and snapshot (subscribed or unsubscribed) on an already-stepped frame
    # performs ZERO forwards and ZERO raster/flood calls.
    sess.set_raster_stream("conn-1", True)
    assert pol.hero_observation(sess.hero_id) is not None
    frame = sess.snapshot()
    assert frame["hero_raster"] is not None
    assert counts == {
        "forward": 0,
        "raster": 0,
        "v8": 0,
        "planes_nb": 0,
        "region_nb": 0,
        "v8_kernel": 0,
    }

    # Stepping N live frames with raster subscribed performs exactly 1 forward/featurize per tick.
    for _ in range(5):
        sess.step()
        f = sess.snapshot()
        assert f["hero_raster"] is not None
    assert counts == {
        "forward": 5,
        "raster": 5,
        "v8": 5,
        "planes_nb": 5,
        "region_nb": 5,
        "v8_kernel": 5,
    }


def test_ego2s_hero_raster_none_in_play_and_replay(bv8_ckpt):
    """Play mode and recorded Watch replays always keep hero_raster=None even when subscribed."""
    from types import SimpleNamespace

    from web.backend.serialize import build_frame

    sess = GameSession(checkpoint=bv8_ckpt)
    sess.set_raster_stream("conn-1", True)
    sess.step()
    assert sess.snapshot()["hero_raster"] is not None

    # Play mode -> hero_raster is None even with raster_subscribers > 0
    sess.set_mode(MODE_PLAY)
    sess.step()
    assert sess.raster_subscribers == 1
    assert sess.snapshot()["hero_raster"] is None

    # Return to live Watch -> hero_raster is present again
    sess.set_mode(MODE_WATCH)
    sess.step()
    live_frame = sess.snapshot()
    assert live_frame["hero_raster"] is not None

    # Recorded Watch replay active -> hero_raster is None even with raster_subscribers > 0
    fake_replay_frame = dict(live_frame)
    fake_replay_frame["hero_raster"] = None
    sess._watch_replay = SimpleNamespace(
        frame=fake_replay_frame,
        info=lambda: {"label": "Replay", "controller": "exact-stop-l1"},
        snapshot=lambda _sess: dict(fake_replay_frame),
    )
    try:
        assert sess.snapshot()["hero_raster"] is None
        assert build_frame(sess)["hero_raster"] is None
    finally:
        sess._watch_replay = None


def test_control_state_input_size_none_safe_for_ego2s(ego_ckpt, bv8_ckpt):
    """control_state input_size stays None-safe and falls back to ego2s obs_spec scalar count."""
    sess_b = GameSession(checkpoint=ego_ckpt)
    sess_b.policy.net.config = None
    assert sess_b.control_state()["input_size"] == 30
    sess_b.policy = None
    assert sess_b.control_state()["input_size"] == 30

    sess_bv8 = GameSession(checkpoint=bv8_ckpt)
    sess_bv8.policy.input_size = None
    sess_bv8.policy.net.config = None
    assert sess_bv8.control_state()["input_size"] == 60
    sess_bv8.policy = None
    assert sess_bv8.control_state()["input_size"] == 60
