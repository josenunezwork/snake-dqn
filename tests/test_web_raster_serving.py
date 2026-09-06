"""Tests for serving an ``obs_spec='raster31v2'`` checkpoint via the web backend.

Verifies that a raster checkpoint loads into a :class:`GameSession`, drives the
game via the shared featurizer, and produces a serialized frame carrying the
raster planes plus valid inspector / net-viz panels — while the existing
``vector61`` serving path stays byte-for-byte unchanged.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("fastapi")

from src.model.obs_spec import (OBS_SPEC_KEY, RASTER31V2,  # noqa: E402
                                RASTER31V2_SHAPES)
from src.model.raster_network import RasterDuelingNetwork  # noqa: E402
from web.backend.session import (MODE_TRAIN, MODE_WATCH,  # noqa: E402
                                 GameSession)

CPU = torch.device("cpu")


@pytest.fixture(autouse=True)
def _restore_global_config():
    """Snapshot/restore the process-global config around every test.

    GameSession initializes the global config (mechanics v2 by default), so
    without this later test files reading the global config without initializing
    their own would see v2 mechanics. Same pattern as test_web_play.
    """
    from src.core import game_config

    prev = game_config._current_config
    yield
    game_config._current_config = prev


def _save_raster_checkpoint(path: str, output_size: int = 6) -> RasterDuelingNetwork:
    """Build a tiny RasterDuelingNetwork and save a contract-v2 raster checkpoint."""
    net = RasterDuelingNetwork(output_size=output_size)
    blob = {
        "dqn_state_dict": net.state_dict(),
        OBS_SPEC_KEY: RASTER31V2,
        "output_size": output_size,
        "total_reward": 42.0,
        "update_counter": 7,
        **RASTER31V2_SHAPES.to_metadata(),
    }
    torch.save(blob, path)
    return net


@pytest.fixture()
def raster_ckpt(tmp_path):
    path = str(tmp_path / "raster.pth")
    _save_raster_checkpoint(path)
    return path


# ---------------------------------------------------------------------------
# Loading + stepping
# ---------------------------------------------------------------------------
class TestRasterSession:
    def test_loads_as_raster_policy(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        assert sess.obs_spec == RASTER31V2
        # Duck-typed ApexPolicy surface the game touches.
        assert sess.policy.training is False
        assert hasattr(sess.policy, "dqn")
        assert callable(sess.policy.dqn)
        assert hasattr(sess.policy, "hero_activations")

    def test_steps_without_error_and_drives_world(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        heads_before = [tuple(s.head) for s in sess.game.snakes]
        alive_before = sum(1 for s in sess.game.snakes if s.is_alive)
        # Sampled every frame rather than at the endpoint alone: an untrained net
        # often emits a constant turn, tracing a closed 4-frame loop that returns
        # every head to its start on each frame divisible by 4 (20 included).
        moved = 0
        for _ in range(20):
            sess.step()  # must never raise
            heads_now = [tuple(s.head) for s in sess.game.snakes]
            moved = max(moved, sum(1 for a, b in zip(heads_before, heads_now) if a != b))
        assert sess.game.frame == 20
        alive_after = sum(1 for s in sess.game.snakes if s.is_alive)
        # The raster policy actually drove the world: snakes moved and/or died
        # (an untrained net crashes most snakes fast — either way it acted).
        assert moved > 0 or alive_after < alive_before

    def test_train_mode_falls_back_to_watch(self, raster_ckpt):
        """Raster checkpoints have no online-training path; train -> watch, loudly."""
        sess = GameSession(checkpoint=raster_ckpt)
        sess.set_mode(MODE_TRAIN)
        assert sess.mode == MODE_WATCH
        # The refusal must be loud (surfaced to the client), not a silent no-op.
        assert sess.last_error is not None
        assert "train" in sess.last_error.lower()
        assert sess.obs_spec == RASTER31V2

    def test_reset_clears_cache_and_keeps_stepping(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        for _ in range(5):
            sess.step()
        sess.reset_game()
        # After reset the frame counter rewinds; stepping must still work.
        for _ in range(5):
            sess.step()
        frame = sess.snapshot()
        assert frame["obs_spec"] == RASTER31V2

    def test_epsilon_pinned_to_zero(self, raster_ckpt):
        """Raster serving must stay strictly greedy so the dispatch queue is 1:1.

        ``AISnake.update`` skips ``policy.dqn()`` for any snake that explores when
        ``record_q_values`` is off (the headless serving case). A skipped call
        would shift the per-frame dispatch queue and hand later snakes a foreign
        snake's raster Q-row. The policy defends against this by clamping epsilon
        to 0 — writes (session init or ``set_epsilon``) are accepted but ignored.
        """
        sess = GameSession(checkpoint=raster_ckpt)
        assert sess.policy.epsilon == 0.0
        # Direct write (mimics session init `policy.epsilon = ...`) is ignored.
        sess.policy.epsilon = 0.7
        assert sess.policy.epsilon == 0.0
        # The public controls path (watch-mode controls panel) is also inert.
        sess.set_epsilon(0.9)
        assert sess.policy.epsilon == 0.0
        # With epsilon forced to 0, every alive snake calls dqn once per frame,
        # so stepping stays healthy no matter what a user requested.
        for _ in range(10):
            sess.step()
        assert sess.game.frame == 10


# ---------------------------------------------------------------------------
# Serialized frame contract
# ---------------------------------------------------------------------------
class TestRasterFrame:
    def test_frame_exposes_obs_spec_and_mechanics(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        sess.step()
        frame = sess.snapshot()
        assert frame["obs_spec"] == RASTER31V2
        assert isinstance(frame["mechanics_version"], int)
        assert frame["session"]["obs_spec"] == RASTER31V2

    def test_frame_protocol_fields(self, raster_ckpt):
        """Every frame carries the protocol contract fields (protocol v2)."""
        sess = GameSession(checkpoint=raster_ckpt)
        sess.step()
        frame = sess.snapshot()
        assert frame["protocol_version"] == 2
        assert frame["checkpoint_name"] == "raster.pth"
        assert frame["architecture"] == "Raster Dueling (raster31v2)"
        assert frame["paused"] is False
        # Sessions without the app's viewer counter degrade to 0.
        assert frame["viewer_count"] == 0
        sess.viewer_count = 3
        sess.set_playing(False)
        frame = sess.snapshot()
        assert frame["viewer_count"] == 3
        assert frame["paused"] is True

    def test_raster_block_gated_on_subscribers(self, raster_ckpt):
        """hero_raster ships only while a client has the Raster tab open.

        A session without the counter attr degrades to always-on (getattr
        default 1), so serialize.py never depends on app.py's half.
        """
        sess = GameSession(checkpoint=raster_ckpt)
        sess.step()
        assert sess.snapshot()["hero_raster"] is not None  # no attr -> default on
        sess.raster_subscribers = 0
        assert sess.snapshot()["hero_raster"] is None
        sess.raster_subscribers = 2
        assert sess.snapshot()["hero_raster"] is not None

    def test_frame_exposes_hero_raster_planes(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        for _ in range(3):
            sess.step()
        hr = sess.snapshot()["hero_raster"]
        assert hr is not None
        # Tactical: two 31x31 uint8 planes + legends.
        assert hr["tactical_size"] == 31
        assert len(hr["tactical_code"]) == 31 and len(hr["tactical_code"][0]) == 31
        assert len(hr["tactical_value"]) == 31 and len(hr["tactical_value"][0]) == 31
        assert len(hr["tactical_channels"]) == 9
        assert hr["tactical_codes"]["own_head"] == 8
        # Strategic: three 25x25 density planes.
        assert hr["strategic_size"] == 25
        assert len(hr["strategic"]) == 3
        assert len(hr["strategic"][0]) == 25 and len(hr["strategic"][0][0]) == 25
        assert len(hr["strategic_channels"]) == 3
        # Scalars + mask.
        assert len(hr["scalars"]) == 26
        assert len(hr["mask"]) == 6
        # Values are JSON-safe primitives.
        assert all(isinstance(v, int) for row in hr["tactical_code"] for v in row)
        assert all(isinstance(v, bool) for v in hr["mask"])

    def test_inspector_and_netviz_valid_for_raster(self, raster_ckpt):
        sess = GameSession(checkpoint=raster_ckpt)
        for _ in range(3):
            sess.step()
        frame = sess.snapshot()
        insp = frame["inspector"]
        assert insp is not None
        # Raster "input" band is the 26-D scalar vector.
        assert insp["input_size"] == 26
        assert len(insp["state"]) == 26
        assert len(insp["q_values"]) == 6
        assert 0 <= insp["chosen"] < 6
        # Dueling decomposition present and consistent.
        nv = frame["netviz"]
        assert nv["value"] is not None
        assert len(nv["advantages"]) == 6
        assert len(nv["output"]) == 6
        assert nv["hidden_count"] > 0
        assert nv["summary"]["status"] == "READY"
        # Honest architecture label + input-band caption: the raster net is not
        # "APEX DQN" and the 26 scalars are not its whole input.
        assert nv["summary"]["architecture"] == "Raster Dueling (raster31v2)"
        assert nv["input_label"] == "Scalars (26 of raster input)"

    def test_inspector_reports_executed_action(self, raster_ckpt):
        """The inspector carries the action the hero actually took (post-mask).

        ``chosen`` stays the raw greedy argmax; ``executed_action`` is
        hero.last_action — the safety-masked/explored action that moved the
        snake on screen.
        """
        sess = GameSession(checkpoint=raster_ckpt)
        for _ in range(3):
            sess.step()
        hero = next(s for s in sess.game.snakes if s.id == sess.hero_id)
        insp = sess.snapshot()["inspector"]
        if insp is None:  # hero died in the warmup steps: nothing to assert
            pytest.skip("hero died during warmup")
        assert insp["executed_action"] == hero.last_action
        assert insp["executed_action"] is not None
        assert 0 <= insp["executed_action"] < 6

    def test_inspector_groups_label_the_raster_scalar_contract(self, raster_ckpt):
        """Raster group labels describe the scalars actually served, not vector58.

        The inspector's ``input`` band for a raster hero is the 26-D featurizer
        scalar vector, so it must be labeled from that contract. Labeling it with
        the hand-crafted vector layout captions real values with wrong semantics
        (e.g. length/log-length rendered as "Direction (one-hot)").
        """
        from src.simd_env.featurizer import SCALARS_DIM

        sess = GameSession(checkpoint=raster_ckpt)
        for _ in range(3):
            sess.step()
        insp = sess.snapshot()["inspector"]
        groups = insp["groups"]

        # Every label captions values that exist: no group may index off the end.
        assert all(g["end"] <= insp["input_size"] for g in groups)
        # The groups tile the scalar vector exactly: no gaps, no overlap, and the
        # tail lands on SCALARS_DIM. Imported (not hardcoded) so a featurizer
        # scalar-contract change fails loudly here instead of silently
        # desynchronizing the labels again.
        assert insp["input_size"] == SCALARS_DIM
        assert [g["start"] for g in groups] == [0] + [g["end"] for g in groups[:-1]]
        assert groups[-1]["end"] == SCALARS_DIM
        # Labels come from the raster contract, not the vector one.
        names = [g["name"] for g in groups]
        assert "Direction (one-hot)" not in names
        assert "Food density (16 sectors)" not in names
        assert any("Wall dist" in n for n in names)
        assert any("Mass rank" in n for n in names)
        # The frontend picks its food/danger sector radars by width==16 + name.
        # Those are vector-only semantics; no raster group may trip that match.
        assert not any(
            g["end"] - g["start"] == 16
            and ("food" in g["name"].lower() or "danger" in g["name"].lower())
            for g in groups
        )

    def test_hero_raster_matches_featurizer(self, raster_ckpt):
        """The served hero raster equals the shared featurizer's output (E=1).

        Compared at frame 0 (before any step) so the live game state the test
        featurizes is exactly the state the session's per-frame cache captured —
        proving the serving path routes the live game through the SAME featurizer
        the trainer uses, hero-row slice included.
        """
        from src.simd_env.featurizer import build_observations
        from src.simd_env.live_adapter import game_state_to_obs_inputs

        sess = GameSession(checkpoint=raster_ckpt)
        hero_row = [s.id for s in sess.game.snakes].index(sess.hero_id)

        inp = game_state_to_obs_inputs(sess.game)
        obs = build_observations(inp)
        expected_code = np.asarray(obs["tactical_uint8"])[0, hero_row, 0]

        hr = sess.snapshot()["hero_raster"]
        np.testing.assert_array_equal(np.asarray(hr["tactical_code"]), expected_code)


# ---------------------------------------------------------------------------
# Vector path stays unchanged
# ---------------------------------------------------------------------------
class TestVectorPathUnchanged:
    def test_vector_checkpoint_has_no_raster_fields(self, tmp_path):
        """A vector61 session reports vector61 and emits no raster planes."""
        from src.training.apex_policy import ApexPolicy

        # Build and save a minimal vector61 checkpoint so the test never depends
        # on a committed champion file being present.
        path = str(tmp_path / "vec.pth")
        policy = ApexPolicy(input_size=61, hidden_size=512, output_size=6, training=True)
        torch.save(policy.get_state_dict(), path)

        sess = GameSession(checkpoint=path)
        assert sess.obs_spec == "vector61"
        for _ in range(3):
            sess.step()
        frame = sess.snapshot()
        assert frame["obs_spec"] == "vector61"
        assert frame["hero_raster"] is None
        assert frame["session"]["obs_spec"] == "vector61"
        # Honest architecture labels on the vector path too.
        assert frame["architecture"] == "Apex DQN (vector61)"
        assert frame["netviz"]["summary"]["architecture"] == "Apex DQN (vector61)"
        assert frame["netviz"]["input_label"] == "Input (state)"
        # Executed action is serialized alongside the greedy argmax.
        assert "executed_action" in frame["inspector"]
        # Vector inspector remains 61-D with the free-space tail.
        assert frame["inspector"]["input_size"] == 61
        assert frame["inspector"]["free_space"] is not None
        assert len(frame["inspector"]["q_values"]) == 6
        # Vector labels are untouched by the raster grouping path, and satisfy
        # the same "a label never captions values that do not exist" invariant.
        groups = frame["inspector"]["groups"]
        assert groups[0]["name"] == "Direction (one-hot)"
        assert groups[-1] == {"name": "Free-space (L/S/R)", "start": 58, "end": 61}
        assert all(g["end"] <= frame["inspector"]["input_size"] for g in groups)


# ---------------------------------------------------------------------------
# serve.py oversubscription pin
# ---------------------------------------------------------------------------
def test_set_num_threads_pinned_on_app_import():
    """Importing the app pins torch to a single intra-op thread (blueprint §6)."""
    import web.backend.app  # noqa: F401  (import triggers the pin)

    assert torch.get_num_threads() == 1
