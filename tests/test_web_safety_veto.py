"""Opt-in (default-off) serving hooks for the free-space safety veto in the web app.

Covers web/backend/safety_veto_serving.py, its single call site in
``GameSession._build_impl``, and the headless serving harness pieces in
research/apex_veto_serving_20261001/serving_run.py that drive the real session.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

pytest.importorskip("fastapi")

from src.evaluation import safety_veto as veto_module  # noqa: E402
from src.evaluation.safety_veto import FreeSpaceVeto  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    ENV_PLAY_AI,
    ENV_WATCH_HERO,
    ServingVetoFlags,
    install_serving_vetoes,
)
from web.backend.session import (  # noqa: E402
    MODE_PLAY,
    MODE_TRAIN,
    MODE_WATCH,
    GameSession,
)

CHAMPION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/champion_a5_freespace_20260621.pth"
)
needs_champion = pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint absent")


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    """No ambient flags, and restore the global config singleton GameSession replaces."""
    from src.core import game_config

    monkeypatch.delenv(ENV_WATCH_HERO, raising=False)
    monkeypatch.delenv(ENV_PLAY_AI, raising=False)
    prev = game_config._current_config
    yield
    game_config._current_config = prev


def wrapped_ids(sess):
    return sorted(int(s.id) for s in sess.game.snakes if getattr(s, "safety_veto", None))


def control(sess, action, value=None):
    from web.backend.app import _apply_control

    return _apply_control(sess, {"type": "control", "action": action, "value": value}, "conn")


class TestFlags:
    def test_default_is_off(self):
        assert ServingVetoFlags() == ServingVetoFlags(watch_hero=False, play_ai=False)
        assert ServingVetoFlags.from_env({}).to_dict() == {"watch_hero": False, "play_ai": False}

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy_values_enable_only_their_flag(self, value):
        assert ServingVetoFlags.from_env({ENV_WATCH_HERO: value}) == ServingVetoFlags(True, False)
        assert ServingVetoFlags.from_env({ENV_PLAY_AI: value}) == ServingVetoFlags(False, True)

    @pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "2", "enabled"])
    def test_anything_else_is_off(self, value):
        flags = ServingVetoFlags.from_env({ENV_WATCH_HERO: value, ENV_PLAY_AI: value})
        assert flags == ServingVetoFlags()

    def test_wrapper_identity_binds_the_unchanged_source(self):
        identity = serving.wrapper_identity()
        source = Path(veto_module.__file__).read_bytes()
        assert identity["source_sha256"] == hashlib.sha256(source).hexdigest()
        assert identity["source_path"] == "src/evaluation/safety_veto.py"
        assert identity["descriptor"] == FreeSpaceVeto().descriptor()
        assert identity["method"] == "free-space-veto/v2-speed-preserving"
        # The strict-gated wrapper bytes are what this branch serves.
        assert identity["source_sha256"] == serving.STRICT_RECEIPT_WRAPPER_SOURCE_SHA256


class TestDefaultOff:
    def test_default_session_wraps_nothing_in_watch_or_play(self):
        sess = GameSession()
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False
        assert sess.safety_veto_state()["scope"] is None
        sess.set_mode(MODE_PLAY)
        assert sess.mode == MODE_PLAY
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_control_state_payload_is_unchanged(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(True, True))
        assert "safety_veto" not in sess.control_state()

    def test_explicit_flags_override_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        assert wrapped_ids(sess) == []


class TestWatchHero:
    def test_env_flag_wraps_only_the_served_hero(self, monkeypatch):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession()
        assert sess.mode == MODE_WATCH
        assert wrapped_ids(sess) == [sess.hero_id] == [int(sess.game.snakes[0].id)]
        state = sess.safety_veto_state()
        assert state["active"] is True and state["scope"] == "watch_hero"
        assert state["wrapped_snake_ids"] == [sess.hero_id]
        assert state["wrapper"]["descriptor"] == FreeSpaceVeto().descriptor()
        json.dumps(state)  # JSON-safe report

    def test_watch_flag_does_not_wrap_play(self, monkeypatch):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession()
        sess.set_mode(MODE_PLAY)
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_counters_track_greedy_decisions_and_survive_reset(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(watch_hero=True))
        hero = sess.game.snakes[0]
        veto = hero.safety_veto
        for _ in range(5):
            sess.step()
        counters = veto.counters.to_dict()
        assert counters["decisions"] == 5
        assert counters["decisions"] == (
            counters["kept_base"] + counters["vetoes_applied"] + counters["fallback_no_spacious"]
        )
        sess.reset_game()  # the browser's new_game: soft reset keeps the snake objects
        assert hero.safety_veto is veto
        sess.step()
        assert veto.counters.decisions == 6

    def test_set_hero_does_not_move_the_wrapper(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(watch_hero=True))
        other = int(sess.game.snakes[1].id)
        sess.set_hero(other)
        assert sess.hero_id == other
        assert wrapped_ids(sess) == [int(sess.game.snakes[0].id)]


class TestPlayAndTrain:
    def test_play_flag_wraps_every_ai_snake_never_the_human(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(play_ai=True))
        assert wrapped_ids(sess) == []  # Watch build: the play flag does nothing
        assert control(sess, "set_mode", "play") is None
        human = sess._find_human()
        assert human is not None and getattr(human, "safety_veto", None) is None
        expected = sorted(int(s.id) for s in sess.game.snakes if s is not human)
        assert wrapped_ids(sess) == expected
        assert len(expected) == len(sess.game.snakes) - 1
        assert sess.safety_veto_state()["scope"] == "play_ai"

    def test_play_rebuild_reinstalls_on_the_new_roster(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(play_ai=True))
        sess.set_mode(MODE_PLAY)
        sess.set_play_opponents(2)
        assert len(sess.game.snakes) == 3
        assert len(wrapped_ids(sess)) == 2

    def test_train_mode_is_never_wrapped(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(True, True))
        sess.set_mode(MODE_TRAIN, override_reward_contract=True)
        if sess.mode != MODE_TRAIN:
            pytest.skip(f"train mode unavailable here: {sess.last_error}")
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_non_vector_policy_is_refused_with_a_reason(self):
        sess = GameSession()
        state = install_serving_vetoes(
            sess.game, sess.policy, "watch", "raster31v2", None, ServingVetoFlags(True, True)
        )
        assert state.active is False
        assert "vector61" in state.reason
        assert wrapped_ids(sess) == []


@needs_champion
class TestServingHarness:
    """The research harness drives the real session (tiny horizons only)."""

    @pytest.fixture()
    def run_mod(self):
        from research.apex_veto_serving_20261001 import serving_run

        return serving_run

    def _ctx(self, run_mod):
        return {
            "checkpoint": CHAMPION,
            "checkpoint_sha256": run_mod.CHAMPION_SHA256,
            "intent_sha256": "0" * 64,
        }

    def test_session_binds_the_champion_sha(self):
        sess = GameSession(checkpoint=str(CHAMPION), safety_veto_flags=ServingVetoFlags(True))
        state = sess.safety_veto_state()
        assert state["checkpoint_sha256"] == serving.STRICT_RECEIPT_CHECKPOINT_SHA256
        assert state["strict_receipt_checkpoint_match"] is True

    def test_watch_episode_counters_equal_decision_frames(self, run_mod):
        record = run_mod.run_watch_episode(0, 11, 40, self._ctx(run_mod))
        assert record["status"] == "complete", record["error"]
        hero = str(record["served"]["hero_id"])
        assert record["veto"]["per_snake"][hero]["decisions"] == (
            record["decision_frames"]["per_snake"][hero]
        )
        assert record["served"]["config_basename"] == "mechanics_v2.yaml"

    def test_play_episode_through_dispatch(self, run_mod):
        ctx = self._ctx(run_mod)
        sess = run_mod.play_session(ctx)
        record = run_mod.run_play_episode(sess, 0, 12, 40, ctx)
        assert record["status"] == "complete", record["error"]
        assert record["episode"]["inputs_sent"] >= 1
        assert record["veto"]["total"]["decisions"] == record["decision_frames"]["total"] > 0
        assert str(record["served"]["human_id"]) not in record["veto"]["per_snake"]

    def test_parity_probe_matches_rollout(self, run_mod):
        probe = run_mod.run_parity_probe(0, 13, 40, self._ctx(run_mod))
        assert probe["status"] == "complete", probe["error"]
        assert probe["first_divergence_frame"] is None
        assert probe["frames_compared"] == 40
        assert probe["trace_sha256_equal"] and probe["veto_counters_equal"]

    def test_seeds_are_fresh_and_deterministic(self, run_mod):
        seeds = run_mod.world_seeds(run_mod.SEED_DOMAIN, run_mod.COUNTS)
        assert seeds == run_mod.world_seeds(run_mod.SEED_DOMAIN, run_mod.COUNTS)
        assert [len(seeds[k]) for k in ("watch", "play", "parity")] == [1, 49, 2]
        assert run_mod.seed_report(seeds)["disjoint"] is True
