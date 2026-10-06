"""Opt-in (default-off) serving hooks for the free-space safety veto in the web app.

Covers web/backend/safety_veto_serving.py, its single call site in
``GameSession._build_impl``, and the headless serving harness pieces in
research/apex_veto_serving_20261001/serving_run.py that drive the real session.
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi")

from src.evaluation import safety_veto as veto_module  # noqa: E402
from src.evaluation.safety_veto import FreeSpaceVeto  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    ENV_PLAY_AI,
    ENV_WATCH_HERO,
    ServingVetoFlags as _ServingVetoFlags,
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
    monkeypatch.delenv(serving.ENV_VARIANT, raising=False)
    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """A real vector61 checkpoint file the receipt binding is re-pinned to (test only).

    Untrained weights saved to ``tmp_path``; ``STRICT_RECEIPT_CHECKPOINT_SHA256`` is
    monkeypatched to its sha256 so the install mechanics run without the champion
    file. The binding check itself is unchanged and still compares bytes.
    """
    sess = GameSession(safety_veto_flags=ServingVetoFlags())
    path = tmp_path / "pinned_vector61.pth"
    torch.save(sess.policy.get_state_dict(), path)
    monkeypatch.setattr(
        serving, "STRICT_RECEIPT_CHECKPOINT_SHA256", hashlib.sha256(path.read_bytes()).hexdigest()
    )
    return str(path)


def wrapped_ids(sess):
    return sorted(int(s.id) for s in sess.game.snakes if getattr(s, "safety_veto", None))


def control(sess, action, value=None):
    from web.backend.app import _apply_control

    return _apply_control(sess, {"type": "control", "action": action, "value": value}, "conn")


def ServingVetoFlags(*args, **kwargs):  # noqa: N802 - test shim with the class's name
    """Explicit flags default to the v2 wrapper in this file (v2 is the rollback path)."""
    kwargs.setdefault("variant", serving.VARIANT_V2)
    return _ServingVetoFlags(*args, **kwargs)


@pytest.fixture(autouse=True)
def _pin_v2_variant(monkeypatch):
    """These tests cover the v2 wrapper path, which is the rollback since the v5 release
    (2026-10-02). Pin the variant so they keep testing v2 regardless of the released default."""
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v2")


@pytest.fixture
def no_default_checkpoint(monkeypatch, tmp_path):
    """Make ``checkpoint=None`` mean untrained weights even where the champion file exists."""
    import web.backend.session as session_module

    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(tmp_path / "absent.pth"))
    # Since the 2026-10-06 release the registry default is frp3-s12 (found on disk); pin the
    # pre-release champion default so checkpoint=None still means untrained weights here.
    from web.backend import served_checkpoint as registry

    monkeypatch.setattr(registry, "CHECKPOINT_RELEASED_DEFAULT", registry.NAME_CHAMPION)


class TestFlags:
    def test_released_defaults(self):
        # Explicit flags stay off by default; the environment default releases the Watch hero.
        assert _ServingVetoFlags() == _ServingVetoFlags(watch_hero=False, play_ai=False)
        assert _ServingVetoFlags.from_env({}).to_dict() == {"watch_hero": True, "play_ai": False}
        assert serving.WATCH_HERO_RELEASED_DEFAULT is True

    @pytest.mark.parametrize("value", ["1", "true", "TRUE", " yes ", "on"])
    def test_truthy_values_enable_their_flag(self, value):
        assert _ServingVetoFlags.from_env({ENV_WATCH_HERO: value}) == _ServingVetoFlags(True, False)
        assert _ServingVetoFlags.from_env({ENV_PLAY_AI: value}) == _ServingVetoFlags(True, True)

    @pytest.mark.parametrize("value", ["0", "false", "no", "off", " OFF "])
    def test_falsy_values_roll_back_the_watch_hero(self, value):
        flags = _ServingVetoFlags.from_env({ENV_WATCH_HERO: value, ENV_PLAY_AI: value})
        assert flags == _ServingVetoFlags(False, False)

    @pytest.mark.parametrize("value", ["", "2", "enabled"])
    def test_unrecognized_values_keep_the_released_default(self, value):
        flags = _ServingVetoFlags.from_env({ENV_WATCH_HERO: value, ENV_PLAY_AI: value})
        assert flags == _ServingVetoFlags(True, False)

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
    def test_rollback_flag_wraps_nothing_in_watch_or_play(self, monkeypatch):
        monkeypatch.setenv(ENV_WATCH_HERO, "0")
        monkeypatch.delenv(ENV_PLAY_AI, raising=False)
        sess = GameSession()
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False
        assert sess.safety_veto_state()["scope"] is None
        sess.set_mode(MODE_PLAY)
        assert sess.mode == MODE_PLAY
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_released_default_wraps_only_watch_hero_never_play_ai(self, monkeypatch, pinned):
        monkeypatch.delenv(ENV_WATCH_HERO, raising=False)
        monkeypatch.delenv(ENV_PLAY_AI, raising=False)
        sess = GameSession(checkpoint=pinned)
        assert wrapped_ids(sess) == [0]
        assert sess.safety_veto_state()["scope"] == "watch_hero"
        sess.set_mode(MODE_PLAY)
        assert wrapped_ids(sess) == []

    def test_control_state_payload_is_unchanged(self):
        sess = GameSession(safety_veto_flags=ServingVetoFlags(True, True))
        assert "safety_veto" not in sess.control_state()

    def test_explicit_flags_override_the_environment(self, monkeypatch):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        assert wrapped_ids(sess) == []


class TestWatchHero:
    def test_env_flag_wraps_only_the_served_hero(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession(checkpoint=pinned)
        assert sess.mode == MODE_WATCH
        assert wrapped_ids(sess) == [sess.hero_id] == [int(sess.game.snakes[0].id)]
        state = sess.safety_veto_state()
        assert state["active"] is True and state["scope"] == "watch_hero"
        assert state["wrapped_snake_ids"] == [sess.hero_id]
        assert state["wrapper"]["descriptor"] == FreeSpaceVeto().descriptor()
        json.dumps(state)  # JSON-safe report

    def test_watch_flag_does_not_wrap_play(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_WATCH_HERO, "1")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_counters_track_greedy_decisions_and_survive_reset(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(watch_hero=True))
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

    def test_set_hero_does_not_move_the_wrapper(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        other = int(sess.game.snakes[1].id)
        sess.set_hero(other)
        assert sess.hero_id == other
        assert wrapped_ids(sess) == [int(sess.game.snakes[0].id)]


class TestPlayAndTrain:
    def test_play_flag_wraps_every_ai_snake_never_the_human(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(play_ai=True))
        assert wrapped_ids(sess) == []  # Watch build: the play flag does nothing
        assert control(sess, "set_mode", "play") is None
        human = sess._find_human()
        assert human is not None and getattr(human, "safety_veto", None) is None
        expected = sorted(int(s.id) for s in sess.game.snakes if s is not human)
        assert wrapped_ids(sess) == expected
        assert len(expected) == len(sess.game.snakes) - 1
        assert sess.safety_veto_state()["scope"] == "play_ai"

    def test_play_rebuild_reinstalls_on_the_new_roster(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(play_ai=True))
        sess.set_mode(MODE_PLAY)
        sess.set_play_opponents(2)
        assert len(sess.game.snakes) == 3
        assert len(wrapped_ids(sess)) == 2

    def test_train_mode_is_never_wrapped(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(True, True))
        sess.set_mode(MODE_TRAIN, override_reward_contract=True)
        if sess.mode != MODE_TRAIN:
            pytest.skip(f"train mode unavailable here: {sess.last_error}")
        assert wrapped_ids(sess) == []
        assert sess.safety_veto_state()["active"] is False

    def test_non_vector_policy_is_refused_with_a_reason(self, no_default_checkpoint):
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        state = install_serving_vetoes(
            sess.game, sess.policy, "watch", "raster31v2", None, ServingVetoFlags(True, True)
        )
        assert state.active is False
        assert "vector61" in state.reason
        assert wrapped_ids(sess) == []


def _other_vector61(tmp_path, name="other_vector61.pth"):
    """A real vector61 checkpoint that is NOT the one the receipt binding names."""
    torch.manual_seed(1234)
    sess = GameSession(safety_veto_flags=ServingVetoFlags())
    path = tmp_path / name
    torch.save(sess.policy.get_state_dict(), path)
    return path


class TestFailClosedBinding:
    """The wrapper is installed only on the strict-gated checkpoint + wrapper bytes."""

    def test_no_checkpoint_untrained_weights_are_never_wrapped(self, no_default_checkpoint):
        sess = GameSession(checkpoint=None, safety_veto_flags=ServingVetoFlags(True, True))
        assert sess.obs_spec == "vector61"
        state = sess.safety_veto_state()
        assert wrapped_ids(sess) == [] and state["active"] is False
        assert state["checkpoint_sha256"] is None
        assert "no strict-gate evidence" in state["reason"]
        sess.set_mode(MODE_PLAY)
        assert wrapped_ids(sess) == [] and "no strict-gate evidence" in sess.safety_veto.reason

    def test_non_champion_vector61_checkpoint_is_never_wrapped(self, tmp_path):
        path = _other_vector61(tmp_path)
        sess = GameSession(checkpoint=str(path), safety_veto_flags=ServingVetoFlags(True, True))
        assert sess.obs_spec == "vector61"
        state = sess.safety_veto_state()
        assert state["checkpoint_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert state["strict_receipt_checkpoint_match"] is False
        assert state["active"] is False and wrapped_ids(sess) == []
        assert "no strict-gate evidence" in state["reason"] and state["wrapper"] is None

    def test_client_load_checkpoint_of_another_vector61_unwraps(
        self, monkeypatch, tmp_path, pinned
    ):
        from web.backend import session as web_session

        saved = tmp_path / "saved"
        saved.mkdir()
        other = _other_vector61(saved)
        monkeypatch.setattr(web_session, "SAVED_DIR", str(saved))
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        assert wrapped_ids(sess) == [int(sess.game.snakes[0].id)]
        assert control(sess, "load_checkpoint", other.name) is None
        assert sess.last_error is None and sess.checkpoint_path.endswith(other.name)
        assert wrapped_ids(sess) == []
        assert "no strict-gate evidence" in sess.safety_veto_state()["reason"]

    def test_changed_wrapper_source_is_never_wrapped(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "STRICT_RECEIPT_WRAPPER_SOURCE_SHA256", "0" * 64)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(True, True))
        state = sess.safety_veto_state()
        assert state["active"] is False and wrapped_ids(sess) == []
        assert "wrapper source sha256" in state["reason"]


class TestBuildLogLine:
    LOGGER = "web.backend.safety_veto_serving"

    def _lines(self, caplog):
        return [r.getMessage() for r in caplog.records if r.name == self.LOGGER]

    def test_one_info_line_per_build_when_requested(self, caplog, pinned):
        caplog.set_level(logging.INFO, logger=self.LOGGER)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        lines = self._lines(caplog)
        assert len(lines) == 1 and lines[0].startswith(serving.LOG_PREFIX)
        assert "active=True" in lines[0] and "scope=watch_hero" in lines[0]
        assert f"wrapped_ids=[{sess.hero_id}]" in lines[0]
        assert "strict_checkpoint_match=True" in lines[0]
        assert serving.STRICT_RECEIPT_WRAPPER_SOURCE_SHA256 in lines[0]
        sess.set_mode(MODE_PLAY)  # a rebuild logs again, with the reason it is off
        lines = self._lines(caplog)
        assert len(lines) == 2 and "active=False" in lines[1]
        assert "no serving veto flag applies to play mode" in lines[1]

    def test_refusal_is_logged_with_its_reason(self, caplog, no_default_checkpoint):
        caplog.set_level(logging.INFO, logger=self.LOGGER)
        GameSession(checkpoint=None, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        (line,) = self._lines(caplog)
        assert "active=False" in line and "no strict-gate evidence" in line

    def test_flags_off_logs_nothing_and_payload_is_unchanged(self, caplog, pinned):
        caplog.set_level(logging.INFO, logger=self.LOGGER)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags())
        sess.set_mode(MODE_PLAY)
        assert self._lines(caplog) == []
        assert "safety_veto" not in sess.control_state()


class TestForcedVetoChangesTheExecutedAction:
    """A cramped hero position: the served wrapper replaces the masked argmax."""

    def test_step_vetoes_the_cramped_argmax(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        game, hero = sess.game, sess.game.snakes[0]
        ss, y = int(hero.segment_size), 10

        def place(snake, cells, direction):
            snake.segments = [(x * ss, yy * ss) for x, yy in cells]
            snake.length, snake.direction, snake.is_alive = len(cells), direction, True

        # Hero at cell (1, y) heading up. Its left turn enters (0, y): a 1-cell nook
        # walled by the arena edge, its own head and two other snakes' bodies.
        place(hero, [(1, y), (1, y + 1), (1, y + 2)], (0, -1))
        place(game.snakes[1], [(0, y - 1), (0, y - 2), (0, y - 3)], (0, -1))
        place(game.snakes[2], [(0, y + 1), (0, y + 2), (0, y + 3)], (0, 1))
        for k, snake in enumerate(game.snakes[3:]):
            place(snake, [(30 + 3 * k, 30), (30 + 3 * k, 31), (30 + 3 * k, 32)], (0, -1))

        class PrefersLeft(torch.nn.Module):  # masked argmax = 0 (left), then straight
            def forward(self, x):
                return torch.tensor([[10.0, 5.0, 4.0, 1.0, 1.0, 1.0]]).expand(x.shape[0], 6)

        sess.policy.dqn = PrefersLeft()
        veto = hero.safety_veto
        seen, apply = [], veto.apply

        def spy(snake, others, masked_q, mask, base):
            action = apply(snake, others, masked_q, mask, base)
            seen.append((int(masked_q.argmax().item()), int(base), int(action)))
            return action

        veto.apply = spy
        sess.step()
        assert seen == [(0, 0, 1)]  # masked argmax left, executed straight
        assert veto.counters.vetoes_applied == 1 and veto.counters.decisions == 1
        assert sess.safety_veto_state()["counters"][str(hero.id)]["vetoes_applied"] == 1
        assert tuple(hero.direction) == (0, -1) and hero.is_alive
        assert tuple(hero.segments[0]) == (1 * ss, (y - 1) * ss)


class TestHarnessControl:
    """serving_run.control: a None reply is not success; ``last_error`` decides."""

    @pytest.fixture()
    def run_mod(self):
        from research.apex_veto_serving_20261001 import serving_run

        return serving_run

    def test_bad_control_value_fails(self, run_mod):
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        run_mod.control(sess, "set_mode", "play")
        assert sess.mode == MODE_PLAY
        with pytest.raises(RuntimeError, match="Invalid set_play_opponents control"):
            run_mod.control(sess, "set_play_opponents", "not-a-number")
        with pytest.raises(RuntimeError, match="last_error"):
            run_mod.control(sess, "no_such_action")

    def test_stale_error_does_not_fail_a_good_control(self, run_mod):
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        sess.last_error = "left over from an earlier control"
        run_mod.control(sess, "set_speed", 12)
        assert sess.last_error is None

    def test_failed_control_marks_the_play_episode_as_error(self, run_mod, monkeypatch):
        sess = GameSession(safety_veto_flags=ServingVetoFlags())
        run_mod.control(sess, "set_mode", "play")
        ctx = {"checkpoint": "untrained", "checkpoint_sha256": None, "intent_sha256": "0" * 64}

        def broken():  # new_game's handler raises inside _apply_control
            raise ValueError("boom")

        monkeypatch.setattr(sess, "reset_game", broken)
        record = run_mod.run_play_episode(sess, 0, 12, 5, ctx)
        assert record["status"] == "error"
        assert "new_game" in record["error"] and "boom" in record["error"]


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
        assert [len(seeds[k]) for k in ("watch", "play", "parity")] == [25, 25, 2]
        assert run_mod.seed_report(seeds)["disjoint"] is True
