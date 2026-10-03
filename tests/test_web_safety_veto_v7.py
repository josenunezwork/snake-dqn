"""SNAKE_SERVE_VETO_VARIANT=v7: opt-in space-preference veto (lambda=4) on the Watch hero.

Covers the v7 branch of web/backend/safety_veto_serving.py: selection by env, the
fail-closed binding to the v7 STRICT_PASS receipt (checkpoint, four veto sources and the
gated lambda), Watch-hero-only scope, Play unaffected, the log line, and that the released
default is still v5 (v7 is not released by this change).
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi")

from src.evaluation.safety_veto import FreeSpaceVeto  # noqa: E402
from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto  # noqa: E402
from src.evaluation.safety_veto_v7 import SpacePreferenceVeto  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    ENV_PLAY_AI,
    ENV_VARIANT,
    ENV_WATCH_HERO,
    ServingVetoFlags,
)
from web.backend.session import MODE_PLAY, GameSession  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHAMPION = REPO / "saved_snakes" / "champion_a5_freespace_20260621.pth"
V7_STRICT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-strict-20261002/run-v1"
)
needs_champion = pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint absent")
needs_v7_intent = pytest.mark.skipif(
    not (V7_STRICT / "intent.json").is_file(), reason="v7 strict intent absent"
)
LOGGER = "web.backend.safety_veto_serving"
V7_METHOD = "free-space-veto/v7-space-preference(lambda=4.0)"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from src.core import game_config

    for key in (ENV_WATCH_HERO, ENV_PLAY_AI, ENV_VARIANT):
        monkeypatch.delenv(key, raising=False)
    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture
def no_default_checkpoint(monkeypatch, tmp_path):
    import web.backend.session as session_module

    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(tmp_path / "absent.pth"))


@pytest.fixture
def pinned(monkeypatch, tmp_path):
    """A real vector61 checkpoint every receipt binding is re-pinned to (test only)."""
    sess = GameSession(safety_veto_flags=ServingVetoFlags())
    path = tmp_path / "pinned_vector61.pth"
    torch.save(sess.policy.get_state_dict(), path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    for name in (
        "STRICT_RECEIPT_CHECKPOINT_SHA256",
        "V5_STRICT_RECEIPT_CHECKPOINT_SHA256",
        "V7_STRICT_RECEIPT_CHECKPOINT_SHA256",
    ):
        monkeypatch.setattr(serving, name, sha)
    return str(path)


def vetoes(sess):
    return {int(s.id): s.safety_veto for s in sess.game.snakes if getattr(s, "safety_veto", None)}


def log_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


V7_FLAGS = ServingVetoFlags(watch_hero=True, play_ai=False, variant="v7")


class TestSelection:
    def test_v7_is_known_and_the_rollback_target(self):
        # v7 was released 2026-10-03 and superseded by v8 the same day; v7 is v8's rollback.
        assert serving.VARIANT_V7 == "v7" and "v7" in serving.VARIANTS
        assert serving.VARIANT_RELEASED_DEFAULT == "v8"
        assert ServingVetoFlags.from_env({ENV_VARIANT: "v7"}).variant == "v7"

    @pytest.mark.parametrize("value", ["v7", " V7 ", "v7\n"])
    def test_env_selects_v7(self, value):
        assert ServingVetoFlags.from_env({ENV_VARIANT: value}).variant == "v7"

    def test_env_v7_wraps_only_the_watch_hero(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=pinned)
        got = vetoes(sess)
        hero = int(sess.game.snakes[0].id)
        assert list(got) == [hero] and len(sess.game.snakes) > 1
        assert type(got[hero]) is SpacePreferenceVeto and got[hero].lam == 4.0
        assert got[hero].method == V7_METHOD
        state = sess.safety_veto_state()
        assert state["active"] is True and state["scope"] == "watch_hero"
        assert state["variant"] == state["variant_requested"] == "v7"
        assert state["reason"] is None
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["strict_receipt_wrapper_match"] is True
        assert state["wrapper"] == serving.wrapper_identity_v7()

    def test_v7_never_affects_play(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False and state["variant"] is None
        assert state["reason"] == "no serving veto flag applies to play mode"

    def test_v7_with_play_ai_keeps_v2_in_play(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v7")
        monkeypatch.setenv(ENV_PLAY_AI, "1")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        got = vetoes(sess)
        assert got and all(type(v) is FreeSpaceVeto for v in got.values())
        assert sess.safety_veto_state()["variant"] == "v2"

    def test_master_switch_off_beats_v7(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_WATCH_HERO, "0")
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=pinned)
        assert vetoes(sess) == {}
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) == {}


class TestReleasedDefaultV7:
    """The v7 release config, now reached by the rollback env SNAKE_SERVE_VETO_VARIANT=v7."""

    def test_unset_env_serves_v7_on_the_watch_hero_only(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=pinned)
        assert type(sess.game.snakes[0].safety_veto) is SpacePreferenceVeto
        assert all(getattr(s, "safety_veto", None) is None for s in sess.game.snakes[1:])
        state = sess.safety_veto_state()
        assert state["variant"] == "v7" and state["active"] is True and state["reason"] is None

    def test_rollback_to_v5_by_env(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v5")
        sess = GameSession(checkpoint=pinned)
        assert type(sess.game.snakes[0].safety_veto) is BoostAwareFreeSpaceVeto
        assert sess.safety_veto_state()["variant"] == "v5"

    def test_v5_and_v2_do_not_depend_on_the_v7_pins(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_CHECKPOINT_SHA256", "0" * 64)
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_SOURCE_SHA256S", {})
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_METHOD", "nope")
        monkeypatch.setenv(ENV_VARIANT, "v5")
        sess = GameSession(checkpoint=pinned)
        assert [type(v) for v in vetoes(sess).values()] == [BoostAwareFreeSpaceVeto]
        monkeypatch.setenv(ENV_VARIANT, "v2")
        sess = GameSession(checkpoint=pinned)
        assert [type(v) for v in vetoes(sess).values()] == [FreeSpaceVeto]


class TestV7Identity:
    def test_identity_matches_the_current_tree_and_the_pins(self):
        identity = serving.wrapper_identity_v7()
        for rel, sha in serving.V7_STRICT_RECEIPT_SOURCE_SHA256S.items():
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha
        assert identity["source_sha256s"] == serving.V7_STRICT_RECEIPT_SOURCE_SHA256S
        assert identity["source_sha256"] == (
            serving.V7_STRICT_RECEIPT_SOURCE_SHA256S["src/evaluation/safety_veto_v7.py"]
        )
        assert identity["source_path"] == "src/evaluation/safety_veto_v7.py"
        assert identity["method"] == serving.V7_STRICT_RECEIPT_METHOD == V7_METHOD
        assert identity["descriptor"] == SpacePreferenceVeto(4.0).descriptor()
        assert identity["descriptor"]["space_preference_lambda"] == 4.0

    def test_v7_pins_agree_with_v5_pins_on_shared_bytes(self):
        assert serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256 == (
            serving.V5_STRICT_RECEIPT_CHECKPOINT_SHA256
        )
        for rel, sha in serving.V5_STRICT_RECEIPT_SOURCE_SHA256S.items():
            assert serving.V7_STRICT_RECEIPT_SOURCE_SHA256S[rel] == sha

    @needs_v7_intent
    def test_pins_equal_the_v7_strict_intent_candidate(self):
        raw = (V7_STRICT / "intent.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == serving.V7_STRICT_INTENT_SHA256
        candidate = json.loads(raw)["candidate"]
        recorded = candidate["wrapper_identity"]
        assert candidate["checkpoint_sha256"] == serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert candidate["wrapper"] == serving.V7_STRICT_RECEIPT_METHOD
        assert recorded["source_sha256s"] == serving.V7_STRICT_RECEIPT_SOURCE_SHA256S
        identity = serving.wrapper_identity_v7()
        for key in ("method", "descriptor", "source_sha256", "source_sha256s"):
            assert identity[key] == recorded[key], key

    @needs_v7_intent
    def test_receipt_sha_and_outcome(self):
        out = V7_STRICT / "output"
        if not (out / "receipt.json").is_file():
            pytest.skip("v7 strict receipt absent")
        sha = hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()
        assert sha == serving.V7_STRICT_RECEIPT_SHA256
        closeout = json.loads((out / "closeout.json").read_text())
        assert closeout["outcome"] == "STRICT_PASS"
        assert closeout["receipt_sha256"] == serving.V7_STRICT_RECEIPT_SHA256
        assert closeout["intent_sha256"] == serving.V7_STRICT_INTENT_SHA256
        assert closeout["audit_report_sha256"] == serving.V7_STRICT_AUDIT_REPORT_SHA256


class TestV7FailClosed:
    def test_untrained_weights_are_never_wrapped(self, no_default_checkpoint):
        sess = GameSession(checkpoint=None, safety_veto_flags=V7_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["reason"] == (
            "no strict-gate evidence for this checkpoint under variant v7 "
            "(sha256 none: untrained weights)"
        )

    def test_non_receipt_checkpoint_is_never_wrapped(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_CHECKPOINT_SHA256", "f" * 64)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V7_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["strict_receipt_checkpoint_match"] is False
        assert state["wrapper"] is None and "under variant v7" in state["reason"]

    @pytest.mark.parametrize("rel", sorted(serving.V7_STRICT_RECEIPT_SOURCE_SHA256S))
    def test_each_changed_source_refuses(self, monkeypatch, pinned, rel):
        pins = dict(serving.V7_STRICT_RECEIPT_SOURCE_SHA256S, **{rel: "0" * 64})
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_SOURCE_SHA256S", pins)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V7_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False
        assert state["reason"].startswith("v7 wrapper source sha256 differs") and rel in (
            state["reason"]
        )

    def test_missing_source_pin_entry_refuses(self, monkeypatch, pinned):
        pins = dict(serving.V7_STRICT_RECEIPT_SOURCE_SHA256S)
        pins["src/evaluation/safety_veto_v99.py"] = "a" * 64
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_SOURCE_SHA256S", pins)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V7_FLAGS)
        assert vetoes(sess) == {}

    def test_ungated_lambda_refuses(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V7_LAMBDA", 2.0)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V7_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False
        assert "v7-space-preference(lambda=2.0)" in state["reason"]
        assert "is not the gated" in state["reason"]

    def test_raster_or_non_apex_policy_is_never_wrapped(self, pinned):
        state = serving.install_serving_vetoes(
            game=None,
            policy=object(),
            mode="watch",
            obs_spec="vector61",
            checkpoint_sha256=serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256,
            flags=V7_FLAGS,
        )
        assert state.active is False and "vector61 Apex policy" in state.reason


class TestV7Served:
    def test_counters_and_diagnostics_track_greedy_decisions(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V7_FLAGS)
        hero = sess.game.snakes[0]
        veto = hero.safety_veto
        assert isinstance(veto, SpacePreferenceVeto)
        for _ in range(6):
            sess.step()
        counters = veto.counters.to_dict()
        assert (
            counters["decisions"]
            == 6
            == (
                counters["kept_base"]
                + counters["vetoes_applied"]
                + counters["fallback_no_spacious"]
            )
        )
        state = sess.safety_veto_state()
        diag = state["diagnostics"][str(hero.id)]
        assert diag["decisions"] == 6 and diag["vetoes_applied"] == counters["vetoes_applied"]
        assert diag["v5"]["decisions"] == 6
        json.dumps(state, allow_nan=False)
        sess.reset_game()  # soft reset keeps the wrapped snake object
        assert hero.safety_veto is veto

    def test_log_line_names_v7(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        hero = int(sess.game.snakes[0].id)
        assert line.startswith(f"{serving.LOG_PREFIX} active=True scope=watch_hero mode=watch ")
        assert f"wrapped_ids=[{hero}]" in line
        v7_sha = serving.V7_STRICT_RECEIPT_SOURCE_SHA256S[serving.V7_SOURCE_PATH]
        assert line.endswith(
            f"strict_checkpoint_match=True wrapper_source_sha256={v7_sha} variant=v7 "
            "variant_requested=v7 wrapper_sources_match=True "
            f"wrapper_method={V7_METHOD} reason=None"
        )

    def test_refusal_is_logged(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setattr(serving, "V7_STRICT_RECEIPT_CHECKPOINT_SHA256", "f" * 64)
        monkeypatch.setenv(ENV_VARIANT, "v7")
        GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        assert "active=False scope=watch_hero" in line and "variant=v7" in line
        assert "reason=no strict-gate evidence for this checkpoint under variant v7" in line


@needs_champion
class TestRealChampion:
    def test_env_v7_binds_the_real_champion(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession(checkpoint=str(CHAMPION))
        state = sess.safety_veto_state()
        assert state["active"] is True and state["variant"] == "v7"
        assert state["checkpoint_sha256"] == serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["strict_receipt_wrapper_match"] is True
        assert state["wrapper"]["method"] == V7_METHOD
        line = log_lines(caplog)[0]
        assert "safety-veto-serving: active=True scope=watch_hero" in line
        assert f"wrapper_method={V7_METHOD}" in line
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) == {}
