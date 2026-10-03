"""SNAKE_SERVE_VETO_VARIANT=v8: opt-in space-and-head veto (lambda=8) on the Watch hero.

Covers the v8 branch of web/backend/safety_veto_serving.py: selection by env, the
fail-closed binding to the v8 STRICT_PASS receipt (checkpoint, seven veto sources, the
gated lambda, head layer on, no diagnostic reference_lambda), Watch-hero-only scope, Play
unaffected, the log line, and that the released default is still v7 (v8 is not released by
this change).
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
from src.evaluation.safety_veto_v7 import SpacePreferenceVeto  # noqa: E402
from src.evaluation.safety_veto_v8 import SpaceAndHeadVeto  # noqa: E402
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
V8_STRICT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v8-strict-20261003/run-v1"
)
needs_champion = pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint absent")
needs_v8_intent = pytest.mark.skipif(
    not (V8_STRICT / "intent.json").is_file(), reason="v8 strict intent absent"
)
LOGGER = "web.backend.safety_veto_serving"
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"


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
        "V8_STRICT_RECEIPT_CHECKPOINT_SHA256",
    ):
        monkeypatch.setattr(serving, name, sha)
    return str(path)


def vetoes(sess):
    return {int(s.id): s.safety_veto for s in sess.game.snakes if getattr(s, "safety_veto", None)}


def log_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


V8_FLAGS = ServingVetoFlags(watch_hero=True, play_ai=False, variant="v8")


class TestSelection:
    def test_v8_is_opt_in_and_v7_stays_the_released_default(self):
        assert serving.VARIANT_V8 == "v8" and "v8" in serving.VARIANTS
        assert serving.VARIANT_RELEASED_DEFAULT == "v7"
        assert ServingVetoFlags().variant == "v7"
        assert ServingVetoFlags.from_env({}).variant == "v7"

    @pytest.mark.parametrize("value", ["v8", " V8 ", "v8\n"])
    def test_env_selects_v8(self, value):
        assert ServingVetoFlags.from_env({ENV_VARIANT: value}).variant == "v8"

    def test_env_v8_wraps_only_the_watch_hero(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v8")
        sess = GameSession(checkpoint=pinned)
        got = vetoes(sess)
        hero = int(sess.game.snakes[0].id)
        assert list(got) == [hero] and len(sess.game.snakes) > 1
        veto = got[hero]
        assert type(veto) is SpaceAndHeadVeto and veto.lam == 8.0
        assert veto.method == V8_METHOD
        # Exactly the strict candidate: head layer on, no diagnostic reference v7.
        assert veto.head_avoidance is True
        assert veto.reference_lambda is None and veto._reference is None
        state = sess.safety_veto_state()
        assert state["active"] is True and state["scope"] == "watch_hero"
        assert state["variant"] == state["variant_requested"] == "v8"
        assert state["reason"] is None
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["strict_receipt_wrapper_match"] is True
        assert state["wrapper"] == serving.wrapper_identity_v8()

    def test_v8_never_affects_play(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v8")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False and state["variant"] is None
        assert state["reason"] == "no serving veto flag applies to play mode"

    def test_v8_with_play_ai_keeps_v2_in_play(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_VARIANT, "v8")
        monkeypatch.setenv(ENV_PLAY_AI, "1")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        got = vetoes(sess)
        assert got and all(type(v) is FreeSpaceVeto for v in got.values())
        assert sess.safety_veto_state()["variant"] == "v2"

    def test_master_switch_off_beats_v8(self, monkeypatch, pinned):
        monkeypatch.setenv(ENV_WATCH_HERO, "0")
        monkeypatch.setenv(ENV_VARIANT, "v8")
        sess = GameSession(checkpoint=pinned)
        assert vetoes(sess) == {}
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) == {}


class TestReleasedDefaultStillV7:
    def test_unset_env_serves_v7_not_v8(self, pinned):
        sess = GameSession(checkpoint=pinned)
        assert type(sess.game.snakes[0].safety_veto) is SpacePreferenceVeto
        assert sess.safety_veto_state()["variant"] == "v7"

    def test_v7_does_not_depend_on_the_v8_pins(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_CHECKPOINT_SHA256", "0" * 64)
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_SOURCE_SHA256S", {})
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_METHOD", "nope")
        sess = GameSession(checkpoint=pinned)
        assert [type(v) for v in vetoes(sess).values()] == [SpacePreferenceVeto]
        assert sess.safety_veto_state()["reason"] is None


class TestV8Identity:
    def test_identity_matches_the_current_tree_and_the_pins(self):
        identity = serving.wrapper_identity_v8()
        assert len(serving.V8_STRICT_RECEIPT_SOURCE_SHA256S) == 7
        for rel, sha in serving.V8_STRICT_RECEIPT_SOURCE_SHA256S.items():
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha
        assert identity["source_sha256s"] == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert identity["source_sha256"] == (
            serving.V8_STRICT_RECEIPT_SOURCE_SHA256S["src/evaluation/safety_veto_v8.py"]
        )
        assert identity["source_path"] == "src/evaluation/safety_veto_v8.py"
        assert identity["method"] == serving.V8_STRICT_RECEIPT_METHOD == V8_METHOD
        assert identity["descriptor"] == SpaceAndHeadVeto(8.0).descriptor()
        assert identity["descriptor"]["space_preference_lambda"] == 8.0
        assert identity["descriptor"]["head_avoidance"] is True
        assert "reference_lambda" not in identity["descriptor"]

    def test_v8_pins_agree_with_v7_pins_on_shared_bytes(self):
        assert serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256 == (
            serving.V7_STRICT_RECEIPT_CHECKPOINT_SHA256
        )
        for rel, sha in serving.V7_STRICT_RECEIPT_SOURCE_SHA256S.items():
            assert serving.V8_STRICT_RECEIPT_SOURCE_SHA256S[rel] == sha

    @needs_v8_intent
    def test_pins_equal_the_v8_strict_intent_candidate(self):
        raw = (V8_STRICT / "intent.json").read_bytes()
        assert hashlib.sha256(raw).hexdigest() == serving.V8_STRICT_INTENT_SHA256
        intent = json.loads(raw)
        candidate = intent["arms"]["candidate"]
        assert candidate["checkpoint_sha256"] == serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert candidate["method"] == serving.V8_STRICT_RECEIPT_METHOD
        assert candidate["source_sha256s"] == serving.V8_STRICT_RECEIPT_SOURCE_SHA256S
        assert "install_space_and_head_veto(hero, 8.0)" in candidate["install"]
        assert "not installed" in candidate["screen_binding"]["reference_lambda"]
        identity = serving.wrapper_identity_v8()
        assert identity["descriptor"] == candidate["descriptor"]
        assert identity["method"] == candidate["method"]
        assert identity["source_sha256"] == candidate["source_sha256"]
        assert identity["source_sha256s"] == candidate["source_sha256s"]
        # The incumbent the gate beat is the released v7.
        assert intent["arms"]["incumbent"]["method"] == serving.V7_STRICT_RECEIPT_METHOD

    @needs_v8_intent
    def test_receipt_sha_and_outcome(self):
        out = V8_STRICT / "output"
        if not (out / "receipt.json").is_file():
            pytest.skip("v8 strict receipt absent")
        sha = hashlib.sha256((out / "receipt.json").read_bytes()).hexdigest()
        assert sha == serving.V8_STRICT_RECEIPT_SHA256
        receipt = json.loads((out / "receipt.json").read_text())
        assert receipt["decision"] == "STOP_PASS"
        assert receipt["intent_sha256"] == serving.V8_STRICT_INTENT_SHA256
        assert receipt["audit_report_sha256"] == serving.V8_STRICT_AUDIT_REPORT_SHA256
        closeout = json.loads((out / "closeout.json").read_text())
        assert closeout["outcome"] == "STRICT_PASS"
        assert closeout["receipt_sha256"] == serving.V8_STRICT_RECEIPT_SHA256
        assert closeout["intent_sha256"] == serving.V8_STRICT_INTENT_SHA256
        assert closeout["audit_report_sha256"] == serving.V8_STRICT_AUDIT_REPORT_SHA256
        audit = (out / "audit" / "audit.json").read_bytes()
        assert hashlib.sha256(audit).hexdigest() == serving.V8_STRICT_AUDIT_REPORT_SHA256


class TestV8FailClosed:
    def test_untrained_weights_are_never_wrapped(self, no_default_checkpoint):
        sess = GameSession(checkpoint=None, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["reason"] == (
            "no strict-gate evidence for this checkpoint under variant v8 "
            "(sha256 none: untrained weights)"
        )

    def test_non_receipt_checkpoint_is_never_wrapped(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_CHECKPOINT_SHA256", "f" * 64)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["strict_receipt_checkpoint_match"] is False
        assert state["wrapper"] is None and "under variant v8" in state["reason"]

    @pytest.mark.parametrize("rel", sorted(serving.V8_STRICT_RECEIPT_SOURCE_SHA256S))
    def test_each_changed_source_refuses(self, monkeypatch, pinned, rel):
        pins = dict(serving.V8_STRICT_RECEIPT_SOURCE_SHA256S, **{rel: "0" * 64})
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_SOURCE_SHA256S", pins)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False
        assert state["reason"].startswith("v8 wrapper source sha256 differs") and rel in (
            state["reason"]
        )

    def test_missing_source_pin_entry_refuses(self, monkeypatch, pinned):
        pins = dict(serving.V8_STRICT_RECEIPT_SOURCE_SHA256S)
        pins["src/evaluation/safety_veto_v99.py"] = "a" * 64
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_SOURCE_SHA256S", pins)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        assert vetoes(sess) == {}

    def test_ungated_lambda_refuses(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V8_LAMBDA", 4.0)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False
        assert "v8-space-and-head(lambda=4.0)" in state["reason"]
        assert "is not the gated" in state["reason"]

    def test_head_layer_off_refuses(self, monkeypatch, pinned):
        real = serving.SpaceAndHeadVeto

        def no_head(lam, head_avoidance=True, reference_lambda=None):
            return real(lam, head_avoidance=False, reference_lambda=reference_lambda)

        monkeypatch.setattr(serving, "SpaceAndHeadVeto", no_head)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False
        assert "head layer is off" in state["reason"]

    def test_reference_lambda_install_refuses(self, monkeypatch, pinned):
        from src.evaluation import safety_veto_v8

        def with_reference(snake):
            return safety_veto_v8.install_space_and_head_veto(snake, 8.0, reference_lambda=4.0)

        monkeypatch.setattr(serving, "_install_v8", with_reference)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False and state["wrapper"] is None
        assert state["strict_receipt_wrapper_match"] is False
        assert state["reason"] == "installed v8 veto is not the gated strict candidate"

    def test_raster_or_non_apex_policy_is_never_wrapped(self, pinned):
        state = serving.install_serving_vetoes(
            game=None,
            policy=object(),
            mode="watch",
            obs_spec="vector61",
            checkpoint_sha256=serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256,
            flags=V8_FLAGS,
        )
        assert state.active is False and "vector61 Apex policy" in state.reason


class TestV8Served:
    def test_counters_and_diagnostics_track_greedy_decisions(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=V8_FLAGS)
        hero = sess.game.snakes[0]
        veto = hero.safety_veto
        assert isinstance(veto, SpaceAndHeadVeto)
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
        assert diag["v7"]["decisions"] == 6 and diag["v7"]["v5"]["decisions"] == 6
        assert diag["reference_lambda"] is None and diag["reference_decisions"] == 0
        assert diag["head_avoidance"] is True
        json.dumps(state, allow_nan=False)
        sess.reset_game()  # soft reset keeps the wrapped snake object
        assert hero.safety_veto is veto

    def test_log_line_names_v8(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setenv(ENV_VARIANT, "v8")
        sess = GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        hero = int(sess.game.snakes[0].id)
        assert line.startswith(f"{serving.LOG_PREFIX} active=True scope=watch_hero mode=watch ")
        assert f"wrapped_ids=[{hero}]" in line
        v8_sha = serving.V8_STRICT_RECEIPT_SOURCE_SHA256S[serving.V8_SOURCE_PATH]
        assert line.endswith(
            f"strict_checkpoint_match=True wrapper_source_sha256={v8_sha} variant=v8 "
            "variant_requested=v8 wrapper_sources_match=True "
            f"wrapper_method={V8_METHOD} reason=None"
        )

    def test_refusal_is_logged(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setattr(serving, "V8_STRICT_RECEIPT_CHECKPOINT_SHA256", "f" * 64)
        monkeypatch.setenv(ENV_VARIANT, "v8")
        GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        assert "active=False scope=watch_hero" in line and "variant=v8" in line
        assert "reason=no strict-gate evidence for this checkpoint under variant v8" in line


@needs_champion
class TestRealChampion:
    def test_env_v8_binds_the_real_champion(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        monkeypatch.setenv(ENV_VARIANT, "v8")
        sess = GameSession(checkpoint=str(CHAMPION))
        state = sess.safety_veto_state()
        assert state["active"] is True and state["variant"] == "v8"
        assert state["checkpoint_sha256"] == serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["strict_receipt_wrapper_match"] is True
        assert state["wrapper"]["method"] == V8_METHOD
        line = log_lines(caplog)[0]
        assert "safety-veto-serving: active=True scope=watch_hero" in line
        assert f"wrapper_method={V8_METHOD}" in line
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) == {}
