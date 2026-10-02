"""SNAKE_SERVE_VETO_VARIANT: v2 (released default) or v5 (boost-aware) on the Watch hero.

Covers the variant selector in web/backend/safety_veto_serving.py: parsing, the v5
fail-closed binding to the v5 STRICT_PASS receipt, every flag/variant combination in
Watch and Play, the log line, and that v2 is unchanged when the variant is unset.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi")

from src.evaluation.safety_veto import FreeSpaceVeto  # noqa: E402
from src.evaluation.safety_veto_v5 import BoostAwareFreeSpaceVeto  # noqa: E402
from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    ENV_PLAY_AI,
    ENV_VARIANT,
    ENV_WATCH_HERO,
    ServingVetoFlags,
)
from web.backend.session import MODE_PLAY, GameSession  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CHAMPION = Path(
    "/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/champion_a5_freespace_20260621.pth"
)
V5_STRICT = Path(
    "/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v5-strict-20261001/run-v1"
)
needs_champion = pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint absent")
needs_v5_intent = pytest.mark.skipif(
    not (V5_STRICT / "intent.json").is_file(), reason="v5 strict intent absent"
)
LOGGER = "web.backend.safety_veto_serving"


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
    """A real vector61 checkpoint both receipt bindings are re-pinned to (test only)."""
    sess = GameSession(safety_veto_flags=ServingVetoFlags())
    path = tmp_path / "pinned_vector61.pth"
    torch.save(sess.policy.get_state_dict(), path)
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(serving, "STRICT_RECEIPT_CHECKPOINT_SHA256", sha)
    monkeypatch.setattr(serving, "V5_STRICT_RECEIPT_CHECKPOINT_SHA256", sha)
    return str(path)


def vetoes(sess):
    return {int(s.id): s.safety_veto for s in sess.game.snakes if getattr(s, "safety_veto", None)}


def set_env(monkeypatch, watch=None, play=None, variant=None):
    for key, value in ((ENV_WATCH_HERO, watch), (ENV_PLAY_AI, play), (ENV_VARIANT, variant)):
        if value is None:
            monkeypatch.delenv(key, raising=False)
        else:
            monkeypatch.setenv(key, value)


def log_lines(caplog):
    return [r.getMessage() for r in caplog.records if r.name == LOGGER]


class TestVariantParsing:
    def test_released_default_is_v2(self):
        assert serving.VARIANT_RELEASED_DEFAULT == "v2"
        assert serving.VARIANTS == ("v2", "v5")
        assert ServingVetoFlags().variant == "v2"
        assert ServingVetoFlags.from_env({}).variant == "v2"

    @pytest.mark.parametrize("value", ["", "   "])
    def test_blank_selects_the_default(self, value):
        assert ServingVetoFlags.from_env({ENV_VARIANT: value}).variant == "v2"

    @pytest.mark.parametrize("value,expected", [("v5", "v5"), (" V5 ", "v5"), ("v2", "v2")])
    def test_known_values(self, value, expected):
        assert ServingVetoFlags.from_env({ENV_VARIANT: value}).variant == expected

    @pytest.mark.parametrize("value", ["v3", "5", "boost", "v5x"])
    def test_unknown_values_are_kept_so_the_build_can_refuse(self, value):
        assert ServingVetoFlags.from_env({ENV_VARIANT: value}).variant == value.lower()

    def test_flags_dict_shape_is_unchanged(self):
        flags = ServingVetoFlags.from_env({ENV_VARIANT: "v5"})
        assert flags.to_dict() == {"watch_hero": True, "play_ai": False}

    def test_variant_does_not_touch_the_master_switch(self):
        off = ServingVetoFlags.from_env({ENV_WATCH_HERO: "0", ENV_VARIANT: "v5"})
        assert off == ServingVetoFlags(False, False, "v5")


class TestV5Identity:
    def test_identity_matches_the_current_tree_and_the_pins(self):
        identity = serving.wrapper_identity_v5()
        for rel, sha in serving.V5_STRICT_RECEIPT_SOURCE_SHA256S.items():
            assert hashlib.sha256((REPO / rel).read_bytes()).hexdigest() == sha
        assert identity["source_sha256s"] == serving.V5_STRICT_RECEIPT_SOURCE_SHA256S
        assert (
            identity["source_sha256"]
            == serving.V5_STRICT_RECEIPT_SOURCE_SHA256S["src/evaluation/safety_veto_v5.py"]
        )
        assert identity["source_path"] == "src/evaluation/safety_veto_v5.py"
        assert identity["method"] == "free-space-veto/v5-boost-aware"
        assert identity["descriptor"] == BoostAwareFreeSpaceVeto().descriptor()

    def test_v5_and_v2_pins_agree_on_shared_bytes(self):
        assert serving.V5_STRICT_RECEIPT_CHECKPOINT_SHA256 == (
            serving.STRICT_RECEIPT_CHECKPOINT_SHA256
        )
        assert serving.V5_STRICT_RECEIPT_SOURCE_SHA256S["src/evaluation/safety_veto.py"] == (
            serving.STRICT_RECEIPT_WRAPPER_SOURCE_SHA256
        )

    @needs_v5_intent
    def test_pins_equal_the_v5_strict_intent_candidate(self):
        intent = json.loads((V5_STRICT / "intent.json").read_text())
        candidate = intent["candidate"]
        recorded = candidate["wrapper_identity"]
        assert candidate["checkpoint_sha256"] == serving.V5_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert recorded["source_sha256s"] == serving.V5_STRICT_RECEIPT_SOURCE_SHA256S
        assert candidate["wrapper_source_sha256"] == recorded["source_sha256"]
        identity = serving.wrapper_identity_v5()
        for key in ("method", "descriptor", "source_sha256", "source_sha256s"):
            assert identity[key] == recorded[key], key

    @needs_v5_intent
    def test_receipt_sha_and_outcome(self):
        receipt = V5_STRICT / "output" / "receipt.json"
        if not receipt.is_file():
            pytest.skip("v5 strict receipt absent")
        sha = hashlib.sha256(receipt.read_bytes()).hexdigest()
        assert sha == serving.V5_STRICT_RECEIPT_SHA256
        closeout = json.loads((V5_STRICT / "output" / "closeout.json").read_text())
        assert closeout["outcome"] == "STRICT_PASS"
        assert closeout["receipt_sha256"] == serving.V5_STRICT_RECEIPT_SHA256


WATCH_VALUES = (None, "1", "0")
PLAY_VALUES = (None, "1")
VARIANT_VALUES = (None, "v2", "v5", "bogus")


class TestAllCombinations:
    """Every WATCH_HERO x PLAY_AI x VARIANT setting, through env-built sessions."""

    @pytest.mark.parametrize(
        "watch,play,variant", list(itertools.product(WATCH_VALUES, PLAY_VALUES, VARIANT_VALUES))
    )
    def test_watch_then_play(self, monkeypatch, pinned, watch, play, variant):
        set_env(monkeypatch, watch, play, variant)
        sess = GameSession(checkpoint=pinned)
        state = sess.safety_veto_state()
        watch_on = watch != "0"
        if not watch_on:
            assert vetoes(sess) == {} and state["scope"] is None
        elif variant == "bogus":
            assert vetoes(sess) == {} and state["active"] is False
            assert "unknown serving veto variant 'bogus'" in state["reason"]
            assert state["variant"] == "bogus"
        else:
            expected = BoostAwareFreeSpaceVeto if variant == "v5" else FreeSpaceVeto
            got = vetoes(sess)
            assert list(got) == [int(sess.game.snakes[0].id)]
            assert type(got[int(sess.game.snakes[0].id)]) is expected
            assert state["variant"] == (variant or "v2") and state["scope"] == "watch_hero"
            assert state["strict_receipt_checkpoint_match"] is True
            assert state["strict_receipt_wrapper_match"] is True
        sess.set_mode(MODE_PLAY)
        assert sess.mode == MODE_PLAY
        play_state = sess.safety_veto_state()
        if play == "1":  # Play AI keeps v2, whatever the variant
            human = sess._find_human()
            got = vetoes(sess)
            assert sorted(got) == sorted(int(s.id) for s in sess.game.snakes if s is not human)
            assert all(type(v) is FreeSpaceVeto for v in got.values())
            assert play_state["variant"] == "v2" and play_state["scope"] == "play_ai"
        else:
            assert vetoes(sess) == {} and play_state["active"] is False
            assert play_state["variant"] is None


class TestV2UnchangedWhenVariantUnset:
    def _run(self, monkeypatch, pinned, variant, frames=25):
        from src.scripts.eval_cli import set_seed

        set_env(monkeypatch, None, None, variant)
        set_seed(4242)
        sess = GameSession(checkpoint=pinned)
        trace = []
        for _ in range(frames):
            sess.step()
            sess.snapshot()
            trace.append(
                [
                    (int(s.id), tuple(int(v) for v in s.head), len(s.segments), bool(s.is_alive))
                    for s in sess.game.snakes
                ]
            )
        return sess, trace

    def test_unset_equals_explicit_v2(self, monkeypatch, pinned):
        a, trace_a = self._run(monkeypatch, pinned, None)
        b, trace_b = self._run(monkeypatch, pinned, "v2")
        assert trace_a == trace_b
        assert a.safety_veto_state() == b.safety_veto_state()
        state = a.safety_veto_state()
        assert state["wrapper"] == serving.wrapper_identity()
        assert state["variant"] == state["variant_requested"] == "v2"
        assert state["diagnostics"] == {}
        assert set(state["wrapper"]) == {"method", "descriptor", "source_path", "source_sha256"}

    def test_v2_does_not_depend_on_the_v5_pins(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V5_STRICT_RECEIPT_CHECKPOINT_SHA256", "0" * 64)
        monkeypatch.setattr(serving, "V5_STRICT_RECEIPT_SOURCE_SHA256S", {})
        sess = GameSession(checkpoint=pinned)
        assert [type(v) for v in vetoes(sess).values()] == [FreeSpaceVeto]

    def test_v2_refusal_text_is_unchanged(self, no_default_checkpoint):
        sess = GameSession(checkpoint=None, safety_veto_flags=ServingVetoFlags(watch_hero=True))
        assert sess.safety_veto_state()["reason"] == (
            "no strict-gate evidence for this checkpoint (sha256 none: untrained weights)"
        )


class TestV5FailClosed:
    def test_untrained_weights_are_never_wrapped(self, no_default_checkpoint):
        flags = ServingVetoFlags(watch_hero=True, variant="v5")
        sess = GameSession(checkpoint=None, safety_veto_flags=flags)
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and "no strict-gate evidence" in state["reason"]
        assert "under variant v5" in state["reason"]

    def test_non_receipt_checkpoint_is_never_wrapped(self, monkeypatch, pinned):
        monkeypatch.setattr(serving, "V5_STRICT_RECEIPT_CHECKPOINT_SHA256", "f" * 64)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(True, False, "v5"))
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["strict_receipt_checkpoint_match"] is False
        assert state["wrapper"] is None and "under variant v5" in state["reason"]

    @pytest.mark.parametrize("rel", sorted(serving.V5_STRICT_RECEIPT_SOURCE_SHA256S))
    def test_each_changed_source_refuses(self, monkeypatch, pinned, rel):
        pins = dict(serving.V5_STRICT_RECEIPT_SOURCE_SHA256S, **{rel: "0" * 64})
        monkeypatch.setattr(serving, "V5_STRICT_RECEIPT_SOURCE_SHA256S", pins)
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(True, False, "v5"))
        state = sess.safety_veto_state()
        assert vetoes(sess) == {} and state["active"] is False
        assert state["strict_receipt_wrapper_match"] is False and rel in state["reason"]

    def test_unknown_variant_refuses_watch_but_not_play(self, pinned):
        flags = ServingVetoFlags(True, True, "v4")
        sess = GameSession(checkpoint=pinned, safety_veto_flags=flags)
        assert vetoes(sess) == {} and "unknown serving veto variant" in sess.safety_veto.reason
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) and all(type(v) is FreeSpaceVeto for v in vetoes(sess).values())


class TestLogLine:
    def test_v5_line_names_variant_and_match_flags(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        set_env(monkeypatch, variant="v5")
        sess = GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        assert line.startswith(serving.LOG_PREFIX) and "active=True" in line
        assert "variant=v5 variant_requested=v5 scope=watch_hero" in line
        assert f"wrapped_ids=[{int(sess.game.snakes[0].id)}]" in line
        assert "strict_checkpoint_match=True wrapper_sources_match=True" in line
        assert "wrapper_method=free-space-veto/v5-boost-aware" in line
        assert serving.V5_STRICT_RECEIPT_SOURCE_SHA256S[serving.V5_SOURCE_PATH] in line

    def test_v2_default_line_names_variant(self, caplog, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        assert "active=True variant=v2 variant_requested=v2 scope=watch_hero" in line
        assert "strict_checkpoint_match=True wrapper_sources_match=True" in line
        assert serving.STRICT_RECEIPT_WRAPPER_SOURCE_SHA256 in line

    def test_refusals_are_logged(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        set_env(monkeypatch, variant="v9")
        GameSession(checkpoint=pinned)
        (line,) = log_lines(caplog)
        assert "active=False variant=v9" in line and "unknown serving veto variant" in line

    def test_off_logs_nothing(self, caplog, monkeypatch, pinned):
        caplog.set_level(logging.INFO, logger=LOGGER)
        set_env(monkeypatch, watch="0", variant="v5")
        sess = GameSession(checkpoint=pinned)
        sess.set_mode(MODE_PLAY)
        assert log_lines(caplog) == [] and vetoes(sess) == {}


class TestV5Served:
    def test_counters_and_diagnostics_track_greedy_decisions(self, pinned):
        flags = ServingVetoFlags(watch_hero=True, variant="v5")
        sess = GameSession(checkpoint=pinned, safety_veto_flags=flags)
        hero = sess.game.snakes[0]
        veto = hero.safety_veto
        assert isinstance(veto, BoostAwareFreeSpaceVeto)
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
        assert state["wrapper"] == serving.wrapper_identity_v5()
        json.dumps(state)
        sess.reset_game()  # soft reset keeps the wrapped snake object
        assert hero.safety_veto is veto

    def test_watch_hero_only_others_unwrapped(self, pinned):
        sess = GameSession(checkpoint=pinned, safety_veto_flags=ServingVetoFlags(True, False, "v5"))
        assert list(vetoes(sess)) == [int(sess.game.snakes[0].id)]
        assert len(sess.game.snakes) > 1


@needs_champion
class TestRealChampion:
    def test_env_v5_binds_the_real_champion(self, monkeypatch, caplog):
        caplog.set_level(logging.INFO, logger=LOGGER)
        set_env(monkeypatch, variant="v5")
        sess = GameSession(checkpoint=str(CHAMPION))
        state = sess.safety_veto_state()
        assert state["active"] is True and state["variant"] == "v5"
        assert state["checkpoint_sha256"] == serving.V5_STRICT_RECEIPT_CHECKPOINT_SHA256
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["strict_receipt_wrapper_match"] is True
        assert "strict_checkpoint_match=True wrapper_sources_match=True" in log_lines(caplog)[0]
        sess.set_mode(MODE_PLAY)
        assert vetoes(sess) == {}  # release config: Play AI stays unwrapped
