"""Focused contract tests for the released T15 mixed Watch roster."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi")

from web.backend import served_checkpoint as registry  # noqa: E402
from web.backend import session as session_module  # noqa: E402
from web.backend.safety_veto_serving import ServingVetoFlags, ServingVetoState  # noqa: E402
from web.backend.served_checkpoint import ServedCheckpointChoice  # noqa: E402
from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_roster_env(monkeypatch):
    for key in (
        "SNAKE_SERVE_WATCH_ROSTER",
        "SNAKE_SERVE_CHECKPOINT",
        "SNAKE_SERVE_VETO_WATCH_HERO",
        "SNAKE_SERVE_VETO_PLAY_AI",
        "SNAKE_SERVE_VETO_VARIANT",
    ):
        monkeypatch.delenv(key, raising=False)


@pytest.fixture(autouse=True)
def _released_checkpoint_fixture(monkeypatch, tmp_path):
    """Use local serialized release-shaped inputs, never workstation artifacts.

    The registry resolver and pin parser are deliberately real.  Only the
    fixture boundary supplies the two release identities: their bytes are
    synthetic, so the hash reader returns the immutable release digest after
    asserting that every requested path is under this test's temporary root.
    """
    from src.model.ego2s_network import Ego2sNet, save_ego2s_checkpoint
    from src.training.apex_policy import ApexPolicy

    b8 = tmp_path / registry.B8_S3_FILENAME
    frp3 = tmp_path / registry.FRP3_S12_FILENAME
    champion = tmp_path / registry.CHAMPION_FILENAME
    save_ego2s_checkpoint(b8, Ego2sNet(local_channels=7, n_scalars=30), {"fixture": "b8"})
    opponent = ApexPolicy(input_size=61, hidden_size=512, output_size=6, training=False)
    torch.save(opponent.get_state_dict(), frp3)
    torch.save(opponent.get_state_dict(), champion)

    local_registry = {
        name: replace(entry, artifact_path=None) for name, entry in registry.REGISTRY.items()
    }
    local_root = str(tmp_path)

    def receipt(files):
        return {
            "verdict": "STRICT_PASS",
            "root": local_root,
            **{key: {"path": f"{key}.json", "sha256": "0" * 64} for key in files},
        }

    pins = {
        "schema": registry.PINS_SCHEMA,
        "checkpoints": {
            registry.NAME_FRP3_S12: {
                "checkpoint_sha256": registry.FRP3_S12_SHA256,
                "veto_variant": registry.V8_VARIANT,
                "veto_method": registry.V8_METHOD,
                "strict_receipt": receipt(registry.STRICT_FILES),
            },
            registry.NAME_B8_S3: {
                "checkpoint_sha256": registry.B8_S3_SHA256,
                "veto_variant": "none",
                "veto_method": "none",
                "strict_receipt": receipt(registry.T7_PIN_FILES),
            },
        },
    }
    pins_path = tmp_path / "pins.json"
    pins_path.write_text(json.dumps(pins))
    expected_hashes = {
        b8: registry.B8_S3_SHA256,
        frp3: registry.FRP3_S12_SHA256,
        champion: registry.CHAMPION_SHA256,
    }

    def fixture_digest(path):
        resolved = Path(path)
        assert resolved in expected_hashes, f"unexpected non-fixture checkpoint access: {path}"
        return expected_hashes[resolved]

    original_load = session_module._load_checkpoint_once

    def load_fixture_checkpoint(path):
        resolved = Path(path)
        assert resolved in expected_hashes, f"unexpected non-fixture checkpoint access: {path}"
        payload, blob, _ = original_load(path)
        return payload, blob, expected_hashes[resolved]

    monkeypatch.setattr(registry, "REGISTRY", local_registry)
    monkeypatch.setattr(registry, "SAVED_DIR", str(tmp_path))
    monkeypatch.setattr(registry, "PINS_PATH", str(pins_path))
    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(champion))
    monkeypatch.setattr(registry, "_sha256_file", fixture_digest)
    monkeypatch.setattr(session_module, "_load_checkpoint_once", load_fixture_checkpoint)


def _ids(session: GameSession) -> list[int]:
    return [int(snake.id) for snake in session.game.snakes]


def test_roster_fixture_survives_historical_artifact_denial_in_subprocess(tmp_path):
    """The local release identities must not mask reads from a developer's archive."""
    historical_roots = (
        "/Users/josenunez/Projects/ml/snake-dqn-artifacts",
        str(Path(registry.REPO_ROOT) / "saved_snakes"),
    )
    hook = tmp_path / "sitecustomize.py"
    hook.write_text(
        "import os, sys\n"
        f"_roots = {historical_roots!r}\n"
        "def _deny(event, args):\n"
        "    if event == 'open' and args and isinstance(args[0], str):\n"
        "        if any(os.path.abspath(args[0]).startswith(root) for root in _roots):\n"
        "            raise PermissionError(f'historical artifact access denied: {args[0]}')\n"
        "sys.addaudithook(_deny)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((str(tmp_path), str(Path.cwd()), env.get("PYTHONPATH", "")))
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            __file__,
            "-k",
            "not historical_artifact_denial_in_subprocess",
        ],
        cwd=Path(__file__).resolve().parents[1],
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_unset_watch_roster_serves_mixed_with_an_unwrapped_hero():
    session = GameSession()
    assert session.watch_roster == "mixed"
    assert session.watch_roster_reason is None
    assert session.game.snakes[0].policy is session.policy
    assert session.game.snakes[0].safety_veto is None
    assert session.watch_opponent_policy is not None
    assert all(snake.policy is session.watch_opponent_policy for snake in session.game.snakes[1:])
    state = session.safety_veto_state()
    assert state["active"] is True
    assert state["scope"] == "watch_opponents"
    assert state["wrapped_snake_ids"] == _ids(session)[1:]
    assert int(session.game.snakes[0].id) not in state["wrapped_snake_ids"]


def test_declared_watch_roster_default_is_mixed():
    from web.backend.served_checkpoint import WATCH_ROSTER_RELEASED_DEFAULT

    assert WATCH_ROSTER_RELEASED_DEFAULT == "mixed"


def test_mixed_watch_has_unwrapped_b8_hero_and_wrapped_frp3_opponents(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    session = GameSession()

    assert session.watch_roster == "mixed"
    assert session.watch_roster_reason is None
    assert session.game.snakes[0].policy is session.policy
    assert session.game.snakes[0].safety_veto is None
    assert session.watch_opponent_policy is not None
    assert all(snake.policy is session.watch_opponent_policy for snake in session.game.snakes[1:])
    state = session.safety_veto_state()
    assert state["active"] is True
    assert state["scope"] == "watch_opponents"
    assert state["wrapped_snake_ids"] == _ids(session)[1:]
    assert set(state["counters"]) == {str(value) for value in _ids(session)[1:]}
    assert "0" not in state["counters"]
    assert session.control_state()["safety_veto"] == state


@pytest.mark.parametrize("value", ("", "garbled"))
def test_unset_and_invalid_roster_words_use_released_mixed_default(monkeypatch, value):
    if value:
        monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", value)
    session = GameSession()
    assert session.watch_roster == "mixed"
    assert session.game.snakes[0].policy is session.policy
    assert session.game.snakes[0].safety_veto is None
    assert session.safety_veto_state()["scope"] == "watch_opponents"
    assert session.safety_veto_state()["wrapped_snake_ids"] == _ids(session)[1:]
    if value == "garbled":
        assert session.watch_roster_reason == (
            "unknown SNAKE_SERVE_WATCH_ROSTER 'garbled'; fell back to mixed"
        )


def test_all_b8_is_an_explicit_rollback(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "all_b8")
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert session.watch_roster_reason is None
    assert all(snake.policy is session.policy for snake in session.game.snakes)
    assert session.safety_veto_state()["active"] is False


def test_mixed_fails_closed_when_fpr3_resolution_fails(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    monkeypatch.setattr(
        session_module,
        "resolve_mixed_watch_opponent",
        lambda: ServedCheckpointChoice(
            requested="frp3-s12", reason="pin checkpoint_sha256 is not the registry sha256"
        ),
    )
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert "FRP3 opponent pin/bytes failed" in str(session.watch_roster_reason)
    assert all(snake.policy is session.policy for snake in session.game.snakes)
    assert session.safety_veto_state()["active"] is False


@pytest.mark.parametrize(
    "choice",
    (
        ServedCheckpointChoice(requested="frp3-s12", reason="missing strict pin"),
        ServedCheckpointChoice(
            requested="frp3-s12", name="champion", path="/tmp/wrong", sha256="0" * 64
        ),
    ),
)
def test_mixed_fails_closed_for_missing_or_wrong_opponent_pin(monkeypatch, choice):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    monkeypatch.setattr(session_module, "resolve_mixed_watch_opponent", lambda: choice)
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert all(snake.policy is session.policy for snake in session.game.snakes)
    assert session.safety_veto_state()["active"] is False


def test_mixed_fails_closed_when_opponent_bytes_change_after_resolution(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    original = session_module._load_checkpoint_once

    def changed_hash(path):
        payload, blob, digest = original(path)
        return payload, blob, "0" * 64 if "frp3_m3" in path else digest

    monkeypatch.setattr(session_module, "_load_checkpoint_once", changed_hash)
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert "bytes changed after pin resolution" in str(session.watch_roster_reason)
    assert all(snake.policy is session.policy for snake in session.game.snakes)


@pytest.mark.parametrize("failure", ("raise", "partial"))
def test_mixed_wrapper_failures_restore_policy_and_report_inactive_state(monkeypatch, failure):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    if failure == "raise":
        monkeypatch.setattr(
            session_module,
            "install_watch_opponent_vetoes",
            lambda *_: (_ for _ in ()).throw(RuntimeError("source failure")),
        )
    else:

        def partial(*_):
            state = ServingVetoState(
                flags=ServingVetoFlags(watch_hero=True, variant="v8"),
                mode="watch",
                obs_spec="vector61",
                checkpoint_sha256="x",
                scope="watch_opponents",
                variant="v8",
            )
            state.vetoes[1] = object()
            return state

        monkeypatch.setattr(session_module, "install_watch_opponent_vetoes", partial)
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert all(snake.policy is session.policy for snake in session.game.snakes)
    state = session.safety_veto_state()
    assert state["active"] is False and state["wrapped_snake_ids"] == []


def test_mixed_opponent_inspection_uses_the_selected_opponent_policy(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    session = GameSession()
    session.set_hero(1)
    inspector = session.snapshot()["inspector"]
    assert inspector is not None and inspector["input_size"] == 61


def test_mixed_is_watch_only_and_rebuilds_on_return(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    session = GameSession()
    session.set_mode(MODE_PLAY)
    assert session.mode == MODE_PLAY
    assert session.watch_roster == "all_b8"
    assert session.safety_veto_state()["scope"] != "watch_opponents"
    session.set_mode(MODE_WATCH)
    assert session.watch_roster == "mixed"
    assert session.safety_veto_state()["wrapped_snake_ids"] == _ids(session)[1:]


@pytest.mark.parametrize("checkpoint", ("frp3-s12", "champion"))
def test_mixed_request_preserves_explicit_checkpoint_behavior(monkeypatch, checkpoint):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    monkeypatch.setenv("SNAKE_SERVE_CHECKPOINT", checkpoint)
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert "requires the registry-selected" in str(session.watch_roster_reason)
    assert session.safety_veto_state()["scope"] == "watch_hero"


def test_mixed_opponents_are_exact_v8(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", "v8")
    session = GameSession()
    state = session.safety_veto_state()
    assert state["variant"] == "v8"
    assert state["scope"] == "watch_opponents"
    assert session.game.snakes[0].safety_veto is None


@pytest.mark.parametrize("variant", ("v2", "v5", "v7"))
def test_mixed_does_not_override_a_legacy_veto_rollback(monkeypatch, variant):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    monkeypatch.setenv("SNAKE_SERVE_VETO_VARIANT", variant)
    session = GameSession()
    assert session.watch_roster == "all_b8"
    assert session.safety_veto_state()["scope"] == "watch_hero"


def test_mixed_speed_ceiling_counts_the_total_roster(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    session = GameSession()
    assert len(session.game.snakes) == 6
    assert session.speed_ceiling == 24.0


@pytest.mark.parametrize("count, ceiling", ((8, 12.0), (12, 12.0)))
def test_mixed_speed_ceiling_at_eight_and_twelve(monkeypatch, count, ceiling):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    original = session_module.load_config

    def roster_config(path):
        config = original(path)
        return replace(config, game=replace(config.game, num_snakes=count))

    monkeypatch.setattr(session_module, "load_config", roster_config)
    session = GameSession()
    assert len(session.game.snakes) == count
    assert session.speed_ceiling == ceiling


def test_mixed_writes_a_roster_labeled_b8_release_watch_monitor(monkeypatch):
    monkeypatch.setenv("SNAKE_SERVE_WATCH_ROSTER", "mixed")
    tick = []
    monkeypatch.setattr(session_module, "watch_tick", lambda **kwargs: tick.append(kwargs))
    session = GameSession()
    session.step()
    assert len(tick) == 1
    assert tick[0]["checkpoint"] == "b8-s3"
    assert tick[0]["roster_mode"] == "mixed"
    assert tick[0]["roster_size"] == len(session.game.snakes)
