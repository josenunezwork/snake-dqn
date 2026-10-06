"""SNAKE_SERVE_CHECKPOINT: the pinned served-checkpoint registry (web/backend/served_checkpoint.py).

Covers: the released default is frp3-s12 since 2026-10-06 (fail closed to the champion; with
no checkpoint named, a pre-swap veto variant v7/v5/v2 serves the champion so those rollbacks
keep their meaning); before the flip the default was the champion (path and behavior
unchanged, including the untrained fallback; tested under ``pre_release``); the frp3-s12 entry is refused while its strict-receipt pin is the
shipped ``null`` placeholder, when the pin is malformed or binds other bytes, and when the
file is missing or its bytes differ (fail closed to the released default, with a reason,
never raised); with a filled pin it is served and the released v8 veto binds it through the
pin (only v8; v7/v5/v2 rollbacks leave it unwrapped); the log line; and the real FRP-v3
checkpoint loading in the live web stack.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
from pathlib import Path

import pytest
import torch

pytest.importorskip("fastapi")

from web.backend import safety_veto_serving as serving  # noqa: E402
from web.backend import served_checkpoint as registry  # noqa: E402
from web.backend.safety_veto_serving import (  # noqa: E402
    ENV_PLAY_AI,
    ENV_VARIANT,
    ENV_WATCH_HERO,
    ServingVetoFlags,
)
from web.backend.served_checkpoint import ENV_CHECKPOINT, NAME_FRP3_S12  # noqa: E402
from web.backend.session import GameSession  # noqa: E402

MAIN_SAVED = Path("/Users/josenunez/Projects/ml/snake-dqn/saved_snakes")
CHAMPION = MAIN_SAVED / registry.CHAMPION_FILENAME
FRP3 = Path(registry.FRP3_S12_ARTIFACT_PATH)
needs_champion = pytest.mark.skipif(not CHAMPION.is_file(), reason="champion checkpoint absent")
needs_frp3 = pytest.mark.skipif(not FRP3.is_file(), reason="FRP-v3 s12 checkpoint absent")
LOGGER = "web.backend.served_checkpoint"
V8_METHOD = "free-space-veto/v8-space-and-head(lambda=8.0)"


@pytest.fixture(autouse=True)
def _isolate(monkeypatch):
    from src.core import game_config

    for key in (ENV_WATCH_HERO, ENV_PLAY_AI, ENV_VARIANT, ENV_CHECKPOINT):
        monkeypatch.delenv(key, raising=False)
    prev = game_config._current_config
    yield
    game_config._current_config = prev


def filled(doc: dict, name: str = NAME_FRP3_S12, sha: str = registry.FRP3_S12_SHA256) -> dict:
    out = copy.deepcopy(doc)
    entry = out["checkpoints"].setdefault(name, {})
    entry.update(
        checkpoint_sha256=sha,
        veto_variant="v8",
        veto_method=V8_METHOD,
        strict_receipt={
            "verdict": "STRICT_PASS",
            "root": "/abs/strict/run-v1",
            "receipt": {"path": "output/receipt.json", "sha256": "a" * 64},
            "intent": {"path": "intent.json", "sha256": "b" * 64},
            "audit_report": {"path": "output/audit/report.json", "sha256": "c" * 64},
        },
    )
    return out


def unfilled(doc: dict) -> dict:
    out = copy.deepcopy(doc)
    out["checkpoints"][NAME_FRP3_S12]["strict_receipt"] = None
    return out


@pytest.fixture
def shipped():
    """The repo pins with the frp3-s12 receipt reset to the shipped placeholder, so these
    tests keep passing after the owner fills and commits the real pin."""
    return unfilled(registry.load_pins())


@pytest.fixture
def use_pins(monkeypatch, tmp_path):
    """Point the registry at a pins document written to a temp file."""

    def apply(doc: dict) -> None:
        path = tmp_path / "pins.json"
        path.write_text(json.dumps(doc))
        monkeypatch.setattr(registry, "PINS_PATH", str(path))

    return apply


@pytest.fixture
def fake_frp3(monkeypatch, tmp_path):
    """Re-point the frp3-s12 registry entry at a small file with a known sha (test only)."""
    path = tmp_path / "artifact" / "apex_mark_u60000.pth"
    path.parent.mkdir()
    path.write_bytes(b"frp3 stand-in bytes")
    sha = hashlib.sha256(path.read_bytes()).hexdigest()
    entry = registry.PinnedCheckpoint(NAME_FRP3_S12, "frp3.pth", sha, artifact_path=str(path))
    monkeypatch.setitem(registry.REGISTRY, NAME_FRP3_S12, entry)
    return path, sha


@pytest.fixture
def pre_release(monkeypatch):
    """The registry as shipped before the 2026-10-06 release flip (champion default)."""
    monkeypatch.setattr(registry, "CHECKPOINT_RELEASED_DEFAULT", registry.NAME_CHAMPION)


def resolve(env: dict, saved: Path, champion: Path, pins: dict):
    return registry.resolve_served_checkpoint(
        environ=env, saved_dir=str(saved), pins=pins, champion_path=str(champion)
    )


class TestShippedState:
    def test_released_default_is_frp3_s12(self):
        assert registry.CHECKPOINT_RELEASED_DEFAULT == registry.NAME_FRP3_S12 == "frp3-s12"
        assert registry.PRE_SWAP_VETO_VARIANTS == ("v7", "v5", "v2")
        assert registry.ENV_VETO_VARIANT == ENV_VARIANT
        import web.backend.session as session_module

        # the champion stays the rollback target at the session's DEFAULT_CHECKPOINT

        assert Path(session_module.DEFAULT_CHECKPOINT).name == registry.CHAMPION_FILENAME
        assert registry.REGISTRY["champion"].sha256 == serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256

    def test_repo_pin_is_the_placeholder_or_a_valid_fill(self, shipped):
        repo = registry.load_pins()
        assert repo["schema"] == registry.PINS_SCHEMA
        entry = repo["checkpoints"][NAME_FRP3_S12]
        assert entry["checkpoint_sha256"] == registry.FRP3_S12_SHA256
        assert (entry["veto_variant"], entry["veto_method"]) == ("v8", V8_METHOD)
        if entry["strict_receipt"] is None:  # shipped state: refused until the owner fills it
            assert registry.v8_gated_checkpoint_sha256s() == []
        else:  # filled by fill_strict_pin.py after the strict gate
            assert registry.strict_pin_problems(NAME_FRP3_S12, repo) == []
            assert entry["strict_receipt"]["root"] != "/smoke-placeholder-not-a-strict-run"
        problems = registry.strict_pin_problems(NAME_FRP3_S12, shipped)
        assert problems == ["strict receipt pin is the unfilled placeholder (null)"]
        assert registry.v8_gated_checkpoint_sha256s(shipped) == []

    def test_frp3_entry_pins_the_artifact(self):
        entry = registry.REGISTRY[NAME_FRP3_S12]
        assert entry.sha256 == "eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723"
        assert entry.requires_strict_pin is True
        assert entry.artifact_path.endswith(
            "frp-v3-20261005/train/arm-M3/seed-12/checkpoints/" "apex_mark_u60000.pth"
        )

    def test_v8_pins_unchanged(self):
        assert serving.V8_STRICT_RECEIPT_SHA256 == (
            "29b1f7f6f095cac1990e8f7eafccc806e498cd503bb964bd083b3436ef2b7507"
        )
        assert serving.V8_STRICT_RECEIPT_CHECKPOINT_SHA256 == registry.CHAMPION_SHA256
        assert serving.V8_LAMBDA == 8.0 and serving.VARIANT_RELEASED_DEFAULT == "v8"


class TestReleasedDefault:
    """After the release flip: unset serves frp3-s12 (pin + bytes verified), fail closed."""

    def test_unset_and_blank_serve_frp3(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        for env in ({}, {ENV_CHECKPOINT: ""}, {ENV_VARIANT: "v8"}, {ENV_VARIANT: " "}):
            choice = resolve(env, tmp_path, champion, filled(shipped, sha=sha))
            assert (choice.name, choice.path, choice.reason) == (NAME_FRP3_S12, str(path), None)
            assert choice.requested is None

    def test_watch_hero_off_still_serves_frp3(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        choice = resolve({ENV_WATCH_HERO: "0"}, tmp_path, tmp_path / "c.pth", filled(shipped, sha=sha))
        assert choice.name == NAME_FRP3_S12 and choice.reason is None

    def test_unknown_variant_keeps_frp3(self, tmp_path, shipped, fake_frp3):
        # the veto module falls back to v8 for an unknown variant, so frp3-s12 stays gated
        _, sha = fake_frp3
        choice = resolve({ENV_VARIANT: "v9"}, tmp_path, tmp_path / "c.pth", filled(shipped, sha=sha))
        assert choice.name == NAME_FRP3_S12

    @pytest.mark.parametrize("word", ["v7", "v5", "v2", " V7 "])
    def test_pre_swap_variant_serves_the_champion(self, tmp_path, shipped, fake_frp3, word):
        _, sha = fake_frp3
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({ENV_VARIANT: word}, tmp_path, champion, filled(shipped, sha=sha))
        assert (choice.name, choice.path) == ("champion", str(champion))
        assert "pre-swap veto rollback" in choice.reason
        assert choice.requested is None

    def test_named_frp3_under_v7_is_still_frp3(self, tmp_path, shipped, fake_frp3):
        # an explicit checkpoint wins: frp3-s12 under v7 is served (unwrapped by the veto hook)
        path, sha = fake_frp3
        env = {ENV_CHECKPOINT: NAME_FRP3_S12, ENV_VARIANT: "v7"}
        choice = resolve(env, tmp_path, tmp_path / "c.pth", filled(shipped, sha=sha))
        assert (choice.name, choice.path, choice.reason) == (NAME_FRP3_S12, str(path), None)

    def test_champion_rollback(self, tmp_path, shipped, fake_frp3):
        _, sha = fake_frp3
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        for env in ({ENV_CHECKPOINT: "champion"}, {ENV_CHECKPOINT: "champion", ENV_VARIANT: "v7"}):
            choice = resolve(env, tmp_path, champion, filled(shipped, sha=sha))
            assert (choice.name, choice.reason) == ("champion", None)

    def test_unfilled_pin_fails_closed_to_the_champion(self, tmp_path, shipped, fake_frp3):
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({}, tmp_path, champion, shipped)
        assert choice.name == "champion" and "serving the champion" in choice.reason

    def test_refused_and_no_champion_means_untrained(self, tmp_path, shipped, fake_frp3):
        choice = resolve({}, tmp_path, tmp_path / "absent.pth", shipped)
        assert choice.path is None and "serving the champion" in choice.reason


@pytest.mark.usefixtures("pre_release")
class TestResolution:
    def test_unset_and_blank_serve_the_champion_path(self, tmp_path, shipped):
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        for env in ({}, {ENV_CHECKPOINT: ""}, {ENV_CHECKPOINT: "  "}):
            choice = resolve(env, tmp_path, champion, shipped)
            assert (choice.name, choice.path, choice.reason) == ("champion", str(champion), None)
            assert choice.requested is None

    def test_missing_champion_means_untrained_as_before(self, tmp_path, shipped):
        choice = resolve({}, tmp_path, tmp_path / "absent.pth", shipped)
        assert choice.path is None and choice.name is None and choice.reason is None

    @pytest.mark.parametrize("word", ["champion", " CHAMPION "])
    def test_champion_by_name(self, tmp_path, shipped, word):
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({ENV_CHECKPOINT: word}, tmp_path, champion, shipped)
        assert (choice.name, choice.requested, choice.reason) == ("champion", "champion", None)

    def test_unknown_name_falls_back_with_reason(self, tmp_path, shipped):
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({ENV_CHECKPOINT: "frp3-s99"}, tmp_path, champion, shipped)
        assert choice.name == "champion" and choice.path == str(champion)
        assert "unknown SNAKE_SERVE_CHECKPOINT 'frp3-s99'" in choice.reason

    def test_unfilled_pin_refuses_frp3(self, tmp_path, shipped, fake_frp3):
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({ENV_CHECKPOINT: NAME_FRP3_S12}, tmp_path, champion, shipped)
        assert choice.name == "champion" and choice.path == str(champion)
        assert choice.requested == NAME_FRP3_S12
        assert "no valid strict-receipt pin" in choice.reason
        assert "strict receipt pin is the unfilled placeholder (null)" in choice.pin_problems

    def test_filled_pin_serves_frp3_from_the_artifact(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        choice = resolve(
            {ENV_CHECKPOINT: NAME_FRP3_S12}, tmp_path, tmp_path / "c", filled(shipped, sha=sha)
        )
        assert (choice.name, choice.path, choice.sha256, choice.reason) == (
            NAME_FRP3_S12,
            str(path),
            sha,
            None,
        )

    def test_saved_snakes_copy_is_preferred(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        saved = tmp_path / "saved"
        saved.mkdir()
        (saved / "frp3.pth").write_bytes(path.read_bytes())
        choice = resolve(
            {ENV_CHECKPOINT: NAME_FRP3_S12}, saved, tmp_path / "c", filled(shipped, sha=sha)
        )
        assert choice.path == str(saved / "frp3.pth") and choice.sha256 == sha

    def test_changed_bytes_are_refused(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        saved = tmp_path / "saved"
        saved.mkdir()
        (saved / "frp3.pth").write_bytes(b"tampered")
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        choice = resolve({ENV_CHECKPOINT: NAME_FRP3_S12}, saved, champion, filled(shipped, sha=sha))
        assert choice.name == "champion" and "not the pinned one" in choice.reason

    def test_missing_file_is_refused(self, tmp_path, shipped, fake_frp3):
        path, sha = fake_frp3
        path.unlink()
        choice = resolve(
            {ENV_CHECKPOINT: NAME_FRP3_S12}, tmp_path, tmp_path / "c", filled(shipped, sha=sha)
        )
        assert choice.path is None and "file not found" in choice.reason

    @pytest.mark.parametrize(
        "mutate,needle",
        [
            (lambda e: e.update(checkpoint_sha256="0" * 64), "registry sha256"),
            (lambda e: e.update(veto_variant="v7"), "veto is not v8"),
            (
                lambda e: e.update(veto_method="free-space-veto/v8-space-and-head(lambda=4.0)"),
                "veto",
            ),
            (lambda e: e["strict_receipt"].update(verdict="STRICT_FAIL"), "verdict"),
            (lambda e: e["strict_receipt"].update(root="relative/root"), "absolute"),
            (lambda e: e["strict_receipt"]["intent"].update(sha256="xyz"), "strict_receipt.intent"),
            (lambda e: e["strict_receipt"]["receipt"].update(path="/abs/receipt.json"), "receipt"),
            (lambda e: e["strict_receipt"].pop("audit_report"), "audit_report"),
            (lambda e: e.update(strict_receipt="filled"), "not a mapping"),
        ],
    )
    def test_malformed_pins_are_refused(self, shipped, mutate, needle):
        doc = filled(shipped)
        mutate(doc["checkpoints"][NAME_FRP3_S12])
        problems = registry.strict_pin_problems(NAME_FRP3_S12, doc)
        assert problems and any(needle in p for p in problems), problems
        assert registry.v8_gated_checkpoint_sha256s(doc) == []

    def test_bad_schema_or_missing_entry_is_refused(self, shipped):
        doc = filled(shipped)
        doc["schema"] = "other/v1"
        assert registry.strict_pin_problems(NAME_FRP3_S12, doc)
        assert registry.strict_pin_problems(
            NAME_FRP3_S12, {"schema": registry.PINS_SCHEMA, "checkpoints": {}}
        )

    def test_unreadable_pins_file_refuses(self, monkeypatch, tmp_path):
        path = tmp_path / "pins.json"
        path.write_text("{not json")
        monkeypatch.setattr(registry, "PINS_PATH", str(path))
        assert registry.load_pins() == {"schema": None, "checkpoints": {}}
        assert registry.strict_pin_problems(NAME_FRP3_S12)
        assert registry.v8_gated_checkpoint_sha256s() == []  # unreadable file: nothing gated

    def test_filled_pin_gates_v8(self, shipped):
        assert registry.v8_gated_checkpoint_sha256s(filled(shipped)) == [registry.FRP3_S12_SHA256]

    def test_released_default_flip_fails_closed_to_the_champion(
        self, monkeypatch, tmp_path, shipped, fake_frp3
    ):
        path, sha = fake_frp3
        monkeypatch.setattr(registry, "CHECKPOINT_RELEASED_DEFAULT", NAME_FRP3_S12)
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        ok = resolve({}, tmp_path, champion, filled(shipped, sha=sha))
        assert ok.name == NAME_FRP3_S12 and ok.reason is None
        refused = resolve({}, tmp_path, champion, shipped)
        assert refused.name == "champion" and "serving the champion" in refused.reason
        rollback = resolve(
            {ENV_CHECKPOINT: "champion"}, tmp_path, champion, filled(shipped, sha=sha)
        )
        assert rollback.name == "champion" and rollback.reason is None


class TestLogging:
    def test_quiet_for_the_plain_default(self, caplog, tmp_path, shipped, pre_release):
        caplog.set_level(logging.INFO, logger=LOGGER)
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        resolve({}, tmp_path, champion, shipped)
        assert not [r for r in caplog.records if r.name == LOGGER]

    def test_logs_requests_and_refusals(self, caplog, tmp_path, shipped, fake_frp3):
        caplog.set_level(logging.INFO, logger=LOGGER)
        champion = tmp_path / "champ.pth"
        champion.write_bytes(b"x")
        resolve({ENV_CHECKPOINT: NAME_FRP3_S12}, tmp_path, champion, shipped)
        (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        assert line.startswith("served-checkpoint: name=champion requested=frp3-s12")
        assert "no valid strict-receipt pin" in line

    def test_logs_a_served_swap_even_unrequested(
        self, caplog, monkeypatch, tmp_path, shipped, fake_frp3
    ):
        caplog.set_level(logging.INFO, logger=LOGGER)
        path, sha = fake_frp3
        monkeypatch.setattr(registry, "CHECKPOINT_RELEASED_DEFAULT", NAME_FRP3_S12)
        resolve({}, tmp_path, tmp_path / "c", filled(shipped, sha=sha))
        (line,) = [r.getMessage() for r in caplog.records if r.name == LOGGER]
        assert f"name=frp3-s12 requested=None sha256={sha}" in line and line.endswith("reason=None")


class TestVetoBinding:
    """checkpoint_match: the swap joins the v8 binding only through a filled pin."""

    def state(self, variant: str, sha: str) -> serving.ServingVetoState:
        flags = ServingVetoFlags(watch_hero=True, play_ai=False, variant=variant)
        state = serving.ServingVetoState(flags, "watch", "vector61", sha)
        state.variant = variant
        return state

    def test_unfilled_pin_never_binds(self, use_pins, shipped):
        use_pins(shipped)
        assert self.state("v8", registry.FRP3_S12_SHA256).checkpoint_match is False
        assert self.state("v8", registry.CHAMPION_SHA256).checkpoint_match is True

    def test_filled_pin_binds_v8_only(self, use_pins, shipped):
        use_pins(filled(shipped))
        assert self.state("v8", registry.FRP3_S12_SHA256).checkpoint_match is True
        for variant in ("v7", "v5", "v2"):
            assert self.state(variant, registry.FRP3_S12_SHA256).checkpoint_match is False
            assert self.state(variant, registry.CHAMPION_SHA256).checkpoint_match is True
        assert self.state("v8", None).checkpoint_match is False
        assert self.state("v8", "f" * 64).checkpoint_match is False


@pytest.fixture
def champion_default(monkeypatch):
    import web.backend.session as session_module

    monkeypatch.setattr(session_module, "DEFAULT_CHECKPOINT", str(CHAMPION))


@needs_champion
@needs_frp3
class TestLiveStack:
    """The real FRP-v3 s12 checkpoint through GameSession() (tiny step counts)."""

    def test_unset_env_serves_frp3_with_v8(self, champion_default):
        sess = GameSession()
        state = sess.safety_veto_state()
        assert sess.served_checkpoint["name"] == NAME_FRP3_S12
        assert sess.served_checkpoint["reason"] is None
        assert state["checkpoint_sha256"] == registry.FRP3_S12_SHA256
        assert state["active"] and state["variant"] == "v8"
        assert state["strict_receipt_checkpoint_match"] is True

    def test_unset_checkpoint_v7_rollback_serves_champion_with_v7(
        self, monkeypatch, champion_default
    ):
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession()
        state = sess.safety_veto_state()
        assert sess.checkpoint_path == str(CHAMPION)
        assert sess.served_checkpoint["name"] == "champion"
        assert state["checkpoint_sha256"] == registry.CHAMPION_SHA256
        assert state["active"] and state["variant"] == "v7"

    def test_unset_env_serves_the_champion_with_v8_pre_release(
        self, champion_default, pre_release
    ):
        sess = GameSession()
        state = sess.safety_veto_state()
        assert sess.checkpoint_path == str(CHAMPION)
        assert state["checkpoint_sha256"] == registry.CHAMPION_SHA256
        assert sess.served_checkpoint["name"] == "champion"
        assert state["active"] and state["variant"] == "v8"

    def test_frp3_is_refused_while_the_pin_is_unfilled(
        self, monkeypatch, use_pins, shipped, champion_default
    ):
        use_pins(shipped)
        monkeypatch.setenv(ENV_CHECKPOINT, NAME_FRP3_S12)
        sess = GameSession()
        assert sess.checkpoint_path == str(CHAMPION)
        assert sess.served_checkpoint["requested"] == NAME_FRP3_S12
        assert "unfilled placeholder" in sess.served_checkpoint["reason"]

    def test_filled_pin_serves_frp3_behind_v8(
        self, monkeypatch, use_pins, shipped, champion_default
    ):
        use_pins(filled(shipped))
        monkeypatch.setenv(ENV_CHECKPOINT, NAME_FRP3_S12)
        sess = GameSession()
        state = sess.safety_veto_state()
        assert state["checkpoint_sha256"] == registry.FRP3_S12_SHA256
        assert sess.served_checkpoint["name"] == NAME_FRP3_S12
        assert state["active"] is True and state["variant"] == "v8"
        assert state["strict_receipt_checkpoint_match"] is True
        assert state["wrapper"]["method"] == V8_METHOD and state["reason"] is None
        hero = int(sess.game.snakes[0].id)
        assert state["wrapped_snake_ids"] == [hero]
        for _ in range(5):
            sess.step()
            frame = sess.snapshot()
        assert frame is not None
        # Mode switches rebuild from the resolved path: Play keeps frp3-s12, unwrapped.
        sess.set_mode("play")
        play = sess.safety_veto_state()
        assert play["checkpoint_sha256"] == registry.FRP3_S12_SHA256 and not play["active"]

    def test_frp3_under_v7_rollback_is_unwrapped(
        self, monkeypatch, use_pins, shipped, champion_default
    ):
        use_pins(filled(shipped))
        monkeypatch.setenv(ENV_CHECKPOINT, NAME_FRP3_S12)
        monkeypatch.setenv(ENV_VARIANT, "v7")
        sess = GameSession()
        state = sess.safety_veto_state()
        assert state["checkpoint_sha256"] == registry.FRP3_S12_SHA256
        assert state["active"] is False and "under variant v7" in state["reason"]

    def test_q_values_are_about_a_quarter_of_the_champions(self, champion_default):
        """The 0.25 head/reward rescale: Q (and V, A) about 4x smaller; argmax-level and
        range-normalized consumers (v7/v8 Qn) are unaffected by the scale."""
        from src.evaluation.safety_veto_v7 import normalized_q

        frp = torch.load(FRP3, map_location="cpu", weights_only=False)
        champ = torch.load(CHAMPION, map_location="cpu", weights_only=False)
        sess = GameSession(checkpoint=str(FRP3))
        net = sess.policy.dqn
        torch.manual_seed(0)
        states = torch.rand(256, 61)
        with torch.no_grad():
            net.load_state_dict(frp["dqn_state_dict"])
            q_frp = net(states)
            net.load_state_dict(champ["dqn_state_dict"])
            q_champ = net(states)
        ratio = float(q_frp.abs().mean() / q_champ.abs().mean())
        assert 0.1 < ratio < 0.6, ratio
        q = [float(v) for v in q_frp[0]]
        scaled = [4.0 * v for v in q]
        cands = [0, 1, 2]
        a, b = normalized_q(q, cands), normalized_q(scaled, cands)
        assert all(abs(a[k] - b[k]) < 1e-9 for k in cands)
