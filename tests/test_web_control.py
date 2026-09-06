"""Tests for the WebSocket control path — the only write surface into the session.

Covers the dispatch table (``_apply_control``), its rejection of malformed
client input, the ``load_checkpoint`` path containment, and that a bad control
message cannot tear down the live stream.
"""

import os
import pathlib
import re
from unittest.mock import MagicMock

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from web.backend import metrics  # noqa: E402
from web.backend.session import GameSession  # noqa: E402

REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
TYPES_TS = REPO_ROOT / "web" / "frontend" / "src" / "types.ts"


@pytest.fixture()
def app_mod():
    """Import web.backend.app lazily.

    Importing it pins torch to one intra-op thread process-wide, which
    tests/test_web_raster_serving.py asserts against; importing at collection
    time would let a later DeviceManager init un-pin it before that assertion.
    """
    import web.backend.app as mod

    return mod


# A representative value per action, so every handler gets something it accepts.
VALID_VALUES = {
    "play": None,
    "pause": None,
    "reset": None,
    "new_game": None,
    "set_speed": 10,
    "set_mode": "watch",
    "set_epsilon": 0.1,
    "set_hero": 0,
    "set_food": 5,
    "load_checkpoint": "x.pth",
    "human_input": "up",
    "human_boost": True,
    "set_play_opponents": 3,
    "save_weights": None,
    "set_raster_stream": {"on": True},
}

# Values a real client could send that the handlers cannot coerce.
MALFORMED_VALUES = [
    ("set_speed", "fast"),
    ("set_speed", None),
    ("set_speed", {}),
    ("set_food", None),
    ("set_food", "abc"),
    ("set_hero", None),
    ("set_epsilon", "x"),
    ("set_play_opponents", "many"),
    ("load_checkpoint", None),
]


@pytest.fixture(autouse=True)
def _restore_global_config():
    """GameSession initializes the process-global config; don't leak it."""
    from src.core import game_config

    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture()
def session():
    return GameSession()


class TestControlDispatchTable:
    """Regression guards for the browser -> GameSession dispatch table."""

    def test_every_handler_targets_a_real_session_method(self, app_mod):
        assert set(VALID_VALUES) == set(app_mod._CONTROL_HANDLERS)
        for action, value in sorted(VALID_VALUES.items()):
            # spec= is load-bearing: a bare MagicMock records a call for a
            # method that does not exist, so a rename would pass vacuously.
            mock = MagicMock(spec=GameSession)
            app_mod._apply_control(mock, {"type": "control", "action": action, "value": value})
            assert len(mock.mock_calls) == 1, f"{action} did not reach one session method"

    def test_frontend_and_backend_action_names_match(self, app_mod):
        if not TYPES_TS.exists():
            pytest.skip("frontend sources not present")
        # Strip // comments first — union members may carry doc comments.
        src = re.sub(r"//[^\n]*", "", TYPES_TS.read_text())
        match = re.search(r'export type ControlAction\s*=\s*((?:\s*\|\s*"[a-z_]+")+)\s*;', src)
        assert match, "could not parse the ControlAction union out of types.ts"
        frontend = set(re.findall(r'"([a-z_]+)"', match.group(1)))
        assert frontend == set(app_mod._CONTROL_HANDLERS)

    def test_unknown_action_is_reported_not_dispatched(self, app_mod):
        mock = MagicMock(spec=GameSession)
        app_mod._apply_control(mock, {"type": "control", "action": "set_speeed", "value": 10})
        assert mock.mock_calls == []
        assert "set_speeed" in mock.last_error

    def test_non_string_action_is_rejected(self, app_mod):
        mock = MagicMock(spec=GameSession)
        app_mod._apply_control(mock, {"type": "control", "action": ["set_speed"], "value": 10})
        assert mock.mock_calls == []


class TestMalformedControlIsRejected:
    """A bad value is recorded, never raised — the caller's stream must survive."""

    @pytest.mark.parametrize("action,value", MALFORMED_VALUES)
    def test_bad_value_sets_last_error_without_raising(self, app_mod, session, action, value):
        session.last_error = None
        app_mod._apply_control(session, {"type": "control", "action": action, "value": value})
        assert session.last_error is not None

    def test_missing_value_key_does_not_raise(self, app_mod, session):
        app_mod._apply_control(session, {"type": "control", "action": "set_speed"})
        assert session.last_error is not None

    @pytest.mark.parametrize("msg", [["control"], "control", None, 42])
    def test_non_dict_message_does_not_raise(self, app_mod, session, msg):
        app_mod._apply_control(session, msg)

    def test_bad_value_leaves_the_setting_untouched(self, app_mod, session):
        session.set_speed(30)
        app_mod._apply_control(session, {"type": "control", "action": "set_speed", "value": "fast"})
        assert session.speed == 30.0

    def test_valid_control_still_applies(self, app_mod, session):
        app_mod._apply_control(session, {"type": "control", "action": "set_speed", "value": 25})
        assert session.speed == 25.0


class TestLoadCheckpointContainment:
    """load_checkpoint feeds torch.load (pickle); it must stay inside SAVED_DIR."""

    @pytest.fixture()
    def torch_load_spy(self, monkeypatch):
        from web.backend import session as session_mod

        calls = []
        real = session_mod.torch.load

        def spy(path, *args, **kwargs):
            calls.append(str(path))
            return real(path, *args, **kwargs)

        monkeypatch.setattr(session_mod.torch, "load", spy)
        return calls

    @pytest.mark.parametrize(
        "name",
        [
            "../../../../etc/hosts",
            "../../../../etc/passwd.pth",
            "/etc/passwd",
            "/tmp/evil.pth",
            "subdir/../../evil.pth",
            "",
            None,
        ],
    )
    def test_escaping_names_are_rejected_before_any_load(self, session, torch_load_spy, name):
        before = session.checkpoint_path
        torch_load_spy.clear()
        session.load_checkpoint(name)
        assert torch_load_spy == [], f"unpickled a rejected path: {torch_load_spy}"
        assert session.checkpoint_path == before
        assert session.last_error is not None

    def test_rejection_does_not_echo_the_requested_path(self, session):
        session.load_checkpoint("/etc/passwd")
        assert "/etc" not in session.last_error

    def test_symlink_out_of_saved_dir_is_rejected(self, session, tmp_path, monkeypatch):
        outside = tmp_path / "outside.pth"
        outside.write_bytes(b"not a checkpoint")
        fake_root = tmp_path / "saved"
        fake_root.mkdir()
        (fake_root / "link.pth").symlink_to(outside)
        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(fake_root))
        before = session.checkpoint_path
        session.load_checkpoint("link.pth")
        assert session.checkpoint_path == before
        assert session.last_error is not None

    def test_non_pth_file_inside_saved_dir_is_rejected(self, session, tmp_path, monkeypatch):
        fake_root = tmp_path / "saved"
        fake_root.mkdir()
        (fake_root / "notes.txt").write_text("hi")
        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(fake_root))
        session.load_checkpoint("notes.txt")
        assert session.last_error is not None

    def test_missing_checkpoint_is_a_clean_error(self, session):
        before = session.checkpoint_path
        session.load_checkpoint("does_not_exist.pth")
        assert session.checkpoint_path == before
        assert "does_not_exist.pth" in session.last_error

    def test_real_basename_still_loads(self, session):
        names = [c["name"] for c in metrics.list_checkpoints()]
        if not names:
            pytest.skip("no checkpoints in saved_snakes/")
        name = os.path.basename(session.checkpoint_path or names[0])
        session.load_checkpoint(name)
        assert session.last_error is None
        assert session.control_state()["checkpoint"] == name


@pytest.fixture()
def ws_client(app_mod):
    """A client for /ws/stream with a session but no lifespan.

    The lifespan starts engine_loop, which broadcasts frames continuously and
    would interleave them with the acks these tests read.
    """
    hub = app_mod.hub
    prev = hub.session
    hub.session = GameSession()
    try:
        yield TestClient(app_mod.app), hub
    finally:
        hub.session = prev
        hub.clients.clear()


class TestWebSocketSurvivesBadControls:
    def test_malformed_control_acks_an_error_and_keeps_the_stream(self, ws_client):
        client, hub = ws_client
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()  # initial snapshot
            ws.send_json({"type": "control", "action": "set_speed", "value": "fast"})
            ack = ws.receive_json()
            assert ack["type"] == "ack"
            assert ack["state"]["error"] is not None
            assert len(hub.clients) == 1

            # The same socket still works afterwards.
            ws.send_json({"type": "control", "action": "set_speed", "value": 30})
            ack = ws.receive_json()
            assert ack["state"]["speed"] == 30.0
            assert len(hub.clients) == 1

    def test_non_object_json_does_not_drop_the_client(self, ws_client):
        client, hub = ws_client
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_json(["not", "an", "object"])
            ws.send_json({"type": "control", "action": "pause", "value": None})
            ack = ws.receive_json()
            assert ack["type"] == "ack"
            assert ack["state"]["playing"] is False
            assert len(hub.clients) == 1

    def test_undecodable_text_does_not_drop_the_client(self, ws_client):
        client, hub = ws_client
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            ws.send_text("}{ not json")
            ws.send_json({"type": "control", "action": "pause", "value": None})
            ack = ws.receive_json()
            assert ack["type"] == "ack"
            assert len(hub.clients) == 1

    def test_traversal_over_the_wire_is_rejected(self, ws_client):
        client, hub = ws_client
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            before = hub.session.checkpoint_path
            ws.send_json(
                {"type": "control", "action": "load_checkpoint", "value": "../../../etc/hosts"}
            )
            ack = ws.receive_json()
            assert ack["state"]["error"] is not None
            assert hub.session.checkpoint_path == before

    def test_disconnect_removes_the_client(self, ws_client):
        client, hub = ws_client
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert len(hub.clients) == 1
        assert len(hub.clients) == 0

    def test_viewer_count_tracks_connections(self, ws_client):
        client, hub = ws_client
        assert hub.session.viewer_count == 0
        with client.websocket_connect("/ws/stream") as ws:
            ws.receive_json()
            assert hub.session.viewer_count == 1
        assert hub.session.viewer_count == 0


class TestSaveWeights:
    """The save_weights control — the safe alternative to destructive rebuilds."""

    def test_save_weights_writes_a_checkpoint_and_replies_info(
        self, app_mod, session, tmp_path, monkeypatch
    ):
        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(tmp_path))
        reply = app_mod._apply_control(
            session, {"type": "control", "action": "save_weights"}, conn_id=1
        )
        assert reply is not None and reply["type"] == "info"
        name = reply["message"].removeprefix("Saved ")
        assert name.startswith("web_train_") and name.endswith(".pth")
        path = tmp_path / name
        assert path.is_file() and path.stat().st_size > 0

    def test_saved_weights_reload_into_a_fresh_policy(self, session, tmp_path, monkeypatch):
        import torch as _torch

        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(tmp_path))
        name = session.save_weights()
        blob = _torch.load(str(tmp_path / name), map_location="cpu", weights_only=False)
        assert "dqn_state_dict" in blob

    def test_two_saves_in_one_second_get_distinct_files(self, session, tmp_path, monkeypatch):
        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(tmp_path))
        monkeypatch.setattr("web.backend.session.time.strftime", lambda _fmt: "web_train_fixed")
        first = session.save_weights()
        second = session.save_weights()
        assert first != second
        assert (tmp_path / first).is_file()
        assert (tmp_path / second).is_file()

    @pytest.mark.parametrize("error_type", [RuntimeError, FileExistsError, KeyboardInterrupt])
    def test_failed_save_removes_only_its_partial_file(
        self, session, tmp_path, monkeypatch, error_type
    ):
        import web.backend.session as session_mod

        monkeypatch.setattr("web.backend.session.SAVED_DIR", str(tmp_path))
        monkeypatch.setattr("web.backend.session.time.strftime", lambda _fmt: "web_train_fixed")
        first = session.save_weights()

        def fail_save(*args, **kwargs):
            raise error_type("serialization failure")

        monkeypatch.setattr(session_mod.torch, "save", fail_save)
        with pytest.raises(error_type, match="serialization failure"):
            session.save_weights()

        assert (tmp_path / first).is_file()
        assert sorted(path.name for path in tmp_path.glob("*.pth")) == [first]

    def test_forward_only_policy_reports_a_clean_error(self, app_mod, session):
        session.policy = object()  # no get_state_dict, like RasterServingPolicy
        session.last_error = None
        reply = app_mod._apply_control(
            session, {"type": "control", "action": "save_weights"}, conn_id=1
        )
        assert reply is None
        assert "forward-only" in session.last_error


class TestSnapshotLocking:
    def test_state_snapshots_take_the_session_lock(self, session):
        class LockProbe:
            enters = 0

            def __enter__(self):
                self.enters += 1

            def __exit__(self, exc_type, exc, traceback):
                return False

        lock = LockProbe()
        session._lock = lock
        session.control_state()
        session.play_state()
        assert lock.enters == 2


class TestSetModeSemantics:
    def test_set_mode_same_mode_is_a_noop(self, session):
        game_before = session.game
        session.set_mode(session.mode)
        assert session.game is game_before  # no rebuild, no arena reset

    def test_train_on_raster_checkpoint_errors_without_rebuilding(self, session):
        from src.model.obs_spec import RASTER31V2

        session.obs_spec = RASTER31V2
        game_before = session.game
        session.set_mode("train")
        assert session.mode == "watch"
        assert session.game is game_before  # the arena survives the click
        assert session.last_error is not None
        assert "forward-only" in session.last_error


class TestRasterStreamSubscription:
    def test_subscribers_counted_per_connection(self, session):
        session.set_raster_stream(1, {"on": True})
        session.set_raster_stream(1, {"on": True})  # idempotent per connection
        assert session.raster_subscribers == 1
        session.set_raster_stream(2, True)  # bare bool accepted too
        assert session.raster_subscribers == 2
        session.set_raster_stream(1, {"on": False})
        assert session.raster_subscribers == 1

    def test_drop_connection_releases_subscription(self, session):
        session.set_raster_stream(7, {"on": True})
        session.drop_connection(7)
        assert session.raster_subscribers == 0


@pytest.fixture()
def play_session(session):
    session.set_mode("play")
    assert session.mode == "play"
    return session


class TestPlayRunOwnership:
    """A live scored run belongs to the connection that started it."""

    def _start_run_as(self, app_mod, session, conn_id):
        reply = app_mod._apply_control(
            session, {"type": "control", "action": "human_input", "value": "up"}, conn_id
        )
        assert reply is None
        assert session.run_started is True
        assert session.run_owner == conn_id

    def test_first_input_claims_ownership(self, app_mod, play_session):
        self._start_run_as(app_mod, play_session, conn_id=111)

    def test_other_connections_destructive_controls_are_rejected(self, app_mod, play_session):
        self._start_run_as(app_mod, play_session, conn_id=111)
        game_before = play_session.game
        for action in ("set_mode", "reset", "new_game", "load_checkpoint"):
            reply = app_mod._apply_control(
                play_session,
                {"type": "control", "action": action, "value": "watch"},
                conn_id=222,
            )
            assert reply == {"type": "error", "message": "Another player's run is in progress"}
        assert play_session.mode == "play"
        assert play_session.game is game_before

    def test_other_connections_steering_is_ignored(self, app_mod, play_session):
        self._start_run_as(app_mod, play_session, conn_id=111)
        human = play_session._find_human()
        direction_before = human.direction
        reply = app_mod._apply_control(
            play_session, {"type": "control", "action": "human_input", "value": "left"}, 222
        )
        assert reply is None
        assert human.direction == direction_before

    def test_owner_keeps_control(self, app_mod, play_session):
        self._start_run_as(app_mod, play_session, conn_id=111)
        reply = app_mod._apply_control(
            play_session, {"type": "control", "action": "new_game", "value": None}, 111
        )
        assert reply is None
        assert play_session.run_started is False  # the reset took effect
        assert play_session.run_owner == 111  # new_game re-claims for the presser

    def test_owner_disconnect_releases_the_run(self, app_mod, play_session):
        self._start_run_as(app_mod, play_session, conn_id=111)
        play_session.drop_connection(111)
        assert play_session.run_owner is None
        # Another connection may now take over.
        reply = app_mod._apply_control(
            play_session, {"type": "control", "action": "reset", "value": None}, 222
        )
        assert reply is None


class TestPlayRunFreeze:
    def test_rejected_first_input_does_not_unfreeze_the_world(self, play_session):
        # The human spawns facing right; "left" is a 180° reversal, rejected.
        play_session.human_input("left")
        assert play_session.run_started is False

        play_session.human_input("up")  # an accepted input starts the run
        assert play_session.run_started is True


class TestSubmissionClaimRelease:
    def _finished_run(self, session):
        session.run_over = True
        session.submitted = False
        session.last_run = {
            "score": 123,
            "length": 5,
            "food_eaten": 2,
            "kills": 0,
            "frames": 100,
            "duration_seconds": 8.3,
            "checkpoint": None,
            "mechanics_version": 1,
        }
        return session

    def test_release_after_failed_write_allows_retry(self, session):
        self._finished_run(session)
        assert session.claim_submission() is not None
        assert session.claim_submission() is None  # double-claim blocked
        session.release_submission()
        assert session.submitted is False
        assert session.claim_submission() is not None  # retry works


@pytest.fixture()
def api_client(app_mod):
    """A TestClient with a fabricated finished run and an in-memory store."""
    from src.data.score_store import ScoreStore

    hub = app_mod.hub
    prev_session, prev_store = hub.session, hub.store
    hub.session = GameSession()
    hub.session.run_over = True
    hub.session.submitted = False
    hub.session.last_run = {
        "score": 123,
        "length": 5,
        "food_eaten": 2,
        "kills": 0,
        "frames": 100,
        "duration_seconds": 8.3,
        "checkpoint": None,
        "mechanics_version": 1,
    }
    hub.store = ScoreStore(":memory:")
    try:
        yield TestClient(app_mod.app), hub
    finally:
        hub.store.close()
        hub.session, hub.store = prev_session, prev_store


class TestScoreSubmissionOrdering:
    """The run is only marked submitted AFTER the DB write succeeds."""

    def test_store_failure_returns_500_and_releases_the_claim(self, api_client, monkeypatch):
        client, hub = api_client

        def boom(*args, **kwargs):
            raise RuntimeError("disk full")

        monkeypatch.setattr(hub.store, "record_game", boom)
        resp = client.post("/api/scores", json={"name": "ada"})
        assert resp.status_code == 500
        assert "Could not save the score" in resp.json()["detail"]
        assert hub.session.submitted is False  # claim released -> retry possible

        monkeypatch.undo()
        resp = client.post("/api/scores", json={"name": "ada"})
        assert resp.status_code == 200
        assert resp.json()["ok"] is True
        assert hub.session.submitted is True

    def test_double_submit_is_a_409(self, api_client):
        client, hub = api_client
        assert client.post("/api/scores", json={"name": "ada"}).status_code == 200
        resp = client.post("/api/scores", json={"name": "ada"})
        assert resp.status_code == 409
        assert "No finished run" in resp.json()["detail"]

    def test_client_id_is_persisted_and_preferred_for_stats(self, api_client):
        client, hub = api_client
        resp = client.post("/api/scores", json={"name": "ada", "client_id": "cid-42"})
        assert resp.status_code == 200
        assert resp.json()["result"]["client_id"] == "cid-42"
        # Personal stats prefer the client_id match even under a new name.
        resp = client.get("/api/players/somebody-else", params={"client_id": "cid-42"})
        body = resp.json()
        assert body["found"] is True
        assert body["player"]["name"] == "ada"

    def test_name_is_normalized_server_side(self, api_client):
        client, hub = api_client
        resp = client.post("/api/scores", json={"name": "  ada   lovelace  "})
        assert resp.status_code == 200
        assert resp.json()["result"]["player_name"] == "ada lovelace"


class _StubWS:
    """Minimal websocket stand-in for broadcast tests."""

    def __init__(self, delay: float = 0.0, dead: bool = False):
        self.delay = delay
        self.dead = dead
        self.sent = []
        self.closed = False

    async def send_text(self, text: str) -> None:
        import asyncio as _asyncio

        if self.dead:
            raise RuntimeError("socket is dead")
        if self.delay:
            await _asyncio.sleep(self.delay)
        self.sent.append(text)

    async def close(self) -> None:
        self.closed = True


class TestConcurrentBroadcast:
    """One slow client must not throttle delivery to everyone else."""

    def test_slow_client_is_dropped_after_two_timeouts(self, app_mod, monkeypatch):
        import asyncio as _asyncio

        monkeypatch.setattr(app_mod, "BROADCAST_SEND_TIMEOUT", 0.05)
        hub = app_mod.Hub()
        fast, slow = _StubWS(), _StubWS(delay=0.5)
        hub.clients = {fast, slow}

        _asyncio.run(hub.broadcast({"type": "frame", "n": 1}))
        assert len(fast.sent) == 1
        assert slow in hub.clients  # one strike: kept

        _asyncio.run(hub.broadcast({"type": "frame", "n": 2}))
        assert len(fast.sent) == 2
        assert slow not in hub.clients  # two strikes: dropped
        assert slow.closed is True

    def test_dead_client_is_removed_and_others_still_receive(self, app_mod):
        import asyncio as _asyncio

        hub = app_mod.Hub()
        ok, dead = _StubWS(), _StubWS(dead=True)
        hub.clients = {ok, dead}

        _asyncio.run(hub.broadcast({"type": "frame"}))
        assert len(ok.sent) == 1
        assert dead not in hub.clients

    def test_good_send_resets_the_timeout_strikes(self, app_mod, monkeypatch):
        import asyncio as _asyncio

        monkeypatch.setattr(app_mod, "BROADCAST_SEND_TIMEOUT", 0.05)
        hub = app_mod.Hub()
        flaky = _StubWS(delay=0.5)
        hub.clients = {flaky}

        _asyncio.run(hub.broadcast({"n": 1}))  # strike 1
        flaky.delay = 0.0
        _asyncio.run(hub.broadcast({"n": 2}))  # good send clears strikes
        flaky.delay = 0.5
        _asyncio.run(hub.broadcast({"n": 3}))  # strike 1 again, not 2
        assert flaky in hub.clients


class TestRewardContractOverride:
    """Deliberate v1->v2 reward fine-tunes go through an explicit override.

    On this branch the served arena runs reward v2 (mechanics_v2.yaml) while the
    champion checkpoint records reward v1, so entering train mode must refuse
    with an OVERRIDABLE error — and re-issuing with the override must enter
    train as a warm-start fine-tune (validator escape hatch), never silently.
    """

    def _requires_mismatch(self, session):
        from web.backend.session import _reward_contract_mismatch

        if not session.checkpoint_path or not _reward_contract_mismatch(session.checkpoint_path):
            pytest.skip("active config's reward contract matches the checkpoint")

    def test_train_refuses_with_overridable_error_and_keeps_arena(self, session):
        self._requires_mismatch(session)
        game_before = session.game
        session.set_mode("train")
        assert session.mode == "watch"
        assert session.game is game_before  # no doomed build, no arena reset
        assert session.last_error is not None
        assert session.last_error_overridable is True
        state = session.control_state()
        assert state["error_overridable"] is True
        assert state["reward_override_active"] is False

    def test_override_enters_train_as_flagged_finetune(self, session):
        self._requires_mismatch(session)
        session.set_mode("train", override_reward_contract=True)
        assert session.mode == "train"
        assert session.last_error is None
        assert session.last_error_overridable is False
        assert session.reward_override_active is True
        assert session.control_state()["reward_override_active"] is True
        # Leaving train drops the fine-tune flag with the training policy.
        session.set_mode("watch")
        assert session.reward_override_active is False

    def test_load_in_train_mode_enforces_and_overrides(self, session):
        self._requires_mismatch(session)
        name = os.path.basename(session.checkpoint_path)
        session.set_mode("train", override_reward_contract=True)
        assert session.mode == "train"
        game_before = session.game
        # A plain (non-override) load of a mismatched checkpoint while training
        # must refuse overridably without touching the arena...
        session.load_checkpoint(name)
        assert session.game is game_before
        assert session.last_error_overridable is True
        # ...and the override form performs the flagged fine-tune load.
        session.load_checkpoint(name, override_reward_contract=True)
        assert session.mode == "train"
        assert session.last_error is None
        assert session.reward_override_active is True

    def test_dict_control_values_reach_the_session(self, app_mod):
        mock = MagicMock(spec=GameSession)
        app_mod._apply_control(
            mock,
            {
                "type": "control",
                "action": "set_mode",
                "value": {"mode": "train", "override_reward_contract": True},
            },
        )
        mock.set_mode.assert_called_once_with("train", True)
        mock2 = MagicMock(spec=GameSession)
        app_mod._apply_control(
            mock2,
            {
                "type": "control",
                "action": "load_checkpoint",
                "value": {"name": "champ.pth", "override_reward_contract": True},
            },
        )
        mock2.load_checkpoint.assert_called_once_with("champ.pth", True)
