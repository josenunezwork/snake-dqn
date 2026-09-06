"""Tests for the web human-play mode: session mechanics + leaderboard endpoints."""

import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

from web.backend.session import MODE_PLAY, MODE_WATCH, GameSession  # noqa: E402


@pytest.fixture(autouse=True)
def _restore_global_config():
    """Snapshot/restore the global config singleton around every test here.

    GameSession initializes the process-global config (mechanics v2 by default
    since the ScoreStore v2 work), so without this, later test files that read
    the global config without initializing their own would see v2 mechanics.
    Same pattern as conftest's setup_config.
    """
    from src.core import game_config

    prev = game_config._current_config
    yield
    game_config._current_config = prev


@pytest.fixture()
def session():
    """A watch-mode session with whatever default checkpoint exists."""
    sess = GameSession()
    yield sess


class TestSessionPlayMode:
    def test_enter_play_mode_creates_human(self, session):
        session.set_mode(MODE_PLAY)
        assert session.mode == MODE_PLAY
        human = session._find_human()
        assert human is not None
        assert human.auto_respawn is False
        assert session.hero_id == human.id
        assert session.run_over is False

    def test_human_input_turns_snake(self, session):
        session.set_mode(MODE_PLAY)
        human = session._find_human()
        human.direction = (1, 0)  # right
        session.human_input("up")
        assert human.direction == (0, -1)

    def test_world_frozen_until_first_input(self, session):
        session.set_mode(MODE_PLAY)
        assert session.run_started is False
        head_before = session._find_human().head
        session._step_play()  # no input yet -> frozen
        assert session._find_human().head == head_before
        assert session.game.frame == 0
        # First input un-freezes the run.
        session.human_input("up")
        assert session.run_started is True
        session._step_play()
        assert session.game.frame == 1

    def test_human_death_finalizes_run(self, session):
        session.set_mode(MODE_PLAY)
        session.run_started = True  # player has started
        human = session._find_human()
        # Simulate a death, then step: the play loop should finalize the run.
        human.length = 7
        human.run_food_eaten = 4
        human.run_kills = 1
        session.run_frames = 120
        human.is_alive = False
        session._step_play()
        assert session.run_over is True
        pending = session.pending_submission()
        assert pending is not None
        assert pending["length"] == 7
        assert pending["food_eaten"] == 4
        assert pending["kills"] == 1
        assert pending["score"] > 0

    def test_claim_submission_is_once_only(self, session):
        session.set_mode(MODE_PLAY)
        session.run_started = True
        human = session._find_human()
        human.is_alive = False
        session._step_play()
        assert session.pending_submission() is not None
        first = session.claim_submission()
        second = session.claim_submission()
        assert first is not None  # claimed the run
        assert second is None  # already claimed -> no double-record
        assert session.pending_submission() is None

    def test_new_game_restarts_run(self, session):
        session.set_mode(MODE_PLAY)
        session.run_started = True
        human = session._find_human()
        human.is_alive = False
        session._step_play()
        assert session.run_over is True
        session.reset_game()
        assert session.run_over is False
        assert session.run_started is False
        assert session.pending_submission() is None
        assert session._find_human().is_alive is True

    def test_play_state_shape(self, session):
        session.set_mode(MODE_PLAY)
        ps = session.play_state()
        assert ps is not None
        for key in ("active", "human_alive", "length", "food_eaten", "kills", "score"):
            assert key in ps

    def test_watch_mode_has_no_play_state(self, session):
        assert session.mode == MODE_WATCH
        assert session.play_state() is None

    def test_leaving_play_mode_clears_human(self, session):
        session.set_mode(MODE_PLAY)
        assert session._find_human() is not None
        session.set_mode(MODE_WATCH)
        assert session._find_human() is None
        assert session.human_id is None

    def test_set_play_opponents_changes_ai_count(self, session):
        session.set_mode(MODE_PLAY)
        session.set_play_opponents(3)
        # 1 human + 3 AI opponents.
        assert len(session.game.snakes) == 4
        assert session._find_human() is not None
        assert session.play_state()["opponents"] == 3
        # Restarting a run keeps the chosen difficulty.
        session.reset_game()
        assert len(session.game.snakes) == 4

    def test_set_play_opponents_clamped(self, session):
        session.set_mode(MODE_PLAY)
        session.set_play_opponents(999)
        assert len(session.game.snakes) <= 12
        session.set_play_opponents(0)
        assert len(session.game.snakes) >= 2


class TestMechanicsVersionConfig:
    """The served game defaults to mechanics v2, opt-out via SNAKE_MECHANICS_V2=0."""

    def test_session_defaults_to_mechanics_v2(self, monkeypatch):
        monkeypatch.delenv("SNAKE_MECHANICS_V2", raising=False)
        sess = GameSession()
        state = sess.control_state()
        assert state["config"] == "mechanics_v2.yaml"
        assert state["mechanics_version"] == 2

    def test_flag_opts_out_to_v1_mechanics(self, monkeypatch):
        monkeypatch.setenv("SNAKE_MECHANICS_V2", "0")
        sess = GameSession()
        state = sess.control_state()
        assert state["config"] == "free_space_v2.yaml"
        assert state["mechanics_version"] == 1

    def test_finalized_run_carries_mechanics_version(self, monkeypatch):
        monkeypatch.delenv("SNAKE_MECHANICS_V2", raising=False)
        sess = GameSession()
        sess.set_mode(MODE_PLAY)
        sess.run_started = True
        human = sess._find_human()
        human.is_alive = False
        sess._step_play()
        assert sess.run_over is True
        assert sess.last_run["mechanics_version"] == 2
        assert sess.pending_submission()["mechanics_version"] == 2

    def test_finalized_run_reports_v1_when_opted_out(self, monkeypatch):
        monkeypatch.setenv("SNAKE_MECHANICS_V2", "0")
        sess = GameSession()
        sess.set_mode(MODE_PLAY)
        sess.run_started = True
        human = sess._find_human()
        human.is_alive = False
        sess._step_play()
        assert sess.last_run["mechanics_version"] == 1


@pytest.fixture()
def client(tmp_path, monkeypatch):
    """A TestClient with an isolated temp scores DB (lifespan builds the app)."""
    monkeypatch.setenv("SNAKE_SCORES_DB", str(tmp_path / "scores.db"))
    # Import inside the fixture so the env var is read at lifespan time.
    from web.backend.app import app, hub

    with TestClient(app) as c:
        yield c, hub


class TestLeaderboardEndpoints:
    def test_leaderboard_starts_empty(self, client):
        c, _ = client
        resp = c.get("/api/leaderboard")
        assert resp.status_code == 200
        data = resp.json()
        assert data["leaderboard"] == []
        assert data["stats"]["total_players"] == 0

    def test_submit_without_run_is_rejected(self, client):
        c, _ = client
        resp = c.post("/api/scores", json={"name": "ada"})
        assert resp.status_code == 409
        assert "No finished run" in resp.json()["detail"]

    def test_submit_records_finalized_run(self, client):
        c, hub = client
        # Inject a server-side finalized run (as if the human just died).
        hub.session.last_run = {
            "score": 250,
            "length": 12,
            "food_eaten": 8,
            "kills": 1,
            "frames": 300,
            "duration_seconds": 20.0,
            "checkpoint": "champion.pth",
        }
        hub.session.run_over = True
        hub.session.submitted = False

        resp = c.post("/api/scores", json={"name": "Ada"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["result"]["player_name"] == "Ada"
        assert body["result"]["score"] == 250
        assert body["leaderboard"][0]["player_name"] == "Ada"

    def test_submit_is_idempotent(self, client):
        c, hub = client
        hub.session.last_run = {
            "score": 100,
            "length": 5,
            "food_eaten": 2,
            "kills": 0,
            "frames": 100,
            "duration_seconds": 5.0,
            "checkpoint": None,
        }
        hub.session.run_over = True
        hub.session.submitted = False

        first = c.post("/api/scores", json={"name": "Bo"})
        second = c.post("/api/scores", json={"name": "Bo"})
        assert first.status_code == 200
        assert first.json()["ok"] is True
        # Idempotent per run: the second submit is rejected with a structured 409.
        assert second.status_code == 409
        # Only one game recorded.
        stats = c.get("/api/players/Bo").json()
        assert stats["found"] is True
        assert stats["player"]["games_played"] == 1

    def test_leaderboard_reflects_submissions(self, client):
        c, hub = client
        for name, score in (("low", 10), ("high", 999), ("mid", 500)):
            hub.session.last_run = {
                "score": score,
                "length": 5,
                "food_eaten": 0,
                "kills": 0,
                "frames": 10,
                "duration_seconds": 1.0,
                "checkpoint": None,
            }
            hub.session.run_over = True
            hub.session.submitted = False
            c.post("/api/scores", json={"name": name})

        board = c.get("/api/leaderboard").json()["leaderboard"]
        assert [e["player_name"] for e in board] == ["high", "mid", "low"]
        assert board[0]["rank"] == 1

    def test_player_lookup_unknown(self, client):
        c, _ = client
        assert c.get("/api/players/nobody").json()["found"] is False

    def test_end_to_end_real_run_to_submit(self, client):
        """Drive a real human game to death, then submit over HTTP and verify."""
        c, hub = client
        sess = hub.session
        sess.set_mode(MODE_PLAY)
        human = sess._find_human()

        # Steer straight up into the wall until the run ends.
        sess.human_input("up")
        frames = 0
        while human.is_alive and frames < 3000:
            sess.human_input("up")
            sess.step()
            frames += 1
        assert not human.is_alive
        assert sess.run_over is True

        pending = sess.pending_submission()
        assert pending is not None
        assert pending["score"] > 0

        resp = c.post("/api/scores", json={"name": "E2E"})
        body = resp.json()
        assert body["ok"] is True
        # Server-authoritative: recorded score matches the finalized run, not any
        # client-supplied value.
        assert body["result"]["score"] == pending["score"]
        assert body["result"]["length"] == pending["length"]

        board = c.get("/api/leaderboard").json()["leaderboard"]
        assert board[0]["player_name"] == "E2E"
        assert board[0]["score"] == pending["score"]
        recent = c.get("/api/scores/recent").json()["recent"]
        assert recent[0]["player_name"] == "E2E"

    def test_recent_scores_newest_first(self, client):
        c, hub = client
        for name in ("first", "second"):
            hub.session.last_run = {
                "score": 50,
                "length": 3,
                "food_eaten": 1,
                "kills": 0,
                "frames": 30,
                "duration_seconds": 2.0,
                "checkpoint": None,
            }
            hub.session.run_over = True
            hub.session.submitted = False
            c.post("/api/scores", json={"name": name})
        recent = c.get("/api/scores/recent").json()["recent"]
        assert [g["player_name"] for g in recent] == ["second", "first"]

    def test_submission_recorded_under_run_mechanics_version(self, client):
        c, hub = client
        hub.session.last_run = {
            "score": 321,
            "length": 9,
            "food_eaten": 5,
            "kills": 0,
            "frames": 200,
            "duration_seconds": 12.0,
            "checkpoint": None,
            "mechanics_version": 2,
        }
        hub.session.run_over = True
        hub.session.submitted = False
        body = c.post("/api/scores", json={"name": "V2"}).json()
        assert body["ok"] is True
        assert body["result"]["mechanics_version"] == 2
        # The historical v1 board is fetchable and does not contain the v2 run.
        v1_board = c.get("/api/leaderboard?mechanics_version=1").json()["leaderboard"]
        assert all(e["player_name"] != "V2" for e in v1_board)
        v2_board = c.get("/api/leaderboard?mechanics_version=2").json()["leaderboard"]
        assert any(e["player_name"] == "V2" for e in v2_board)

    def test_limit_is_clamped(self, client):
        c, _ = client
        # Absurd limits must not error; they clamp into range.
        assert c.get("/api/leaderboard?limit=99999").status_code == 200
        assert c.get("/api/leaderboard?limit=-5").status_code == 200
        assert c.get("/api/scores/recent?limit=0").status_code == 200
