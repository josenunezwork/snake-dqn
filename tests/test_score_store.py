"""Tests for the SQLite-backed human-play leaderboard (src/data/score_store.py)."""

import sqlite3

import pytest

from src.data.score_store import SCHEMA_VERSION, GlobalStats, ScoreStore, compute_score


@pytest.fixture()
def store():
    s = ScoreStore(":memory:")
    yield s
    s.close()


class TestComputeScore:
    def test_monotonic_in_each_component(self):
        base = compute_score(length=5, food_eaten=2, kills=0, frames=100)
        assert compute_score(length=6, food_eaten=2, kills=0, frames=100) > base
        assert compute_score(length=5, food_eaten=3, kills=0, frames=100) > base
        assert compute_score(length=5, food_eaten=2, kills=1, frames=100) > base
        assert compute_score(length=5, food_eaten=2, kills=0, frames=200) > base

    def test_kills_weighted_above_food(self):
        # A kill should be worth more than a single pellet.
        assert compute_score(1, 1, 0, 0) < compute_score(1, 0, 1, 0)

    def test_negative_inputs_clamped_non_negative(self):
        assert compute_score(-5, -5, -5, -5) >= 0

    def test_returns_int(self):
        assert isinstance(compute_score(5, 5, 5, 555), int)


class TestRecordGame:
    def test_record_creates_player_and_game(self, store):
        result = store.record_game("Ada", length=10, food_eaten=8, kills=1, frames=400)
        assert result.player_name == "Ada"
        assert result.length == 10
        assert result.score == compute_score(10, 8, 1, 400)
        stats = store.player_stats("Ada")
        assert stats is not None
        assert stats.games_played == 1
        assert stats.best_score == result.score

    def test_explicit_score_overrides_formula(self, store):
        result = store.record_game("Bo", length=3, score=9999)
        assert result.score == 9999

    def test_name_is_sanitized(self, store):
        result = store.record_game("   spaced   out   ", length=2)
        assert result.player_name == "spaced out"

    def test_blank_name_becomes_anonymous(self, store):
        result = store.record_game("   ", length=2)
        assert result.player_name == "anonymous"

    def test_long_name_capped(self, store):
        result = store.record_game("x" * 100, length=2)
        assert len(result.player_name) == 32

    def test_aggregates_accumulate_across_games(self, store):
        store.record_game("Ada", length=5, score=100)
        store.record_game("Ada", length=8, score=250)
        store.record_game("Ada", length=2, score=40)
        stats = store.player_stats("Ada")
        assert stats.games_played == 3
        assert stats.best_score == 250
        assert stats.total_score == 390
        assert stats.average_score == pytest.approx(130.0)

    def test_length_floored_to_one(self, store):
        result = store.record_game("Ada", length=0)
        assert result.length == 1


class TestLeaderboard:
    def test_one_row_per_player_best_game(self, store):
        store.record_game("Ada", length=5, score=100)
        store.record_game("Ada", length=9, score=300)  # Ada's best
        store.record_game("Bo", length=7, score=200)
        board = store.leaderboard()
        assert [e.player_name for e in board] == ["Ada", "Bo"]
        assert board[0].score == 300
        assert board[0].rank == 1
        assert board[1].rank == 2

    def test_respects_limit(self, store):
        for i in range(5):
            store.record_game(f"p{i}", length=2, score=i * 10)
        assert len(store.leaderboard(limit=3)) == 3

    def test_empty_leaderboard(self, store):
        assert store.leaderboard() == []

    def test_ranked_descending(self, store):
        store.record_game("low", length=2, score=10)
        store.record_game("high", length=2, score=999)
        store.record_game("mid", length=2, score=500)
        scores = [e.score for e in store.leaderboard()]
        assert scores == sorted(scores, reverse=True)

    def test_tie_break_prefers_earliest_best_game(self, store):
        # Same player, same best score, inserted latest-first: the leaderboard
        # must surface the EARLIEST achievement's created_at (deterministically).
        store.record_game("ada", length=1, score=100, created_at="2020-12-31T00:00:00+00:00")
        store.record_game("ada", length=1, score=100, created_at="2020-01-01T00:00:00+00:00")
        store.record_game("ada", length=1, score=100, created_at="2021-06-15T00:00:00+00:00")
        entry = store.leaderboard()[0]
        assert entry.score == 100
        assert entry.created_at == "2020-01-01T00:00:00+00:00"


class TestRecentGamesAndGlobalStats:
    def test_recent_games_newest_first(self, store):
        store.record_game("a", length=2, score=1)
        store.record_game("b", length=2, score=2)
        recent = store.recent_games()
        assert recent[0].player_name == "b"
        assert recent[1].player_name == "a"

    def test_global_stats_empty(self, store):
        gs = store.global_stats()
        assert isinstance(gs, GlobalStats)
        assert gs.total_players == 0
        assert gs.total_games == 0
        assert gs.best_player is None

    def test_global_stats_populated(self, store):
        store.record_game("Ada", length=2, score=100)
        store.record_game("Bo", length=2, score=300)
        store.record_game("Bo", length=2, score=50)
        gs = store.global_stats()
        assert gs.total_players == 2
        assert gs.total_games == 3
        assert gs.best_score == 300
        assert gs.best_player == "Bo"
        assert gs.average_score == pytest.approx(150.0)


class TestPersistenceAndLifecycle:
    def test_survives_reopen(self, tmp_path):
        db = str(tmp_path / "scores.db")
        with ScoreStore(db) as s:
            s.record_game("Ada", length=10, score=500)
        with ScoreStore(db) as s:
            assert s.player_stats("Ada").best_score == 500
            assert s.leaderboard()[0].player_name == "Ada"

    def test_unknown_player_returns_none(self, store):
        assert store.player_stats("nobody") is None

    def test_to_dict_is_json_friendly(self, store):
        result = store.record_game("Ada", length=10, food_eaten=3, score=500)
        d = result.to_dict()
        assert d["player_name"] == "Ada" and d["score"] == 500
        entry = store.leaderboard()[0].to_dict()
        assert entry["rank"] == 1
        stats = store.player_stats("Ada").to_dict()
        assert "average_score" in stats

    def test_close_is_idempotent(self, store):
        store.close()
        store.close()


class TestMechanicsVersioning:
    """Schema v2: per-mechanics-version score recording and leaderboards."""

    def test_record_uses_active_version_by_default(self, store, monkeypatch):
        monkeypatch.setattr("src.data.score_store._active_mechanics_version", lambda: 2)
        result = store.record_game("Ada", length=5, score=100)
        assert result.mechanics_version == 2

    def test_explicit_version_overrides_active(self, store, monkeypatch):
        monkeypatch.setattr("src.data.score_store._active_mechanics_version", lambda: 2)
        result = store.record_game("Ada", length=5, score=100, mechanics_version=1)
        assert result.mechanics_version == 1

    def test_leaderboard_filters_by_version(self, store):
        store.record_game("old", length=5, score=900, mechanics_version=1)
        store.record_game("new", length=5, score=100, mechanics_version=2)
        v1 = store.leaderboard(mechanics_version=1)
        v2 = store.leaderboard(mechanics_version=2)
        assert [e.player_name for e in v1] == ["old"]
        assert [e.player_name for e in v2] == ["new"]
        assert v1[0].mechanics_version == 1
        assert v2[0].mechanics_version == 2

    def test_leaderboard_default_is_active_version_board(self, store, monkeypatch):
        monkeypatch.setattr("src.data.score_store._active_mechanics_version", lambda: 2)
        store.record_game("old", length=5, score=900, mechanics_version=1)
        store.record_game("new", length=5, score=100)  # records at active = 2
        board = store.leaderboard()
        assert [e.player_name for e in board] == ["new"]
        # The v1 board stays fetchable (read-only historical view).
        assert [e.player_name for e in store.leaderboard(mechanics_version=1)] == ["old"]

    def test_player_best_is_per_version(self, store):
        # Same player: a huge v1 score must not shadow their v2 board entry.
        store.record_game("Ada", length=5, score=900, mechanics_version=1)
        store.record_game("Ada", length=5, score=50, mechanics_version=2)
        v2 = store.leaderboard(mechanics_version=2)
        assert v2[0].player_name == "Ada"
        assert v2[0].score == 50

    def test_recent_games_mixed_feed_by_default(self, store):
        store.record_game("old", length=2, score=10, mechanics_version=1)
        store.record_game("new", length=2, score=20, mechanics_version=2)
        recent = store.recent_games()
        assert [g.player_name for g in recent] == ["new", "old"]
        assert [g.mechanics_version for g in recent] == [2, 1]

    def test_recent_games_explicit_version_filter(self, store):
        store.record_game("old", length=2, score=10, mechanics_version=1)
        store.record_game("new", length=2, score=20, mechanics_version=2)
        assert [g.player_name for g in store.recent_games(mechanics_version=1)] == ["old"]
        assert [g.player_name for g in store.recent_games(mechanics_version=2)] == ["new"]

    def test_to_dict_includes_mechanics_version(self, store):
        store.record_game("Ada", length=5, score=100, mechanics_version=2)
        assert store.recent_games()[0].to_dict()["mechanics_version"] == 2
        assert store.leaderboard(mechanics_version=2)[0].to_dict()["mechanics_version"] == 2


# The exact games/players DDL that shipped with schema v1 (no mechanics_version
# column), used to build a realistic pre-migration fixture database.
_V1_GAMES_DDL = """
CREATE TABLE games (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    player_id        INTEGER NOT NULL REFERENCES players(id) ON DELETE CASCADE,
    score            INTEGER NOT NULL,
    length           INTEGER NOT NULL,
    food_eaten       INTEGER NOT NULL DEFAULT 0,
    kills            INTEGER NOT NULL DEFAULT 0,
    frames           INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL    NOT NULL DEFAULT 0.0,
    mode             TEXT    NOT NULL DEFAULT 'play',
    checkpoint       TEXT,
    created_at       TEXT    NOT NULL
)
"""

_V1_PLAYERS_DDL = """
CREATE TABLE players (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    name           TEXT    NOT NULL UNIQUE,
    created_at     TEXT    NOT NULL,
    last_played_at TEXT,
    games_played   INTEGER NOT NULL DEFAULT 0,
    best_score     INTEGER NOT NULL DEFAULT 0,
    total_score    INTEGER NOT NULL DEFAULT 0
)
"""


def _make_v1_fixture_db(path: str) -> None:
    """Create a scores.db exactly as schema v1 wrote it, with two games."""
    conn = sqlite3.connect(path)
    try:
        conn.execute(_V1_PLAYERS_DDL)
        conn.execute(_V1_GAMES_DDL)
        conn.execute(
            "INSERT INTO players (id, name, created_at, games_played, best_score, total_score) "
            "VALUES (1, 'Ada', '2026-06-30T00:00:00+00:00', 2, 500, 800)"
        )
        conn.execute(
            "INSERT INTO games (player_id, score, length, food_eaten, kills, frames, "
            "duration_seconds, mode, checkpoint, created_at) VALUES "
            "(1, 500, 12, 8, 1, 400, 30.0, 'play', 'champion.pth', '2026-06-30T00:00:00+00:00')"
        )
        conn.execute(
            "INSERT INTO games (player_id, score, length, food_eaten, kills, frames, "
            "duration_seconds, mode, checkpoint, created_at) VALUES "
            "(1, 300, 8, 4, 0, 250, 18.0, 'play', 'champion.pth', '2026-07-01T00:00:00+00:00')"
        )
        conn.execute("PRAGMA user_version = 1")
        conn.commit()
    finally:
        conn.close()


class TestV1ToV2Migration:
    """Opening a v1 database migrates it additively; nothing is lost."""

    def test_v1_rows_preserved_with_default_version_1(self, tmp_path):
        db = str(tmp_path / "scores.db")
        _make_v1_fixture_db(db)
        with ScoreStore(db) as store:
            recent = store.recent_games()
            assert len(recent) == 2
            assert all(g.mechanics_version == 1 for g in recent)
            assert {g.score for g in recent} == {500, 300}
            # Historical v1 board intact, aggregates untouched.
            board = store.leaderboard(mechanics_version=1)
            assert board[0].player_name == "Ada"
            assert board[0].score == 500
            assert store.player_stats("Ada").games_played == 2

    def test_migration_bumps_user_version(self, tmp_path):
        db = str(tmp_path / "scores.db")
        _make_v1_fixture_db(db)
        ScoreStore(db).close()
        conn = sqlite3.connect(db)
        try:
            assert conn.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
        finally:
            conn.close()

    def test_migration_is_idempotent_across_reopens(self, tmp_path):
        db = str(tmp_path / "scores.db")
        _make_v1_fixture_db(db)
        ScoreStore(db).close()
        with ScoreStore(db) as store:  # second open: ALTER must not re-run/fail
            assert len(store.recent_games()) == 2

    def test_v2_scores_do_not_pollute_v1_board(self, tmp_path):
        db = str(tmp_path / "scores.db")
        _make_v1_fixture_db(db)
        with ScoreStore(db) as store:
            store.record_game("Bo", length=20, score=9999, mechanics_version=2)
            v1 = store.leaderboard(mechanics_version=1)
            assert [e.player_name for e in v1] == ["Ada"]
            v2 = store.leaderboard(mechanics_version=2)
            assert [e.player_name for e in v2] == ["Bo"]
