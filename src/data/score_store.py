"""SQLite-backed store for human play sessions and the high-score leaderboard.

This is deliberately separate from :mod:`src.data.memory_db_handler` (the large
Apex-DQN experience-replay database). Human scores are a small, user-facing
concern with a very different access pattern — a handful of writes per game and
cheap ranked reads — so they get their own tidy database and schema.

Design goals:
    * One row per completed game in ``games``; denormalized per-player aggregates
      in ``players`` so the common "how am I doing" read is a single lookup.
    * Every query is parameterized (no string interpolation into SQL).
    * Thread-safe: the web engine steps the game on a background thread while
      control handlers write scores from another, so the connection is shared
      under a lock with ``check_same_thread=False``.
    * ``ScoreStore(":memory:")`` gives a throwaway store for tests.

Scoring policy lives in :func:`compute_score` so the definition of "score" is in
one place and independently testable.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import List, Optional

__all__ = [
    "ScoreStore",
    "GameResult",
    "LeaderboardEntry",
    "PlayerStats",
    "GlobalStats",
    "compute_score",
    "DEFAULT_SCORES_DB",
]

# Runtime DB lives next to the replay DB at the repo root by default. It is
# gitignored (``*.db``). Override with the ``SNAKE_SCORES_DB`` env var or the
# constructor argument (tests pass ``":memory:"``).
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_SCORES_DB = os.environ.get("SNAKE_SCORES_DB", os.path.join(_REPO_ROOT, "scores.db"))

# Schema v2 (blueprint §5.4): games gained a ``mechanics_version`` column so the
# leaderboard can be versioned — v1 scores stay readable (historical, read-only
# by convention) while v2 sessions rank on a fresh board. Migration is additive
# (ALTER TABLE with DEFAULT 1); existing rows are never rewritten or dropped.
SCHEMA_VERSION = 2

# Score weights. Snake length is the headline number; food and kills add flavor,
# and a small survival term rewards staying alive without dominating the score.
_FOOD_WEIGHT = 10
_KILL_WEIGHT = 50
_SURVIVAL_PER_FRAME = 0.1


def compute_score(length: int, food_eaten: int, kills: int, frames: int) -> int:
    """Combine a run's stats into a single integer score.

    The formula favors length and aggression while giving a gentle bonus for
    survival. Kept pure and centralized so the leaderboard, the live HUD, and
    tests all agree on what "score" means.

    Args:
        length: Final snake length (segments).
        food_eaten: Food pellets consumed this run.
        kills: Enemy snakes killed this run.
        frames: Frames the snake survived.

    Returns:
        A non-negative integer score.
    """
    score = (
        max(0, int(length)) * _FOOD_WEIGHT
        + max(0, int(food_eaten)) * _FOOD_WEIGHT
        + max(0, int(kills)) * _KILL_WEIGHT
        + int(max(0, int(frames)) * _SURVIVAL_PER_FRAME)
    )
    return int(score)


@dataclass(frozen=True)
class GameResult:
    """A single completed game as stored in the ``games`` table."""

    id: int
    player_name: str
    score: int
    length: int
    food_eaten: int
    kills: int
    frames: int
    duration_seconds: float
    mode: str
    checkpoint: Optional[str]
    created_at: str
    mechanics_version: int = 1

    def to_dict(self) -> dict:
        """Return a JSON-serializable dict (used by the web API)."""
        return asdict(self)


@dataclass(frozen=True)
class LeaderboardEntry:
    """A player's best game, for ranked leaderboard display."""

    rank: int
    player_name: str
    score: int
    length: int
    food_eaten: int
    kills: int
    frames: int
    created_at: str
    mechanics_version: int = 1

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class PlayerStats:
    """Denormalized per-player aggregates from the ``players`` table."""

    name: str
    games_played: int
    best_score: int
    total_score: int
    created_at: str
    last_played_at: Optional[str]

    @property
    def average_score(self) -> float:
        return self.total_score / self.games_played if self.games_played else 0.0

    def to_dict(self) -> dict:
        data = asdict(self)
        data["average_score"] = self.average_score
        return data


@dataclass(frozen=True)
class GlobalStats:
    """Whole-leaderboard summary for a dashboard header."""

    total_players: int
    total_games: int
    best_score: int
    best_player: Optional[str]
    average_score: float

    def to_dict(self) -> dict:
        return asdict(self)


def _active_mechanics_version() -> int:
    """Return the mechanics version of the currently active game config.

    Falls back to 1 (legacy mechanics) when no config has been initialized,
    e.g. when a CLI tool opens the store outside a running game.

    Returns:
        The active ``game.mechanics_version`` (currently 1 or 2).
    """
    try:
        from src.core.game_config import get_config

        return int(get_config().game.mechanics_version)
    except Exception:
        return 1


def _utc_now_iso() -> str:
    """Timezone-aware UTC timestamp in ISO-8601 (seconds precision)."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _sanitize_name(name: str) -> str:
    """Normalize a player name: trim, collapse blanks, cap length."""
    cleaned = " ".join(str(name).split()).strip()
    if not cleaned:
        cleaned = "anonymous"
    return cleaned[:32]


class ScoreStore:
    """Persistent leaderboard for human play sessions.

    Usage:
        >>> store = ScoreStore(":memory:")
        >>> result = store.record_game("ada", length=12, food_eaten=8, kills=1, frames=430)
        >>> [e.player_name for e in store.leaderboard()]
        ['ada']
        >>> store.close()

    Can also be used as a context manager::

        with ScoreStore(path) as store:
            store.record_game(...)
    """

    def __init__(self, db_path: str = DEFAULT_SCORES_DB) -> None:
        """Open (creating if needed) the scores database.

        Args:
            db_path: Filesystem path, or ``":memory:"`` for an ephemeral store.
        """
        self.db_path = db_path
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA foreign_keys=ON;")
            self._create_schema()

    # -- schema -------------------------------------------------------------
    def _create_schema(self) -> None:
        cur = self._conn
        cur.execute("""
            CREATE TABLE IF NOT EXISTS players (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                name           TEXT    NOT NULL UNIQUE,
                created_at     TEXT    NOT NULL,
                last_played_at TEXT,
                games_played   INTEGER NOT NULL DEFAULT 0,
                best_score     INTEGER NOT NULL DEFAULT 0,
                total_score    INTEGER NOT NULL DEFAULT 0
            )
            """)
        cur.execute("""
            CREATE TABLE IF NOT EXISTS games (
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
                created_at       TEXT    NOT NULL,
                mechanics_version INTEGER NOT NULL DEFAULT 1
            )
            """)
        # v1 -> v2 migration: purely additive. Pre-existing databases lack the
        # mechanics_version column; ALTER TABLE backfills every historical row
        # with DEFAULT 1 (legacy mechanics) and never touches existing data.
        cols = {row["name"] for row in cur.execute("PRAGMA table_info(games)").fetchall()}
        if "mechanics_version" not in cols:
            cur.execute("ALTER TABLE games ADD COLUMN mechanics_version INTEGER NOT NULL DEFAULT 1")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_score ON games(score DESC)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_games_player ON games(player_id)")
        cur.execute(
            "CREATE INDEX IF NOT EXISTS idx_games_mechanics "
            "ON games(mechanics_version, score DESC)"
        )
        cur.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        self._conn.commit()

    # -- writes -------------------------------------------------------------
    def record_game(
        self,
        player_name: str,
        *,
        length: int,
        food_eaten: int = 0,
        kills: int = 0,
        frames: int = 0,
        duration_seconds: float = 0.0,
        mode: str = "play",
        checkpoint: Optional[str] = None,
        score: Optional[int] = None,
        created_at: Optional[str] = None,
        mechanics_version: Optional[int] = None,
    ) -> GameResult:
        """Persist one completed game and update the player's aggregates.

        The player row is created on first sighting. Score defaults to
        :func:`compute_score` of the run stats but may be passed explicitly.

        Args:
            player_name: Display name (trimmed, capped at 32 chars).
            length: Final snake length.
            food_eaten: Food pellets eaten.
            kills: Enemy kills.
            frames: Frames survived.
            duration_seconds: Wall-clock seconds the run lasted.
            mode: Game mode label (e.g. ``"play"``).
            checkpoint: Name of the AI checkpoint the human played against.
            score: Explicit score; computed from stats when ``None``.
            created_at: ISO timestamp; defaults to now (UTC).
            mechanics_version: Game-mechanics version the run was played under;
                defaults to the active config's version (1 when no config).

        Returns:
            The stored :class:`GameResult`.
        """
        name = _sanitize_name(player_name)
        when = created_at or _utc_now_iso()
        length = max(1, int(length))
        food_eaten = max(0, int(food_eaten))
        kills = max(0, int(kills))
        frames = max(0, int(frames))
        duration_seconds = max(0.0, float(duration_seconds))
        if score is None:
            score = compute_score(length, food_eaten, kills, frames)
        score = max(0, int(score))
        if mechanics_version is None:
            mechanics_version = _active_mechanics_version()
        mechanics_version = max(1, int(mechanics_version))

        with self._lock:
            cur = self._conn
            cur.execute(
                "INSERT INTO players (name, created_at, last_played_at) VALUES (?, ?, ?) "
                "ON CONFLICT(name) DO UPDATE SET last_played_at = excluded.last_played_at",
                (name, when, when),
            )
            row = cur.execute("SELECT id FROM players WHERE name = ?", (name,)).fetchone()
            player_id = int(row["id"])

            cursor = cur.execute(
                "INSERT INTO games (player_id, score, length, food_eaten, kills, frames, "
                "duration_seconds, mode, checkpoint, created_at, mechanics_version) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    player_id,
                    score,
                    length,
                    food_eaten,
                    kills,
                    frames,
                    duration_seconds,
                    mode,
                    checkpoint,
                    when,
                    mechanics_version,
                ),
            )
            game_id = int(cursor.lastrowid)

            cur.execute(
                "UPDATE players SET games_played = games_played + 1, "
                "total_score = total_score + ?, "
                "best_score = MAX(best_score, ?), "
                "last_played_at = ? WHERE id = ?",
                (score, score, when, player_id),
            )
            self._conn.commit()

        return GameResult(
            id=game_id,
            player_name=name,
            score=score,
            length=length,
            food_eaten=food_eaten,
            kills=kills,
            frames=frames,
            duration_seconds=duration_seconds,
            mode=mode,
            checkpoint=checkpoint,
            created_at=when,
            mechanics_version=mechanics_version,
        )

    # -- reads --------------------------------------------------------------
    def leaderboard(
        self, limit: int = 10, mechanics_version: Optional[int] = None
    ) -> List[LeaderboardEntry]:
        """Return the top players by their single best game, per mechanics version.

        One row per player (their best run), ranked by score descending. Ties
        break toward the earlier achievement. The board is versioned: only games
        played under one mechanics version are ranked together, so v2 sessions
        get a fresh board while the v1 board stays fetchable read-only.

        Args:
            limit: Maximum rows to return.
            mechanics_version: Which version's board to return. Defaults to the
                active config's version (so a v2 session shows the v2 board);
                pass ``1`` explicitly for the historical v1 board.

        Returns:
            Ranked :class:`LeaderboardEntry` list.
        """
        limit = max(1, int(limit))
        if mechanics_version is None:
            mechanics_version = _active_mechanics_version()
        mechanics_version = max(1, int(mechanics_version))
        with self._lock:
            # ROW_NUMBER picks each player's single best game deterministically —
            # highest score, and among ties the earliest achievement (then lowest
            # id). A bare GROUP BY would collapse tied best-score games and surface
            # an arbitrary row's created_at, contradicting the earlier-achievement
            # contract above.
            rows = self._conn.execute(
                """
                SELECT p.name AS player_name, g.score, g.length, g.food_eaten,
                       g.kills, g.frames, g.created_at, g.mechanics_version
                FROM (
                    SELECT *, ROW_NUMBER() OVER (
                        PARTITION BY player_id
                        ORDER BY score DESC, created_at ASC, id ASC
                    ) AS rn
                    FROM games
                    WHERE mechanics_version = ?
                ) g
                JOIN players p ON p.id = g.player_id
                WHERE g.rn = 1
                ORDER BY g.score DESC, g.created_at ASC, g.id ASC
                LIMIT ?
                """,
                (mechanics_version, limit),
            ).fetchall()
        return [
            LeaderboardEntry(
                rank=i + 1,
                player_name=r["player_name"],
                score=int(r["score"]),
                length=int(r["length"]),
                food_eaten=int(r["food_eaten"]),
                kills=int(r["kills"]),
                frames=int(r["frames"]),
                created_at=r["created_at"],
                mechanics_version=int(r["mechanics_version"]),
            )
            for i, r in enumerate(rows)
        ]

    def recent_games(
        self, limit: int = 20, mechanics_version: Optional[int] = None
    ) -> List[GameResult]:
        """Return the most recently recorded games, newest first.

        Args:
            limit: Maximum rows to return.
            mechanics_version: When given, only games played under that
                mechanics version; ``None`` (default) returns the mixed feed.

        Returns:
            Newest-first :class:`GameResult` list.
        """
        limit = max(1, int(limit))
        where = ""
        params: tuple = (limit,)
        if mechanics_version is not None:
            where = "WHERE g.mechanics_version = ?"
            params = (max(1, int(mechanics_version)), limit)
        with self._lock:
            rows = self._conn.execute(
                f"""
                SELECT g.id, p.name AS player_name, g.score, g.length, g.food_eaten,
                       g.kills, g.frames, g.duration_seconds, g.mode, g.checkpoint,
                       g.created_at, g.mechanics_version
                FROM games g JOIN players p ON p.id = g.player_id
                {where}
                ORDER BY g.id DESC LIMIT ?
                """,
                params,
            ).fetchall()
        return [self._row_to_result(r) for r in rows]

    def player_stats(self, name: str) -> Optional[PlayerStats]:
        """Return aggregates for one player, or ``None`` if unknown."""
        clean = _sanitize_name(name)
        with self._lock:
            row = self._conn.execute(
                "SELECT name, games_played, best_score, total_score, created_at, "
                "last_played_at FROM players WHERE name = ?",
                (clean,),
            ).fetchone()
        if row is None:
            return None
        return PlayerStats(
            name=row["name"],
            games_played=int(row["games_played"]),
            best_score=int(row["best_score"]),
            total_score=int(row["total_score"]),
            created_at=row["created_at"],
            last_played_at=row["last_played_at"],
        )

    def global_stats(self) -> GlobalStats:
        """Return a whole-leaderboard summary."""
        with self._lock:
            players = int(self._conn.execute("SELECT COUNT(*) AS c FROM players").fetchone()["c"])
            grow = self._conn.execute(
                "SELECT COUNT(*) AS c, COALESCE(AVG(score), 0.0) AS avg FROM games"
            ).fetchone()
            best = self._conn.execute(
                "SELECT p.name AS name, g.score AS score FROM games g "
                "JOIN players p ON p.id = g.player_id ORDER BY g.score DESC, g.id ASC LIMIT 1"
            ).fetchone()
        return GlobalStats(
            total_players=players,
            total_games=int(grow["c"]),
            best_score=int(best["score"]) if best else 0,
            best_player=best["name"] if best else None,
            average_score=float(grow["avg"]),
        )

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        """Close the underlying connection (idempotent)."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass

    def __enter__(self) -> "ScoreStore":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    @staticmethod
    def _row_to_result(r: sqlite3.Row) -> GameResult:
        return GameResult(
            id=int(r["id"]),
            player_name=r["player_name"],
            score=int(r["score"]),
            length=int(r["length"]),
            food_eaten=int(r["food_eaten"]),
            kills=int(r["kills"]),
            frames=int(r["frames"]),
            duration_seconds=float(r["duration_seconds"]),
            mode=r["mode"],
            checkpoint=r["checkpoint"],
            created_at=r["created_at"],
            mechanics_version=int(r["mechanics_version"]),
        )
