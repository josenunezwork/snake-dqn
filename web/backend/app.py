"""FastAPI app: streams the live game over a WebSocket and serves controls + metrics.

Run from the repo root:
    ./venv/bin/python -m uvicorn web.backend.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import os
from typing import Set

import torch
from fastapi import Body, FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data.score_store import DEFAULT_SCORES_DB, ScoreStore
from web.backend import metrics
from web.backend.session import SAVED_DIR, GameSession

# Pin torch to one intra-op thread for the whole server process. The per-frame
# inference (raster or vector) is a tiny forward pass; torch's default of one
# thread per core oversubscribes the CPU against the event loop (measured ~3x
# slowdown). Set at import so it applies under uvicorn regardless of launcher.
torch.set_num_threads(1)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIST = os.path.join(REPO_ROOT, "web", "frontend", "dist")


class Hub:
    """Holds the single shared session + connected websocket clients."""

    def __init__(self) -> None:
        self.session: GameSession | None = None
        self.clients: Set[WebSocket] = set()
        self.engine_task: asyncio.Task | None = None
        self.store: ScoreStore | None = None

    async def broadcast(self, message: dict) -> None:
        dead = []
        for ws in list(self.clients):
            try:
                await ws.send_json(message)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.clients.discard(ws)

    async def engine_loop(self) -> None:
        """Step the game and broadcast a frame at the session's speed.

        With no connected clients the whole tick is skipped — the game already
        pauses without viewers, and serializing a frame runs a network forward
        pass for the inspector, so there is no point computing one nobody sees.
        """
        assert self.session is not None
        while True:
            sess = self.session
            try:
                if self.clients:
                    if sess.playing:
                        await asyncio.to_thread(sess.step)
                    frame = await asyncio.to_thread(sess.snapshot)
                    await self.broadcast(frame)
            except Exception as exc:  # never let the loop die
                await self.broadcast({"type": "error", "message": str(exc)})
            await asyncio.sleep(1.0 / max(1.0, sess.speed))


hub = Hub()


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):
    hub.store = ScoreStore(os.environ.get("SNAKE_SCORES_DB", DEFAULT_SCORES_DB))
    hub.session = await asyncio.to_thread(GameSession)
    hub.engine_task = asyncio.create_task(hub.engine_loop())
    try:
        yield
    finally:
        if hub.engine_task:
            hub.engine_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await hub.engine_task
        if hub.store is not None:
            hub.store.close()


app = FastAPI(title="snake-dqn web", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
async def health() -> dict:
    return {"ok": True, "session": hub.session is not None}


@app.get("/api/checkpoints")
async def checkpoints() -> dict:
    return {"checkpoints": metrics.list_checkpoints(), "dir": SAVED_DIR}


@app.get("/api/metrics")
async def get_metrics() -> dict:
    return await asyncio.to_thread(metrics.dashboard)


@app.get("/api/state")
async def state() -> dict:
    if hub.session is None:
        return {"ready": False}
    return {"ready": True, **hub.session.control_state()}


def _clamp_limit(limit: int, default: int = 10, hi: int = 100) -> int:
    """Clamp a client-supplied row limit into a sane range."""
    try:
        return max(1, min(hi, int(limit)))
    except (TypeError, ValueError):
        return default


@app.get("/api/leaderboard")
async def leaderboard(limit: int = 10, mechanics_version: int | None = None) -> dict:
    """Top players by best game, plus a whole-board summary.

    The board is versioned by game mechanics: it defaults to the active
    mechanics version's board; pass ``?mechanics_version=1`` for the read-only
    historical v1 board.
    """
    if hub.store is None:
        return {"leaderboard": [], "stats": None}
    limit = _clamp_limit(limit)
    board = await asyncio.to_thread(hub.store.leaderboard, limit, mechanics_version)
    stats = await asyncio.to_thread(hub.store.global_stats)
    return {
        "leaderboard": [e.to_dict() for e in board],
        "stats": stats.to_dict(),
    }


@app.get("/api/scores/recent")
async def recent_scores(limit: int = 8) -> dict:
    """The most recently recorded games, newest first."""
    if hub.store is None:
        return {"recent": []}
    limit = _clamp_limit(limit, default=8)
    games = await asyncio.to_thread(hub.store.recent_games, limit)
    return {"recent": [g.to_dict() for g in games]}


@app.get("/api/players/{name}")
async def player(name: str) -> dict:
    """Aggregate stats for one player (or ``found: False``)."""
    if hub.store is None:
        return {"found": False}
    stats = await asyncio.to_thread(hub.store.player_stats, name)
    return {"found": stats is not None, "player": stats.to_dict() if stats else None}


@app.post("/api/scores")
async def submit_score(payload: dict = Body(...)) -> dict:
    """Record the current finalized human run under a name (server-authoritative).

    The client sends only a name; the score and stats come from the session's
    finalized run so scores cannot be spoofed. Idempotent per run: a second POST
    after submission is a no-op that reports ``ok: False``.
    """
    if hub.session is None or hub.store is None:
        return {"ok": False, "error": "no active session"}
    name = str(payload.get("name", "")).strip() or "anonymous"
    # Atomically claim the run so a double-click / retry can't double-record it.
    pending = hub.session.claim_submission()
    if pending is None:
        return {"ok": False, "error": "no finished run to submit"}
    result = await asyncio.to_thread(
        hub.store.record_game,
        name,
        length=pending["length"],
        food_eaten=pending["food_eaten"],
        kills=pending["kills"],
        frames=pending["frames"],
        duration_seconds=pending["duration_seconds"],
        score=pending["score"],
        checkpoint=pending.get("checkpoint"),
        mechanics_version=pending.get("mechanics_version"),
    )
    board = await asyncio.to_thread(hub.store.leaderboard, 10)
    return {
        "ok": True,
        "result": result.to_dict(),
        "leaderboard": [e.to_dict() for e in board],
    }


# WebSocket control actions -> how to apply them to the session. Kept as a table
# so adding a control is one line and the dispatch stays flat.
_CONTROL_HANDLERS = {
    "play": lambda s, v: s.set_playing(True),
    "pause": lambda s, v: s.set_playing(False),
    "reset": lambda s, v: s.reset_game(),
    "new_game": lambda s, v: s.reset_game(),
    "set_speed": lambda s, v: s.set_speed(v),
    "set_mode": lambda s, v: s.set_mode(v),
    "set_epsilon": lambda s, v: s.set_epsilon(v),
    "set_hero": lambda s, v: s.set_hero(v),
    "set_food": lambda s, v: s.set_food(v),
    "load_checkpoint": lambda s, v: s.load_checkpoint(v),
    "human_input": lambda s, v: s.human_input(v),
    "human_boost": lambda s, v: s.human_boost(v),
    "set_play_opponents": lambda s, v: s.set_play_opponents(v),
}


def _apply_control(session: GameSession, msg: dict) -> None:
    handler = _CONTROL_HANDLERS.get(msg.get("action"))
    if handler is not None:
        handler(session, msg.get("value"))


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    try:
        if hub.session is not None:
            await ws.send_json(await asyncio.to_thread(hub.session.snapshot))
        while True:
            msg = await ws.receive_json()
            if hub.session is not None and msg.get("type") == "control":
                await asyncio.to_thread(_apply_control, hub.session, msg)
                await ws.send_json({"type": "ack", "state": hub.session.control_state()})
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        hub.clients.discard(ws)


# Serve the built frontend (production). In dev, Vite serves on :5173 and proxies.
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
else:

    @app.get("/")
    async def root() -> JSONResponse:
        return JSONResponse(
            {
                "message": "snake-dqn backend running. Build the frontend "
                "(cd web/frontend && npm install && npm run build) or run it in dev "
                "(npm run dev on :5173).",
                "api": [
                    "/api/health",
                    "/api/state",
                    "/api/checkpoints",
                    "/api/metrics",
                    "/ws/stream",
                ],
            }
        )
