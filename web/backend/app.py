"""FastAPI app: streams the live game over a WebSocket and serves controls + metrics.

Run from the repo root:
    ./venv/bin/python -m uvicorn web.backend.app:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
from typing import Dict, Optional, Set

import torch
from fastapi import Body, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from src.data.score_store import DEFAULT_SCORES_DB, ScoreStore
from web.backend import metrics
from web.backend.session import (
    MODE_PLAY,
    MODE_TRAIN,
    PROTOCOL_VERSION,
    SAVED_DIR,
    GameSession,
)

# Pin torch to one intra-op thread for the whole server process. The per-frame
# inference (raster or vector) is a tiny forward pass; torch's default of one
# thread per core oversubscribes the CPU against the event loop (measured ~3x
# slowdown). Set at import so it applies under uvicorn regardless of launcher.
torch.set_num_threads(1)

logger = logging.getLogger(__name__)

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
FRONTEND_DIST = os.path.join(REPO_ROOT, "web", "frontend", "dist")


# How long one client may block one frame delivery before the send is abandoned.
BROADCAST_SEND_TIMEOUT = 1.0
# Consecutive timed-out sends after which a chronically slow client is dropped.
BROADCAST_MAX_TIMEOUTS = 2
# While paused, heartbeat frames go out at ~1 Hz instead of the full tick rate.
PAUSED_HEARTBEAT_SECONDS = 1.0


class Hub:
    """Holds the single shared session + connected websocket clients."""

    def __init__(self) -> None:
        self.session: GameSession | None = None
        self.clients: Set[WebSocket] = set()
        self.engine_task: asyncio.Task | None = None
        self.store: ScoreStore | None = None
        # Consecutive send-timeout strikes per client (reset on any good send).
        self._send_timeouts: Dict[WebSocket, int] = {}

    def sync_viewer_count(self) -> None:
        """Mirror the connected-client count onto the session for the frame payload."""
        if self.session is not None:
            self.session.viewer_count = len(self.clients)

    async def broadcast(self, message: dict) -> None:
        """Send one frame to every client concurrently, never serially.

        The payload is serialized ONCE, and each send gets its own timeout so a
        slow client (congested link, backgrounded tab) cannot backpressure the
        engine tick for everyone else. A client that times out
        ``BROADCAST_MAX_TIMEOUTS`` frames in a row is disconnected.
        """
        clients = list(self.clients)
        if not clients:
            return
        text = json.dumps(message)
        results = await asyncio.gather(*(self._send_frame(ws, text) for ws in clients))
        for ws, status in zip(clients, results):
            if status == "ok":
                self._send_timeouts.pop(ws, None)
            elif status == "timeout":
                strikes = self._send_timeouts.get(ws, 0) + 1
                self._send_timeouts[ws] = strikes
                if strikes >= BROADCAST_MAX_TIMEOUTS:
                    await self._drop_client(ws)
            else:  # dead socket
                await self._drop_client(ws)

    async def _send_frame(self, ws: WebSocket, text: str) -> str:
        """Send one pre-serialized frame; classify the outcome, never raise."""
        try:
            await asyncio.wait_for(ws.send_text(text), timeout=BROADCAST_SEND_TIMEOUT)
            return "ok"
        except asyncio.TimeoutError:
            return "timeout"
        except Exception:
            return "dead"

    async def _drop_client(self, ws: WebSocket) -> None:
        """Remove a dead/chronically-slow client and release its session state."""
        self.clients.discard(ws)
        self._send_timeouts.pop(ws, None)
        self.sync_viewer_count()
        if self.session is not None:
            self.session.drop_connection(id(ws))
        with contextlib.suppress(Exception):
            await ws.close()

    async def engine_loop(self) -> None:
        """Step the game and broadcast a frame at the session's speed.

        With no connected clients the whole tick is skipped — the game already
        pauses without viewers, and serializing a frame runs a network forward
        pass for the inspector, so there is no point computing one nobody sees.

        While the session is paused nothing advances, so identical frames are
        not re-broadcast at the tick rate (that duplicate spam used to scroll a
        paused user's telemetry history away): one frame goes out right after
        pausing, then heartbeats at ~1 Hz.
        """
        assert self.session is not None
        last_paused_send = 0.0
        while True:
            sess = self.session
            try:
                if self.clients:
                    if sess.playing:
                        last_paused_send = 0.0
                        await asyncio.to_thread(sess.step)
                        frame = await asyncio.to_thread(sess.snapshot)
                        await self.broadcast(frame)
                    elif time.monotonic() - last_paused_send >= PAUSED_HEARTBEAT_SECONDS:
                        last_paused_send = time.monotonic()
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
    return {"ok": True, "session": hub.session is not None, "protocol_version": PROTOCOL_VERSION}


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
async def player(name: str, client_id: Optional[str] = None) -> dict:
    """Aggregate stats for one player (or ``found: False``).

    When ``client_id`` is given, the lookup prefers the player who last
    submitted with that id (renames keep working), falling back to name match.
    """
    if hub.store is None:
        return {"found": False}
    stats = await asyncio.to_thread(hub.store.player_stats, name, _clean_client_id(client_id))
    return {"found": stats is not None, "player": stats.to_dict() if stats else None}


def _clean_client_id(client_id: object) -> Optional[str]:
    """Normalize an optional opaque client id: trimmed, capped at 64 chars, or None."""
    if client_id is None:
        return None
    cleaned = str(client_id).strip()[:64]
    return cleaned or None


@app.post("/api/scores")
async def submit_score(payload: dict = Body(...)) -> dict:
    """Record the current finalized human run under a name (server-authoritative).

    The client sends a name (plus an optional opaque ``client_id`` used for
    "me" highlighting across renames); the score and stats come from the
    session's finalized run so scores cannot be spoofed. Idempotent per run: a
    second POST after a successful submission is a 409. The run is only marked
    submitted once the DB write succeeds — a store failure releases the claim
    so the player can retry.
    """
    if hub.session is None or hub.store is None:
        raise HTTPException(status_code=503, detail="No active session")
    name = " ".join(str(payload.get("name", "")).split()) or "anonymous"
    client_id = _clean_client_id(payload.get("client_id"))
    # Atomically claim the run so a double-click / retry can't double-record it.
    pending = hub.session.claim_submission()
    if pending is None:
        raise HTTPException(status_code=409, detail="No finished run to submit")
    try:
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
            client_id=client_id,
        )
    except Exception as exc:
        # The write failed: un-claim the run so it stays submittable, and give
        # the client a structured error it can distinguish from a network one.
        hub.session.release_submission()
        logger.exception("score submission failed")
        raise HTTPException(status_code=500, detail=f"Could not save the score: {exc}")
    board = await asyncio.to_thread(hub.store.leaderboard, 10)
    return {
        "ok": True,
        "result": result.to_dict(),
        "leaderboard": [e.to_dict() for e in board],
    }


# WebSocket control actions -> how to apply them to the session. Kept as a table
# so adding a control is one line and the dispatch stays flat. Handlers take
# (session, value, conn_id); most ignore conn_id, which exists for the
# per-connection controls (raster subscription, play-run ownership).
_CONTROL_HANDLERS = {
    "play": lambda s, v, c: s.set_playing(True),
    "pause": lambda s, v, c: s.set_playing(False),
    "reset": lambda s, v, c: s.reset_game(),
    "new_game": lambda s, v, c: s.reset_game(),
    "set_speed": lambda s, v, c: s.set_speed(v),
    # set_mode / load_checkpoint accept either a plain string value or
    # {"mode"/"name": ..., "override_reward_contract": true} — the dict form is
    # the client's explicit consent to fine-tune across reward contracts.
    "set_mode": lambda s, v, c: (
        s.set_mode(v.get("mode"), bool(v.get("override_reward_contract")))
        if isinstance(v, dict)
        else s.set_mode(v)
    ),
    "set_epsilon": lambda s, v, c: s.set_epsilon(v),
    "set_hero": lambda s, v, c: s.set_hero(v),
    "set_food": lambda s, v, c: s.set_food(v),
    "load_checkpoint": lambda s, v, c: (
        s.load_checkpoint(v.get("name"), bool(v.get("override_reward_contract")))
        if isinstance(v, dict)
        else s.load_checkpoint(v)
    ),
    "human_input": lambda s, v, c: s.human_input(v),
    "human_boost": lambda s, v, c: s.human_boost(v),
    "set_play_opponents": lambda s, v, c: s.set_play_opponents(v),
    "save_weights": lambda s, v, c: s.save_weights(),
    "set_raster_stream": lambda s, v, c: s.set_raster_stream(c, v),
}

# Controls that rebuild or reset the shared game — locked to the run owner
# while a scored play run is live.
_DESTRUCTIVE_ACTIONS = frozenset({"set_mode", "reset", "new_game", "load_checkpoint"})
# Steering inputs — silently ignored from non-owner connections during a run.
_STEERING_ACTIONS = frozenset({"human_input", "human_boost"})


def _foreign_run_active(session: GameSession, conn_id: object) -> bool:
    """True when another connection's scored play run is currently live."""
    owner = getattr(session, "run_owner", None)
    return (
        session.mode == MODE_PLAY
        and session.run_started
        and not session.run_over
        and owner is not None
        and owner != conn_id
    )


def _apply_control(session: GameSession, msg: object, conn_id: object = None) -> Optional[dict]:
    """Apply one control message to the session, rejecting anything malformed.

    Everything here is untrusted client input, so a bad shape, an unknown action
    or a value a handler cannot use is recorded on ``session.last_error`` (which
    ``control_state`` already ships back in the ack) instead of raised — one bad
    message must not take down the caller's stream.

    Returns:
        An optional per-connection reply frame (``{"type": "info"|"error", ...}``)
        the caller should send to THIS connection only, or None.
    """
    if not isinstance(msg, dict):
        logger.warning("ws control: ignoring non-object message of type %s", type(msg).__name__)
        return None
    action = msg.get("action")
    handler = _CONTROL_HANDLERS.get(action) if isinstance(action, str) else None
    if handler is None:
        logger.warning("ws control: unknown action %r", action)
        session.last_error = f"Unknown control action: {action!r}"
        return None
    # Play-run ownership: while a scored run started by another connection is
    # live, that run cannot be destroyed (mode switch, reset, checkpoint swap)
    # or steered from other tabs.
    if conn_id is not None and _foreign_run_active(session, conn_id):
        if action in _DESTRUCTIVE_ACTIONS:
            return {"type": "error", "message": "Another player's run is in progress"}
        if action in _STEERING_ACTIONS:
            return None
    try:
        result = handler(session, msg.get("value"), conn_id)
    except Exception as exc:
        logger.warning("ws control: %r failed", action, exc_info=True)
        session.last_error = f"Invalid {action} control: {exc}"
        return None
    # Record who owns the scored run: whoever pressed new_game, or (if nobody
    # did) whoever's first accepted input started it.
    if conn_id is not None and session.mode == MODE_PLAY:
        if action == "new_game":
            session.run_owner = conn_id
        elif (
            action == "human_input"
            and session.run_started
            and getattr(session, "run_owner", None) is None
        ):
            session.run_owner = conn_id
    if action == "save_weights" and isinstance(result, str):
        return {"type": "info", "message": f"Saved {result}"}
    return None


@app.websocket("/ws/stream")
async def ws_stream(ws: WebSocket) -> None:
    await ws.accept()
    hub.clients.add(ws)
    hub.sync_viewer_count()
    conn_id = id(ws)
    try:
        if hub.session is not None:
            await ws.send_json(await asyncio.to_thread(hub.session.snapshot))
        while True:
            try:
                msg = await ws.receive_json()
            except (ValueError, KeyError, TypeError):
                # Undecodable frame: drop the message, keep the connection. An
                # empty hub.clients pauses engine_loop for every other viewer.
                logger.warning("ws control: dropping undecodable frame", exc_info=True)
                continue
            if hub.session is None or not isinstance(msg, dict) or msg.get("type") != "control":
                continue
            reply = await asyncio.to_thread(_apply_control, hub.session, msg, conn_id)
            if reply is not None:
                await ws.send_json(reply)
            await ws.send_json({"type": "ack", "state": hub.session.control_state()})
    except WebSocketDisconnect:
        pass
    except Exception:
        logger.exception("ws_stream failed")
    finally:
        hub.clients.discard(ws)
        hub._send_timeouts.pop(ws, None)
        hub.sync_viewer_count()
        sess = hub.session
        if sess is not None:
            sess.drop_connection(conn_id)
            if not hub.clients and sess.mode == MODE_TRAIN:
                # The last viewer left mid-training. engine_loop already skips
                # empty-client ticks; pause explicitly so a later viewer's
                # connect doesn't silently resume mutating the policy weights.
                sess.set_playing(False)


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
