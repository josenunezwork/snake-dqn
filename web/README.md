# snake-dqn web app

The interactive UI for snake-dqn — a **server-authoritative** web app that replaced
the retired PyQt5 desktop GUI. The Python engine (`GameState` plus the selected
vector or raster policy) is the
single source of truth; the React frontend is a pure view that renders frames streamed
over a WebSocket. No game logic or model inference runs in the browser.

**Served setup (T34 CLOSED, 2026-11-04 PT):** Play runs `1` human against live AI opponents
(default `5`, adjustable `1–11`), defaulting to bare **B8 seed 3** (`b8-s3`, no veto; **Classic B8**,
released 2026-10-17) with an opt-in **"Field: Classic B8 / Mixed"** selector (`Mixed` = `frp3-s12 + v8`
opponents; leaderboard partitioned by field; force Classic with `SNAKE_SERVE_PLAY_ROSTER=all_b8`,
`b3789463`), while live Watch fallback (`SNAKE_WATCH_SOURCE=live`) runs a bare `b8-s3` hero against
the `mixed` `5× frp3-s12 + v8` roster. Watch plays the **132-session confirmed `exact-stop-l1`
replay bank** (`SNAKE_WATCH_SOURCE=replay`,
`SNAKE_WATCH_REPLAY_DIR=.../watch-replay-bank-live-132-l1`,
`SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l1`: `.watch.gz` gzipped canonical JSONL recordings from
banks `14l1–24l1` visibly labelled `"Recorded assisted hero"`, confirmed by `Watch6-L1-V1`
`GO_L1_SUPERIOR` and Mac `24 fps` qualification runs 2+3), with live bare `b8-s3` as automatic
fallback (`SNAKE_WATCH_SOURCE=live`; rollback banks `watch-replay-bank-live-36-l1` and
`watch-replay-bank-live-132` remain on disk). Replay is gated on
clean mixed-Watch6 `b8-s3` (`_is_clean_mixed_watch6_b8()`), so env-var checkpoint or roster
rollbacks (`SNAKE_SERVE_CHECKPOINT=frp3-s12|champion` or `SNAKE_SERVE_WATCH_ROSTER=all_b8`)
automatically run live. Full map of modules, the WebSocket
protocol, and every `SNAKE_SERVE_*` / `SNAKE_WATCH_*` env var:
[docs/architecture/WEB_MAP.md](../docs/architecture/WEB_MAP.md).

```
React + TS (canvas)  ──WebSocket──►  FastAPI  ──►  GameState · selected policy
   (pure view)          frames /         (web/backend)      (reused src/, unchanged)
                        controls
```

## Run it

```bash
# 1) backend deps (already in the project requirements.txt)
./venv/bin/pip install fastapi "uvicorn[standard]" websockets

# 2) build the frontend once
cd web/frontend && npm install && npm run build && cd ../..

# 3) serve everything (API + WebSocket + built UI) on one port
./venv/bin/python web/serve.py            # -> http://127.0.0.1:8000
#   PORT=9000 ./venv/bin/python web/serve.py
```

Frontend dev mode (hot reload), with the backend running separately:

```bash
./venv/bin/python web/serve.py            # backend on :8000
cd web/frontend && npm run dev            # UI on :5173 (proxies /api + /ws to :8000)
cd web/frontend && npm test               # frontend unit tests (vitest)
```

## What's in it (the panels)

The arena is always on screen; the other six are tabs beside it.

- **Game** — live arena rendered on a `<canvas>` from the streamed frame; hero snake outlined.
- **Play** — *you* steer a snake against the trained AI (arrows/WASD, space to boost) with a
  **SQLite leaderboard**. Server-authoritative: the run freezes until your first key, your death
  ends the run (AI opponents keep respawning), the score is computed on the server from
  length/food/kills/survival (can't be spoofed), and you pick the opponent count for difficulty.
- **Inspector** — the active policy's scalar state, dueling Q/value/advantages, and replay search
  override markers (`"Move from search"`); `ego2s-bv8` (`b8-s3`) exposes its 60-D scalar band,
  `raster31v2` exposes its 26-D scalars, and `vector61` exposes the 61-D free-space vector.
- **Raster** — what a `raster31v2` snake sees: the heading-rotated ego-centric 31×31 tactical and
  25×25 strategic stacks (head centred, facing up), as a colour-coded composite or a single isolated
  channel (brightness = the value byte). Shows a placeholder on `ego2s-*` (`b8-s3`) and `vector61`.
- **Network** — live dueling-Q details and, where the served policy exposes them, activation
  views with top action, margin, and activity.
- **Dashboard** — eval leaderboard parsed from `logs/eval_*.json` + the checkpoint inventory.
- **Controls** — play/pause, reset, speed, epsilon, hero selector, checkpoint loader, and a
  watch/train toggle (train rebuilds legacy `vector61` policies in training mode and is refused on
  forward-only `ego2s-*` / `b8-s3`, `raster31v2`, and `raster31v3` checkpoints before touching
  active Watch replay).

### Play / leaderboard REST

- `GET /api/leaderboard?limit=` — top players by best game + a whole-board summary.
- `GET /api/scores/recent?limit=` — most recent runs, newest first.
- `GET /api/players/{name}` — one player's aggregates.
- `POST /api/scores` `{name}` — record the session's finalized run (server-authoritative,
  atomic + idempotent per run). Scores live in `scores.db` ([`ScoreStore`](../src/data/score_store.py)),
  separate from the large experience-replay DB.

## Layout

```
web/
  serve.py              single-port launcher (backend + built frontend)
  backend/
    app.py              FastAPI: /ws/stream broadcast loop + REST + static serve
    session.py          GameSession: owns GameState + selected policy, controls
    serialize.py        GameState -> frame DTO (the wire contract); PyQt-free
    served_checkpoint.py  pinned registry (b8-s3 / frp3-s12 / champion), env-selected
    safety_veto_serving.py  v2/v5/v7/v8 veto wrappers (vector61 only)
    ego2s_policy.py, raster_policy.py  serving policies for raster/ego checkpoints
    watch_replay.py     verified snake-watch-replay/v1 precomputed Watch replay bank
    pathb*.py, serving_speed.py, release_monitoring.py  see WEB_MAP.md
    metrics.py          dashboard data from saved_snakes/ + logs/eval_*.json
    state_labels.py     vector feature grouping for the inspector
  frontend/             Vite + React + TypeScript (canvas view, no extra UI deps)
    src/components/     GameCanvas, Play, Controls, Inspector, EgoRasterViewer,
                        NetworkVisualizer, Dashboard, ...
    src/*.test.ts(x)    vitest unit tests (api error paths, Play flow, keys, Slider)
```

The backend imports nothing from PyQt — it runs headless on a server. The human-play
score store is `src/data/score_store.py`; the play mode lives in `session.py` (mode `"play"`).
