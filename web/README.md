# snake-dqn web app

The interactive UI for snake-dqn — a **server-authoritative** web app that replaced
the retired PyQt5 desktop GUI. The Python engine (`GameState` plus the selected
vector or raster policy) is the
single source of truth; the React frontend is a pure view that renders frames streamed
over a WebSocket. No game logic or model inference runs in the browser.

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
./venv/bin/python web/serve.py            # -> http://localhost:8000
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
- **Inspector** — the active policy's state and Q-values; vector checkpoints expose the 61-D
  free-space features and raster checkpoints expose their ego-raster observation.
- **Raster** — what a `raster31v2` snake sees: the heading-rotated ego-centric 31×31 tactical
  stack (head centred, facing up), as a colour-coded composite or a single isolated channel
  (brightness = the value byte). Shows a placeholder when the served policy is the 61-D vector.
- **Network** — live dueling-Q details and, where the served policy exposes them, activation
  views with top action, margin, and activity.
- **Dashboard** — eval leaderboard parsed from `logs/eval_*.json` + the checkpoint inventory.
- **Controls** — play/pause, reset, speed, epsilon, hero selector, checkpoint loader, and a
  watch/train toggle (train rebuilds the policy in training mode).

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
    metrics.py          dashboard data from saved_snakes/ + logs/eval_*.json
    state_labels.py     vector feature grouping for the inspector
  frontend/             Vite + React + TypeScript (canvas view, no extra UI deps)
    src/components/     GameCanvas, Play, Controls, Inspector, EgoRasterViewer,
                        NetworkVisualizer, Dashboard, ...
    src/*.test.ts(x)    vitest unit tests (api error paths, Play flow, keys, Slider)
```

The backend imports nothing from PyQt — it runs headless on a server. The human-play
score store is `src/data/score_store.py`; the play mode lives in `session.py` (mode `"play"`).
