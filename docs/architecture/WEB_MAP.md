# Web app map (`web/`)

Read-only map of the server-authoritative web app, written 2026-10-20 from the files in
`web/`. The Python engine owns all game state and inference; the React frontend is a
pure view over one WebSocket. Quick-start: [`web/README.md`](../../web/README.md).

## How to run

```bash
cd web/frontend && npm install && npm run build && cd ../..   # build UI once
./venv/bin/python web/serve.py              # API + WS + built UI on http://127.0.0.1:8000
PORT=9000 ./venv/bin/python web/serve.py    # other port
# dev: backend as above, then
cd web/frontend && npm run dev              # :5173, proxies /api and /ws to :8000 (vite.config.ts)
cd web/frontend && npm test                 # vitest
```

`serve.py` binds `127.0.0.1`, pins `torch.set_num_threads(1)`, and runs
`web.backend.app:app` with uvicorn (`log_level="warning"`). If `web/frontend/dist`
is missing, `GET /` returns a JSON hint instead of the UI.

## Backend modules (`web/backend/`)

| Module | Role |
|---|---|
| [`app.py`](../../web/backend/app.py) | FastAPI app: hub with the single shared `GameSession`, broadcast engine loop, REST routes, `/ws/stream`, control dispatch table `_CONTROL_HANDLERS`, static serve of `frontend/dist`. |
| [`session.py`](../../web/backend/session.py) | `GameSession`: owns one `GameState` + selected policy; modes watch / play / train; resolves served checkpoint and Watch roster; installs vetoes, Path B hook, speed ceiling. |
| [`serialize.py`](../../web/backend/serialize.py) | `GameState` -> frame DTO (`type: "frame"`); the wire contract. `PROTOCOL_VERSION = 2`. |
| [`served_checkpoint.py`](../../web/backend/served_checkpoint.py) | Pinned registry (`champion`, `frp3-s12`, `b8-s3`) and fail-closed resolution from `SNAKE_SERVE_CHECKPOINT`; Watch roster constants. |
| [`served_checkpoint_pins.json`](../../web/backend/served_checkpoint_pins.json) | Strict-gate receipt pins (sha256 + evidence paths) authorizing non-champion checkpoints. |
| [`safety_veto_serving.py`](../../web/backend/safety_veto_serving.py) | Free-space safety veto wrappers (v2/v5/v7/v8) for vector61 policies, each bound fail closed to its strict receipt hashes. |
| [`ego2s_policy.py`](../../web/backend/ego2s_policy.py) | `Ego2sServingPolicy`: forward-only serving of `ego2s-*` students (b8-s3 is served through this path, per `session.py`). |
| [`raster_policy.py`](../../web/backend/raster_policy.py) | `RasterServingPolicy`: serves `raster31v2` checkpoints with the training featurizer. |
| [`pathb.py`](../../web/backend/pathb.py), [`pathb_controller.py`](../../web/backend/pathb_controller.py) | T34 Path B staged assistance, default OFF; only for the released mixed Watch6 b8-s3 hero on macOS. |
| [`serving_speed.py`](../../web/backend/serving_speed.py) | Frame-rate ceilings for expensive policies (`DEFAULT_SERVER_SPEED_CEILING = 120.0`). |
| [`release_monitoring.py`](../../web/backend/release_monitoring.py) | JSONL telemetry (prefix `b8-release-monitor:`) to stderr for the B8 release window. |
| [`checkpoints.py`](../../web/backend/checkpoints.py) | Containment-safe catalog of loadable checkpoints (`saved_snakes/*.pth` etc.). |
| [`metrics.py`](../../web/backend/metrics.py) | Dashboard data: checkpoint inventory + `logs/eval_*.json` (legacy and repaired-gate schemas). |
| [`state_labels.py`](../../web/backend/state_labels.py) | Vector feature grouping for the Inspector. |

## REST

`GET /api/health`, `/api/checkpoints`, `/api/metrics`, `/api/state`,
`/api/leaderboard?limit=`, `/api/scores/recent?limit=`, `/api/players/{name}`;
`POST /api/scores` `{name}` (server-authoritative; scores in `scores.db`, path override
`SNAKE_SCORES_DB`).

## WebSocket protocol (`/ws/stream`)

Server -> client:

- on connect: one snapshot frame; then broadcast `{"type": "frame", ...}` (from `serialize.py`, protocol v2).
- after each control: `{"type": "ack", "state": <session.control_state()>}`.
- `{"type": "info"|"error", "message": ...}` replies (e.g. save_weights, foreign play run, loop errors).

Client -> server: `{"type": "control", "action": <name>, "value": <any>}`. Actions
(`app.py` `_CONTROL_HANDLERS`): `play`, `pause`, `reset`, `new_game`, `set_speed`,
`set_mode` (string or `{mode, override_reward_contract}`), `set_epsilon`, `set_hero`,
`set_food`, `load_checkpoint` (string or `{name, override_reward_contract}`),
`human_input`, `human_boost`, `set_play_opponents`, `save_weights`,
`set_raster_stream` (per-connection opt-in). While another connection owns a live
scored Play run, `set_mode`/`reset`/`new_game`/`load_checkpoint` are refused and
steering is ignored. Unknown/malformed controls set `session.last_error`, never raise.

## Choosing the served agent (env vars, read per session build; restart to apply)

| Env var | Default | Meaning |
|---|---|---|
| `SNAKE_SERVE_CHECKPOINT` | `b8-s3` (released 2026-10-17) | Registry name: `b8-s3`, `frp3-s12`, `champion`. Unknown name -> default with a `reason`. Non-champion names need a filled pin and matching sha256; b8-s3 failing falls back to verified frp3-s12, then champion. Rollback: `=frp3-s12` or `=champion`. |
| `SNAKE_SERVE_VETO_VARIANT` | `v8` | Watch-hero veto wrapper `v2`/`v5`/`v7`/`v8` (vector61 only). b8-s3 pin has `veto_variant: none` -> served bare. With no checkpoint named, `v7`/`v5`/`v2` serve the champion (pre-swap rollback meaning). |
| `SNAKE_SERVE_VETO_WATCH_HERO` | ON | Master switch for the Watch-hero veto; `0/false/no/off` disables. |
| `SNAKE_SERVE_VETO_PLAY_AI` | OFF | `1/true/yes/on` wraps Play AI snakes with v2. |
| `SNAKE_SERVE_WATCH_ROSTER` | `mixed` | `mixed` (frp3-s12 opponent) or `all_b8` (rollback). |
| `SNAKE_SERVE_PATHB` | OFF | Path B assistance, only when eligible (see `pathb.eligible`). |
| `SNAKE_MECHANICS_V2` | ON | `0/false/no` serves legacy v1 mechanics. |
| `SNAKE_SCORES_DB` | `DEFAULT_SCORES_DB` | Play leaderboard SQLite path. |
| `PORT` | `8000` | `serve.py` port. |

The client's `load_checkpoint` control and an explicit `GameSession(checkpoint=...)`
bypass the registry. Resolution logs one line with prefix `served-checkpoint:`.

## Frontend (`web/frontend/`, Vite + React + TS)

- `src/App.tsx`: arena always visible; tabs Controls, Play, Inspector, Raster, Network,
  Dashboard (persisted in `localStorage` `snake_tab` and URL hash). Raster tab subscribes via `set_raster_stream`.
- `src/hooks/useGameSocket.ts`: WebSocket client (handles `frame`, `ack`, `info`/`error`); `useViewSettings.ts`: view prefs.
- `src/api.ts`: REST client; `keys.ts`: key bindings; `identity.ts`: player identity; `types.ts`: wire types.
- `src/components/`: `GameCanvas` (rAF renderer), `Play`, `Controls`, `Inspector`,
  `EgoRasterViewer`, `NetworkVisualizer`, `Dashboard`, telemetry/explanation widgets
  (`TelemetryBand`, `Sparkline`, `SectorRadar`, `SteeringWheel`, `DecisionTrace`,
  `DecisionNarrator`, `DuelingSplit`, `EpisodeSummary`), UX chrome (`CommandPalette`,
  `HelpOverlay`, `WelcomeTour`, `ViewMenu`, `ArenaLegend`, `Toast`, `Confetti`, `InfoDot`, `Slider`).
- `src/lib/`: `narrate`, `decisionMargin`, `runSummary`, `fuzzy`, `glossary`, `presets`.
- Tests: `*.test.ts(x)` beside sources (vitest).
