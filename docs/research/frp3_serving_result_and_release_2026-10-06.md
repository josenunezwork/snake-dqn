# FRP-v3 s12 checkpoint swap: web serving qualification and release (2026-10-06)

**Outcome: `SERVING_PASS`** (`serving_path_qualified = true`, audit criteria S1-S6 all pass).
Released on local main as a checkpoint swap: the served default is now
**frp3-s12 + v8** (it was champion_a5 + v8).

## Evidence chain

| Step | Result | Evidence |
|---|---|---|
| Strict gate (Tier 2) | `STRICT_PASS`, stop look 2, n = 207/mix | `frp3-m3s12-strict-20261005/run-v1`, receipt `5107f3fc…acc5b3`; see `frp3_strict_result_2026-10-06.md` |
| Strict pin | filled with `fill_strict_pin.py` (commit `9318547`) | `web/backend/served_checkpoint_pins.json` |
| Web serving qualification | `SERVING_PASS` | `frp3-s12-serving-20261006/run-v1` |

Serving run identity: protocol `research/frp3_checkpoint_serving_20261005/protocol.md` (sha
`91634617…727c6`, pinned), code commit `9318547` (clean tree), intent sha `b10de533…2ff5`,
receipt sha `4a05cbb0…0c04`, audit report `audit/audit.json` sha `4bf7a616…0dc7`.
Run on the Mac: one shared CPU slot, 1 thread, AC power and lid open, inside tmux with
caffeinate. Wall time 1607 s; 144,994 frames; 0 failed episodes.

## What the run measured (serving correctness, not skill)

- **Watch (25 x 5000 frames)**: the release environment (`SNAKE_SERVE_CHECKPOINT=frp3-s12`)
  served the pinned bytes (`eec144bf…3723`) with the v8 wrapper on the hero only; 125,000
  decisions, 305 vetoes applied (rate 0.00244), 88 head-risk vetoes, 60 v7 re-rank changes,
  18 hero deaths (Watch respawns), mean wrapper cost 0.61 ms per decision.
- **Play (25 episodes)**: unwrapped; all 25 ended by the stand-in human's death (19,994 frames).
- **Parity (2 probes x 5000 frames)**: session vs SIMD rollout of the same identity, no
  divergence, equal traces and counters; label `rerank-exercised` (20 vetoes, 9 re-rank
  changes, 0 head-risk vetoes in the probes).
- **S6**: before the release flip, the empty environment served champion + v8; unknown names
  and an unfilled pin fell back to the champion with a reason; frp3-s12 under `v7` or
  `SNAKE_SERVE_VETO_WATCH_HERO=0` was served unwrapped.

## Release (checkpoint swap)

- `web/backend/served_checkpoint.py`: `CHECKPOINT_RELEASED_DEFAULT = "frp3-s12"`. It stays
  pin-gated: if the pin or the bytes do not verify, the champion is served with a reason.
- The checkpoint is copied (gitignored) to `saved_snakes/frp3_m3_s12_u60000_20261005.pth`;
  the artifact path is the fallback.
- The veto is unchanged: v8 (lambda 8, head layer on).
- **Rollback to the champion:** restart with `SNAKE_SERVE_CHECKPOINT=champion` (no code change).
- **Veto rollback keeps its meaning:** when no checkpoint is named,
  `SNAKE_SERVE_VETO_VARIANT=v7` (or `v5` / `v2`) now serves the **champion** with that
  variant, which is the pre-swap configuration those receipts bind. The frp3-s12 default was
  gated with v8 only. This is a release-step addition the qualification protocol did not
  cover. With `SNAKE_SERVE_CHECKPOINT=frp3-s12` named explicitly, an older variant still
  serves frp3-s12 unwrapped (as S6 checked). `SNAKE_SERVE_VETO_WATCH_HERO=0` serves frp3-s12
  unwrapped.
- Expected stderr on a default start: `served-checkpoint: name=frp3-s12 requested=None
  sha256=eec144bf…3723 path=…/saved_snakes/frp3_m3_s12_u60000_20261005.pth reason=None` and
  `safety-veto-serving: active=True scope=watch_hero … strict_checkpoint_match=True … variant=v8`.

## UI follow-up shipped with the release

The checkpoint's Q-values are about 4x smaller than the champion's. The fixed confidence
cutoffs in the steering-wheel hub and the narrator (`confident > 1`, `moderate > 0.3`) would
label about 97% of its decisions "close call". The cutoffs now apply to the relative margin
(best - second) / (best - worst), using 0.25 and 0.10. These values were calibrated on 1500
champion Watch frames to reproduce the old split (14.6 / 43.9 / 41.5% against the old
13.9 / 44.0 / 42.1%); FRP-v3 gives 21 / 45 / 34% (commit `9e99c72`).

## Non-claims

- The serving run makes no skill claim; the skill evidence is the strict gate.
- The strict gate passed only under survival band v2. Under the old per-mix 0.05 band, the
  scripted survival lower bound (-0.060) would have failed (disclosed in the strict result).
- Train mode from a frp3-s12 session is not qualified.
