---
name: snake-dqn-serving-release
description: |
  Use whenever you touch what the snake-dqn web app serves in /Users/josenunez/Projects/ml/snake-dqn:
  web/backend/served_checkpoint.py, served_checkpoint_pins.json, safety_veto_serving.py, Watch/Play
  roster selection, SNAKE_SERVE_* environment variables, scripts/monitor_b8_release.py, a serving
  qualification (T12/T15 pattern), a release or rollback of the served default, or when answering "what is
  served right now and how do I roll it back". It records the registry, released defaults, rollback
  switches, monitoring thresholds and what a release requires.
---

# Snake DQN: serving registry, release and rollback

Current truth: `docs/STATUS.md` "Released agent". Release records:
`docs/research/b8_s3_release_2026-10-17.md` (+ `b8_s3_monitoring_erratum_2026-10-17.md`,
`b8_s3_release_erratum_mixed_watch_2026-10-17.md`), earlier
`docs/research/frp3_serving_result_and_release_2026-10-06.md`. Hash-bound records are never edited; add a
sibling erratum.

## 1. What is served (as of 2026-10-17)

| Role | Name | Checkpoint | SHA-256 | Policy / veto |
|---|---|---|---|---|
| Watch hero (slot 0), Play opponents' default checkpoint | `b8-s3` | `saved_snakes/b8_s3_t7_20261016.pth` (gitignored) or `snake-dqn-artifacts/redesign-m4b-mac-20261013/B8_s3/final.pth` | `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8` | `Ego2sServingPolicy`, obs `ego2s-bv8`, **no veto** |
| Watch opponents (slots 1..N-1, mixed roster) | `frp3-s12` | `saved_snakes/frp3_m3_s12_u60000_20261005.pth` | `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723` | vector61 + **v8 veto** (`free-space-veto/v8-space-and-head(lambda=8.0)`) |
| Legacy rollback | `champion` | `saved_snakes/champion_a5_freespace_20260621.pth` | `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` | vector61 + v8 (or v7/v5/v2 by variant) |

- `CHECKPOINT_RELEASED_DEFAULT = NAME_B8_S3`; `WATCH_ROSTER_RELEASED_DEFAULT = WATCH_ROSTER_MIXED`
  (`web/backend/served_checkpoint.py`). Watch default = bare B8 hero vs frp3-s12+v8 opponents
  (owner-approved after T15 `SERVING_PASS`, T15B spec).
- Release claim, verbatim: **"more mass at equal survival against the served agent, with no veto."** Not
  "safer". Late-game deaths are at parity with frp3-s12+v8. Tier 2 margin was 0.034 at SE 0.023.
- B8 was trained with v8 labels and consumes live v8-derived input features (`LiveV8Source`), yet is
  served without a wrapper; the inspector's "v8 veto features (5 per action)" group is inputs, not a veto.
  The B8 hero is never in `wrapped_ids` and has no veto counter; opponent counters are labelled as such.
- Train mode is **refused** for ego2s (drops to Watch, "served forward-only"). Leaderboard rows before and
  after 2026-10-17 are not comparable (heavier Play opponents); scored runs record the served name.

## 2. The registry (`web/backend/served_checkpoint.py`)

- `REGISTRY` has exactly three `PinnedCheckpoint`s: `champion` (`requires_strict_pin=False`), `frp3-s12`
  (pin files `receipt`, `intent`, `audit_report`; veto v8), `b8-s3` (pin files `decision`,
  `audit_report`, `independent_review`; `veto_variant="none"`).
- `resolve_served_checkpoint(environ, saved_dir, pins, champion_path)` **never raises**: unknown name ->
  released default with a `reason`; pin invalid or bytes mismatch -> fail closed down the chain
  b8-s3 -> verified frp3-s12 -> champion (`_fallback_after_refusal`), always with a recorded reason.
  Candidate paths: `saved_snakes/<filename>` first, then the pinned artifact path; bytes are re-hashed.
- `strict_pin_problems(name)` validates `served_checkpoint_pins.json` (schema
  `served-checkpoint-strict-pins/v1`): checkpoint sha, veto variant/method, `strict_receipt.verdict ==
  "STRICT_PASS"`, absolute `root`, relative `path` + 64-hex sha for each required file. A `null`
  receipt is the shipped placeholder and refuses the name.
- `v8_gated_checkpoint_sha256s()` extends the v8 serving veto's checkpoint binding to pinned v8 entries
  (frp3-s12), so `safety_veto_serving.ServingVetoState.checkpoint_match` wraps it; B8 is absent by design.
- `resolve_mixed_watch_opponent()` reuses the same strict-pin + byte checks for the mixed roster; any
  opponent refusal restores **all_b8** with a visible reason, never a partial mixed arena.
- Env is read on every `GameSession()` construction; a session keeps one checkpoint for its lifetime.
  Explicit `GameSession(checkpoint=...)` and the client's `load_checkpoint` bypass the registry.
- One INFO line per resolution to stderr: `served-checkpoint: name=... requested=... sha256=... path=...
  reason=...` and `served-checkpoint: watch_roster=... hero=... opponent=... opponent_pins=... reason=...`;
  the veto layer logs `safety-veto-serving: active=... scope=... wrapped_ids=[...] ... reason=...`.
  Expected lines for default/all_b8/rollbacks are recorded in the release docs; compare against them.
- Tests: `tests/test_web_served_checkpoint.py`, `tests/test_web_watch_roster.py`,
  `tests/test_apex_veto_serving_audit.py`. Historical T13/T13b tests pin `all_b8` explicitly.

## 3. Environment switches (restart the server; no code change)

| Variable | Values | Meaning |
|---|---|---|
| `SNAKE_SERVE_CHECKPOINT` | unset/blank -> `b8-s3`; `frp3-s12`; `champion`; other -> default + reason | which registry checkpoint the default session serves |
| `SNAKE_SERVE_WATCH_ROSTER` | unset/blank -> `mixed`; `all_b8`; invalid -> mixed + reason | Watch exhibition roster |
| `SNAKE_SERVE_VETO_VARIANT` | unset -> `v8`; `v7`/`v5`/`v2` | Watch-hero veto variant for vector61 policies. With **no checkpoint named**, a pre-swap variant makes the default yield to **champion + that variant** (`PRE_SWAP_VETO_VARIANTS`); explicit `frp3-s12` with v7/v5/v2 is served **unwrapped** |
| `SNAKE_SERVE_VETO_WATCH_HERO` | unset -> on; `0/false/no/off` -> off | wrapper-off switch (vector61 hero only; B8 reports `active=False`, reason "applies only to a vector61 Apex policy") |
| `SNAKE_SERVE_VETO_PLAY_AI` | unset -> off; `1` -> on | wrap Play AI (v2 behaviour) |

Rollbacks:
```bash
SNAKE_SERVE_CHECKPOINT=frp3-s12 ./venv/bin/python web/serve.py     # frp3-s12 + v8 (previous release)
SNAKE_SERVE_CHECKPOINT=champion ./venv/bin/python web/serve.py     # champion + v8
SNAKE_SERVE_WATCH_ROSTER=all_b8 ./venv/bin/python web/serve.py     # keep B8, all-B8 Watch roster
```
Rollback triggers: any served fallback reason or ego2s exception; roster-specific hero deaths/1k above
the table below; default-speed update p95 > 1.5x the matched qualification value.

## 4. Monitoring (`scripts/monitor_b8_release.py`)

```bash
mkdir -p logs
./venv/bin/python web/serve.py 2>> logs/b8-release.stderr.log       # retain stderr across restarts
./venv/bin/python scripts/monitor_b8_release.py logs/b8-release.stderr.log
```
Reads `b8-release-monitor:` JSON records; exit 0 = measured metrics within bounds, exit 2 = trigger or
incomplete/invalid telemetry (a short startup smoke returns 2 for missing warmed latency samples; that is
correct, not a failure). It never stops/restarts servers. Baselines are per roster mode and size and
are never pooled or borrowed:

| Watch roster | Driven snakes | Deaths/1k baseline | Roll back when > | Update p95 baseline (ms) / trigger |
|---|---:|---:|---:|---|
| mixed | 6 | 0.272 (T15) | 0.544 | 7.792 / 11.688 |
| mixed | 8 | none -> `NOT_HEALTHY` | - | 9.519 / 14.279 |
| mixed | 12 | none -> `NOT_HEALTHY` | - | 13.141 / 19.712 |
| all_b8 | 6 | 0.424 (T12) | 0.848 | T12B baselines |
| all_b8 | 8 | 0.496 (T13b) | 0.992 | no latency baseline |
| all_b8 | 12 | 0.768 (T13) | 1.536 | T12B baselines |

Equality does not trigger. Latency excludes the first 200 updates after build/rebuild/reset and uses
eligible 12 fps samples; deaths and decisions count at every speed. Observation window per roster
mode+size: >= 7 days from first retained valid tick AND 100,000 actual hero decisions, whichever ends
later. Owner check-in **2026-10-24**. Review late-death complaints without automatic rollback.

Speed ceilings (processing budgets, not cadence promises): <= 6 driven AI -> 24 fps; 7-12 -> 12 fps;
larger -> `max(1, floor(144/n))`. Qualified only for Watch6/12, Play6/12 (T12B) and mixed Watch6/8/12
latency (T15).

## 5. What a release requires (the chain that produced B8 s3)

1. **Skill evidence**: Tier 1 `STRICT_PASS` and Tier 2 PASS on fresh banks with independent audit and
   review (T7; `research/redesign_t7_phase_r_20261016/results/decision.json`, hashes in the release doc).
   For vector61 changes: the sequential strict gate `STRICT_PASS` (FRP-v3 pattern).
2. **Registry entry + pin**: add the `PinnedCheckpoint`, gitignored copy under `saved_snakes/`, artifact
   path; fill `served_checkpoint_pins.json` only with the study's `fill_strict_pin.py`
   (`research/t12_b8_serving_20261017/`, `research/frp3_checkpoint_serving_20261005/`) after
   decision + audit + review verify. Until filled, the name is refused (fail closed).
3. **Mac serving qualification**, pre-registered protocol with its sha pinned in the harness, run with the
   candidate **selected by env**, default unchanged. B8's eight checks
   (`docs/research/t12_b8_serving_qualification_2026-10-17.md`): (1) live-observation identity vs SIMD
   (9/9 H5000, 3/3 H10000, parity probes with no differing frame), (2) veto absent and says so,
   (3) registry/pin/rollback matrix incl. corrupt-bytes and unfilled-pin fallbacks and 100-frame live
   rollbacks, (4) latency and UI-speed ceilings (zero overruns at default and enforced ceilings; failed
   once, repaired by T12B check-4-only amendment), (5) 25 Watch H5000 episodes -> monitoring baseline,
   (6) 25 real-control Play episodes incl. score API, (7) UI/confidence sample, (8) Train refusal.
   Independent audit must return `SERVING_PASS`. A roster change needs its own qualification (T15: six
   checks, identity, opponent-only veto scope, fallbacks, latency, continuous Watch6 baseline).
4. **Owner authorization** (explicit; advisor memo `docs/research/advice/2026-10-17_release_decision.md`,
   spec `docs/research/specs/t12c_release_spec_2026-10-17.md`). A Phase R label or SERVING_PASS does not
   itself release anything.
5. **Flip + verify + record**: change only `CHECKPOINT_RELEASED_DEFAULT` / `WATCH_ROSTER_RELEASED_DEFAULT`
   (no policy math, obs-spec, engine, veto flag or model-byte changes); start default and every rollback
   configuration live with `SNAKE_SERVE_*` cleared (except the explicit selector under test), owned
   loopback port, temp scores DB, CPU one BLAS thread; record exact stderr identity lines, 8 WebSocket
   frames, health/state endpoints; stop and reap every server you started; never touch other tmux
   sessions/processes; preserve the `saved_snakes` symlink.
6. **Tests + publication**: full local suite `./venv/bin/python -m pytest -q` (incl. slow, one BLAS
   thread), focused serving/monitor tests, flake8/black/isort, `git diff --check`; frontend tests/build
   only if frontend files changed; push to `main` and the task branch; GitHub CI green on the published
   SHA. Write the release record with disclosures, monitoring thresholds, rollback commands and evidence
   manifest (`research/t12c_release_20261017/results/`, `research/t15b_watch_default_20261017/results/`);
   update `docs/STATUS.md` and `docs/research/INDEX.md`.

## 6. Do not

- Edit `b8_s3_release_2026-10-17.md`, T12/T15 reports, protocols or pins' bound evidence; add errata.
- Fill a pin by hand, point it at a non-`STRICT_PASS` verdict, or relax `strict_pin_problems`.
- Call B8 "safer", "veto-wrapped", or describe all-B8/Play rosters as gated (they are ungated monitoring
  baselines).
- Pool monitoring across roster modes/sizes, or reuse the 6-snake 0.848 trigger for other sizes.
- Serve Train for ego2s, change speed clamps, or alter `Ego2sServingPolicy` forward paths without a fresh
  qualification (T12 "fresh-qualification boundary for any repair").
- Leave a server, tmux session or monitoring job running after a verification task unless the owner asked.

## Before you say done
- Report the exact `served-checkpoint:` / `safety-veto-serving:` lines observed for every configuration
  you started, their PIDs and that they were stopped.
- State which env var restores the previous behaviour and confirm it was exercised live.
- If anything about the served path changed, name the qualification that covers it or say it is unqualified.
