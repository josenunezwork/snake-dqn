# T12: B8 seed 3 extended Mac serving qualification

Pre-registration, 2026-10-17. This is a new sibling of the immutable frp3 protocol.
Authority: `docs/research/specs/t12_b8_serving_qual_spec_2026-10-17.md` and
`docs/research/advice/2026-10-17_release_decision.md`. This job qualifies a path; it does
not release it. `CHECKPOINT_RELEASED_DEFAULT` remains `NAME_FRP3_S12` throughout.
The SHA256 of these exact bytes is pinned independently in `serving_run.py` and
`serving_audit.py` before any non-smoke episode. Changes after a real start require a
sibling erratum, preserved failed evidence, and a new run/seed namespace for producer defects.

## Identity and environment

Candidate: registry `b8-s3`, T7 seed 3 SHA256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`,
`Ego2sServingPolicy`, observation `ego2s-bv8`, epsilon zero, forward-only, no veto.
Every candidate session is constructed as `GameSession()` with no checkpoint argument,
under `SNAKE_SERVE_CHECKPOINT=b8-s3`. The three `SNAKE_SERVE_VETO_*` variables are unset:
Watch hero flag is on by its released default, Play flag off, variant v8. Ordinary served
runs use batched forwards; identity comparisons use explicit rowwise forwards.
The new pin tool binds T7 decision, isolated audit and independent review bytes; it refuses
non-PASS audit, failure to recompute STRICT_PASS, mismatched seed/checkpoint and overwrite.
The candidate file is copied without overwriting to the gitignored saved-snakes location.

Runs use this Mac's project venv and CPU, one torch/BLAS thread per process, at most four
owned qualification worker processes. AC power and open lid are checked before work units.
No unrelated process or tmux session is modified, signaled, paused, resumed or killed.
Latency runs alone after all our other compute stops, and only after a read-only tmux
check establishes that `t11-lanes` has finished (absent, or all its workload panes exited).
If it is present and active, poll; elapsed time never authorizes interruption.

## Seeds, registration and immutable evidence

Namespace `t12-b8-serving-20261017/v1`. For purpose and zero-based index i, the world seed
is `uint32_be(sha256("<namespace>|<purpose>|<i>")[:4])`.
Purposes/counts: watch 25; play 25; parity 2; latency 4; confidence 1; ui 1;
rollback 2; controls 1; p2_h5000 9; p2_h10000 3 (73 globally unique seeds).
A pre-episode freshness audit must pass against every registered/exclusion bank covered by
T7's freshness audit, T7's final and rejected banks, and visible repository/artifact JSON
and JSONL evidence. It binds the T7 audit and mandatory source hashes and records scan
coverage and limitations. It excludes its own newly written registration/run evidence.
No inference about unrecorded external experiments or opaque binaries is made. If seeds
collide before gameplay, preserve the rejected allocation and add a sibling allocation
erratum with a fresh namespace before starting; no outcome-driven reseeding is permitted.
Smoke uses a separate `t12-b8-serving-20261017/smoke-v1` domain, at most 100 frames per
probe and one episode of each kind, and never qualifies.

The real intent is create-only, written on a clean tracked tree after implementation and
review/tests; it records commit, protocol hash, source/config and checkpoint hashes,
strict pin, environment, seed audit hash and exact bank. Every result is create-only and
bound to the intent SHA, seed, candidate SHA, actual counts, errors and elapsed time.
A final receipt inventories result files/hashes. The independent audit verifies those
bytes, completeness and criteria below; PASS booleans alone are not sufficient evidence.
Long workers are polled to completion. A failed worker is preserved; no silent retry.

## Execution and checks (all numbered checks must pass)

1. **Live observation identity.** Run `tests/test_ego_live_bv8.py` and
   `tests/test_ego_live_identity.py`. Repeat P2 no-veto record parity on the exact resolved
   candidate: nine fresh H5000 worlds (three per frozen/scripted/mixed gate roster), then
   three fresh H10000 worlds (one per mix). Reuse the existing P2 record comparison,
   gate profile and roster design, with `ego2s_mask="advisory"`, rowwise forwards and no
   veto. All 12 full records must be identical after the established SIMD-only provenance
   exclusion. Also compare two H5000 served-session traces (frozen then mixed gate roster)
   with `run_simd_eval` on `promotion-v2-watch-rect`: GameSession constructs the candidate
   via env, hero terminal, opponents attached to the same gate roster, rowwise forward,
   advisory mask, no veto. Compare every post-frame cell signature (alive/head/logical
   length and food cells) and require equal trace digests, 5000 frames and zero divergence.
   Gate-roster parity is distinct from the all-B8 served episodes below.
2. **No veto, both guards.** Every Watch/Play candidate is unwrapped. Watch's installation
   is inactive and its reason contains `applies only to a vector61 Apex policy`; stderr
   shows `active=False`. No veto counters/diagnostics or wrappers are active, B8's SHA is
   absent from `v8_gated_checkpoint_sha256s`, and its pin says `veto_variant: none`.
3. **Registry and rollback.** Unit matrix: empty env, champion, frp3-s12, b8-s3, unknown,
   unfilled pin, corrupt candidate bytes, and no-name `VETO_VARIANT=v7`. Invalid B8 falls
   back to verified frp3-s12, otherwise champion, with reason. Empty env remains released
   frp3-s12; v7 no-name remains champion+v7. Live rollbacks construct sessions via
   `SNAKE_SERVE_CHECKPOINT=frp3-s12` and `=champion`, step 100 frames each and capture
   stderr/identity: the expected checkpoint and active v8 hero wrapper; Play unwrapped.
4. **Latency.** Four cases: Watch 6/12 total AI snakes; Play 6/12 total snakes (one human,
   5/11 AI), every AI sharing the pinned B8 policy. Use actual batched serving. Warm 200
   advancing frames outside timing after each construction; measure 1500 advancing frames
   per case, restart Play when human dies (no dead/frozen tails). Time whole
   `GameState.update`, and separately step+snapshot processing. Record raw samples,
   p50/p95/max, roster/identity, restart count, quiet-machine/tmux/power observations.
   Pass requires update p95 < 66.667 ms in each case: at least 20% margin below the default
   12fps budget of 83.333 ms. Record the highest integer UI speed 1..60 whose processing
   budget exceeds the observed full-tick maximum, with a 120-frame validation using the
   server's sleep-after-work cadence; if validation overruns, lower to the new measured
   bound and validate once more. Record failure if that validation still overruns.
   This is a processing-budget ceiling, not a claim of achieved requested fps: the server
   sleeps 1/speed after work; disclose measured/estimated attained cadence. Network/client
   rendering is outside the update measurement. No slider change unless evidence requires
   it; report the ceiling and default margin.
5. **Served roster and monitoring.** Run 25 Watch episodes at H5000 with step then snapshot,
   all-respawn as served, all snakes B8. Require 125000 stepped frames, no last_error,
   correct identity and no wrappers. Record actual hero decisions/deaths and per-decision
   logical length distribution (histogram and quantiles), aggregate deaths/1000 decisions,
   alive frames and final lengths. Rates/lengths are descriptive, never skill criteria.
   Disclose that T7 gated scripted/frozen/mixed rosters, not this all-B8 self-play roster.
6. **Play.** 25 fresh episodes through real control dispatch, seeded frp3/v2 stand-in human,
   each until human death or H5000. Require no last_error, unwrapped B8 AI, no human rows in
   policy observations/dispatch, no AI driving of the human, correct termination/counts.
   Separately exercise play_opponents at 1/5/11, and real `/api/players` + score submission
   and retrieval with an isolated temporary database. Scored rows persist served checkpoint
   name `b8-s3`; old rows remain distinguishable (nullable legacy name). Record endpoint
   responses and stored row. Never modify the user's leaderboard database.
7. **UI and confidence.** Sample exactly 1500 usable inspector frames from a fresh B8 Watch session. Apply the actual UI
   finite-Q relative margin `(best-second)/(best-worst)` (zero for spread <=1e-9) with
   confident >.25, moderate >.10, otherwise close; report counts/percentages, missing and
   nonfinite values. Record attempted/missing dead-hero frames separately (maximum 3000 attempts). Require
   1500 usable Q vectors and at least two buckets populated; recalibrate only if all collapse to one bucket, then preserve the
   first result and register the change before a new qualification sample. Confirm live
   browser rendering of narrator, steering-wheel hub, ego2s-bv8 scalar groups and absence
   of veto counters using a separate owned server and isolated score DB. Save screenshot
   and DOM/visible-check evidence plus the browser URL and candidate backend identity.
8. **Train refusal.** Send Train control for B8; require mode Watch, policy training false,
   and the forward-only/drop-to-Watch message. Train is not qualified for release.

## Decision, reporting and release draft

Audit outcomes: SERVING_PASS if every check passes; SERVING_FAIL for a completed failed
criterion; INVALID for missing/corrupt/unbound evidence. Smoke is SMOKE_ONLY. Any failed
numbered check means HOLD, fix only the serving defect and re-run under a new registered
producer version; T7's gate is not revisited. Do not conceal a failed run.

Publish `docs/research/t12_b8_serving_qualification_2026-10-17.md` with all eight results,
evidence hashes, tests/flake8, latency/cadence, baseline, errors and limitations. Include a
DRAFT release record only, with claim: **more mass at equal survival against the served
agent, with no veto** (not safer). Disclose v8 training labels and live v8 input features,
no serving veto, Tier 2 margin .034 above threshold at SE .023, self-play roster ungated,
late deaths at parity, Train unqualified, changed Play leaderboard comparability.
The prospective one-line release change is
`CHECKPOINT_RELEASED_DEFAULT = "b8-s3"`; DO NOT APPLY IT in this job.
Expected future stderr: `served-checkpoint: name=b8-s3 requested=None sha256=3e649783...`
and `safety-veto-serving: active=False ... reason=the safety veto applies only to a vector61 Apex policy`.
Rollbacks: env frp3-s12 or champion and restart. Monitoring for one week or 100000 hero
Watch decisions, whichever later: rollback on fallback/ego2s exception, deaths/1000 above
2x baseline, or default-speed p95 above 1.5x baseline. Review late-death complaints;
seven-day owner check-in. Record qualification status separately from released state.

Git: work on t12-20261017 in the named worktree, commit in bounded steps; after EACH commit
fetch origin, merge origin/main, push HEAD:main, then push t12-20261017. Preserve unrelated
changes and immutable evidence. Final message: eight-check table, latency numbers,
monitoring baseline, any FAIL and cause, final commit SHA. Released default unchanged.
