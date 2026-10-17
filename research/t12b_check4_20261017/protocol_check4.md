# T12B amendment: Check 4 latency and serving-speed ceiling

This pinned, check-4-only amendment follows the failed T12 B8 serving
qualification recorded in `docs/research/t12_b8_serving_qualification_2026-10-17.md`
and the repair authority in
`docs/research/specs/t12b_check4_fix_spec_2026-10-17.md`. Checks 1--3 and
5--8 remain PASS only through the prior, hash-bound T12 audit; they are not
re-run here. This amendment does not alter `CHECKPOINT_RELEASED_DEFAULT`.

## Identity, scope, and immutable binding

The candidate is registry `b8-s3`, SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`,
served with `SNAKE_SERVE_CHECKPOINT=b8-s3`. The measured path is the live
`GameSession` path: all AI use the registry-selected `Ego2sServingPolicy` and
its existing batched serving forward. This amendment must not change policy
math, observation math, batching, action selection, or the forward path.

The preserved diagnostic of the failed Watch-12 attempts found rare GC pauses
(27.8 ms and 38.9 ms among the retained tails) rather than a steady policy
cost. The repair is therefore the roster-driven server scheduling ceiling; it
does not tune process-global GC or move work out of the measured tick.

Before any measurement, `run.py prepare` creates a fresh create-only intent on
a clean tracked tree. It binds these exact protocol bytes, all producer source
hashes, candidate bytes and selection, the unchanged released default, the
fresh seed-audit bytes, preflight-validation bytes, and the prior T12 audit and
receipt. Each result binds the intent digest and retains raw samples. The
amendment SHA is independently pinned in both producer and audit. A final
receipt binds every result file. Existing paths are never overwritten.

## Fresh seeds

Namespace: `t12b-check4-20261017/v1`. The only registered purpose is
`latency`, with four ordered seeds generated as
`uint32_be(sha256("<namespace>|latency|<i>")[:4])`, i=0..3. The seed audit
must reject collisions with the original T12 73-seed allocation, its actual
latency seeds, the T12B diagnostic allocation, and the registered/excluded
coverage inherited from the prior audit. It may not exclude broad new-artifact
areas to hide a collision.

## Quiet-Mac precondition

The real measurement runs only on a quiet Mac, on AC power with an open lid.
Before and after each case, the producer records a read-only process snapshot
and every tmux pane. It blocks if another training or evaluation workload is
present. It never signals, stops, changes, or attaches to a foreign process or
tmux session. The qualification process itself is the only permitted workload.

## Check 4 procedure and pass rule

Cases, in seed order, are Watch-6, Watch-12, Play-6, and Play-12. Watch has
6/12 driven AI. Play has one human and 5/11 driven AI. Every AI shares the
pinned B8 serving policy. For each construction and after every Play reset,
advance 200 live frames outside timing. Then retain exactly 1,500 advancing
measurement samples of both (a) wall time inside `GameState.update` and (b)
the complete server processing order `GameSession.step(); GameSession.snapshot()`.
The latter deliberately excludes websocket transmission and client rendering.
Record p50/p95/max and raw samples. A terminal human frame is a valid sample;
restart and warm the new Play run before its next sample.

For every case, pass requires update p95 below 66.667 ms. Also run exactly
120 paced live ticks at the default speed: call `session.set_speed(12)` and
record the actual accepted speed; zero ticks may meet or exceed its processing
budget. Do not retry this default-speed probe.

For the server-ceiling probe, call `session.set_speed(120)` in a new live
session. Record the actual accepted (clamped) speed and ceiling exposed by the
server, then run 120 paced ticks with its sleep-after-work cadence. Zero
overruns are required. If this ceiling attempt fails, preserve that failed
attempt. At most one retry is permitted only after a separately declared,
server-enforced lower ceiling is in effect; it must be strictly lower than the
first actual ceiling, and a fresh `set_speed(120)` must prove the new actual
clamp before its 120 ticks. Merely requesting a lower speed does not validate a
higher server ceiling. No default-speed retry is permitted.

The initial production ceiling is determined only by the existing serving
function and the actual driven-AI count: 24 fps for at most six driven AI, 12
fps for seven through twelve, and `max(1, floor(144/n))` above twelve. The
recorded ceiling is a server processing ceiling, not a claim that requested
cadence was attained; record actual cadence separately.

`SERVING_PASS` for this amendment requires all four cases to satisfy all three
conditions: update p95 threshold, zero default-12 overruns, and zero
server-ceiling overruns. Missing, stale, unbound, or malformed evidence is
`INVALID`; a completed failed condition is `SERVING_FAIL`/HOLD. Persist each
completed case immediately so a later failure cannot erase its evidence.

## Reporting boundary

The report retains the original Watch-12 failed 24/17 fps attempts and adds
this amendment's diagnosis, server-side repair, actual clamped ceilings, and
new Check-4 verdict. It says explicitly that the protocol measures live server
step-and-snapshot processing and excludes network/client rendering. It retains
the release-default prohibition.
