# T15 mixed Watch roster: Mac serving qualification

Pre-registration, 2026-10-17. Authority:
`docs/research/specs/t15_mixed_roster_qual_spec_2026-10-17.md` and step 2 of
`docs/research/advice/2026-10-17_after_t13.md`. This new sibling reuses the T12
qualification structure and T12B timing procedure. It does not amend their bytes or
re-run the T7 promotion gate. Qualification is separate from release: the checkpoint
default remains b8-s3 and the Watch roster default remains all_b8. Only the explicit
environment selection `SNAKE_SERVE_WATCH_ROSTER=mixed` enables the candidate.
The SHA256 of these exact bytes must be independently pinned in the producer and
audit and committed before any non-smoke episode.

## Identity and execution environment

Hero: registry b8-s3, SHA256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`,
Ego2sServingPolicy, ego2s-bv8, epsilon zero, forward-only, no serving veto.
Every opponent: separately pin-verified registry frp3-s12, SHA256
`eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723`,
vector61, epsilon zero, exact v8 method
`free-space-veto/v8-space-and-head(lambda=8.0)`, head avoidance on and no
reference lambda. Serving scope must be watch_opponents, with IDs 1 through N-1.
Mixed qualification constructs GameSession without an explicit checkpoint argument;
the checkpoint selector is unset or explicitly b8-s3, the roster selector is mixed,
and the three veto override variables are unset. Roster-size overrides change only
the constructor's config object. They do not change checked-in configs.

Use this Mac, project venv, CPU, one torch/BLAS thread per process and at most four
owned episode/identity workers. Ordinary serving and timing retain normal forwards;
identity uses explicit rowwise forwards to avoid batch-shape numeric differences.
Offline probes suppress release-monitor writes and isolate the scores database.
AC power and open lid are required for real compute. Never interact with any tmux
session or signal, stop, pause, resume, or kill a process not started by this task.
Latency runs alone after all owned compute and UI servers stop. Record read-only
process and power/lid observations before and after each latency case; block while
another training/evaluation/Python workload is active. Do not change GC settings.

## Seeds and immutable evidence

Namespace `t15-mixed-roster-20261017/v1`. For purpose and zero-based index i,
allocate `uint32_be(sha256("<namespace>|<purpose>|<i>")[:4])`.
Purposes/counts: watch 25, parity 3, latency 3, ui 1, controls 1 (33 unique seeds).
Watch uses N6/H5000. Parity uses N6/H5000, N8/H5000, N12/H10000 in that order.
Latency uses N6, N8, N12 in that order. Smoke uses a separate smoke namespace,
at most 100 frames per probe, and never qualifies.

Before real gameplay, require a passing freshness audit over the inherited T7
mandatory evidence, final/rejected allocations, historical training exclusions,
T12/T12B/T13/T13b allocations, and visible research/artifact JSON/JSONL evidence.
Record exact scan exclusions and limitations; opaque or unrecorded external work
is outside that claim. Preserve rejected allocations and register a new sibling
allocation if a collision occurs before gameplay; never reseed based on outcomes.

Prepare a create-only intent on a clean tracked tree after implementation, tests and
review. Bind commit, protocol, producer/audit, source/config/UI closure, both checkpoint
bytes and pins, defaults, environment, seed audit and preflight validation. Preserve
the pre-existing untracked saved_snakes symlink. Every result is create-only and binds
intent hash, seed, role/count/horizon, actual work and errors. Worker exceptions become
failed evidence, never a silent retry. A final receipt binds all raw results and UI
evidence. An independent audit checks counts, values, identities and hashes rather
than accepting producer PASS booleans. Any producer/source change after real start
requires a sibling erratum and a new run/seed namespace, preserving prior evidence.

## Required checks

1. **Observation and execution identity.** Run the ego live bv8 and identity tests.
   Compare the three registered real mixed GameSession traces with the grid eval
   engine. Pin the same evaluation world/profile and roster size on both sides.
   For parity only, make the hero terminal to match the evaluator; continuous Watch
   is checked separately below. A harness-local evaluator adapter assigns exact v8
   to opponent slots 1..N-1, because the evaluator's public veto option targets the
   hero. Record that adapter and every lifecycle/config override explicitly. Compare
   hero ego2s and opponent vector61 observations, Q values/actions, and post-frame
   world signatures with exact hashes/counts and first-difference evidence. All
   registered frames and both role comparisons must agree; an unexplained tie or
   missing role observation fails. No evaluator or feature source bytes are changed.
2. **Veto scope.** In the real mixed serving path the hero never has a wrapper or
   counter; every opponent has the exact pinned v8 wrapper and counter. Check at
   construction and throughout all 25 continuous Watch episodes, including respawns.
   Stderr must report active=True scope=watch_opponents and the actual wrapped IDs.
   Registry and wrapper source pins must match. No claim that the hero carries a veto.
3. **Selection, fallback and Play regression.** Exercise a unit and real-session
   matrix covering unset/all_b8/mixed/unknown roster, explicit B8/FRP3/champion,
   unknown checkpoint, veto override combinations, missing/unfilled/wrong FRP3 pin,
   corrupt/unavailable FRP3 bytes, changed wrapper source, mode/rebuild transitions,
   and checkpoint replacement. A mixed opponent refusal must restore all-B8 with a
   visible reason and no partial wrappers. B8 hero refusal preserves the existing
   checkpoint fallback chain. Play remains its existing homogeneous selected policy
   and existing opt-in veto semantics; Train remains refused for B8. Controls use
   the registered control seed and bounded probes up to 100 frames per matrix row.
4. **Latency and enforced speed.** Watch6/8/12 each receives 200 advancing warm-up
   frames and exactly 1500 measured advancing frames. Retain raw wall times for
   GameState.update and complete GameSession.step plus snapshot processing, and
   p50/p95/max. Require update p95 < 66.667 ms in each case. In separate fresh
   sessions of that case's seed, warm 200 frames and collect 120 paced ticks after
   set_speed(12), and separately after set_speed(120). The accepted speed, published
   ceiling and actual total AI count must agree; zero processing times may meet or
   exceed either budget. Initial ceilings remain 24 fps through six total AI and
   12 fps through twelve. The mixed roster counts every AI, not just the B8 policy's
   single row. No default-speed retry. A failed ceiling may be repaired only by a
   separately committed sibling amendment, server-enforced lower mixed ceiling and
   new timing registration; no unregistered retry or merely lower requested speed.
   Report achieved cadence separately: the server sleeps a full interval after its
   work. Network transmission and client rendering are outside the measured tick.
5. **Continuous Watch stability.** Run 25 fresh N6/H5000 episodes with the actual
   all-respawn serving lifecycle, step then snapshot every frame. Require exactly
   125000 advancing frames, no exceptions/last_error, stable mixed checkpoint and
   wrapper identity, and only the hero in B8 observations/dispatch. Retain actual
   hero decisions/deaths, alive frames, per-decision logical-length histogram and
   quantiles, opponent counters, and descriptive deaths per 1000 decisions. These
   are a mixed Watch monitoring baseline, not a new skill or promotion gate.
6. **Live UI.** Use an owned loopback server with isolated score database and the
   registered UI seed. Observe the live Watch UI showing opponent veto counters,
   their opponent-only scope and no hero veto counter. Retain screenshot, visible
   DOM and backend/session identity, plus URL. Check the UI speed controls consume
   the server ceiling. Stop only the server this task started after evidence capture.

## Audit, report and draft erratum

SERVING_PASS requires all six checks. A completed failed criterion is SERVING_FAIL
and HOLD; missing, corrupt or unbound evidence is INVALID. Smoke is SMOKE_ONLY.
Preserve failures; do not change serving defaults regardless of result.

Publish `docs/research/t15_mixed_roster_qualification_2026-10-17.md` with the check
table, raw evidence bindings, tests, latency/cadence/ceilings, baseline, limitations,
expected stderr and proposed one-line future default change:
`WATCH_ROSTER_RELEASED_DEFAULT = "mixed"` (DO NOT APPLY).

Include a clearly DRAFT release erratum: Watch's B8 hero remains unwrapped;
exhibition opponents use frp3-s12 plus v8. T7's claim remains more mass at equal
survival against the served agent, with no hero veto, not safer. This roster
qualification does not isolate opponent checkpoint from wrapper effects or confer
new promotion evidence. Play, checkpoint default, training, and checkpoint bytes
are unchanged. Publish roster-specific monitoring baseline and prospective latency
thresholds; distinguish them from the existing all-B8 release monitor. No monitoring
or default change is authorized by this qualification. Record rollback to all_b8.

Git: branch t15-20261017; bounded commits; after each commit fetch origin, merge
origin/main, push HEAD:main, then push t15-20261017. Never broad-stage unrelated
changes. Final message: check table, latency, proposed default change and final SHA.
