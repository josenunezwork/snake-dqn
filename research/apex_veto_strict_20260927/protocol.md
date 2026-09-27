# Apex + free-space veto: Tier-2 strict challenge (pre-registration)

Status: **pre-registered Tier-2 design, written 2026-09-27 before any calibration or
final episode.** Harness: `strict_run.py` beside this file. Governance:
[`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md)
(Tier 2). A `STRICT_PASS` produces a receipt only. It changes no champion file, default,
config or deployment. Release is a separate, explicit action.

Revision before `prepare` (no intent, calibration or final data exists yet): after review,
the final wall cap rose from 30000 s to 45000 s on the measured screen timings (see
"Runtime projection"); the run parent now holds both CPU slot locks for the whole run; the
smoke and Tier-2 output roots are separated; the serving stage is named for what it is; the
independent audit binds the candidate identity to the source; and a non-claim covers
single-hero wrapping. No statistical design element changed.

## Question and decision it informs

Does serving the incumbent Apex checkpoint behind the opt-in free-space veto improve the
H5000 mass integral against the unwrapped incumbent, strictly (Holm superiority in >= 2
of 3 mixes plus scripted non-inferiority plus behavioral bands), on fresh worlds? The
answer decides whether a later release action may consider deploying the wrapper.

## Arms and identities

| Role | Checkpoint | Wrapper |
|---|---|---|
| Incumbent | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c53919dd59416c145cf0ba7c4faf1c7f298eebbb1723146807d747ac93` | none |
| Candidate | the same bytes | `free-space-veto/v2-speed-preserving` (`src/evaluation/safety_veto.py`, hero only) |

The candidate identity is the triple {checkpoint sha256, wrapper method and descriptor,
sha256 of `src/evaluation/safety_veto.py`}. `prepare` binds all three into `intent.json`.
Every candidate record must carry `probes.safety_veto` with that exact descriptor, and no
incumbent record may carry it. The shape check is `validate_strict_world_record(...,
candidate_wrapper=...)` in `src/evaluation/strict_promotion.py`; the default path of that
validator is unchanged. The independent audit binds the triple to the frozen source on its
own: `intent.candidate.wrapper_source_sha256` must equal the source-closure hash of
`src/evaluation/safety_veto.py`, and every candidate record must carry that sha256 and a
`probes.safety_veto` equal (without `counters`) to the intent's wrapper descriptor.

## Profile, mixes and rosters

- Profile `promotion-v2-watch-rect`, H5000, digest `d396d3ed…0e8b`. Pinned config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3…15aa5`).
- Mixes frozen, scripted and mixed. Rosters are built by `dev_screen._design_rows`, which is
  `strict_promotion.materialize_rosters` with the strict pilot's checkpoint pool order and
  mixed-slot rules. Before writing the intent, `prepare` rebuilds the strict pilot's 48
  development rosters and requires identity parity (`dev_screen.roster_parity_report`).
- Serving stage roster: the web Watch deployment of a vector61 checkpoint runs one shared
  policy for every snake, so serving episodes use a self-play roster (five incumbent
  opponents without the wrapper) under `mix_id` `serving-selfplay`, hero with the wrapper.

## Worlds (all new namespaces)

Seeds are `uint32_be(sha256("<domain>|worlds|<i>")[:4])`, the `dev_screen` recipe.

| Namespace (domain) | Count | Use |
|---|---|---|
| `apex-veto-strict-dev-v1` | 16 | incumbent-only calibration, 3 mixes |
| `apex-veto-strict-final-v1` | ordered bank, Nmax 300 | final paired worlds; the first N are played |
| `apex-veto-strict-serving-v1` | 50 | serving-compatibility episodes |

At construction `prepare` fails closed unless the seeds are unique within and across the
three namespaces and disjoint from every set `dev_screen.disjointness_report` checks (the
five task-aligned challenger namespaces, the strict pilot's observed development and pilot
seeds, small integers 0..999) and from the first 1000 seeds of `apex-safety-screen-v1`
(the Tier-1 screen). World identity includes the seed, so seed disjointness implies
world-identity disjointness. The namespace registry proposed by the governance draft is
not implemented; this in-code check is its stand-in (disclosed gap).

## Calibration stage (incumbent only, before any final data)

The incumbent plays the 16 dev worlds x 3 mixes (48 episodes). From those records:

- reference means of `mass_integral` and `survival_fraction` per mix;
- absolute non-inferiority margin `delta_NI = 0.03 x (dev incumbent scripted mean mass)`,
  the E2 initial proposal; the run stops `INVALID_STOP` if it is not positive;
- behavioral bands on candidate final records: per mix, the candidate's mean
  `survival_fraction` must lie in `[dev_mean - 0.02, dev_mean + 1]`.

`calibration.json` is written create-only before the final stage starts.

## Sizing (frozen in the intent)

- MDE = 20 mass-integral units absolute per mix. This is a declared product threshold.
  **Disclosure:** it was declared after the Tier-1 screen
  (`apex-safety-screen-20260926/run-v1`) was seen: B-A means +43.3 frozen, +55.1
  scripted, +47.7 mixed, one-sided Holm rejection in 3/3 mixes. That screen stopped at its
  deadline during the determinism control (decision `INCOMPLETE`), but all 40 A/B pairs per
  mix completed.
- The screen's paired B-A mass deltas are the independent pilot (disjoint worlds, E2
  permits a pilot-derived n from independent data). `prepare` recomputes them from the raw
  screen records (not the summary), cross-checks them against the summary, and freezes
  `N = max(40, paired_delta_pilot_size(deltas, {m: 20})["required_final_worlds"])`.
  With the current screen data the per-mix paired SDs are 86.8 / 100.2 / 92.3 and
  N = 235 (scripted binds).
- Kill criterion: if N > 300 the intent records `STOP_INFEASIBLE` and no episode runs.
- N is computed once. No later stage may change it.

## Final stage and decision

N paired worlds per mix; the incumbent and the candidate each play every world (6N
episodes). Primary metric: per-world paired delta `mass_integral` (candidate minus
incumbent), ordered by world index.

- Decision: `eval_stats.strict_promotion_decision(deltas_by_mix, scripted_mix="scripted",
  absolute_delta_ni=delta_NI)` (family alpha 0.05, one-sided Holm, >= 2 of 3 mixes, and
  scripted lower bound > -delta_NI), plus the behavioral bands above.
- Outcomes:
  - `STRICT_PASS`: complete, audit passed, decision valid and passing, all bands in.
  - `STRICT_FAIL`: complete, audit passed, decision valid but not passing or a band out.
  - `STOP_INFEASIBLE`: N > 300 at prepare; nothing runs.
  - `INCOMPLETE`: a deadline or wall-cap stop, or remaining caps plus the 120 s handoff do
    not fit before a stage starts. No decision is reported.
  - `INVALID_STOP`: any other failure (child crash, RSS or free-RAM breach, source or
    output drift, audit failure, producer/audit disagreement, invalid decision).

## Serving stage: rollout-harness self-play compatibility check

The directory and namespace keep the name `serving`, but this stage is a **rollout-harness
self-play compatibility check**, not serving evidence. It plays 50 candidate episodes on the
serving namespace through the same `tournament_eval.rollout(profile=...)` code as the final
stage (the Watch-mode `GameState` transition, runtime mode `watch`), with the veto on the
hero and the self-play roster. It runs no web backend or session code. Checks: all 50
complete the H5000 horizon without error and pass the candidate record-shape validator.

The intent, closeout and receipt record `serving_stage_kind: "rollout-harness self-play
compatibility check"`, `serving_stage_exercises_web_path: false` and
`serving_path_qualified: false`. The web backend (`web/backend/session.py` `GameSession`)
cannot host the wrapper without new serving code. Remaining work before a release: an
opt-in install of `FreeSpaceVeto` on the served hero in Watch (and on AI snakes in Play,
where human dispatch bypasses the policy), a per-episode serving receipt schema for
vector61 + wrapper (the strict `_serving` validator accepts only raster31v3 receipts), a
web serving run of 50 episodes under that schema, and, for any deployment that wraps more
than the one hero (every Watch snake sharing the policy, or every AI snake in Play), its
own strict evaluation.

## Execution envelope

- Stages in order: calibration, final, serving, audit child, closeout. No retry.
- Numeric stages shard episodes across two worker processes. Worker k holds shared CPU slot
  lock `cpu-slot-{k+1}.lock` under
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909` (flock, the
  supervisor helpers' lock root and release helper). Shard assignment is deterministic:
  pair (or episode) index j goes to worker `j % 2`.
- The run parent acquires **both** slot locks (all-or-nothing, 180 s) before it creates
  `output/`, and holds them until closeout, so no other job can take a slot between
  stages. Worker k inherits the parent's open lock file through `Popen(pass_fds=...)`
  (`--slot-fd`) and asserts it with a non-blocking flock on that descriptor (flock belongs
  to the open file description, which parent and worker share). Workers never unlock.
  `output/started.json` lists the held lock files.
- Each worker: CPU, torch 2 intra-op / 1 inter-op threads, process-group RSS <= 8 GiB,
  system available RAM >= 9.6 GiB, heartbeat after every episode, write-once records.
- Stage wall caps: calibration 1800 s, final **45000 s**, serving 3600 s, audit 900 s, plus
  a separate 120 s handoff reserve. Before a stage starts, the remaining time to the intent
  deadline must cover this stage's and every later stage's caps plus 120 s, so the intent
  deadline must be at least 51,420 s (14 h 17 min) after `prepare`.
- Output root `/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-strict-20260927/run-v1`
  (create-only; outside `ongoing-research-20260913`). Nothing is written elsewhere.
  `prepare` and `run` refuse a non-smoke intent on any other root, and refuse a smoke
  intent on that root or on any root named `run-*`.
- `run` checks available RAM and acquires both slots before creating `output/`, so a
  transient shortage or a busy slot is not an attempt. After `output/started.json`
  exists, every stop is final: a crash or a watchdog breach is `INVALID_STOP`; a wall-cap
  or worker deadline stop is `INCOMPLETE`.
- No other CPU load may run on the host during the run (the slot locks exclude only jobs
  that use the shared supervisor).

### Runtime projection (measured basis, fixed before any data)

The design's first estimate (18-30 s per episode, final cap 30000 s) understated the
measured cost. The Tier-1 screen (`apex-safety-screen-20260926/run-v1`) ran **one** worker
at torch 2 intra-op / 1 inter-op. Its 240 A/B records measured these rollout wall times
(mean / max, seconds):

| Mix | Incumbent (A) | Candidate (B) |
|---|---|---|
| frozen | 47.7 / 234.7 | 52.4 / 271.3 |
| scripted | 17.2 / 42.1 | 20.0 / 52.3 |
| mixed | 24.5 / 92.8 | 28.0 / 90.7 |

Overall mean 31.6 s per episode. One world triplet (A+B in all three mixes) costs 189.8 s.
The screen's 240 episodes took 8,582 s from first start to last finish against 7,590 s of
summed rollout time (overhead factor 1.13). The same world took 35.6 s at the start of that
run and 148.9 s at its end, so host load can slow episodes several-fold.

With N = 235 the final stage is 1,410 episodes, 705 per worker: 235 x 189.8 / 2 = 22,302 s
of rollout per worker. With a declared overhead allowance of 1.15 that is 25,647 s. The
worker budget under the former 30000 s cap (29,970 s) would leave about 14% headroom
after overhead, and two concurrent workers were never measured. The final cap is
therefore **45000 s** (worker budget 44,970 s): the overhead-inclusive projection is 57%
of it, which leaves room for about a 1.75x slowdown from 2-way contention or tail episodes.
`prepare` recomputes this projection from the raw screen records, records it in
`intent.caps.final_runtime_projection`, and refuses to write the intent if the projection
exceeds 70% of the worker budget. The cap changed before any calibration or final episode
and before any intent; it changes no statistical design element (N, MDE, margins,
decision, bands).

## Audit stage

Before the audit the producer writes `output/producer-outcome.json`, its pre-audit outcome
claim (`STRICT_PASS` or `STRICT_FAIL` from the decision and bands). Then, within the 900 s
audit cap:

1. **Independent audit** (gating): `python -I strict_audit.py --root <root> --out
   <root>/output/audit --pilot-root <screen run>`, stdlib only, imports nothing from the
   repo, bound in the intent and the source closure. Exit 0 with `audit.json` status `PASS`
   is a pass; exit 1 is a fail; anything else is a harness error (`INVALID_STOP`). It
   re-derives namespaces, world identities, calibration, sizing N, pairing and the decision
   (its own Student-t, Holm and NI), and checks every producer claim, including the
   pre-audit outcome.
2. **Producer self-check** (gating, run after the independent audit so it is not part of
   the audited evidence): `strict_run.py audit`. It re-hashes the source closure, the
   checkpoint snapshots and every record against its shard report (no missing or extra
   records), rebuilds the rosters, applies the shared record-shape validator
   (`validate_strict_world_record`, candidate wrapper required, incumbent wrapper
   forbidden), recomputes calibration and the decision, and cross-checks the frozen reducer
   against a second plain-mean / integrated-t implementation.

Scope follows governance "Audit scope": gated rules cover integrity, pairing, identity,
denominators, the profile digest and the decision recomputation. Veto counters versus
`decision_frames`, death causes and wall times are reported, not gated. Every self-check
rule has a positive test on real Tier-1 screen records (copied to
`tests/fixtures/apex_veto_strict_runner/`) and a negative test on a mutated copy
(`tests/test_apex_veto_strict_run.py`).

The final outcome needs both audits to pass and the self-check's recomputed pass/fail to
equal the producer's; otherwise `INVALID_STOP`.

## Dry-run before GO

`prepare --smoke-frames <=500` writes to its own root, by default
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-strict-20260927/smoke-v1`
(never `run-v1`; `prepare` refuses a smoke intent on the Tier-2 root). It builds a
plumbing intent on its own namespace
(`apex-veto-strict-smoke-v1`, never a Tier-2 world): one scripted world, one incumbent and
one candidate episode on the legacy truncated path, then `run` exercises both workers, the
slot locks, the supervisor, the independent audit, the self-check and closeout
(`SMOKE_NO_DECISION`). On smoke the independent audit runs as `strict_audit.py --smoke`
(governance rule 5): the H5000 record, sizing, calibration and decision rules are replaced
by legacy-path record rules (smoke world and roster, envelope, wrapper identity, counters at
most the smoke frame cap), while the shard plan and hashes, identity, counters and artifact
rules are the same code as the full audit, and no document may claim a Tier-2 outcome. It
gates the smoke exactly as the full audit gates the run; the full audit refuses a smoke
intent (`mode.not_smoke`) and `--smoke` refuses a non-smoke intent (`smoke.intent`). The
H5000 path through both audits is exercised before GO by
`tests/test_apex_veto_strict_integration.py`: `strict_run.run` on a real intent with
in-process workers returning real Tier-1 screen records, audited by the real
`strict_audit.py` and self-check children (STRICT_PASS and STRICT_FAIL runs).

## Non-claims

No champion file, default, config or deployment changes on any outcome. A `STRICT_PASS`
receipt does not qualify the web serving path. Tier-1 screen numbers are not evidence here.

The result covers a **single wrapped hero against unwrapped opponents** under
`promotion-v2-watch-rect` H5000 only, in both the final stage and the self-play
compatibility check. The configuration a release would most likely ship (the wrapper on
every Watch snake that shares the policy, or on every AI snake in Play) is not measured.
Any deployment that wraps several or all snakes needs its own evaluation.
