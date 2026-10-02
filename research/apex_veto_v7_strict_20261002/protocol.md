# Apex + v7 space-preference veto (lambda=4.0) vs Apex + released v5 veto: Tier-2 strict challenge (pre-registration)

Status: **pre-registered Tier-2 design, written 2026-10-02 after the v7 Tier-1 screen
(`apex-veto-v7-screen-20261002/run-v1`) closed with `RECOMMEND_STRICT_GATE`, before any
calibration or final episode of this study and before any intent.** Harness:
`strict_run.py` beside this file (independent audit `strict_audit.py`). Governance:
[`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md)
(Tier 2). A `STRICT_PASS` produces a receipt only. It changes no champion file, default,
config, released veto or deployment. Release is a separate, explicit action.

This package is a copy of the v5 strict package
(`research/apex_veto_v5_strict_20261001/{strict_run.py,strict_audit.py,protocol.md}`, bound
by the v5 `STRICT_PASS` receipt), parametrized for the new arms. Those files are not
modified and not imported. Every hardening of the v5 package is kept (monotonic-clock
heartbeat, AC-power guard, two-slot sharding with inherited slot locks, stage caps, runtime
projection, create-only writes, no retry, screen-pilot gating with source parity, an
independent stdlib audit plus a producer self-check).

## Question and decision it informs

Does replacing the released v5 boost-aware veto with the v7 space-preference veto at
`lambda = 4.0` on the served Apex hero improve the H5000 mass integral, strictly (Holm
superiority in >= 2 of 3 mixes plus scripted non-inferiority plus behavioral bands), on fresh
worlds? The answer decides whether a later release action may consider replacing v5 with v7
(which would also need its own web serving qualification).

## Precondition: the v7 Tier-1 screen and its lambda

`prepare` refuses to write an intent unless the v7 screen
(`/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-screen-20261002/run-v1`)
has finished with:

- `receipt.json` `decision` == `RECOMMEND_STRICT_GATE` and
  `screen_decision_before_self_check` == `RECOMMEND_STRICT_GATE`;
- `receipt.json` `self_check.passes` == true with no failures;
- `receipt.json` `summary_sha256` equal to the bytes of `summary.json`, and every A/B
  record the sizing reads listed in `receipt.records` with the same sha256;
- 40 A/B pairs per mix, A records carrying the v5 probe and B records the v7 probe with
  `space_preference_lambda == 4.0`, and the per-mix deltas recomputed from the raw records
  equal to `summary.json` `per_mix.<mix>.primary_mass_integral.deltas_B_minus_A`;
- source parity (`pilot.screen_source_parity`): the screen `intent.json` is the one
  `receipt.json` binds (`intent_sha256`), its `git.commit` is the pre-declared screen commit
  `900385fed868a1b35f308871d211aeafc318e4c2` with an empty `git.dirty_paths`, and every
  veto module each arm runs read at that commit (`git show <commit>:<path>`) has the same
  sha256 as the strict arm's frozen `wrapper_identity.source_sha256s`, so N is sized from the
  variance of the same candidate and incumbent bytes;
- lambda binding (`pilot.lambda_binding`): the screen receipt's `lambda` and
  `sweep_binding.selected_lambda` are 4.0; `sweep_binding.summary_path` is the DEV sweep
  summary
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-lambda-sweep-20261002/run-v1/summary.json`
  and `summary_sha256` its bytes; that summary is a real (non-smoke)
  `apex-veto-v7-lambda-sweep/v1` summary of `apex-veto-v7-dev-v1`, `selection.status ==
  SELECTED` with `passes` at `selected_lambda == 4.0`, its `intent_sha256` is the sweep
  `intent.json`, and its source commit (summary and sweep intent, clean) is the screen
  commit; the screen intent's arm B description and hypothesis name `lambda=4.0` and the
  sweep summary sha256.

Any other screen outcome means this study does not run. The independent audit re-checks
the same gates (`pilot.pairs`, `pilot.screen_receipt`), recomputes source parity itself from
the screen intent and `git show` (`pilot.screen_source_parity`) and the lambda binding from
the receipt, the sweep summary and both intents (`pilot.lambda_binding`), comparing each
with what the producer recorded.

## Arms and identities

| Role | Checkpoint | Wrapper (hero only) |
|---|---|---|
| Incumbent (released) | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c5…d747ac93` | `free-space-veto/v5-boost-aware`: `safety_veto_v5.py` `d86d084e…ec86c`, `safety_veto.py` `1b62d15c…c428`, `safety_veto_v3.py` `ed3a6d86…be1` |
| Candidate | the same bytes | `free-space-veto/v7-space-preference(lambda=4.0)`: `safety_veto_v7.py` plus the three modules above |

Both arms are wrapped and both install through
`dev_screen.hero_veto_installer(...)` around `tournament_eval.rollout(hero_safety_veto=True)`:
the built-in vector61 guard runs first, then the incumbent's `install_boost_aware_veto`
(v5) or the candidate's `install_space_preference_veto(hero, 4.0)` (v7) replaces rollout's
v2 veto on the hero. Opponents never carry a veto. This is the v7 screen's A/B install.

Each arm's identity is {checkpoint sha256, wrapper method and descriptor, sha256 of every
veto module the arm executes}. `safety_veto_v7.py` imports `safety_veto.py`,
`safety_veto_v3.py` and `safety_veto_v5.py` (and `src/game/snake_state.py`, a game module);
v5 imports v2 and v3; v3 imports v2 (plus `src/core/game_config.py` and
`src/game/game_logic.py`). The identities bind the veto modules (incumbent: v5, v2, v3;
candidate: v7, v2, v3, v5); the non-veto imports are bound by the source closure (all of
`src/`), not by the arm identity (disclosed). `prepare`, `validate_intent`, the self-check
and the independent audit require the incumbent's three sha256s to equal the released bytes
above (`incumbent_released_source_sha256s`) and the candidate descriptor's
`space_preference_lambda` to be 4.0.

`prepare` binds both identities into `intent.json` (`incumbent` / `candidate`:
`wrapper`, `wrapper_identity` with `descriptor`, `source_sha256`, `source_sha256s`, and
`wrapper_source_sha256`) and requires each module's sha256 to equal its source-closure
entry. Every record envelope carries `safety_veto: true`, its arm's `wrapper`,
`wrapper_source_sha256`, `wrapper_source_sha256s` and `veto_diagnostics` of its own kind
(incumbent: the v5 counters; candidate: the v7 counters with the nested v5 counters;
values reported only). Every record's `probes.safety_veto` must equal (without `counters`)
its arm's descriptor and carry exactly the seven v2 counters with `decisions = kept_base +
vetoes_applied + fallback_no_spacious`. The producer checks the shape with
`validate_strict_world_record(..., candidate_wrapper=<arm descriptor>)` for both arms; the
independent audit re-derives the same rules (`identity.wrapper_source_bound_to_closure`,
`identity.arm_records_bound`, `identity.wrapper_source_sha256`).

## Profile, mixes and rosters

- Profile `promotion-v2-watch-rect`, H5000, digest `d396d3ed…0e8b`. Pinned config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3…15aa5`).
- Mixes frozen, scripted and mixed, rosters from `dev_screen._design_rows` (the strict
  pilot's checkpoint pool order and mixed-slot rules). Before writing the intent, `prepare`
  rebuilds the strict pilot's 48 development rosters and requires identity parity.
- Serving stage roster: `serving-selfplay`, five unwrapped champion opponents, hero with the
  candidate (v7) veto.

## Worlds (all new namespaces)

Seeds are `uint32_be(sha256("<domain>|worlds|<i>")[:4])`, the `dev_screen` recipe.

| Namespace (domain) | Count | Use |
|---|---|---|
| `apex-veto-v7-strict-dev-v1` | 16 | incumbent-only calibration, 3 mixes |
| `apex-veto-v7-strict-final-v1` | ordered bank, Nmax 300 | final paired worlds; the first N are played |
| `apex-veto-v7-strict-serving-v1` | 50 | serving-compatibility episodes |
| `apex-veto-v7-strict-smoke-v1` | 1 | plumbing smoke only, never a Tier-2 world |

`prepare` fails closed unless the seeds are unique within and across the three namespaces,
disjoint from every set `dev_screen.disjointness_report` checks (the task-aligned challenger
namespaces, the strict pilot's observed seeds, seeds 0..999), and disjoint from the first
1000 seeds of every earlier domain/purpose of this recipe:

- `apex-safety-screen-v1/worlds` (the v2 Tier-1 screen; also the SIMD H5000 parity worlds);
- `apex-veto-strict-{dev,final,serving}-v{1,2,3}` and `apex-veto-strict-smoke-v1`;
- both web serving lanes and their smokes, `apex-veto-web-serving{,-smoke}-v1` and
  `apex-veto-v5-web-serving{,-smoke}-v1`, purposes `watch`, `play`, `parity` (and `worlds`);
- the v3, v4, v5, v6 and v7 screens and their smokes (`apex-veto-v3-screen-v1`, `-v2`,
  `apex-veto-v{3,4,5,6,7}-screen-smoke-v1`, `apex-veto-v{4,5,6,7}-screen-v1`); the SIMD v5
  parity replayed v5 screen worlds;
- the v5 strict banks and smoke `apex-veto-v5-strict-{dev,final,serving,smoke}-v1`;
- the v7 DEV lambda sweep `apex-veto-v7-dev-v1` and its smoke `apex-veto-v7-dev-smoke-v1`;
- `trap-horizon-dev-v1`, `trap-horizon-dev-smoke-v1`, `trap-horizon-v5-dev-v1`,
  `trap-horizon-v5-dev-smoke-v1`;
- this study's smoke domain, `apex-veto-v7-strict-smoke-v1`.

World identity includes the seed, so seed disjointness implies world-identity disjointness.
The governance namespace registry is not implemented; this in-code check is its stand-in
(disclosed gap). The independent audit recomputes the same exclusion list
(`namespaces.fresh_and_disjoint`).

## Calibration stage (incumbent only, before any final data)

The incumbent (champion + v5) plays the 16 dev worlds x 3 mixes (48 episodes). From those
records:

- reference means of `mass_integral` and `survival_fraction` per mix;
- absolute non-inferiority margin `delta_NI = 0.03 x (dev incumbent scripted mean mass)`;
  the run stops `INVALID_STOP` if it is not positive;
- behavioral bands on candidate final records: per mix, the candidate's mean
  `survival_fraction` must lie in `[dev_mean - 0.02, dev_mean + 1]`.

`calibration.json` is written create-only before the final stage starts.

## Sizing (frozen in the intent)

- MDE = **30 mass-integral units absolute per mix**.
- **Disclosure: the MDE was chosen after the v7 screen finished and after its result was
  seen.** The v2 and v5 strict gates used MDE 20 (declared before their screens' results).
  The v7 screen's paired B-A standard deviations are 139.9 (frozen), 152.9 (scripted) and
  135.3 (mixed), so MDE 20 would need N = 547 > Nmax 300 (`STOP_INFEASIBLE`). With these
  SDs, MDE 25/30/35/40 give N = 350/243/179/137.
- **Rule (amendment, before `prepare`):** the smallest MDE in 5-unit steps whose N fits
  Nmax = 300 and the 70% runtime limit, which is 30 (N = 243; 25 needs 350). This replaces
  an earlier draft value of 40, which a reviewer flagged as a choice rather than a necessity.
  The observed screen effect (+134.8 / +153.9 / +153.9) did not set the value and is not
  evidence here. The MDE only affects power; the decision is Holm superiority over 0 on
  fresh worlds with N frozen before any final data. The intent records `mde_basis`,
  `mde_chosen_after_v7_screen: true`, `v7_screen_complete_before_mde: true` and the
  infeasible alternative.
- The v7 screen's paired B-A mass deltas are the independent pilot (disjoint worlds). Its A
  arm is exactly this study's incumbent (champion + v5) and its B arm exactly the candidate
  (champion + v7 at 4.0), so its B-A variance is the right planning variance. `prepare`
  recomputes the deltas from the raw screen records (not the summary), cross-checks them
  against the summary and the receipt, and freezes
  `N = max(40, paired_delta_pilot_size(deltas, {m: 30})["required_final_worlds"])`
  (243 from the closed screen: scripted binds).
- Kill criterion: if N > 300 the intent records `STOP_INFEASIBLE` and no episode runs.
- N is computed once. No later stage may change it.
- Record-shape fixtures: one complete frozen A/B pair of the v7 screen (world index 0) and
  the A/B pair of the v7 screen's GO smoke were copied into
  `tests/fixtures/apex_veto_v7_strict/` (sha256 in `provenance.json`); they only test
  record shapes.

## Final stage and decision

N paired worlds per mix; the incumbent and the candidate each play every world (6N
episodes). Primary metric: per-world paired delta `mass_integral` (candidate minus
incumbent, i.e. v7 minus v5), ordered by world index.

- Decision: `eval_stats.strict_promotion_decision(deltas_by_mix, scripted_mix="scripted",
  absolute_delta_ni=delta_NI)` (family alpha 0.05, one-sided Holm, >= 2 of 3 mixes, and
  scripted lower bound > -delta_NI), plus the behavioral bands above.
- Outcomes:
  - `STRICT_PASS`: complete, audit passed, decision valid and passing, all bands in.
  - `STRICT_FAIL`: complete, audit passed, decision valid but not passing or a band out.
  - `STOP_INFEASIBLE`: N > 300 at prepare; nothing runs.
  - `INCOMPLETE`: a deadline or wall-cap stop, or remaining caps plus the 120 s handoff do
    not fit before a stage starts. No decision is reported.
  - `INVALID_STOP`: any other failure (child crash, RSS or free-RAM breach, stale
    heartbeat, source or output drift, audit failure, producer/audit disagreement, invalid
    decision).

## Serving stage: rollout-harness self-play compatibility check

This stage is a **rollout-harness self-play compatibility check**, not serving evidence. It
plays 50 candidate (v7) episodes on the serving namespace through the same
`tournament_eval.rollout(profile=...)` code as the final stage (Watch-mode `GameState`
transition, runtime mode `watch`), with the v7 veto on the hero and five unwrapped champion
opponents. It runs no web backend or session code. Checks: all 50 complete the H5000
horizon and pass the candidate record-shape validator.

The intent, closeout and receipt record `serving_stage_kind: "rollout-harness self-play
compatibility check"`, `serving_stage_exercises_web_path: false` and
`serving_path_qualified: false`. Today `web/backend/safety_veto_serving.py` serves v2 or
v5 (`SNAKE_SERVE_VETO_VARIANT`) on the Watch hero and cannot install v7. Before any v7
release: an opt-in v7 variant there, a per-episode web serving receipt for the v7
descriptor and source closure, and a separate 50-episode web serving run with its own audit
(the pattern of `research/apex_veto_v5_serving_20261001`), which should also measure v7's
extra per-decision cost in the served frame loop. Any deployment that wraps more than the
one hero needs its own strict evaluation.

## Execution envelope (unchanged from the v5 package)

- Stages in order: calibration, final, serving, audit children, closeout. No retry.
- Numeric stages shard episodes across two worker processes; pair (or episode) index j goes
  to worker `j % 2`. The run parent acquires **both** shared CPU slot locks
  (`cpu-slot-{1,2}.lock` under
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909`, all-or-nothing,
  180 s) before it creates `output/`, and holds them until closeout. Worker k inherits the
  parent's open lock file (`--slot-fd`) and asserts it with a non-blocking flock; workers
  never unlock. `output/started.json` lists the held lock files.
- Each worker: CPU, torch 2 intra-op / 1 inter-op threads, process-group RSS <= 8 GiB,
  system available RAM >= 9.6 GiB, a heartbeat after every episode, write-once records.
- Heartbeat staleness (600 s) is measured on the system monotonic clock, which pauses while
  the host sleeps; a missing or unreadable beat counts from stage start.
- A Tier-2 `run` refuses to start on battery (macOS `pmset`), before any record exists. The
  host must stay on AC with the lid open.
- Stage wall caps: calibration 1800 s, final 45000 s, serving 3600 s, audit 900 s, plus a
  separate 120 s handoff reserve. Before a stage starts, the remaining time to the intent
  deadline must cover this stage's and every later stage's caps plus 120 s, so the intent
  deadline must be at least 51,420 s (14 h 17 min) after `prepare`.
- Output root `/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-strict-20261002/run-v1`
  (create-only; outside `ongoing-research-20260913`). `prepare` and `run` refuse a non-smoke
  intent on any other root, and refuse a smoke intent on that root or on any root named
  `run-*`.
- `run` checks available RAM, AC power and both slots before creating `output/`, so a
  transient shortage or a busy slot is not an attempt. After `output/started.json` exists,
  every stop is final: a crash or a watchdog breach is `INVALID_STOP`; a wall-cap or worker
  deadline stop is `INCOMPLETE`.
- No other CPU load may run on the host during the run.

### Runtime projection (checked at prepare)

`prepare` recomputes the per-worker final-stage projection from the v7 screen's measured
A/B episode wall times (one worker, 2 intra-op / 1 inter-op): N x (sum over mixes of the mean
A and mean B episode time) / 2 workers x 1.15 overhead. It records the projection in
`intent.caps.final_runtime_projection` and refuses to write the intent if it exceeds 70% of
the 44,970 s worker budget. From the closed screen: v5 episodes average 29.8 / 16.3 / 21.9 s
and v7 episodes 38.0 / 22.2 / 29.6 s (frozen / scripted / mixed), 157.7 s per world triplet;
N = 243 projects to about 22,000 s per worker (49% of the budget). Two concurrent workers
were never measured, which is why the 70% limit leaves room for contention. Expected wall
clock for the whole run: about 0.2 h calibration, 6.1 h final, 0.2 h serving, under 0.25 h
audit (about 6.8 h of the 14 h 17 min the caps reserve).

## Audit stage

Before the audit the producer writes `output/producer-outcome.json`, its pre-audit outcome
claim (`STRICT_PASS` or `STRICT_FAIL` from the decision and bands). Then, within the 900 s
audit cap:

1. **Independent audit** (gating): `python -I strict_audit.py --root <root> --out
   <root>/output/audit --pilot-root <v7 screen run> --sweep-summary <sweep summary>`,
   stdlib only, imports nothing from the repo, bound in the intent and the source closure.
   Exit 0 with `audit.json` status `PASS` is a pass; exit 1 is a fail; anything else is a
   harness error (`INVALID_STOP`). It re-derives namespaces, world identities, both arm
   identities against the source closure and the released v5 bytes, the pilot gate, source
   parity, the lambda binding, calibration, sizing N, pairing and the decision (its own
   Student-t, Holm and NI), and checks every producer claim, including the pre-audit outcome,
   the MDE (40) and the lambda (4.0) wherever they are claimed.
2. **Producer self-check** (gating, after the independent audit): `strict_run.py audit`. It
   re-hashes the source closure, the checkpoint snapshots and every record against its shard
   report, recomputes both arm identities from the frozen sources (incumbent sources must be
   the released v5 bytes, candidate lambda 4.0), rebuilds the rosters, applies
   `validate_strict_world_record` with each arm's descriptor, checks each arm's
   `veto_diagnostics` kind, recomputes calibration and the decision, and cross-checks the
   frozen reducer against a second implementation.

Gated rules cover integrity, pairing, identity, denominators, the profile digest, the pilot
gate and the decision recomputation. Veto counters versus `decision_frames`, the v5 landing
and v7 re-rank counters (including `rerank_changes_tail_release_driven`), death causes and
wall times are reported, not gated. Rules are tested positively on real record shapes and
negatively on mutated copies (`tests/test_apex_veto_v7_strict_run.py`,
`tests/test_apex_veto_v7_strict_audit.py`, `tests/test_apex_veto_v7_strict_source_parity.py`):
fixtures are one complete frozen A/B pair of the v7 screen (H5000; v5 probe and v5
diagnostics on A, v7 probe and v7 diagnostics on B) and the A/B pair of the v7 screen's GO
smoke (legacy 500-frame path, lambda 4.0), copied to `tests/fixtures/apex_veto_v7_strict/`
with their sha256 in `provenance.json`. Every test that calls a runner entry point first
replaces `tournament_eval.rollout`, `dev_screen.run_episode` and
`strict_run.run_unit_episode` with functions that raise; the pipeline tests then install an
in-process fake episode that returns relabelled fixture records
(`tests/test_apex_veto_v7_strict_integration.py`: STRICT_PASS, STRICT_FAIL and a smoke,
audited by the real `strict_audit.py` and self-check children).

The final outcome needs both audits to pass and the self-check's recomputed pass/fail to
equal the producer's; otherwise `INVALID_STOP`.

## Dry-run before GO

`prepare --smoke-frames <=500` writes to its own root (default
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v7-strict-20261002/smoke-v1`,
never `run-v1`). It builds a plumbing intent on `apex-veto-v7-strict-smoke-v1`: one scripted
world, one incumbent (v5) and one candidate (v7, lambda 4.0) episode on the legacy truncated
path, then `run` exercises both workers, the slot locks, the supervisor, the independent
audit (`--smoke`), the self-check and closeout (`SMOKE_NO_DECISION`). The smoke does not
read the screen. The full audit refuses a smoke intent (`mode.not_smoke`) and `--smoke`
refuses a non-smoke intent (`smoke.intent`).

The plumbing smoke that counts as GO evidence must run from a clean tree at the commit the
GO binds (its intent `source_closure.dirty` empty and `source_closure.commit` equal to that
commit). A smoke from a dirty tree is a development check only.

## Non-claims

No champion file, default, config, released veto or deployment changes on any outcome. A
`STRICT_PASS` receipt does not qualify the web serving path. v7 screen numbers are not
evidence here (pilot variance only; their observed effect informed the MDE choice, as
disclosed). The result covers **lambda = 4.0 only, a single wrapped hero against unwrapped
opponents** under `promotion-v2-watch-rect` H5000 only. A `STRICT_FAIL` does not show that
v7 harms the hero; say "harm" only when a confidence bound excludes zero. The released
veto modules (`safety_veto.py`, `safety_veto_v3.py`, `safety_veto_v5.py`) and
`safety_veto_v7.py` are not modified by this study; `prepare`, `validate_intent`, the
self-check and the independent audit all require the released incumbent sha256s.
