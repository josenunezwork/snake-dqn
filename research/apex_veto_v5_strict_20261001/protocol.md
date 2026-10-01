# Apex + v5 boost-aware veto vs Apex + released v2 veto: Tier-2 strict challenge (pre-registration)

Status: **pre-registered Tier-2 design, written 2026-10-01 while the v5 Tier-1 screen
(`apex-veto-v5-screen-20261001/run-v1`) was still running, before any calibration or final
episode of this study and before any intent.** Harness: `strict_run.py` beside this file
(independent audit `strict_audit.py`). Governance:
[`docs/research/governance_tiers_2026-09-26.md`](../../docs/research/governance_tiers_2026-09-26.md)
(Tier 2). A `STRICT_PASS` produces a receipt only. It changes no champion file, default,
config, released veto or deployment. Release is a separate, explicit action.

This package is a copy of the v2 strict package
(`research/apex_veto_strict_20260927/{strict_run.py,strict_audit.py,protocol.md}`, bound by
the run-v3 `STRICT_PASS` receipt), parametrized for the new arms. Those files are not
modified and not imported. Every hardening of the v2 package is kept (monotonic-clock
heartbeat, AC-power guard, two-slot sharding with inherited slot locks, stage caps,
create-only writes, no retry, an independent stdlib audit plus a producer self-check).

## Question and decision it informs

Does replacing the released v2 free-space veto with the v5 boost-aware veto on the served
Apex hero improve the H5000 mass integral, strictly (Holm superiority in >= 2 of 3 mixes plus
scripted non-inferiority plus behavioral bands), on fresh worlds? The answer decides whether
a later release action may consider replacing v2 with v5 (which would also need its own web
serving qualification).

## Precondition: the v5 Tier-1 screen

`prepare` refuses to write an intent unless the v5 screen
(`/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v5-screen-20261001/run-v1`)
has finished with:

- `receipt.json` `decision` == `RECOMMEND_STRICT_GATE` and
  `screen_decision_before_self_check` == `RECOMMEND_STRICT_GATE`;
- `receipt.json` `self_check.passes` == true with no failures;
- `receipt.json` `summary_sha256` equal to the bytes of `summary.json`, and every A/B
  record the sizing reads listed in `receipt.records` with the same sha256;
- 40 A/B pairs per mix, and the per-mix deltas recomputed from the raw records equal to
  `summary.json` `per_mix.<mix>.primary_mass_integral.deltas_B_minus_A`.

Any other screen outcome (`NOT_ADVANCED`, `INCOMPLETE`, `INVALID_*`, a failed self-check)
means this study does not run. The independent audit re-checks the same gate
(`pilot.screen_receipt`).

## Arms and identities

| Role | Checkpoint | Wrapper (hero only) |
|---|---|---|
| Incumbent (released) | `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c5…d747ac93` | `free-space-veto/v2-speed-preserving`, `src/evaluation/safety_veto.py` sha256 `1b62d15c48987584533efbefdf3fd84e1e3f2e22cfe70dbb82b43ec57169c428` |
| Candidate | the same bytes | `free-space-veto/v5-boost-aware`, `src/evaluation/safety_veto_v5.py` |

Both arms are wrapped. The incumbent uses `tournament_eval.rollout(hero_safety_veto=True)`
(its built-in v2 install). The candidate runs the same call inside
`dev_screen.hero_veto_installer(install_boost_aware_veto)`: the built-in vector61 guard
runs first and the v5 veto replaces the v2 one on the hero. Opponents never carry a veto.

Each arm's identity is {checkpoint sha256, wrapper method and descriptor, sha256 of every
veto module the arm executes}:

- incumbent: `src/evaluation/safety_veto.py`, which `prepare` requires to equal the released
  sha256 above (`incumbent_released_source_sha256`);
- candidate: `src/evaluation/safety_veto_v5.py` (primary) plus the modules it imports,
  `src/evaluation/safety_veto.py` (v2) and `src/evaluation/safety_veto_v3.py` (v3's
  `grid_for`, `static_blocked`, `tail_aware_reachable`).

`prepare` binds both identities into `intent.json` (`incumbent` / `candidate`:
`wrapper`, `wrapper_identity` with `descriptor`, `source_sha256`, `source_sha256s`, and
`wrapper_source_sha256`) and requires each module's sha256 to equal its source-closure
entry. Every record envelope carries `safety_veto: true`, its arm's `wrapper`,
`wrapper_source_sha256` and `wrapper_source_sha256s`; candidate envelopes also carry
`veto_diagnostics` (the v5 counters, reported only). Every record's `probes.safety_veto`
must equal (without `counters`) its arm's descriptor and carry exactly the seven v2
counters with `decisions = kept_base + vetoes_applied + fallback_no_spacious`. The
producer checks the shape with `validate_strict_world_record(..., candidate_wrapper=<arm
descriptor>)` for both arms; the independent audit re-derives the same rules
(`identity.wrapper_source_bound_to_closure`, `identity.arm_records_bound`,
`identity.wrapper_source_sha256`).

## Profile, mixes and rosters

- Profile `promotion-v2-watch-rect`, H5000, digest `d396d3ed…0e8b`. Pinned config
  `research/apex_safety_20260926/deployment.yaml` (sha256 `4146baa3…15aa5`).
- Mixes frozen, scripted and mixed, rosters from `dev_screen._design_rows` (the strict
  pilot's checkpoint pool order and mixed-slot rules). Before writing the intent, `prepare`
  rebuilds the strict pilot's 48 development rosters and requires identity parity.
- Serving stage roster: `serving-selfplay`, five unwrapped champion opponents, hero with the
  candidate (v5) veto.

## Worlds (all new namespaces)

Seeds are `uint32_be(sha256("<domain>|worlds|<i>")[:4])`, the `dev_screen` recipe.

| Namespace (domain) | Count | Use |
|---|---|---|
| `apex-veto-v5-strict-dev-v1` | 16 | incumbent-only calibration, 3 mixes |
| `apex-veto-v5-strict-final-v1` | ordered bank, Nmax 300 | final paired worlds; the first N are played |
| `apex-veto-v5-strict-serving-v1` | 50 | serving-compatibility episodes |
| `apex-veto-v5-strict-smoke-v1` | 1 | plumbing smoke only, never a Tier-2 world |

`prepare` fails closed unless the seeds are unique within and across the three namespaces,
disjoint from every set `dev_screen.disjointness_report` checks (the task-aligned challenger
namespaces, the strict pilot's observed seeds, seeds 0..999), and disjoint from the first
1000 seeds of every earlier domain/purpose of this recipe:

- `apex-safety-screen-v1/worlds` (the v2 Tier-1 screen; also the SIMD H5000 parity worlds);
- `apex-veto-strict-{dev,final,serving}-v{1,2,3}/worlds` (all v2 strict banks, including the
  two invalid runs) and `apex-veto-strict-smoke-v1`;
- `apex-veto-web-serving-v1` and `apex-veto-web-serving-smoke-v1` (`watch`, `play`, `parity`);
- `apex-veto-v3-screen-v1`, `apex-veto-v3-screen-v2`, `apex-veto-v3-screen-smoke-v1`;
- `apex-veto-v4-screen-v1`, `apex-veto-v4-screen-smoke-v1`;
- `apex-veto-v5-screen-v1` (the pilot), `apex-veto-v5-screen-smoke-v1`;
- `trap-horizon-dev-v1`, `trap-horizon-dev-smoke-v1`;
- this study's smoke domain, `apex-veto-v5-strict-smoke-v1`.

World identity includes the seed, so seed disjointness implies world-identity disjointness.
The governance namespace registry is not implemented; this in-code check is its stand-in
(disclosed gap). The independent audit recomputes the same exclusion list
(`namespaces.fresh_and_disjoint`).

## Calibration stage (incumbent only, before any final data)

The incumbent (champion + v2) plays the 16 dev worlds x 3 mixes (48 episodes). From those
records:

- reference means of `mass_integral` and `survival_fraction` per mix;
- absolute non-inferiority margin `delta_NI = 0.03 x (dev incumbent scripted mean mass)`;
  the run stops `INVALID_STOP` if it is not positive;
- behavioral bands on candidate final records: per mix, the candidate's mean
  `survival_fraction` must lie in `[dev_mean - 0.02, dev_mean + 1]`.

`calibration.json` is written create-only before the final stage starts.

## Sizing (frozen in the intent)

- MDE = **20 mass-integral units absolute per mix**, the same declared product threshold as
  the v2 strict gate, carried over unchanged. **Disclosure:** it was declared while the v5
  screen was running and before its result existed. Two complete records of that screen
  (one frozen A/B pair, world index 0) were copied into
  `tests/fixtures/apex_veto_v5_strict/` as record-shape fixtures; they were not analyzed and
  play no role in any number here. The intent records `mde_basis` and
  `v5_screen_complete_before_mde: false`.
- The v5 screen's paired B-A mass deltas are the independent pilot (disjoint worlds). Its A
  arm is exactly this study's incumbent (champion + v2) and its B arm exactly the candidate
  (champion + v5), so its B-A variance is the right planning variance. `prepare` recomputes
  the deltas from the raw screen records (not the summary), cross-checks them against the
  summary and the receipt, and freezes
  `N = max(40, paired_delta_pilot_size(deltas, {m: 20})["required_final_worlds"])`.
- Kill criterion: if N > 300 the intent records `STOP_INFEASIBLE` and no episode runs.
- N is computed once. No later stage may change it.

## Final stage and decision

N paired worlds per mix; the incumbent and the candidate each play every world (6N
episodes). Primary metric: per-world paired delta `mass_integral` (candidate minus
incumbent, i.e. v5 minus v2), ordered by world index.

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

As in the v2 package, this stage is a **rollout-harness self-play compatibility check**, not
serving evidence. It plays 50 candidate (v5) episodes on the serving namespace through the
same `tournament_eval.rollout(profile=...)` code as the final stage (Watch-mode `GameState`
transition, runtime mode `watch`), with the v5 veto on the hero and five unwrapped champion
opponents. It runs no web backend or session code. Checks: all 50 complete the H5000
horizon and pass the candidate record-shape validator.

The intent, closeout and receipt record `serving_stage_kind: "rollout-harness self-play
compatibility check"`, `serving_stage_exercises_web_path: false` and
`serving_path_qualified: false`. Today `web/backend/safety_veto_serving.py` installs only the
v2 `FreeSpaceVeto` on the served Watch hero. Before any v5 release: an opt-in v5 install
there, a per-episode web serving receipt for the v5 descriptor and source closure, and a
separate 50-episode web serving run with its own audit (the pattern of
`research/apex_veto_serving_20261001`). Any deployment that wraps more than the one hero
needs its own strict evaluation.

## Execution envelope (unchanged from the v2 package)

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
- Output root `/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v5-strict-20261001/run-v1`
  (create-only; outside `ongoing-research-20260913`). `prepare` and `run` refuse a non-smoke
  intent on any other root, and refuse a smoke intent on that root or on any root named
  `run-*`.
- `run` checks available RAM, AC power and both slots before creating `output/`, so a
  transient shortage or a busy slot is not an attempt. After `output/started.json` exists,
  every stop is final: a crash or a watchdog breach is `INVALID_STOP`; a wall-cap or worker
  deadline stop is `INCOMPLETE`.
- No other CPU load may run on the host during the run.

### Runtime projection (checked at prepare)

`prepare` recomputes the per-worker final-stage projection from the v5 screen's measured A/B
episode wall times (one worker, 2 intra-op / 1 inter-op): N x (sum over mixes of the mean A
and mean B episode time) / 2 workers x 1.15 overhead (the v2 screen measured 1.13). It records
the projection in `intent.caps.final_runtime_projection` and refuses to write the intent if
it exceeds 70% of the 44,970 s worker budget. If it does, this protocol needs an amendment
before any data; the cap is not raised silently. For scale: the v2 gate measured 189.8 s per
world triplet (N = 235: 57% of the budget); the v5 screen protocol expects about 21 s per v2
episode.

## Audit stage

Before the audit the producer writes `output/producer-outcome.json`, its pre-audit outcome
claim (`STRICT_PASS` or `STRICT_FAIL` from the decision and bands). Then, within the 900 s
audit cap:

1. **Independent audit** (gating): `python -I strict_audit.py --root <root> --out
   <root>/output/audit --pilot-root <v5 screen run>`, stdlib only, imports nothing from the
   repo, bound in the intent and the source closure. Exit 0 with `audit.json` status `PASS`
   is a pass; exit 1 is a fail; anything else is a harness error (`INVALID_STOP`). It
   re-derives namespaces, world identities, both arm identities against the source closure,
   the pilot gate, calibration, sizing N, pairing and the decision (its own Student-t, Holm
   and NI), and checks every producer claim, including the pre-audit outcome.
2. **Producer self-check** (gating, after the independent audit): `strict_run.py audit`. It
   re-hashes the source closure, the checkpoint snapshots and every record against its shard
   report, recomputes both arm identities from the frozen sources, rebuilds the rosters,
   applies `validate_strict_world_record` with each arm's descriptor, recomputes calibration
   and the decision, and cross-checks the frozen reducer against a second implementation.

Gated rules cover integrity, pairing, identity, denominators, the profile digest, the pilot
gate and the decision recomputation. Veto counters versus `decision_frames`, the v5 landing
counters, death causes and wall times are reported, not gated. Rules are tested positively
on real record shapes and negatively on mutated copies
(`tests/test_apex_veto_v5_strict_run.py`, `tests/test_apex_veto_v5_strict_audit.py`):
fixtures are one complete frozen A/B pair of the v5 screen (H5000; v2 probe on A, v5 probe
and `veto_diagnostics` on B) and one A/B pair of a v5 screen smoke (legacy 500-frame path),
copied to `tests/fixtures/apex_veto_v5_strict/` with their sha256 in `provenance.json`.
Every test that calls a runner entry point first replaces `tournament_eval.rollout`,
`dev_screen.run_episode` and `strict_run.run_unit_episode` with functions that raise; the
pipeline tests then install an in-process fake episode that returns relabelled fixture
records (`tests/test_apex_veto_v5_strict_integration.py`: STRICT_PASS, STRICT_FAIL and a
smoke, audited by the real `strict_audit.py` and self-check children).

The final outcome needs both audits to pass and the self-check's recomputed pass/fail to
equal the producer's; otherwise `INVALID_STOP`.

## Dry-run before GO

`prepare --smoke-frames <=500` writes to its own root (default
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/apex-veto-v5-strict-20261001/smoke-v1`,
never `run-v1`). It builds a plumbing intent on `apex-veto-v5-strict-smoke-v1`: one scripted
world, one incumbent (v2) and one candidate (v5) episode on the legacy truncated path, then
`run` exercises both workers, the slot locks, the supervisor, the independent audit
(`--smoke`), the self-check and closeout (`SMOKE_NO_DECISION`). The smoke does not need the
v5 screen to be finished. The full audit refuses a smoke intent (`mode.not_smoke`) and
`--smoke` refuses a non-smoke intent (`smoke.intent`).

## Non-claims

No champion file, default, config, released veto or deployment changes on any outcome. A
`STRICT_PASS` receipt does not qualify the web serving path. v5 screen numbers are not
evidence here (pilot variance only). The result covers a **single wrapped hero against
unwrapped opponents** under `promotion-v2-watch-rect` H5000 only. A `STRICT_FAIL` does not
show that v5 harms the hero; say "harm" only when a confidence bound excludes zero.
`src/evaluation/safety_veto.py` (released v2) is not modified by this study; `prepare`,
`validate_intent`, the self-check and the independent audit all require its released
sha256.
