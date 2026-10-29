# P watch-lead successor — DRAFT pending target canary and independent review

**DRAFT, NOT RATIFIED. No successor dispatch, data collection, allocation date,
scientific manifest, completed runtime estimate, or release authority.** Astra is
the designer. This document deliberately precedes target-host measurements only
as a prospective design; executable successor construction and dating wait for
those measurements and independent review. The engineering probe is separate.

## Authority and closed predecessor

Use T34 `SUCCESS_CONTRACT.md`, `DESIGN_V2_SHADOW.md` (+ Amendment 1),
`P7_STATISTICS.md`, the strategy ruling's P premise, and the standing resume/2×
rules in `AGENTS.md`. Superseded historical bare-only goals do not close Claim B.
P 09e is INVALID/INCOMPLETE (21/32 complete sessions; completion cap exhausted).
Those 21 sessions and the entire old P bank remain CLOSED: no outcomes opened,
no P0 descriptives, no variance/power estimation, no ranking, no reuse or pooling,
no third completion, and no reinterpretation at N=21. The closeout's operational
timing evidence motivates a target-host profile; it supplies no efficacy result.

## Fixed prospective question and design

Does delivering the same W1/K1 search recommendation 48 frames ahead reduce hero
deaths by at least 10%, without more than 5% relative horizon-mass loss, compared
with bare B8 s3 in ordinary continuously respawning mixed Watch6 sessions?

| Item | Prospective commitment |
| --- | --- |
| Endpoint | Exactly 32 paired sessions × 3 arms × 6,000 ordinary Watch ticks; fixed N, one terminal Mac look |
| Arms | bare B8 s3 comparator; P48 sole primary; P0 descriptive only, never rescues P48 |
| Hero | B8 s3 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`, fp32, CPU |
| Opponent | Mixed Watch6 frozen FRP3 s12 `eec144bf92509a42664e2d650b8741f7d7774b86d7b6601d012bdd9425dd3723` with existing v8 roster semantics |
| Mechanics | Actual GameSession, mechanics v2, ordinary respawns, private copied RNG; no action tape |
| Search | Existing `watch_lead_p/engine.py`, `live.py`, `protocol.py` semantics: H300, trigger alive/L≥600, ≥5-frame root spacing, W1/K1, delays 0/48, same acceptance/drop/fallback and 16 MiB root limit |
| Scientific unit | Complete three-arm paired session; all arms of a session on one host CPU model and software identity, matching initial state |
| Proposed fresh namespace | `a100-dev/p-watch-lead-successor-v1/`, canonical `world_seed("p-watch-lead-successor-v1", i)`, i=0..31; freeze exact allocation and collision audit only after engineering acceptance |
| Exclusions | Entire predecessor bank (32 + engineering indices 32..35), all `p-successor-profile-v1` and `p-successor-profile-v2` engineering seeds and any later engineering stress seeds, protected registry and full available historical seed inventories, TRAIN01 monitor/nomination |

Pre-data seed audit must reject a whole colliding allocation; increment a
prospectively recorded round key, never replace one selected seed. Document audit
coverage and opaque/unrecorded history limits. No successor seed is executed
during profiling. The fixed 32 is the inherited design proposal, not a power
claim derived from unobserved or closed P data. Any sample-size/design change
requires independent prospective ratification before allocation/data.

## One-look statistics and decision order

Retain the old P statistic definition, independently check it before freezing the
new reducer. For each paired session i, let d_bi,d_pi be hero deaths over all
6,000 ticks, and m_bi,m_pi be mean mass over the full horizon (dead ticks zero).
Benefit = `1 - mean(d_pi)/mean(d_bi)`. Relative mass is
`mean_i((m_pi-m_bi)/m_bi)`, an equal-session paired mean, not a pooled mass ratio.
Resample whole matched sessions, 10,000 bootstrap replicates, RNG seed 3431,
quantile method `inverted_cdf`. Benefit limits use 0.10/0.90 quantiles; mass limits
0.025/0.975. For every zero-baseline-death resample, use -infinity for the lower-bound
sample and +infinity for the upper-bound sample, regardless of candidate deaths,
as in the frozen P implementation; do not silently change that statistic.
Any baseline session mass <= 0 is invalid. These conservative extended-real
limits mean unsupported benefit is INCONCLUSIVE unless the mass STOP rule applies.
Nonfinite/invalid payloads, changed identity/recipe, duplicates, foreign units,
or missing sessions take precedence as INVALID/INCOMPLETE.

1. INVALID/INCOMPLETE first, including exhausted time/completion attempts with
   fewer than 32 verified paired units. No partial-union effect computation.
2. STOP this delayed controller if supported benefit UB90 < 10% OR mass upper95 < −5%.
3. GO to real paced development if benefit ≥ 10%, LB90 > 0, and mass lower95 ≥ −5%.
4. Otherwise INCONCLUSIVE, with no automatic expansion, seed replacement or
   checkpoint/controller selection. P0 cannot rescue the primary.

This is development evidence only. GO admits actual Mac paced development,
not timing/service qualification, strict promotion, serving changes or release.
The T34 Watch product contract and Mac strict/serving chain remain binding later;
only the owner may authorize release.

## Dependency graph and measurement-derived envelope — fill after canary

| Order | Work / placement | Acceptance / decision enabled |
| --- | --- | --- |
| 1 | Sealed short exact-P engineering probe, actual Colab L4 CPU host; optional separately priced A100 host comparison | Terminal PASS, verified timing exports, stop_verified and disappearance, full fixed worker grid and trajectory hash agreement; measure host shape, CPU/wall, contention and rates |
| 2 | Target-platform excluded long-body/full-session and late-branch engineering stress, if step 1 lacks tail coverage | Must measure construction/finalization, mature and long-body costs and costly H300 alternatives; no full-study price from a 64-frame window alone |
| 3 | Fill formula terms below, choose useful measured worker count and bounded disjoint session shards; independent Argon code review and Fable prospective ratification | Freeze inputs/namespace/endpoint/reducer/shards/per-attempt envelopes before any successor data |
| 4 | Implement and offline-test successor resume/export/reducer, then same-pin target canary | Current source/manifest pin, corruption/foreign/duplicate refusal, exact skips, full-N look gate, all pass |
| 5 | Operator only: dispatch fresh session shards through qualified Colab supervisor | All account assignments ≤3; no occupied slot assumed or stopped; no automatic scientific completion |

For host h and measured concurrency w, record `s[h,w,a]` = maximum measured
inclusive seconds per frame for arm a over the frozen relevant engineering
regimes, `c[h,w,a]` = measured per-arm construction + finalization cost,
`q[h,w]` = measured contention factor at w / serial (do not apply twice if s is
already measured at w), `L[h,w,a] >= 1` = measured tail-stress/full-session uplift,
`q_residual[h,w,a] >= 1` = additional contention uplift whenever the measured
window's time-weighted active concurrency is below w or sustained search pressure
is not demonstrated (it is never assumed to be 1 from a nominal worker label),
and setup/export/control terms from the actual supervisor receipt. All are
**UNFILLED** until target measurements exist. Do not credit GPU acceleration:
the science executes on CPU on both accelerator rentals.

```
arm_forecast[h,w,a] = 6000 * s[h,w,a] * L[h,w,a] * q_residual[h,w,a] + c[h,w,a]
shard_forecast[j] = measured_scheduler_makespan(
    frozen_session_indices[j], all_three_arms, arm_forecast[h,w], w)
    + allocation + warmup + credential_launch + rate_calibration
    + worst_staging + hardware_check + bootstrap + poll + worst_export + 70 seconds
base_timeout[j] = ceil(shard_forecast[j] / 60) minutes
effective_timeout[j] = 2 * base_timeout[j]
base_cap[j] = ceil_to_0.01(base_timeout[j] / 60 * prepriced_rate_ceiling[h])
effective_cap[j] = 2 * base_cap[j]
base_reserve[j] = 0.20 * base_cap[j]
effective_reserve[j] = 2 * base_reserve[j]
aggregate_runtime_hours = sum_j(shard_forecast[j]) / 3600
units_forecast = sum_j(shard_forecast[j] / 3600 * calibrated_rate[h])
series_guard = sum_over_frozen_original_and_2_completion_envelopes(effective_cap)
```

Keep measured estimates/rates/N unchanged by the 2× rule. Include setup, idle,
telemetry and exports in whole-session caps; fit exports inside both VM and Mac
deadlines, including the fixed 45/25-second allowances. Source ≤32 MiB compressed
and unpacked, staging ≤300 s and native probe transfer ≤150 s; exports bounded
and completion-marked. A large cap does not guarantee Colab uptime. No current
39-unit/day or per-night rationing applies; registered science caps and timeout
guards remain fixed after data. Compare price per useful complete paired session
and time to full N, not accelerator labels or speedup alone.

The exact number of shards, indices per shard, workers, host, per-attempt timeout,
rate ceilings, retry envelopes and series totals remain formulas/placeholders.
Do not manufacture an executable main manifest or put a date of qualification on
this draft. Choose shards small enough to complete with demonstrated tail margin
inside available runtime lifetimes; account-wide parallelism is at most three,
including other projects. Never change a shard boundary based on outcomes.

## Resumable per-session shard contract to implement after measurements

Freeze canonical unit ID = namespace + allocation index, allocation digest,
controller/science fingerprint, input hashes, host/software identity, and exact
session index assignment per shard. Use session-grouped FIFO tasks (i,P0),
(i,P48),(i,bare) before the next session so whole units retire throughout a shard. A unit is one complete three-arm session.
Each session is published immediately as an immutable atomic payload and public
construction metadata, SHA sidecars and completion marker LAST. An interrupted
session is discarded whole; no arm reuse, trajectory stitching or partial-arm
resume is proposed. Engineering snapshots never become scientific units.

First attempt uploads an empty verified ledger. Later ledgers are made on the
Mac from path/size/hash and public metadata only, never effect estimates.
Runtime skips exactly Mac-verified IDs, refuses foreign namespace/allocation,
source mismatch, changed initial pairing, duplicate IDs and corrupt/partial units,
and never recomputes a verified session into the result set. Do not reuse the old
P module-level SEEDS/NAMESPACE or its old ledger: successor wrappers and admission
must explicitly bind the new allocation. Same-platform pairing is mandatory.

Every main manifest uses the standing uploaded `resume` block with
`unit_kind: paired-Watch6-session`, `ledger_path: inputs/verified-ledger.json`,
`skip_verified: true`, `max_outcome_blind_retries: 2`, wired to the actual runtime
path. Define paths consistently with supervisor input destination semantics.
One original plus at most two MANUAL completion attempts PER SHARD, globally
bounded by the frozen series envelopes, only after verified mechanical failure,
verified stop/disappearance and metadata-only missing-unit checks. Mechanical
triggers: timeout/lease expiry, RPC/transport failure, process crash or incomplete
export with integrity refusal; no scientific failure retry. Use fresh manifests,
output directories, broker state and receipts. Preserve every attempt. The ops
wrapper's exact accounting allocation retry is separately accounted; it never
becomes a scientific completion retry. No retry launches automatically.

The registered Mac reducer alone validates the complete unique union of indices
0..31 across all shards/attempts, refuses duplicates and foreign units, then takes
one terminal look. Remote summaries and per-session outcomes stay sealed until
that gate. Stop/exhaustion with N<32 is INVALID/INCOMPLETE, never INCONCLUSIVE.

Before ratification: empty-ledger, exact skip, corrupted hash, partial marker,
foreign namespace/allocation, duplicate union, missing index, interruption,
cleanup, serial/parallel trajectory identity, host pairing and no-partial-look
tests; current-pin engineering/transport receipt; independent cross-review.
The present preparation implements only the engineering probe, not this successor.
