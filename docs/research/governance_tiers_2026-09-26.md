# Tiered research governance, 2026-09-26

Status: **policy draft**. It governs new work that starts on or after 2026-09-26. It does
not reopen, relabel or re-audit any closed study, and it changes no default code path.

Grounding: [portfolio review](experiment_portfolio_review_2026-09-26.md) §1, §3, §5 and §7,
the [strict pilot README](../experiments/task_aligned_strict_pilot_2026-09-26/README.md), and
the [bridge audit-only README](../experiments/task_aligned_bridge_audit_only_2026-09-26/README.md).

## Why tiers

Over Sep 6-26 one governance weight applied to everything (portfolio review §5):

- 12 of 13 invalid studies failed on tooling or rules, not science. Examples include a
  63-character SHA literal, a JSON tuple/list round-trip, heartbeat field names, subTest row
  counts, a watchdog and a KeyError.
- Zero-recovery windows turned each small defect into a new admission. The 6102 serving
  bridge took five admissions over about 1.5 days, and then a roughly 1-hour pilot retired
  the candidate.
- Audits failed on auditor bugs, not producer bugs (anchor_retention, bridge v4,
  native_pretrap, strict pilot).
- Deployment-profile evidence came last. Only 3 of 169 non-foundations experiments ran at
  H5000, and only the strict pilot measured a challenger against Apex.

The fix is to match governance weight to what a result can change. Only Tier 2 can change
the deployed agent, so only Tier 2 carries the full freeze, admission and independent audit.

## Tier summary

| | Tier 0: saved-data diagnostic | Tier 1: development screen | Tier 2: strict promotion |
|---|---|---|---|
| Purpose | Answer a question from data already on disk | Decide where to spend effort; kill clear losers | Change the incumbent or a default |
| Can change defaults or promote | No | No | Yes, and only this tier |
| New simulation | None | Yes, on Tier-1 namespaces only | Yes, on fresh Tier-2 namespaces |
| Typical cost | < 1 min wall, read-only | Minutes to a few hours | Hours to days |
| Pre-registration | One-line question in the receipt | Light intent file (below) | Full existing E2 pre-registration |
| Admission | None | None | Existing admission and freeze |
| Audit | None required | Self-check plus audit dry-run; independent audit optional | Independent audit, required |
| Recovery | Rerun freely | One mechanical-defect recovery | Audit-only re-audit of saved evidence (below) |
| Label on results | `diagnostic` | `screen (non-authoritative)` | Existing strict verdicts |

A result is never promoted between tiers. Tier-1 numbers are never Tier-2 evidence. They may
only inform a Tier-2 design, for example as a variance estimate for power.

## Tier 0: saved-data diagnostics

**Applies when** all of the following hold:

- the work only reads records, logs or checkpoints that already exist;
- it runs no new episodes, training or checkpoint forwards on new worlds;
- it finishes in under about a minute on one machine;
- it writes nothing under any frozen artifact root.

Example: the Apex H5000 death census over the saved strict-pilot incumbent records
(`pilot-v1/output/{development,pilot}/producer/records/incumbent-*.json`). It showed 85 of 96
deaths were self-collision, 9 head-on, 1 enemy body and 1 wall. That census sized the
serving-time safety layer (review §7 item 2, step 1) without any admission.

**Required artifacts:** one receipt (JSON or a short README section) with:

- the question, written before the numbers are read;
- input paths with sha256;
- the script path and git SHA (or the inline command);
- the output numbers.

**Rules:**

- Outputs are labelled `diagnostic`. They can motivate a Tier-1 screen but cannot be a
  screen or promotion result.
- Rerun freely. There is no recovery budget, because nothing is admitted.
- Do not chain diagnostics whose only output is the next diagnostic (review §7, stop doing).
  Each Tier-0 receipt names the decision it informs.

## Tier 1: development screens

**Applies when** new simulation is needed and the result only steers effort. Examples:

- Apex vs Apex plus a capped flood-fill veto at H5000 across the three mixes. This is the
  serving-time safety question in review §7 item 2, step 2. The harness is
  `research/apex_safety_20260926/dev_screen.py` (pre-registration in `protocol.md` beside
  it). It predates the registry below: its worlds use the domain `apex-safety-screen-v1`
  rather than a `screen/<screen_id>/<purpose>` name, it writes `intent.json` and
  `summary.json` (the summary plays the receipt's role), and its disjointness check is
  in-code rather than against a registry. Its seeds should be the registry's first entry.
- An A/A (Apex vs Apex) run to measure null variance for a Tier-2 design (review §7 item 1).
- A training lineage screen (see the statistics standards below).
- The mandatory H5000 check against Apex before multi-day investment (below).

**Required artifacts**, all in one screen directory:

1. `intent.json`, written and hashed **before** any episode runs:
   - `screen_id`, owner, git SHA and a dirty-tree flag;
   - hypothesis and the decision it informs (continue, stop, or design a Tier-2 run);
   - arms, with checkpoint paths and sha256 (the incumbent is always
     `champion_a5_freespace_20260621.pth`, sha256 `43d4e2c5…d747ac93`);
   - evaluation profile name and digest, horizon and mixes;
   - the seed namespace name and the ordered world seeds;
   - primary metric, estimator and decision rule;
   - a compute cap (episodes and wall time).
2. `receipt.json`, written after the run:
   - the intent sha256;
   - per-episode record paths and sha256;
   - counts of episodes, frames and failures;
   - wall time;
   - the primary estimate with its CI;
   - the decision taken under the pre-stated rule.
3. The per-episode records themselves, in the same shape the strict producer writes (keys
   such as `seed`, `world_identity`, `evaluation_profile_digest`, `deaths`,
   `survival_fraction`, `mass_integral`, `max_mass`, `mean_mass_alive`, `kills`,
   `denominators`). Then Tier-1 data can be read by the same loaders as Tier-2 data.

**Rules:**

- Every Tier-1 result is labelled `screen (non-authoritative)`. A Tier-1 result never
  changes a default and never promotes.
- New behavior under test is opt-in and default-off in code. A screen passes it explicitly.
- Secondary metrics may be reported, but the decision uses only the pre-stated primary.
- **One mechanical-defect recovery is allowed per screen.** A mechanical defect is a
  tooling fault that does not depend on outcomes: a crash, a schema or field-name mismatch,
  a receipt or hashing bug, or an audit-rule bug. The recovery reruns the same `intent.json`
  (same arms, seeds, metric and rule) under a `recovery-1` sub-directory. It records the
  defect, the fix diff and whether any outcome of the failed attempt was seen. A second
  defect, or any change to arms, seeds, metric or rule, needs a new intent and a new
  namespace allocation.
- An independent audit is optional. Before the real run, though, the screen's own checks
  must pass on a smoke output (see "Audit scope").

## Tier 2: strict promotion

**Applies when** a result could change the incumbent, a default code path or a deployment
profile. The only admissible outcome is the existing strict E2 gate, as implemented by
`src/evaluation/strict_promotion.py`, `src/evaluation/protocol.py`, `src/scripts/eval_stats.py`
and the `rollout()` path of `src/scripts/tournament_eval.py`. The profile is
`promotion-v2-watch-rect` at H5000 over the frozen, scripted and mixed mixes. This policy does
not weaken any part of it.

**Required artifacts:** as today. These are the full pre-registration (margins, MDE, count
bounds, kill criterion), frozen source closure, materialized rosters, an admission, a create-only
producer, a sealed output and an independent audit.

**Additional requirements from this policy:**

- **Fresh final worlds.** Tier-2 `pilot`, `final` and `serving` seeds must never have
  appeared in any Tier-0, Tier-1 or earlier Tier-2 run. They are checked against the
  namespace registry at construction time (below).
- **Tier-1 prerequisite.** A candidate reaches Tier 2 only after a recorded Tier-1 H5000 check
  against Apex that was not a clear loss (below). The 6102 pilot is the precedent. It lost all
  three mixes (−30.2, −66.4, −34.9 mean mass; 5 of 16 world wins each), and that was visible
  from an hour of H5000 play.
- **Feasibility first.** Pre-registration includes a power estimate from development-only
  variance, such as a Tier-1 A/A run. If the required worlds exceed the cap at the stated MDE,
  do not start. The strict pilot measured a paired Δ SD of 56–118 against a 10% MDE of 3.9–7.3,
  giving 552–8,676 worlds per mix. Until throughput or a pre-registered variance-reduced design
  changes that, most Tier-2 challenges are infeasible by construction.
- **Audit-only recovery.** If the producer completed naturally and only the audit rule is
  defective, one audit-only re-audit of the unchanged saved evidence is allowed. It must use
  an audit fix that has been reviewed and unit-tested against the real saved record shapes.
  The original failed audit is preserved and never relabelled. The bridge audit-only PASS
  on admission-v4 is the precedent. Any producer-side defect still needs a new admission.

## Namespace allocation and construction-time checks

Freshness overlaps were found late in the past: 11 of 3,072 rows, 1 of 127,506 inputs, and 48
raw geometries only discovered after training finished (review §5). The rule is to check
at construction, not at admission.

1. **One registry.** Keep one append-only registry of every allocated world-seed set, for
   example `research/namespace_registry.jsonl`. Each line records the tier, the owning
   `screen_id` or intent sha256, the namespace name, the ordered seeds or a seed-derivation
   rule plus count, and the sha256 of the materialized list. Allocation appends. Nothing is
   ever deleted, including for failed or abandoned runs.
2. **Disjoint Tier-1 namespaces.** Tier-1 screens use their own namespace names,
   `screen/<screen_id>/<purpose>`. They never use the strict names (`training`,
   `development`, `shakedown`, `pilot`, `final`, `serving`). Two screens never share a
   namespace. A recovery reuses its parent screen's namespace. It does not allocate one.
3. **Deterministic derivation.** Derive seeds from a documented rule, for example
   uint32 from `sha256("tier1/<screen_id>/<purpose>/<i>")`, then drop collisions against the
   registry. This makes an allocation reproducible from the intent alone.
4. **Checks at construction time.** The code that builds a screen or strict run does this
   before writing `intent.json`, and fails closed:
   - seeds are unique within each namespace;
   - there is no intersection with any registered seed from any tier (Tier 1 against
     everything; Tier-2 `pilot`, `final` and `serving` against everything, including Tier 1);
   - where world identity is richer than the seed (geometry, roster), the same check runs
     on the materialized world identity;
   - the registry line is appended in the same step, so two concurrent constructions cannot
     both claim a seed.
5. **Read-only reuse.** Tier 0 may read any tier's saved records, because it creates no
   worlds. Tier 1 may reuse Tier-2 `development` records as a baseline (Apex-only episodes)
   when the profile digest matches. It may not run new episodes on Tier-2 seeds.
6. **Tier-2 construction** keeps the existing `_seed_namespaces` disjointness check in
   `strict_promotion.py` and adds the registry intersection check.

The registry file and its checker are not implemented by this lane (see "Not yet done").

## What must be pre-registered

| Item | Tier 0 | Tier 1 | Tier 2 |
|---|---|---|---|
| Question and the decision it informs | Yes, one line | Yes | Yes |
| Arms and checkpoint sha256 | n/a | Yes | Yes (frozen) |
| Profile, horizon, mixes | n/a | Yes | Yes (frozen) |
| World seeds or namespace | n/a | Yes, registry-checked | Yes, registry-checked and materialized |
| Primary metric and estimator | n/a | Yes | Yes |
| Decision rule | n/a | Yes (continue, stop or design) | Yes (margins, MDE, bands) |
| Checkpoints evaluated | n/a | Yes, for training screens | Yes |
| Compute cap | n/a | Yes | Yes, with kill criterion |
| Power or feasibility estimate | n/a | Recommended | Required, from development-only variance |
| Audit rules and their tests | n/a | Self-checks only | Yes, frozen with the source closure |

Anything not pre-registered may be reported only as exploratory. Selection steps informed by
outcomes (which arm, cohort or checkpoint goes forward) are disclosed with the number of
alternatives looked at, as the past write-ups did.

## Audit scope

Audits failed on auditor bugs more often than they caught producer bugs. Two recent cases
show the pattern:

- **`lifecycle_frames` (strict pilot, INVALID_STOP_AUDIT_FAILED).** The audit required
  `lifecycle_frames == 240000` for a completed stage. That counter is charged only by the
  bridge's live-parity probe, which the strict pilot does not install. Every real record
  therefore carried `lifecycle_frames` 0. A reviewer had flagged the rule as a NIT, and the
  integration pass then tightened it incorrectly. One test against one real development
  record would have failed before the run.
- **subTest row counts (bridge admission-v4).** The audit expected one passing `call` row per
  test node (75) and found 82. pytest 9.0.1 logs each passing `unittest.subTest` as an extra
  `SubtestReport` call row, and schema 1 of the producer did not record the report class. One
  test against real pytest output of the frozen fixtures would have caught it.

Rules:

1. **Scope to the decision path.** An audit verifies what the decision depends on:
   - artifact integrity (hashes, seals, create-only);
   - pairing and world identity;
   - denominators and episode counts;
   - profile digest;
   - an independent recomputation of the decision statistic from the raw per-episode records.
   Counters the decision does not use are reported, not gated.
2. **Name a source of truth for every rule.** Each audit rule states which code path writes
   or charges the field it checks, and under what configuration. A rule on a counter whose
   writer is not installed in the audited configuration is a defect.
3. **Share record-shape code, not decision logic.** The producer and audit import one record
   schema and loader module: field names, types and report classes. Then shape drift between
   them cannot happen. The audit recomputes the decision statistic in its own code. Where it
   must reuse the frozen reducer, it cross-checks the reducer against a minimal second
   implementation (a plain mean and t-CI over the paired deltas).
4. **Test rules against real record shapes before use.** Each rule has a positive test on a
   record captured from the real producer path and a negative test on a mutated copy. The
   positive record is either a saved record (for example a strict-pilot `incumbent-*.json`)
   or a smoke output. Synthetic dicts written from the spec do not count as the positive test.
5. **Audit dry-run before GO.** Before any Tier-1 or Tier-2 run, run the full producer and
   audit end to end on a smoke output (at most 2 episodes × 500 frames on CPU) from the same
   source. Tier 2 requires this. For Tier 1, the screen's self-checks do it. Any change to an
   audit rule after review, including a "tightening", reruns the dry-run.
6. **Independence where it counts.** Tier 2 keeps an independent auditor. Tier 1 needs only
   self-checks plus the dry-run. Tier 0 needs none.

## Screen statistics standards (Tier 1)

Past screens used 2–3 correlated seeds, 4–32 worlds and all-seed conjunction gates. At an 80%
per-seed pass rate, a true effect passes a three-seed conjunction only about 51% of the time.
Most "negative" verdicts meant "not established" (review §5).

- **Seeds (training screens).** Use at least 5 independent training seeds per arm, each from a
  fresh initialization. Re-seeded descendants of shared parents do not count. When a warm
  start is the point of the study, use at least 5 independent parents, or treat the parent as
  the unit and say so.
- **Unit and pairing.** All arms play the same screen worlds. The per-seed estimate is the
  mean paired Δ (arm minus reference) over those worlds.
- **Pooling.** The primary estimate pools per-seed estimates with a random-effects CI: a
  Student-t CI over per-seed estimates with k−1 df, or DerSimonian–Laird, reporting τ².
  All-seed conjunctions and sign counts may be secondary, never primary.
- **Checkpoints.** Evaluate at least 3 pre-declared checkpoints per run (for example 1/3, 2/3
  and the full dose), or use the area under the learning curve as the primary. The
  acquire-then-lose dynamics seen in BC→TD handoffs make a single final checkpoint
  misleading.
- **Serving-time interventions (no training, for example the veto).** The unit is the world,
  paired across arms, on all three mixes. Report per-mix paired Δ and a mix-stratified pooled
  estimate (mean of the per-mix means) with its CI. Use at least 16 worlds per mix.
- **Wording.** Report "harm" or "no effect" only when the CI excludes the effect size of
  interest. Otherwise report "not established".
- **Tooling.** Use `paired_stats`, `paired_delta_test` and `paired_delta_pilot_size` in
  `src/scripts/eval_stats.py` instead of ad hoc arithmetic. `src/evaluation/screen_stats.py`
  adds the pooled and sequential tools these standards call for:
  `stratified_mean_of_means` (the serving-time mix-stratified estimate and 90% clear-loser
  flag), `random_effects_pool` with `per_seed_effects` (DerSimonian–Laird, τ², Hartung–Knapp
  CI), `crossed_seed_world_mean` (seeds sharing one world bank), `learning_curve_contrast`
  (AUC and last-k checkpoints), gate-pass planning, and an O'Brien–Fleming-type futility
  plan (`futility_plan`, `run_futility_sequence`). Gate sizing inputs from saved pilot
  records are in `research/gate_calibration_20260926/`.

## Rule: a cheap H5000 check against Apex before multi-day investment

Before any candidate lineage, architecture or intervention gets more than one day of
compute or effort, run one Tier-1 H5000 check against Apex. Serving-bridge qualification and
any Tier-2 preparation count as that kind of investment.

**This section is the normative definition of the mandatory Tier-1 "not a clear loss"
check.** Other documents cite it and do not redefine it. In particular, the sequential
futility looks in `research/gate_calibration_20260926/README.md` belong to a Tier-2 gate and
are not this check.

- **Design.**
  - Profile `promotion-v2-watch-rect`, H5000, frozen, scripted and mixed mixes.
  - At least 16 fresh Tier-1 worlds per mix (the count is fixed before the run), with the
    candidate and Apex paired on each world: at 16, 48 candidate plus 48 Apex episodes, 96
    in total. The Apex arm may come from a cache for the same worlds and code revision.
  - Primary metric: paired Δ `mass_integral`, the same field the strict pilot pairs in
    `strict_promotion.py`.
  - Estimator: the equal-weight mean of the three per-mix mean Δs, with a two-sided 90%
    Welch–Satterthwaite t CI, computed by `src.evaluation.screen_stats.stratified_mean_of_means`
    (`confidence=0.90`). Its `upper_below_zero` and `strata_upper_below_zero` fields are the
    triggers below.
- **Serving.** A research-only harness loads the candidate directly, without full bridge
  governance. If no such harness can serve the candidate, build it first. It takes hours,
  where a bridge takes days.
- **Cost.** Apex took 853 s for 48 H5000 episodes on CPU (about 18 s each), so 96 episodes
  take roughly 30–50 min, depending on candidate speed. The vectorized eval engine is not a
  large saving for this check: once `src/simd_env/eval_engine.py` can featurize vector61, the
  projection in `docs/research/simd_vector61_plan_2026-09-26.md` ("Measured cost") is only
  about 3–4× cheaper than live for all-vector61 mixes, not 100×. That figure is not measured
  end to end, and action parity is not yet shown. Plan world counts on live cost until both
  exist.
- **Decision (clear-loser filter).**
  - Trigger: if the upper bound of the pooled 90% CI is below 0 (`upper_below_zero`), stop or
    redesign the candidate.
  - If any single mix's 90% CI upper bound is below 0 (`strata_upper_below_zero` non-empty),
    continuing needs a written reason in the receipt.
  - Otherwise the candidate may proceed.
- **Resolution.** With the pilot's per-mix SDs (56, 118, 56) and n=16, the pooled standard
  error is about 12, so the 90% half-width is about 21. The check detects only large
  deficits. Passing it is **not** evidence of improvement.
- **Precedent.** 6102's pilot means (−30.2, −66.4, −34.9) pool to about −43.8, which this
  check would have caught in under an hour. The line instead got about 1.5 days of bridge
  qualification and two continuation lineages first.

## Applying this now

- Tier 0, done: the Apex H5000 death census from saved pilot records. Self-collision caused
  85/96 deaths, alive fraction was 0.29–0.37, and per-world mass SD was 56–88.
- Tier 1, next: Apex vs Apex plus the capped flood-fill veto
  (`src/evaluation/safety_veto.py`, mirroring `_apply_free_space_veto` in
  `src/game/scripted_snake.py`), opt-in and default-off. The harness and pre-registration
  are built (`research/apex_safety_20260926/`); the 264-episode run has not been made.
- Tier 1, alongside it: an A/A run for null variance, as input to any future Tier-2
  feasibility estimate.
- Tier 2: only if the Tier-1 veto screen clears the H5000 rule **and** a feasibility estimate
  shows the required worlds fit the cap.

## Not yet done

- The namespace registry file and its construction-time checker are specified here but
  not implemented.
- The shared record-schema module for the producer and audit is not implemented. Neither
  are the real-record unit tests for the existing strict audit rules.
- Nothing in `src/evaluation/strict_promotion.py` or the existing gate changes because of
  this document. The Tier-2 additions (registry check, audit-only recovery, Tier-1
  prerequisite) need their own reviewed change.
