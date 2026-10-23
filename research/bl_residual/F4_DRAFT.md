# F4 draft — nested rescue-life dose study for a frozen-B8 residual

**Status: DRAFT FOR ASTRA REVIEW — UNRATIFIED.** This is a proposed Tier-1
development registration, not authority to collect labels, fit a model, allocate
Colab, run the gameplay screen, change serving, or release a checkpoint. F4 must
be reviewed by Astra and then independently judged before an immutable
registration replaces this draft. In particular, **no factory label may be fitted
before that judgment.**

## Question and scope

Does increasing the number of independent, rescue-bearing natural lives from 256
to 1,024 and then 4,096 make the same frozen-B8 residual materially stronger
than bare B8 s3 in fresh paired H5000 gameplay? The comparator is bare B8 s3,
SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
The endpoint is full-horizon terminal-hero mass integral (dead frames are zero),
not AUC, agreement, a teacher result, or an offline label metric.

This is Claim A development evidence only. A screen result neither promotes a
candidate nor changes the served B8 s3; a later fresh Mac strict gate, serving
qualification, and owner approval remain required. Path B, the Round-23 Mac
schedule, and all serving configuration are outside this package.

The proposal implements the Round-24 judgment's new mechanism: exact
counterfactual labels from natural B8 lives, a zero-initialised residual over
frozen B8 features, ordinary-row preservation, and a dose contrast. It is not a
rebranded vector61 fine-tune, a learned simulator, a decision-time veto/search,
or a substitute for the strict gate.

## Preconditions that must be sealed before any data or fit

1. F0 must have passed its digest and bare re-roll audits, and F1 must have chosen
   a qualified label placement. Its immutable, per-life exports must identify
   exact-simulator code, B8 and opponent checkpoints, kernel attestation,
   platform, parent world, life, stream, sampling probability, full digest, and
   the verified resume ledger. A missing or foreign/duplicate completion marker
   makes the tranche INCOMPLETE or INVALID as applicable.
2. F2 must publish its T5 action-interface disposition. The 4,096-life
   collection is purchased only if F2 does **not** strike volume as evidence
   (`25 -> 100%` selected-action-survival change has a positive lower bound), or
   if the 1,024-life fit beats the 256-life fit in the F3 harness. The latter
   condition is prospective and does not turn an offline result into gameplay
   evidence.
3. Before tranche 1 is read, freeze and hash the F3 implementation, the B8 s3
   checkpoint, exact label schema, parent allocation order, split assignment,
   feature-extraction code, the evaluated controller export path, the evaluator,
   and this replacement registration. A synthetic canary may test plumbing but
   cannot be admitted to this look.
4. Allocate fresh development and later strict namespaces separately. Run the
   documented seed-collision audit before gameplay; one collision rejects the
   whole allocation and takes the next consecutive round. The label/training,
   held-out, development-screen, and strict-gate parent-world sets are mutually
   disjoint.
5. F0 and F3 reuse the existing exact path unchanged: `b2_engine.snapshot()` /
   restore, `clone_world_exact`, `full_digest`, and `exact_stop.run`. They do not
   introduce a second simulator, approximate branch engine, or altered mechanics.

## Data unit, ancestry, and nested doses

A rescue-bearing life is one hero death in a natural parent world for which the
factory records at least one exact alternative that survives the registered H300
horizon. All restored roots, complete-branch audits, H1440 extensions, and any
future descendant remain on their parent world's side of every split. Roots,
rows, branches, frames, and alternative actions do not create additional
independent lives.

The named doses are nested **selected support** counts of rescue-bearing lives,
not a promise that collection itself stops at 256, 1,024, or 4,096 lives:

| Dose | Selected support lives | Fit-eligible lives after the fixed 64-life reserve | Purpose |
|---|---:|---:|---|
| D256 | 256 | 192 | readiness floor and first fit |
| D1024 | 1,024 | 960 | lower-dose gameplay comparator |
| D4096 | 4,096 | 4,032 | top dose and primary dose contrast |

`research/bl_residual/data.py:allocate_lives` consumes only identity, stream,
rescue-bearing flag, and an append-only `allocation_index`. Its five-fold parent
hash keeps every descendant together; fold 0 is the held-out side. At D256 it
reserves the first 16 fold-0 S-Watch6 lives and then 48 other fold-0 lives, for a
64-life reserve. The fit side is the allocation-index prefix from the other
folds. Later tranches append indices only, preserving the original catalog and
the D256/D1024 prefixes. A five-fold selection will normally require a collection
pool larger than the selected support (D256 may require about 320 collected lives
to yield 64 fold-0 plus 192 fit lives). The allocation receipt must report the
actual collected, selected, and unused counts by stream and parent; it may not
replace missing lives with an outcome-selected parent. If the collection cap is
reached without the required selected support, the tranche is `INSUFFICIENT`.
That result grants neither refill nor a cap increase.

The rescue-life catalog and the extra non-rescue natural archive are distinct
inputs. The latter contains ordinary natural-stream rows from all eligible
held-out parents and supplies the natural-prevalence denominator; it neither
changes a dose nor fills the rescue-life reserve.

The fit inputs are only the F0 factory products: decisive rows with exact
counterfactual outcomes and ordinary complete-action rows carrying H300 mass.
`NOT_EVALUATED` and `UNKNOWN` alternatives are never treated as deaths or as
negative preferences. H1440 audit outcomes, held-out labels, and screen worlds
remain sealed until the corresponding candidate is frozen.

## Candidate and frozen training recipe

Every dose starts from the same B8 s3 weights and adds a linear six-logit residual
to B8's Q logits in the same forward pass. The frozen 256-dimensional fusion
embedding is `research/a100_a2/model.py:b8_features`; all B8 parameters are
non-trainable. The residual's weight and bias are initialised exactly to zero, so
the pre-fit controller is behaviorally bare B8. The exported candidate is one
B8-compatible checkpoint containing `Q_B8 + residual(features)`, with no
decision-time teacher, search, veto, or opponent-ID input.

The actual F3 recipe below must be bound by its source and recipe digests before
ratification:

| Field | Proposed frozen value |
|---|---|
| Optimizer / numeric mode | Adam, fp32 |
| Learning rate | 0.001 |
| Optimizer updates | 2,000 for every dose |
| Batch size | 256 |
| Fit seed | 3403 |
| Decisive objective | Every exact, legal, served-selectable survivor is preferred to every exact, legal, served-selectable dying action on an exact bare-death root |
| Pair weight | `sampling_weight * (1 + positive mass gap)`; mass gap is capped at 4 and the final weight at 5 |
| Ordinary objective | Complete ordinary exact H300 mass preference, weight 1.0 |
| Retention objective | KL from corrected to B8 Q distribution on ordinary rows, weight 1.0 |
| Residual regularizer | L2, weight 0.0001 |
| Small-dose treatment | Repeat the deterministic fit sampler as needed to reach 2,000 updates; do not add updates for larger doses |
| Selection | No early stop, checkpoint selection, seed swap, recipe sweep, or data-dependent threshold |

The served action uses B8's advisory mask and the normal-action fallback when its
mask is empty. `NOT_EVALUATED` and `UNKNOWN` actions are never dying labels or
negative preferences. F3 hashes its sampling/repetition order, non-finite
handling, and checkpoint schema. Any change is a new design requiring review
before labels are read.

Each fit must emit a recipe/source digest, optimiser/update counters, all
tripwire records, input allocation/ledger hashes, and exactly one final
checkpoint. Non-finite loss, Q, or gradient; illegal action; or more than 90%
one-action share in a 100k-decision window kills that fit. The T34 fit-drift
tripwires also apply: more than 5% disagreement from the previous student in any
length decile, boost share more than 1.5 times the previous student, or held-out
decisive agreement below 70% on the registered 32-world probe kills that fit.

`fuse_residual` exports a normal B8 checkpoint by folding the residual into the
dueling value/advantage algebra. This is a float32 operation reassociation, not
a bit-exact identity claim. The held-out admission check therefore reloads the
exported checkpoint and evaluates its fused Q outputs, rather than relying on
the training wrapper.

## Offline held-out check — admission only

For each final dose candidate, report held-out selected-action regret and rescue
retention on the selected held-out rescue IDs; report harmful-override rate on
the separate natural archive whose denominator is all eligible decisions from
held-out parents, not decisive rows or overrides. The p/q sampling weight is
capped at 5 and its effective sample size (ESS) is reported. A changed selection
whose candidate or B8 outcome is `UNKNOWN` is charged conservatively to the
harmful upper rate.

Its predeclared offline admission condition is held-out rescue retention at a
harmful-override upper rate no greater than **0.01 per 1,000 natural-stream
decisions**. Admission is unavailable, rather than passed, unless the complete
natural archive and reloaded fused-checkpoint predictions are present. Report a
10-bin score-reliability table, ECE, and Brier score as explicitly uncalibrated
diagnostics; no score is called a calibrated probability.

Failure of this check prevents that candidate from entering the gameplay screen.
Passing it authorizes only the already-registered development evaluation; it is
not a GO, a dose result, a release claim, or evidence from which to choose another
checkpoint.

## Fresh paired gameplay development screen

After candidates are frozen, create one fresh collision-audited Tier-1 namespace
with 128 terminal-hero H5000 worlds in each of the scripted, frozen, and mixed
rosters. The screen has 384 paired worlds. For every world, bare B8 s3, D256,
D1024, and D4096 use the same ordered roster, hero slot, mechanics, wrapper, and
initial state. Each arm starts from an exact clone; no arm's outcome can alter
another's state. The bank is a single fixed-N, one-look screen with no extensions
or added worlds after outcome access.

The primary efficacy values are paired per-world mass-integral differences:
`G_top = D4096 - B8` and `G_dose = D4096 - D1024`. Equal-weight the three mixes.
Use the fixed-controller paired-world frame in P7: report every per-mix estimate,
standard error, one-sided 90% lower and upper bounds, and the pooled effect. The
planned inference is a paired-world, mix-stratified analysis with one-sided 90%
bounds; a hierarchical paired bootstrap is sensitivity only. The final registration
must state the exact covariance-aware pooling calculation if it reuses world IDs
across mixes. It must apply Holm correction across every inference-bearing member
of the declared efficacy, mix, survival, and product-guard families (and across
looks, if any) before judging the corresponding bar. It must not apply a
post-hoc favorable subset of the guards.

For D4096 vs B8, the skill and product guards are all binding:

- pooled mass point estimate at least **+40** and Holm-adjusted LB90 **> 0**;
- no mix has Holm-adjusted mass UB90 **< 0**;
- survival guard A: pooled alive-frame margin at least −0.05 and single-mix
  margin at least −0.075, both vs B8 s3;
- survival guard B: scripted no-death gain at least **+0.03**;
- delivered boost share in **[0.5%, 10%]**;
- on fresh continuous-respawn mixed Watch6 sessions versus B8, deaths/1k has a
  paired 95% interval excluding zero in the favorable direction (target at least
  20% fewer), and full-horizon hero-mass relative-difference lower 95% bound is
  at least −5%. Median/p95 length and late large-body deaths are reported.

The D4096-vs-D1024 dose contrast passes only if its paired mass point estimate is
at least **+20** and its Holm-adjusted LB90 is **> 0**. This is a dose effect,
not a substitute for `G_top`. Every result reports survival/product guards for
both candidates; a product failure cannot be offset by a mass gain.

## Ordered result labels

The immutable analyzer must use this precedence, after verifying the complete
ledger and one permitted look:

1. `INVALID` — bad digest/lineage/seed audit/parity, a foreign or duplicate unit,
   an unsealed output, wrong controller identity, or unregistered recipe.
2. `INCOMPLETE` — a required fit, holdout reserve, screen unit, product session,
   audit, receipt, or required statistic is absent. Do not shrink N.
3. `INSUFFICIENT` — the registered collection cap was reached without the
   selected support needed for the named dose. Do not refill, resize the dose, or
   increase the cap from this result.
4. `KILL_FIT` — a P7 fit/controller tripwire fires. It kills that fit/arm; it is
   not recast as an integrity-invalid result.
5. `KILL_DOSE_CLOSED` — `UB90(G_dose) <= 0` after the registered Holm treatment.
   This closes the distillation lane as designed; it does not purchase more
   labels, rerun a fit, or try another head.
6. `GO_TOP_DOSE_DEV` — all D4096-vs-B8 skill, survival, product, multiplicity,
   and controller-integrity bars pass and the D4096-vs-D1024 +20/LB90 bar passes.
   This buys only a separately registered strict-gate plan.
7. `INCONCLUSIVE` — every other complete result, including a positive dose trend
   that misses +20, a top-dose effect missing +40/LB90, or a guard failure. It
   grants no extra collection or release.

## Power evidence and limitation

This draft uses observed, not invented, variability as a feasibility reference.
The completed L2′ paired mass analysis reports standard errors 16.455 (frozen,
n=64), 11.960 (mixed, n=64), and 38.566 (scripted, n=32); its pooled-equivalent
mass SE is 14.534. T23 reports five equal-mix training-seed effects with sample
SD 22.277 and SE 9.962, but it is a different five-seed training-recipe screen,
not a fixed-controller dose comparison. Neither source supplies the variance of
`D4096 - D1024`, which is the F4 contrast that matters.

Accordingly, 128/mix is the Round-24 fixed development-bank commitment, not a
claimed power guarantee. Before ratification, Astra must either derive the
paired-dose power from a sealed, admissible variance source without opening the
new screen, or explicitly retain this fixed-N falsifier and record its operating
characteristic limitation. This draft does not convert L2′ or T23 outcomes into
new evidence or re-score either bank.

## Compute, transport, and caps

Label collection is CPU exact-simulator work; GPU hardware is not used to create
labels. F1 selects CPU runtime or re-qualified L4 placement. Fits may use an L4;
an A100 fit is allowed only after a cached fit demonstrates at least fivefold
speedup. All Colab work uses the qualified CLI tree recorded in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`
(currently commit `f21a92edb86550df5263dbed4be7e2ff0bfa4c17`, tree
`553af3f692515b2bf66173356fe4d099acc96942`), passes the clean-venv import
preflight at `tools/ops/import_preflight.py`, and is dispatched by Claude only
through `/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/dispatch.sh`.

Every future manifest binds its sources and input hashes, an empty then
Mac-verified resume ledger, and the required `resume` declaration (`unit_kind`,
ledger path, `skip_verified: true`, two outcome-blind completion attempts). It
uses fresh job/output/state names, immutable per-life exports, SHA sidecars,
completion markers last, telemetry, terminal receipt verification, and explicit
runtime release. It may never mount Drive or directly invoke the Colab allocator.

| Item | Base envelope | Effective envelope | Interpretation |
|---|---:|---:|---|
| F0/F1 canaries | Judgment-controlled; no F4 increase | 2x once | Separate prerequisites, not a fit budget |
| D256 label tranche | <=40 u | <=80 u | Current Round-24 ruling; first cap wins |
| F3 harness before labels and judgment | 0 u | 0 u | Synthetic plumbing only; factory-label fits are refused |
| Fits across the F4 screen after ratification | <=10 u | **<=20 u** | The ruling's unqualified `fits <=10 u` is base and the owner’s 2x rule applies once |
| D1024/D4096 collection | Not admitted by this draft | Not admitted by this draft | Requires F1 placement forecast and F2/256-to-1024 condition, then a reviewed immutable registration with base/effective caps before data |
| Development gameplay bank | Mac-only development work | 0 Colab u | Scheduled only in a permitted Mac gap; no timing or strict gate is armed |

“Effective” is exactly twice “base,” applied once to a forecast/cap/reserve/timeout;
it does not double N, update count, measured rates, or the two completion retries.
The D1024/D4096 missing envelopes are a deliberate review block rather than a
license to use the judgment's broad cost ranges as a cap. Any ratified successor
must state both envelopes and retain the owner’s first-cap-wins rule.

The root job accepts an explicit staged requested-dose list. A later-dose
allocation/dispatch may create only the new dose while retaining the immutable
checkpoint references for every verified lower dose; it must never recompute or
replace a verified D256 or D1024 fit.

F3's existing gameplay hook is intentionally not the scientific decision
analyzer: it emits `NOT_A_DECISION`, records boost share as `UNAVAILABLE` because
the current evaluator has no action trace, and refuses a non-fixture screen until
a hash-bound ratified F4 successor is supplied. The implementation of the
scientific analyzer and the fixed continuous-respawn product-session N therefore
remain explicit F4 review requirements, not claims that this draft has already
armed them.

## Review questions for Astra and the independent judge

1. Confirm the selected-support convention: 256/1,024/4,096 selected lives with
   the fixed 64-life reserve (192/960/4,032 fit-eligible), while collected and
   unused lives are reported separately. Astra may instead recommend a
   prospective collector-quota split that removes this fold-selection waste.
2. Bind the F3 source and recipe digests, including the actual capped-gap,
   natural-archive, and harmful-override definitions.
3. Confirm the covariance-aware paired-world/Holm calculation for the shared
   128/mix bank and whether the Watch6 product-session count must be fixed here
   rather than by the successor registration.
4. Approve or replace the fixed-N power justification without inspecting new
   labels or gameplay, and set the D1024/D4096 base/effective collection caps.

## Source evidence

- Round-24 authority: `JUDGMENT.md` §§3.2–3.6 and §4 (nested lives, F2/F3/F4
  sequence, +20/+40 kill numbers, 128/mix H5000 bank, first-tranche and fit
  envelopes); `JUDGMENT_fable_prior_0010.md` §3.5 agrees on the core plan.
- B8 embedding interface: `research/a100_a2/model.py:b8_features` returns the
  frozen 256-D fuse embedding.
- T34 bars and fixed-controller inference:
  `docs/research/final_run_20261019/SUCCESS_CONTRACT.md` §§1–4 and
  `docs/research/final_run_20261019/P7_STATISTICS.md` §§1–5.
- Observed variability only:
  `docs/research/results/L2prime_mac_analysis_20261022.json` and
  `research/t23_r0_screen_20261017/results/t23_verdict.json`.
- Transport and resume contract: `AGENTS.md` “Operations — Resume-by-default”
  and `/Users/josenunez/Projects/ml/snake-dqn-acctfix/tools/ops/README.md`.
