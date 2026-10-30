# T34 spatial doom learnability smoke — 2026-10-30

Status: FROZEN BEFORE FITTING. Prepared under the owner's explicit work order;
operator execute-time scientific/operational admission still applies. No dispatch,
merge, serving change, gameplay, new labels, or pilot-spend authorization here.
This is a saved-data representation diagnostic, not a policy efficacy test.

## Question and fixed data

Can a randomly initialized full-arena spatial U-Net learn H300 death risk on the
existing C/T searched roots? Compare a frozen B8-feature MLP on exactly the same
roots, split, targets, schedule and assessment endpoints. The earlier ~0.6 probe
was a different endpoint; its value is context, not this experiment's comparator.

Use only the supplied revision-02/attempt-02 stage-0001 bank and stage2/attempt-01
eight scored chunks. Require all 2,560 roots, unique root IDs, exact coverage and
legal-mask/B8-action agreement. Reuse the bank's whole-world train/assessment
table (2,048/512 roots); all lives, lookbacks, actions and suffixes of a world
stay together. No random row split, validation on assessment, resplitting,
checkpoint selection, augmentation, hyperparameter search or second seed.
Train on all train roots; primary endpoints use assessment roots with L >= 600.
All-length and individual-action endpoints are descriptive.

The 50,000 unsearched anchors have neither full-arena snapshots nor H300 labels.
Exclude them from both arms. Do not impute doom from death-lookback metadata,
ordinary-root status, or future trajectory. Models see current observations only;
world IDs, partition, tick, root kind, lookback and death metadata are never inputs.

Bank SHA256: `112a2aa4c369ecd0e0f735db410e9ac5a7d330aca09846a50d1a734e6b3175f0`.
Scored stage-0001..0008 SHA256, in order:

```
3b86db00fdbe11e7c20b82992599599a4265bebf47a68aa94be9b84fccb6162c
44ef1bdf31274585eab2f6130e1913faadbf72f07b16d360a54c23c5e8bc1277
6684e30e752ebbee00c302dd308750edacae648bf96ffc7e96b5704655076186
1b2387507412ced28db804bb35c460d7a8e3a9dd62215691c34943768a1d26e3
6fc27dbd38f1da9d03411e1ca1c61d067663c6d32daf2af20b9f1fdf2807a689
d5f49020006e7b3f07d78ea6cb8b7263dba396714615dbcaf08270913dbd4da1
e28d8637b9cd39501523984d94f7b1588905fa6b50998f2298d766b6aaa707eb
cce958ae9a90b19d4e77b5fa471d5b09f8f08e269baed592f63b0512e2937d0b
```

## Labels, models and fitting

Death = 1 - survive within the bank's H300 continuation, with the forced first
action followed by the recorded B8 continuation/opponents. Use train-world
`survive_sel_raw` suffixes 0..3 only: masked BCE against their mean death rate
is exactly BCE averaged over the four binary labels. Assessment uses independent
`survive_score_raw` suffixes 4..7 only. Illegal actions never enter loss or AUC.
No mass auxiliary target in this cheapest smoke.

Spatial: pilot `spatial-student-v1-16p36s` observation, preserving full 83x145 arena,
body ordering, food, scalars and hero-head gather. A smaller residual U-Net with
widths 12/24/48/96 (half the pilot channels), identical three downsample/upsample
stages and 2x4 deep-context pooling; 128-unit head, six independent doom logits.
Random initialization, no B8 weights or features in this arm.

B8 baseline: canonical B8 s3 SHA256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`;
frozen 256-D post-fusion representation from exact bank ego2s-bv8 observations.
Verify recomputed Q against saved Q (absolute tolerance 1e-3, rtol 0).
Standardize features with train-only mean/std (floor 1e-6), then train
256 -> 128 ReLU -> 6 doom logits. No baseline trunk updates.

Each arm: one seed 343010, 3,000 AdamW updates, batch 32, lr 3e-4,
weight decay 1e-4, gradient norm clip 5; uniform train-root sampling with
replacement using an independent CPU generator seed 343011. Same batch indices
for both arms. Float32, no autocast/compile/TF32, no class weighting or learning
rate schedule. Final update only. Save full model, optimizer, CPU/CUDA RNG,
sampler RNG, step, source/data/config identity and hardware every 250 updates
and on SIGUSR1. Finite loss, parameters and gradients required; numerical
failures are INVALID, never a science KILL or automatic retry.

## One Mac scoring look and decision

No held-out labels are uploaded to the GPU. Runtime exports only final assessment
logits and full-state training snapshots. Only the registered Mac `aggregate`
entrypoint combines both final predictions with the locally sealed assessment
labels; it creates its result once, after complete verified exports exist.

Report (1) pooled per-legal-action H300 doom ROC AUC and six individual-action
AUCs; (2) root-level B8-action doom ROC AUC: one prediction at the stored B8 action
per root. For each endpoint, expand each prediction across that root/action's
four binary scoring-suffix outcomes. Thus the target is empirical H300 death
risk, not arbitrary majority-vote doom; each root/action contributes equally,
and suffix replication does not create independent bootstrap units. Ties get
half credit. Report positive/negative counts and parent-world support.

Resample assessment parent worlds with replacement, retaining ALL their roots,
actions and suffixes, 2,000 replicates, NumPy seed 343012. Report two-sided 95%
percentile CIs and paired spatial-minus-B8 AUC CIs using the same draws. Never
bootstrap rows/actions/suffixes independently. Undefined replicates are reported;
require >= 95% valid replicates. Primary endpoints each require positives and
negatives in >= 10 distinct assessment worlds; otherwise INCOMPLETE_SUPPORT.

Precedence: INVALID (integrity/numerics) > INCOMPLETE (missing final updates,
predictions or units) > INCOMPLETE_SUPPORT > science bars. On the spatial
late-game assessment endpoints, let A = min(pooled per-action AUC, B8-action AUC):
GO if A >= 0.85; GRAY if 0.75 <= A < 0.85; KILL if A < 0.75. Equality at 0.75
is GRAY; equality at 0.85 is GO. Point estimates decide the smoke; CIs quantify
uncertainty and are mandatory. B8 comparison and all-length sensitivity do not
change the bars. GO admits only planning the next student test, never a release.

## Runtime, transport and completion

H100-first / A100-fallback only, via qualified tools/ops/dispatch.sh. No RunPod.
Useful compute forecast <= 45 min; the scientific worker stops at 45 min from
entry (all loading/fitting/inference included), preserving a resumable snapshot.
An unfinished fit is INCOMPLETE, never an abbreviated scored fit. Whole session
base estimate/timeout 60 min; owner 2x effective runaway timeout 120 min. Rates
are provisional ceilings H100 12 / A100 8 u/h (not throughput measurements).
Base/effective caps H100 12/24 u, A100 8/16 u; base/effective reserves 2/4 u
H100, 4/3 and 8/3 u A100. Units are informational, not a rationing gate.
The <1 A100-hour premise refers to useful original smoke compute; setup,
canaries, owner 2x guards and any mechanical completions are reported separately.
No claim of measured GPU throughput before the same-pin hardware canary.

Separate same-pin same-input family canary: synthetic labels/observations only,
three real forward/backward/optimizer units per arm plus final inference and
SIGUSR1/resume tests offline. Base/effective whole-session 15/30 min, base/effective
H100 cap 3/6 u, A100 cap 2/4 u; reserves follow one-sixth of cap. Its output is
excluded from science. Real data files remain staged unchanged for transport
coverage but are not fitted by the canary. Bind genuine stopped H100 and A100
receipts before main admission. Never fabricate a receipt or waive qualification.

First attempt uploads an empty verified ledger. Namespace
`t34-doom-auc-20261030/v1`, frozen allocation index 0. Deterministic immutable
model/step unit IDs, payload + SHA sidecar + completion marker published last.
Mac ledger builder uses receipt-bound path/size/hash only, validates identity,
refuses duplicates/foreign allocations, and uploads exactly verified units.
Resume the latest verified full-state snapshot of each arm; never recompute a
verified final prediction. Cross-hardware/library training resume is refused.
One original plus at most two operator-controlled completion attempts, eligible
only after verified mechanical failure, verified stop, and missing work; fresh
manifest/output/receipt each time. No outcome-dependent extension or automatic
completion retry. Preserve the original N, 3,000-update schedules and scoring look.

Preparation must retain exact raw input hashes, lossless compact conversion
checks, split audit, sparse source and import preflight, measured Mac tiny real
entrypoint timings, size/transport projections, tests, review, dry-run output
and exact live commands. SIGUSR1 has a 15 s checkpoint grace; completed-work
watchdog <= 900 s; progress <= 60 s; export cap 256 MiB. No dispatch in this task.
