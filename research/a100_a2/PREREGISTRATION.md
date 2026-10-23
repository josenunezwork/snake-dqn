# A2 scale and target falsifier — frozen before label access

Owner-authorized A2 preparation, 2026-10-22; authority: external big-lever/PLAN_A100.md
§3 A2 and round-20/JUDGMENT.md, read in full. Dispatch is Claude's alone after
independent Codex-V review and same-pin A100 transport canary. This intent is
committed before this package opens any historical label payload. No sealed study
or gate outputs may be read. Exploratory historical diagnostics only; no serving claim.

## Inputs and ancestry

T5/T8 features SHA e4ecbe991061c90bb2ae3cdb4b236d0025f1219d61fa806b7e96f43962d9e02a:
19,976 states, 103,264 legal branches, 342 parent worlds. Preserve the published
five folds verbatim. Join raw observations by canonical state identity from the
hash-checked T4e bank, never by row order alone. Recorded branch death within
300 is D300. Recorded alive steps define event interval (alive_steps + 1);
survivors at 300 are right-censored. Horizons: 30, 60, 100, 200, 300.

Second table: only the explicitly permitted excluded-A5 diagnostic exports used
by T35 (manifests-final-3ebf3991/a5-l4-s{0,1}-output), 39 worlds / 190,779
anchors / 296 searched rows / 225 decisive rows in 21 worlds. Never discover
other T34 study directories. Existing frozen-feature cache is permitted; raw
observations, if absent, are reconstructed on Colab from these exact excluded
exports through the audited replay seam, with no new gameplay labels. Train on
searched branches only (not untested anchors); report all anchor counts. Keep
T35's five ancestry-grouped folds; all branches/descendants share a parent fold.
A5 is a separate table, never pooled into T5. Any failed identity, count, hash or
replay validation is INVALID; missing evidence is INCOMPLETE, never N shrink.

## Frozen factorial and fit endpoint

Two seeds: 20261022, 20261023. Encoders: C0 B8-s3 frozen 256-D trunk + linear
head; S-random B8-shaped trainable encoder; S-B8 same initialized from pinned
B8-s3 (SHA 3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8);
M-random approximately 6.5M, widths 64/128/128 and one extra 3x3 block per tower,
local bottleneck 256, global bottleneck 256, fuse 512. No new planes.
Each encoder has D300 (6 logits) and hazards (6 x 5 logits). For each held-out
fold, SHA256(namespace + parent-world) orders other worlds; ceil(25/50/100%
world count) prefixes are nested and shared across arms/seeds. No outer-fold
rows in optimization, standardization or model selection.

Fixed final endpoint: 20 epochs, batch 256 states, AdamW lr 0.001, weight decay
0.0001, fp32, gradient clip 10, equal observed legal-branch weight, no class
reweighting, no early stopping, no checkpoint selection. Hazard likelihood sums
conditional event/non-event BCE through the event or censor endpoint; unknown
intervals contribute zero. D300 = 1-product(1-hazard); single-horizon scores at
earlier horizons are explicitly unavailable. Nonfinite loss/grad => INVALID.
480 completed fit units = 2 tables x 4 encoders x 2 heads x 3 doses x 2 seeds x 5 folds.
Synthetic canary executes this same loop with one epoch / batch 2; synthetic
results can never be admitted to the registered look.

## One look and frozen kill rules

Only the Mac aggregator opens the complete union of OOF predictions after every
unit is verified. Average the two seed probabilities before metrics. Report D300
AUC/Brier, horizon-specific AUC/Brier (hazards), A5 decisive teacher agreement
(survivors p>=0.5 ranked by recorded B8 Q; otherwise max survival, lowest-index
ties), parent and rescue-bearing world counts, and 50->100% AUC slopes per arm.
World-cluster paired percentile bootstrap: 2,000 resamples, seed 20261022, 95% CI;
undefined AUC resamples are excluded and their count reported. No branch bootstrap.

Primary scale contrast = M-random minus S-random, D300 head, 100% T5 dose.
Scale/volume on this label type CLOSED iff delta AUC < 0.02 AND paired LB <= 0
AND M-random 50->100% AUC slope < 0.01. Report S-B8 and hazard contrasts and
A5 separately as sensitivities, never substitute a favorable arm for primary.
Hazards retained in WP3 only for >=10% relative D300 Brier improvement against
matched single-horizon arm; otherwise keep single-horizon. Primary target decision
uses S-random at 100% T5, other contrasts descriptive. Equality at 10% retains
hazards. No served-gain or fresh-bank inference; WP4 after scale closure may
only be funded on WP2 counterfactual labels. A5's 21 rescue-bearing worlds cannot
meet WP2's 64-live requirement and cannot authorize that gate.

## Caps, transport, and recovery

A100-40 only; forecast 2 allocated hours (~10.8 units at historical 5.4u/h;
live rate checked by operator). PLAN's explicit effective envelope overrides
arithmetic extrapolation: base 2h/20u, effective **4.8h/48u**, applied once.
Conservative debit admission 10u/h, reserve base 4u -> effective 8u within 48u.
Canary <=30min/3u inside the same series envelope; main <=258min/45u, reserve
8u, so canary+main <=288min/48u. Setup/export/failed attempts count. First cap
wins. No automatic retry or cap increase. Two completion attempts at most, only
verified mechanical interruption + verified stop + missing units, within original
remaining series time/units. Fresh manifests/outputs/receipts every attempt.

Unit = complete final fold fit (namespace + frozen allocation index); no partial
optimizer trajectory reused. Publish immutable prediction payload + SHA sidecar
+ completion marker last. Empty uploaded ledger on attempt 1. Mac verifies only
path/size/hash into ledger; refuse foreign namespace/allocation and duplicates,
skip exactly verified unit IDs without recomputation. Aggregator rejects duplicate,
missing or foreign units. Remote metrics stay sealed. No N extension.

Pin published commit and qualified research/colab_cli tree, bind every input SHA,
Python3.12/cu126, <=3 account assignments, file telemetry at <=120 seconds,
<=2GiB bounded immutable export. Claude dispatches via tools/ops/dispatch.sh only.
A100 family requires same-pin real transport canary, verified terminal exports and
confirmed backend release. Qualification is not inferred from offline tests.
Mac preflight: synthetic tiny inputs, one process/thread, no xdist, no long fits.
