# T4e RETURN pre-registration amendment — 2026-10-16

## Amendment 1, before any RETURN run

This sibling amends `t4e_prereg_2026-10-16.md` under the binding
`t4e_calibration_spec_2026-10-16.md` and
`../advice/2026-10-16_return_arm_review.md`. The original pre-registration is
SHA-bound by the completed collection and remains byte-for-byte unchanged
(`d404e5d95e3671d12b0c0761246862448b0f84a1fbc7ffecc3046ac1b1fdd6a0`).
This amendment supersedes its W=1.0/no-calibration paragraph and the older
within-10-percentage-point descriptive readout. No 8M RETURN campaign run has
started. The explicitly authorized, isolated calibration smokes are not campaign
finals, evaluation results, or promotion evidence.

### Calibration and fixed weight

**Fixed W = 0.0032** (two significant figures). The required five-weight grid
did not bracket 0.5: its smallest pooled initial ratio was 1.383 at W=0.01.
An additional short W=0.001 pair measured 0.179 pooled; log-log interpolation
between these measured points gave 0.0031830591. The rounded weight was then
confirmed on both initializations: **0.646 / 0.466**, pooled **0.548**.

All 14 calibration smokes completed exactly 300 updates, with no halt or flags
and actor exit codes `[0, 0]`. Each used MPS; no CPU fallback was needed.

| B8 init | W | Initial ratio (1–50) | Middle (126–175) | Last (251–300) | max abs Q |
|---|---:|---:|---:|---:|---:|
| s0 | 1 | 4.354 | 20.678 | 38.609 | 11.852 |
| s0 | 0.2 | 2.815 | 10.906 | 12.088 | 12.441 |
| s0 | 0.05 | 2.586 | 5.515 | 6.143 | 12.871 |
| s0 | 0.02 | 2.366 | 3.400 | 4.289 | 13.330 |
| s0 | 0.01 | 1.694 | 2.051 | 3.000 | 13.609 |
| s0 | 0.001 | 0.174 | 0.588 | 0.454 | 14.488 |
| s0 confirmation | **0.0032** | **0.646** | 0.861 | 1.073 | 14.321 |
| s2 | 1 | 3.516 | 20.477 | 35.252 | 21.057 |
| s2 | 0.2 | 1.942 | 10.450 | 10.514 | 19.729 |
| s2 | 0.05 | 1.911 | 4.401 | 6.484 | 18.749 |
| s2 | 0.02 | 2.054 | 3.524 | 5.152 | 18.015 |
| s2 | 0.01 | 1.129 | 2.142 | 2.817 | 17.480 |
| s2 | 0.001 | 0.183 | 0.290 | 0.320 | 20.925 |
| s2 confirmation | **0.0032** | **0.466** | 1.011 | 1.409 | 17.704 |

Raw per-update logs, exact argv, source hashes, final smoke checkpoints, and
normal-cleanup summaries are under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t4e-calibration-20261016/`.
The persisted summary is `../t4e_calibration_result_2026-10-16.json`.
The chosen weight is substantially below the advisor's expectation; the measured
gradient ratios, rather than that expectation, determine the registered value.

Each grid smoke uses the actual two-actor RETURN trainer, real training bank,
and B8_s0 or B8_s2 final, on MPS. Training stops through normal trainer cleanup
after 300 staged/measured updates; final flush applies exactly update 300.
There is also a 200,000-transition ceiling. The optimizer, target network,
replay, and actors are newly initialized: these are weights-only fine-tunes.
The production recipe is unchanged except the calibrated W and opt-in
`--return-gradient-telemetry`; the smoke changes only the stopping boundary.

The initial ratio is the geometric mean of the same-update pre-clip ratios
over updates 1–50; trend windows are updates 126–175 and 251–300. Seeds have
equal weight when pooling their initial geometric means. Weight selection
uses log-log interpolation to a pooled ratio of 0.5, rounded to two significant
figures, with confirmation from both initializations. No survival, evaluation,
held-out labels, or final checkpoint performance enters weight selection.

### Batch composition and telemetry

The observed batch holds **256 on-policy anchor rows + 128 additional return
rows + 768 agent TD rows** on every measured update. This is the unchanged M6
CTRL/RETURN recipe with `--batch-scale 2 --label-frac 1 --return-frac 0.5`.
The advisor/calibration spec's “128 anchor” count is the unscaled base count;
the actual scale-2 anchor block has 256 rows. We retain that complete block,
so the essential condition is satisfied: return rows do not replace or split
the anchor rows. No anchor-batch reduction was introduced to force the
specification's unscaled number.

`--return-gradient-telemetry` measures
`||grad(W * return_loss)|| / ||grad(td_loss)||` on shared online parameters
from the same update, before clipping. It uses `autograd.grad` without
mutating optimizer gradients or drawing RNG values. Per-100k log rows report
the arithmetic mean of these ratios, both mean norms, all three row counts,
and `max_abs_q` (maximum absolute online Q over sampled agent current states,
all six actions, in the interval; same scope as existing `qmax`). Calibration
also retains every update. The flag defaults off; six reference/fast trace
modes preserve network tensors, RNG, and existing stable telemetry exactly.

A separate short logging smoke completed 386 updates / 100,224 transitions
with no halt or flags and actor exits `[0, 0]`. Its first per-100k row contains
all requested fields: ratio 1.215411 (arithmetic interval mean), weighted-return
norm 0.738188, TD norm 0.956695, batch 256/128/768, and max|Q| 15.402152.
This interval mean includes post-initialization drift and is not the first-50
calibration statistic.

### Advisor's registered diagnostic readouts

The advisor's four threshold bullets are adopted verbatim:

> - held-out decisive gap <= -3 Q (B8 starts at +0.45 median; T1 p90 is +2.55);
> - training-set gap of similar size, so the two agree within a factor of two, which rules out
>   memorisation;
> - held-out outside-state gap unchanged within +/- 0.5, which rules out spillover in Q space;
> - consequential change (c) within 2 points of the 0.99% drift baseline.

At 8M report the held-out decisive **median** of
`Q(fatal) - Q(best surviving)` with its training-set median and distribution,
and the held-out outside-state gap `Q(B8 action) - max(Q(other actions))` change
from initialization. The training/held-out decisive gaps must have the same
negative sign and their absolute medians must agree within a factor of two.
The consequential-change band is within two percentage points of 0.99%
(physical rate range 0–2.99%). Report each seed and the five-seed mean,
denominators, and **max|Q| per 100k transitions**, including the final window.
These diagnostics accompany the existing primary GO/KILL/EXTEND rule; they do
not replace it or authorize training, extension, or promotion.

Keep return Huber delta 100 and TD delta 1. Targets remain “a then B8 for 300”;
the continuation can undervalue a student's later improved escape, while the
survivor bootstrap uses the student's moving target network. This staleness is
disclosed and left unchanged.

### World exclusion and unstarted launch boundary

`../t4e_calibration_seed_audit_2026-10-16.json` binds all six collection plans,
both published bank manifests, and the seed-generator/evaluator sources.
All 1,200 collection world/seed pairs are represented and are disjoint from
rounds 68 and 69. Published training worlds (312) and held-out worlds (30) are
disjoint. Round 68 uses 96 scripted seeds; the frozen/mixed 32-world mixes use
prefixes of that roster. Round 69 reserves its own 96 scripted seeds. Both
rosters have zero overlap with every collection seed, hence with both banks.

The generic unfiltered scan is preserved separately: it matches previously
written seed-audit receipts and the collection transcript. Those documentary
hits are not bank-world reuse. Refresh the full seed audit with exact
documentary exclusions at evaluation launch, as the original prereg requires.

Both tracked and artifact `lane0.sh`/`lane1.sh` retain explicit `--device mps`,
the original seed split and 8M command, create-only output refusal, and
`--check` behavior. They use the new `launch_inputs_amendment1.json` binding,
the calibrated W, and gradient telemetry. The original launch receipt is
preserved. Only `--check` is executed here; the 8M scripts remain unstarted.

## Deviation 1 (planner, 2026-10-16): RETURN halted at ~0.8M, no screen data
Both started seeds (s10, s11) showed greedy boost rising 1.4% -> 34% (CTRL ~0.1%), return/TD grad
ratio rising 1.3 -> 2.5-3.6, max|Q| stuck at ~16-30 (CTRL climbs to 110-140). The planner stopped the
lanes at ~0.7-0.8M and set the partial runs aside as `HALTED_boost_RETURN_s1{0,1}` (never probed).
No RETURN final exists; the screen has not run. A bug check and an advisor review precede any
relaunch, which will be pre-registered as a new arm/amendment.
