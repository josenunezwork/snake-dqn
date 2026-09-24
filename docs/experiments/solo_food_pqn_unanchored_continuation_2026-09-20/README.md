# CA: longer PQN continuation with and without a restored teacher anchor

**Status: COMPLETE_AND_AUDITED.** After 128 further native
updates from BY’s released mark-64 parent, `no_ce` retained H64 behavior in only one
of three fresh-runtime seeds. Restoring the teacher anchor retained behavior in all
three. `no_ce` did not beat the anchored comparator or its own mark-64 start by the
predeclared paired-world rule in any seed, so the strong outcome is false. This is an
objective-package result for a short H64 benchmark, not a policy promotion or an
architecture ranking.

## Frozen comparison and recovery boundary

CA maps fresh runtime seeds 5101–5103 to BY released-anchor parents 5001–5003 at mark
64. Both arms inherit the same authenticated parent network tensors, full Adam state
and ages, update count, odometer, and tripwire counters; fresh runtime world/action
RNG streams start at frame 0. This is an experimental model/Adam warm start, not an
exact simulator continuation or canonical CLI resume.

The only arm difference is teacher weight: `no_ce` uses weight 0 while
`reanchor_ce1` uses weight 1. Both retain Q scale 0.1, native TD updates, teacher-row
sampler, masks, optimizer, epsilon, H64 horizon, and observation contract. Each runs
128 new updates, to total mark 192. Train rows remain separate from games; the reused
BV 1,526-row scheduled-live held archive is diagnostic only, not a fresh final test.
Greedy games use fresh worlds 2026106600–2026106607 and 96 lanes. Eight worlds are
the paired statistical unit; lanes and seeds are never pooled.

The 5102 `no_ce` first attempt stopped at the 70-second wall after 118 confirmed
updates. It was preserved. Its replacement restarted from the authenticated BY parent
and same CA seed, not from a partial checkpoint; the saved 118-prefix diagnostic
bytes and model/Adam digests were exact. R4 repaired a zero-update `Path` startup
error. R5 reconstructed the missing 5102 reanchor mark-192 report after its guarded
frames and Q archives had completed, performing no forward pass, optimizer update, or
simulator step. These operational repairs did not change worlds, criteria, optimizer,
or selection rules.

## Actual H64 behavior

Each cell is ambient food / survival fraction / endpoint survivors. Mark 128 is
descriptive; mark 192 governs the declared result.

| Seed | Initial 64 | No CE: 128 → 192 | Reanchor: 128 → 192 |
|---|---|---|---|
| 5101 | 11.927 / 0.999837 / 95 | 10.302 / 1.000000 / 96 → 11.719 / 1.000000 / 96 | 11.854 / 1.000000 / 96 → 12.000 / 1.000000 / 96 |
| 5102 | 11.510 / 1.000000 / 96 | 10.698 / 0.999512 / 95 → 9.469 / 0.999837 / 95 | 11.677 / 1.000000 / 96 → 11.750 / 1.000000 / 96 |
| 5103 | 11.781 / 1.000000 / 96 | 9.760 / 0.994954 / 95 → 4.698 / 1.000000 / 96 | 11.635 / 0.999512 / 95 → 11.740 / 0.999512 / 95 |

| Seed | No CE − reanchor final food, 95% CI | No CE − initial food, 95% CI | Mark-192 result |
|---|---|---|---|
| 5101 | -0.281 [-0.936, 0.373] | -0.208 [-0.819, 0.402] | no-CE retention passes; relative and beyond-initial fail |
| 5102 | -2.281 [-2.726, -1.836] | -2.042 [-2.584, -1.499] | no-CE retention, relative, and beyond-initial fail |
| 5103 | -7.042 [-7.605, -6.478] | -7.083 [-7.657, -6.510] | no-CE retention, relative, and beyond-initial fail |

All reanchor retention checks pass 3/3. No-CE retains 1/3, relative benefit passes
0/3, beyond-initial benefit passes 0/3, and the strong outcome is false. The
calibration’s teacher food is 12.645833 and random-safe food is 1.5, both with 96
endpoints; it establishes headroom rather than selecting an arm.

The 13-entry failure ledger is preserved exactly: seed 5101 fails the no-CE versus
reanchor food lower-bound and no-CE versus-initial lower-bound gates. Seed 5102 fails
no-CE final food versus 95% initial, versus 75% teacher, and versus mark-128 minus 5%
teacher, plus both relative and beyond-initial lower-bound gates. Seed 5103 fails
those three food-retention gates, all-12-pose-cell food, and both relative and
beyond-initial lower-bound gates. All no-CE survival and endpoint bounds pass; those
facts do not compensate for failed food rules.

The fixed lanes illustrate individual trajectories, not pooled proof: seed 5103 lane
1 has no-CE mark-192 food 2 while alive, compared with reanchor food 11 while alive.
Conversely, seed 5102 lane 35 has no-CE food 12 versus reanchor 11 even though the
aggregate comparison is worse; neither example is a cherry-picked group conclusion.

## Training versus held fit

Values are train / held action accuracy at marks 64, 128, and 192. The held rows were
reused from BV and did not enter training or select checkpoints.

| Seed | No CE: 64 → 128 → 192 | Reanchor: 64 → 128 → 192 |
|---|---|---|
| 5101 | 98.014% / 90.760% → 94.368% / 84.207% → 97.363% / 87.811% | 98.014% / 90.760% → 99.967% / 90.105% → 99.967% / 90.039% |
| 5102 | 98.568% / 91.022% → 96.810% / 86.763% → 81.120% / 84.600% | 98.568% / 91.022% → 99.967% / 89.843% → 99.967% / 90.367% |
| 5103 | 98.307% / 89.318% → 93.913% / 83.879% → 60.221% / 73.722% | 98.307% / 89.318% → 99.902% / 88.270% → 99.902% / 89.318% |

Fit drift is compatible with the behavioral split but does not establish its cause.
It cannot turn a retained reanchor arm into evidence that teacher labels are globally
optimal, nor rescue the no-CE failures.

## Curves and representative gameplay

- [Learning and loss curves](learning-curves.png)
- [Fixed-bank behavior comparison](behavior-comparison.png)
- [Fixed representative gameplay evidence](fixed-gameplay-evidence.png)

Root visually reviewed all three figures: they are legible, unclipped, and their
fixed lanes 0, 35, and 1 agree with the saved endpoints.

## Execution and limits

CA’s original 730-second cap is historical. The preserved wall-stop recovery amended
the study reservation to 900 seconds without changing scientific gates. Qualification
7.897 seconds is likewise a historical pre-R3 subtotal; the later recovery accounting
is retained in the artifacts. The independent science and receipt audits, visual QA, resource rollup, and closeout
all passed. The final ledger records 29 physical attempts: 25 successful logical
science jobs and four preserved failures. It charged 498.70716814749176 / 900 science
seconds and 10.438 / 120 qualification seconds, with 768 accepted updates and
196,364 accepted transitions. Peak RSS was 1,252,638,720 bytes, minimum available
memory 34,358,198,272 bytes, and peak MPS-driver allocation 1,220,182,016 bytes.

The next selected question is a one-factor food-reward comparison, 1.0 versus 0.1,
from all BY released mark-64 models with fresh Adam, matched 64 updates, and three
fresh seeds. It has no outcome in this report.

This result only localizes a failure after a 128-update continuation package on fresh
H64 worlds from inherited BY model/Adam state. It does not identify the reward or
objective mechanism, generalize to longer games or opponents, or show that Apex is
inherently superior. Apex remains the operational incumbent until the shared tournament
gate supports replacement; its larger historical training budget is confounded.

## Primary records

- [CA final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/analysis-r2/analysis.json) — SHA-256 `db60b3fd8e6e0f461124336c007268d845edb6cd2e9acb0c6edcdca9f2cca8f1`
- [CA frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/design.md)
- [CA R3 prefix proof](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/retry-prefix-proof.json)
- [CA R5 evaluation recovery](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/eval-recovery-r5.json)
- [CA qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/qualification-accounting.json)
- [CA science audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/science-audit.json)
- [CA independent receipt audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/independent-audit.json)
- [CA resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/resource-rollup.json)
- [CA visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/visual-qa.json)
- [CA closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-unanchored-continuation-r1/closeout.json) — SHA-256 `7dd255a24b258845e2520485139fc99ef20480c846a37b2efd3dc187b6b677c3`
- [BY released-anchor parent study](../solo_food_pqn_anchor_release_2026-09-20/README.md)
