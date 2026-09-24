# BY: PQN fixed versus released teacher anchor

**Status: COMPLETE_AND_AUDITED.** Both the fixed-anchor control and the
released-anchor arm retained the short food-routing task in all three fresh learner
seeds. Releasing the teacher anchor did not show a reliable food benefit over the
fixed control in any seed, and only seed 5001 had a positive final-versus-initial
food interval. The declared strong outcome is false. This is not a causal result
about Q margins, a policy promotion, or an architecture comparison.

## Frozen comparison

BY reuses the three AN supervised checkpoint-500 parents and starts fresh native RNG
streams: 5001 → AN 3401, 5002 → AN 3402, 5003 → AN 3403. It contrasts a fixed teacher
anchor with a released anchor under the same native PQN recipe and exactly matched
first eight Adam updates. Both arms multiply every Q output by 0.1, and the
teacher cross-entropy uses the original logits, `Q / 0.1`. Fixed uses teacher weight 1
through all 64 updates. Released uses weight 1 at updates 1–8, `(33-u)/25` at
updates 9–32, and zero at updates 33–64, giving 32 fully unanchored updates. This schedule is the
only paired intervention. The identity of the first eight updates is an actual saved
proof, not an assumption based only on configuration. Each arm runs 64 updates and
is evaluated at marks 0, 8, 32, and 64; stored checkpoints also include marks 1, 2,
4, and 16. The study yields 384 updates and 98,210 valid hero transitions.

Train rows are independent of the reused BV 1,526-row conditional scheduled-live
held archive, which is reported only and did not enter training. Greedy gameplay uses
fresh worlds 2026106400–2026106407, with 96 H64 lanes. Eight worlds are the paired
unit; lanes and seeds are never pooled. The pretraining AN models are reused, but the
three native learner RNG streams are fresh.

The [Deep Q-learning from Demonstrations](https://arxiv.org/abs/1704.03732) and
[Kickstarting Deep Reinforcement Learning](https://arxiv.org/abs/1803.03835) papers
motivate examining how a teacher signal is maintained. BY does not replicate either
method and adds neither method to this learner.

## Final behavior

The fresh-world teacher/random calibration passed all six declared checks. The teacher
collected 12.656 food per game versus 1.260 for random-safe; both survived all 96 games.
The eight-world paired teacher-minus-random food interval was [10.922, 11.869].

Each cell gives ambient food / survival fraction / endpoint survivors. Both arms pass
the frozen retention package in every seed. The final relative-benefit gate requires
the lower bound of released-minus-fixed food to exceed zero; it passes in 0 of 3 seeds.

| Seed | Fixed 0 → 8 → 32 → 64 | Released 0 → 8 → 32 → 64 |
|---|---|---|
| 5001 | 11.188/1.000000/96 → 11.417/1.000000/96 → 11.656/1.000000/96 → 11.688/0.994954/94 | 11.188/1.000000/96 → 11.417/1.000000/96 → 11.656/1.000000/96 → 11.781/1.000000/96 |
| 5002 | 11.344/1.000000/96 → 11.875/1.000000/96 → 11.510/1.000000/96 → 11.865/0.996908/95 | 11.344/1.000000/96 → 11.875/1.000000/96 → 11.802/0.997233/95 → 11.635/0.996419/95 |
| 5003 | 11.406/0.997721/95 → 11.219/0.998535/95 → 10.969/0.999674/95 → 11.583/0.996419/94 | 11.406/0.997721/95 → 11.219/0.998535/95 → 11.354/1.000000/96 → 11.417/0.997721/95 |

| Seed | Released − fixed final food, 95% CI | Released − initial final food, 95% CI | Outcome |
|---|---|---|---|
| 5001 | 0.094 [-0.278, 0.465] | 0.594 [0.101, 1.087] | retention passes; beyond-initial passes; relative benefit fails |
| 5002 | -0.229 [-0.676, 0.217] | 0.292 [-0.035, 0.618] | retention passes; relative benefit and beyond-initial fail |
| 5003 | -0.167 [-0.722, 0.388] | 0.010 [-0.597, 0.618] | retention passes; relative benefit and beyond-initial fail |

Thus both arms retain the behavior 3/3, but released-anchor relative benefit is 0/3
and final-beyond-initial food is 1/3. The frozen all-seed treatment criterion and
strong outcome are false. A retained fixed control does not make a null released-
versus-fixed interval evidence that the two objectives are equivalent.

## Train and held action agreement

Values below are train / held accuracy. Held fit is descriptive on the reused BV
conditional-live archive; it did not select a mark or alter gameplay criteria.

| Seed | Fixed: 0 → 8 → 32 → 64 | Released: 0 → 8 → 32 → 64 |
|---|---|---|
| 5001 | 99.967%/88.794% → 99.967%/90.695% → 99.967%/89.122% → 99.967%/91.547% | 99.967%/88.794% → 99.967%/90.695% → 99.967%/91.612% → 98.014%/90.760% |
| 5002 | 99.967%/89.187% → 99.870%/90.236% → 99.837%/90.695% → 99.935%/91.809% | 99.967%/89.187% → 99.870%/90.236% → 99.967%/91.088% → 98.568%/91.022% |
| 5003 | 99.902%/88.204% → 99.870%/89.056% → 99.902%/87.484% → 99.902%/90.170% | 99.902%/88.204% → 99.870%/89.056% → 99.544%/89.450% → 98.307%/89.318% |

The two arms share the exact first-eight prefix, so their mark-8 rows are the same.
Later fit differences do not establish a common-Q-margin mechanism or explain the
non-positive relative-benefit intervals.

## Curves and fixed gameplay

The figures show the predeclared marks and fixed representative lanes. In seed 5002,
fixed lane 35 at released mark 64 collects five food while still alive; it is a
low-food loop/trajectory, not a death. The figures are behavioral evidence for this
study, not checkpoint-selection material.

- [Learning curves](learning-curves.png)
- [Fixed-bank behavior comparison](behavior-comparison.png)
- [Fixed representative gameplay evidence](fixed-gameplay-evidence.png)

## Execution, scope, and next question

Qualification completed 45 guarded tests in 4.159 seconds. An additional 0.3 seconds
of unguarded tiny fixtures is charged as a preserved incident, not qualification and
not ML work. Two discarded MPS updates and zero discarded CPU updates are recorded.
The 29-job science schedule completed in 285.9425303339958 / 730 seconds. The
resource rollup recorded peak RSS 1,168,621,568 bytes, minimum available memory
34,575,450,112 bytes, and peak MPS-driver allocation 1,224,310,784 bytes. The
resource rollup, scientific review, and corrected closeout passed; the audit checked
33,337 hashes.

The next selected question is BZ: teacher-alignment and margin changes from the
common mark-8 fixed rows to mark 64, with a late mark-32-to-64 descriptive
localization. Teacher labels are not assumed to be optimal ground truth. It is a
saved-row diagnostic and does not involve new training. BY does not show that released
anchors cannot help a different task, horizon, or package.

Apex remains the operational incumbent until the shared tournament gate supports a
replacement. Its much larger training budget is a confound, not evidence of
architectural superiority.

## Primary records

- [BY frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/design.md)
- [BY intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/intent.json)
- [BY final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/analysis/analysis.json)
- [BY qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/qualification-complete.json)
- [BY science receipts](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/supervisor-runs)
- [BY resource rollup](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/resource-rollup.json)
- [BY science review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/science-review.json)
- [BY corrected closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-anchor-release/closeout-corrected.json) — SHA-256 `e67ef0c0be58fb776c60647facf3f985635787e6fbe1b64184794771eb538ba0`
- [BX output-scale result](../solo_food_pqn_output_scale_2026-09-20/README.md)
