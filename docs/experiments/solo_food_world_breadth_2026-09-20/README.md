# BT: fixed row-budget comparison of narrow versus broad own-policy worlds

**Status: COMPLETE_AND_AUDITED.** At the fixed 1,536-row and 500-update
budget, spreading own-policy examples across sixteen worlds did **not** reliably
improve held decisions or H512 food collection over concentrating the same number of
rows in eight worlds. Both learned arms still played the short, reachable-food task
well and retained the frozen parent behavior, but the predeclared breadth result
requires every lineage to pass. It passed in none.

BT follows BR, where selected own-policy rows fitted strongly but held reliability
did not pass, and BS, where most remaining strict held errors were immediately worse
food choices rather than distance ties. This experiment changes world breadth and
per-world temporal density only; it does not test a new observation, network,
optimizer, teacher, or PQN update.

## Frozen comparison and boundaries

Fresh optimization/sampler seeds 2026094601–2026094603 map to the same admitted
BO/BM warm-start composites 4401–4403 and their frozen food parents 3401–3403. They
are independent fine-tunes from those parents, not warm starts from BR and not
whole-learner reinitializations. The shared safety head, architecture, observation
transforms, native six-action mask, simulator, and teacher contract remain frozen.

A single H512 collection across training worlds 2026105300–2026105315 supplied four
headings and three balanced reachable-food lanes per world. Each arm selected exactly
1,536 rows, mixed 128 original-retention and 128 own-policy rows per batch, consumed
128,000 example presentations, and made 500 updates. Adam configuration and sampler
positions matched between arms; optimizer state differed as gradients accumulated.

| Arm | World breadth and temporal density | Selected rows |
|---|---|---:|
| Narrow control | First 8 worlds × 12 lanes × 16 rows | 1,536 |
| Wide coverage | All 16 worlds × 12 lanes × 8 rows | 1,536 |

Thus this is a breadth-versus-depth test, not an isolated world-identity test. On
shared lanes, wide used the first eight rows of narrow's sixteen-row permutation.
The actual captured training support was 6,144 rows per seed; all 3,072 held rows per
seed came from worlds 2026105400–2026105407, and all gameplay lanes came from separate
worlds 2026105500–2026105507. No held or gameplay record entered gradients. There
were no replacement worlds or dead-row substitutions; ordinary minibatch resampling
is part of the frozen learner.

The primary statistical unit is eight paired gameplay-world clusters within a seed;
twelve lanes in a world are not independent replicates. Every gate had to pass for
each of the three seeds separately, with no pooling.

## Final fit: selected rows fit; held breadth criteria fail

Both arms fitted their selected training rows at mark 500. None of the three wide
policies met the
inherited held accuracy, macro-recall, and every-world criteria. It also missed both
comparative requirements: at least 20% narrow-to-wide equal-world error reduction and
at least 0.05 absolute reduction from the starting parent.

| Seed | Narrow train acc / macro | Wide train acc / macro | Narrow held acc / macro / error | Wide held acc / macro / error | Wide vs narrow error reduction | Start minus wide error |
|---|---|---|---|---|---:|---:|
| 4601 | 99.674% / 99.528% | 99.609% / 99.572% | 88.314% / 86.928% / 11.686% | 88.249% / 86.884% / 11.751% | -0.557% | 4.134 pp |
| 4602 | 99.674% / 99.681% | 99.479% / 99.503% | 88.900% / 87.119% / 11.100% | 88.737% / 87.611% / 11.263% | -1.466% | 3.711 pp |
| 4603 | 99.674% / 99.636% | 99.870% / 99.864% | 88.184% / 86.598% / 11.816% | 88.770% / 87.177% / 11.230% | 4.959% | 4.329 pp |

The wide arm reduced held error only in seed 4603, and by 4.959% rather than the
required 20%; its 4.329-point improvement from the starting parent was also below the
five-point threshold. This keeps the `coverage_fit` and `held_fit_advantage` gates
false for all three seeds. Selected-row fit therefore does not establish held-policy
reliability.

## Greedy gameplay: good absolute play, no reliable breadth benefit

All policies below use greedy gameplay on the fresh bank. Values are mean ambient food,
survival fraction over the horizon, surviving lanes out of 96, and mean mass integral.
All control and wide absolute/retention gates at H256 and H512 passed, as did all six
BL package-safety compatibility gates. H512 zero-food lanes were zero for baseline,
narrow, and wide in every seed.

| Seed | Role | H256: food / time / end / mass | H512: food / time / end / mass |
|---|---|---|---|
| 4601 | Parent | 42.125 / 1.000000 / 96 / 22.977091 | 77.552 / 0.998739 / 95 / 42.213420 |
| 4601 | Narrow | 43.885 / 1.000000 / 96 / 23.699707 | 84.271 / 1.000000 / 96 / 44.443807 |
| 4601 | Wide | 43.354 / 0.996989 / 95 / 23.598511 | 83.177 / 0.993286 / 95 / 43.821452 |
| 4602 | Parent | 42.188 / 1.000000 / 96 / 23.073608 | 78.396 / 1.000000 / 96 / 42.354451 |
| 4602 | Narrow | 43.156 / 1.000000 / 96 / 23.481689 | 81.927 / 1.000000 / 96 / 43.735107 |
| 4602 | Wide | 43.719 / 1.000000 / 96 / 23.570516 | 84.177 / 1.000000 / 96 / 44.381205 |
| 4603 | Parent | 41.417 / 1.000000 / 96 / 22.655192 | 78.125 / 1.000000 / 96 / 41.874552 |
| 4603 | Narrow | 43.583 / 1.000000 / 96 / 23.767700 | 82.917 / 1.000000 / 96 / 44.151713 |
| 4603 | Wide | 42.458 / 1.000000 / 96 / 23.299398 | 80.219 / 0.999207 / 95 / 42.876139 |

The teacher and random-safe anchors passed calibration and leave room for learned
performance. They were executed on the same fresh gameplay bank; values use the same
food, time, endpoint, and mass definitions as the learned table.

| Anchor | H256: food / time / end / mass | H512: food / time / end / mass |
|---|---|---|
| Scripted teacher | 44.854 / 1.000000 / 96 / 24.423462 | 86.990 / 0.994527 / 93 / 45.292684 |
| Random-safe | 4.729 / 0.996582 / 94 / 3.505168 | 8.219 / 0.936096 / 66 / 4.934896 |

The positive-result gate additionally required wide H512 food at least 3% above
narrow, a paired eight-world food interval wholly above zero, and no additional
zero-food lanes. The zero-lane condition passed, but both effect requirements failed
everywhere:

| Seed | Wide minus narrow H512 food | Paired 95% CI | World wins / losses | Result |
|---|---:|---|---:|---|
| 4601 | -1.094 | [-3.667, 1.479] | 4 / 4 | ratio and interval fail |
| 4602 | 2.250 | [-0.638, 5.138] | 6 / 2 | ratio and interval fail |
| 4603 | -2.698 | [-5.217, -0.179] | 2 / 6 | ratio and interval fail; interval is negative |

The same predeclared failure ledger appears for each seed: `coverage_fit`,
`held_fit_advantage`, and `wide_H512_benefit`. The result is not evidence that
sixteen worlds are harmful in general; it is evidence that this fixed-row,
fixed-update breadth package did not produce a reliable advantage over its matched
eight-world control.

## Evidence and execution record

The calibration passed before learner work. All 30 science jobs completed naturally;
two supervisor process-monitor races (`gameplay-coverage-seed2026094601` and analysis)
were recovered as confirmed exit 0 and are not failed jobs. The audited resource
record is 860.0889573331224 seconds against the reserved 1,970 seconds, with peak
recursive RSS 2,182,955,008 bytes, minimum available memory 36,792,205,312 bytes,
and training MPS-driver peak 1,164,722,176 bytes across the 3,000 study-update
samples. Scientific review confirmed every inherited H256/H512 absolute, retention,
and safety gate passed; it also confirmed the strict held-fit and breadth-benefit
failures in all three seeds. All nine saved plots passed visual QA.

Qualification preserves the 4.613-second failed hash-convention invocation and the
9.293-second repaired 48-test pass: 13.906 seconds of the 120-second allowance. Its
two disposable MPS optimizer updates are outside the 3,000 study updates across the
six learners. There were no failed science attempts.

The following copies preserve the analysis evidence used here:

| Seed | Fit curve | Gameplay comparison | Representative gameplay |
|---|---|---|---|
| 4601 | [curve](learning-fit-seed2026094601.png) | [comparison](gameplay-comparison-seed2026094601.png) | [representative](representative-gameplay-seed2026094601.png) |
| 4602 | [curve](learning-fit-seed2026094602.png) | [comparison](gameplay-comparison-seed2026094602.png) | [representative](representative-gameplay-seed2026094602.png) |
| 4603 | [curve](learning-fit-seed2026094603.png) | [comparison](gameplay-comparison-seed2026094603.png) | [representative](representative-gameplay-seed2026094603.png) |

## Scope and next question

BT makes no claim about PQN, full-game opponents, or policy replacement. Apex remains
the operational incumbent until the shared tournament gate supports replacement; its
larger historical training budget is not an architecture comparison or proof of
inherent superiority.

The next bounded hypothesis is whether a CoordConv-style positional representation
could improve the strict immediate-food decision errors seen on held own-policy rows.
It is only a proposed question: it is not frozen, run, or an outcome of BT.

## Primary records

- [BT frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/design.md)
- [BT frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/intent.json)
- [Final analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/analysis/analysis.json)
- [Qualification accounting](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/qualification-accounting.json)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/qualification-complete.json)
- [Calibration report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/calibration/report.json)
- [BT receipt directory](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/supervisor-runs)
- [BR own-policy coverage report](../solo_food_own_policy_coverage_2026-09-20/README.md)
- [BS error-geometry diagnostic](../solo_food_error_geometry_diagnostic_2026-09-20/README.md)

- [Audited closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/closeout.json)
- [Receipt and resource audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/independent-audit.json)
- [Independent science review](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/science-review.json)
- [Nine-figure visual QA](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-world-breadth-r1/visual-qa.json)
