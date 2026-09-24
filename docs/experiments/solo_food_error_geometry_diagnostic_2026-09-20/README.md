# BS: held coverage errors are mostly one-step worse choices, not distance ties

**Status: COMPLETE_AND_AUDITED.** BS diagnoses the saved BR fit arrays; it
runs no new rollout, inference, training, or optimizer update. In every final
coverage held set, strictly worse immediate food-direction choices outnumber
equal-distance ties: 220/42 in seed 4501, 200/43 in seed 4502, and 248/57 in seed
4503. Thus the BR held-fit failure cannot be explained solely by a convention for
breaking equal-distance food ties. This is a one-step geometric certificate, not a
claim about downstream trajectory value, observation sufficiency, or a learner fix.

BR remains `overall_success: false` with unchanged criteria. BS is saved-array
analysis of its starting, control, and coverage marks. It does not change BR’s
behavioral evidence, frozen thresholds, or policy status. See the [BR completed
comparison](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_own_policy_coverage_2026-09-20/README.md)
for actual learning and gameplay results.

## What the certificate tests

For each valid parent GreedyFood mismatch, the analyzer compares the target action’s
one-step distance to the nearest food against the predicted action’s distance. It
classifies the row as `tie`, `worse`, `unavailable`, or boost. `Worse` means the
predicted normal-direction action has greater immediate food distance than the target
under the saved state and resolved action contract. The certificate intentionally does
not infer collision risk, future food, later body geometry, or eventual return.

A boost prediction has no normal-direction comparison and is reported separately.
An unavailable result has insufficient eligible comparison evidence. Counts therefore
sum to the evaluated rows rather than asserting every mismatch has a geometric
certificate.

## Final held error geometry

The table reports all final held partitions from the fixed BR arrays. The start
checkpoint is shared by control and coverage at mark 0; final control and coverage
are both shown at mark 500. `Correct` plus the four remaining categories equals the
3,072 held rows in each cell.

| BR seed | Mark / arm | Correct | Tie | Strictly worse | Unavailable | Boost | Error majority |
|---|---|---:|---:|---:|---:|---:|---|
| 2026094501 | Start / shared 0 | 2,546 | 65 | 277 | 183 | 1 | worse |
| 2026094501 | Control / 500 | 2,569 | 61 | 265 | 177 | 0 | worse |
| 2026094501 | Coverage / 500 | 2,662 | 42 | 220 | 148 | 0 | worse |
| 2026094502 | Start / shared 0 | 2,585 | 56 | 326 | 103 | 2 | worse |
| 2026094502 | Control / 500 | 2,590 | 71 | 301 | 110 | 0 | worse |
| 2026094502 | Coverage / 500 | 2,738 | 43 | 200 | 91 | 0 | worse |
| 2026094503 | Start / shared 0 | 2,626 | 79 | 295 | 67 | 5 | worse |
| 2026094503 | Control / 500 | 2,625 | 84 | 293 | 66 | 4 | worse |
| 2026094503 | Coverage / 500 | 2,702 | 57 | 248 | 65 | 0 | worse |

The final coverage strict mistakes are the relevant diagnostic population: 410, 334,
and 370 rows across seeds 4501–3, of which 220/200/248 are strictly worse and
42/43/57 are ties. The remaining 148/91/65 are unavailable. There are no final
coverage boost errors. This separates the observed fit gap from a “teacher ties made
any answer correct” explanation; it does not identify why the network chose the
worse immediate action.

## Exact-input conflict audit

Exact effective-input groups were checked independently within training and held
partitions, then combined. There are zero conflicting target groups and zero
cross-split exact matches in all three seeds. `Repeat rows` are dependent duplicate
records, not independent examples. The absence of conflicts means these saved arrays
do not establish an unavoidable deterministic label contradiction; it does not prove
that the representation, data coverage, finite network, or optimizer is sufficient.

| Seed | Train groups / repeat rows / conflicts | Held groups / repeat rows / conflicts | Combined groups / repeat rows / cross-split matches |
|---|---:|---:|---:|
| 2026094501 | 1,519 / 17 / 0 | 3,013 / 59 / 0 | 4,532 / 76 / 0 |
| 2026094502 | 1,536 / 0 / 0 | 3,013 / 59 / 0 | 4,549 / 59 / 0 |
| 2026094503 | 1,495 / 41 / 0 | 3,061 / 11 / 0 | 4,556 / 52 / 0 |

## Fixed held examples and partition curves

The first declared worse held example occurs at frame 0 in every seed. It is selected
by frozen order, not for visual appeal: seed 4501 lane 9 predicts action 1 where the
target is 0; seed 4502 lane 6 predicts 1 where the target is 0; and seed 4503 lane
12 predicts 0 where the target is 1. These figures make the saved state, masks, and
one-step comparison inspectable; they do not simulate the alternative action.

| Seed | Final partition curve | First worse held example |
|---|---|---|
| 2026094501 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/partition-learning-seed2026094501.png) | [frame 0, lane 9](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/held-example-seed2026094501.png) |
| 2026094502 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/partition-learning-seed2026094502.png) | [frame 0, lane 6](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/held-example-seed2026094502.png) |
| 2026094503 | [curve](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/partition-learning-seed2026094503.png) | [frame 0, lane 12](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_error_geometry_diagnostic_2026-09-20/held-example-seed2026094503.png) |

## Execution, boundary, and next decision

Qualification passed nine tests in 1.385 seconds. One CPU analysis completed in
4.156399958 seconds against its 60-second science cap. Peak RSS was 1,114,849,280
bytes and minimum available memory was 39,025,868,800 bytes. A monitor recorded one
confirmed-exit race; it was not a failed science job. The independent audit rehashed all 3,324 bound inputs, confirmed source and input
identity before and after execution, and passed all 48 row partitions. Root visually
inspected all six figures; their report copies are byte-identical. No retries or
failed science jobs occurred.

The next prospective comparison will keep the same 1,536 training rows, 500 updates,
parents, and model while comparing eight versus sixteen training worlds. It has not
been frozen or executed. BS cannot establish a policy improvement, PQN progress,
architecture superiority, or promotion. Apex remains the operational incumbent until
the shared tournament gate supports replacement; its larger historical training
budget remains a confound.

## Primary records

- [BS intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-error-geometry-diagnostic/intent.json)
  — SHA-256 `201bbb5166c12d45baa2d617f6b5ba912c901e2c6174bd6364ced31a057c5003`
- [BS completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-error-geometry-diagnostic/analysis/analysis.json)
  — SHA-256 `cc99d950fcc564f3cebc13fab9a5159d7fb37659f86bdba75a19b96ad735232d`
- [BS qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-error-geometry-diagnostic/qualification-complete.json)
  — SHA-256 `3b5e842d3770dd6b2fc07c047b89ced1585aa63f11c4462cce21d8bba30a323c`
