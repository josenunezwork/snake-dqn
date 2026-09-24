# Saved label geometry supports a parent-consistent safety target, not a learned improvement (BC)

BC finds strong incompatibility in all three held-out seeds: strict food-ranking disagreements make up more than half of both the normal parent-versus-teacher decision population and the reachability candidate's errors. This supports testing a parent-consistent safety target before another learning experiment. It does not show that the target improves greedy gameplay, resolves the held-veto generalization gap, or constitutes new model progress.

The key distinction is deliberate. BA's ordinary loss is designed to preserve the frozen parent distribution, so the prevalence of food-ranking disagreements on ordinary rows is expected. It does **not** explain away the held-veto/safety generalization failure. BC separates those parent-ranking conflicts from cases where an engineered roomy-action filter excludes the parent's choice and the parent's best retained normal action agrees with the teacher (clean safety).

## Read-only saved-state decomposition

BC reuses the three BA final-500 candidate/control and mark-zero parent Q archives, with their 3,072 training rows and separate 1,536 held-out rows. It performs no model forward call, optimizer update, simulation step, new label collection, or gameplay. It reconstructs the saved length and reachability cap, derives the roomy normal set from the stored reachable counts, and checks the existing masks, labels, task/step joins, parent-Q equality, and tie behavior. It relies on the hash-bound BA saved-action replay qualification and does not repeat that replay.

The seven partitions are exhaustive and disjoint: parent already matches teacher; clean safety filter; safety plus strict food disagreement; allowed strict food disagreement; excluded tie only; allowed tie only; and boost-domain mismatch. A strict food disagreement is not an equal-parent-Q tie. Boost selections are never treated as normal-space exclusions.

| Seed | Split | Parent already teacher | Clean safety | Safety + strict food | Allowed strict food | Boost / tie-only |
|---|---|---:|---:|---:|---:|---:|
| 2026094101 | Train | 2,564 | 98 | 13 | 396 | 1 / 0 |
| 2026094101 | Held-out | 1,159 | 117 | 43 | 217 | 0 / 0 |
| 2026094102 | Train | 2,582 | 90 | 13 | 385 | 2 / 0 |
| 2026094102 | Held-out | 1,285 | 38 | 7 | 206 | 0 / 0 |
| 2026094103 | Train | 2,586 | 69 | 19 | 398 | 0 / 0 |
| 2026094103 | Held-out | 1,305 | 46 | 13 | 172 | 0 / 0 |

No held seed has an empty normal set, boost-domain mismatch, or tie-only disagreement. The absence of those cases is why the frozen parent-consistent safety target remains interpretable for the saved decision population.

## Safety fit and strict-ranking error composition

The table reports the normal parent/teacher disagreement population `D`, strict food-ranking rows within it, candidate errors within `D`, and candidate errors on strict-ranking rows. Clean safety fit is the candidate's teacher accuracy inside the clean-safety partition; the error count is its complement in rows, not a separate learning result.

| Seed | Split | D | Strict food rows | Candidate errors in D | Strict-ranking candidate errors | Clean safety: rows / teacher fit / errors |
|---|---|---:|---:|---:|---:|---:|
| 2026094101 | Train | 507 | 409 | 397 | 392 | 98 / 94.90% / 5 |
| 2026094101 | Held-out | 377 | 260 | 329 | 248 | 117 / 30.77% / 81 |
| 2026094102 | Train | 488 | 398 | 385 | 383 | 90 / 97.78% / 2 |
| 2026094102 | Held-out | 251 | 213 | 226 | 205 | 38 / 44.74% / 21 |
| 2026094103 | Train | 486 | 417 | 385 | 383 | 69 / 97.10% / 2 |
| 2026094103 | Held-out | 231 | 185 | 209 | 175 | 46 / 26.09% / 34 |

Held strict-food shares are 260/377, 213/251, and 185/231. Their candidate-error shares are 248/329, 205/226, and 175/209. Each exceeds one half, satisfying BC's predeclared strong-incompatibility criterion in 3/3 held seeds. The alternative compatible-failure criterion does not pass: clean safety is not the majority of candidate errors in all three held seeds, despite its substantial held-out fit drop.

Clean-safety candidate teacher accuracy is 94.90%, 97.78%, and 97.10% on training rows but 30.77%, 44.74%, and 26.09% on held-out rows. That drop remains an important safety-generalization diagnostic. It does not permit replacing the ordinary parent-preservation target with the scripted teacher's food ranking: the ordinary loss intentionally preserves the parent, and a majority of strict food disagreements is therefore expected rather than a resolution of the mechanism failure.

## What the result supports next

The parent-consistent safety oracle alters a parent action only when the roomy filter excludes it, then chooses the parent's highest-Q retained normal action. BC supports qualifying this oracle in greedy gameplay before any new learning comparison. The oracle is not learned, does not change the full six-action mask, and has no demonstrated gameplay benefit yet.

The next work is a bounded parent-consistent safety-oracle greedy-gameplay test. First screen the oracle on BA worlds while reusing the completed parent baselines. A positive screen then needs a fresh paired parent/oracle world bank before using the target in a matched learning experiment. This audit does not establish missing observation information, causal rescue, a safe model replacement, or equivalence of objectives.

## Evidence and boundaries

The one CPU analysis completed in 5.167595125 seconds of the 30-second science cap. Qualification passed six tests in 1.699 seconds of the 120-second cap. The source revision is unchanged at `56e0e92434ae510a84ebaf6f58cf41c3e4421b04`; source input and criteria hashes are recorded in the saved report. The resource audit passed with zero failed attempts: peak recursive RSS was 775,110,656 bytes (0.72 GiB) and minimum host available memory was 34,805,252,096 bytes (32.42 GiB). This was CPU-only analysis; no MPS allocation measurement was reported.

BC's held-out error-composition figure is a static diagnostic. BA's learning curves and representative gameplay remain the behavioral evidence for the model checkpoints.

Apex remains the operational incumbent pending the shared tournament gate. Its much larger historical training budget confounds architecture comparisons and is not evidence of inherent superiority.

- [Frozen BC intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-label-geometry/intent.json)
- [Complete BC report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-label-geometry/analysis/report.json)
- [BC resource audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-label-geometry/resource-rollup.json)
- [BC held-out error composition](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-label-geometry/analysis/error-composition.png)
- [BA objective learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/objective-learning-curve.png)
- [BA fit learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/fit-learning-curve.png)
- [BA greedy gameplay learning curve](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/gameplay-learning-curve.png)
- [BA representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-reachability-r1/analysis/representative-gameplay.png)
