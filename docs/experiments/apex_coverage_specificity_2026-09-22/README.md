# Apex coverage specificity — 2026-09-22

> **Outcome:** exact saved-record coverage is **uninformative** for the four
> selected loops. The sealed study took `EXACT_COVERAGE_UNINFORMATIVE`: exact
> absence is common in successful post-first gameplay, so it does not separate
> the loops from successful references. The prior `FAILURE_GUARD` remains, and
> Apex remains the incumbent.

This is a retrospective, saved-record comparison for readers familiar with the
[second-food continuation](../apex_second_food_continuation_2026-09-22/README.md).
It tests whether exact H64-continuation replay coverage can distinguish four
selected recurrent loops from 284 successful saved games. It does not evaluate
a new policy, change training, or explain why a loop occurs.

## Result

All validity checks passed. The comparison assigns equal weight to each saved
game, across all 284 successful references and four selected loops. Every
stratum/view/axis median coverage fraction is `0.000`.

The compact table reports zero-game rates for the three exact-match axes as
`input / chosen action / sampled chosen action`. `full` considers each game's
complete eligible post-first record; `fixed4` uses the first four unique
input-mask/action pairs in frame order. All stratum/view/axis summaries are
available in the unchanged [table.md](table.md).

| Seed and bank | Success N | fixed4 eligible N | Full zero-game rate (input / action / sampled) | fixed4 zero-game rate (input / action / sampled) |
| --- | ---: | ---: | --- | --- |
| 2026099201 known | 48 | 47 | 93.8% / 95.8% / 95.8% | 100.0% / 100.0% / 100.0% |
| 2026099201 fresh | 48 | 48 | 91.7% / 95.8% / 95.8% | 100.0% / 100.0% / 100.0% |
| 2026099202 known | 48 | 47 | 100.0% / 100.0% / 100.0% | 100.0% / 100.0% / 100.0% |
| 2026099202 fresh | 47 | 47 | 91.5% / 95.7% / 95.7% | 100.0% / 100.0% / 100.0% |
| 2026099203 known | 46 | 45 | 91.3% / 93.5% / 93.5% | 97.8% / 97.8% / 97.8% |
| 2026099203 fresh | 47 | 47 | 95.7% / 95.7% / 95.7% | 100.0% / 100.0% / 100.0% |

Chosen-action absence is 93.5–100.0% for the full view and 97.8–100.0% for
`fixed4`. Exact absence therefore cannot be used as a discriminator, causal
explanation, or training-admission criterion. It supplies no basis to train on
the held loops or to promote a policy.

For the prior learner context, use the existing
[TD/Q learning curve](../apex_second_food_continuation_2026-09-22/td-q-learning-curve.png)
and [representative fresh gameplay paths](../apex_second_food_continuation_2026-09-22/representative-fresh-paths.png).
Those figures are linked, not copied or re-rendered.

## Scope and provenance

This completed study generated no games, updates, or forward passes. Guarded
execution took 1.2556108339922503 seconds; peak child RSS was 390,299,648 bytes
and minimum available memory was 35,836,690,432 bytes.

- Source head: `8f96153`
- Intent SHA-256: `d8e7f41424f3ee89c0782899aafcceb3f767a117e7a4169c546c04724c3a2ee2`
- [Completed analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-coverage-specificity/analysis.json): `PASS`, SHA-256 `a54ae791dc795c3c871705d898705e6ef0c817acc096ff4b233cf92c276555d7`
- [Independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-coverage-specificity/audit.json): `PASS`, SHA-256 `f8967b2e496cc484c17953c5ecc8438e0c53e61c7c4ad4419af467902fdf0eab`
- [Sealed closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/apex-coverage-specificity/closeout.json): `COMPLETE_AUDITED_EXACT_COVERAGE_UNINFORMATIVE`, SHA-256 `cd585651a8fd5e3574cfdadcf3b4b4cf19e16f0d5112a83931fba825caeb51c6`
- `table.md` is a byte-for-byte copy of the completed source table, SHA-256 `a50bbbb9ddf33d7173de3751b84f0653bf65746159f4a3135728d9724e4473a1`.

The next valid step, if pursued, is a separately designed bounded
collection-policy question. This census itself admits no new training.
