# CE: exact saved-action diagnosis of CD H128 deaths

**Status: COMPLETE_AND_AUDITED.** Every one of the 54 deaths recorded in
CD's H128 archive replays as self-collision under the saved action. All 54 receive the
fallback classification; none is a wall or enemy collision, and none is an advisory
safe-but-fatal event. This describes the terminal transitions. It does not show that an
earlier state was inevitably lost or that another action would have rescued it.

## Scope and method

CE reads the eight completed CD H128 raw14 archives: teacher and random-safe anchors,
then the original BY released-64 and continued CC mark-1024 policies for each of
seeds 2026095301–03. It
replays the recorded action and verifies actions, masks, food events, validity, alive,
mass, length, boost, done, death cause, heads, and directions. It retains 16 saved
predecessor states for each terminal event: 864 predecessor states across 54 deaths.

The replay loads no model, makes no policy decision, changes no optimizer, and computes
no new score. It is a saved-evidence diagnostic, not a fresh game, counterfactual
search, policy evaluation, or promotion test. CD's `overall_success: false` result
therefore remains unchanged.

## Full death record

| Role | Seed | Deaths | First death frame | Surviving lanes / 96 | Verified predecessor windows |
| --- | ---: | ---: | ---: | ---: | ---: |
| Teacher | 0 | 4 | 77 | 92 | 4 |
| Random-safe | 0 | 0 | — | 96 | 0 |
| Original | 2026095301 | 7 | 58 | 89 | 7 |
| Original | 2026095302 | 11 | 81 | 85 | 11 |
| Original | 2026095303 | 12 | 86 | 84 | 12 |
| Continued | 2026095301 | 4 | 109 | 92 | 4 |
| Continued | 2026095302 | 6 | 53 | 90 | 6 |
| Continued | 2026095303 | 10 | 59 | 86 | 10 |

All 54 `death_cause` values are 2 (self-collision). The archive contains zero wall,
enemy, or other terminal causes, zero advisory-safe-fatal cases, and 54 fallback cases.
“Fallback” means no action at the fatal prepared state was both legal and
advisory-safe, so the resolver permitted a legal action despite that empty safe set. It does not certify safety before the fatal state, assess policy
quality, or establish whether a non-recorded action would have changed the outcome.

The continued policy has fewer recorded deaths than its paired original in each seed
(4 vs 7, 6 vs 11, and 10 vs 12), but CE does not attach an effect estimate or causal
interpretation to those counts. CD's fresh-bank, eight-world comparisons remain the
behavioral evidence.

## Guarded execution and next study

Qualification passed five tests in 2.452 of 60 seconds. The one serialized saved-action
replay completed naturally in 9.2092 of 40 science seconds, with no model load,
optimizer update, policy inference, or score calculation. Its receipt records 308,248,576
bytes maximum RSS and 38,694,043,648 bytes minimum available memory. The independent audit and root closeout passed; all 442 hashed files checked by
the auditor matched. Including qualification, peak RSS was 318,996,480 bytes.

The prospective next test is a single fresh H128 replication with 32 worlds and 384
lanes. It will use the scaled endpoint threshold of 348 survivors (87/96) and an
at-most-eight-endpoint relative drop (2/96). It will not pool CD worlds or rescue CD's
failure. If it fails, the plan stops larger replication and investigates the learning,
representation, and horizon boundary instead. This plan has no result yet.

Apex remains the operational incumbent. Its larger historical training budget does not
make it inherently superior, and CE makes no architecture comparison or promotion claim.

## Primary records

- [CE saved-action report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/evidence/report.json)
- [CE frozen design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/design.json)
- [CE qualification](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/qualification-complete.json)
- [CE replay receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/supervisor-runs/replay/receipt.json)
- [CD all-seed H128 comparison](../solo_food_pqn_h128_retention_2026-09-20/allseed-comparison.png)
- [CD fixed gameplay evidence](../solo_food_pqn_h128_retention_2026-09-20/fixed-gameplay.png)
- [CC learning curve](../solo_food_pqn_training_dose_2026-09-20/learning-curves.png)

- [CE independent audit](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/independent-audit.json)
- [CE closeout](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-h128-deaths/closeout.json)
