# CZ: enemy tactical-access diagnostic

**Status: complete and independently audited.** All final policies
passed their independent absolute food/survival packages in both solo and S2.
The stricter learned-progress package passed in one visible seed and no blind
seeds; the separate visible-versus-blind enemy-access benefit passed in one of
three seeds. The frozen all-three learning and access conclusions are false.

## Fixed paired scope

CZ restored CV length-access final mark-4096 parents 2026096001--2026096003 into
fresh S2 seeds 2026096101--2026096103. `enemy_blind` and `enemy_visible` retained
the parent model, Adam state, learned length inputs, and native PQN recipe.
Both arms zeroed only the newly enabled enemy-input weights and matching Adam
moments before training; their initial model and optimizer digests were equal. The
sole treatment difference was tactical enemy channels 1--3 (body, head, predicted
movement). Both arms trained for 512 updates to 4608; equal update budgets did
not force equal valid-row experience after policy feedback changed trajectories.

Fresh masked-greedy evaluation used 32 new worlds (2026110000--2026110031), four
headings, and three food placements: 384 lanes. S2 has one native GreedyFood
opponent. H64/H128/H256 are exact prefixes. Initial exposure and every final S2
exposure were sufficient in all seeds, so close-range interaction is observed;
proximity remains descriptive rather than a causal explanation.

## Final behavior and frozen decisions

Every one of the 12 final arm-by-condition packages passed all 17 absolute
conditions and all 36 cell subchecks: 204/204 booleans and 432/432 cells. H256
entries below are mean ambient food, survival fraction, and endpoints.

| Training seed | Arm | Solo: food / time / endpoints | S2: food / time / endpoints | Learned progress | Solo retention | Enemy-access benefit |
| --- | --- | --- | --- | --- | --- | --- |
| 2026096101 | Blind | 46.838542 / 1.000000 / 384 | 45.221354 / .963470 / 361 | failed | passed | — |
| 2026096101 | Visible | 42.638021 / .999430 / 382 | 41.195312 / .970256 / 364 | failed | failed | failed |
| 2026096102 | Blind | 38.638021 / .999939 / 383 | 38.572917 / .962199 / 359 | failed | failed | — |
| 2026096102 | Visible | 51.992188 / 1.000000 / 384 | 51.132812 / .980540 / 373 | passed | passed | passed |
| 2026096103 | Blind | 44.041667 / .999257 / 382 | 43.013021 / .975199 / 369 | failed | failed | — |
| 2026096103 | Visible | 46.718750 / .999044 / 383 | 45.468750 / .970144 / 364 | failed | passed | failed |

The learned S2-progress package requires at least eight additional endpoints from the equal initial
policy, a strictly positive paired endpoint-CI lower bound, a food-CI lower bound
above minus 5% of the initial mean, and survival time within 0.01 of the initial
value. Blind passed 0/3; visible passed only seed 6102. The corresponding
visible access benefit requires final visible S2 to exceed blind S2 on the
endpoint CI and retain food: seed 6102 passed, seeds 6101 and 6103 failed.

The endpoint and food 95% CIs below are shown as `[lower, upper]`, based on 32
paired world means with df 31. No lanes or seeds are pooled.

| Seed | Blind learning endpoint / food | Visible learning endpoint / food | Visible minus blind endpoint / food |
| --- | --- | --- | --- |
| 6101 | [-.028554, .028554] / [-.970516, 1.335100] | [-.023983, .039608] / [-5.226293, -2.461207] | [-.033538, .049163] / [-5.396775, -2.655308] |
| 6102 | [-.021229, .057688] / [-2.603380, -.422661] | [.020043, .089332] / [9.801530, 12.292220] | [.005051, .067866] / [11.520981, 13.598811] |
| 6103 | [.002383, .070534] / [-3.372828, -1.127172] | [-.011729, .058604] / [-1.111125, 1.522583] | [-.041686, .015645] / [1.480282, 3.431176] |

This is a screening result, not a familywise population claim or a causal account
of enemy inputs. It does not establish that body, head, motion, shared food,
action masking, or any other single factor caused the observed differences. There
is no best-seed substitution: the three-seed conjunctions remain false.

The saved reductions include [behavioral learning curves](behavioral-learning-curves.png),
[training loss](trainingloss.png),
and [fixed representative gameplay](representative-fixed-gameplay.png).
They are descriptive and do not replace the population criteria.

## Boundary and provenance

CZ makes no promotion claim. Apex remains incumbent, and its greater historical
training dose remains a confound rather than evidence of inherent superiority.
CZ made no new teacher-label fit measurement; native TD regression fit and
greedy fresh-world behavior remain separate evidence. The independent audit
passed 5,319 hashes across 35 jobs after preserving two audit attempts. Attempt
1's six failures came solely from treating training reports as evaluation reports
when checking initial-food equality; the repair restricted that check to `eval-`
reports. There were no scientific failures or reruns. Scientific work consumed
2,640.068946 seconds; qualification used 13.814851 seconds. Peak RSS was
1,320,321,024 bytes, minimum available memory was 29,336,879,104 bytes, and peak
MPS driver use was 1,157,087,232 bytes. The two audit receipts used 47.894351
seconds within their 60-second budget.

- [CZ frozen intent](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-access/intent.json)
- [CZ numerical analysis](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-access/analysis/analysis.json)
- [CZ qualification evidence](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-access/qualification-evidence.json)
- [CZ independent audit attempt 2](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-pqn-enemy-access/audit-attempt2.json)
- [CX completed one-opponent screen](../solo_food_pqn_one_opponent_2026-09-21/README.md)
- [CV completed fixed-cohort evidence](../solo_food_pqn_length_access_2026-09-21/README.md)
