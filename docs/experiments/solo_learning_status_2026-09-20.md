# Solo-learning status: evidence map

**Current milestone:** [CW's completed H768 duration screen](solo_food_pqn_h768_length_confirmation_2026-09-21/README.md) passes all 37 declared food-and-survival conditions and 72 placement-cell checks in each of the three fixed CV length-access finals on 32 fresh worlds. At H768, mean food is 122.51, 116.85, and 135.34; endpoint survivors are 353, 364, and 383 out of 384; mean survival fractions are .97917, .99001, and .99785. Every seed continues collecting at least 32 food during frames 513--768. This establishes the declared longer-solo milestone for this warm-start cohort, not opponent competence or promotion.

CW is complete and independently audited: six scientific jobs, no new training, 565.686 seconds of a 1,200-second cap, and 1,977 audited hashes. The first physical audit had a source-path typo; the corrected second audit passed while preserving both attempts and reusing every completed game. Qualification passed 36 tests and exact H512 raw-prefix parity in 12.269 seconds. Mass and boost behavior remain distinct: seed6003 boosts on 48.35% of frames and has mean mass integral 12.52, compared with 60.97 for seed6002. Food and survival therefore do not establish equal body growth.

[CV's matched explicit-length continuation](solo_food_pqn_length_access_2026-09-21/README.md) remains the training evidence: its absolute H512 package passed 3/3 treatment finals and 2/3 controls, but the separate strict relative package passed only 1/3 and CV's overall conjunction remains false. Its 24 completed jobs added 3,072 updates and 785,926 valid hero transitions; the learning curves retain midpoint degradation and final recovery. CW selected that full three-policy family from CV outcomes and adds fresh-world duration evidence, not a new causal length comparison. Apex remains incumbent pending the shared tournament gate; its larger historical training dose does not establish inherent superiority.

[CX's completed one-opponent screen](solo_food_pqn_one_opponent_2026-09-21/README.md) preserves the solo milestone while locating the next boundary. All three fixed policies pass all 17 absolute conditions in both paired H256 conditions and all 72 cell subchecks. However, each fails all three paired-retention requirements: endpoint survivors change from 383/382/384 solo to 362/355/359 with one GreedyFood opponent, and food-retention confidence bounds also miss the declared margin. Every policy has adequate encounter exposure in all 32 worlds; the all-three CX screen is false. All 11 jobs and the first independent audit completed, with no new learning or repeated gameplay. [CY's completed saved-trace census](solo_food_pqn_one_opponent_death_census_2026-09-21/README.md) records 70 head collisions, five enemy-body collisions, and one self-collision in S2. All 70 head deaths had a nonempty advisory set. Most of the food-count deficit occurs when the solo hero remains alive after its paired S2 hero dies. The independently audited 1.045-second reduction and corrected plots introduce no new games, learning, or change to CX's result.

The preceding [CS H512 screen](solo_food_pqn_h512_control_confirmation_2026-09-21/README.md) had passed in only 1/3 fixed CR controls. [CT](solo_food_pqn_h512_terminal_diagnostic_2026-09-21/README.md) replayed all 98 recorded deaths under advisory fallback (97 self-collisions, one wall), and [CU](solo_food_pqn_fatal_onset_2026-09-21/README.md) found choices remained through offset 1 in every saved window. Those diagnostics motivated CV but do not prove counterfactual rescue or a representation cause.

This map is a recovery guide, not a pooled benchmark. Each linked study has its own fixed parents,
data, worlds, horizon, criteria, and statistical unit. “Passes 3/3” means every predeclared per-seed
requirement for that study passed; it does not transfer a result to another policy family or task.

## Recovered exploration-floor baseline

The earlier [solo exploration-floor experiment](solo_exploration_floor_2026-09-13/README.md)
is complete and independently audited. Its unchanged all-seed rule required all 24
learning conditions and all 12 C2 baseline conditions. It passed 16/24 and 6/12,
respectively: `STAGE_A_SCREEN_NOT_ESTABLISHED`, with no promotion eligibility.

| Training seed | Final floor-.10 food / mass integral / survival | Learning checks | C2 checks |
|---|---|---|---|
| 2026091901 | 1.375 / 1.761475 / 1.0 | 5/8 | 4/4 |
| 2026091902 | 5.000 / 3.334900 / 1.0 | 5/8 | 1/4 |
| 2026091903 | 2.375 / 3.341175 / 1.0 | 6/8 | 1/4 |

The three training receipts, final v5 evaluator receipts, analysis, and independent
v2 audit establish completion. The immutable early intent still says implementation
pending and there is no separate historical closeout file; neither means the completed
runs are missing. Earlier failed evaluator attempts remain part of the repair history.
The completed evidence is reused and the original thresholds are preserved.

## What has been established

| Line | Evidence | What it establishes | Boundary that remains |
|---|---|---|---|
| Food representation | [AN point-versus-raster](solo_food_point_model_2026-09-14/README.md) | A wide food-only raster can learn short H64 food seeking; the compact point model did not fit or play reliably. | H64 games all survived, so this was not a survival stress test. It also changed representation and capacity together. |
| Supervised correction | [BM fresh shared-head](solo_food_shared_head_fresh_init_confirmation_2026-09-19/README.md), [BN H256 evaluation](solo_food_longer_solo_confirmation_2026-09-19/README.md), and [BO H512 confirmation](solo_food_h512_confirmation_2026-09-20/README.md) | A separately trained safety head corrected the defined labels while preserving a frozen food parent. All three fixed policies then passed fresh H512 worlds with 96/96 survivors each and mean food of 77.01, 77.73, and 77.93. | The head receives per-action reachability summaries and learns with supervised labels and a parent-action ranking objective. This is a different input/objective package from native PQN. Each H512 cohort still includes one zero-food lane; aggregate success does not mean every individual behavior is solved. No promotion follows. |
| Native PQN recovery | [CC training dose](solo_food_pqn_training_dose_2026-09-20/README.md) | Continuing the unchanged native PQN recipe to 1,024 updates recovered food relative to its mark-64 parent and met its recovery and survival-retention checks in all three seeds. | Final food exceeded the original released reference with a positive paired interval in only 1/3 seeds, so the all-seed primary result failed. Training and held label fit are descriptive references, separate from greedy fresh-world play. |
| Unanchored continuation | [CA no-CE versus restored-CE](solo_food_pqn_unanchored_continuation_2026-09-20/README.md) | Restoring teacher cross-entropy retained the short-game behavioral package in 3/3 seeds; the unanchored arm retained it in 1/3. | The restored anchor did not produce a reliable relative or beyond-initial improvement. This isolates neither a general PQN cure nor a causal reason for the result. |
| H128 retention | [CF 32-world replication](solo_food_pqn_h128_replication_2026-09-20/README.md) | CC's fixed mark-1024 policies retained H128 food and time across all three seeds. In CF, every H64 condition passed and H128 food/time conditions passed in all three. | Absolute H128 endpoint survival passed only 2/3 seeds: CF seed 5302 ended 342/384, below the unchanged 348 floor. The all-three requirement therefore fails; no averaging or pooling rescues it. |
| Death diagnosis | [CE saved-action H128 deaths](solo_food_pqn_h128_deaths_2026-09-20/README.md) | The 54 recorded H128 deaths replayed exactly as self-collisions under fallback; none was classified as wall, enemy, or advisory-safe fatal. | Replay classification does not prove earlier inevitability, a counterfactual rescue, or the cause of the learned policy's choice. |
| Native H128 milestone | [CG training-horizon comparison](solo_food_pqn_training_horizon_2026-09-20/README.md) | Both fixed final arms pass all 17 food/survival-retention checks in all three seeds on 32 fresh worlds. The six learners added 6,144 updates and 1,568,614 valid hero transitions. | Longer-cap survival superiority passes 0/3; CG overall success remains false. This is a warm-start, bounded H128 result, not from-scratch mastery, long-game reliability, or promotion. |
| Body-aware native PQN | [CK body access](solo_food_pqn_body_access_2026-09-20/README.md) | All six finals retain H64/H128 behavior and H256 food, including late food. Body-aware versus control endpoint intervals are positive in seeds 5501 and 5502 and negative in 5503. | All six miss the 348/384 H256 endpoint floor. Body-arm absolute reliability passes 0/3; its complete relative-benefit rule passes 1/3. This does not establish a reliable benefit from the body-input package. |
| Trace continuation | [CL matched trace length](solo_food_pqn_credit_trace_2026-09-20/README.md) | With equal 512-update doses from shared body-aware parents, current λ=.65 passes H256 absolute criteria in 2/3 seeds; λ=.95 passes in 1/3. | The complete paired benefit rule passes 0/3. Longer-trace seed5602 has 384/384 survivors but only 1.7474 food, below random 4.6771. Longer traces are not a reliable repair. |
| Selector-cut continuation | [CN matched selector-cut](solo_food_pqn_selector_cut_2026-09-20/README.md) | At equal λ=.95 continuation dose, selector-cut passed the H256 absolute package in 2/3 seeds versus native 1/3. | The endpoint-benefit lower CI was below zero in all three seeds; CN overall failed. The 346/384 selector-cut endpoint count in seed5702 remained below the unchanged 348 floor. |
| Loss-shape continuation | [CO Huber versus half-MSE](solo_food_pqn_loss_shape_2026-09-20/README.md) | At equal 512-update continuation doses from shared CK parents, all three half-MSE finals passed the 16 absolute H64/H128/H256 conditions. H256 food was 37.76, 41.05, and 38.23, with 368, 376, and 374 survivors out of 384. | Huber passed the absolute package in 2/3 seeds; half-MSE's paired-effect package failed in all three. Its food retention bounds failed against the parent in seed5801 and Huber in seeds5802/5803. This is a warm-start cohort result on one fresh bank, not a universal loss recommendation. |



## How to interpret fit and gameplay

- **Training fit** asks whether a network matches labels in rows it was optimized on. **Held fit** asks
  the same question on a separated label set. Neither metric is greedy gameplay.
- **Greedy fresh-world gameplay** measures food, survival time, and endpoint survival in a bank not
  used for that study's optimization. The native-PQN reports keep its fresh gameplay separate from
  the archived teacher-label reference sets.
- The safety-head results and native-PQN results answer different questions. The former learns
  corrections over frozen food-Q behavior; the latter updates native TD objectives from warm-start
  parents and then tests unanchored or re-anchored continuation packages. They must not be merged
  into a single “model success” score.
- A two-seed survival result is still a failure when the frozen rule requires all three. CD and CF
  preserve that distinction even where food or relative-retention statistics look favorable.

## Operational decision

Apex remains the operational incumbent. It was trained for a much larger historical budget, which
is a confound rather than evidence of inherent architectural superiority. None of the studies above
ran the shared tournament promotion gate or earned a replacement decision.
Historical Apex logs include runs with hundreds of thousands of learner updates, but
those partial run logs do not establish the incumbent checkpoint's final cumulative
budget. As the [counter audit](../state_and_system_review_2026-09-06.md)
explains, Apex CLI steps and PQN hero transitions are different units. Compare actual
optimizer updates, replay/hero rows, and wall time separately; the current native PQN
line also inherits supervised food training and is not a from-scratch comparison.

CG is complete and audited, including the two preserved operational partials and
their successful saved-evidence verifications. Its H128 absolute milestone does not
erase CF's previous failed replication or establish a causal benefit from the longer
training cap. The subsequent [CH H256 confirmation](solo_food_pqn_h256_confirmation_2026-09-20/README.md)
completed only the scripted-anchor screen: H256 teacher food remained strong, while the
frozen survival-time and endpoint calibration checks failed. All six learned CH roles
were consequently not run. This is an H256 anchor limitation, not evidence that either
CG learned arm regressed. CH is complete and independently audited, with its failed calibration preserved. The separate [CI H256 diagnostic](solo_food_pqn_h256_diagnostic_2026-09-20/README.md) is also complete and audited: all six policies retain food but fail H256 time and endpoint floors. It reused CH anchors and did not rerun them. All H64/H128 diagnostic conditions pass. The subsequent [CJ replay](solo_food_pqn_h256_deaths_2026-09-20/README.md) exactly reproduces all 607 self-collision deaths under empty-advisory fallback; none was advisory-safe fatal. This supports testing earlier body-aware choices, without proving a rescue counterfactual or a causal representation failure.

CK completed that matched body-access test with 3,072 additional optimizer updates
and 778,144 valid hero transitions. Its negative all-seed result is preserved.
CL changes only λ=.65 versus λ=.95 within each pair, restores each complete CK
body-aware network and Adam state, and gives both arms 512 further updates on
fresh training RNG streams. Its fresh 32-world evaluation retains the absolute
food/survival criteria and requires paired endpoint benefit plus food retention.
The three starting policies and completed anchors are measured once; operational
report repairs adopt completed evidence without replay. An earlier
[λ=1 short-task screen](solo_food_lambda1_2026-09-13/README.md) also failed its
all-seed rule, but used a length-one task and reused outcome-informed controls.
It is relevant negative history, not the same survival question as CL.

CL is complete and independently audited, with all 24 current receipts reconciled and all four prior physical attempts preserved. Its six learners added 3,072 updates and 781,367 valid hero transitions. CM is a saved-array diagnostic, with no new policy claim or tournament promotion.

CM is complete and audited: all 3,072 CL rollout batches and 781,367 transitions were reduced without new learning or gameplay. This is diagnostic evidence, not a new policy result.

CO is complete and independently audited. Its six learners added 3,072 optimizer
updates and 781,625 valid hero transitions. The source-metric audit recomputed every
absolute decision, all 216 cell subchecks, and every paired relative confidence
interval; its full input closure passed byte verification. The fixed final cohort
passes the absolute screen even though CO's stricter relative primary failed.
CP completed that duration and fresh-world check without further optimization:
H384 food was 54.88, 59.55, and 56.15; endpoint survivors were 344, 358, and 368
out of 384. Its independent audit passed, with all six completed scientific jobs
reused. It preserves CO's failed primary, CH's failed calibration, and the
distinction between the same three training lineages and new evaluation worlds.

[CQ's saved-trajectory census](solo_food_pqn_h384_death_census_2026-09-20/README.md)
is complete and independently audited. All 82 learned-policy deaths in CP were
self-collisions, including 53 after frame 256. It performed no new learning,
inference, or gameplay and does not prove an earlier rescue was possible.

CR gives each of those full CO model/Adam parents two paired 512-update
continuations with fresh training streams: cap 256 and cap 384. It keeps native
Q(lambda=.65), half-MSE, the body/food inputs, and the six-action contract fixed.
Qualification passed 42 tests, exact first-update agreement, and CPU gameplay
parity. The smoke confirmed post-256 exposure only in the longer-cap arm. Fresh
384-lane greedy evaluations and the unchanged 22-condition absolute package
remain separate from the paired endpoint-benefit and food-retention rules. The
science budget was 6,600 seconds; heavy jobs remained serialized.

CR is complete and independently audited: 24/24 scientific jobs, 3,072 new
optimizer updates, 785,237 valid hero transitions, and 2,180.344256 seconds of
science. Its fixed final control cohort passes all 22 conditions in every seed.
The treatment's seed 5902 fails H128 food, H256 food, and food during frames
129–256 despite 382/384 H384 survivors. The final independent audit checked
7,850 hashes; Sol separately recomputed every final gate and paired interval.
Midpoint behavior was not monotone and no midpoint substituted for a final.
The proposed CS H512 screen retains every shorter-prefix requirement and adds
seven longer-horizon checks, including an absolute mean-food floor of 64. The
fixed control-family choice is outcome-informed; no opponent or tournament
result follows from this solo-food milestone.
