# Solo food-learning progress

This is a readable map of the audited solo-food research line through September
21. It distinguishes greedy behavior on fresh worlds from training, supervised
fit, and saved-trace diagnostics. Each linked experiment fixes its own parents,
worlds, horizon, and decision rule; a pass in one study does not pool with or
transfer automatically to another.

## Where the line began

The original [exploration-floor screen](solo_exploration_floor_2026-09-13/README.md)
did not establish a remedy. Its frozen all-seed rule required 24 learning and 12
C2 baseline conditions; it passed 16/24 and 6/12. Final floor-.10 food was 1.375,
5.000, and 2.375 across the three seeds. That is an audited failure of the
specified exploration-floor intervention, not proof that exploration is the only
limitation.

A short [food microbenchmark](solo_food_microbenchmark_2026-09-13/README.md)
then separated two learning methods on a narrow food-reaching task. Supervised
behavior cloning passed all three seeds; native PQN passed one of three. The
comparison established a useful short-task contrast, not an algorithm-equivalent
ranking: behavior cloning used frozen teacher trajectories and 500 Adam updates,
while native PQN used its own online experience and valid-hero-step clocks. The
methods shared the simulator, raster observation, network, and action contract,
but differed in objective, experience, and optimization budget. Their comparison
therefore identifies a learning-method gap on that task, not an architecture ranking.

This distinction has remained important. A supervised safety-head sequence could
correct defined labels while preserving a frozen food parent, and its fixed
policies passed its H512 screen. That is evidence for that supervised
input/objective package, not evidence that native PQN has solved the same problem.
Conversely, native PQN reports separate archived TD or teacher-label fit from
greedy fresh-world gameplay. Falling loss or good label agreement alone is not a
behavioral result.

## What current solo behavior supports

The current warm-start cohort comes from the audited [CV explicit-length
continuation](solo_food_pqn_length_access_2026-09-21/README.md). It added two
logical-length scalars while keeping the model family and native PQN continuation
recipe otherwise matched. CV's treatment passed its H512 absolute package in all
three seeds, but its strict paired relative package passed only one; CV's overall
conjunction is false. It therefore supplies a fixed cohort for later screens,
not a clean all-seed causal win for length access.

The audited [CW H768 confirmation](solo_food_pqn_h768_length_confirmation_2026-09-21/README.md)
then evaluated all three fixed CV treatment finals without further optimization.
On 32 fresh worlds with 384 lanes per seed, every policy passed all 37 declared
food-and-survival conditions and all 72 placement-cell checks. At H768, mean
ambient food was 122.5078125, 116.8541667, and 135.3385417; endpoint survivors
were 353, 364, and 383 of 384; and mean survival fractions were .9791667,
.9900140, and .9978468. Each seed also passed the frames-513--768 late-food floor.
This is the strongest current evidence for reliable solo food-seeking and duration
within this fixed cohort and fresh bank.

It has clear limits. CW is a solo duration screen, not a new training result or
an opponent test. Its food and survival outcomes do not establish equal body
growth: the same reports show materially different boost use and mass integrals.
The CV family was chosen from earlier outcomes, so CW confirms a fixed cohort's
retention at a longer horizon rather than estimating a general representation
effect.

## Where opponent behavior fails

The audited [CX one-opponent screen](solo_food_pqn_one_opponent_2026-09-21/README.md)
introduced one native GreedyFood opponent on fresh paired H256 worlds. All three
policies still passed every one of the 17 absolute conditions in solo and
one-opponent conditions, including all 72 placement-cell checks. Food remained
above the independently specified baseline margin. Yet all three failed every
paired retention check: endpoints fell from 383/382/384 solo to 362/355/359 with
an opponent, and the food-retention confidence bounds missed their declared
margins. Exposure was adequate in all 32 worlds for every seed, so the screen is
not inconclusive from weak contact. The all-three CX result is false.

The audited [CY saved-death census](solo_food_pqn_one_opponent_death_census_2026-09-21/README.md)
describes the saved CX traces without running new games or changing the policy
result. Of 76 S2 deaths, 70 were head collisions, five enemy-body collisions, and
one self-collision. The 70 recorded head deaths had nonempty advisory sets; the
other six were fallback cases. Most paired food-count deficit occurred after the
S2 hero had died while its solo counterpart remained alive. These observations do
not show that a different action would have rescued a trajectory, nor do they
identify enemy perception, shared food, action masking, or collision handling as
the cause.

## What is active now

[CZ enemy tactical access](solo_food_pqn_enemy_access_2026-09-21/README.md) has
completed its frozen numerical matrix and passed independent audit. It trained
three fresh paired continuations from the CV finals: `enemy_blind` versus
`enemy_visible`, differing only in tactical enemy body/head/predicted-motion
channels. Every final arm passed its own solo and S2 absolute package. The stricter
learned-progress package passed in one visible seed and no blind seed; the separate
visible-versus-blind benefit passed in that same one seed. The predeclared all-three
learning and access conclusions are false. Equal update budgets still produced
different valid-row experience, and no individual seed substitutes for the
all-three decision. These results remain descriptive until the independent audit
closes and do not establish a causal account of the opponent failures.

Apex remains the operational incumbent. Its larger historical training dose is a
confound, not evidence that Apex is inherently better. None of these screens ran
the shared promotion tournament or authorizes replacement.
