# Panel decision memo: next direction for snake-dqn

Date: 2026-10-15. Chair's synthesis of a two-expert panel: game mechanics and theory, and deep RL.
Scope: read-only. Nothing was trained, modified or pushed. The chair checked every load-bearing
code citation against `/home/user/snake-dqn` at commit cd8eded, the Oct 8 state. Everything about
the Mac trainer `train_m3c.py` is inferred from its ancestor
`research/redesign_m3_20261008/train_m3.py` plus the flags quoted in the digests. Those items are
marked inferred.

Long forms in this folder: `round1_mechanics.md`, `round1_rl.md`, `round2_mechanics.md`,
`round2_rl.md`, `librarian_answers.md`. Source report:
`../mac_session/REPORT_mac_session_2026-10-12_to_14.md`.

---

## 1. Bottom line

B8 already beats the served agent frp3-s12 + v8 on H5000 mass by +128.7, lower bound +101.5, with
survival on par on all three mixes. The one failing clause compares scripted alive-frame survival
with champion_a5 + v8 on one seed's 32 worlds, at a threshold near one standard error, against an
agent that never lives at the lengths where B8 dies. Both experts now agree on the mechanism: the
deaths are late neck-sealing enclosures that B8's learner cannot value, because the Huber loss at
beta 1 fits a median target at a death delta of about 100 Q units, the on-policy margin anchor is a
keep-your-argmax trust region, and replay content cycles with the segment clock. No static feature
tried so far predicts these deaths before the seal, so new inputs are last, not first. The panel
recommends one Mac day of measurement with no training, then a five-seed learner-regime screen,
then rollout-labelled late-game training if the owner rules that training-time labels are not a
veto. The owner must settle the guard clause before any new gate data, because it decides whether
the target is a learned long-snake skill or a smaller snake.

---

## 2. Shared diagnosis

Claims both experts accept after two rounds. Confidence is the chair's reading after checking the
cited code.

| # | Claim | Evidence | Confidence |
|---|---|---|---|
| S1 | The late deaths are sealed-neck enclosures. The census trap flag marks the seal, not the commitment. | 85% of 75 deaths are enclosures. Median trap 31 decisions. 41 of 64 trap onsets had no one-step escape. Trap flag is the one-step tail-aware region in `ego_raster_b.py` dropping below L. A 1,000-cell body partitions the 145 x 83 arena routinely. | High |
| S2 | The whole scripted gap sits at L >= 800. Below 800 B8 is as safe per decision as the champion. | 49 of 75 deaths at L >= 800, where B8 spends 42% of 659,595 decisions. Below 800: 26 observed vs 29 expected at champion rates. | High |
| S3 | The scripted mix is hardest because of greedy_food mechanics, not because of a missing veto. | `ai_snake.py:60-120`: opponents static within a frame, no body avoidance. They hug the hero, die on its head path, drop whole-body corpses, respawn anywhere within 2 frames. Shadow v8 agreed with B8 at 75 of 75 fatal decisions. | High for mechanism, unmeasured in share |
| S4 | champion_a5 + v8's plateau at 800-870 is growth rate times horizon. Its 0 deaths there are uninformative. | Length-blind above L 150, free space capped at 160. Reaches 800 near frame 4,000. 5.6% of decisions at >= 800. Expected deaths at B8's hazard: 1.2. P of 0 is about 0.3. | High |
| S5 | Under the H5000 integral, B8's growth is roughly break-even against its own late hazard. A growth governor is a pure mass loss that serves only the guard. | Death at frame 3,900 and L 1,000 costs about 220 mass. 49 such deaths cost about 67 over 160 episodes. Growth 800 to 1,200 adds about 80 when alive. Measured hazard slope 2.7e-7 per cell per frame is below the break-even 1e-6. | Medium |
| S6 | The TD loss is a median regressor at the death scale. Risk below 50% in aliased state classes is nearly invisible. D20 and DEATHW could not work. | `train_m3.py:430` `smooth_l1_loss(qa, y)`, default beta 1.0. Phi = L/10, Phi(death) = 0, death value -3. Death delta about 100 at L 1,000. Fixed point for class death rate p < 0.5 is q_survive - p/(1-p). Chair verified the code. | High for mechanism, medium for magnitude |
| S7 | "Tried and null" overstates. M6 was underpowered for the effect it sought. | 5 x 32 paired: SE of a survival difference 0.056. Power 63% for +0.11, 23% for +0.05. DEATHW 90% CI covers +0.13. | High |
| S8 | The v8-label margin anchor is a keep-your-argmax trust region. M6 fine-tunes almost certainly ran lambda 1.0 for all 8M. | `--onpolicy-labels --label-frac 1.0`. v8 differs from B8 on 4.5e-5 of decisions, never at a fatal one. `Schedules.anchor_hold` default 8M, verified. M6 argv sets only `--anchor-floor 0.1`. Margin 0.063 vs Q 50-150. | High on mechanism, lambda column unread |
| S9 | The replay defect is cyclic non-stationarity plus rare-event supply, not aggregate starvation. `death_len_mean` is phase-confounded. | All 128 worlds reset per 5,000-frame segment, `train_m3.py:259-301`. FIFO 500k holds about 1,950 frames. Late scripted states are about 30% of transitions time-averaged. DEATHW smoke `death_len_mean` 628 contradicts the unverified 200-235 reading. 1.28M period is inferred. | Medium |
| S10 | The guard clause is weak both ways and mis-targeted. The report's 0.64 target mixes two metrics. | `example_rules.py:85-91,122`: alive-frame surv5, seed 0's bank, champion baseline, -0.05, tolerance 1e-9. Chair verified. Census 0.531/0.656/0.688 are episode counts on a different bank. | High |
| S11 | Perception is not the first lever. Most tail-topology quantities are already inputs. No boundary-TTL input exists. | bv8 carries v7's tail-release landing area capped at 2L. ego2s-b carries uncapped tail-aware region sizes and a K=24 reach plane. M7: 99 hand features add +0.0125 at H100, +0.035 at H300. Not-tight AUC at H100 about 0.59. | Medium |
| S12 | Exact counterfactual rollouts are feasible in all three eval mixes. | greedy_food is deterministic. `fast_anchors.py:111` hashes random_safe by world seed, slot, frame. Frozen-pool opponents are argmax nets. Chair verified. Only humans break determinism. | High |
| S13 | No serving-time planner or hand rule. Training-time labels only. The served artifact stays a bare network. | Both memos. A serving-time rollout cannot run against humans in Play mode. | Agreed |
| S14 | Every screen is 5 paired seeds x >= 96 scripted worlds, reporting episode-level and alive-frame survival on the same bank, plus mass vs the incumbent. | SE table in section 4. | Agreed |

### 2.1 The numbers behind S2 and S4

Census bank, round 52, 32 scripted worlds per arm, episode-level, no veto for B8.

| Agent | Survival | Deaths | Mass | Median L at death |
|---|---|---|---|---|
| B8 s0 | 0.656 | 11 | 652.3 | 1,106 |
| B8 s1 | 0.375 | 20 | 477.9 | 793 |
| B8 s2 | 0.625 | 12 | 637.0 | 1,218 |
| B8 s3 | 0.531 | 15 | 563.9 | 972 |
| B8 s4 | 0.469 | 17 | 522.5 | 934 |
| B8 pooled | 0.531 | 75 | 570.7 | 974 |
| frp3-s12 + v8 | 0.656 | 11 | 454.6 | 643 |
| champion_a5 + v8 | 0.688 | 10 | 368.7 | 520 |

Hazard, derived from the census counts.

| Regime | B8 deaths | Share of B8 decisions | Hazard per 1,000 decisions |
|---|---|---|---|
| L < 800 | 26 | 58% | 0.068 |
| L >= 800 | 49 | 42% | 0.177 |
| champion at L >= 800 | 0 | 5.6% of its decisions | 1.2 expected at B8's rate |
| incumbent at L >= 800 | 2 | 25% of its decisions | 6 expected at B8's rate |

### 2.2 The numbers behind S6

Huber at beta 1 for a state class the network cannot separate, death delta 100.

| Class death probability | Expected value loss | Huber fixed-point shift |
|---|---|---|
| 0.05 | 5.0 | 0.05 |
| 0.20 | 20.0 | 0.25 |
| 0.40 | 40.0 | 0.67 |
| 0.69, inside a trap | 69.0 | death side |

Inside traps the class death rate is about 0.69, 74 fatal of 108, so the median fit already values
the inside of a trap. The deficit is before the seal, where classes are barely separable.

### 2.3 Phase R look 2 as recorded

| Clause | Value | Threshold | Status |
|---|---|---|---|
| Efficacy, mass vs incumbent, 5 seeds | +128.7, SE 16.4, LB +101.5 | LB > 0 | pass |
| c2 scripted surv5 vs incumbent, 5 seeds pooled | -0.003 | >= -0.03 | pass |
| c3 frozen surv5 vs incumbent | +0.011 | >= -0.03 | pass |
| c3 mixed surv5 vs incumbent | +0.038 | >= -0.03 | pass |
| c5 mi10 | +74.3 | >= 20 | pass |
| c6 guard mass vs champion, seed 0 | +136.8 | > 0 | pass |
| c6 guard scripted surv5 vs champion, seed 0 | -0.156 | >= -0.05 | FAIL |

### 2.4 What the guard gap probably is in clause units

Both experts reconstructed the alive-frame gap from the census. A late death near frame 3,500
costs about 0.3 alive-frame. An early death costs about 0.6 to 0.7. The reconstruction gives B8
near 0.80 and the champion near 0.86 to 0.88 on the census bank, a gap of -0.04 to -0.08. The RL
expert's SE of 0.075 for a 32 vs 32 alive-frame difference is the right one for the clause
metric; the mechanics expert's 0.12 is the episode-level SE. The look-2 value of -0.156 is then
about 1.3 SE below the reconstructed gap and 2.1 SE below zero. The chair's reading: the gap vs the
champion is probably real and at the margin, with an unlucky draw on top. The per-world look-2
records on the Mac settle this at zero compute cost and must be read first.

### 2.5 Two chair observations the experts did not make

First, the proposed amendment "comparator = served agent at -0.05" is weaker than clause c2, which
already protects scripted alive-frame survival vs the served agent at -0.03 pooled over 5 seeds.
In substance the amendment deletes the champion comparator and adds floors. The owner should
decide it under that description.

Second, the reward composition has an unresolved inconsistency. If the PBRS term is kept and the
mass term is kappa(1-gamma)Phi(L), then steady-state Q is near 0 and the death target is near
-Phi(L) - 3. Yet qmax reached 120 to 153 and B8_s1's run ended at 134 to 139. That matches Q
near Phi(L), which is what a mass term without the PBRS term gives. One of the two readings of
the Mac reward is wrong. The Huber conclusion survives either way because the death delta is
about Phi(L) in both cases, but the implementation of the Huber fix should be chosen after reading
the m6 source.

---

## 3. Where they still disagree and how to settle it

| Point | Mechanics | RL | Chair's call | Settle by |
|---|---|---|---|---|
| Size of the Huber-knee effect | Positive but modest. A caution tax on whole classes, with a mass cost. Pre-seal classes are barely separable, not-tight AUC 0.59. | Largest single available change. Sub-50% risk becomes visible. D20 and DEATHW type levers start to work on top. | Mechanics is better supported on magnitude. The median fit already values the inside of traps, and the separable middle zone is narrow. RL is right on priority: it is the cheapest untested lever with a verified mechanism. Expect +0.03 to +0.08, not a cure. | T2 primary endpoint, deaths per 1k at L >= 800, and the Q-gap value check from T1 rerun on the new checkpoints. |
| n-step 16 to 32 | Worth an arm now, with one actor at eps 0. | Conditional: only if the commitment horizon from T1 is at or below about 40 decisions. | RL. The arm is code-level because n_step is hard-coded at `train_m3.py:454`, and its mechanism depends on a horizon that T1 measures within a day. Hold it until T1 reports. | T1's H_LAD distribution. |
| Rollout labels: source and density | Labels from late scripted snapshots as the anchor stream. Labels cannot be produced against humans. | Sparse decisive labels only, across all three mixes, with the lambda schedule audited. Dense labels would re-create the trust region. | RL on both counts. Determinism holds in frozen and mixed worlds too, chair verified. Dense on-policy labels are exactly the defect in S8. | T3 reports label density and the fraction equal to the student's argmax. |
| Tail-topology inputs | Demoted to conditional after conceding most quantities exist. Keeps the per-action-at-commitment objection. | Gated third: rises only if T1 shows per-action separation and the probe fails. | Both now gate it on the same test. Order is moot until T1. | T1 rule R3 and the probe AUC. |
| Growth governor form | A reward cap on L is objective-level and legitimate. | A cap leaves eating free, so it does not govern. Needs a negative term above L_cap. | RL is right on the mechanics. Both rank it last and conditional. Moot unless the owner keeps a champion-level floor and hazard rises with L at fixed frame. | T1 hazard crosstab, rule R6. |
| Sealed pocket size | 30 to 60 cells, from the 31-frame median. | Unmeasured. If the region at the flag is hundreds of cells, in-pocket play is half the mechanism. | RL. It is an inference. The census series hold the region size at onset. | Read the `_series.npz` region column at trap onset. |
| gamma 0.999 and terminal hero | Gamma does not change the growth-vs-death ratio under the mass reward. | Withdrawn in round 2 after checking that death is terminal in the TD target. | Resolved in favour of mechanics. The arm-C gamma change was confounded with three other changes. | None needed. |

The remaining disagreements are about magnitude and ordering, not direction. Nothing in the plan
below depends on resolving them before T1 runs.

---

## 4. Recommended direction and the first three experiments

### 4.1 Direction

Treat the campaign as a value-learning problem on a rare delayed catastrophe, in a regime the
training distribution under-samples at the wrong phase. Measure before training. Fix the learner's
loss regime and anchor first because they cost nothing and have verified mechanisms. Bring the
counterfactual contrast to the sealing decision through training-time rollout labels, never
through a serving-time rule. Add inputs only on evidence from the action-contrastive test. Do not
make the snake smaller unless the owner decides the product wants a smaller snake.

Standard error and minimum detectable effect for the screen designs. MDE is for 80% power,
one-sided alpha 0.05, equal to 2.49 times SE.

| Design | SE of a survival difference | MDE |
|---|---|---|
| 32 vs 32, episode-level | 0.125 | 0.31 |
| 32 vs 32, alive-frame | about 0.075 | about 0.19 |
| 5 x 32 paired, episode-level | 0.056 | 0.14 |
| 5 x 96 paired, episode-level | 0.032 | 0.08 |
| 5 x 192 paired, episode-level | 0.023 | 0.057 |

### 4.2 T1. Measurement day. No training.

Veto in spirit: no. Owner decision needed: none.

Hypothesis. The fatal choice is made before the trap flag, at a decision where a surviving
alternative exists and B8's Q does not rank it higher. If so, the limit is value learning, and T2
and T3 are the levers. If no alternative exists within 300 decisions for most deaths, the deaths
are environmental and the guard question dominates.

Precondition. GridBatchSim deep-copy parity: replay 2 census worlds for 1,000 frames from a deep
copy and compare world digests frame by frame. Must be bitwise identical. Time the copy and a
300-frame rollout at E=1 and at E=16. The Oct 3 bench gave 0.49 ms per BatchSim copy at E=1.

Part A. Exact-replay commitment and contrast study. Data: the 75 B8 deaths from the census, 5
finals x 32 round-52 scripted worlds, plus the 11 incumbent and 10 champion deaths as reference.
Procedure per death at frame f: scan decisions f-300 to f-1 at stride 10. At each, for every
masked-legal action including boosts, clone the sim, apply the action, continue with B8 greedy at
eps 0 for 300 frames or to frame 5,000, and record survival. The last avoidable decision LAD is the
last decision with a surviving alternative. Refine LAD at stride 1 inside the last positive window.
H_LAD is frames from LAD to death. Record at LAD and at the trap-flag frame: Q of the chosen action,
Q of the best surviving action, their gap; the bv8 and ego2s-b scalars of both actions; whether the
surviving action is a boost; whether its next cell's tail-aware component contains the hero's tail;
the minimum own-segment TTL on each component's boundary; the owner of the cell whose occupation
split the head's component from the tail's component, hero or opponent slot; whether a live
opponent head was inside the final pocket at trap onset; region size at onset and its series;
frame of death. Rollout budget: about 75 x 30 x 4 x 300 plus refinement, roughly 4M env-frames,
a few Mac hours single-threaded.

Part B. Metric reconciliation and hazard, zero compute. From the Phase R look-2 shard records:
per-world surv5 and mi5 for B8_s0 and champion_a5 + v8 on seed 0's scripted bank, and the
incumbent on all five banks. From the census per-death rows and `_series.npz`: frame of death for
all 96 deaths, alive-frame survival for all seven arms on the census bank, and hazard per 1,000
decisions by L bucket 600-800, 800-900, 900-1,000, 1,000-1,100, 1,100+ crossed with frame bucket
2,000-3,000, 3,000-4,000, 4,000-5,000. Also region size at trap onset.

Part C. Representation probe. Re-run the M7 capture on the 9 arms with a forward hook on the 256-d
fuse layer, `ego2s_network.py:57-59`, about 150 to 235 s per arm. Fit logistic regression,
5-fold by world, for death within 50, 100 and 300 decisions at L >= 800. Compare with the scalar
GBM at 0.91, 0.811, 0.695 and LR at 0.888, 0.825, 0.726. Add a contrastive fit: LAD states vs
matched non-LAD states at similar L and frame.

Optional, if Mac time remains. Resume the seed-ensemble track of workflow wf_2ee9a0af-657, mean-Q
and majority vote of the 5 B8 finals on 96 scripted worlds, about 1 hour. Run champion_a5 + v8 on
the other four Phase R banks, both metrics, as a diagnostic.

Routing rules, committed before running.

| Rule | Condition | Consequence |
|---|---|---|
| R1 | >= 60% of the 49 late deaths have a surviving alternative within 150 decisions and median H_LAD <= 100 | T3 is eligible |
| R1' | < 30% have any alternative within 300, or median H_LAD > 300 | T3 is off. Deaths are environmental or need long planning. Guard question dominates. |
| R2 | Q-gap at LAD < 2 Q units in a majority, or probe AUC >= 0.75 at H100 | Value learning is the limit. T2 is the main lever. |
| R2' | Q-gap > 5 favouring the fatal action and probe AUC < 0.65 | Perception is the limit. Tail-topology inputs take T3's slot. |
| R3 | Surviving action differs on tail-in-region or boundary TTL in >= 70% of LADs | Tail-topology inputs GO as a later arm. Below 50%: NO_GO. |
| R4 | A boost at LAD survives in >= 20% of late deaths | Include boost labels in T3 |
| R5 | Reconstructed alive-frame gap vs champion in [-0.08, -0.02] | Any amendment memo reports the guard failure as at the margin, one bank |
| R6 | Hazard at fixed frame bucket rises >= 2x from L 800-900 to 1,100+ | A reward-side governor can pay. Flat: drop every governor form. |

Cost: about one Mac day including code, reusing `census.py` on branch census-b8 and the M7 capture
in the m7-feature-auc worktree. No pod.

### 4.3 T2. Learner-regime screen: Huber knee x anchor off.

Veto in spirit: no. Owner decision needed: none if Mac-only.

Hypothesis. Putting the death delta inside the quadratic regime of the TD loss, and removing the
on-policy margin trust region, lets Q rank the sealing action below its alternatives. Predicted
effect on scripted survival: +0.03 to +0.08 with a mass cost under 30.

Arms. All are 8M fine-tunes from the B8 finals with the M6 CTRL flags: `--gamma-start 0.998
--eps-end-at 1`, batch-scale 2, obs-pack, 128 worlds, 2 actors.

| Arm | Change | Implementation |
|---|---|---|
| CTRL | none | Reuse M6 CTRL_s10 to s14 finals, re-probed on the new bank. Valid because the new arms differ only by the lever. |
| HUBER | TD loss beta 100 at the current Q scale, effectively MSE below residual 100 | One flag on the loss. Fallback beta 30 if td diverges. Re-base the qmax tripwire. |
| ANCHOR0 | lambda 0 and label_frac 0 from step 0 | `--anchor-hold 0 --anchor-floor-at 1 --anchor-floor 0 --label-frac 0`. Note: `--anchor-floor 0` alone leaves lambda at 1.0 during the hold. |
| HUBER+ANCHOR0 | both | as above |

The chair prefers the beta route over the reward x0.01 route. Rescaling rewards and the output
layer by 0.01 is exact for Q, but it shrinks the output-layer weights 100x under an Adam step size
that does not shrink, which risks instability. If the m6 source shows the PBRS term is kept and Q
sits near 0, beta 100 still covers a death delta of 100.

Seeds and worlds. 5 paired seeds: fine-tune seed 10+k from B8_s{k} final. Probe with no veto on 96
scripted worlds per seed on a fresh dev-bank namespace, not round 52, plus 32 frozen and 32 mixed
per seed. 480 paired scripted episodes per arm.

Endpoints. Primary: paired mean difference vs CTRL in scripted episode-level survival, SE about
0.032. Co-primary, reported always: the same difference on alive-frame surv5. Secondary: mass vs
frp3-s12 + v8 on 32 worlds x 3 mixes; deaths per 1,000 decisions at L >= 800 from the census
harness on the primary arm; hold-out v8 agreement at 8M from the trainer log; episode-level
survival on frozen and mixed. Interaction HUBER x ANCHOR0 with its SE, about 0.045.

Stopping rule. One look at 96 worlds. GO if the primary >= +0.06 with mass delta >= -30, no other
mix down more than 0.05, and hold-out agreement >= 0.6. KILL if the primary <= 0. If the primary is
between 0 and +0.06, extend once to 192 worlds and stop. Adopt a single arm only if its own cell
passes. Collapse tripwire: hold-out agreement below 0.6 or mass drop beyond 60 kills the arm.

Mechanistic check. Rerun T1 Part A's Q-gap readout on the winning checkpoints over the census
deaths. The Q-gap must open at least 10 decisions before death in a majority of late deaths for the
Huber mechanism to be credited. If ANCHOR0 alone moves survival, M6 is relabelled uninformative
about reward levers.

Conditional fifth arm. NSTEP24, one actor at eps 0, only if T1 reports median H_LAD <= 40. It is
code-level because n_step is hard-coded.

Cost. 3 new arms x 5 seeds x 8M = 120M transitions, about 6.2 h at 5.4k per second, about 3.5 h
wall in two lanes if RAM allows. Probes: 20 seed-arms x 160 worlds, about 2.3 h. Code: one loss
flag, half a day. Pod optional at $3 to $5 for a pinned 4090 with no fallbacks.

Follow-on if GO. Retrain 20M from scratch with the winning recipe, 5 seeds, about 1 h each at
5.4k per second, and enter a fresh-bank Phase R under whatever rule the owner has fixed.

### 4.4 T3. Rollout-labelled late-game training, training-time only. Gated by T1 rule R1.

Veto in spirit: borderline. The served artifact is a bare network and the lookahead exists only in
the training data, as the vetoed teacher already did for distillation. Disclose it in the result
doc. Owner decision needed: yes, see section 5.

Hypothesis. The information that separates the fatal action from the surviving one at the
commitment decision is "what happens if I keep going this way for 30 to 100 steps", which no
static feature carries. Exact rollouts supply that contrast as labels without bootstrapping.

Step 1, ceiling test. B8 plus a depth-300 rollout shadow at L >= 600 decisions on 96 scripted
worlds, 1 to 2 seeds. For each masked-legal action run a B8-greedy rollout. Choose the action whose
rollout survives longest, ties to B8's argmax. Report episode-level and alive-frame survival and
mass. GO to step 2 only if scripted episode-level survival >= 0.65, SE about 0.05. If B8 plus
lookahead does not survive more, its labels cannot help. Cost about 8 to 10 Mac hours at 0.4 ms per
env-frame batched, to be re-estimated from T1's timing.

Step 2, labelled training. Build a snapshot bank of late states, frame >= 2,500 and L >= 700, from
B8 play in all three mixes at the training ratio 75% scripted and 25% mixed, plus frozen. At a
sparse subset of decisions, region drop, tight flag, or every 50th decision at L >= 700, roll out
each legal action with B8 greedy for 300 frames. Keep only decisive states: the chosen action dies
within 300 frames and some alternative survives. Label is the surviving action set. Fine-tune from
the B8 finals for 8M with the TD loss at the T2-chosen regime, v8 labels off, and the margin loss on
decisive labels only, lambda 1.0 for 2M then floor 0.1. Restart 25% of training worlds from the
snapshot bank. Arms: CTRL at the T2 regime, LABELS, LABELS plus regression of Q(chosen) toward the
realised 300-frame rollout return. 5 paired seeds. Probe as T2 plus a transfer check on frozen and
mixed.

Stopping rule. As T2: GO at primary >= +0.06 with mass >= -30 and no mix down more than 0.05; KILL
at <= 0; extend once to 192 otherwise. Report label density, the fraction of labels equal to the
student's argmax, hold-out agreement and the margin-loss trend for the S18-type conflict.

Cost. 2 to 3 days of code: sim clone, batched rollouts, labeler, LabelBuffer swap, snapshot
restarts. About 4 h Mac training and 3 to 6 h of rollouts. Pod optional.

### 4.5 Conditional levers after T1 to T3

| Lever | Trigger | Cost |
|---|---|---|
| Replay reservoirs keyed on mix, segment phase and L; segment desync | Checkpoint-phase probe shows survival or mass swing above 0.03 or 20 across one 1.28M cycle | About 1 day trainer code |
| Calibrated death-probability auxiliary head | Probe AUC between 0.65 and 0.85 at H100 | About 1 day |
| Tail-topology inputs: per-action tail-in-region, boundary TTL, free run ahead, global own-TTL plane | T1 rule R3 GO and R2' | Featurizer, numba kernel, parity tests, 5 x 20M retrain, about 7 h |
| Boost labels | T1 rule R4 | Falls out of T3 |
| Corpse-bait discount | >= 50% of traps began while eating corpse within 2 cells of own body | Reward flag, 8M fine-tunes |
| Reward-side growth governor | Owner keeps a champion-level floor and T1 rule R6 fires | Trivial; concedes the mass lead |
| Hand rules at serving time | Excluded | Veto in spirit |

---

## 5. Decisions only the owner can make

### 5.1 The champion guard clause

What it is. Clause c6: mean over seed 0's 32 scripted worlds of alive-frame surv5 of the hero minus
champion_a5 + v8, threshold -0.05, tolerance 1e-9. SE about 0.075 on this metric. Under equal
safety it fails about 20% of the time; under a true -0.10 harm the sequential design's own table
puts P(GO) near 0.39. Its comparator never lives at L >= 800, so it mostly measures whether the
candidate stayed small.

Options.

| Option | What changes | Consequence for B8 |
|---|---|---|
| A. Keep the comparator, fix the power | Champion guard re-registered on all five banks, >= 96 worlds, pooled, same -0.05 on alive-frame | B8 probably fails on episode-level and is near the margin on alive-frame. Campaign continues with T1 to T3 aiming at a learned long-snake skill. |
| B. Remove the champion comparator, add floors | Rely on c2 vs the served agent at -0.03 pooled, which already exists. Add a per-seed floor of -0.15 vs the served agent on surv5. Add an absolute pooled episode-level floor of 0.50 on every mix. Report length-matched hazard as a secondary. | B8 as trained is likely promotable in one fresh-bank Phase R. It passes the 0.50 floor narrowly at 0.531 on the census bank. |
| C. Keep the comparator, switch to episode-level | Product-faithful if the goal is "rarely dies visibly" in Watch mode | B8 fails clearly on scripted: 0.531 vs 0.688. |

Discipline that applies to any option: amend once, prospectively, before any new gate data; fresh
banks; never re-score B8's look-2 data; publish a sensitivity table of B8 under the old and new
clause on both metrics; disclose the PARTIAL and its role in motivating the change; measure the
champion on all five banks now as a diagnostic; no second amendment after a second failure.

Panel recommendation. B if the owner's goal is the one written in STATUS.md, "beat frp3-s12 + v8
without a veto". A or C if the product value is that the Watch snake rarely dies against bots. The
chair adds: under B, run the fresh-bank Phase R on B8 as-is in parallel with T1 and T2, since it is
Mac CPU time with no training and nothing in T1 or T2 changes B8. Under A or C, state the product
goal explicitly in the pre-registration so the clause has a justification other than history.

### 5.2 The planner question

Does a training-time rollout planner that produces action labels count as "no veto"? Options: yes
with disclosure, serving stays a bare network; or no, nothing beyond TD and the existing v8 labels.
Also: may the label bank include frozen and mixed worlds?

Panel recommendation. Yes with disclosure, on both counts. The student was already distilled from a
vetoed teacher. A serving-time lookahead remains excluded; it is a veto in spirit and cannot run
against humans in Play mode. If the owner says no, T3 is off and the campaign rests on T2 and the
replay levers.

### 5.3 Pod spend

T1 is Mac-only. T2 is about 9 Mac hours or about $3 to $5 on a pinned 4090 with no fallbacks; 10 of
the last 11 pod attempts failed or fell back. STATUS.md's rule requires a projected 4x speed-up for
pod use; the one successful 4090 gave 6.0x on training but CPU-poor hosts slow the actors.

Panel recommendation. Mac for T1 and T2. Cap this round at $10. Use a pod only for T3's rollouts or
to parallelise T2 if the owner wants results in one day rather than two.

### 5.4 Candidate eligibility for the next Phase R

The M4 rule names a 20M candidate. T2 and T3 produce 8M fine-tunes on top of 20M. Options: allow
fine-tuned checkpoints with full provenance, or require a 20M from-scratch retrain with the winning
recipe before entry.

Panel recommendation. Pre-register either, but decide before T2 reports. The chair prefers the
20M retrain: it costs about 1 h per seed at 5.4k per second and removes a provenance argument.

### 5.5 Getting the record off one machine

Local main has four docs-only commits and all working code sits on branch m6, with census-b8,
perf-integrate-results and the m7-feature-auc worktree commits unmerged. Nothing has been pushed
since Oct 8.

Panel recommendation. Push main, m6, census-b8, perf-integrate-results and m7-feature-auc before
the next experiment. Fix the census README wording the verifier flagged: 12 of the 30 shadow-v8
disagreements fall inside 11 death episodes, 69 to 2,869 decisions before death.

---

## 6. Context the panel still needs that is not on this machine

### 6.1 On the Mac already

| Priority | Item | Why |
|---|---|---|
| Must | `train_m3c.py` on branch m6: TD loss and beta; n-step; the exact reward composition, PBRS kept or not, death value; replay capacity and sampling; whether actors are staggered; LabelBuffer sampling; the lambda column of M6's `log.jsonl` | Every T2 arm is a flag or a one-line change only if these hold. The qmax inconsistency in 2.5 and the M6 reading both turn on it. |
| Must | Phase R look-2 shard records: per-world surv5 and mi5 for B8_s0 and champion_a5 + v8 on seed 0's scripted bank; incumbent per bank | Settles what the clause measured and whether -0.156 is a marginal gap plus noise. Zero compute. |
| Must | Census `runs/*.json` per-death rows and `_series.npz`: frame of death, region size at trap onset, boost involvement, L and frame per decision | Alive-frame survival on the census bank, the hazard by L x frame crosstab, and T1 Part B all read from these. |
| Should | CTRL_s10 `log.jsonl` and `telemetry.jsonl` over 8M | Verifies the 1.28M sawtooth period and the actor phase, replaces the phase-confounded `death_len_mean` reading. |
| Should | M7 capture rows and code in the m7-feature-auc worktree | Input to the T1 Part C probe. |
| Nice | L60 probes every 5M and D20 probes with their cycle phases; Mac RAM; `--obs-pack` bytes per row | A free 12-point learning curve against phase; bounds reservoir capacity. |

### 6.2 Needs a new measurement

| Priority | Item | Why |
|---|---|---|
| Must | GridBatchSim deep-copy parity and timing | Precondition for T1 and T3. |
| Must | T1 Part A: commitment and contrast rollouts on the 96 census deaths | The one measurement that separates policy error from fate and value error from input deficit. |
| Must | T1 Part C: logistic probe on B8's 256-d fuse features | Zero-training test of whether the representation already carries the death signal. |
| Should | Training death records over one full 1.28M cycle: hero L, frame in segment, mix, eps flag | Quantifies the L >= 800 death supply and eps contamination. Needs a small trainer change. |
| Should | Checkpoint-phase probe: 8 checkpoints over one 1.28M cycle, 96 scripted worlds each | Decides reservoirs-only vs reservoirs plus desync. About 8 minutes of training plus probes. |
| Should | champion_a5 + v8 and frp3-s12 + v8 on all five Phase R banks, both metrics | Fixes the comparator estimate for any amendment and the sensitivity table. Diagnostic only. |
| Should | Seed-ensemble eval of the 5 B8 finals on 96 scripted worlds | Near-independent seed deaths, correlation 0.035, predict a large gain if errors are idiosyncratic. Resumable from wf_2ee9a0af-657. |
| Nice | Corpse share of intake at L >= 800, kills per episode, live greedy snakes inside the hero's region at onset | Decides the corpse-bait lever and quantifies the intrusion mechanism. |
| Nice | Which opponent mix the served Watch mode actually uses | Decides which mix the product cares about for option C in 5.1. |

---

## 7. What the librarian answered that changed the picture

| Finding | Effect on the panel |
|---|---|
| The guard clause is alive-frame surv5 on seed 0's Phase R bank. The census 0.531/0.656/0.688 are episode counts on round 52. The report's 0.64 target subtracts the margin from the wrong metric. | Both experts recalibrated the guard gap to about -0.05 in clause units and the RL expert dropped the claim of a -0.10 to -0.15 clause-metric gap. |
| `Schedules.anchor_hold` defaults to 8M and the M6 argv sets only `--anchor-floor 0.1`. | M6 fine-tunes almost certainly ran lambda 1.0 throughout. Every M6 arm was a trust-region experiment. The anchor-off arm became mandatory in T2. |
| bv8 already carries v7's tail-release landing area, capped at 2L. ego2s-b carries uncapped tail-aware region sizes. No boundary-TTL input exists. | The mechanics expert demoted tail-topology inputs from rank 2 to conditional. |
| n-step is hard-coded at 5. No Huber-beta or reward-scale flag appears in any argv. Replay sampling is uniform. | "Zero-code" levers are small code changes. The n-step arm became conditional. |
| The reward is PBRS with Phi(death) = 0 plus death -3 plus a mass term described as kappa(1-gamma)Phi(L). | Confirms the death delta of about Phi(L). Exposes the qmax inconsistency in 2.5. |
| No per-world, per-episode or per-decision number is in the digests: death frames, the hazard crosstab, the lambda column. The census did only one-step counterfactuals. | Part B of T1 is a zero-compute artifact read, and the commitment study is genuinely new. |
| DEATHW smoke `death_len_mean` 628 contradicts the unverified 200-235 reading for CTRL_s10. | Supports the phase-confound view over the starvation view. |
| Counterfactual replays are exact in all three mixes, and the census code already deep-copies GridBatchSim every step. BatchSim copy 0.49 ms at E=1. | T1 is a half-day to one-day Mac job, not a research project. T3's label bank can span all mixes. |
| The 1.28M cycle and in-phase actors are inferences from a qmax sawtooth, not code readings. | The replay-cycling claim is held at medium confidence pending CTRL_s10's log. |
| "41 of 64 trap onsets unavoidable" is a census finding in the raw JSON, not a literature-track claim. | Upgraded from unverified to verified. Strengthens S1. |
| The census verifier flagged the README's "v8 would not have changed any death": 12 of 30 disagreements fall inside 11 death episodes well before death. | The 0 of 75 at fatal decisions stands; the README wording should be fixed before pushing. |
| The champion was never run on seeds 1 to 4's banks. The incumbent was. | The comparator's estimate rests on 32 plus 32 worlds; measuring it on five banks is cheap and needed for any amendment. |

---

## Appendix. Code facts the chair verified at cd8eded

| Fact | Location |
|---|---|
| `smooth_l1_loss(qa, y)` with default beta 1.0; grad clip 10; target sync every 1,000 updates | `research/redesign_m3_20261008/train_m3.py:430, 436-443` |
| `n_step = 5` hard-coded | `train_m3.py:454` |
| `Schedules.anchor_hold = 8_000_000`, lambda 1.0 for t <= hold | `train_m3.py:75-96` |
| Death flushes the pending n-step window as terminal | `train_m3.py:529-531` |
| eps random action uniform over mask-legal actions incl. boosts | `train_m3.py:497-500` |
| Guard clause: `_mix(guard, "surv5", "scripted") >= -0.05`, tolerance 1e-9, guard cell seeds `[seeds[0]]`, baseline champion | `research/sequential_phase_r/example_rules.py:85-91, 122` |
| surv5 reads `prefix_h5000.survival_fraction`; `survival_fraction = alive_frames / scored_horizon` | `hooks.py:140-147`; `src/evaluation/metrics.py:133` |
| Reward: `gamma*Phi(L') - Phi(L)` with Phi(death) = 0, death -3, kill 0.3 per victim length | `src/core/reward_events.py`; `src/simd_env/grid_sim.py:845-868` |
| Region scalars uncapped, tail-aware by TTL > 1 | `src/simd_env/ego_raster_b.py:23-35` |
| Fuse layer `Linear(256+128+n_scalars, 256) -> LayerNorm -> ReLU` | `src/model/ego2s_network.py:57-59` |
| random_safe hashed by world seed, slot, frame | `src/simd_env/fast_anchors.py:111-117` |
| Census: 75 deaths, 67 self, 85.3% enclosures, 41 of 64 onsets unavoidable one step later, 32 of 108 traps escaped | raw census JSON, `scratchpad/p10.txt` line 10 |
| Sequential design OC: harm_guard_scripted -0.10 gives P(GO) 0.39 to 0.47 | `docs/research/sequential_phase_r_design_2026-10-07.md:406-449` |
