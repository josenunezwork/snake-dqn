# Round 1 — deep RL / ML panelist memo (role: rl)

Read-only design discussion, 2026-10-14. Sources: the Mac-session report, digest blocks 0, 2, 7, 8, 9,
and the GitHub-state repository at /home/user/snake-dqn (Oct 8, M1–M3). The M4–M6 trainer
`train_m3c.py` is not in this repo; everything about it is inferred from `train_m3.py` (the frozen M3
algorithm it descends from), the flags quoted in the digests, and the telemetry numbers. Each such
inference is marked. I did not run anything and did not touch the repository.

## 0. Summary

B8's scripted-world deaths are a rare, delayed, catastrophic outcome: ~0.18 deaths per 1,000 decisions
at L >= 800, with the trap sealed a median 31 decisions before death. The frozen M3 learner that the
M4–M6 trainers inherit is close to the worst possible instrument for learning the value of such an
event, for four code-visible reasons, in order of how much I would bet on each:

1. **The TD loss is a median regressor at the scale of a death.** `train_m3.py:430` uses
   `smooth_l1_loss(qa, y)` with the default `beta = 1.0`. With Phi(L) = L/10 (`src/core/reward_events.py:30`)
   the death delta at L ~ 1,000 is ~100 Q units and qmax is 120–150 (digests). Every death-adjacent TD
   error sits in Huber's linear regime, where the gradient is sign(e) regardless of |e|. For a class of
   states the network cannot separate, the fixed point is the **median** target, which is the survive
   value whenever P(death | state) < 0.5: a trap with a 20% death probability lowers Q by ~0.25 instead of
   the ~20 that expected value demands (factor ~80). Compounded through 5-step bootstraps, any sub-50%
   death risk is zeroed at every hop. This predicts exactly what the census shows: Q and v8 only "see"
   the death when it is certain, i.e. at the fatal decision. It also predicts, mechanically, the null of
   D20 (−3 → −15 on an implicit −100) and DEATHW (×2 Phi(L)): both change a target whose gradient is
   already saturated. TIGHT (−0.4 per tight decision) *was* in the quadratic regime and did change
   behaviour (tight fraction 2.8% → 2.1%) without changing survival, which is what a symptom-cost should
   do. Confidence: high that the mechanism exists in the visible code; medium-high that train_m3c.py kept
   it (td 0.06–0.34 with qmax 120–153 is consistent with a mostly-quadratic loss plus a rare linear tail).
2. **Credit cannot reach the decision that matters.** n-step 5; target sync every 1,000 updates =
   128k agent transitions (~500 frames at 2 actors x 128 worlds); the 500k FIFO holds ~1,950 frames,
   i.e. ~3.9 target syncs; each stored transition is sampled 3 times (384 agent rows per 128 new ones).
   Propagating a terminal 31 decisions back needs >= 6 bootstrap generations, so along any single
   trajectory the chain cannot complete before its trap-entry transitions are flushed. Everything beyond
   ~20 decisions must come from generalisation across other L >= 800 deaths (~60–95 per 1M transitions,
   ~37 in the buffer at any moment), and item 1 discards their signal.
3. **The replay distribution oscillates with period 1.28M.** `build_segment` resets all E worlds at
   each 5,000-frame segment; L >= 800 is reached around frame 2,900; the FIFO is smaller than one
   segment pair. So the buffer swings from ~0% to ~80% late-game states every 1.28M transitions (the
   qmax sawtooth the literature track saw). This is **not** aggregate starvation (the time-averaged
   L >= 800 share in replay is roughly the eval share, ~35–42%); it is cyclic non-stationarity with a
   two-sync memory, and it makes checkpoint quality a function of phase (20M / 40M / 60M land at phases
   0.6 / 0.25 / 0.875 of a cycle). The `death_len_mean ~ 200–235 near the end of CTRL_s10` reading is at
   phase 0.25 (frame ~1,250, all heroes young) and is confounded by this.
4. **The anchor defends the fatal action.** `--anchor-mode v8margin --onpolicy-labels --label-frac 1.0`
   makes the DQfD margin target the student's own v8-filtered argmax. v8 differs from B8 on 4.5e-5 of
   decisions and agreed at 75/75 fatal decisions, so the margin is a "keep your current argmax by a
   margin" trust region on every on-policy state, including the trap-forming ones. If the M6 fine-tunes
   inherited the B8 anchor schedule (hold 8M, floor at 15M) without override, lambda was 1.0 for the whole
   8M. CTRL fine-tunes reproduce B8's pooled scripted survival to three decimals (0.531), which is what a
   strong trust region predicts.

What this says about the four questions: it is primarily a **value-learning / credit-assignment**
failure on a rare catastrophic event (items 1–2), aggravated by a data-pipeline defect (item 3) and a
regulariser (item 4). Perception is **unproven either way**: the network's Linear(64·8·8) layer sees the
whole 31x31 crop (the "±15 cells receptive field" claim is wrong as stated; the crop is 23 ahead, 7
behind, ±15 lateral), the 8-cell global map covers the arena, and GBM on the scalar inputs already
reaches AUC 0.91 / 0.81 / 0.70 at H = 50 / 100 / 300 while 99 hand features add +0.01–0.035. What the
inputs lack is *topology* at 15–60 cells (is my tail in this action's region? how long is this
corridor?), which the game-mechanics panelist covers. It is not an exploration-coverage problem (the
agent experiences the traps); it is a *contrast* problem: it never sees what the other action would
have done at the sealing decision.

Statistics: M6 at 5 x 32 paired worlds had SE(diff) 0.056, so an 80%-power MDE of 0.14 and 63% power
for the +0.11 the guard needs; DEATHW's 90% CI [-0.07, +0.13] contains the needed effect. "Tried and
null" overstates: those levers were not detectable with that design, and under item 1 they were not
expected to work anyway. The Phase R guard is the least informative clause in the rule (32 vs 32 worlds
on seed 0's bank, point threshold, SE ~0.125, ~34% false-fail under equality) and happens to be the one
that failed; but the independent census (160 vs 32) corroborates a real gap of order −0.10 to −0.15 vs
champion_a5 + v8. B8 vs the served agent on scripted worlds is roughly −0.03 to −0.07 when both
measurements are pooled.

Priorities, in order: (A) one afternoon of measurements that confirm or kill items 1–4 without
training; (B) a 5-paired-seed x 96-world screen of the zero-code levers (reward/Q scale so the death
sits at Huber's knee; n-step 16–32; anchor off; gamma 0.999 and terminal-hero training); (C) data levers
(death-window and length-stratified reservoir, desynchronised segments, snapshot restarts) if the
checkpoint-phase and death-histogram measurements implicate them; (D) auxiliary calibrated
death-probability head (also the representation diagnostic); (E) rollout-labelled expert iteration
through the existing LabelBuffer only after a ceiling test shows B8 + lookahead survives more;
(F) perception last and only if the linear probe shows the representation is the limit.

## 1. Diagnosis of the four nulls (question 3a)

### 1.1 Why the reward and length levers were null

| Lever | What it changed | Why item 1 predicts null | Power |
|---|---|---|---|
| D20 (`--death-value -15`) | death constant −3 → −15 on top of the implicit −Phi(L) ~ −100 | +12% on a target whose gradient is sign-saturated | 1 seed, 32 worlds: SE ~0.12 |
| DEATHW (death x 2 Phi(L)) | −100 → −200 at L = 1,000 | same; identical gradient per death row | 5 x 32 paired: SE 0.060, 90% CI [−0.07, +0.13] |
| TIGHT (−c(1−gamma)Phi(L) per tight decision, L >= 600) | −0.4 per tight decision | in the quadratic regime, so it *was* fitted and tight_frac fell 2.8% → 2.1%; tightness is a symptom of length | 5 x 32 paired: SE 0.055 |
| L60 (60M transitions) | 3x data | changes neither the per-transition death supply nor the median fit nor the 5-step reach; checkpoint phase differs (20M at 0.6, 40M at 0.25, 60M at 0.875 of a cycle) | 1 seed |

The common factor: all four change the *magnitude* or *amount* of a signal that, under a Huber-beta-1
loss at Q scale ~100 with sub-50% death probability, never moves Q at the states where the choice is
still open. A death that is 20% likely within the bootstrap horizon is worth −20 Q units in expectation
and −0.25 under the median fit. The alternative action's advantage (a few pellets, ~0.1 each, plus
(1−gamma)Phi per step ~0.2) is of the same order as 0.25, so the ranking is noise.

### 1.2 The training-starvation hypothesis, against the code

The literature track's claim has three parts; two are right in a different sense than stated.

- "Each actor runs 128 worlds through 5,000-frame segments that start together" — confirmed in
  `train_m3.py:build_segment` (new `GridBatchSim` per segment, all worlds at frame 0). Inferred for
  train_m3c.py from the 1.28M qmax sawtooth = 2 actors x 128 x 5,000.
- "Replay 500k FIFO smaller than one segment pair; L >= 800 flushed ~1,950 frames after the next
  reset" — arithmetic confirmed (500k / 256 transitions per frame = 1,953 frames). But this is not
  starvation: averaged over a cycle the L >= 800 share of sampled rows is ~0.42 x 0.84 ~ 35%, the same as
  eval exposure. What it is: the buffer is ~100% early-game for ~1,500 updates and ~80% late-game for
  ~1,000 updates, alternating, with 2–4 target syncs per phase. A rare-event value learned in one
  phase has nothing holding it during the next.
- "Training mean death length 200–235 vs eval median 974" — the quoted number is from the last
  100k-transition interval of an 8M run, which is phase 0.25 of a cycle (frame ~1,250; every hero was
  reset 1,250 frames earlier and is at L ~ 200–300). It cannot be compared with an episode-level eval
  median. The smoke's cumulative death_len_mean of 628 at 1M (frames 0–3,900) is the better hint and it
  is consistent with training deaths happening at roughly eval lengths.
- "Each transition sampled ~3 times" — confirmed: 384 agent rows per update / 128 new transitions.

A second data defect the track did not name: at eps = 0.01 the hero takes ~21 random actions per
world-segment at L >= 800 (2,100 decisions). The resolved mask blocks only immediate collisions, not
pockets. If even 1% of those are fatal, eps-induced deaths (~0.21 per world-segment) rival the genuine
enclosure deaths (~0.37), and their uncorrected 5-step windows teach "this state was deadly" at the
wrong states. Untested; the measurement below carries an eps flag for this reason.

### 1.3 One afternoon of measurements (no training)

All on the Mac, each < 1.5 h, none is gate evidence.

1. **Training death histogram.** From one 8M fine-tune's actor telemetry (add a 20-line per-death
   record if train_m3c.py lacks it: hero length, frame within segment, mix, whether an eps action
   occurred in the last 5 decisions). Tabulate deaths per 1,000 decisions by length bin x mix x
   eps-flag over >= 1 full cycle, and compare with the census hazard (B8: 0.068/1k below 800,
   0.177/1k above). Kills "starvation" if the L >= 800 death supply per decision matches eval; confirms
   the eps contamination if the eps-flagged share at L >= 800 exceeds ~30%.
2. **Checkpoint-phase probe.** Fine-tune B8_s0 for one cycle (1.28M transitions, ~4–5 min at 5.4k/s)
   with a checkpoint every 160k; probe the 8 checkpoints on 96 scripted worlds. If scripted survival
   or mass swings by more than ~0.05 / 30 with phase, item 3 is confirmed as harmful and desynchronised
   segments or a multi-cycle replay is the fix; if flat, drop the replay-size lever.
3. **Linear probe on B8's own features.** The M7 capture already holds ~166k L >= 800 rows with
   death-within-{50,100,300} labels. One forward pass per row (~10 min) gives the 256-d fuse features;
   fit a logistic / small GBM out-of-fold by world. If AUC at H = 100 is >= the scalar GBM's 0.81 while
   the Q-gap at those rows does not separate chosen from alternative actions, the representation is not
   the limit and the problem is value learning (items 1, 2, 4). If AUC is well below 0.81, perception is
   on the table.
4. **Counterfactual rollouts on the 75 deaths.** Scripted worlds are deterministic given the hero's
   actions (`fast_anchors.random_safe` keys its RNG by world seed, slot, transition; greedy_food is a
   deterministic kernel). Clone the sim (`observation_probe.clone_sim` = deepcopy, 0.49 ms at E = 1 for
   BatchSim; GridBatchSim to be verified) at each of the 60 decisions before death, play each legal
   alternative with B8 greedy for 300 frames, record survival and the Q-gap between the chosen and the
   best surviving action. Output: the last-avoidable-decision distribution for the long-snake regime
   (the trap-horizon doc measured median 4 frames for the short champion under a static-world model;
   the long-snake answer is unknown) and whether Q "knew". ~75 x 60 x 5 x 300 env-frames, under 1.5 h.
5. **Seed-ensemble eval.** Mean-Q over the 5 B8 finals on 96 scripted worlds (~40 min). Seeds die in
   nearly independent worlds (pairwise r = 0.035). If the ensemble survives >= 0.62, the deaths are
   idiosyncratic value errors and a distilled ensemble is a cheap no-veto candidate; if it survives
   ~0.53, the errors are shared and structural.
6. **Champion on all five Phase R banks** (160 scripted episodes, ~1 h). Fixes the comparator's SE from
   0.08 to 0.037. Diagnostic only; see section 5 for how it may and may not be used.

## 2. Which kind of failure (question 3b)

- **Value estimation / credit assignment (primary).** Items 1, 2 and 4 are all mechanisms by which a
  rare catastrophe fails to change Q at the states where the choice is open. The decisive evidence is
  the census pattern itself: Q (and v8) recognise the death only when it is certain. A median
  regressor cannot do anything else. The L-dependent hazard is learnable in principle from the inputs
  (L is a scalar input via region_frac = R/L; GBM gets 0.81 at H = 100), so this is not "cannot
  represent the hazard", it is "the loss does not reward representing it".
  Implication: fix the loss regime and the reach (levers 1–4) before anything else; they are nearly
  free and the screen is cheap.
- **Data distribution (secondary).** Not starvation in aggregate, but cyclic non-stationarity with a
  short memory, plus a rare-event supply of ~60–95 L >= 800 deaths per 1M transitions each used 3
  times. Implication: a death-window reservoir and length stratification (and desynchronised segment
  phases) raise the effective supply by an order of magnitude at trivial cost; snapshot restarts are
  the heavier version.
- **Perception (unproven).** The crop is global over 31x31 via the FC layer, the 8-cell global map
  covers the arena, and hand features add little to death prediction. What is missing is explicit
  topology (tail-in-region, corridor length) at 15–60 cells, which neither raster makes cheap for a
  3-layer conv. Measurement 3 decides. Implication if it fails: add topology scalars (cheap) before a
  mid-resolution plane (expensive).
- **Exploration (contrast, not coverage).** The agent experiences the traps; it never sees the
  counterfactual at the sealing decision, so the value *difference* between actions must be inferred
  from generalisation. Snapshot restarts with forced alternatives, and rollout labels, supply the
  contrast. Implication: these are data-efficiency tools, not the first fix.

Two objective-level facts belong here too. (i) gamma = 0.998 values a death at L = 1,000 as −Phi(L) −3
~ −103, equivalent to ~500 frames of holding; with a terminal hero in the eval profile
(`runtime_contract.hero_terminal`; the training sim respawns, `allow_respawn=True`), the H5000 metric
loses the remaining 2,100 frames at frame 2,900, i.e. ~4x more. The training objective is more
reckless than the metric. (ii) The reward's growth terms (+0.1 per pellet, +0.2 per step at L = 1,000)
are in Huber's quadratic regime and are fitted accurately. B8 is therefore an accurate mass-greedy
agent that is blind to sub-50% death risk, which is precisely its phenotype.

## 3. Ranked levers (question 3c)

Screen design used below unless stated: 5 paired 8M fine-tunes from B8_s0..s4 (as M6), scored no-veto
on 96 scripted worlds per seed (480 per arm; SE(diff) ~0.032 on survival), plus deaths per 1,000
decisions at L >= 800 as the secondary (Poisson; 49 vs ~29 deaths is 2.3 sigma). GO on the primary
needs paired mean delta >= +0.06 (1.9 SE) with mass delta >= −30; KILL at delta <= 0; otherwise extend
to 192 worlds. Each screen is ~2 h of training at 5.4k/s plus ~2 h of probes.

1. **Put the death at Huber's knee** (rank 1; veto-in-spirit: no). Scale rewards by 0.01 and the output
   layer of the warm-start checkpoint by 0.01 (exact), rescale the margin constant, keep beta = 1 (or
   equivalently beta ~ 100 on the old scale; or MSE with reward scale 0.01). Fits: turns the median fit
   into an expectation fit at the death scale, so P(death) = 0.2 now costs ~20 instead of ~0.25.
   Expected: the largest single change in late-game caution available; may cost some mass. Cost: a
   flag and a one-off checkpoint rescale; re-base the qmax tripwire. Risk: larger effective gradients on
   death rows (grad-norm clip 10 then lets a death row dominate its batch); start with beta ~ 30–50 if
   the ×0.01 variant is unstable. Go/no-go: the screen above, plus measurement 4 re-run on the new
   checkpoint: the Q-gap must open >= 10 decisions before death on a majority of the 49 L >= 800 deaths.
2. **n-step 16–32 returns** (rank 2; no). The trap median is 31 decisions; with n = 24 the terminal
   enters the return of the sealing decision directly instead of through 5–6 bootstraps. Pair with
   eps = 0 on one actor (or tree-backup correction) because off-policy contamination of a 24-step
   window at eps = 0.01 is ~21%. Cost: a parameter in the vectorised n-step builder (inferred to
   exist). Risk: return variance; interacts with lever 1 (without it the longer window still fits the
   median). Go/no-go: same screen, run as a 2x2 with lever 1 if compute allows (4 arms x 5 seeds).
3. **Anchor off in fine-tunes** (rank 3; no). `--anchor-floor 0` from step 0 (and confirm the lambda
   schedule actually used in M6; if lambda was 1.0 throughout, every M6 arm was a trust-region
   experiment). Fits item 4. Cost: zero. Risk: drift from the distilled prior; M3-C's collapse was
   reward-driven, not anchor-driven, so the risk is moderate. Go/no-go: paired screen; also report
   hold-out v8 agreement to catch collapse.
4. **Death-window and length-stratified replay, desynchronised segments** (rank 4; no). Two extra
   `Replay` objects: a reservoir of every transition within 100 decisions of a death (tiny, keeps the
   whole run) and a length-stratified reservoir for L >= 800, sampled at fixed shares (e.g. 20% / 30% /
   50% uniform FIFO). Stagger world phases (random first-segment length per world) so the FIFO's
   composition is stationary. Fits items 2–3 and the rare-event supply. Cost: ~1 day; RAM is the
   constraint for a plain bigger FIFO (--obs-pack: 6–7 GB per 500k), so stratification beats size.
   Risk: stale off-policy rows in a 20M run (mild in fine-tunes). Go/no-go: measurement 2 first; if the
   phase swing is < 0.03 skip the desync and keep only the reservoirs; then the screen.
5. **Calibrated death-probability auxiliary head** (rank 5; no). A sigmoid head on the fuse features
   trained with cross-entropy on hindsight death-within-{32, 128, 300} labels for L >= 600 rows (a
   proper scoring rule, so unlike the Q head it *does* learn sub-50% risks). Use it (a) as the
   representation diagnostic (its out-of-fold AUC vs the GBM's 0.81), (b) as a shaping input to the
   value, e.g. Q = Q_survive − p_death·Phi(L), or simply as an auxiliary loss. Cost: ~1 day. Risk:
   label leakage via eps deaths (use the eps flag). Go/no-go: AUC >= 0.85 at H = 100 on census worlds
   is GO for using it in the value; then the screen.
6. **Horizon alignment: gamma 0.999 and terminal-hero segments** (rank 6; no). Doubles the value of
   survival relative to growth (death cost ~ 1,000 frames of holding instead of ~500) and removes the
   respawn-vs-terminal mismatch with the eval metric. This is the legitimate, learned form of a growth
   governor; it will cost mass and the question is how much. Cost: flags; qmax tripwire re-based.
   Risk: slower, noisier TD at gamma 0.999 with n = 5 (pair with lever 2). Go/no-go: screen; GO requires
   survival +0.06 **and** mass >= incumbent + 60 so that c1 (+20) stays safe on fresh banks.
7. **Rollout-labelled expert iteration through the existing LabelBuffer** (rank 7; borderline). The
   trainer already has on-policy label collection and a margin loss; swap the one-step v8 labeler for a
   depth-100–300 rollout planner on L >= 600 decisions (scripted worlds are exact), label sparsely
   (tight or region-drop states, or 1 in 50 decisions), and train the margin only where the rollouts
   show a decisive survival difference. Serving stays a bare network. Cost: 2–3 days (clone, batched
   rollouts, labeler). Risk: see section 6; the labels are only learnable if the representation can
   separate them, and they encode privileged opponent futures. Go/no-go: **ceiling test first**:
   B8 + the same lookahead as a shadow policy on 96 scripted worlds must survive >= 0.65; if the planner
   itself does not close the gap its labels cannot. Then the screen.

Not ranked but worth a line: **seed-ensemble distillation** (measurement 5 decides; distillation
infrastructure exists from M2b); **prioritised replay** (generic alternative to lever 4; TD-error
priority will over-sample eps deaths and the Huber tail, so stratification is the better first step);
**distributional / CVaR heads** (fixes item 1 differently and offers a risk-sensitive argmax that is
still a bare network; costlier than lever 1 and the lit track's objection, that it costs mass, is a
feature for the guard and a cost for c1); **mid-resolution plane / dilated local stack** (only if
measurement 3 fails; the cheaper version is topology scalars, which is the mechanics panelist's
territory); **snapshot restarts from pre-trap states** (heavier sibling of lever 4; supplies the
counterfactual contrast; needs GridBatchSim and Vector61Runtime state restore).

## 4. Statistics and methodology (question 3d)

**Sample size.** Per-world survival is Bernoulli with p ~ 0.55, variance ~0.25. Pairing by *world*
buys almost nothing (seed-pair r = 0.035); pairing by *seed* (fine-tunes from the same B8 finals, as
M6 did) removes the large seed-quality component (B8 seeds span 0.38–0.66) and is the right design.
Then SE(diff) = sqrt(2 x 0.25 / n) with n worlds per arm: n = 160 → 0.056 (M6's observed 0.055–0.060);
n = 480 → 0.032; n = 960 → 0.023. MDE at 80% power, one-sided alpha 0.05 (z = 2.49): 0.14, 0.080,
0.057. To detect +0.10 at 90% power needs ~430 worlds per arm, i.e. 5 seeds x 96 (the session's rule)
is adequate for +0.10 and not for +0.05 (~1,700). Use deaths per 1,000 decisions at L >= 800 as the
secondary endpoint: it is a count over ~280k decisions rather than 160 coin flips and is 1.3–1.5x more
efficient for the same hazard change. Snapshot-start probes (begin at a census L ~ 800 state, run 2,100
frames) multiply events per compute by ~2.4x and are legitimate for screens (never for the gate).

**Was M6 able to see a real effect?** Only a large one, with worse-than-even odds: 63% power for the
full +0.11 the guard needs, 23% for +0.05. The interim 4-seed "TIGHT clearly harmful" and the final
"null" are both over-readings of SE-0.06 contrasts; DEATHW's 90% CI [−0.07, +0.13] contains the needed
effect. The L60 and D20 single-seed screens (SE ~0.12) could not have detected anything short of a
transformation. Under item 1 none of these levers were expected to work, so the correct status is
"untested in a design that could have seen them, and mechanistically unlikely".

**Reading the PARTIAL.** B8 beat the served agent on the primary with LB +101 and 5/5 seeds, and was
non-inferior on survival against it on all three mixes with 160-vs-160 power (scripted −0.003, thr
−0.03). The one failing clause compares B8 seed 0 with champion_a5 + v8 on seed 0's 32-world bank as a
point clause (`frp_v4_like`: guard cell `seeds=[seeds[0]]`): SE ~0.125, so it fails ~34% of the time
when the two agents are equally safe and ~50% at a true −0.05 gap. The OC table shows the same clause
lets a true −0.10 guard harm through 39% of the time. It is weak in both directions. The census (160
vs 32, different worlds) nevertheless gives −0.157 +/− 0.091 against the champion and −0.125 +/− 0.065
against the incumbent (vs Phase R's −0.003 +/− 0.056), so the honest pooled picture is: B8 is probably
−0.03 to −0.07 below the served agent and ~−0.10 to −0.15 below champion_a5 on scripted worlds, with
wide uncertainty on both. Note the mechanics point that the champion barely enters the long-snake
regime (5.6% of decisions at L >= 800, 0 deaths where B8's hazard predicts ~1.2): survival over 5,000
frames conflates exposure with hazard, and the guard mostly measures "did you stay small".

**Is a prospective amendment legitimate?** It can be, under all of these: (1) the amendment is written,
reviewed and committed before any new gate data exists; (2) the new Phase R runs on a fresh bank
namespace, so B8's known numbers are not re-scored (re-scoring the existing look-2 data under a new rule
and calling it GO is p-hacking, full stop); (3) the comparator change (champion_a5 → the served agent)
is justified by the stated goal, which predates B8, and the margin (−0.05) and the one-sided
protective-margin machinery are left as they are, not tuned to B8's numbers; (4) the guard is estimated
the way c2 is (all five banks, candidate pooled vs comparator pooled) rather than on one bank; (5) a
catastrophic floor survives: e.g. pooled scripted survival vs incumbent >= −0.05 **and** no seed below
−0.15, which B8_s1-like seeds (0.38) would fail; (6) the B8 PARTIAL and the fact that it motivated the
amendment are disclosed in the amendment and in every later result; (7) this is the one survival-guard
change for the redesign line: a second one after a second failure is a pattern, not a protocol. What
would make it p-hacking: choosing the comparator or the margin because B8 passes it; iterating; or
promoting on the existing data. The owner's real decision is whether "never ship a snake that dies
more against bots than the June champion did" is still a product value; if it is, keep the clause but
fix its power (five banks), and accept that B8 as trained probably fails it.

## 5. The guard clause (question 4)

Methodologically the clause is defective independent of B8: one seed, one bank, 32 vs 32, a point
threshold at a margin smaller than half its SE, and a comparator that lives in a different length
regime. Its false-fail rate under equality (~34%) is larger than the protection it offers against a
real −0.10 harm (61%). The sequential design's §9 OC already shows the guard costing GO power at
+65 with no harm present. I would (a) keep a scripted-survival protection in the rule, because the
census shows a real gap exists; (b) re-specify it prospectively as a pooled five-bank clause against
the served agent with a per-seed catastrophic floor, for a fresh-bank Phase R of whatever candidate
comes out of the screens; (c) measure champion_a5 + v8 on all five banks now as a diagnostic, and
publish the sensitivity table (B8 under the old and the new clause) so nobody can later claim the
amendment was invisible; (d) not re-gate the existing B8 finals under the new clause and call it GO.
If the owner's goal is literally "beat frp3-s12 + v8 without a veto", the clause against champion_a5
is the only thing standing between B8 and promotion, and it was never the goal's comparator; the
legitimate path to say so is the amendment above plus a fresh confirmatory gate, not a re-score.

## 6. The planner question (question 3e)

Training-time labels only: the served artifact is a bare network with no search at inference, so
"no veto at serving time" is preserved literally and in the same sense the project already accepts
(the student was distilled from a teacher *with* v8). What must be disclosed is that the policy was
trained with privileged simulator lookahead. Ways it fails to transfer, in order of likelihood:

1. **Representation.** If the student cannot separate the states where the planner's action differs
   from its own (measurement 3 says whether it can), the labels are unlearnable noise. M3-D's
   signature ("fits TRAIN, fails fresh-world transfer; more labels barely help") is exactly this and is
   the main reason to run the probe first.
2. **Privileged futures.** Scripted opponents are deterministic given the state and the transition
   index, so a rollout "knows" what greedy_food and random_safe will do. Labels that are right only
   because of that will be brittle against the frozen vector61 pool and against humans in Play mode.
   Mitigate by labelling with the expectation over K randomised opponent continuations and keeping only
   decisive labels.
3. **Ceiling.** If B8 + the same lookahead as a shadow does not survive more, the labels cannot help.
   Run the shadow on 96 scripted worlds before building the labeler.
4. **Loss conflict.** The margin loss moves argmaxes; TD moves values. M3-D's S18 found the
   imitation-vs-critical-correction conflict; apply the margin only on decisive labels and keep its
   weight below the TD's, or distil into the value instead (regress Q toward the rollout's realised
   return, which lever 1 makes meaningful).
5. **Coverage.** 75 deaths x 60 decisions is ~4,500 labels. Expert iteration needs the labeler running
   continuously on sampled L >= 600 states (and on snapshot restarts), not on the census alone.
6. **Compounding on B8's own weaknesses.** Rollouts that continue with B8 will report "all actions
   die" wherever B8 would die from any continuation; iterate (expert iteration proper) rather than
   label once.

## 7. Disagreements with the current plan

1. The replay fix is framed as starvation and sized as "a much larger replay". The aggregate share is
   not starved; the defects are cyclic non-stationarity, rare-event supply and 5-step reach. Measure
   the phase swing before buying RAM; stratified reservoirs and desynchronised segments are the cheaper
   and more targeted fix.
2. "Tried and null" for D20 / DEATHW / TIGHT / L60 should read "not detectable at 5 x 32 (power 63% for
   the needed effect) and mechanistically not expected to work under a Huber-beta-1 loss at Q scale
   100". The loss regime is the untested lever with the largest predicted effect and zero cost.
3. The anchor is treated as settled infrastructure. With on-policy v8 labels at label_frac 1.0 it is a
   self-imitation trust region on exactly the states that need to change; the M6 fine-tunes may have
   run it at lambda 1.0 throughout. It needs an ablation before any more fine-tune screens.
4. The literature track's "network cannot see far enough (±15 cells)" is wrong as a description of the
   architecture (the FC layer is global over the 31x31 crop; 23 ahead / 7 behind / ±15 lateral) and
   the M7 AUC study says hand features add little. Perception is last, after the linear probe, as the
   report already has it, but for the right reason.
5. `death_len_mean` should not be used as the starvation check: it is per-interval and confounded by
   segment phase. Use the full-cycle death histogram by length x phase x eps.
6. The guard target "0.64" is a dev-bank number with SE 0.08, and the gate's guard is 32 vs 32. Any
   claim that a lever "closes the gap" needs the comparator measured on all banks and a 5 x 96 screen.
7. The growth governor should not be a serving-time mechanism (that is a veto in spirit); its
   legitimate form is objective-level (gamma 0.999, terminal-hero training, survival weight), tested
   as a lever with the mass cost reported.
8. The planner track should start with the ceiling test (B8 + lookahead as a shadow) and the linear
   probe, not with the labeler.

## 8. Context still needed before committing

| # | Item | Where | Priority |
|---|---|---|---|
| 1 | train_m3c.py (branch m6): TD loss and beta; n-step; replay capacity and sampling; whether all worlds reset per segment and whether actors stagger; LabelBuffer sampling; anchor schedule defaults as applied in M6 (lambda over the 8M); margin value | mac artifacts (repo branch m6) | must |
| 2 | Per-length hazard: deaths per 1,000 decisions in L bins (0–400, 400–800, 800–1000, 1000–1200, 1200+) and time-at-L for B8 seeds, incumbent, champion on the census worlds | mac artifacts: b8-death-census-20261014 captures / results.json | must |
| 3 | Training death records (length, segment frame, mix, eps-in-last-5) over >= 1 full cycle of an 8M fine-tune | mac artifacts telemetry.jsonl if logged; else new measurement | must |
| 4 | log.jsonl of CTRL_s10 (qmax, td, deaths_per_1k vs transitions) to confirm the 1.28M period and the checkpoint phases | mac artifacts redesign-m6-mac-20261014 | should |
| 5 | Phase R look-2.json per-cell numbers: B8_s0 and champion scripted survival on seed 0's bank; incumbent per bank | mac artifacts redesign-m4-phase-r-nv-20261013 | must |
| 6 | Champion_a5 + v8 scripted survival on all five Phase R banks | new measurement (~1 h) | should |
| 7 | Linear probe on B8 fuse features using the M7 capture rows and labels | new measurement (~1 h) | must |
| 8 | Checkpoint-phase probe (8 checkpoints over one 1.28M cycle, 96 scripted worlds each) | new measurement (~1.5 h) | must |
| 9 | Counterfactual rollouts on the 75 deaths: last-avoidable decision and Q-gap traces; also verifies GridBatchSim deepcopy cost | new measurement (~1.5 h) | should |
| 10 | Seed-ensemble (mean-Q of 5 B8 finals) on 96 scripted worlds | new measurement (~40 min) | should |
| 11 | M5 L60 intermediate probes (every 5M) vs checkpoint phase | mac artifacts redesign-m5-probes-20261013 | nice |
| 12 | Mac RAM and --obs-pack bytes per transition (feasible replay capacity) | mac transcript digests / artifacts | nice |
| 13 | Eval profile `hero_terminal` and training `allow_respawn` as used by train_m3c.py (objective mismatch) | repo here (eval_engine, runtime_contract) + mac trainer | should |
| 14 | Whether the vectorised n-step builder and Replay in train_m3c.py parameterise n and capacity (cost of levers 2 and 4) | mac artifacts (branch m6) | should |
