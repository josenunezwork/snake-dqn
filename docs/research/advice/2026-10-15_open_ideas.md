# Open ideas for the B8 late-death problem (advice, 2026-10-15)

Advisory note from the cloud session, second request from the local session. Not gate evidence.
Read at origin/main `aee9a64` with the T3 screen (LABELS, LABELS_KEEPV8, LABELS_KEEPV8_LO) running.
Ranked by expected value per Mac-hour. Numbers are the repo's own measurements.

## 1. Speed-ups

**1a. Batched hero forward for screen probes (likely 5-10x).** Probes run a rowwise ego2s
forward per world at H5000 over 160 worlds per checkpoint. Rowwise CPU acting is about 1.7k
steps/s because the forward dominates, while the sim alone does 15-17k hero steps/s and MPS does
127k forwards/s. Run each mix as one GridBatchSim with E = 96 and one MPS forward per step for
screens; keep rowwise for strict gates and serving qualification. Pairing stays exact because
every arm and CTRL use the same path. Declare it once per screen, never mixed within one.

**1b. Probe the primary first, stop on KILL.** Probe the scripted mix for all five seeds of an
arm first. If the paired primary is <= 0 the arm is KILL and its frozen and mixed probes never
run. Pre-register as a futility stop only; GO still needs every mix.

**1c. Batch mixed and frozen rollouts in the labeller.** The bank doc records them at E = 1 per
action, 3.9-4.7 s per batch, while scripted batches run all alternatives as one sim. The merged
vector61 opponent forward already stacks checkpoints, so K actions can be K environments of one
sim as in Part A. Those worlds are 32.5% of the roster.

**1d. Dedupe adjacent decisive snapshots.** Neighbouring snapshots in a death window share almost
the same surviving set. Keep every fifth decision in the last 100 before death (T1's median LAD
horizon) and every tenth before that: half the rollouts for nearly the same label diversity.

**1e. Intermediate futility checkpoints.** Probe the 4M checkpoint on 32 scripted worlds. A paired
scripted episode-survival delta below -0.05 at 4M never recovered in M6, T2 or T2b. Pre-register
as a non-binding lane kill; it saves half a lane per losing arm.

**1f. Learner headroom.** A lane caps near 3.3k transitions/s on the 39 ms MPS update at replay
ratio 4; batch-scale 2 sits near 5k. `--learner-precision bf16` and `--compile-learner` exist and
need a parity measurement before use. A third lane pays only if MPS sits under about 60% busy.

## 2. Methods for a rare, delayed, self-inflicted catastrophe

**2a. Expert iteration: relabel on the student's own trajectories.** The bank came from B8's
death windows under B8's continuation. After fine-tuning, the student dies elsewhere and the
surviving sets are stale (launch review, risk 3). Run the dense shadow policy on each student's
final, collect a fresh death-window bank, fine-tune again. One cycle is about 2 Mac-hours at the
measured 5.9k decisive rows/hour and the trainer flags exist. This is the most likely route from
"labels help a little" to "labels close the gap".

**2b. Return regression on rolled-out branches (the reserved LABELS_RETURN).** Each decisive
state already has six 300-frame continuations with known outcomes. Regress Q(s, a) toward each
branch's realised discounted return instead of only a margin, so the fatal action gets its true
-Phi(L) target 100-300 decisions early instead of waiting for TD to carry it through the replay
cycle. The collector only needs per-action death frame and final length.

**2c. Boost handling, cheapest first.** Label the smallest sufficient set: surviving non-boost
actions when any exist (29 of 46 LADs), the full surviving set otherwise. That removes most of the
12-17% boost rate the smoke showed, since 56% of surviving labels are boosts that were usually
unnecessary. Next, select rollout labels by survival then final length so a surviving boost loses
the tie to a surviving normal move. For the 17 boost-only escapes, label a short boost macro
(boost k frames, then normal): those escapes are bursts, not sustained boost.

**2d. Snapshot-restart curriculum.** The 10.9k decisive states are prepared world clones. Start
10-20% of actor episodes from them so the student meets the seal region hundreds of times more
often than natural play provides, with ordinary TD and no labels. This is the standard fix for
rare delayed catastrophes in long-horizon games. Needs an actor flag and a rule for counting
those transitions.

**2e. Counterfactual transitions into replay.** The six branches per decisive state are real
transitions with real rewards and terminals. Store the fatal branch's last 30 frames and the best
survivor's first 30 in packed form (about 2 GB for the bank) and add them to replay at modest
weight, so TD sees the death and its avoidance side by side.

**2f. Break the segment clock.** Replay content cycles with the 5000-frame segment clock (1.3M
period) and confounded death-length telemetry. Stagger segment starts per world by a random
offset. One line, must be re-controlled, removes a known artefact.

**2g. Auxiliary death-probability head.** Train P(death within 100 decisions | s, a) from rollout
outcomes as an auxiliary loss on the shared trunk, for representation first. Re-ranking actions
with it at serving is a judgement call under the no-veto ruling; a distributional or CVaR head
inside the network is the cleaner form. Lower priority than 2a-2d.

## 3. Game knowledge that transfers

Strong slither-style bots keep a long body safe with four habits: never enter a region whose free
area is below body length (bv8 already carries area and landing inputs); when enclosed, coil by
following the own tail at a fixed gap, because tail cells free as the body moves, so a pocket of
area about L is survivable if entered as a spiral rather than a line; hug walls at large length so
one side is permanently safe; boost in short bursts to cross a closing gap or break an
encirclement, never into unseen space.

Two cheap featurizer signals follow if an input experiment is ever reopened: the narrowest gap
between the current region and the outside, and whether a k-frame boost reaches outside the
component before the nearest sealer does. T1's R3 NO_GO on tail topology is consistent with
this: escapes are a ranking problem, but survival once enclosed is a coiling problem the inputs do
not describe, and half the late deaths are self-sealed (13 hero, 13 opponent, 16 none).

## 4. Evaluation and gating waste

- The mass clause (>= -30 on every mix) is judged on 32-world frozen and mixed banks with a world
  SE of 15-20 mass units, so a true -10 can KILL a good arm by noise. Use 64 worlds for those
  mixes or pool mass across mixes with a per-mix floor of -60.
- EXTEND costs as much as the initial probe. Pre-register an early GO at look 1 for large effects
  and keep EXTEND for the band between KILL and GO.
- The guard clause decision (option B) is still open; Phase R cannot be pre-registered without it.
- Holdout agreement is not logged by the multi-actor trainer yet appears in rules: log it or drop it.
- Hash-pinned plan tests fail off the Mac on 1e-17 float noise, so the ubuntu CI runner can never
  pass them. Mark them Mac-only so CI can be made meaningful.
