# T3 interim: why the labels cost mass, and the next lever (advice, 2026-10-16)

Advisory note from the cloud session, fifth request, on the four-seed T3 interim. Not gate
evidence. Numbers are the local session's interim table and the T1/T3 docs.

## 1. Why ~11k margin labels cost ~90 mass and bought little survival

The data point that settles it is LABELS_KEEPV8_LO: CTRL's own anchor weight, half the anchor
rows still on-policy v8, boost rate unchanged, and still -88 scripted mass with survival -0.005.
So the loss is not boosting, not anchor strength, and not the missing v8 anchor. It is what the
bank rows say and where they say it.

**The surviving-set margin is asymmetric and non-specific.** On a decisive row it has one
active term: push every non-surviving action, which always includes B8's greedy move, at least
one margin below the best survivor. On a counter row B8's action is already in the set, so the
hinge is satisfied and contributes no gradient. Nothing in the loss ever says "the greedy move
is right here". Over 11k crowded late-game states (L >= 700, frame >= 2000) seen hundreds of
times, the network learns the only rule its inputs let it express: de-prefer the greedy move in
crowded late states. The specific escape needs a discrimination the bv8 inputs do not carry at
a 300-frame horizon (the fuse probe is AUC 0.85, not 0.99), so the rule generalises to
look-alike safe states. That is general caution: slower growth, roughly the same deaths.

**The margin is probably far too large for the job.** T1 found the fatal action ahead by a
median 0.45 Q units (p90 2.55). The run's margin constant comes from `margin.json` and was set
for distillation from teacher Q values, not for correcting half-unit ranking errors. If it is
several Q units, the hinge forces Q(greedy) several units below the survivors on every bank
state and bank_agree of 0.95-0.995 says the student complied. Overshoot of that size has to
spill into neighbouring states. Report the constant; if it is above about 1.5, this is the
first thing to fix.

Secondary contributors, in the order I would weight them: label density 61% (the set is wide,
so the margin mostly says "not B8's action"); counter rows too few and inert (above); bank
built under B8's continuation. Staleness is not the main story while LO barely moves survival.

## 2. Ranked next levers

1. **Recalibrate the margin and use bank v2 (prefer-nonboost).** `--label-bank-margin-scale`
   exists. Set it so the effective margin is about 1.0 Q unit (between the T1 median and p75
   gap), keep the LABELS schedule, bank v2. One arm, five seeds, 8M (the checkpoint-curve doc shows 4M does not retain the
   8M effect: CTRL4M is -0.14 episode survival below CTRL8M), with `--update-every 2` for
   wall time if its A/B passed: about 3-4 Mac-hours. This is a direct test of the overshoot reading and
   the cheapest thing that could turn -72 mass into something near -30.
2. **Return regression on recorded branch outcomes.** Symmetric by construction: every rolled
   action gets a magnitude-calibrated target, the fatal one -Phi(L)-3, the good ones their
   realised return, and counter rows carry real information. The T4 tooling note is right that
   the discounted mass sum must be recorded during rollout; the v1 bank cannot supply it. Cost:
   one collection with outcome recording (about 2 Mac-hours per 10k rows), the loss term, five
   8M runs. About one day. Highest expected value if lever 1 only halves the mass loss.
3. **Snapshot-restart curriculum.** Start 10-20% of actor episodes from decisive-state world
   clones so TD itself sees the death and the escape in the same region with real outcomes.
   No labels, no margin, no asymmetry. Needs an actor reset-from-snapshot path and world
   clones (reconstruct by deterministic replay of the 1,400 bank worlds, roughly 40 CPU-minutes
   on 12 workers, or record clones in the next collection). About one day.
4. **Expert iteration from the student.** Fixes staleness only. Worth it after lever 1 or 2
   shows a positive survival delta without a mass fail, not before.
5. **Behaviour-clone the shadow on its own trajectories.** Not as framed: the dense shadow
   changed 75 of 300k decisions, so a BC dataset is 99.97% "do what B8 does" and relearns B8.
   Its value is as the on-policy data source for lever 2, with the changed decisions and their
   windows up-weighted.
6. **Narrower single-action labels.** Still asymmetric, and the best survivor under B8's
   continuation is not necessarily best. Skip; bank v2 already narrows the set where it matters.

On the registered rule: LABELS (+0.042) and KEEPV8 (+0.021) are EXTEND, LO is KILL. The
extension cannot move a -72 mass clause with a world SE near 15, so run it only to close the
record; the decision is already made by mass.

## 3. Cheap diagnostics while the last seed finishes

- **Margin overshoot (minutes).** Print the margin constant. On 2,000 bank rows, compute
  max Q(surviving) minus Q(B8's action) under CTRL and under each LABELS final. If the student's
  separation is several units where 0.5 would do, overshoot is confirmed.
- **Mass decomposition (minutes, from probe rows).** Split the -72 into earlier deaths versus
  smaller size while alive: compare death frames and death lengths LABELS versus CTRL, and mean
  alive length at frames 2000, 3000, 4000 on worlds where both survive. Caution shows as lower
  alive length with similar death frames.
- **Where the arms disagree (under an hour, CPU).** Replay CTRL's recorded round-62 scripted
  trajectories through the LABELS final and bucket disagreements by length and by -max Q. If
  most disagreements sit in ordinary states (L < 700, low -max Q), the rule generalised far
  beyond the bank.
