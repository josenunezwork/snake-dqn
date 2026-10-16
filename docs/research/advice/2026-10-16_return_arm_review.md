# RETURN arm design review (advice, 2026-10-16)

Advisory note from the cloud session, ninth request, on `specs/t4e_return_arm_spec_2026-10-16.md`
and the return-regression tooling doc. Not gate evidence. T4c = KILL with consequential change
3.8% against a 0.99% drift baseline, so the fork to return regression stands.

## 1. What is risky, in order

**The loss weight, not the delta, is the hazard.** The planner's addendum already separates
`--return-huber-delta 100` from the TD delta of 1. That is right: with delta 1 the loss is
linear beyond one unit and fits a median, which erases the magnitude the arm exists to supply.
But at delta 100 the per-row gradient equals the error, and return errors start around 30-80 Q
while TD rows are capped at 1. At `--return-weight 1.0` with 128 return rows against 384 TD
rows, the return term dominates the update by roughly 20-50x per row. T2's delta-100 arm hit
the action-collapse tripwire on one seed with far less pressure. Calibrate the weight, do not
guess it: in the smoke, log the gradient-norm ratio of the return term to the TD term at
initialisation and set W so the ratio is about 0.5. My expectation is W in 0.02-0.05.
Pre-register W from that measurement and report max |Q| per 100k as the drift readout.

**Target staleness is acceptable and should be disclosed, not fixed.** Branches run "a then B8
for 300". For a decisive escape the student may later continue better than B8 did, so the
realised part under-values escapes slightly; it never over-values the fatal action, which is the
direction that matters. The bootstrap `gamma^T V_target(s_T)` uses the student's own target net
and moves with training; that is standard and carries 0.55 of a survivor's target, so the
terminal observations must be in the bank, which the tooling confirms.

**Huber 100 versus MSE.** Keep Huber at 100: it is quadratic across the whole range the targets
occupy, so it behaves as MSE there, and it caps the few errors beyond 100 that an early,
mis-scaled Q head produces. MSE adds nothing and removes that cap.

**Two smaller checks.** First, the on-policy anchor stays at `--label-frac 1`, so the return rows
are extra, not a replacement; confirm the batch composition log shows 128 anchor plus 128 return
rows, not 128 split. Second, bank states must not appear in the held-out set by world, as the
spec says; also exclude their worlds from round 68's seed audit, since the collection namespace
is fresh but the probe bank must not overlap it.

## 2. The single most informative readout

**Held-out decisive Q-gap at 8M**: median of Q(fatal) minus Q(best surviving) over decisive
states held out by world, with the training-set gap and the held-out outside-state gap
(Q(B8 action) minus max of the others) beside it. Pre-register:

- held-out decisive gap <= -3 Q (B8 starts at +0.45 median; T1 p90 is +2.55);
- training-set gap of similar size, so the two agree within a factor of two, which rules out
  memorisation;
- held-out outside-state gap unchanged within +/- 0.5, which rules out spillover in Q space;
- consequential change (c) within 2 points of the 0.99% drift baseline.

If the first holds and (c) holds but survival does not move, the labels are right and the
remaining problem is which states the student reaches, which is the whack-a-mole reading and
the expert-iteration arm from the tighten addendum. If the first fails while the training-set
gap succeeds, the magnitudes do not generalise from bv8, and the campaign moves to inputs as the
spec's fork already says.
