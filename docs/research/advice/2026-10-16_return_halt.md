# RETURN arm halt: why it boosts, what to check, what to change (advice, 2026-10-16)

Advisory note from the cloud session, tenth request, on the RETURN arm halted at 0.8M with the
greedy boost rate at 34% and the return loss still at 200. Not gate evidence. The halt was right.

## 1. Hypotheses, ranked

**H1. One-step-deviation targets flatten the action landscape, and boost rides the flattening.**
A branch is "a, then B8 for 300". In the shaped convention the trainer uses, the PBRS part
telescopes, so a steady survivor's realised return is about 0 whatever safe first action it took,
and the bootstrap adds the same V(s_T) to all of them. So at every state, every safe action gets
nearly the same target: the regression says Q(s, a) is about V(s) for all safe a. That erases the
small advantages the on-policy Q encoded (which direction reaches food sooner) and it is
universal because outside states are half the rows and five of their six branches are safe
deviations. One boost frame costs no segment (the cost accrues per k frames of sustained boost),
so y(boost a) equals y(normal a) and the regression pulls Q(boost) up to Q(normal). On a flat
landscape that flips the argmax to boost often; the student then boosts for runs, pays the
sustained cost the branch never priced, and only on-policy TD can push back (TD loss rising
0.01 to 0.2 is that push starting).

**H2. The boost-escape signal is real at decisive states and generalises.** 17 of 46 T1 escapes
were boost-only, so y(boost) exceeds y(normal) by tens of Q at some decisive rows. Combined with
H1 there is nothing in the data saying boost is ever worse. Same mechanism, extra push.

**H3. The fatal rows are the unfit part.** A Huber-100 mean of 200-320 is a fatal fraction of
15-20% carrying errors of 55-75 Q with everything else fit. With W = 0.0032 a fatal row pushes
about 0.2 per row against hundreds of TD and anchor rows that hold its look-alikes at 0. The
magnitude never lands, which is the discriminability limit again, and it will not land at any W
that does not also distort the look-alikes.

**H4. Small max |Q| is probably timing, not a defect.** CTRL reaches 110-140 by 1-2M; this run
stopped at 0.8M. Check before treating it as a symptom.

**H5. Index or convention mismatch: low prior.** Twelve sim tests and the B8 control branches
cover the formula and B8's own index, but B8 boosts 0.1% of the time, so a permutation among
boost indices would pass those checks.

**H6. Early bootstrap near zero: not a problem.** V near 0 is the correct shaped value of steady
survival; survivor targets near 0 are right, not collapsed.

## 2. Cheapest decisive checks

- H1: on the bank, the distribution of y(a) minus y(a_B8) over safe branches. If over 90% lie
  within 1 Q, confirmed. Also y(boost a) minus y(normal a) for the same direction: expect 0 on
  outside states. Minutes, no training.
- H2: fraction of decisive states whose best branch is a boost, and its margin over the best
  normal branch. Minutes.
- H3: per-family return loss (fatal versus survivor rows) from the halted run's logs, and
  Q(fatal action) on 200 bank states at 0.8M versus at init. Minutes.
- H4: CTRL max |Q| at 0.8M from its logs. Seconds.
- H5: for 20 states, re-roll each boost branch one frame and compare head displacement with the
  boost speed; recompute 20 targets by hand from the reward sequence. Under an hour.

## 3. What to change: regress contrasts, not levels, and only where contrast exists

Lowering W does not fix H1, it slows it. Decisive-only rows remove half the flattening but keep
the other half. The change that removes the mechanism:

- **Contrast target.** For each branch, regress Q(s, a) minus Q(s, a_B8) toward y(a) minus
  y(a_B8), with Q(s, a_B8) detached so TD keeps the absolute scale and the bank moves only the
  deviating action relative to B8's.
- **Contrast weight.** w_a = min(1, |y(a) minus y(a_B8)| / 10). Near-tie branches get no weight,
  so the flattening cannot happen; fatal branches (minus 55 to 99) and boost-only escapes keep
  full weight and full magnitude. This drops about five of six rows; re-measure the gradient
  ratio and reset W, which will rise.
- **Boost macro branches.** Roll out boost actions as k frames of boost (k = the cost period)
  then B8, so the sustained cost is priced where boost-only escapes are labelled.
- **Tripwire.** Greedy boost rate above 5% at any 100k halts the run, pre-registered.

This is a new pre-registered arm, not a resumption of the halted one.
