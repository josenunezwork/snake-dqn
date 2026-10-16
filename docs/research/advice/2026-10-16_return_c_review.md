# RETURN_C pre-launch review (advice, 2026-10-16)

Advisory note from the cloud session, eleventh request. Not gate evidence. The bug check
(no bug; decisive boost targets -4.7 versus non-boost -35.8; outside boost-minus-normal gap
-0.16 Q; same-state CTRL Q near 9 with large negative tails) confirms H1 and H2.

## 1. One correction before launch: do not anchor on a_B8

At a decisive state a_B8 is the fatal action. With stopgrad on Q(s, a_B8), the contrast sets
every survivor to Q(fatal) plus about 70 Q. TD holds Q(fatal) near its look-alike value, not at
-76, so survivors are pushed to about +70 at bank states, an absurd value in the shaped
convention where steady survival is near 0, and boost branches (87% survive) get the biggest
lift. That is the boost drift again, now with magnitude.

Anchor instead on the **recorded action with the highest target**, a_ref = argmax_a y(a), with
stopgrad on its Q. Its Q is a normal-valued action that TD estimates well, the fatal action is
pushed down to Q(a_ref) minus 70, and nothing is inflated. At outside states a_ref is usually
a_B8 anyway. Every legal action was rolled out, so a_ref is always a recorded branch. Do not use
the student's current greedy action: it makes the target non-stationary and buys nothing, since
all pairwise differences are consistent whichever recorded action anchors them.

## 2. Make outside rows resist, not flatten

With weight min(1, |dy|/10), an outside row at dy = -0.16 carries weight 0.016: the loss is
one-sided in practice, exactly the T3 shape. Replace the weighting with an epsilon-insensitive
Huber: zero loss while |contrast error| <= 2 Q, Huber-100 beyond, uniform weight 1.0 on every
row. Near-ties then carry no gradient (no flattening), but a 10-unit spillover onto an ordinary
action is resisted at full weight. Recalibrate W to the gradient ratio after this change, since
far fewer rows are active.

## 3. Boost pressure and the tripwire

The decisive boost advantage is real and, with a_ref as anchor, boost is the reference rather
than the thing lifted, which removes most of the broad push. Spillover to look-alikes remains
possible and the epsilon rows are the guard. 5% is too loose: CTRL and B8 sit at 0.1-0.2%, and
the dense ceiling changed 75 of 300k decisions. Pre-register two: greedy boost rate at L < 700
above 1% halts (escapes do not live there, so that is pure spillover); overall rate above 2%
halts. Report the rate split by length bucket per 100k.

## 4. Early warning at 0.5M-1M

**Held-out decisive Q-gap**, Q(fatal) minus Q(best surviving) on decisive states held out by
world, at 0.5M and 1M, beside two guards: the held-out outside-state gap (unchanged within
0.5 Q) and the L < 700 boost rate (under 1%). The gap should be moving from +0.45 toward -3 by
1M. If training-row contrast error falls while the held-out gap does not move, the magnitudes
are being memorised, not learned, and the run should stop there rather than at 8M.
