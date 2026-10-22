# Prospective capacity amendment 02 — before production collection

METRICS.md and AMENDMENT_01 remain immutable. The excluded eight-decision Mac engineering
preflight exercised one L800 scripted H300 branch in 2.184720 CPU-seconds (300 transitions).
That cost makes the original all-pass 2,412,000-decision workload too large for a <=90-minute
L4 session under the planned host slowdown sensitivity. No production world has been run.
This amendment is based on engineering cost, not a selected efficacy or realism outcome.

For all production arms and mixes, sample t=0,200,400,...,4600 (24 possible roots per episode),
replacing the original 50-decision grid. Still require alive hero, L>=800, the same cheap
prefilter, and the same full H300 bare death check. The original H5000/t<=4700 observability
rule is retained; 4600 is the final multiple of 200 satisfying it. Exact arm-A recall now
refers to THIS sampled grid. All other metric definitions, feature order, classifier split,
72-episode allocation, six workers, and all-live-decision length/efficiency denominators stay.
A rare short-lived doom interval may be missed; report the stride and do not extrapolate
moment rates to every-decision event counts.

Worst-case simulator-decision ceiling is 72*(5000+24*300)=878400, before setup/analysis.
Real cost may exceed a tiny-fixture forecast: first supervisor cap wins, and any truncation
is INCOMPLETE. Bind this amendment's SHA-256 in the manifest and every exported run record.
