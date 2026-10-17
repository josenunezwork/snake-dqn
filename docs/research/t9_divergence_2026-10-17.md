# T9: fatal versus surviving divergence after the LAD (2026-10-17)

development diagnostic, not gate evidence

## Finding

There is no clean, consistent early separator among the requested quantities.
The strongest later differences are **the head's connected free-region size and
hero-first arrival-race areas at horizons 128/256**. Local body counts, tail
distance, boundary ownership and food-following do not consistently separate the
branches. Both branches commonly occupy pockets bounded mostly by the hero's own
body; the surviving branch usually does **not** remain in the largest component.
These findings qualify the advisor's hypothesis rather than establish missing
information in the observation. No training, observation/reward changes,
promotion or serving changes were made.

All 46 late T1 deaths with a LAD were reconstructed. Each pair starts at the same
prepared state: the fatal branch follows recorded actions; the survivor takes
T1's highest-Q surviving action once, then the original B8 checkpoint's greedy
policy. Every original fatal death and surviving endpoint was reproduced.

## Which quantities separate, and when

Differences below are **fatal minus surviving**, in cells unless noted. IQR is
reported as [Q1, Q3], not the width Q3−Q1. The descriptive consistency criterion
is at least 75% of available pairs sharing a strict sign, including ties in the
denominator, and at least 20 valid pairs. This is an exploratory scan over the
101 offsets, not a significance test, action selector or predictive validation.

- At offset 1, race-16 already has median −2 [−31.25, 1.75], but only 25/46
  (54.3%) differences are negative. Own-cell count within radius 15 has median
  −1.5 [−7, 5.5], negative in 26/46 (56.5%). Race-32 is −0.5 [−29.25, 2]; race-256
  is +0.5 [−2, 1.75]. These early signs are small and inconsistent.
- The first 75% directional separation is **offset 49**: race-128 and race-256
  are both smaller in **20/26 pairs (76.9%)**, with median **−48**. Their IQRs
  are **[−476.5, −1]** and **[−524.5, −1]**, respectively. That first crossing
  lasts only one offset; both satisfy the same negative-direction criterion
  throughout **offsets 65–83**, 19 consecutive offsets. Each qualifies at 20 of
  the 100 positive offsets, the most among these measurements.
- Head-region size has its first nonzero paired median at offset 6. Its first
  75% separation is **offset 65**, smaller in **20/26 pairs (76.9%)**, median
  **−29.5 [−1002, −1]**. It meets the criterion at offsets 65–73 and 75–83
  (18 offsets). Race-64 qualifies at 13 offsets; race-16 at three and race-32 at
  two. No other requested quantity reaches the criterion.
- At offset 50, head-region size is **−32 [−1003.5, 0.75]**, negative in 18/26;
  race-128 is **−48 [−508, −0.25]** and race-256 **−48 [−540.5, −0.25]**, both
  negative in 19/26. At offset 100, head-region size is **−25 [−157.75, −0.25]**,
  negative in 16/22; race-128/256 are **−18.5 [−160, 1.5]**, negative in 15/22.

The live paired denominators at offsets **0, 10, 25, 50, 100** are respectively
**46, 36, 31, 26, 22**. Later comparisons condition on the fatal branch still being
alive; they cannot stand in for all 46 original cases, or establish that a useful
signal was available at the intervention. The same worlds can appear under
multiple B8 arms, so the pairs are not 46 independent world draws.

## Testing the proposed pocket story

At the shared LAD, the median own-body boundary fraction is 0.8975; 43/46 regions
have a majority-own boundary. At offset 50 this is true in 25/26 fatal and 26/26
surviving states; at offset 100 it is 22/22 for both. The paired median difference
in own-body boundary fraction is zero at every requested offset. Being surrounded
by one's own body is common to both branches rather than a distinguishing sign.

The head is in the largest component in 14/46 LAD states. At offset 10 the counts
are 2/36 fatal versus 8/36 surviving; at offset 50, 0/26 versus 5/26; at offset 100,
0/22 versus 2/22. Survivors are generally in a different or larger pocket, not
necessarily the globally largest component.

The combined "moving toward nearest food, and that food is in a smaller component"
counts are 11/46 fatal versus 13/46 surviving at the LAD; 12/36 versus 18/36 at
D10; 10/31 versus 9/31 at D25; 14/26 versus 8/26 at D50; and 10/22 versus 8/22 at
D100. There is no consistently stronger fatal food-following pattern.

Static BFS reaches the own tail in only 9/46 shared LAD states. Both branches
have a finite tail distance in 2/36 pairs at D10 and zero pairs at D25/D50/D100.
Numeric BFS deltas therefore become unavailable; they are not zero. The separate
`tail_reachable` row retains the unreachable states. Manhattan tail distance and
radius-15/30 own/opponent density have mixed signs rather than a stable separator.

## Outside the 31×31 local window

The known intervention point is the LAD. Its shared head-region boundary extends
beyond 15 cells (Chebyshev distance) in **40/46 cases (87.0%)**.

For post-intervention divergence, the fixed analysis convention is the first
positive offset where the fatal head is outside the largest component and its
region is strictly smaller than the surviving head's region. It resolves
**42/46 cases** within D1–D100; four remain unresolved. At this first resolved
point, the fatal pocket boundary extends beyond 15 cells in **23/42 (54.8%)**;
19/42 boundaries fit within the window. Thus the observed outside share is
**23/46 (50.0%) of the full cohort**, with **4/46 unresolved**, not counted as
inside. Resolved offset quartiles are [2, 4, 18].

This is a **pocket-boundary extent statistic**, not localization of a causally
necessary separator cell. A pocket can have a local neck and a distant wall.
The 87.0% LAD figure and 54.8% post-divergence figure concern different states;
neither alone proves that the decisive information is outside the observation.

## Measurement and replay definitions

The complete executable contract is in
[`research/t9_divergence_20261017/README.md`](../../research/t9_divergence_20261017/README.md).
Offset 0 is the prepared LAD before either action; offset k is the prepared live
state after k transitions. Every integer offset through 100 or death is retained.
No live geometry is fabricated after death. Additional unmeasured transitions
validate the original fatal death and T1 survivor endpoint (115–300 transitions).

Static components use four-neighbor connectivity without a size cap: empty and
own-body TTL≤1 cells are free, opponents block, and only the hero head is opened
as a seed. Largest-component ties count as largest. BFS uses the same board and
returns null when the actual tail is blocked or disconnected. Radius counts use
Chebyshev distance and unique occupied cells, including heads. Boundary fractions
count exposed cardinal edges against own body, opponents or arena exterior.
Nearest food uses Manhattan distance with lexicographic (x,y) ties; moving toward
it means the chosen action's first next cell decreases that distance. A smaller
food component is smaller than the largest free component.

Arrival races are a separate optimistic diagnostic: the full board, hero and all
live opponent heads as unit-speed sources, and all owners' current segments
released at their current TTL. Arrival relaxes as `max(previous+1, destination_TTL)`;
source heads initialize at zero. Counts require hero arrival strictly earlier
than every opponent, within 16/32/64/128/256 steps; ties are excluded. Waiting is
permitted, but future growth, new trails, respawns and collision feasibility are
not modeled. These are access counts rather than guaranteed Snake paths.

## Complete requested paired summaries

All values below are paired fatal-minus-surviving differences. Boolean differences
are −1/0/+1, not percentage points; boundary fractions are proportions. The final
column is the number of negative / tied / positive pairs. NumPy linear quartiles
are used. Tail-BFS rows exclude unavailable distances pairwise.

| Quantity | Offset | Paired n | Fatal − surviving median [Q1, Q3] | − / 0 / + |
|---|---:|---:|---|---|
| head_region | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| head_region | 10 | 36 | -7 [-861.5, 1.25] | 21 / 3 / 12 |
| head_region | 25 | 31 | -2 [-214.5, 2] | 17 / 4 / 10 |
| head_region | 50 | 26 | -32 [-1004, 0.75] | 18 / 1 / 7 |
| head_region | 100 | 22 | -25 [-157.8, -0.25] | 16 / 1 / 5 |
| largest_region | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| largest_region | 10 | 36 | 0 [-2, 0.25] | 15 / 12 / 9 |
| largest_region | 25 | 31 | 0 [-1, 3] | 11 / 9 / 11 |
| largest_region | 50 | 26 | 0 [-2.75, 21.25] | 12 / 5 / 9 |
| largest_region | 100 | 22 | 0 [-1.75, 36] | 8 / 5 / 9 |
| head_in_largest | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| head_in_largest | 10 | 36 | 0 [0, 0] | 6 / 30 / 0 |
| head_in_largest | 25 | 31 | 0 [0, 0] | 4 / 27 / 0 |
| head_in_largest | 50 | 26 | 0 [0, 0] | 5 / 21 / 0 |
| head_in_largest | 100 | 22 | 0 [0, 0] | 2 / 20 / 0 |
| tail_bfs | 0 | 9 | 0 [0, 0] | 0 / 9 / 0 |
| tail_bfs | 10 | 2 | -15 [-18.5, -11.5] | 2 / 0 / 0 |
| tail_bfs | 25 | 0 | unavailable | 0 / 0 / 0 |
| tail_bfs | 50 | 0 | unavailable | 0 / 0 / 0 |
| tail_bfs | 100 | 0 | unavailable | 0 / 0 / 0 |
| tail_reachable | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| tail_reachable | 10 | 36 | 0 [0, 0] | 1 / 35 / 0 |
| tail_reachable | 25 | 31 | 0 [0, 0] | 2 / 29 / 0 |
| tail_reachable | 50 | 26 | 0 [0, 0] | 3 / 23 / 0 |
| tail_reachable | 100 | 22 | 0 [0, 0] | 1 / 21 / 0 |
| tail_manhattan | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| tail_manhattan | 10 | 36 | 0 [-4, 4.25] | 17 / 6 / 13 |
| tail_manhattan | 25 | 31 | 0 [-5, 11] | 14 / 2 / 15 |
| tail_manhattan | 50 | 26 | -1 [-5, 12.75] | 15 / 2 / 9 |
| tail_manhattan | 100 | 22 | -1 [-11.75, 2.75] | 13 / 2 / 7 |
| own_cells_r15 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| own_cells_r15 | 10 | 36 | 0 [-6.5, 12.75] | 17 / 2 / 17 |
| own_cells_r15 | 25 | 31 | 7 [-19.5, 25.5] | 11 / 0 / 20 |
| own_cells_r15 | 50 | 26 | -12 [-32, 25.5] | 14 / 0 / 12 |
| own_cells_r15 | 100 | 22 | 3.5 [-11.5, 28.5] | 10 / 1 / 11 |
| own_cells_r30 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| own_cells_r30 | 10 | 36 | 0 [-28, 27.25] | 17 / 3 / 16 |
| own_cells_r30 | 25 | 31 | 6 [-48, 20] | 13 / 1 / 17 |
| own_cells_r30 | 50 | 26 | 1.5 [-41.25, 58] | 12 / 1 / 13 |
| own_cells_r30 | 100 | 22 | 1 [-46.25, 31.5] | 9 / 1 / 12 |
| opponent_cells_r15 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| opponent_cells_r15 | 10 | 36 | 0 [-0.5, 2.5] | 9 / 16 / 11 |
| opponent_cells_r15 | 25 | 31 | 0 [0, 1] | 5 / 17 / 9 |
| opponent_cells_r15 | 50 | 26 | 0 [0, 0.75] | 3 / 16 / 7 |
| opponent_cells_r15 | 100 | 22 | 0 [-1, 0] | 7 / 11 / 4 |
| opponent_cells_r30 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| opponent_cells_r30 | 10 | 36 | 0 [0, 3.75] | 8 / 16 / 12 |
| opponent_cells_r30 | 25 | 31 | 0 [0, 3.5] | 7 / 14 / 10 |
| opponent_cells_r30 | 50 | 26 | 0.5 [0, 8.5] | 5 / 8 / 13 |
| opponent_cells_r30 | 100 | 22 | 0 [-10, 0] | 8 / 9 / 5 |
| boundary_own | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| boundary_own | 10 | 36 | 0 [-0.003134, 0.006757] | 11 / 16 / 9 |
| boundary_own | 25 | 31 | 0 [0, 0.03984] | 4 / 15 / 12 |
| boundary_own | 50 | 26 | 0 [0, 0] | 5 / 15 / 6 |
| boundary_own | 100 | 22 | 0 [0, 0] | 1 / 18 / 3 |
| boundary_opponent | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| boundary_opponent | 10 | 36 | 0 [0, 0.001518] | 8 / 18 / 10 |
| boundary_opponent | 25 | 31 | 0 [-0.02503, 0] | 11 / 17 / 3 |
| boundary_opponent | 50 | 26 | 0 [-0.004386, 0] | 7 / 16 / 3 |
| boundary_opponent | 100 | 22 | 0 [0, 0] | 2 / 20 / 0 |
| boundary_wall | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| boundary_wall | 10 | 36 | 0 [0, 0] | 6 / 26 / 4 |
| boundary_wall | 25 | 31 | 0 [0, 0] | 4 / 25 / 2 |
| boundary_wall | 50 | 26 | 0 [0, 0] | 5 / 19 / 2 |
| boundary_wall | 100 | 22 | 0 [0, 0] | 3 / 18 / 1 |
| toward_food | 0 | 46 | 0 [-0.75, 1] | 12 / 21 / 13 |
| toward_food | 10 | 36 | 0 [-1, 0] | 10 / 22 / 4 |
| toward_food | 25 | 31 | 0 [-1, 1] | 12 / 10 / 9 |
| toward_food | 50 | 26 | 0 [0, 0.75] | 5 / 14 / 7 |
| toward_food | 100 | 22 | 0 [-0.75, 1] | 6 / 9 / 7 |
| food_in_smaller_component | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| food_in_smaller_component | 10 | 36 | 0 [0, 0] | 5 / 25 / 6 |
| food_in_smaller_component | 25 | 31 | 0 [0, 0] | 1 / 24 / 6 |
| food_in_smaller_component | 50 | 26 | 0 [0, 0] | 0 / 22 / 4 |
| food_in_smaller_component | 100 | 22 | 0 [0, 0] | 2 / 16 / 4 |
| race_16 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| race_16 | 10 | 36 | -14.5 [-76.25, 13] | 21 / 1 / 14 |
| race_16 | 25 | 31 | 9 [-71, 16.5] | 12 / 1 / 18 |
| race_16 | 50 | 26 | -18.5 [-145.8, 17.5] | 18 / 0 / 8 |
| race_16 | 100 | 22 | -13 [-56.5, 30.5] | 12 / 2 / 8 |
| race_32 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| race_32 | 10 | 36 | -2 [-152, 16.5] | 19 / 1 / 16 |
| race_32 | 25 | 31 | 4 [-263, 46] | 13 / 0 / 18 |
| race_32 | 50 | 26 | -17.5 [-337, 12.5] | 17 / 0 / 9 |
| race_32 | 100 | 22 | 0 [-136.2, 18] | 10 / 2 / 10 |
| race_64 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| race_64 | 10 | 36 | -2.5 [-200.8, 4.75] | 19 / 3 / 14 |
| race_64 | 25 | 31 | 0 [-599.5, 5] | 15 / 2 / 14 |
| race_64 | 50 | 26 | -35 [-423.8, 0.75] | 18 / 1 / 7 |
| race_64 | 100 | 22 | -25 [-163.5, -0.5] | 16 / 1 / 5 |
| race_128 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| race_128 | 10 | 36 | -1 [-200.8, 2.25] | 19 / 3 / 14 |
| race_128 | 25 | 31 | 0 [-682, 3] | 15 / 3 / 13 |
| race_128 | 50 | 26 | -48 [-508, -0.25] | 19 / 1 / 6 |
| race_128 | 100 | 22 | -18.5 [-160, 1.5] | 15 / 1 / 6 |
| race_256 | 0 | 46 | 0 [0, 0] | 0 / 46 / 0 |
| race_256 | 10 | 36 | -0.5 [-200.8, 2.25] | 18 / 3 / 15 |
| race_256 | 25 | 31 | 0 [-682, 3] | 15 / 3 / 13 |
| race_256 | 50 | 26 | -48 [-540.5, -0.25] | 19 / 1 / 6 |
| race_256 | 100 | 22 | -18.5 [-160, 1.5] | 15 / 1 / 6 |

## Verification, cost and provenance

- Execution: one CPU process, one numerical thread, **117.34 s** for
  all 46 pairs; 7,528 live trace states. No session or existing process was managed.
- Exact checks: 46/46 LAD Q vectors and both chosen/survivor geometry dictionaries;
  46/46 fatal deaths; 46/46 original survivor endpoints; unchanged source clones;
  178,523 pre-action length checks. Input/source hashes unchanged during execution.
- Artifact audit: PASS, 46 unique source identities, complete trace offset ranges,
  same origin metrics, physical metric invariants, 511 input/source hashes checked.
- Focused validation: **43 tests passed**, covering T1 replay/clone machinery and
  new topology, race, trace timing and summaries. Flake8 and Black checks passed.
  The compiled flood exactly matches its retained heap reference on 20 randomized
  synthetic worlds, including additional horizon/other-owner release tests.
- A real paired pilot before/after acceleration matched every output field except
  timing, including prepared state hashes, actions and endpoints (18.52 → 3.11 s).
- Launch commit: `c173e63cfe732fafff536314bd538f10fa3a818b`. Full artifact root:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t9-divergence-20261017`.
  `manifest.json`, `rows.jsonl`, `completion.json`, `audit.json`, `summary.json`,
  `tables.md`, `report.log` retain the evidence. The full run log is the sibling
  `t9-divergence-20261017-run.log`. Raw artifacts remain local.
- Committed compact evidence:
  [`research/t9_divergence_20261017/results.json`](../../research/t9_divergence_20261017/results.json),
  including all requested offsets, key separation offsets, case-level spatial
  results, checkpoint/source-row hashes and output hashes. The report source is
  bound separately from the frozen trace run.
- Rows SHA256: `8fa30d4b52929f739d1eefce16ac781253fee548232e622c0f17b9c36630f1a6`.
- Manifest SHA256: `df3533531960c313bb71371127a59410a0e791cc1af4ef9d6b82c0557de91e71`.

Reproduce with the study/report commands in the README, then
`./venv/bin/python -m research.t9_divergence_20261017.audit --out <artifact-root>`.
This is a development diagnostic on selected T1 deaths and their B8-continuation
alternatives. It is not a promotion test, an observation-sufficiency proof, or a
claim that another policy could not survive a censored or fatal branch.
