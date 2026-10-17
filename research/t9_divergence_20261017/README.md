# T9 divergence diagnostic

This implements only step 0 of the 2026-10-17 advice. It does not train a model,
change observations or rewards, run a promotion gate, or change serving.

Run from the repository root:

```sh
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  ./venv/bin/python -m research.t9_divergence_20261017.study
./venv/bin/python -m research.t9_divergence_20261017.report \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t9-divergence-20261017
```

The runner uses one CPU process and one numerical thread. The output directory
must not already exist. A separate pilot can use `--limit 1 --out /tmp/t9-pilot`.
It never manages sessions or other processes. All inputs are read-only.

## Measurement contract

- Cohort: all 46 T1 rows with death length at least 800 and an observed LAD.
  Reconstruct each source world with the original census checkpoint/seed/roster,
  replay recorded hero actions, and check every pre-action length. At the LAD,
  require exact Q, chosen action, highest-Q surviving action (lower action wins
  ties), and both T1 action-geometry dictionaries. Clone the prepared state twice.
- Fatal branch: the recorded action series. Surviving branch: the recorded
  highest-Q surviving action followed by that checkpoint's B8 greedy policy.
  Validate the fatal decision and the original T1 survivor endpoint beyond the
  measurement horizon. These additional control transitions are not trace data.
- Offset 0 is the prepared LAD before intervention. Offset k is the prepared
  live state after k transitions. Capture every integer offset through 100 or
  death; death itself has no live head geometry. Later offsets are missing,
  never zero-filled or carried forward. Report paired n at every summary offset.
- Static topology follows T1: four-neighbor connectivity; empty and own TTL<=1
  cells free; opponent cells blocked; current hero head opened as a seed. Compute
  the head component and largest component on this same uncapped board. Ties for
  largest count as largest. Tail BFS uses this board, with null for unreachable;
  Manhattan distance is unconditional. A growing body's tail can remain blocked.
- Radius 15/30 means Chebyshev distance, a square consistent with the 31x31 local
  crop. Counts are unique occupied cells, including heads. Boundary fractions
  count exposed cardinal edges, categorized by adjacent own body, opponent body,
  or arena exterior. Outside-window checks use the adjacent boundary cell's
  Chebyshev distance from the hero head, greater than 15.
- Nearest food minimizes Manhattan distance, with lexicographic (x,y) ties.
  Moving toward food means the action's first next cell decreases that distance;
  normal and boost variants share this first cell. Smaller food component means
  smaller than the largest free component. No food gives a null component flag.
- Arrival race is separate from static topology: full-board, unit-speed hero and
  live opponent sources, all current body segments releasing at their TTL.
  Earliest arrival uses `max(previous+1, destination_TTL)`; waiting is optimistic.
  No future growth, new trails, scripted opponent choices, respawns, or collisions
  are simulated by the flood. Source heads seed at time zero; other floods do
  not get those cells opened early. Count cells reached within 16/32/64/128/256
  steps strictly before every opponent; ties are excluded. This measures spatial
  access, not guaranteed feasible Snake paths or survival.

## Analysis contract

Paired differences are fatal minus surviving. Quartiles use NumPy's linear
interpolation. Null BFS/food values are excluded pairwise with explicit n.
Each metric has negative/zero/positive counts, including ties in the denominator.
The descriptive early-consistency scan reports the first offset with a nonzero
median, and the first with at least 75% of valid pairs sharing a strict sign,
requiring at least 20 pairs. This is an exploratory summary, not an inferential
test or promotion rule. Offset-100 estimates condition on both branches still
being alive and cannot represent deaths that already occurred.

The spec does not define a causal cell-identification rule for a "decisive
divergence point." We operationalize it as the first positive offset at which
the fatal head is outside the largest component and its region is strictly
smaller than the surviving head's region. Report the pocket boundary extending
beyond 15 cells at that point, both among resolved cases and over all 46; retain
unresolved cases explicitly. Also report boundary extent at the shared LAD.
Boundary extent alone does not prove that an unseen boundary cell caused the
fatal action or that the observation lacks sufficient predictive information.

`manifest.json` pins code, spec, source rows, census series and checkpoints;
`rows.jsonl` retains traces and replay checks; `completion.json` binds outputs
and confirms source/input stability; `summary.json` retains every offset and
`tables.md` the five requested offsets. Existing research inputs stay untouched.
