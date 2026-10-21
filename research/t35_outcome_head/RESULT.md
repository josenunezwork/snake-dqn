# T35 outcome-head probe: result

**Registered verdict: KILL** (2026-10-20, Pacific). The rule is in PREREGISTRATION.md §6 (frozen in
`89c610e3` before any model saw the data). The verdict is not relabelled.

- Run: `snake-dqn-artifacts/t35-outcome-head-20261020/registered-02/result.json`
  (sha256 `62b644b97c9cf1b07dff1a69d40664201c3fa565d737559e70374fb017bba34f`), log `registered-02.log`.
- Code: `3e9cd3d5`. The two commits after the registration only fix plumbing: population std (a
  preflight warning) and multiple export roots plus a 39-world guard. The first launch
  (`registered-01`) stopped on a duplicate world id before any feature was built or any model was
  fit, because the shared root also held cpu-profile diagnostic worlds.
- Compute: Mac, nice 10, 2 worker processes × 1 thread. Features took 156 s, fits 31 min, 34 min in
  total, under the 1 h cap. The preflight took 18 s on 2 worlds. **Colab: 0 units.** The Round 0
  runtime was not touched, and `colab sessions` shows only that L4.

## Integrity (all pass)

| Check | Value |
|---|---|
| Worlds replayed / dropped | 39 / 0 |
| Rows (hero decisions) | 190,779 |
| Replayed B8 Q vs exported Q: rows > 1e-3 | 0 % (max abs diff 5.3e-4) |
| Searched / decisive rows / worlds with decisive rows | 296 / 225 / 21 |

## Primary results (225 held-out decisive rows, 1,350 (row, action) pairs, 21 worlds)

| Metric | Value | World-bootstrap 95 % CI | GO bar | KILL bar |
|---|---|---|---|---|
| Per-action survival AUC, teacher arm | **0.747** | 0.705 – 0.792 | ≥ 0.80, LB > 0.77 | ≤ 0.77 |
| Outcome-argmax = `a_star` | **46.2 %** | 39.1 – 52.2 % | ≥ 85 % | < 70 % |
| AUC, own-continuation control arm | 0.428 | — | | |
| Margin teacher − control | +0.320 | 0.267 – 0.380 | ≥ 0.03 | |

Both KILL clauses fire on their own: AUC ≤ 0.77 and agreement < 70 %.

## Secondary results (reported only)

| | AUC pooled | AUC without immediate deaths | Within-row AUC | Agreement | Chosen action survives | Mass Spearman |
|---|---|---|---|---|---|---|
| Teacher arm, ridge (primary) | 0.747 | 0.716 | 0.747 | 46.2 % | 60.9 % | 0.60 |
| Teacher arm, seeds 1 and 2 | 0.747 | 0.716 | 0.747 | 46.2 % | 60.9 % | 0.60 |
| Teacher arm, Adam recipe as the expert wrote it (5 epochs) | 0.550 | 0.519 | 0.541 | 29.3 % | 50.2 % | 0.27 |
| Control arm (own continuation) | 0.428 | 0.361 | 0.346 | 0.0 % | 0.0 % | 0.39 |
| B8 Q used as a survival score | 0.639 | 0.567 | — | 0 % (argmax Q) | — | — |
| Advisory mask used as a survival score | 0.637 | — | — | — | — | — |

Inner CV chose L2 = 0.1 for the teacher arm in 4 of 5 folds (0.01 in the other) and 0.01 or 0.001
for the control arm. Each fold trained on 227–243 teacher rows, and on 138k–148k control rows that
held only 900–1,200 death rows, from 3–4 hero deaths.

## Reading

- On these states, a linear read-out of B8's frozen trunk recovers the 300-step outcome only at
  about the reactive ceiling (0.747, against T8/T5b's 0.742–0.769). It is clearly above B8's own Q
  (0.639) and the advisory mask (0.637). It is far from usable as a decision rule: it picks the
  teacher's action on 46 % of rows and a surviving action on 61 %.
- The control arm is weak by construction. Only about 4 hero deaths supply its negatives, and its
  AUC falls below 0.5. So the large margin shows that the search outcomes carry information that
  bare continuation in this bank does not. It does not show that the teacher arm is good.
- **Caveat: the probe is data-starved.** Expert 3 assumed about 0.65M outcome rows. Only 296 rows
  carry outcomes (PREREGISTRATION §2), so each fold trains on about 235 rows. The KILL is the
  registered answer for this representation **and this amount of data**. It does not prove that the
  horizon cannot be recovered from the state. By expert 3's own protocol, the next step after a KILL
  would be one trainable-trunk attempt. With about 240 rows and a 1.6M-parameter trunk, that needs a
  larger outcome bank, such as Round 0's searched rows, and its own registration. On the current
  evidence the lever does not earn an R1 slot, so planner/Path B stays the route to the horizon.
