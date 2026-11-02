# T35 Auxiliary Outcome-Head Probe and Two-Deviation Rescue Probe

## Question and Goal

After the 2026-10-20 A100/GPU expert consultation ([`t34-a100-advice.md`](t34-a100-advice.md)), two cheap Mac-only probes were pre-registered under `T35` on 2026-10-20 (Pacific) to decide whether two GPU-scale levers deserved funding:

1. **T35 outcome-head probe (`research/t35_outcome_head`):** Can B8 s3's **frozen** 256-D trunk (fuse vector) linearly predict the teacher's exported 300-step per-action outcomes (survive, mean mass) well enough to choose the teacher's action, and is that signal more than the same head learns from B8's own continuation (control arm)? Source: GPU expert 3's "cheapest experiment" (`research/t35_outcome_head/PREREGISTRATION.md` §1).
2. **T35 two-deviation rescue probe (`research/t35_two_deviation`):** T3 dense search (one deviation from B8 greedy) is the ceiling mechanism; in the T34 shadow M2 truth scan, 33 of 75 historical T3 action changes had no single deviation in their D=300 window that survives 300 steps ("all-die events"). Are some all-die events rescuable with **two** deviations? If yes, a deeper GPU search oracle has a customer; if no, do not fund GPU search now. Source: GPU expert 2 (`research/t35_two_deviation/PREREGISTRATION.md`, "Question").

---

## Timeline of Steps and Versions

| Date (PT) | Step / Package | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-20 | `t35_outcome_head` `PREREGISTRATION.md` | Linear head `Linear(256, 12)` on the frozen B8 s3 fuse vector, full-batch L-BFGS with inner-CV L2, world-level 5-fold CV over the 39 A5 v2 export worlds. Pre-data premise correction: only **296** rows carry tested per-action outcomes (**225** decisive rows in **21** worlds), not expert 3's assumed ~0.65M. Frozen in `89c610e3`. | **REGISTERED** |
| 2026-10-20 | `t35_outcome_head` `RESULT.md` | Registered run `registered-02` (code `3e9cd3d5`; `registered-01` stopped on a duplicate world id before any fit). Mac, 2 procs × 1 thread, 34 min total, **0 Colab units**. Integrity: 39/0 worlds replayed/dropped, 190,779 rows, 0 % rows with Q diff > 1e-3. | **KILL** — AUC 0.747, agreement 46.2 % |
| 2026-10-20 | `t35_two_deviation` `PREREGISTRATION.md` + Amendment 1 | Exact CPU `GridBatchSim`/T3 rollout path; layer-1 expansion, beam k = 8, layer-2 expansion, rescue = alive at H=300 and confirmed alive at `min(t2 + 600, 5000)`. Amendment 1 (pre-data): decision unit = distinct world; primary = 7 events (earliest all-die event of each of the 7 worlds), EXTEND = the other 26; W ≥ 3/7 → REAL, W ≤ 1/7 → KILL, W = 2/7 → EXTEND. Frozen in `321ba7c7`. | **REGISTERED** |
| 2026-10-20 | `t35_two_deviation` `RESULT_PRIMARY.md` | 7 primary events; Mac, 3 single-thread procs, nice 10, 18:59–19:11 PT (~12 min, cap 3 h), 2,048,954 row-steps, **0 Colab units**. | **EXTEND** — W = 2/7, U = 0 |
| 2026-10-20 | `t35_two_deviation` `RESULT_EXTEND.md` | 26 EXTEND events, 19:18–19:37 PT, 3 procs, cap 7200 s, 0 Colab units; approval by an Opus stand-in (Fable usage limit). | **KILL** — W_pool = 2/7, 26/26 EXTEND events NOT_RESCUED |

---

## Final Verdict and Key Numbers

### 1. Outcome-head probe — **KILL** (`research/t35_outcome_head/RESULT.md`)

Run: `snake-dqn-artifacts/t35-outcome-head-20261020/registered-02/result.json` (sha256 `62b644b97c9cf1b07dff1a69d40664201c3fa565d737559e70374fb017bba34f`).

Primary results on 225 held-out decisive rows (1,350 (row, action) pairs, 21 worlds):

| Metric | Value | World-bootstrap 95 % CI | GO bar | KILL bar |
|---|---|---|---|---|
| Per-action survival AUC, teacher arm | **0.747** | 0.705 – 0.792 | ≥ 0.80, LB > 0.77 | ≤ 0.77 |
| Outcome-argmax = `a_star` | **46.2 %** | 39.1 – 52.2 % | ≥ 85 % | < 70 % |
| AUC, own-continuation control arm | 0.428 | — | | |
| Margin teacher − control | +0.320 | 0.267 – 0.380 | ≥ 0.03 | |

Both KILL clauses fire independently (AUC ≤ 0.77 and agreement < 70 %). Secondary (reported only): chosen action survives on 60.9 % of rows; B8 Q used as a survival score AUC 0.639; advisory mask AUC 0.637; the expert-literal Adam recipe (5 epochs) AUC 0.550.

### 2. Two-deviation rescue probe — **KILL** (`research/t35_two_deviation/RESULT_PRIMARY.md`, `RESULT_EXTEND.md`)

Primary summary: `snake-dqn-artifacts/t35-two-deviation-20261020/primary/summary-primary.json` (sha256 `69acca9d11f76cb11337c91df60110424b3f8af9348a11ab2f52b1782c428a90`).

| Event | Status | Rescue found by the registered beam | Row-steps | Search s |
|---|---|---|---:|---:|
| `b8_s0_21@3707` | NOT_RESCUED | — | 370,039 | 345 |
| `b8_s0_8@4318` | NOT_RESCUED | — | 326,941 | 312 |
| `b8_s3_19@4693` | RESCUED_DEPTH2 | (4813, 3) → (4965, 0), rank 1 (ranks 1–4 rescued); confirm 35 frames to episode end | 141,293 | 159 |
| `b8_s3_26@3647` | NOT_RESCUED | — | 380,981 | 351 |
| `b8_s3_6@4562` | NOT_RESCUED | — | 156,174 | 155 |
| `b8_s3_8@4246` | NOT_RESCUED | — | 285,910 | 284 |
| `b8_s3_9@4188` | RESCUED_DEPTH2 | (4198, 0) → (4204, 3), rank 3; confirm 600 frames | 387,616 | 370 |

Primary totals: rescued 2, not rescued 5, unknown 0; row-steps per rescue 1,024,477.

Pooled verdict (APPROVAL_EXTEND C3, fixed pre-data): **KILL — W_pool = 2/7 worlds**; EXTEND 26/26 events NOT_RESCUED, U_pool = 0; W_novel = 0; W_uncensored = 1 (`b8_s3_9`).

**Known-path caveat (Amendment 1 §A1.7):** both rescues are in the only two worlds where T3's own history already showed a multi-deviation escape. The `b8_s3_9@4188` rescue **is** the known A1.7 path; the `b8_s3_19@4693` rescue starts from a different first deviation and is episode-end censored (reach 35 steps). In the five worlds without a known path the beam found no rescue and no layer-2 candidate alive at t2 + 300.

---

## What Was Learned

1. **Frozen-trunk outcome head:** a linear read-out of B8's frozen trunk recovers the 300-step outcome only at about the reactive ceiling (0.747 vs T8/T5b's 0.742–0.769), above B8's own Q (0.639) but far from a usable decision rule (picks the teacher action on 46 % of rows, a surviving action on 61 %). The probe is **data-starved** (~235 training rows per fold), so the KILL is the registered answer for this representation and this amount of data, not a proof that the horizon cannot be recovered from the state. The lever does not earn an R1 slot; planner/Path B stays the route to the horizon (`RESULT.md`, "Reading").
2. **Two deviations:** the registered beam reproduced only already-known escapes and found no novel two-deviation rescue. The pooled KILL means the registered beam heuristic found ≤ 2 rescued worlds, so GPU search is not funded now; it is a statement about the registered beam, not a proof that no two-deviation rescue exists (`RESULT_EXTEND.md`; Amendment 1 §A1.3).

---

## Why It Closed / Current Status

- **Status:** **CLOSED, both KILL on 2026-10-20.** Neither probe used Colab units (both Mac CPU). GPU two-deviation search was not funded (`research/t35_two_deviation/README.md`, removed in consolidation; see git c4c4ab41).

---

## Kept Files in This Lineage

- [`research/t35_outcome_head/PREREGISTRATION.md`](../../../research/t35_outcome_head/PREREGISTRATION.md) — outcome-head registration (premise correction, recipe, GO/KILL bars).
- [`research/t35_outcome_head/RESULT.md`](../../../research/t35_outcome_head/RESULT.md) — registered KILL (AUC 0.747, agreement 46.2 %).
- [`research/t35_two_deviation/PREREGISTRATION.md`](../../../research/t35_two_deviation/PREREGISTRATION.md) — two-deviation registration plus Amendment 1.
- [`research/t35_two_deviation/RESULT_PRIMARY.md`](../../../research/t35_two_deviation/RESULT_PRIMARY.md) — primary EXTEND (W = 2/7).
- [`research/t35_two_deviation/RESULT_EXTEND.md`](../../../research/t35_two_deviation/RESULT_EXTEND.md) — pooled KILL (W_pool = 2/7).
- Package READMEs `research/t35_outcome_head/README.md` and `research/t35_two_deviation/README.md` were removed in consolidation; see git c4c4ab41.
