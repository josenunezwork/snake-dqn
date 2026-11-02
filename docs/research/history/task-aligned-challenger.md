# Task-Aligned CZ `enemy_visible` Challenger & 6102 Continuation Series (2026-09-24 to 2026-09-26)

## Question & Goal

Following the September 2026 four-track Astra research review (`docs/research_review_2026-09-24/README.md`), this lineage tested whether continuing the three CZ `enemy_visible` update-`4608` PQN `raster31v3` parents (`2026096101`, `2026096102`, `2026096103`) under a task-aligned mixed rollout schedule (`[solo, s2, s2, s6, s6]`, `H1024` hero episodes, `E16 × T16`, `Q scale 0.1`, restricted tactical channels `0–5` and scalars `0, 1, 12–14`) would improve total-horizon living mass over the frozen parents while retaining solo competence, and whether the nominated challenger could then pass serving-bridge qualification and beat the incumbent Apex/`vector61` (`champion_a5_freespace_20260621.pth`) at `H5000` on `promotion-v2-watch-rect` (`research/task_aligned_20260924/protocol.md`, `docs/research/task_aligned_challenger_design_draft_2026-09-24.md`, `docs/research/task_aligned_challenger_serving_plan_2026-09-24.md` (removed in consolidation; see git c4c4ab41)).

---

## Timeline of Steps & Versions

| Date | Step / Run | What Was Run | Verdict |
|---|---|---|---|
| 2026-09-24 | **4-Track Research Review** (`docs/research_review_2026-09-24/README.md`) | Four Astra reviewers (learning, evaluation, operations, strategy) attributed stalled progress mainly to task/distribution mismatch, unresolved TD retention and a fragmented process; root prioritized one task-distribution package screen against frozen parents, starting from all three CZ `enemy_visible` parents. | **Direction adopted (`task_aligned_20260924`)** |
| 2026-09-24 | **First-Lineage Exploratory Parallel Preview** (`docs/experiments/task_aligned_first_lineage_preview_2026-09-24/README.md` (removed in consolidation; see git c4c4ab41)) | CPU preview of lineage `2026096101` at update `35,613` vs its update-`4,608` parent on `32` paired `solo` and `32` paired `mixed` `H3000` worlds (`128` episodes, `415.039 s` guarded) while lineage 2 trained on MPS. | **`COMPLETE_AUDITED_EXPLORATORY_MIXED_RESULT`** (`mixed` mass `+16.44`, `solo` mass `-15.05`) |
| 2026-09-24 | **3-Lineage Task-Aligned CZ Continuation (`admission-v2`)** (`docs/experiments/task_aligned_cz_integration_2026-09-24/README.md`) | Continued `2026096101`, `2026096102`, `2026096103` for `31,005` rollouts each (`93,015` scientific updates / `23,776,196` hero transitions; `36,853.56` guarded s), evaluated on `32` paired `H3000` worlds × `3` profiles (`solo`, `food_pressure`, `mixed`; `768` episodes). | **`INCONCLUSIVE_SCREEN` (audit `PASS`)** — only **`2026096102`** passed every gain and solo-retention check and was nominated as challenger |
| 2026-09-24 | **Further Continuation of `2026096102` (`35,613 → 66,618`)** (`docs/experiments/task_aligned_continuation_2026-09-24/README.md`) | Continued `2026096102` another `31,005` fresh-game rollouts/updates (`7,933,551` valid hero transitions; `12,692.35` guarded s) and evaluated against its update-`35,613` source on `32` paired `H3000` worlds × `3` profiles (`12` cells, `384` episodes). | **`NEGATIVE_CONTINUATION_SCREEN` (audit `PASS`)** — solo mass `-35.152` (`CI95 [-45.794, -24.509]`); update-`35,613` `6102` preserved |
| 2026-09-24 | **Serving Bridge Qualification `v1`** (`docs/experiments/task_aligned_bridge_qualification_2026-09-24/README.md` (removed in consolidation; see git c4c4ab41), source `aa3d0dd0`) | Controller preflight for the original `6102` serving bridge. | **`CLOSED_INVALID_STOP_CONTROLLER_PREFLIGHT_IDENTITY_MISMATCH`** (`0` stages admitted; a manually copied 63-hex-character candidate literal rejected the correct intent at `runner.py:129–133`) |
| 2026-09-24 | **Serving Bridge Replacement `v2`** (`docs/experiments/task_aligned_bridge_replacement_2026-09-24/README.md` (removed in consolidation; see git c4c4ab41), `failure-source-review.md`, source `b22d59c5`) | Repaired identity guard; capture child ran (`1` child, `0` completed stages, `1.026` guarded s). | **`CLOSED_INVALID_STOP_CAPTURE_JSON_ROUNDTRIP_MISMATCH`** (`capture.py:580` compared in-memory tuple `(9, 31, 31)` with JSON-reloaded list `[9, 31, 31]`) |
| 2026-09-25 | **Serving Bridge Saved-Prefix Recovery `v3` (`admission-v3`)** (`docs/experiments/task_aligned_bridge_saved_prefix_recovery_2026-09-25/README.md` (removed in consolidation; see git c4c4ab41), source `595e17b3`) | Saved-prefix handoff (`4` synthetic cases) + `75` fixture tests passed (`79` cases, `1.858` guarded s); live shakedown then started. | **`CLOSED_INVALID_STOP_SHAKEDOWN_HEARTBEAT_SCHEMA`** (watchdog stop at `20.05 s` with `1,966` completed frames persisted in the first cell; writer emitted wall-clock `time`, supervisor expects numeric `monotonic`) |
| 2026-09-25 | **Serving Bridge Remaining Qualification `v4` (`admission-v4`)** (`docs/experiments/task_aligned_bridge_remaining_qualification_2026-09-25/README.md` (removed in consolidation; see git c4c4ab41), source `344a7974`) | `7` new heartbeat cases (`0.41 s`), `12` `H5000` shakedown episodes (`334.41 s`), `1` Watch episode (`52.82 s`) and `1` Play episode terminating at `24` frames (`1.03 s`); shakedown used world indices `2` and `3`, shared with Watch/Play (`388.87 s` total guarded). | **`CLOSED_INVALID_STOP_AUDIT_FAILED_LIVE_RESULTS_PROVISIONAL`** (`audit.py:970` rejected `82` passed `call` reports for `75` fixture nodes; the `7` extras match `unittest.subTest` reports under pytest 9.0.1) |
| 2026-09-26 | **Serving Bridge Audit-Only Verification `v5` (`audit-only-v5`)** (`docs/experiments/task_aligned_bridge_audit_only_2026-09-26/README.md` (removed in consolidation; see git c4c4ab41), source `9495dc7`) | Read-only corrected audit of saved `admission-v4` evidence (`87` JSON reads, child exit after `0.234 s`; no checkpoints, models, games or pytest). | **`AUDIT_ONLY_PASS`** (all `14` cells validated; fixture subtests `3 + 4`, provenance `schema1-structural-inference`) |
| 2026-09-26 | **Strict Challenge Pilot vs Incumbent Apex (`pilot-v1`)** (`docs/experiments/task_aligned_strict_pilot_2026-09-26/README.md` (removed in consolidation; see git c4c4ab41), source `56547b4`) | `48` Apex-only development episodes (`16` worlds × `3` mixes) and `96` paired pilot episodes (`16` worlds × `3` mixes × candidate/incumbent) at `H5000` on `promotion-v2-watch-rect` (`3,828 s`, `720,000` completed frames). | **`INVALID_STOP_AUDIT_FAILED`; unaudited producer outcome `PILOT_STOP_INFEASIBLE_AT_FIXED_MDE` — LINEAGE RETIRED** (required `N = 8,676` > cap `120`; `6102` trailed Apex by `-30.2` to `-66.4` mass) |

---

## Final Verdict & Key Numbers

**Final Lineage Verdict:** **6102 / task-aligned challenger line RETIRED (2026-09-26)** under the pre-registered kill criterion (pilot needed > `120` worlds, and the audit failed; either alone retires the line) (`docs/experiments/task_aligned_strict_pilot_2026-09-26/README.md` (removed in consolidation; see git c4c4ab41); `docs/research/experiment_portfolio_review_2026-09-26.md`). Apex/`vector61` remains the incumbent.

### 1. 3-Lineage Task-Aligned CZ Continuation (`4,608 → 35,613`, `H3000`, `N = 32` Paired Worlds/Profile)
Primary sources: `docs/experiments/task_aligned_cz_integration_2026-09-24/README.md` and `docs/experiments/task_aligned_cz_integration_2026-09-24/all-cells.md` (saved `decision.json` file SHA-256 `abb1adb87f8d38bf325d2799a8326c34b8fff3f0685c7ef20d7858db87eb27f2`, as recorded in the README; the run directory is no longer in `snake-dqn-artifacts`).

Mass columns are full-horizon living mass per frame; food, alive % and boost % are parent → candidate from `all-cells.md`.

| Lineage (Update `35,613` vs `4,608`) | Profile | Parent Mass | Final Mass | Paired Delta (`CI95`) | Ambient Food | Alive % | Boost % | Saved Check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`2026096101`** | `solo` | `62.494` | `60.401` | `-2.093` (`[-25.720, +21.534]`) | `203.781 → 462.750` | `47.67 → 69.25` | `0.10 → 35.54` | Mass retention not established; food/alive pass |
| **`2026096101`** | `food_pressure` | `23.908` | `48.325` | **`+24.417`** (`[+10.893, +37.941]`) | `130.188 → 390.812` | `27.18 → 54.96` | `0.08 → 28.49` | **Gain pass** |
| **`2026096101`** | `mixed` | `35.725` | `50.256` | **`+14.531`** (`[+0.716, +28.345]`) | `158.281 → 400.219` | `33.72 → 58.13` | `0.11 → 29.84` | **Gain pass** |
| **`2026096102`** *(challenger)* | `solo` | `36.516` | `74.462` | **`+37.946`** (`[+25.150, +50.742]`) | `303.125 → 445.969` | `56.15 → 77.87` | `21.95 → 34.73` | **All retention checks pass** |
| **`2026096102`** *(challenger)* | `food_pressure` | `15.327` | `55.203` | **`+39.876`** (`[+25.825, +53.927]`) | `180.062 → 353.625` | `31.52 → 57.77` | `11.93 → 27.26` | **Gain pass** |
| **`2026096102`** *(challenger)* | `mixed` | `17.136` | `55.239` | **`+38.104`** (`[+25.394, +50.813]`) | `205.375 → 349.188` | `35.79 → 58.50` | `13.73 → 27.32` | **Gain pass** |
| **`2026096103`** | `solo` | `72.291` | `55.092` | `-17.199` (`[-33.093, -1.304]`) | `261.844 → 226.812` | `51.75 → 50.40` | `3.69 → 5.84` | Mass, food and alive retention not established (upper bound `-1.304` above the `-3.615` clear-harm boundary, so not clear harm) |
| **`2026096103`** | `food_pressure` | `19.950` | `33.599` | **`+13.649`** (`[+4.002, +23.295]`) | `131.750 → 169.531` | `24.70 → 33.50` | `1.31 → 3.61` | **Gain pass** |
| **`2026096103`** | `mixed` | `36.031` | `40.374` | `+4.343` (`[-9.863, +18.548]`) | `175.469 → 183.250` | `33.69 → 37.77` | `1.98 → 4.44` | Gain not established |

*Note on the parallel preview (`docs/experiments/task_aligned_first_lineage_preview_2026-09-24/README.md` (removed in consolidation; see git c4c4ab41)):* On its separate preview worlds, `2026096101` (`update 35,613` vs `4,608`) moved `mixed` full-horizon living mass from `32.34` to `48.78` (`+16.44`, `20/1/11` W/T/L; food `145.41 → 391.38`; frames alive `953.78 → 1696.63`; boost `0.001042 → 0.288875`) while `solo` living mass moved from `81.22` to `66.18` (`-15.05`, `17/0/15` W/T/L; food `231.53 → 494.75`; frames alive `1619.91 → 2238.16`; boost `0.000979 → 0.385438`).

### 2. Further Continuation of `2026096102` (`35,613 → 66,618`, `H3000`, `N = 32` Paired Worlds/Profile)
Primary sources: `docs/experiments/task_aligned_continuation_2026-09-24/README.md`, `docs/experiments/task_aligned_continuation_2026-09-24/presentation/all-cells.md` and `.../presentation/paired-world-differences.md` (saved decision SHA-256 `247d4add564ec1e52f569c227dbb99baed6a50a4bd485dea095004b04a4683c3`, as recorded in the README). W/T/L counts below are derived from the signs of the per-world mass differences in `paired-world-differences.md`.

- **`solo`:** Mean living mass fell from **`78.750`** (`35,613`) to **`43.598`** (`66,618`), paired delta **`-35.152`** (`CI95 [-45.794, -24.509]`, below the `-3.938` clear-harm bound; `4/0/28` W/T/L), while food rose (`470.9 → 660.1`), alive fraction rose (`0.806 → 0.946`) and boost fraction rose (`0.362 → 0.646`).
- **`food_pressure`:** Mean living mass moved from **`52.952`** to **`47.885`**, paired delta **`-5.067`** (`CI95 [-16.924, +6.790]`, `14/0/18` W/T/L), with food `338.1 → 610.3`, alive fraction `0.565 → 0.821`, and boost fraction `0.270 → 0.555`.
- **`mixed`:** Mean living mass moved from **`58.036`** to **`50.737`**, paired delta **`-7.299`** (`CI95 [-21.090, +6.492]`, `14/0/18` W/T/L), with food `363.4 → 629.1`, alive fraction `0.612 → 0.848`, and boost fraction `0.287 → 0.576`.
- **Outcome:** `NEGATIVE_CONTINUATION_SCREEN` (solo clear harm); the original update-`35,613` `2026096102` checkpoint (`SHA-256 8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`) remained the sole nominated challenger.

### 3. Serving Bridge Qualification (`v1`–`v5`) & Strict Challenge Pilot (`pilot-v1`) vs Incumbent Apex (`H5000`)
Primary sources: `docs/experiments/task_aligned_bridge_replacement_2026-09-24/failure-source-review.md`, `docs/experiments/task_aligned_bridge_remaining_qualification_2026-09-25/README.md` (removed in consolidation; see git c4c4ab41), `docs/experiments/task_aligned_bridge_audit_only_2026-09-26/README.md` (removed in consolidation; see git c4c4ab41) (audit-only `output/report.json` SHA-256 `41d0b092ab32fdcac4dd78aae696f0f250c1a4e737fbe2b835b7efa15f490eee`), and `docs/experiments/task_aligned_strict_pilot_2026-09-26/README.md` (removed in consolidation; see git c4c4ab41).

- **Bridge Qualification (`audit-only-v5`):** The corrected audit passed the saved `admission-v4` interface/accounting evidence: carried `4` handoff + `75` fixture passes (fixture subtests `3 + 4`), `7` heartbeat cases, `12` `H5000` shakedown episodes, `1` Watch episode and `1` Play episode (`14` live episodes; shakedown world indices `2`/`3` shared with Watch and Play). It does not establish better gameplay or independent world coverage.
- **Strict Challenge Pilot (`pilot-v1`, `16` development + `16` paired pilot worlds per mix at `H5000` on `promotion-v2-watch-rect`, unaudited):**
  - **`frozen` mix:** Apex dev mean mass `52.61` (10% MDE `5.26`), paired-difference `SD 56.3` → required **`N = 1,072`**. Pilot descriptive mean (candidate − Apex): **`-30.2`** mass (`6102` won `5` of `16` worlds).
  - **`scripted` mix:** Apex dev mean mass `38.82` (10% MDE `3.88`), paired-difference `SD 118.3` → required **`N = 8,676`**. Pilot descriptive mean: **`-66.4`** mass (`5` of `16` worlds won).
  - **`mixed` mix:** Apex dev mean mass `72.65` (10% MDE `7.26`), paired-difference `SD 55.8` → required **`N = 552`**. Pilot descriptive mean: **`-34.9`** mass (`5` of `16` worlds won).
  - Required final worlds: **`8,676`** against a pre-registered cap of `120`.
  - *(Audit note:* `pilot-v1` recorded `INVALID_STOP_AUDIT_FAILED` because the audit required `lifecycle_frames` `240000` for the development stage, but `lifecycle_frames` is charged only by the bridge live-parity probe, which the strict pilot does not install. Both the `N > 120` result and the failure independently trigger the pre-registered retirement criterion.)

---

## What Was Learned & Why It Closed

1. **Further continuation traded living mass for food, survival and boost:** Continuing `2026096102` from update `35,613` to `66,618` raised solo alive fraction (`0.806 → 0.946`) and food (`470.9 → 660.1`) and raised boost fraction to `0.555–0.646` (from `0.270–0.362`), while full-horizon living mass fell `-35.15` in solo and `-5.07` / `-7.30` (CIs crossing zero) in the opponent profiles. Higher boost co-occurred with lower mass, but boost was not intervened on, so the study does not establish it as the cause.
2. **Task-aligned PQN `raster31v3` (`6102`) remained weaker than incumbent `vector61` Apex at `H5000`:** Despite `+37.9` over its own CZ parent at `H3000`, `6102` trailed Apex by `-30.2` (`frozen`), `-66.4` (`scripted`) and `-34.9` (`mixed`) mass in the unaudited pilot.
3. **A strict 10%-MDE comparison at the deployed profile needs thousands of worlds (`N = 552–8,676` per mix):** The observed noise-to-mean ratio was `1–3`. Future strict gates should plan on much higher simulation throughput (the vectorized-sim direction) or a variance-reducing evaluation design pre-registered before data.

**Status:** **CLOSED / RETIRED (2026-09-26).**

---

## Kept Lineage Documents (`KEEP`)

- **4-Track Research Review (2026-09-24):**
  - [`docs/research_review_2026-09-24/README.md`](../../research_review_2026-09-24/README.md)
  - [`docs/research_review_2026-09-24/evaluation.md`](../../research_review_2026-09-24/evaluation.md)
  - [`docs/research_review_2026-09-24/learning.md`](../../research_review_2026-09-24/learning.md)
  - [`docs/research_review_2026-09-24/operations.md`](../../research_review_2026-09-24/operations.md)
  - [`docs/research_review_2026-09-24/strategy.md`](../../research_review_2026-09-24/strategy.md)
- **3-Lineage Task-Aligned CZ Integration (`4,608 → 35,613`):**
  - [`research/task_aligned_20260924/README.md`](../../../research/task_aligned_20260924/README.md)
  - [`research/task_aligned_20260924/protocol.md`](../../../research/task_aligned_20260924/protocol.md)
  - [`research/task_aligned_20260924/protocol-evaluation.md`](../../../research/task_aligned_20260924/protocol-evaluation.md)
  - [`research/task_aligned_20260924/contracts-learning.md`](../../../research/task_aligned_20260924/contracts-learning.md)
  - [`research/task_aligned_20260924/contracts-operations.md`](../../../research/task_aligned_20260924/contracts-operations.md)
  - [`research/task_aligned_20260924/contracts-qualification.md`](../../../research/task_aligned_20260924/contracts-qualification.md)
  - [`research/task_aligned_20260924/review-notes.md`](../../../research/task_aligned_20260924/review-notes.md)
  - [`docs/experiments/task_aligned_cz_integration_2026-09-24/README.md`](../../experiments/task_aligned_cz_integration_2026-09-24/README.md)
  - [`docs/experiments/task_aligned_cz_integration_2026-09-24/all-cells.md`](../../experiments/task_aligned_cz_integration_2026-09-24/all-cells.md)
- **Further Continuation of `2026096102` (`35,613 → 66,618`):**
  - [`research/task_aligned_continuation_20260924/README.md`](../../../research/task_aligned_continuation_20260924/README.md)
  - [`research/task_aligned_continuation_20260924/protocol.md`](../../../research/task_aligned_continuation_20260924/protocol.md)
  - [`research/task_aligned_continuation_20260924/protocol-evaluation.md`](../../../research/task_aligned_continuation_20260924/protocol-evaluation.md)
  - [`research/task_aligned_continuation_20260924/contracts-learning.md`](../../../research/task_aligned_continuation_20260924/contracts-learning.md)
  - [`docs/experiments/task_aligned_continuation_2026-09-24/README.md`](../../experiments/task_aligned_continuation_2026-09-24/README.md)
  - [`docs/experiments/task_aligned_continuation_2026-09-24/presentation/all-cells.md`](../../experiments/task_aligned_continuation_2026-09-24/presentation/all-cells.md)
  - [`docs/experiments/task_aligned_continuation_2026-09-24/presentation/paired-world-differences.md`](../../experiments/task_aligned_continuation_2026-09-24/presentation/paired-world-differences.md)
  - [`docs/experiments/task_aligned_continuation_2026-09-24/presentation/all-cases.md`](../../experiments/task_aligned_continuation_2026-09-24/presentation/all-cases.md)
- **Serving Bridge & Challenger Portfolio Synthesis:**
  - [`docs/experiments/task_aligned_bridge_replacement_2026-09-24/failure-source-review.md`](../../experiments/task_aligned_bridge_replacement_2026-09-24/failure-source-review.md)
  - [`docs/research/task_aligned_challenger_design_draft_2026-09-24.md`](../task_aligned_challenger_design_draft_2026-09-24.md)
  - [`docs/research/experiment_portfolio_review_2026-09-26.md`](../experiment_portfolio_review_2026-09-26.md)
