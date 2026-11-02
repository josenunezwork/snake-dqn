# Task-Aligned CZ `enemy_visible` Challenger & 6102 Continuation Series (2026-09-24 to 2026-09-26)

## Question & Goal

Following the September 2026 four-track research review (`docs/research_review_2026-09-24/README.md`), this lineage tested whether continuing the three Study CZ `enemy_visible` update-`4,608` PQN `raster31v3` parents (`2026096101`, `2026096102`, `2026096103`) under a task-aligned mixed rollout schedule (`[solo, s2, s2, s6, s6]`, rollout horizon `H = 1,024`, `E16 × T16`, `Q scale = 0.1`, restricted tactical channels `0–5` and scalars `0, 1, 12–14`) could bridge the gap between short solo/2-snake screens (`H = 128–512`) and the 6-snake deployment environment, pass serving-bridge qualification, and defeat the operational `vector61` incumbent (`champion_a5_freespace_20260621.pth`, Apex DQN) on the shared `H = 5,000` `promotion-v2-watch-rect` strict tournament gate (`research/task_aligned_20260924/protocol.md`, `docs/research/task_aligned_challenger_design_draft_2026-09-24.md`).

---

## Timeline of Steps & Versions

| Date | Step / Run | What Was Run | Verdict |
|---|---|---|---|
| 2026-09-24 | **4-Track Research Review** (`docs/research_review_2026-09-24/README.md`) | Audited `187` completed experiments across evaluation, learning, operations, and strategy; identified horizon/occupancy mismatch (`H128–512` solo/2-snake screens vs 6-snake `H5000` deployment) and recommended task-aligned continuation from Study CZ `enemy_visible` parents. | **ADOPTED (`task_aligned_20260924`)** |
| 2026-09-24 | **First-Lineage Exploratory Parallel Preview** (`task-aligned-first-lineage-preview-20260924/run/stages/audit/output/comparison.json`) | CPU preview of lineage `2026096101` at update `35,613` vs its update-`4,608` parent across `32` paired `solo` and `32` paired `mixed` `H = 3,000` worlds (`128` episodes, `415.04 s`) while lineage 2 trained on MPS. | **`COMPLETE_AUDITED_EXPLORATORY_MIXED_RESULT`** (`mixed` mass `+16.44`, `solo` mass `-15.05`) |
| 2026-09-24 | **3-Lineage Task-Aligned CZ Continuation (`admission-v2`)** (`docs/experiments/task_aligned_cz_integration_2026-09-24/README.md`) | Continued `2026096101`, `2026096102`, and `2026096103` for `31,005` updates each (`4,608 → 35,613`; `93,015` updates / `23,776,196` hero transitions in `36,853.56 s`), evaluated on `32` paired `H = 3,000` worlds × `3` profiles (`solo`, `food_pressure`, `mixed`; `768` episodes). | **`INCONCLUSIVE_SCREEN` (`audit: PASS`)** — lineage **`2026096102`** passed all 3 profile bars and was nominated as challenger |
| 2026-09-24 | **Further Continuation of `2026096102` (`35,613 → 66,618`)** (`docs/experiments/task_aligned_continuation_2026-09-24/README.md`) | Continued `2026096102` another `31,005` fresh-game updates (`7,933,551` transitions in `12,692.35 s`) and evaluated against its update-`35,613` parent on `32` fresh paired `H = 3,000` worlds × `3` profiles (`384` episodes). | **`NEGATIVE_CONTINUATION_SCREEN` (`audit: PASS`)** — solo mass `-35.152` (`CI95 [-45.794, -24.509]`); update-`35,613` `6102` preserved |
| 2026-09-24 | **Serving Bridge Qualification `v1`** (`task-aligned-bridge-qualification-20260924/run/stages/audit/output/closeout.json`, source `aa3d0dd0`) | Preflight for live/BatchSim serving bridge of restricted `raster31v3` `6102` (`body capacity 10,400`, `Q scale 0.1`, uint64 training-seed exclusion). | **`CLOSED_INVALID_STOP_CONTROLLER_PREFLIGHT_IDENTITY_MISMATCH`** (`0` stages run; 63-char truncated SHA hex literal in `runner.py:129-133`) |
| 2026-09-24 | **Serving Bridge Replacement `v2` (`admission-v2`)** (`docs/experiments/task_aligned_bridge_replacement_2026-09-24/failure-source-review.md`, source `b22d59c5`) | Re-ran preflight and capture stage (`1.03 s`, `1` stage executed). | **`CLOSED_INVALID_STOP_CAPTURE_JSON_ROUNDTRIP_MISMATCH`** (`capture.py:580` compared tuple `(9, 31, 31)` vs JSON-reloaded list `[9, 31, 31]`) |
| 2026-09-25 | **Serving Bridge Saved-Prefix Recovery `v3` (`admission-v3`)** (`task-aligned-bridge- qualification-20260924/admission-v3/run/stages/audit/output/closeout.json`, source `595e17b3`) | Reused verified `v2` capture prefix (`4/4` handoff checks + `75/75` fixture tests in `1.86 s`); launched 12-episode `H = 5,000` shakedown. | **`CLOSED_INVALID_STOP_SHAKEDOWN_HEARTBEAT_SCHEMA`** (stopped `1,966` frames into cell 1 at `20.05 s`; writer emitted wall-clock `time` instead of `monotonic`) |
| 2026-09-25 | **Serving Bridge Remaining Qualification `v4` (`admission-v4`)** (`task-aligned-bridge-qualification-20260924/admission-v4/run/stages/audit/output/closeout.json`, source `344a7974`) | Completed `7/7` heartbeat tests (`0.22 s`), `12/12` `H = 5,000` shakedown episodes (`334.41 s`), `1` Watch `H = 5,000` episode (`52.82 s`), and `1` Play `24`-frame episode (`1.03 s`) across shared worlds `2` and `3` (`388.87 s` total). | **`CLOSED_INVALID_STOP_AUDIT_FAILED_LIVE_RESULTS_PROVISIONAL`** (`audit.py:970` rejected `82` passed `call` rows for `75` pytest nodes due to `7` `unittest.subTest` reports in pytest 9.0.1) |
| 2026-09-26 | **Serving Bridge Audit-Only Verification `v5` (`audit-only-v5`)** (`task-aligned-bridge-qualification-20260924/audit-only-v5/output/report.json`, source `9495dc7`) | Read-only re-audit of frozen `v2`/`v3`/`v4` evidence (`87` JSON reads in `0.234 s`; zero model loads or gameplay). | **`AUDIT_ONLY_PASS`** (`75` tests + `3 + 4` subtests and `14/14` live episodes verified) |
| 2026-09-26 | **Strict Challenge Pilot vs Incumbent Apex (`pilot-v1`)** (`task-aligned-strict-challenge-20260926/pilot-v1/run/stages/audit/output/closeout.json`, source `56547b4`) | Evaluated `48` Apex-only dev episodes (`16` worlds × `3` mixes) and `96` paired pilot episodes (`16` worlds × `3` mixes × `6102` vs `Apex`) at `H = 5,000` on `promotion-v2-watch-rect` (`3,828 s`, `720,000` frames). | **`PILOT_STOP_INFEASIBLE_AT_FIXED_MDE` / LINEAGE RETIRED** (required `N = 8,676 > N_max = 120`; `6102` trailed Apex by `-30.2` to `-66.4` mass/frame) |

---

## Final Verdict & Key Numbers

**Final Lineage Verdict:** **`PILOT_STOP_INFEASIBLE_AT_FIXED_MDE` — LINEAGE RETIRED (2026-09-26)** (`docs/research/experiment_portfolio_review_2026-09-26.md`).

### 1. 3-Lineage Task-Aligned CZ Continuation (`4,608 → 35,613`, `H = 3,000`, `N = 32` Paired Worlds/Profile)
Primary sources: `docs/experiments/task_aligned_cz_integration_2026-09-24/README.md`, `docs/experiments/task_aligned_cz_integration_2026-09-24/all-cells.md`, and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/task-aligned-cz-integration-20260924/admission-v2/run/analysis/output/decision.json` (SHA-256 `abb1adb875bb494649c1a1bcff4e6a98a4a6e7762c7233f4dd50b7c7415f7b4f`).

| Lineage (Update `35,613` vs `4,608`) | Profile | Parent Mass | Final Mass | Paired Delta (`CI95`) | Alive Frames (`Parent → Final`) | Food (`Parent → Final`) | Boost Fraction (`Parent → Final`) | Rule Check |
|---|---|---:|---:|---:|---:|---:|---:|---|
| **`2026096101`** | `solo` | `62.494` | `60.401` | `-2.093` (`[-25.720, +21.534]`) | `1417.0 → 2266.2` | `184.4 → 527.1` | `0.001 → 0.409` | Retention not established |
| **`2026096101`** | `food_pressure` | `23.908` | `48.325` | **`+24.417`** (`[+10.893, +37.941]`) | `827.5 → 1880.1` | `117.1 → 418.4` | `0.001 → 0.305` | **PASS** |
| **`2026096101`** | `mixed` | `35.725` | `50.256` | **`+14.531`** (`[+0.716, +28.345]`) | `1027.7 → 1766.8` | `160.3 → 409.5` | `0.001 → 0.303` | **PASS** |
| **`2026096102`** *(challenger)* | `solo` | `36.516` | `74.462` | **`+37.946`** (`[+25.150, +50.742]`) | `1070.3 → 2394.4` | `109.7 → 455.2` | `0.001 → 0.351` | **PASS (Retention & Gain)** |
| **`2026096102`** *(challenger)* | `food_pressure` | `15.327` | `55.203` | **`+39.876`** (`[+25.825, +53.927]`) | `556.8 → 2016.2` | `69.9 → 396.3` | `0.001 → 0.289` | **PASS** |
| **`2026096102`** *(challenger)* | `mixed` | `17.136` | `55.239` | **`+38.104`** (`[+25.394, +50.813]`) | `573.8 → 1919.6` | `76.5 → 384.3` | `0.002 → 0.280` | **PASS** |
| **`2026096103`** | `solo` | `72.291` | `55.092` | `-17.199` (`[-33.093, -1.304]`) | `1556.0 → 2033.2` | `204.6 → 457.5` | `0.001 → 0.382` | **FAIL (Clear Harm)** |
| **`2026096103`** | `food_pressure` | `19.950` | `33.599` | **`+13.649`** (`[+4.002, +23.295]`) | `772.9 → 1456.6` | `97.9 → 279.6` | `0.001 → 0.244` | **PASS** |
| **`2026096103`** | `mixed` | `36.031` | `40.374` | `+4.343` (`[-9.863, +18.548]`) | `946.4 → 1564.1` | `154.0 → 326.8` | `0.001 → 0.264` | Gain not established |

*Note on the parallel preview (`task-aligned-first-lineage-preview-20260924/run/stages/audit/output/comparison.json`, SHA-256 `6876aa2298bd4bf7c8ca9a0c8d2ec4b362a68452be890e8484f3045b9d5d6fd0`):* On its separate preview seed bank, `2026096101` (`update 35,613` vs `4,608`) moved `mixed` living mass from `32.34` to `48.78` (`+16.44`, `20/1/11` W/T/L, food `145.41 → 391.38`, alive frames `953.78 → 1696.63`, boost `0.0010 → 0.2889`) while `solo` living mass moved from `81.22` to `66.18` (`-15.05`, `13/0/19` W/T/L, food `231.53 → 494.75`, alive frames `1619.91 → 2238.16`, boost `0.0010 → 0.3854`).

### 2. Further Continuation of `2026096102` (`35,613 → 66,618`, `H = 3,000`, `N = 32` Fresh Paired Worlds/Profile)
Primary sources: `docs/experiments/task_aligned_continuation_2026-09-24/README.md`, `docs/experiments/task_aligned_continuation_2026-09-24/presentation/all-cells.md`, and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/task-aligned-continuation-20260924/admission-v2/run/analysis/output/decision.json` (SHA-256 `247d4add3883a4769e7db201f9bd5e19534b65726b7d40a8a3d809b300efcf9f`).

- **`solo`:** Mean living mass fell from **`78.750`** (`35,613`) to **`43.598`** (`66,618`), paired delta **`-35.152`** (`CI95 [-45.794, -24.509]`, well below the `-3.938` clear-harm bound; `3/0/29` W/T/L), even though food increased (`470.9 → 660.1`) and alive fraction increased (`0.806 → 0.946`), because boost fraction nearly doubled (`0.362 → 0.646`).
- **`food_pressure`:** Mean living mass moved from **`52.952`** to **`47.885`**, paired delta **`-5.067`** (`CI95 [-19.442, +9.308]`, `13/0/19` W/T/L), with food `338.1 → 610.3`, alive fraction `0.565 → 0.821`, and boost fraction `0.270 → 0.555`.
- **`mixed`:** Mean living mass moved from **`58.036`** to **`50.737`**, paired delta **`-7.299`** (`CI95 [-20.785, +6.188]`, `13/0/19` W/T/L), with food `363.4 → 629.1`, alive fraction `0.612 → 0.848`, and boost fraction `0.287 → 0.576`.
- **Outcome:** `NEGATIVE_CONTINUATION_SCREEN`; the original update-`35,613` `2026096102` checkpoint (`SHA-256 8d67915ef1ea5dbb0804ffd8bf9874dedb27c95878bacdd5b5e5ae77682aa46f`) remained the sole nominated challenger.

### 3. Serving Bridge Qualification (`v1–v5`) & Strict Challenge Pilot (`pilot-v1`) vs Incumbent Apex (`H = 5,000`)
Primary sources: `docs/experiments/task_aligned_bridge_replacement_2026-09-24/failure-source-review.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/task-aligned-bridge-qualification-20260924/audit-only-v5/output/report.json` (SHA-256 `41d0b092d480a710396601cb349181fa748d88dcdf882efee2fb5855ba9d399c`), and `/Users/josenunez/Projects/ml/snake-dqn-artifacts/task-aligned-strict-challenge-20260926/pilot-v1/run/stages/audit/output/closeout.json` (plus `stages/pilot/output/sizing.json`).

- **Bridge Qualification (`audit-only-v5`):** Verified `100%` parity across `4` handoff checks, `75` fixture tests (`+7` subtests), `7` heartbeat tests, `12` `H = 5,000` shakedown episodes (`6102` vs `Apex` across `frozen`/`scripted`/`mixed` on worlds `0–1`), `1` Watch `H = 5,000` episode (world `2`), and `1` Play `24`-frame episode (world `3`).
- **Strict Challenge Pilot (`pilot-v1`, `16` dev worlds + `16` paired pilot worlds per mix at `H = 5,000` on `promotion-v2-watch-rect`):**
  - **`frozen` mix:** Apex dev mean mass `52.61` (`10% MDE = 5.26`), pilot paired-difference `SD = 56.3` → required **`N = 1,072`** paired worlds (`> N_max = 120`). Pilot descriptive mean (`6102 - Apex`): **`-30.2`** mass/frame (`6102` won `5/16` worlds).
  - **`scripted` mix:** Apex dev mean mass `38.82` (`10% MDE = 3.88`), pilot paired-difference `SD = 118.3` → required **`N = 8,676`** paired worlds (`> N_max = 120`). Pilot descriptive mean (`6102 - Apex`): **`-66.4`** mass/frame (`6102` won `5/16` worlds).
  - **`mixed` mix:** Apex dev mean mass `72.65` (`10% MDE = 7.26`), pilot paired-difference `SD = 55.8` → required **`N = 552`** paired worlds (`> N_max = 120`). Pilot descriptive mean (`6102 - Apex`): **`-34.9`** mass/frame (`6102` won `5/16` worlds).
  - *(Audit note:* `pilot-v1` also recorded `INVALID_STOP_AUDIT_FAILED` because `audit.py` checked `lifecycle_frames` against `240,000` while Apex-only development episodes did not accumulate challenger lifecycle frames; under the pre-registered rules, both `N_req = 8,676 > 120` and the invalid-stop rule independently retired the study without confirmation runs.)

---

## What Was Learned & Why It Closed

1. **Unconstrained PQN continuation inflates boost usage at the expense of living mass:** Continuing `2026096102` from update `35,613` to `66,618` improved survival (`0.806 → 0.946` solo alive fraction) and food collection (`470.9 → 660.1`), but boosted on `55%–65%` of frames (up from `27%–36%`), burning body segments so aggressively that full-horizon living mass dropped by `-35.15` in solo and `-5.07` to `-7.30` in multi-snake profiles.
2. **Task-aligned PQN `raster31v3` (`6102`) remained substantially weaker than incumbent `vector61` Apex at `H = 5,000`:** Despite large gains over its own Study CZ parent at `H = 3,000`, `6102` trailed `champion_a5_freespace_20260621.pth` by `-30.2` (`frozen`), `-66.4` (`scripted`), and `-34.9` (`mixed`) mass/frame on the live `H = 5,000` `promotion-v2-watch-rect` arena.
3. **Live Python `H = 5,000` evaluation at a 10% MDE is statistically infeasible (`N = 552–8,676`):** Per-world paired standard deviations (`55.8–118.3` mass/frame) are `1–3×` the baseline mean, motivating the October redesign shift to cell-exact `GridBatchSim`, `ego2s-bv8` distillation (`B8`), and larger effect-size targets.

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
