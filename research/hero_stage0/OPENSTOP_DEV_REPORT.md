# In-Sample Dev Filter Report: `open-stop-h32-nb` (`K=1` and `K=2`) vs `bare` and `exact-stop-h48-nb`

- **Evaluation Type:** **IN-SAMPLE DEV FILTER ONLY (NOT A DECISIVE GATE)** — evaluated on the 48 worlds of `t34-hero-probe-20261103-v1` (`H=6,000`, `288,000` frames per arm) that were previously inspected in Stage 0b (`DEATH_AUDIT.json`) and Stage 1A (`STAGE1A_READOUT.json`) to diagnose `b8-s3` self-enclosure and design the `open-stop-h32-nb` certificate.
- **Date:** 2026-11-04
- **Author:** Argon (implementer)
- **Worktree / Branch:** `/Users/josenunez/Projects/ml/snake-dqn-hero-stage0` (`argon-hero-stage0-1103`)
- **Readout Receipt:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-openstop-dev/OPENSTOP_DEV_READOUT.json` (committed copy at `research/hero_stage0/OPENSTOP_DEV_READOUT.json`)
- **Explicit Pre-Registered Dev Filter Decision:**
  - **Explicit Verdict: `CONTINUE` (with `open-stop-h32-nb` `K=1` as the primary candidate for a fresh-world confirmation gate)** under the per-decision (`per-frame`) override rate definition (`0.1215% <= 0.5%` for `K=1`, `0.1493% <= 0.5%` for `K=2`), where both variants pass all three pre-registered dev-filter bars (`override_rate <= 0.5%`, `self-enclosure deaths removed >= 55%`, `L<600 deaths removed >= 70%`).
  - **Denominator Transparency Note (`per-decision` vs `per-root` override rate):** Because `open-stop-h32-nb` triggers every `interval = 5` alive decisions (`57,611` search roots across `288,000` decisions), the per-root override rate is `5x` the per-decision override rate (`0.6075%` per root for `K=1` and `0.7464%` per root for `K=2`, compared to `0.5953%` per root [`0.1191%` per decision] for Stage 1A `exact-stop-h48-nb`). If the `<= 0.5%` bar were interpreted per `interval=5` search root across all lengths rather than per decision, the verdict would be **`STOP`** (`350 / 57,611 = 0.6075% > 0.50%` by `62` overrides, although at `L<600` both `K=1` [`0.3557%`] and `K=2` [`0.3565%`] are `<= 0.50%` per root, and within each length band `K=1` has a *lower* per-root override rate than Stage 1A `exact-stop-h48-nb`). Both denominators are reported in full below.

---

## 1. Executive Summary & Pre-Registered Dev Filter Scorecard (IN-SAMPLE)

### 1.1 Pre-Registered Dev Filter Bars (48 Worlds, `H=6,000`, IN-SAMPLE)

Pre-registered rule: **continue to a fresh-world gate only if `override rate <= 0.5%` AND `self-enclosure deaths removed >= 55%` AND `L<600 deaths removed >= 70%` for at least one variant; otherwise `STOP`.**

| Pre-Registered Dev Filter Bar | Requirement | Stage 1A `exact-stop-h48-nb` (`K=2`, IN-SAMPLE ref) | `open-stop-h32-nb` (`K=1`, IN-SAMPLE) | `open-stop-h32-nb` (`K=2`, IN-SAMPLE) |
|---|---:|---:|---:|---:|
| **1a. Override rate (per decision / frame)** | `<= 0.50%` (`<= 1,440 / 288,000`) | `0.1191%` (`343 / 288,000`) — **PASS** | **`0.1215%` (`350 / 288,000`) — PASS** | **`0.1493%` (`430 / 288,000`) — PASS** |
| *1b. Override rate (per `interval=5` root, all `L`)* | *(if `<= 0.50%` per root: `<= 288 / 57,611`)* | *`0.5953%` (`343 / 57,618`)* | *`0.6075%` (`350 / 57,611`)* | *`0.7464%` (`430 / 57,611`)* |
| *1c. Override rate (per `interval=5` root, `L<600`)* | *(diagnostic)* | *`0.4035%` (`165 / 40,889`)* | *`0.3557%` (`120 / 33,740`)* | *`0.3565%` (`122 / 34,218`)* |
| **2. Self-enclosure deaths removed vs `bare`** | `>= 55.0%` (from `53` in `bare`) | `28.30%` (`53 -> 38`, `-15`) — **FAIL** | **`69.81%` (`53 -> 16`, `-37`) — PASS** | **`60.38%` (`53 -> 21`, `-32`) — PASS** |
| **3. `L<600` deaths removed vs `bare`** | `>= 70.0%` (from `45` in `bare`) | `64.44%` (`45 -> 16`, `-29`) — **FAIL** | **`84.44%` (`45 -> 7`, `-38`) — PASS** | **`86.67%` (`45 -> 6`, `-39`) — PASS** |
| **Overall Dev Filter Verdict (per-decision rate)** | All 3 bars pass | `FAIL` (`1/3`) | **`CONTINUE` (`3/3` PASS)** | **`CONTINUE` (`3/3` PASS)** |

### 1.2 Headline Performance Comparison Across All 4 Arms (IN-SAMPLE, 48 Worlds, `288,000` Frames/Arm)

| Metric (IN-SAMPLE, 48 paired worlds at `H=6,000`) | `bare` (`b8-s3`) | Stage 1A `exact-stop-h48-nb` (`K=2`) | `open-stop-h32-nb` (`K=1`) | `open-stop-h32-nb` (`K=2`) |
|---|---:|---:|---:|---:|
| **Mean mass (`mass_integral`)** | `371.27` (`±16.64` SE) | `446.80` (`±19.92` SE) | **`545.69` (`±21.03` SE)** | **`526.35` (`±18.13` SE)** |
| **Paired mass $\Delta$ vs `bare` (95% CI)** | — | `+75.52` (`[+33.36, +117.68]`) | **`+174.41` (`[+123.85, +224.97]`)** | **`+155.08` (`[+110.61, +199.54]`)** |
| **Relative mass gain vs `bare` (Fieller 95% CI)** | — | `+20.34%` (`[+8.54%, +33.56%]`) | **`+46.98%` (`[+31.59%, +64.51%]`)** | **`+41.77%` (`[+28.13%, +57.38%]`)** |
| **Paired mass $\Delta$ vs `exact-stop-h48-nb` (95% CI)** | `-75.52` | — | **`+98.89` (`[+52.29, +145.49]`)** | **`+79.55` (`[+31.43, +127.68]`)** |
| **Total deaths (`ALL`)** | `78` (`0.2708 / 1k`) | `40` (`-48.72%`) | **`17` (`-78.21%` vs `bare`, `-57.50%` vs `h48`)** | **`23` (`-70.51%` vs `bare`, `-42.50%` vs `h48`)** |
| **`L<600` deaths** | `45` (`0.1562 / 1k`) | `16` (`-64.44%`) | **`7` (`-84.44%` vs `bare`, `-56.25%` vs `h48`)** | **`6` (`-86.67%` vs `bare`, `-62.50%` vs `h48`)** |
| **`L>=600` deaths** | `33` (`0.1146 / 1k`) | `24` (`-27.27%`) | **`10` (`-69.70%` vs `bare`, `-58.33%` vs `h48`)** | **`17` (`-48.48%` vs `bare`, `-29.17%` vs `h48`)** |
| **Self-enclosure deaths (`ALL`)** | `53` | `38` (`-28.30%`) | **`16` (`-69.81%` vs `bare`, `-57.89%` vs `h48`)** | **`21` (`-60.38%` vs `bare`, `-44.74%` vs `h48`)** |
| **Opponent collision deaths (`ALL`)** | `25` (`4` head-on, `21` body) | `2` (`0` head-on, `2` body) | **`1` (`0` head-on, `1` body; `-96.00%`)** | **`2` (`0` head-on, `2` body; `-92.00%`)** |
| **Alive frames at `L>=600` (share of `288,000`)** | `59,036` (`20.50%`) | `83,701` (`29.06%`) | **`119,451` (`41.48%`, `2.02x` `bare`)** | **`117,052` (`40.64%`, `1.98x` `bare`)** |
| **Pooled boost share ($\Delta$ vs `bare`)** | `0.1951%` (`562`) | `0.2278%` (`+0.0326 pp`) | **`0.1958%` (`564`, `+0.0007 pp`)** | `0.2826%` (`814`, `+0.0875 pp`) |
| **Harmful overrides (`H=32` raw cond. majority)** | — | `0.5831%` (`2/343` at `H=48`) | **`0.5714%` (`2/350`)** | **`0.6977%` (`3/430`)** |
| **Harmful overrides (`H=48` raw cond. majority)** | — | `0.5831%` (`2/343`) | **`1.1429%` (`4/350`; `0/120` at `L<600`)** | `1.8605%` (`8/430`; `1/122` at `L<600`) |
| **8-worker selection `ms / root` (`s / world`)** | `0.00 ms` (`0.0s`) | `261.79 ms` (`314.24s`) | **`103.97 ms` (`124.78s`, `2.52x` faster)** | `211.99 ms` (`254.44s`, `1.23x` faster) |
| **1-worker uncontended `ms / root` (`s / world`, W0)** | `0.00 ms` (`0.0s`) | `147.72 ms` (`177.26s`) | **`60.44 ms` (`72.52s`, `2.44x` faster)** | `128.09 ms` (`153.70s`, `1.15x` faster) |

---

## 2. Controller Specification & `R >= L` Formula Verification

### 2.1 `open-stop-h32-nb` Mechanism (`research/hero_stage0/open_stop.py`)

`open-stop-h32-nb` modifies `exact-stop-h48-nb` (`research/hero_stage0/stage1a.py`) as follows:

1. **Trigger & Horizon:** Fires at `interval = 5` alive decisions for all lengths `L >= 1` (`DenseTrigger(min_length=1, interval=5)`), rolling out candidate branches for `H = 32` steps with full Watch6 opponent stepping (`1` `b8-s3` hero + `5` `frp3-s12 + v8` opponents under `branch_scope("mixed")` and compiled `configure("kernel", stats)`).
2. **Rollout Certificate (`alive_and_open`):** A branch rollout under suffix $s$ is certified (`survives = True`) iff:
   - The hero is alive on **every step `1..32`** (`raw_survives = True`), **AND**
   - At terminal step `H = 32`, the hero state is `un_enclosed`, defined as:
     $$\text{un\_enclosed}(s_{32}) \iff \Big(\text{v8 best\_safe\_area}(s_{32}) \ge \min(\max(4, L_{32}), 160)\Big) \;\wedge\; \Big(L_{32} < 160 \;\lor\; \max_{a \in \text{valid}(s_{32})} \text{region\_frac}(s_{32}, a) \ge 0.25\Big)$$
3. **Verify-First Action Ordering (`evaluation_order_nb`):**
   - Evaluates `bare = argmax_{a in legal} Q(s, a)` first across the selection suffix(es) (`K=1`: `(0,)`; `K=2`: `(0, 1)`). If `bare` is certified on all selection suffixes, stops immediately and returns `bare` with 1 branch evaluated.
   - If `bare` fails the certificate on any selection suffix, evaluates remaining legal **non-boost** actions (`0, 1, 2`) in descending `Q(s, a)` order, stopping at the first action certified on all selection suffixes.
   - Only if all legal non-boost actions fail, evaluates remaining legal **boost** actions (`3, 4, 5`) in descending `Q(s, a)` order.
4. **No-Futile-Override Rule When All Actions Fail Certificate (`all_die_roots`):**
   - Each branch rollout records `frames_survived = 32` if `raw_survives` is true (`alive` on steps `1..32`, even if enclosed at step `32`), else `frames_executed - 1` (`0..31`).
   - If no legal action satisfies the `alive_and_open` certificate (`all_die_roots`), `ceiling._choose` selects by `max(frames_survived)` with tie-break in `bare`-first order. Consequently, if `bare` already survives `32` raw steps (`frames_survived = 32`) inside a sealed pocket where no action is `un_enclosed`, `open-stop-h32-nb` **keeps `bare`** rather than making a futile override inside the sealed pocket.

### 2.2 Verification of `region_frac >= 0.25` $\iff$ `R >= L` in `src/simd_env/ego_raster_b.py`

In `src/simd_env/ego_raster_b.py` (lines 240–248):
- `reg = region_sizes(view, rows, backend=backend)` calls the numba flood-fill kernel `ego_raster_nb.region_sizes_into` (`src/simd_env/ego_raster_nb.py`, lines 302–373), which computes for each of the 3 directional candidate head cells `(left, straight, right)` the exact integer 4-connected free-space component size $R \in \{0, \dots, W \times H\}$ on the full $84 \times 84$ arena grid (treating cells with `ttl > 1` as blocked and `ttl <= 1` as vacating/free).
- Line 246 of `src/simd_env/ego_raster_b.py` computes:
  ```python
  denom = np.maximum(lengths, 1.0)
  out[:, 11:14] = np.minimum(reg / denom[:, None], 4.0) / 4.0
  ```
  (which appears at scalar indices `23:26` in `ego2s-bv8` via `src/simd_env/ego_raster_bv8.py`).
- Because `min(R / L, 4.0) / 4.0` is strictly monotonic in $R/L$ on $[0, 4]$ and equals $0.25$ at $R/L = 1.0$ (where `float32(0.25)` and `float64(0.25)` are exact powers of two in IEEE-754), `region_frac >= 0.25` is mathematically and bit-identically equivalent to the integer test `R >= L`.
- `evaluate_terminal_openness(sim)` (`research/hero_stage0/open_stop.py`, lines 145–151) computes both `max_pocket_r >= length` (`R >= L`) and `max_region_frac >= 0.25` over all advisory-valid actions (`advisory[a] == True`, mapping action $a \in \{0..5\}$ to direction $a \bmod 3$) and raises a `RuntimeError` if they ever disagree. Across all `188,960` selection rollouts and `6,240` scoring rollouts in this 48-world evaluation, `R >= L` and `region_frac >= 0.25` agreed `100.00%` of the time (`0` discrepancies).

### 2.3 Verification of Reused Stage 1A `bare` and `exact-stop-h48-nb` Arms

All 48 cached world files in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a/worlds/stage1a-world-00.json`..`47.json` were verified before reuse:
- All 48 worlds match `allocation("t34-hero-probe-20261103-v1", 48)` (`seeds_sha256 = 290bd1fc9be68d5f72db781b6f390e68f401a08415241432d910ba68fd21f7b7`, `horizon = 6000`, `mirror_identity_frames = 6000`).
- Re-running `run_stage1a_arm(..., "bare", horizon=6000, audit_live_mirror=False)` on World 0 reproduced `mass_sum = 1881614`, `deaths = 1`, `death_lengths = [644]`, `boost_decisions = 2` bit-for-bit.
- Furthermore, 200-frame live-mirror parity checks (`audit_live_mirror=True` vs `False`) on World 0 for both `open-stop-h32-nb-k1` and `open-stop-h32-nb-k2` confirmed `200/200` mirror identity frames and bit-identical `mass_sum`, `deaths`, and `interventions`.

---

## 3. Deaths by Length Band and Cause (IN-SAMPLE, 48 Worlds, `H=6,000`)

### 3.1 Deaths by Primary Length Regime (`L<600`, `L>=600`, `ALL`) and Cause

| Arm (IN-SAMPLE, 48 worlds) | Regime | Total Deaths | Deaths / 1k Frames (95% CI) | Self-Body Enclosure | Opponent Head-On | Opponent Body | Opponent Total | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`bare` (`b8-s3`)** | `L<600` | `45` | `0.1562` (`[0.1040, 0.2085]`) | `27` (`60.0%`) | `4` (`8.9%`) | `14` (`31.1%`) | `18` (`40.0%`) | `0` |
| | `L>=600` | `33` | `0.1146` (`[0.0919, 0.1373]`) | `26` (`78.8%`) | `0` (`0.0%`) | `7` (`21.2%`) | `7` (`21.2%`) | `0` |
| | **`ALL`** | **`78`** | **`0.2708` (`[0.2266, 0.3151]`)** | **`53` (`67.9%`)** | **`4` (`5.1%`)** | **`21` (`26.9%`)** | **`25` (`32.1%`)** | **`0`** |
| **Stage 1A `exact-stop-h48-nb` (`K=2`)** | `L<600` | `16` (`-64.44%`) | `0.0556` (`[0.0267, 0.0844]`) | `15` (`93.8%`) | `0` (`0.0%`) | `1` (`6.2%`) | `1` (`6.2%`) | `0` |
| | `L>=600` | `24` (`-27.27%`) | `0.0833` (`[0.0589, 0.1078]`) | `23` (`95.8%`) | `0` (`0.0%`) | `1` (`4.2%`) | `1` (`4.2%`) | `0` |
| | **`ALL`** | **`40` (`-48.72%`)** | **`0.1389` (`[0.1084, 0.1694]`)** | **`38` (`95.0%`)** | **`0` (`0.0%`)** | **`2` (`5.0%`)** | **`2` (`5.0%`)** | **`0`** |
| **`open-stop-h32-nb` (`K=1`)** | `L<600` | **`7` (`-84.44%`)** | **`0.0243` (`[0.0044, 0.0442]`)** | **`7` (`100.0%`)** | **`0` (`0.0%`)** | **`0` (`0.0%`)** | **`0` (`0.0%`)** | `0` |
| | `L>=600` | **`10` (`-69.70%`)** | **`0.0347` (`[0.0149, 0.0546]`)** | **`9` (`90.0%`)** | **`0` (`0.0%`)** | **`1` (`10.0%`)** | **`1` (`10.0%`)** | `0` |
| | **`ALL`** | **`17` (`-78.21%`)** | **`0.0590` (`[0.0317, 0.0863]`)** | **`16` (`94.1%`)** | **`0` (`0.0%`)** | **`1` (`5.9%`)** | **`1` (`5.9%`)** | **`0`** |
| **`open-stop-h32-nb` (`K=2`)** | `L<600` | **`6` (`-86.67%`)** | **`0.0208` (`[0.0047, 0.0370]`)** | **`5` (`83.3%`)** | **`0` (`0.0%`)** | **`1` (`16.7%`)** | **`1` (`16.7%`)** | `0` |
| | `L>=600` | **`17` (`-48.48%`)** | **`0.0590` (`[0.0356, 0.0824]`)** | **`16` (`94.1%`)** | **`0` (`0.0%`)** | **`1` (`5.9%`)** | **`1` (`5.9%`)** | `0` |
| | **`ALL`** | **`23` (`-70.51%`)** | **`0.0799` (`[0.0516, 0.1081]`)** | **`21` (`91.3%`)** | **`0` (`0.0%`)** | **`2` (`8.7%`)** | **`2` (`8.7%`)** | **`0`** |

### 3.2 Fine-Grained Sub-Band Breakdown (`L<200`, `200<=L<600`, `600<=L<800`, `L>=800`)

| Length Sub-Band | Cause Class | `bare` | Stage 1A `exact-stop-h48-nb` | `open-stop-h32-nb` (`K=1`) | `open-stop-h32-nb` (`K=2`) | `K=1` vs `bare` | `K=1` vs `h48-nb` | `K=2` vs `bare` | `K=2` vs `h48-nb` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`L<200`** | Total | `17` | `1` | **`1`** | **`1`** | `-16` (`-94.1%`) | `0` (`0.0%`) | `-16` (`-94.1%`) | `0` (`0.0%`) |
| | Self-body enclosure | `8` | `1` | **`1`** | **`0`** | `-7` (`-87.5%`) | `0` | `-8` (`-100.0%`) | `-1` |
| | Opponent collision (`head-on` + `body`) | `9` (`3+6`) | `0` | **`0`** | **`1` (`0+1`)** | `-9` (`-100.0%`) | `0` | `-8` (`-88.9%`) | `+1` |
| **`200<=L<600`** | Total | `28` | `15` | **`6`** | **`5`** | `-22` (`-78.6%`) | `-9` (`-60.0%`) | `-23` (`-82.1%`) | `-10` (`-66.7%`) |
| | Self-body enclosure | `19` | `14` | **`6`** | **`5`** | `-13` (`-68.4%`) | `-8` (`-57.1%`) | `-14` (`-73.7%`) | `-9` (`-64.3%`) |
| | Opponent collision (`head-on` + `body`) | `9` (`1+8`) | `1` (`0+1`) | **`0`** | **`0`** | `-9` (`-100.0%`) | `-1` (`-100.0%`) | `-9` (`-100.0%`) | `-1` (`-100.0%`) |
| **`600<=L<800`** | Total | `12` | `9` | **`4`** | **`5`** | `-8` (`-66.7%`) | `-5` (`-55.6%`) | `-7` (`-58.3%`) | `-4` (`-44.4%`) |
| | Self-body enclosure | `9` | `9` | **`3`** | **`4`** | `-6` (`-66.7%`) | `-6` (`-66.7%`) | `-5` (`-55.6%`) | `-5` (`-55.6%`) |
| | Opponent collision (`head-on` + `body`) | `3` (`0+3`) | `0` | **`1` (`0+1`)** | **`1` (`0+1`)** | `-2` (`-66.7%`) | `+1` | `-2` (`-66.7%`) | `+1` |
| **`L>=800`** | Total | `21` | `15` | **`6`** | **`12`** | `-15` (`-71.4%`) | `-9` (`-60.0%`) | `-9` (`-42.9%`) | `-3` (`-20.0%`) |
| | Self-body enclosure | `17` | `14` | **`6`** | **`12`** | `-11` (`-64.7%`) | `-8` (`-57.1%`) | `-5` (`-29.4%`) | `-2` (`-14.3%`) |
| | Opponent collision (`head-on` + `body`) | `4` (`0+4`) | `1` (`0+1`) | **`0`** | **`0`** | `-4` (`-100.0%`) | `-1` (`-100.0%`) | `-4` (`-100.0%`) | `-1` (`-100.0%`) |

### 3.3 Exposure-Adjusted Death Rates (Per 1,000 Alive Frames in Each Length Band)

Because `open-stop-h32-nb` keeps the hero alive at `L<600` (`84.4%`–`86.7%` fewer `L<600` deaths), the hero spends **twice as many frames at `L>=600`** (`119,451` alive frames for `K=1` and `117,052` for `K=2`, vs `59,036` for `bare` and `83,701` for `exact-stop-h48-nb`). Normalizing by alive frames in each band shows an even larger reduction in hazard rate:

- **`L<600` deaths per 1k `L<600` alive frames:**
  - `bare`: `0.1966` (`45 / 228,886`)
  - `exact-stop-h48-nb`: `0.0783` (`16 / 204,259`, `-60.16%` vs `bare`)
  - `open-stop-h32-nb (K=1)`: **`0.0415` (`7 / 168,532`, `-78.87%` vs `bare`, `-46.98%` vs `h48-nb`)**
  - `open-stop-h32-nb (K=2)`: **`0.0351` (`6 / 170,925`, `-82.15%` vs `bare`, `-55.19%` vs `h48-nb`)**
- **`L>=600` deaths per 1k `L>=600` alive frames:**
  - `bare`: `0.5590` (`33 / 59,036`)
  - `exact-stop-h48-nb`: `0.2867` (`24 / 83,701`, `-48.71%` vs `bare`)
  - `open-stop-h32-nb (K=1)`: **`0.0837` (`10 / 119,451`, `-85.02%` vs `bare`, `-70.80%` vs `h48-nb`)**
  - `open-stop-h32-nb (K=2)`: **`0.1452` (`17 / 117,052`, `-74.02%` vs `bare`, `-49.35%` vs `h48-nb`)**

### 3.4 Why `exact-stop-h48-nb` Failed on `200<=L<600` Self-Enclosure and Why `open-stop-h32-nb` Fixed It

In Stage 1A (`exact-stop-h48-nb`), `14` of the `16` remaining `L<600` deaths were `200<=L<600` self-body enclosures (`19 -> 14`, only `-26.3%` removed) because a snake at `L in [200, 600)` that enters a pocket of `80..250` cells survives all `H=48` steps inside the pocket (`survives = True` under raw survival) and then dies of self-enclosure at step `55..180`. With the terminal openness certificate (`v8 best_safe_area >= min(max(4, L), 160)` and `R >= L` for `L >= 160`) checked at `H = 32`, `open-stop-h32-nb` rejects pocket-entry branches as soon as the pocket seals within `32` steps, cutting `200<=L<600` self-enclosure deaths from `19` (`bare`) / `14` (`h48-nb`) down to **`6` (`K=1`, `-68.4%` vs `bare`)** and **`5` (`K=2`, `-73.7%` vs `bare`)**, while simultaneously cutting `L>=600` self-enclosure deaths from `26` (`bare`) / `23` (`h48-nb`) down to **`9` (`K=1`, `-65.4%` vs `bare`)**.

---

## 4. Override Rate, Root Certificate Breakdown, Boost Share, and Independent-Suffix Harmful-Override Scoring (IN-SAMPLE)

### 4.1 Search Root Certificate Breakdown & Override Rate

| Search / Override Metric (IN-SAMPLE, 48 worlds) | Stage 1A `exact-stop-h48-nb` (`K=2`) | `open-stop-h32-nb` (`K=1`) | `open-stop-h32-nb` (`K=2`) |
|---|---:|---:|---:|
| Total decisions (`H=6,000 x 48`) | `288,000` | `288,000` | `288,000` |
| Decisions at `L<600` / `L>=600` | `204,320` / `83,680` | `168,584` / `119,416` | `170,976` / `117,024` |
| Total search roots (`interval=5`) | `57,618` | `57,611` | `57,611` |
| Search roots at `L<600` / `L>=600` | `40,889` / `16,729` | `33,740` / `23,871` | `34,218` / `23,393` |
| Roots where `bare` is certified (`alive + un-enclosed`) | `56,914` (`98.78%` raw alive) | `55,889` (`97.01%`) | `55,413` (`96.18%`) |
| Roots where `bare` survives `1..32` raw (`bare_raw_survived_roots`) | — | `57,435` (`99.69%`) | `57,354` (`99.55%`) |
| Roots where `bare` fails **only by terminal enclosure at `H=32`** | — | `1,546` (`2.68%`) | `1,941` (`3.37%`) |
| Roots where `bare` fails **by raw death within `1..32`** | `704` (`1.22%` within `1..48`) | `176` (`0.31%`) | `257` (`0.45%`) |
| Roots where **no** action is certified (`all_die_roots`) | `478` (`0.83%`) | `1,418` (`2.46%`) | `1,832` (`3.18%`) |
| **Total overrides (`chosen != bare`)** | **`343`** | **`350`** | **`430`** |
| Overrides at `L<600` / `L>=600` | `165` / `178` | `120` / `230` | `122` / `308` |
| Overrides triggered by `bare` terminal enclosure only | `0` | `268` (`76.57%` of overrides) | `312` (`72.56%` of overrides) |
| Overrides triggered by `bare` raw death within `1..32` | `343` (within `1..48`) | `82` (`23.43%` of overrides) | `118` (`27.44%` of overrides) |
| Overrides to non-boost (`0..2`) / to boost (`3..5`) | `273` (`79.6%`) / `70` (`20.4%`) | `268` (`76.6%`) / `82` (`23.4%`) | `312` (`72.6%`) / `118` (`27.4%`) |
| **Override rate per decision (overall)** | **`0.1191%`** | **`0.1215%`** | **`0.1493%`** |
| Override rate per decision (`L<600` / `L>=600`) | `0.0808%` / `0.2127%` | `0.0712%` / `0.1926%` | `0.0714%` / `0.2632%` |
| **Override rate per search root (overall)** | **`0.5953%`** | **`0.6075%`** | **`0.7464%`** |
| Override rate per search root (`L<600` / `L>=600`) | `0.4035%` / `1.0640%` | **`0.3557%` / `0.9635%`** | `0.3565%` / `1.3166%` |

**Key structural observations on override rate:**
1. **No futile overrides inside already-enclosed states:** In `open-stop-h32-nb (K=1)`, `bare` fails the terminal openness check (while surviving `32` steps raw) on `1,546` roots, but the controller only overrides on `268` (`17.3%`) of those roots — namely the roots where a genuinely `un_enclosed` alternative exists! On the remaining `1,278` enclosed roots where *all* actions remain enclosed at `H=32`, `frames_survived = 32` for `bare` ensures `ceiling._choose` keeps `bare` (`0` futile overrides).
2. **Simpson's paradox vs Stage 1A on per-root override rate:** In *both* length bands (`L<600` and `L>=600`), `open-stop-h32-nb (K=1)` has a **lower** per-root override rate than Stage 1A `exact-stop-h48-nb` (`0.3557%` vs `0.4035%` at `L<600`, and `0.9635%` vs `1.0640%` at `L>=600`). Its overall per-root override rate (`0.6075%` vs `0.5953%`, a difference of `+7` overrides over `288,000` frames) is slightly higher only because `K=1` survives into `L>=600` so much more often (`23,871` roots at `L>=600` vs `16,729` in Stage 1A).

### 4.2 Boost Share Delta vs `bare`

| Arm (IN-SAMPLE, 48 worlds) | Total Boost Decisions | Pooled Boost Share (`ALL`) | Pooled Boost Share (`L<600`) | Pooled Boost Share (`L>=600`) | Pooled $\Delta$ vs `bare` (pp) | Paired World Mean $\Delta$ vs `bare` (95% CI, pp) |
|---|---:|---:|---:|---:|---:|---:|
| `bare` (`b8-s3`) | `562 / 288,000` | `0.1951%` | `0.0795%` (`182 / 228,976`) | `0.6438%` (`380 / 59,024`) | — | — |
| Stage 1A `exact-stop-h48-nb` (`K=2`) | `656 / 288,000` | `0.2278%` | `0.0984%` (`201 / 204,320`) | `0.5437%` (`455 / 83,680`) | `+0.0326 pp` | `+0.0326 pp` (`[-0.0332, +0.0985]`) |
| **`open-stop-h32-nb` (`K=1`)** | **`564 / 288,000`** | **`0.1958%`** | **`0.0801%` (`135 / 168,584`)** | **`0.3592%` (`429 / 119,416`)** | **`+0.0007 pp`** | **`+0.0007 pp` (`[-0.0715, +0.0729]`)** |
| **`open-stop-h32-nb` (`K=2`)** | `814 / 288,000` | `0.2826%` | `0.0719%` (`123 / 170,976`) | `0.5905%` (`691 / 117,024`) | `+0.0875 pp` | `+0.0875 pp` (`[+0.0022, +0.1728]`) |

### 4.3 Independent-Suffix Harmful-Override Scoring (`OPENSTOP_SCORING_SUFFIXES = (4, 5, 6, 7)`)

Every override (`350` in `K=1`, `430` in `K=2`) was scored on 4 independent opponent-tiebreak RNG suffixes `(4, 5, 6, 7)` disjoint from the selection suffixes `(0, 1)`. We report three evaluation modes on those 4 suffixes:
1. **`H=32` raw survival** (whether `chosen` dies within `32` steps where `bare` survives `32` steps),
2. **`H=48` raw survival** (Stage 1A's `48`-step horizon, rolling the same 4 independent suffixes out to `H=48`), and
3. **`H=32` certified survival** (whether `chosen` fails `alive_and_open` at `H=32` where `bare` passes `alive_and_open` at `H=32`).

| Variant & Scoring Mode (4 independent suffixes `4..7`) | Regime | Overrides Scored | Mean `S_score(bare)` | Mean `S_score(chosen)` | Harmful Cond. Majority (`n_diff > n_bare/2`) | Harmful Strict Majority (`n_diff >= 3/4`) | Harmful `>= 2/4` (`n_diff >= 2`) | Harmful `>= 1/4` (`n_diff >= 1`) | `n_diff` Histogram (`0 / 1 / 2 / 3 / 4`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Stage 1A `exact-stop-h48-nb` (`H=48` raw)** | `ALL` | `343` | `0.6254` | `0.9424` | `2` (`0.58%`) | `1` (`0.29%`) | `2` (`0.58%`) | `4` (`1.17%`) | `339 / 2 / 1 / 0 / 1` |
| **`open-stop-h32-nb (K=1)` — `H=32` raw** | **`ALL`** | **`350`** | **`0.8264`** | **`0.9086`** | **`2` (`0.57%`)** | **`1` (`0.29%`)** | **`3` (`0.86%`)** | **`8` (`2.29%`)** | **`342 / 5 / 2 / 1 / 0`** |
| | `L<600` | `120` | `0.7771` | `0.8979` | **`0` (`0.00%`)** | **`0` (`0.00%`)** | **`0` (`0.00%`)** | `1` (`0.83%`) | `119 / 1 / 0 / 0 / 0` |
| | `L>=600` | `230` | `0.8522` | `0.9141` | `2` (`0.87%`) | `1` (`0.43%`) | `3` (`1.30%`) | `7` (`3.04%`) | `223 / 4 / 2 / 1 / 0` |
| **`open-stop-h32-nb (K=1)` — `H=48` raw** | `ALL` | `350` | `0.7571` | `0.8521` | `4` (`1.14%`) | `2` (`0.57%`) | `8` (`2.29%`) | `22` (`6.29%`) | `328 / 14 / 6 / 2 / 0` |
| | `L<600` | `120` | `0.7000` | `0.8375` | **`0` (`0.00%`)** | **`0` (`0.00%`)** | `2` (`1.67%`) | `5` (`4.17%`) | `115 / 3 / 2 / 0 / 0` |
| | `L>=600` | `230` | `0.7870` | `0.8598` | `4` (`1.74%`) | `2` (`0.87%`) | `6` (`2.61%`) | `17` (`7.39%`) | `213 / 11 / 4 / 2 / 0` |
| **`open-stop-h32-nb (K=1)` — `H=32` certified** | `ALL` | `350` | `0.3600` | `0.7386` | `23` (`6.57%`) | `7` (`2.00%`) | `16` (`4.57%`) | `40` (`11.43%`) | `310 / 24 / 9 / 1 / 6` |
| | `L<600` | `120` | `0.3833` | `0.7708` | `5` (`4.17%`) | `0` (`0.00%`) | `1` (`0.83%`) | `9` (`7.50%`) | `111 / 8 / 1 / 0 / 0` |
| | `L>=600` | `230` | `0.3478` | `0.7217` | `18` (`7.83%`) | `7` (`3.04%`) | `15` (`6.52%`) | `31` (`13.48%`) | `199 / 16 / 8 / 1 / 6` |
| **`open-stop-h32-nb (K=2)` — `H=32` raw** | **`ALL`** | **`430`** | **`0.8070`** | **`0.9169`** | **`3` (`0.70%`)** | **`2` (`0.47%`)** | **`3` (`0.70%`)** | **`6` (`1.40%`)** | **`424 / 3 / 1 / 0 / 2`** |
| | `L<600` | `122` | `0.7910` | `0.9160` | `1` (`0.82%`) | `1` (`0.82%`) | `1` (`0.82%`) | `1` (`0.82%`) | `121 / 0 / 0 / 0 / 1` |
| | `L>=600` | `308` | `0.8133` | `0.9172` | `2` (`0.65%`) | `1` (`0.32%`) | `2` (`0.65%`) | `5` (`1.62%`) | `303 / 3 / 1 / 0 / 1` |
| **`open-stop-h32-nb (K=2)` — `H=48` raw** | `ALL` | `430` | `0.7459` | `0.8547` | `8` (`1.86%`) | `4` (`0.93%`) | `8` (`1.86%`) | `19` (`4.42%`) | `411 / 11 / 4 / 3 / 1` |
| | `L<600` | `122` | `0.7049` | `0.8525` | `1` (`0.82%`) | `1` (`0.82%`) | `4` (`3.28%`) | `6` (`4.92%`) | `116 / 2 / 3 / 1 / 0` |
| | `L>=600` | `308` | `0.7622` | `0.8555` | `7` (`2.27%`) | `3` (`0.97%`) | `4` (`1.30%`) | `13` (`4.22%`) | `295 / 9 / 1 / 2 / 1` |
| **`open-stop-h32-nb (K=2)` — `H=32` certified** | `ALL` | `430` | `0.4169` | `0.7733` | `19` (`4.42%`) | `3` (`0.70%`) | `10` (`2.33%`) | `40` (`9.30%`) | `390 / 30 / 7 / 3 / 0` |
| | `L<600` | `122` | `0.4529` | `0.7561` | `7` (`5.74%`) | `2` (`1.64%`) | `3` (`2.46%`) | `12` (`9.84%`) | `110 / 9 / 1 / 2 / 0` |
| | `L>=600` | `308` | `0.4026` | `0.7800` | `12` (`3.90%`) | `1` (`0.32%`) | `7` (`2.27%`) | `28` (`9.09%`) | `280 / 21 / 6 / 1 / 0` |

Note why `mean S_score(bare)` on `H=32` raw survival is `0.8264` (`K=1`) / `0.8070` (`K=2`) on override frames: `76.6%` (`268/350` in `K=1`) of overrides occur when `bare` **survives all `32` steps raw** (`S_score_raw(bare) = 1.0`) but enters a **sealed pocket** at step `32` (`S_score_certified(bare) = 0.3600`, which drops to `0.7571` raw survival by step `48` as the pocket begins to run out of space). Even on raw survival, `chosen` improves `32`-step survival from `0.8264 -> 0.9086` and `48`-step survival from `0.7571 -> 0.8521`, while doubling certified open survival at step `32` from `0.3600 -> 0.7386`.

---

## 5. Why `K=1` Outperforms `K=2` under the Terminal Openness Certificate (IN-SAMPLE)

A critical empirical finding of this dev screen is that **`open-stop-h32-nb (K=1)` strictly dominates `open-stop-h32-nb (K=2)` on almost every metric**:
- **Fewer total deaths:** `17` (`K=1`) vs `23` (`K=2`), driven by `L>=600` deaths (`10` in `K=1` vs `17` in `K=2`, and `6` vs `12` at `L>=800`).
- **More self-enclosure deaths removed:** `69.81%` (`53 -> 16` in `K=1`) vs `60.38%` (`53 -> 21` in `K=2`).
- **Higher mean mass:** `545.69` (`+46.98%` vs `bare`) in `K=1` vs `526.35` (`+41.77%` vs `bare`) in `K=2`.
- **Fewer overrides:** `350` (`0.1215%` per decision, `0.6075%` per root) in `K=1` vs `430` (`0.1493%` per decision, `0.7464%` per root) in `K=2`.
- **Zero boost inflation:** `+0.0007 pp` (`564` vs `562` boost decisions) in `K=1` vs `+0.0875 pp` (`814` vs `562` boost decisions) in `K=2`.
- **Half the compute cost:** `103.97 ms/root` (`124.78s/world` on 8 workers; `60.44 ms/root`, `72.52s/world` on 1 worker) in `K=1` vs `211.99 ms/root` (`254.44s/world` on 8 workers; `128.09 ms/root`, `153.70s/world` on 1 worker) in `K=2`.

**Why does `K=2` degrade `L>=600` performance relative to `K=1`?**
Because `K=2` requires `bare` (and each non-boost alternative) to satisfy the terminal openness certificate (`R >= L` at step `32`) across **both** opponent RNG suffixes `(0, 1)` simultaneously (`AND` conjunction):
1. At `L >= 600` (where $R \ge L$ requires a connected component of `600..1,200+` free cells at step `32`), an opponent snake's random tie-break on suffix `1` occasionally pinches a corridor at step `32`, causing `bare` to fail `un_enclosed` on `1,941` roots in `K=2` vs `1,546` roots in `K=1` (`+395` extra root failures).
2. Moreover, when `bare` fails, a safe non-boost alternative that is certified open on suffix `0` may also be rejected on suffix `1`, forcing `K=2` to fall through to **boost** alternatives (`118` overrides to boost in `K=2` vs `82` in `K=1`, raising `L>=600` boost decisions from `429` in `K=1` to `691` in `K=2`).
3. Thus `K=1` avoids false-positive enclosure rejections from taking the worst of two 32-step rollouts while halving the per-root rollout cost.

---

## 6. Compute Cost & Latency Profile (`ms` per Root and `s` per World)

| Timing Metric (`H=6,000` per world, `1,200` roots/world) | `bare` (`b8-s3`) | Stage 1A `exact-stop-h48-nb` (`K=2`, `H=48`) | `open-stop-h32-nb` (`K=1`, `H=32`) | `open-stop-h32-nb` (`K=2`, `H=32`) |
|---|---:|---:|---:|---:|
| **Mean branches evaluated / root** | `0.000` | `1.0250` | **`1.0807`** | `1.1036` |
| **Mean selection rollouts / root** | `0.000` | `2.0460` | **`1.0807`** | `2.1992` |
| **Total selection env frames (48 worlds)** | `0` | `5,582,592` | **`1,981,622` (`-64.5%` vs `h48`)** | `4,022,510` (`-27.9%` vs `h48`) |
| **8-worker selection `ms / root` (mean across 48 worlds)** | `0.00 ms` | `261.79 ms` | **`103.97 ms` (`2.52x` faster)** | `211.99 ms` (`1.23x` faster) |
| **8-worker selection `ms / session frame`** | `0.00 ms` | `52.37 ms` | **`20.80 ms`** | `42.41 ms` |
| **8-worker selection wall `s / world` (mean ± SD)** | `0.00s` | `314.24s ± 5.13s` | **`124.78s ± 11.92s`** | `254.44s ± 21.00s` |
| **8-worker selection wall `s / world` (median / p95 / max)** | `0.00s` | `314.13s / 322.19s / 325.52s` | **`119.56s / 150.36s / 156.80s`** | `251.16s / 292.35s / 301.48s` |
| **8-worker total session wall `s / world` (mean)** | `140.65s` (with live mirror) | `477.52s` (with live mirror) | **`192.59s` (grid sim + scoring)** | `324.74s` (grid sim + scoring) |
| **1-worker uncontended `ms / root` (World 0, `H=6,000`)** | `0.00 ms` | `147.72 ms` | **`60.44 ms` (`2.44x` faster)** | `128.09 ms` (`1.15x` faster) |
| **1-worker uncontended `ms / session frame` (World 0)** | `0.00 ms` | `29.54 ms` | **`12.09 ms` (< `41.7 ms` 24 FPS budget)** | `25.62 ms` (< `41.7 ms` 24 FPS budget) |
| **1-worker uncontended selection wall `s / world` (World 0)** | `0.00s` | `177.26s` | **`72.52s`** | `153.70s` |

---

## 7. Explicit Dev Filter Decision

- **Verdict:** **`CONTINUE`** (proceed to a fresh-world out-of-sample confirmation gate with **`open-stop-h32-nb` (`K=1`)**, which passes all three pre-registered dev-filter bars under the per-decision override rate: `override_rate = 0.1215% <= 0.50%`, `self_enclosure_deaths_removed = 69.81% >= 55.0%`, `L<600_deaths_removed = 84.44% >= 70.0%`).
- **Caveat on override-rate denominator:** If the owner's `<= 0.5%` override-rate bar was intended per `interval=5` search root (`57,611` roots) rather than per decision (`288,000` decisions), `open-stop-h32-nb (K=1)` is at `0.6075%` per root overall (`0.3557%` per root at `L<600`, `0.9635%` per root at `L>=600`), which would trigger `STOP` unless the per-decision denominator (`0.1215%`) is ratified before fresh-world gate dispatch.
