# T34 Hero Stage 1A — Teacher Ceiling Evaluation (`exact-stop-h48-nb`)

- **Date:** 2026-11-04
- **Implementer:** Argon
- **Worktree / Branch:** `/Users/josenunez/Projects/ml/snake-dqn-hero-stage0` (`argon-hero-stage0-1103`, base `403f2615`)
- **Artifact Directory:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a`
- **Primary JSON Receipt:** [`STAGE1A_READOUT.json`](file:///Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a/STAGE1A_READOUT.json) (committed copy at [`research/hero_stage0/STAGE1A_READOUT.json`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/hero_stage0/STAGE1A_READOUT.json))
- **Per-World Paired Receipts:** [`worlds/stage1a-world-00.json`](file:///Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a/worlds/stage1a-world-00.json) .. [`worlds/stage1a-world-47.json`](file:///Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage1a/worlds/stage1a-world-47.json)
- **World Namespace:** `t34-hero-probe-20261103-v1` (`N = 48` fresh paired worlds, `H = 6,000` frames/world, `288,000` session frames per arm; identical seeds and initial states as Stage 0b)
- **Live / Grid Mirror Verification:** `288,000 / 288,000` frames (`100.0%`) verified exact live `GameState` $\leftrightarrow$ `GridBatchSim` identity on `bare` AND `288,000 / 288,000` frames (`100.0%`) on `exact-stop-h48-nb` (`576,000` total audited frames, `0` divergences)

---

## 1. Explicit Stage 1A Verdict: **FAIL**

| Pre-Registered Bar | Threshold | Measured Value (`exact-stop-h48-nb` vs `bare`) | Status |
|---|---:|---:|---|
| **1. `L < 600` deaths removed** | **`>= 70.0%`** | **`64.44%`** (`29 / 45` deaths removed; `45 -> 16`, `-2.56 pp` short of `31.5 / 45`) | **FAIL** |
| **2. Delivered boost-share change** | **`<= +0.50 pp`** | **`+0.0326 pp`** (`0.1951% -> 0.2278%`; 95% CI `[-0.0488, +0.1140] pp`) | **PASS** |
| **3. Harmful overrides (4 independent scoring suffixes `4, 5, 6, 7`)** | **`<= 1.00%`** | **`0.8746%`** (`3 / 343`) under strict majority of all 4 scoring suffixes (`>= 3/4`)<br>**`1.1662%`** (`4 / 343`) under conditional majority where `b8-s3` survives (`> n_bare_survives / 2`) or `>= 2/4` | **PASS** (`>= 3/4`) / **FAIL** (conditional majority & `>= 2/4`) |
| **Overall Stage 1A Gate** | **All 3 bars pass** | **Fails Bar 1 (`64.44% < 70.0%`), and exceeds Bar 3 (`1.17% > 1.0%`) under conditional-majority / `>= 2/4` scoring** | **FAIL** |

### Why `exact-stop-h48-nb` Missed the `>= 70%` `L < 600` Death-Removal Bar
- In Stage 0b ([`argon-hero-stage0/REPORT.md`](file:///Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-hero-stage0/REPORT.md)), `bare` (`b8-s3`) suffered **`45` `L < 600` deaths** across the 48 worlds: **`27` (`60.0%`) `self-body enclosure`** and **`18` (`40.0%`) opponent collisions** (`14` `opponent body`, `4` `opponent head-on`).
- In Stage 1A, `exact-stop-h48-nb` (`H = 48`, `interval = 5`, `L >= 1`, `K = 2` selection suffixes, non-boost-first fallback) removed **`17 / 18` (`94.44%`) of `L < 600` opponent-collision deaths** (`14 -> 1` `opponent body`, `4 -> 0` `opponent head-on`), leaving only `1` `L < 600` opponent-collision death across all `288,000` frames.
- However, `exact-stop-h48-nb` removed only **`12 / 27` (`44.44%`) of `L < 600` `self-body enclosure` deaths** (`27 -> 15`). Of the `16` remaining `L < 600` deaths under `exact-stop-h48-nb`, **`15 / 16` (`93.75%`) are `self-body enclosure`**.
- This matches the Stage 0b counterfactual audit finding: `11 / 45` (`24.4%`) of `bare` `L < 600` deaths were already enclosed beyond any `50`-step escape (`t = -50`), and with a `48`-step lookahead horizon (`H = 48`), a pocket/coil commitment whose fatal collision occurs more than `48` steps after the entry decision survives both `H = 48` selection rollouts and therefore does not trigger an override at entry.

---

## 2. Controller Specification & Implementation

Implemented in [`research/hero_stage0/protocol.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/hero_stage0/protocol.py) and [`research/hero_stage0/stage1a.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/hero_stage0/stage1a.py), reusing [`research/bl_served_search/exact_stop.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/bl_served_search/exact_stop.py), [`research/bl_served_search/ceiling.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/bl_served_search/ceiling.py), [`research/t3_rollout_labels_20261015/bank_engine.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/t3_rollout_labels_20261015/bank_engine.py), and [`research/t3_rollout_labels_20261015/trigger.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/t3_rollout_labels_20261015/trigger.py):

1. **Trigger & Reset Semantics (`interval = 5`, `L >= 1`):**
   - Uses `DenseTrigger(interval=5, min_length=1)`, matching `exact_stop_alllen` in [`research/watch6_alllen_screen/protocol.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/watch6_alllen_screen/protocol.py) and [`research/watch6_alllen_screen/engine.py`](file:///Users/josenunez/Projects/ml/snake-dqn-hero-stage0/research/watch6_alllen_screen/engine.py).
   - Resets `DenseTrigger` on initial spawn and on every hero death/respawn (`last_fire = None`), so the first alive decision of every life fires (`gap >= 5` thereafter).
2. **Action Evaluation Order & Early-Stop (`exact-stop-h48-nb`):**
   - At every triggered root (`L >= 1`, `interval = 5`), computes `b8-s3` Q-values and legal actions via `ceiling._legal(root)` (`self_collision_Check` + `step_guard` $3\times 3$ opponent-head check).
   - Evaluates `bare_action = argmax_a Q(s, a)` first under `K = 2` deterministic selection suffixes `STAGE1A_SELECTION_SUFFIXES = (0, 1)` for `H = 48` exact-sim steps (`rollout_all(sim_b, root, a, [0, 1], 48, ...)`).
   - If `bare_action` survives `H = 48` on **both** selection suffixes `(0, 1)`, keeps `bare_action` immediately (`1` branch / `2` rollouts evaluated).
   - Otherwise, orders the remaining legal actions (`a != bare_action`) with **all legal non-boost actions (`0..2`) sorted by descending `b8-s3` Q-value before any legal boost actions (`3..5`) sorted by descending `b8-s3` Q-value**, and evaluates them one by one under selection suffixes `(0, 1)` via `exact_stop.run(..., order=order, suffixes=(0, 1), horizon=48)`, stopping at the first candidate that survives `H = 48` on both selection suffixes.
   - If no legal action survives both selection suffixes (`all_die_roots`), `exact_stop.run` delegates to `ceiling._choose` (`(S, T, L, Q)` tie-break across all evaluated legal actions).
3. **Independent Override Scoring (`STAGE1A_SCORING_SUFFIXES = (4, 5, 6, 7)`):**
   - Whenever `chosen_action != bare_action` (`override`), clones the pre-step root state before advancing the world and runs `H = 48` exact-sim rollouts for both `bare_action` and `chosen_action` across `4` independent scoring suffixes `(4, 5, 6, 7)` (disjoint from selection suffixes `(0, 1)`).
   - Records per-suffix survival vectors `bare_survives_vec` and `chosen_survives_vec`, the count of scoring suffixes where the override dies within `H = 48` while `b8-s3` survives (`n_override_dies_where_b8_survives`), and both majority definitions of `harmful`:
     - **Conditional majority among `b8-s3`-surviving scoring suffixes:** `n_override_dies_where_b8_survives > n_bare_survives / 2` (with `n_bare_survives > 0`).
     - **Strict majority of all 4 scoring suffixes (`>= 3 / 4`):** `n_override_dies_where_b8_survives >= 3`.
     - **Half or more of all 4 scoring suffixes (`>= 2 / 4`):** `n_override_dies_where_b8_survives >= 2`.

---

## 3. Deaths `L < 600` and `L >= 600` (`N = 48` Paired Worlds, `H = 6,000`)

### 3.1 Death Counts, Rates per 1k Frames, and Paired Reductions

| Metric | `bare` (`b8-s3`) | `exact-stop-h48-nb` | Paired Change / Reduction |
|---|---:|---:|---:|
| **Session frames (`48 x 6,000`)** | `288,000` | `288,000` | `0` |
| **Alive frames (post-step)** | `287,922` (`99.973%`) | `287,960` (`99.986%`) | `+38` (`+0.013 pp`) |
| — Alive frames at `L < 600` | `228,886` (`79.496%`) | `204,259` (`70.933%`) | `-24,627` (`-8.563 pp`) |
| — Alive frames at `L >= 600` | `59,036` (`20.504%`) | `83,701` (`29.067%`) | `+24,665` (`+8.563 pp`) |
| **Pre-step hero decisions (`288,000` total)** | `288,000` | `288,000` | `0` |
| — Decisions at `L < 600` | `228,976` (`79.506%`) | `204,320` (`70.944%`) | `-24,656` (`-8.561 pp`) |
| — Decisions at `L >= 600` | `59,024` (`20.494%`) | `83,680` (`29.056%`) | `+24,656` (`+8.561 pp`) |
| **`L < 600` deaths (count)** | **`45`** (`0.9375` / world) | **`16`** (`0.3333` / world) | **`-29` (`64.44%` of `bare` `L < 600` deaths removed)** |
| **`L < 600` deaths per 1k session frames** | **`0.15625`** (95% CI `[0.1040, 0.2085]`) | **`0.05556`** (95% CI `[0.0267, 0.0844]`) | **`-0.10069`** (95% CI `[-0.1523, -0.0491]`, `-64.44%`) |
| **`L < 600` deaths per 1k `L < 600` alive frames** | **`0.19660`** | **`0.07833`** | **`-0.11827`** (`-60.16%` in-band hazard reduction) |
| **`L >= 600` deaths (count)** | **`33`** (`0.6875` / world) | **`24`** (`0.5000` / world) | **`-9` (`27.27%` of `bare` `L >= 600` deaths removed)** |
| **`L >= 600` deaths per 1k session frames** | **`0.11458`** (95% CI `[0.0919, 0.1373]`) | **`0.08333`** (95% CI `[0.0589, 0.1078]`) | **`-0.03125`** (95% CI `[-0.0623, -0.0002]`, `-27.27%`) |
| **`L >= 600` deaths per 1k `L >= 600` alive frames** | **`0.55898`** | **`0.28673`** | **`-0.27225`** (`-48.71%` in-band hazard reduction) |
| **`L >= 800` deaths (sub-band count)** | `21` (`0.07292` / 1k session) | `15` (`0.05208` / 1k session) | `-6` (`-28.57%`) |
| **Total deaths (`ALL`, `L >= 1` count)** | **`78`** (`1.6250` / world) | **`40`** (`0.8333` / world) | **`-38` (`48.72%` of total `bare` deaths removed)** |
| **Total deaths per 1k session frames** | **`0.27083`** (95% CI `[0.2266, 0.3151]`) | **`0.13889`** (95% CI `[0.1084, 0.1694]`) | **`-0.13194`** (95% CI `[-0.1798, -0.0841]`, `-48.72%`) |
| **Total deaths per 1k alive frames** | **`0.27091`** | **`0.13891`** | **`-0.13200`** (`-48.72%`) |
| **Paired worlds win / tie / loss (fewer `L < 600` deaths)** | — | — | **`21` win / `23` tie / `4` loss** |
| **Paired worlds win / tie / loss (fewer total deaths)** | — | — | **`29` win / `16` tie / `3` loss** |

### 3.2 Death Breakdown by Length Sub-Band and Cause

| Length Band | `bare` Deaths | `exact-stop-h48-nb` Deaths | Deaths Removed (`%`) | `bare` Causes (`self` / `opp_body` / `head_on` / `wall`) | `exact-stop-h48-nb` Causes (`self` / `opp_body` / `head_on` / `wall`) |
|---|---:|---:|---:|---|---|
| **`L < 200`** | `12` | `6` | `6 / 12` (`50.00%`) | `5` / `6` / `1` / `0` | `6` / `0` / `0` / `0` |
| **`200 <= L < 600`** | `33` | `10` | `23 / 33` (`69.70%`) | `22` / `8` / `3` / `0` | `9` / `1` / `0` / `0` |
| **Subtotal `L < 600`** | **`45`** | **`16`** | **`29 / 45` (`64.44%`)** | **`27` (`60.0%`) / `14` (`31.1%`) / `4` (`8.9%`) / `0`** | **`15` (`93.8%`) / `1` (`6.2%`) / `0` / `0`** |
| **`600 <= L < 800`** | `12` | `9` | `3 / 12` (`25.00%`) | `9` / `3` / `0` / `0` | `9` / `0` / `0` / `0` |
| **`L >= 800`** | `21` | `15` | `6 / 21` (`28.57%`) | `17` / `4` / `0` / `0` | `14` / `1` / `0` / `0` |
| **Subtotal `L >= 600`** | **`33`** | **`24`** | **`9 / 33` (`27.27%`)** | **`26` (`78.8%`) / `7` (`21.2%`) / `0` / `0`** | **`23` (`95.8%`) / `1` (`4.2%`) / `0` / `0`** |
| **Total (`ALL`, `L >= 1`)** | **`78`** | **`40`** | **`38 / 78` (`48.72%`)** | **`53` (`67.9%`) / `21` (`26.9%`) / `4` (`5.1%`) / `0`** | **`38` (`95.0%`) / `2` (`5.0%`) / `0` / `0`** |

- **By death mechanism across `L < 600`:**
  - `opponent body`: `14 -> 1` (**`92.86%` removed**)
  - `opponent head-on`: `4 -> 0` (**`100.00%` removed**)
  - Combined `L < 600` opponent collisions: `18 -> 1` (**`94.44%` removed**)
  - `self-body enclosure`: `27 -> 15` (**`44.44%` removed**)
- **By death mechanism across `ALL` (`L >= 1`):**
  - Combined opponent collisions: `25 -> 2` (**`92.00%` removed**)
  - `self-body enclosure`: `53 -> 38` (**`28.30%` removed** in raw count; in-band self-enclosure per-alive-frame rate drops from `0.1180 / 1k` to `0.0734 / 1k` at `L < 600` and from `0.4404 / 1k` to `0.2748 / 1k` at `L >= 600`).

---

## 4. Delivered Boost-Share Change

| Band | `bare` Boost Decisions / Decisions | `bare` Boost Share | `exact-stop-h48-nb` Boost Decisions / Decisions | `exact-stop-h48-nb` Boost Share | Delivered Change (`pp`) |
|---|---:|---:|---:|---:|---:|
| **`L < 600`** | `182 / 228,976` | `0.0795%` | `201 / 204,320` | `0.0984%` | **`+0.0189 pp`** |
| **`L >= 600`** | `380 / 59,024` | `0.6438%` | `455 / 83,680` | `0.5437%` | **`-0.1001 pp`** |
| **Overall (`ALL`, `L >= 1`)** | **`562 / 288,000`** | **`0.1951%`** | **`656 / 288,000`** | **`0.2278%`** | **`+0.0326 pp`** (95% CI `[-0.0488, +0.1140] pp`) |

- Because `exact-stop-h48-nb` tests all legal non-boost actions (`0..2`) before any boost action (`3..5`) and shifts `8.56%` of total session frames from `L < 600` into `L >= 600` (where `b8-s3` naturally boosts more often, while `exact-stop-h48-nb`'s in-band `L >= 600` boost share is actually `0.1001 pp` *lower* than `bare`), the overall delivered boost-share change is **`+0.0326 pp`**, well below the **`<= +0.50 pp`** bar (**PASS**).

---

## 5. Override Rate Overall and by `L` Band

| Metric / Band | Search Roots (`interval = 5`) | Decisions (`every frame`) | Overrides (`chosen != bare`) | Override Rate per Search Root | Override Rate per Decision | Overrides to Non-Boost (`0..2`) | Overrides to Boost (`3..5`) |
|---|---:|---:|---:|---:|---:|---:|---:|
| **`L < 600`** | `40,889` | `204,320` | **`165`** | **`0.4035%`** | **`0.0808%`** | `135` (`81.82%`) | `30` (`18.18%`) |
| — `L < 200` | — | — | `35` | — | — | `28` (`80.00%`) | `7` (`20.00%`) |
| — `200 <= L < 600` | — | — | `130` | — | — | `107` (`82.31%`) | `23` (`17.69%`) |
| **`L >= 600`** | `16,729` | `83,680` | **`178`** | **`1.0640%`** | **`0.2127%`** | `138` (`77.53%`) | `40` (`22.47%`) |
| — `600 <= L < 800` | — | — | `67` | — | — | `48` (`71.64%`) | `19` (`28.36%`) |
| — `L >= 800` | — | — | `111` | — | — | `90` (`81.08%`) | `21` (`18.92%`) |
| **Overall (`ALL`, `L >= 1`)** | **`57,618`** | **`288,000`** | **`343`** | **`0.5953%`** | **`0.1191%`** | **`273` (`79.59%`)** | **`70` (`20.41%`)** |

### Root Classification Across All `57,618` Search Roots
- **`bare` survived both selection suffixes `(0, 1)` (`S_sel = 1.0`):** `56,914 / 57,618` roots (**`98.778%`**) — early-stopped after evaluating `1` branch (`2` rollouts).
- **`bare` failed at least one selection suffix (`S_sel < 1.0`):** `704 / 57,618` roots (**`1.222%`**):
  - **Alternative action survived both selection suffixes (`chosen_survived_selection_both = True`):** **`226` overrides** (`0.392%` of all roots; `121` at `L < 600`, `105` at `L >= 600`; `186` to non-boost `0..2`, `40` to boost `3..5`).
  - **No legal action survived both selection suffixes (`all_die_roots = 478`, `0.830%` of all roots):**
    - `bare` remained selected under `ceiling._choose` `(S, T, L, Q)` tie-break: **`361` roots**.
    - Alternative action won `ceiling._choose` `(S, T, L, Q)` tie-break (`chosen_survived_selection_both = False`): **`117` overrides** (`44` at `L < 600`, `73` at `L >= 600`; `87` to non-boost `0..2`, `30` to boost `3..5`).

---

## 6. Independent Override Scoring (`4` Disjoint Scoring Suffixes `4, 5, 6, 7`)

Every one of the `343` overrides was independently scored from the cloned pre-step root state across `4` scoring suffixes `(4, 5, 6, 7)` for `H = 48` steps (`2,744` scoring rollouts, `127,932` scoring env frames).

### 6.1 Distribution of Differential Scoring Deaths (`n_override_dies_where_b8_survives` out of 4)

| `n_override_dies_where_b8_survives` (out of 4 scoring suffixes) | Override Count (`N = 343`) | Share of Overrides (`%`) |
|---:|---:|---:|
| **`0 / 4`** (override never dies on any scoring suffix where `b8-s3` survives) | `327` | **`95.335%`** |
| **`1 / 4`** | `12` | **`3.499%`** |
| **`2 / 4`** | `1` | **`0.292%`** |
| **`3 / 4`** | `2` | **`0.583%`** |
| **`4 / 4`** | `1` | **`0.292%`** |

### 6.2 Harmful-Override Rate by Definition and Length Band

The prompt defines `harmful` as: *"the override dies within H=48 on a majority of scoring suffixes where the b8-s3 action would not."* Because "a majority of scoring suffixes where the b8-s3 action would not" can be read either as a **strict majority of all 4 scoring suffixes (`>= 3 / 4` suffixes where override dies and `b8-s3` survives)** or as a **conditional majority among the subset of scoring suffixes on which `b8-s3` survives (`n_diff > n_bare_survives / 2`)**, we report all three exact definitions transparently:

| Harmful Definition | Overall (`343` overrides) | `L < 600` (`165` overrides) | `L >= 600` (`178` overrides) | Among `survived_both` (`226` overrides) | Among `all_die` Fallback (`117` overrides) | Status vs `<= 1.00%` Bar |
|---|---:|---:|---:|---:|---:|---|
| **Strict majority of 4 scoring suffixes (`>= 3 / 4`)** | **`3 / 343` (`0.8746%`)** | `2 / 165` (`1.2121%`) | `1 / 178` (`0.5618%`) | `2 / 226` (`0.8850%`) | `1 / 117` (`0.8547%`) | **PASS (`0.87% <= 1.00%`)** |
| **Conditional majority where `b8-s3` survives (`n_diff > n_bare_survives / 2`)** | **`4 / 343` (`1.1662%`)** | `2 / 165` (`1.2121%`) | `2 / 178` (`1.1236%`) | `3 / 226` (`1.3274%`) | `1 / 117` (`0.8547%`) | **FAIL (`1.17% > 1.00%`)** |
| **Half or more of 4 scoring suffixes (`>= 2 / 4`)** | **`4 / 343` (`1.1662%`)** | `2 / 165` (`1.2121%`) | `2 / 178` (`1.1236%`) | `3 / 226` (`1.3274%`) | `1 / 117` (`0.8547%`) | **FAIL (`1.17% > 1.00%`)** |
| **Any scoring suffix (`>= 1 / 4`)** | `16 / 343` (`4.6647%`) | `5 / 165` (`3.0303%`) | `11 / 178` (`6.1798%`) | `13 / 226` (`5.7522%`) | `3 / 117` (`2.5641%`) | Diagnostic |

- **Mean survival rate across the 4 independent scoring suffixes (`343` overrides):**
  - `bare` (`b8-s3` argmax): **`31.56%`** (`433 / 1,372` scoring rollouts survived `H = 48`)
  - `exact-stop-h48-nb` (`chosen_action`): **`69.02%`** (`947 / 1,372` scoring rollouts survived `H = 48`)
  - Mean scoring survival gain on overridden states: **`+37.46 pp`** (`+0.3746` survival fraction).

### 6.3 Exact Audit of the `5` Overrides with `n_diff >= 2` or Conditional Majority

| World | Frame | Length `L` | `bare -> chosen` | Selection Survival `(0, 1)` (`bare` $\to$ `chosen`) | Scoring Survival `(4, 5, 6, 7)` (`bare` $\to$ `chosen`) | `n_diff / 4` | `>= 3/4` | Cond. Maj. | `>= 2/4` | Mechanism Note |
|---:|---:|---:|---|---|---|---|---:|---|---|---|
| `13` | `3156` | `665` | `2 -> 0` | `[T, F] -> [T, T]` | `[T, T, T, T] -> [F, F, F, F]` | `4 / 4` | **Yes** | **Yes** | **Yes** | `bare` died on selection suffix `1`; `chosen=0` survived `(0, 1)` but died at step `35` on all 4 scoring suffixes while `bare` survived all 4. |
| `18` | `4730` | `329` | `1 -> 0` | `[T, F] -> [F, T]` | `[T, T, T, T] -> [F, F, T, F]` | `3 / 4` | **Yes** | **Yes** | **Yes** | `all_die` fallback: both `bare=1` and `chosen=0` had `S_sel = 0.5`; `chosen=0` won `T` tiebreak on selection suffixes but died on `3/4` scoring suffixes where `bare` survived all 4. |
| `20` | `896` | `159` | `1 -> 0` | `[T, F] -> [T, T]` | `[T, T, T, F] -> [F, F, F, F]` | `3 / 4` | **Yes** | **Yes** | **Yes** | `bare` died on selection suffix `1`; `chosen=0` survived `(0, 1)` but died on all 4 scoring suffixes while `bare` survived `3/4`. |
| `32` | `3531` | `716` | `0 -> 3` | `[F, F] -> [T, T]` | `[T, F, F, F] -> [F, T, T, T]` | `1 / 4` | No | **Yes** | No | `bare` died on `2/2` selection and `3/4` scoring suffixes (`S_score = 0.25`), whereas `chosen=3` survived `2/2` selection and `3/4` scoring suffixes (`S_score = 0.75`); counted under conditional majority only because on the single scoring suffix (`4`) where `bare` survived, `chosen` died (`1/1 > 0.5`). |
| `40` | `5563` | `603` | `1 -> 3` | `[T, F] -> [T, T]` | `[T, T, T, T] -> [T, F, F, T]` | `2 / 4` | No | No | **Yes** | `bare` died on selection suffix `1`; `chosen=3` survived `(0, 1)` and `2/4` scoring suffixes while `bare` survived `4/4` (`2/4` is an exact tie, not a strict majority of 4). |

---

## 7. Hero Mass & Search Compute per World (`N = 48`, `H = 6,000`)

### 7.1 Hero Mass (Session-Integral Mean Length over `6,000` Frames)

| Metric | `bare` (`b8-s3`) | `exact-stop-h48-nb` | Paired Delta (`exact-stop-h48-nb - bare`) |
|---|---:|---:|---:|
| **Mean Hero Mass** | **`371.2745`** | **`446.7951`** | **`+75.5206`** (`+20.34%`) |
| **SD across 48 worlds** | `115.2907` | `138.0405` | `144.7740` |
| **SE of mean (`N = 48`)** | `16.6408` | `19.9244` | `20.8963` |
| **95% CI** | `[337.80, 404.75]` | `[406.71, 486.88]` | **`[+33.48, +117.56]`** (Fieller relative: `[+8.64%, +33.31%]`) |
| **90% CI** | `[343.35, 399.20]` | `[413.36, 480.23]` | **`[+40.46, +110.58]`** |
| **Paired worlds win / tie / loss (mass)** | — | — | **`34` win / `0` tie / `14` loss** |

### 7.2 Search Compute and Wall Time per World (`8` Workers, `nice -n 15`, 7 Single-Thread Caps)

| Metric | Mean per World (`N = 48`) | SD | Min | Median (`p50`) | `p90` | `p95` | Max | Cumulative (`48` Worlds) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Search roots (`interval = 5`, `L >= 1`)** | `1,200.375` | `0.489` | `1,200` | `1,200` | `1,201` | `1,201` | `1,201` | `57,618` |
| **Branches evaluated (selection)** | `1,254.083` | `22.790` | `1,210` | `1,253.5` | `1,277.3` | `1,286.6` | `1,334` | `60,196` (`1.045` / root) |
| **Branches skipped by early-stop** | `5,934.042` | `25.887` | `5,834` | `5,936.5` | `5,959.6` | `5,976.9` | `5,986` | `284,834` (`82.55%` of legal) |
| **Selection rollouts (`K = 2`, `H = 48`)** | `2,508.167` | `45.579` | `2,420` | `2,507.0` | `2,554.6` | `2,573.2` | `2,668` | `120,392` (`2.089` / root) |
| **Selection rollout env frames** | `119,619.2` | `1,896.8` | `115,921` | `119,664` | `121,490` | `122,428` | `125,959` | `5,741,720` (`19.94x` base) |
| **Selection search wall time (`s` / world)** | **`314.243 s`** (`5.24 min`) | `5.130 s` | `304.785 s` | **`314.126 s`** | `321.297 s` | **`322.185 s`** | `325.523 s` | `15,083.68 s` (`261.8 ms` / root) |
| **Independent scoring rollouts (`K = 4`, `2` arms)** | `57.167` | `40.483` | `8` | `56.0` | `96.0` | `117.2` | `232` | `2,744` (`343` overrides) |
| **Independent scoring env frames** | `2,665.25` | `1,878.8` | `369` | `2,664.0` | `4,462.6` | `5,355.8` | `10,858` | `127,932` |
| **Independent scoring wall time (`s` / world)** | **`6.492 s`** | `4.572 s` | `0.875 s` | **`6.418 s`** | `10.973 s` | **`11.714 s`** | `25.650 s` | `311.63 s` (`908.5 ms` / override) |
| **Total `exact-stop-h48-nb` world wall time (`s`)** | **`477.522 s`** (`7.96 min`) | `10.175 s` | `457.648 s` | `477.269 s` | `489.035 s` | `492.426 s` | `508.843 s` | `22,921.05 s` |
| **Total `bare` world wall time (`s`)** | **`140.649 s`** (`2.34 min`) | `3.596 s` | `133.903 s` | `140.427 s` | `145.676 s` | `146.944 s` | `147.624 s` | `6,751.14 s` |
| **End-to-end 48-world wall-clock (`8` workers)** | — | — | — | — | — | — | — | **`3,757.253 s` (`62.62 min`)** |

---

## 8. Per-World Paired Table (`N = 48` Worlds, `H = 6,000`)

| World | Seed | `bare` `L<600` Deaths | `h48-nb` `L<600` Deaths | `bare` `L>=600` Deaths | `h48-nb` `L>=600` Deaths | `bare` Total Deaths | `h48-nb` Total Deaths | `bare` Mass | `h48-nb` Mass | $\Delta$ Mass | `bare` Boost Share | `h48-nb` Boost Share | Search Roots | Overrides | Search Time |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 00 | `2062932752` | 0 | 0 | 1 | 1 | 1 | 1 | 313.60 | 324.09 | +10.49 | 0.13% | 0.17% | 1201 | 4 | 316.5s |
| 01 | `3785340679` | 2 | 0 | 1 | 0 | 3 | 0 | 251.57 | 679.11 | +427.54 | 0.10% | 0.07% | 1200 | 7 | 308.3s |
| 02 | `4218827148` | 1 | 0 | 0 | 0 | 1 | 0 | 485.12 | 512.38 | +27.25 | 0.20% | 0.25% | 1200 | 7 | 313.8s |
| 03 | `2415454477` | 1 | 0 | 1 | 0 | 2 | 0 | 421.52 | 649.75 | +228.22 | 0.13% | 0.28% | 1200 | 2 | 313.7s |
| 04 | `2977219390` | 1 | 1 | 1 | 0 | 2 | 1 | 440.01 | 332.33 | -107.67 | 0.12% | 0.15% | 1200 | 7 | 309.0s |
| 05 | `2628643480` | 0 | 0 | 1 | 1 | 1 | 1 | 413.63 | 397.38 | -16.25 | 0.08% | 0.08% | 1200 | 5 | 308.9s |
| 06 | `756767462` | 2 | 0 | 0 | 1 | 2 | 1 | 213.10 | 397.00 | +183.89 | 0.23% | 0.12% | 1200 | 8 | 312.5s |
| 07 | `1154321447` | 2 | 0 | 0 | 1 | 2 | 1 | 291.29 | 534.60 | +243.32 | 0.13% | 0.47% | 1200 | 11 | 314.5s |
| 08 | `945847054` | 1 | 0 | 0 | 0 | 1 | 0 | 363.37 | 686.60 | +323.23 | 0.37% | 0.03% | 1200 | 2 | 310.5s |
| 09 | `975190140` | 3 | 2 | 0 | 0 | 3 | 2 | 228.60 | 249.58 | +20.99 | 0.05% | 0.10% | 1201 | 13 | 306.0s |
| 10 | `3107027377` | 0 | 0 | 1 | 1 | 1 | 1 | 459.48 | 452.09 | -7.39 | 0.10% | 0.13% | 1201 | 3 | 307.7s |
| 11 | `1895355368` | 0 | 1 | 1 | 0 | 1 | 1 | 329.72 | 313.70 | -16.02 | 0.00% | 0.05% | 1200 | 5 | 313.9s |
| 12 | `4050160226` | 0 | 0 | 0 | 0 | 0 | 0 | 694.33 | 691.93 | -2.39 | 0.23% | 1.07% | 1200 | 5 | 313.1s |
| 13 | `2435685882` | 0 | 0 | 1 | 1 | 1 | 1 | 353.95 | 445.80 | +91.85 | 0.02% | 0.23% | 1200 | 8 | 313.0s |
| 14 | `1456278919` | 4 | 0 | 0 | 1 | 4 | 1 | 190.68 | 534.72 | +344.03 | 0.07% | 0.32% | 1201 | 7 | 304.9s |
| 15 | `1559494153` | 0 | 0 | 1 | 1 | 1 | 1 | 353.54 | 371.80 | +18.27 | 0.65% | 0.62% | 1200 | 9 | 304.8s |
| 16 | `3399611963` | 0 | 1 | 1 | 1 | 1 | 2 | 338.11 | 295.60 | -42.51 | 0.03% | 0.03% | 1201 | 8 | 321.0s |
| 17 | `3423899073` | 0 | 1 | 1 | 1 | 1 | 2 | 342.44 | 262.18 | -80.26 | 0.05% | 0.45% | 1201 | 10 | 316.0s |
| 18 | `3122211886` | 1 | 1 | 0 | 0 | 1 | 1 | 308.24 | 322.09 | +13.86 | 0.20% | 0.35% | 1201 | 10 | 313.1s |
| 19 | `2818026180` | 4 | 1 | 0 | 0 | 4 | 1 | 127.44 | 361.75 | +234.31 | 0.15% | 0.07% | 1201 | 6 | 313.9s |
| 20 | `2123997879` | 1 | 0 | 1 | 1 | 2 | 1 | 293.88 | 517.18 | +223.30 | 0.08% | 0.10% | 1201 | 4 | 322.7s |
| 21 | `244645779` | 0 | 0 | 1 | 0 | 1 | 0 | 344.01 | 533.54 | +189.53 | 0.77% | 0.03% | 1200 | 8 | 315.7s |
| 22 | `4180156207` | 1 | 1 | 0 | 0 | 1 | 1 | 328.31 | 308.95 | -19.37 | 0.10% | 0.13% | 1200 | 3 | 305.1s |
| 23 | `2057192298` | 0 | 0 | 1 | 0 | 1 | 0 | 380.72 | 656.09 | +275.37 | 0.10% | 0.13% | 1200 | 4 | 313.3s |
| 24 | `4042611737` | 0 | 0 | 1 | 0 | 1 | 0 | 405.96 | 535.46 | +129.51 | 0.57% | 0.15% | 1200 | 4 | 314.3s |
| 25 | `3060406266` | 0 | 0 | 1 | 1 | 1 | 1 | 550.25 | 411.56 | -138.69 | 0.15% | 0.53% | 1200 | 7 | 325.5s |
| 26 | `3913171354` | 1 | 1 | 0 | 0 | 1 | 1 | 359.56 | 301.78 | -57.78 | 0.42% | 0.15% | 1200 | 4 | 308.6s |
| 27 | `3803481744` | 2 | 0 | 1 | 0 | 3 | 0 | 355.10 | 601.71 | +246.61 | 0.03% | 0.35% | 1200 | 2 | 307.7s |
| 28 | `102410241` | 0 | 0 | 1 | 0 | 1 | 0 | 515.74 | 647.40 | +131.66 | 0.67% | 0.02% | 1200 | 1 | 321.8s |
| 29 | `690481545` | 3 | 0 | 0 | 1 | 3 | 1 | 254.34 | 402.29 | +147.96 | 0.07% | 0.07% | 1200 | 11 | 317.1s |
| 30 | `2500794294` | 0 | 0 | 0 | 1 | 0 | 1 | 599.10 | 329.27 | -269.84 | 0.07% | 0.33% | 1201 | 12 | 314.6s |
| 31 | `2778312588` | 0 | 0 | 1 | 1 | 1 | 1 | 361.85 | 395.72 | +33.86 | 0.08% | 0.05% | 1201 | 4 | 309.7s |
| 32 | `863838712` | 2 | 1 | 0 | 1 | 2 | 2 | 228.57 | 289.17 | +60.59 | 0.00% | 0.33% | 1200 | 17 | 322.4s |
| 33 | `4228541725` | 0 | 0 | 1 | 0 | 1 | 0 | 457.53 | 522.45 | +64.93 | 0.28% | 0.23% | 1200 | 7 | 312.4s |
| 34 | `4209784803` | 1 | 0 | 1 | 0 | 2 | 0 | 404.54 | 700.89 | +296.35 | 0.08% | 0.15% | 1200 | 4 | 308.3s |
| 35 | `488502840` | 2 | 1 | 0 | 0 | 2 | 1 | 328.31 | 356.61 | +28.30 | 0.12% | 0.28% | 1201 | 7 | 316.2s |
| 36 | `3162122039` | 1 | 0 | 1 | 1 | 2 | 1 | 440.72 | 571.15 | +130.43 | 0.35% | 0.23% | 1201 | 5 | 318.2s |
| 37 | `253126262` | 1 | 0 | 1 | 1 | 2 | 1 | 544.55 | 363.99 | -180.56 | 0.82% | 0.27% | 1201 | 8 | 315.4s |
| 38 | `751040636` | 0 | 0 | 1 | 0 | 1 | 0 | 521.07 | 615.40 | +94.32 | 0.15% | 0.05% | 1200 | 1 | 320.9s |
| 39 | `4049636215` | 2 | 0 | 1 | 1 | 3 | 1 | 210.72 | 330.19 | +119.47 | 0.28% | 0.12% | 1200 | 9 | 315.9s |
| 40 | `1353532238` | 1 | 0 | 1 | 1 | 2 | 1 | 279.24 | 352.43 | +73.19 | 0.25% | 0.20% | 1201 | 6 | 320.3s |
| 41 | `1481878349` | 0 | 0 | 1 | 1 | 1 | 1 | 556.19 | 474.21 | -81.98 | 0.20% | 0.17% | 1200 | 7 | 317.1s |
| 42 | `3999407681` | 1 | 2 | 1 | 0 | 2 | 2 | 367.14 | 206.02 | -161.12 | 0.13% | 0.25% | 1201 | 29 | 321.2s |
| 43 | `1650198235` | 2 | 2 | 1 | 0 | 3 | 2 | 268.97 | 273.08 | +4.11 | 0.02% | 0.13% | 1200 | 9 | 316.9s |
| 44 | `3654334958` | 1 | 0 | 1 | 1 | 2 | 1 | 294.49 | 367.30 | +72.81 | 0.28% | 0.77% | 1201 | 12 | 313.7s |
| 45 | `1631312248` | 1 | 0 | 1 | 1 | 2 | 1 | 340.03 | 467.51 | +127.49 | 0.17% | 0.13% | 1200 | 8 | 321.5s |
| 46 | `3284701936` | 0 | 0 | 1 | 1 | 1 | 1 | 410.74 | 493.51 | +82.77 | 0.05% | 0.48% | 1201 | 11 | 316.7s |
| 47 | `143115146` | 0 | 0 | 1 | 0 | 1 | 0 | 499.83 | 606.84 | +107.01 | 0.03% | 0.05% | 1200 | 2 | 315.6s |
| **Total / Mean** | **`48` worlds** | **`45`** | **`16`** | **`33`** | **`24`** | **`78`** | **`40`** | **`371.27`** | **`446.80`** | **`+75.52`** | **`0.20%`** | **`0.23%`** | **`57,618`** | **`343`** | **`314.2s`** |
