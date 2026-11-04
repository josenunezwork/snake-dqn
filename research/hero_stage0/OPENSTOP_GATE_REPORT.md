# Decisive Out-of-Sample Gate Report: `open-stop-h32-nb` (`K=1`) vs `bare` (`b8-s3`)

- **Single Verdict:** **`PASS`** (`5 / 5` pre-registered decisive gate bars passed on `N = 96` fresh Watch6-v2 worlds at `H = 6,000`)
- **Registration Authority:** `/Users/josenunez/Projects/ml/snake-dqn/docs/research/final_run_20261019/CAMPAIGN_LOG.md` row `2026-11-04 PT 09:30`
- **Date:** 2026-11-04
- **Operator / Implementer:** Argon
- **Worktree / Branch:** `/Users/josenunez/Projects/ml/snake-dqn-hero-stage0` (`argon-hero-stage0-1103`)
- **Controller Code Freeze Commit:** `42a48def4a6ed993cefa40827e2a477c8e3e7798` (`42a48def`)
- **Pre-Execution Allocation & Seed-Audit Commit:** `fd7d90f678103f38b5373ea0b718f9f2ea8ddad9` (`fd7d90f6`, committed before running any world)
- **Namespace:** `t34-openstop-gate-20261104-v1` (`N = 96` worlds, `H = 6,000` frames/world, `576,000` frames/arm, `1,152,000` total live/Grid mirror-verified frames)
- **Readout Receipt:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/argon-openstop-gate/OPENSTOP_GATE_READOUT.json` (committed copy at `research/hero_stage0/OPENSTOP_GATE_READOUT.json`, SHA-256 `fe590865ef91ce326054348f40ba38c26c5253090bf27122d409d2aa41436be8`)

---

## 1. Pre-Execution Allocation, Seed Disjointness Audit, and Training-Exclusion Registration

Before simulating any world in `t34-openstop-gate-20261104-v1`, the 96-world Watch6-v2 allocation and its disjointness audit were committed at `fd7d90f678103f38b5373ea0b718f9f2ea8ddad9`:

| Artifact / Field | Value |
|---|---|
| **Namespace** | `t34-openstop-gate-20261104-v1` |
| **World count / Horizon** | `N = 96`, `H = 6,000` (`576,000` frames per arm) |
| **Constructor & World Contract** | `research.watch_replay.product.open_watch` (`mechanics_version = 2`, `continuous_respawn = True`, hero slot `0 = b8-s3`, opponent slots `1..5 = frp3-s12 + v8` with `lambda = 8.0`, live/Grid mirror audited on every frame) |
| **Pre-execution commit hash** | `fd7d90f678103f38b5373ea0b718f9f2ea8ddad9` |
| **`seeds_sha256`** | `29c478d83f096389f4e7b18f33994cd4640c078dffd5a1b1d511b2aead02e44e` |
| **`BANK_OPENSTOP_GATE.json` SHA-256** | `18fde3be1ba4f95b5e9d3fc7008712cfa6b2268f84131e8b4acb16df16a0702e` |
| **`training_exclusions_openstop_gate.json` SHA-256** | `1033a38a04187809a1af47d67a1d81e62614b724f782c9f179863285224b6daf` (`96` sorted seeds; `144` combined with `t34-hero-probe-20261103-v1`) |
| **Disjointness vs `t34-hero-probe-20261103-v1`** | `0` collisions (`48` allocated + `512` prefix seeds checked) |
| **Disjointness vs `watch6_alllen_screen`** | `0` collisions (`771` unique seeds across main, canary, preflight, benchmarks, 256-prefixes) |
| **Disjointness vs `watch6_l1_v1` primary & reserve** | `0` collisions (`3,075` unique seeds across primary, reserve, canary, fixtures; no outputs opened) |
| **Disjointness vs `watch_replay` banks** | `0` collisions (`18,957` unique seeds across 74 namespaces + 13 fixtures; no live bank files touched) |
| **Disjointness vs C1, R6 census, & 15 committed banks** | `0` collisions (`26,357` unique seeds) |
| **Disjointness vs 7 historical FRP `training_exclusions.json`** | `0` collisions (up to `1,135,019` excluded seeds per inventory) |
| **Single-thread `rg` literal scan (repo + artifacts)** | `0` candidate hits; both positive controls (`17222702` from committed bank and `2062932752` from `t34-hero-probe-20261103-v1`) detected |
| **Live/Grid mirror verification** | `576,000 / 576,000` frames in `bare` + `576,000 / 576,000` frames in `open-stop-h32-nb` (`K=1`) = `1,152,000 / 1,152,000` frames bit-identical |

---

## 2. Pre-Registered Decisive Gate Scorecard (`N = 96` Worlds, `H = 6,000`)

Pre-registered decision rule: **`PASS` iff `L<600 deaths removed >= 70%` AND `total deaths removed >= 55%` AND `paired mass LB95 >= -5% of bare mean` AND `harmful overrides <= 1.0% (H=32 conditional majority on 4 independent suffixes)` AND `boost share delta <= +0.5 pp`.**

| # | Pre-Registered Gate Bar | Requirement | `bare` (`b8-s3`) | `open-stop-h32-nb` (`K=1`) | Observed Effect & 95% Confidence Interval | Bar Verdict |
|---:|---|---|---:|---:|---:|---:|
| **1** | **`L<600` deaths removed** | `>= 70.0%` | `117` (`0.2031 / 1k`) | `19` (`0.0330 / 1k`) | **`83.76%` removed** (`-98` deaths; Fieller 95% CI `[72.91%, 93.37%]`; paired $\Delta = -0.1701 / 1\text{k}$, 95% CI `[-0.2142, -0.1260]`) | **`PASS`** |
| **2** | **Total deaths removed** | `>= 55.0%` | `169` (`0.2934 / 1k`) | `51` (`0.0885 / 1k`) | **`69.82%` removed** (`-118` deaths; Fieller 95% CI `[61.58%, 77.40%]`; paired $\Delta = -0.2049 / 1\text{k}$, 95% CI `[-0.2442, -0.1655]`) | **`PASS`** |
| **3** | **Paired mass LB95 vs `bare` mean** | `>= -5.0%` (`>= -17.28` mass) | `345.60 ± 13.56` | `522.32 ± 15.11` | **LB95 = `+40.63%` of `bare` mean** (`+140.40`; Fieller LB95 = **`+38.63%`**; paired $\Delta = +176.72 \pm 18.30$, 95% CI `[+140.40, +213.04]`; rel. gain **`+51.13%`**, Fieller 95% CI `[+38.63%, +65.23%]`) | **`PASS`** |
| **4** | **Harmful overrides (`H=32` cond. majority on suffixes `4,5,6,7`)** | `<= 1.00%` | — | `7 / 799` overrides | **`0.8761%`** (Clopper-Pearson 95% CI `[0.3529%, 1.7967%]`; strict majority `>= 3/4`: `5 / 799 = 0.6258%`) | **`PASS`** |
| **5** | **Boost share $\Delta$ vs `bare`** | `<= +0.50 pp` | `0.1717%` (`989 / 576,000`) | `0.2227%` (`1,283 / 576,000`) | **`+0.0510 pp` pooled** (paired world mean $\Delta = +0.0510\text{ pp}$, 95% CI `[-0.0063, +0.1083] pp`) | **`PASS`** |
| | **Single Gate Verdict** | **All 5 bars pass** | | | **`5 / 5` bars passed (`79` wins / `1` tie / `16` losses on paired world mass)** | **`PASS`** |

### Headline Paired Summary (`96` Worlds, `576,000` Frames/Arm)

| Metric | `bare` (`b8-s3`) | `open-stop-h32-nb` (`K=1`) | Paired Difference / Ratio (95% CI) |
|---|---:|---:|---:|
| **Mean mass (`mass_integral`)** | `345.60` (`±13.56` SE, 95% CI `[318.69, 372.51]`) | **`522.32` (`±15.11` SE, 95% CI `[492.31, 552.32]`)** | **`+176.72` (`[+140.40, +213.04]`, `+51.13%`, Fieller `[+38.63%, +65.23%]`)** |
| **Mean max length per world** | `778.75` | **`1,047.54`** | **`+268.79` (`+34.52%`)** |
| **Alive frames (`ALL` / `L<600` / `L>=600`)** | `575,831` / `473,661` / `102,170` (`17.74%` at `L>=600`) | **`575,949` / `343,160` / `232,789` (`40.42%` at `L>=600`)** | **`2.28x` alive frames at `L>=600`** (`+130,619` frames) |
| **Total deaths (`ALL`)** | `169` (`0.2934 / 1k`) | **`51` (`0.0885 / 1k`)** | **`-118` (`-69.82%`, Fieller `[-77.40%, -61.58%]`)** |
| **`L<600` deaths** | `117` (`0.2031 / 1k`) | **`19` (`0.0330 / 1k`)** | **`-98` (`-83.76%`, Fieller `[-93.37%, -72.91%]`)** |
| **`L>=600` deaths** | `52` (`0.0903 / 1k`) | **`32` (`0.0556 / 1k`)** | **`-20` (`-38.46%` unadjusted; `-72.99%` per 1k `L>=600` alive frames)** |
| **Self-enclosure deaths (`ALL` / `L<600` / `L>=600`)** | `102` / `64` / `38` | **`47` / `17` / `30`** | **`-55` (`-53.92%` all, Fieller `[40.60%, 65.95%]`; `-73.44%` at `L<600`)** |
| **Opponent collision deaths (`ALL` / `L<600` / `L>=600`)** | `67` (`6` head-on, `61` body) / `53` / `14` | **`4` (`0` head-on, `4` body) / `2` / `2`** | **`-63` (`-94.03%` all; `-96.23%` at `L<600`; `-85.71%` at `L>=600`)** |
| **World W / T / L (higher mass)** | — | — | **`79` wins / `1` tie / `16` losses** (`82.3%` win rate) |
| **World W / T / L (fewer total deaths)** | — | — | **`74` wins / `16` ties / `6` losses** |
| **World W / T / L (fewer `L<600` deaths)** | — | — | **`58` wins / `34` ties / `4` losses** |

---

## 3. Deaths by Cause and Length Band (`N = 96` Worlds, `H = 6,000`)

### 3.1 Primary Length Regimes (`L<600`, `L>=600`, `ALL`)

| Arm | Regime | Total Deaths | Deaths / 1k Session Frames (95% CI) | Deaths / 1k Alive Frames in Band | Self-Body Enclosure | Opponent Head-On | Opponent Body | Opponent Total | Wall |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`bare` (`b8-s3`)** | `L<600` | `117` | `0.2031` (`[0.1633, 0.2430]`) | `0.2470` (`117 / 473,661`) | `64` (`54.7%`) | `6` (`5.1%`) | `47` (`40.2%`) | `53` (`45.3%`) | `0` |
| | `L>=600` | `52` | `0.0903` (`[0.0727, 0.1079]`) | `0.5090` (`52 / 102,170`) | `38` (`73.1%`) | `0` (`0.0%`) | `14` (`26.9%`) | `14` (`26.9%`) | `0` |
| | **`ALL`** | **`169`** | **`0.2934` (`[0.2575, 0.3293]`)** | **`0.2935` (`169 / 575,831`)** | **`102` (`60.4%`)** | **`6` (`3.6%`)** | **`61` (`36.1%`)** | **`67` (`39.6%`)** | **`0`** |
| **`open-stop-h32-nb` (`K=1`)** | `L<600` | **`19` (`-83.76%`)** | **`0.0330` (`[0.0136, 0.0524]`)** | **`0.0554` (`19 / 343,160`, `-77.59%`)** | **`17` (`-73.44%`)** | **`0` (`-100.0%`)** | **`2` (`-95.74%`)** | **`2` (`-96.23%`)** | `0` |
| | `L>=600` | **`32` (`-38.46%`)** | **`0.0556` (`[0.0396, 0.0716]`)** | **`0.1375` (`32 / 232,789`, `-72.99%`)** | **`30` (`-21.05%`)** | **`0` (`0.0%`)** | **`2` (`-85.71%`)** | **`2` (`-85.71%`)** | `0` |
| | **`ALL`** | **`51` (`-69.82%`)** | **`0.0885` (`[0.0667, 0.1104]`)** | **`0.0885` (`51 / 575,949`, `-69.83%`)** | **`47` (`-53.92%`)** | **`0` (`-100.0%`)** | **`4` (`-93.44%`)** | **`4` (`-94.03%`)** | **`0`** |

### 3.2 Fine-Grained Length Sub-Bands (`L<200`, `200<=L<600`, `600<=L<800`, `L>=800`)

| Length Sub-Band | Cause Class | `bare` (`b8-s3`) | `open-stop-h32-nb` (`K=1`) | Count Change | `%` Removed vs `bare` |
|---|---|---:|---:|---:|---:|
| **`L<200`** | **Total** | **`30`** | **`0`** | **`-30`** | **`100.00%`** |
| | Self-body enclosure | `11` | `0` | `-11` | `100.00%` |
| | Opponent head-on | `2` | `0` | `-2` | `100.00%` |
| | Opponent body | `17` | `0` | `-17` | `100.00%` |
| **`200<=L<600`** | **Total** | **`87`** | **`19`** | **`-68`** | **`78.16%`** |
| | Self-body enclosure | `53` | `17` | `-36` | `67.92%` |
| | Opponent head-on | `4` | `0` | `-4` | `100.00%` |
| | Opponent body | `30` | `2` | `-28` | `93.33%` |
| **`600<=L<800`** | **Total** | **`25`** | **`9`** | **`-16`** | **`64.00%`** |
| | Self-body enclosure | `17` | `8` | `-9` | `52.94%` |
| | Opponent head-on | `0` | `0` | `0` | — |
| | Opponent body | `8` | `1` | `-7` | `87.50%` |
| **`L>=800`** | **Total** | **`27`** | **`23`** | **`-4`** | **`14.81%`** |
| | Self-body enclosure | `21` | `22` | `+1` | `-4.76%` |
| | Opponent head-on | `0` | `0` | `0` | — |
| | Opponent body | `6` | `1` | `-5` | `83.33%` |
| **`ALL` (`L>=1`)** | **Total** | **`169`** | **`51`** | **`-118`** | **`69.82%`** |
| | Self-body enclosure | `102` | `47` | `-55` | `53.92%` |
| | Opponent head-on | `6` | `0` | `-6` | `100.00%` |
| | Opponent body | `61` | `4` | `-57` | `93.44%` |
| | Wall | `0` | `0` | `0` | — |

Note that because `open-stop-h32-nb` (`K=1`) eliminates `100%` of `L<200` deaths (`30 -> 0`), `78.2%` of `200<=L<600` deaths (`87 -> 19`), and `64.0%` of `600<=L<800` deaths (`25 -> 9`), the hero spends **`2.28x` as many frames at `L>=600`** (`232,789` vs `102,170` frames) and reaches a mean max length of **`1,047.54`** (vs `778.75` for `bare`). Per 1,000 alive frames at `L>=600`, the `L>=600` death rate drops by **`72.99%`** (`0.5090 -> 0.1375` deaths / 1k `L>=600` alive frames), and the `L>=600` self-enclosure hazard drops by **`65.36%`** (`0.3719 -> 0.1289` / 1k `L>=600` alive frames).

---

## 4. Override Rates, Root Certificate Breakdown, and Independent-Suffix Harmful-Override Scoring

### 4.1 Override Rates (Per Decision and Per Root) & Root Certificate Breakdown

| Metric (`N = 96` worlds, `576,000` frames) | Overall (`ALL`) | `L < 600` | `L >= 600` |
|---|---:|---:|---:|
| **Total hero decisions** | `576,000` | `343,271` (`59.59%`) | `232,729` (`40.41%`) |
| **Total search roots (`interval = 5`)** | `115,230` (`1,200.31 / world`) | `68,714` (`59.63%`) | `46,516` (`40.37%`) |
| **Roots where `bare` is certified (`alive + un-enclosed` at `H=32`)** | `110,863` (`96.2102%`) | — | — |
| **Roots where `bare` survives `1..32` raw (`bare_raw_survived_roots`)** | `114,701` (`99.5409%`) | — | — |
| **Roots where `bare` fails only by terminal enclosure at `H=32`** | `3,838` (`3.3307%`) | — | — |
| **Roots where `bare` fails by raw death within `1..32`** | `529` (`0.4591%`) | — | — |
| **Roots where no action is certified (`all_die_roots`, keeps `bare` if raw-alive)** | `3,724` (`3.2318%`) | — | — |
| **Total overrides (`chosen != bare`)** | **`799` (`8.32 / world`)** | **`261` (`32.67%`)** | **`538` (`67.33%`)** |
| **Overrides triggered by `bare` terminal enclosure only** | `565` (`70.71%` of overrides) | — | — |
| **Overrides triggered by `bare` raw death within `1..32`** | `234` (`29.29%` of overrides) | — | — |
| **Overrides to non-boost (`0..2`) / to boost (`3..5`)** | `615` (`76.97%`) / `184` (`23.03%`) | — | — |
| **Override rate per decision (`interventions / decisions`)** | **`0.1387%`** (`799 / 576,000`) | **`0.0760%`** (`261 / 343,271`) | **`0.2312%`** (`538 / 232,729`) |
| **Override rate per search root (`interventions / search_roots`)** | **`0.6934%`** (`799 / 115,230`) | **`0.3798%`** (`261 / 68,714`) | **`1.1566%`** (`538 / 46,516`) |

### 4.2 Independent-Suffix Harmful-Override Scoring (`OPENSTOP_SCORING_SUFFIXES = (4, 5, 6, 7)`)

All `799` overrides across the 96 worlds were scored on the 4 independent opponent-tiebreak suffixes `(4, 5, 6, 7)` (disjoint from selection suffix `(0,)`):

| Evaluation Mode on Suffixes `(4, 5, 6, 7)` | Regime | Overrides Scored | Mean `S_score(bare)` | Mean `S_score(chosen)` | Harmful Cond. Majority (`n_diff > n_bare/2`) | Harmful Strict Majority (`n_diff >= 3/4`) | Harmful `>= 2/4` (`n_diff >= 2`) | Harmful `>= 1/4` (`n_diff >= 1`) | `n_diff` Histogram (`0 / 1 / 2 / 3 / 4`) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **`H=32` raw survival (Gate Bar 4)** | **`ALL`** | **`799`** | **`0.7707`** | **`0.8739`** | **`7` (`0.8761%`, CP95 `[0.35%, 1.80%]`)** | **`5` (`0.6258%`)** | **`8` (`1.0013%`)** | **`18` (`2.2528%`)** | **`781 / 10 / 3 / 4 / 1`** |
| | `L<600` | `261` | `0.7385` | `0.8812` | **`1` (`0.3831%`)** | **`1` (`0.3831%`)** | `2` (`0.7663%`) | `5` (`1.9157%`) | `256 / 3 / 1 / 1 / 0` |
| | `L>=600` | `538` | `0.7862` | `0.8704` | `6` (`1.1152%`) | `4` (`0.7435%`) | `6` (`1.1152%`) | `13` (`2.4164%`) | `525 / 7 / 2 / 3 / 1` |
| **`H=48` raw survival (diagnostic)** | `ALL` | `799` | `0.6931` | `0.8073` | `14` (`1.7522%`) | `8` (`1.0013%`) | `17` (`2.1277%`) | `42` (`5.2566%`) | `757 / 25 / 9 / 4 / 4` |
| | `L<600` | `261` | `0.6121` | `0.7807` | `5` (`1.9157%`) | `3` (`1.1494%`) | `7` (`2.6820%`) | `14` (`5.3640%`) | `247 / 7 / 4 / 1 / 2` |
| | `L>=600` | `538` | `0.7323` | `0.8202` | `9` (`1.6729%`) | `5` (`0.9294%`) | `10` (`1.8587%`) | `28` (`5.2045%`) | `510 / 18 / 5 / 3 / 2` |
| **`H=32` certified open survival (diagnostic)** | `ALL` | `799` | `0.3120` | `0.7028` | `33` (`4.1302%`) | `9` (`1.1264%`) | `27` (`3.3792%`) | `61` (`7.6345%`) | `738 / 34 / 18 / 6 / 3` |
| | `L<600` | `261` | `0.3295` | `0.7174` | `10` (`3.8314%`) | `2` (`0.7663%`) | `8` (`3.0651%`) | `18` (`6.8966%`) | `243 / 10 / 6 / 0 / 2` |
| | `L>=600` | `538` | `0.3034` | `0.6956` | `23` (`4.2751%`) | `7` (`1.3011%`) | `19` (`3.5316%`) | `43` (`7.9926%`) | `495 / 24 / 12 / 6 / 1` |

---

## 5. Compute Cost per World (`N = 96` Worlds, 8 Workers, `nice -n 15`, 7 Single-Thread Caps)

| Cost / Timing Metric (`H = 6,000` frames/world, `1,200.31` roots/world) | `bare` (`b8-s3`) | `open-stop-h32-nb` (`K=1`) |
|---|---:|---:|
| **Mean branches evaluated / root** | `0.0000` | **`1.1028`** (`127,078` branches / `115,230` roots; `491,918` skipped) |
| **Mean selection rollouts / root** | `0.0000` | **`1.1028`** (`127,078` rollouts) |
| **Total selection environment frames (96 worlds)** | `0` | **`4,031,704`** (`41,997` env frames / world) |
| **8-worker selection `ms / root` (mean across `115,230` roots)** | `0.00 ms` | **`100.24 ms / root`** |
| **8-worker selection `ms / session frame`** | `0.00 ms` | **`20.05 ms / frame`** |
| **8-worker selection wall time per world (mean ± SD)** | `0.00s` | **`120.32s ± 12.65s`** (`min 101.86s`, `median 117.15s`, `p90 135.90s`, `p95 138.99s`, `max 171.95s`) |
| **8-worker override-scoring wall time per world (mean)** | `0.00s` | **`8.32s`** (`median 7.26s`, `p95 14.52s`, `max 43.32s`) |
| **8-worker full session wall time per world (with live/Grid mirror)** | `150.02s` (`median 144.69s`, `p95 180.52s`) | **`292.78s`** (`median 289.41s`, `p95 317.51s`, `max 425.08s`) |
| **Total wall clock for 96 paired worlds on 8 Mac workers** | — | **`5,396.93s` (`89.95 min`, `56.22s` wall per paired world)** |

---

## 6. Single Gate Verdict

**`PASS`**
