# Watch6 Live `GameState` Search Confirmation (`v1–v3`), Replay Bank Release, and All-Length (`L >= 1`) Extension

## Question & Goal

Establish confirmation-grade product efficacy for synchronous `H = 300` exact-stop search (`exact-stop-l2-prime` at `L >= 600`, and subsequently `exact-stop-l1` at `L >= 1`) in the actual shipped mixed Watch6 `GameSession` environment (`1` `b8-s3` hero vs `5 × frp3-s12 + v8(lambda = 8)` opponents, continuous respawn, mechanics v2, `H = 14,400` frames = `10 min` at `24 fps`, verified every frame against an independent cell-exact `GridBatchSim` mirror) under the binding Watch product contract (`docs/research/final_run_20261019/SUCCESS_CONTRACT.md` §3: hero `deaths / 1k frames` 95% paired Student-t `UB95 < 0` and full-horizon pooled mass ratio 95% paired Fieller `LB95 >= -5.0%`), and ship precomputed deterministic `.watch.gz` replay banks (`research/watch_replay/`) so live Watch serves confirmed search-assisted play at `24 fps` without real-time search overruns (`research/watch6_search_efficacy/README.md`, `docs/research/ROADMAP.md`).

---

## Timeline of Versions & Steps

| Date | Step / Version | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-29 | **Watch6 Search Efficacy `V1`** (`research/watch6_search_efficacy/REGISTRATION.md`) | Registered `N = 96` paired `H = 14,400` Watch6 sessions (`exact-stop-l2-prime` vs `bare` `b8-s3`) across `16` L4 shards (`6` pairs/shard). Superseded pre-data before any main-study dispatch to add portable release-evidence bundling and `<= 12`-transfer archive staging. | **SUPERSEDED PRE-DATA BY `V2`** (`research/watch6_search_efficacy/RULING.md`) |
| 2026-10-30 | **Watch6 Search Efficacy `V2` Canary & Forecast Fix** (`research/watch6_search_efficacy/REGISTRATION_V2.md`, `docs/research/final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md`, `docs/research/watch6_forecast_fix_status_2026-10-30.md`) | Ratified `V2` (`RULING.md`). Genuine L4 canary (`watch6-canary-transport01`, receipt `eacbda01...`) measured `4.018 s` worst `H = 300` branch batch and `0.0443/0.0444 s/frame` main loops; plugging the theoretical `2,880 × 6` branch ceiling into the `V2` formula yielded `1,494` base / `2,988` doubled min (`49.8 h > 22 h` cap), blocking all `18` shard layouts. Owner-authorized empirical forecast (`61d60578`) scaled the observed `p95` across `24` `H = 6,000` `W = 4` sessions (`2,883.46 s × 2.4 + 637.69 s = 7,557.99 s/pair`), sizing `24` L4 `W = 4` shards (`209` base / `418` doubled min). | **CANARY PASS / EMPIRICAL FORECAST ADOPTED (`24 × 4` L4 `W = 4`)** |
| 2026-10-30 to 2026-10-31 | **Watch6 Search Efficacy `V2` Main Run** (`research/watch6_search_efficacy_v3/REGISTRATION_V3.md` §1) | Dispatched `24` L4 shards (`96` pairs); `88/96` paired sessions exported cleanly, but shard `5` exhausted its `3` attempts on pre-start staging/RPC transport failures without starting compute. All `88` `V2` pair payloads were permanently sealed uninspected. | **`INVALID_OR_INCOMPLETE`** (`88/96` pairs; outcome-blind transport stop on shard `5`) |
| 2026-10-31 to 2026-11-01 | **Watch6 Search Efficacy `V3` Confirmation** (`research/watch6_search_efficacy_v3/REGISTRATION_V3.md`, `research/watch6_l1_confirm/REGISTRATION_L1_V1.md` §1.1) | Following Astra's pre-data review (`JUDGE_QUESTION.md`, `ebacfb8d`), `V3` rev 2 (`N = 96` fresh primary seeds `202610310000–202610310095`, `24 × 4` L4 `W = 4` shards) enforced serialized staging (`watch6-staging.lock`), separated positively proven pre-`ready.json` pre-start failures from compute-phase external-transport failures, and verified all `96/96` primary pairs (`100%` `14,400/14,400` live/Grid mirror parity; shard wall time `mean = 10,312.4 s`, `p95 = 12,058.8 s` at `1.94 u/h`). | **`GO_EFFICACY_ONLY` (`52135312`)** — `deaths/1k` **`-0.1250`** (`CI95 [-0.1532, -0.0968]`, `-45.0%`); mass **`+63.74%`** (`Fieller CI95 [+49.84%, +78.52%]`) |
| 2026-10-31 to 2026-11-01 | **132-Session `exact-stop-l2-prime` Watch Replay Bank Release** (`research/watch_replay/README.md`, `docs/research/ROADMAP.md` §1.1) | Generated and mirror-verified `132` complete `H = 6,000` `exact-stop-l2-prime` `.watch.gz` sessions across banks `01–11` (`watch-replay-bank-live-132`), passed Mac `24 fps` serving qualification (`fec81656`: cold session `< 0.45 s`, concurrent live fallback `update p95 = 8.78 ms <= 11.688 ms`, `0/120` paced overruns), and enabled live serving (`f5912f44`). | **SHIPPED IN PRODUCTION WATCH (`2026-11-01`, `f5912f44`)** |
| 2026-10-31 | **R6 `L < 600` Trigger Census** (`research/r6_l600_census/README.md`) | Harvested `384` `L < 600` roots from `64` Watch6-v2 sessions (`192` death-precursor + `192` length-matched survival controls) and evaluated `6`-action × `H = 500` exact-stop rollouts plus 5-fold grouped logistic triggers (`snake-dqn-artifacts/t34-sol/argon-mac-checks/r6-census/RESULT.json`). | **KILL** (`34.90%` roots had `>= +5` harvest gain, but lethal flips were `7.69% > 0.5%` and OOF ROC-AUC was `0.474–0.521`, passing `0` threshold bars) |
| 2026-10-31 | **R7 Macro-Action Fallback Screen** (`research/r7_macro_fallback/README.md`) | Tested `18` `10`-step spatial macro sequences (`wall_hug`, `coil_tail`, `max_clearance` × `2` boost modes × `3` reentry modes) on `239` `1-step-all-die` train roots (`91` at `L >= 600`) and `200` ordinary survival roots (`snake-dqn-artifacts/t34-sol/argon-r7-repack/R7_READOUT.json`). | **KILL (`3/5` bars)** — `L >= 800` `all_die_sel0` rescue `33.0% vs 9.0%` and Mac `p95 = 0.3158 ms` passed, but `L >= 600` lift was `+3.85 pp < +10.0 pp` with `2 > 0` ordinary lethal flips |
| 2026-10-31 to 2026-11-01 | **All-Length (`L >= 1`) Watch6 Screen (`watch6_alllen_screen`)** (`research/watch6_alllen_screen/README.md`) | Tested removing the `L >= 600` gate so the identical `exact_stop.run` (`H = 300`) runs at `L >= 1` (`exact-stop-l1`) across `48` paired `H = 6,000` Watch6 sessions (`snake-dqn-artifacts/t34-sol/argon-alllen-a100/analysis.json`); also generated `36` `exact-stop-l1` replay sessions (`banks 14l1–16l1`, `395eacbb`). | **`ADOPT_EXACT_STOP_L_GE_1_FOR_NEW_BANKS`** (`L < 600` deaths `34 → 1`; total `deaths/1k` `0.1597 → 0.0521` [`-67.4%`]; mass `461.16 → 559.43` [`+21.31%`]) |
| 2026-11-01 | **Watch6-L1-V1 3-Arm Confirmation (`watch6_l1_confirm`)** (`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`, `docs/research/ROADMAP.md` §2) | Registered and dispatched 3-arm (`bare`, `exact-stop-l2-prime`, `exact-stop-l1`) fixed-sequence confirmation on `N = 96` fresh paired worlds (`202611010000–202611010095`) at `H = 14,400` (`12 × 6` A100 `W = 6` + `6 × 4` L4 `W = 4` = `18` single-wave shards; Fable R2 `APPROVE-WITH-CONDITIONS`, running in `t34-sol/l1v1-main-01` at `a3135483`). | **IN FLIGHT (`2026-11-01`)** |

---

## Final Verdict & Key Numbers

### 1. Watch6 `V3` Confirmation (`exact-stop-l2-prime` vs `bare` `b8-s3`, `N = 96`, `H = 14,400`) — **`GO_EFFICACY_ONLY`**
Primary sources: `research/watch6_search_efficacy_v3/REGISTRATION_V3.md`, `research/watch6_l1_confirm/REGISTRATION_L1_V1.md` §1.1, and `docs/research/ROADMAP.md` §1.1 (commit `52135312`).

| Metric (`N = 96` Paired `H = 14,400` Watch6 Sessions) | `bare` (`b8-s3`) | `exact-stop-l2-prime` (`L >= 600`, `H = 300`) | Paired Estimate (`l2_prime - bare`) | Two-Sided 95% CI | Contract Bar (`SUCCESS_CONTRACT.md` §3) | Verdict |
|---|---:|---:|---:|---:|---|---|
| **Hero `deaths / 1k frames`** | `0.2778` | `0.1528` | **`-0.1250`** (`-45.0%` relative) | **`[-0.1532, -0.0968]`** (paired Student-t, `df = 95`) | `UB95 < 0` | **PASS** |
| **Full-horizon mean hero mass** | `346.66` | `567.62` | **`+220.96`** (`+63.74%` pooled ratio) | **`[+49.84%, +78.52%]`** (paired Fieller) | `LB95 >= -5.0%` | **PASS** |
| **Live `GameState` ↔ `GridBatchSim` mirror parity** | `14,400 / 14,400` | `14,400 / 14,400` | `100%` (`96/96` primary pairs) | — | `0` mismatches | **PASS** |

### 2. Shipped 132-Session Watch Replay Bank (`watch-replay-bank-live-132`) & Mac Serving Qualification
Primary sources: `research/watch_replay/README.md` and `docs/research/ROADMAP.md` §1.1 (commits `fec81656`, `f5912f44`).

- **Production Replay Bank (`watch-replay-bank-live-132`):** `132` verified `H = 6,000` (`4m 10s` at `24 fps`) `exact-stop-l2-prime` `.watch.gz` recordings across banks `01–11` (`> 9 hours` of unique recorded Watch gameplay), cycle-shuffled without immediate boundary repeats, with visible `"Recorded assisted hero"` / `"Move from search"` UI badges and automatic live bare `b8-s3` fallback.
- **Mac `24 fps` Serving Qualification (`qualify.py`, commit `fec81656`):**
  - Cold replay session start (including gzip decompression + full SHA-256 verification): **`< 0.45 s`** (bar `<= 1.0 s`).
  - Concurrent live fallback (`GameSession.update` while background verifier runs at `<= 20%` GIL duty cycle): **`p95 = 8.78 ms`** (bar `<= 11.688 ms`), **`0 / 120`** paced `24 fps` (`41.67 ms`) overruns.

### 3. Early-Game (`L < 600`) & Macro Follow-Ups (`r6`, `r7`, `watch6_alllen_screen`)
Primary sources: `research/r6_l600_census/README.md` (`snake-dqn-artifacts/t34-sol/argon-mac-checks/r6-census/RESULT.json`), `research/r7_macro_fallback/README.md` (`snake-dqn-artifacts/t34-sol/argon-r7-repack/R7_READOUT.json`), and `research/watch6_alllen_screen/README.md` (`snake-dqn-artifacts/t34-sol/argon-alllen-a100/analysis.json`).

- **Why `exact-stop-l2-prime`'s remaining deaths occurred:** In `watch6_alllen_screen` (`48` paired `H = 6,000` sessions), `34` of `l2_prime`'s `46` deaths (`73.9%`) occurred at `L < 600` solely because the `L >= 600` gate prevented `exact_stop.run` from firing; on `L < 600` frames `l2_prime` was bitwise identical to `bare` (`34` deaths each before the first `L >= 600` trigger).
- **All-Length (`L >= 1`) Screen (`exact-stop-l1`, `N = 48` paired `H = 6,000` sessions):**
  - **`L < 600` deaths:** Cut from **`34`** (`bare` and `l2_prime`) to **`1`** (`-97.1%`).
  - **Total `deaths / 1k frames`:** **`0.0521`** (`15` deaths) vs **`0.1597`** (`46` deaths, `l2_prime`) and **`0.2882`** (`83` deaths, `bare`) — paired delta vs `l2_prime`: **`-0.1076 / 1k`** (`-67.4%`, `CI95 [-0.1609, -0.0544]`); vs `bare`: **`-0.2361 / 1k`** (`-81.9%`, `CI95 [-0.3056, -0.1667]`).
  - **Full-horizon mean hero mass:** **`559.43`** (`l1`) vs **`461.16`** (`l2_prime`, **`+98.27`**, Fieller **`+21.31%`** `[+10.08%, +34.79%]`) and **`351.07`** (`bare`, **`+208.35`**, Fieller **`+59.35%`** `[+42.26%, +79.77%]`).
  - **Search trigger rate & compute:** Trigger fraction rose only from `5.61%` (`l2_prime`) to `6.81%` (`l1`, `+1.20 pp`), adding `+18.7%` wall time (`426.3 s → 505.8 s` per `H = 6,000` session on A100 `W = 6`).

---

## What Was Learned & Current Status

1. **Synchronous `H = 300` exact-stop search delivers massive, confirmation-grade survival and mass gains in live Watch6:** On `N = 96` fresh `10`-minute (`H = 14,400`) sessions (`Watch6-V3`), `exact-stop-l2-prime` cut hero deaths by **`45.0%`** (`-0.1250 / 1k`) and increased full-horizon mean mass by **`+63.74%`** (`346.66 → 567.62`) with `100%` frame-by-frame live/`GridBatchSim` mirror parity.
2. **Precomputed `.watch.gz` replay banks decouple heavy search from Mac `24 fps` serving:** Because `H = 300` exact-stop search takes `~0.4–4.0 s` on triggered frames, serving pre-verified `.watch.gz` replay banks (`watch-replay-bank-live-132`) delivers `24 fps` playback with `< 9 ms` `p95` frame latency and zero overruns.
3. **Removing the `L >= 600` gate (`exact-stop-l1`) eliminates `97%` of early-game deaths without new features or heuristics:** Scalar triggers (`r6_l600_census`, ROC-AUC `~0.50`) and open-loop `10`-step spatial macros (`r7_macro_fallback`, `+3.85 pp` lift) both failed, whereas simply running the existing `exact_stop.run` verifier at all lengths (`L >= 1`) fired on only `+1.20%` more frames and cut total deaths by another **`67.4%`** beyond `l2_prime` on `48` paired `H = 6,000` sessions.

**Current Status:** **LIVE IN PRODUCTION & ACTIVE CONFIRMATION.** Watch serves the confirmed `132`-session `exact-stop-l2-prime` replay bank (`watch-replay-bank-live-132`, commit `f5912f44`), while `Watch6-L1-V1` (`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`, `N = 96`, `H = 14,400`) runs on Colab to confirm `exact-stop-l1` (`L >= 1`) against both `bare` and `exact-stop-l2-prime`.

---

## Kept Lineage Documents (`KEEP`)

- **Campaign Roadmap & Feasibility / Forecast Notes:**
  - [`docs/research/ROADMAP.md`](../ROADMAP.md)
  - [`docs/research/final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md`](../final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md)
  - [`docs/research/watch6_forecast_fix_status_2026-10-30.md`](../watch6_forecast_fix_status_2026-10-30.md)
- **Watch6 Search Efficacy (`V1`, `V2`, `V3`):**
  - [`research/watch6_search_efficacy/README.md`](../../../research/watch6_search_efficacy/README.md)
  - [`research/watch6_search_efficacy/REGISTRATION.md`](../../../research/watch6_search_efficacy/REGISTRATION.md)
  - [`research/watch6_search_efficacy/REGISTRATION_V2.md`](../../../research/watch6_search_efficacy/REGISTRATION_V2.md)
  - [`research/watch6_search_efficacy/RULING.md`](../../../research/watch6_search_efficacy/RULING.md)
  - [`research/watch6_search_efficacy_v3/REGISTRATION_V3.md`](../../../research/watch6_search_efficacy_v3/REGISTRATION_V3.md)
  - [`research/watch6_search_efficacy_v3/JUDGE_QUESTION.md`](../../../research/watch6_search_efficacy_v3/JUDGE_QUESTION.md)
- **Production Watch Replay Bank Infrastructure:**
  - [`research/watch_replay/README.md`](../../../research/watch_replay/README.md)
- **Early-Game (`L < 600`), Macro Fallback, All-Length (`L >= 1`) Screen & `Watch6-L1-V1` Confirmation:**
  - [`research/r6_l600_census/README.md`](../../../research/r6_l600_census/README.md)
  - [`research/r7_macro_fallback/README.md`](../../../research/r7_macro_fallback/README.md)
  - [`research/watch6_alllen_screen/README.md`](../../../research/watch6_alllen_screen/README.md)
  - [`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`](../../../research/watch6_l1_confirm/REGISTRATION_L1_V1.md)
