# Watch6 Live `GameState` Search Confirmation (`v1–v3`), Replay Bank Release, and All-Length (`L >= 1`) Extension

## Question & Goal

Establish confirmation-grade product efficacy for synchronous `H = 300` exact-stop search (`exact-stop-l2-prime` at `L >= 600`, and subsequently `exact-stop-l1` at `L >= 1`) in the actual shipped mixed Watch6 `GameSession` environment (`1` `b8-s3` hero vs `5 × frp3-s12 + v8(lambda = 8)` opponents, continuous respawn, mechanics v2, `H = 14,400` frames = `10 min` at `24 fps`, verified every frame against an independent cell-exact `GridBatchSim` mirror) under the binding Watch product contract (`docs/research/final_run_20261019/SUCCESS_CONTRACT.md` §3: hero `deaths / 1k frames` 95% paired Student-t `UB95 < 0` and full-horizon pooled mass ratio 95% paired Fieller `LB95 >= -5.0%`), and ship precomputed deterministic `.watch.gz` replay banks (`research/watch_replay/`) so live Watch serves confirmed search-assisted play at `24 fps` without real-time search overruns (`research/watch6_search_efficacy/README.md`, `docs/research/ROADMAP.md`).

---

## Timeline of Versions & Steps

| Date | Step / Version | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-30 | **Watch6 Search Efficacy `V1`** (`research/watch6_search_efficacy/REGISTRATION.md`) | Registered `N = 96` paired `H = 14,400` Watch6 sessions (`exact-stop-l2-prime` vs `bare` `b8-s3`), six operational shard partitions on L4 `W = 4` (CPU science, GPU unused). | **SUPERSEDED PRE-DATA BY `V2`** — Astra's pre-data ruling ratified the prospective V2 design with a measured same-pin L4 canary before any main run (`research/watch6_search_efficacy/RULING.md`) |
| 2026-10-30 | **Watch6 `V2` canary, feasibility block & forecast fix** (`research/watch6_search_efficacy/REGISTRATION_V2.md`, `docs/research/final_run_20261019/WATCH6_SHARD_FEASIBILITY_20261030.md`, `docs/research/watch6_forecast_fix_status_2026-10-30.md`) | Genuine L4 `transport01` canary (receipt SHA256 `eacbda01…`) measured `4.018 s` worst W4 branch batch and `0.0443 / 0.0444 s/frame` main loops; under the registered physical-maximum forecast, W4 needed `1,494` base / `2,988` effective minutes (`49.8 h`) vs the `660 / 1,320` min (`22 h`) ceiling, so all `18` supported layouts failed (`NOT_READY`). Owner-authorized empirical forecast (package pin `61d60578`): linear p95 of `24` full `H = 6,000` replay sessions (`2,883.46 s`) × `14,400/6,000` + `637.69 s` W4 bare-loop cost = `7,557.99 s/pair`, sizing W4 at `12` shards × `8` pairs (`398` base / `796` doubled min/shard). | **NOT_READY → EMPIRICAL FORECAST PREPARED** (pending a new same-pin canary) |
| 2026-10-30 to 2026-10-31 | **Watch6 Search Efficacy `V2` Main Run** (`research/watch6_search_efficacy_v3/REGISTRATION_V3.md` context, `docs/research/final_run_20261019/CAMPAIGN_LOG.md` 2026-10-31 row) | `88/96` paired sessions exported; shard 5 (the sixth shard) lost all three attempts to pre-start staging/RPC transport failures without starting compute. All `88` V2 pair payloads stay sealed and unread; Astra denied extra attempts or `88+8` reuse. | **`INVALID_OR_INCOMPLETE`** (`88/96`, `0` outcomes read) |
| 2026-10-31 to 2026-11-01 | **Watch6 Search Efficacy `V3` Confirmation** (`research/watch6_search_efficacy_v3/REGISTRATION_V3.md`, `JUDGE_QUESTION.md`, `research/watch6_l1_confirm/REGISTRATION_L1_V1.md` §5.1) | After Astra rejected V3 rev 1 pre-data, V3 rev 2 kept the V2 science with a fresh SHA256-derived `96`-seed primary bank (no reserves), `24 × 4` L4 `W = 4` shards (`209` base / `418` doubled min), serialized staging (`watch6-staging.lock`) and traceback-verified external-transport resume rules. All `96/96` primary pairs verified (`0` reserve substitutions, `2,764,800/2,764,800` mirror frames); `24/24` shards PASS (whole-shard `mean = 10,312.4 s`, `p95 = 12,058.8 s`; calibrated L4 rate `1.94 u/h`). Final look via V6.1 accounting adapter `75471246`. | **`GO_EFFICACY_ONLY` (2026-11-01 14:40 PT, `52135312`)** — `deaths/1k` **`-0.125`** (`CI95 [-0.153, -0.097]`); mass **`+63.7%`** (`Fieller CI95 [+49.8%, +78.5%]`) |
| 2026-11-01 | **132-Session `exact-stop-l2-prime` Watch Replay Bank Release** (`research/watch_replay/README.md`, `docs/STATUS.md`, `docs/research/ROADMAP.md`) | `watch-replay-bank-live-132`: `132` confirmed `H = 6,000` `exact-stop-l2-prime` `.watch.gz` sessions from banks `01–08` and `10–12`; Mac `24 fps` qualification (`qualify.py`, merged at `fec81656`): `132/132` inventory parity, cold readiness `0.349–0.435 s` (`< 0.45 s`), concurrent live fallback `update p95 = 8.78 ms <= 11.688 ms`, `0/120` paced overruns; released at `f5912f44`. | **SERVED IN WATCH (released 2026-11-01 ~16:00 PT, `f5912f44`)** |
| 2026-10-31 | **R6 `L < 600` safe-harvest census** (`research/r6_l600_census/README.md`) | Counterfactual census on `64` fresh Watch6-v2 worlds / `384` roots, plus 5-fold world-disjoint OOF separability probes (`snake-dqn-artifacts/t34-sol/argon-mac-checks/r6-census/RESULT.json`). | **KILL** (`34.90%` = `134/384` roots gained `>= +5` H500 mass, but non-bare H500 lethal flips were `7.69%`, OOF ROC-AUC `0.474–0.521`, `0` threshold points passed) |
| 2026-10-31 | **R7 macro-option fallback** (`research/r7_macro_fallback/README.md`) | Pruned depth-2 (`K = 15` ticks) deterministic spatial macro-options `{O_0..O_5}` on `1-step-all-die` C/T `train` roots, `0` assessment roots opened (`snake-dqn-artifacts/t34-sol/argon-r7-repack/R7_READOUT.json`). | **KILL (`3/5` bars)** — Mac 1-thread p95 `0.3158 ms` and `L >= 800` `all_die_sel0` rescue `33.0%` vs `9.0%` passed; total `L >= 600` lift `+3.85 pp < +10.0 pp` and `2 > 0` ordinary lethal flips failed |
| 2026-10-31 to 2026-11-01 | **All-Length (`L >= 1`) Watch6 Screen** (`research/watch6_alllen_screen/README.md`) | Removed the `L >= 600` gate (`DenseTrigger(min_length=1, interval=5)`, `H = 300`) across `48` paired `H = 6,000` Watch6-v2 sessions (`snake-dqn-artifacts/t34-sol/argon-alllen-a100/analysis.json`); then `36/36` `exact-stop-l1` replay sessions in banks `14l1–16l1` verified into `watch-replay-bank-live-168` (`395eacbb`). | **`ADOPT` for new replay banks** (`L < 600` deaths `34 → 1` vs `L >= 600`; `deaths/1k` `0.1597 → 0.0521` [`-67.4%`]; mass `461.16 → 559.43` [Fieller `+21.31%`]) |
| 2026-11-01 | **Watch6-L1-V1 3-arm confirmation** (`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`, `docs/STATUS.md`) | 3-arm (`bare`, `exact-stop-l2-prime`, `exact-stop-l1`) confirmation on `N = 96` fresh SHA256-derived paired worlds at `H = 14,400` (`12` A100 `W = 6` + `6` L4 `W = 4` = `18` single-wave shards); ratified on Fable R2 (`APPROVE-WITH-CONDITIONS`, `93c77f43`), `l1v1-packet-02` canaries PASS, running as `l1v1-main-01` (`a3135483`). | **IN FLIGHT (as of base snapshot)** |

---

## Final Verdict & Key Numbers

### 1. Watch6 `V3` Confirmation (`exact-stop-l2-prime` vs `bare` `b8-s3`, `N = 96`, `H = 14,400`) — **`GO_EFFICACY_ONLY`**
Primary sources: `docs/STATUS.md` and `snake-dqn-artifacts/t34-sol/argon-v3-accounting-v6-fix1/final-look-v3/efficacy-decision.json` (verdict `GO_EFFICACY_ONLY`, `replay_serving_decision = ELIGIBLE_FOR_REPLAY_QUALIFICATION`). Absolute arm levels below are pooled from that file's `session_diagnostics` (deaths and `mass_sum` over `1,382,400` frames per arm).

| Metric (`N = 96` paired `H = 14,400` sessions) | `bare` (`b8-s3`) | `exact-stop-l2-prime` (`L >= 600`, `H = 300`) | Paired estimate | Two-sided 95% CI | Contract bar (`SUCCESS_CONTRACT.md` §3) | Verdict |
|---|---:|---:|---:|---:|---|---|
| **Hero `deaths / 1k frames`** | `0.3248` (`449` deaths) | `0.1997` (`276` deaths) | **`-0.1251`** (`-38.5%` relative, `death_reduction_fraction`) | **`[-0.1535, -0.0968]`** (paired Student-t) | `UB95 < 0` | **PASS** |
| **Full-horizon pooled mean hero mass** | `399.92` | `654.57` | **`+63.68%`** pooled ratio | **`[+49.79%, +78.50%]`** (paired Fieller) | `LB95 >= -5.0%` | **PASS** |
| **Live `GameState` ↔ `GridBatchSim` mirror parity** | `14,400 / 14,400` per session | `14,400 / 14,400` per session | `96/96` pairs, `2,764,800/2,764,800` frames | — | `0` mismatches | **PASS** |

### 2. Served 132-Session Watch Replay Bank (`watch-replay-bank-live-132`) & Mac Serving Qualification
Primary sources: `research/watch_replay/README.md`, `docs/STATUS.md`, `docs/research/ROADMAP.md` (commits `fec81656`, `f5912f44`).

- **Served bank:** `132` confirmed `H = 6,000` (`250 s` at `24 fps`) `exact-stop-l2-prime` `.watch.gz` recordings from banks `01–08` and `10–12`, labelled **"Recorded assisted hero"**, with search overrides marked "Move from search" in the inspector and live bare `b8-s3` fallback. `ReplayBank` shuffles each cycle without reusing a unit within a cycle or immediately repeating the boundary unit.
- **Mac `24 fps` qualification (`qualify.py`, merged at `fec81656`):** `132/132` inventory parity; cold session readiness `0.349–0.435 s` (`< 0.45 s`); concurrent cold-verification live-B8 fallback `update p95 = 8.78 ms <= 11.688 ms` (verifier yields the GIL at `<= 20%` duty cycle); `0/120` paced-`24 fps` overruns (`step p99 = 11.34 ms <= 41.67 ms`).

### 3. Early-Game (`L < 600`) & Macro Follow-Ups (`r6`, `r7`, `watch6_alllen_screen`)
Primary sources: `research/r6_l600_census/README.md`, `research/r7_macro_fallback/README.md`, `research/watch6_alllen_screen/README.md`.

- **Where `exact-stop-l2-prime`'s deaths occurred (all-length screen, `48` paired `H = 6,000` sessions):** `34` of `l2_prime`'s `46` deaths were at `L < 600` (`bare`: `53` of `83`).
- **All-Length (`L >= 1`) Screen (`exact-stop-l1`):**
  - **`L < 600` deaths:** `53` (`bare`) / `34` (`l2_prime`) → **`1`**.
  - **`deaths / 1k frames`:** **`0.0521`** (`15` deaths) vs `0.1597` (`46`, `l2_prime`) and `0.2882` (`83`, `bare`) — paired delta vs `l2_prime`: **`-0.1076`** (`-67.4%`, `CI95 [-0.1609, -0.0544]`); vs `bare`: **`-0.2361`** (`-81.9%`, `CI95 [-0.2828, -0.1895]`).
  - **Full-horizon mean hero mass:** **`559.43`** vs `461.16` (`l2_prime`, **`+98.27`**, Fieller **`+21.31%`** `[+10.08%, +34.79%]`) and `351.07` (`bare`, **`+208.37`**, Fieller **`+59.35%`** `[+45.24%, +75.82%]`).
  - **Search cost:** `1,268.5` mean branches/session vs `449.6` for `l2_prime` (`2.821×`).

---

## What Was Learned & Current Status

1. **Synchronous `H = 300` exact-stop search gives confirmation-grade survival and mass gains in live Watch6:** on `N = 96` fresh `H = 14,400` sessions (Watch6 V3), `exact-stop-l2-prime` changed hero deaths by `-0.125/1k` (`-38.5%`) and pooled full-horizon mass by `+63.7%`, with `100%` live/`GridBatchSim` mirror parity (`efficacy-decision.json`, `docs/STATUS.md`).
2. **Precomputed `.watch.gz` replay banks decouple heavy search from Mac `24 fps` serving:** the worst `H = 300` W4 branch batch measured `4.018 s` on the L4 canary, while the replay path qualified on the Mac with live-fallback `update p95 = 8.78 ms` and `0/120` paced overruns.
3. **Removing the `L >= 600` gate (`exact-stop-l1`) removed almost all early-game deaths:** learned/scalar triggers (`r6_l600_census`, OOF ROC-AUC `0.474–0.521`) and macro-option fallbacks (`r7_macro_fallback`, `+3.85 pp` lift) both failed, whereas running the existing `exact_stop.run` at all lengths cut `L < 600` deaths `34 → 1` and total `deaths/1k` by a further `67.4%` vs `l2_prime` on `48` paired `H = 6,000` sessions, at `2.821×` the branches.

**Current Status (at base snapshot c4c4ab41):** **SERVED & CONFIRMATION IN FLIGHT.** Watch serves the confirmed `132`-session `exact-stop-l2-prime` replay bank (`watch-replay-bank-live-132`, commit `f5912f44`), while `Watch6-L1-V1` (`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`, `N = 96`, `H = 14,400`) runs on Colab to confirm `exact-stop-l1` (`L >= 1`) against both `bare` and `exact-stop-l2-prime`.

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
