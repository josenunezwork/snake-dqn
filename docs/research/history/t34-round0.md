# T34 Round 0 Sparse-Teacher (`teacher_v2`) & Round 0-D Dense-Teacher Premise Tests (`t34-round0`)

## Question & Goal

Does the deterministic T34 v2 search teacher (`research.t34_exit_b8.teacher_v2`: $\beta = 1$, `H = 300`, shadow depth `D = 300`, `bisection-radius3-prepared-cache`) in `t34_round0`, or the contingent dense search teacher (T3's `DenseTrigger(min_length=600, interval=5)`) in `t34_round0d`, meet the registered GO clauses against bare `b8-s3` — teacher scripted no-death survival `>= 0.66` with paired gain `>= +0.10`, equal-mix pooled mass gain `>= +40` with efficacy `LB90 > 0`, delivered boost share in `[0.005, 0.10]`, and survival guard A (pooled alive-frame delta `LB95 > -0.05`, each mix `LB95 > -0.075`, absolute alive-frame fraction `>= 0.30` per mix) — across 240 paired `H = 5,000` worlds (`96` scripted, `72` frozen, `72` mixed) (`research/t34_round0/PREREGISTRATION_ROUND0.md`)?

---

## Timeline of Versions & Steps

| Date | Package / Stage | What Was Built / Run | Verdict |
|---|---|---|---|
| 2026-10-20 | `t34_round0` base (`PREREGISTRATION_ROUND0.md`, `OC_ROUND0.json`, `VERIFICATION_A6_FINAL.json`) | Registered the 240-pair `H = 5,000` Round 0 study (`BANK_ROUND0.json`, `SEED_AUDIT_ROUND0.json`), ran 20,000-rep operating characteristics (`oc.py` -> `OC_ROUND0.json`, sha `66c0594e...` as recorded in `OPERATING_CHARACTERISTICS.md`), and verified excluded Mac `A6` preflight (`7f7bc147`, `18.69 s`, `50/50` focused tests). | **Base draft verified (`7f7bc147`)** — OC worst protected-regression joint error `0.00750–0.03670 <= 0.06`; joint mass+survival pass at true `+40` is `0.239–0.262`. |
| 2026-10-20 | `ERRATUM_A5_ADDENDUM_B.md` & `ERRATUM_INTEGRITY_SIZING.md` (`8900d2b5`) | Replaced the `3,000` row-steps/s throughput floor with `>= 50` completed worlds/h/lane on Colab L4 `K = 12 W1` (`54.37` and `57.17` worlds/h measured in `A5`), added create-only remote `refs/tags/t34-round0/*` claims + append-only repo ledger (`ledger/t34-round0-study-01.jsonl`), `ceil(10%)` stratified Mac replay, and 3-way physical worker parity (`09d00b7a...`). | **Integrity & sizing verified (`8900d2b5`)** — `a6-03` completed in `13.34 s`; `68/68` Round 0 tests passed. |
| 2026-10-20 | `PREREGISTRATION_ROUND0_FIXES1.md`, `ERRATUM_FIXES1.md`, `FIX-FIRST #2` (`838ae93e` -> `b60b0e3a`) | Implemented `FIX-FIRST #1` and `#2`: canonical `REGISTRATION_ROUND0.json` on `origin/main`, dedicated fault exception classes in `research/t34_exit_b8/faults.py`, `schedule_complete.json` pre-reservation guard, 48-hour dispatch deadline, and `4:3:3` stratified `100/100/40`-pair chunks (`27.2 u` cap + `5.44 u` reserve = `32.64 u` initial / `65.28 u` max rerun liability). | **Ratified & dispatched (`b60b0e3a`)** — `a6-fixes2-01` completed in `23.80 s`; `139/139` Round 0 + adjacent tests passed (`41.64 s`). |
| 2026-10-20 to 2026-10-21 | `t34_round0_erratum` (`ERRATUM_MECHANICS_LABEL.md`, `attest_mechanics.py`) | Ran single-fixture mechanics attestation (`ATTESTED_MECHANICS_V2`, sha `a27aa164...`) proving that Round 0 executed `research/apex_safety_20260926/deployment.yaml` (`game.mechanics_version: 2`, sha `4146baa3...`) despite `BANK_ROUND0.json`'s metadata label `profile.mechanics_version: 1`. | **`ATTESTED_MECHANICS_V2` PASS** — pre-analysis provenance erratum recorded without opening study outcomes. |
| 2026-10-20 | `t34_round0_amend` (`AMENDMENT_BOOST_FLOOR.json`, `30b2739a`) | Ratified pre-look boost-floor amendment replacing the absolute `[0.005, 0.10]` delivered-boost floor (bare B8 s3 delivered boost share `0.12%` over `190,779` decisions, below the `0.5%` floor; `docs/research/final_run_20261019/CAMPAIGN_LOG.md` 10-20 ~19:55) with the relative `0.5 * bare_share <= teacher_share <= 0.10` clause (`rule.py`, `decide.py`). | **Ratified pre-look (`30b2739a`)** — closed unexecuted on pair outcomes when Round 0 settled `KILL` pre-look. |
| 2026-10-20 to 2026-10-21 | `t34_round0d` (`PREREGISTRATION_ROUND0D.md`, `VERIFICATION_ROUND0D.md`) | Built contingent 240-world dense-teacher (`DenseTrigger(600, 5)`, `D = 300`) fallback package (`272` fresh disjoint seeds in `BANK_ROUND0D.json`), re-scoped with the relative boost floor, 12-pair wave-quantized straggler scheduling, `120 u` hard total ceiling (`chunk_admission_ceiling = 120 - replay_cap`), and `A5-D` L4 fidelity/rate gates (`89/89` tests pass). | **Prepared (`DRAFT`, `89/89` tests pass)** — never activated because Round 0 settled `KILL / INTEGRITY_FAILURE` rather than a governing scientific `STOP`/`INCONCLUSIVE`. |
| 2026-10-21 | `t34-round0-study-01` terminal settlement (`ledger/t34-round0-study-01.jsonl`) | During post-chunk replay/analysis transition, the append-only replay-ledger publication barrier was violated before `analyze` reservation (`KILL / INTEGRITY_FAILURE`). | **`KILL / INTEGRITY_FAILURE` (pre-look)** — halted before any outcome look; all 240 pair outcomes remained sealed and unused. |

---

## Final Verdict & Key Numbers

### 1. Operating Characteristics (`research/t34_round0/OC_ROUND0.json`, sha `66c0594ec359...` as recorded in `research/t34_round0/OPERATING_CHARACTERISTICS.md`)
- **Inputs & Simulation Grid (`oc.py`, seed `202610200`, `20,000` reps/cell):** Resampled historical FRP-v2 Phase 1 (`480` worlds/mix; mass SD `240.38 / 197.06 / 180.71`, alive-fraction delta SD `0.3738 / 0.3363 / 0.2954`) and Phase 2 (`160` worlds/mix; mass SD `247.15 / 213.92 / 176.14`, alive-fraction delta SD `0.3566 / 0.3493 / 0.2717`) pools (`research/survival_band_v2_20261006/survival_pools_20261006.json`, sha `6295400a...`) plus three normal stress models (`alive SD in {0.25, 0.35, 0.50}`, within-world mass/survival $\rho = 0.97$) across the `96 / 72 / 72` design:
  - **Protected-Regression Type-I Error (`<= 0.06` bar):** Passed across all 5 models — worst-cell joint pass rate under a true `-0.05` pooled or `-0.075` single-mix survival regression was **`0.02165`** (Wilson 95% UB `0.02376`) in Phase 1, **`0.02325`** (`0.02543`) in Phase 2, and **`0.03670` / `0.01855` / `0.00750`** in the `SD .25 / .35 / .50` normal models (`research/t34_round0/OC_ROUND0.json`).
  - **Power & Boundary Disclosure at True `+40` Mass (`0` survival change):** Because the point bar `mass gain >= +40` bisects a candidate whose true effect is exactly `+40`, mass-only pass probability at true `+40` was **`0.49855` / `0.49845`** (Phase 1 / Phase 2), and joint mass + survival-band pass probability was **`0.23945` / `0.26175`** (`0.41455` / `0.20120` / `0.05155` for normal `SD .25 / .35 / .50`). Mass-only pass reached **`0.93195` / `0.92905`** at true `+60` and **`0.99865` / `0.99795`** at true `+80` (`research/t34_round0/OC_ROUND0.json`).

### 2. A5/A6 Sizing, Worker Parity & `FIX-FIRST #2` Schedule (`research/t34_round0/`)
- **L4 `A5` Teacher Rate & Mac `A6` Preflight (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/a6-fixes2-01/`):**
  - Saved `A5` L4 benchmarks (`K = 12` CPU inference processes, `1` thread/process, `W1` under `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/`) measured **`57.173` and `54.372` completed worlds/hour/lane** (exceeding the `>= 50` worlds/h floor in [`research/t34_round0/ERRATUM_A5_ADDENDUM_B.md`](../../../research/t34_round0/ERRATUM_A5_ADDENDUM_B.md)).
  - Fresh excluded `A6` (`a6-fixes2-01` at `b60b0e3a`, `2` H500 pairs on CPU `1T`, `23.801 s` wall) timed bare rollout at `9.191 s` (pair 1) / `3.045 s` (pair 2), teacher rollout at `3.943 s` / `5.491 s`, and metric replay at `0.155–0.195 s`/arm, scaling (`10x` H500->H5000, `3x` CPU factor `/ K12`) to three `4:3:3` chunks: **`chunk-000` (`100` pairs, `166 min` est / `332 min` timeout, `11.07 u` cap + `2.21 u` reserve)**, **`chunk-001` (`100` pairs, `166 / 332 min`, `11.07 + 2.21 u`)**, and **`chunk-002` (`40` pairs, `76 / 152 min`, `5.07 + 1.01 u`)** = **`32.64 u` initial / `65.28 u` max liability** (`research/t34_round0/VERIFICATION_FIXES2.json`, [`research/t34_round0/README.md`](../../../research/t34_round0/README.md)).
- **3-Way Physical Worker Parity (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/worker-parity-01/receipt.json`):** Direct serial, 1-worker spawned, and 2-worker spawned executions of the two excluded H500 fixtures produced identical normalized payload SHA-256 **`09d00b7a304d861f880d52e14c8ad9d22567770667ae6991574586110fd12917`** across all six-snake action traces, state hashes, and replay metrics.

### 3. Contingent Round 0-D Build & Pre-Look `KILL / INTEGRITY_FAILURE` Settlement
- **Round 0-D (`research/t34_round0d/`):** Allocated `272` fresh disjoint seeds (`240` selection + `2` A6 + `30` A5-D in `BANK_ROUND0D.json`), reproduced T3 dense census decisions (`env 0` triggers `1913/1918/1923`, `env 1` decision `2592: 1 -> 4`, `env 6` decision `3751: 1 -> 0`), and verified 12-pair wave-quantized straggler scheduling under a hard **`120 u` total ceiling** (`89/89` tests passing in `tests/test_t34_round0d.py`) ([`research/t34_round0d/VERIFICATION_ROUND0D.md`](../../../research/t34_round0d/VERIFICATION_ROUND0D.md)).
- **Round 0 Settlement (`2026-10-21`):** Round 0 halted **pre-look** with **`KILL / INTEGRITY_FAILURE`** on unpublished replay-ledger ordering (`research/t34_round0/ledger/t34-round0-study-01.jsonl`, [`docs/research/final_run_20261019/CAMPAIGN_LOG.md`](../final_run_20261019/CAMPAIGN_LOG.md)). Because the failure was an operational integrity `KILL` rather than a governing scientific `STOP` or `INCONCLUSIVE`, pair outcomes were never opened, `t34_round0_amend` closed unexecuted on pair outcomes, and `t34_round0d` was never activated.

---

## What Was Learned & Why It Closed

1. **Manual ledger publication barriers between phases are brittle:** All 240 pairs sealed `COMPLETE` and the Mac replay passed (24 worlds), but the operator did not publish the replay's ledger append before the sole analysis; `history.require_chunks -> require_published_history` raised as its first check, before any pair record was read, and `analyze` maps `HistoryError` to `KILL` (`docs/research/final_run_20261019/CAMPAIGN_LOG.md`, 2026-10-21 ~00:20 PT). No Round 0 outcome was computed or read.
2. **Delivered-boost guards must be relative to the baseline checkpoint's natural boost rate:** `b8-s3`'s delivered boost share is only `0.12%` (`190,779` decisions; CAMPAIGN_LOG 10-20 ~19:55), making the historical `[0.005, 0.10]` absolute floor (`0.5%`) an unintended automatic `STOP` unless replaced by the relative `0.5 * bare_share <= teacher_share <= 0.10` guard (`research/t34_round0_amend/AMENDMENT_BOOST_FLOOR.json`).
3. **Status — CLOSED (`KILL / INTEGRITY_FAILURE` pre-look, 2026-10-21).** Five `research/t34_round0/` preregistration/erratum markdown files are path-bound in `research/t34_round0/REGISTRATION_ROUND0.json` (four of them also named in `research/t34_round0d/pins.py`); the remaining kept files are listed below.

---

## Kept Files of the Lineage

- [`research/t34_round0/ERRATUM_A5_ADDENDUM_B.md`](../../../research/t34_round0/ERRATUM_A5_ADDENDUM_B.md)
- [`research/t34_round0/ERRATUM_FIXES1.md`](../../../research/t34_round0/ERRATUM_FIXES1.md)
- [`research/t34_round0/ERRATUM_INTEGRITY_SIZING.md`](../../../research/t34_round0/ERRATUM_INTEGRITY_SIZING.md)
- [`research/t34_round0/ERRATUM_MECHANICS_LABEL.md`](../../../research/t34_round0/ERRATUM_MECHANICS_LABEL.md)
- [`research/t34_round0/PREREGISTRATION_ROUND0.md`](../../../research/t34_round0/PREREGISTRATION_ROUND0.md)
- [`research/t34_round0/PREREGISTRATION_ROUND0_FIXES1.md`](../../../research/t34_round0/PREREGISTRATION_ROUND0_FIXES1.md)
- [`research/t34_round0/README.md`](../../../research/t34_round0/README.md)
- [`research/t34_round0d/PREREGISTRATION_ROUND0D.md`](../../../research/t34_round0d/PREREGISTRATION_ROUND0D.md)
- [`research/t34_round0d/VERIFICATION_ROUND0D.md`](../../../research/t34_round0d/VERIFICATION_ROUND0D.md)
