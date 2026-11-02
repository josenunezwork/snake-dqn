# B8 Seed 3 Strict Gate (T7), Serving Qualification & Release (T12/T12b/T12c), Product Roster Audits (T13/T13b), and Mixed Watch Qualification & Default Switch (T15/T15b)

## Question and Goal

This lineage governed the promotion, Mac web-serving qualification, release, product-roster evaluation, and Watch exhibition configuration of the bare `ego2s-bv8` CNN policy **`b8-s3`** (`Ego2sNet`, trained with `v8` labels and receiving 30 live `v8` free-space scalars, served with `veto_variant: none`). It answered four sequential questions:
1. **Strict Gate (`T7`)**: Can the 5-seed `B8` family clear every Tier 1 and Tier 2 promotion clause on a fresh, one-look paired world bank (`5,760` H10000 episodes + `45` H5000 prefix controls) against `frp3-s12 + v8` and `champion_a5 + v8` without a serving veto?
2. **Mac Serving Qualification & Release (`T12`, `T12b`, `T12c`)**: Does the selected candidate (`b8-s3`, sha256 `3e649783...`) preserve bit-exact live observation/action parity, veto absence, fail-closed pin fallback (`b8-s3 -> frp3-s12 + v8 -> champion`), Play/UI/Train contracts, and tick-pacing budgets across 6- and 12-snake Watch and Play sessions on the Mac?
3. **Product Roster Audits (`T13`, `T13b`)**: In continuous-respawn Watch (`H5000`, `25` paired fresh worlds per roster), how does homogeneous `all_b8` compare against legacy `all_frp3` (hero-only `v8`), `all_champion` (hero-only `v8`), and a `mixed` exhibition roster (bare `b8-s3` hero at slot 0 vs `frp3-s12 + v8` opponents at slots `1..N-1`) at 12 snakes (`T13`) and at the 6-snake product default plus 8-snake `all_b8` (`T13b`)?
4. **Mixed Watch Qualification, Default Switch, and Next Research Lanes (`T15`, `T15b`, Oct 17 Advice)**: Can `SNAKE_SERVE_WATCH_ROSTER=mixed` pass full Mac serving qualification (`T15`) and become the owner-approved Watch default (`T15b`), and what training lanes remain once bare `B8` is served?

## Timeline of Steps and Versions

| Date | Step / Version | What Was Run | Verdict |
|---|---|---|---|
| 2026-10-16 | `T7` Fresh-Bank Phase R (`7e173435`) | One-look Mac CPU strict gate (`5,760` H10000 episodes + `45` H5000 controls across `960` fresh worlds in `redesign-t7-phase-r-b8/v1`, `6.828 h` wall time). Candidate rule selected `b8-s3` (`3e649783...`). | **`STRICT_PASS` (Tier 1 & Tier 2 PASS)** |
| 2026-10-17 | Release Decision Advice | Delegated cloud-advisor ruling on `T7`: release `b8-s3` as served Watch hero conditional on an 8-check extended Mac serving qualification (`S1–S8`) and 7-day / `100,000`-decision rollback monitoring. | **RELEASE CONDITIONAL ON `T12`** |
| 2026-10-17 | `T12` Extended Mac Serving Qualification (`70534958`) | Evaluated `b8-s3` across Checks 1–8 (`73` fresh seeds in `SEED_AUDIT_T12_V2.json`, `25` Watch6 H5000 + `25` Play episodes). Checks 1–3 and 5–8 passed; Check 4 failed `Watch12` paced probes (`1/120` overrun at `24 fps` and `1/120` at `17 fps` retry). | **`SERVING_FAIL` / HOLD (Check 4 only)** |
| 2026-10-17 | `T12b` Check-4-Only Repair & Rerun (`25b25bf0`, `23c571de`) | Diagnosed `Watch12` overruns as `27–43 ms` GC pauses overlapping snapshot construction rather than steady inference (`5.14 ms` forward, `1.17 ms` v8 flood). Added server-enforced ego2s speed ceiling (`24 fps` for `<= 6` driven AI, `12 fps` for `7..12`). Re-ran Check 4 (`4` fresh seeds, `1,500` frames + `120`-tick paced probes at `12 fps` and ceiling). | **`SERVING_PASS` (`0/120` overruns)** |
| 2026-10-17 | `T12c` Bare `b8-s3` Release | Flipped `CHECKPOINT_RELEASED_DEFAULT = "b8-s3"`, verified live startup and rollbacks (`frp3-s12`, `champion`), published release record (`b8_s3_release_2026-10-17.md`), and shipped `scripts/monitor_b8_release.py`. | **RELEASED (`b8-s3`)** |
| 2026-10-17 | `T13` 12-Snake Product Roster Audit (`56e20060`) | Descriptive 12-snake `H5000` measurement (`25` paired fresh worlds × `4` rosters = `100` episodes, `6,000,000` decisions, `27.02 min` wall time). Showed `all_b8` had the healthiest whole arena (`0.712` deaths/1k, `1,646.70` mass) but higher hero deaths (`0.768`/1k) than hero-vetoed `all_frp3` (`0.368`/1k); `mixed` improved B8 hero deaths to `0.624`/1k and mass to `190.67`. | **COMPLETED (Independent Review `PASS`)** |
| 2026-10-17 | Post-`T13` Advice & Decision Rule | Noted that `configs/mechanics_v2.yaml` defaults to `6` snakes, not `12`; ordered `T13b` at `6` snakes (plus `all_b8` at `8`), fixed the pre-data Watch roster selection rule (`f36a20d3`), and ordered a per-roster-size monitoring erratum. | **RULE FIXED BEFORE DATA** |
| 2026-10-17 | `T13b` 6-Snake & 8-Snake Roster Audit (`0fbb7c2c`) | Measured `4` six-snake rosters + `8`-snake `all_b8` (`25` paired fresh worlds × `5` cells = `125` H5000 episodes, `4,000,000` decisions, `22.35 min`). At 6 snakes, `all_b8` vs `mixed` hero death gap was `+0.096`/1k (`<= 0.10`), but `all_b8` vs `all_frp3` hero p95 length was `596` vs `723` (`-127` cells, outside `100`). Published `b8_s3_monitoring_erratum_2026-10-17.md`. | **`PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`** |
| 2026-10-17 | `T15` Mixed Watch Roster Qualification (`9cc59403`) | Implemented `SNAKE_SERVE_WATCH_ROSTER=mixed` (`scope=watch_opponents`, `wrapped_ids=[1..N-1]`, hero unwrapped) and ran 6-check Mac qualification (`33` fresh seeds in `t15-mixed-roster-20261017/v1`). Sibling `run-v1-assembly` fixed `ui-000.json` `wall_seconds` (`38.903733 s` from retained UTC logs). | **`SERVING_PASS` (All 6 Checks PASS)** |
| 2026-10-17 | `T15b` Mixed Watch Default Switch | Owner-approved switch `WATCH_ROSTER_RELEASED_DEFAULT = WATCH_ROSTER_MIXED`; verified default and `all_b8` rollback startups (`research/t15b_watch_default_20261017/results/startups.json`); published `b8_s3_release_erratum_mixed_watch_2026-10-17.md` (`182` Python + `178` frontend tests pass). | **RELEASED (`mixed` Watch Default)** |
| 2026-10-17 | Post-Release Next-Lane Advice (`open_next`, `new_lane`) | Cloud-advisor memos diagnosing late-game (`L >= 800`) pocket sealing outside the `31x31` crop (`open_next`) and evaluating GPU-enabled self-play leagues, large teachers, and short-horizon expert iteration (`new_lane`). | **ADVISORY (Superseded by T34 ExIt-B8)** |

## Final Verdict and Key Numbers

- **`T7` Phase R Strict Gate (`STRICT_PASS`)**:
  - **Pooled 5-seed efficacy (`H5000` mass vs `frp3-s12 + v8`)**: **`+138.666306`** (`SE 5.993054`, `LB90 +129.477718 > 0`, `5/5` positive seeds; `H10000` mass `+93.746685`, `SE 13.104930`, `45/45` prefix controls identical).
  - **Tier 1 survival vs `frp3-s12 + v8`**: frozen alive-frame diff **`+0.046215`** (`SE 0.013547`), mixed **`+0.042840`** (`SE 0.012882`), scripted **`-0.006909`** (`SE 0.012254`, threshold `>= -0.03`); absolute episode survival `0.743750` (frozen), `0.785417` (mixed), `0.538542` (scripted) (`>= 0.50`).
  - **Tier 2 survival vs `champion_a5 + v8`**: scripted episode survival diff **`-0.015625`** (`SE 0.022947`, threshold `>= -0.05`, margin `+0.034375`, `960` paired worlds).
  - **Selected candidate**: **Seed 3 (`b8-s3`)**, equal-mix mass effect **`+149.650051`** (`80%` WS lower bound `+131.939914`), checkpoint SHA-256 `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
  - Primary source: [`docs/research/t7_phase_r_result_2026-10-16.md`](../t7_phase_r_result_2026-10-16.md).
- **`T12` + `T12b` + `T12c` Bare `b8-s3` Serving Qualification and Release (`SERVING_PASS` -> `RELEASE`)**:
  - **Live parity & veto absence (Checks 1–3, 5–8)**: `9/9` H5000 and `3/3` H10000 P2 records bit-identical; `2/2` H5000 served-vs-SIMD parity probes (`0` differing frames); `active=False` with `veto_variant: none`; `25` all-B8 Watch6 episodes (`125,000` hero decisions, `53` deaths = **`0.424` deaths/1k**, length p50 `172` / p95 `528` / max `867` / mean `208.22`); `25` Play episodes (`17,878` frames, `0` errors); `1,500` UI confidence frames (`78.47%` confident, `10.53%` moderate, `11.00%` close).
  - **Latency & speed ceilings (`T12b` Check 4 `SERVING_PASS`)**: `GameState.update` p50 / p95 / max (ms) over `1,500` frames: `Watch6` (`6` AI) **`5.967 / 8.048 / 30.710`** (ceiling `24 fps`), `Watch12` (`12` AI) **`12.621 / 15.417 / 42.616`** (ceiling `12 fps`), `Play6` (`5` AI) **`4.589 / 5.790 / 31.626`** (ceiling `24 fps`), `Play12` (`11` AI) **`9.115 / 11.727 / 40.362`** (ceiling `12 fps`); **`0 / 120` overruns** across all 8 default (`12 fps`) and ceiling paced probes.
  - Primary sources: [`docs/research/t12_b8_serving_qualification_2026-10-17.md`](../t12_b8_serving_qualification_2026-10-17.md), [`docs/research/b8_s3_release_2026-10-17.md`](../b8_s3_release_2026-10-17.md), [`research/t12_b8_serving_20261017/results/summary.json`](../../../research/t12_b8_serving_20261017/results/summary.json), [`research/t12b_check4_20261017/results/summary.json`](../../../research/t12b_check4_20261017/results/summary.json), and [`research/t12c_release_20261017/results/startups.json`](../../../research/t12c_release_20261017/results/startups.json).
- **`T13` (12-Snake) and `T13b` (6-Snake & 8-Snake) Product-Roster Measurements (`25` Paired `H5000` Episodes per Cell)**:

| Study & Snakes | Roster (`hero` / `arena`) | Hero Deaths / 1k | Hero Length p50 / p95 / max | Hero H5000 Mass | Hero Boost | Arena Deaths / 1k | Arena H5000 Mass (Per Snake) |
|---|---|---:|---:|---:|---:|---:|---:|
| `T13` (`12` snakes) | `all_b8` (bare) | `0.768` | `96 / 333 / 488` | `121.51` | `0.24%` | `0.712` | `1646.70` (`137.22`) |
| `T13` (`12` snakes) | `all_frp3` (hero `+v8`) | `0.368` | `191 / 653 / 996` | `239.73` | `2.50%` | `2.109` | `1072.01` (`89.33`) |
| `T13` (`12` snakes) | `all_champion` (hero `+v8`) | `0.360` | `150 / 476 / 625` | `183.57` | `12.63%` | `1.389` | `994.93` (`82.91`) |
| `T13` (`12` snakes) | `mixed` (`b8` hero / `frp3+v8` opp) | `0.624` | `143 / 594 / 883` | `190.67` | `0.19%` | `0.729` | `1509.57` (`125.80`) |
| `T13b` (`6` snakes) | `all_b8` (bare) | `0.392` | `189 / 596 / 1078` | `236.91` | `0.15%` | `0.404` | `1324.33` (`220.72`) |
| `T13b` (`6` snakes) | `all_frp3` (hero `+v8`) | `0.184` | `249 / 723 / 1014` | `293.73` | `2.12%` | `1.159` | `824.57` (`137.43`) |
| `T13b` (`6` snakes) | `all_champion` (hero `+v8`) | `0.120` | `290 / 674 / 793` | `311.96` | `9.49%` | `1.015` | `812.59` (`135.43`) |
| `T13b` (`6` snakes) | `mixed` (`b8` hero / `frp3+v8` opp) | `0.296` | `271 / 791 / 1019` | `316.76` | `0.17%` | `0.348` | `1222.94` (`203.82`) |
| `T13b` (`8` snakes) | `all_b8` (bare) | `0.496` | `145 / 555 / 1037` | `195.05` | `0.25%` | `0.498` | `1475.06` (`184.38`) |

  - **Key `T13` (`12`-snake) paired bootstrap differences (`10,000` resamples, seed `20261017`, `summary.json` sha256 `7aadfa65...`)**:
    - `all_b8` vs `all_frp3` (hero): deaths/1k `+0.400 [+0.232, +0.552]`, median length `-95 [-141, -51.975]`, p95 `-320 [-420, -174]`, H5000 mass `-118.220 [-164.867, -73.064]`, kills made `-261` (`39` vs `300`), self-deaths `+36` (`59/96` vs `23/46`).
    - `all_b8` vs `all_frp3` (arena): deaths/1k `-1.397 [-1.482, -1.315]`, H5000 mass `+574.691 [+535.524, +612.970]`.
    - `all_b8` vs `mixed` (hero): deaths/1k `+0.144 [+0.024, +0.264]`, median length `-47 [-73, -23]`, p95 `-261 [-375, -67]`, H5000 mass `-69.162 [-109.264, -32.843]`.
  - **Key `T13b` (`6`-snake) rule evaluation (`RULE_RESULT.json`, `summary.json` sha256 `aac66199...`)**:
    - `all_b8` (`0.392`/1k) minus `mixed` (`0.296`/1k) hero deaths = **`+0.096`/1k** (`95% CI [+0.008, +0.184]`, within `0.10` bar), with `34/49` `all_b8` hero deaths vs `21/37` `mixed` hero deaths caused by self-collision.
    - `all_b8` (`596`) minus `all_frp3` (`723`) hero p95 length = **`-127` cells** (`95% CI [-299, +68.025]`, outside `100`-cell bar) -> **`PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`**.
  - Primary sources: [`research/t13_product_roster_20261017/results/summary.json`](../../../research/t13_product_roster_20261017/results/summary.json), [`docs/research/t13b_six_snake_roster_2026-10-17_results.md`](../t13b_six_snake_roster_2026-10-17_results.md), [`research/t13b_six_snake_20261017/results/summary.json`](../../../research/t13b_six_snake_20261017/results/summary.json), and [`docs/research/b8_s3_monitoring_erratum_2026-10-17.md`](../b8_s3_monitoring_erratum_2026-10-17.md).
- **`T15` + `T15b` Mixed Watch Roster Qualification and Default Switch (`SERVING_PASS` -> `RELEASE`)**:
  - **Parity & Veto Scope (Checks 1–3, 5–6)**: Exact `20,000`-frame world/role parity across `N6/H5000`, `N8/H5000`, and `N12/H10000` (`7,992` active hero decisions); hero slot `0` always bare (`wrapped_ids=[1..N-1]`, `scope=watch_opponents`); `25` fresh `N6/H5000` episodes (`125,000` hero decisions, `34` hero deaths = **`0.272` deaths/1k**, rollback trigger `> 0.544`/1k, length p50 **`248`**, p95 **`841`**, max **`1,153`**; opponents made `625,000` decisions with `3,288` vetoes).
  - **Mixed Watch Latency (Check 4)**: `GameState.update` p50 / p95 / max (ms) over `1,500` frames: `Watch6` **`6.027 / 7.792 / 34.076`** (`1.5×` trigger `11.688 ms`, ceiling `24 fps`), `Watch8` **`7.504 / 9.519 / 32.845`** (`1.5×` trigger `14.279 ms`, ceiling `12 fps`), `Watch12` **`11.329 / 13.141 / 36.511`** (`1.5×` trigger `19.712 ms`, ceiling `12 fps`); **`0 / 120` overruns** across all 6 paced probes.
  - Primary sources: [`docs/research/t15_mixed_roster_qualification_2026-10-17.md`](../t15_mixed_roster_qualification_2026-10-17.md), [`docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md`](../b8_s3_release_erratum_mixed_watch_2026-10-17.md), [`research/t15_mixed_roster_20261017/results/summary.json`](../../../research/t15_mixed_roster_20261017/results/summary.json), and [`research/t15b_watch_default_20261017/results/startups.json`](../../../research/t15b_watch_default_20261017/results/startups.json).

## What Was Learned

1. **Legacy Watch's hero dominance was an artifact of asymmetric `v8` wrapping**: In legacy `all_frp3` and `all_champion` Watch, only hero slot `0` carried the `v8` safety veto while the remaining `5` or `11` copies ran unwrapped, causing massive opponent churn (`2.109` arena deaths/1k at 12 snakes, `1.159`/1k at 6 snakes) that the protected hero farmed (`300` hero kills in `T13`). Homogeneous `all_b8` cut arena deaths by `65–66%` and raised total arena mass by `+53%` (`T13`) to `+61%` (`T13b`), but made the unwrapped hero face much larger, longer-lived opponents.
2. **Pairing a bare `b8-s3` hero with `frp3-s12 + v8` opponents (`mixed`) gave the best hero exhibition without wrapping the hero**: At 6 snakes (`T13b`), `mixed` lowered B8 hero deaths from `0.392` to `0.296`/1k (`0.272`/1k in `T15`), lifted hero median length from `189` to `271` (p95 `596` -> `791`, H5000 mass `236.91` -> `316.76`), and preserved low arena death rates (`0.348`/1k).
3. **`Watch12` high-speed tick overruns were GC pause tails during snapshot serialization, not neural forward saturation**: `T12b` showed `b8-s3` batched forwards (`5.14 ms`) and `v8` floods (`1.17 ms`) took `< 7.5 ms` for 12 snakes, while Python GC pauses (`27.8–42.9 ms`) during snapshot construction caused isolated overruns above `17–24 fps` budgets. Clamping ego2s server speed to `24 fps` (`<= 6` AI) and `12 fps` (`7..12` AI) eliminated overruns (`0/120`) without changing policy math.
4. **Short-length self-collisions (`L ~ 100–190`) vs late pocket sealing (`L >= 800`)**: `T13`/`T13b` and the Oct 17 advisory memos (`2026-10-17_after_t13.md`, `2026-10-17_open_next.md`, `2026-10-17_new_lane.md`) distinguished short-horizon crowding self-collisions (`59/96` hero deaths in `T13`, `34/49` in `T13b`) and low boost usage (`0.15–0.24%`) from late-game pocket sealing outside the `31x31` local crop, motivating the subsequent T34 ExIt-B8 campaign.

## Why It Closed / Current Status

**Closed and Released (2026-10-17); Watch Hero Later Upgraded in T34 (2026-10-30/11-01)**. Bare `b8-s3` remains the live served Play opponent policy (`CHECKPOINT_RELEASED_DEFAULT = "b8-s3"`) and the live fallback controller for Watch. On 2026-10-30/11-01, Watch's primary exhibition path was upgraded to serve the confirmed 132-session `exact-stop-l2-prime` replay bank (`watch-replay-bank-live-132`) with live bare `b8-s3` fallback (`docs/STATUS.md`).

## Kept Lineage Files

- **Strict Gate & Serving Qualification Reports**:
  - [`docs/research/t7_phase_r_result_2026-10-16.md`](../t7_phase_r_result_2026-10-16.md) — `T7` Phase R `STRICT_PASS` report (`b8-s3` selection).
  - [`docs/research/t12_b8_serving_qualification_2026-10-17.md`](../t12_b8_serving_qualification_2026-10-17.md) — `T12` (`HOLD` on Check 4) and `T12b` (`SERVING_PASS`) qualification report.
  - [`docs/research/t15_mixed_roster_qualification_2026-10-17.md`](../t15_mixed_roster_qualification_2026-10-17.md) — `T15` mixed Watch roster qualification report (`SERVING_PASS`).
- **Release Records, Errata, and Roster Results**:
  - [`docs/research/b8_s3_release_2026-10-17.md`](../b8_s3_release_2026-10-17.md) — Original `b8-s3` release record (`sha256: 29b7d71b...`).
  - [`docs/research/b8_s3_monitoring_erratum_2026-10-17.md`](../b8_s3_monitoring_erratum_2026-10-17.md) — Per-roster-size (`Watch6`/`8`/`12`) death-threshold monitoring erratum.
  - [`docs/research/b8_s3_release_erratum_mixed_watch_2026-10-17.md`](../b8_s3_release_erratum_mixed_watch_2026-10-17.md) — Owner-approved `T15b` release erratum switching Watch default to `mixed`.
  - [`docs/research/t13b_six_snake_roster_2026-10-17.md`](../t13b_six_snake_roster_2026-10-17.md) — Pre-registered `T13b` 6-snake Watch roster decision rule (`sha256: 6a79dbfe...`).
  - [`docs/research/t13b_six_snake_roster_2026-10-17_results.md`](../t13b_six_snake_roster_2026-10-17_results.md) — `T13b` 6-snake and 8-snake roster results (`PLAN_MIXED_PENDING_ROSTER_QUALIFICATION`).
- **Kept Specifications**:
  - [`docs/research/specs/t12_b8_serving_qual_spec_2026-10-17.md`](../specs/t12_b8_serving_qual_spec_2026-10-17.md) — `T12` extended Mac serving qualification spec.
  - [`docs/research/specs/t13_roster_spec_2026-10-17.md`](../specs/t13_roster_spec_2026-10-17.md) — `T13` 12-snake product-roster spec (`sha256: 3b779e7d...`).
  - [`docs/research/specs/t13b_six_snake_spec_2026-10-17.md`](../specs/t13b_six_snake_spec_2026-10-17.md) — `T13b` 6-snake product-roster spec (`sha256: cb21c1e2...`).
- **Pinned Protocols & Independent Audit Reviews**:
  - [`research/t12_b8_serving_20261017/protocol.md`](../../../research/t12_b8_serving_20261017/protocol.md) (`sha256: 3d8b5167...`) and [`research/t12_b8_serving_20261017/results/REVIEW.md`](../../../research/t12_b8_serving_20261017/results/REVIEW.md).
  - [`research/t12b_check4_20261017/protocol_check4.md`](../../../research/t12b_check4_20261017/protocol_check4.md) (`sha256: ce753f69...`) and [`research/t12b_check4_20261017/results/REVIEW.md`](../../../research/t12b_check4_20261017/results/REVIEW.md).
  - [`research/t13_product_roster_20261017/protocol.md`](../../../research/t13_product_roster_20261017/protocol.md) — `T13` 12-snake measurement protocol.
  - [`research/t13b_six_snake_20261017/protocol.md`](../../../research/t13b_six_snake_20261017/protocol.md) — `T13b` 6-snake measurement protocol.
  - [`research/t15_mixed_roster_20261017/protocol.md`](../../../research/t15_mixed_roster_20261017/protocol.md) (`sha256: 0b882fc2...`), [`research/t15_mixed_roster_20261017/results/REVIEW.md`](../../../research/t15_mixed_roster_20261017/results/REVIEW.md), and [`research/t15_mixed_roster_20261017/results/UI_ENVELOPE_CORRECTION.md`](../../../research/t15_mixed_roster_20261017/results/UI_ENVELOPE_CORRECTION.md).
