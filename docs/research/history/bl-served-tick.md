# Big-Lever Served-Tick Lookahead Assay, Mac Y1/L1-prime Latency Rulers, and Paced Identity (`bl-served-tick`)

## Question / Goal

Establish whether inference-time lookahead over `b8-s3` improves 6-snake H5000 performance (`bl-served-tick`) and can execute within the Mac 24 fps (`41.67 ms/frame`) serving budget under rigorous three-oracle paced identity (`bl_paced_identity`, X7/Z-2), quiet-host dense H300 search rulers (`bl_mac_latency`, Y1/L1'), host-load/supervisor/desktop pause discipline (`served_tick_probe.py`), and authoritative-game vs inspector DTO process separation (`SERVED_DTO_SPLIT_README.md`).

## Timeline of Versions / Steps

| Date | Step / Package | What Was Run / Changed | Verdict / Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | `research/bl_paced_identity` (X7 & Round-09 Z-2) | Built the three-oracle paced search harness (`PacedController`, `ReplayGame`, 18+96 root semantic check), Mac `preflight-04` (`1.078 s`, 43 tests), and Round-09 Z-2 two-cutoff nanosecond deadlines (`publish_cutoff_ns = T_r - 2 ticks`, `consume_cutoff_ns = T_r + 1 tick`) + 4-case source-bound `clock_seam` proof (72 tests in `2.76 s`) | `BUILD_VERIFIED_NOT_QUALIFIED` (`0 u` Colab) |
| 2026-10-21 | `research/bl_mac_latency` (Y1 & L1' Ruler) | Measured 270 dense H300 `ExactT3Operator` batches across 9 frozen Y1 fixtures (`y1-mac-01/mac-ruler.json`) and 10 CPU/MPS L1' co-resident Watch6 cells (`cells-01.json`) before native `v8` optimization; quiet repeats 02/03 stopped on macOS background daemons (`mediaanalysisd`, `analyticsd`) | `NOT_READY(INSTRUMENT_MISSING)`; nominated cell: **none** |
| 2026-10-21 | Offline Served-Tick Efficacy Assay (`BL_SERVED_TICK_RESULTS.md`) | Evaluated 1-step root-to-child tick lookahead (`T1-M0` over `b8-s3`, `T1-C1` over `C1`) on 64 paired H5000 worlds/mix (`192` worlds) in the 6-snake served arena (`cap=10400`) | **`POSITIVE_LEVER`** (`T1-M0 - M0 = +45.67` equal-mix mass, `-36.4%` mixed deaths/1k) |
| 2026-10-27 | `SERVED_TICK_README.md`, `HOST_LOAD_AMENDMENT`, `REMOTE_SUPERVISION_AMENDMENT`, `ASTRA_DESKTOP_RULING`, `DESKTOP_APPS_AMENDMENT` | Built Mac 24 fps `served_tick_probe.py` (5,400 ticks across `BARE_B8`, `H8_K2_D16`, `H16_K2_D16`, `H16_K3_D32`, `H24_K3_D48`); invalidated `r1` (`163.93%` host CPU from `cargo check`) and `r2` (`115.63%` host CPU) for `H16_K3_D32` sign-off while confirming `H24_K3_D48` latency elimination (`p95 = 48.03–48.20 ms > 35.0 ms`); added `--pause-other-agent-work`, `--pause-remote-supervisors`, and allowlisted `--pause-desktop-apps` (`38` processes, `41.3%` ps CPU in `dryrun-desktop-20261027`) | `H24_K3_D48` eliminated; host-load/desktop guards ratified |
| 2026-10-27/28 | `SERVED_DTO_SPLIT_README.md` & `SUPV2_WIRING_20261028.md` | Wired `tools/ops/supervision_v2.py` (`/tmp/snake-dqn-supervision.lease`, `maxproc` headroom) into `served_tick_probe.py`; implemented `dto_split_w0_k0` (separate worker process for inspector forwards/DTO serialization + 4 Hz inspector stride `SNAKE_INSPECTOR_DTO_STRIDE=6`, 501 tests passed in `13.57 s`) | Cut `H16_K3_D32` `dto_serialize_ms` `5.65 -> 0.94 ms` and 1,800-tick `p95` `37.65 -> 33.24 ms` (`<= 35.0 ms`) |

## Final Verdict and Key Numbers

- **Served-distribution 1-step tick lookahead efficacy (`research/big_lever_20261021/BL_SERVED_TICK_RESULTS.md`):**
  - **`T1-M0` vs `M0` (`b8-s3`):** **`+45.67` equal-mix paired H5000 mass** (`scripted +53.97`, `frozen +34.55`, `mixed +48.49`), clearing both the $\ge +20$ equal-mix bar and $\ge +15$ mixed bar, while cutting deaths per 1k alive frames by **`-26.6%`** (`scripted`), **`-23.1%`** (`frozen`), and **`-36.4%`** (`mixed`: `1.316` vs `2.070`).
  - **Why `N=1` (`cap=1000`) `B2` failed previously:** In solo `N=1` (`cap=1000`), `b8-s3` already survived `95.6%` of episodes at `997.43 / 1000` mean mass (zero headroom), whereas in the 6-snake served arena (`cap=10400`) `b8-s3` dies `1.32–2.07` times per 1k frames.
  - **Interaction with `C1` (`T1-C1`):** `T1-C1 - M0 = +20.23` equal-mix (`+42.98` mixed, `-12.88` scripted), showing lookahead rescues `C1` on mixed (`-5.73 -> +42.98`) but `T1-M0` (`+45.67`) strictly dominates.
- **Mac Y1 dense H300 ruler & L1' co-resident Watch6 diagnostic before native `v8` (`research/bl_mac_latency/RESULT_L1.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/`):**
  - **Owner L1' bars:** Admission-to-visibility `S p99`: `W2 <= 0.83 s`, `W3 <= 1.25 s`, `W4 <= 1.458 s`; valid consumption `C p99 <= 333.33 ms` (`2.4 roots/s`) or `541.67 ms` (`1.6 roots/s`); $\ge 29$ or $\ge 20$ valid consumptions per 12 s burst; `0` frame overruns and Watch6 `update p95 <= 11.688 ms`.
  - **Y1 CPU `W=1` H300 batch latency `p50 / p95` ($n=30$/fixture, `y1-mac-01/mac-ruler.json` sha256 `502f1eb6...`):** `frozen-800` **`10.557 / 10.607 s`**, `frozen-1200` **`22.749 / 23.311 s`**, `frozen-1500` **`21.221 / 25.814 s`**, `mixed-800` **`7.603 / 7.642 s`**, `mixed-1200` **`10.823 / 10.925 s`**, `mixed-1500` **`10.332 / 10.382 s`**, `scripted-800` **`0.611 / 0.616 s`**, `scripted-1200` **`0.649 / 0.681 s`**, `scripted-1500` **`0.595 / 0.626 s`**.
  - **L1' co-resident stress cells (`cells-01.json` sha256 `b8983ab8...`):** `S p99` was **`22.580–27.244 s`** on CPU (`update p95` `39.744–48.271 ms`, `280–390` burst overruns) and **`34.600–41.202 s`** on MPS (`update p95` `30.405–41.039 ms`, `291–364` burst overruns). Verdict: **`NOT_READY(INSTRUMENT_MISSING)`** (motivated the `K-1` native `v8` kernel).
- **Three-oracle paced identity (`research/bl_paced_identity/README.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/preflight-04/`):**
  - Enforced two-cutoff integer-nanosecond deadlines (`publish_cutoff_ns = T_r - 2 ticks`, `consume_cutoff_ns = T_r + 1 tick`) and a 4-case source-bound `clock_seam` proof (`on-time`, `publication-late`, `consumption-late`, `wrong-frame`), verified by **`72` passing tests in `2.76 s`** (`preflight-04/receipt.json` sha256 `47340cbb...`, `manifest.json` sha256 `4bff4aab...`).
- **Mac 24 fps served-tick probe & DTO split (`research/bl_served_search/SERVED_TICK_README.md`, `ASTRA_DESKTOP_RULING_20261027.md`, `SUPV2_WIRING_20261028.md`):**
  - **24 fps pass bars (5,400 ticks = 3 mixes $\times$ 3 seeds $\times$ 600 frames):** `p95 <= 35.0 ms`, `p99 <= 41.67 ms`, `frac_gt_41_67ms <= 0.01`, `count_gt_50ms == 0`, and $L \ge 600$ `p95 <= 38.0 ms`.
  - **Candidate elimination & DTO split:** `H24_K3_D48` was eliminated (`p95 = 48.03–48.20 ms > 35.0 ms`, max `83.27 ms`). Splitting inspector payload generation (`SNAKE_INSPECTOR_DTO_STRIDE=6` / `dto_split_w0_k0`, verified by `501` tests in `13.57 s`) cut `dto_serialize_ms` from **`5.65 ms` to `0.94 ms`** (`p95` `7.89 -> 4.92 ms`), reducing `H16_K3_D32` 1,800-tick `p95` from **`37.65 ms` (`FAIL`) to `33.24 ms` (`PASS` $\le 35.0\text{ ms}$, max `40.47 ms`)**, and achieving **`p95 = 34.43 ms`** (`p50 = 16.58 ms`, `p99 = 38.68 ms`, `0` frames $> 50\text{ ms}`, `0.20% > 41.67 ms`, max `45.28 ms`) over 5,400 quiet-host ticks (`served-tick-20261028-dto-s6`).

## What Was Learned

1. Evaluating survival/search levers in `N=1` (`cap=1000`) was a structural false negative because `b8-s3` saturates the 1,000-mass cap at `95.6%` survival; in the 6-snake served arena (`cap=10400`), even 1-ply tick lookahead (`T1-M0`) adds `+45.67` mass and cuts mixed death rate by `36.4%`.
2. Before native `v8` kernel compilation (`S2`/`K-1`), stock-Python 5-opponent `v8` flood-fill vetoes made `frozen` and `mixed` H300 batches $12\times$–$38\times$ slower (`7.6–25.8 s`) than `scripted` (`0.60–0.68 s`).
3. Reliable Mac 24 fps timing requires dual-counter host-load gates (`host_non_idle_cpu_pct <= 100%` and `other_processes_cpu_pct_est <= 25%`), lease-fenced `SIGSTOP`/`SIGCONT` supervision of background agent/Colab/desktop processes (`supervision_v2.py`), and decoupling 24 fps game stepping from 4 Hz inspector DTO serialization.

## Why It Closed / Current Status

**Completed and integrated into the Watch6 confirmation line.** The positive `T1-M0` result (`+45.67` mass) launched the 2-ply `bl-served-search-l2` (`+106.79` H5000 / `+131.20` H10000 mass) and `watch6-confirmation` (`exact-stop-l2-prime`, promoted to Watch replay serving on 2026-11-01) studies, while `supervision_v2.py` and the DTO split remain in the repository.

## Kept Files in This Lineage

- [`research/bl_mac_latency/README.md`](../../../research/bl_mac_latency/README.md)
- [`research/bl_mac_latency/RESULT_L1.md`](../../../research/bl_mac_latency/RESULT_L1.md)
- [`research/bl_paced_identity/README.md`](../../../research/bl_paced_identity/README.md)
- [`research/bl_served_search/ASTRA_DESKTOP_RULING_20261027.md`](../../../research/bl_served_search/ASTRA_DESKTOP_RULING_20261027.md)
- [`research/bl_served_search/DESKTOP_APPS_AMENDMENT_20261027.md`](../../../research/bl_served_search/DESKTOP_APPS_AMENDMENT_20261027.md)
- [`research/bl_served_search/HOST_LOAD_AMENDMENT_20261027.md`](../../../research/bl_served_search/HOST_LOAD_AMENDMENT_20261027.md)
- [`research/bl_served_search/REMOTE_SUPERVISION_AMENDMENT_20261027.md`](../../../research/bl_served_search/REMOTE_SUPERVISION_AMENDMENT_20261027.md)
- [`research/bl_served_search/SERVED_TICK_README.md`](../../../research/bl_served_search/SERVED_TICK_README.md)
- [`research/bl_served_search/SUPV2_WIRING_20261028.md`](../../../research/bl_served_search/SUPV2_WIRING_20261028.md)
