# Big-Lever Served-Tick Timing Diagnostics, Mac Y1/L1-prime Latency Rulers, and Paced Identity (`bl-served-tick`)

## Question / Goal

Can exact search over `b8-s3` be served on the owner's Mac at the 24 fps Watch6 cadence? This lineage built the measurement instruments rather than an efficacy result: a three-oracle paced-identity harness (`bl_paced_identity`, X7 and Round-09 Z-2), a Mac dense H300 search ruler plus co-resident L1′ latency diagnostics (`bl_mac_latency`, Y1/L1′), a fixed served-tick timing panel (`served_tick_probe.py`) with host-load, desktop-application and remote-supervision admission policies, and an opt-in authoritative-game / inspector-DTO process split (`dto_split_w0_k0`). None of these packages claims search efficacy or grants serving/release authority.

## Timeline of Versions / Steps

| Date | Step / Package | What Was Run / Changed | Verdict / Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | `research/bl_paced_identity` (X7) | Built advisor 03's three-oracle paced package; Mac `preflight-04` took `1.078488625 s` (one spawned worker, 26 synthetic interleaving fixtures, eight unpaced GridBatchSim ticks). The production 18+96 root bank was **not** run. Final X7-only pytest: `43 passed in 1.38 s` (`PREFLIGHT.md`, removed in consolidation; see git c4c4ab41). | `BUILD_VERIFIED_NOT_QUALIFIED` (no paced study, no Colab call) |
| 2026-10-21 | Round-09 Z-2 (`ROUND09_Z2.md`) | Added two required integer-nanosecond request fields (`publish_cutoff_ns` = target tick − 2 ticks; `consume_cutoff_ns` = target tick + 1 tick) and the four-case fake-clock `clock_seam` proof (on-time, publication-late, consumption-late, wrong-frame); package test selection passed `72` tests in `2.76 s` (removed in consolidation; see git c4c4ab41). | Implementation only; no paced runs or timing measurements |
| 2026-10-21 | `research/bl_mac_latency` (Y1 ruler & L1′ cells) | 270 actual dense H300 `ExactT3Operator` batches (9 frozen fixtures × 30, CPU `W=1`) in `y1-mac-01/mac-ruler.json`, plus 10 CPU/MPS L1′ stress cells (`cells-01.json`). The Y1 run overlapped foreign research workers; quiet attempt 02 refused before data (mediaanalysisd/Spotlight) and attempt 03 stopped after five batches when analyticsd used 63.7% of one core. | `NOT_READY(INSTRUMENT_MISSING)`; nominated cell: **none**; Y1 quiet-host acceptance `NOT_READY` |
| 2026-10-26/27 | `SERVED_TICK_README.md` (Build 0) | Froze the bare served Watch6 timing panel: three seeds × four arms (timer-only, inline Watch6, fresh-process Watch6, real Hub/WebSocket) = 12 cells, 200 warm-up + 2,400 measured ticks per cell at 24 fps; PASS requires zero absolute overruns and `GameState.update` p95 `<= 11.688 ms` in every required cell; base 60 / effective 120 min runaway envelope. Added an opt-in W1/K1 contention arm. | Engineering diagnostic only |
| 2026-10-27 | `HOST_LOAD_AMENDMENT_20261027.md` (`classified-v1`, `--classified-host-load`) | Classified host processes: foreign workload/unknown processes at `>= 20%` of one core abort the attempt; macOS service + renderer background aborts only at strictly `> 150%` of one core for strictly `> 60 s`. No process is stopped, reniced or disabled to manufacture quiet conditions. | Prospective amendment for opted-in attempts |
| 2026-10-27 | `ASTRA_DESKTOP_RULING_20261027.md` + `DESKTOP_APPS_AMENDMENT_20261027.md` (`classified-desktop-v1`) | Attempt `panel-argon-classified-08` became INVALID after `259.08 s` (cells 00–02 of 12) on an Outlook process at `26.2%` of one core; attempt 07 became INVALID after `24.38 s` on a `ps` process at `28.0%`. Astra conditionally admitted an explicitly enrolled `desktop_app` class inside the shared `> 150%` / `> 60 s` background ceiling. | Attempts 07 and 08 remain INVALID; desktop policy conditionally admitted |
| 2026-10-27 | `REMOTE_SUPERVISION_AMENDMENT_20261027.md` (`classified-supervised-v2`) | Offline evidence implementation for co-running a supervisor; the existing attempt-07 root (PID 7425) could not be made admissible without prohibited live attach/restart, so the implementation refuses live supervision. | `IMPLEMENTATION INCOMPLETE — LIVE ADMISSION UNAVAILABLE` |
| 2026-10-27 | `SERVED_DTO_SPLIT_README.md` (`dto_split_w0_k0`; removed in consolidation, see git c4c4ab41) | Opt-in arm moving stock inspector forwards, `build_frame` and WebSocket-shaped JSON construction into a second spawned interpreter; new manifest `served_dto_split_manifest.json` keeps the same three seeds and 200 + 2,400 tick exposure. Final scoped suite `501 passed, 1 skipped in 13.57 s` (21 new split checks). | Implementation handoff; the panel was **not** run |
| 2026-10-28 | `SUPV2_WIRING_20261028.md` (`classified-desktop-supervised-v2`) | Composed the desktop and supervision policies offline (host schema `served-tick-host/v5`); no new aggregate ceiling. | `PREPARATORY IMPLEMENTATION ONLY`; live co-running admission and the 600-second qualification unavailable |

## Final Verdict and Key Numbers

- **No efficacy or serving result in this lineage.** Every package here is an engineering instrument or diagnostic; none claims search efficacy, qualifies serving latency, or authorizes a serving change (source READMEs and amendments listed below).
- **Mac Y1 dense H300 ruler & L1′ diagnostic (`research/bl_mac_latency/RESULT_L1.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/`):**
  - **Owner L1′ bars:** expanded `S p99`: `W2 <= 0.830 s`, `W3 <= 1.250 s`, `W4 <= 1.458 s`; W1/K1 `C p99` ceilings `333/542 ms` (unrounded `333.333/541.667 ms`, i.e. 2.4 or 1.6 roots/s); coverage `>= 29` (or `>= 20` sensitivity) valid consumptions per 12 s burst; zero overruns and Watch6 update p95 `<= 11.688 ms`.
  - **Y1 CPU `W=1` H300 batch latency `p50 / p95` ($n=30$/fixture, contended run; `y1-mac-01/mac-ruler.json` sha256 `502f1eb6...`):** `frozen-800` **`10.557 / 10.607 s`**, `frozen-1200` **`22.749 / 23.311 s`**, `frozen-1500` **`21.221 / 25.814 s`**, `mixed-800` **`7.603 / 7.642 s`**, `mixed-1200` **`10.823 / 10.925 s`**, `mixed-1500` **`10.332 / 10.382 s`**, `scripted-800` **`0.611 / 0.616 s`**, `scripted-1200` **`0.649 / 0.681 s`**, `scripted-1500` **`0.595 / 0.626 s`**.
  - **L1′ stress cells (`cells-01.json` sha256 `b8983ab8...`):** `S p99` was **`22.580–27.244 s`** on CPU (`update p95` `39.744–48.271 ms`, `280–390` burst overruns) and **`34.600–41.202 s`** on MPS (`update p95` `30.405–41.039 ms`, `291–364` burst overruns). The probes had no production root provider, so registered coverage is `NOT_MEASURED`. Verdict: **`NOT_READY(INSTRUMENT_MISSING)`**, measured with the stock-Python opponent-veto hook before the native S2/K-1 veto was enabled (`research/bl_mac_latency/README.md`).
- **Three-oracle paced identity (`research/bl_paced_identity/README.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-x7/preflight-04/`):**
  - `preflight-04` exports: `receipt.json` sha256 `47340cbb...`, `manifest.json` sha256 `4bff4aab...`. Preflight PASS is `PRECHECK_ONLY`; the joined receipt remains INCOMPLETE, L3 identity prerequisites BLOCKED, paced oracle NOT_RUN.
  - Round-09 Z-2 added the two-cutoff integer-nanosecond deadlines and the four-case `clock_seam` proof (`72` tests in `2.76 s`); the earlier preflight exports do not verify the changed implementation.
- **Served-tick panel admission (`research/bl_served_search/SERVED_TICK_README.md`, `ASTRA_DESKTOP_RULING_20261027.md`):** classified attempts 07 and 08 ended INVALID on foreign-load guards before completing the 12-cell panel (08 completed `cell-00`–`cell-02` in `259.08 s`); no cells may be salvaged, rescored or combined. The DTO split and combined-policy wiring were implemented and tested offline but no panel result is recorded in these sources.

## What Was Learned

1. Without a qualified production reachable-root provider, the L1′ stress cells cannot measure valid consumption latency `C` or coverage; micro-timing alone yields `NOT_READY` and no nominated cell (`RESULT_L1.md`).
2. With the stock-Python opponent-veto hook, `frozen` and `mixed` H300 batches had p50 `7.6–22.7 s` versus `0.60–0.65 s` for `scripted` (roughly `12×–38×` slower), on a contended host (`RESULT_L1.md`).
3. Reliable Mac timing needs explicit host-load classification rather than manufacturing a quiet host: per-process `>= 20%` guards for foreign/unknown work, a shared `> 150%` / `> 60 s` background ceiling for macOS services, renderers and enrolled desktop apps, and authenticated evidence before any supervisor may co-run (`HOST_LOAD_AMENDMENT_20261027.md`, `DESKTOP_APPS_AMENDMENT_20261027.md`, `REMOTE_SUPERVISION_AMENDMENT_20261027.md`).

## Why It Closed / Current Status

The instruments and policies remain in `research/bl_served_search/`, `research/bl_mac_latency/` and `research/bl_paced_identity/`. As of these sources, the full bare served-tick panel had not completed validly, live remote-supervision admission was unavailable, and the DTO-split panel had not been run. Release remains the owner's decision.

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
