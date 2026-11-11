# Big-Lever Served Search Ceiling (`L0′` / `L2′`) and PLAN_v3 Follow-Up Packages (`M0` / `B2` / `C1`), plus `X5` Bare Mac Timing (`bl-served-search-l2`)

**Dates:** 2026-10-21 to 2026-10-22
**Verdict:** Offline exact-stop 300-step search (`L2′`) beat bare `b8-s3` by `+83.00` pooled H5000 mass on `160` paired development worlds (`FUND_SYSTEMS`), while L4 CPU placement was `NOT_DISPATCHABLE`, live co-resident serving failed `M1` (`KEEP_B8`), `B2`/`C1` stayed prepared without running, and `X5` ended `INCOMPLETE` / `NOT_READY`.

## What was tried
- Tested `L2′` (development-only offline ceiling for the unchanged T3 dense-search rule: check every fifth decision at length `>= 600`, evaluate legal actions to depth `300` with `exact_stop` early exit on the first survivor, vs bare B8 s3) across `160` paired H5000 worlds (`64` frozen, `64` mixed, `32` scripted).
- Ran `L0′` saved-smoke deadline bookkeeping on `80` last-avoidable-decision (`LAD`) records (`34` rescue-capable, `41` delay-only) and measured L4 vs Mac native+exact-stop batch throughput against the `R >= 2.18` batches/s admission bar.
- Ran `M0` (rescue lead-time census on the `160` `L2′` worlds) and prepared `B2` (`32` Watch6 `14,400`-decision trajectories for `L >= 800` natural escape roots), `C1` (`24` paired `14,400`-frame Watch6 sessions), and `X5` (`research/bl_x5`, bare-B8 Mac timing adapter).

## What happened
- **Throughput rulers:** Stock-hook L4 search measured `0.166` batches/s vs the `2.18` admission bar (`NOT_DISPATCHABLE` on L4), whereas the Mac native+exact-stop ruler passed with zero-tolerance chosen-action parity, admitting Mac `L2′` execution after correcting the opponent-veto premise (`veto_slots=()`).
- **`L2′` offline ceiling (`FUND_SYSTEMS`, development only):** Cleared the `>= 40` pooled point and `LB90 > 0` frozen/mixed bars: pooled-equivalent `+83.00` (`+83.0`, `LB90 +64.12` / `+64.1`, `UB90 +101.87`), frozen `+56.12` (`LB90 +34.81`), mixed `+27.60` (`LB90 +12.11`), scripted `+165.27` (`LB90 +114.77`).
- **`L2′` survival & edits:** Pooled no-death delta was `+0.198` (`LB90 +0.153`) and alive-frame fraction `+0.075` (`LB90 +0.058`), changing `54` / `21` / `79` actions (`0.18` / `0.07` / `0.52` per `1,000` decisions) for `13` / `5` / `10` (`28` total) productive rescues and `0` newly caused deaths.
- **Follow-ups:** `L0′` logged `0/62` own-deadline hits (`29/34` D300 reachable at `1` s). `M0` recorded `L* = 72`, `F = 0.944`, before co-resident serving failed `M1` (`FINAL FAIL` / `KEEP_B8`). `B2` (`52` mocked tests) and `C1` were `PREPARED, NOT RUN`; `X5` stopped after `55.12 s` for overlapping the K-1 audit (`INCOMPLETE` / `NOT_READY`).

## Why
Evaluating bare B8 first and stopping at the first H300 survivor (`exact_stop`) preserved exact chosen actions while cutting Mac runtime enough to measure `160` worlds; the `+83.00` gain came from sparse (`0.07–0.52` per `1,000` moves) self-trap rescues with `0` newly induced deaths, while stock-hook L4 CPU rollouts (`0.166` batches/s) missed the `2.18` threshold.

## What to reuse / what not to repeat
- Reuse `exact_stop` ordering (test bare B8 action first, then descending Q, stopping on the first H300 survivor) for zero-tolerance speedup of survival-ranked rollout teachers.
- Ensure snapshot-based branch workers serialize the immutable `AppConfig` alongside simulator state (`WatchDriver.snapshot()` erratum in `B2`); otherwise fresh spawned workers revert to mechanics v1 defaults and fail rule checks.
- Quote `frozen`/`mixed` gains (`+28` to `+56` times delivered fraction) rather than the `+83` pooled headline, and do not place uncompiled Python search loops on Colab L4 vCPUs.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-served-search-l2.md` | Primary sources: [L2prime_mac_analysis_20261022.json](../results/L2prime_mac_analysis_20261022.json), [PREREGISTRATION_L2.md](../../../research/bl_served_search/PREREGISTRATION_L2.md), [bl_served_search/README.md](../../../research/bl_served_search/README.md)
