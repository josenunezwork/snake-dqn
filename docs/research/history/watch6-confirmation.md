# Watch6 Live `GameState` Search Confirmation (`v1–v3`), Replay Bank Release, and All-Length (`L >= 1`) Extension

**Dates:** 2026-10-30 to 2026-11-01
**Verdict:** Confirmed that 300-step exact-stop lookahead (`exact-stop-l2-prime`) cuts hero deaths and boosts mass in live Watch6 (`GO_EFFICACY_ONLY`), shipped the 132-session precomputed Watch replay bank, and launched confirmation for all-length search (`Watch6-L1-V1` `IN FLIGHT`).

## What was tried
- Tested `exact-stop-l2-prime` (depth `H = 300` exact survival search every 5th frame when body length `L >= 600`) against bare `b8-s3` (the served convolutional snake network with no wrapper) across `N = 96` fresh paired `H = 14,400`-frame (`10 min` at `24 fps`) Watch6 sessions (`1` hero vs `5 × frp3-s12 + v8(lambda = 8)` opponents), mirrored cell-for-cell against `GridBatchSim`.
- Built precomputed `.watch.gz` deterministic replay banks (`research/watch_replay/`) so browser Watch serves confirmed search-assisted play at `24 fps` with live bare `b8-s3` fallback during cold startup verification.
- Tested two shortcuts for remaining deaths: `R6` (a 64-world / 384-root census and classifier trigger for safe mass harvesting below `L < 600`) and `R7` (depth-2, `K = 15`-tick deterministic spatial macro-options on 1-step-all-die states).
- Tested `exact-stop-l1` (removing the `L >= 600` gate to run `H = 300` search at all lengths `L >= 1` every 5th frame) on `48` paired `H = 6,000` Watch6-v2 sessions, then launched the 3-arm `Watch6-L1-V1` confirmation (`N = 96`, `H = 14,400`).

## What happened
- **Watch6 `V2`:** Ended `INVALID_OR_INCOMPLETE` (`88/96` sessions exported after shard 5 lost 3 attempts to pre-start staging/RPC failures; all `88` pairs stayed sealed with `0` outcomes read).
- **Watch6 `V3` (`GO_EFFICACY_ONLY`, `ELIGIBLE_FOR_REPLAY_QUALIFICATION`):** Passed across `96/96` pairs (`2,764,800/2,764,800` mirror frames, `0` mismatches): `deaths / 1k frames` fell `0.3248 → 0.1997` (`449 → 276`, `-0.1251` / `-0.125`, `-38.5%`, `CI95 [-0.1535, -0.0968] < 0`), and pooled mass rose `399.92 → 654.57` (`+63.68%` / `+63.7%`, Fieller `CI95 [+49.79%, +78.50%] >= -5.0%`).
- **Replay serving (`SERVED IN WATCH`):** Qualified `132/132` `H = 6,000` recordings (`watch-replay-bank-live-132`) on Mac at `24 fps` with cold readiness `0.349–0.435 s` (`< 0.45 s`), live-fallback `update p95 = 8.78 ms <= 11.688 ms`, and `0/120` paced overruns (`step p99 = 11.34 ms <= 41.67 ms`).
- **Shortcuts (`R6`, `R7`) & `exact-stop-l1`:** `R6` (`KILL`: `7.69%` lethal flips, OOF ROC-AUC `0.474–0.521`) and `R7` (`KILL` on `3/5` bars: `+3.85 pp < +10.0 pp` lift, `2 > 0` ordinary lethal flips) failed.
- **All-length search (`ADOPT`):** `exact-stop-l1` cut `L < 600` deaths `34 → 1` (`deaths/1k` `0.1597 → 0.0521`, `-67.4%`; mass `461.16 → 559.43`, `+98.27` / `+21.31%`) at `2.821×` branches (`1,268.5` vs `449.6`), launching `Watch6-L1-V1` (`IN FLIGHT`).

## Why
Most deaths remaining under `exact-stop-l2-prime` (`34` of `46` in the 48-session screen) occurred before length `600`, where local scalar classifiers failed (`0.474–0.521` ROC-AUC), whereas exact `H = 300` rollouts at `L >= 1` caught early self-traps directly; worst-case live search batches (`4.018 s` on L4) required `.watch.gz` replays to keep `24 fps` smooth.

## What to reuse / what not to repeat
- Reuse `.watch.gz` verified replay banks (`research/watch_replay/`) with cooperative GIL-yielding verification (`<= 20%` duty cycle) to serve slow lookahead controllers at `24 fps` with automatic live `b8-s3` fallback.
- Size Colab shard wall caps from empirical p95 runtimes (`7,557.99 s/pair`) rather than theoretical physical-maximum branch counts, which forecast `49.8 h` against a `22 h` ceiling and blocked all `18` candidate layouts.
- Keep incomplete confirmation banks (`88/96` in `V2`) sealed and unread rather than salvaging partial shards; do not repeat static scalar triggers (`R6`) or short 2-step spatial macro-options (`R7`) for `L < 600` or all-die escapes.

## Full record
Original: `git show 90b3db94:docs/research/history/watch6-confirmation.md` | Primary sources: [REGISTRATION_V3.md](../../../research/watch6_search_efficacy_v3/REGISTRATION_V3.md), [watch_replay/README.md](../../../research/watch_replay/README.md), [REGISTRATION_L1_V1.md](../../../research/watch6_l1_confirm/REGISTRATION_L1_V1.md)
