# L2′ FUND_SYSTEMS — independent Fable advice for the owner (2026-10-22, ~16:30 PT)

Read-only. Inputs: `docs/research/results/L2prime_mac_analysis_20261022.json`, the 160 per-world rows in
`big-lever/exp/bl-l2/mac-amendment-01/chunk-0*/shard.json` (my own recount), `terminal.json`, LIVE_FACTS, PLAN_v2, round-11..14
judgments, the opponent-premise judgment, K-1 `timing.json`, L1′ RESULT, SC §4 amendment. Not the round-15 judge; not a release OK.

## 1. How strong is it, and what could mislead

**Strong on sign, moderate on size, and it is not a served number.** The run is clean: 8/8 chunks `TERMINAL_VALID`, mode A
exact-stop + stock, zero veto counters, pinned source, one look, lease released, 5.2 Mac wall-hours. Per-world deltas are **never
negative** (0 of 160) and `common_alive_productivity_delta` ≈ 0: the mechanism is purely a last-300-frame escape when bare B8 is
provably doomed; it changes nothing else (0.07–0.52 actions per 1,000). Baseline deaths: frozen 23 → 10, mixed 11 → 6, scripted
17 → 7 (28 rescues). So the direction is as certain as this campaign gets. Things that make the *magnitude* soft or misleading:

- **Zero-inflated, heavy right tail.** Frozen: 47/64 worlds exactly 0, 17 positive; the top 5 worlds are 61 % of the gain
  (drop top 3 → +36). Mixed: 57/64 zeros, 7 positive, top 5 = 98 % of the gain (drop top 3 → +9.5). A t-interval on this shape is
  liberal; treat LB90 as "comfortably > 0", not as a calibrated floor. Each rescue is worth 130–680 mass because an H5000 fixed
  horizon credits every later frame to one escape.
- **Scripted carries the headline.** Pooled +83.0 is the equal-weight mean of +56 / +28 / +165. Frozen+mixed alone average
  **+41.9** — at the +40 bar, not above it, before any delivery loss. The served Watch game has no scripted opponents.
- **Opponent-roster mismatch is real and unmeasured.** The bank uses historical T3 rosters (Apex checkpoints without veto,
  `random_safe`, `greedy_food`); the served Watch6 roster is 5 × FRP3-s12 **with v8**. Deaths were 85 % `DEATH_SELF`
  (self-enclosure, consistent with the B8 census), which is encouraging for transfer, but `baseline_sealer_strata` is `unresolved`
  for all 51 deaths — the analysis cannot say whether opponents or the hero closed the pocket, which is exactly what decides whether
  v8-avoiding opponents create the same rescuable situations.
- **It is unpaced.** `consumed_before_deadline` is UNAVAILABLE by design; every intervention "completed". Path B's smoke delivered
  0/34 rescues in time. Gain = 100 % rescues ⇒ today's served value of this result is **zero** until delivery is shown.
- Minor: exact-sim certificates assume the server simulator is bit-exact with the search (known float-tie class); continuous-respawn
  Watch values a rescue by remaining life, not by H5000 frames — same sign, different scale.

## 2. The single biggest risk, and the cheapest measurement

**Delivery under the paced clock.** Every unit of this gain sits inside a window between "bare is provably doomed within 300
frames" and "last decision at which an escape still exists". The smoke measured leads of 224–433 frames (9–18 s) and first improving
evaluations arriving 7–30 s later; K-1 (20.1×) + exact stop now give ~0.56 s per common batch at W1 (1.793 batches/s), but a doom
root costs up to 4–6 branches (~2–3 s) and the W4 tail cap is 1.458 s. Break-even arithmetic: pooled +83 needs ≈ 50 % of rescues
delivered to keep a +40 point and ≈ 70 % to keep LB90 > 0 on a fresh bank of this size.

Cheapest retirement, in order, 0 units:
1. **Lead census on the 28 L2′ rescues (0 u, existing shards, Sol ~3 h + Codex-V).** For each rescue, from `batch_diagnostics`:
   first batch where bare dies within H300 → last batch where the chosen alternative still survives → lead in frames; also the
   branch count needed at the deciding batch. Pre-register: if the median lead < 3 × the native doom-batch latency (≈ 150 frames),
   served exact search is closed on this Mac without building anything.
2. If it passes: **WP-4 sparse future-root burst** (already specced, was conditional on K-1 ≥ 5× — now satisfied) + the Sat 2-h
   fixture bars already in PLAN_v2 §3, then a **paced replay of those 28 worlds** (W ≤ 4, K ≤ 8; ≈ 1–2 Mac-h): delivered rescues / 28.
   Bar ≥ 60 %.

## 3. Sequence, next 3–5 days (Thu 22 → Mon 27), and odds

| When | Item | Cost |
|---|---|---|
| Thu night | Lead census (above); register the delivery bar pre-data; un-stop WP-4 build (Sol-C) | 0 u |
| Fri | WP-4 build + Codex-V review; K-1 kernel enabled for any run with vetoed opponents | 0 u |
| Sat AM | WP-4 fixture measurement (2 leased Mac-h, PLAN_v2 bars) | 0 u |
| Sat PM–Sun | Paced replay of the 28 rescue worlds (delivered fraction) | ~2 Mac-h |
| Sun–Mon | **New registration**: 64-world served-roster bank (5 × FRP3-s12+v8, continuous-respawn-like or H5000 mixed), bare vs exact-stop dense, unpaced, native ON — answers "does the rescue value exist in the product game" (~6–10 Mac-h) | 0 u |
| Week 2 | Only if both pass: L1″ → L3 → L4 → strict/product/serving per PLAN_v2 §5; earliest qualification Nov 3–7 | Mac serial |

Honest odds of a strict-gate-qualified, owner-approved served release: **0 % in the next 5 days** (≈ 80 serial Mac-hours remain,
no bar moves). **≈ 10 % by Nov 7**: P(lead census passes) ~.6 × P(≥ 60 % delivered) ~.5 × P(served-roster value at bar) ~.5 ×
P(strict + product + serving all clear mechanically) ~.6. PLAN_v2's 10–15 % stands; this result raised the ceiling's certainty,
not the delivery odds.

## 4. What to stop now

- **Stop all Colab/L4 L2′ work** (native-L4 ruler, mode-B forecasts, replication bank): the Mac did the whole bank in 5.2 h; the sign
  is settled; replication buys nothing until delivery exists.
- **Stop Round 0 re-registration and Round 0-D**: L2′ is the offline ceiling they were meant to measure. Close them.
- **Stop O2 / harvest lanes until delivery is measured**: harvesting needs alternatives evaluated at non-doom roots, which is the
  4–6× cost that exact stop removes; it fights the only serving path that is close to feasible.
- **Stop quoting "+83".** The number the product game resembles is frozen/mixed +28..+56 (LB90 +12/+35) times delivered fraction.
- Keep: B8 s3 served, bars unchanged, P1 on its own slot (independent of this result), round-13 V0 ladder only after the lead census.
