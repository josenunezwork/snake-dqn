# Big-Lever F0 Hindsight Label Factory, F1 Placement Ruler, and D256/D1024/D4096 Production

**Dates:** 2026-10-23 to 2026-10-25
**Verdict:** Closed with Mac tranches `D256` (`1,566` units across `864` parents) and `D1024` (`3,865` units across `2,160` rows) `COMPLETE`, while `D4096` was sealed `NOT_RUN_BY_RULE` after `F4` Look `R-A` `STOP_FUNDING`, and both the L4 CPU tail lane and GPU label premise (`Step -1`, `CLOSE_NO_BOUND`) closed.

## What was tried
- Built the `F0` hindsight label factory (`research/bl_labels`) for bare `b8-s3` across S-gate (`H=5,000`) and `Watch6` (`H=14,400`), branching counterfactual `H=300` actions from 20-step ring snapshots at up to `90` doom + `8` ordinary roots/life with `10%` bare-reroll and `10%` `H=1,440` continuation audits.
- Ran Colab CPU Canary A (`36` session parents), prepared the `F1` 64-root placement ruler (Colab CPU vs 12-vCPU L4), and calibrated Mac `W12` vs `W6` throughput (`>= 1.20` bar) plus rolling-median slowdown and `>= 30 s` foreign-process persistence guards.
- Produced nested Mac label tranches `D256` (`864` parents: `192` fit + `64` reserve lives), `D1024` (`2,160` fold `1–4` parents), and early-started sealed `D4096` (`7,020` fold `1–4` rows), while testing incremental ring trimming (`equiv-02`) and the `Step -1` GPU vs Mac `W12` speed ruler (`>= 4x` bar).

## What happened
- **Canary A (`INCOMPLETE_BUT_INFORMATIVE`) & `W12` calibration (`PASS`):** Canary A verified `44/44` receipt-listed payloads (`31/36` parents in `6,465.88 s`, `1,012` restores, `82` bare rerolls, `0` mismatches; `1.2%` restore overhead). Paired Mac `W12` vs `W6` achieved `1.36x` wall throughput (`>= 1.20` bar) with `0` thermal trips.
- **`D256` Mac production (`COMPLETE`):** Completed all `864` parents (`1,566` immutable units) across `8` attempts in `16,333.70 s` (`4.537 h`, `~85 parents/h` at `W12`, `0` thermal trips).
- **`D1024` Mac production (`COMPLETE`):** After `equiv-02` matched all life payloads byte-for-byte (`FAIL` only on session metadata `beq-gate-scripted-00`), `D1024` ran OLD `61900af2` at `W6` and completed all `2,160` rows (`3,865` immutable units, `0` pending) across `2` attempts (resuming from `362` units after a `7,584.94 s` preemption).
- **L4/GPU closures & `D4096` (`NOT_RUN_BY_RULE`):** L4 v2 failed `C6` (`12` sidecar-download timeouts -> `13` attempts vs limit `3`), and `Step -1` measured `3.251x` optimistic / `1.751x` pessimistic vs Mac `W12` (`W12_BELOW_4`, `CLOSE_NO_BOUND`). `D4096` (`60` parents completed in `5,400 s` of `468,000 s` cap) was sealed `NOT_RUN_BY_RULE` after `D1024` `R-A` returned `STOP_FUNDING`.

## Why
20-step ring snapshots plus Q-ordered `exact_stop` kept restore/replay overhead to `1.2%` (`47.2%` natural play, `33.5%` doom branches), letting Mac `W12` run fast enough that an A100 GPU missed `>= 4x` (`3.251x`); single-life `1.30x` slowdown checks had tripped falsely (`~10%`/life) on Apple Silicon E-core scheduling until replaced by `4 * width` rolling medians.

## What to reuse / what not to repeat
- Reuse atomic per-life/session `.sha256` + `.complete` exports with verified union ledgers so preempted multi-attempt runs (`8` in `D256`, `2` in `D1024`) skip completed lives with zero recomputation.
- Guard Apple Silicon `W12` worker pools by aggregate wall throughput (`1.36x` vs `W6`), `4 * width` rolling medians, and `>= 30 s` (`7`-tick) `(PID, birth_tick)` persistence rather than single-life per-worker latencies.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-labels.md` | Primary sources: [label-factory.md](../../notes/label-factory.md), [bl_labels/README.md](../../../research/bl_labels/README.md), [INTENT.md](../../../research/bl_labels/INTENT.md)
