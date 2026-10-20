# After the M2/M3 NO-GOs: what T34 should do next (Fable, 2026-10-20 ~01:30 PT)

Advice only. Read-only probes over the 64 M2 bank receipts joined to the T3 dense per-world receipts, and the
M3 `lad-*.json` receipts; no jobs. The registered NO-GOs stand; what follows is what they actually measured.

## 1. The 33 alive-extending changes are noise, and the surviving worlds are fully recovered

Joining each T3 change to its world outcome (T3 dense: 22 worlds with changes, 16 survived H5000, 6 died):

- **All 16 surviving worlds contain only truth-capable changes** (one world has one extender among four), and
  **bisection recovered every truth-capable change in all 16** (16/16 worlds, 20/20 changes).
- **All 33 extenders sit in the 6 worlds that died anyway**, always as a tail after the truth-capable rescues
  (patterns TTTTxxxxxx, TTTTTxxxxxxxx, ...). Their alive-time gain over greedy is **1 / 2 / 4 frames**
  (quartiles) on a 63-230-frame remaining life. They carry no information and no survival.
- So T3's +16 surviving worlds are explained entirely by the 42 truth-capable rescues, of which bisection found
  39 (93%) and all that mattered. The registered 90%-of-75 bar counted 33 targets T3 itself never achieved.

**Label target.** A decisive label is `(t, a*, F)` only where some alternative survives the full D horizon
(truth-capable). Where none does, emit **no decisive label** (export the doom as "no escape within D", weight 0;
do not teach "longest alive"). Because the escape island is wide (t* minus first detection: 20 / 60 / 273
decisions), add same-target labels at stride ~10 across [first, t*] with per-decision F; they are nearly free
(each is one expansion already paid by the search or ≤1,500 row-steps). Confirm labels stay KL-anchor rows.
**Drop** the stride-10+edge teacher rule: 2x the work (31k vs 17k/world) for lower recovery (88% vs 93%).

**Check against my estimates.** I forecast 6-8 expansions and ~10k row-steps per doom; measured 10-14
expansions (radius-3 refinement) and 6.5k charged rows median — fine. I forecast bisection within 10 decisions of
the LAD in ~65% and >30 off in ~12%; measured p50 3, p95 145 — consistent. Non-monotone islands cost 3/42
misses, as predicted; they did not cost a surviving world.

## 2. Path B: the deadline miss is the engine, not the idea

M3 timed the **serial T3 live-continuation path**: 53 s for 8,086 row-steps on frozen rosters = **6.6 ms per
row-step**, versus 0.89 ms on the scripted dense bank (bisection fresh, p50 5.8 s) and the T33 lockstep engine's
~0.2-0.7 ms (3 threads). My one-thread estimate (1-1.5 s per 5-branch expansion) was right for the scripted engine
and 5-10x optimistic for frozen/mixed on this path, because frozen opponents go through the rowwise live-veto
adapter. Three changes, all mechanical:

1. **Earliest-first expansion.** In every one of the 42 truth-capable intervals the first detection decision is
   itself positive (bisection's lower bound never moved below `first`). Path B does not need the LAD: expand once
   at `t_live + k` (k = search latency in frames), take the highest-Q survivor, schedule it, re-fork. One expansion
   = ≤1,500 row-steps: **~1-2 s** on the T33 engine with ONNX B8 (0.14 ms), ~10 s even on today's serial path. A
   second expansion only if the first finds no survivor (extender territory: then do nothing).
2. **D = 500** (design allows ≤500; upkeep is 9-22 ms/tick regardless of D; catch-up ~3-7 s once per re-fork).
3. **Batched single-world engine** (T33 lockstep at E=5-6, frozen V61+v8 in numba) + ONNX Runtime B8.

Reachability, lead = (D-1-H_LAD)/fps against T1's late-death H_LAD quartiles [21, 100, 254]:

| Search latency | D=300, 24 fps | D=300, 12 fps | D=500, 24 fps | D=500, 12 fps |
|---|---:|---:|---:|---:|
| 2 s (batched engine) | ~75% | ~80% | ≥94% | ≥94% |
| 10 s (serial path, earliest-first) | ~45% | ~65% | ~88% | ≥94% |
| 53 s (serial path, full LAD search, as measured) | 0% | 0% | ~10% | ~50% |

(≥94% = all T1 deaths with an observed LAD ≤300; the three censored LADs are unknown.) Path B is feasible at
D=500 even on the current engine; with the batched engine it reaches essentially every death T1 says is avoidable.
Hero deaths/1k and the contract's §3-§4 bars remain to be measured, not assumed.

## 3. Decision

- **Keep Claim B as the lead** (contract §5 already orders it first). Build Path B v2 with earliest-first
  expansion, D=500, T33 engine + ONNX; one thread, ≤1 outstanding request, scheduled "at frame t take a*".
- **Teacher v2 continues** with the corrected label target (§1); budget 17-20k row-steps/world holds, so an
  800-world round is ~1-3 u on an L4 or free on the Mac. Do **not** fund R1 until the measurement below reads
  GO: if the assisted hero cannot reproduce T3's survival online, the labels are the wrong target too.
- **Drop:** the 33-extender target; the stride-10+edge rule; the serial live-continuation search path; D=300 for
  Path B; any post-hoc relabelling of M2 — register M2b instead.

## 4. The single pre-registrable measurement

**M2b/Path B online paired probe on the T3 dense bank** (64 worlds, both B8 seeds, same seeds/rosters): the
real asynchronous Path B v2 controller on one thread under real 24 fps and 12 fps pacing versus bare B8 — the
bank where T3's synchronous teacher reached 54/64. Pre-register: GO = **≥ 50/64 episode survival** (≥75% of
T3's +16) with mass ≥ +60 and every miss/fallback counted; also report recovered truth-capable rescues (expect
≥ 39/42) and search latency p50/p95. Cost ≈ 64 × 5,000 frames × ~15 ms ≈ 1.5 h per cadence on two threads, no
spend. GO validates both the label target and Path B reachability at once and unlocks the contract-§2/§3 fresh
banks; NO-GO at both cadences kills Claim B and sends R1 back to the owner as research only.
