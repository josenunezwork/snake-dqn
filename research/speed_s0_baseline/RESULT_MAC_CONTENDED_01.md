# Speed program S0: matched baseline, Mac run 01 (CONTENDED HOST)

Excluded diagnostic (speed_vs_constraints JUDGMENT §3 S0 / §5 D1). Not A5, Round 0, gate,
retention or serving evidence.

- Receipt: `mac-contended-01/s0-baseline.json` (+ `.sha256`, `.complete`); log `mac-contended-01.log`
- Host: Apple M5 Pro, **CONTENDED** — registered 6-worker Phase 0 job + other lanes running,
  load average ~12 on 18 cores; S0 ran as 1 process, 1 thread, `nice -n 10`. Absolute
  times are contended-host numbers; stage *shares* are the D1 input.
- Engine: unchanged T34 teacher/shadow (`teacher_v2.run_worlds`, H300/D300, beta 1.0, CPU);
  engine-source digest `3b9f1564…` (main engine == branch base d62070e5).
- Fixture: 3 mixes x {short, long (900/600/400/300/150/150)}, 300 decisions each, cold build ->
  clone/reset -> first fork per case. Coverage round: 10 opponent deaths, 10 respawns,
  7 shadow forks (6 cold + 1 re-fork), 1 doom search (long-mixed), 0 hero deaths.
- Rounds: 1 coverage + 3 plain + 2 instrumented; profiler overhead ratio 0.94-1.06 per case.
- Identity: all 6 rounds byte-identical (18 files: exports, .sha256, .complete). These
  hashes are now the frozen reference `research/speed_s0_baseline/REFERENCE_MAC.json`.

## Throughput (plain rounds, median)

103.1 s per 6-world round -> 209 completed worlds/h, 155 active row-steps/s,
6.44 ms per active row-step (16,021 row-steps/round; long-mixed alone = 11,526, 72 s).

## D1 answer (deadline Wed 10-21 09:00 PT)

Exclusive share of worker wall time, long bodies (median of 2 instrumented rounds):

| mix      | digest | v8 (opponent veto hook) | digest + v8 | >= 40%? |
|----------|-------:|------------------------:|------------:|:-------:|
| scripted | 50.5%  | 0.0% (no veto opponents)| 50.5%       | yes     |
| frozen   | 17.6%  | 56.5%                   | 74.1%       | yes     |
| mixed    | 4.9%   | 67.0%                   | 72.0%       | yes     |

**D1 on this host: digest + v8 >= 40% on all three long-body mixes -> S1 and S2 both continue.**
Digest alone is >= 25% only on long-scripted (50.5%); it is 17.6% (frozen) and 4.9% (mixed).

Pooled over the whole fixture: v8 56.8%, inference 13.5%, digest 11.9%, other 5.7%,
raster 4.6%, sim 3.4%, opponent actions 1.7%, hero bv8 features 1.1%, everything else < 1%.
Short bodies: digest 26-49%, v8 0-30%.

## Caveats for the speed program

- This does not match the earlier 6-s smoke (digest 35.5%, v8 22.2%). The difference is the
  doom search: long-mixed search rows (10,139 of 16,021 row-steps) are ~67% v8 and ~5% digest.
  The more search-heavy production is, the more S2 (native v8) is worth and the less S1
  (digest v4) is. Amdahl on the pooled shares: digest 4x -> 1.10x; v8 2x -> 1.40x; both -> 1.60x.
- The fixture contains a single doom search and a single re-fork. It is deterministic,
  so it is a valid identity ruler, but the pooled stage *weights* depend on how often search
  happens in production. Use the per-case shares for planning.
- Contended host: absolute walls will move with the Phase 0 load; re-run quiet (or on L4)
  before quoting speed-ups. Matched speed-ups must compare plain-round medians run under the same load.
