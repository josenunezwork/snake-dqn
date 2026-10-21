# Path B v2 deliverability smoke — 2026-10-21

**Stopping-rule verdict: STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY.**
Do not start operator-02. Both configurations fall below 50%; do not run the paced study.

DEVELOPMENT ONLY — excluded from every registered verdict. This result changes no registration,
product thread contract, serving configuration, or release. Next steps follow the precommitted
stopping rule; this report does not perform a section 4 amendment or admit operator-02.

| Configuration | Delivered improvements / 75 T3 changes | Horizon-surviving rescues delivered | Historical rescue intervals rescued / 34 | M3 delivered / 5 |
|---|---:|---:|---:|---:|
| N=1 | 7/75 = 9.333333% | 0 | 0/34 | 0/5 |
| N=6 | 0/75 = 0.000000% | 0 | 0/34 | 0/5 |

The 75 changes occupy 22 original arm/environment trajectories. Of the 75 original chosen
actions, 34 survived the historical D300 horizon and 41 were delay-only. The historical 54/64
number is episode survival, not the delivery denominator. Controller-labelled rescue horizons
are up to H500 and truncate at terminal frame 5,000, as in the unchanged adapter.

## Execution and interpretation

- Run source: `c417687364eaf058d11d87efb57019f1e95e919a`; stopping-rule commit: `7622d62e` (committed/pushed before data).
- Wall time: 3432.912 seconds (57.215 minutes), cap 6,900 seconds.
- Mac CPU, 24 fps absolute deadlines, N=1 followed by N=6. Numerical pools capped at one.
- N=6 uses one existing controller worker plus five helpers; main tick thread makes seven active compute threads at most.
- All periodic AC, open-lid and thermal checks passed. Phase 0 and other owner jobs remained active; this is a contended development measurement.
- Only the new smoke package changed. Path B source, registrations, frozen worktrees, existing processes, tmux sessions, checkpoints and serving pins were untouched. The sealed Round 0 study directory was never opened.

Each interval reconstructs earlier original T3 overrides, then starts a fresh actual v2 adapter
at max(previous change + 1, doom - 499). Cold shadow fill is charged from before adapter
construction. The original controller runs without a search wait at any tick; the interval ends
at its first applied change or original greedy doom. An actual application also needs a result
completed before its absolute 24 fps frame deadline to count as delivered. This prevents tick
overruns donating extra search time. These conditional cold-root replays measure opportunity
delivery; they do not establish continuous-trajectory efficacy or quiet-host serving timing.

The six-thread wrapper partitions chronological branch points, calls the unchanged native
evaluator, latches the first complete winner, and cooperatively drains sibling chunks. Drain
cost is charged to publication. Results apply to this wrapper and the recorded load.

The operational LAD below is the exact backward-scanned last H500 **improving** decision:
either a surviving alternative or a strict survival delay. It is separate from the preserved
historical D300 survival-only LAD. Lead uses the live frame observed when worker doom detection
actually finishes, rather than its older source-frame metadata. Bracketed lead also uses the
ideal 24 fps wall-clock frame at that moment; negative means its ideal deadline already passed.
Diagnostic LAD scans run outside
paced windows and provide no search-result cache.

| Timing/accounting | N=1 | N=6 |
|---|---:|---:|
| Completed improving evaluations already late (all 80 dooms) | 475 | 166 |
| Controller late discards (all 80 dooms) | 0 | 0 |
| Controller stale discards (all 80 dooms) | 0 | 0 |
| Tick deadline overruns (all 80 dooms) | 293 | 21077 |

Delivered intervention kinds: `{"1": {"delay": 7}, "6": {}}`.
All scored controller states have zero worker, invalid-result and unconditional-fault counts.
Searches still active at the original doom are cooperatively cancelled by adapter close. The
engine records that drain as INVALID; these terminal shutdown records are retained separately
from the scored controller state. No failed unit was retried.

## M3 pacing

The target was real 24 fps pacing. The achieved rates below expose failures to sustain that
pace; absolute 24 fps deadlines still govern delivery even when scored ticks fall behind.

| M3 doom | Achieved fps N=1 / N=6 | Tick overruns N=1 / N=6 |
|---|---:|---:|
| M3-frozen-2026101934-0 | 24.024 / 5.692 | 57 / 195 |
| M3-frozen-2026101934-1 | 24.022 / 5.486 | 12 / 246 |
| M3-frozen-2026101934-2 | 23.989 / 15.917 | 59 / 68 |
| M3-mixed-2026101934-0 | 23.995 / 4.969 | 33 / 322 |
| M3-mixed-2026101935-1 | 24.012 / 5.333 | 18 / 314 |

## Per-doom measurements

Pairs are N=1 / N=6. First improving time is seconds from observed doom detection to the first
fully evaluated improving candidate, including candidates whose frame has passed. `—` means none
observed through cooperative shutdown/drain. First-improvement timing can include a drain completion. Delivery is `D` for executed delay, `R` for executed rescue, `—` for none.
Late evaluations and controller late/stale discards are distinct counters.

| Doom ID | Original doom | Observed [ideal-24fps] lead | First full improvement, s | Delivered | Late evaluations | Controller late/stale |
|---|---:|---:|---:|---|---:|---|
| M3-frozen-2026101934-0 | 2146 | 324 [325] / 305 [305] | — / — | — / — | 0 / 0 | 0/0 ; 0/0 |
| M3-frozen-2026101934-1 | 2796 | 329 [330] / 325 [325] | — / — | — / — | 0 / 0 | 0/0 ; 0/0 |
| M3-frozen-2026101934-2 | 4672 | 276 [276] / 245 [245] | — / — | — / — | 0 / 0 | 0/0 ; 0/0 |
| M3-mixed-2026101934-0 | 2178 | 359 [359] / 354 [354] | — / — | — / — | 0 / 0 | 0/0 ; 0/0 |
| M3-mixed-2026101935-1 | 2956 | 346 [346] / 353 [353] | — / 75.134 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_11-c00 | 4198 | 425 [425] / 421 [422] | 17.583 / 22.275 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s0_12-c00 | 4877 | 420 [421] / 433 [434] | — / 27.714 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_15-c00 | 4177 | 413 [414] / 425 [425] | 17.003 / 23.185 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s0_17-c00 | 4610 | 419 [420] / 427 [428] | — / 29.833 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_21-c00 | 3080 | 401 [402] / 413 [414] | — / 22.981 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_21-c01 | 3968 | 423 [424] / 427 [428] | 13.886 / 13.182 | — / — | 13 / 5 | 0/0 ; 0/0 |
| b8_s0_21-c02 | 3973 | 247 [247] / 251 [252] | 9.107 / 7.904 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s0_21-c03 | 3976 | 251 [252] / 258 [259] | 8.586 / 12.284 | — / — | 5 / 3 | 0/0 ; 0/0 |
| b8_s0_21-c04 | 3977 | 240 [241] / 242 [243] | 8.455 / — | — / — | 1 / 0 | 0/0 ; 0/0 |
| b8_s0_21-c05 | 3995 | 243 [243] / 252 [253] | 7.525 / 13.618 | — / — | 16 / 1 | 0/0 ; 0/0 |
| b8_s0_21-c06 | 4006 | 224 [225] / 234 [235] | 7.136 / — | — / — | 6 / 0 | 0/0 ; 0/0 |
| b8_s0_21-c07 | 4023 | 232 [233] / 244 [245] | 7.215 / 7.471 | — / — | 9 / 4 | 0/0 ; 0/0 |
| b8_s0_21-c08 | 4026 | 195 [196] / 202 [203] | 8.403 / — | — / — | 1 / 0 | 0/0 ; 0/0 |
| b8_s0_21-c09 | 4027 | 177 [178] / 183 [184] | 7.845 / 6.283 | — / — | 2 / 1 | 0/0 ; 0/0 |
| b8_s0_21-c10 | 4028 | 114 [114] / 117 [118] | 3.401 / 5.890 | — / — | 2 / 1 | 0/0 ; 0/0 |
| b8_s0_22-c00 | 3694 | 423 [423] / 437 [437] | — / 26.482 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_3-c00 | 4362 | 422 [423] / 425 [426] | 14.854 / 24.982 | — / — | 8 / 6 | 0/0 ; 0/0 |
| b8_s0_5-c00 | 2041 | 417 [417] / 427 [427] | — / 28.163 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s0_8-c00 | 4553 | 431 [432] / 433 [434] | 15.772 / 19.691 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s0_8-c01 | 4566 | 260 [261] / 253 [253] | 5.638 / 7.382 | — / — | 11 / 4 | 0/0 ; 0/0 |
| b8_s0_8-c02 | 4590 | 273 [274] / 273 [274] | 10.288 / — | — / — | 7 / 0 | 0/0 ; 0/0 |
| b8_s0_8-c03 | 4596 | 222 [222] / 217 [218] | 5.774 / — | — / — | 14 / 0 | 0/0 ; 0/0 |
| b8_s0_8-c04 | 4598 | 192 [192] / 188 [189] | 7.164 / — | — / — | 9 / 0 | 0/0 ; 0/0 |
| b8_s0_8-c05 | 4600 | 139 [140] / 134 [134] | 6.322 / — | — / — | 3 / 0 | 0/0 ; 0/0 |
| b8_s3_1-c00 | 2889 | 428 [429] / 435 [436] | — / 29.194 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_1-c01 | 3478 | 427 [427] / 434 [435] | — / 21.103 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_12-c00 | 3492 | 433 [434] / 421 [422] | 11.734 / 15.526 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s3_13-c00 | 2208 | 416 [417] / 424 [425] | 12.866 / 14.873 | — / — | 5 / 4 | 0/0 ; 0/0 |
| b8_s3_15-c00 | 2636 | 424 [424] / 427 [428] | — / 28.616 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_17-c00 | 4531 | 416 [416] / 416 [417] | 16.322 / 21.140 | — / — | 8 / 4 | 0/0 ; 0/0 |
| b8_s3_19-c00 | 4939 | 403 [403] / 407 [408] | 14.118 / 19.186 | — / — | 9 / 6 | 0/0 ; 0/0 |
| b8_s3_19-c01 | 4969 | 238 [238] / 242 [243] | 9.140 / 12.058 | — / — | 11 / 1 | 0/0 ; 0/0 |
| b8_s3_19-c02 | 4974 | 195 [195] / 197 [198] | 6.232 / — | — / — | 9 / 0 | 0/0 ; 0/0 |
| b8_s3_19-c03 | 4989 | 215 [216] / 217 [218] | 8.453 / — | — / — | 11 / 0 | 0/0 ; 0/0 |
| b8_s3_21-c00 | 1983 | 417 [418] / 421 [422] | — / 24.417 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_26-c00 | 3282 | 422 [422] / 421 [422] | — / 29.990 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_26-c01 | 3834 | 413 [414] / 426 [426] | 12.223 / 20.803 | — / — | 14 / 3 | 0/0 ; 0/0 |
| b8_s3_26-c02 | 3915 | 318 [319] / 321 [322] | 9.549 / 10.710 | — / — | 15 / 6 | 0/0 ; 0/0 |
| b8_s3_26-c03 | 3946 | 272 [272] / 275 [276] | 6.575 / 12.276 | — / — | 5 / 1 | 0/0 ; 0/0 |
| b8_s3_26-c04 | 3948 | 224 [224] / 222 [223] | 9.471 / — | — / — | 3 / 0 | 0/0 ; 0/0 |
| b8_s3_26-c05 | 3950 | 77 [78] / 78 [78] | 2.704 / 2.836 | — / — | 3 / 2 | 0/0 ; 0/0 |
| b8_s3_26-c06 | 3951 | 72 [73] / 73 [74] | 3.022 / — | — / — | 1 / 0 | 0/0 ; 0/0 |
| b8_s3_29-c00 | 2151 | 424 [425] / 432 [433] | — / 12.816 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_30-c00 | 3846 | 424 [425] / 434 [435] | — / 26.753 | — / — | 0 / 4 | 0/0 ; 0/0 |
| b8_s3_6-c00 | 4050 | 426 [426] / 430 [431] | 15.818 / 21.352 | — / — | 7 / 4 | 0/0 ; 0/0 |
| b8_s3_6-c01 | 4660 | 428 [429] / 427 [428] | 13.213 / 18.360 | — / — | 19 / 6 | 0/0 ; 0/0 |
| b8_s3_6-c02 | 4684 | 273 [273] / 266 [267] | 6.265 / 13.858 | — / — | 16 / 2 | 0/0 ; 0/0 |
| b8_s3_6-c03 | 4759 | 305 [306] / 301 [302] | 12.255 / — | — / — | 2 / 0 | 0/0 ; 0/0 |
| b8_s3_6-c04 | 4761 | 169 [170] / 169 [170] | 4.026 / — | D / — | 3 / 0 | 0/0 ; 0/0 |
| b8_s3_6-c05 | 4764 | 143 [144] / 144 [145] | 3.998 / 6.483 | — / — | 5 / 1 | 0/0 ; 0/0 |
| b8_s3_6-c06 | 4766 | 136 [137] / 136 [136] | 4.609 / — | — / — | 1 / 0 | 0/0 ; 0/0 |
| b8_s3_6-c07 | 4768 | 53 [54] / 51 [52] | 1.386 / — | — / — | 5 / 0 | 0/0 ; 0/0 |
| b8_s3_6-c08 | 4769 | 38 [39] / 37 [38] | 1.091 / 1.577 | D / — | 4 / 2 | 0/0 ; 0/0 |
| b8_s3_6-c09 | 4773 | 12 [12] / 11 [12] | 0.283 / — | D / — | 1 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c00 | 4407 | 422 [423] / 420 [420] | 10.791 / 22.617 | — / — | 19 / 1 | 0/0 ; 0/0 |
| b8_s3_8-c01 | 4437 | 279 [279] / 278 [279] | 6.756 / 9.444 | D / — | 14 / 2 | 0/0 ; 0/0 |
| b8_s3_8-c02 | 4457 | 264 [264] / 260 [261] | 5.287 / 12.072 | — / — | 13 / 2 | 0/0 ; 0/0 |
| b8_s3_8-c03 | 4468 | 249 [249] / 248 [248] | 5.308 / 9.224 | D / — | 15 / 7 | 0/0 ; 0/0 |
| b8_s3_8-c04 | 4503 | 274 [275] / 266 [266] | 7.132 / 12.007 | — / — | 18 / 2 | 0/0 ; 0/0 |
| b8_s3_8-c05 | 4507 | 220 [221] / 219 [219] | 6.357 / 9.894 | — / — | 11 / 2 | 0/0 ; 0/0 |
| b8_s3_8-c06 | 4513 | 215 [215] / 215 [216] | 7.605 / — | — / — | 10 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c07 | 4521 | 132 [133] / 131 [131] | 2.744 / 4.596 | D / — | 5 / 2 | 0/0 ; 0/0 |
| b8_s3_8-c08 | 4522 | 113 [114] / 110 [111] | 4.148 / — | — / — | 5 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c09 | 4525 | 84 [84] / 82 [82] | 2.406 / — | — / — | 6 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c10 | 4526 | 80 [81] / 78 [79] | 3.264 / — | — / — | 3 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c11 | 4527 | 54 [55] / 53 [54] | 2.119 / — | — / — | 4 / 0 | 0/0 ; 0/0 |
| b8_s3_8-c12 | 4529 | 22 [22] / 22 [22] | 0.650 / 1.363 | — / — | 5 / 1 | 0/0 ; 0/0 |
| b8_s3_9-c00 | 4416 | 422 [422] / 420 [421] | 13.414 / 20.489 | — / — | 14 / 4 | 0/0 ; 0/0 |
| b8_s3_9-c01 | 4487 | 300 [300] / 294 [295] | — / 16.478 | — / — | 0 / 1 | 0/0 ; 0/0 |
| b8_s3_9-c02 | 4497 | 248 [249] / 245 [246] | 5.246 / 7.933 | — / — | 6 / 1 | 0/0 ; 0/0 |
| b8_s3_9-c03 | 4498 | 181 [181] / 176 [177] | 5.377 / — | — / — | 5 / 0 | 0/0 ; 0/0 |
| b8_s3_9-c04 | 4500 | 162 [162] / 161 [162] | 5.386 / — | D / — | 4 / 0 | 0/0 ; 0/0 |
| b8_s3_9-c05 | 4501 | 60 [61] / 60 [61] | 2.310 / — | — / — | 6 / 0 | 0/0 ; 0/0 |
| b8_s3_9-c06 | 4504 | 54 [55] / 54 [55] | 1.498 / 2.700 | — / — | 4 / 1 | 0/0 ; 0/0 |
| b8_s3_9-c07 | 4506 | 30 [30] / 31 [31] | 1.163 / 1.591 | — / — | 4 / 1 | 0/0 ; 0/0 |

## Evidence and verification

- Artifact root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-pathb-smoke-20261021`.
- `receipt.json` SHA-256: `d493fe5824f44db138e92fac282cd1d6141cb0432613d5a1b8982fef5994e336`.
- `start.json` SHA-256: `91d4236888da77024bf1a06c825ad3aeeb762f3634281333af6f3283541a2fe6`.
- Audit: PASS, 160/160 rows, 80 paired dooms, 255 source hashes and 50 input hashes verified.
- Targeted package tests: 5 passed. Native serial/six-thread outcomes matched; first-winner cancellation, fixed denominator and later on-time application checks passed.
- Black, isort and flake8 passed on the new package. No broader tests were run.
- Raw `n1-*.json`, `n6-*.json` and `lad-*.json` retain all per-doom fields and outcome details; raw artifacts remain outside Git.

The current operator-02 chain remains unstarted by this task. Any future engineering revision
needs its own development measurement and prospective product/registration process.
