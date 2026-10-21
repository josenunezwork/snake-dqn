# E4/E2 implementation and Mac preflight — 2026-10-21

Status: implementation verified locally; NO Colab call or scientific pilot dispatched.
This development experiment is excluded from every registered verdict.
Kill/sizing rules were committed in `3bedc5de` before gameplay. B8 reference SHA256:
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.

## Real entrypoints

All probes used the primary Python 3.12 venv, `nice -n 10`, CPU, and
OMP/OPENBLAS/MKL/VECLIB/NUMBA thread caps of 1. Training/ruler probes used two
worlds and one collector process: at most two numerical threads including parent.
Phase 0 and other Mac activity remained running. These are contended operational
measurements, not serving latency, research efficacy, or L4 throughput.

| Probe | Work | Wall seconds | Collection seconds | Learner/rollout seconds |
|---|---:|---:|---:|---:|
| R real training | 400 hero transitions; 4 critic-only + 1 actor update | 11.8189875420 | 1.5303175000 | 2.7185627491 |
| F real training | same | 10.7081145830 | 1.5720484579 | 1.8384132510 |
| Explicit R snapshot resume | restore completed 400-transition boundary | 3.2128612500 | preserved | preserved |

R resume reproduced the endpoint SHA256 exactly:
`1dd647c837dc079d58d54060826e57ffae1386abcfeb4beb0ffe14b504befb29`.
Tests also replay the next actual native-v8 simulator transition from a serialized
snapshot and prove grouped versus partitioned simulator trajectories match.
An independent two-rollout RNN check found collected/replayed log probabilities
identical after refreshing hidden state from the stored 32-step context.

The real long-body ruler probe used 3 warm-up steps and 12 measured steps,
2 worlds, 1 process: 24 measured hero transitions / 0.0893689160 s =
268.5497494446 active transitions/s. Including startup/calibration/warm-up/close,
24 / 3.6129783751 s = 6.6427189727/s. It is marked `scientific:false` and cannot
admit E2. Scientific E4 compares 6 versus 12 workers on 128 worlds; its runtime
inclusive allocation denominator also apportions observed bootstrap costs.
Supervisor final transfer/stop remains separately reported, not hidden as active work.

Three excluded H40 screen fixtures ran B8 through the exact v1 promotion profile:
scripted 0.1002767080 s, frozen 2.1601545840 s, mixed 0.0853296250 s.
A deliberately naive H5000 linear extrapolation for 96/32/32 worlds x 3 controllers
is about 8.5 Mac CPU hours, before candidate/long-body differences. This includes
a cold frozen-kernel compile and is NOT a qualified upper bound. Run the excluded
full-H5000 parity fixtures after endpoints return and revise the operational screen
forecast before admission. Screen on the Mac at one thread; reserve no extra L4
allocation for it unless separately materialized within the existing global cap.

## Forecast and first-cap bounds

L4 is priced at 2 units/allocation-hour, including setup and export. Scientific
E4 estimate 15 min / 0.5 u; hard timeout 30 min (at that rate at most 1 u),
registered unit cap <=2 u. E2 reserves two 4.5-hour / 9-unit arm envelopes plus
1 hour / 2 units unallocated reserve: total <=10 L4 allocation-hours /20 u.
The reserve is not an extra funded tranche. Mac screen consumes no Colab units.

Post-E4 generator uses `ceil(1.25 * transitions / rate / 60 + 15)` minutes per arm,
and refuses estimates >135 min (the supervisor requires a 2x timeout margin).
The 25% learner/checkpoint allowance is an assumption; CUDA learner speed is
unmeasured. At 2k allocated hero/s, 8M forecasts 99 min /3.3 u per arm;
at 4k/s, optional 20M forecasts 120 min /4 u per arm. Both fit the hard envelopes.
20M requires an explicit pre-data switch AND E4 >=4k/s; default remains 8M.
Below 1k/s the fixed rule shrinks to 4M. Below roughly 694/s even 4M cannot fit
this conservative envelope, so preparation refuses rather than overrunning it.
No automatic retries, extension after outcomes, or replacement seed.

Periodic immutable full snapshots are exported every 120 seconds at a completed
rollout boundary; a running rollout and transfer latency add to that interval.
Recovery restores optimizer/refit phase, counters, RNG, recurrent context/windows,
GridBatchSim arrays/RNG and frozen opponent runtime/veto memory. Resume identity
binds target, E4, B8, intent, bank, source commit, seed, topology and roster flags.
An explicit operator recovery must price the original attempt plus the remainder
inside the SAME arm/global caps and 48-hour window; the code never dispatches it.

## Fresh bank and audit

The committed BANK.json contains 96 scripted /32 frozen /32 mixed paired worlds,
plus 6 excluded preflight/parity fixtures. Valid artifact bank SHA:
`dec04505c932d47e127c925299f0f491984996663dfaf4a9558bf96624897e19`.
Valid safe audit SHA:
`5b1a9d29a983c61879461705d38de4ba1bc9cc912c6d373ff3baa94fe6082d66`.
It scanned 114,498 approved text files, reconstructed 28,256 historical seeds,
and checked 18 published inventories and 7 FRP exclusions. The later concurrent
E6/E7 namespace's 11,224 seeds were checked separately with zero collisions;
see `latest-main-collision-supplement.json` in the artifact root.
Opaque binaries, unrecorded runs and sealed output bytes are excluded limitations.

Deviation: the first reused scanner's exclusion glob failed a synthetic test.
It may have opened files inside the prohibited live directory while searching for
seed tokens; no study outcome was inspected. Those receipts are INVALID and
preserved with `screen-bank/INVALIDATION.json`. The replacement prunes directories
before traversal and passes explicit approved files to rg; its unreadable-directory
sentinel test proves neither traversal nor subprocess admission enters that tree.
Use ONLY `screen-bank-v2` and the dispatch handoff's final manifests.

## Verification and dispatch boundary

Targeted package + supervisor export selection: 59 passed, 13 deselected in 4.76 s.
Black(100), isort, flake8, Bash syntax, and diff whitespace checks passed.
Sparse source delivery is additionally verified through the unchanged supervisor's
archive builder and an isolated Python import/actual native opponent step.
The handoff records its final source pin, archive size and validation receipt.

The artifact root is:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e2/`.
`DISPATCH.txt` contains absolute commands, final manifest paths and input hashes.
Only E4 is immediately dispatchable; R/F manifests are locally validated templates
that fail closed until a genuine immutable E4 receipt is materialized.
Strict gate, serving qualification and release remain separate; nothing was served.
