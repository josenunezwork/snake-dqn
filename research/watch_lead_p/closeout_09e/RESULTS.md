# P watch-lead, attempt 09e: offline closeout

**Outcome: INVALID/INCOMPLETE. No scientific decision. No outcome metrics opened or computed.**

2026-10-29 · Astra implementer/analyst · worktree `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p` · branch `astra-argon-falsifier-p-20261028` · starting HEAD `d83b9d6d1d31932a7d2d01b22a14058ab1a798c0` · runtime pin `5028a996bb94ec47823627cc113ab6dad6073157`.

09e was the last registered manual completion. The attempt receipt records `RPC_EXEC_POLL_TIMEOUT`, `Refused('deadline reached during telemetry recovery')`, `stop_verified=true`, `cleanup_error=null`, and elapsed **34,156.61816912505 s (569.276969 min)** under a 570-minute timeout. This is receipt-backed historical stop evidence; no live provider check was made. The registered endpoint remains 32 paired sessions × three arms × 6,000 frames. The comparator is bare B8 s3; P48 is primary and P0 descriptive. This closeout confers no controller efficacy, timing, strict-gate, serving or release result.

## Binding rules and the exhausted-attempt disposition

Authority: `research/watch_lead_p/README.md`, frozen `protocol.py` and `resume.py`, the seal-bound `packet-09/reseal-05/retry-accounting.json`, and the 2026-10-28 strategy `RULING.md` §4 P. The runtime source closure is unchanged by this closeout.

Exact binding text:

> “The Mac reducer takes exactly one terminal look over all 32 verified paired sessions.” — README.md:17

> “INVALID/INCOMPLETE first; then STOP this delayed controller if UB90(benefit)<10% or mass upper95<−5%; GO to real paced development if benefit≥10%, LB90>0 and mass lower95≥−5%; otherwise INCONCLUSIVE, with no automatic expansion.” — README.md:21; also protocol.py DECISION_ORDER and strategy RULING.md §4 P

> “Resume units are whole three-arm paired sessions” and “Interrupted partial sessions are discarded and never stitched together.” — README.md:25

> “There is one original attempt and at most two manual completion attempts, only after verified mechanical failure, verified stop and missing-unit checks.” — README.md:27

> “The one Mac terminal aggregation uses the complete 32-unit union across attempts, with duplicate/foreign refusal.” — README.md:37

> “Time-cap INCOMPLETE of a running job: completing to the registered N under the pre-registered attempt cap is allowed; raising N or the cap mid-series is not.” — docs/research/advice/2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md:133–134

The executable rule agrees: `resume.py:211–221` requires indices 0–31 before calling either statistic function and states `# Do not inspect effect estimates on a partial union.` `protocol.py:80–81` returns `INVALID/INCOMPLETE` when `complete` is false. `INCONCLUSIVE` is a later, valid-complete-data branch; it is not the missing-data label.

The packet explicitly records `prepared_completion_ordinal=2`, `maximum_completion_attempts=2`, `completion_attempts_remaining_after_execution=0`, and `preparation_09e.remaining_after_09e_dispatch=0`. Original 06b and completion 09c both have zero retained exports. 09c was conservatively charged as completion 1 despite failing at allocation/warmup; that accounting is already frozen, and is not refunded here. 09e consumed completion 2.

**Consequence:** no registered analysis of the 21 completed sessions is allowed, including P0 descriptives, death rates, relative mass, bootstrap limits or benefit estimates. None was run. The administrative status is `INVALID/INCOMPLETE`, with no GO/STOP efficacy conclusion. No third completion or smaller-N reinterpretation is authorized. P contains no separate sentence prescribing a specific successor registration on exhaustion; the need for new prospective authority follows from the fixed N, fixed look and exhausted cap. The proposal below uses a fresh registration/namespace/bank and does not reopen this bank. Any exceptional future reuse would need explicit separate authority; this closeout supplies none.

## Export integrity and import

**Completed sessions: 00–20 inclusive (21). Missing whole sessions: 21–31 inclusive (11).** `P_RUNTIME completed` counts whole paired sessions, not individual arms. Thus **63 complete arm runs are retained**, and 33 arm slots belong to missing whole sessions. Some work on those slots occurred, but no arm-only file is an eligible resume unit.

Independent byte audit verified all **42 receipt-listed payload exports** (21 session JSON files and 21 construction-metadata JSON files), their digest/name sidecars and exact `complete\n` markers: **126 retained files**, all accounted for, no partial or orphan local export files. All 306 shipped archive files match both the runtime-pin Git blobs and current source bytes. The scientific recipe fingerprint is `acb5d0a6492c50a4ee382837f2bf4f115a305bd4779282626da5d2815b30170c`; construction metadata agrees with the sealed ledger, including the adapter and mechanics-reader hashes and mechanics v2. The packet seal, actual dispatch manifest against its receipt, archive hash, and four uploaded input identities also verify.

The source packet dispatch manifest and actual dispatched manifest have different hashes because the actual manifest contains admission/accounting expansion. The audit verifies the actual retained manifest against the receipt, rather than pretending these are identical.

All 126 files were copied byte-for-byte into `evidence/imported/units/` below this summary. The registered `run ledger` entrypoint produced `evidence/imported/verified-ledger.json`; its content equals the independent metadata-only ledger. Registered `read_ledger` verified exactly indices 0–20. No `aggregate` entrypoint was invoked. `audit.json` inventories every original/imported component with path, size and SHA-256.

Per-arm completion for retained sessions is certified by the frozen worker/publisher: `run.py:worker` runs `run_arm` at the fixed horizon, `publish` requires all three arms and equal initial state, and markers publish only after a whole unit. Integrity verification here deliberately does not deserialize scientific payloads. It is custody/source/publication verification, not a new content-level adjudication of scientific validity. There is no final 32-session inventory or remote terminal receipt in 09e.

## Per-arm runtime evidence

All values below come from `P_PROGRESS` in the retained ops log; duplicate poll-tail emissions were deduplicated (4,962 unique pulses). They are **last observed gameplay elapsed seconds, hence lower bounds**, not exact finish times. The logger emits only after a tick crosses its 60-second interval and emits no arm-finish event. Construction, final digest and coordinator export are outside that gameplay timer. No remaining branch, death or mass outcomes were inspected.

| Arm | Forecast seconds/arm | Completed-session min / median / max last elapsed | Median / forecast |
| --- | ---: | ---: | ---: |
| P0 | 4637.065570 | 300.1 / 14309.8 / 19367.4 | ≥3.086× |
| P48 | 621.630275 | 300.1 / 1949.0 / 2745.0 | ≥3.135× |
| bare | 59.015770 | 240.1 / 300.2 / 300.3 | ≥5.087× |

| Completed session | P0 seconds ≥ | P48 seconds ≥ | bare seconds ≥ |
| --- | ---: | ---: | ---: |
| 00 | 18093.6 | 2369.0 | 300.3 |
| 01 | 14964.2 | 1949.0 | 300.2 |
| 02 | 17538.5 | 2082.8 | 300.1 |
| 03 | 19367.4 | 2546.0 | 240.1 |
| 04 | 3958.4 | 761.0 | 300.1 |
| 05 | 360.3 | 300.1 | 300.2 |
| 06 | 13206.1 | 1775.7 | 300.1 |
| 07 | 17512.9 | 2270.3 | 300.2 |
| 08 | 17260.1 | 2289.6 | 300.2 |
| 09 | 9965.3 | 1790.4 | 300.1 |
| 10 | 300.1 | 300.3 | 300.1 |
| 11 | 300.1 | 300.1 | 300.3 |
| 12 | 5327.6 | 835.5 | 300.2 |
| 13 | 14309.8 | 1942.4 | 300.1 |
| 14 | 6432.3 | 1047.1 | 300.2 |
| 15 | 16408.8 | 2211.0 | 300.1 |
| 16 | 6816.6 | 2745.0 | 300.3 |
| 17 | 17755.4 | 2331.4 | 300.1 |
| 18 | 15234.2 | 2614.4 | 300.2 |
| 19 | 6494.4 | 1126.9 | 300.3 |
| 20 | 15294.2 | 2009.7 | 300.1 |

For missing sessions, cells show **last observed frame / elapsed seconds**, not completion. “No pulse” means no retained observation, not proof that an arm never started. No missing session has any eligible retained partial payload; completion of individual arms without a whole session cannot be established exactly from these logs.

| Missing session | P0 last pulse | P48 last pulse | bare last pulse |
| --- | --- | --- | --- |
| 21 | 5302/6000, 16224.9 s | 5946/6000, 2625.2 s | 5519/6000, 300.1 s |
| 22 | 5992/6000, 8888.1 s | 5996/6000, 300.2 s | No pulse |
| 23 | 4626/6000, 9958.6 s | 5963/6000, 2099.9 s | No pulse |
| 24 | 5997/6000, 4453.6 s | 5997/6000, 797.4 s | No pulse |
| 25 | 4792/6000, 9886.1 s | 4428/6000, 1088.7 s | No pulse |
| 26 | 4596/6000, 9428.4 s | 4251/6000, 1103.2 s | No pulse |
| 27 | 4448/6000, 8639.0 s | 3754/6000, 748.4 s | No pulse |
| 28 | 4199/6000, 7244.9 s | 2742/6000, 207.9 s | No pulse |
| 29 | 5993/6000, 2777.3 s | No pulse | No pulse |
| 30 | 5763/6000, 300.3 s | No pulse | No pulse |
| 31 | 3592/6000, 3169.9 s | No pulse | No pulse |

The last retained runtime pulse is 33,755.4 s with completed=21/skipped=0. First observed publication milestones were 1 session at 8,504.3 s; 11 at 19,372.4 s; 12 at 22,912.8 s; 20 at 32,182.7 s; 21 at 32,613.0 s. These are telemetry observation times, not exact publication timestamps. Session 21 P0 was last observed at frame 5,302 after 16,224.9 s; no interpolation can turn it into a complete unit.

## Why the forecast failed

1. **The target host was never throughput-qualified.** The frozen forecast explicitly says “Contended Mac forecast; target L4 throughput unqualified.” Its projected makespan was 17,094.119074851274 s (284.901984580855 min), built from three natural L600–603 roots and a first-transition forced-death fixture, using maximum root costs and a 59.01577 s bare arm. Completed bare arms alone reached median last elapsed 300.2 s, over 5× the full bare forecast. This isolates a substantial host/workload throughput mismatch even without examining search outcomes. The relative contributions of CPU model, effective cores/SMT, worker contention, process priority and full-session length distribution cannot be identified: retained evidence does not include a usable CPU topology/utilization profile.

2. **Search dominates the long tail.** P0 median last elapsed is 14,309.8 s and maximum 19,367.4 s, versus 4,637.06557 s forecast. P48 median is 1,949.0 s and maximum 2,745.0 s, versus 621.630275 s. Short early-L600 roots and first-step injected death do not price full long-body sessions or several late-dying H300 branches. This is a representativeness limitation already disclosed in README:33, not evidence of a specific observed branch/death frequency. No outcome-dependent per-root census was taken.

3. **Scheduling magnified the expensive-arm tail.** Eleven single-thread CPU workers execute cohort-ordered P0/P48/bare tasks. P0 for a later cohort can overlap unfinished P0 from an earlier cohort; the queue has no all-session barrier. An early expensive P0 (session 21) remained unfinished while other sessions published. Runtime workers inherit parent nice 15 and are lowered to absolute nice 19. This reserves supervisor preference but provides no speedup. GPU model does not accelerate this CPU-only implementation.

4. **The terminal failure is operational, not an efficacy failure.** The receipt has 136 failed renewal calls (and two failed control calls); many renewal RPCs reached about 120 s, and the final failure was deadline expiry during telemetry recovery. Setup/upload measurement is only 83.891442 s, including 73.616407 s transfers, versus the forecast’s 1,200 s allowance. Worker progress continued while RPCs were slow. Summed call durations overlap remote execution and other calls, so they cannot be subtracted from wall time to invent a transport-free throughput. No exact causal split between host contention and RPC load is available.

**Correct interpretation of “~2×”:** receipt elapsed / forecast = **1.99815024217×**, but the attempt completed only 21/32 units. The full-study runtime is right-censored; it was not measured at 2×. An illustrative deterministic schedule using each completed arm’s maximum last pulse +60 s gives 68,617.8 s (19.0605 h), versus the 4.7484 h forecast. This assumes a common expensive arm across all sessions and no unobserved tail, excludes variable initialization/finalization, and is neither an estimator of the actual missing work nor a safe upper bound.

## Successor proposal only — nothing prepared for dispatch

This is a proposal, not a registration, amendment, manifest, funding ruling, or new run. The old 32-session bank remains closed to analysis and further execution under P. The active strategy permits this delivered-search premise question; older closed vector61/veto or bare late-death lines are not being reopened.

| Priority / dependency | Proposed work and placement | Envelope and prospective acceptance | Decision enabled |
| --- | --- | --- | --- |
| 1 | Fresh excluded full-session timing/profile fixtures on the actual candidate host, with fixed CPU math and B8/opponent inputs. Benchmark useful work at 4/6/8/11 workers, preserving coordinator headroom. Record CPU topology, runnable load, RSS, CPU vs wall, phase times and exact per-arm finish events; include long bodies and late-death/all-legal-branch engineering stress. | Freeze a finite engineering plan and 2× timeout before any fixture; engineering must not read or reuse P outcome payloads. No numerical host speedup is credited until measured. | Qualified per-host useful completed sessions/hour and a truthful full-bank forecast. |
| 2, after 1 | A new fixed-N, one-look development registration on a new namespace/bank; retain the same controllers/statistics/thresholds unless a separately motivated change is declared prospectively. Prefer up to three disjoint session shards, subject to the **account-wide maximum of three including all other jobs**. Do not assume three slots are free or stop another job. | Illustrative sizing: observed-max-plus-60 three-arm gameplay cost is 22,592.7 s (6.27575 h). With one session per worker, an 11-session shard is roughly 6.6091 h including 1,200 s setup/export. A provisional **8 h base / 16 h effective per 11/11/10-session shard** leaves some margin on this sensitivity; replace it with measured target-host forecasting and tail stress before registration. It is not a completion guarantee. Freeze full endpoint, per-shard envelopes, one original + at most two mechanical completions, and exhaustion status before data. | A complete development answer without reinterpreting fast-finished sessions as the bank. |
| 3, optional after 1 | A100 or H100 host only if its **CPU capacity** yields better measured time to a complete paired session, or after a separately reviewed GPU implementation exists. CPU-only runtime is also a candidate if qualified and its debit measured. | Compare actual host shape/available memory and aggregate runtime-hours at measured debit. A100/H100 labels alone justify no multiplier; GPU utilization/speedup for current code is zero by design. Any GPU port changes floating-point/platform semantics and needs explicit new identity/parity validation. | Faster completion if demonstrated, rather than purchasing an idle accelerator. |

At the observed P L4 rate **1.54 u/h**, three illustrative shards at the 6.6091 h sensitivity consume about **19.8273 aggregate runtime-h / 30.534 units**; three 8 h base guards correspond to 24 runtime-h / 36.96 units and three 16 h effective guards to 48 runtime-h / 73.92 units. These are exposure arithmetic, not unit budgets or permission to use occupied slots. The old attempt’s modelled own debit is 14.6114422 units at receipt elapsed; provider crosscheck and other-job debit must remain separate. A100/H100 successor rates are not measured here. Use live calibration at future dispatch; no invented H100 price or GPU throughput is used.

For a single L4 host, the 19.0605 h illustrative schedule implies at least about **20 h base / 40 h effective** before additional tail stress. That is a poor dependency on one VM; a 40-hour guard does not provide 40 hours of Colab availability. Colab documents variable hardware/limits and Pro+ execution up to 24 hours subject to availability and units: [official Colab FAQ](https://research.google.com/colaboratory/faq.html). Prefer bounded resumable shards over a giant timeout. Do not retrospectively enlarge 09e.

Whole-session three-arm units remain the scientific/ledger identity. A successor may add immutable arm-level operational checkpoints to avoid losing an expensive finished arm, but must first specify same-platform pairing, canonical arm IDs, full-state/RNG identity, collision/duplicate refusal, marker-last publication, skipped-unit accounting and a fixed union reducer. It must forbid stitching interrupted trajectories or recomputing verified arms, and register arm-granularity recovery before data. Resume, partial completion and whole-bank exhaustion must be offline-tested. Do not change W1/K1, H300/H6000, trigger, fallback, precision or scientific thresholds as a throughput fix.

## Verification and preserved boundaries

- Focused audit tests: `python -m pytest -q research/watch_lead_p/closeout_09e/test_audit.py` under CPU/single-thread caps and `nice -n 15`: **6 passed** (0.04 s test time). They verify opaque-payload handling and corruption/partial/symlink refusal.
- `python -m black --check --line-length 100 research/watch_lead_p/closeout_09e`, `python -m isort --check-only --profile black research/watch_lead_p/closeout_09e`, and `python -m flake8 research/watch_lead_p/closeout_09e`: passed.
- Real offline audit entrypoint: `python -m research.watch_lead_p.closeout_09e.audit --artifacts /Users/josenunez/Projects/ml/snake-dqn-artifacts --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-closeout-09e/evidence`: verified/imported 21 sessions and 126 components; no look.
- Registered ledger construction and `read_ledger` verification: independent and registered ledgers equal; indices exactly 0–20. Exact argv is recorded in `evidence/ledger-command.json`.
- The interpreter is `/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python`. Existing runtime/science code and hash-bound documents, manifests and receipts were not edited. No gameplay, benchmark, aggregate, training, Colab command, live session, other job or ops-state write occurred. The original untracked `saved_snakes` item is preserved. The existing external prompt.txt/retry.out are preserved.
- Independent Argon review and Git publication evidence are recorded separately after the source commit; no claim of cross-review acceptance is made by this initial report.

## Evidence identities

| Evidence | SHA-256 |
| --- | --- |
| `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-closeout-09e/evidence/audit.json` | `3fa4096c32e723b3ccb4c0001286bc61e8566fe4ac62459f5f67461ba7908703` |
| `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-closeout-09e/evidence/arm-timings.csv` | `0997b50b939a5194fc4b6be79e2134cd2f223dc24554b8f4a530e5c2eb0e05d5` |
| `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/astra-p-closeout-09e/evidence/imported/verified-ledger.json` | `bd635bee6650fd71ddb11e79698a3cf982cbbd947a95a72e2b169a1c1fb1b445` |
| `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-watch-lead-1028/attempt-09e/output/receipt.json` | `03d0e60d297118a8c27f316e42ab70af8d2e7568a59f3fa9704b80c64629a822` |
| `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/sandbox-p-watch-lead-20261028-09e.log` | `b74a84b768303d77b6338603bd6f08dcf2e44422b7e260d39d2a689a12975f55` |
| `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p/research/watch_lead_p/packet-09/reseal-05/retry-accounting.json` | `23263b519a2ea1a43db25b5b9afb035e8f7c71f16c5505774b677b88e7e60760` |
| `/Users/josenunez/Projects/ml/snake-dqn-falsifier-p/research/watch_lead_p/packet-09/reseal-05/forecast.json` | `bd789c9897b0946ed4630c0313027ebf219706d3e08f169beaea592e24851ea8` |

Full component hashes and original evidence paths are in `evidence/audit.json`. Scientific payload copies stay outside Git. Branch publication is limited to the closeout auditor, focused tests, this report, and verification/review evidence; never main.
