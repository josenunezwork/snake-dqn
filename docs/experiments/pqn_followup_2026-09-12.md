# PQN follow-up execution report — 2026-09-12

Status: **B5 v2 ended `INCONCLUSIVE_STOP` (parent exit 2); the independent raw
audit is accepted**. This report records the completed implementation,
verification, and terminal-execution evidence. The result does not identify an
RL winner, authorize evaluation or promotion, change `main`, or prescribe an
automatic rerun.

## Audience, task, and outcome

**Audience.** The root integrator, B5 reviewer, and a future maintainer reading
the durable evidence.

**Task.** Implement and verify opt-in per-environment PQN autoreset, then run a
frozen three-pair matched throughput screen with a controlled barrier arm.

**Outcome.** B1–B4 are accepted. B5 stopped after its first pair because the
second child process could not be confirmed as exited by the parent monitor.
Four arms were left unrun, without replacement or retry. The protocol therefore
returns `INCONCLUSIVE_STOP`; there is no accepted pair, paired median, or
advance decision.

## Evidence and authority

- Campaign ledger: [`campaign.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/campaign.json).
- Frozen B0 opt-in contract:
  [`contract.md`](pqn_per_env_autoreset_v1/contract.md).
- Approved A3 timing analysis:
  [`a3-execution-analysis.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-timing-calibration-v1/review/a3-execution-analysis.md).
- Accepted B4 receipt:
  [`b4-verification-receipt-20260912-v1.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/b4-verification-receipt-20260912-v1.json).
- B5 v2 frozen boundary and terminal result:
  [`boundary.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/executions/matched-20260912-v2/boundary.json),
  [`results.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/executions/matched-20260912-v2/results.json),
  and [`root-post-exit-check.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/executions/matched-20260912-v2/root-post-exit-check.json).
- B5 closure, which hashes all 27 executed files unchanged:
  [`root-execution-closure-20260912-v2.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/root-execution-closure-20260912-v2.json)
  (SHA-256 `75c1e9ff25240e7c8ed0d37f510714989db2559129a73242e8cc2d6003891865`).
- Accepted independent raw outcome audit:
  [`independent-outcome-audit-20260912.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/review/independent-outcome-audit-20260912.md)
  (SHA-256 `3e7dabc8a874a9a43da0435cbb108b20eb23a037d8039409f1921e0ff0b4fb69`).
- Pair-one parent receipts:
  [`batch barrier`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/executions/matched-20260912-v2/pair-1-batch_barrier_v1.parent-receipt.json)
  and
  [`per-environment autoreset`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/executions/matched-20260912-v2/pair-1-per_env_autoreset_v1.parent-receipt.json).

`tournament_eval.py` remains the only promotion authority. B5 performs no
quality evaluation and cannot substitute for that gate.

## Completed implementation and verification

The accepted B1–B4 work supplies masked simulator reset, per-environment
assignment and lifecycle metadata, trainer provenance, serving evidence, and
the independent autoreset-oracle suite.

| Area | Accepted evidence |
| --- | --- |
| Serving lifecycle (`9265eb59`) | B3S final suite: 108 passed in 13.66 s, covering both native modes through real serving receipts and strict final consumption. |
| Default compatibility and autoreset oracles (`9265eb59`) | B4 final: 9 passed in 1.23 s; default parity reported zero known differences and 44 additive fields. |
| Full non-slow regression suite (`9245c003`) | 2,602 passed, 6 skipped, 3 slow deselected in 91.38 s. |
| Slow parity (`9245c003`) | 3 passed in 111.71 s. |
| Native smoke paths (`9265eb59`) | CPU and MPS smokes passed for both modes. |

The full and slow suites ran on `9245c003c4fd507dcc97adcee1b8ea2980ee3419`.
Focused serving, B4 recapture, and native smoke evidence ran on frozen source
`9265eb59b58ab752aa8586922d430754e910cb44`, which also supplied the B5 source
and runtime. Test isolation occurred before that B5 execution. Only the final
report and any supervisor repair are post-experiment.

Runtime defaults remain legacy `batch_barrier_v1 + continuous_env_rng_v1` with
`raster31v2`. The `corrected-v3` and `per_env_autoreset_v1` path is opt-in.
The original clean `main` base `950d583b9eb0c3e316edd0f651954d133334ac1a`
remains unchanged.

## A2/A3 timing boundary

A2/A3 remains approved timing evidence only. Both calibration arms completed on
clean source `7cb66a9437f63b89cd5dc8731b5f8577ce1c9765`; the predeclared
process-wall formula calculated a future 500,000-useful-hero-transition cap of
822 seconds. It is neither a throughput comparison nor a learning-quality,
evaluation, promotion, or release result.

The earlier five X0 wall-stopped arms remain descriptive context. They used the
older continuous-RNG, legacy-sampler, dynamic-pool setup and must not be
retrofitted with the 822-second cap or treated as paired quality evidence.

## B5 v2 terminal result

The frozen execution root is
`x0-autoreset-matched-v1/executions/matched-20260912-v2`. It planned three
serial pairs, each with a 200,000-useful-hero-step target. The parent process
exited with code 2 and recorded `INCONCLUSIVE_STOP` after pair one's second arm.

| Arm | Worker result | Parent result | Useful steps / updates | Wall time |
| --- | --- | --- | ---: | ---: |
| Pair 1, `batch_barrier_v1` | Completed at useful budget | Completed | 200,115 / 197 | 221.623155 s |
| Pair 1, `per_env_autoreset_v1` | Terminal receipt completed at useful budget | Failed: `child_exit_unconfirmed` from a process-monitor race; child return code unavailable | 200,376 / 173 | 226.953658 s worker; 228.652235 s parent |
| Pairs 2–3, both modes | Unrun | `prior_child_exit_unconfirmed` | — | — |

The root post-exit check later observed both CPU locks reacquired and the
reported child no longer alive, but that observation does not repair the parent
monitor's terminal record. The controlling protocol result remains
`INCONCLUSIVE_STOP`.

The two workers recorded 400,491 useful hero steps in total, but only the
barrier arm's 200,115 steps belong to an accepted completed arm. For each worker
terminal record, sampled draws, unique sampled rows, and useful hero transitions
were equal, so each draw ratio was 1.0 within its own arm. Their first 97
semantic rows matched exactly before the first reset. These facts validate the
sampled implementation boundary; they do not create an accepted pair after the
parent failure.

The recorded throughputs are descriptive only: 54,177.10 useful hero steps per
process-wall minute for the barrier arm and 52,580.11 for the autoreset arm,
or -2.948% for autoreset. Since the pair was not accepted, the paired median is
null and the predeclared 15% advance criterion was never evaluable.

| Resource observation | Barrier | Autoreset worker |
| --- | ---: | ---: |
| Peak RSS | 3.06 GB | 2.71 GB |
| Peak MPS driver allocation | 1.213 GB | 1.203 GB |
| Minimum available system memory | at least 27.5 GB | at least 27.5 GB |

Both terminal workers stayed within the frozen RSS, MPS-driver, and available
memory caps. No numerical tripwire or evaluation was recorded.

## What this establishes—and what it does not

The useful implementation result is that the accepted autoreset path has
default-parity, focused oracle, serving, CPU, and MPS evidence, and that one
frozen matched pair produced terminal worker receipts within the resource caps.
The parent process-exit race prevented that pair from becoming a valid throughput
comparison under the protocol.

This experiment therefore does not establish an autoreset throughput gain, a
policy-quality gain, an RL winner, or a promotion candidate. The future
dynamic-pool, PPO, and quality-gate directions remain conditional and have not
started. Any later work needs a newly frozen decision and must retain this
protocol stop rather than silently continuing it.

## Post-experiment supervisor repair

The B5 diagnosis found a monitor weakness: a pinned H0 `NoSuchProcess` race
entered signal-before-reap, while the original OS error details were discarded.
The exact OS error is therefore unknown. The future-only repair is implemented
in `src/scripts/pqn_correctness_diagnostic.py` with tests: attempt a bounded
`Popen.wait()` before signalling, retain the error details, and continue to fail
closed when an actual exit code is unavailable.

The durable diagnosis is
[`b5-process-monitor-race-diagnosis-20260912.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/review/b5-process-monitor-race-diagnosis-20260912.md)
(SHA-256 `06c1361ee1e28c16b2ceb70723d5f201363f40b4e01184a2dc1d9a93e4a61c76`).

The post-experiment repair source digest is
`5f471e07cb369852c326b50775f9fc493608123fb28a97dae84155b9a3fe8a52`.
Its focused verification recorded 43 passes in 12.58 s at
[`supervisor-fix-focused-v4.log`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/verification-20260912/supervisor-fix-focused-v4.log).
Direct review approved the exact source and test hashes, and the two-file repair
is committed as `b51b1588a3698b0ab20b051c98d83317b5dcc0f9`. The full suite was
not rerun after this post-experiment repair; the earlier full-suite evidence
therefore remains scoped to the frozen B5 source.

The repair receipt is
[`supervisor-repair-receipt.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/verification-20260912/supervisor-repair-receipt.json)
(SHA-256 `738f0dfd22c30ce32013db2d7fa94032cff11beceb6ca6adc927663d98e92999`),
with the direct review at
[`b5-future-supervisor-fix-review-20260912.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/review/b5-future-supervisor-fix-review-20260912.md).

That repair is post-experiment and does not alter frozen source `9265eb59`, the
B5 receipt, or the `INCONCLUSIVE_STOP` result. It does not authorize a retry.
