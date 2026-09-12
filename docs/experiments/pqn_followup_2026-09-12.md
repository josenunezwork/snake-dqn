# PQN follow-up rolling handoff — 2026-09-12

Status: **prepared for the B5 execution handoff; B5 is not frozen or executable
yet**. This continuity note records the reviewed integration state and the
remaining autoreset throughput screen. It does not report a learned-policy
comparison, quality evaluation, promotion, release, or a change to `main`.

## Audience, prerequisites, and intended outcome

**Audience.** The root integrator and the B5 runner/reviewer.

**Prerequisites.** Read the frozen per-environment autoreset contract, the
campaign ledger, the approved timing analysis, and the B5 preparation note.
The B5 driver must remain blocked until its required B3 and B4 evidence is
byte-pinned.

**Task.** Complete final review and recapture work, repair and freeze the B5
controls, then execute the predeclared three-pair throughput screen only after
its freeze prerequisites are satisfied.

**Desired outcome.** A complete, paired throughput receipt that either advances
the narrow utilization question or records `INCONCLUSIVE_STOP`. It cannot name
a PQN winner or authorize promotion; held-out evaluation remains a separate,
future task.

## Evidence and authority

- Campaign status: [`campaign.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/campaign.json).
- Frozen B0 behavior: [`contract.md`](../pqn_per_env_autoreset_v1/contract.md).
- A2 timing execution and approved A3 analysis:
  [`results.json`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-timing-calibration-v1/executions/calibration-20260909-v1/results.json)
  and
  [`a3-execution-analysis.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-timing-calibration-v1/review/a3-execution-analysis.md).
- B4 verification boundary: [`b4-verification-plan.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/review/b4-verification-plan.md).
- B5 create-only preparation: [`README.md`](/Users/josenunez/Projects/ml/snake-dqn-artifacts/pqn-followup-20260909/x0-autoreset-matched-v1/README.md).

`tournament_eval.py` remains the project promotion authority. B5 has no
evaluation step and cannot substitute for that gate.

## Rolling wave table

| Wave | Current state | Evidence and handoff meaning |
| --- | --- | --- |
| A2 + A3 | Complete and approved, timing only | Both lifecycle-complete calibration arms ran on clean source `7cb66a9437f63b89cd5dc8731b5f8577ce1c9765`: arm 1 recorded 353,647 useful hero steps in 333.459239958 s; arm 2 recorded 357,152 in 469.213795292 s. The formula was predeclared; its calculated result is `ceil(1.25 * max(projected 500k wall)) = 822 s`. No evaluation started and no quality or promotion conclusion follows. |
| B1 | Accepted and integrated | The masked `BatchSim` reset API is accepted at integrated commit `2c2f5b19f8914014efe0eff1649a19a79618696a`; the ledger records 21 focused passes. |
| B2 | Accepted and integrated | Per-environment self-play assignment/exploration work is accepted through integrated commits `24aa532` and `bc7aab2`; the ledger records 125 focused passes and 19 independent new-file passes. |
| B3C | Accepted and integrated | Lifecycle/checkpoint metadata source `ed6c18593534f9a53f926694a2c682e26f881ffe` is integrated as `1cdd7f03020b6b30863133c4d3141ce0552f11cb`; the ledger records 8 focused passes plus one actual-old-checkpoint adapter pass. |
| B3T | Approved and integrated | Reviewed source `cc410b463621945a3a2fa95f3e73ca24d9621876` is integrated as `f12c40c49c749844eca5e2921e698843f2266a62`, with the native-provenance correction at `8f412be`. The authoritative tailed logs record 36 passes in 2.51 s and 59 passes in 2.52 s; earlier 33/52 builder summaries are not evidence. |
| B3S | Final review pending | Native serving source `3f1b32119de78ad14b2441514baf75f461b86952` is integrated as `9245c003c4fd507dcc97adcee1b8ea2980ee3419`. Its strict final suite recorded 108 passes in 14.01 s, covering both native modes through real serving receipts and strict final consumption. |
| B4 | Recapture pending | Test/oracle commit `7ab167b07c43a16d2aab7d7fa03c50c1a7c99eda` supplies nine focused oracles. The older candidate-parity capture has zero known diffs, but the final candidate recapture and acceptance remain pending. |
| B5 | Controls under repair; unrun | The matched three-pair, 200,000-useful-hero-step throughput screen has no frozen execution boundary, terminal arm receipt, or outcome. |

## A2/A3 timing boundary

The predeclared process-wall formula calculates the 822-second cap for a future
500,000-useful-hero-transition arm. It includes parent pre-spawn through
confirmed reap, including startup and save. It is not a throughput comparison
or a training-quality measurement. The approved A3 receipt verifies the source
and receipt chain and explicitly withholds promotion authority.

Keep the original five X0 wall-stopped arms as descriptive historical context.
They used continuous RNG, the legacy sampler, and a dynamic pool; they are not
paired quality evidence and must not be reinterpreted under, or retrofitted
with, the new 822-second calibration cap.

## B5 execution contract to retain

After the mandatory B3/B4/source/checkpoint/hardware pins are supplied, run
three fresh root-seed pairs serially. Each pair compares:

1. `batch_barrier_v1 + derived_env_episode_v1`; and
2. `per_env_autoreset_v1 + derived_env_episode_v1`.

Each arm uses `corrected-v3`, `raster31v3`, mechanics/reward v2,
`sgd_epochs=1`, `flip_augment=false`, `pool_capacity=1`, disabled pool
admission, and the same preloaded frozen snapshot. World, assignment, action,
and SGD streams are derived from the paired root according to the frozen B5
preparation note. Run one MPS learner at a time and retain every terminal arm,
including failures; there are no replacements, retries, or extensions.

The advance condition is deliberately narrow. Every completed pair must show at
least 15% autoreset useful-hero-throughput gain and a draw ratio within 1% of
the barrier arm. Any failed or incomplete pair is `INCONCLUSIVE_STOP`. Either
outcome is operational evidence only, not learned-policy evidence.

## Current verification boundary

At this handoff, the root is running the full non-slow suite on integrated source
`9245c003c4fd507dcc97adcee1b8ea2980ee3419`. That run has no result recorded
here. Slow parity, the bounded CPU smoke, and the MPS smoke remain pending.
These checks, B3S final review, B4's final candidate recapture, and repaired B5
controls must finish before a B5 execution boundary can be frozen.

## Results still needed before this handoff can close

1. B3S final-review evidence, B4's final candidate recapture and accepted
   verification receipt, and the remaining root verification results.
2. Repaired B5 driver controls and a finalized B5 protocol/manifest boundary:
   three fresh disjoint root seeds, checkpoint and preloaded-snapshot proofs,
   and the hardware/monitor receipt required by the preparation note.
3. Six terminal B5 arm receipts: useful hero steps, process wall time, optimizer
   draws and unique rows, resource/incident outcome, resolved mode/configuration,
   realized stream identities, and source/checkpoint hashes.
4. The three raw paired throughput gains and draw ratios, their paired median,
   and the resulting advance or `INCONCLUSIVE_STOP` decision. Preserve all
   incomplete or failed outcomes without a replacement run.

Until these items exist, this is a prepared execution handoff rather than a
completed experiment report. The original clean `main` base
`950d583b9eb0c3e316edd0f651954d133334ac1a` remains unchanged; no publication is
requested by this handoff.
