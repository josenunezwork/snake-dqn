# Solo diagnostic infrastructure — September 13, 2026

**Status: integrated locally on `main` at `e083116`; no remote push.** The isolated
candidate passed the final independent audit: 107 checks, with zero failures. This
work adds a tightly scoped diagnostic capability; it has not launched solo training,
demonstrated learning, or changed promotion authority.

## What the candidate adds

The opt-in `sole_snake_death_or_frame_cap_v1` lifecycle completes an S1 environment
when hero slot 0 dies or reaches the frame cap. The death-causing transition remains
valid and terminal. Subsequent padded inactive rows remain invalid, so they do not
enter loss, food telemetry, action counts, or the useful-step odometer. The selected
environment resets only at the following rollout boundary through the established
per-environment derived-seed reset path.

Solo checkpoints carry the native lifecycle-v2 metadata, including the explicit
completion mode and terminal semantics. The option is CLI-only: shared YAML rejects
it. Default corrected-v3 checkpoints and histories retain their v1 shape.

The matching `solo-watch-diagnostic-v1` evaluator requires one snake, an empty
opponent roster, a 5,000-frame full horizon, and fresh-reset fixed-ring storage of
capacity 5,002. It stamps diagnostic-only authority. The serving path rejects a
solo checkpoint, and the normal tournament and strict-promotion paths reject the
solo profile, so it cannot be represented as a competitive or promotion evaluation.

## Verification evidence

The four qualified standard suites total 441 passing tests. A separate nine-test
mutation baseline also passed before mutations were run.

| Suite | Result | Peak recursive RSS |
| --- | ---: | ---: |
| Training lifecycle and CLI | 133 passed in 6.09 s | 726,335,488 B |
| Profile, evaluation, storage, and serving | 122 passed in 9.25 s | 565,067,776 B |
| Trainer, Watch phase, and lifecycle regression | 132 passed in 10.64 s | 774,569,984 B |
| Native config, checkpoint, and resume | 54 passed in 1.08 s | 471,482,368 B |

The largest qualified-suite process tree was 0.721 GiB. Across these phases the
lowest recorded host-available memory was 26.70 GiB; resource guards reported no
violation.

The default behavior capture matches clean `633cf6f` on all 10 checked contracts,
including promotion profile, lifecycle, target and sampler contracts, checkpoint
shape, and default CLI behavior. That protects existing non-solo behavior while the
new lifecycle is opt-in.

Mutation proof killed all six planned mutants with no survivors. The mutations cover
sole-death completion, constructor lifecycle validation, the raster31v3 fence,
history metadata, the native checkpoint marker, and v2 terminal semantics. The
largest mutation process tree was 443,138,048 B (0.413 GiB), with no resource-guard
violation. The V1 wrapper-argument and V2 overlay-collection failures are retained
as failed harness attempts; only the V3 executions supply the six kill claims.

## Evidence

- Candidate source: [trainer](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/src/training/pqn_trainer.py>), [lifecycle contract](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/src/training/pqn_lifecycle.py>), [CLI](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/src/scripts/train_pqn.py>), [diagnostic profile](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/src/evaluation/protocol.py>), [SIMD evaluator](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/src/simd_env/eval_engine.py>), and [serving guard](</Users/josenunez/Projects/ml/snake-dqn-solo-diagnostic/web/backend/session.py>)
- [Qualification source manifest](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/qualification-manifest-v6.json>), [default parity](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/default-contract-parity-v2.json>), and [four-suite results](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/training-lifecycle-cli-v2/result.json>) ([profile](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/profile-evaluation-storage-serving-v3/result.json>), [trainer](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/trainer-watch-lifecycle-v3/result.json>), [native](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/native-v2-roundtrip-config-resume-v5/result.json>))
- [Mutation baseline](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/mutation-baseline-v1/result.json>), [V3 mutation summary](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/mutation-summary-v3.json>), [V1 failed attempt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/mutant-death-v1/mutation-result.json>), and [V2 failed attempt](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/mutant-death-v2/mutation-result.json>)

The final [standard-library audit](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/audit.md>)
independently recomputed the 441 standard tests, nine-test baseline, six mutation
kills, and 10 default-parity checks against the clean current `e083116` candidate.
See its [machine-readable result](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/audit.json>)
and [auditor](</Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-infrastructure/audit.py>).

A later frozen solo protocol may use this capability only after root-owned source
selection and freeze complete. It remains diagnostic-only and does not authorize a
promotion or incumbent change.
