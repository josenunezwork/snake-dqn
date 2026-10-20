# T34 retention FIX-FIRST #3 verification — 2026-10-20

Status: **implemented, independently reviewed, and verified; pre-data draft remains unratified**.
Authority: [operator rulings](../../docs/research/specs/2026-10-20_t34_retention_fixes3.md).

No efficacy rows, qualifying timing, T3 pre-check, training, live service, tmux,
cloud allocation, serving change, or checkpoint/pin modification was performed.
All execution was on the Mac with numerical pools fixed to one thread; tests
used no xdist workers. At most two single-thread verification commands ran together.

## Implemented behavior

- Each phase has its own contention ledger, with a fixed registered-cap denominator.
  Timing uses its existing 40-hour elapsed cap (2,880 seconds contention allowance),
  pre-check uses 8 hours (576 seconds), and study uses 40 hours (2,880 seconds).
  The timing forecast of approximately 0.36 hours is not a registered cap.
  Timing remains charged to the overall study wall/CPU budget, but its contention
  is not carried into the study phase's separate contention ledger.
- Each contended sample adds exactly five seconds. Tick reads and repeated
  summaries add nothing. The two-percent budget is enforced only at phase end;
  `between_rows` continues even if an interim total exceeds that allowance.
- Load above the measured admission baseline plus 3.0 must persist continuously
  for more than 60 seconds. Other processes above 50% CPU use the same strict
  greater-than-60-second persistence rule. Recovery resets persistence. Samples
  record the measured baseline, threshold, CPU count, and process evidence.
- Every registered phase requires 120 seconds of quiet host telemetry before
  output reservation and gameplay. Raw CPU/load excess refuses even during the
  persistence grace period. Refusal propagates `HostAdmissionError`, exits
  nonzero, constructs no world, and writes no output root or terminal record.
  Quiet admission time precedes the phase's elapsed debit.
- Sampling closes before final validation. Publication is fenced under the
  sampler lock after shutdown, including late successes and exceptions from an
  in-flight host-tool call. A late sample cannot alter a saved PASS ledger.
- The analyzer verifies supplied cap-normalized ledgers, including timing ledgers
  whose probe flags claim quiet. Missing or incompatible accounting fails closed.
- The prospective preregistration Amendment 4 explicitly limits unconditional
  KILL faults to bare/paced. Failed or missing unpaced Colab rows remain
  descriptive and cannot affect the verdict.

## Verification

Commands below ran from `/Users/josenunez/Projects/ml/snake-dqn-retention-fixes3`.

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1

"$PY" -m pytest -q -n0 \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py \
  tests/test_t34_retention_fixes*.py \
  tests/test_t34_pathb_controller_v2.py \
  tests/test_t34_pathb_adapter_v2.py \
  tests/test_t34_pathb_precheck_v2.py tests/test_t34_fast_digest.py
# 179 passed in 16.37 seconds, no skips/deselections.

CHECKS=(research/t34_pathb/{host_guard_v2,retention_v2,analyze_v2}.py
  tests/test_t34_pathb_{analysis_v2,retention_v2}.py
  tests/test_t34_retention_fixes_runner.py
  tests/test_t34_retention_fixes2_{guard,runner}.py
  tests/test_t34_retention_fixes3_{analyzer,guard,runner}.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
# PASS for all changed Python files/tests.

"$PY" -m black --workers 1 --check --line-length 100 src web/backend
"$PY" -m isort --profile black --check-only src web/backend
"$PY" -m flake8 --jobs 1 src web/backend
# PASS; Black checked 149 files.

"$PY" -m research.t34_pathb.retention_v2 --help
# PASS, actual CLI imports/parser only; no gameplay.
```

The original retention/adjacent selection was retained and expanded. Earlier
integration runs passed 165 and 170 tests; after the supplied-ledger hardening,
58 focused tests passed. The final 179 includes the incoming main digest changes.
Independent read-only review passed 17 new regressions before the last ledger
hardening, reproduced the slow-sampler race, and confirmed its fix with no
remaining blocking finding. The final suite includes the added ledger regression.
No full-repository pytest claim is made.

## Refreshed draft bindings

Source implementation was committed and published before planning regeneration.
The development-only OC uses historical development records, produces no gameplay,
and retains N, thresholds, banks, and the seed audit. An incoming main merge added
full-state digest changes during publication; `oc-01` was preserved and superseded
by a fresh `oc-02` against that merged closure. This was source-binding refresh,
not a scientific retry or a changed decision rule.

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes3-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-02.json" --state "$E/oc-02-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS; 20,000 repetitions, zero failed cells, efficacy_rows_produced=false.
# oc-01 used the same options with oc-01 paths and also passed 20,000 / zero.
# oc-02 copied to RETENTION_OC_V2.json and committed before draft build.

"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-01.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Copied to REGISTRATION_RETENTION_V2.json.
"$PY" -m research.t34_pathb.registration_retention_v2 validate
# {"valid": true, "authorized": false, "mode": "draft"}
```

Draft source commit: `01375add3ca96741269b4718041381df1d6fed63`.
Its 206-file source closure matches the current implementation and the completed
OC. The final draft pins committed OC bytes and remains unauthorized for execution.

| Artifact | SHA-256 |
|---|---|
| `REGISTRATION_RETENTION_V2.json` | `d51775b28f07d1f8c4a245490227741a4cf3e0fc5a19e12fc57ec49aff529e60` |
| `RETENTION_OC_V2.json` | `0e2bd35bce5e48ebd6867c85c911d427b1539b53a2e63be7b9c9804e95205537` |
| `RETENTION_BANK_V2.json` (unchanged) | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| `SEED_AUDIT_RETENTION_V2.json` (unchanged) | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

The authorized draft preregistration/profile/planning package was refreshed
prospectively. Historical verification, result receipts, released artifacts,
serving pins, and frozen banks were preserved. Ratification, final-source trace,
real quiet-host admission, timing, pre-check, and efficacy remain operator work
outside this implementation task. No such phase was launched.
