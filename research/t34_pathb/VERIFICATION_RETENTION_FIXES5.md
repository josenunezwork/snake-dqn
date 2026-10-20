# T34 retention FIX-FIRST #5 implementation and verification

Implemented [FIX-FIRST #5](../../docs/research/specs/2026-10-20_t34_retention_fixes5.md)
prospectively in `/Users/josenunez/Projects/ml/snake-dqn-retention-fixes5`, branch
`t34-retention-fixes5-20261020`. Initial tree: clean at
`0845070527e57cf43a64186db6e076039f2fc9f2`.

The registration remains **DRAFT_PENDING_OPERATOR_RATIFICATION**, `ratified: false`.
Validation grants no execution authority. No real timing, reused-T3 pre-check,
retention efficacy, training, cloud allocation, service, or tmux action ran.
All execution was local on the Mac, numerical pools capped to one thread,
pytest serial, and the reservation race test limited to two Python threads.

## Behavior and changed files

- `retention_v2.py`: validates one absolute canonical `study_root`, its sole
  `timing_attempt_parent: <study_root>/timing`, and `timing_max_attempts: 3`.
  Reserves consecutive create-only `attempt-01` through `attempt-03` before
  host admission. Admission refusal consumes an attempt. Rejects a fourth.
- `retention_v2.py`: timing launch and pre-check/efficacy admission enumerate
  the entire parent. Earlier attempts must be `INCONCLUSIVE-retry` or
  `ADMISSION_REFUSED`; latest exact on-disk PASS alone can admit later phases.
  KILL-timing, KILL, plain INCONCLUSIVE, missing/incomplete/foreign history,
  inconsistent verdicts, changed evidence, gaps, links, and source failure
  refuse further timing and efficacy admission. PASS prevents another launch.
- `retention_v2.py`: each diagnostic binds its path, registration digest,
  source commit, and terminal evidence hashes. A create-only
  `timing_finalized.json` binds that diagnostic after the source fence closes
  and failures are recorded. An unfinished attempt blocks another invocation.
  Invocation-specific reservation ownership prevents a rejected duplicate from
  finalizing another caller's pending attempt. Timing failure uses canonical
  `KILL-timing` in the gate, decision, and attempt.
- `registration_retention_v2.py`: builds and validates the same fixed parent
  and cap contract; `build --study-root` selects the root prospectively.
- `PREREGISTRATION_RETENTION_V2.md`: adds prospective Amendment 6, superseding
  unrestricted fresh-directory retry language. The unauthorized draft is the
  explicitly requested planning document update; no ratified document changed.
- `README_V2.md`: updates operator timing paths and retry/terminal guidance.
- `tests/test_t34_retention_fixes5.py`: adds 54 synthetic regression cases,
  including real producer and source-fence lifecycle seams, terminal history
  blocking every phase, latest PASS, cap, corruption, and concurrent reservation.
- `tests/test_t34_pathb_retention_v2.py`, `test_t34_retention_fixes3_runner.py`,
  `test_t34_retention_fixes4.py`, and `test_t34_retention_fixes_runner.py`:
  supply the newly required policy and numbered paths in existing fixtures.
- `RETENTION_OC_V2.json`: fresh development-only OC against the new source.
- `REGISTRATION_RETENTION_V2.json`: rebuilt unauthorized draft with current
  source, preregistration, OC, and timing-parent bindings.
- This file: verification and operator handoff.

## Verification commands and outcomes

All commands below ran from the named worktree. Use the existing primary venv:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes5
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLACK_NUM_WORKERS=1

"$PY" -m pytest -q -n0 \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py tests/test_t34_retention_fixes*.py \
  tests/test_t34_pathb_controller_v2.py tests/test_t34_pathb_adapter_v2.py \
  tests/test_t34_pathb_precheck_v2.py tests/test_t34_fast_digest.py
# Final: 259 passed in 17.73s; zero skips/deselections.
```

The original 205-case selection passed in 28.53s before the new test file was
added. The initial new-file run passed 45 in 1.16s. An intermediate complete
selection passed 255 in 17.68s. Review then identified the source-finalization
race; its first focused run had 76 passes and two failures from a synthetic
Fence missing its `check()` method. Completing that test double produced:

```bash
"$PY" -m pytest -q -n0 tests/test_t34_retention_fixes5.py \
  tests/test_t34_retention_fixes4.py
# 78 passed in 1.36s, followed by the final 259-case run above.
```

Independent read-only review found the finalization race and re-reviewed its
repair with no remaining blocking findings. The reviewer ran no independent
compute or tests. The new lifecycle test checks refusal while the first
attempt's diagnostic exists but source finalization is pending, and continued
refusal after final source failure.

```bash
CHECKS=(research/t34_pathb/{retention_v2,registration_retention_v2}.py
  tests/test_t34_retention_fixes5.py tests/test_t34_retention_fixes4.py
  tests/test_t34_retention_fixes3_runner.py tests/test_t34_retention_fixes_runner.py
  tests/test_t34_pathb_retention_v2.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
# PASS, all seven changed Python files.

"$PY" -m black --workers 1 --check --line-length 100 src web/backend
"$PY" -m isort --profile black --check-only src web/backend
"$PY" -m flake8 --jobs 1 src web/backend
# PASS; Black checked 149 files.

"$PY" -m research.t34_pathb.retention_v2 --help
"$PY" -m research.t34_pathb.registration_retention_v2 build --help
# PASS, actual parser/import paths; no gameplay.
```

## Source publication and rebuilt draft

Implementation commit: `5e6cfa06`. Merged incoming main's campaign-log update
at `cab7edeb7980d049439c1df79fe7e07ff986e80d` and published main and the branch.
No conflict occurred; incoming changes did not alter the 206-file source closure.

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes5-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-01.json" --state "$E/oc-01-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS: 20,000 repetitions; zero failed cells; efficacy_rows_produced=false.
```

The new OC records source commit `cab7edeb7980d049439c1df79fe7e07ff986e80d`.
Copied its exact bytes to `RETENTION_OC_V2.json`, committed and published at
`e56c80cda39d8e680d4d8eb9a2ac05644ea4fe4e` before building the draft.

```bash
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-01.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Copied its exact bytes to REGISTRATION_RETENTION_V2.json.
"$PY" -m research.t34_pathb.registration_retention_v2 validate
# {"valid": true, "authorized": false, "mode": "draft"}
```

Draft source commit: `e56c80cda39d8e680d4d8eb9a2ac05644ea4fe4e`.
The source closure matches the freshly executed OC; no old identity receipt was
reused. Default registered study root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-01`.
No attempts were created under that production root by this implementation.

| Artifact | SHA-256 |
|---|---|
| `REGISTRATION_RETENTION_V2.json` | `4110750ddbc6098b583698a07aed6e16e55144cfe4ed72c2b31c80ddc09da7df` |
| `RETENTION_OC_V2.json` | `bf93b96a85b16f7a38f4b34f86994dd780e4c80111a24b8de85d688111434361` |
| `RETENTION_PROFILE_V2.json` (unchanged) | `2ff16b5d4223c9a1057db21e87786d83a0b6f6c0ba1c47338281c27d097ca4fd` |
| `RETENTION_BANK_V2.json` (unchanged) | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| `SEED_AUDIT_RETENTION_V2.json` (unchanged) | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

Operator review command (read-only; no execution authority):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes5
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t34_pathb.registration_retention_v2 validate
```

No implementation deviations or unresolved review findings. These are local
synthetic/source-binding checks, not live timing, efficacy, strict-gate, or
serving qualification. Ratification remains separate. No tmux sessions, live
services, released checkpoints, serving pins, frozen banks, seed audits, or
historical hash-bound receipts/documents were modified. Only the expressly
requested prospective draft/preregistration and OC planning package was refreshed.
