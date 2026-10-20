# T34 retention FIX-FIRST #4 implementation and verification

Implemented the operator's [FIX-FIRST #4 rulings](../../docs/research/specs/2026-10-20_t34_retention_fixes4.md)
prospectively. The registration remains **DRAFT_PENDING_OPERATOR_RATIFICATION**,
with `ratified: false`; validation grants no execution authority. No real timing
gate, reused-T3 pre-check, retention efficacy row, training, cloud allocation,
service, or tmux action was started.

Worktree: `/Users/josenunez/Projects/ml/snake-dqn-retention-fixes4`.
Branch: `t34-retention-fixes4-20261020`. Initial tree was clean at `d817a582`.
Implementation commit: `efe0b06e`. It was merged with incoming main at
`c058aeb2de1168caeecccaa1aea3756c5dad7f37` and published to main and the branch.
The draft's final source commit is
`8ca4170fc6df030108a8f1e4178cad089b6f79e2`; its 206-file closure matches both
the verified implementation and the refreshed development-only OC. Subsequent
incoming changes to unrelated fanout files left that closure unchanged.

## Changed files and behavior

Paths below are relative to the repository root.

| File | Change |
|---|---|
| `research/t34_pathb/retention_v2.py` | Enforces the one-hour timing cap in guard, probe, boundary, and final checks; validates timing/pre-check evidence; records each timing attempt create-only. |
| `research/t34_pathb/analyze_v2.py` | Uses phase caps `{timing: 3600, precheck: 28800, study: 144000}`; rejects mismatched ledgers; scored timing contention precedes timing bars as `INCONCLUSIVE-retry`. |
| `research/t34_pathb/timing_v2.py` | Separates warmup and scored tick flags and reports scored windows from `finally`, including failed probes. |
| `research/t34_pathb/host_guard_v2.py` | Caps the admission load baseline at `(os.cpu_count() or 1)/4` and records the ceiling. |
| `research/t34_pathb/registration_retention_v2.py` | Adds `timing_max_elapsed_hours: 1.0` to draft construction and validation; runtime validation also requires it. |
| `research/t34_pathb/RETENTION_PROFILE_V2.json` | Freezes the one-hour timing cap. |
| `research/t34_pathb/PREREGISTRATION_RETENTION_V2.md` | Adds prospective Amendment 5 and replaces superseded timing/baseline language. |
| `research/t34_pathb/README_V2.md` | Updates operator caps, zero-contention admission, and manual retry evidence guidance. |
| `tests/test_t34_retention_fixes4.py` | Adds 24 synthetic regressions for caps, validators, ledgers, baseline ceilings, scored contention, and attempt preservation. |
| `tests/test_t34_pathb_analysis_v2.py` | Makes ledger fixtures phase-specific and updates superseded timing expectations. |
| `tests/test_t34_pathb_registration_retention_v2.py` | Adds the timing cap to draft profile fixtures. |
| `tests/test_t34_pathb_retention_v2.py` | Supplies the timing cap and valid pre-check ledgers to registration fixtures. |
| `tests/test_t34_retention_fixes3_analyzer.py` | Retains cap-normalization coverage using unscored contention under the new timing cap. |
| `tests/test_t34_retention_fixes3_guard.py` | Makes CPU counts explicit in historical baseline persistence tests. |
| `tests/test_t34_retention_fixes3_runner.py` | Requires the timing guard to receive 3600 seconds. |
| `research/t34_pathb/RETENTION_OC_V2.json` | Refreshes the completed development-only OC against the final source closure. |
| `research/t34_pathb/REGISTRATION_RETENTION_V2.json` | Rebuilds the unauthorized draft with new cap and current bindings. |
| `research/t34_pathb/VERIFICATION_RETENTION_FIXES4.md` | Records verification and artifact identities. |

Timing uses a 72-second aggregate contention budget, while pre-check and study
use 576 and 2880 seconds respectively. Zero contended scored timing ticks is a
separate requirement. The runner checks all sampled records against scored
windows after closing the sampler, so a sample inside the final tick cannot
escape the cached tick flags. Window evidence survives a later row-boundary
host exception. Both paths produce `INCONCLUSIVE-retry`, block pre-check and
efficacy admission, and preserve attempt diagnostics. Unconditional Mac
controller faults retain KILL precedence.

Admission refusal remains a nonzero refusal with no terminal phase receipt or
`decision.json`; only diagnostic `timing_attempt.json` is written. Each manual
invocation requires a fresh directory and cannot overwrite a prior attempt.
There is no automatic retry. Timing elapsed/CPU still debit study totals;
pre-check retains its separate allocation.

## Verification commands and results

All commands ran locally on the Mac with one numerical thread and serial pytest;
no more than two CPU threads were allocated. Use the existing primary venv:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes4
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLACK_NUM_WORKERS=1

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
# Final merged-source result: 205 passed in 16.67s; no skips/deselections.
```

The previous 179-case selection is retained, with assertions adjusted only for
the superseding rules and fixtures updated for the new required cap/ledger.
Adding 24 new cases initially produced 203 passed in 16.54s. Incoming main added
two digest cases, producing the final 205 above. The first integration run,
before the new cases were added, had 175 passed / four fixture failures; those
were corrected by supplying valid pre-check ledgers and avoiding missing
optional source metadata in fault diagnostics. A focused correction run was:

```bash
"$PY" -m pytest -q -n0 tests/test_t34_retention_fixes4.py \
  tests/test_t34_pathb_retention_v2.py tests/test_t34_retention_fixes_runner.py
# 51 passed in 2.15s before the final three new regression cases were added.
```

Final style checks:

```bash
CHECKS=(research/t34_pathb/{analyze_v2,host_guard_v2,registration_retention_v2,retention_v2,timing_v2}.py
  tests/test_t34_pathb_{analysis_v2,registration_retention_v2,retention_v2}.py
  tests/test_t34_retention_fixes3_{analyzer,guard,runner}.py
  tests/test_t34_retention_fixes4.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
# PASS: all 12 changed Python files. One docstring line-length error was fixed.

"$PY" -m black --workers 1 --check --line-length 100 src web/backend
"$PY" -m isort --profile black --check-only src web/backend
"$PY" -m flake8 --jobs 1 src web/backend
# PASS; Black checked 149 files.

"$PY" -m research.t34_pathb.retention_v2 --help
"$PY" -m research.t34_pathb.registration_retention_v2 --help
# PASS: actual module imports/parsers, without gameplay.
```

Independent read-only review identified final-tick sample loss and loss of the
retry classification on a host exception. Both were fixed and covered by
`test_final_scored_sample_is_preserved_through_probe_or_boundary_exit`.
Re-review found no remaining blocking issue. These are synthetic/local checks;
no live timing, strict gate, serving qualification, or full-repository pytest
claim is made. Host sampling retains its registered five-second resolution.

## Rebuilt draft and evidence

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes4-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-01.json" --state "$E/oc-01-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS: 20,000 repetitions; zero failed cells; efficacy_rows_produced=false.
# Copied to RETENTION_OC_V2.json, committed, and published before draft build.

"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-01.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Copied to REGISTRATION_RETENTION_V2.json.
"$PY" -m research.t34_pathb.registration_retention_v2 validate
# {"valid": true, "authorized": false, "mode": "draft"}
```

| Artifact | SHA-256 |
|---|---|
| `REGISTRATION_RETENTION_V2.json` | `7270144598c3567a33177129ee7d1bd3f72042bd3bc6bbd68dd979a42b57f8c1` |
| `RETENTION_OC_V2.json` | `9a270030405e71f7927d31bd5489e5a428a26cd7880281e33bfbc711651c94fd` |
| `RETENTION_PROFILE_V2.json` | `2ff16b5d4223c9a1057db21e87786d83a0b6f6c0ba1c47338281c27d097ca4fd` |
| `RETENTION_BANK_V2.json` (unchanged) | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| `SEED_AUDIT_RETENTION_V2.json` (unchanged) | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

Only the explicitly authorized prospective draft/profile/planning package was
refreshed. Frozen banks, seed audit, historical receipts and verification files,
released checkpoints, serving pins, and running jobs were preserved. There are
no implementation deviations or unresolved review findings. Ratification and
all real gameplay phases remain outside this implementation task.
