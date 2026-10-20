# T34 retention FIX-FIRST #6 — implementation and verification

Implemented [FIX-FIRST #6](../../docs/research/specs/2026-10-20_t34_retention_fixes6.md)
in `/Users/josenunez/Projects/ml/snake-dqn-retention-fixes6`, branch
`t34-retention-fixes6-20261020`. Initial tree was clean at
`45a089ecad6e9bebc2075b0bdaa8d629ed053e1f`.

The planning package remains **DRAFT_PENDING_OPERATOR_RATIFICATION**, with
`ratified: false`. Validation grants no execution authority. No real timing,
precheck, efficacy, training, services, tmux, cloud work, or production admission
claims were run. All compute was local on the Mac. Numerical pools were capped
to one thread; tests were serial. Parallel specialist work was limited to
read-only review and one bounded test-file owner.

## Behavior and files

- `retention_history_v2.py` (new): registered authoring-checkout JSONL ledger;
  complete terminal file/seal hash verification; fresh `origin/main` fetch and
  byte equality; append-only published Git-history verification; durable
  post-source-fence append. The authoring checkout must differ from the detached
  executable checkout, so ledger writes never weaken the source fence.
- `retention_history_v2.py`: before directory creation, publishes one unique,
  non-force, create-only remote admission tag per fixed phase/attempt under
  `refs/tags/t34-retention/<study_id>/<phase>`. This closes hard-kill plus local
  deletion before terminal sealing. The claim binds the registration/path and
  declares unmeasured unfinished interruption Claim B KILL. A duplicate or
  unavailable remote refuses before gameplay. Terminal records still require
  operator commit and push of the ledger to main before proceeding.
- `retention_v2.py`: create-only fixed `<study_root>/precheck` and
  `<study_root>/efficacy`; efficacy requires the original fixed precheck receipt.
  Directory reservation precedes host admission and cannot be reused. Published
  terminal history is required before all further phase/attempt admissions.
- `retention_v2.py`: BaseException and SIGINT/SIGTERM/SIGHUP handling; unmeasured
  interruption/operator abort KILL; source/abort markers invalidate preserved
  decisions and block downstream admission. Late abort after saved GO cannot
  preserve an effective GO. Finalization failure preserves evidence, invalidates
  the result, and blocks progress without rewriting immutable receipts.
- `host_guard_v2.py`: attaches actual recorded measurements only when its
  measured contention/power rules interrupt a phase. These retain the registered
  INCONCLUSIVE/retry semantics. Sampler errors, missing telemetry, and unrelated
  earlier contention do not excuse interruption.
- `registration_retention_v2.py`: builds and validates the fixed study-specific
  ledger policy. The new module is included by the actual executable closure.
- `PREREGISTRATION_RETENTION_V2.md`: prospective Amendment 7 states the fixed
  paths, publication, durable admission, and abort rules. This expressly requested
  draft update changes no ratified registration or historical receipt.
- `README_V2.md`: updates efficacy path and publication/operator sequence.
- `tests/test_t34_retention_fixes6.py` (new): 23 synthetic cases covering fixed
  paths, actual temporary bare-Git origin publication, remote claim consumption,
  deletion/tampering, published rewrites, late aborts, measured interruptions,
  original precheck admission, and descriptive analysis preserving KILL.
- Existing focused test fixtures now isolate the new Git transport while their
  original local contracts remain tested. Cap/source failures expect KILL under
  the new prospective rule. No frozen bank, profile, or seed audit changed.

## Final verification

Run from the worktree root using the primary interpreter; no second venv:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes6
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 BLACK_NUM_WORKERS=1

"$PY" -m pytest -q -n0 \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py tests/test_t34_pathb_timing_v2.py \
  tests/test_t34_retention_fixes*.py \
  tests/test_t34_pathb_controller_v2.py tests/test_t34_pathb_adapter_v2.py \
  tests/test_t34_pathb_precheck_v2.py tests/test_t34_fast_digest.py --tb=short
# 282 passed in 27.05s: existing 259 + new 23; no skips/deselections.

CHECKS=(research/t34_pathb/{retention_history_v2,retention_v2,host_guard_v2,registration_retention_v2}.py
  tests/test_t34_pathb_retention_v2.py tests/test_t34_pathb_analysis_v2.py
  tests/test_t34_retention_fixes2_runner.py tests/test_t34_retention_fixes3_runner.py
  tests/test_t34_retention_fixes4.py tests/test_t34_retention_fixes5.py
  tests/test_t34_retention_fixes6.py tests/test_t34_retention_fixes_runner.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
# PASS: all 12 changed Python files.

"$PY" -m black --workers 1 --check --line-length 100 src web/backend
"$PY" -m isort --profile black --check-only src web/backend
"$PY" -m flake8 --jobs 1 src web/backend
# PASS: Black checked 149 files.

"$PY" -m research.t34_pathb.retention_v2 --help
"$PY" -m research.t34_pathb.registration_retention_v2 build --help
# PASS: real parser/import paths, no gameplay.
```

Development runs first exposed expected old-fixture boundary failures; subsequent
runs reached 116/118, then 258/259, then 279/280 as stricter abort/publication
contracts were added. All were resolved before the final 282-case passing run.
The independent static reviewer found late-abort and hard-kill deletion paths;
repairs and regression tests closed them. Final review: no remaining blocking
findings. GitHub tag publication was not exercised against the production remote;
the create-only protocol was tested against an actual local bare Git remote.

## Published source and rebuilt planning artifacts

Implementation: `04cacd62`. Merge of incoming campaign-log/handoff-only updates:
`b15f80e21e71441b7b92b3f29ad6edf7c3a40de2`. Both published to main and the branch.
These incoming documentation updates are outside the executable source closure.

Independent verification also confirmed the executable closure contains 207
files including `retention_history_v2.py`; all four changed executable modules
passed `py_compile`. No changed files are under `src/` or `web/` and there are
no new training/serving imports. The verifier independently checked final test
logs, the 12-file formatting/lint selection, and diff hygiene.

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes6-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-01.json" --state "$E/oc-01-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS: 20,000 repetitions, zero failed cells, efficacy_rows_produced=false.
# Exact receipt copied to research/t34_pathb/RETENTION_OC_V2.json.
# Committed/published at 97232d51c94cfa1cbe00839c45601dfa632508fc.

"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-01.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Exact bytes copied to research/t34_pathb/REGISTRATION_RETENTION_V2.json.
"$PY" -m research.t34_pathb.registration_retention_v2 validate
# {"valid": true, "authorized": false, "mode": "draft"}
```

Draft source commit: `97232d51c94cfa1cbe00839c45601dfa632508fc`.
Its 207-file source closure equals the fresh OC's closure, and the retained
OC is committed before the draft source freeze. There was no reuse of stale
identity receipts. Registered study root remains:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-v2-operator-01`.
Registered authoring repo is this worktree, and ledger path is
`research/t34_pathb/ledger/t34-retention-v2-operator-01.jsonl`.
No production phase directory or admission tag was created by this task.

| Artifact | SHA-256 |
|---|---|
| `REGISTRATION_RETENTION_V2.json` | `db5d996098f73a71d9e084612630419f0d2d3285ff27bfafcd37ea99e8c978ca` |
| `RETENTION_OC_V2.json` | `870060a17f0a2ad31af5d84910998c19ab42afab924cbb3c2254687fc809e180` |
| `RETENTION_PROFILE_V2.json` (unchanged) | `2ff16b5d4223c9a1057db21e87786d83a0b6f6c0ba1c47338281c27d097ca4fd` |
| `RETENTION_BANK_V2.json` (unchanged) | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| `SEED_AUDIT_RETENTION_V2.json` (unchanged) | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

Operator read-only review command (no execution authority):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes6
SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t34_pathb.registration_retention_v2 validate
```

No remaining implementation deviations or blocking review findings. The remote
admission claim is the necessary hard-kill-safe extension of terminal history;
its prospective rule is recorded in Amendment 7. These are synthetic/local Git,
source-binding and development-only OC checks, not measured timing, efficacy,
strict-gate or serving qualification. Ratification remains separate. No tmux,
live service, released checkpoint, serving pin, frozen bank, seed audit, or
historical hash-bound file was modified. Only the expressly requested draft
preregistration/registration and development OC planning package was refreshed.
