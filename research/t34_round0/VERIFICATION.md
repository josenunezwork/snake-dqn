# Round 0 implementation verification — 2026-10-20

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. Implementation is prepared;
scientific execution, ratification, spend and release have not occurred. The
operator runbook is [README.md](README.md). The original draft and bank/audit
bytes remain intact; [the A5 erratum](ERRATUM_A5_ADDENDUM_B.md) applies the later
binding Amendment 2 addendum B.

## Final source and bounded execution

Final executable source: `7f7bc147b669d87d786a5dff0d25e786ae2a6470`.
[VERIFICATION_A6_FINAL.json](VERIFICATION_A6_FINAL.json) binds that source inventory,
the current retention-bank bytes, the original bank/audit and the tabular audit.
It reports PASS: two excluded H500 paired fixtures, one CPU thread, exact released
B8 s3, 18.685644915996818 seconds. The raw create-only evidence is under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-20261020/a6-04/`.
The chunk SHA256 is
`941dcd5952fa2185233c65a1eb5103929af69fe583487ffc053fc9f11757ca07`.
[SOURCE_QUALIFICATION_FINAL.json](SOURCE_QUALIFICATION_FINAL.json) records local
source preparation only: 829 files, 3,352,335 compressed bytes and 13,338,868
unpacked bytes, below both 32 MiB bounds, with no allocation. Raw archive and
plan are under the sibling external `source-03/` directory.
The earlier A6 and source receipts remain available; they are historical source
snapshots, not substitutes for the final receipts.

All work ran on the Mac with at most two concurrent numerical workers, each
limited to one numerical thread. No live Colab operation or tmux command was
issued. No selection-bank arm was run. A6 is the expressly specified excluded
preflight, not scientific evidence. No checkpoint, served pin or live worker was
changed by this implementation.

## Passing checks

All pytest commands used this prefix (the final focused run also used the CPU
setting):

```bash
export T34_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
```

Final focused verification after the A5 and CUDA integration fixes:

```bash
"$T34_PY" -m pytest -q \
  tests/test_t34_round0_analysis.py tests/test_t34_round0_manifest.py \
  tests/test_t34_round0_registration.py tests/test_t34_round0_runner.py \
  tests/test_t34_batch_benchmark.py tests/test_t34_shadow_profile_receipts.py
```

**50 passed in 22.95 seconds**. This covers numerical decisions and integrity
refusals, incomplete exports, fault priority, 100k action windows, both replay
arms, current retention collisions, manifest caps, source admission, current A5
acceptance and the native CUDA opponent-dispatch opt-out without executing CUDA.

Integration verification of the teacher batching changes and opt-out:

```bash
"$T34_PY" -m pytest -q \
  tests/test_t34_round0_analysis.py tests/test_t34_round0_manifest.py \
  tests/test_t34_round0_registration.py tests/test_t34_round0_runner.py \
  tests/test_t34_shadow_batching.py tests/test_t34_batch_vector_device.py \
  tests/test_t34_shadow_batch_oracle.py
```

**52 passed in 175.00 seconds**. This run preceded the A5 admission erratum; the
final 50-test run above verifies that subsequent change. The oracle tests exercise
the existing serial-versus-batched teacher seam, not Round 0 selection worlds.

Final Black, isort, flake8, shell syntax and whitespace checks passed:

```bash
T34_FILES=(research/t34_round0 research/t34_exit_b8/teacher_v2.py \
  tests/test_t34_round0_analysis.py tests/test_t34_round0_manifest.py \
  tests/test_t34_round0_registration.py tests/test_t34_round0_runner.py)
"$T34_PY" -m black --check --workers 1 --line-length 100 "${T34_FILES[@]}"
"$T34_PY" -m isort --check-only --profile black --line-length 100 "${T34_FILES[@]}"
"$T34_PY" -m flake8 --max-line-length 100 "${T34_FILES[@]}"
bash -n research/t34_round0/run.sh
git diff --check
```

The project `make PY="$T34_PY" lint` also passed earlier (149 core/backend files).
The operating-characteristic simulation is development-only, 20,000 repetitions
per model/grid. The validator checks its current source/pool hashes and protected
regression error. It does not claim adequate joint power at the point thresholds;
see [OPERATING_CHARACTERISTICS.md](OPERATING_CHARACTERISTICS.md).

## Full-suite result and isolated failures

The complete non-slow suite was run on fixed commit
`d83fc1c785eb6d3d6b7114661de40e582567939c`, before the final concurrent-main
integration and the two integration fixes:

```bash
"$T34_PY" -m pytest -q -n 2 -m 'not slow'
```

**8,603 passed, 29 skipped, 28 failed, two warnings in 2,343.06 seconds**.
Log: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-20261020/pytest-nonslow-d83fc1c7.log`.
This is not an all-green full-suite claim. Earlier superseded duplicate test runs
were stopped only by their verified owned PIDs and are not counted as passing.

An independent verifier reran all 28 failed nodes at the original clean commit
`4c19a83e0b0826d7463e752775d7688f6f0e1865` in detached worktree
`/Users/josenunez/Projects/ml/snake-dqn-round0-baseline-check-20261020`, using one
CPU thread. Result: **15 failed, 13 passed in 25.83 seconds**.

- All 13 Q1 admission failures reproduce on the starting commit: the historical
  fixture's allocation-dispatch assumption has expired and requires reconciliation.
- Both T22 frozen-policy failures reproduce on the starting commit: its existing
  frozen `v8_decide` API does not accept the passed `membership` argument.
- The remaining 13 compile tests pass on the starting commit and fail in this
  worktree. Isolated current rerun: **13 failed in 19.84 seconds**. The common
  error is Torch Inductor's generated library failing to load
  `@rpath/libc++.1.dylib`; eager fallback then fails compilation assertions.

The compile/test/conftest source files are byte-identical between the two commits.
A fresh TorchInductor cache reproduced one representative case: baseline **1
passed in 15.76 seconds**, current **1 failed in 6.69 seconds**. `otool -L`
showed direct `/usr/lib/libc++.1.dylib` linkage in the baseline generated library
versus unresolved `@rpath/libc++.1.dylib` in the current one. This excludes stale
shared cache and xdist-only explanations. The evidence indicates a worktree/toolchain
code-generation and linkage difference; its complete cause remains unresolved.
No wheel, global environment or unrelated compiler code was modified to suppress it.

Independent logs:

- `/Users/josenunez/ml-snake-dqn-artifacts-t34-baseline-20261020.log`
- `/Users/josenunez/ml-snake-dqn-artifacts-t34-current-perf-compile-20261020.log`
- `/Users/josenunez/ml-snake-dqn-artifacts-t34-inductor-baseline-bv8-fresh-20261020.log`
- `/Users/josenunez/ml-snake-dqn-artifacts-t34-inductor-current-bv8-fresh-20261020.log`

## Review and operator hold

Independent read-only review covered the producer, analysis authority, supervisor
receipts, source/seed audits and final upstream teacher integration. The CUDA
opponent recipe and superseded A5 floor findings were fixed and reviewed; no
remaining material finding was reported.

Actual A5 evidence, ratification and production manifests are deliberately absent.
The README gives exact later ratify, manifest, supervisor validation, dispatch,
Mac replay and analysis commands. Missing evidence holds dispatch. Both arms
remain paired on one runtime; a GO would be development premise evidence only.
Mac strict gating, product tests, serving qualification and release remain separate.
