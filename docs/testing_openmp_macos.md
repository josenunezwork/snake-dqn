# macOS pytest OpenMP routing

Run tests normally, using the project venv:

```sh
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 4 -m 'not slow'
```

On macOS, the root `conftest.py` restarts pytest once with torch's `lib` directory
first in the child's `DYLD_LIBRARY_PATH`, before collection. Both torch and
scikit-learn then resolve `libomp.dylib` to the same image. This covers serial
pytest, local xdist workers, and Python subprocesses that inherit this environment.
It does not modify the shell environment, installed wheels, research sources,
thread settings, checkpoints, or existing result artifacts. Runtime routing is macOS-only.
`KMP_DUPLICATE_LIB_OK` is not set.

The guard in `tests/openmp_runtime.py` uses dyld's image enumeration API, rather
than package imports or threadpool metadata. `tests/conftest.py` checks after
collection and before/after each test, including lazy imports. If two distinct
libomp images appear, it fails with both paths. The Mac regression test imports
torch then sklearn in a fresh interpreter, runs threaded torch matrix products
and sklearn histogram boosting plus NumPy array formatting, and verifies that
exactly one runtime is loaded. Its child also enables `DYLD_PRINT_LIBRARIES`.

This choice keeps the research source closure intact. An xdist group alone
cannot isolate these imports: every worker collects every test module, and some
research modules import both libraries themselves.

## Diagnostic evidence (2026-10-18)

Environment: Python 3.12.3, torch 2.9.1, NumPy 1.26.4, SciPy 1.14.1,
scikit-learn 1.9.1. The unrouted mixed-import/threaded-work reproducer loaded:

- torch's `libomp.dylib`, UUID `B2B68D0E-8E1F-3AE8-8A86-8CAA76F48C9E`
- sklearn's `libomp.dylib`, UUID `E3A31AB3-3AE5-3371-87D0-7FD870A41A0D`

The routed reproducer loaded only the first image. Both bounded threaded probes
exited successfully; they reproduced duplicate loading, not a deterministic
segfault. The guard rejected the unrouted process with both library paths.

Six historical crash reports from 20:08:58 through 20:12:30 PT show SIGSEGV in
`raw_array_assign_scalar` / `PyArray_AssignRawScalar`. Four contain both runtime
UUIDs; two contain only torch's runtime. Duplicate loading is confirmed, but
these observations do not establish it as the sole cause of every native crash.
A passing suite establishes the outcome of that run, not immunity to every
possible native failure.

Raw DYLD logs, crash-report extracts and pytest output are stored outside Git at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/openmp-tests-20261018/`.

## Failure reporting and closed shared memory

`pytest.ini` no longer enables `--showlocals` globally. An exception traceback can
retain NumPy views after an actor test's `finally` block closes their mmap-backed
shared memory. Printing those locals dereferences unmapped memory. Disabling
locals formatting preserves the actual assertion and traceback while avoiding
that diagnostic read. Do not enable `--showlocals` for shared-memory actor tests.

The first full non-slow four-worker run, with only runtime routing, recorded
7,903 passed, 26 skipped and 8 failed in 438.80 seconds, including two worker
SIGSEGVs. Both new crash reports contained only torch's libomp. Both crashed
nodes were the `hero` and `learner-seats-3` variants of
`tests/test_t21_actor_control.py::test_actor_fifo_marker_resume_preserves_ring_and_nstep_continuation`.
Their stacks entered `repr_locals` and NumPy `array_repr`.

The tests keep direct mmap-backed arrays at lines 223/252/272 and close their
mappings in `finally` at lines 246/266/284. A focused rerun with
`--no-showlocals --tb=short` returned two ordinary assertion failures comparing
`produced_monotonic`, with zero worker crashes. The assertions remain unchanged.
The new failure-reporting regression raises a deliberate assertion after closing
an mmap with a retained ndarray view, and requires pytest to exit 1 and print the
assertion instead of crashing. The final focused four-worker run recorded
3 passed, 2 ordinary failures and zero native crashes.

This is evidence for two separate issues: duplicate runtime loading and unsafe
failure-local formatting. The duplicate-runtime hypothesis alone does not
explain the observed SIGSEGVs. No research code or expected result was altered.

## Final verification

The final command was:

```sh
SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 4 -m 'not slow'
```

Result: **7,904 passed, 26 skipped, 8 ordinary failures**, 4 warnings,
440.69 seconds; **zero native worker crashes**, zero worker replacements, and
zero duplicate-runtime guard failures. The full suite is not green. Remaining
failures are actor telemetry comparisons in M6/T4/T21, the T15 requirement for a
built frontend bundle, and a manual-load serving fallback expectation in this
fresh worktree. These assertions and resources were not changed for this task.

Before/after native counts: six archived reports (four with duplicate runtimes);
two crashes in the runtime-routing-only full run (both with a single runtime);
zero in the final full run. The archived reports span multiple earlier runs and
are not a matched single-run baseline. Both full runs used source base
`66c9135eab39620094f6336aa0b7db8594af501f` plus this task's test-harness changes.
The second full run was necessary after the first exposed the independent
failure-local formatting crash. Logs are `full-nonslow.log` and
`full-final-nonslow.log` in the artifact directory above.
