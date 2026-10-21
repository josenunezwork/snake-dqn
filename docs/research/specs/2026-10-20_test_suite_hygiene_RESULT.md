# Test-suite hygiene result — 2026-10-20

Implementation of [the hygiene spec](2026-10-20_test_suite_hygiene.md).

## Scope and baseline

- Worktree: `/Users/josenunez/Projects/ml/snake-dqn-test-hygiene`.
- Branch: `t34-test-hygiene-20261020`.
- Original source commit: `73a2d794422154b65639ded90260acfd31e2bd2d` (clean checkout).
- Baseline full suite: **35 failed, 8,971 passed, 31 skipped**, 2 warnings,
  **2,681.97 seconds**. Of the failures, **20 were non-Inductor and 15 were
  Inductor-dependent**, including two slow end-to-end compile tests.
- First full verification on upstream `511e075bfe23da1b1895dd1dbc235aa4bc8012f7`
  plus the six-file test patch: **15 failed, 8,994 passed, 30 skipped**, 2 warnings,
  **2,900.98 seconds**. All 20 original non-Inductor failures passed; the failure set
  exactly matched the original 15 Inductor cases. Both added regressions passed.
- `test_t33_round2_parallel.py::test_v8_serial_parallel_painted_owner_bytes_after_reuse`
  was conditionally skipped in the baseline and executed successfully in this run;
  no skip marker or condition was changed by this implementation.
- Main advanced during verification. Upstream changes were fast-forwarded to
  `e301cfeb` without conflicts or changes to our test patch. The upstream changes
  include two existing binding documents (`DESIGN_V2_SHADOW.md` and
  `SUCCESS_CONTRACT.md`); those bytes were pulled from main, not edited by this task.
  All 897 original inventory entries were unchanged before that fast-forward.
  A fresh 936-file protected/source inventory was then captured for the merged run.
- Raw logs, JUnit reports and the protected-source digest inventory are outside Git:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_`.

All pytest invocations use `OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu`,
`./venv/bin/python`, `-p no:cacheprovider`, and at most `-n 2`. The worktree's ignored
`venv` symlink points at the existing primary-checkout environment; no environment,
compiler, installed wheel, or loader configuration was repaired.

## Every original non-Inductor failure

Each row identifies one original failing test. The actor tests retain exact comparison
of deterministic message fields, array dtypes, arrays and observation rings; the existing
telemetry validator separately checks integer accounting and monotonic production stamps.

| Original failing node | Cause | Test-only fix / disposition |
| --- | --- | --- |
| `tests/test_perf_actor.py::test_actor_messages_identical_across_impls[plain]` | Compared `produced_monotonic = time.perf_counter()` across separate executions. | Reuse `WALL_CLOCK_ACTOR_FIELDS`; validate telemetry per run and retain all deterministic parity assertions. |
| `tests/test_perf_actor.py::test_actor_messages_identical_across_impls[shield-labels]` | Compared `produced_monotonic = time.perf_counter()` across separate executions. | Reuse `WALL_CLOCK_ACTOR_FIELDS`; validate telemetry per run and retain all deterministic parity assertions. |
| `tests/test_perf_opp.py::test_actor_messages_identical_with_merged_cpu` | Compared `produced_monotonic = time.perf_counter()` across separate executions. | Reuse `WALL_CLOCK_ACTOR_FIELDS`; validate telemetry per run and retain all deterministic parity assertions. |
| `tests/test_profile_m3c.py::test_actor_messages_unchanged_by_profiling[cprofile]` | Compared `produced_monotonic = time.perf_counter()` across separate executions. | Reuse `WALL_CLOCK_ACTOR_FIELDS`; validate telemetry per run and retain all deterministic parity assertions. |
| `tests/test_profile_m3c.py::test_actor_messages_unchanged_by_profiling[pyinstrument]` | Compared `produced_monotonic = time.perf_counter()` across separate executions. | Reuse `WALL_CLOCK_ACTOR_FIELDS`; validate telemetry per run and retain all deterministic parity assertions. |
| `tests/test_q1_admission_integration.py::test_cli_materializes_120_bound_jobs_and_real_admission_checks` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_a5_refusal_writes_complete_forecast_without_queue_manifests` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[study-plan]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[mac]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[historical-mac]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[timing]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[source]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[packet]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[q0]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[inventory]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[owner]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[amendment]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_q1_admission_integration.py::test_admission_rejects_mutated_bound_evidence[plan-amendment]` | Synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. | Pin the fixture clock to its historical dispatch window; keep the real expiry and admission validators. |
| `tests/test_t22_frozen_pool.py::test_all_six_real_policy_streams_match_frozen_source_and_recording` | Current feature adapter forwarded `membership` to the frozen pre-membership `v8_decide` oracle. | Use a test-side historical scalar adapter calling the byte-pinned old kernel directly; retain recorded action and bytewise parity checks. |
| `tests/test_t22_frozen_pool.py::test_all_six_real_policy_streams_match_prechange_v8_on_physical_long_bodies` | Current feature adapter forwarded `membership` to the frozen pre-membership `v8_decide` oracle. | Use a test-side historical scalar adapter calling the byte-pinned old kernel directly; retain recorded action and bytewise parity checks. |

## Every original Inductor failure retained

No blanket macOS, compiler, or module skip was added. The actual prerequisite is a
successful shape-specific compiled-library load; several production wrappers consume
that exception and fall back before their test assertions. A platform-only `skipif`
would hide unrelated regressions, and a late blanket exception hook would violate the
spec's targeted `skipif` restriction. These failures remain visible and are listed
individually below. No attempt was made to repair the known environment issue.

The baseline shows direct `ImportError: Library not loaded: @rpath/libc++.1.dylib`
from `torch._inductor.codecache` or `BackendCompilerFailed` followed by the eager fallback.
Zero compiled/verification/graph counts and missing compilation status are consequences
of that failed initialization, not separate numerical-policy changes.

| Original failing node | Cause / disposition |
| --- | --- |
| `tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[bv8-1.0]` | Known macOS Inductor C++ library loader failure; retained. |
| `tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[b-1.0]` | Known macOS Inductor C++ library loader failure; retained. |
| `tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[b-3.0]` | Known macOS Inductor C++ library loader failure; retained. |
| `tests/test_perf_compile.py::test_compiled_updates_stay_close_to_eager[loss]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_compiled_updates_stay_close_to_eager[forward]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_first_steps_verified_then_compiled` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_amp_compiled_steps_pass_the_rounding_floor_check[bf16]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_amp_compiled_steps_pass_the_rounding_floor_check[fp16]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_graph_count_is_per_learner` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_recompile_storm_falls_back_to_eager` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_hero_forward_matches_eager_without_recompiles[30]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_hero_forward_matches_eager_without_recompiles[60]` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_hero_forward_picks_up_in_place_weight_updates` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_actor_compiled_hero_forward_end_to_end` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |
| `tests/test_perf_compile.py::test_trainer_runs_with_compiled_learner_and_actor` | Compiled execution unavailable after the known Inductor loader failure; fallback/status assertion remains enforced. |

The compiled actor's later message-comparison loop also used the stale wall-clock
comparison. It now uses the same telemetry contract and explicit message-key equality.
Its compilation prerequisite remains enforced; this host's loader failure still prevents
that later loop from running. This change does **not** claim compiled action parity.

## Verification

Targeted results (all commands run from the assigned worktree):

| Selection | Result |
| --- | --- |
| `tests/test_perf_actor.py tests/test_perf_opp.py tests/test_profile_m3c.py` | 64 passed, 11.04s |
| `tests/test_q1_admission_integration.py` | 14 passed, 4.67s |
| `tests/test_t22_frozen_pool.py` | 11 passed, 3.84s |
| `make PY=./venv/bin/python lint` | Black checked 149 production files; isort and flake8 passed |
| Black `--check --line-length 100`, isort `--profile black --check-only`, flake8 on all six changed Python files | Passed |
| `git diff --check` | Passed |

The two added regressions cover historical dispatch expiry and direct frozen-kernel
live/dead-row handling. No parity tolerance, numerical assertion, production guard,
registered fixture, frozen reference file, or recorded action stream was relaxed.
Independent read-only review found no remaining issues.

**Final merged full suite: 15 failed, 9,082 passed, 30 skipped**, 2 warnings,
**2,885.06 seconds (48m05s)**. **Zero non-Inductor failures.** All 20 original
non-Inductor nodes passed without skipping, both added regressions passed, and the
remaining failure-node set exactly equals the 15 original Inductor-dependent nodes.
The merged run includes the new upstream packages and tests.

The tested base is `e301cfeb30edc8639335cc336754688630887fe4`, with the six-file test patch bound by
SHA-256 `aef93f54fd1c020fea08f0beaf44d7e9e1b3c1742ea100118ab79607cde564dd`.
`tested-source.json` in the artifact directory records every changed test file's hash.
All six hashes still match the tested bytes.

The later upstream fast-forward to `9cf59466634eb9e43c54043a381c01f26c08004f` preserves documentation, result notes,
and the upstream Round 0 ledger record. No Python, shell, configuration, or test
code changed in that fast-forward. All **936** source/protected digests captured
before the merged full run still match afterward. The only baseline-to-upstream
binding-document changes are the two owner-side documents identified above; this
task made no edits to them. The newly pulled ledger was also left unchanged.

`git diff --stat origin/main` contains only the six test files before staging this
new result document. `git diff --name-only origin/main -- research/t34_round0
research/t34_pathb research/t34_exit_b8` is empty. No implementation changes touch
production, simulator, reward, model, training, serving pins, or frozen-reference
bytes. No existing tmux sessions or running research jobs were stopped or modified.

Independent code review and independent artifact/digest verification passed. The
macOS Inductor environment remains unrepaired and its 15 failures remain visible;
compiled action parity is not claimed on this host.

## Exact verification commands

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short --junitxml=/tmp/test-hygiene-baseline.xml > /tmp/test-hygiene-baseline.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short tests/test_perf_actor.py tests/test_perf_opp.py tests/test_profile_m3c.py --junitxml=/tmp/test-hygiene-actor.xml > /tmp/test-hygiene-actor.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short tests/test_q1_admission_integration.py --junitxml=/tmp/test-hygiene-q1.xml > /tmp/test-hygiene-q1.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short tests/test_t22_frozen_pool.py --junitxml=/tmp/test-hygiene-t22.xml > /tmp/test-hygiene-t22.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short --junitxml=/tmp/test-hygiene-final.xml > /tmp/test-hygiene-final.log 2>&1
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -p no:cacheprovider -n 2 -q --tb=short --junitxml=/tmp/test-hygiene-merged.xml > /tmp/test-hygiene-merged.log 2>&1
make PY=./venv/bin/python lint > /tmp/test-hygiene-lint.log 2>&1
./venv/bin/python -m black --check --line-length 100 tests/test_perf_actor.py tests/test_perf_compile.py tests/test_perf_opp.py tests/test_profile_m3c.py tests/test_q1_admission_integration.py tests/test_t22_frozen_pool.py
./venv/bin/python -m isort --profile black --check-only tests/test_perf_actor.py tests/test_perf_compile.py tests/test_perf_opp.py tests/test_profile_m3c.py tests/test_q1_admission_integration.py tests/test_t22_frozen_pool.py
./venv/bin/python -m flake8 tests/test_perf_actor.py tests/test_perf_compile.py tests/test_perf_opp.py tests/test_profile_m3c.py tests/test_q1_admission_integration.py tests/test_t22_frozen_pool.py
git diff --check
```
