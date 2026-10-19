# A4 wider test-suite limitations — 2026-10-18

The final scoped Q0/Q1/queue run passed **339 tests**. The additional whole-repo
non-slow run was not green: **22 failed, 7678 passed, 26 skipped, 5 warnings in
398.64 seconds**. It used three xdist workers and also encountered two native
worker crashes while pytest was formatting NumPy arrays. This record keeps that
failure separate from the scoped A4 and manifest admission evidence.

One directly related failure was corrected: the synthetic Q1 serialization test
monkeypatches world execution in-process, so the new spawned-worker default
attempted to load synthetic checkpoint bytes in a fresh process. The test now
explicitly uses workers=1 for its serialization-only fixture. The real serial/
spawn proof remains the independently run reserved-world entrypoint. The fixed
serialization test passed in isolation (1 passed in 2.88 seconds) and in the
339-test scoped suite. No product change was needed for this fixture repair.

An independent verifier checked the remaining failures against Git 17a927c9.
All seven affected non-Q1 test files and the corresponding trainer, compile,
serving, and T15 runner sources were byte-identical across baseline and A4.
A clean detached baseline worktree reproduced four representative failures
(M6 produced_monotonic mismatch, T4 actor-payload key mismatch, missing T15
frontend dist, and the serving-fallback expectation), with **4 failed, 3 passed
in 12.72 seconds**. The same failures reproduced on the A4 checkout. The baseline
worktree was clean and was removed; logs remain outside Git.

The compile failures report TorchInductor dlopen errors for
@rpath/libc++.1.dylib / missing LC_RPATH. A separate isolated current run had
3 compile cases fail, including with a fresh compiler cache; the representative
baseline comparison had those three compile cases pass. Compile source bytes
are unchanged, so this evidence supports a local compiler/runtime or test-order
issue, not an A4 compile implementation change. It does not prove every compile
failure reproduces in every baseline context. No global compiler or environment
repair was attempted here.

The actor-control failure reproduced as an ordinary assertion failure and then
as a native exit-139 crash during short traceback generation. The snapshot test
passed alone (1 passed in 0.55 seconds). This is evidence of native/test-context
instability in unchanged actor paths; it is not a clean qualification of those
paths. A whole-repo green gate is therefore **not claimed**.

Exact broader command (from the A4 worktree):

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 3 -m 'not slow'
```

The verifier's representative command was run in both checkouts:

```bash
env PYTHONHASHSEED=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
 -p no:faulthandler --tb=short --disable-warnings \
 tests/test_m6_reward.py::test_levers_on_change_only_the_rewards_by_the_lever_terms \
 tests/test_t4_restart_actor.py::test_restart_default_actor_trace_matches_frozen_pre_t6_baseline \
 tests/test_t15_run.py::test_source_closure_binds_html_css_and_the_actual_built_ui_bundle \
 tests/test_web_served_checkpoint.py::TestLiveStack::test_manual_load_after_b8_fallback_restores_the_requested_v7_variant \
 tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch
```

Diagnostic evidence is under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/q1-a4-20261018-r2/`:
repro-baseline-unchanged.log, repro-legacy.log, repro-perf-compile.log,
repro-perf-compile-repeat.log, repro-perf-compile-clean-cache.log,
repro-t21-control.log, repro-t21-control-short.log, repro-t21-snapshot.log,
and repro-q1-analysis-fixed.log. The broad log is r1/test-fast.log.
An independent existing pytest process (PID 94825, a separate parent and command)
was observed during verification and left untouched. The capacity proof is a
bounded planning measurement with conservative scaling, not an isolated machine
benchmark or Colab hardware qualification.

Original broad-run failure list (before the Q1 fixture correction):

```text
FAILED tests/test_m6_reward.py::test_levers_off_is_the_pre_m6_actor_bitwise[b8]
FAILED tests/test_m6_reward.py::test_levers_off_is_the_pre_m6_actor_bitwise[plain]
FAILED tests/test_m6_reward.py::test_levers_on_change_only_the_rewards_by_the_lever_terms
FAILED tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[bv8-1.0]
FAILED tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[b-1.0]
FAILED tests/test_perf_compile.py::test_compiled_loss_terms_close_on_fixed_batch[b-3.0]
FAILED tests/test_perf_compile.py::test_compiled_updates_stay_close_to_eager[loss]
FAILED tests/test_perf_compile.py::test_compiled_updates_stay_close_to_eager[forward]
FAILED tests/test_perf_compile.py::test_first_steps_verified_then_compiled - ...
FAILED tests/test_perf_compile.py::test_amp_compiled_steps_pass_the_rounding_floor_check[bf16]
FAILED tests/test_perf_compile.py::test_amp_compiled_steps_pass_the_rounding_floor_check[fp16]
FAILED tests/test_perf_compile.py::test_graph_count_is_per_learner - assert (...
FAILED tests/test_perf_compile.py::test_recompile_storm_falls_back_to_eager
FAILED tests/test_perf_compile.py::test_hero_forward_matches_eager_without_recompiles[30]
FAILED tests/test_perf_compile.py::test_hero_forward_matches_eager_without_recompiles[60]
FAILED tests/test_perf_compile.py::test_hero_forward_picks_up_in_place_weight_updates
FAILED tests/test_q1_analysis.py::test_serialized_evaluator_shards_are_accepted_without_rehashing_file_whitespace
FAILED tests/test_t15_run.py::test_source_closure_binds_html_css_and_the_actual_built_ui_bundle
FAILED tests/test_t21_actor_control.py::test_actor_fifo_marker_resume_preserves_ring_and_nstep_continuation[hero]
FAILED tests/test_t4_restart_actor.py::test_restart_default_actor_trace_matches_frozen_pre_t6_baseline
FAILED tests/test_web_served_checkpoint.py::TestLiveStack::test_manual_load_after_b8_fallback_restores_the_requested_v7_variant
FAILED tests/test_t21_actor_snapshot.py::test_actor_state_round_trip_restores_sim_runtime_nstep_rng_and_network
```
