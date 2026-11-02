# Architecture, Specifications, and Test-Suite Hygiene (`architecture-and-specs`)

## Question and Goal

What are the core architectural maps, simulation/algorithm specifications, the macOS OpenMP test-runtime contract, and deterministic test-fixture specifications for the `snake-dqn` repository, and how was the full pytest suite repaired in October 2026 (`2026-10-20_test_suite_hygiene`) to eliminate non-Inductor test regressions without modifying any production, research, or configuration source files?

## Timeline of Versions and Steps

| Date | Step / Artifact | What Was Run / Specified | Verdict / Status |
|---|---|---|---|
| 2026-07 – 2026-09 | [`docs/ml_algorithm.md`](../../ml_algorithm.md), [`docs/simd_env_spec.md`](../../simd_env_spec.md) | Design reference for the Apex vector stack (`ApexNetwork`, feedforward dueling DQN, Double DQN + n-step targets, prioritized replay; the raster/PQN track is deferred to `docs/ml_redesign_blueprint_2026-07.md`), and the file:line parity spec of the live simulator's v1/v2 per-frame dynamics for a bit-exact vectorized reimplementation (`src/simd_env/`). | **ACTIVE REFERENCE** |
| 2026-09 | [`docs/codebase_quality_2026-09.md`](../../codebase_quality_2026-09.md) | September 2026 codebase quality, module boundary, and maintainability assessment. | **HISTORICAL REFERENCE** |
| 2026-10-18 | [`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md) | Diagnosed duplicate loading of two distinct `libomp.dylib` images (torch's and scikit-learn's) on macOS; the root [`tests/conftest.py`](../../../tests/conftest.py) now restarts pytest once with torch's `lib` first in `DYLD_LIBRARY_PATH`, and `tests/openmp_runtime.py` fails a test if two libomp images appear. `KMP_DUPLICATE_LIB_OK` is not set; [`pytest.ini`](../../../pytest.ini) no longer enables `--showlocals`. | **ACTIVE CONTRACT** |
| 2026-10-19 – 2026-10-28 | [`tests/fixtures/`](../../../tests/fixtures/) (`phase0_teacher_speed`, `t34_shadow_batching`, `t34_fast_digest`, `ct-supervisor-06`, `d4096_start_metadata`) | Frozen test oracles/fixtures: Phase 0 search-teacher speed oracle (`teacher.py` at `45069720`), frozen T34 serial shadow oracle (`a832e281`), frozen T34 full-digest oracle (`shadow.py` at `c97e6d7d`), raw qualified producer corpus packet 06 for `ct_supervisor` goldens, and D4096 metadata copied 2026-10-25 for the R-A hash-convention regression. | **ACTIVE FIXTURES** |
| 2026-10-20 | Test-suite hygiene pass (`t34-test-hygiene-20261020`, `73a2d794` → `e301cfeb`) | Full-suite baseline (`8,971` passed, `35` failed, `31` skipped in `2,681.97 s`) and 6-file test-only repair (final merged suite `9,082` passed, `15` Inductor-only failed, `30` skipped in `2,885.06 s`) in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_`. | **PASS** (`0` non-Inductor failures; `0` source bytes changed) |
| 2026-10-20 | [`docs/architecture/WEB_MAP.md`](../../architecture/WEB_MAP.md), [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md) | Read-only maps of the server-authoritative web app (`web/`) and of `tests/` (as of main `3d57df7f`: 556 entries, 545 `test_*.py` files). | **ACTIVE REFERENCE** |

## Final Verdict and Key Numbers

The architecture and specification documents remain active reference manuals for the Apex vector stack, the simulator parity contract, the web app, and the test suite. The October 20, 2026 test-suite hygiene pass (`t34-test-hygiene-20261020`, base commit `73a2d794` → final commit `e301cfeb`) achieved its primary goal of **0 non-Inductor pytest failures** across the entire repository while modifying **0 bytes** of production, research, script, web, or configuration source files:

- **Baseline full pytest run (`baseline.xml`):** `8,971` passed, `35` failed (`20` non-Inductor + `15` macOS TorchInductor C++ loader failures), `31` skipped, `0` errors in **`2,681.97 s`** (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/baseline.xml`).
- **First post-fix full run (`final.xml`, upstream `511e075b` + patch):** `8,994` passed, `15` failed, `30` skipped in `2,900.98 s`.
- **Final merged full run (`merged.xml`, base `e301cfeb` + patch):** `9,082` passed, `15` failed (`0` non-Inductor + the `15` original macOS TorchInductor `@rpath/libc++.1.dylib` failures), `30` skipped, `0` errors in **`2,885.06 s`** (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/merged.xml`).
- **Targeted verification (`actor.xml`, `q1.xml`, `t22.xml`):** actor/opp/profile tests `64` passed in `11.04 s`; `test_q1_admission_integration.py` `14` passed in `4.67 s`; `test_t22_frozen_pool.py` `11` passed in `3.84 s`; lint passed ([`docs/research/specs/2026-10-20_test_suite_hygiene_RESULT.md`] (removed in consolidation; see git c4c4ab41)).
- **Protected-source invariance (`protected-merged-baseline.json`, `tested-source.json`):** all `936` source/protected digests captured before the merged full run still matched afterward; the 6-file test patch (`tests/test_perf_actor.py`, `tests/test_perf_compile.py`, `tests/test_perf_opp.py`, `tests/test_profile_m3c.py`, `tests/test_q1_admission_integration.py`, `tests/test_t22_frozen_pool.py`) is bound by SHA-256 `aef93f54fd1c020fea08f0beaf44d7e9e1b3c1742ea100118ab79607cde564dd` ([`docs/research/specs/2026-10-20_test_suite_hygiene_RESULT.md`] (removed in consolidation; see git c4c4ab41)).

### Root Causes of the 20 Repaired Non-Inductor Failures

1. **Monotonic clock comparison across separate executions (5 failures in [`tests/test_perf_actor.py`](../../../tests/test_perf_actor.py), [`tests/test_perf_opp.py`](../../../tests/test_perf_opp.py), and [`tests/test_profile_m3c.py`](../../../tests/test_profile_m3c.py)):** actor-message parity tests compared `produced_monotonic = time.perf_counter()` across runs. Fix: reuse `WALL_CLOCK_ACTOR_FIELDS`, validate telemetry per run, and keep all deterministic parity assertions (fields, dtypes, arrays, observation rings).
2. **Expired synthetic dispatch horizon in Q1 admission fixtures (13 failures in [`tests/test_q1_admission_integration.py`](../../../tests/test_q1_admission_integration.py)):** the synthetic A5 dispatch horizon expired against the real wall clock before the intended admission assertion. Fix: pin the fixture clock to its historical dispatch window, keeping the real expiry and admission validators.
3. **`membership` forwarded to the frozen pre-membership `v8_decide` oracle (2 failures in [`tests/test_t22_frozen_pool.py`](../../../tests/test_t22_frozen_pool.py)):** the current feature adapter passed `membership` to the frozen oracle. Fix: a test-side historical scalar adapter calls the byte-pinned old kernel directly; recorded-action and bytewise parity checks are retained.

The 15 remaining failures, all in [`tests/test_perf_compile.py`](../../../tests/test_perf_compile.py), were left visible (no blanket skip): `ImportError: Library not loaded: @rpath/libc++.1.dylib` from `torch._inductor.codecache` or `BackendCompilerFailed` followed by eager fallback. The environment issue was not repaired and compiled action parity is not claimed on this host ([`docs/research/specs/2026-10-20_test_suite_hygiene_RESULT.md`] (removed in consolidation; see git c4c4ab41); see also [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md) §5).

## What Was Learned

- **Separate wall-clock telemetry from deterministic transition payloads in parity tests:** Embedding `time.perf_counter()` inside actor queue messages is useful for lag monitoring, but deterministic lockstep tests must strip wall-clock timestamps while asserting byte identity on simulation, observation, RNG, and tensor fields.
- **Never let synthetic fixtures depend on the real wall clock:** pin the fixture clock to its historical window so unit tests do not silently expire after a study closes.
- **macOS native runtimes need explicit routing, not suppression:** duplicate libomp loading (torch + scikit-learn) is handled by `DYLD_LIBRARY_PATH` routing in `tests/conftest.py` plus a dyld image guard, without `KMP_DUPLICATE_LIB_OK` ([`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md)); the TorchInductor `@rpath/libc++.1.dylib` loader failure remains unresolved ([`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md)).

## Current Status

**Active reference lineage.** The architectural maps, algorithm/environment specs, OpenMP guide, and fixture READMEs remain active; the one-off `2026-10-20_test_suite_hygiene` spec and result notes were consolidated into this record and their raw XML/log/hash artifacts remain preserved at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_`.

## Keep Files in This Lineage

- [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md) — Test suite organization, markers, fixtures, and macOS OpenMP/Inductor caveats.
- [`docs/architecture/WEB_MAP.md`](../../architecture/WEB_MAP.md) — FastAPI backend + React/TypeScript frontend architecture, WebSocket protocol, and serving modes.
- [`docs/codebase_quality_2026-09.md`](../../codebase_quality_2026-09.md) — September 2026 codebase quality assessment.
- [`docs/ml_algorithm.md`](../../ml_algorithm.md) — Design reference for the Apex vector stack (dueling DQN, Double DQN + n-step, PER).
- [`docs/simd_env_spec.md`](../../simd_env_spec.md) — File:line parity spec of the live simulator's v1/v2 dynamics for the vectorized `src/simd_env/` reimplementation.
- [`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md) — macOS pytest OpenMP routing (duplicate torch/sklearn `libomp` images) and `--showlocals` shared-memory note.
- [`tests/fixtures/ct-supervisor-06/README.md`](../../../tests/fixtures/ct-supervisor-06/README.md) — Raw qualified producer corpus, packet 06 (`ct_supervisor` goldens).
- [`tests/fixtures/d4096_start_metadata/README.md`](../../../tests/fixtures/d4096_start_metadata/README.md) — D4096 metadata copied for the R-A hash-convention regression.
- [`tests/fixtures/phase0_teacher_speed/README.md`](../../../tests/fixtures/phase0_teacher_speed/README.md) — Phase 0 search-teacher speed oracle.
- [`tests/fixtures/t34_fast_digest/README.md`](../../../tests/fixtures/t34_fast_digest/README.md) — Frozen T34 full-digest oracle.
- [`tests/fixtures/t34_shadow_batching/README.md`](../../../tests/fixtures/t34_shadow_batching/README.md) — Frozen T34 serial shadow oracle.
