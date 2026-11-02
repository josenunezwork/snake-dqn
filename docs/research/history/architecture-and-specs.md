# Architecture, Specifications, and Test-Suite Hygiene (`architecture-and-specs`)

## Question and Goal

What are the core architectural maps, simulation/algorithm specifications, macOS OpenMP/TorchInductor test contracts, and deterministic test-fixture specifications for the `snake-dqn` repository, and how was the full pytest suite repaired in October 2026 (`2026-10-20_test_suite_hygiene`) to eliminate non-Inductor test regressions without modifying any production, research, or configuration source files?

## Timeline of Versions and Steps

| Date | Step / Artifact | What Was Run / Specified | Verdict / Status |
|---|---|---|---|
| 2026-06 – 2026-07 | [`docs/ml_algorithm.md`](../../ml_algorithm.md), [`docs/simd_env_spec.md`](../../simd_env_spec.md) | Reference documentation for the Apex 61-D Dueling Double DQN + PER + N-step stack (`ApexNetwork`, `ApexPolicy`, `SharedPrioritizedBuffer`) and the vectorized `BatchSim` + `raster31v2` + `RasterDuelingNetwork` + PQN blueprint stack (`src/simd_env/`). | **ACTIVE REFERENCE** |
| 2026-09 | [`docs/codebase_quality_2026-09.md`](../../codebase_quality_2026-09.md) | September 2026 codebase quality, module boundary, and maintainability assessment. | **HISTORICAL REFERENCE** |
| 2026-10-16 | [`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md) | Documented the macOS `libomp.dylib` vs `libiomp5.dylib` dual-OpenMP abort when `torch` and `numba` load in the same process; standardized `KMP_DUPLICATE_LIB_OK=TRUE` in [`pytest.ini`](../../../pytest.ini) and [`tests/conftest.py`](../../../tests/conftest.py). | **ACTIVE CONTRACT** |
| 2026-10-19 – 2026-10-21 | [`tests/fixtures/`](../../../tests/fixtures/) (`phase0_teacher_speed`, `t34_shadow_batching`, `t34_fast_digest`, `ct-supervisor-06`, `d4096_start_metadata`) | Frozen test-fixture bundles and SHA-256 verification sidecars for T34 Phase 0 teacher throughput, shadow batching, fast-digest checks, Colab CLI supervisor 06, and D4096 start metadata. | **ACTIVE FIXTURES** |
| 2026-10-20 | Test-suite hygiene pass (`t34-test-hygiene-20261020`, `73a2d794` → `e301cfeb`) | Full-suite baseline (`8,971` passed, `35` failed, `31` skipped in `2,681.97 s`) and 6-file test-only repair (`9,082` passed, `15` Inductor-only failed, `30` skipped in `2,885.06 s`) in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_`. | **PASS** (`0` non-Inductor failures; `0` source bytes changed) |
| 2026-10-25 | [`docs/architecture/WEB_MAP.md`](../../architecture/WEB_MAP.md), [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md) | Comprehensive architectural maps of the FastAPI + React/TypeScript web platform (`web/serve.py`, `web/backend/`, `web/frontend/`) and the 240-file test suite (`tests/`). | **ACTIVE REFERENCE** |

## Final Verdict and Key Numbers

The architecture and specification documents remain active reference manuals for the repository's three model stacks (`ApexNetwork`, `Ego2sNet`, `RasterDuelingNetwork`), the FastAPI/React web serving stack, and the test suite. The October 20, 2026 test-suite hygiene pass (`t34-test-hygiene-20261020`, base commit `73a2d794` → final commit `e301cfeb`) achieved its primary goal of **0 non-Inductor pytest failures** across the entire repository while modifying **0 bytes** of production, research, script, web, or configuration source files:

- **Baseline full pytest run (`baseline.xml`):** `8,971` passed, `35` failed (`20` non-Inductor + `15` macOS TorchInductor C++ loader failures), `31` skipped, `0` errors in **`2,681.97 s`** (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/baseline.xml`).
- **Post-fix full pytest run (`final.xml`):** `9,082` passed, `15` failed (`0` non-Inductor + `15` retained macOS TorchInductor `@rpath/libc++.1.dylib` failures), `30` skipped, `0` errors in **`2,885.06 s`** (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/final.xml`).
- **Targeted non-Inductor verification (`actor.xml`, `q1.xml`, `t22.xml`, `publication.json`):** `46` passed, `0` failed, `0` skipped in **`121.66 s`** across the 6 modified test files (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/publication.json`).
- **Protected-source invariance (`protected-baseline.json`, `artifact-sha256.json`):** `936` protected files under `src/`, `web/`, `research/`, `configs/`, `saved_snakes/`, and `scripts/` had `0` SHA-256 differences; the 6-file test patch (`tests/test_perf_actor.py`, `tests/test_perf_compile.py`, `tests/test_perf_opp.py`, `tests/test_profile_m3c.py`, `tests/test_q1_admission_integration.py`, `tests/test_t22_frozen_pool.py`) had SHA-256 `aef93f54fd1c020fea08f0beaf44d7e9e1b3c1742ea100118ab79607cde564dd` (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_/artifact-sha256.json`).

### Root Causes of the 20 Repaired Non-Inductor Failures

1. **Monotonic clock comparison across sequential trainer runs (5 failures in [`tests/test_perf_actor.py`](../../../tests/test_perf_actor.py), [`tests/test_perf_compile.py`](../../../tests/test_perf_compile.py), [`tests/test_perf_opp.py`](../../../tests/test_perf_opp.py), and [`tests/test_profile_m3c.py`](../../../tests/test_profile_m3c.py)):** After [`research/redesign_m3c_20261008/train_m3c.py`](../../../research/redesign_m3c_20261008/train_m3c.py) added `produced_monotonic = time.perf_counter()` to each actor payload (`msg`) for wall-age telemetry, bit-identity tests comparing raw `msg` dicts across two consecutive runs failed on `produced_monotonic`. The tests were updated to validate `produced_monotonic` as a finite float and exclude only that wall-clock timestamp from cross-run equality assertions.
2. **Hardcoded wall-clock dispatch horizon in Q1 admission fixtures (13 failures in [`tests/test_q1_admission_integration.py`](../../../tests/test_q1_admission_integration.py)):** Synthetic test fixtures generated an A5 dispatch schedule anchored to `2026-10-18T12:00:00Z` (`horizon_hours = 36`, ending `2026-10-20T00:00:00Z`), while [`research/q0q1_20261017/qualification.py`](../../../research/q0q1_20261017/qualification.py) compares the schedule end against `time.time()`. Shifting the synthetic test fixture start date to `2099-10-18T12:00:00Z` restored deterministic pass behavior.
3. **Kwarg forwarding to frozen pre-membership v8 reference oracle (2 failures in [`tests/test_t22_frozen_pool.py`](../../../tests/test_t22_frozen_pool.py)):** T24 added `membership` and `unique_counts` kwargs to `src.simd_env.v8_nb.v8_decide`, whereas the frozen pre-T24 reference kernel (`tests/fixtures/t24_v8_reference.py`) takes 18 positional arguments without `membership`. Filtering unsupported kwargs in the test wrapper (`_delegate_frozen_v8`) when delegating to the frozen reference restored exact parity.

The 15 remaining failures in [`tests/test_perf_compile.py`](../../../tests/test_perf_compile.py), [`tests/test_perf_learner.py`](../../../tests/test_perf_learner.py), and [`tests/test_t18_h100pod.py`](../../../tests/test_t18_h100pod.py) were left untouched per specification because they stem from macOS `clang++` TorchInductor shared-library linking (`dlopen` missing `LC_RPATH` for `@rpath/libc++.1.dylib`), documented in [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md).

## What Was Learned

- **Separate wall-clock telemetry from deterministic transition payloads in parity tests:** Embedding `time.perf_counter()` inside actor queue messages is useful for lag monitoring, but deterministic lockstep tests must strip wall-clock timestamps while asserting byte identity on simulation, observation, RNG, and tensor fields.
- **Never anchor unit-test fixtures to near-term calendar dates when validators check `time.time()`:** Synthetic schedule tests should either inject a clock or use far-future horizons (`2099-*`) so unit tests do not silently expire two days after a study closes.
- **macOS OpenMP and TorchInductor require explicit environment guards:** Dual-OpenMP loading (`torch` + `numba`) is prevented repository-wide by `KMP_DUPLICATE_LIB_OK=TRUE` ([`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md)), while CPU TorchInductor C++ compilation on macOS requires explicit `CXX=/usr/bin/clang++` and `OMP_PREFIX` / rpath flags ([`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md)).

## Current Status

**Active reference lineage.** The architectural maps, algorithm/environment specs, OpenMP guide, and fixture READMEs remain active; the one-off `2026-10-20_test_suite_hygiene` spec and result notes were consolidated into this record and their raw XML/log/hash artifacts remain preserved at `/Users/josenunez/Projects/ml/snake-dqn-artifacts/test-suite-hygiene-20261020-xeyheao_`.

## Keep Files in This Lineage

- [`docs/architecture/TESTS_MAP.md`](../../architecture/TESTS_MAP.md) — Test suite organization, markers, fixtures, and macOS OpenMP/Inductor caveats.
- [`docs/architecture/WEB_MAP.md`](../../architecture/WEB_MAP.md) — FastAPI backend + React/TypeScript frontend architecture, WebSocket protocol, and serving modes.
- [`docs/codebase_quality_2026-09.md`](../../codebase_quality_2026-09.md) — September 2026 codebase quality assessment.
- [`docs/ml_algorithm.md`](../../ml_algorithm.md) — Reference specification of the Apex 61-D Dueling Double DQN + PER + N-step algorithm.
- [`docs/simd_env_spec.md`](../../simd_env_spec.md) — Reference specification of the vectorized `BatchSim` simulator and `raster31v2` observation contract.
- [`docs/testing_openmp_macos.md`](../../testing_openmp_macos.md) — Root cause and configuration fix for macOS OpenMP runtime collisions (`libomp` vs `libiomp5`).
- [`tests/fixtures/ct-supervisor-06/README.md`](../../../tests/fixtures/ct-supervisor-06/README.md) — Fixture contract for Colab CLI supervisor 06 tests.
- [`tests/fixtures/d4096_start_metadata/README.md`](../../../tests/fixtures/d4096_start_metadata/README.md) — Fixture contract for D4096 start-state metadata verification.
- [`tests/fixtures/phase0_teacher_speed/README.md`](../../../tests/fixtures/phase0_teacher_speed/README.md) — Fixture contract for T34 Phase 0 teacher throughput tests.
- [`tests/fixtures/t34_fast_digest/README.md`](../../../tests/fixtures/t34_fast_digest/README.md) — Fixture contract for T34 fast-digest parity checks.
- [`tests/fixtures/t34_shadow_batching/README.md`](../../../tests/fixtures/t34_shadow_batching/README.md) — Fixture contract for T34 shadow-batching equivalence checks.
