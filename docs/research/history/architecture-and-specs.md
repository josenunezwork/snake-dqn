# Architecture, Specifications, and Test-Suite Hygiene (`architecture-and-specs`)

**Dates:** 2026-07 to 2026-10-28
**Verdict:** Active reference lineage (`ml_algorithm.md`, `simd_env_spec.md`, `WEB_MAP.md`, `TESTS_MAP.md`, `testing_openmp_macos.md`, `tests/fixtures/`) and `PASS` on the `2026-10-20` test-suite hygiene repair (`9,082` passed, `0` non-Inductor failures, `0` production/research source bytes modified).

## What was tried
- Documented the Apex vector stack (`ml_algorithm.md`), file:line v1/v2 `GameState` parity spec for `src/simd_env/` (`simd_env_spec.md`), September codebase assessment (`codebase_quality_2026-09.md`), and read-only maps of `web/` (`WEB_MAP.md`) and `tests/` (`TESTS_MAP.md`, `556` entries / `545` `test_*.py` files).
- Repaired duplicate macOS `libomp.dylib` loading (torch vs scikit-learn) in `tests/conftest.py` and `tests/openmp_runtime.py` (`testing_openmp_macos.md`) and pinned `5` frozen regression fixtures under `tests/fixtures/`.
- Executed a test-only hygiene repair (`t34-test-hygiene-20261020`, `73a2d794 -> e301cfeb`) across `6` test files while verifying byte invariance across `936` protected source digests.

## What happened
- **macOS OpenMP contract (`ACTIVE`):** Routing torch's `lib` first in `DYLD_LIBRARY_PATH` via single-restart `tests/conftest.py` + dyld image guard eliminated duplicate `libomp.dylib` crashes without setting `KMP_DUPLICATE_LIB_OK` (and dropped `--showlocals` from `pytest.ini`).
- **Repaired `20` non-Inductor failures:** Fixed `5` `time.perf_counter()` parity diffs (`test_perf_actor.py`, `test_perf_opp.py`, `test_profile_m3c.py`), `13` wall-clock-expired Q1 A5 fixtures (`test_q1_admission_integration.py`), and `2` `membership` signature mismatches against the frozen `v8_decide` oracle (`test_t22_frozen_pool.py`).
- **Full-suite outcome (`0` source bytes changed):** Lifted the full suite from `8,971` passed / `35` failed (`2,681.97 s`) to **`9,082` passed, `15` Inductor-only failed (`@rpath/libc++.1.dylib` in `test_perf_compile.py`, left visible), `30` skipped (`2,885.06 s`)** with `936` protected source digests unchanged.

## Why
Lockstep actor parity tests had compared live `time.perf_counter()` stamps across runs, Q1 admission tests compared a frozen historical dispatch window against real wall time, and `v8` feature adapters added `membership` after the `t22` oracle was pinned.

## What to reuse / what not to repeat
- Strip wall-clock telemetry (`WALL_CLOCK_ACTOR_FIELDS`) from deterministic payload parity assertions and pin synthetic test clocks so historical admission tests never expire.
- Route macOS OpenMP explicitly via `DYLD_LIBRARY_PATH` + dyld image checks rather than `KMP_DUPLICATE_LIB_OK`, and keep environment-specific TorchInductor failures visible rather than blanket-skipping.

## Full record
Original: `git show 90b3db94:docs/research/history/architecture-and-specs.md` | Primary sources: [TESTS_MAP.md](../../architecture/TESTS_MAP.md), [WEB_MAP.md](../../architecture/WEB_MAP.md), [simd_env_spec.md](../../simd_env_spec.md)
