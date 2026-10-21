# Reorg proposals: scope 21 (tests/)

Proposals only. Nothing was moved. Map: [docs/architecture/TESTS_MAP.md](../../../architecture/TESTS_MAP.md).

1. **Do not move test files into subdirectories** (for example `tests/t34/`). Receipts pin
   `tests/test_t34*.py`-style globs and hash `tests/test_*.py` paths, so moves break
   re-verification of ratified studies. If grouping is wanted, use pytest markers or `-k`
   selections, or a conftest-registered collection alias, and keep the paths.
2. **Register the known-failure groups** as an explicit `xfail`/skip list (for example a
   `tests/known_failures.txt` read by `conftest.py` under an opt-in env var). Then "8,060 passed,
   14 failed" becomes a green, auditable run. Needs owner OK because conftest is receipt-adjacent.
3. **Inductor libc++**: add a skip-guard to the compile tests when the generated library cannot
   resolve `@rpath/libc++.1.dylib`, or document the per-process system-runtime env var from
   `research/t34_exit_b8_fit/README.md` in `tests/README.md`.
4. **Fix the `tests/README.md` claim** that `addopts` sets `--showlocals`. `pytest.ini`
   explicitly does not set it, because of the segfault risk.
5. **Retire or annotate stale receipt pins**. The 2026-09-24 `task_aligned_*` JSONs hash test
   files that no longer exist. Add a one-line note in `docs/research/` saying they are not
   re-verifiable.
6. **Use the unused markers** (`integration`, `unit`, `requires_gpu`) or drop them. Today they
   select nothing.
7. **Mark the Q1 admission and T22 frozen-pool tests as expired-fixture** with a reason string,
   so their failures don't hide new ones.
