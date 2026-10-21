# Spec — test-suite hygiene (2026-10-20)

**Goal.** The full pytest suite on main has ~15 non-Inductor failures (was 3). Make them pass, or — when a test
encodes a contract that was deliberately changed — update the test to the current contract with a one-line reason.
The ~13 Inductor `@rpath/libc++` loader failures are a known environment issue: do NOT fix; mark them with a
targeted skip only if a clean, platform-gated `skipif` (macOS + torch Inductor loader error) is possible, otherwise
leave them and list them.

**Hard limits (frozen registrations).** Do not modify anything under `research/t34_round0/`, `research/t34_pathb/`,
`research/t34_exit_b8/`, or any module in their import closures (their ratified bindings hash those bytes; the
running Round 0 and the ratified retention study execute from frozen worktrees, but main must stay byte-identical
for these paths). If a failure is in or caused by those paths, do not change them — report it.
Do not change gameplay/sim/reward/model numerics. Tests-first: prefer fixing tests, fixtures or harness glue.

**Compute.** The Mac is running Phase 0 (6 workers). Run pytest with `-p no:cacheprovider -n 2` at most (or serial),
`OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`. Use `./venv/bin/python -m pytest`.

**Done when.** (1) A list of every originally failing test with cause + fix/disposition in
`docs/research/specs/2026-10-20_test_suite_hygiene_RESULT.md`; (2) full suite: 0 non-Inductor failures;
(3) `git diff --stat origin/main` shows no changes in the protected paths; (4) merged to main and pushed
(branch `t34-test-hygiene-20261020`).
