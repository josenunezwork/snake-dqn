# T34 retention second FIX-FIRST verification — 2026-10-20

Status: **implemented and verified, pre-data; draft remains unratified**.
No retention efficacy, T3 pre-check, qualifying timing, training, cloud allocation,
service, tmux, or serving operation was run. The explicitly authorized unpaced
cost-only smoke is excluded from every study estimate and contract bar.

Authority: [second FIX-FIRST specification](../../docs/research/specs/2026-10-20_t34_retention_fixes2.md)
and the existing [Amendment 2 addendum](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md).
Historical verification files remain unchanged. The authorized draft preregistration,
profile, planning OC, and draft registration were updated prospectively.

## Implementation and independent review

- `source_binding_v2.py` walks Python imports deterministically from all six
  registered entry points, including function-scope imports, package initializers,
  relative imports and literal dynamic imports. It rejects unresolved repository
  modules and nonliteral dynamic imports. The current closure contains **180
  files**, including 91 under `src/`, 16 under `web/`, 4 under
  `research/t34_shadow/`, and 10 configuration files. Generated OC output is bound
  separately to avoid self-reference.
- Registered execution requires a clean detached worktree at the exact ratified
  source commit. A background source monitor supplies cached tick checks; final
  checks cover analysis and cleanup. `source_failure.json` invalidates a preserved
  decision. Timing/pre-check admission also requires a matching create-only
  `source_validated.json`, produced only after the monitor closes successfully;
  a failure sidecar prevents reuse after source restoration.
- Host measurement runs in a lightweight sampler. Tick checks only read cached
  state; no host pause occurs in collection or between ticks. Every tick retains
  its contention flag. Pauses occur at row boundaries and fence outstanding
  assistance as counted `host_pause` fallbacks. Rows are never removed or retried.
- Matching process names alone do not contend: another process must exceed 50%
  CPU for more than 60 seconds. Piecewise sample timestamps account for cumulative
  contended time, including pauses without ticks. Up to 2% is allowed/reported;
  more is INCONCLUSIVE. AC/lid/thermal faults pause between rows and become
  INCONCLUSIVE at the registered 120-second persistence limit.
- Profile, preregistration, registration and runner agree on 48 terminal
  worlds/mix, 32 Watch sessions, 40 elapsed h / 60 CPU-h / $0 study limits.
  The T3 pre-check has its own 8-hour elapsed cap and no study debit. Timing
  remains a study debit. Final checks charge analysis against study caps.
- Independent read-only review reported no remaining actionable code findings.
  It checked the real closure and separately passed 52 focused cost/runner/analyzer
  tests. Detection operates at the registered polling cadences.

## Verification commands and results

All Python commands used the primary project venv on the Mac, CPU device, and
single native numerical pools. No xdist fan-out was used.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-retention-fixes2
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1

# Original 106-test selection, retained and expanded, plus the new regressions:
"$PY" -m pytest -q \
  tests/test_t34_pathb_registration_retention_v2.py \
  tests/test_t34_pathb_retention_oc_v2.py \
  tests/test_t34_pathb_retention_v2.py \
  tests/test_t34_pathb_analysis_v2.py \
  tests/test_t34_pathb_timing_v2.py \
  tests/test_t34_retention_fixes_analyzer.py \
  tests/test_t34_retention_fixes_power.py \
  tests/test_t34_retention_fixes_runner.py \
  tests/test_t34_retention_fixes2_runner.py \
  tests/test_t34_retention_fixes2_guard.py \
  tests/test_t34_retention_fixes2_source.py \
  tests/test_t34_retention_fixes2_discard.py
# 129 passed in 13.72 s; no skips/deselections.

"$PY" -m pytest -q tests/test_t34_retention_fixes2_cost.py
# 4 passed. Total retention coverage across these selections: 133 passed.

"$PY" -m pytest -q \
  tests/test_t34_pathb_controller_v2.py \
  tests/test_t34_pathb_adapter_v2.py \
  tests/test_t34_pathb_precheck_v2.py
# 21 passed; total retention plus adjacent coverage: 154 passed.
```

Black at line length 100, isort with the Black profile, flake8 with one job,
and `git diff --check` passed on the changed Python files/tests. The final
combined selection was also rerun after the cost fixture stabilized; see the
final closeout entry below. No full-repository test claim is made.

## Planning package regeneration

The implementation was committed/published before the new development-only OC.
The OC was then committed before building the draft, so the draft's pinned source
commit includes the exact separately bound OC bytes visible in a detached checkout.

```bash
E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes2-20261020
"$PY" -m research.t34_pathb.retention_oc_v2 \
  --out "$E/oc-01.json" --state "$E/oc-01-state.json" \
  --reps 20000 --seed 3420261020 --max-seconds 60
# PASS; 20,000 completed; zero failed cells. No resume/retry.
# Copied to RETENTION_OC_V2.json and committed before draft generation.
"$PY" -m research.t34_pathb.registration_retention_v2 build \
  --out "$E/draft-01.json" \
  --checkpoints research/t34_pathb/RETENTION_CHECKPOINTS_V2.json
# Copied to REGISTRATION_RETENTION_V2.json.
"$PY" -m research.t34_pathb.registration_retention_v2 validate
# {"valid": true, "authorized": false, "mode": "draft"}
```

Draft source commit: `2e8a6e7c92e173db3564dd73fc85485e5ae01d7f`.
The 180 source hashes equal the current import/data closure. The source, source
commit, OC and draft byte bindings were verified. The terminal/product/timing
bank and seed audit bytes were preserved; no seed allocation or N change occurred.

| Artifact | SHA-256 |
|---|---|
| `REGISTRATION_RETENTION_V2.json` / `draft-01.json` | `d3ce6952258a62d89eb885f2273e44c839c6c2c161fa7302ee34769917ffdc3d` |
| `RETENTION_OC_V2.json` / `oc-01.json` | `a60e94a54c85fd9e1e614a6d3ff3897d3ece724dd5e720d9a58a8f9bb583006b` |
| `RETENTION_BANK_V2.json` | `af54fd499e99575bcc960c78fc3c52d756f8e1c1a52ca65edb95bb5b013f6d95` |
| `SEED_AUDIT_RETENTION_V2.json` | `7d895340b742d2011fde9e2731c3e73c1b58484023b3e4afbf66e769704f3b51` |

## Bounded, contended cost-only smoke

`cost_smoke_v2.py` uses three deterministic development seeds disjoint from the
registered terminal, product and timing bank. It calls the actual unpaced adapter
and settle boundary, caps numerical pools at one (main plus one assistance worker),
checks AC/open lid/thermal state, and records operational costs without efficacy
metrics. The three 64-frame prefixes are deliberately censored cost fixtures, not
H5000 study rows. All three were measured once in the corrected invocation.

```bash
/usr/bin/caffeinate -dims env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t34_pathb.cost_smoke_v2 \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-retention-fixes2-20261020/cost-02 \
  --max-elapsed-seconds 600 --max-frames-per-world 64
```

Receipt: `cost-02/cost-smoke.json`, SHA-256
`b215a2898737a1dc59ebe63e3ab1d8fd0e0d73fd2c9a6ff163f1106859214bf5`.
Its label is **CONTENDED cost-only excluded from all efficacy**. Observed total
smoke time was 14.586303 seconds. The owned Python and caffeinate processes exited.

| Mix | Executed frames | Elapsed incl. setup/close | Process CPU |
|---|---:|---:|---:|
| scripted | 64 | 3.410745 s | 3.332779 s |
| frozen | 64 | 5.952250 s | 5.746993 s |
| mixed | 64 | 4.556938 s | 4.548227 s |

Source-identity limitation: `cost-02/execution-provenance.json` (SHA-256
`5cecceb7a652274333e3eda930b08414cf41bc47df14e46e0ef282c9cd7230d3`)
distinguishes the launch-time disk hash from process-attested execution. The shell
printed final smoke script hash
`7a6102b1a9d4abd349b44e4e05a1cd7ddfb3d012f2e9963335f784ff6bc719b7`
immediately before launch and no edits followed through completion, but the process
did not self-record its loaded-source hash. Earlier metadata hashes and its observed
Git HEAD are stale preparation values, not a frozen execution binding. This smoke
is cost-only operational evidence, not a registered hash-bound scientific run.

The frame-rate extrapolated mean is **362.498247 seconds per H5000 terminal
world**. Applying that terminal proxy to all 144 terminal worlds and 32 Watch
sessions yields **17.722136 h unpaced**. Adding the conservative full-horizon
pacing allocation of 20.370 h and timing allocation of 0.361 h gives **38.453136 h**
against the 40-hour cap (about 1.55 h remaining). The separate pre-check is excluded.
**N remains 48/mix and 32 sessions.**

This forecast is uncertain. Short prefixes do not measure late-length search cost,
terminal deaths can shorten pacing, and Watch uses a different live bridge that
was not sampled. The proxy is below 40 h, but does not establish that the full study
will fit. The first registered cap still stops the study INCONCLUSIVE.

Deviation preserved: the first cost-only attempt loaded an earlier smoke runner
that continued after terminal death. It was interrupted before producing a cost
receipt and mechanically corrected within its original 30-minute allowance; it
was not used in the forecast. `cost-01/interrupted.json` preserves its command,
PIDs, approximate 5m40s elapsed debit, bytecode identifier and reason, SHA-256
`8a4a03361247bc9c9acc35a98d4d62cce55fc64a565fa2243a5cd2070664a89b`.
No efficacy row was generated, discarded or retried. Both attempts remain outside
all study estimates; the corrected prefix costs alone inform the budget forecast.

## Final style check

```bash
CHECKS=(research/t34_pathb/{adapter_v2,controller_v2,analyze_v2,registration_retention_v2,retention_v2,timing_v2,source_binding_v2,host_guard_v2,cost_smoke_v2}.py
  tests/test_t34_pathb_{analysis_v2,registration_retention_v2,retention_v2}.py
  tests/test_t34_retention_fixes_runner.py tests/test_t34_retention_fixes2_*.py)
"$PY" -m black --workers 1 --check --line-length 100 "${CHECKS[@]}"
"$PY" -m isort --profile black --check-only "${CHECKS[@]}"
"$PY" -m flake8 --jobs 1 "${CHECKS[@]}"
git diff --check
```

All 18 changed Python files/tests passed these final checks.

## Operator boundary and closeout

The package remains DRAFT and unauthorized for scientific execution. The operator
must produce a final-source PASS trace receipt, ratify/publish the JSON, create a
fresh clean worktree detached at its source commit, pass timing first, and then
run the separate T3 pre-check before the fixed-N study. The exact sequence is in
[README_V2.md](README_V2.md#retention-registration-package). Do not reuse these
cost fixtures as efficacy, timing, pre-check or release evidence.

No live service or tmux action occurred. Released checkpoints, serving pins,
other worktrees/jobs and ratified hash-bound evidence were not modified. Changes
to the preregistration/profile/OC/registration are the explicitly requested
prospective draft alignment. No cloud resources were allocated.

Final closeout: one combined invocation of the 16 test files listed above passed
**154 tests in 16.98 seconds**, with no skips or deselections. Draft validation
returned `valid: true`, `authorized: false`; final source/profile consistency
assertions passed with all 180 bindings matching. The independent review had no
remaining actionable code findings. Implementation and planning package commits
were pushed to `origin/main` and `t34-retention-fixes2-20261020`; the final
publication commit is recorded in the operator handoff.
