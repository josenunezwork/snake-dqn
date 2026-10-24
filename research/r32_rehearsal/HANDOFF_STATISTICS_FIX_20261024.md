# R32 heavy-tail call repair — 2026-10-24

Mechanical fix verified and published. **Main remains HOLD for a same-pin canary.**
No Colab allocation, dispatch, stop, scientific look, gameplay, native fit, serving
change or release was performed. The live D1024 factory was left untouched.

## Source and method

Implementation commit: `f91bd771`. Frozen runtime pin:
`4f11670a9aec42ed1307e2cfaefcd31688e50276`, branch
`t34-r32-statistics-pin-20261024`, checkout
`/Users/josenunez/Projects/ml/snake-dqn-r32-statistics-pin-20261024`.
An ancestry-only merge (`a9e1c18c`) publishes that pin on main without replacing
main's newer independent transport changes. The frozen pin retains qualified
transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.

The only runtime change from `c1ab2750` passes the protocol's
`synthetic_sensitivity_model.heavy_tail_df` (exactly **5**) to both observed and
remaining `Generator.standard_t` draws in `statistics.run_frozen_unit`. Normal
draws receive no new parameters. Seed, marginal-SD scaling, complete paired
blocks, Gaussian conditional probability, final estimator, guards, trial counts,
and `RETAIN_FIXED_DESIGN` remain unchanged. Authority is round-32 `JUDGMENT.md`
section 5 and the existing frozen `STATISTICS_PROTOCOL.json`/`STATISTICS.md`.
Those documents were not edited.

Changed files are `research/r32_rehearsal/statistics.py`,
`tests/test_r32_rehearsal.py`, and `tests/test_r32_package.py`. The new direct
regression checks both df arguments. The full-main regression executes every
decision/lifecycle/statistics unit with all eleven statistics scenarios reduced
to the protocol's 200-trial local preflight in a separate synthetic allocation.

## Verification and package

Artifact base:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation`.
New pack: `pack-4f11670a-01`.

- Serial Mac tests: **23 passed in 23.06 s**. Exact command below.
- Real Bash main entrypoint: **28/28 units**, eleven statistics scenarios at
  200 trials each, **9.095768250059336 s** worker elapsed. Immutable receipt:
  `preflight-4f11670a-01/output/receipt.json`, SHA256
  `3ad6aa75f5173f74d55f732d3c7ac46f3af4ff22d1b35d8715fee255878ce88b`.
- Black (`--check --workers 1 --line-length 100`), isort
  (`--check-only --profile black`), flake8 (`--jobs 1`) for all three changed
  Python files and `git diff --check`: PASS.
- Standard `package build`: PASS, including a fresh isolated Python 3.12.15
  dependency/import check: 88 import statements, 34 local modules. Existing
  `consumer-02/trace.json` source/checkpoint hashes still match the frozen pin;
  no consumer fit was rerun during the factory window.
- Fresh canary supervisor validation: PASS. Canary dispatch dry-run:
  `PASS_DRY_RUN r32-cpu-canary-4f11670a-01 allocations=0 live_accounting=NOT_CHECKED`.
- Full `statistics.py` call-site audit found no other omitted required random
  or statistical function arguments. The tiny run exercises normal, heavy-tail,
  correlated, heterogeneous, guard-boundary and partial-null branches. It does
  not establish large-N calibration, throughput or remote execution.
- Independent Codex-V light ran the two new tests: **2 passed in 9.79 s**.
  Final artifact review is recorded separately in the new pack's
  `CODEX_V_LIGHT.json`.

All local numerical work used absolute nice 15, CPU, one numerical thread and
no xdist. No full repository/native suite or timing measurement was run.

```bash
env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
 -m pytest -q -p no:xdist tests/test_r32_rehearsal.py tests/test_r32_package.py
```

`LOCAL_VERIFICATION.json` in the new pack records this command and the exact
main-mode Bash smoke environment. `pack.json`, both input specs, empty verified
ledgers and all ordinary build proofs remain immutable.

## Existing PASS canary cannot admit this pin

The requested existing receipt is
`pack-c1ab2750-02/canary-output/receipt.json`, SHA256
`f0ffbb3ca2acc43732a4bfeeaad31707221d6f501e89fbcdbe4edcffbdf078c2`.
It is PASS with verified stop, but its executable pin is `c1ab2750`.
`research.colab_cli.canary.check` requires both the receipt proof and saved
canary manifest to match the main commit and exact uploaded input hashes.
The earlier directory-fix handoff also requires same-pin proof.

Actual attempts to bind the old receipt through `package admit` and validate
the new main both refused with:
`family canary source commit or transport identity differs`.
The requested main dispatch dry-run refused as well. Evidence is in
`ADMIT_EXISTING_CANARY_REFUSAL.json`, `VALIDATE_EXISTING_CANARY_REFUSAL.json`, and
`MAIN_DRY_RUN_REFUSAL.json`. `main-existing-canary-BLOCKED.json` preserves the
requested binding for inspection only; never dispatch it. No `main.json`,
main/canary output directory, state directory or dispatch reservation was created.
The gate was not weakened and the old receipt/manifest/export bytes were not edited.

## Shared envelope and exact operator sequence

`SERIES_ACCOUNTING.json` binds all four existing supervisor receipts, each with
verified stop and positive own rate 0.08 u/h: the first failed `7b114eaa` canary,
the failed `c1ab2750-01` canary, the PASS `c1ab2750-02` canary, and the failed
`c1ab2750` main. Prior total: **16.838035205563454 min / 0.022450713607417936 u**.
Reserve a full fresh canary **20 min / 0.02666666666666667 u**. The adjusted main
template is `admission/main-template.json`: base/effective **125/250 min**,
effective cap **0.3333333333333333 u**, included effective reserve
**0.05555555555555555 u**, estimate 115 min, CPU 0.08 u/h, 64 MiB export cap,
180 s lease. The original generated 268-minute template remains untouched and
must not be used. Worst-case series total is **286.8380352055635 min /
0.3824507136074179 u**, below the unchanged **288 min / 0.384 u** ceiling.
Main remains 100,000 trials for each critical scenario and 10,000 for each
sensitivity; ledgers are empty for the new source allocation. Old partial
exports remain sealed and are not reused under a foreign allocation.

The fresh canary and main are prepared only. Use the qualified frozen R30 broker
for dispatch; the new R32 runtime leaf has the older wrapper without `--dry-run`.
Live account liabilities, priority, positive debit and capacity still require
normal operator admission. No automatic retry or fresh scientific look is granted.

Initialize exact paths:

```bash
set -e
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
R32=/Users/josenunez/Projects/ml/snake-dqn-r32-statistics-pin-20261024
BROKER=/Users/josenunez/Projects/ml/snake-dqn-r30-l4-lifecycle-20261024
BASE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-32/implementation
PACK=$BASE/pack-4f11670a-01
export PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH
export OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
cd "$R32"
```

The following fresh-canary command was run only with `--dry-run`. It is the
operator's prerequisite before a main can be admitted; no such dispatch was run:

```bash
"$BROKER/tools/ops/dispatch.sh" r32-cpu-canary-4f11670a-01 "$PACK/canary-template.json" \
 --broker "$BROKER" --cwd "$BROKER" \
 --state-dir "$BASE/state-cpu-canary-4f11670a-01"
```

Only after that exact new-pin canary is PASS with verified stop and all normal
admission checks are satisfied, admit, validate, dry-run, then dispatch main:

```bash
nice -n 15 "$PY" -m research.r32_rehearsal.package admit \
 --pack "$PACK/admission" --receipt "$PACK/canary-output/receipt.json" \
 --out "$PACK/main.json"
nice -n 15 "$PY" -m research.colab_cli.supervisor \
 --state-dir "$BASE/state-cpu-main-4f11670a-01" \
 --colab /Users/josenunez/.local/bin/colab validate "$PACK/main.json"
"$BROKER/tools/ops/dispatch.sh" r32-cpu-main-4f11670a-01 "$PACK/main.json" \
 --broker "$BROKER" --cwd "$BROKER" \
 --state-dir "$BASE/state-cpu-main-4f11670a-01" \
 --dispatch-txt "$PACK/main.DISPATCH.txt" --dry-run
"$BROKER/tools/ops/dispatch.sh" r32-cpu-main-4f11670a-01 "$PACK/main.json" \
 --broker "$BROKER" --cwd "$BROKER" \
 --state-dir "$BASE/state-cpu-main-4f11670a-01" \
 --dispatch-txt "$PACK/main.DISPATCH.txt"
```

These main commands are conditional future operator instructions, not a claim
that main admission or a successful main dry-run exists today. The existing
old-pin receipt cannot satisfy them. The generated Mac consumer plan is excluded;
retain the pre-existing guarded Mac plan and factory fence. Strict, registry and
serving readiness remain blocked separately.

Untouched tmux sessions observed: `git-sync`,
`mac-bl-f0-d1024-resume-old-eq02-01-7b7b4ca2cd23`, `renicer`,
`sol-astra-i0unblock`, `sol-r30upload`, `sol-r32stat`, and `watchd`.
Released checkpoints, serving pins, factory processes/leases, and all previous
hash-bound files and receipts were not modified.
