# O1 implementation and excluded preflight evidence

Prepared only. **No Colab calls, no dispatch, no study-world gameplay, no scientific
look or GO/KILL result.** The sealed Round 0 study directory was never opened.

## Published before data

| Step | Commit |
| --- | --- |
| Preregistration + 55 collision-audited identities | `292d6410d33100560a7e8e782c2ae0e563d9ddf4` |
| Initial implementation | `adc4aa4de0bf7cf30c863c23754dd82e8b52b2b7` |
| Initial frozen registration, before run-01 | `df59622ce6983e0665905a0779450601d61d5355` |
| Timing-only correction, before run-02 | `7277dc545b76a7efdcaa182610559220e5258687` |
| Corrected sibling frozen registration, before run-02 | `198718f0d3caed183b8b000aeca114e407bf0e20` |

Execute/validate only the corrected sibling registration
`REGISTRATION_O1_TIMING_V2.json`, SHA-256
`e8e9be8f9c6eee77aada9d71cceadabb825056c8449d0260ff1a8aed8194c608`.
It binds 241 source/data files, B8 s3, all 48 study worlds and seven excluded
fixtures. The original registration and run-01 are retained unchanged.

## Verification

Targeted tests at `nice -n 10`, one numerical thread, CPU and `-n 0`:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -n 0 \
  research/t34_o1_harvest \
  tests/test_colab_cli_supervisor.py::test_existing_output_directory_is_refused_as_nonfresh \
  tests/test_t34_transport_qualification.py::test_record_matches_actual_accepted_source
```

Final result on the corrected frozen source: **26 passed in 3.11 s**.
Log: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/targeted-tests-final.log`.
No broad suite was run. Black (100 columns), isort, flake8,
`bash -n research/t34_o1_harvest/launch.sh`, and `git diff --check` passed.

Coverage includes exact KILL/GO boundaries; ambient/trail gains rejected;
cumulative food/boost accounting; missing branches/worlds; bare ties; source
clone isolation and reacting synthetic opponents; observer detachment parity;
FIFO overflow/reinsertion; real spawn task ordering; reserve accounting;
qualified manifest caps; deterministic immutable export discovery.
Read-only independent review found no remaining blocker, including the timing
correction. Synthetic spawn evidence concerns transport and deterministic task
merge; the three real H5000 observer-parity worlds remain a runtime requirement.

## Mandatory Mac entrypoint preflight

From `/Users/josenunez/Projects/ml/snake-dqn-o1`, the actual shell entrypoint:

```bash
/usr/bin/time -p nice -n 10 env \
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  T34_O1_PREFLIGHT=1 T34_O1_WORKERS=1 \
  T34_O1_PYTHON=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  T34_O1_REGISTRATION=/Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_o1_harvest/REGISTRATION_O1_TIMING_V2.json \
  T34_O1_CHECKPOINT=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  T34_O1_OUTPUT_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-02 \
  bash research/t34_o1_harvest/launch.sh \
  > /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-02.log 2>&1
```

Completed 2 excluded worlds × 40 base steps, 4 roots, 15 advisory-legal branches × H8
= 120 branch steps. Source preservation and food accounting passed. Actual
niceness 15 (the shell already had positive niceness; command adds 10), each
numerical thread environment 1; CPU only. Host was on AC, lid open, no reported
thermal warning; the fixture is not quiet-host Path B latency evidence.

- **Wall:** 3.27 s (`/usr/bin/time`); runner: 2.913637958 s.
- Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-02/receipt.json`.
- Receipt SHA-256: `2536f49d3a872538cabb914b71377bb2d2188d985def4162f984e085379c1037`.
- Closeout including wall time and both attempts: `preflight/receipt.json` in the same artifact study root.
- Worst recurring base step: 0.002984046 s; branch row-step: 0.003596802 s.
- **Extrapolated L4:** 2278.563912 s = 37.976065 min = 1.26586884 u at 2 u/h.
  Includes 20 min setup/export, rounded worker waves, charged per-world startup,
  11 workers and 2× caution. This is a forecast, not measured L4 throughput.

Run-01 completed 14.40 s wall but produced an over-cap 9.357196 u forecast because
first-step compilation was amortized over only 40 steps. Both its receipt/log and
the original frozen registration remain. See `PREFLIGHT_TIMING_ERRATUM.md`.
No hypothesis, seed, root, attribution threshold, look or cap changed.

## Prepared Colab manifest

`research/t34_o1_harvest/MANIFEST_L4.json`, SHA-256
`bca6d17c1235d997eba7cb2f67be3f1389df0324d09cbe4c7200bb1cb0aefb42`.
Pinned source commit `7277dc545b76a7efdcaa182610559220e5258687`;
registration/public-input catalog/preflight receipt are hash-bound uploads.

One L4, at least 12 vCPUs, 11 one-thread CPU workers; **estimate 38 min**, timeout 240 min,
cap 8 u at 2 u/h, reserve 1.6 u, ordinary allowance 192 min. Timing fixture ≤1 u / 30 min
including consumed setup. Export ≤64 MiB, lease 600 s, final-only evidence.
Output directory (still fresh):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/colab-census-01`.

The real `make_manifest` CLI passed qualified-transport inventory comparison,
pinned blob verification, `research.colab_cli.supervisor.validate` and
`research.colab_cli.source.prepare`. These are local operations; they made no
Colab call. Reproduce preparation with a NEW manifest output filename:

```bash
nice -n 10 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  -m research.t34_o1_harvest.make_manifest \
  --registration /Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_o1_harvest/REGISTRATION_O1_TIMING_V2.json \
  --catalog /Users/josenunez/Projects/ml/snake-dqn-o1/research/t34_exit_b8/PUBLIC_INPUTS.json \
  --preflight-receipt /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/preflight/run-02/receipt.json \
  --commit 7277dc545b76a7efdcaa182610559220e5258687 \
  --output-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-o1-20261021/colab-census-01 \
  --out /absolute/fresh/manifest.json
```

Before any later dispatch the operator must record current account assignments
and positive reviewed rates in `budget.other_assignments`; unknown endpoints
fail closed and the account cap is 3. The live excluded timing fixture and three
H5000 observer-parity worlds must pass before the 48 study worlds. No strict,
serving, product efficacy or release qualification is claimed.

Observed existing tmux sessions, untouched: `git-sync`, `phase0-selection`,
`t34-claima-d0`, `t34-round0-autochain`, `t34-round0-c001`.
