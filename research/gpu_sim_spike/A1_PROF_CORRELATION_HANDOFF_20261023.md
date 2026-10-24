# A1-PROF correlation repair: operator handoff

Prepared only; no allocation or dispatch. Implementation worktree:
`/Users/josenunez/Projects/ml/snake-dqn-a1-prof`, branch
`t34-a1-prof-20261023`. Published source/manifest pin:
`2edf7ed6bb63361d594c773827ff8e080ef07ec8`, containing repair `54b96b27`.
The audit/handoff commit follows this source pin without changing its runtime
source closure. Both are published to main and the implementation branch.

The mechanical correction is documented in
`A1_PROF_CORRELATION_REPAIR_20261023.md`. `a1_prof.py` now exports attributed/raw
and unattributed counts, time and shares, retaining the missing remainder in a
profiler-only `unattributed` bucket. More than 2% unattributed kernel time
refuses the diagnostic; malformed/missing raw evidence remains an integrity
refusal. The 2% bar bounds uncertainty in stage fractions to two percentage
points of summed kernel time. `a1_prof_pack.py` uses `PYTORCH_ALLOC_CONF` and
supports fresh numeric dispatch suffixes. Timing paths, engine and transport
are unchanged. Independent review receipt:
`CODEX_V_A1_PROF_CORRELATION_20261023.json`.

## Fresh immutable artifacts

Root: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02`.

| Artifact | SHA256 |
| --- | --- |
| `a1-prof-canary.json` | `ac52c7ca8ccd65c884029d6cd546af0ae0fa1028142906784063aa90bf51420d` |
| `a1-prof-main-template.json` | `3b83c75ab43a8de05b9847fb805c6f384809e63f7e607ce069f91ad33ac3096c` |
| `DISPATCH.txt` | `16f056f90d1cb214536473d2a588517ec4cc04b6f6ce6640775fb5e8484cc00e` |
| `mac-preflight/mac-preflight.json` | `2268479ae0f6551478b631dad62ce0dabb1949c6e0d422aa23f82985bcaa373a` |
| `mac-preflight/runtime/output/a1-prof.json` | `8eb04f0114051a55fcdb234fa8d8ed335d2317efdd021447ac1a3de61a79d23d` |
| `mac-preflight-analysis.json` | `a7271b8a6459776e8d6dbb7ff20911887058f08af9b753b058fe840cc7564948` |

State and output directories are unused. Both manifests bind the same seven
uploaded catalog/ledger inputs, empty pin-bound ledgers, qualified transport,
and new successful Mac preflight. Main is an unbound template until the new
same-pin canary PASS, verified exports and verified release exist. Prior canary
receipt, log, outputs, manifests and state remain unchanged.

Caps/forecasts remain: A100 only, >=12 vCPU, ceiling 10 u/hour; canary estimate
10 min, wall 30 min, cap 5 u plus 1 u reserve, workload 900 s; main estimate
30 min, wall 90 min, cap 20 u plus 4 u reserve, workload 2,400 s. These are the
existing doubled per-attempt envelopes; estimates and rates are not doubled.
The old failed run remains charged in historical accounting. No caps in the
old package are amended. Dispatch refreshes live account rates and assignments.

## Verification performed

From the implementation worktree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m pytest -q \
  research/gpu_sim_spike/test_a1_prof.py research/gpu_sim_spike/test_a1_prof_units.py \
  research/gpu_sim_spike/test_a1_prof_pack.py tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py
```

**188 passed, 1 CUDA-only skip in 42.66 s.** Black (100 columns), isort check
(Black profile), flake8 passed on all four changed Python files. Both Bash
entrypoints passed `bash -n`; `git diff --check` passed. An AST comparison to
`31d98ee9` confirmed `sync`, `ExclusiveTiming`, `instrument`, `metrics`, and
`measure_cell` unchanged. Tests cover the reported 17,843/17,910 count gap,
count versus time coverage, exact 2% boundary, above-threshold time, missing raw
trace, overcount integrity, export reconciliation and fresh dispatch names.

```bash
./venv/bin/python -m research.gpu_sim_spike.a1_prof_preflight \
  --commit 2edf7ed6bb63361d594c773827ff8e080ef07ec8 \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02/mac-preflight
./venv/bin/python -m research.gpu_sim_spike.a1_prof_pack \
  --commit 2edf7ed6bb63361d594c773827ff8e080ef07ec8 \
  --artifact-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02 \
  --dispatch-suffix 02 --validate
./venv/bin/python -m research.gpu_sim_spike.a1_prof_analyze --smoke \
  --commit 2edf7ed6bb63361d594c773827ff8e080ef07ec8 \
  --exports /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02/mac-preflight/runtime/output \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02/mac-preflight-analysis.json
```

All three commands passed. Mac real-entrypoint elapsed: **18.091278707957827 s**;
three CPU E1 rosters completed, sidecars verified, complete smoke union analyzed.
Baseline seconds/tick: scripted 0.20991337503073737, frozen
0.38989837498714525, mixed 0.3313161116869499. These are CPU portability
measurements, not A100 forecasts or new GPU evidence. The old 99.626% count
coverage does not establish that its missing kernel time was below 2%.

## Exact operator sequence — prepared, not executed

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-a1-prof
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/a1-prof-a100-20261023-02
PIN=2edf7ed6bb63361d594c773827ff8e080ef07ec8
cd "$REPO"
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-canary-02 "$ART/a1-prof-canary.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-canary" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

Wait for the new canary PASS, verified exports and verified release. Then:

```bash
"$REPO/venv/bin/python" -m research.gpu_sim_spike.a1_prof_pack \
  --commit "$PIN" --artifact-dir "$ART" \
  --canary-receipt "$ART/canary-output/receipt.json" --validate
"$REPO/tools/ops/dispatch.sh" a1-prof-a100-main-02 "$ART/a1-prof-main.json" \
  --broker "$REPO" --cwd "$REPO" --state-dir "$ART/state-main" \
  --dispatch-txt "$ART/DISPATCH.txt"
```

CUDA correlation/time coverage, production-shape memory and current hardware
selfchecks require those operator-run jobs. No tmux sessions, live workers,
released checkpoints, serving pins, legacy queue, qualified transport or
historical hash-bound files were modified by this repair. Unrelated incoming
main changes were merged without editing their files and are outside this
mechanical repair's light review.
