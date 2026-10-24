# GPU engine sprint: A1-PROF v4 and static cap160 handoff

Prepared, tested, committed and published. No Colab allocation or job was launched.
Runtime pin: `313758255ee5b9f1422a0202c14a80cc32554857` (implementation commit),
contained in origin/main and `t34-gpu-flood-20261023`. Implementation merge:
`51cd5552d02fb7e9219425fd83f6d3892cba2448`.

## Frozen dispatch checkout

Use `/Users/josenunez/Projects/ml/snake-dqn-gpu-flood-frozen-01`, detached at the
runtime pin above. It contains the new engine/package source and the qualified
transport tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`. The latest-main merge
brought in an unrelated launcher change with transport tree
`7d4f2df30382c7808d5c96d1787e84b50bee9f5a`; the existing qualification guard
correctly refuses it. This handoff preserves the guard by using the published
implementation pin and its original qualified transport. Do not edit the frozen
checkout or regenerate these manifests against a newer source pin.

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-01`.
`VERIFICATION.json` with SHA/complete sidecars records source identity, independent
Codex-V LIGHT review, tests, preflight hashes and package hashes.

## Implementation

- `flood_cuda.{cu,cpp,py}`: indexed world occupancy, eight cooperative warp queries
  per block, cap160/641-entry bounded static queue, fail-closed status and
  per-native-launch FUNCTION records/current-stream event diagnostics. Release
  BFS and ego components remain separate. Native selfcheck includes distinct
  E2 worlds, blocked/OOB/duplicate queries, cap32 saturation, and 159/160/161
  component boundaries.
- `event_cuda.{cu,cpp,py}`, `a1_prof.py`, `a1_prof_analyze.py`: gated native scopes,
  NVTX, fixed event pools, separate plain/stage/native-event/minimal-trace passes,
  all names/counts/times, per-E/roster shares/remainders, 10% reconciliation,
  bounded immutable gzip traces and hash-verifying Mac reducer.
- `vector_device.py`: original dense default plus explicit `static` candidate.
  Candidate uses CPU sector LUT/pixel-space float64 intermediates/trend and
  canonical dead-row zeros. These are declared CPU-correctness repairs, because
  the original dense port failed full-byte parity outside free-space columns.
  This A/B is not claimed as a pure same-math optimization.
- `sparse_{bench,pack,preflight,units,analyze}.py`, `sparse_bench.sh`, `colab.sh`:
  matched reset-inclusive alternating E1024 dense/static ticks, all-roster CPU
  byte oracle, recorded-state NPZ/config/carry exports, portable immutable resume
  units, Mac medians/reducer, and enforced prerequisite receipts. Any candidate
  feature mismatch invalidates; any main E1024 roster below 2x stops the route.
- Tests: strict full61/carry/dead/duplicate/rotated-ring/E2 CPU checks, native
  CUDA-gated oracle, profiler inventory/trace accounting, canary/template staging
  and resume contracts. Legacy E1 template tests now assert the current
  supervisor's refusal without qualified-family producer exports.

## Verification

One low-priority test process, one thread per numerical library:

```bash
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  nice -n 10 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q \
  research/gpu_sim_spike tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py
```

317 passed, 4 CUDA-only skips in 67.78 s. Black 100-column check, isort Black profile,
flake8 on all 16 changed Python files; Bash syntax on runtime/generated helpers;
`git diff --check`: PASS. Codex-V LIGHT read-only final review found no remaining
actionable P1/P2 and accepted ordered canary preparation, pending native proof.

Real Bash entrypoint preflights at the exact published pin, serial nice 15:
A1-PROF 20.626692417 s; sparse 4.334873667 s. Sparse static full61 matched every byte
for all 18 rows across scripted/frozen/mixed, with immutable replay states.
These are Mac CPU portability/parity results, not GPU speed or native proof.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-gpu-flood-frozen-01
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-01
PIN=313758255ee5b9f1422a0202c14a80cc32554857
cd "$REPO"
# Already completed; these paths are immutable and must not be rerun in place:
# nice -n 15 "$PY" -m research.gpu_sim_spike.a1_prof_preflight --commit "$PIN" --root "$ART/a1-prof-v4/mac-preflight"
# nice -n 15 "$PY" -m research.gpu_sim_spike.sparse_preflight --commit "$PIN" --root "$ART/sparse-cap160/mac-preflight"
# nice -n 15 "$PY" -m research.gpu_sim_spike.a1_prof_pack --commit "$PIN" --artifact-dir "$ART/a1-prof-v4" --dispatch-suffix 0401 --validate
# nice -n 15 "$PY" -m research.gpu_sim_spike.sparse_pack --commit "$PIN" --artifact-dir "$ART/sparse-cap160" --validate
```

## Envelopes

| Packet | Forecast incl.setup/export | Base wall/units (included reserve) | Effective wall/units (included reserve) |
|---|---|---|---|
| A1-PROF v4 |30 min / 3.85-5 u|36 min / 6 u (1 u)|72 min / 12 u (2 u)|
| Sparse cap160 |90 min / 11.55-15 u|108 min / 18 u (3 u)|216 min / 36 u (6 u)|

A100 only, >=12 vCPU, verified rate <=10 u/h. Effective envelopes doubled once.
Canary/main caps respectively 24 min / 4 u then 48 min / 8 u for profiling, and 24 min / 4 u
then 192 min / 32 u for sparse. Reserve is included. Estimates/rates/N/retry maximum
are not doubled. Original plus at most 2 manual mechanical completion attempts
share the series envelope; unused balance must be frozen before a fresh attempt.
Prior Q1 spend, sealed T0, global round-28 caps, account balance/liabilities and
calendar or scientific stops still bind. No new T0 or additional funding is implied.

## Exact dispatch commands, not executed

First instrument the baseline. Ops refreshes actual assignment/rate evidence.
These commands use the frozen qualified broker and inherited low priority:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-gpu-flood-frozen-01
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-engine-sprint-20261023-01
export PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"
cd "$REPO"
nice -n 15 "$REPO/tools/ops/dispatch.sh" a1-prof-a100-canary-0401 \
  "$ART/a1-prof-v4/a1-prof-canary.json" --broker "$REPO" --cwd "$REPO" \
  --state-dir "$ART/a1-prof-v4/state-canary" \
  --dispatch-txt "$ART/a1-prof-v4/DISPATCH.txt"
```

After same-pin profiler canary PASS, complete exports, <=10% diagnostic
reconciliation on every cell and verified release, bind/dispatch profiler main:

```bash
nice -n 15 bash "$ART/a1-prof-v4/dispatch-main-after-canary.sh"
```

Then, after the profiler main finishes/releases, start the sparse canary. Its
helper validates the ORIGINAL broker receipt together with its adjacent saved
manifest/exports; copying only receipt.json is not valid proof. It creates the
actual sparse canary from the preserved template before normal ops dispatch.

```bash
export A1_PROF_V4_RECEIPT="$ART/a1-prof-v4/canary-output/receipt.json"
nice -n 15 bash "$ART/sparse-cap160/dispatch-canary-after-a1-prof-v4.sh"
# Only after sparse canary PASS, verified exports and verified release:
nice -n 15 bash "$ART/sparse-cap160/dispatch-main-after-canary.sh"
```

The main helper binds the exact sparse canary receipt and A1 prerequisite to the
actual immutable main manifest and validates before dispatch. Scientific failure
is never mechanically retried. No job has yet consumed Colab units.

For completed native exports, run the Mac reducers (create-only output names):

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
PIN=313758255ee5b9f1422a0202c14a80cc32554857
nice -n 15 "$PY" -m research.gpu_sim_spike.a1_prof_analyze --commit "$PIN" \
  --exports "$ART/a1-prof-v4/main-output/exports" --output "$ART/a1-prof-v4/main-analysis.json"
nice -n 15 "$PY" -m research.gpu_sim_spike.sparse_analyze --commit "$PIN" \
  --exports "$ART/sparse-cap160/main-output/exports" --output "$ART/sparse-cap160/main-analysis.json"
```

Native CUDA compilation/races, GPU feature bytes, diagnostic overhead/reconciliation,
2x tick speedup and 2,000 world-steps/s remain unverified. The 72-hour all-roster
round-28 bar and CPU label-verification contract remain binding. No tmux/live jobs,
released checkpoints, serving pins or hash-bound past evidence were modified.
Authored changes are confined to research/gpu_sim_spike; unrelated incoming main
changes were merged unchanged.
