# MOVE2 operator handoff — prepared, not dispatched

**Codex-V LIGHT: ACCEPT_PACKET_BENCHMARK_ONLY.** Fresh packet under Astra's
`ADMIT_MOVE_BENCHMARK_ONLY` ruling; native A100 parity and A100 speed are unrun.
The old MOVE1 result remains **INVALID: MOVE_CPU_GRID_MISMATCH**. All nine files
bound by its original VERIFICATION.json were independently reverified unchanged.

Authoring worktree `/Users/josenunez/Projects/ml/snake-dqn-gpu-move`, branch
`t34-gpu-move-20261023`. The runtime source is published and merged through this
branch into main; final publication identity is in the operator's final message
and the sibling artifact VERIFICATION.json.

## Exact identities

- Dispatch checkout: `/Users/josenunez/Projects/ml/snake-dqn-gpu-move-frozen-03`.
- Source pin: `97a61d97fbbdfb3e790bf95ecb1f76d0325ee021`, branch
  `t34-gpu-move-runtime-20261023-03`.
- Qualified transport pin: `313758255ee5b9f1422a0202c14a80cc32554857`;
  `research/colab_cli` tree `d0d92bdb572b770284a1fea6152417ffa5f06bea`.
- Source delta from old frozen MOVE `866fa48e`: exactly fourteen added MOVE2
  files, zero edits to old files, simulator, kernels or transport.
- Artifact root:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02`.
- Ruling SHA-256: `51b637243a57db2a64baa0d80174386994e3e9f71d545589c31628ad58b4948b`.
- Independent review: artifact `CODEX_V_LIGHT.json`, SHA-256
  `e0fcf3b6eb8f8e076e20fb55eee257cf82130c8c973053bd0e9d9b22dbcb0fb7`.

The dispatch checkout is clean and has an ignored `venv` symlink to the existing
primary checkout venv. Its generated helpers use that interpreter. Current main's
newer transport/ops work is preserved; dispatch uses the frozen qualified source.

## Packet behavior and evidence

Admission is complete frozen-device identity: original tensor inventories,
shapes, dtypes and complete byte hashes, full capacities/padding, typed Python
lifecycle/configuration state and RNG cursors. Timing uses the independently
vendored pre-MOVE simulator itself as the reference stopwatch. Warmups restore
the complete recorded state, including dtype changes and fields first allocated
by a step. Three interleaved warmed repeats compare identical recorded inputs;
separate stage attribution never enters the authoritative plain-step timing.

CPU diagnostics retain float32 versus float64 reward, ordered food/corpse
association and counter-RNG versus MT differences. The immutable discrepancy
register binds seven CPU source hashes, frozen/candidate hashes, configurations,
initial state/action descriptors, original reward bytes, aligned food/corpse
hashes and random-state consumption. Only the frozen baseline's independently
reproduced discrepancy fingerprints are classified as inherited. New or changed
differences stop admission. None of these diagnostics establish CPU-exact
mechanics, natural ancestry, authoritative rewards, labels or policy identity.

The current-pin A100 canary compiles and calls both indexed CUDA entry points,
asserts CUDA tensors and no fallback, synchronizes execution and compares the
complete state against the frozen device. Ten executable event fixtures cover
ready/no-ready/crowded respawns, duplicate candidates/occupancy, full-capacity long
wrapped rings, simultaneous deaths and ordered corpse/food, boost/tail release,
RNG overflow, dead/padded rows and lifecycle transitions. Witnesses require the
advertised event to occur. All three E64 rosters also run the full-step ruler and
a real CUDA policy action outside the isolated simulator timing. Main retains
E64/E256/E1024 and all three rosters; every E1024 roster requires at least 2x.
Mac verdict precedence is INVALID, INCOMPLETE, STOP_ROUTE, PASS.

The successful actual Bash Mac smoke is `mac-preflight-02/mac-preflight.json`:
exit 0, three E1 roster cells, **19.66856983298203 seconds**. All ten event fixtures
passed CPU portability; their native status is **NOT_RUN**. The three compact
timing cells satisfy the reducer's complete cell schema. Total smoke exports:
**371,657 bytes, 24 files including sidecars**. Export ceiling is **2 MiB**; no
large traces or full-state binaries are produced.

Canary and main upload the same ten hash-verified inputs, including both empty
resume ledgers, the ruling, discrepancy register and Mac diagnostic fingerprints.
Immutable payload/SHA/completion publication and namespace/allocation rejection
remain mandatory. The qualified supervisor validated `packet/move2-canary.json`.
`packet/move2-main-template.json` has `scientific_admission=false`; there is no
runnable main manifest before the helper binds a genuine same-pin broker canary
PASS, producer proof, export inventory and verified release.

The first local MOVE2 smoke at `4f416dfd` found a missing synchronization helper
and stopped before timing completion. Its log and diagnostics are preserved in
`mac-preflight/` and `mac-preflight-01-mechanical-failure.json`. The corrected
source uses a local CUDA synchronizer. This was a mechanical Mac preparation
failure, not native parity evidence or a reset of the old scientific INVALID.

## Closing verification

From the exact frozen checkout, serial CPU/thread caps at one and nice 15:
**372 passed, 21 CUDA-only skips, 108.02 seconds**. The complete new MOVE2 slice
contributed 21 passing tests. Black, isort, flake8, Python compilation, Bash syntax
and Git whitespace checks passed. Source-qualified real-entrypoint preflight and
manifest validation passed; native tests were not relabelled from skips.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-gpu-move-frozen-03
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 "$PY" -m pytest -q \
  research/gpu_sim_spike tests/test_colab_cli_canary.py \
  tests/test_colab_cli_schema_failures.py tests/test_ops_dispatch.py
nice -n 15 "$PY" -m black --check --line-length 100 \
  research/gpu_sim_spike/move2_*.py research/gpu_sim_spike/test_move2_*.py
nice -n 15 "$PY" -m isort --check-only --profile black \
  research/gpu_sim_spike/move2_*.py research/gpu_sim_spike/test_move2_*.py
nice -n 15 "$PY" -m flake8 \
  research/gpu_sim_spike/move2_*.py research/gpu_sim_spike/test_move2_*.py
bash -n research/gpu_sim_spike/move2_bench.sh
git diff --check
```

Preparation commands below were already executed successfully. Their paths are
create-only evidence; do not rerun the Mac preflight in place.

```bash
export PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02
PIN=97a61d97fbbdfb3e790bf95ecb1f76d0325ee021
env SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 nice -n 15 "$PY" \
  -m research.gpu_sim_spike.move2_preflight --commit "$PIN" --root "$ART/mac-preflight-02"
nice -n 15 "$PY" -m research.gpu_sim_spike.move2_pack --commit "$PIN" \
  --artifact-dir "$ART/packet" --preflight "$ART/mac-preflight-02/mac-preflight.json" --validate
```

## Exact operator commands — not executed

Before dispatch, record the ruling, remaining Q4/global balances, original
T0+72/T0+120 deadlines and live liabilities. Existing remaining balances and
deadlines dominate the inherited ceilings below; this packet adds no allowance.
Ops refreshes assignment/rate evidence through the normal qualified dispatch.

```bash
# Canary only. Prepared helper uses tools/ops/dispatch.sh and the qualified broker.
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02/packet/dispatch-canary.sh

# Main only after that canary has native PASS, verified exports and verified release.
# This helper binds and validates the actual receipt before reaching dispatch.
nice -n 15 bash /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02/packet/dispatch-main-after-canary.sh
```

Fresh names are `gpu-move2-canary-97a61d97` and `gpu-move2-main-97a61d97`.
Separate `packet/state-canary`, `packet/state-main`, `packet/canary-output` and
`packet/main-output` do not exist at preparation. No old output path is reused.
The main helper targets `packet/canary-output/receipt.json` and refuses absent,
foreign, incomplete, unreleased or failed canary evidence.

| Slice | Forecast | Wall ceiling | Unit ceiling | Reserve included |
|---|---:|---:|---:|---:|
| Canary | 10 min | 24 min | 4 u | 2/3 u |
| Main | 80 min | 192 min | 32 u | 16/3 u |
| Existing series total | 90 min | 216 min | 36 u | 6 u |

Base envelope remains 108 minutes/18 units including 3 reserve; effective
envelope is the owner's existing 2x value above. A100 only, at least 12 vCPU,
verified positive rate at most 10 u/hour. First cap wins. One original plus at
most two manual outcome-blind mechanical completions share the series envelope,
only after verified mechanical failure/stop and missing units. Fresh manifests,
outputs and receipts are required; verified units are skipped exactly. No
automatic retry. **Actual allocated wall: 0 seconds; GPU debit: 0 units.**

No engine qualification, integrated 2,000/s or 20,000/s result, proposal/label
ingestion, training, strict/product/serving result or release follows. No live
tmux/process, released checkpoint, serving pin, old hash-bound evidence or
qualification registry was modified. See MOVE2_PROTOCOL_20261023.md for the frozen
prospective rules and the artifact VERIFICATION.json for exact closing commands.
