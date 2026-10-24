# A100 next packages — prepared only

No Colab allocation, workload, dispatch, stop, or tmux action was performed by
this preparation. Serving, released checkpoints, and historical receipts remain
unchanged. The old V8 STOP_ROUTE remains closed.

## Frozen identities

- Implementation worktree: `/Users/josenunez/Projects/ml/snake-dqn-gpu-v8`,
  branch `t34-gpu-v8-20261023`.
- V8 runtime worktree: `/Users/josenunez/Projects/ml/snake-dqn-gpu-v8-split-frozen-01`,
  published branch `t34-gpu-v8-split-runtime-20261023`, runtime commit
  `f2c83d124658d31c6bd28fabf367732fe1fde89f`.
- V8 adds only the split harness/tests to `85180521e87ffc3ddc79296bfb09dcd2c89fdc62`;
  all 14 bound component hashes match that original pin. Newer main-branch
  geometry kernels are deliberately excluded from this frozen runtime.
- Qualified transport pin: `313758255ee5b9f1422a0202c14a80cc32554857`.
- Prepared V8 root:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-split-20261023-01`.
- MOVE2 recovery helper commit on main:
  `51f49206d7d59fa0dfa33e7e6d23d3b4f4202e1f`.
- MOVE2 prepared manifest:
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02/packet-fresh-move2-02/move2-main.json`,
  SHA256 `71475b877a5ddfdec44938bce112f33adee7e6057ebb2530021dbd8fc504e375`.
  Runtime `97a61d97fbbdfb3e790bf95ecb1f76d0325ee021` and existing PASS canary
  remain unchanged. Only job/output namespace changes. Do not use fresh-move2-01.

## Verification

Executed in the frozen V8 runtime with the primary venv, CPU forced and numerical
threads one:

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m pytest -q research/gpu_sim_spike/test_v8_split_contract.py \
  research/gpu_sim_spike/test_v8_split_packet.py research/gpu_sim_spike/test_v8_split_parity.py \
  research/gpu_sim_spike/test_vector_geometry.py research/gpu_sim_spike/test_v8_indexed.py \
  research/gpu_sim_spike/test_v8_bench_contract.py research/gpu_sim_spike/test_v8_packet.py
"$PY" -m black --check --line-length 100 research/gpu_sim_spike/v8_split_*.py research/gpu_sim_spike/test_v8_split_*.py
"$PY" -m isort --profile black --check-only research/gpu_sim_spike/v8_split_*.py research/gpu_sim_spike/test_v8_split_*.py
"$PY" -m flake8 research/gpu_sim_spike/v8_split_*.py research/gpu_sim_spike/test_v8_split_*.py
bash -n research/gpu_sim_spike/v8_split_bench.sh
git diff --check
```

28 passed, nine CUDA-only cases skipped. Formatting/lint/shell checks passed.
Codex-V LIGHT closure found no remaining findings in the reviewed scope.
Published-pin real Bash Mac preflight completed all three E1 rosters and three
arms in 18.134282290993724 seconds. It checked independent CPU feature/diagnostic
bytes, full same-action state/configuration/carry, and G/GV identity.
Real immutable exported-fixture/ledger recovery skipped all three verified
units with `measure` replaced by a recomputation trap; zero units recomputed.
The qualified supervisor's real offline canary validator passed. Canary/main
output directories and ops state namespaces remain fresh.

MOVE2 helper tests:

```bash
"$PY" -m pytest -q -p no:xdist tests/test_ops_redispatch_fresh.py \
  tests/test_colab_cli_warmup.py tests/test_colab_cli_schema_failures.py
```

70 passed. The kernel-creation failure has one failed external readiness exec,
no hardware admission, terminal, uploads or exports, and verified release.
The first internal helper probe timed out; the next was stopped by the guard.
Useful workload GPU time was zero. Allocation elapsed-rate accounting is retained:
55.95090962498216 seconds / 0.1554191934027282 units at the frozen 10 u/h rate;
unchanged account usage does not establish zero billing.

## Exact operator order (not executed)

Before dispatch the operator records original T0/deadlines, original-series
spend/liabilities, Q2/Q4 remaining allowance and live unencumbered balance per
the ruling. The smallest remaining allowance wins. Use only one sprint A100;
wait for verified terminal release before the next command.

```bash
export PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-v8-split-20261023-01

# 1. Fresh current-pin V8 canary.
nice -n 15 bash "$ART/dispatch-canary.sh"

# 2. Only after native PASS, independent PASS_CANARY, valid exports and release.
# This wrapper refuses absent/invalid evidence or a main forecast >20 minutes.
nice -n 15 bash "$ART/dispatch-main-after-canary.sh"

# 3. After V8's terminal receipt and verified release, MOVE2 main recovery.
MOVE_REPO=/Users/josenunez/Projects/ml/snake-dqn-gpu-move-frozen-03
MOVE_PACKET=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-02/packet-fresh-move2-02
"$MOVE_REPO/tools/ops/dispatch.sh" gpu-move-a100-20261023-02-main-97a61d97-fresh-move2-02 \
  "$MOVE_PACKET/move2-main.json" --broker "$MOVE_REPO" --cwd "$MOVE_REPO" \
  --state-dir "$MOVE_PACKET/state" --dispatch-txt "$MOVE_PACKET/DISPATCH.txt"
```

| Package | Effective wall cap | Unit cap | Forecast/estimate |
|---|---:|---:|---|
| V8 canary | 24 min | 4 | <=10 min |
| V8 main | 48 min | 8 | <=20 min; refined from new canary |
| V8 combined, all attempts | 72 min | 12 | 30 min / <=5 units prospective |
| MOVE2 fresh main | 192 min | 32 | unchanged 80 min registered estimate |
| MOVE2 combined, all attempts | 216 min | 36 | inherited series envelope |

Included reserves are not additional allowance. Prior failed allocation and
canaries remain charged to their original series. No second doubling, new
clock, scientific retry or serving authorization is created. V8 main completes
all nine cells unless a registered E1024 G STOP_SPLIT occurs. Native A100
compilation/parity/timing and the eventual small-export bound remain unmeasured
until the fresh canary. Neither package establishes CPU engine qualification,
labels, integrated speed PASS, training admission, or release.
