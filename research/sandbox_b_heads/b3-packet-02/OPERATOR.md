# B-3 operator handoff

Prepared only; no A100 allocation was made. Runtime pin: `7f84306e30a257592b6d74cae71af27a12f47d9b`. The ten-line sandbox record and the empty verified ledger are committed alongside this note. The full plan and F4 source-only comparison are in [B3.md](../B3.md); [B3_PACKET02_FIXES.md](../B3_PACKET02_FIXES.md) supersedes its affected geometry and veto descriptions. Packet 01 is preserved unchanged.

Offline admission (verification recorded in VERIFICATION.json; repeat immediately before dispatch):

```sh
cd /Users/josenunez/Projects/ml/snake-dqn-sbxB
nice -n 15 env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./venv/bin/python -m tools.sandbox.preflight research/sandbox_b_heads/b3-packet-02/sandbox-manifest.json
nice -n 15 env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu ./tools/ops/dispatch.sh sandbox-b3-a100-packet02-attempt01 /Users/josenunez/Projects/ml/snake-dqn-sbxB/research/sandbox_b_heads/b3-packet-02/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-sbxB --cwd /Users/josenunez/Projects/ml/snake-dqn-sbxB --dry-run
```

Exact live dispatch command (operator executes once):

```sh
nice -n 15 env OMP_NUM_THREADS=1 OMP_THREAD_LIMIT=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu /Users/josenunez/Projects/ml/snake-dqn-sbxB/tools/ops/dispatch.sh sandbox-b3-a100-packet02-attempt01 /Users/josenunez/Projects/ml/snake-dqn-sbxB/research/sandbox_b_heads/b3-packet-02/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-sbxB --cwd /Users/josenunez/Projects/ml/snake-dqn-sbxB
```

The host CPU setting above applies to preflight/control; the sealed runtime packet explicitly sets `SNAKE_DQN_DEVICE=cuda` for the A100 job. Dispatch checks live account rates/capacity and protected-study coexistence separately; dry-run does not prove those conditions. Output is a fresh `/Users/josenunez/Projects/ml/snake-dqn-artifacts/a100-sandbox-b3-20261026/a100-packet02-attempt-01`. Do not overwrite an output or retry automatically.

One assignment, no retained chain, 60-minute base / 120-minute effective runaway guard, A100-only, lease 180 seconds, export cap 1 GiB. The required compatibility accounting envelope is 16 units at an 8 u/hour rate ceiling (120 minutes), reserve 0; this is derived from the wall guard, not a resource-rationing decision. The broker checks actual live accounting at admission. All 12 immutable units and all 432 sweep points must be exported. Unknown base votes preserve B8. Every veto requires a strictly safer alternative; equal, worse, and all-zero alternatives preserve base. Threshold zero preserves base actions exactly. `promising` means micro retention >=0.70, ordinary flip_rate <=1%, harm count <=1. Every other point remains reported. These are simulator-D development diagnostics, with no crossing or release authorization.
