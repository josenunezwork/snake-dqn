# T22 frozen-pool CPU microbenchmark

`bench_frozen_pool.py` measures the real six-policy pool with one CPU thread,
128 worlds and three learner seats. The paired receipts exclude 200 warm-up
frames and measure 30 frames. All six groups have 64 live rows per measured
frame. This is a uniform, immutable pool assignment with straight-moving learner
seats, not a training-throughput measurement or a representative late-game distribution.
The source oracle is unedited `league_pool.py` at `4c1e1b2` (SHA-256
`18c3b2e7257af0410069d0f344f3bbe37dbee0f08d5a8e53938b87667575919c`).

Use the established project venv (`SNAKE_PYTHON` below), with all host thread
limits set to one. No GPU, actor subprocess, training update or pod is involved.

```bash
export SNAKE_PYTHON=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
"$SNAKE_PYTHON" research/t22_frozen_pool_20261017/bench_frozen_pool.py \
  --worlds 128 --warmup 200 --frames 30 \
  --checkpoint-dir /Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  --reference --out /tmp/t22_reference.json
# Repeat without --reference and with --out /tmp/t22_current.json.
```

Final receipts are `results_final_reference.json` and `results_final_current.json`.
Both full action streams hash to
`b02339badab1dc36d12d76d4ec13ebb100f268e062f5eb38621588b2af726b14`.
The receipt binds the benchmark, executed pool source and checkpoint identities.
Reference identity validation reads current-tree dependency paths; the separately
recorded executed source hash identifies the historical dispatcher actually loaded.

| Policy | Before ms/frame | After ms/frame |
|---|---:|---:|
| B8 | 28.307 | 27.491 |
| FRP bare | 0.230 | 0.237 |
| FRP + v8 | 6.449 | 0.397 |
| Champion + v8 | 5.846 | 0.337 |
| Greedy food | 0.062 | 0.063 |
| Random safe | 0.082 | 0.036 |
| Entire dispatch | 41.109 | 28.687 |

The instrumented native wall phases, not cProfile self times, establish attribution.
B8 CPU network forward costs 26.651/26.847 ms before/after; it runs on CUDA in
the target configuration. FRP/champion decision-time vector features cost
0.457/0.411 ms within their old veto totals of 6.256/5.606 ms. Their complete
compiled veto kernels cost 0.195/0.150 ms. Baseline B8 raster costs 1.510 ms,
including 0.415 ms of v8 scalars; after sharing, B8 v8 scalars cost 0.421 ms.
Separate learner raster costs 3.980 ms; combined learner+B8 raster costs
4.294 ms. These are nested phase timings: never add inclusive parents and children.
Runtime post-step carry costs 2.276/1.739 ms, controlling 384/192 rows; that work
is outside dispatch. Preparation plus mask retrieval costs 0.143/0.106 ms.

`project_a100.py --drive-root <a100-n3-01>` produces `a100_projection.json`.
It uses each row's own matched actor counters, removes the Mac B8 CPU forward,
charges the extra shared-raster cost back to the pool, and keeps CUDA time unchanged:

- old host pool = 41.109 - 26.651 = 14.458 ms;
- new host pool = 28.687 - 26.847 + max(0, 4.294 - 3.980) = 2.154 ms;
- remaining host ratio = 0.14898;
- projection = baseline rate / (1 - host pool fraction + host pool fraction × ratio).

This gives **13,578 tr/s at a4** and **14,833 tr/s at a6**. The a4 input rate
is the saved scored 5,937.953; the a6 input is the owner's approximate 7,720
and scales linearly if a different rate window is selected. Cumulative host-pool
fractions are 66.12% and 56.35%, respectively. No unmeasured savings outside
pool dispatch are credited. Even zero host-pool cost would cap this restricted
model near 17,525/17,686 tr/s. The 20,900 target is not established. Mac-to-x86
relative speedup, GPU contention, setup, and rate-window effects remain unmeasured.

The independent test recording in `tests/fixtures/t22` is separate from this
microbenchmark: 96 frames, 12 worlds, two independent simulators/runtimes, real
policies, all-policy death/respawn, and exact feature, runtime-memory and world-RNG
comparisons against a vendored, SHA-checked pre-change module. `verification.txt`
records the bounded CPU regression suite. No CUDA action-parity claim is made.
