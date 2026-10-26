# Job A: resident collector + learner feasibility

Authority: `docs/research/governance_a100_sandbox_20261026/ADOPTED.md`, Fable §3 Job A,
and Astra §3 rank 1. This package is simulator-D engineering evidence only. It
never selects a policy by efficacy, exports gameplay scores, or crosses a
checkpoint into Mac evaluation. Claude alone dispatches. No per-job judge,
registration, or timing-canary requirement is introduced here.

## Frozen workload and interpretation

One sampled learner hero and five `greedy_food` scripted opponents run through
`DeviceSim` and `DevicePolicy.actions/after_step`. The actor copies B8 s3's CNN,
fusion trunk and six-action advantage head; its new scalar value head starts at
zero. All actor parameters train in a minimal fp32 PPO update (Adam 0.0003,
clip 0.2, value coefficient 0.5, entropy coefficient 0.01, grad norm 0.5). GAE uses
gamma 0.99/lambda 0.95. There is one epoch of minibatches of at most 256 rows.
There is no teacher forward, KL constraint, E8 critic or recurrent state.

Each shape (E128 and separately E1024) has a 128-step horizon, one excluded
warmup cycle, then **three fixed measured cycles** in each of:

1. cached learner-only: frozen synthetic tensors already on GPU;
2. collection-only: real actor, legal masks, v8 observation features, terminal and
   time-limit handling, device resets, PPO buffer writes, and GAE;
3. integration: fresh collection plus the complete PPO update.

E128 matches P1's world/horizon dimensions. The scripted roster, fresh zero
critic, PPO recipe and simulator-D distribution do not reproduce P1. E1024 is a
separate engineering recipe. Six snake seats produce **one useful hero row per
world/tick**, never six. Cached rows are reported as processed rows, not useful
collected rows. There is no automatic comparison to a historical L4 number.
A positive resident-PPO charter requires a separately justified, matched
completion-date improvement over the qualified P1 path; absent that, this assay
only establishes correctness, capacity, timings and the next engineering binder.

Collection and learning keep world/observation/rollout/optimizer tensors on GPU.
Python launches kernels and reads scalar control checks, as the existing engine
does; this is not a CUDA-graph or host-free-control claim. CUDA uses indexed MOVE,
R2 caches, static flood and native geometry flags. It does not use T4A's
output-retaining instrumentation wrapper or claim CPU simulator equality.

## Inputs, state and completion

The sole numerical file input is published B8 s3:
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
Worlds are generated on device using uint64 SHA256 seeds from
`a100-dev/sandbox-a/<canonical decimal index>`. Six disjoint, million-index-spaced
allocations bind each shape/phase. A seventh 48-seed allocation at 9,000,000 is
reserved for the tiny same-device resume proof. The canonical uint64 seed is
stored with its unchanged signed int64 bit pattern. DeviceSim's mixer inherently
uses 63 effective bits; preflight additionally rejects zero, effective aliases,
and intersections with the registry under that normalization. Every possible
reset seed, not only initial roots, is checked before data.

The finite run needs 513 possible episode roots per world/cell. On hero death or
at 5,000 ticks, an independent blank device scratch simulator generates fresh
roots. Every environment-leading tensor, including RNG and lazy overflow flags,
is merged only into reset rows. Live rows remain byte-identical. Time limits
bootstrap the pre-reset critic but cut the GAE trace; deaths do not bootstrap.
The logical body capacity is 10,400 with a 1,024-entry physical ring. Physical
capacity overflow, spawn-budget exhaustion, nonfinite loss/gradient, or resume
mismatch makes the assay invalid; it is never silently accepted.

Phase boundaries export complete actor, optimizer, simulator padding/carry,
prepared state, RNGs, episode/horizon/cycle counters, reset keys and frozen recipe.
For cached learning, the non-observation tensors are retained and observations
are reconstructed exactly from a dedicated seed (74031), avoiding a multi-GB
random-byte export. `restore_unit_checkpoint` restores the complete boundary;
tests compare the next optimizer update byte-for-byte. Mid-rollout continuation
is not advertised. Each phase is an independent resumable unit.

Payloads are create-only with SHA sidecars and completion markers. A logical unit
requires **both** checkpoint tar and export-inclusive timing JSON. Ledger creation
reads the prior broker manifest/receipt identity and path/size/hash metadata only,
then refuses old source/recipe, missing timing, duplicates, and foreign IDs.
Verified complete phases are skipped exactly. A new attempt uses fresh output,
manifest and receipt identities. There are no automatic completion retries;
Claude may prepare at most two after verified mechanical failure, stop, and
missing units. `postprocess aggregate` refuses missing/duplicate/foreign units and
requires terminal PASS and verified release. A union spanning allocations is
explicitly marked incomplete for the same-allocation comparison.

Phase reports include collection/GAE/reset time, learner time, local immutable
export time, setup/warmup, peak device memory, and useful rows/s with export.
The Mac aggregator also charges bootstrap, supervisor transfer and release using
the terminal receipt. Neither steady-state speed nor a CLI exit is complete-job
execution evidence.

## Local verification

`LOCAL_VERIFICATION.json` binds runtime source hashes, B8 hash, command, platform,
full-arena E2/E4 measurements and same-platform resume hashes. The final focused
suite has 11 passing tests. All test/smoke processes used nice 15, one math thread,
and no xdist. The full-arena smoke used three measured cycles and one warmup per
phase, horizon 3, minibatch 6, and a four-tick episode limit to exercise resets.
It completed all six cells and the resume proof. No CUDA execution occurred.

| CPU/Mac shape | Cached cycle | Collection cycle | Integrated cycle | Integrated hero rows/s, including local export |
|---|---:|---:|---:|---:|
| E2, H3 | 0.00694 s | 0.16124 s | 0.16253 s | 32.85 |
| E4, H3 | 0.01395 s | 0.25619 s | 0.26247 s | 42.54 |

These small CPU measurements are not an A100 throughput forecast. The proposed
base one-hour/effective two-hour session envelope is Astra's conservative
engineering allowance. No current A100 whole-job estimate has been measured by
this package. The broker's legacy field `measured_estimate_seconds` carries that
prospective allowance; it must not be cited as a measured A100 result. The only
purpose of the rate-priced legacy budget fields is the doubled runaway guard,
not an economic unit allocation or study funding ceiling.

Reproduce correctness:

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 ./venv/bin/python -m pytest -q -o addopts='' research/sandbox_a_resident
./venv/bin/python -m black --check --line-length 100 research/sandbox_a_resident
./venv/bin/python -m isort --check-only --profile black research/sandbox_a_resident
./venv/bin/python -m flake8 research/sandbox_a_resident
bash -n research/sandbox_a_resident/job.sh
```

## Operator packaging

The source pin must already be published. Rebase onto main containing the shared
sandbox infrastructure before preparing the final packet. The packet generator
never calls Colab, reads protected outputs, or modifies the transport.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  nice -n 15 ./venv/bin/python -m research.sandbox_a_resident.prepare \
  --packet "$PWD/research/sandbox_a_resident/packet-01" --commit "$(git rev-parse HEAD)" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --estimate-seconds 3600
```

Commit the ten-line `packet-01/sandbox.json` and its bound broker packet before
running `python -m research.sandbox_a_resident.preflight` on the packet directory.
The operator handoff must record the actual shared preflight and real
`tools/ops/dispatch.sh --dry-run` outputs. A blocked dry-run is not a PASS and does
not introduce a judge requirement. Resolve any adoption/transport admission
mismatch through the infrastructure owner, keeping `research/colab_cli` tree
`5833e54e9c5396121cbbf99a5cd4e52006e564fd` unchanged.
