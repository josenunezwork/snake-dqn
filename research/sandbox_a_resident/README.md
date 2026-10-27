# Job A: resident collector + learner feasibility

## Current runtime: A-3

[`A3.md`](A3.md) is the current prospective sandbox protocol. It keeps the
resident collector/learner assay as simulator-D development evidence and adds
the Argon-ranked collection repairs: CUDA-event phase profiling, vectorized
initial food on reset, indexed deferred-bootstrap/reset work, batched anchors,
static-subgraph compilation with an explicit eager fallback record, and an
E2048/E4096 extension. It does not touch F5 or `research/colab_cli`.

The fixed finite matrix is 24 cells: E128/E1024/E2048/E4096 × episode limit
64/512 × cached/collection/integrated, each with H128, one excluded warmup, and
three measured cycles. `compile_mode=default` is bound prospectively. Results
must retain per-cycle and aggregate phase ms/tick, compile status/fallback
records, and rows/s. The packet uses fresh allocation roots starting at
120,000,000 with 3,000,000-index spacing and a proof allocation at 200,000,000;
these intervals exceed the maximum 4,096-world root range and are checked for
collisions before data.

The intended output root is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/a3-attempt-01/output`.
Its four-hour/eight-hour base/effective timeout is only a technical runaway
guard. A-3 source pin, test, preflight, dry-run, and execution evidence are
recorded separately; this README makes no A100 performance claim.

## Historical runtime: A-2

The A-2 successor was `packet-a2/reseal-03`. See
[`A2_RESEAL03.md`](A2_RESEAL03.md) for the initial-age schedule correction,
per-cycle validity rule, and verification that supersedes the rejected
`packet-a2/reseal-02` schedule. Earlier packets and verification files are preserved.

[`A2.md`](A2.md) is the authoritative A-2 runtime protocol. It supersedes the
A-1 runtime description below while preserving it and every A-1 packet,
verification file, receipt, and export as historical evidence.

A-2 is still simulator-D engineering evidence under
[`ADOPTED.md`](../../docs/research/governance_a100_sandbox_20261026/ADOPTED.md).
It does not touch F5 or `research/colab_cli`; it does not select a checkpoint,
authorize serving, or establish gameplay efficacy. It runs 12 finite cells:
E128/E1024 × episode limit 64/512 × cached/collection/integrated, with H128,
one excluded warmup, and three measured cycles per cell. The A-2 recipe uses a
2,048-row minibatch, fresh allocation offsets 20,000,000 through 31,000,000,
and same-device proof offset 90,000,000. Its packet is `packet-a2`, its output
root is `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/a2-attempt-01/output`,
and its four-hour/eight-hour envelope is a technical runaway guard.

The A-2 timing cells record TF32 enabled with deterministic algorithms disabled.
The separate exact-resume proof records TF32 disabled with deterministic
algorithms enabled. Timing mode is not a whole-training bit-equality claim;
only the same-platform deterministic proof makes that claim. The A-2 manifest
uses eager `compile_mode=none`; the explicit compiler hook is reserved for a
future fresh packet, so this sweep makes no compiled GPU timing claim. Actual
source pins, test results, local timings, preflight, and dispatch dry-run
evidence are recorded in `VERIFICATION_A2.*`, not inferred from this README.

## Historical A-1 runtime description and evidence

The remaining text describes A-1 and its packet-02 path. It is retained for
provenance and must not be used as an A-2 dispatch recipe or performance claim.

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
Python launches kernels and compacts reset/truncation indices. Overflow/spawn/
allocation checks transfer once per horizon; nonfinite loss/gradient checks transfer
once per update, with distribution validation disabled inside minibatches.
Dynamic CUDA compaction and existing simulator/R2 scalar synchronizations remain; this is not a CUDA-graph or host-free-control claim. CUDA uses indexed MOVE,
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
at 64 ticks, a compact blank device simulator generates fresh roots for only the
reset worlds. An explicit environment-state field list, including RNG and overflow
flags, is merged only into reset rows (including E == S). Scratch never prepares;
the next observation prepares the merged simulator once. Live rows remain byte-
identical during reset. Truncation observations/critic forwards compact only live
truncated worlds, so simultaneous terminal heroes are never observed. Each measured
production collection/integrated cell must report at least one truncated row.
The 64-tick limit is an explicit revised engineering recipe, not a 5,000-tick
survival or P1-equivalence claim. Time limits
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
then refuses old source/recipe, corrupt completed pairs, duplicates, foreign
files/receipt entries, protected path tokens and symlinked input paths. A TAR
completion marker without its later JSON completion marker is incomplete and is
skipped, preserving earlier completed units; the inverse marker ordering is
corruption. Known partial publication files and parity/summary metadata do not
block resume. Arbitrary hex fragments containing `f5` are permitted.
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

`LOCAL_VERIFICATION_02.json` binds the repaired runtime source hashes, B8 hash,
command, platform, full-arena E2/E4 smoke and exact same-platform resume hashes.
The 25 resident regression cases cover compact reset work and frozen-oracle world
bytes, E == S constants, simultaneous death/truncation, time-limit reachability,
boundary fault checks, nonfinite refusal, off-device restore, interrupted exports,
foreign/duplicate inventory, protected paths and symlinks. Every test/smoke uses
nice 15 and one math thread, without xdist. The real entrypoint smoke completes
all six cells with H3, a four-tick time limit, three measured cycles and one
excluded warmup, on the full production arena. No CUDA execution occurred.
`LOCAL_VERIFICATION.json`, `VERIFICATION.txt` and packet-01 retain their original
historical evidence; their old source pins/timings do not qualify packet-02.

The packet-02 base four-hour/effective eight-hour envelope is a generous
prospective engineering allowance, including setup, compilation, export and
release. There is no measured A100 whole-job forecast. The ten-line manifest explicitly labels the doubled base as a `runaway-guard`;
legacy broker forecast/rate-priced budget fields carry that technical guard only: Colab time/units are not a budget, and this package introduces
no economic unit ceiling or per-job timing-canary requirement. The finite endpoint
remains one warmup plus three measured cycles per shape/phase; the timeout does
not extend that endpoint.

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

The dispatchable packet-02 seal is `packet-02/reseal-02/` (see
`packet-02/FINAL.txt`). The parent seal is preserved after its offline validation
timeout. Job A preflight retains all static shared checks and its full resident
suite; the adapter runs the shared suite separately, avoiding duplicate suites
inside the 60-second dispatch validation bound.

The source pin must already be published. Rebase onto main containing the shared
sandbox infrastructure before preparing the final packet. The packet generator
never calls Colab, reads protected outputs, or modifies the transport.

```sh
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  nice -n 15 ./venv/bin/python -m research.sandbox_a_resident.prepare \
  --packet "$PWD/research/sandbox_a_resident/packet-02" --commit "$(git rev-parse HEAD)" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth \
  --base-seconds 14400
```

Commit the ten-line `packet-02/sandbox.json` and its bound broker packet before
running `python -m research.sandbox_a_resident.preflight` on the packet directory.
The operator handoff must record the actual shared preflight and real
`tools/ops/dispatch.sh --dry-run` outputs. A blocked dry-run is not a PASS and does
not introduce a judge requirement. Resolve any adoption/transport admission
mismatch through the infrastructure owner, using the explicit `sandbox_admission` route from main commit `a4027fa3`, while
keeping `research/colab_cli` tree `5833e54e9c5396121cbbf99a5cd4e52006e564fd`
unchanged. The adapter replaces only the family timing-canary requirement; the
qualified broker retains allocation, accounting, leases, transfer and release.
Packet-02 declares no probe mode or fabricated canary receipt.
