# T33 many-worlds search teacher

Development-only distillation labels. The Phase 0 implementation and original
registration are imported read-only. No serving or promotion authority.
See `protocol.md` for the prospective calibration and inference checks.

## 1. Search-budget ablation

The default factorial is H32/H64 x N4/N8 x {masked-max Q, dueling V}, scale 1,
epsilon .02. The full reference is N8/H128. CRN shares epsilon RNG streams
across first actions at each rollout index; cloned food RNG and deterministic
seed/slot/frame opponent randomness are shared across first actions too.
The reference scorer preserves Phase 0's rowwise B8 evaluation. An additional
original independent-action N8/H128 reference records the CRN policy change.
There is no automatic choice of a winning recipe.

Full calibration command (operator only; three single-thread workers, fresh
output required; runtime is unmeasured, approximately an hour is the spec's
planning estimate):

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-batched-teacher
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 ./venv/bin/python -m research.t33_batched_teacher.ablation --decisions 300 --workers 3 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/calibration-v1.json
```

Only the three audited calibration fixtures are used. Eligible bare-B8
decisions are sampled without replacement, then replayed without changing
visitation according to a variant's action. Each result includes both
reference conventions, scores, agreement, regret, timings, and identities.
The existing bank allocation digest and passing audit/file binding are checked
without opening absolute paths embedded in the Phase 0 audit.

`--margin-gate FLOAT` optionally searches only below that legal-Q margin;
skipped decisions still enter agreement and regret accounting. Synthetic
`--smoke --decisions 3 --workers 1` uses the disjoint L800 timing fixtures,
one per mix. Smoke results cannot establish quality of a cheap budget.

Use `ABLATION_BUDGET_SMOKE_V2.json`. The two original smoke receipts are
preserved but invalidated by `ABLATION_SMOKE_ERRATUM.md`: a callback name
shadowed the leaf-mode option before the regression test caught it. In the
corrected three-fixture smoke, H64/N8 agreed with the CRN reference on 3/3
choices, H32/N8 on 2/3, H32/N4 on 2/3, and H64/N4 on 1/3. This is a smoke,
not evidence to select a cheaper recipe. V and masked-max leaves happened to
choose the same actions here; the unequal-value unit fixture verifies that
their numerical scores actually differ.

## 2. Resident teacher and host observations

`BatchedTeacher.choose(prepared_worlds)` keeps `W * 6 * N` branch rows in stable
source/action/rollout order. Illegal first-action rows remain allocated and
inactive; dead rows are masked. Each source must be a prepared Watch seam
with compatible simulator/configuration/runtime parameters and identical hero
weights. Simulator arrays, RNGs, ordered food, and Vector61 carry are copied.
Frozen models are grouped across worlds; deterministic anchors receive the
complete per-row source-seed vector. There are no Q tie rechecks or reference
search reruns. Batching/precision can change near-tie choices; no universal
trajectory or cross-device identity is claimed.

`ObservationBuffer` owns one aligned byte slab. All seven uint8 31x31 local
planes and four uint8 37x37 global planes are written directly by Numba
`prange` kernels with private scratch. Sixty scalar features remain fp32.
Stable Torch/NumPy views share storage. CUDA uses pinned host storage and one
nonblocking H2D slab copy for B8 each step; each frozen Vector61 model also has
its own fixed input batch/copy. CPU/MPS use eager fp32 through `FixedInference`.

Real fixture tests cover all three mixes at N8/H128 with the independent frozen
Phase 0 oracle, two-source calls, source/runtime/RNG immutability, both CRN
conventions, max/V leaves, dead/illegal rows and repeat determinism. CRN off
matches the Phase 0 sampling policy; CRN on is checked against the separate
rowwise calibration scorer. Direct observation bytes match the established
builder, including dead rows, rotations, boosting and borders. One-versus-two
Numba-thread byte parity passes.

The carry profile in `RUNTIME_CARRY_BENCH_MAC_20261019_V4.json` uses 48 branches,
four frames and three rounds, warmup excluded. Canonical exclusive carry
median was 9.978 ms, T33 one-thread 10.823 ms, T33 two-thread 9.462 ms. Whole-step
medians were 76.275/76.955/78.638 ms respectively. There is **no stable total
Mac speedup**. Earlier attempts/receipts are retained, including the rejected
per-environment scratch allocation design. The final one-thread path invokes
the original kernel once; parallel partitions reuse scratch across environments.

## 3. Device inference qualification

`FixedInference` owns copied immutable weights and a fixed resident input shape.
CUDA uses fp16 raster normalization/autocast plus
`torch.compile(mode="reduce-overhead", dynamic=False, fullgraph=True)`.
Compilation failures propagate. Frozen Vector61 opponents remain fp32, with
fixed compiled batches on CUDA; the B8 fp16 screen does not qualify a separate
opponent precision change.

The calibration collectors use 1,000 actual prepared states, split over the
three disjoint calibration worlds. CPU and MPS each matched same-device fp32
eager on 1,000/1,000 masked and raw argmax choices, with Q error zero. Receipts:
`INFERENCE_CPU_1000_V2.json`, `INFERENCE_MPS_1000.json`. These are fallback
checks. **CUDA fp16/compile agreement remains unverified.** The L4 entrypoint
requires at least 99.5% agreement, both raw and masked, on at least 1,000 states,
then repeats qualification at each admitted actual resident model batch shape.

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 ./venv/bin/python -m research.t33_batched_teacher.qualify_inference --states 1000 --batch-size 64 --device mps --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/inference-mps-repeat.json
```

## 4. Throughput and L4 handoff

The real `bench.sh` entrypoint completed a tiny CPU N1/H4 preflight and
three-round N8/H128 probes on CPU and MPS, one native thread per process,
warmup excluded. The CPU/MPS probes ran concurrently, with other work on the
host; these are observed fixture rates, not quiet-host or controlled speedups.
The first source is scripted, the second adds frozen, the third adds mixed.

| Source worlds | Branch/model rows | CPU decisions/s | MPS decisions/s |
|---:|---:|---:|---:|
| 1 | 48 | 0.290370 | 0.903570 |
| 2 | 96 | 0.278810 | 0.724329 |
| 3 | 144 | 0.279005 | 0.800748 |

The final measured receipts and immutable export sidecars live in
`bench_mac_cpu_final/` and `bench_mac_mps_final/`; the initial probes and
`preflight_cpu/` remain preserved. These use repeated L800 timing
snapshots, not newly collected independent labels. The 100–200 decisions/s L4
target and one-million-label completion remain unverified.

The benchmark requests **256/1024/4096 source worlds**, not branch rows.
The current simulator ring capacity is 10,400; conservative host peak estimates
are about 30/120/481 GiB for those N8 shapes, including clone headroom. It checks
host and CUDA memory before expansion and reports rejected shapes explicitly.
If all requested shapes are rejected, the packaged job measures a separately
labeled 64-source fallback. That point does **not** qualify the requested
resident shapes. `--fallback-worlds 0` disables this behavior. A real tiny
memory-refusal/fallback preflight is retained as `FALLBACK_PREFLIGHT.json`.

The 1,200-second measurement budget is divided over admitted cells, with JIT,
compile, qualification and warmup separate. It logs host/H2D/GPU milliseconds
per rollout step, completed searched decisions/s, GPU utilization snapshots,
and exact resident counts. CUDA events synchronize before accounting.
Incomplete iterations are excluded from all rate and phase numerators and
denominators, with their elapsed time retained separately. All outputs use
payload + SHA-256 + completion markers, verified by the supervisor export reader.

The fastest fully measured CUDA resident cell decides the next proposal:

- Host >=60% of completed wall: consider three L4 runtimes and conditional step 5.
- GPU >=60%: consider one separately operated 20-minute A100 probe.
- Otherwise: mixed, no accelerator conclusion.

**No remote run has occurred, so step 5 is deferred.** No GPU observation port
or simulator GPU port was implemented.

### Public inputs and manifest status

`PUBLIC_INPUTS.json` is the portable six-checkpoint catalog; `INPUTS_REQUIRED.json`
lists the local files, exact sizes and hashes Claude needs to publish. No public
release URLs were available in the inspected checkout, and none were invented
or published. All six URLs are null. The committed
`L4_MANIFEST_PUBLICATION_BLOCKED_V2.json` pins implementation commit
`7cbf8e974a6e7cbe9cce638ddd1f216e07305f34` and uploads only that catalog.
Readiness validation refuses it before dispatch. The explicit six-local-upload
preflight manifest, `L4_LOCAL_VALIDATION_ONLY_V2.json`, passed the supervisor's pure
local validator and is not a production launch manifest.

The draft envelope is L4 only, >=12 vCPUs, 12 Numba threads, 30-minute estimate
including setup/measurement, 60-minute timeout, 2 units cap and 0.4 units reserve,
16 MiB export cap, 180-second lease. The 2-units/hour rate is a dated planning
input, not a live price verification. Setup, CUDA compile, and actual memory/time
costs need the operator's live review; this package grants no allocation authority.

After Claude supplies all six public HTTPS release URLs in a **new** catalog,
preserving every name/destination/size/hash, these are the exact operator commands.
The public-input helper verifies the complete catalog before opening a URL,
downloads with bounded chunks/timeouts, and checks exact size and SHA-256.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-batched-teacher
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m research.t33_batched_teacher.make_manifest --commit 7cbf8e974a6e7cbe9cce638ddd1f216e07305f34 --public-catalog /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/PUBLIC_INPUTS_PUBLISHED.json --output-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/l4-throughput-v2 --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/l4-throughput-public-v2.json --validate
"$PY" -m research.colab_cli.supervisor --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/supervisor-l4-v2 --colab /Users/josenunez/.local/bin/colab run /Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-batched-teacher/l4-throughput-public-v2.json
```

These commands are a handoff only. This implementation did not call Colab,
RunPod, tmux, serving, asset publication, or either excluded worktree. Only the
requested Git branch was pushed. No checkpoint or Phase 0 artifact was written.

The first draft manifests remain as historical, publication-blocked templates.
Use V2: it pins the Numba bootstrap correction. Phase 0's import-time cap
previously initialized Numba's maximum pool at one, so a two-thread real
entrypoint smoke failed before movement. Importing Numba before Phase 0
preserves the launcher-configured ceiling; the corrected two-thread real
entrypoint and a subprocess regression test now pass. The 12-thread CUDA
runtime remains untested. The one-thread N8/H128 measurements above are from
committed source before this import-order-only correction.

## Verification closeout

`VERIFICATION_FINAL.json` records the exact final commands, test counts, export
hashes, source bindings, and unverified boundaries. The T33 suite has 46 passes
and one intentionally gated thread test; that test separately passes with two
Numba threads. The affected Phase 0/raster/Vector61/anchor/grid regressions have
151 passes and one slow deselection. Serial versus two-worker ablation CLI
rows match after excluding only elapsed timings. Black, isort, flake8, Bash
syntax, and `git diff --check` pass. Seven export triples are hash-verified.
The full repository suite was not run; verification stayed within the bounded
research and affected simulator/featurizer surface.
