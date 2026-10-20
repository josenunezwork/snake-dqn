# T34 ExIt-B8 fit preparation

## v2 fit cache (2026-10-20)

Implements [the v2 fit-cache spec](../../docs/research/specs/2026-10-19_t34_v2_fitcache.md)
under [DESIGN_V2_SHADOW.md and Amendment 1](../../docs/research/final_run_20261019/DESIGN_V2_SHADOW.md)
and [SUCCESS_CONTRACT.md](../../docs/research/final_run_20261019/SUCCESS_CONTRACT.md).
The normal fit/preflight entrypoint now regenerates **once**, before the first
optimizer update, into a local memory-mapped observation cache. It also caches
the held-out replay stream when the source supplies one. Subsequent epochs use
the original deterministic sampler, batch shuffle, fp32 loss and Adam recipe.
The low-level `fit(cache_dir=None)` path remains the regeneration reference.

Raster planes remain lossless uint8, including optional full-sight planes.
Scalar/Q arrays retain their original dtypes. S1b omits the tower cache; S1a
includes it. The receipt reports measured payload and disk sizes per row rather
than assuming the spec's approximate GB estimates. Cache manifests bind caller
input/checkpoint/source identities and SHA-256 of payloads; incomplete, changed,
or mismatched caches are refused. A failed build never publishes a complete cache.
Cache directories are create-only. `--cache-dir` selects a local SSD directory
to build or validate/reuse; the default is `OUT/observation-cache`. Keep it out
of the supervisor's exported artifact payload. Delete it only after retaining
the fit receipt; it is reproducible data, not a candidate checkpoint.

Sampling still keeps every searched row and an equal number of anchors. The
source's optional bank-wide inventory behavior is preserved. Caching requires
an epoch-invariant round export: epochs change sampling, not world/row facts.
No epoch-varying augmentation or teacher recollection is supported. CUDA uses
one batch of pinned prefetch on a copy stream (current plus next batch at most),
without DataLoader processes or CPU worker threads. CPU uses ordinary tensors.
The CUDA path is implemented but is not qualified by this Mac-only task.

`normalize_row` accepts the following **replayed observation** additions:

- `label_kind="shadow_confirm"`: maps to unsearched, nondecisive KL anchors;
  hinge, advantage-MSE and auxiliary losses are masked out. Old death-window
  and decisive flags cannot turn these into independent decisive labels.
- `label_kind="lad"` or `"transition"`: requires `t_star`, `a_star`, `F`,
  and explicit `death_window_id`, with replayed `decision` equal to `t_star`.
  Maps to searched decisive death-window rows; each window retains weight 10
  (5 for older rounds) and its internal decisive ratio of 20. Supply the normal
  six-action `forbidden_mask`, or tested `per_action` outcomes from which dying
  actions can be derived. The shadow helper's `F` means surviving alternatives;
  it must never be passed directly to the forbidden-action hinge.

The current main tree has v1 world exports and v2 **measurement diagnostics**,
but no production v2 export schema or `ReplaySource` factory. This implementation
defines the fit-side adapter boundary and tests it; a production producer must
still supply hash-verified observations, previous-student Q, round/split metadata,
and the preregistered probe. Measurement receipts alone are not fit exports.

Run these exact commands sequentially on the Mac. They create only small local
smoke artifacts and do not allocate services, touch tmux, or change serving:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-fitcache
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
B8=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
ARTIFACTS=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-v2-fitcache-20261020
mkdir -p "$ARTIFACTS"
RUN=$(mktemp -d "$ARTIFACTS/operator.XXXXXX")
"$PY" -m research.t34_exit_b8_fit.cache_smoke --checkpoint "$B8" --out "$RUN/s1b"
"$PY" -m research.t34_exit_b8_fit.cache_smoke --checkpoint "$B8" --tower --out "$RUN/s1a"
"$PY" -m pytest -q tests/test_t34_fitcache.py tests/test_t34_fitcache_recipe.py \
  tests/test_t34_exit_fit.py tests/test_t34_exit_tools.py \
  tests/test_t34_exit_tower.py tests/test_t34_exit_jobs.py
"$PY" -m black --workers 1 --check --line-length 100 research/t34_exit_b8_fit \
  tests/test_t34_fitcache.py tests/test_t34_fitcache_recipe.py
"$PY" -m isort --profile black --check-only research/t34_exit_b8_fit \
  tests/test_t34_fitcache.py tests/test_t34_fitcache_recipe.py
"$PY" -m flake8 --jobs 1 research/t34_exit_b8_fit \
  tests/test_t34_fitcache.py tests/test_t34_fitcache_recipe.py
git diff --check
```

To exercise the actual Bash preflight entrypoint with its fixed 200-update
limit and the new default cache path:

```bash
mkdir -p "$RUN/script-inputs/checkpoints"
ln -s "$B8" "$RUN/script-inputs/checkpoints/B8_s3_final.pth"
env PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH" \
  T34_FIT_INPUT_MODE=local-preflight T34_FIT_INPUT_DIR="$RUN/script-inputs" \
  T34_FIT_OUTPUT_DIR="$RUN/bash-output" T34_FIT_DEVICE=cpu \
  T34_FIT_FAKE_ENGINE=1 bash research/t34_exit_b8_fit/fit.sh
```

The two smokes use two synthetic worlds and eight actual optimizer updates,
optionally initialized from read-only B8. `timing.json` records cache regeneration
seconds/row, bytes/row, epoch minutes, peak process RSS and VRAM. GPU busy is
null with an explicit reason on the CPU Mac; no GPU utilization is inferred.
These timings establish plumbing only, not full-bank throughput or efficacy.
`tests/fixtures/t34_fitcache/data_pre_cache.py` is the byte-frozen loader from
`65df507a` (SHA-256 `144098b9688fb52c1e9c57558612b467ea26d4883694a9a41b72ba9faed68596`).
Parity tests compare every batch byte and every Adam loss/parameter update for
three epochs, both with and without bank-wide inventory sampling.

Implements P4, P5 and the fit-only P6 entrypoint from
[`2026-10-19_exit_fit_prep.md`](../../docs/research/specs/2026-10-19_exit_fit_prep.md),
following the binding
[`FINAL_PLAN.md`](../../docs/research/final_run_20261019/FINAL_PLAN.md).
The [input addendum](../../docs/research/specs/2026-10-19_exit_inputs_addendum.md)
supersedes the original checkpoint-publication blocker: B8 and the league are
already in the `s1-inputs-20261019` public release. No new release is needed.

S1b (unchanged B8 architecture, no auxiliary head) is primary. S1a adds the
full-arena tower. No TD term is used. Preparation authorizes no production
training round, allocation, promotion, or serving release.

## Fit and replay contract

`interface.py` is the fit-owned boundary. The sibling `research/t34_exit_b8/`
package owns the teacher exports and replay regenerator; this package never
modifies it. A production factory supplies `replay_epoch(epoch)` and
`replay_heldout(epoch)` streams of `ReplayWorld` objects. Observations are
regenerated once for the cache (each epoch in the reference path). A whole world must be verified before its rows are
exposed, with matching expected/actual SHA-256 proof, a `chunk_id`, and rows
containing:

- `local` uint8 `(7,31,31)`, `global` uint8 `(4,37,37)`, `scalars` float `(60,)`;
- `q_prev` from the previous frozen student, `action_star`, six-action
  `forbidden_mask`, `searched`, `decisive`, `round_age`, `length`, `world_id`;
- `death_window` and explicit `death_window_id` for window rows;
- optional `full_sight` uint8 `(4,88,152)` for S1a;
- optional `hindsight_log_free_region` and its boolean
  `hindsight_log_free_region_mask` together for auxiliary supervision.

Export provenance must bind all six snakes' action traces, RNG stream IDs and
periodic sim hashes as specified in FINAL_PLAN §5. The engine must reject a
world's replay before yielding any observation if a recorded hash diverges.
The fit loader drops/reports failed worlds and kills a chunk above 2% drops.
It retains only a world and bounded minibatches, never the full observation bank.

Keep all searched rows plus an equal number of unsearched on-policy rows. Sources
with `training_row_counts(epoch) -> (searched, unsearched)` use exact, uniform
bank-wide sampling without storing the observation bank. The counts must describe
the verified retained stream, excluding hash-rejected worlds. Count drift
invalidates the entire fit (including when world drops are at most 2%); no candidate
is exported. Without this optional metadata each world must independently have
enough unsearched rows for its searched rows. Current-round weights are
1.0 and older weights 0.5. Every death window has total weight 10 (5 for older
rounds), including its internal decisive/nondecisive ratio of 20:1. Compute
weights before minibatching. The weighted loss divides by row count so batches
of older rows retain their half weight.

Loss is KL(previous || student) at tau 0.05, forbidden-action hinge at margin
0.1 (empty set contributes zero), plus 0.01 advantage MSE. KL uses centered
log-partition arithmetic to preserve an exact zero gradient at identical Q;
tests compare its value and gradient to an independent float64 oracle. This
prevents Adam from amplifying float32 self-distillation roundoff.

Adam uses 2.5e-5 for pretrained weights and 1e-4 for new tower/head weights.
Defaults are batch 1024 and five epochs; production accepts four to six epochs.
Epoch selection ranks held-out decisive agreement, then 32-world probe survival,
then earlier epoch. This rule must be bound in the production preregistration
before fitting. Held-out decisive agreement currently uses the **raw six-action
argmax**; register that choice before production data. The legacy sweep also
reports advisory-masked action diagnostics. The probe hook must supply unique, complete H5000 development
worlds, previous/current actions and lengths, and no gate data. It runs the
all-length decile, one-action-share, boost-share and decisive-agreement
tripwires from FINAL_PLAN §8. A local fake establishes wiring only; its synthetic
probe fields carry no scientific or promotion claim.

## Tower and serving boundary

`network.py` owns arch `exit-fit-tower/1`. Its 4-plane, 152-wide × 88-high tower
has stride-2 3×3 convs 16→32→64→64 and a 256-dimensional embedding. Zero fusion
weights preserve B8's original GEMM and Q bytes while still allowing training
of the new fusion weights. Eval executes the full tower even when fusion is
zero, so the later A5 benchmark measures its actual compute cost and validates
its input shape. Candidate exports strip the hindsight head.

`LiveFullSight` feeds the same plane painter through the independent live
`EgoGridView` producer. `serving.py` provides an explicitly constructed candidate
policy (`ExitFitTowerServingPolicy`), with frame preparation, action dispatch,
rowwise evaluation and Q inspection. It does not register a served default.
The production B8 loader continues refusing this new arch; use the research
loader for a tower candidate. Candidate serving qualification remains separate.

## Exact local commands

Run compute checks sequentially. All preparation checks use the existing Mac
venv, at most two CPU threads, and no tmux, Phase 0/S1 manipulation, Colab,
pods, dispatch or paid services. Future L4 launches explicitly require CUDA.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-fitcache
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
B8=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes/b8_s3_t7_20261016.pth
T3=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t3-rollout-labels-20261015/bank/bank-00000.npz
mkdir -p /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-fit-prep-20261019
OUT=$(mktemp -d /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-fit-prep-20261019/acceptance.XXXXXX)
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
```

Prepare 10,000 **existing actual T3 observation states** for A3 and A4 without
new gameplay. The converter records the original shard and output hashes.

```bash
"$PY" -m research.t34_exit_b8_fit.prepare_a3_rows \
  --source "$T3" --out "$OUT/a3-rows" --states 10000 --batch 64
"$PY" -m research.t34_exit_b8_fit.check_a3 \
  --b8 "$B8" --rows "$OUT/a3-rows" --states 10000 --batch 64 --out "$OUT/a3.json"
"$PY" -m research.t34_exit_b8_fit.check_a4 \
  --checkpoint "$B8" --rows "$OUT/a3-rows" --states 1024 --batch 32 \
  --updates 20 --out "$OUT/a4.json"
```

A3's actual-state check establishes initialization Q identity. It does not
establish full-sight parity on the future P2 bank; the paired live/sim tests are
separate below. A4 executes genuine optimizer updates and compares the fitted
network with the frozen B8 initializer (tolerance 1e-5).

Run the required **tiny** sweep smoke (nine actual fits, one update per pair):

```bash
"$PY" -m research.t34_exit_b8_fit.sweep --checkpoint "$B8" --bank "$T3" \
  --max-rows 32 --updates 1 --batch 4 --out "$OUT/sweep-smoke.json"
```

The **owner-run full sweep**, deliberately not run during prep:

```bash
"$PY" -m research.t34_exit_b8_fit.sweep --checkpoint "$B8" --bank "$T3" \
  --epochs 5 --batch 1024 --device cpu --out "$OUT/sweep-full.json"
```

The grid is tau `{0.02,0.05,0.1}` × margin `{0.05,0.1,0.25}`. Every cell resets
B8 and computes actual all-length disagreement, per-decile disagreement and
boost share, with additional advisory-masked diagnostics. It streams raw or
losslessly packed T3 NPZ rows. This legacy sensitivity sweep uses uniform T3
rows and surviving-set labels; it does not replace the production DAgger
weighting, complete H5000 tripwire probe, or preregistered epoch selection.

Paired live/sim byte identity and focused tests:

```bash
"$PY" -m pytest -q tests/test_t34_exit_tower.py tests/test_t34_exit_fit.py \
  tests/test_t34_exit_tools.py tests/test_t34_exit_jobs.py

# Run serially; some existing parity tests intentionally switch numba to two threads.
NUMBA_NUM_THREADS=2 "$PY" -m pytest -q -m 'not slow'
"$PY" -m black --workers 2 --check --line-length 100 research/t34_exit_b8_fit \
  tests/test_t34_exit_fit.py tests/test_t34_exit_tools.py \
  tests/test_t34_exit_tower.py tests/test_t34_exit_jobs.py
"$PY" -m isort --profile black --check-only research/t34_exit_b8_fit \
  tests/test_t34_exit_fit.py tests/test_t34_exit_tools.py \
  tests/test_t34_exit_tower.py tests/test_t34_exit_jobs.py
"$PY" -m flake8 --jobs 2 research/t34_exit_b8_fit \
  tests/test_t34_exit_fit.py tests/test_t34_exit_tools.py \
  tests/test_t34_exit_tower.py tests/test_t34_exit_jobs.py
bash -n research/t34_exit_b8_fit/fit.sh
git diff --check
```

Run the **actual Bash entrypoint** with the explicitly local fake regenerator.
It loads B8 read-only, performs 200 optimizer updates over two tiny fake worlds,
and exports a candidate, selected training state and receipt with SHA/complete
sidecars. The fake is refused on a Colab runtime. These outputs are smoke
artifacts, never a candidate for evaluation or release.

```bash
mkdir -p "$OUT/script-inputs/checkpoints"
ln -s "$B8" "$OUT/script-inputs/checkpoints/B8_s3_final.pth"
env PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH" \
  T34_FIT_INPUT_MODE=local-preflight T34_FIT_INPUT_DIR="$OUT/script-inputs" \
  T34_FIT_OUTPUT_DIR="$OUT/runtime-a6/output" T34_FIT_DEVICE=cpu \
  T34_FIT_FAKE_ENGINE=1 bash research/t34_exit_b8_fit/fit.sh
```

Create a blocked manifest now; this command does not allocate anything:

```bash
"$PY" -m research.t34_exit_b8_fit.make_manifest \
  --commit "$(git rev-parse HEAD)" --output-dir "$OUT/l4-not-allocated" \
  --out "$OUT/fit-manifest-blocked.json"
```

Adding `--validate` with no binding is an **expected refusal**, because the
engine export and preregistered probe are pending. B8 and the league URLs are
already populated. Bulk checkpoint bytes are downloaded from the existing
public release, and all five extracted league members are hash checked.
The supervisor uploads only tiny control catalogs.

## Production adapter handoff (not executed during preparation)

The engine lane supplies a small immutable `engine-binding.json`. Its two
factories, imported from the pinned repository commit, have these signatures:

```python
def source_factory(*, exports, binding, previous_model):
    # Return a ReplaySource, is_fake=False, with replay_epoch/replay_heldout.
    ...

def probe_factory(*, binding, previous_model):
    # Return hook(student_model, epoch) -> diagnostics.ProbeResult.
    ...
```

The binding contains `approved: true` only after the relevant owner decisions,
`arm` (`S1b` primary or `S1a`), `source_factory` and `probe_factory` as
`module:function`, and the full `preregistration_sha256`. Binding `seed` defaults
to 0 and fixes new-parameter initialization and fit sampling. `inputs[0]` binds the
engine payload with `name`, `url` (HTTPS), `dest: engine-export/payload.json`,
exact `size_bytes`, `sha256`, and `complete_marker: true`. The runtime verifies
payload bytes against both the catalog and local completion sidecar before
calling the factory. Worlds within the payload retain the plan's six traces,
RNG IDs and per-250-frame hashes; the adapter validates them before yielding.

For a later DAgger round, optional `inputs[1]` binds the previous student's
public checkpoint at `checkpoints/student_previous.pth`. The fitter can warm
start either a plain student or a trained tower. Only genuinely new tower/head
parameters receive the new-parameter learning rate. Gate worlds remain excluded
from every fit and probe source. Sources must keep emitted row arrays immutable
for the lifetime of the row objects, even after advancing to another world.

After these actual inputs and factories exist, the owner can prepare and
validate a preflight manifest locally. Use an actual binding path in this
example; no file is manufactured with placeholder approval or hashes:

```bash
BINDING=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-fit-prep-20261019/engine-binding.json
"$PY" -m research.t34_exit_b8_fit.make_manifest \
  --commit "$(git rev-parse HEAD)" --engine-binding "$BINDING" \
  --output-dir "$OUT/l4-preflight-not-allocated" \
  --out "$OUT/fit-preflight-ready.json" --validate
```

For `--mode fit`, the binding additionally fixes `worlds`, `epochs` (4–6), a
unique `job_id`, and measured/priced `fit_resources`: `est_minutes`,
`timeout_minutes`, `unit_cap`, `rate_units_per_hour`, `reserve_units`. All setup,
transfer and cleanup time must fit inside the approved cap; timeout must be at
least twice the estimate. Binding `other_assignments` is forwarded to the
supervisor budget for its existing account-wide assignment/rate checks.
The 15-minute estimate / 30-minute timeout / 1-unit
default (including 0.2 units reserved) applies only to the tiny preflight and is
provisional until the real engine preflight is timed. Full fitting has no guessed
default budget.

```bash
"$PY" -m research.t34_exit_b8_fit.make_manifest --mode fit \
  --commit "$(git rev-parse HEAD)" --engine-binding "$BINDING" \
  --output-dir "$OUT/l4-fit-not-allocated" --out "$OUT/fit-ready.json" --validate
```

Both commands use the supervisor's pure local validation function. They never
run its allocator. L4-only admission, CUDA, lease 600 seconds, the declared wall
and unit caps, and a 64 MiB per-artifact export limit are retained. Dispatch
still depends on the final plan's funding and operational gates.

The selected training artifact includes the selected full model (with training
head if present), optimizer, CPU torch RNG, epoch and update count. It declares
`exact_resume_qualified: false`: no resume entrypoint, CUDA/source-RNG restoration
or continuation parity has been qualified. Candidate files use their real
architecture loader and strip the auxiliary head. Stops retain `FIT_STOPPED`
evidence and never export a candidate. Epoch regeneration timings exclude time
spent fitting while the data generator is suspended.

## Acceptance status

| Check | Status / evidence |
| --- | --- |
| A3 zero-init identity | PASS: full tower executed on 10,000 existing T3 states, max Q delta 0 |
| A4 actual no-tower self-distillation | PASS: 1,024 actual states, 20 Adam updates, max Q drift 0, no argmax changes |
| T3 grid smoke | PASS: nine actual fits on 32 saved rows, one update each; full sweep reserved for owner |
| P5 independent live/sim byte tests | PASS in focused suite; paired simulator/live producers, plus trained reload and live dispatch checks |
| P4 core and fit artifact integration | PASS: 44 focused tests, including bank-wide sampling and export delivery |
| Full non-slow suite | Completed: 8,173 passed, 29 skipped, 17 failed, 101 deselected; follow-up attribution below |
| A6 two-world / 200-update shared entrypoint | PASS: actual Bash entrypoint, B8 loaded, 200 updates, 2.723 seconds; fake engine is wiring evidence only |
| Export delivery | PASS: supervisor local export reader verifies all three artifacts; candidate reload and selected training-state tensors agree |
| Checkpoint input delivery | Existing public B8/league release; no new publication |
| Production replay/probe integration | Depends on sibling engine export/factories and preregistered probe |
| P2-bank planes, A2, real L4 run, gate, release | Outside this prep run; not claimed |

The full non-slow run took 4,100.38 seconds. Thirteen failures were the existing
`torch.compile` module's macOS `@rpath/libc++.1.dylib` loading error. Its complete
non-slow module subsequently passed **26 tests, 2 deselected** with the system C++
runtime supplied only to that process. A fresh cache alone did not fix the loader;
using the Homebrew C++ runtime caused a collection-time bus error and is not the
recommended command. No shared library, global cache, or global configuration was
modified. The successful follow-up command was:

```bash
env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  CXX=/usr/bin/clang++ DYLD_LIBRARY_PATH=/usr/lib \
  TORCHINDUCTOR_CACHE_DIR="$OUT/compile-cache-apple-clang" \
  "$PY" -m pytest -q -m 'not slow' tests/test_perf_compile.py
```

Four failures remain inherited from starting commit
`e8a4425f88dd32cc0e706a0685c30889ef576dc0`; all four reproduced in a clean,
detached checkout of that commit with identical errors:

- S1 admission's bound timing receipt rejects the earlier T33 source-hash change.
- Two T22 frozen-oracle tests forward `membership` to a frozen function without
  that parameter.
- T4 snapshot graft rejects differing `_food_array` cache capacities.

Affected sources, tests, frozen fixtures and bound receipts are byte-identical
between that baseline and this T34 implementation. They remain unchanged under
this prep task's no-S1/no-Phase0 rules. **The whole suite is not green**; the T34
focused suite and the compiler follow-up are green. The long suite began before
the final isolated T34 refinements; the final 44-test focused run covers those
refinements. [VERIFICATION.json](VERIFICATION.json) records exact commands,
results, source identities and hashes of the local evidence files.
