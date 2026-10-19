# T33 batched continuation teacher — round two

Development-only distillation labels. Implementation follows
`docs/research/specs/2026-10-19_batched_teacher_round2.md` and its named Fable
hot-path review. Phase 0 and all original receipts remain unchanged.
See [OPTIMIZATION_LOG.md](OPTIMIZATION_LOG.md) for measured gains, regressions,
Mac lockstep stage timings, L4 projections and remaining qualification limits.
[ROUND2_PROTOCOL.md](ROUND2_PROTOCOL.md) prospectively defines descriptive fits.
The final test and source-hash evidence is in
[ROUND2_VERIFICATION.json](ROUND2_VERIFICATION.json).

`BatchedTeacher.choose(prepared_worlds)` uses a fixed W×6×N resident layout.
It compacts int64 body rings to maximum segment count plus H+1, gathers mutable
state without per-row deep copies, and keeps one leader per equivalent rollout
until epsilon requires a fork. A fork copies frame, food/RNG, masks, V61 carry
and trend memory before movement, including accumulated mass. If the leader
explores, a quiet representative retains the greedy continuation. All rows draw
their epsilon schedule while dormant. H≤128 shares a canonical 128-step schedule;
H>128 is a separate family. CRN shares hero exploration draws across first actions.
Food RNG starts cloned and anchors remain seed/slot/frame deterministic.

This pre-drawn schedule intentionally differs from Phase 0's conditional draws.
Exact Phase 0 choice/score tests use epsilon zero; nonzero-epsilon tests compare
lazy and eager T33 under the independent documented schedule oracle. Batched
BLAS and CUDA fp16 trunk can change near ties; there is no cross-device trajectory
identity claim. Initial Q is computed once per source, with a separate one-row
adapter. Hero and frozen-opponent forward shapes remain fixed as leaders change.

Prepared sources must have matching configurations and hero weights, valid grid
invariants, and no overlapping live body cells. Unequal food-mirror capacities
are padded before merging. Dormant rows skip v8/V61/opponent host work.
GridBatchSim maintains an ordered numeric food mirror, compiled head/tail and
corpse queue updates, a numeric noncontested train-mode consume path, and internal
mask views. Order-sensitive contested/nontraining consumption and CPython RNG
spawning retain their established order.

`ObservationBuffer` owns reusable Torch/NumPy views of one byte slab. It clears
selected rows and stale tails, writes raster channels directly, and retains
private local-flood/region scratch across calls. Canonical NumPy scalar helpers
still allocate temporaries; this is not a globally allocation-free simulator.
Shared v8 callers retain painted membership. T33 explicitly selects the owner
map route, with a painted oracle and thread-private Numba workspaces.

The prefix ablation runs ONE full CRN continuation and reads mass and next-boundary
Q checkpoints for every H/N/max/V variant. It retains the independent-action
Phase 0 oracle as a second full run. Initial calibration uses uniform reservoir
sampling during one bare replay, retaining only the requested snapshots; an
optional identity-bound cache lets subsequent calibrations replay selected seams
without enumerating eligible indices again. This changes sample indices from
the older two-pass sampler. No result from the separately running calibration-v1
was read or used.

Leaf scale has **segment-frames per raw network-Q unit**. The report fits a
nonnegative through-origin slope from leaf Q to remaining full-horizon mass,
separately per H/N/leaf, and reports scale 0, 1 and fitted in-sample agreement and
regret. It selects no recipe. The three-decision H128 smoke in
`ROUND2_ABLATION_SMOKE.json` is synthetic timing evidence, not label-quality proof.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-batched-teacher
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu nice -n 10 ./venv/bin/python -m research.t33_batched_teacher.ablation --decisions 300 --workers 3 --eligible-cache /absolute/fresh-or-existing/cache --out /absolute/fresh/calibration-round2.json
```

`FixedInference` uses fp32 on CPU/MPS and fp16 towers/fusion with an fp32 dueling
head on CUDA. CUDA compile failures propagate. Qualification needs ≥1,000 actual
calibration states, ≥99.5% masked argmax agreement, and the actual resident shape.
Raw agreement and the fraction of legal margins below .01 remain descriptive.
`submit/collect` fence slab ownership. `DoubleBufferedTeacher` owns two isolated
resident batches, slabs and CUDA streams with two single-thread host lanes;
its CPU orchestration is tested, CUDA overlap is not yet measured.

The benchmark accepts `--resident-rows` divisible by 6N. The updated draft requests
6,144 / 24,576 / 49,992 resident rows at N4/H32, strictly as timing shapes pending
budget calibration. Admission counts unique array allocations, compact rings,
clone peaks and a conservative GPU activation reserve before expansion. A rejected
shape stays rejected; a smaller fallback is labeled separately. GPU cropping was
dropped following Fable; the raster transfer is smaller than a full board.

L4 remains **unlaunched and publication blocked**: all six checkpoint URLs in
`PUBLIC_INPUTS.json` are null. Updated draft manifests do not publish assets or
operate cloud sessions. CUDA shape qualification, memory estimates, compiled
execution and A/B overlap require separately authorized hardware verification.
The current draft is
[L4_MANIFEST_PUBLICATION_BLOCKED_ROUND2.json](L4_MANIFEST_PUBLICATION_BLOCKED_ROUND2.json).
No main-branch integration, serving, release or checkpoint publication occurs here.
Older round-one receipts and blocked manifests remain historical; use the round-two
log and new blocked draft for the current implementation status.
