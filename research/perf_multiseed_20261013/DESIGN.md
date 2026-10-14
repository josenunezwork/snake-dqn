# Multi-seed (stacked) ego2s learner: design, evidence, go/no-go

Perf pass item 5, 2026-10-13. Research prototype only: it is **not** wired into
`research/redesign_m3c_20261008/train_m3c.py`, and no default anywhere changed.
**All throughput numbers below are NOISY.** They were measured on an Apple M5 Pro while the
M5 screen `m5-L60` trained on the same MPS GPU (`--device mps`) and other jobs shared the CPU.
The load average was 8 to 12 on 18 cores. `bench_multiseed.py` re-runs every number in one
command (see "Re-running").

## Recommendation: NO-GO for integration now

1. **Mac (where training runs today).** The vmap stacking that was asked for is *slower*
   than updating the K models one after another: 0.45x (K = 1) to 0.86x (K = 4) on MPS (one
   noisier replicate reached 1.10x at K = 4) and 0.85x to 0.96x on CPU. The hand-grouped
   forward added here does the same math and is bitwise equal to vmap on MPS. It is 1.16x to
   1.42x faster than in-process sequential updates at K = 2 and 4.
   But it is only about 3 to 16% faster than the status quo, which is K separate processes
   sharing the GPU (`procs`, K = 2). In the one K = 4 `procs` measurement, the separate
   processes were *faster* than the grouped forward. At this noise level the gain is not
   worth a trainer rewrite.
2. **Pods.** The one measured M3-C pod run was CPU-bound, not GPU-bound. That was RTX 4090
   attempt 1 (`PREREGISTRATION_M3C.md` amendment A1), with a 13.6-vCPU quota and 4 runs
   totalling 9.1k transitions/s. The learners then needed about 9.1k / 128 = 71 updates/s x
   49.7 GFLOP, roughly 3.5 TFLOP/s of useful work. That is about 4% of the 4090's
   ~83 TFLOP/s fp32 peak. Raising GPU utilisation does not raise throughput while actors are
   the bottleneck. The remaining gains are one CUDA context instead of K (a few hundred MB
   of GPU memory each) and fewer kernel launches.
3. **Cost and risk.** Integration needs a lockstep update cadence across seeds, padded or
   masked batches (per-seed `n` and `m` vary), per-seed pipelining, schedules and publication,
   and shared fate (one crash stops K seeds). The result is also not bitwise the single-seed
   trainer (see Numerics), so for governance it is a new implementation.

**Revisit (conditional GO)** only if a planned pod configuration becomes GPU-bound, for
example once the vectorized actor and numba featurizer make actors much cheaper per
transition. Run the one-command CUDA bench below on that pod's GPU and integrate only if
all of these hold:

- (a) `grouped` K = 4 (or 8) `seed_updates_per_s` is at least 1.5x `procs` at the same K;
- (b) the projected per-GPU learner demand is at least 70% of what `procs` delivers at that
  K, i.e. the pod would actually be learner-bound (runs x per-run transitions/s / 128
  updates/s);
- (c) CUDA numerics hold: argmax agreement 1.0 and drift ratio below 1e-2 against
  same-device sequential.

Try the cheaper levers first: more processes per GPU (the status quo), the CUDA MPS daemon,
and CUDA graphs / `torch.compile` on the single-seed update (perf item `perf-compile`).

## What was built (`stacked_learner.py`)

- `StackedEgo2s` stacks K `Ego2sNet` parameter sets (`torch.func.stack_module_state`) as
  leaf tensors with a leading K dimension. It works for both obs specs the trainer uses:
  ego2s-b (30 scalars) and ego2s-bv8 (60 scalars), both with local planes (7, 31, 31).
  Two batched forwards:
  - `forward="vmap"` (the default): `vmap(functional_call)` over the stacked parameters.
  - `forward="grouped"`: the same network written by hand. Convolution activations stay in
    a channel layout `(N, K*C, H, W)` with one `groups=K` convolution per layer. Linears are
    `baddbmm`, and LayerNorm normalizes each row and then applies seed k's affine. The
    uint8 planes are permuted once.
- `StackedLearner.update` is the per-seed `learner_m3c` T1 update for all K seeds at once:
  Double-DQN n-step smooth-L1 TD toward the target network's value of the online argmax
  over the legal next actions, plus `lam` x the M4-A `v8margin` anchor on the label rows.
  The anchor is the production `anchor_loss_margin` under vmap. The target networks are
  stacked too.
- Per-seed independence:
  - **Loss:** the sum over seeds is backpropagated, and seeds share no parameter, so each
    `grad[k]` is seed k's gradient alone.
  - **Clip:** `clip_per_seed_` applies `get_total_norm` and the `clip_grad_norm_`
    coefficient to each seed separately.
  - **Adam:** `StackedAdam` steps per-seed views of stacked `exp_avg` / `exp_avg_sq` with one
    step counter per seed. It uses `torch.optim.adam.adam`, the function `Adam.step` calls,
    in single-tensor, foreach or fused mode.
  - **Target sync:** each seed syncs on its own update count.
  - **Failure isolation:** a non-finite loss leaves that seed out of the optimizer step and
    halts it. The other seeds keep stepping.
- Per-seed checkpoints: `seed_nets(k)` returns plain `Ego2sNet`s that
  `save_ego2s_checkpoint` writes in the existing format. `StackedAdam.seed_state(k)` gives
  the optimizer state and `load_seed` restores a seed. `precision="bf16"` uses autocast.
  fp16 is refused, because it would need one GradScaler per seed.
- `single_update` is the single-model reference: `learner_m3c` T1 `_update_fast` +
  `_finalize`, fp32, v8margin, without pipelining.

## Correctness evidence (`tests/test_perf_multiseed.py`, 27 tests, ~40 s CPU)

Bitwise (`torch.equal`) on CPU:

- `single_update` equals `learner_m3c.Learner` T1 on the batches the Learner itself sampled,
  including ring-overrun compaction. This holds with plain and with fused Adam.
- The stacked loss pieces (masked argmax with its no-legal-action fallback, and the
  anchor), the per-seed clip and `StackedAdam` (including a step that leaves a seed out)
  equal the single-model versions given equal inputs or gradients.
- Isolation holds for both forwards:
  - changing seed 1's data leaves seeds 0 and 2 unchanged;
  - a NaN in seed 1 halts seed 1 with its weights from before that step;
  - seeds 0 and 2 then match the clean run.

K stacked updates vs K separate `single_update` runs are compared with a tolerance. Setup:
CPU, K = 3, 6 updates, 24 + 8 rows per seed, target sync every 3 updates; worst seed shown.
A drift ratio is `|theta_stacked - theta_ref| / |theta_ref - theta_0|`. The tolerance is
the tests' `PARITY_TOL`; Q is measured on 64 held-out rows.

| obs spec | forward | drift ratio | param max abs | Q max abs | TD rel | argmax agree |
|---|---|---|---|---|---|---|
| ego2s-b | vmap | 4.9e-5 | 1.0e-5 | 2.0e-6 | 3.0e-6 | 1.0 |
| ego2s-b | grouped | 2.8e-6 | 5.8e-7 | 9.5e-7 | 3.2e-7 | 1.0 |
| ego2s-bv8 | vmap | 8.5e-4 | 9.6e-5 | 6.7e-5 | 9.6e-6 | 1.0 |
| ego2s-bv8 | grouped | 2.3e-6 | 2.8e-7 | 7.3e-7 | 2.3e-7 | 1.0 |
| tolerance | | 1e-2 | 2e-3 | 2e-3 | 1e-3 | = 1.0 |

Deliberate breaks fail the same check by orders of magnitude. Across the breaks (the weakest
is `global_clip`):

- drift ratio 0.19 to 1.40;
- TD rel 0.09 to 2.4;
- argmax agreement 0.36 to 0.80.

The breaks are: batch mix-up, shared Adam moments, target sync from seed 0, no target sync,
global clip, wrong anchor weight, and seed/channel-block layout mix-up in the grouped
forward. The grouped forward is also checked directly against each seed's own
`Ego2sNet.forward` for both obs specs. The LayerNorm affines are randomized per seed in that
check so a shared affine would show; the max abs diff is at most 1e-5 x scale and the
argmax is equal.

`mutation_check.py` applies 12 source mutations to `stacked_learner.py`. Each targets one
property:

- the anchor weight;
- the halt mask;
- the per-seed clip and its epsilon;
- the argmax fallback;
- shared Adam step counters;
- four grouped-forward layout and affine bugs;
- the target-sync source;
- the TD reduction axis.

All 12 were caught, each by an assertion failure in the tests.

Numerics at the real batch size come from the bench's `numerics` block: 3 updates at
384 + 128 rows, K = 2, fused Adam, Q on 512 held-out rows.

| device | arm vs same-device sequential | drift ratio | Q max abs (Q mean 0.48) | argmax agree |
|---|---|---|---|---|
| CPU | stacked (vmap) | 3.8e-5 | 2.7e-6 | 1.0 |
| CPU | grouped | 1.2e-5 | 2.0e-6 | 1.0 |
| MPS | stacked (vmap) | 2.9e-4 | 3.4e-5 | 1.0 |
| MPS | grouped | 2.9e-4 | 3.4e-5 | 1.0 |

- On MPS, vmap and grouped are **bitwise equal** (`stacked_vs_grouped_params`). They run
  the same grouped kernels and differ only in layout copies.
- On CPU they differ by 3.8e-6 max abs after 3 updates.
- Neither is bitwise the single-model update. The difference is float reassociation:
  grouped convolution and bmm against convolution and addmm. Adam can amplify it on
  near-zero gradients. MPS stacked against CPU sequential is 6.5e-3 drift and 1.0e-3 Q max
  abs. Stacking adds only the 2.9e-4 above; the rest is MPS-vs-CPU arithmetic.

**torch.func on MPS (torch 2.9.1) works:**

- No vmap fallback was hit (checked with `_set_vmap_fallback_warning_enabled(True)` on both
  obs specs).
- No CPU fallback was used (`PYTORCH_ENABLE_MPS_FALLBACK` is unset, so an unsupported op
  would raise).
- The MPS parity test passes for both forwards.

It is just slow: see the next section.

## Throughput (NOISY; per-seed learner updates/s; real batch 384 agent + 128 anchor rows)

One seed update is 49.7 GFLOP (FlopCounterMode; convolution forward and backward are 85%).
All runs used bv8, fused Adam, 1 torch thread and a host-to-device upload per call. Results:
`results/bench_multiseed_{mps,cpu}_20261013_noisy.json`, plus two earlier MPS replicates
`results/bench_multiseed_mps_rep{1,2}_20261013_noisy.json`. The replicates came from the
pre-final script revision: same timing code, schema label v1, no FLOP fields. Rep2's
`procs` K = 4 started 4 worker processes, above this pass's 2-process budget.

MPS, per-seed updates/s. Each cell gives final run / rep1 / rep2 (load averages at start:
11.5 / 8.4 / 12.0).

| K | stacked (vmap) | grouped | sequential | procs (K processes) |
|---|---|---|---|---|
| 1 | 9.6 / 10.3 / 9.0 | 23.1 / 23.4 / 19.2 | 21.3 / 22.9 / 19.9 | 20.2 / 21.8 / - |
| 2 | 14.0 / 15.4 / 10.4 | 30.2 / 31.5 / 20.2 | 21.3 / 23.3 / 15.7 | 28.1 / 27.1 / 19.6 |
| 4 | 18.8 / 19.1 / 16.9 | 27.6 / 28.1 / 17.8 | 21.8 / 23.4 / 15.4 | - / - / 28.3 |

Ratios on MPS:

| comparison | K = 1 | K = 2 | K = 4 |
|---|---|---|---|
| grouped / sequential | 0.97 to 1.08 | 1.29 to 1.42 | 1.16 to 1.27 |
| vmap / sequential | 0.45 | 0.66 | 0.82 to 1.10 |
| grouped / procs | | 1.03 to 1.16 | 0.63 (rep2 only) |

No MPS arm achieved more than 1.5 TFLOP/s of useful work, so the MPS learner is bound by
latency and launch overhead, not compute. This matches perf-learner's `--batch-scale 4`,
which gave 1.85x with one model and a 4x batch.

CPU (1 thread, one call per round, 3 rounds), per-seed updates/s:

| K | stacked | grouped | sequential |
|---|---|---|---|
| 1 | 1.03 | 1.09 | 1.08 |
| 2 | 0.92 | 0.96 | 1.09 |
| 4 | 0.92 | 0.95 | 1.08 |

The CPU is compute-bound (0.05 TFLOP/s), so stacking only adds layout work.

**Why vmap loses:** its convolution batching rule folds K into the channel dimension and
unfolds it again around every convolution. Each time, that is a float copy of the
activations; the first local layer alone is 512 x 32 x 31 x 31 floats per seed. On MPS
those copies cost about as much as the convolutions themselves. vmap is 2.1x to 2.4x slower
than grouped at K = 1 and 1.9x to 2.2x slower at K = 2. At K = 2 the two produce bitwise-equal
weights on MPS, so the gap is the layout copies, not the kernels.

## Integration design (only if the CUDA criteria above pass)

- **Process layout.** One learner process per GPU owns K seeds. Each seed keeps everything
  it has today:
  - its own actor processes (2 each), `SharedObsRing`s / `MultiRing`, `Replay` and
    `LabelBuffer`;
  - its own RNG streams, seeded exactly as a single-seed run of that seed;
  - its own schedules (eps, gamma, anchor lambda) keyed on its own transition count. `lam`
    may be a `(K,)` tensor.

  Only the forward, backward and optimizer step are shared.
- **Cadence (lockstep with a due mask).** One stacked update fires when any seed has
  `UPDATE_EVERY` (128) new transitions. Seeds that are not due are left out of the step
  through the existing `active` mask: `StackedAdam`, the counters and target sync are
  already per seed. That keeps every seed's replay ratio at exactly 1 update per 128
  transitions, as in `train_m3c`. The cost is wasted forward/backward compute for seeds
  that are not due, so actor rates should be balanced (same actors and worlds per seed).
- **Staging.** Extend `learner_m3c._Staging` to a K-seed block written in the grouped
  layout `(rows, K, C, H, W)` (no device permute), with one upload per stacked update. Each
  seed's `n` (ring-overrun compaction) and `m` (label re-check) vary, so pad to
  `a_rows` / `d_rows` and carry per-row weights:
  - TD becomes `sum(w*l)/sum(w)`;
  - the anchor needs a masked-mean variant of `anchor_loss_margin`, because the production
    function uses `.mean()`;
  - the result equals the compacted single-seed loss up to reassociation.

  Resampling to fill the rows would change the RNG stream instead.
- **Pipelining.** As in T1: gather update n+1 on the host while the device runs update n.
  One host sync reads all K finite flags. Host-side assembly stays per seed and serial in
  one thread. At ~2.6 ms per seed (`bench_m3c` `host_batch_ms`), K = 8 costs ~21 ms per
  stacked update, i.e. at most ~48 updates/s or ~6k transitions/s per seed. That is
  above today's 2.6k/s actor rate, so it is not a new bottleneck up to K = 8.
- **Publication.** Weights go to each seed's actors via `seed_state_dict(k)` every
  `actor_sync_updates` of *that* seed.
- **Checkpointing.** Per seed, into that seed's own out dir:
  - weights in the existing ego2s format (`seed_nets(k)` + `save_ego2s_checkpoint`);
  - `StackedAdam.seed_state(k)` mapped to a `torch.optim.Adam` state dict;
  - replay and RNG state as today.

  A seed's artifacts then look like a single-seed run, and it can resume alone.
- **Failure isolation.**
  - A non-finite loss halts only that seed (tested). Add compaction: re-stack without
    halted seeds so they stop costing compute.
  - An actor crash for seed k halts seed k only.
  - A learner crash or OOM still stops all K seeds. That shared fate is the main
    operational cost, so checkpoint every seed often. GPU memory grows linearly with K.
- **Governance.** Not bitwise the single-seed learner (numbers above). A stacked-trained
  seed is a different implementation of the same recipe, in the same class as Mac vs pod,
  so pre-register it and do not mix stacked and single-seed seeds within one Phase R
  comparison.
- **Not covered by the prototype.** fp16 (needs one GradScaler per seed), per-seed learning
  rates, and the demo-stream (`--demo-data`) rows (only label rows were exercised; the rows
  are the same layout).

## Re-running (quiet Mac, or CUDA)

```bash
./venv/bin/python -m pytest -q tests/test_perf_multiseed.py              # 27 tests, ~40 s
./venv/bin/python research/perf_multiseed_20261013/mutation_check.py     # 12/12, ~50 s
# Mac (quiet), MPS then CPU:
./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \
    --devices mps --ks 1,2,4 --arms stacked,grouped,sequential,procs --procs-ks 1,2,4 \
    --rounds 7 --note "quiet Mac" --out /tmp/bench_multiseed_mps_quiet.json
./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \
    --devices cpu --ks 1,2,4 --rounds-cpu 3 --note "quiet Mac" --out /tmp/bench_cpu_quiet.json
# CUDA pod (the go/no-go run; criteria (a)-(c) above):
./venv/bin/python research/perf_multiseed_20261013/bench_multiseed.py \
    --devices cuda --ks 1,2,4,8 --arms stacked,grouped,sequential,procs --calls 50 \
    --note "quiet CUDA" --out bench_multiseed_cuda.json
```
