# Advice: why the A100 smoke is slower than the Mac, and how to make Colab worth it (2026-10-17)

Advisory note with code (cloud advisor), requested by the owner through the local session.
Branch `colab-slower-20261017`; main untouched. Inputs: the smoke-02 rows and telemetry the
local session reported (A100-SXM4-40GB, 12 vCPU / 6 physical, torch 2.9.1+cu126, runner pin
`6b7e280`), `train_m3c.py` (actor loop, ingest and shutdown), `learner_m3c.py` (the fast
update), `obs_pack.py`, `research/perf_quiet_20261013/RESULTS.md` (Mac and 4090 rows), and
two measurements I made on a 2.1 GHz Xeon vCPU in this session (below; a Colab-class core,
not the A100 box itself, so read them as shapes and ratios, not as the A100's numbers).

## Short answer

Colab is not slow because of the GPU. It is slow because every part of this trainer that is
not a GPU kernel runs on one slow x86 core at a time, and the recipe as configured gives
those cores a lot to do. Two bottlenecks, one per row type:

- **Hero rows are actor-bound** on six physical cores that are about 3.7x slower per
  128-world frame than the Mac's (63 ms vs about 17 ms). Two actors cannot beat a laptop, and
  eight oversubscribe six cores. No GPU change touches this.
- **Self-play rows are learner-bound at 44 to 45 updates/s** (22 ms per update, 2 and 4
  actors identical, queues full, `learner_wait` 4 s of 148 s). About half of each 22 ms is
  the GPU (fp32 convolutions on a chip whose fp32 rate is a quarter of a 4090's), and about
  half is CPU-only time in the learner process during which the GPU idles (GPU utilization
  54% mean on a learner-bound row): the host-side sample and gather, `--obs-pack` unpacking,
  and the one-deep pipeline's synchronization.

The levers that follow are all on the self-play path, because that is the only path whose
transitions are cheap enough per core to clear 21,600/s on six cores. In order of certainty:
drop `--obs-pack` on Colab (RAM is not the constraint there), raise the batch scale (bigger
batches at the same replay ratio: more transitions per 22 ms update), try `bf16` autocast on
the A100 (its tensor-core ratio is 16x, not the 4090's 2x), and run as many actors as there
are physical cores but no more. Then, if the learner still binds, pipeline the host gather
one update ahead (a contained code change, sketched below). I have pre-registered these as a
smoke-03 matrix in the runner and pointed the notebook at it; one 150 s row per lever and one
control, same budget as smoke-02. I also fixed the `actor did not exit cleanly` rows: they
were a shutdown bug, not a measurement problem, and the rates stand.

## What the numbers say

**Hero rows.** 4,062 tr/s from two actors at 128 worlds is 15.9 frames/s per actor, 63 ms
per hero frame. The trainer's own Mac measurement is about 17 ms for a 128-world actor
frame (`train_m3c.py`, perf-opp). Scaling: 2 -> 4 actors doubled the rate (8,014), 4 -> 8
added 14% (9,138) with CPU at 60%: the second hyperthread of each core adds little to a
numba loop, and eight actors plus the learner plus the sampler share six cores. The per-frame
cost is the numba simulator, the compiled v8 shield, the featurizer, and, with `--obs-pack`,
packing each row (3 ms per 128-row frame on my Xeon). The CUDA hero forward adds one
host-device round trip per frame but is not the bulk. This matches the pods (EPYC 2.4x
slower per core than the Mac, about 4.1k tr/s per run); Colab's vCPUs are slower still.

**Self-play rows.** 11,086 (64 worlds, 2 actors), about 11,500 (128 worlds, 2 actors) and
11,341 (128 worlds, 4 actors): the same number regardless of actors and worlds, queues full
at depth 8, `learn_s` 125 to 130 of 148 s. That is a learner cap at 44 to 45 updates/s with
about 257 transitions per update (`--batch-scale 2`: one update per 256 ingested rows). The
Mac's learner does the same update in about 48 ms on MPS (21 updates/s, 5.4k tr/s); the
4090 did it in 11.5 ms with CUDA graphs and batch scale 2 (22.2k ceiling). The A100 at 22 ms
sits between them, and the smoke's GPU utilization (54% mean on the learner-bound rows, 71
to 75% peak) says the GPU is idle for almost half of each update.

**Where the 22 ms go.** Per update at batch scale 2 the fast learner gathers 1,792 rows
(768 agent transitions, current and next observation, plus 256 label rows): 21.9 MB of raw
planes written into pinned staging, unpacked from the 3,064-byte packed form when
`--obs-pack` is on, then one upload, then the compiled loss (one CUDA graph), backward,
gradient clipping, fused Adam. The pipeline is one deep: the host gather for update i+1
overlaps the GPU's update i, then `_finalize` waits for i's finite-loss flag before launching
i+1. So the period is roughly max(host gather, GPU step) plus the non-overlapped launch and
sync work. Measured on my Xeon core (single thread, same code path, random valid rows):

| rings | batch scale | rows gathered | MB / update | host gather ms | host-only cap (tr/s) |
|---|---|---|---|---|---|
| packed | 1 | 896 | 10.9 | 6.4 | 20.0k |
| packed | 2 | 1,792 | 21.9 | 7.0 | 36.6k |
| packed | 4 | 3,584 | 43.7 | 29.2 | 17.5k |
| packed | 8 | 7,168 | 87.5 | 41.6 | 24.6k |
| raw | 1 | 896 | 10.9 | 2.5 | 50.9k |
| raw | 2 | 1,792 | 21.9 | 4.6 | 55.6k |
| raw | 4 | 3,584 | 43.7 | 10.5 | 48.6k |
| raw | 8 | 7,168 | 87.5 | 24.9 | 41.0k |

Two things to take from it. The packed gather falls off a cliff at batch scale 4 (the
staging buffer leaves the cache and the unpack kernel's scattered writes pay for it), so
raising the batch scale with `--obs-pack` on would buy nothing on a Colab core. Raw rings cost
less than half at every scale. On the actor side the same kernel is heavier: packing one
768-row self-play frame took 18.9 ms on this core against 0.8 ms for a raw copy, and a
self-play actor frame is where the six cores' time goes once the learner is lifted.

**The GPU leg.** The network is 1.64M parameters and 22.5 MFLOP per forward row (19.4
convolution, 3.1 linear). One batch-scale-2 update is about 4,600 row-forward equivalents
(three forwards over 768 rows, one over 256 anchor rows, backward about twice a forward over
1,024 rows), roughly 104 GFLOP. At 22 ms that is 4.7 TFLOP/s achieved. The A100's fp32
CUDA-core peak is 19.5 TFLOP/s (the 4090's is 82.6, which is why the 4090 is faster here
despite being the cheaper card); its TF32 tensor-core rate is 156 and bf16 is 312. cuDNN
uses TF32 for convolutions by default, but a 7-channel input conv at 31x31 is not a shape
tensor cores like, and the linear layers run true fp32 unless `set_float32_matmul_precision`
is raised (nothing in `train_m3c`/`learner_m3c` does). So on the A100, unlike the 4090, the
GPU leg of the update is plausibly compute-bound at fp32 at these batch sizes, which is why
the earlier "launch-bound, 1.2 to 1.5x between chips" reading understates the per-update gap
for fp32 on the A100; it holds again once the arithmetic moves to tensor cores.

**Policy lag.** With queues full, the actors are back-pressured and the data they deliver is
staler: lag 4.4 at two actors, 9.0 at four, with no throughput gain. Beyond the learner's
capacity, extra actors only age the data. Match actors to what the learner consumes.

## What I changed on this branch

1. **Shutdown fix (`train_m3c.py`, `stop_actors`).** Both 128-world self-play rows were
   marked `invalid_sample: actor did not exit cleanly` although the trainer returned 0. Cause:
   the learner set the stop event, drained each queue once, then joined each actor for 30 s.
   An actor whose queue was full (the normal state of a learner-bound run) had handed its
   last message to `multiprocessing.Queue`'s feeder thread; the process joins that thread at
   exit, and the thread blocks on a full pipe nobody reads any more. The join timed out, the
   actor was terminated, exit code -15. Reproduced here with a two-process repro (one drain:
   still alive after 8 s; drain while joining: exits in 0.1 s). The fix drains while joining
   under one shared 30 s deadline, then terminates whatever is left. Covered by
   `tests/test_m3c_actor_shutdown.py` (back-pressured actors exit 0 quickly; a stuck process
   is still terminated). The smoke-02 self-play rates were real; only the flag was wrong.
2. **A pre-registered smoke-03 matrix (`run_smoke.py --matrix smoke03`).** Six 150 s rows,
   all 128-world self-play, each one lever: `sp_w128_a4_pack` (control, equal to smoke-02's
   `selfplay_w128_a4`), `sp_w128_a4_raw` (no `--obs-pack`), `sp_w128_a4_raw_bs4`
   (`--batch-scale 4`), `sp_w128_a4_raw_bf16` (`--learner-precision bf16`), `sp_w128_a6_raw`
   (six actors, one per physical core), `sp_w128_a6_raw_bs4` (the combination I expect to
   win). Config keys `obs_pack`, `batch_scale`, `precision` drive the trainer flags; the
   smoke-02 matrix is unchanged and remains the default; `compare.py` requires the rows of
   whichever matrix the results record. The notebook parameter cell gains `MATRIX` and now
   points at `smoke-03` / `smoke03`. The runner also records `ram_bytes`.
3. **RAM check for raw rings.** With `--obs-pack` off the four- and six-actor rings are
   10.6 to 11.2 GB (file-backed under `--scratch-dir`, so no `/dev/shm` limit) against 2.8 to
   3.0 GB packed. The A100 runtime has tens of GB; the recorded `ram_bytes` confirms it.

What I did not change: the recipe. Batch scale 4 and bf16 are throughput arms in a
throughput smoke; a quality A/B (as batch scale 2 had) stays a separate, pre-registered
decision, and nothing here touches the served artifact or the gates.

## How to read smoke-03

- `a4_raw` vs `a4_pack`: the host-side cost of packing on both sides. Expect the learner cap
  to rise (fewer host ms per update) and, once the learner is no longer the limit, the actor
  frames to get cheaper too.
- `a4_raw_bs4` vs `a4_raw`: if updates/s stays near flat while transitions per update
  doubles, the GPU leg is not the limit and the ceiling roughly doubles; if updates/s halves,
  the GPU leg (fp32 compute) is the limit and bf16 is the next lever.
- `a4_raw_bf16` vs `a4_raw`: whether A100 tensor cores pay. On the 4090 autocast lost (the
  casts cost more than the 2x they bought); on the A100 the ratio is 16x. If it wins by much,
  the GPU leg was compute-bound and the H100 question below has a different answer than it
  did for the 4090.
- `a6_raw` vs `a4_raw`: whether the actors become the limit once the learner is lifted, and
  what six cores deliver. `a6_raw_bs4` is the combined row and the one to compare with the
  21,600/s bar. Watch `policy_lag` in all of them: a row that clears 4x with lag above the
  smoke-02 control's is clearing it with staler data.
- Utilization: GPU mean well above 54% with CPU under 80% means the learner's GPU leg is
  binding; CPU near 100% on six cores with GPU low means the actors are, and the only further
  lever is fewer, larger actor frames (256 worlds) or cheaper frames.

## If the learner still binds after smoke-03: pipeline the gather

The remaining host work can be hidden rather than reduced. The fast learner already double
buffers its staging; the change is to produce batch i+1's gather while update i is in flight
and before `_finalize` waits on it, so the period becomes max(GPU step, host gather) instead
of their sum plus sync. Two ways, both contained to `learner_m3c.py`:

- **Two-deep in the same thread.** Sample i+1 immediately after launching i (before the
  finite-loss wait), keeping the RNG order (sampling is the only RNG consumer). The rows for
  i+1 are then drawn from the replay as it stood one update earlier, which the multi-actor
  trainer already tolerates (it is not bit-reproducible against ingest timing). Keep it
  behind a flag (`--learner-prefetch`) so the reference-equality tests keep their path.
- **A gather thread.** `unpack_into` is `nogil` numba and the fancy-index copies release the
  GIL, so a worker thread can fill the other staging buffer while the main thread ingests
  actor messages and launches; the main thread's Python work is small per message.

Expected effect on the A100 at batch scale 2, raw rings: period from about 22 ms to about
the GPU leg (10 to 12 ms fp32, less with bf16), 80 to 100 updates/s, 20 to 25k tr/s before
any batch-scale change; with batch scale 4 the ceiling is then set by the six actor cores.

## Could an H100 do better, and what would an H100 setup look like?

Partly, and only after the host-side work above is done. The H100 changes one leg of one of
the two bottlenecks:

- **Actors: no change.** Both runtimes give the same class of x86 cores; the hero rows stay
  at 63 ms per frame and the self-play frames stay CPU work. Before paying for an H100,
  read `cpu_physical` and `ram_bytes` from the runner's environment record on that runtime
  (my earlier note assumed the same vCPU count; the record settles it).
- **Learner, GPU leg.** H100 SXM fp32 is 67 TFLOP/s against the A100's 19.5 (3.4x); TF32
  495 vs 156; bf16 990 vs 312. If smoke-03 shows the GPU leg compute-bound at fp32 (the
  `bs4` row halving updates/s, the `bf16` row winning big), an H100 cuts that leg to roughly
  a third: about 4 ms per batch-scale-2 update. The host leg is unchanged, so without the
  pipelining change the period only falls from about 22 ms to about 14 ms (70 updates/s,
  18k tr/s at batch scale 2): still under 4x. With raw rings and the prefetch, the H100 makes
  the learner effectively free and the six cores set the ceiling, which they also set on the
  A100 once its leg is under the actors' frame time.
- **Where the H100 is the right chip.** Lane B's teacher (10 to 30M parameters, higher
  replay ratio) and several league members' learners sharing one GPU: those are GPU-bound by
  construction and the H100's 2.5 to 3x per update becomes wall-clock. A single 1.6M-parameter
  learner fed by six x86 actors is not that workload.

So an H100-tailored setup is the same setup as an A100-tailored one (raw rings, batch scale
chosen by the A/B, bf16 if smoke-03 shows it pays, actors equal to physical cores, the gather
pipelined), plus one more thing that only the H100 makes sensible: running two or three league
members' learners on the same GPU with their actor sets splitting the cores, since one
member's learner no longer fills the chip. At Colab's rates (about 13 units/h H100 vs 9 to
12 A100) that only pays once the per-member learner is GPU-bound. My runtime recommendation
stands: A100 for smoke-03 and Lane A; H100 for Lane B's teacher arm and for shared-GPU
multi-member runs if smoke-03's bf16 and bs4 rows show the GPU leg mattering.

## Decision

Run smoke-03 on the A100 (notebook `RUN_NAME = 'smoke-03'`, `MATRIX = 'smoke03'`, pin the
merge commit of this branch). If `sp_w128_a6_raw_bs4` clears 21,600/s with a policy lag no
worse than the control, Colab is worth it for Lane A under that configuration, pending the
batch-scale-4 quality A/B on the Mac. If it falls short but `a4_raw` lifted the learner cap
and the `bs4` row kept updates/s, the gather pipelining above is the next change and I would
make it. If the `a6` rows show six cores saturated below the bar, Colab's ceiling for this
trainer is the core count and the honest answer is more runs per session, not faster ones.
