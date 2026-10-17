# Advice: review of the T17 plan (parallel gather, ingest cleanup, clean exit, smoke-03) (2026-10-17)

Advisory note (cloud advisor), requested by the owner through the local session. Reviewed:
`docs/research/t16_colab_smoke02_results_2026-10-17.md` (diagnosis, main `8c3ddfe`) and
`docs/research/specs/t17_colab_learner_gather_spec_2026-10-17.md` (spec, main `3eed856`),
against my own reading in `2026-10-17_colab_slower.md` (same branch, `colab-slower-20261017`)
and the two measurements there (host gather per update and per-frame packing on a 2.1 GHz
Xeon core; the learner's FLOP count). Codex is implementing the spec on
`t17-colab-gather-20261017`; where this branch overlaps it, I say which to take.

## Verdict

Agree with the diagnosis and with the shape of the plan: self-play on the A100 is bound by the
learner's host thread, updates/s is the right metric, more actors only age the data, the H100
does not help until the GPU leg binds, and the -15 exits were the drain-once shutdown. The
updates/s target (84 at an unchanged 256 transitions per update) is the honest bar.

I would change five things, in order of how much they matter:

1. **Measure the learner alone on the A100 before writing the thread pool (a P0, five
   minutes of credits).** The spec's 19 to 23 ms of host work per update is a profile inside
   the live trainer; my isolated measurement of the same `_sample_fast` path at batch scale 2
   is 7.0 ms with packed rings and 4.6 ms raw on a Colab-class core. Something accounts for
   the 3x: Colab's core, the file-backed mmap rings (page-cache reads through 4 KB mappings
   over 2.8 GB of rings), ingest work attributed to the update, or cProfile overhead on
   numba calls. `research/perf_learner_20261013/bench_learner.py --devices cuda` already
   measures ms/update with no actors, eager and compiled, batch scale 1/2/4, fp32/bf16; run it
   on the A100 and add a raw-ring cell. It splits the 22 ms into host and GPU exactly, and it
   says whether 84 is reachable at all: if the GPU leg is 8 to 11 ms as the diagnosis
   estimates, a perfect P1 leaves a period of about 11 to 14 ms (70 to 90 updates/s), on the
   bar, not over it. The bench also settles the `/dev/shm` versus file-backed question by
   pointing `--ring-dir` at each (check `df /dev/shm` on the runtime first).

2. **Turn `--obs-pack` off on Colab before parallelizing its unpack.** It is a RAM lever for
   the pods (6.6 GB versus 16.2 GB per run); the A100 runtime has tens of GB and the four- and
   six-actor raw rings are 10.6 to 11.2 GB. On my core it cuts the learner's gather from 7.0
   to 4.6 ms at batch scale 2 and from 29.2 to 10.5 ms at batch scale 4 (the packed gather
   falls off a cache cliff there), and it removes 18.9 ms of packing from every 768-row
   self-play actor frame (a raw copy is 0.8 ms), which is where the six cores' time goes once
   the learner is lifted. Zero code, byte-identical batches by the obs-pack contract. P1 then
   parallelizes a smaller thing, and if P1 slips, raw rings alone may carry the a4 row most
   of the way. Make the control row packed and every other row raw.

3. **P1: chunk by rows, not by ring, and keep the pipeline in view.** Per-ring threads give
   two-way parallelism on a two-actor row and three-way on three; the spec's `a2 g3` cannot
   use its third thread. Split each gather's destination slice into N contiguous chunks
   across rings (the staging rows are disjoint either way) so N follows `--gather-threads`.
   And note what P1 cannot do: the fast learner is already one deep (the gather for i+1
   overlaps the GPU's update i; `_finalize` then waits), so the period is max(gather, GPU)
   plus the non-overlapped launch, clip, optimizer and sync work. Once the gather is under the
   GPU leg, P1 is done and the next millisecond comes from the GPU leg (bf16 autocast: the
   A100's tensor-core ratio is 16x, not the 4090's 2x where autocast lost) or from a two-deep
   pipeline (sample i+1 before waiting on i; RNG order unchanged; behind a flag because the
   rows are then drawn from the replay as it stood one update earlier). I would add a bf16
   row to smoke-03 now rather than discover after P1 that the GPU leg is the new cap.

4. **P4: shorter rows, more actors, same budget.** Four 600 s rows plus grace and setup do
   not fit the 3,600 s runner cap or the notebook budget without raising both; 300 s rows do,
   and smoke-02's 150 s windows were already steady (the log rate at 148 s matched the
   steady-state rate on every row). Two and three actors can show the learner cap moving but
   cannot show Colab clearing 4x: three actors need a 768-row self-play frame every 106 ms,
   and the back-pressured smoke-02 frames were about 133 ms (true capacity unknown). Add
   four- and six-actor raw rows so the actor ceiling is measured in the same session; keep
   `--actor-queue-frames 4` (it halves the lag the a4 row showed). Pass on updates/s as
   specified, and also require `policy_lag_mean` no worse than the control: a row that
   clears 84 with staler data is not the same recipe. My branch pre-registers such a matrix
   (`run_smoke.py --matrix smoke03`: packed control, raw, raw + batch scale 4, raw + bf16,
   six actors raw, six actors raw + batch scale 4; config keys `obs_pack`, `batch_scale`,
   `precision`); the spec's `--suite` and `gather_threads` key are the same mechanism. Fold
   the row set into whichever lands; I have no attachment to the flag name.

5. **Batch scale 4 is a throughput arm, not only a recipe arm.** The spec defers it as a
   recipe change, which is right for the league's recipe. For the smoke it is the lever that
   doubles transitions per update at a near-constant update period when the period is
   host-bound, and it is the fallback if P1 lands at 70 updates/s. One raw batch-scale-4 row
   costs 300 s and tells you whether the fallback exists. The quality A/B stays separate and
   pre-registered as before.

## Point by point

- **Diagnosis.** Agreed on all four points. One refinement: the GPU leg is not only
  launch-bound on the A100. The update is about 104 GFLOP at batch scale 2 (22.5 MFLOP per
  forward row, 1.64M parameters); at 22 ms that is 4.7 TFLOP/s achieved against an fp32
  CUDA-core peak of 19.5, and the linear layers run true fp32 (no `set_float32_matmul_precision`
  anywhere in the trainer). The 4090 is faster per update because its fp32 rate is 4x the
  A100's, not because it launches faster. That is why bf16 belongs in smoke-03 and why the
  H100 answer flips from "no" to "yes, for the GPU leg" only if the bf16 row shows that leg
  compute-bound.
- **P1.** Agreed, with the row-chunking change above and a note that `NUMBA_NUM_THREADS` is
  irrelevant to `nogil` kernels called from Python threads (it governs `prange`, which these
  kernels do not use). The byte-identity test is the right invariant; test it with rejected
  rows present (the compaction path) and with the label gather's re-check.
- **P2.** Agreed; small but free. The bincount is per message, not per update (about 15
  messages/s on the self-play rows), so expect a few percent, not a step. The halt condition
  can stay exact with a running count.
- **P3.** Done on this branch: `train_m3c.stop_actors` drains all queues while joining under
  one shared 30 s deadline, then terminates what is left; `tests/test_m3c_actor_shutdown.py`
  has the two tests the spec asks for (back-pressured actors exit 0 within seconds; a stuck
  process is still terminated). The actor side already has the `put(timeout=1.0)` loop that
  re-checks `stop_evt` (`train_m3c.py`, the send loop), so no actor change is needed. Take
  this commit for P3 and spend Codex's time on P1.
- **P4.** See item 4. Also: the cProfile window is useful but keep it off the scored rows
  (profile overhead on 15 numba-heavy messages per second is small, on the learner's Python
  it is not); and report `learn_s` per update beside updates/s, since the two together say
  whether the learner was busy or waiting.
- **Deferred items.** The GPU-resident ring is the real fix if P1 and raw rings do not reach
  the bar, and it is simpler than the spec's "packed ring with on-device unpack": store raw
  rows on the device (the full 500k-row replay is 6.1 GB of the A100's 40 GB), upload each
  actor message's rows at ingest (11.5k rows/s is 140 MB/s raw, nothing), and gather by
  `index_select`; no unpack kernel to write, and the host does no per-update tensor work at
  all. League mode stays a prerequisite for Lane A and should be measured after, as the
  spec says: its per-slot forwards and frozen-pool opponents change the actor frame, and the
  smoke numbers do not transfer to it unmeasured.

## Is this the best use of the Colab credits toward a better agent?

The credits, yes: smoke-03 costs about 6 units and the T17 work costs engineering time, not
units. The time is the question. The agent gets better through the league design (opponent
pool, slot count, PBT rules, the gates), not through 4x; 4x only decides where the league
runs. So I would time-box T17: P0, raw rings, P1 as specified, P3 from this branch, one
smoke-03 session. If that session clears 84 updates/s with lag no worse than the control,
Colab hosts Lane A and the batch-scale A/B follows. If it lands at 60 to 80, do not start a
second week of learner engineering: run Lane A on the Mac at 1x (it is learner-bound at the
same 21 updates/s either way, and the Mac's data is not stale), and spend the Colab credits on
Lane B's teacher arm, which is GPU-bound by construction and is the one workload where the
A100, and later the H100, pay for themselves without any of this.

One more thing I would do differently, cheaply: measure the Mac's own learner with the same
bench cell. The Mac is learner-bound at 48 ms per update on MPS; if its host gather is small
on an M-series core, P1 does nothing for the Mac and the comparison baseline stays 21
updates/s. If it is not small, P1 speeds up the Mac too, and the 4x bar moves.
