# Spec: T33 round 2 — implement the Fable hot-path review

Owner: Claude. Implementer: Sol. Reviewer: Fable. Base: your HEAD 76f16ced.

**Brief:** `/Users/josenunez/Projects/ml/snake-dqn-artifacts/t33-reviews/fable-hotpath-review-1.md` (read it fully;
it measured ~110 us host per row-step vs ~1 us GPU, i.e. host-bound ~100x; memory blocker; leaf-scale units;
ablation prefix trick; lazy fork; merge deep-copies; serial `_v8_rows`; V61 `_food_arrays`; per-row RNG/ε loops;
obs buffer zeroing/allocs; GridBatchSim `ufunc.at`/Python loops; GPU head precision; plus correctness traps).

Your Mac measurement: N8/H128 ~0.28 decisions/s CPU, ~0.8 MPS (under concurrent load). Target remains
~100-200 decisions/s per L4 host (~12 vCPU) at the budget the ablation selects.

A Claude-run 300-decision ablation (your command, calibration-v1) is in progress; do not depend on its result,
but implement the **single-run prefix ablation** so future calibrations cost ~10x less, and make the leaf
scale a calibrated, documented quantity (units!).

## Required outcome

Implement the review's items in its gain order, each with a micro-benchmark before/after (rows/s for the
affected stage) recorded in `research/t33_batched_teacher/OPTIMIZATION_LOG.md`:
1. Memory: re-cap body rings to `max(seg_count)+H` at merge; int64 kept without copies; size benchmark shapes
   in rows with a memory admission check (target ~50k resident rows on an L4 host).
2. Lazy fork (leaders until hero ε fires), pre-drawn ε arrays, RNG independence proven by test.
3. `merge_worlds` without per-row deep copies.
4. `_v8_rows` prange with thread-local scratch reading `owner_pad`.
5. V61: vectorize `_food_arrays`, skip inactive rows, keep the Apex MLP on device.
6. Obs buffer: no whole-slab zeroing, no per-step allocs, parallel region kernel, single initial Q per world.
7. GridBatchSim hot loops (`ufunc.at`, corpse/consume/spawn) — numba where safe, bit-identical stepping.
8. GPU: fp32 dueling head (fp16 trunk), double-buffered A/B overlap; drop GPU-crop step 5.
Then a **256 / 1,024-world Mac lockstep profile** (<= 4 threads; short) reporting per-stage µs/row-step and
decisions/s, and an updated projection for one L4 host (12 vCPU) and for 3 L4s.

Tests: bit-identical stepping vs reference sim; batched teacher choices identical to the reference on fixtures
at equal search params (document fp16/tie exceptions); RNG independence under fork; dead-row masking;
uint8 clamps. Existing tests green; Black/isort/flake8. Commit + push `t33-batched-teacher-20261019`.

## Rules

Same as round 1: only this worktree; no Colab/RunPod/tmux/serving/publishing; never touch the Phase 0 or S1
worktrees/artifacts; <= 4 threads locally (a Claude ablation also runs at nice 10 with 3 workers).
Final message: OPTIMIZATION_LOG summary table, Mac lockstep decisions/s, L4 projection, updated L4 manifest
status, unverified items.
