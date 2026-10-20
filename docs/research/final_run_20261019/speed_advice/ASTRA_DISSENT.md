# Astra's points that differ from Fable's GPU answers (2026-10-19 ~22:00 PT)

Full texts: /Users/josenunez/Projects/ml/snake-dqn-speed-perf/docs/research/advice/2026-10-19_t34_gpu_astra_perf.md
and /Users/josenunez/Projects/ml/snake-dqn-speed-search/docs/research/advice/2026-10-19_t34_gpu_astra_search.md

Both Astra advisors agree: teacher on L4 first, Path B on Mac CPU, no H100/G4 without measured benefit.
They differ from your answers on:

1. **Fits are the real GPU case.** "Yes, especially for batched student fits": once the loader caches
   observations and keeps the GPU fed, compare a complete fit on L4 vs A100-40 (A100-40 at 5.4 u/h wins if
   >2.7x faster whole-job). Illustration: a 4 h L4 fit = 8 u vs 3x faster A100 = 7.2 u. (Fable estimated fits
   at ~1-2 min of GPU / ~350 TFLOP, i.e. trivial on any GPU.)
2. **Colab CPU-only / High-RAM runtime** (CLI acceptance observed 0.08 u/h; ceiling 0.5 u/h; cores/RAM
   unknown) could beat L4 for the host-bound teacher if it reaches >1/4 of L4 complete-world throughput,
   including CPU CNN inference.
3. **Bisection is not certified.** Rescue availability need not be monotone in time, so bisection on "some
   alternative survives" may not find the true last avoidable decision; sparse LAD labels need extra coverage
   (M2's intervention-recovery check), and "free" overlapping confirm labels are not independent decisive
   examples.
