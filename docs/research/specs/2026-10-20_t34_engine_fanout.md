# T34 teacher engine: process fan-out + fast digest (Sol spec, 2026-10-20 ~14:05 PT)

Evidence: docs/research/advice/2026-10-20_t34_a5_speed_fable.md (Fable systems speed review of A5 #2): Mac serial
row-step 5.84 ms = full_digest 3.73 ms (64%, recursive Python walk + repr of MT states) + evaluate 0.93 + opponent
dispatch/v8 0.76 + numba sim.step 0.27; lockstep W=8 2.2 ms vs serial 1.0 (deepcopy merge/scatter); on Colab ~95%
single-threaded CPython, 2 of 12 vCPU used, GPU idle; CUDA batch-1 evaluate 4.35 ms. Portfolio lane L0
(docs/research/final_run_20261019/CAMPAIGN_LOG.md). Owner: all Colab pre-approved.

1. Fan-out: the benchmark and label entrypoints (research/t34_exit_b8/{bench.sh,label.sh,benchmark.py}) run K
   independent single-thread processes (K default = nproc, overridable), each on a disjoint shard of worlds, CPU
   device, OMP/MKL/OPENBLAS/NUMBA/VECLIB=1, no GPU; relax the benchmark.py guards (~L224-238) that force the
   2-worker x 5-thread layout. Aggregate per-process reports into the lane report (completed worlds/h, row-steps/s,
   per-process stage breakdown, nproc, RAM). Keep the CUDA batched path available but OFF by default on CPU.
2. Fast digest: replace the recursive Python full_digest with an O(bytes) digest over contiguous numpy buffers
   (bodies, ordered food, RNG states via getstate()->bytes, opponent carry arrays) with a FIXED field order; it must
   detect the same divergences: prove with tests that every field change flips the digest, and that M1-style
   identity (shadow vs independent source) still reports 0 differences on the M1 fixtures. Exports/labels must stay
   byte-identical to the current implementation on the Mac fixtures (digest values may change only if they are not
   exported; if they ARE exported, version the field and document).
3. Skip lockstep merge/scatter when running on CPU (serial per world is faster).
4. Mac profile receipt (<=2 threads; Phase 0 running): serial row-step before/after; 2-process fan-out smoke.
5. Manifests: rebuild A5 L4 manifests with K=12 processes x 1 thread, CPU device, worlds per process ~4, workload
   <= 45 min, session timeout >= workload + 25 min (setup + export) — the A5 #2 30-min wall killed the export; plus
   a CPU-only-runtime variant that records nproc/RAM first (rate: use observed provider rate or 0.5 u/h ceiling).
   Validate only; leave dispatch commands.
Rules: Mac only; no live Colab/tmux; <=2 CPU threads; commit/push often; merge to main; run test_t33*/test_t34*
(-n 2) + new tests; report commit SHA, before/after ms per row-step.
