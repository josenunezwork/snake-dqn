# Speed vs constraints — owner question (2026-10-20 ~21:30 PT)

## Owner's words
"Is this why we're so slow? … ask if we should stop to increase speed. I don't know if I care about a lot of the things
restricting us, and I'd hope every part is optimized."

## Goal
A better served snake than bare B8 s3 (the current Watch hero), as soon as possible; the owner cares about results, not
process for its own sake, and wants the system fast.

## Open question
Should the project pause or reshape its current work to make the engine and search much faster — and which of the current
restrictions (bit-exact parity with the Mac reference sim, per-step state hashing/fingerprinting, full replayability and
receipts, Mac-only strict gates, serial Python opponents, freeze-before-data review loops) are actually needed for a
trustworthy "this snake is better" claim, which could be relaxed or redesigned, and what would you do, in what order,
starting tonight?

## Facts (verify in the repo; cite file:line)
- Where time goes (Mac profile of one teacher/search row-step, before the 10-20 digest fix):
  `docs/research/advice/2026-10-20_t34_a5_speed_fable.md` — full_digest (per-step state hash) ~64%, B8 forward ~16%
  (22 MFLOP; batch-1 CUDA slower than CPU), v8 opponent hooks ~13%, numba sim step ~5%. Digest since made 1.71× faster
  (`research/t34_exit_b8/ENGINE_FANOUT.md`); production L4 layout 12 single-thread procs → ~55 teacher worlds/lane-h.
- Serving budget: per-update p95 ~8.4 ms measured vs 11.69 ms contract (`docs/research/t12_b8_serving_qualification_2026-10-17.md`);
  T3-dense search ~1.3 s/decision vs ~0.2 s real-time budget (`docs/research/specs/2026-10-20_t34_nogo_branch_plan.md`);
  T3-dense acting beat B8 (+92 mass, 54/64 vs 38/64 scripted survival; FINAL_PLAN.md §2). Path B serves a cheaper search.
- Simulator: `src/simd_env/grid_sim.py` (NumPy+numba, but ordered food lists / per-env Python RNG / event resolution stay
  scalar Python, docstring :1-40), reference `batch_sim.py`, parity harness `parity.py`; v8 opponents `vector61_policy.py`,
  `v8_nb.py`; GPU expert analyses `docs/research/advice/2026-10-20_a100/gpu_expert_{1,2,3}.md` (device port ~3 weeks;
  thread-per-world numba.cuda design; break-even arguments).
- Why the restrictions exist: `docs/research/final_run_20261019/{SUCCESS_CONTRACT.md, P7_STATISTICS.md, CAMPAIGN_LOG.md}`,
  memories of past failures (self-play eval inflated skill; float ties Mac vs pod; best-save freezes) — see
  `docs/research/GLOSSARY.md`, `docs/TIMELINE.md`.
- In flight now: Round 0 (Colab, verdict ~Thu), Path B paced retention (Mac, starts after Phase 0 ~Wed, ~18–19 h),
  Claim A loss amendment, Round 0-D rescope; Colab lanes plan `docs/research/advice/2026-10-20_colab_lanes/JUDGMENT.md`.
- Resources: Mac M5 Pro 18 cores (Phase 0 uses 6 now); Colab ~1,150 units (L4 12 vCPU ~1.7 u/h; G4 48 vCPU ~8.7 u/h);
  implementers: Codex Sol and Claude/Opus agents (many can run in parallel).

## Rules for your answer
Read-only (no edits/commits/jobs; only trivial Mac probes, 1 thread, nice 10, < 30 s). Full answer in your final message.
Be concrete: what to change, expected speedup with arithmetic, engineering time, risks to the validity of a "better snake"
claim, and whether in-flight studies (Round 0, Path B) should continue unchanged while speed work proceeds in parallel.
Label verified vs estimated with file:line. Make a recommendation; do not list options for someone else to pick.
