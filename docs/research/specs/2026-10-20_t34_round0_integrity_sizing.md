# T34 Round 0 — integrity rules + A5-based sizing before ratification (Sol spec, 2026-10-20 ~15:40 PT)

A5 v2 PASSED (GO >= 50 completed worlds/h/lane): L4 lanes 57.2 and 54.4 worlds/h, 377/432 active row-steps/s,
16.3k/20.1k row-steps/world, export ~8.5-8.9 MB/world, RAM peak ~0.5 GB; layout K=12 single-thread CPU-inference
processes, W1 (receipts: snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s*-output/
exports/benchmark.json). The Colab CPU-only runtime has nproc=2 (37 row-steps/s aggregate) -> not used.
Update the DRAFT Round 0 package (research/t34_round0/, main) before review:
1. Layout: the A5 PASS layout (K=12 x 1 thread, W1, CPU inference on L4); both arms of a pair on the same runtime.
2. Sizing: chunks from measured worlds/h (teacher arm ~55/h/lane; bare arm cost measured in the A6 smoke) so
   est <= 180 min, session timeout >= workload + 25 min, export per chunk < 2 GiB at ~9 MB/world x1.25 margin;
   <= 3 lanes concurrently (account cap; other Colab jobs share it); recompute the unit cap (2.0 u/h priced, 20%
   reserve) and state it.
3. Integrity rules (same as the ratified Path B package, research/t34_pathb/retention_history_v2.py and README_V2):
   fixed phase/chunk directories under one study root (create-only), a remote create-only tag per dispatch/phase,
   an append-only receipt-hash ledger in the repo that must be committed+pushed before analysis, frozen worktree at
   the ratified source commit, operator abort / unexplained interruption = KILL, one look, decision.json sole verdict.
   Colab chunk failures: one outcome-independent mechanical rerun per chunk under a fresh attempt id (preserve
   both), never after looking at outcomes.
4. Mac replay of a 10% sample per chunk (A2-style), reported; > 2% dropped worlds in a chunk -> stop and report.
5. Tests; draft validates; leave exact ratify + dispatch commands. Do NOT ratify or dispatch.
Rules: Mac only; no live Colab/tmux; <=2 CPU threads; commit/push often; merge to main; report SHA.
