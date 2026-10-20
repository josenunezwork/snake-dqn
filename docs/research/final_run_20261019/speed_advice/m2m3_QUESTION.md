# T34 after the M2/M3 NO-GOs — open question (2026-10-20 ~00:20 PT)

GOAL. Ship a clearly stronger Watch hero than bare B8 s3 under the owner-ratified T34 contract
(docs/research/final_run_20261019/{SUCCESS_CONTRACT,DESIGN_V2_SHADOW,P7_STATISTICS}.md on main):
Claim A = bare distilled student, Claim B = labelled live-assisted hero (one extra Mac thread, never blocks a
tick, bare fallback). The owner has delegated the decision on how to proceed to Claude; your advice decides it.

MEASURED FACTS (verify; full detail in /Users/josenunez/Projects/ml/snake-dqn-shadow/research/t34_shadow/RESULTS.md
and receipts under /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-shadow-20261019/):
- M1 GO: a shadow continuation (D=300) stays bit-identical to the real Watch game, 15,000 frames, 3 mixes.
- M2 registered gate NO-GO (stands; no post-hoc relabel): replaying the T3 dense bank (64 worlds, 75 T3 action
  changes in 22 worlds; T3 dense: 54/64 vs 38/64 scripted survival, +92 mass), bisection recovered 39/75 (52%) at
  17.3k row-steps/world (23x cheaper than 400k); stride-10+edge teacher rule 37/75 (49%) at 31.2k (12.8x). Exact
  stride-1 truth found a horizon-surviving alternative in only 42/75 intervals; on those 42 bisection got 39 (93%),
  teacher 37 (88%). The other 33 T3 changes extended alive time without surviving the 300-frame label horizon.
  LAD error p50/p95: bisection 3/145 decisions, teacher 0/41.
- M3 NO-GO for the current serial T3 continuation path: a LAD search on live Watch rosters takes 30-63 s per doom
  on one Mac thread (frozen p50 53 s, mixed 33 s; n=5) versus 12.5 s (24 fps) / 24.9 s (12 fps) of D=300 lead ->
  0/49 T1 late deaths reachable. Shadow upkeep is cheap (confirm+advance 9-22 ms/tick; 300-frame catch-up
  1.6-4.3 s). Measured under Phase 0 load (6 workers). M4: ONNX Runtime B8 batch-1 0.14 ms (2.5x PyTorch), max
  |dQ| 3.8e-5, 0 action diffs in sample. T33 has a batched lockstep engine (research/t33_batched_teacher) never
  used for this path. Earlier advice: docs/research/final_run_20261019/speed_advice/*.md,
  docs/research/advice/2026-10-19_t34_speed_astra_*.md.
- Sol builds: teacher v2 (shadow + Amendment-1 labels, batching, L4/Mac entrypoints) running; fit cache done;
  Path B v2 not started. Colab transport qualified; nothing spending.

OPEN QUESTION. Given these results, what should T34 do next — for the teacher labels (what target should a
label encode, and what about the 33 alive-extending changes) and for Path B (can a live search reach the
decisions that matter in time, and how) — and what single pre-registrable measurement should decide it?
Give estimates with reasoning; say plainly if a part of the plan should be dropped.

RULES. Advice only; read files, tiny read-only probes (<=60 s, <=2 threads); no jobs, tmux, Colab or paid
services; do not disturb Phase 0 or the running Sol build. One answer file, <=1,000 words.
