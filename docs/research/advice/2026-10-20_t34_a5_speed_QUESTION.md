# Speed review: T34 v2 teacher on Colab (2026-10-20 ~13:55 PT)

GOAL. Label teacher worlds (DESIGN_V2 Amendment 2 teacher, research/t34_exit_b8/teacher_v2.py) and run Round 0
(research/t34_round0/, 240 paired worlds) and the unpaced Path B arm on Colab cheaply and fast. Owner pre-approved all
Colab use and wants ~1,160 units spent usefully by ~11-20. A5 v2 GO = >= 50 completed worlds/h per L4 lane.

MEASURED (verify):
- A5 #1 (serial shadow fill): ~50 active row-steps/s per lane, 0 decisions in 600 s
  (snake-dqn-artifacts/t34-a5-20261020/l4-*-output/exports/).
- Shadow batching (byte-identical; research/t34_exit_b8 VERIFICATION_SHADOW_BATCHING.md, profile in
  snake-dqn-artifacts/t34-shadow-batch-20261020/): Mac 1 thread, 32 worlds x 1,000 frames: serial 136 row-steps/s,
  batched 110 (-19%); 86,008 row-steps / 32 worlds per 1,000 frames.
- A5 #2 (batched, 2 L4 lanes x 2 workers x 5 threads, 256 roots/worker, CUDA): telemetry in
  snake-dqn-artifacts/t34-shadow-batch-20261020/final-l4-{0,1}-supervisor.log and final-l4-*-output/poll-*.json:
  38,912 executed row-steps in 1,384 s per worker (~28/s/worker, ~56/lane), 0/256 worlds done; one 256-world batched
  step ~9 s (~35 ms per world-step). Session hit the 30-min wall cap before export (manifest timeout too tight).
- L4 runtime: 12 vCPU (~6 physical); Colab CPU-only runtime <= 0.5 u/h (cores unknown).

OPEN QUESTION. Where does the ~35 ms per world-step on Colab (vs ~7 ms on one Mac thread) actually go, and what
configuration and/or code change gets the teacher to a high completed-worlds/hour per unit on Colab? Give estimates,
the single cheapest measurement to confirm, and the exact runtime layout (processes, threads, worlds per process,
GPU or CPU-only runtime) you would dispatch next.

RULES. Advice only; read files; tiny read-only probes (<=60 s, <=2 threads); no jobs, tmux or Colab. One answer
file, <=900 words.
