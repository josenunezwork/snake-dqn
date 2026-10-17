# T16 Mac baseline for the Colab throughput comparison (2026-10-17)

Throughput smoke only; not gate, quality or serving evidence. Requested by the advisor's Colab review
(`docs/research/advice/2026-10-17_colab_review.md`, item 9): the self-play rows count every driven snake's
transitions, so they need a matching Mac baseline.

Same command as `research/t16_colab/run_smoke.trainer_command` (B8 recipe, batch-scale 2, obs-pack, seed 3,
init B8_s3 final), on the Mac with `--device mps`, 128 worlds, 2 actors, `--max-wall-seconds 180`, no compile
(MPS), measured with `run_smoke.measured_rate` (steady-state window after the first learner update).

| Config | Transitions/s | Tail (last 60 s) | Learner updates/s |
|---|---:|---:|---:|
| hero-only | 5,283 | 5,280 | 20.6 |
| self-play (`--all-snakes-learner`) | 5,459 | 5,454 | 21.3 |

Both are learner-bound at the same ~21 updates/s, so the Mac's rate does not change with self-play. The 4x bar
for Colab is about 21,100 (hero-only) and 21,800 (self-play) transitions/s; compare.py's single 21,600 bar is
within 1-3% of both. Raw logs: `snake-dqn-artifacts/t16-mac-baseline/`.
