# Quiet-Mac benchmarks (2026-10-13 22:32–22:58 PT, M5 Pro, torch 2.9.1, MPS)

No training/probe jobs; one stuck system daemon (coreaudiod) used one core throughout.
Raw JSON / logs: `snake-dqn-artifacts/perf-quiet-mac-20261013/`. Code: branch `perf-integrate`.

## Learner (bench_learner, bv8, real batch 384 + 128)
| config | ms/update | tr/s ceiling | vs baseline |
|---|---|---|---|
| baseline | 35.0 | 3,658 | 1.00× |
| t1 (`--learner-fast`) | 30.4 | 4,212 | 1.15× |
| t1 + `--batch-scale 2` | 44.7 / 256 tr | 5,730 | 1.57× |
| t1 + `--batch-scale 4` | 85.1 / 512 tr | 6,016 | 1.64× |

## Actor (bench_actor, single actor, B8 flags)
| worlds | python v61 | numba v61 | × | numba + compile-actor |
|---|---|---|---|---|
| 64 | 4,210 | 7,845 | 1.9 | 8,615 |
| 128 | 4,962 | 10,176 | 2.1 | 10,600 |
| 256 | 5,337 | 11,300 | 2.1 | 11,631 |
| 64, long bodies | 1,184 | 6,264 | 5.3 | — |
| 128, long bodies | 1,336 | 7,324 | 5.5 | — |

## End-to-end trainer (B8 + `--v61-backend numba --learner-fast --obs-pack`, 700k tr, steady)
| worlds × actors | tr/s | learner busy |
|---|---|---|
| 64×2 | 3,726 | 0.91 |
| 128×2 | 3,898 | 0.94 |
| 256×2 | 3,968 | 0.96 |
| 128×3 / 128×4 / 256×4 | 3,919 / 3,924 / 3,986 | 0.95–0.97 |
| 128×2 + batch-scale 2 | 5,364 | 0.90 |
| 256×3 + batch-scale 2 | 5,337 | 0.93 |

Learner-bound in every configuration: more actors / worlds do not help. Recommended Mac
config: `--worlds 128 --actors 2 --v61-backend numba --learner-fast --obs-pack` (+ `--batch-scale 2`
only if the A/B at `snake-dqn-artifacts/perf-ab-k2-20261014/` shows equal policy quality).
