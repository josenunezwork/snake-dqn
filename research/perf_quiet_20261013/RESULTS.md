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

# CUDA benchmarks (RunPod RTX 4090 secure, x86 EPYC, torch 2.5.1+cu124; 2026-10-14 ~05:40–07:10 UTC)

Raw: `snake-dqn-artifacts/perf-cuda-bench-20261013-try2/out/` (runtime image) and
`perf-cuda-compile-20261014-try1/out/` (devel image, gcc available). Both pods deleted & verified; ~$0.5 total.

| learner (bv8) | ms/update | tr/s ceiling | note |
|---|---|---|---|
| baseline | 14.9 | 8,603 | |
| t1 | 12.5 (13.5–14.6 in the 2nd session) | 10,273 | |
| t1 fp16 / bf16 | 15.7 / 15.4 | ~8.2–8.3k | slower: no precision tier |
| t1 + batch-scale 2 / 4 | 15.6 / 22.4 | 16,375 / 22,873 | semantics change (A/B) |
| t1 + compile default | 13.7 | 9,364 | 1.07× vs eager twin |
| **t1 + compile reduce-overhead** | **8.4** | **15,170** | **1.74× vs eager twin**; drift 0.040 of update norm ≈ 1-ulp init floor 0.045 |
| t1 + k2 + compile reduce-overhead | 11.5 | 22,212 | |
| max-autotune-no-cudagraphs | — | — | first-steps check rejected (grad mismatch) → eager |

Compile needs a C compiler: the training *runtime* image has none (all compile cells fell back
to eager in session 1); use `pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel`.

| actor (single, numba, CUDA hero fwd) | 64 | 128 | 256 worlds |
|---|---|---|---|
| ego2s-b | 4,251 | 5,424 | 6,234 hero tr/s |
| B8 recipe | 3,338 | 4,056 | 4,540 |
| B8, long bodies | 2,614 | 2,987 | 3,285 |
| `--compile-actor` | no gain (±5 %) | | |

Pod EPYC cores are ~2.4× slower per actor than the M5 Pro: pods are actor-bound -> 3–4 actors/run.

**Pod production flags:** devel image; `--learner-fast --compile-learner reduce-overhead --obs-pack
--v61-backend numba` fp32, `--worlds 128`, `--actors 3`–`4`; `--batch-scale 2` only after the A/B.
**Mac production flags:** `--learner-fast --obs-pack --v61-backend numba --worlds 128 --actors 2`
(no compile on MPS: wrong gradients in torch 2.9.1, auto-fallback).

# Batch-scale A/B (item 3; Mac, B8 recipe, seed 0, 20M, `--learner-fast --obs-pack --v61-backend numba --worlds 64`, both arms concurrently)

No-veto dev probe, 32 worlds × 3 mixes vs frp3-s12 + v8 (434.9); raw `snake-dqn-artifacts/perf-ab-k2-20261014/`, probes `perf-ab-probes-20261014/`.

| arm | 10M | final | survival frozen / scripted / mixed | wall |
|---|---|---|---|---|
| K1 | 494.2 (+59) | 527.3 (+92.4, SE 28) | 0.81 / 0.62 / 0.75 | 2.10 h |
| K2 (`--batch-scale 2`) | 535.5 (+101) | 537.5 (+102.6, SE 27) | 0.69 / 0.59 / 0.81 | 1.49 h |

K2 − K1 = +10 (well inside one SE; one seed each; B8's own seed spread was +63…+132): no
evidence of a quality loss, 1.41× faster wall-clock. **Adopted: `--batch-scale 2` in the production
recipe** (Mac and pods), with the single-seed caveat.
