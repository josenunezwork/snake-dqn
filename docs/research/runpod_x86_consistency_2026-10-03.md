# RunPod x86 consistency check (2026-10-03)

> **Superseded — see [STATUS.md](../STATUS.md) (2026-10-08):** the later Mac/pod divergence diagnosis (branch `diag-divergence`) shows that bitwise identity fails at random on exact float32 Q ties, at about 1 in 50 H10000 episodes. This document's all-identical sample is preserved as dated evidence.

**Question:** do eval episodes run on RunPod Linux x86 CPUs give results bit-identical to
the same episodes recorded on the Mac (Apple M5 Pro)?

**Answer: yes, for every episode tested.** 28 of 28 H5000 episodes matched the saved Mac
records exactly: 20 on the live engine and 8 on the SIMD vector61 engine. Raw network
Q-values are **not** bit-identical across the two machines. The differences are at the
ULP level, around 5e-5 absolute, and they flipped no decision in any of these episodes.

Artifacts (write-once): `snake-dqn-artifacts/runpod-xcheck-20261003/run-v1/`.
- `comparison.json`: the per-episode verdicts and diffs.
- `pod/`: the raw pod outputs and `env.json`.
- `mac_rerun/`: the Mac Q reference and Mac reruns.
- `harness/`: the scripts used.
- `receipt.json`: the cost record.

## Setup

| | Mac | RunPod pod |
|---|---|---|
| CPU | Apple M5 Pro (arm64) | AMD EPYC 7713, cpu3c, 2 vCPU, secure cloud |
| OS / Python | macOS, 3.12.3 | Linux 6.17 glibc 2.41, `python:3.12-slim` (3.12.15) |
| torch | 2.9.1 (Accelerate BLAS, no MKL-DNN) | 2.9.1+cpu (MKL 2024.2, oneDNN 3.7.1, AVX2) |
| numpy | 1.26.4 | 1.26.4 |
| Pinned extras | pydantic 2.12.5, PyYAML 6.0.3, psutil 5.9.8, openskill 6.2.0 | same pins |
| Threads per episode | 2 (as in the screens) | 2, one episode at a time |

- **Code:** `git archive a0cd04b` (tracked files only) plus the champion checkpoint (sha256
  `43d4e2c5…ac93`, verified on the pod).
- **Config and profile:** pinned `deployment.yaml` (sha256 `4146baa3…`) and the
  `promotion-v2-watch-rect` profile, horizon 5000, with its digest checked.

## Episodes

Only the **scripted** mix was replayable. Its rosters are the champion plus scripted
anchors. The frozen and mixed rosters need three other checkpoints, which were not approved
for upload.

| Target (saved Mac records) | Veto | Live | SIMD |
|---|---|---|---|
| v7 screen arm B | v7 λ=4 | 8/8 identical | 4/4 identical |
| v8 screen arm B | v8 λ=8 (ref λ=4) | 8/8 identical | 4/4 identical |
| v8 screen arm A | v7 λ=4 | 4/4 identical | — |
| **Total** | | **20/20** | **8/8** |

Worlds were taken in `world_index` order: the first 8 (or 4) of the scripted mix.

Comparison rules were the same as `research/simd_parity_v7v8_h5000_20261003/h5000_check.py`:
- **Live:** the whole `record` (canonical JSON, type-strict) matched, as did `veto_diagnostics`
  with every key containing `seconds` removed, and the arm, roster, hero and method fields.
- **SIMD:** the whole pod SIMD record (timing stripped) matched the saved Mac SIMD record from
  `simd-v7v8-parity-20261003/run-v1`. Its SIMD-only keys removed, it also matched the pod's
  own live record for the same world (8/8).
- **Frame-level:** for one world (v7-B, seed 2982303806), a per-frame digest of the hero's
  selection state and mask matched the Mac rerun on all 5000 frames, so no frame differs.

The outcomes covered a useful range: hero deaths in 15/20 live episodes, and mass integrals
from 43 to 522.

### Q-values on fixed inputs

The test used 256 inputs drawn from `default_rng(20261003)`, uniform on [0,1) with 61
features, plus an all-zeros row and an all-ones row.

| Forward | Bit-identical | Elements differing | Max abs diff | Max rel diff | Max ULP | Argmax flips |
|---|---|---|---|---|---|---|
| Batch-1 (`InferenceAgent.q_values`) | no | 1334/1536 | 4.8e-5 | 2.0e-4 | 2176 | 0/256 |
| Batched (256 rows) | no | 1408/1536 | 8.4e-5 | 2.9e-4 | 3072 | 0/256 |

The BLAS libraries differ (Accelerate vs MKL/oneDNN), so floating-point reduction order
differs. Episodes are still identical because no decision in these 28 episodes was close
enough to a tie for about 5e-5 to change it:
- the action argmax;
- the veto's λ-weighted space preference;
- the v8 head-risk re-rank.

This is **empirical, not guaranteed.** Over thousands of episodes a near-tie could flip a
decision, and that episode would then diverge.

## Speed and cost

- **Live episodes:** the pod ran 1813 s in total against 508 s for the Mac records, a median
  of 3.66× slower per episode (range 2.96–3.99×). This matches the earlier probe (about 3.6×
  slower per core).
- **SIMD episodes:** these ran one world per call on the pod, 45–93 s each. That is not
  comparable with the Mac's 4-world batches, which took 50–69 s per batch.
- **Live throughput per dollar:** cpu3c costs $0.03/vCPU-hr. One 2-thread episode takes about
  91 s on the pod, so it costs about $0.0015. A 32-vCPU pod ($0.96/hr) would run 16 episodes
  at once, about 630 episodes/hr, or roughly 660 episodes per dollar.
- **Capacity:** cpu3c had no capacity at 32, 16, 8 or 4 vCPU (9 create attempts were refused
  and no pod was created). Only the 2-vCPU size started, so the run was serial.
- **Cost:** the pod ran 43.4 min (16:54:51–17:38:12 UTC) at $0.06/hr. **Total cost: $0.047**
  (balance $49.9378 → $49.8912). `GET /pods` was empty at the end.
- **Safety nets:**
  - An 80-minute local watchdog was armed and was stopped after the delete.
  - The channel was up at +27 s.
  - Results were pulled every 2.5 min.
  - All 52 pod files were downloaded before the delete.

## Recommendation

- **Tier-1 screens and sweeps: OK to run on RunPod cpu3c**, live or SIMD, with this setup:
  - torch 2.9.1+cpu, numpy 1.26.4, Python 3.12;
  - 2 torch threads per episode;
  - the same pinned config.

  A screen's records should be treated as a self-contained x86 run. Its paired arms and
  determinism controls must run on the same platform, and it records its platform in the
  intent.
- **Strict gates (Tier-2): not yet.** Q-values differ at about 5e-5, so bit-equality with
  Mac-recorded evidence is not guaranteed at scale. Two paths are open:
  - run a whole gate, every arm and control, on one platform, and name the platform in the
    pre-registration;
  - first run a larger cross-check covering all mixes, which needs approval to upload the 3
    pool checkpoints, over about 200 or more worlds, and confirm 0 divergences.
- **Capacity:** check it before planning large runs. cpu3c at 32 vCPU was unavailable today.
  The cpu5c flavor previously failed to start.
