# Better GPU? Where hardware changes the shadow-design teacher and fits (Fable, 2026-10-19 ~21:30 PT)

Advice only; no new probes. Sources: colab SKILL.md GPU table; t16 results (L4 and A100-80GB both
**12 vCPU / 6 physical cores**; H100 and G4 vCPU counts **unknown**, recorded nowhere in the skill or
colab_cli notes); FINAL_PLAN §3-6; my first-answer probes.

## 1. Where the work is

Per world under the shadow design: ~5,000 shadow frames plus 2-20 expansions of <=1,500 row-steps =
**~10-35k row-steps**. Each costs host work (sim, hero raster, five opponents' V61 + v8 + MLP) of ~0.25-0.65
thread-ms on a Mac core, plus one B8 forward: 0.38 ms on a CPU thread, **~1 us on an L4**. The shadow is a
serial chain per world, so the GPU helps only when 100-300 root worlds advance in lockstep and B8 batches
across them — which T33/P1 already does. With B8 off the CPU every runtime is **host-bound**: six physical
cores at 2.4-3.7x a Mac core. An A100/H100/G4 buys the same six cores plus a GPU that is >95% idle.

Consequence: an L4 lane does ~8-15k row-steps/s host-bound, i.e. **~1,000-3,000 worlds/h**; an 800-world
round is under an hour and ~1-2 u, not 60 u. The Mac does 800 worlds in ~1-3 h on three threads (CPU B8
included) at zero units. Colab is needed only while the Mac is busy, and then an L4 is the right GPU; an
A100-80 costs 3.75x more per world at the same wall time.

## 2. Fits are not GPU-bound either

A fit is 1 M rows x 5 epochs x ~70 MFLOP (fwd+bwd) ≈ 350 TFLOP: **~1-2 min of L4 time**. The plan's cost is
the streaming replay regeneration (host, <=30 min/epoch, five times per fit). A bigger GPU does nothing for
it; two things do:

- **Cache observations once per fit.** Without the S1a tower a row is 12.4 KB: 1 M rows = **12.4 GB**, fits
  the runtime disk and most of its RAM. The tower's 53 KB planes make it 65 GB: regenerate only those for
  S1a, or store them 2:1 downsampled (they are stride-2 convolved anyway). Regen runs once (~15 min); epochs
  take minutes; a fit is ~0.5-1 u.
- **More vCPUs**, not more GPU, if regen must stay streaming — the one reason to learn H100/G4 nproc.

## 3. Units per useful result

| Item | Plan (L4) | L4, shadow | A100-80, shadow | Mac |
|---|---:|---:|---:|---:|
| 800-world label round | 60 u / ~10 h | **1-3 u / <1 h** | 5-10 u, same wall | 0 u / 1-3 h |
| One fit (S1b, 1 M rows) | 8 u | **0.5-1 u** (cached obs) | 2-4 u, no faster | — |
| Round 0 (240 worlds) | 16 u | ~1 u | ~3 u | 0 u / ~1 h |

The R0-R3 cap drops from ~270 u to **~15-25 u**; the binding resource becomes Sol's prep time and label and
distillation design, not compute.

## 4. Where a different GPU would matter — nothing funded

Only a self-play/PPO line (thousands of envs through B8 every frame) or a CUDA port of the world step and
featurizers (2-4 weeks) would make GPU class matter. Path B on the Mac: MPS is slower than CPU at batch 1
(~1 ms launch); the one off-CPU option worth a 10-minute probe is **CoreML/ANE** for batch-1 B8
(~0.2-0.4 ms, zero CPU) — not needed, the shadow costs ~1.5-2.5 ms per tick. G4 (cu128, 8.7 u/h, same host):
never for this.

## 5. Measure first

1. **Seconds per world** (shadow + expansions, mixed roster) on one Mac thread and on one L4 lane, W=256
   lockstep (~0.3 u): decides Mac-only vs L4; chunk sizes in worlds/h.
2. **Regen seconds per world and bytes per row** (S1b, S1a): decides cache-once vs streaming.
3. **GPU % during an L4 fit** (expect <30%).
4. Only if 1-2 show the host short: `nproc`/RAM on an H100 or G4 at attach (~0.5 u); keep
   `allowed_gpus=[L4]` until then.

Recommendation: stay L4-only, re-derive chunk sizes from worlds/h, shift Colab budget from labelling to
fits with cached observations, and label Round 0 on the Mac if it is free.
