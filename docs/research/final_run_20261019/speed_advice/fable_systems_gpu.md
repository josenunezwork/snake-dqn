# T34 follow-up — would a better GPU help? (Fable systems, 2026-10-19 ~21:30 PT)

Short answer: **no, not for the teacher and not for the fits as designed.** Under the shadow design
both are host-core and I/O bound; the GPU is a few percent busy on an L4 and would be ~1% busy on
an A100/G4. Spend units on host cores and on a cached fit bank, not on a bigger GPU.

## Where the work is

**Teacher.** Per world under the shadow design: ~5k frames of ordinary lockstep play plus 2-20
expands ≈ 10-35k row-steps (fable_systems.md). Per row-step on a Colab vCPU (2.4-3.7x slower
than a Mac core): ~250-370 us of host work (sim ~26, obs ~27, opponents' features/veto, assembly)
versus ~1-1.5 us of B8 on an L4 at batch 256-2k (22.5 MFLOP/row, fp32). Host:GPU ≈ 200:1 per
core, ~35:1 with six physical cores. One L4 lane ≈ 35k × 300 us / 6 cores ≈ **2 s per world**,
800 worlds ≈ 30 min ≈ **1 u per labelling round** (2-3 u with a 2-3x derate). The Mac's three
threads do the same in 1-3 h for free. An A100-80 or G4 buys the same CPUs (plan §42: "the same 6
cores") at 3.5-5x the rate — it makes labels **4-5x more expensive per world** and no faster.

**Fits.** The plan budgets ≤30 min × 5 epochs of *replay regeneration* (1 M rows × 65 KB fp32 does
not fit RAM) — the fit is regen-bound, not GPU-bound. GPU arithmetic is trivial: forward + backward
≈ 70 MFLOP/row → 1 M rows ≈ 70 TFLOP ≈ 1-3 min/epoch on an L4 (fp32, batch 1,024). Fix the
loader, not the GPU: regenerate **once** into uint8 planes (12.4 KB/row → **12.4 GB per 1 M
rows**) on the runtime's local disk, memory-map, read it 5 times. Regeneration needs no policy
forwards (actions come from the exported traces), only sim + obs ≈ 60-150 us/frame → 800 worlds
≈ 10-20 min on 6 cores, once. A round's fit is then ~15-30 min of L4 ≈ **1 u**; S1a's full-arena
tower (~20 MFLOP/row) only doubles the GPU minutes. An A100 would cut ~10 GPU-minutes to ~4 at
3-4x the rate: a loss.

**Campaign units.** Labels 1-3 u + fit 1-2 u + probe per round → **≈ 3-5 u/round, 15-30 u for
R0-R3**, versus the plan's 150-270. The cap can stay; most of it will not be spent.

## What a different kind of hardware *would* change

- **More host cores per unit** is the only lever: labelling and regeneration scale with core-seconds.
  If an A100/G4 runtime exposes more vCPUs than the L4's 12 (unknown — the Colab notes record 12
  vCPU for L4 only; `nproc`/`free -g` on each runtime is a 1-minute, <0.2 u probe), compare
  vCPU/(u/h): L4 = 12/1.7 ≈ 7; an A100-80 would need > 50 vCPU to match. RunPod 32-vCPU CPU pods
  are the cheapest core-seconds if the owner ever relaxes the Colab preference; the Mac is free.
- **Many fits in parallel**: one 80-96 GB GPU can host 6-8 concurrent S1a/S1b/τ-sweep fits
  (< 4 GB each) on a cached bank; units are a wash versus one L4 per fit, the gain is wall time
  and fewer sessions. Only worth it if A4's sweep grows beyond ~4 arms.
- **H100**: scarce, unmeasured, and nothing here is GPU-bound — do not queue on it.
- **GPU sim port**: the shadow design removed its justification; keep it stopped.
- **Path B** runs on the Mac CPU (MPS at batch 1 is slower than the 0.29 ms CPU forward).

## Measure first

1. One L4 lane: worlds/s of the shadow teacher at W = 256 lockstep worlds (expect ~0.5 world/s,
   GPU busy < 10%). GPU busy < 15% closes the bigger-GPU question by data.
2. Regenerate-once cost and the cached fit's GPU minutes/epoch (S1b, S1a) on an L4 with the uint8
   bank on local disk; expect < 3 min/epoch.
3. `nproc`, `free -g`, local disk on an A100-80 and a G4 runtime (< 0.2 u each) to settle
   cores-per-unit; today their vCPU count is an assumption.

Flag: §42's "GPU busy 10-25% labelling, 40-60% fits" will read ~2-5% and ~20-40% once the loader
is cached; that is correct behaviour, not a problem.
