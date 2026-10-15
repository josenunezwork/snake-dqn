# Throughput, RunPod 4x, and T4 shape (advice, 2026-10-15)

Advisory note from the cloud session, fourth request. Not gate evidence. Sources: the quiet-Mac
and CUDA benchmarks (`research/perf_quiet_20261013/RESULTS.md`), the M6 pod plan notes, the T4 pod
dry run, the T4 tooling doc, the trainer and learner. Rates are the repo's own measurements.

## 1. More transitions/s on the Mac

**Diagnosis.** A lane is learner-bound on the one MPS device, and both lanes share it. Quiet
Mac: one lane at 128x2 with batch-scale 2 does 5,364/s with the learner 90% busy, and more
actors or worlds change nothing. Two contended lanes at 3.1k each sum to 6.2k, 1.15x one quiet
lane: the second lane adds almost nothing. The probe watcher costs 2 of 18 cores and only
matters if it touches MPS; the batched MPS probe measured 1.17x because it competes with the
learners for the same device. CPU is not the constraint (8 of 18 cores busy).

**Levers, in order.**

1. **Update cadence for fine-tunes, expected 1.7-1.9x per lane.** The learner updates once per
   128 ingested transitions at 384 agent rows (replay ratio 3; K2 doubles both sides). A
   fine-tune from a converged B8 does not need every transition replayed three times. Add a
   default-off `--update-every M`: one update per M x 128 x K transitions at the same batch.
   M=2 halves updates per transition; actors have 4x headroom (one numba actor at 128 worlds
   supplies 10k/s). It changes dynamics like K2 did, so run the same one-seed paired A/B before
   adopting. Expected: ~9-10k/s on a quiet lane, ~5.5k each on two contended lanes, 8M in 24 min.
2. **Shorter fine-tunes if the data allow.** T3 writes 2M, 4M and 6M checkpoints. Probe them on
   the 32-world dev bank now, at zero training cost, for the effect-versus-length curve. If 4M
   carries 80% of the 8M effect, T4 cycles run at 4M.
3. **Probes on CPU, in parallel, off MPS.** While lanes train, run rowwise probes as 6-8
   single-thread processes on the idle cores instead of 2, and keep MPS for the learners. That
   is a 3-4x probe speed-up here; the batched MPS path is for a quiet GPU only.
4. **Skip:** batch-scale 4 (+5% over K2), bf16 (no tier on CUDA, untested on MPS), compile on
   MPS (wrong gradients in torch 2.9.1), a third lane (same GPU).

## 2. Clearing 4x on RunPod cheaply

**Why one pod fails.** Pods are actor-bound: EPYC cores run the actor 2.4x slower than the Mac,
so a run makes ~4.1k/s with 3 actors however fast the learner is. Five concurrent runs on one
4090 give ~20k/s. The step is 15 runs, 5.4 h on the Mac at 6.2k/s. One pod needs three waves,
about 100 min plus overhead: 2.5-3x. Compile cannot help an actor-bound run, and its CUDA-graph
pools (4.8 GB/run) caused the M6 OOMs.

**Plan that clears 4x: three pods in parallel, five runs each, one arm per pod.**

| Item | Value |
|---|---|
| Per pod | 1 x 4090, >= 32 vCPU, >= 64 GB RAM, devel image, eager learner, no compile |
| Per run | `--actors 3 --worlds 128 --learner-fast --obs-pack --batch-scale 2`; 4.3 vCPU, 2.7 GB VRAM |
| Per pod aggregate | ~20k/s (5 x 4.1k); M6 measured 24.6k with 7 runs on one card |
| Wall per pod | 32.5 min compute + 10-20 min setup and transfer |
| Step wall | ~50 min for all 15 runs vs 5.4 h on the Mac: 6-7x |
| Cost | 3 x ~0.8 h x $0.95, about $2.3; two-hour lifetime cap $5.7 |

The 4x rule is step wall-clock, so record it as three pods against two Mac lanes. Keep the
30k-transition preflight on every pod. A pod that fails capacity or preflight sends its five
runs back to the Mac without touching the other two; two pods alone are still about 4x. With a
48-64 vCPU host in stock, two pods of 7 eager runs (19 GB VRAM) also clear 4x. Do not use
CPU-only actors (1.7k/s on the Mac, worse on EPYC). Four actors per run lifts a run to ~5.4k
only when 5.3 vCPU per run is spare, and five such runs approach the eager learner ceiling.

The dry run's 2.5-3.97x compared one pod with one 27-43 min Mac lane. The same arithmetic with
three pods against the measured 3.1k contended lane rate gives the numbers above.

## 3. T4 shape after T3 reports

**Recipe.** Whichever T3 arm meets the rule, with bank v2 (prefer-nonboost) replacing bank v1 for
every T4 cycle. If no arm is GO but LABELS_KEEPV8_LO shows positive scripted survival with no
mass drop beyond -30, cycle 1 is that arm with bank v2, the placeholder already in the pod plan.
If only the lambda-1.0 arm moved survival and both lost mass, do not iterate: the labels are
teaching caution, and the next lever is return regression on the recorded branch outcomes.

**Cycle.** Collect a death-window bank from each of the five students' own deaths (about 2
Mac-hours per 10k decisive rows; 2k per student pooled is enough), relabel prefer-nonboost,
union with the previous bank capped near 20k rows with older rows at weight 0.5, fine-tune from
the student rather than from B8, probe on a fresh round. Record the bank lineage each cycle.

**Length.** 4M per cycle if the T3 checkpoint probes show 80% of the 8M effect by 4M, else 8M.
`bank_agree` reached 0.92 by 200k in the smoke, so the anchor is learned early; the transitions
are for TD to re-settle around the new argmax.

**Cycles.** Two, then decide. A third only if cycle 2 added at least +0.03 pooled scripted
episode survival over cycle 1 with no mass loss beyond -30. Stop at any cycle whose dev screen
passes Tier 1 (c2 >= -0.03 vs the served agent, per-seed floor, 0.50 floor) and sits within one
SE (0.032) of Tier 2.

**20M retrain** at that point, not before: from scratch with the winning recipe and the final
cycle's bank, five seeds, then fresh-bank Phase R under the amended guard. About 1 h per seed on
a quiet Mac lane, so 3 h on two lanes; one pod wave would be under 4x and stays on the Mac unless
waived. Rough calendar from T3's report: cycle 1 about 5 h, cycle 2 about 5 h, retrain 3 h.
