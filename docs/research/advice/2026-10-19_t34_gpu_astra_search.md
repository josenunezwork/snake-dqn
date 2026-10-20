# Better hardware for T34: L4 teachers first; larger GPUs conditional

2026-10-19 · game-AI search advice · source snapshot `aa4f1124ba04`, including approved shadow v2

**Yes, especially for batched student fits; the shadow teacher primarily needs efficient host cores and enough independent worlds to feed its GPU. Keep Path B on Mac CPU. Select hardware by units per completed, validated result.** Advice only: no launch, registration, budget amendment or hardware substitution is authorized here.

Under shadow reuse, each trajectory still has serial state→policy→state dependencies. More GPU cannot parallelize its future timesteps. Batch **independent worlds and branches** through one inference service; host workers retain simulator, observations, opponent carry and v8 work. Avoid oversubscribing physical cores. Larger host RAM can hold snapshots and observation caches; GPU VRAM cannot accelerate Numba. A CPU-heavy runtime could therefore beat a premium GPU runtime. The [earlier profiles](2026-10-19_t34_speed_astra_search.md) are different workloads, including missing opponent vetoes; their GPU fractions are not shadow measurements.

Rates below are **owner-supplied planning inputs**, superseding older advice:

| Runtime | Units/hour | Whole-job speedup over L4 needed to save units |
|---|---:|---:|
| L4 | 2.0 (observed estimate ~1.7) | baseline |
| A100-40GB | 5.4 | >2.70× |
| A100-80GB | 7.5 | >3.75× |
| G4, RTX PRO 6000, ~96GB | 8.7 | >4.35× |
| H100 | unknown, >7.5 | >rate/2 |

Equal overhead assumed; include setup/export in actual comparisons. Using 1.7 for L4 raises these thresholds. `~/.claude/skills/colab/{SKILL.md,*.md}` records L4's 12 vCPUs; [T16](../t16_colab_smoke02_results_2026-10-17.md) records both A100 sizes at 12, with six physical cores on A100-40. H100/G4 vCPU counts and host-RAM capacities remain unverified; CPU High-RAM specifications also remain unknown. VRAM sizes above are not host RAM. Colab [resources vary](https://research.google.com/colaboratory/faq.html#resource-limits).

**Teacher economics.** Illustratively, 800 accepted worlds × 40k active steps/world at 4k steps/s takes 2.22 runtime-hours: **4.44 L4 units, 0.0056/world**. These are conditional assumptions, not measured shadow throughput. An A100-40 delivering 2× costs 6.0 units; G4 delivering 3× costs 6.44. Count rejected worlds, re-forks and all required branch outcomes. Bisection is a search heuristic: rescue availability need not be monotone, so certifying the last avoidable decision needs additional coverage. Approved sparse labels still need [shadow v2's M2 coverage check](../final_run_20261019/DESIGN_V2_SHADOW.md). “Free” overlapping confirmations do not create independent decisive examples.

CPU-only deserves comparison: [CLI acceptance](../../colab-cli.md) observed 0.08 u/h, but retains a **0.5-u/h ceiling**, not a High-RAM quote. At that ceiling it beats 2-u/h L4 if complete-world throughput exceeds one-quarter of L4's, including CPU CNN inference. Use measured core performance and available RAM, not vCPU count alone.

**Fits.** The [plan](../final_run_20261019/FINAL_PLAN.md) uses batch 1,024, 4–6 epochs and observation regeneration every epoch. Profile replay/decode, host-to-device copies and optimizer work separately; cache reusable observations in RAM/local scratch where feasible, prefetch and overlap. Keep precision, batches, data order and objectives fixed. Try A100-40 first only after the loader feeds L4 continuously. Illustratively, a four-hour L4 fit costs 8 units; a genuine 3× whole-fit acceleration costs 7.2 on A100-40 before overhead. Larger VRAM helps only if measured activation/cache demand needs it. H100/G4 follow only with demonstrated cost or deadline benefit; G4 needs cu128, others cu126. Defer TPU/GPU-simulator ports.

**Measure in order:** shadow/action identity across intended GPU batch shapes and decisive-label coverage → representative long-body, real-veto worlds → complete fits. Record useful worlds/hour, steps/world, interventions/death, identity drops, physical-core occupancy, RAM, batch sizes, GPU time and loader waits. First use the existing A5 L4 allowance; premium comparisons require a priced amendment. Add setup/export plus 20% reserve, first cap wins, through the CLI supervisor; preserve the three-assignment cap.

For Path B, measure Mac CPU catch-up/expansion tails and on-time rescues under the [contract](../final_run_20261019/SUCCESS_CONTRACT.md), with state/RNG-bound scheduled actions and bare fallback. Remote GPUs add latency; faster local cores/native kernels matter more. Strict/serving qualification stays Mac-only. RunPod remains deferred unless measured ≥4× Mac end-to-end and separately authorized. Prepaid units do not make idle premium hardware free.
