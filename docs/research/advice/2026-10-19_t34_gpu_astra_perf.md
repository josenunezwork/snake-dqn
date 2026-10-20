# T34 hardware: L4 teacher, conditional A100 fits, CPU serving

2026-10-19 · simulator performance advice · source `7e64b04f` (main).

**Yes: a better GPU could shorten cached, GPU-bound student fits. Start the shadow teacher on L4; optimize host work and reuse first. Keep Path B on the Mac CPU.** This follows the approved [shadow design](../final_run_20261019/DESIGN_V2_SHADOW.md); advice only, no allocation or budget amendment.

Shadow reuse removes repeated verification but still executes sequential simulation, observations, opponent features/vetoes and CNN forwards. Batch independent worlds and compact active branches into bounded inference buckets; overlap host preparation with GPU work across independent batches. A larger GPU cannot parallelize successive frames of one trajectory. Sparse doom events may actually reduce batch occupancy. If GPU work occupies 20% of the critical path, a 4× faster GPU yields only **1.18× overall**, with unchanged host work; this is an illustration, not a measured shadow profile. The older [T33 profile](2026-10-19_t34_speed_astra_perf.md) used N4/H32/MPS and excluded serving-veto opponents.

Owner-supplied rates supersede older pricing. [T16](../t16_colab_smoke02_results_2026-10-17.md) and `~/.claude/skills/colab/SKILL.md` establish:

| Runtime | Units/hour | Documented host capacity | Whole-job speedup needed to beat L4 cost |
|---|---:|---|---:|
| L4 | 2 planned; ~1.7 observed | 12 vCPU | 1× |
| A100-40GB | 5.4 | 12 vCPU / 6 physical cores | >2.70× |
| A100-80GB | 7.5 | 12 vCPU | >3.75× |
| H100 | Unknown; >7.5, scarce | Unknown | >rate/2 |
| G4, RTX PRO 6000 | 8.7 | Unknown | >4.35× |

System RAM is unverified in the inspected notes; G4's ~96 GB is **VRAM**, not host RAM. It needs cu128. Do not assume every premium GPU buys six cores, or more cores. Inspect affinity, physical topology, available RAM and disk at admission. [Colab guarantees no fixed hardware allocation](https://research.google.com/colaboratory/faq.html#resource-limits). At L4's actual 1.7 rate, premium break-even ratios rise another 18%.

**Teacher economics:** `units/accepted-world = rate × total billed hours / accepted worlds`. Illustratively, 800 worlds at 30k executed steps/world and 5k steps/s require 1.33 lane-hours. Including 15 minutes setup/export and reserving 20% of the envelope gives **4.0 units, 0.005/world** on L4. At M2's 80k-step ceiling and 3k/s, that becomes **15.5 units, 0.0194/world**. These are conditional envelopes, not forecasts; count refills, all LAD expansions and rejected exports. Parallel lanes shorten elapsed time, not aggregate units. Price recovered interventions too: bisection's escape predicate need not be monotone, so M2's intervention-recovery check matters.

**Fits are the stronger GPU case.** [FINAL_PLAN §5](../final_run_20261019/FINAL_PLAN.md) regenerates observations each epoch; the [fit loader at `31ecbe07`](https://github.com/josenunezwork/snake-dqn/blob/31ecbe076b21d1620bbdbebab681896ecc718dcb/research/t34_exit_b8_fit/data.py#L157-L228) does this synchronously. Faster convolution cannot remove that CPU wait. Propose a hash-validated, lossless RAM/local-SSD cache and bounded pinned prefetch first; these are absent today. Preserve sample order, precision and fit recipe. From the [tensor shapes](https://github.com/josenunezwork/snake-dqn/blob/31ecbe076b21d1620bbdbebab681896ecc718dcb/research/t34_exit_b8_fit/interface.py#L99-L106), one million uint8-raster rows occupy approximately **12.4 GB for S1b**, **66 GB with S1a's tower**, before metadata; 96 GB VRAM is unnecessary merely to stage batches. Then compare identical complete fits on L4 and A100-40. Illustratively, an all-inclusive four-hour L4 fit costs 8 units: one-hour A100-40 costs 5.4; two-hour costs 10.8. A100-80/G4/H100 need measured capacity or deadline benefits. Online-RL update rates do not price these fits.

**Measure first:** M1–M3, then matched A5 worlds: work/world, accepted worlds/hour, CPU stages, inference/transfer/queue time, batch occupancy, RSS; fits add regeneration, training, validation/probe time and peak VRAM. Preserve fp32 and verify action/state/RNG identity when changing devices or batch shapes ([PyTorch caveat](https://docs.pytorch.org/docs/main/notes/numerical_accuracy.html)).

Mac CPU/BNNS is the Path B candidate; remote GPUs cannot remove its tick latency. For host-heavy work, compare CPU High-RAM against L4: [0.5 units/hour ceiling](../../colab-cli-pricing-notes-2026-10-18.md), cores/RAM unknown, cheaper only above one-quarter L4 throughput. RunPod needs measured ≥4× Mac speed including transfers. Use approved `colab_cli` caps/receipts, release promptly; retain Mac-only strict qualification.
