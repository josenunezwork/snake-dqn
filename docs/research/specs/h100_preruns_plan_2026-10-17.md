# H100 pre-runs: merged plan before the long Colab H100 run (2026-10-17)

Merged from three Astra advisor personas (gpt-6-astra via Codex CLI):
`docs/research/advice/2026-10-17_h100_preruns_{gpu,rl,league}.md`. Throughput context:
`docs/research/t16_colab_smoke02_results_2026-10-17.md`. League design:
`docs/research/specs/lane_a_league_design_2026-10-17.md`.

## Where the three agree

1. All-six self-play speed is infrastructure evidence only. The long run is a league with N=3 learner seats
   plus frozen-pool seats, which the trainer cannot do yet. That slice is the shared prerequisite.
2. Fixed recipe R0 = fp32, bs2, LR 1e-4, 256 transitions/update. bf16 (R1) and bs4+sqrt (R2, LR 2e-4, 512
   per update) are separately screened arms, never silent swaps.
3. Count learner decisions D, replay rows R, optimizer updates U; report mean and p95 policy lag.
4. RunPod H100 now for short capped packets (<= 60 min, <= $10 each); a short Colab portability check on the
   owner's runtime is mandatory before the long allocation (Colab vCPUs differ).
5. No short experiment promotes. A stronger agent still needs the fresh Mac strict gate vs B8.

## Ordered steps

| # | Step | Where | Status / cost | Decides |
|---|---|---|---|---|
| 1 | H100 learner ceiling (isolated P0 cells) + all-six smoke matrix | RunPod H100 | running (pod_bench T18, cap $8) | can an H100 reach >= 84 upd/s fp32 bs2 at all; H100 vs A100 per unit |
| 2 | Actor CUDA graphs (`--compile-actor`) x CUDA MPS daemon, a4 self-play | RunPod H100 | after 1, <= $4 | recover the isolated vs full-run gap without recipe change |
| 3 | bf16 in-run reversal diagnosis (profile learner vs actor forward; bf16 learner with fp32 actors) | RunPod H100 | after 1, <= $4 | whether bf16 earns a quality arm |
| 4 | **Code: N=1/3/6 mixed-seat mode + frozen BV8 pool dispatch + D/R/U/p95-lag counters** (league slice, no PFSP yet) | Codex, Mac | starts now | prerequisite for 5-8 |
| 5 | Mac N=3 baseline (same counters) | Mac | after 4 | the honest 4x denominator for N=3 |
| 6 | N=3 H100 throughput, a3 vs a4, with the initial frozen pool | RunPod H100 | after 4-5, <= $4 | actor count, queue depth for the long run |
| 7 | R0 learning screen: five B8-initialized 8M fine-tunes at N=3 vs static pool, Mac fixed-world evaluation vs B8 | RunPod H100 (<= $3/seed) + Mac eval | after 6; pre-registered | does the league opponent mix fine-tune B8 usefully at all |
| 8 | PFSP sampler + payoff adapter, then the short league probe (L3/L4 of the league doc) | Codex + pod + Mac | after 7 GO | adaptive league vs static pool |
| 9 | Colab portability check (<= 45 min, owner's H100) | Colab | when owner is home | final actor count; units/hour economics |
| 10 | Long Colab H100 run: five fresh 20M fine-tunes from B8 under the selected recipe/roster | Colab | after 9 | candidate for the strict Mac gate |

## Decision rules carried from the advisors (summary)

- Step 1: if H100 isolated fp32 bs2 < 84.4 upd/s, no fp32 4x claim on H100 either; consider R2 or Mac/A100
  economics. H100 must beat A100 useful work by >= 1.65x to be worth its ~1.5x unit rate.
- Steps 2-3: adopt actor graphs / MPS only if each passes its own gain and correctness checks; bf16 only if
  the reversal is explained, gains >= 20% in-run, and later passes a quality screen.
- Step 6: pick a4 over a3 only with >= 10% more N=3 throughput and lag no worse.
- Step 7: GO/KILL/EXTEND thresholds per `2026-10-17_h100_preruns_rl.md` section 6 (to be pre-registered with
  hashes before data).
