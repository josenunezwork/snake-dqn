# A100 advice round (2026-10-20 PT): folder index

The question: would Colab A100 runtimes get a better served snake sooner than the L4 plan? The folder also holds one
related ruling on label ordering.

## Verdict
1. **No A100 this campaign, and no A100 placement test either.** All 10 Astra answers, 3 Fable reviewers and 3 Fable
   GPU experts agreed.
2. The teacher and Round 0 are bound by the host CPU, and A100 and L4 hosts both have 12 vCPUs. The GPU is at most a
   ~1.15x lever, while an A100 costs ~3.2x the units of an L4.
3. Reopen only if the first phase-timed L4 Claim A fit takes ≥ 60 min with the GPU leg ≥ 40%. Then run one matched
   L4 vs A100-40 packet (cap 12 u). Adopt at ≥ 1.25x and ≥ 1 h saved.

## Read in this order
| # | File | What it is |
|---|---|---|
| 1 | [SYNTHESIS.md](SYNTHESIS.md) | Claude's synthesis: verdict, the levers that matter more (Claim A pipeline, Path B pacing, §5 ordering, NO-GO plan), corrections to the brief |
| 2 | [FACTS_AND_QUESTION.md](FACTS_AND_QUESTION.md) | The facts brief and the open question that every advisor received |
| 3 | [REVIEW_A_evidence.md](REVIEW_A_evidence.md) | Independent Fable review A: audit of every cited file:line in the advice |
| 4 | [REVIEW_B_decision.md](REVIEW_B_decision.md) | Independent Fable review B: the decision itself |
| 5 | [REVIEW_C_adversarial.md](REVIEW_C_adversarial.md) | Independent Fable review C: what all ten Astra answers missed |
| 6 | [astra_p1_systems.md](astra_p1_systems.md) … [astra_p5_colab_ops.md](astra_p5_colab_ops.md) | 5 Astra lenses × 2 calls each, with cross-examination: P1 systems, [P2 GPU search](astra_p2_gpu_search.md), [P3 distillation](astra_p3_distill.md), [P4 critical path](astra_p4_critical_path.md), P5 Colab ops |
| 7 | [GPU_EXPERT_PROMPTS.md](GPU_EXPERT_PROMPTS.md) | Round 2: the Fable-written prompts for an open-ended GPU rethink |
| 8 | [gpu_expert_1.md](gpu_expert_1.md), [gpu_expert_2.md](gpu_expert_2.md), [gpu_expert_3.md](gpu_expert_3.md) | Round 2 answers from 3 independent Fable GPU experts (simulator, search algorithm, learning/network). They agree there is no A100 and no GPU search port this campaign |
| 9 | [RULING_label_ordering.md](RULING_label_ordering.md) | A separate Fable judge ruling: R1 label collection may follow a published Round 0 GO, in parallel with Path B. It became the §5 amendment in `d5b11e3a` |

## Related
The no-A100 decision is recorded in `docs/research/final_run_20261019/HANDOFF_2026-10-20.md` (update ~18:30 PT). The
NO-GO branch plan is `docs/research/specs/2026-10-20_t34_nogo_branch_plan.md`.
