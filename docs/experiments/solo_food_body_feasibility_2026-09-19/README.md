# No exact target contradiction found for the observed body-only inputs (AX)

AX found no negative-cycle contradiction in the saved training or held-out body-visible input groups. Every group is feasible for both frozen target systems at both margin 0 and margin 0.001. This rules out one narrow explanation for AW's incomplete result: an exact, observed group of rows does not demand mutually impossible additive six-action corrections. It does not show that the residual network can represent or learn a feasible correction, that it will generalize beyond these rows, or that it will behave well on states induced by its own greedy actions.

This is a static feasibility audit, not new learning or gameplay. It reuses the three original AT cohorts and AW parent mark-zero Q arrays; it performs no checkpoint inference, optimizer update, policy action selection, collection, or checkpoint selection. The behavioral context remains [AW](/Users/josenunez/Projects/ml/snake-dqn/docs/experiments/solo_food_policy_preservation_2026-09-19/README.md), where preservation improved ordinary-action retention but failed the complete three-seed mechanism gate.

## Fixed constraint question

For each AT cohort 3701–3703, mapped to AW seeds 2026093801–2026093803, AX separately groups train, held-out, and combined rows by the exact float32 bytes of the actual body/wall residual projection:

- tactical channels 0, 6, and 7;
- strategic channel 2; and
- scalar indices 0, 1, 8, 9, 10, and 11.

The saved parent Q values remain row-specific. For every exact input group, AX asks whether a shared six-coordinate additive correction can satisfy two target systems with each row's external legal-action mask:

1. Teacher-all: the SpaceTeacher action ranks above every other legal action.
2. Preserve-plus-veto: ordinary rows preserve the parent distribution exactly across legal actions, while veto rows rank the SpaceTeacher action highest.

The ranking inequality is c[j] − c[y] ≤ baseQ[y] − baseQ[j] − margin. Ordinary preservation fixes c[j] − c[k] = 0 for each legal pair. A Bellman–Ford difference-constraint solver uses six vertices, float64 arithmetic, negative-cycle tolerance 1e-9, and verifies any feasible correction against the original constraints. The two margins, 0 and 0.001, were frozen before execution. Margin 0 allows ties, so its feasibility does not prove native greedy argmax behavior.

## Every cohort and split

A repeated group has more than one row with an exactly equal projected body-visible input. Full-observation distinctions count repeated projected groups whose original full observations differ; no group has more than one action mask. The feasibility count is the same for both target systems and both margins: all groups feasible, zero infeasible, zero affected rows, and zero affected veto rows.

| AW cohort | Split | Rows | Ordinary / veto | Exact groups | Repeated groups | Full-observation distinctions | Mask-diverse groups | Feasible groups at both targets and margins | Infeasible groups / affected rows |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 2026093801 | Train | 1,536 | 1,469 / 67 | 1,490 | 10 | 10 | 0 | 1,490 | 0 / 0 |
| 2026093801 | Held-out | 1,536 | 1,411 / 125 | 1,467 | 6 | 6 | 0 | 1,467 | 0 / 0 |
| 2026093801 | Combined | 3,072 | 2,880 / 192 | 2,957 | 16 | 16 | 0 | 2,957 | 0 / 0 |
| 2026093802 | Train | 1,536 | 1,458 / 78 | 1,516 | 6 | 6 | 0 | 1,516 | 0 / 0 |
| 2026093802 | Held-out | 1,536 | 1,481 / 55 | 1,534 | 2 | 2 | 0 | 1,534 | 0 / 0 |
| 2026093802 | Combined | 3,072 | 2,939 / 133 | 3,050 | 8 | 8 | 0 | 3,050 | 0 / 0 |
| 2026093803 | Train | 1,536 | 1,483 / 53 | 1,534 | 2 | 2 | 0 | 1,534 | 0 / 0 |
| 2026093803 | Held-out | 1,536 | 1,432 / 104 | 1,455 | 20 | 20 | 0 | 1,455 | 0 / 0 |
| 2026093803 | Combined | 3,072 | 2,915 / 157 | 2,988 | 23 | 23 | 0 | 2,988 | 0 / 0 |

The result is identical across the two predeclared targets and both margins. Each of the 17,991 split-specific group checks is feasible at margin 0 and 0.001; no infeasibility witness exists in the saved groups archive. This count includes the combined analysis of the same train and held-out rows and must not be treated as 17,991 independent observations.

Train-to-held exact projected-input overlap is only 0, 0, and 1 ordinary rows for cohorts 3801, 3802, and 3803 respectively. Thus, the audit sees nearly disjoint body-visible signatures across those historical splits. That makes it a poor test of coverage or generalization despite feasibility in both splits.

No repeated training group contains a veto row, and no group in any split mixes ordinary and veto rows. The exact-preservation constraints therefore never compete with an escape target within the same observed group. This further limits what the audit can say about AW's practical food/escape tradeoff.

## Coverage limits

The selected cohorts contain all eight worlds per split, with 192 selected rows in every world, but veto support is uneven. In the combined cohorts, veto rows by heading are 27/102/26/37, 43/28/25/37, and 61/14/26/56 for seeds 3801–3803. Veto teacher turn labels are also uneven: 24/96/72, 28/48/57, and 23/39/95 for left/straight/right.

World-level veto support ranges from 5 to 67 rows in cohort 3801, 4 to 18 in cohort 3802, and 2 to 35 in cohort 3803. Cohort 3801's held-out world 2026102505 alone contributes 67 of 192 combined veto rows. These are selected, parent-driven historical rows rather than natural prevalence. Exact grouping can also miss near-aliases and unseen states. The absence of a contradiction within these signatures is therefore not evidence that the body-only representation has enough capacity or coverage for greedy gameplay.

## Resource and integrity evidence

The one CPU audit completed naturally in 4.9273 of its 45-second science cap. It recorded peak process RSS of 441,942,016 bytes and minimum available host memory of 35,148,152,832 bytes. The source and driver remained unchanged; the audit used the existing two CPU thread/inter-op one, 4 GiB RSS, 12 GiB available-memory, and 20-second heartbeat guards. No MPS work ran.

Fourteen focused qualification tests passed in 1.794 of the 60-second qualification cap, with no failed attempt. Including qualification, peak recursive RSS was 565,084,160 bytes and minimum available memory was 35,148,152,832 bytes. The frozen source revision is 56e0e92434ae510a84ebaf6f58cf41c3e4421b04. The parent checkpoints and the six-action, external-mask action contract are preserved exactly. AX has zero optimizer updates and no promotion eligibility.

Apex remains the operational incumbent, while its substantially larger historical training budget means it is not evidence of inherent architectural superiority. AX is not an Apex comparison and cannot support replacement.

## What this resolves next

The evidence removes an exact observed-input conflict as the immediate explanation for AW's failure, but leaves coverage and learnability open. The next coverage experiment is proposed, not frozen or run: compare the existing eight-world training data with the same data plus eight fresh worlds. Both arms will use the same preservation objective, network, parent, optimizer settings, and number of updates, with three fresh residual initialization seeds. Fresh, separate fit and gameplay banks will measure escape generalization and food retention. Added worlds do not guarantee more useful escape examples; their observed support must be reported.

- [Frozen AX design](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-correction-feasibility/design.md)
- [Full feasibility report](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-correction-feasibility/analysis/report.json)
- [Per-group results and feasibility witnesses](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-correction-feasibility/analysis/groups.json)
- [Qualification completion](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-correction-feasibility/qualification-complete.json)
- [Audit receipt](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-correction-feasibility/supervisor-runs/audit/receipt.json)
- [AW objective curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/objective-learning-curve.png)
- [AW fit and preservation curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/fit-and-preservation-curve.png)
- [AW greedy-gameplay curves](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/gameplay-learning-curve.png)
- [AW representative gameplay](/Users/josenunez/Projects/ml/snake-dqn-artifacts/ongoing-research-20260913/solo-food-body-policy-preservation/analysis-r1/representative-gameplay.png)

The difference-constraint construction follows MIT's [lecture on Bellman–Ford and difference constraints](https://ocw.mit.edu/courses/6-046j-introduction-to-algorithms-sma-5503-fall-2005/resources/lecture-18-shortest-paths-ii-bellman-ford-linear-programming-difference-constraints/). The lecture supports the mathematical test; it does not establish the network-capacity or gameplay claims above.
