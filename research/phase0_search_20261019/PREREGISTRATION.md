# Phase 0 search-teacher premise test

**Status: APPROVED_PHASE0_20261019.** Approved by Claude (planner) 2026-10-19 ~13:05 PT under the owner's chat approval ("APPROVE", ~12:30 PT) of the search-teacher path, before any selection data. Protocol and settings exactly as registered below (N=8, H=128, eps .02, margin threshold 0, length gate 600, leaf scale 1); resource envelope: Mac only, six one-thread CPU workers (batched CPU teacher), 432,000 s wall cap, first cap wins; no Colab/RunPod for this attempt.

Original status: **DRAFT_PENDING_OWNER_REVIEW**. Implementation and disjoint local CPU
preflight only are authorized. Selection gameplay requires Claude's separate approval
of the final hash-bound plan. No training, allocation, dispatch, serving or release
is authorized by this document. This is development evidence, never a strict gate.

## Question and prospective protocol

Does exact-simulator search over bare B8 improve terminal-hero H5000 full-horizon
mass integral enough to justify distillation? Comparator is bare B8 seed 3,
ego2s-bv8, without a veto. The input checkpoint is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth`,
expected **whole-file** SHA-256
`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
The implementation must verify this pin rather than infer identity from the filename.

Fresh allocation uses namespace `phase0-search-teacher-20261019/v1`, whole-round
rejection on any collision, and separate selection/calibration/timing/parity/smoke
purposes. Selection is 96 scripted + 32 frozen + 32 mixed worlds, strict-design
ordered rosters, the existing promotion-v2-watch-rect profile, GridBatchSim with
terminal slot-0 hero and ordinary opponent respawn. Both arms of each world run
on the same worker and CPU model. Dead frames contribute zero; denominator is
always 5000. No world/seed/checkpoint replacement, extension, or interim efficacy
read is allowed. Operational progress contains identities and counts only.

## Search and preflight selection (before selection data)

The prospective pilot chooses between N=8 or 16, H=128 or 256, epsilon=0.02,
using disjoint timing fixtures and calibration worlds only. Start with N=8/H=128;
increase only if the measured cost remains practical. Search at every decision
with length >=600, or a legal-mask B8 best-minus-second-best Q margin below
a threshold fitted on disjoint bare-B8 calibration decisions. Target approximately
10% total searched decisions; if the mandatory length trigger alone exceeds 10%,
report that floor and disable additional margin admissions instead of hiding it.
Ties use the lowest action index. Each legal first action receives N independent
continuations keyed by world, decision, first action, and rollout. The realized
post-transition hero mass is summed for H steps (zero after death), then a scaled
masked-max B8 Q leaf is added only while alive. The final scale and all selected
parameters will be recorded here before approval and selection gameplay.

Rollout opponents use their actual strict-roster policies; scripted anchors are
exact and frozen-network forwards are batched across replica rows. Outer B8 uses
rowwise CPU advisory-mask inference, including the strict fallback. Batch-forward
numerics are part of the search recipe, not a bit-identity claim for network Q.
Simulator stepping and snapshot state must match the strict implementation.

## Endpoint and one-look decisions

For each world, D = search mass integral - bare B8 mass integral. Report the mean
and survival (no-death fraction) per mix. The pooled effect is the **equal-weight
mean of the three mix means**, not the 96/32/32 raw-world average. With one fixed
B8 network, worlds within mix are the independent paired units: pooled variance
is sum(s_mix^2/n_mix)/9, with Welch-Satterthwaite degrees of freedom rounded
down to an integer for the existing Student-t quantile helper (conservative). LB90 and UB90
are one-sided 90% Student-t bounds. They are not across-training-seed estimates.

At the single complete-bank look, apply in order:

1. INVALID/INCOMPLETE for missing, duplicate, nonfinite, identity-mismatched,
   nonconforming or unfinished evidence; this is not a scientific result.
2. STOP_TRAINING if UB90 pooled mass effect < +40 **or** search scripted
   no-death survival < bare B8 scripted no-death survival +0.05.
3. GO if search scripted no-death survival >=0.66 **and** pooled paired mass
   effect >=+80 **and** LB90 >+40.
4. INCONCLUSIVE otherwise. No extra look or automatic retry.

STOP takes precedence if its survival condition overlaps a mass-based GO.
A GO authorizes planning only, never distillation execution or a serving change.

## Execution, evidence, and caps

Mac CPU only for the implementer's timing/tests/smoke, one math thread per
process, maximum six concurrent workers. Each preflight invocation stays under a
few minutes. Selection is not launched by the implementer. The future runner
must enforce AC, open lid and thermal admission on macOS, bound wall time,
create-only plan and per-world partials, validate identity on resume, and merge
in bank order identically to serial. Both paired arms are committed as one
resumable unit. A cap stop is INCOMPLETE, never a scientific verdict.

Linux CPU portability is for later owner operation only. No pod is allocated
here. A future CPU-pod plan must first replay reserved parity worlds against a
Mac reference, with same-state Q/mask/action traces and <=4-ULP tie diagnostics;
unexplained divergence fails closed. Rates/prices and the >=4x placement rule
must be revalidated before any spend. Strict gates and serving remain on Mac.

Artifacts live outside Git under `snake-dqn-artifacts/phase0-search-20261019/`;
small preflight receipts and allocation/audit may be committed. Bind source
closure, checkpoint bytes, rosters, profile, config, bank, audit and this draft
in the final plan. Source/spec bytes already bound by prior studies stay intact.

## Final preflight selection and measured budget

Selected before any selection data: **N=8, H=128, epsilon=0.02, margin
threshold=0, length gate=600, leaf scale=1 mass-frame per Q unit**. The score
uses max over legal leaf actions at the next prepared Watch decision. The
smallest suggested N/H already costs 4.1-5.4 s per forced search, so no larger
recipe was piloted. Outer B8 is rowwise; rollout hero remains rowwise and frozen
opponents are batched. No veto is applied.

Disjoint calibration: three worlds, 11,446 alive decisions, 20.37 s. The mandatory
L>=600 trigger alone covered 43.395%; the ~10% total search target is infeasible
with that OR rule on these fixtures. Threshold zero disables additional margin
admissions. This feasibility limit is disclosed before approval, without changing
the required length trigger or inspecting any selection outcome.

`PREFLIGHT.json` binds raw timing/calibration/scaling receipts and forecast
assumptions. The central forecast is 337.34 serial-equivalent core-hours, 88.85 h
on six Mac workers (533.13 aggregate worker-hours), or 106.63 h with 20% reserve.
Proposed owner-review Mac limits: six one-thread workers, 120 wall hours per
attempt (432,000 s), first applicable cap wins. An all-decisions-searched scenario
is 1,201 serial hours, so completion is not guaranteed by this proposed cap.
A stopped/incomplete bank is INVALID_INCOMPLETE, not a scientific conclusion.

A hypothetical 32-vCPU scenario at historical 2.4x per-core slowdown gives 25.30 h
and $22.52-$24.04 at the dated $0.89-$0.95/h planning rates, before setup/reserve.
Its 3.51x projected speedup does not establish >=4x placement admission. Current
price, CPU throughput and cross-platform replay remain unverified. No CPU-pod
execution or spending envelope is ratified by this draft. See README for exact
future commands and scope boundaries. No selection outcomes exist from this work.
