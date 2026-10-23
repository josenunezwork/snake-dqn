# A3 — actor-enabled P1 learner ceiling

Authority: owner task 2026-10-22, `PLAN_A100.md` §3 A3 and round-20 judgment,
mirrored in `docs/research/advice/2026-10-22_a100_program/`. Owner opted into
A100 work. This implements that operational assay and freezes its otherwise
unspecified measurement details before production data. It does not amend P1,
consume its sealed evaluation outputs, qualify its critic, or adopt a new recipe.

Question: can the present P1 learner possibly deliver 10,000 hero transitions/s
with collection free? Comparator: the unchanged P1 code at the pack's commit.
Run the same packet on A100-40 and L4, with actor enabled, F arm, 128 worlds ×
128 steps = 16,384 transitions. Keep fp32, P1 loss/GAE/learning rates, clipping,
two PPO epochs and KL stop. Explicit benchmark-only `critic_refit_updates=0`
activates the actor; this grants no P1 training or qualification authority.

## Frozen measurement allocation

Seven independent units, in fixed order: baseline cached learning, collection,
complete cycle; optimized cached learning, collection, complete cycle; fixed-batch
gradient comparison. Each timing unit has one excluded warm cycle and three
measured repeats. Warm CUDA before measuring, synchronize timing boundaries,
use one numerical-library thread per worker and the P1 twelve-worker collector.
Record collection, staging/H2D, combined forward/backward/Adam, actual P1 action
window/E8 diagnostic scoring, durable receipt export time, and sampled GPU busy.
Cycle totals include collection + staging + learning + scoring. Export is measured
separately once per immutable unit; this assay exports receipts, not a training
checkpoint. GPU utilization is sampled every ~0.2 s while each measured phase runs;
missing samples remain null, never zero. Do not treat busy sampling as timing proof.

Cached repeats use the identical on-policy batch already resident on the device,
identical copied actor parameters and warmed Adam state; cloning/preparation stays
outside the timer. Report median and conservative p95 (maximum of three samples).
Both actual Adam steps must run for a valid two-epoch ceiling. A legitimate KL stop
is retained as `INCOMPLETE`, never a fast learner success. The comparison captures
every parameter's actual gradient after clipping, immediately before each Adam
step; timing runs count steps without copying gradients.

The optimization is separate and local to this package. It reuses the actual
teacher logits for diagnostic B8 actions and logically flattens time/world rows
in 1,024-row forward chunks, without altering the single full-batch PPO loss or
gradient reduction. Chunking bounds forward workspace; all autograd activations
still survive until backward. An OOM is `INCOMPLETE`; never reduce the 128×128
shape. Kernel/reduction reassociation is explicitly reported. Elementwise fp32
gradient acceptance is `torch.allclose(atol=2e-5, rtol=2e-4)` for every parameter
in both epochs, on the same fixed real rollout and initial state. The original
P1 modules remain the baseline; passing this comparison does not adopt the fix.

Timing worlds use an independent assay namespace at seed 20271022, excluding the
published development bank seeds. Temperature follows the pinned P1 independent
no-step calibration, at assay seed + 104729, once per attempt; its value/rule is
exported. Both accelerators use the same source, seed and calibration algorithm.
The tiny operational workload is one world × three steps, serial, labelled tiny
and rejected by the production aggregator. It cannot contribute A3 evidence.

## Decisions and caps

First validate both complete seven-unit ledgers using path/size/hash without
opening outcomes. Claude then takes the single Mac aggregation look. Foreign
namespace/allocation, duplicate IDs, changed hashes, tiny rows, absent samples,
wrong dimensions, missing actor updates or failed execution => INVALID/INCOMPLETE,
never a pass. **Baseline cached median > 1.6384 s/rollout => PAUSE_WP5 for a learner
fix before any collector port**, even if the optimized candidate is faster.
Equality does not trigger the strictly-greater-than kill. A value at or below
the ceiling means only CEILING_NOT_FALSIFIED, never that a full collector will
reach 10 k/s. Unexplained gradient change => REJECT_OPTIMIZATION. No GPU timing is
strict/product/serving evidence. No interaction with amendment-04 or its one look.

| Allocation | Base wall/cap/reserve | Effective wall/cap/reserve |
|---|---|---|
| A100 canary | 8 min / 1.5 u / 0.3 u | 16 min / 3 u / 0.6 u |
| A100 assay | 52 min / 14.5 u / 2.9 u | 104 min / 29 u / 5.8 u |
| L4 canary | 10 min / 0.5 u / 0.1 u | 20 min / 1 u / 0.2 u |
| L4 assay | 50 min / 3.5 u / 0.7 u | 100 min / 7 u / 1.4 u |
| **Whole A3 series** | **2 h / 20 u / 4 u** | **4 h / 40 u / 8 u** |

The owner's 2× rule is applied exactly once. Caps include setup/export, failed
allocations, canaries, accounting retry and all completion attempts; first cap wins.
Expected allocation is one A100 hour plus one L4 hour, including canaries. Forecast
A100 at 3.5× L4 = 7 u/h (round-20); charge A100 at a conservative 10 u/h ceiling
(5× L4) and L4 at 2 u/h. Expected combined cost ~9 u; these are forecasts, not
observed debit. A100-40 historical rate ~5.4 u/h; record actual hardware/rate from
the broker. No H100 substitution. Account cap three, including notebook workers;
P1 Y3/source, A1 and A2 have priority over A3.

## Resume and transport

An independent completed assay cell is the export unit, identified by the source-
bound namespace and frozen allocation index. Publish payload + SHA sidecar + marker
last. Start with an empty ledger; the runtime skips exactly Mac-verified entries.
Ledger building reads names/sizes/hashes only; outcomes remain sealed until all
seven units on both targets are verified. No weights-only or trajectory-resume
claim is made. The registered aggregator rejects duplicate IDs and foreign data.

Mechanical failure (RPC stall, lost runtime/lease, transfer failure, OOM, process
failure or cap expiry) is INCOMPLETE. After verified stop and a missing-unit audit,
at most two outcome-blind completion attempts may be prepared, with fresh manifests,
output/state directories and receipts, the same allocation, and only missing units.
Each attempt's envelope is at most the original lane envelope and the remaining
series wall/units; subtract all prior canary/attempt/retry debit before preparing it.
No cap increases, N changes, automatic scientific retries or automatic completion
retries. A changed ledger requires a fresh same-input target canary and its cost is
charged to the remaining envelope. The existing exact accounting allocation retry
in dispatch.sh is unchanged and also charged. If remaining headroom cannot cover
a canary plus completion, stop INCOMPLETE and refer to the judge.

Only Claude dispatches via `tools/ops/dispatch.sh`. Both targets need a passing
same-pin, same-entrypoint, same-input real CUDA family canary, including verified
exports and backend release, before binding the main manifest. A100 canary ≤16 min/3 u;
L4 canary ≤20 min/1 u here
(also within the transport's 30-min limit). A stalled canary stays INCOMPLETE.
Current qualified supervisor already supports A100/cu126. No transport executable
changes are required; source qualification is checked against the live ops registry.
A100 platform proof is the A3 A100 canary; L4 proof cannot substitute. If the broker
tree later changes, Claude must use the documented `tools/ops/requalify.sh plan`
and reviewed `execute` process before regenerating/binding packs.
