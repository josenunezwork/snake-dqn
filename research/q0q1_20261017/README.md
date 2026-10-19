# Q0/Q1: B8 preservation factorial (2026-10-17)

This directory contains the owner-approved Q0/Q1 implementation. It answers one bounded
question before any adaptive league work: can a B8 weights-only fine-tune preserve
or improve fixed-opponent H5000 mass when its schedule and collection package are
varied independently?

The intended users are the study owner and an operator already familiar with the
M3-C trainer, immutable queue manifests, and T23's terminal-hero evaluator.  Read
[PREREGISTRATION.md](PREREGISTRATION.md) before creating, queuing, or interpreting
an artifact. The owner approved `APPROVED_Q0Q1_20261017` and the A1 envelope on
2026-10-18. Claude made A2 as a planner amendment under the standing approval
"everything approved for today dont ask me again"; the appended erratum corrects
the earlier attribution of an explicit owner request. The approved queue path still requires passing Q0 audits, both measured
Q0b packets, parity and a complete budget forecast. Implementation tests do not
qualify a GPU or claim a scientific result. The older `jobs/` manifests remain
draft historical examples; use `jobs_approved/` and the generator below.

## Contents and boundaries

| Item | Purpose | Status |
|---|---|---|
| `PREREGISTRATION.md` | Frozen-before-data Q0/Q1 hypothesis, qualification gates, factorial, evaluation and decision rules | A3 inclusive budget and lag operationalization; A2 attribution erratum retained |
| `q1_bank.py` | Creates Q1's five fresh 64/32/32 H5000 terminal-hero banks and their pre-data collision audit | Committed allocation/audit; no runtime or evaluation has run |
| `q1_evaluate.py` | CPU shard evaluator for A--D and the one reusable B8 identity on the same registered bank | Implementation; development only |
| `q1_parity.py` | Four-world Mac/Colab diagnostic parity comparison | Implementation; a passing probe would not be a strict gate |
| `jobs/` | One queue-schema JSON object per proposed job, generated from the pinned dependency graph | Pending owner review; not enqueued |

The package has two gates.  Q0a is a descriptive audit of saved T23/B8 artifacts;
it cannot reverse T23's 8M KILL or produce a gameplay efficacy GO.  Q0b qualifies
the exact code, GPU runtime, actor count, counters, and capacity needed by Q1.  A
missing or failed Q0 acceptance check blocks Q1; it is not permission to alter a
slow cell, substitute a seed, or shorten a bank.

## Registered trainer interfaces

The named packages are explicit arguments so old behavior remains the default:

```text
--schedule-package restart|mature        # default: restart
--collection-profile legacy|task-aligned-n1  # default: legacy
```

`mature` fixes gamma to `.998`, epsilon to `.01`, and the anchor weight to `.1`
from learner decision zero through 8M.  It retains positive schedule spans, so it
does not create a zero epsilon-decay denominator.  `task-aligned-n1` requires a
multi-actor run with `--learner-seats 1` and a fresh namespace beginning
`q1-task-aligned-n1-20261017`; it is not an alias for legacy N=1.

The queue manifest, rather than this README, must provide the reviewed complete
command after Q0b freezes its GPU-specific actor/world/queue configuration.  The
queue accepts one JSON object per `enqueue`, never a JSON array.  Every generated
job must bind source/configuration/checkpoint/opponent/profile/bank hashes, actual
unit rate, hardware class, runtime limit, dependencies, and the specific Q1 arm
and seed.  It must use create-only output paths and preserve failed receipts.

## Expected acceptance artifacts

Before Q1, the owner should be able to inspect:

1. Q0a hash and zero-update B8 fidelity results, plus a clearly marked descriptive
   2M/4M report only for snapshots that actually exist.  Its 480 mass-decomposition
   policy-episode cap and 160-world B8 observation-capture reuse are separately
   accounted; this documentation claims neither has run.
2. Q0b oracle and parity results, then one L4 and one A100 timing packet containing
   every schedule-by-collector cell, warm-up, both scored windows, counters, lag,
   ring-overrun status, exit status, and the measured price/rate.
3. A fresh Q1 collision audit, immutable world-bank receipt, runtime-local parity
   receipt, and final training receipt for every A--D seed before its evaluation
   rows can be analyzed.

The final Q1 evaluator uses exactly five policy identities per bank: A, B, C, D,
and B8.  Because the B8 hero row is evaluated once and reused only inside that
same bank and manifest, the total is `5 × 128 × 5 = 3,200` episodes, not 6,400.
It evaluates with one CPU thread per world and accepts only a
`q1-analysis-input/v1` assembly of the committed bank, collision audit, and full
immutable shard receipts.  Its output remains a Colab development screen.  Any
advancement still requires a fresh Mac Phase R procedure and separately qualified
serving work.

## Common invalid states

Do not repair these by rerunning or silently changing the workload:

- A checkpoint fallback, wrong wrapper/roster identity, action-mask disagreement,
  frozen-seat replay/label contamination, missing B8 row, or failed parity is
  `INVALID/INCOMPLETE`.
- A runtime that cannot run the common qualified configuration is ineligible.  Do
  not lower actor count for one arm or silently replace L4 with A100.
- An exhausted time, unit, or aggregate CPU/GPU cap is `INCOMPLETE`.  Record its
  observed price and usage; an unknown CPU price is never entered as zero.
- A numerical or behavioral training tripwire is scientific `KILL` for that arm,
  with no substitute seed or automatic retry.

For the complete outcome order, preservation rule, and deterministic selection
tie-break, use [PREREGISTRATION.md](PREREGISTRATION.md#q1-analysis-and-ordered-decisions).

## A3 operator sequence (2026-10-18)

**Current outcome: both preserved pre-A3 r6 hardware packets qualify under A3;
Q1 is REFUSED on GPU-hours and CPU-hours. Do not enqueue L4 r7 or Q1.**
The [operational receipt](Q0B_A3_REQUALIFICATION_20261018.md) records exact hashes,
retrospective admission disclosure, inclusive Q0 debit T, every forecast cap and
the actual generator refusal. This supersedes the earlier L4 r7 replacement and
enqueue sequence; its approved manifest is retained unchanged as historical evidence.
Both workers remain alive and idle under the owner's instruction.

A3 fixes Q0's cap at final inclusive T and the combined unit envelope at T+135,
with the original 35-unit reserve: usable combined units are T+100. Final
accounting includes all 12 attempts (including L4 after midnight UTC) and the
two disclosed historical conservative closure bounds. T is
68.63035524611341 units; combined nominal is 203.63035524611342 and usable is
168.63035524611342. The unchanged 19.2/24 GPU-hour and 32/40 CPU-hour envelopes
still bind. The complete forecast is 29.468238339864364 GPU-hours and
67.11074987439922 CPU-hours; combined units 134.30072755019802 and Q1 training
32.14652514466276 units fit their limits. Runtime admission fails independently.

The Mac-only `requalify_q0b` accepts exactly the two preserved pre-A3 r6
hardware/job/failure identities, rechecks source, parity, pricing and queue
provenance and writes a new archive. Existing evidence remains immutable. It
charges actual claim-to-terminal time once and derives conservative Q1 overhead.
The unchanged A100 A3 archive is admitted with its exact historical tool digest;
the new L4 archive is `q0b/capacity-l4-20261018r6-a3/result.tar.gz`.
The failed original queue states remain failed. The generator can omit either
failed dependency only after the A3 receipt and accounting bind the same exact
claim and terminal hashes. No done receipt is manufactured.

The following reproduces the actual local admission checks using fresh review
paths. The two review output paths must not exist; replace them with new paths
for another independent replay. Accounting is read-only with respect to the
queue. The generator is expected to refuse the unchanged runtime envelopes and
emit no manifests. These commands do not dispatch work.

```bash
set -euo pipefail
cd /Users/josenunez/Projects/ml/snake-dqn-q0b-l4-a3
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
ADMISSION="$DRIVE/q0q1-20261017/admission"
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
"$PY" -m research.q0q1_20261017.q0_accounting \
  --drive-root "$DRIVE" --pricing "$ADMISSION/observed-pricing.json" \
  --out /tmp/observed-q0-accounting-a3-r6-review.json
"$PY" -m research.q0q1_20261017.make_q1_jobs \
  --commit "$(git rev-parse HEAD)" \
  --inventory-receipt "$ADMISSION/inventory-receipt.json" \
  --q0-mass-archive "$DRIVE/q0q1-20261017/q0/audit-t23-mass-a100r5/result.tar.gz" \
  --q0-init-archive "$DRIVE/q0q1-20261017/q0/audit-b8-init-a100r6/result.tar.gz" \
  --q0b-l4-receipt "$DRIVE/q0q1-20261017/q0b/capacity-l4-20261018r6-a3/result.tar.gz" \
  --q0b-a100-receipt "$DRIVE/q0q1-20261017/q0b/capacity-a100-20261018r6-a3/result.tar.gz" \
  --mac-parity-reference "$ADMISSION/mac-parity-reference.json" \
  --q0-accounting /tmp/observed-q0-accounting-a3-r6-review.json \
  --q0b-l4-job-id q0b-capacity-l4-20261018r6 \
  --q0b-a100-job-id q0b-capacity-a100-20261018r6 \
  --drive-output-dir /content/drive/MyDrive/t16_colab/q0q1-20261017/q1/jobs_approved_a3_r6_review \
  --out-dir /tmp/q1-jobs-a3-r6-review
```

The actual run used committed source `8a8a17f5ceaa7026182e565ff996078ad1974c78`,
accounting `admission/observed-q0-accounting-a3-final-r6.json` and NEW output
`q1/jobs_approved_a3_r6_final` under `q0q1-20261017` on Drive. That directory
contains staged evidence, exact invocation/log, the refused plan and hash-bound
forecast. It contains **zero manifests**; 120-manifest validation and enqueue
commands do not apply. No additional qualification or training is authorized.

## Historical draft manifests and local verification

The checked-in `jobs/` directory contains 122 individual queue JSON objects:
2 dependent Q0 CPU audits, 20 L4 arm/seed trainers, 20 B8 CPU shards, and 80
candidate CPU shards. They pin a real implementation commit and remain
`DRAFT_PENDING_OWNER_REVIEW`. No job has been enqueued. Measured Q0b, budget,
training-runtime parity, Mac reference, and owner-admission bindings are absent
until those artifacts exist; the adapter refuses draft execution.

Q0 mass and fidelity jobs allow respectively 240 and 120 script minutes. With
up to 30 setup minutes per job, their combined attempt cap fits the eight-hour
Q0 CPU envelope. Each L4 attempt permits 90 script minutes, including the trainer's
5,100-second cap and five-minute export reserve, plus the queue's at-most-30-minute
setup. Global caps still apply first; these per-job maxima are not additional budget.

After a later owner-authorized run, extract the validated immutable job archives
into one local evidence directory and analyze all receipts with:

```bash
python -m research.q0q1_20261017.q1_analyze --shard-root /path/to/extracted/evidence --out q1-analysis.json
```

The analyzer reads shard receipts and retained `training-outcome.json` files.
It validates each arm separately. Missing or corrupt evidence makes that arm
`INVALID/INCOMPLETE`; a verified scientific tripwire makes it `KILL`. No subset
of surviving seeds supplies an arm interval. Contrasts requiring an incomplete
arm are unavailable. Shared B8 failures invalidate every dependent arm.
