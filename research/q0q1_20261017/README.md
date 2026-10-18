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
| `PREREGISTRATION.md` | Frozen-before-data Q0/Q1 hypothesis, qualification gates, factorial, evaluation and decision rules | A2 with attribution erratum; A1 envelope unchanged |
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

## Non-interactive Q1 operator sequence (2026-10-18)

Q0 mass uses `q0/audit-t23-mass-a100r5/result.tar.gz`; Q0 init uses
`q0/audit-b8-init-a100r6/result.tar.gz`. The init r6 audit is COMPLETE.
The capacity attempts are `q0b-capacity-a100-20261018r6` and
`q0b-capacity-l4-20261018r6`; wait for **both** packets to be QUALIFIED.
The L4 runtime still needs the owner's Drive authorization click. Do not enqueue
old r2/r3/r4/r5 replacements or rerun an audit already COMPLETE.

The admission directory already holds the hash-bound `inventory-receipt.json`,
`mac-parity-reference.json`, and `observed-pricing.json`. Keep them immutable.
The inventory receipt SHA-256 is
`2f9d74980fc8ed6a683deb52c4933eaf0516bdbcb3dea072d1b27d03ced542aa`.
A2's admission is `OWNER_ADMISSION_20261018_A2.json`, bound to the complete
registration including its append-only authorization erratum. The original A1
admission remains unchanged and deliberately fails current-binding validation.
Both the generator and runtime validate the latest registration amendment and
its digest. A2 retains A1's 40-unit Q0 and 175-unit combined envelope, 20% reserve,
and every other cap.

Run this sequence from the clean published checkout after both packets qualify.
It has no prompts. `q0_accounting` reads every Q0/Q0b claim and attempt receipt
since the study date (including attempts after UTC midnight), including failed
generations, and prints the charged seconds,
hardware, price, and evidence behind each contribution. Pending jobs carry no
attempt debit; active or unaccountable claims block final accounting. Setup,
execution, copy/export and cleanup inside an attempt are included in its inclusive
queue elapsed time. Missing terminal evidence is never silently treated as zero.
The measured CPU overhead receipt records three real Mac runs of the empty
evaluator shard, with staging, archive publication and cleanup; no selection
world is scored. The budget uses three times the slowest Mac run for projected
x86 work, plus the largest receipt-backed queue setup/prelude (an A100 proxy),
plus the registered 60-second export/analysis/cleanup reserve. These forecast
components are disclosed separately from direct measurements. Per-episode
gameplay/parity cost still comes from the qualifying runtime receipt.
The abandoned CPU claim is conservatively bounded by its worker's explicit idle
heartbeat; mass r2 generation 0 is bounded by the operator's recorded verified
runtime disconnection. Neither is presented as an exact terminal measurement.
The current pricing file describes A100 5.3 units/hour as observed, L4 2.0 as an
owner-supplied planning rate, and CPU 0.5 as a conservative estimate. Those
provenance limits and all conservative bounds remain disclosed in the accounting
artifact; rates in a file named "observed" do not become observations
by virtue of that filename.

```bash
set -euo pipefail
cd /Users/josenunez/Projects/ml/snake-dqn-q1-admission
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
ADMISSION="$DRIVE/q0q1-20261017/admission"
JOBS="$DRIVE/q0q1-20261017/q1/jobs_approved"
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

"$PY" -m research.q0q1_20261017.q0_accounting \
  --drive-root "$DRIVE" \
  --pricing "$ADMISSION/observed-pricing.json" \
  --out "$ADMISSION/observed-q0-accounting.json"

"$PY" -m research.q0q1_20261017.make_q1_jobs \
  --commit "$(git rev-parse HEAD)" \
  --inventory-receipt "$ADMISSION/inventory-receipt.json" \
  --q0-mass-archive "$DRIVE/q0q1-20261017/q0/audit-t23-mass-a100r5/result.tar.gz" \
  --q0-init-archive "$DRIVE/q0q1-20261017/q0/audit-b8-init-a100r6/result.tar.gz" \
  --q0b-l4-receipt "$DRIVE/q0q1-20261017/q0b/capacity-l4-20261018r6/result.tar.gz" \
  --q0b-a100-receipt "$DRIVE/q0q1-20261017/q0b/capacity-a100-20261018r6/result.tar.gz" \
  --mac-parity-reference "$ADMISSION/mac-parity-reference.json" \
  --q0-accounting "$ADMISSION/observed-q0-accounting.json" \
  --q0b-l4-job-id q0b-capacity-l4-20261018r6 \
  --q0b-a100-job-id q0b-capacity-a100-20261018r6 \
  --out-dir "$JOBS"
```

Both tools refuse to overwrite existing outputs. The generator verifies both Q0
COMPLETE receipts, both qualified four-cell packets, identical source/recipe
bindings, inventory exclusions, Mac/runtime parity, current A2 owner admission,
and the complete A1-envelope forecast. It writes exactly 20 training and 100
evaluation manifests, copied evidence and the study plan. An INCOMPLETE packet,
unaccountable attempt, failed parity, or over-cap forecast stops the sequence.
Preserve such evidence; do not fill missing numbers, retry, replace seeds, change
actors, or substitute a checkpoint to make admission pass.

Wait for the generated evidence and study plan to synchronize to Drive before
running this separate enqueue block. The sequence stops on the first error:

```bash
q() { "$PY" research/colab_queue/q.py --drive-root "$DRIVE" "$@"; }
for job in "$JOBS"/q1-train-*.json; do q enqueue "$job"; done
for job in "$JOBS"/q1-eval-b8-*.json; do q enqueue "$job"; done
for arm in a b c d; do
  for job in "$JOBS"/q1-eval-"$arm"-*.json; do q enqueue "$job"; done
done
```

Keep at most three runtimes, initially two qualified L4 learners and one CPU
worker. Q1 remains one L4 cohort; A100 measures the registered comparison only.
Each evaluation job uses one CPU thread per world, with parallelism across the
100 immutable shards. A passing four-world parity probe is development evidence.
No command here promotes a checkpoint or changes serving.

**Budget evidence at this repair:** closed attempt receipts plus the two explicit
closure bounds already charge approximately **63.59 units** before A100 r6 and
L4 r6. This exceeds the unchanged **40-unit Q0 cap**. Qualification alone does
not clear that cap. The commands above preserve that refusal; no budget change
or dispatch is authorized by this tooling repair.

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
