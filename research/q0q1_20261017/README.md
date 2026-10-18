# Q0/Q1: B8 preservation factorial (2026-10-17)

This directory is the proposed, unexecuted Q0/Q1 package.  It answers one bounded
question before any adaptive league work: can a B8 weights-only fine-tune preserve
or improve fixed-opponent H5000 mass when its schedule and collection package are
varied independently?

The intended users are the study owner and an operator already familiar with the
M3-C trainer, immutable queue manifests, and T23's terminal-hero evaluator.  Read
[PREREGISTRATION.md](PREREGISTRATION.md) before creating, queuing, or interpreting
an artifact.  The registration is deliberately `DRAFT_PENDING_OWNER_REVIEW`; it
does not authorize training, Colab, pod, tmux, release, or promotion work.

## Contents and boundaries

| Item | Purpose | Status |
|---|---|---|
| `PREREGISTRATION.md` | Frozen-before-data Q0/Q1 hypothesis, qualification gates, factorial, evaluation and decision rules | Draft pending owner review |
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

## Local verification and future operator commands

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
