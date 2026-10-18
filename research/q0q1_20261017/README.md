# Q0/Q1: B8 preservation factorial (2026-10-17)

This directory contains the owner-approved Q0/Q1 implementation. It answers one bounded
question before any adaptive league work: can a B8 weights-only fine-tune preserve
or improve fixed-opponent H5000 mass when its schedule and collection package are
varied independently?

The intended users are the study owner and an operator already familiar with the
M3-C trainer, immutable queue manifests, and T23's terminal-hero evaluator.  Read
[PREREGISTRATION.md](PREREGISTRATION.md) before creating, queuing, or interpreting
an artifact. The owner approved `APPROVED_Q0Q1_20261017` and Amendment A1 on
2026-10-18. The approved queue path still requires passing Q0 audits, both measured
Q0b packets, parity and a complete budget forecast. Implementation tests do not
qualify a GPU or claim a scientific result. The older `jobs/` manifests remain
draft historical examples; use `jobs_approved/` and the generator below.

## Contents and boundaries

| Item | Purpose | Status |
|---|---|---|
| `PREREGISTRATION.md` | Frozen-before-data Q0/Q1 hypothesis, qualification gates, factorial, evaluation and decision rules | Owner approved, Amendment A1 |
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

## Approved operator sequence (2026-10-18)

These commands are for the owner to run. Preparing this implementation does not
enqueue work, restart a worker, access a Colab session or touch tmux. Keep at most
three runtimes, initially two L4 learners and one CPU worker. Q1 remains an L4
cohort; the A100 packet measures the registered comparison only.

From the Mac checkout, establish paths and create the source/bank-bound reference:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-adv-league
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
export SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes
export SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
ADMISSION="$DRIVE/q0q1-20261017/admission"
"$PY" -m research.q0q1_20261017.make_q1_mac_parity_reference \
  --b8 "$SNAKE_CKPT_DIR/b8_s3_t7_20261016.pth" \
  --out "$ADMISSION/mac-parity-reference.json"
```

Record actual observed debit rates. These are data inputs, not another owner
approval. Never substitute zero for an unknown CPU price or treat a catalog
projection as an observation. The following prompts produce the synchronized
pricing input and SHA sidecars for it and the Mac reference:

```bash
printf 'Observed L4 units/hour: '; read -r L4_RATE
printf 'Observed A100 units/hour: '; read -r A100_RATE
printf 'Observed CPU units/hour: '; read -r CPU_RATE
printf 'Observation source/date: '; read -r RATE_EVIDENCE
"$PY" -m research.q0q1_20261017.prepare_q0b_inputs \
  --l4-units-per-hour "$L4_RATE" --a100-units-per-hour "$A100_RATE" \
  --cpu-units-per-hour "$CPU_RATE" --evidence "$RATE_EVIDENCE" \
  --mac-reference "$ADMISSION/mac-parity-reference.json" \
  --out "$ADMISSION/observed-pricing.json"
```

After Drive has synchronized these files, enqueue the fidelity audit first,
then the two hardware packets. The mass job already exists; do not enqueue it
again. Dependencies hold the packets until the named audits complete, and
scientific COMPLETE checks hold qualification/admission closed on a failed audit.

```bash
q() { "$PY" research/colab_queue/q.py --drive-root "$DRIVE" "$@"; }
q enqueue research/q0q1_20261017/jobs_approved/q0-audit-b8-init-a100r2.json
q enqueue research/q0q1_20261017/jobs_approved/q0b-capacity-l4-20261018.json
q enqueue research/q0q1_20261017/jobs_approved/q0b-capacity-a100-20261018.json
```

Each hardware packet has one attempt, four real learner cells, at least 120 seconds
warm-up per cell after all actors/replay/compilation are ready, and two complete
scored windows of at least 180 seconds each. Observed boundary delays are included
in throughput denominators. The inclusive 60-minute cap conservatively counts
queue arbitration and setup and leaves 60 seconds for export. Timeout, stale ring
rows, drops, nonfinite updates, nonzero actor exits, parity failure or a missing
measurement exports INCOMPLETE; it does not retry or shorten a window. CUDA B8
inference is compared against the retained old pool; vector61 and v8 kernels
remain on CPU as in the actual actor stack.

After both packet archives and their SHA sidecars have synchronized, record the
actual complete Q0 debit (audits, packets, setup, failures, copies and cleanup)
and measured per-evaluation-job CPU startup/copy/export overhead. Include any
remaining analysis/cleanup allowance in that overhead. Values may be entered from
the queue attempt receipts and runtime debit observations; hours for an unused
allocation class may be zero, but the observed CPU price must be positive.

```bash
"$PY" - "$ADMISSION/observed-q0-accounting.json" <<'PY'
import datetime, json, pathlib, sys
path = pathlib.Path(sys.argv[1])
if path.exists():
    raise SystemExit(f'refusing to replace {path}')
fields = {
    'q0_units': 'Total observed Q0 units including failed attempts/setup/cleanup: ',
    'q0_gpu_hours': 'Total observed Q0 GPU allocation hours: ',
    'q0_cpu_hours': 'Total observed Q0 CPU allocation hours: ',
    'cpu_unit_rate_per_hour': 'Observed CPU units/hour (same observation as packets): ',
    'cpu_eval_job_overhead_seconds': 'Measured CPU overhead seconds per eval job, including reserves: ',
}
# /dev/tty remains interactive when Python source is supplied through stdin.
value = {}
with open('/dev/tty', 'r+') as tty:
    for key, prompt in fields.items():
        tty.write(prompt); tty.flush()
        value[key] = float(tty.readline().strip())
value['observed_at'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
with path.open('x') as stream:
    json.dump(value, stream, sort_keys=True, indent=2)
    stream.write('\n')
PY
```

Generate the complete study packet with all evidence hashes. This checks both Q0
scientific receipts, both four-cell qualification packets, identical measured
sources/recipes, Mac/runtime parity, owner admission and the A1 forecast. It
requires clean immutable HEAD. It writes one study plan, copied evidence and
exactly 120 individual job JSONs; it never enqueues them. The output directory
must be new.

```bash
JOBS="$DRIVE/q0q1-20261017/q1/jobs_approved"
"$PY" -m research.q0q1_20261017.make_q1_jobs \
  --commit "$(git rev-parse HEAD)" \
  --q0-mass-archive "$DRIVE/q0q1-20261017/q0/audit-t23-mass-a100r/result.tar.gz" \
  --q0-init-archive "$DRIVE/q0q1-20261017/q0/audit-b8-init-a100r2/result.tar.gz" \
  --q0b-l4-receipt "$DRIVE/q0q1-20261017/q0b/capacity-l4-20261018/result.tar.gz" \
  --q0b-a100-receipt "$DRIVE/q0q1-20261017/q0b/capacity-a100-20261018/result.tar.gz" \
  --mac-parity-reference "$ADMISSION/mac-parity-reference.json" \
  --q0-accounting "$ADMISSION/observed-q0-accounting.json" \
  --q0b-l4-job-id q0b-capacity-l4-20261018 \
  --q0b-a100-job-id q0b-capacity-a100-20261018 \
  --out-dir "$JOBS"
```

Wait for the generated evidence and study plan to synchronize before enqueueing.
Publish training, reusable B8 shards, then candidate shards in that order:

```bash
for job in "$JOBS"/q1-train-*.json; do q enqueue "$job" || break; done
for job in "$JOBS"/q1-eval-b8-*.json; do q enqueue "$job" || break; done
for arm in a b c d; do
  for job in "$JOBS"/q1-eval-"$arm"-*.json; do q enqueue "$job" || break; done
done
```

The generator does not fill missing measured values. An INCOMPLETE packet or an
over-budget forecast blocks Q1. Preserve the attempt; do not requeue automatically,
replace seeds, adjust one arm's actor count or substitute a checkpoint. A1 sets
Q0 to 40 units and the combined envelope to 175 units, with 20% reserved (140
usable); the 24 GPU-hour, 40 CPU-hour and Q1 60-unit training caps remain in force.

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
