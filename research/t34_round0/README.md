# T34 Round 0 preparation and operator runbook

Latest implementation evidence: [VERIFICATION_FIXES1.md](VERIFICATION_FIXES1.md).
Latest rules: [FIX-FIRST #1 erratum](ERRATUM_FIXES1.md) and
[preregistration supplement](PREREGISTRATION_ROUND0_FIXES1.md).
Historical implementation evidence: [VERIFICATION_INTEGRITY_SIZING.md](VERIFICATION_INTEGRITY_SIZING.md).

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. This is a 240-pair development
premise test against bare B8 s3, not a release. The original
[preregistration](PREREGISTRATION_ROUND0.md), [A5 amendment](ERRATUM_A5_ADDENDUM_B.md),
and [integrity/sizing amendment](ERRATUM_INTEGRITY_SIZING.md) apply except where
the FIX-FIRST #1 supplement supersedes them. First wall/unit cap is
**INCONCLUSIVE**, without extension. Abort or unexplained interruption is KILL.
Original hash-bound documents,
banks and receipts stay unchanged. No Round 0 labels go to fitting.

Production layout is the A5 PASS recipe: **L4, K=12 processes, one CPU inference
thread each, W1**; both arms of each pair run in one process/runtime. Preparation
and excluded A6 run on the Mac with one worker and one numerical thread. This
implementation does not ratify, dispatch, touch live Colab/tmux or change serving.

## Reproduce local preparation

Use fresh artifact paths for every excluded A6 reproduction. The saved bank/audit
already exist: never reallocate selection worlds. Commit all source files before
A6 so its inventory includes the full implementation. The optional setup argument
below is a planning allowance and must cover the measured A5 setup/export/cleanup
elapsed time. The adapter refuses a smaller value.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0-fixes1
export AUTHOR_ROOT="$PWD"
export PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export PREP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020
export A6="$PREP/a6-reproduction-01"
export A5ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991
"$PY" -m research.t34_round0.bank --preflight-inputs "$A6/inputs"
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$PY" -m research.t34_round0.runner --preflight \
  --bank "$A6/inputs/bank.json" --audit "$A6/inputs/audit.json" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --output-dir "$A6/output" --runtime-id t34-round0-a6-reproduction-01 \
  --start 0 --count 2 --device cpu --threads 1 --seconds 600
"$PY" -m research.t34_round0.sizing \
  --a5-benchmarks "$A5ROOT/a5-l4-s0-output/exports/benchmark.json" \
                  "$A5ROOT/a5-l4-s1-output/exports/benchmark.json" \
  --preflight-dir "$A6/output" --setup-minutes 15 --out "$A6/A5_RECEIPT.json"
"$PY" -m research.t34_round0.validate --preflight-dir "$A6/output" \
  --a5 "$A6/A5_RECEIPT.json" --out "$A6/VERIFICATION.json"
```

Sizing reads the two actual A5 v2 reports and their supervisor PASS/release
receipts, uses the slower teacher throughput, and binds separately timed bare and
metric replay costs from A6. H500 measurements are conservatively scaled by 10
for H5000 and by 3 for the L4 CPU planning projection; only those CPU costs divide
by K=12. These are explicit assumptions, not measured paired L4 throughput.
Chunk estimate <=180 min; timeout >=estimate+25 min and >=2*ceil(estimate).
Exports use 9 MiB/world *1.25 and stay strictly below 2 GiB. Forecast prices every
timeout at 2.0 u/h, holds an additional 20%, and shows both initial and maximum
one-rerun-per-chunk liabilities. No unknown debit is represented as zero.

## Rebuild the draft without gameplay

Saved measured A5/A6 component costs can price a draft after a source change.
They do not qualify the new source. This command performs local validation only
and builds the chunk plan, supervisor-schema manifests and a registration whose
DRAFT status prevents production admission. Choose a fresh output path:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0-fixes1
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.t34_round0.draft \
  --a5 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-integrity-20261020/a6-03/A5_RECEIPT.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-fixes1-20261020/draft-01 \
  --study-root /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01
```

The execution view interleaves 4 scripted / 3 frozen / 3 mixed worlds per stratum;
the maximum feasible chunk size rounds down to a multiple of ten. Bank and audit
files remain byte-identical. Each chunk has exact mix proportions.

## Future ratification and frozen execution checkout

**Not executed by this implementation.** An independent Fable judge must approve
ratification under campaign authority and the operator must record it in
CAMPAIGN_LOG.md. Review the concrete draft sizing first. A6 must match the final
source closure; after executable/rule changes repeat the excluded A6 with a fresh
path. Commit/push implementation before approval. Ratification creates a separate
receipt; it never edits the original preregistration.

```bash
cd "$AUTHOR_ROOT"
export SOURCE="$(git rev-parse HEAD)"
export E=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-study-01
export FROZEN=/Users/josenunez/Projects/ml/snake-dqn-round0-frozen-01
: "${T34_OPERATOR:?Set the independent approving judge identity}"
"$PY" -m research.t34_round0.registration --operator "$T34_OPERATOR" \
  --study-root "$E" --a5 "$A6/A5_RECEIPT.json" \
  --preflight-record "$A6/VERIFICATION.json" \
  --out "$AUTHOR_ROOT/research/t34_round0/REGISTRATION_ROUND0.json"
git add -- research/t34_round0/REGISTRATION_ROUND0.json
git commit -m 'Ratify T34 Round 0 fixed study root and deadline'
git fetch origin
git merge origin/main
git push origin HEAD:main
git push origin t34-round0-fixes1-20261020
export RATIFICATION="$AUTHOR_ROOT/research/t34_round0/REGISTRATION_ROUND0.json"
git worktree add --detach "$FROZEN" "$SOURCE"
cd "$FROZEN"
"$PY" -m research.t34_round0.make_manifest --a5 "$A6/A5_RECEIPT.json" \
  --ratification "$RATIFICATION" --commit "$SOURCE" \
  --output-root "$E" --out-dir "$E/manifests" \
  --lanes 1 --other-assignments '{}' --setup-minutes 15
```

`{}` is valid only when no other account assignment exists. Otherwise replace it
with every existing endpoint's reviewed positive units/h rate; lanes plus these
assignments cannot exceed three. The broker rechecks account-wide admission.
Use one lane for the serial publication-barrier runbook below. Manifests can
assign up to three lane IDs, but every completed phase must publish before the
next admission. This conservative serial dispatch does not claim concurrent
Round 0 speedup. The ratification freezes exact chunk slices and all sizing;
changing roots, costs, setup or partition after ratification is refused. Admission
requires this exact canonical registration JSON on origin/main and rejects any
origin `t34-round0/*` claim tag with a different root. The frozen execution checkout
stays at SOURCE while the authoring checkout commits the separate registration.

Every chunk must dispatch within 48 hours of ratification and finish within its
registered timeout. A missing chunk at the deadline is KILL. Mechanical recovery
does not extend the deadline. Do not open chunk exports before all chunks are
terminal; analysis refuses an incomplete schedule while its deadline is pending.

## Future explicit dispatch and publication barrier

Use only the Round 0 wrapper around the qualified supervisor. Never call the
Colab job CLI or the broker run command directly. The wrapper makes a unique
create-only remote admission tag, reserves the exact attempt directory, uploads
the claim, monitors the frozen worktree, preserves every result/failure, seals
all files, and appends its terminal hash to the authoring repo ledger. It never
automatically retries. Missing/unavailable remote publication refuses work.

```bash
cd "$FROZEN"
export M0="$E/manifests/t34-round0-chunk-000-attempt-01-l1.json"
"$PY" -m research.t34_round0.dispatch run --manifest "$M0" \
  --ratification "$RATIFICATION" \
  --state-dir "$E/supervisor-chunk-000-attempt-01"
```

After **every** dispatch, replay, or analysis, publish its ledger append before
continuing. Keep the authoring checkout separate from the frozen checkout:

```bash
cd "$AUTHOR_ROOT"
export LEDGER="research/t34_round0/ledger/$(basename "$E").jsonl"
git add -- "$LEDGER"
git commit -m 'Preserve T34 Round 0 terminal receipt history'
git fetch origin
git merge origin/main
git push origin HEAD:main
git push origin t34-round0-fixes1-20261020
cd "$FROZEN"
```

For the next generated manifest, use chunk-001 and a fresh corresponding state
path, then publish again. Repeat for every forecast chunk. Inspect only progress,
errors, hashes and operational terminal status; outcome values stay unopened.
A supervisor CLI exit alone proves nothing: admission requires its immutable
receipt, successful terminal, all exports and verified runtime disappearance.

One manual rerun per chunk is permitted only when its terminal seal says
MECHANICAL_FAILURE: verified release plus a bound operational error, including
export/serialization failure or failed artifact download. Preserve the first attempt and publish its ledger first;
then use the same original manifest with `--mechanical-rerun --outcomes-unseen`
and a fresh state path ending attempt-02. The wrapper selects the fixed attempt-02
path. No third attempt or rerun after replay/analysis. Setup, failure and rerun
are all priced inside the frozen total liability.

An explained unconditional controller fault blocks further dispatch and replay
returns STOP_CONTROLLER_FAULT without requiring a complete chunk. The sole
analysis still verifies immutable fault identity and emits the original STOP.
First wall/unit cap is INCONCLUSIVE, with no extension or cap retry. Operator
abort, source drift, missed scheduling deadline and unexplained interruption are KILL.
After a cap, skip replay and invoke the sole analysis directly; it settles from
the published operational history without opening outcomes or requiring replay.
A hard kill leaves an unfinished remote claim that still blocks admission even
if local files are deleted. Do not repair/delete/relabel such history. No world
replacement, scientific resume, selection-bank extension or outcome-based retry.

## Future Mac replay and sole fixed-N decision

Use the latest COMPLETE attempt for each chunk. The example below assumes the
first attempts of two chunks succeeded; adjust only attempt-01 to the admitted
attempt-02 if mechanical recovery was used. Exact chunk count comes from the
frozen forecast. Keep every failed attempt at its fixed location.

```bash
cd "$FROZEN"
export C0="$E/chunks/chunk-000/attempt-01/job/exports/round0"
export C1="$E/chunks/chunk-001/attempt-01/job/exports/round0"
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$PY" -m research.t34_round0.replay --bank research/t34_round0/BANK_ROUND0.json \
  --audit research/t34_round0/SEED_AUDIT_ROUND0.json --ratification "$RATIFICATION" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --chunk-dirs "$C0" "$C1" --out "$E/replay/replay.json"
# Publish the ledger using the block above before analysis.
"$PY" -m research.t34_round0.analyze --bank research/t34_round0/BANK_ROUND0.json \
  --audit research/t34_round0/SEED_AUDIT_ROUND0.json --ratification "$RATIFICATION" \
  --chunks "$C0" "$C1" --replay "$E/replay/replay.json" --out "$E/analysis/decision.json"
# Publish the terminal ledger again before reporting the decision.
```

Replay selects the ceil(10% of allocation) canonical worlds **within each
chunk**, stratified by mix with largest-remainder quotas and bank-order ties, both arms, before inspecting outcomes. Reports include allocated,
completed and dropped counts plus all divergences. >2% drops in any chunk stop
and report; any missing pair still blocks the whole 240-pair look. Trace replay
is distinct from a Mac strict gate or cross-platform greedy-policy equivalence.

Only `analysis/decision.json` is the verdict. Imported consumer APIs also require
a reserved phase; a second analysis or copied output path cannot create another
look. Published immutable history and remote claims are checked even if source
or local artifacts were restored. Numerical summaries grant no evidence authority.
Development GO only admits the next planning decision; strict gate, serving
qualification and release remain separate.
