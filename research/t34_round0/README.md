# T34 Round 0 preparation and operator runbook

Status: **DRAFT_PENDING_OPERATOR_RATIFICATION**. This package prepares the
240-pair teacher premise test; implementing it does not authorize scientific
execution. [Preregistration](PREREGISTRATION_ROUND0.md) fixes the bank, statistics,
ordered GO/STOP/INCONCLUSIVE rules, and isolation. [OC note](OPERATING_CHARACTERISTICS.md)
discloses power and cost limitations. No labels from this study go to fitting.

The operator supplies the actual A5 receipt; no rate or successful rehearsal is
fabricated. Existing public release `s1-inputs-20261019` supplies pinned B8 and
frozen checkpoints through `research/t34_exit_b8/PUBLIC_INPUTS.json`. All Colab
operations use the already-qualified supervisor, never direct job CLI commands.

## Local preparation (Mac only, no campaign arm)

Use the project interpreter and one numerical thread. These commands are also the
reproduction instructions for the excluded two-world H500 preflight. The saved
bank/audit must already exist; do not rerun allocation against its own committed
seed inventory. All outputs must be fresh. `source_hashes` uses the tracked
inventory, so all implementation files must be committed or staged before A6.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
export T34_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export T34_ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-20261020
export T34_A6="$T34_ART/a6-reproduction-01"
"$T34_PY" -m research.t34_round0.bank --preflight-inputs "$T34_A6/inputs"
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$T34_PY" -m research.t34_round0.runner --preflight \
  --bank "$T34_A6/inputs/bank.json" --audit "$T34_A6/inputs/audit.json" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --output-dir "$T34_A6/output" --runtime-id t34-round0-a6-reproduction-01 \
  --start 0 --count 2 --device cpu --threads 1 --seconds 600
"$T34_PY" -m research.t34_round0.validate --preflight-dir "$T34_A6/output" \
  --out "$T34_A6/VERIFICATION.json"
```

The validator is read-only except for its create-only report. A6 worlds are stored
under the separate excluded preflight namespaces. It verifies trace/pair hashes,
two H500 worlds, CPU/one-thread identity and the executable source inventory.
No scientific performance claim follows from A6. Any executable/rule change
requires a new A6 receipt before ratification; existing evidence stays intact.

[TABULAR_SEED_AUDIT.json](TABULAR_SEED_AUDIT.json) supplements the original text/
JSON collision audit with all 52 visible CSV/TSV files across research/, docs/,
scripts/ and artifact history. All 242 selection/fixture seeds are clear. The
supplement is bound into the executable registration inventory; the original
audit's bytes remain unchanged.
Admission also rechecks the current committed retention-bank seed union; its
bytes join the source inventory. This covers retention additions merged after
the original audit without changing or replacing any Round 0 world.

## Ratification after A5, before data

The operator places a real, immutable receipt at `$T34_ART/A5_RECEIPT.json`:

```json
{
  "schema": "t34-round0-a5/v1",
  "paired_worlds_per_hour": "REPLACE_WITH_MEASURED_NUMBER",
  "active_row_steps_per_second": "REPLACE_WITH_MEASURED_NUMBER_AT_LEAST_3000",
  "two_lane_rehearsal_complete": true,
  "measured_export_bytes_per_pair": "REPLACE_WITH_MEASURED_NUMBER",
  "evidence_sha256": "REPLACE_WITH_64_HEX_DIGEST_OF_REAL_A5_EVIDENCE"
}
```

The strings above deliberately fail validation; use actual JSON numbers and the
real evidence digest. Measure the actual paired entrypoint's throughput and
setup/export/cleanup time. The receipt must describe the frozen recipe and a clean
two-lane rehearsal; lane 3 requires that rehearsal first. Keep its raw evidence
with the receipt. Unknown/missing evidence holds dispatch.

A5 is a separately authorized operator measurement, supplied as an input to this
package. This package does not authorize or expose an unratified L4 benchmark
mode. A teacher-only A5 worlds/hour number must not be passed off as paired
throughput: the operator must account for bare evaluation and metric regeneration
using excluded development measurements, with the recipe and derivation in the
hash-bound evidence. If those measurements are unavailable, preparation stays on
hold; running selection worlds to manufacture A5 would violate the registration.

With clean committed source and reviewed evidence, the delegated operator runs:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0
export T34_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export T34_ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-round0-20261020
export T34_A6_RECORD="$T34_ART/a6-02/VERIFICATION.json"
: "${T34_OPERATOR:?Set the approving operator name}"
"$T34_PY" -m research.t34_round0.registration --operator "$T34_OPERATOR" \
  --a5 "$T34_ART/A5_RECEIPT.json" --preflight-record "$T34_A6_RECORD" \
  --out "$T34_ART/RATIFICATION.json"
```

Ratification creates a sibling receipt; it does not edit the draft or claim a
scientific result. It binds the full source commit/inventory, exact preregistration,
bank/audit, A6 and A5. Prepare immutable manifests with measured setup minutes and
the reviewed account endpoint-to-positive-rate map. `{}` is valid only when the
operator has verified there are no other account assignments. Start with one lane;
the generator supports up to three minus existing assignments.

```bash
: "${T34_A5_SETUP_MINUTES:?Set measured setup/export/cleanup minutes per chunk}"
: "${T34_OTHER_ASSIGNMENTS:?Set reviewed JSON endpoint-to-rate map}"
"$T34_PY" -m research.t34_round0.make_manifest \
  --a5 "$T34_ART/A5_RECEIPT.json" --ratification "$T34_ART/RATIFICATION.json" \
  --commit "$(git rev-parse HEAD)" --lanes 1 \
  --other-assignments "$T34_OTHER_ASSIGNMENTS" --setup-minutes "$T34_A5_SETUP_MINUTES" \
  --output-root "$T34_ART/dispatch-01" --out-dir "$T34_ART/dispatch-01/manifests"
cat "$T34_ART/dispatch-01/manifests/FORECAST.json"
```

Every manifest is locally validated by the supervisor. No A5 measurement means
no manifest package. Each chunk has estimate ≤180 min, timeout 360 min, L4-only,
2.0 units/hour, 12.0 operational units plus 2.4 reserve = 14.4 units. For K chunks,
worst reserved envelope = 14.4K units. Expected units price all pairs and setup;
the last chunk uses its actual count. Review this forecast and available account
credits **before** dispatch. The original ~16-unit planning estimate is not an
approved substitute for the computed worst case. No retry is automatic.

The generator also exercises the supervisor's local bounded source-archive
preparation with the exact ratified file inventory. Production requires a Git
checkout of the pinned commit. If GitHub delivery fails and the supervisor installs
its sparse fallback without Git metadata, this package refuses before gameplay;
fallback execution would need a separately verified source-identity adapter.

## Validate and dispatch (operator only)

The following commands are the later ratified run, never part of implementation.
Publish the pinned source before allocation. Keep the Mac on AC with the lid open;
the foreground launcher applies the guards and caffeinate. The loop is serial,
so it uses one lane; to add lanes prepare a new prospective plan with reviewed
account assignments and at most three in total. Use unique state directories.

```bash
export T34_LAUNCH=/Users/josenunez/Projects/ml/snake-dqn-round0/research/colab_cli/launch.sh
export T34_COLAB=/Users/josenunez/.local/bin/colab
for T34_JOB in "$T34_ART"/dispatch-01/manifests/t34-round0-c*.json; do
  T34_JOB_ID="$(basename "$T34_JOB" .json)"
  "$T34_LAUNCH" --check --state-dir "$T34_ART/state-01/$T34_JOB_ID" \
    --colab "$T34_COLAB" validate "$T34_JOB"
done
```

After ratification, review of the forecast, and explicit dispatch authorization:

```bash
for T34_JOB in "$T34_ART"/dispatch-01/manifests/t34-round0-c*.json; do
  T34_JOB_ID="$(basename "$T34_JOB" .json)"
  "$T34_LAUNCH" --state-dir "$T34_ART/state-01/$T34_JOB_ID" \
    --colab "$T34_COLAB" run "$T34_JOB" || break
done
```

Do not read outcomes mid-run. Monitor operational progress, caps, errors and
immutable hashes only. Success requires each terminal supervisor receipt with
`status=PASS`, `stop_verified=true`, hash-verified exports and no matching account
assignment. A client exit code or a scheduled task is not execution evidence.

## Mac replay and the only analysis look

After every scientific chunk completes, freeze the imported outputs. The chunk
directories are `$T34_ART/dispatch-01/<job-id>/exports/round0`, with each supervisor
receipt and original manifest two levels above. The analyzer checks this complete
provenance chain. After Phase 0 frees capacity, run the 10% replay on the Mac with
the same one-thread caps as A6. Supply every chunk, including no-data failed chunks
when preserving fault evidence; never omit an unfavorable pair.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-round0
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu
T34_CHUNKS=("$T34_ART"/dispatch-01/t34-round0-c*/exports/round0)
SNAKE_CKPT_DIR=/Users/josenunez/Projects/ml/snake-dqn/saved_snakes \
  "$T34_PY" -m research.t34_round0.replay \
  --bank research/t34_round0/BANK_ROUND0.json --audit research/t34_round0/SEED_AUDIT_ROUND0.json \
  --ratification "$T34_ART/RATIFICATION.json" \
  --checkpoint /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --chunk-dirs "${T34_CHUNKS[@]}" --out "$T34_ART/MAC_REPLAY.json"
"$T34_PY" -m research.t34_round0.analyze --chunks "${T34_CHUNKS[@]}" \
  --bank research/t34_round0/BANK_ROUND0.json --audit research/t34_round0/SEED_AUDIT_ROUND0.json \
  --ratification "$T34_ART/RATIFICATION.json" --replay "$T34_ART/MAC_REPLAY.json" \
  --out "$T34_ART/DECISION.json"
```

Replay selects first 10 scripted / 7 frozen / 7 mixed worlds, both arms, solely by
the frozen bank order. Any divergence is reported; no drops/replacements. A GO is
development premise evidence only. Strict gating, product-contract tests, serving
qualification and release remain separate authorized steps.
