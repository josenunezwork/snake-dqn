# F4 D1024 prospective transfer exception and attempt-04 runbook

Status: implementation and offline preparation only; **live qualification HOLD**.
This sibling update supersedes the transfer-bar and projection instructions in
F4_THROUGHPUT_REPAIR_20261024.md and F4_D1024_TRANSPORT_REQUALIFICATION.md for
this prospective D1024 chain only. Prior acceptance-1792897885 stays REFUSE.

Authority: Astra JUDGMENT_F4_D1024_PROSPECTIVE_TRANSFER_EXCEPTION_20261024.md,
SHA256 `f57080bdc08b6f0dc439adb0082ea5e00725d0cfd2521fdaffde6b864f1cc809`.
The operator recorded the ruling in CAMPAIGN_LOG at 20:45 PT, before execution.
Independent source/hash ACCEPT and matching plan review remain required.

The shared policy binds the exact F4 D1024 inventory, main/canary modes, dose
1024, L4/CUDA and immutable production hashes (including the metadata archive
that binds the frozen recipe). Its acceptance context permits only the exact
synthetic inventory in the final-only L4 acceptance arm. CPU acceptance and
other jobs have no exception. Missing authority grants no exception; foreign,
malformed, or mismatched declarations refuse. Consumers rederive the policy.

| Clock | Default | This authority only |
| --- | ---: | ---: |
| Verified transfer | 150 s | base 150 / effective 300 s |
| Whole staging | 300 s | 300 s, independently enforced |

The unchanged staging clock includes parent setup, transfers, source delivery,
renewals, credential rotation, verification, cleanup and recovery. No work or
clock is moved or reset. Per-operation bounds, lease/watchdog, timeout,
accounting, mutation/replay refusal, credentials, exports and verified stops
are unchanged. Transfer/source components must still fit within elapsed time.

Historical replacement acceptance observed transfer **221.68512525002006 s**
and staging **235.52283416606952 s**. The old approximately 110-second forecast
is obsolete and diagnostic only. These observations show feasibility and do
not qualify a future run. Actual fresh acceptance and canary must pass both
independent gates. Normal canary/main still need successful terminal, export,
receipt and stop proof; the acceptance-only intentional watchdog exception
cannot be reused for a production job.

Before execution, the operator must verify from metadata that no D1024 main
fit or R-A has started and no terminal exclusion was sealed. Otherwise HOLD to
the judge without opening outcomes. Drain D4096 at completed-life boundaries,
verify quiescence, exclude competing staging for acceptance and every later
staging window, and refresh account/rate admission. This task performs none
of those live actions. Pinned running lanes and prior evidence stay intact.

## Fresh source-bound plan and exactly one acceptance

The delivered empty account map is **REVIEW_ONLY**, not current account evidence.
The plan is NOT_RUN with CPU20/L4-30 envelopes and the unchanged synthetic full
inventory. If the operator's current reviewed assignments differ from `{}`,
create a fresh plan using that reviewed map, obtain independent review, and
leave the delivered plan immutable. Do not execute the empty-map plan against
a nonempty account. No automatic repeat is authorized.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-transport-repair
PY="$REPO/venv/bin/python"
AUTH=f57080bdc08b6f0dc439adb0082ea5e00725d0cfd2521fdaffde6b864f1cc809
# Use only if a new plan is required for the actual reviewed account map:
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent /Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/f4/transfer-exception-20261024-01/operator-plan \
  --d1024-inventory --transfer-authority-sha256 "$AUTH"

# Set PLAN to the exact independently reviewed ops-review.json.
PLAN=/absolute/path/to/independently-reviewed/ops-review.json
# Operator only: after pre-fit metadata, drain/quiescence, account and source review:
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED

# Only after independent actual PASS, both verified stops and registry publication:
"$PY" -m research.colab_cli.acceptance --repo "$REPO" \
  --verify-qualified-record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json \
  --required-inventory-sha256 b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea \
  --required-transfer-authority-sha256 "$AUTH" \
  --reject-tree d0d92bdb572b770284a1fea6152417ffa5f06bea
```

## Attempt-04: the existing single canary and main grant

No additional canary/main retry is granted. Attempt 04 names fresh create-only
paths; it does not count as an additional permission. Keep exact data/recipe,
seed 3403, fp32, 2,000 updates, sampler and final checkpoint. L4/CUDA is fixed.
The original scientific sample sizes, thresholds, R-A/R-B looks and strikes
branch are unchanged. No Mac fit, alternate GPU, Drive, public label sharing,
splitting or recompression is authorized.

```sh
export SOURCE=/Users/josenunez/Projects/ml/snake-dqn-transport-repair
export PIN=$(git -C "$SOURCE" rev-parse HEAD)
export D1024_UNION_SHA=f5ec94ccdfae90133362a9072227d83125ae0ce661c44471904ce57333b610b8
export F4_D1024_ATTEMPT=04
RUN="$SOURCE/research/bl_residual/RUN_D1024_FIT_20261024.sh"
"$RUN" make-q
"$RUN" preflight
"$RUN" prepare
"$RUN" upload-check
"$RUN" canary-dry-run
# Independent packet review; operator verifies a drained/exclusive staging window:
"$RUN" canary
# Only after actual transfer <=300, staging <=300, every gate and verified stop:
"$RUN" seal-main
"$RUN" fit-dry-run
# Independent packet review and a fresh drained/exclusive staging window:
"$RUN" fit
"$RUN" publish-fit
"$RUN" verify-fit-publication
# Operator establishes exclusive Mac priority for census and R-A:
export CENSUS_PIN=$(git -C "$SOURCE" rev-parse HEAD)
"$RUN" census
"$RUN" publish-census
"$RUN" verify-census
export RA_PIN=$(git -C "$SOURCE" rev-parse HEAD)
"$RUN" ra-prepare
# Independent packet review, then print and execute the registered command immediately:
"$RUN" ra-command
```

Qualification target: Sun 2026-10-25 06:00 PT; escalate by 04:00 if the path is
no longer credible, and immediately on any failed gate. Fit/publication target
noon; census/packet target 14:00. R-A starts immediately when lawful prerequisites
are complete and no later than **Sun 10-25 18:00 PT**. These targets expand no cap.
F2 volume_struck=true retains R6.3 STRIKES: mean retention improvement >=0.05,
parent-bootstrap LB90>0 and D1024 H=0. If no lawful recovery meets the deadline,
record unrecoverable R-A HOLD and apply B6 within a completed-life boundary:
create-only d4096-not-run-by-rule.json, preserve/exclude the sealed tranche,
never fit/publish/open it under F4; downstream NOT_RUN_BY_RULE. A transport
failure is not scientific KILL or STOP_FUNDING. No sealed look/exclusion reopens.
Release and serving still require owner approval.
