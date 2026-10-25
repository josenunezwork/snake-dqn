# F4 D1024 transport-repair requalification

Status: **NOT QUALIFIED.** This package prepares the one source-bound
`final-only` acceptance that the operator must run by Sun 2026-10-25 06:00 PT.
It creates no PASS record, does not allocate a runtime by itself, and does not
authorize a D1024 canary or main dispatch.

The acceptance remains the established final-only profile: its 4.0-unit total
envelope and 0.8-unit reserve, CPU/L4 arms, watchdog/stop/export gates, and
verified 64 MiB L4 final-artifact download are unchanged. The repaired source
also must stage the synthetic D1024 inventory within the existing bars:

| Input role | Transport | Bytes |
| --- | --- | ---: |
| `b8.pth` | HTTPS public input | 6,562,005 |
| `factory-verified-ledger.json` | verified Contents upload | 2,405,362 |
| `train-1024.npz` | verified Contents upload | 45,397,643 |
| `metadata.tar` | verified Contents upload | 2,211,840 |

The three local files are sparse deterministic synthetic bytes created in the
fresh acceptance state directory. They do not read, copy, hash, or expose F4
payload bytes. `b8.pth` retains the D1024 manifest's public HTTPS role and
pinned public digest. This checks the whole input count, sizes, and transport
roles without interpreting science outcomes. The D1024 contract digest is
`b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea`.
The normal acceptance probe and wrapper are additional transport fixtures, so
the acceptance's total staged-file count is conservatively greater than the
four-row D1024 inventory; the required synthetic subset remains exact.

The stage remains qualified only when its measured `upload-measurement.json`
reports `transfer_seconds <= 150`, `elapsed_seconds <= 300`, and a finite,
nonnegative `source_delivery_seconds` whose components do not exceed elapsed
time. The acceptance operator also checks receipt records for every synthetic
local input (`remote` under `/inputs/`, verified status, expected byte count and
digest). A source-delivery delay, directory setup, renewal, verification, or
control RPC remains part of the 300-second staging elapsed clock; none is moved
outside the evidence.

The acceptance code records `d1024-synthetic-full-inventory-staging` as a
required L4 gate. It refuses a PASS unless each local synthetic file has a
receipt-backed verified upload with its local SHA-256 and exact byte count, the
B8 HTTPS role remains manifest-bound, and its public-input preflight exists.
The consumer verifier also calls the complete shared final-only admission check,
so all ordinary safety gates, both clean stop receipts, and the verified 64 MiB
final artifact remain required alongside this new gate.

## Operator procedure

Use the reviewed repair commit in a clean dedicated checkout. Do not alter
`research/colab_cli` between plan creation, review, execution, and source-bound
record publication. The account-rate JSON is a reviewed snapshot of every
existing assignment; `{}` is valid only when the account actually has none.

Before qualification, canary, or main upload staging, the operator must drain
D4096 at a life boundary and ensure no other supervisor is staging. Resume
collection after staging completes or the attempt terminates, subject to the
ordinary guards. Census and R-A retain exclusive Mac priority. This handoff
does not stop collection or operate its lease.

```sh
REPO=/absolute/path/to/snake-dqn-transport-repair
PY="$REPO/venv/bin/python"
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
STATE_PARENT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/f4-d1024-transport-requalification-20261025

cd "$REPO"
git status --short
git rev-parse HEAD HEAD:research/colab_cli

# Run the reviewed offline test/lint selection first. This step has no Colab calls.
nice -n 15 env OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OMP_THREAD_LIMIT=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  SNAKE_DQN_DEVICE=cpu "$PY" -m pytest -q -p no:xdist \
  tests/test_colab_cli_*.py

# Planning creates a fresh, reviewable final-only plan and synthetic inputs only.
"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" --state-parent "$STATE_PARENT" \
  --d1024-inventory
PLAN=/absolute/path/printed/by/the/plan/ops-review.json

# After independent review of that exact plan, execute exactly once.
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The operator independently reviews the result, both broker receipts, all
required final-only gate observations, synthetic-input receipt rows, timing
measurements, and verified backend disappearance. A failed, interrupted, or
late run stays immutable and is a **HOLD**. It returns to the judge; this
package grants neither an automatic retry nor a substitute platform.

`tools/ops/requalify.py` publishes the mutable qualification registry only after
the actual `acceptance-result.json` is `PASS`, `qualified: true`, and
`return_code: 0`. Its registry includes the acceptance-result SHA-256,
qualified commit, and CLI tree. The acceptance result adds the same source
identity plus the D1024 synthetic-inventory contract. No operator may construct
or edit a PASS record from current hashes.

## Consumer gate

The D1024 package/runbook must call this before making a new canary or main
package:

```sh
"$PY" -m research.colab_cli.acceptance --repo "$REPO" \
  --verify-qualified-record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json \
  --required-inventory-sha256 b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea \
  --reject-tree d0d92bdb572b770284a1fea6152417ffa5f06bea
```

The verifier rejects a missing/tampered receipt, a non-PASS result, copied
source hashes, an acceptance commit whose CLI tree differs from the record, a
current tree/source mismatch, an omitted/wrong synthetic inventory, and the
previous D1024 tree. It deliberately permits commits that differ outside
`research/colab_cli`, provided their CLI subtree is exactly the accepted one.
Only after this verifier and the independent acceptance review pass may the
separate D1024 package prepare the fresh `03` canary; that canary must then
meet the same unchanged 150/300-second bars before a newly sealed main path.

The package also retains its existing publication requirement: the selected
source commit must be on `origin/main`. A pushed repair branch alone does not
authorize make-q or dispatch. If the complete test gate prevents merging, keep
the source and acceptance plan reviewable on the branch and report HOLD; do
not bypass the publication check.

After publication and actual reviewed acceptance, use the create-only attempt:

```sh
export SOURCE="$REPO"
export PIN=$(git -C "$SOURCE" rev-parse HEAD)
# Set this to the already published, independently verified final-union digest.
export D1024_UNION_SHA='<verified-final-union-sha256>'
export F4_D1024_ATTEMPT=03
RUN="$REPO/research/bl_residual/RUN_D1024_FIT_20261024.sh"
"$RUN" make-q
"$RUN" preflight
"$RUN" prepare
"$RUN" upload-check
"$RUN" canary-dry-run
# Operator, after independent review and the drain/staging checks:
"$RUN" canary
# Only after timing PASS and verified stop of that exact canary:
"$RUN" seal-main
"$RUN" fit-dry-run
# Operator dispatches the registered main once:
"$RUN" fit
```

No command in this sequence was executed against the real F4 payloads during
the transport repair. The prior failed attempts and their liabilities remain
in the series; attempt `03` never overwrites either earlier package.
