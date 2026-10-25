# F4 materially corrected transport: prospective attempt 04

Status: **REVIEW ONLY / NOT QUALIFIED**. This correction implements the recovery
condition in `JUDGMENT_F4_REQUALIFICATION_FAILURE_0bd6f058.md`, SHA-256
`2832a35c06983b3667db3e5ef5669d11df83a7f3b6cd4004c36f6dc7c4c78b0c`.
The previous acceptance remains immutable and INCOMPLETE. No scientific result,
qualification registry, payload, fit recipe, placement, threshold, namespace,
look rule, or serving configuration changes here. The required bars remain
**150 seconds verified transfer and 300 seconds whole staging**.

## Bottleneck and material correction

The failed L4 measurement was 207.348055665032 seconds transfer and
220.22179129195865 seconds staging. Receipt SHA-256:
`8bb428854e51f727f564d085a2c7c9a3cd1cfeabd56d9eb9d23cebc9083f0e3c`.
Measurement SHA-256:
`714b430f2e26ede64ae913cfcbd0b465d14297cb1e2ce7d7119ee81e964e7b08`.
The 45,397,643-byte synthetic input took 169.245689 seconds: 82.339631 seconds
of serial PUT calls, 51.585162 seconds of foreground renewal/recovery, and
30.840436 seconds of cleanup calls. The renewal total already contains the
nested recovery-idle interval. Background heartbeats overlap foreground work;
their elapsed times must not be added to historical transfer time.

The corrected transport:

- Retains 2 MiB PUTs and the official CLI. Exactly two workers upload distinct,
  immutable chunk files per window. No chunk-size/API assumption is changed.
  Every worker explicitly inherits the shared staging deadline. Workers never
  renew through kernel RPCs, and every worker finishes before assembly/cleanup.
- Renews during staging with the existing generation-fenced Contents heartbeat,
  throttled by the same 30-second coordinator clock. The independent watchdog
  heartbeat, credential rotation, retained-assignment checks, forced state
  reads, session/account limits and first-cap-wins deadlines remain. Outside
  staging, ordinary kernel renewal remains unchanged.
- Checks each chunk SHA-256 and the complete digest/size, fsyncs and atomically
  publishes, then removes UUID parts in the same single-dispatch assembly.
  The existing durable acknowledgement/idle-proof recovery remains mandatory;
  uncertain assembly is never blindly replayed. Failure cleanup remains bounded
  and cannot race a late worker PUT. Small files retain remote digest verification.
- Records synthetic staging errors separately from release/calibration evidence.
  A genuine 150-second refusal remains INCOMPLETE with its actual exception;
  valid provider calibration is no longer overwritten with a missing-rate claim.

## Offline evidence and limits

`tests/fixtures/f4_upload_latency_0bd6f058.json` contains only receipt metadata:
all seven upload sizes/times, their PUT latencies, control-operation totals and
the five heartbeat latencies. No private payload bytes are included.
`tests/test_colab_cli_upload_throughput.py` reads the frozen reviewed serial
implementation from Git object `0bd6f058a34afb5817ff30b61c2a333ad9758fd4`
and executes both implementations on full-size deterministic synthetic files
through a fake CLI filesystem. Serial call/control costs are charged from the
receipt; the two-worker model charges the maximum saved PUT duration per pair.
The 39.505134-second outlier remains in the model. Both implementations perform
real local hashing and assembly; modeled network waits use virtual time.

| Receipt-latency scenario | Transfer seconds | Staging seconds |
| --- | ---: | ---: |
| Reviewed serial transport replay | 207.348056 | 220.221791 |
| Corrected path before replacement Contents renewals | 89.160950 | 102.033713 |
| Corrected path with two due measured Contents renewals | 91.199103 | 104.071867 |
| Corrected path plus an additional entire worst observed heartbeat | 128.225664 | 141.098428 |

The final row adds 37.026561 seconds to the corrected replay, including its
normal renewals. It is an allowance scenario, **not a guaranteed upper bound**.
Actual contention, credential rotation or slower calls can still fail closed.
In particular a foreground staging control is capped at 30 seconds; that
37-second heartbeat would fail if encountered as a foreground call. No timing
is moved outside the transfer/staging wrappers. These tests do not qualify
the source. Only the fresh complete CPU/L4 acceptance can do that.

The focused suite also checks worker joining on failure, chunk/full digest
refusal, atomic publication, shared worker deadlines, source mutation, lease
loss, staging renewal selection, and the exact failed measurement on normal
and watchdog acceptance paths. Broader test logs and baseline exceptions must
accompany the source-bound review packet.

## Fresh acceptance and attempt-04 runbook

Prepare the full-inventory final-only plan from the clean published corrected
commit using `tools/ops/requalify.py plan --d1024-inventory`. The review packet's
empty account map is explicitly **REVIEW_ONLY**, not current account evidence.
The operator must verify all actual assignments/rates and obtain independent
review of the matching frozen map/plan. If the map differs, prepare a fresh
plan for review before any execution. Never edit a saved plan or manifest.

Keep CPU 20-minute and L4 30-minute envelopes, all safety/export/release gates,
and the exact synthetic inventory. Before any acceptance or subsequent upload
staging, the operator drains D4096 at a completed-life boundary and excludes
other heavy/staging work. This implementation does not operate collection.
The corrected source and plan need Astra ACCEPT before the one fresh complete
acceptance. Failure returns to the judge with cleanup evidence; no automatic
retry or diagnosis-only replay is authorized.

After actual acceptance PASS, independently review the full result, timing,
source bindings, both verified stops and qualification registry. The following
commands are **operator-only instructions, not actions performed by this task**:

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-transport-repair
PY="$REPO/venv/bin/python"
PLAN=/absolute/path/to/independently-reviewed/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED

# Only after reviewed PASS and publication of this corrected source on main:
"$PY" -m research.colab_cli.acceptance --repo "$REPO" \
  --verify-qualified-record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json \
  --required-inventory-sha256 b9eca552d1dc72349e7e303859a87da6317ce0abc0d5635dcecfb9a7151c72ea \
  --reject-tree d0d92bdb572b770284a1fea6152417ffa5f06bea

export SOURCE="$REPO"
export PIN=$(git -C "$SOURCE" rev-parse HEAD)
export D1024_UNION_SHA='<already-published-verified-final-union-sha256>'
export F4_D1024_ATTEMPT=04
RUN="$REPO/research/bl_residual/RUN_D1024_FIT_20261024.sh"
"$RUN" make-q
"$RUN" preflight
"$RUN" prepare
"$RUN" upload-check
"$RUN" canary-dry-run
# Independent review; operator excludes other staging/heavy work, then one canary:
"$RUN" canary
# Only after that exact canary's actual timing PASS and verified stop:
"$RUN" seal-main
"$RUN" fit-dry-run
# Independent review, then the one registered fit:
"$RUN" fit
"$RUN" publish-fit
"$RUN" verify-fit-publication
# After operator establishes exclusive Mac census/R-A priority:
export CENSUS_PIN=$(git -C "$SOURCE" rev-parse HEAD)
"$RUN" census
"$RUN" publish-census
"$RUN" verify-census
export RA_PIN=$(git -C "$SOURCE" rev-parse HEAD)
"$RUN" ra-prepare
"$RUN" ra-command  # prints the registered look command; does not execute it
```

Attempt `04` is a fresh create-only artifact suffix, not a fourth canary grant.
It yields `f4/d1024-pack-04`, `d1024-preflight-04.json`,
`d1024-canary-state-04`, `d1024-fit-state-04`, and the dedicated
`snake-dqn-f4-d1024-qualified-04.noindex` checkout. It does not consume or expand
the ruling's conditional pending canary permission. Prior evidence stays intact.

Qualification target is Sun 2026-10-25 06:00 PT; escalate at 04:00 if no credible
path. Immutable fit/publication is due noon, census/R-A packet 14:00, and the
existing F2/R-A trigger/cutoff is 18:00. Missing the lawful prerequisite chain
returns to the judge and the registered HOLD/NOT_RUN_BY_RULE/discard procedure;
it never authorizes a late or substitute look. Census/R-A require exclusive Mac
priority. Release/serving changes remain owner-only.
