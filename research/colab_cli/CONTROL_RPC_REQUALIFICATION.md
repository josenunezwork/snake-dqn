# Control-RPC stall fix: requalification handoff

## Outcome and boundary

This branch changes executable `research/colab_cli` control-path behavior. It is
**not qualified** until a fresh, source-bound requalification passes. Do not merge
it, dispatch science, or replace the qualified-transport record from the offline
test result alone.

This handoff is for the Colab operator and the independent Codex-V reviewer. It
assumes they can inspect a clean Git checkout, prepare the current
endpoint-to-positive-rate map, and operate the approved `tools/ops` wrapper. It
produces either a source-bound requalification record or a failure record; it
does not authorize F1-bis or change its frozen placement/counting rule.

The observed F1 decision recorded six CPU and three L4 `exec` timeouts at about
120 seconds. Those commands recovered, but the ruling says a recovered fault is
still counted. It also records one distinct L4 `exec` failure at 36.74 seconds;
the available decision record does not establish its child/provider cause. The
review retains it as an unresolved non-timeout control failure; the historical
receipt did not save its exception. Future `CONTROL_RPC_ERROR` call records
include sanitized exception details. The B2 transport investigation located the
costly stall in a payload-free kernel `exec` after the checkpoint chunks had already
arrived, rather than in checkpoint bandwidth.

The fix being reviewed uses short control RPC attempts, a durable request
identity, and runtime `.started`/`.done` markers. After an uncertain response,
the supervisor may re-issue only the same request ID, and only after independent
idle proof. A completed marker returns its saved result. A started-without-done
marker means the command is still pending: bounded polls may observe its done
marker, but expiry fails closed and never replays the command body. Receipt and
call evidence must classify these failures as `CONTROL_RPC_TIMEOUT` or
`CONTROL_RPC_ERROR`, separately from data-path failures.

The F1 rule remains frozen:

- Every control-path stall, including a recovered timeout, is a counted fault.
- A platform passes the new requalification only with zero counted faults of any
  class and a successful injected-retry proof showing that no unit ran twice.
- F1-bis uses the frozen F1 ruler, fresh output/state directories, its existing
  25% CPU-to-L4 placement rule, and the same stall rule. It may start no earlier
  than Monday, October 26.

The controlling ruling is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-24/JUDGMENT_PLACEMENT_MAC.md`
section 4. The transport diagnosis is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-b2/JUDGMENT_B2_CANARY_TRANSPORT.md`.

## Offline review gate

Run these light, single-process regressions at the final committed branch SHA.
They exercise the observed 120-second timeout fixture, request-ID marker
handling, independent-idle recovery, and receipt fault classification. The
commands are checks only; they make no Colab allocation.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-stallfix
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
cd "$REPO"
git fetch origin
git status --short
git rev-parse HEAD origin/ops-rpc-stall-fix-20261023
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  "$PY" -m pytest -q -p no:xdist \
  tests/test_colab_cli_rpc_recovery.py \
  tests/test_colab_cli_fault_properties.py \
  tests/test_colab_cli_supervisor.py \
  tests/test_colab_cli_warmup.py \
  tests/test_colab_cli_dispatch_warmup.py \
  tests/test_colab_cli_latency_policy.py \
  tests/test_ops_qualification.py
git diff --check
```

Before reviewing a live plan, require an empty `git status --short`, the local
`HEAD` equal to `origin/ops-rpc-stall-fix-20261023`, and source-manifest review
at that exact SHA. Do not change any executable transport byte after the plan is
saved. The reviewer must inspect the test evidence that a lost response causes
one request identity, `.started` precedes the command, `.done` is published
only after the saved result, and the retry executes the command body once.

## Existing wrapper: exact final-only requalification commands

`tools/ops/requalify.sh` is the supported wrapper for the current normal
`final-only` acceptance. It creates a review-only plan and consumes it exactly
once at execution. A failed or interrupted execution requires a new plan; never
rerun its execution marker. The account map must include every currently listed
assignment at its reviewed positive rate. `{}` is valid only for an actually
empty account.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-stallfix
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
STATE_PARENT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/control-rpc-stall-requalification-20261023

"$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent "$STATE_PARENT"

# Inspect the printed plan, its exact manifests, source inventory, account map,
# rate/cap calculations, and fresh output directories. Then use that exact file.
PLAN=/absolute/path/printed/by/plan/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The wrapper's existing `final-only` plan is a necessary supervisor acceptance,
not the ruling's F1-bank requalification. Its implementation builds one CPU arm
and one L4 arm under the historical final-only envelope; it does not create six
F1 sessions, inject a request retry into an F1 unit, or decide the F1 counted
fault census. A wrapper PASS alone therefore does not admit F1-bis.

For a PASS, independently review `acceptance-result.json`, both platform
receipts, every call record carrying a control-RPC classification, source
inventory, verified export hashes, and backend session disappearance. A process
exit code or a recovered request does not erase a counted fault.

## Required six-session F1-bank qualification

After Codex-V accepts the exact committed source, run the following plan through
a newly prepared, reviewed F1-specific harness:

| Platform | Sessions | Duration | Required verdict evidence |
| --- | ---: | ---: | --- |
| CPU runtime | 3 | at least 60 minutes each | Frozen F1 bank loop, complete per-session fault census, zero counted faults, verified release |
| L4 runtime | 3 | at least 60 minutes each | Frozen F1 bank loop, complete per-session fault census, zero counted faults, verified release |

Each session needs a unique job ID, state directory, output directory,
allocation namespace, immutable receipt, and its own verified ledger. Keep all
six raw call/fault records. Attach the deterministic injected-response-loss
regression as a separate proof record, retaining request and unit identity: one
`.started`, one `.done`, one committed result, and one execution count. The six
clean sessions can additionally re-issue a completed request ID without injecting
a transport failure. An intentional transport fault must never be hidden or
waived inside their zero-fault census: an injected or real recovered fault there
still fails the platform. Any extra live injection session needs its own priced
manifest inside the same envelope.

The budget is per platform: base cap at most 2 units including reserve, then the
owner's multiplier may be applied once for an effective cap at most 4 units
including reserve. The first wall or unit cap still wins. Record base and
effective envelopes before the first allocation; do not double measured rates,
the bank, or the retry maximum, and do not raise a cap mid-series.

Current repository materials cannot execute that plan yet:

1. `tools/ops/requalify.py` only constructs the regular one-CPU/one-L4
   `final-only` acceptance; it has no `3 + 3` F1-loop profile or injected-retry
   result checker.
2. The available F1 package is explicitly HOLD. Its CPU family probe is 40
   minutes while the qualified family-canary gate accepts no more than 30
   minutes, as recorded in
   `research/bl_labels/F1_REFUND_HANDOFF_20261023.md` and the frozen
   `f1-pack-10/DISPATCH.txt`.
3. The held F1 L4 template prices its ordinary ruler at 1.54 units/hour. Three
   one-hour L4 sessions cost 4.62 units before reserve. That exceeds the
   ruling's 4-unit effective aggregate platform cap if “per platform” means the
   stated three-session series, so the held template cannot be reused. The
   ruling does not say that the cap is per session; a Fable/judge clarification
   is required before any plan treats it that way.

Consequently, there is no supported, executable F1 dispatch command in this
checkout. The operator must not use the reserved pack-10 dispatch lines or edit
their caps. A new reviewed F1 requalification manifest/harness must first bind
the frozen bank, six fresh namespaces, the fault census, the duplicate-unit
assertion, and a rate/cap envelope that actually fits the ruling. Until then,
the six-session stage is **BLOCKED_NOT_PREPARED**, not a failed or waived
qualification.

## F1-bis after a clean requalification

Only after the new six-session package passes and the independent reviewer
records the source-bound evidence may the operator prepare F1-bis. It must use
the frozen bank digest
`a8c1dc4bbe1fd579d59b69d6ca6ca9dbd2814b088f74277f008f504f6d1bbb0d`,
new output/state paths, the frozen ruler, and the existing analyzer's full
attempt/fault census. Include the old attempts only as their recorded classes;
do not reclassify the historical recovered timeouts as benign. A clean platform
still joins production only after the separate measured-Mac-equivalence gate
specified by the ruling.

No current command is supplied for F1-bis because the required F1-specific
requalification harness is absent and pack-10 is HOLD. This document does not
amend a sealed F1 manifest, create a new allocation, or authorize a merge.

## Sources checked

- `research/colab_cli/REQUALIFICATION.md`, `ACCEPTANCE.md`, and
  `tools/ops/requalify.py` / `requalify.sh` for the supported plan/execute
  interface and its source-bound, one-use behavior.
- `research/bl_labels/F1_HANDOFF.md`, `F1_REFUND_HANDOFF_20261023.md`,
  `f1_manifests.py`, and `f1_ruler.py` for the frozen-bank identity, existing
  held package, session shape, and present harness limitation.
- The F1 decision record and the two external ruling/diagnosis files named
  above for the observed timeout counts, fault-counting rule, six-session
  requirement, caps, and F1-bis timing.
