> **2026-10-29 owner-authorized offline re-qualification:** Unit caps, prices,
> balances, reserves and rate drift are informational warnings, including a
> depleted balance; provider allocation decides availability. Historical unit
> limits below describe prior evidence only. Offline fake-provider qualification
> is authorized for this change; no live acceptance is claimed. The mutable ops
> registry pins the new transport tree with `independent_review_status: PENDING`;
> dispatch refuses until independent review sets `ACCEPT`. Prior live evidence
> remains historical and does not qualify these changed bytes.

# Kernel notebook UUID correction — 2026-10-22

Status: **NOT QUALIFIED; fresh final-only requalification required.**
This sibling supersedes the operator handoff in `KERNEL_READINESS_REQUALIFICATION.md`.
The failed acceptances, existing receipts, qualification registry and old P1 packs
remain immutable. Publish this correction only to
`supervisor-kernel-ambiguity-20261022`; do not merge it to main before requalification.

## Root cause and evidence

Acceptance root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-readiness-requalification-20261022/requalify-4wxbp7pk/acceptance-1792737517`.
The CPU arm at `af326ab3ffdf5bfdfdebb83dc6a95be1fd2adf38` ended INCOMPLETE,
failure class SUPERVISOR, with one failed exec (0.303217083 seconds), no
Contents transfers, no exports, no terminal workload, verified stop, no cleanup
error, and zero remaining assignments. L4 was not reached.

The sibling lane's `sessions.assignment.json` retains notebook identity
`8a64755c-57f4-43d1-92d9-59159b2cb7a0`. Production `CLI.__init__` creates
`uuid.uuid4().hex`, and `_call_once` explicitly exports that 32-character value
to both allocation and exec subprocesses. `install_hooks` parses it with
`uuid.UUID`, then `guarded_assign` writes `str(desired_hash)` (hyphenated) to
the durable record before returning. The first kernel POST compared the
hyphenated record against the hex environment string. These are the same UUID
but unequal strings. This is a deterministic formatting defect, rather than
missing export or a record written before the identity exists. The failed
receipt does not contain the original exec environment; the source and offline
reproduction establish that path without claiming the environment was logged.

## Correction and regression

`cli_guard._same_notebook_uuid` requires string inputs and parses both as UUIDs
before comparing identity. Missing, malformed and foreign values refuse. The
endpoint, URL, session/config and proxy-token checks are unchanged, as are the
saved kernel-ID fence and the process-local uncertain-POST fence. No retry,
allocation, kernel deletion, cap or scientific change is added.

`tests/test_colab_cli_guard.py::test_assignment_then_first_exec_replays_cpu_uuid_refusal`
uses the exact retained UUID, production `CLI.call` / `_call_once`, the generated
sitecustomize file, and separate subprocesses for new, first exec and duplicate
exec. `tests/fixtures/colab_identity_replay.py` replaces provider/network and
StateStore dependencies with offline stand-ins. It verifies the hook writes the
hyphenated record, the hex environment passes first creation after correction,
the created ID is durable, and a fresh process cannot issue a duplicate POST.
It also covers a hyphenated environment. Restoring the old string comparison
produced **1 failed (hex), 1 passed (hyphenated)** with the recorded terminal
GuardError; the correction passes both. The additional malformed/missing/foreign
UUID selection passes. These offline proofs do not grant live qualification.

Independent Codex-V receipt (outside the qualified source tree):
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-uuid-20261022/codex-v-review.json`.

## Requalification commands

Use the final published branch tip, clean and unchanged between plan and execute.
The old plan is consumed. Prepare a fresh reviewed endpoint-to-positive-rate map
from current account inventory; an empty map is valid only when no assignments
are live. Preserve final-only, all eleven gates, and the existing **4.0-unit total
with 0.8-unit reserve**. Do not treat offline verification as a passing acceptance.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn-kernelamb
cd "$REPO"
git fetch origin
git status --short
git rev-parse HEAD origin/supervisor-kernel-ambiguity-20261022
# Operator reviews current sessions/usage and writes the current rate map.
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
PLAN=$("$REPO/tools/ops/requalify.sh" plan --broker "$REPO" \
  --other-assignments "$ACCOUNT_MAP" \
  --state-parent /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/kernel-uuid-requalification-20261022)
cat "$PLAN"
# After reviewing this fresh saved plan and its bound manifests:
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Require CPU and L4 PASS, exact source identity, all eleven observed gates,
verified exports and verified backend disappearance. Only PASS updates the
qualification registry. No live acceptance or production dispatch is performed
by this preparation. The independent P1 retry uses the already-qualified CLI
tree `553af3f692515b2bf66173356fe4d099acc96942`, not this changed transport.
