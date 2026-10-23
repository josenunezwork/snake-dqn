# Acceptance 1792762502: elapsed control-path timeouts, no read-count defect

Verdict: **L4 FIRST_FAILED_ATTEMPT; no read-operation-count sizing fix.**
CPU's standard acceptance arm reports PASS. L4 reports INCOMPLETE, and the
combined acceptance is not qualified. This is an operational diagnosis, not a
new qualification look, fault waiver, or permission to dispatch another attempt.
No executable transport, caps, protections, counting rules or past evidence
are changed by this diagnostic commit.

## Evidence

Acceptance root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/control-rpc-stall-requalification-20261023/requalify-p6wss65b/acceptance-1792762502`.
Its manifest pins `a18bf6b4f606f023bfd7c035e68a48e64d7f7457`. The investigation
starts from runner branch head `865b9ce6` without changing its executable bytes.

| Artifact relative to acceptance root | SHA256 |
| --- | --- |
| `acceptance-result.json` | `a441372fd5ba8ab824a6bbbee28defcec54c42d8789765a3e4eaf503e6989262` |
| `outputs/l4/receipt.json` | `78fae4b656429f034c1a95d41bb6a9bc8614acd963b3230f5044169bb31ee20b` |
| `outputs/cpu/receipt.json` | `c4528d37a249bd0688d778f5a464c9678d4edbc08515857fb74f56749325e26d` |

The L4 receipt's zero-based `cli_calls` records supply direct control-path evidence:

| Index | Evidence | Interpretation |
| ---: | --- | --- |
| 58 | `CONTROL_RPC_TIMEOUT`; 30.013058167 s elapsed against 29.999986625 s; request `1f91df004dc24083b1d55a61f2b2f7bb` | Real bounded exec timeout; index 59 subsequently completes under the same request ID in 2.338153916 s. Recovery does not erase the fault. |
| 108, 109 | Completed execs taking 14.405958500 s and 14.386045208 s | Slow replies below their limits; these completed calls alone are not faults. |
| 116 | Lease upload completes in 31.536221042 s within its 59.999995708 s allowance | Nearby Contents latency; no claim of a lease fault from this record. |
| 117 | `CONTROL_RPC_TIMEOUT`; 30.006209542 s elapsed against 29.999997709 s; request `954d47ea5c2747e0a109e9017ca8d81d` | Second real bounded exec timeout, without a later exec recovery in the receipt. |
| 119, 120 | Stop and account sessions read complete | Cleanup verified; this cannot turn the attempt into a pass. |

Both failed execs have normal telemetry code SHA256
`215cd22cf01f5c19ad5853eece74397239ea3fafa1783d810eb204b7c9d386d9`.
`AcceptanceObserver.__call__` runs the L4 transfer-ready wait / final-artifact
download / watchdog path; the silence injection is in the separate CPU path
(`research/colab_cli/acceptance_faults.py`). Neither L4 timeout is an expected
silence injection. CPU's explicit silence-test timeout has the different code
hash `6a1d3577b125bd06e67f9829b901cc5a64a64e87c3ddedef1b2329903dfdc181`.

L4's terminal fields are `error = GuardError('read helper operation budget
exhausted')`, `failure_class = SUPERVISOR`, elapsed 520.607787750 s,
`stop_verified = true`, and `cleanup_error = null`. The generic SUPERVISOR
category does not overwrite the two specific call-level timeout classifications.
There is also a failed pre-runtime upload at index 7; it is retained as recorded,
and this diagnosis introduces no new exemption for it.

The supervisor console log is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/requal-stallfix.log`
(SHA256 `d2b785028c612a0513d651fd478da8a6513f139d224febdacb45a23ffd4463cd`).
Lines 1–3 report CPU progress and successful completion. Lines 4–5 report
`REFUSE: acceptance failed; qualification registry unchanged` and `EXIT=1`.
It contains no L4 helper traceback, GET duration, or provider error. The L4
receipt, rather than an absent console traceback, establishes the two timeouts.
No stronger child/provider/tunnel attribution is justified by the saved logs.

## Budget arithmetic and mechanism

There is **no cumulative read-operation counter** in `cli_guard._run_read_helper`.
Each invocation creates a new monotonic deadline, then performs at most three
idempotent helper attempts sharing that deadline, including backoff. The literal
error is raised when this invocation's time is exhausted; it is not an Nth-read
refusal. The function chains its last error, but the receipt only saves the outer
representation, so the historical helper's exact underlying exception is absent.

`CLI.poll_rpc` invokes `kernel_idle` after an uncertain exec, passing
`min(policy.RPC_SECONDS, remaining operation time)`. Therefore the arithmetic is:

- Exec attempt: **at most 30 s**.
- Idle helper: **min(150 s helper ceiling, 30 s caller allowance, remaining
  telemetry/session/priced time) = at most 30 s**, renewed independently on
  each helper invocation. A first helper attempt consuming all 30 s leaves
  **30 - 30 = 0 s** for a retry; the guard then refuses. Backoff cannot add time.
- Telemetry recovery: **300 s total** across exec attempts, marker polls, idle
  probes and jitter, narrowed by session and price. The number of exec recovery
  attempts remains three. An unknown kernel remains fenced.
- This L4 manifest: **30 min × 60 = 1,800 s** wall cap; its **1.0 u / 2.0 u/h
  × 3,600 = 1,800 s** priced cap. The supervisor keeps its unchanged 45 s
  cleanup reserve, giving an operational wall deadline at **1,800 - 45 =
  1,755 s**. Recorded elapsed **520.6078 < 1,755 < 1,800 s**.
  Session cap exhaustion is not the terminal cause established here.
- C1's later 160-min loop profile has **160 × 60 = 9,600 s** manifest wall cap
  and **9,600 - 45 = 9,555 s** operational wall deadline. It still
  gives each idle helper at most 30 s, not 9,600 s or an accumulated read quota.

`policy.MAX_REQUEST_POLLS = 300` is a separate **per-request** marker-poll
ceiling. L4 has 62 exec records over 30 request IDs (27 IDs with two polls, and
one each with four, three and one). Its maximum observed four polls is below
300. L4's complete census is 122 calls: 62 exec, 19 sessions, 19 usage, 19 upload,
one version, one new, one stop. CPU passed with **136 calls and 73 execs**.
Those observed counts cannot consume a cumulative guard quota that does not exist.

The code path and terminal error are consistent with an idle-read time budget
being consumed after the second exec timeout. Its exact historical duration and
last exception are **inferred, not logged**. The new regression deterministically
reproduces this path with a helper that consumes 30 s; it does not assert that
this duration was measured in the original attempt. The direct evidence for
failure is the two non-injected control exec timeouts. No read-count budget is
re-derived or increased, and no timeout, fence, request identity, lease, retry,
wall/unit cap, release check or frozen counting protection is removed.

## Counted-or-not under section 7

Authority: `big-lever/round-24/JUDGMENT_PLACEMENT_MAC.md`, section 7 C1.3(3)
and C1.4, not the old budget-blocked paragraph in the original stall-fix handoff.

**A guard budget refusal is a counted fault.** C1.3 counts every failed,
exhausted, fenced or timed-out call of any class, recovered or not. A locally
enforced elapsed-time refusal is an exhausted/failed helper operation; it is
not a clean observation or a ruled carve-out. Even a genuine local sizing bug
would not acquire a new exemption by being called packaging or SUPERVISOR.
The generic receipt category and missing separate helper-call record limit
exact fault cardinality, not platform failure classification. Do not invent a
precise aggregate count by double-counting the terminal failure and its cause.

Independently, indices 58 and 117 each already satisfy the frozen timeout rule.
Under C1.4's explicit acceptance prerequisite this is **L4's first failed
attempt**. CPU's reported PASS is independent and this L4 failure does not fail CPU.
The six-session profile has not passed. A second L4 failed attempt closes that
platform's Colab label lane for October. A second attempt requires a fresh plan
and namespaces at the same envelope. At most one intervening mechanical fix is
allowed and any such fix needs Codex-V review; the text does not require a fix
before a same-code second attempt. This diagnostic commit supplies **no transport
fix** or live pass, and no second attempt is launched here. An unchanged-source
rerun still risks closing the L4 lane if the observed control latency recurs.

## Verification and operator commands

`tests/fixtures/colab_control_faults_acceptance_1792762502.json` preserves all
122 calls' operation/timing/status/request fields and binds the original receipt
hash; exception argv is omitted. The tests replay those counts, show fresh
per-read deadlines beyond the L4 and CPU observed counts, preserve the recovered
fault in the frozen census, and reproduce a timed-out exec followed by helper
time exhaustion with the kernel fence and original wall/price caps intact.

Light offline checks, no allocation:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-stallfix
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest -q -p no:xdist \
  tests/test_colab_cli_acceptance_stall_diagnosis.py \
  tests/test_colab_cli_guard.py tests/test_colab_cli_rpc_recovery.py \
  tests/test_colab_cli_latency_policy.py
```

The following are the existing requalification commands for an **eligible fresh
second-attempt plan**, preserving the current exact prerequisite pin. They are
not an instruction to rerun the consumed failed plan, and are not executed here.
The operator supplies the actually current, reviewed account map (empty only
when the account is empty); preserve all old manifests, outputs and receipts.

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-stallfix
BROKER=/Users/josenunez/Projects/ml/snake-dqn-stallfix-acceptance2
# Fresh, clean original transport pin; never reset the working branch.
git -C "$REPO" worktree add --detach "$BROKER" \
  a18bf6b4f606f023bfd7c035e68a48e64d7f7457
ACCOUNT_MAP=/absolute/path/to/reviewed-current-account-rates.json
STATE_PARENT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/control-rpc-stall-second-attempt-20261023
"$REPO/tools/ops/requalify.sh" plan --broker "$BROKER" \
  --other-assignments "$ACCOUNT_MAP" --state-parent "$STATE_PARENT"

# Review the newly printed plan and committed source; use its exact fresh path.
PLAN=/absolute/path/printed/by/plan/ops-review.json
"$REPO/tools/ops/requalify.sh" execute --plan "$PLAN" \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The existing six-session admission has two source/acceptance constraints to
resolve before calling it dispatch-ready:

- `research/rpc_qualification/admission.py:require_prerequisite` hard-pins
  `TRANSPORT_PIN` to `a18bf6b4...`; planning directly at this diagnostic commit's
  HEAD would yield a different commit and be refused, even though transport
  bytes are unchanged. The detached original-pin broker above preserves that
  binding. A future executable remedy must update its binding through review.
- The same function calls frozen `transport_faults` on every acceptance receipt
  without distinguishing expected CPU acceptance injections. This observed CPU
  arm reports PASS but has intentional download-rejection and silence-test
  failures, so the current prerequisite checker would refuse it too. This is
  a separate existing admission limitation, not evidence for a cumulative quota
  or an exemption for L4. Its resolution needs protocol/source review and must
  preserve the zero-real-fault rule; this commit makes no admission edits.

After a source-bound final-only prerequisite with zero counted faults passes
and those admission constraints are satisfied, the operator can prepare a fresh
six-session pack using the primary venv's
`python -m research.rpc_qualification.manifests prepare --out <fresh-absolute-pack>`
from this worktree. Review its generated `DISPATCH.txt` and Codex-V receipt before
dispatch. Preserve the C1 envelopes, ledgers, injection separation and account
cap; no production or F1-bis admission follows from this offline diagnosis.
F1-bis remains no earlier than 2026-10-26.
