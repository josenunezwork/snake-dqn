# Colab CLI acceptance qualification

This is the mandatory, bounded qualification before routing a new research job
through `research.colab_cli`. It is not a study run, does not alter the Q1 Drive
queue, and does not become a pass until its create-only records exist and
receive independent review.

There are two acceptance profiles. The CLI defaults to `final-only`. Every plan
and execution result records the selected profile and the current source
identity it is bound to; a PASS from a different source identity does not admit
a later source revision.

| Profile | When it is required | Qualification boundary |
| --- | --- | --- |
| `final-only` (default) | S1 and terminal-artifact-only jobs | Requires all operational safety gates plus a measured, hash-verified download of one exactly 64 MiB final artifact. It does not qualify snapshots or resume. |
| `full-state` | Before a job uses a mid-run full-state snapshot or claims exact resume | Requires the shared operational safety gates and the 600 MiB busy-snapshot/client-loss, exact-resume, and checkpoint-overhead gates. |

S1 exports only terminal weights and receipts, with a 64 MiB per-attempt bound;
it has no mid-run full-state snapshot and makes no exact-resume claim. Its
pre-allocation admission validator requires a `final-only` PASS bound to the
current source. The S1 manifest must upload that acceptance result as the
hash-bound input destination `cli-acceptance-result.json`. A prepared plan, a
`NOT_REQUIRED` full-state row, a different-profile result, or a PASS bound to
another source does not admit S1.

## Caps and account boundary

Use the supplied 2026-10-18 18:38 PT account context only as a preparation
baseline: 1,174.37 compute units, aggregate usage 0.00 units/hour, and zero
active assignments. It is not a current measurement. Before execution the
supervisor queries `colab --auth oauth2 usage` and `sessions`. Every assignment
already present must appear in a reviewed endpoint-to-rate map (or a current
broker record); an unknown endpoint is a refusal. The account-wide maximum
remains three assignments.

The whole qualification has a hard cap of 4.0 units, including a 0.8-unit
(20%) reserve; only 3.2 units are usable. The first wall-clock or unit cap
reached stops the relevant job. CPU is at most 20 minutes with a conservative
priced upper bound of 0.5 units/hour by default; L4 is at most 30 minutes with
a conservative priced upper bound of 2.0 units/hour by default. The
`--cpu-rate`/`--cpu-rate-units-per-hour` and
`--l4-rate`/`--l4-rate-units-per-hour` options override those *pricing
ceilings* before allocation. They are used for admission, liability
reservation, and the hard session cap, never replaced by a lower provider
observation. There are no retries, purchases, replacement allocations, or
automatic CPU/L4 fallback.

After each allocation, the supervisor performs one bounded provider-rate
calibration: up to 120 seconds, three stable positive reads, five seconds
apart, with stability within 0.01 units/hour. It subtracts the reconciled rate
of every other accounted assignment from the provider account rate, allowing
provider lag. The resulting observed rate must be at or below the priced
ceiling plus 0.05 units/hour. A missing, nonpositive, unstable, unexplained,
or over-ceiling observation stops the arm and leaves qualification
`INCOMPLETE`. Once calibration passes, later provider-rate drift is checked
against the observed rate, while pricing and caps remain bound to the
conservative priced ceiling.

## Prepare only

Preparation creates unique absolute output directories, pinned manifests,
input hashes, and a `NOT_RUN` plan. It never invokes `colab`. Use the profile
explicitly when preparing review material, even though `final-only` is the
default.

```sh
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn/research/colab_cli/launch.sh
"$LAUNCH" acceptance \
  --profile final-only \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance
```

The default plan prices CPU as `20 / 60 * 0.5` and L4 as
`30 / 60 * 2.0 = 1.0`; both must fit the usable 3.2 units. Use a reviewed
`--other-assignments /absolute/path/other-assignments.json` only when current
`sessions` is nonempty; that JSON must list every pre-existing endpoint and its
reviewed positive rate. Re-run preparation rather than reuse any manifest or
output directory.

No remote duration or debit was measured by preparation. A Mac preflight on
2026-10-18 measured three real units of 1,000 deterministic toy updates at
0.021951, 0.022130, and 0.022009 seconds with one worker. These are transport
fixtures, not M3-C training estimates.

## Execute after independent review only

After Q1 is complete and an independent reviewer accepts the generated plan,
the operator may execute the reviewed `final-only` profile. This calls the
supervisor sequentially, first CPU then L4, using `new`/`exec`/`download`/`stop`;
it does not use `drivemount`, runtime `auth`, or browser clicks.

```sh
"$LAUNCH" acceptance \
  --profile final-only \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance \
  --execute-after-review \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The command writes a create-only `acceptance-result.json` beside the initial
`acceptance-plan.json`; the plan is never rewritten. A process exit of zero is
not evidence of kernel success.

### `final-only` required gates

| Gate | Required evidence | Pass condition |
| --- | --- | --- |
| Accounted concurrent spend | `account-assignment-spend` ledger | Every pre-existing endpoint has a reviewed positive rate; the receipt separates acceptance elapsed-rate charge from the measured other-assignment baseline. |
| Rate calibration | CPU and L4 calibration samples and receipt fields | Stable positive, baseline-subtracted observed rate is within the priced ceiling plus tolerance; caps remain the priced ceiling. |
| Fresh OAuth refresh | New Terminal `whoami` and `usage` after token refresh | Reusable OAuth works without a browser prompt. |
| Silence timeout | Small job has a gap greater than 60 seconds and the supervisor records an explicit timeout | Timeout is bounded and does not blindly resubmit execution. |
| Browser attachment | `sessions` before and after `url` | Same assignment; `url` creates no second allocation. |
| Client loss/reconnect | Small-job generation fence, remote job identity, and one completion record | Reconnect observes the existing job without duplicate work. |
| Failure and corrupt export | Intentional child-exit receipt and corrupt-download rejection record | Failure is terminal and a wrong hash never becomes a local artifact. |
| Final artifact download | Exactly 64 MiB immutable final artifact, SHA-256 sidecar, completion marker, verified local hash, and recorded bytes/second lower bound | The complete final-sized artifact downloads, verifies, and records the throughput lower bound used for S1 admission. |
| Mac-loss watchdog unassign | L4 lease expiry, checkpoint/grace evidence, and backend endpoint disappearance within 300 seconds | Simulated Mac renewal loss causes the VM-side release path to checkpoint and unassign before broker fallback cleanup; a `?` label or trainer exit alone fails. |
| Backend stop disappearance | `stop`, then independent `sessions`/`status` observation | The exact assignment disappears; a successful CLI exit alone is insufficient. |

The `final-only` receipt must record `busy-full-snapshot`, `exact-resume`,
and `checkpoint-overhead` as `NOT_REQUIRED`, not as `PASS`. This profile
neither uploads a 600 MiB snapshot nor establishes full-state transport,
trainer resume, or checkpoint-overhead behavior.

### `full-state` extension

Run `--profile full-state` before a job uses mid-run full-state snapshots or
asserts exact resume. It retains the shared operational safety gates and
requires the 600 MiB busy-snapshot/client-loss transfer, an uninterrupted
versus resumed full-state hash comparison (optimizer, replay, actors, target,
RNG, and counters; weights alone are insufficient), and the paired
fixed-count checkpoint-overhead measurement at or below 2%. The exactly-64-MiB
final-artifact download is the S1 `final-only` admission measurement and is
`NOT_REQUIRED` in `full-state`. Missing required evidence makes the profile
`INCOMPLETE`.

## Supplied acceptance evidence: run 6 is INCOMPLETE

Live acceptance run 6 used source commit `fe0fd684` and directory
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792384051`.
It ended `INCOMPLETE` after 1,157 seconds during the busy 600 MiB
full-snapshot/client-loss phase: `acceptance_faults.py:447` (`_wait_file`)
raised `Refused('kernel completion file not observed; refusing exec resubmission')`.
Observed gates were account-assignment-spend, fresh-oauth-refresh,
silence-timeout, url-no-second-allocation, failure-and-corrupt-export, and
`stop_verified: true`. Small uploads achieved only about 2--8 kB/s effective
throughput (about 1.3--1.7 seconds per few-KiB file), where per-call overhead
dominated.

These observations do not qualify either profile. In particular, run 6 did
not PASS `final-only`, did not measure the required 64 MiB final-download
throughput lower bound, and cannot admit S1.

When a completion file is absent, the refusal now identifies the remote file,
the named acceptance phase, and the elapsed wait before it declines to
resubmit a possibly busy kernel operation. The diagnostic does not turn an
uncertain completion into success.

## Supplied acceptance evidence: run 7 is INCOMPLETE

Run 7 used source `84eedcba`, profile `final-only`, and directory
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792394444`.
Its `acceptance-result.json` and CPU/L4 `receipt.json` files show CPU **PASS**
with every CPU gate observed and stop verified. L4 was **INCOMPLETE** after
259 seconds of its 30-minute cap, with stop verified and
`Refused('deadline reached during telemetry recovery')`. A heartbeat upload
took 41.5 seconds, followed by an exec that consumed its full 120-second
timeout. The probe never reached `transfer_ready`; no final download was
attempted, and both final-artifact-download and mac-loss-watchdog-unassign
remain INCOMPLETE.

The code confirms that telemetry formerly shared one 120-second allowance
between the exec attempt and recovery: a full exec timeout left no time for
independent kernel-idle proof, regardless of the manifest lease. Telemetry
now has a separate **300-second total operation budget**, with each exec
still bounded to **120 seconds**, at most three attempts, and independent
idle proof required before resubmission. Explicit shorter caller budgets,
whole-session wall limits, and priced unit limits still win.
Per-attempt deadlines are passed directly to exec rather than temporarily
narrowing the shared session deadline, so they cannot starve independent
Contents heartbeat uploads. Contents heartbeats continue through transport
stalls; the L4 **180-second lease is unchanged**, including the deliberate
Mac-loss exercise, and its
VM-side checkpoint and assignment-disappearance gate still requires release
within **300 seconds after lease expiry**, before broker fallback cleanup.
This avoids extending actual Mac-loss spending to solve a read-recovery
budget problem. No unit cap, allocation count, or qualification gate changes.
Offline Fake-Colab regression coverage does not qualify a live acceptance
run or admit S1; run 7 remains INCOMPLETE.

Any missing artifact, incomplete provider-rate calibration, unaccounted
session, unexplained provider-rate change, timeout ambiguity, backend
persistence, cleanup error, or supervisor feature that cannot produce the
listed evidence is `INCOMPLETE` and fails the selected profile.
