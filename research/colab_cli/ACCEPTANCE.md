# Colab CLI acceptance qualification

This is the mandatory, bounded qualification before routing a new research job through `research.colab_cli`. It implements the operating-model advice dated 2026-10-18. It is not a study run, does not alter the Q1 Drive queue, and does not constitute a live pass until the records below exist and are independently reviewed.

Use the supplied 2026-10-18 account context only as a budget baseline: 1,178.30 compute units, aggregate usage 6.84 units/hour, and two active notebook assignments. It is not a current measurement. Before execution the supervisor queries `colab --auth oauth2 usage` and `sessions`. Every assignment already present must appear in a reviewed endpoint-to-rate map (or a current broker record); an unknown endpoint is a refusal. The receipt records the session-attributed acceptance spend and the priced baseline of those other assignments, cross-checked against the account balance delta. The account-wide maximum remains three assignments.

The whole qualification has a hard cap of 4.0 units, including a 0.8-unit (20%) reserve; only 3.2 units are usable. The first wall-clock or unit cap reached stops the relevant job. CPU is at most 20 minutes and is admitted only after its unit rate is measured. L4 is at most 30 minutes at the provisional 2.0 units/hour rate (1.0 unit). There are no retries, purchases, replacement allocations, or automatic CPU/L4 fallback. If CPU expected cost plus 1.0 unit exceeds 3.2 units, do not execute.

## Prepare only

This creates unique, absolute output directories, pinned manifests, input hashes, and a `NOT_RUN` plan. It never invokes `colab`.

```sh
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn/research/colab_cli/launch.sh
# JSON object: {"<pre-existing endpoint>": <reviewed units/hour>, ...}
OTHER_ASSIGNMENTS=/absolute/path/other-assignments.json
"$LAUNCH" acceptance \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance \
  --cpu-rate-units-per-hour MEASURED_CPU_RATE \
  --other-assignments "$OTHER_ASSIGNMENTS"
```

Unknown CPU price blocks manifest creation. `other-assignments.json` may be `{}` only when `sessions` is empty before allocation; it must list every pre-existing session endpoint otherwise. The generated plan records expected CPU cost as `20 / 60 * MEASURED_CPU_RATE`, L4 as `30 / 60 * 2.0 = 1.0`, and both must fit the usable 3.2 units. Re-run preparation rather than reuse any manifest or output directory.

The probe calibrates a fixed deterministic synthetic useful-work loop, then runs the same fixed count before and after the 600 MiB immutable snapshot/export handoff. `checkpoint-overhead.json` records calibration samples, fixed iteration count, baseline, second pass, snapshot-write time, and computed overhead. This measures transport behavior only; it does not establish trainer throughput or exact trainer resume. The expected remote cost is CPU `MEASURED_CPU_RATE / 3` plus L4 `1.0` unit, and the measured synthetic overhead must be at most 2% for the acceptance gate to pass.

## Step costs and timing evidence

No remote duration or debit was measured by this implementation task. A Mac preflight on 2026-10-18 measured
three real units of 1,000 deterministic toy updates at **0.021951, 0.022130, and 0.022009 seconds**, one worker.
The runtime repeats that calibration on its actual CPU and uses the same fixed update count for both
approximately 90-second passes. These are transport fixtures, not M3-C training estimates.

For every billed step record `seconds * admitted_rate / 3600`; `r` below is the known CPU ceiling. These expected
planning allowances are not measurements or additional caps. Overlapping work is charged once. Setup, deliberate
idle time, export tails, and cleanup all count against the arm's whole-session cap. Exhaustion fails qualification
without another allocation.

| Step | Expected allowance | CPU expected units | L4 expected units | Measurement artifact |
| --- | ---: | ---: | ---: | --- |
| Mac archive/hash validation | Before allocation | 0 | 0 | `input-hashes.json` |
| Allocation, Python/wheel/dependency setup | Up to 8 min | `8*r/60` | 0.267 | usage/elapsed and bootstrap log |
| Calibration + fixed baseline | About 90 s plus calibration | `r/40` plus calibration | 0.050 plus calibration | `checkpoint-overhead.json` |
| Snapshot creation, paired work, busy transfer/client loss | About 120 s, overlapping | `r/30` | 0.067 | observer timings, hashes, transfer ack |
| Silent kernel execution | 65 s, overlaps baseline | `65*r/3600` | 0.036 | observer timeout/elapsed |
| Separate silent child fixture | 65 s | `65*r/3600` | 0.036 | `silence.json` |
| Child exit, imported toy resume, final verification | About 30 s | `r/120` | 0.017 | child/resume/receipt artifacts |
| L4 lease-loss checkpoint and unassignment | About 201 s; L4 only | Not applicable | 0.112 | verified checkpoint and backend absence |
| CPU whole-session maximum | 20 min | **`r/3`** | Not applicable | receipt and account debit |
| L4 whole-session maximum | 30 min | Not applicable | **1.0** | receipt and account debit |

The CPU acceptance lease is 300 seconds for the intentional 120-second busy-kernel/client-loss exercise. The L4
fault arm uses a separate 180-second lease. The script does not change either running notebook worker's lifetime.
OAuth refresh occurs in memory on the Mac. The watchdog receives only a short-lived, narrowly scoped release
credential generated for that runtime; the owner's refresh token and token file are never uploaded or rewritten.

## Execute after review only

After Q1 is complete and an independent reviewer accepts the generated plan, the operator may run the same command with the exact review marker below. This calls the supervisor sequentially, first CPU then L4, using `new/exec/download/stop`; it does not use `drivemount`, runtime `auth`, or browser clicks.

```sh
"$LAUNCH" acceptance \
  --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance \
  --cpu-rate-units-per-hour MEASURED_CPU_RATE \
  --other-assignments "$OTHER_ASSIGNMENTS" \
  --execute-after-review --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

The probe pauses at its hash-bound transfer-ready state while the Mac performs the bounded one-way snapshot export handoff, then resumes the same measured loop. That one export is the paired checkpoint-overhead measurement. The later full upload/download roundtrip is recorded separately as transport-integrity evidence, outside that overhead gate. It separately exports a toy full-state checkpoint; the runtime-side resume entrypoint reloads it and continues the remaining deterministic steps for an independent hash comparison. The command writes a create-only `acceptance-result.json` beside the initial `acceptance-plan.json`; the plan is never rewritten. A process exit of zero is not evidence of a kernel success.

| Gate | Required evidence | Pass condition |
| --- | --- | --- |
| Fresh credentials | New Terminal `whoami` and `usage` after token refresh | Reusable OAuth works without a browser prompt. |
| Busy full snapshot transfer | One-way handoff timing plus separately timed full roundtrip, with local SHA-256 verification and atomic rename | The paired overhead covers only the required export; roundtrip hashes match while the kernel has a supervised child. |
| Silence timeout | Probe has a gap greater than 60 seconds and supervisor records an explicit timeout | Timeout is bounded and does not blindly resubmit execution. |
| Browser attachment | `sessions` before and after `url` | Same assignment; `url` creates no second allocation. |
| Client loss/reconnect | Generation fence, remote job identity, and one completion record | Reconnect observes the existing job without duplicate work. |
| Failure/corrupt export | Intentional child exit receipt and corrupt-download rejection record | Failure is terminal and wrong hash never becomes a local artifact. |
| Exact resume | Full-state hash for uninterrupted and resumed runs | Optimizer, replay, actors, target, RNG, and counters match; weights alone are insufficient. |
| Accounted concurrent spend | `account-assignment-spend` ledger | Every pre-existing endpoint has a reviewed positive rate; receipt separates acceptance elapsed-rate charge from the measured other-assignment baseline. |
| Mac loss watchdog | Exported checkpoint during the 15-second grace plus backend endpoint disappearance within 300 seconds | Simulated Mac renewal loss leaves the VM-side release path to checkpoint and unassign. The exact endpoint must disappear before broker fallback cleanup; a `?` label or trainer exit alone fails. |
| Checkpoint overhead | Paired fixed-count synthetic-work baseline and transfer pass | Measured paired transport overhead is at most 2%; this is not a trainer-throughput claim. |
| Backend release | `stop`, then independent `sessions`/`status` observation | Assignment disappears; a successful CLI exit alone is insufficient. |

Any missing artifact, unmeasured CPU rate, unaccounted session, mismatch, timeout ambiguity, backend persistence, cleanup error, or supervisor feature that cannot produce the listed evidence is `INCOMPLETE` and fails qualification. Do not route research workloads through the CLI until every row is verified.
