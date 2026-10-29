# Ops tooling

All operational state lives in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops`.
`OPS_DIR` overrides that location for isolated tests. Logs and receipts are retained;
a successful shell exit alone does not prove a completed job or backend release.
The examples below use `REPO=/Users/josenunez/Projects/ml/snake-dqn`.

## Dispatch a Colab job

```bash
"$REPO/tools/ops/dispatch.sh" NAME /absolute/job.json \
  --broker "$REPO" --cwd "$REPO" --state-dir /absolute/fresh-state \
  --dispatch-txt /absolute/DISPATCH.txt
```

`dispatch.sh` refuses BLOCK/HOLD/"do not run" instructions, false scientific admission,
list-typed `budget.other_assignments`, reused names, and failed validation. It checks
that the broker's and manifest commit's `research/colab_cli` Git trees match the
qualification registry and
refuses staged, unstaged, or untracked transport changes. Unrelated commits do not
invalidate the registry. Dispatch preserves a create-only manifest snapshot before
validation and launch, so concurrent edits to the source manifest cannot change the
job being monitored. The broker remains responsible for scientific admission,
canaries, usage measurements, leases, verified exports, and release.

Dispatch serializes account allocation using `ops/alloc.lock` with POSIX `flock`
(via Python `fcntl`, so the macOS host needs no external `flock` binary). The lock
covers launch through allocation success/failure, with a maximum ten-minute hold.
Immediately before each launch (including the one authorized allocation retry),
dispatch reads `colab --auth oauth2 sessions` under that lock and joins every live
endpoint to `~/.config/colab-cli/snake/active-*.json`. A manifest declaration alone
cannot establish broker ownership. Unknown endpoints, ambiguous records, or unregistered hardware refuse launch.
Missing calibrated prices and price differences produce structured warnings. L4 is registered at
1.54 u/hour and CPU at 0.08 u/hour; A100 uses its broker-record rate. For records
with a priced ceiling, accounting uses the observed calibrated rate; an unavailable
rate is recorded as unknown with a warning and zero estimated units.

The wrapper replaces only `budget.other_assignments` in a create-only
`ops/manifests/NAME.live.json` sibling, validates that exact sibling, and checks
its hash before submitting it. `ops/dispatch/NAME/ATTEMPT.accounting.json` records
the original and refreshed manifest paths/SHA256, raw inventory, active-record
hashes, and refreshed rates. The original snapshot and source pack remain
unchanged. Departed manifest declarations disappear and new broker-owned endpoints
are included. The supervisor still checks account count, stale broker records, and membership
changes during allocation. Baseline price differences, unit caps and reserves
produce structured `WARNING` lines on stderr and retain usage/spend evidence.
Only a depleted balance (`<= 0`) refuses admission.

An unconfirmed allocation reaches the time limit without authorizing a retry; the
supervisor continues to own cleanup. A submitted job records its `EXIT=` marker
when the supervisor exits; an uncertain tmux submission is labelled in the log.
Logs are registered with `watchd.sh`, which is started if needed.

The sole automatic retry is the exact supervisor accounting failure
`AccountingError('assignment endpoints changed during allocation')`, after the
failed attempt has terminated and its receipt proves cleanup/release. All other
errors, uncertain release, and a second failure stop. The retry uses a saved copy
of the original manifest with a new job ID (bounded to 64 characters) and fresh
output directory; the original
manifest, output, and log remain evidence. Review both attempts when computing spend.
This narrowly authorized retry is in the ops wrapper; transport code is unchanged.
The `resume` declaration does not expand this automatic retry or launch completion
jobs. The package/operator owns the separately registered outcome-blind completion
attempts and accounts for every attempt against the registered series envelope.

## H100-first GPU preference (owner rule, 2026-10-28)

New hardware-generic GPU packets may declare:

```json
"requirements": {
  "gpu": "H100", "allowed_gpus": ["H100"],
  "prefer": ["H100", "A100"], "min_vcpu": 2, "est_minutes": 10
}
```

Price the primary at the provisional **12 u/h H100 ceiling**, with a cap that
covers the **whole timeout** and a timeout at least twice the estimate. The
wrapper derives an **8 u/h A100** sibling (covers 40/80 GB), reducing its cap to
`timeout_minutes * 8 / 60`, with distinct job,
dispatch, state, output, log and receipt identities. It does not enlarge submitted
caps. Both manifests are sealed, reserved and validated before the first launch;
`--dry-run` validates both with zero allocations or name reservations. The future
sandbox A assay, TRAIN-01 and sandbox B/B2 (also used by B3) builders emit this
preference and freeze their new pre-data operational envelopes. Existing sealed
packets and host-bound L4 builders retain their declared hardware.

New builder caps come from each variant's complete timeout and rate, even when
the builder receives a larger old cap. Each recorded base cap is the price of
the base timeout; the effective timeout and cap are twice those base values.
Sandbox A and TRAIN-01 compile native kernels for both `8.0` and `9.0`.
GPU-sprint H100 canaries require a native-compile receipt that includes `9.0`;
historical `8.0;8.9` receipts remain valid only for their supported hardware.

Fallback occurs once, only after the installed 0.7.4 CLI's exact exit-1/412
allocation rejection or the measured H100 assign-POST `ColabRequestError` ending
in `Service Unavailable`/503, supervisor termination, and independent proof that no
endpoint was assigned and no VM compute or transfers occurred. Under the broker's
account lock, dispatch verifies the exact pending/config identity, absence of all
durable endpoint records, a GET-only notebook recovery returning no endpoint, and
unchanged account membership below three assignments. It archives that pending
record in `unavailable.json` before retiring just that proven empty intent. The
supervisor's original incomplete receipt remains unchanged; it does not acquire
an invented `stop_verified` flag. Unknown errors, timeout, 400 entitlement errors,
uncertain recovery, changed inventory, or any endpoint forbid fallback. Preference
packets never use the legacy accounting retry after an endpoint failure.

For a retained pending record for any accelerator from either recognized error, use
`python tools/ops/reconcile_gpu_pending.py --manifest <executed-live-manifest>
--state-dir <state-root> --log <terminated-dispatch-log> --pending <exact-account-pending>
--proof <fresh-archive.json> --broker <qualified-broker> --dry-run`. The preview runs
GET-only recovery and sessions under the same ops/account locks and prints proof;
it leaves the pending record and archive path untouched. Replace `--dry-run` with
`--live` for the operator to repeat every check, create the immutable archive
(including the full pending payload and SHA), then unlink only that pending path.
No allocation, stop, active-record mutation, fallback dispatch, or receipt rewrite
is performed by this command. Any recovered endpoint, inventory drift, other
pending intent, archive collision or evidence mismatch keeps the pending fence.
The assign-POST 503 error must name the exact accelerator in the executed manifest;
an error for another accelerator is refused. The proof records that requested
accelerator. This retirement support does not change dispatch preferences or grant
permission to substitute hardware: a sealed A100-only packet stays A100-only after
retirement. An H100 retry needs a newly sealed and admitted successor with fresh
job/output/state/receipt identities and an H100-priced envelope (and any required
hardware-specific canary evidence); never edit the original packet.
Both automatic fallback reconciliation and this operator tool poll a contended
account broker lock once per second within their remaining deadline (at most
600 seconds including preceding work). They retain the ops-then-account lock
order. Deadline expiry refuses without publishing proof, retiring the pending
record or starting fallback; contention never relaxes any evidence check.
Each pre-admitted GPU variant gets its own allocation/admission deadline of
`OPS_ALLOC_LOCK_TIMEOUT_SECONDS` (default and maximum 600 seconds). The ops lock
stays held across primary reconciliation and sibling admission, with a total bound
of twice that value for the pair. The sibling repeats live accounting and full
validation within its fresh window; it never reuses earlier host preflight proof.
The sandbox validation subprocess still has its separate 180-second maximum,
clipped to the current variant's remaining time. Reconciliation must finish within
the primary's deadline before any sibling window begins. No failure or timeout
within a variant refreshes that variant's deadline.
The operator tool's `--timeout-seconds` may shorten its 600-second total bound;
zero, nonfinite values and values above 600 are refused.
The qualified supervisor refuses **all new allocations** while any pending record
exists, even when fewer than three sessions are active. Historical sealed brokers
remain fenced until this shared account record is reconciled; no packet change
is necessary to remove that account-wide blocker.

`gpu-preference.json` binds both admissions. Each submitted variant publishes a
separate `gpu-receipt.json` after supervisor exit, binding its original receipt,
requested GPU, observed `hardware.json` (when available), calibrated accounting,
and `colab usage` before/after. Allocation success alone is not execution success.
The original supervisor still owns rate calibration, export verification and stop.

Ordinary science mains still need a target-platform family canary for **each**
variant. `requirements.hardware_canaries` may map `H100` and `A100` to their
respective `{path, sha256}` receipt bindings; both undergo unchanged supervisor
admission. No unqualified hardware substitution is inferred from an A100 receipt.
The A1 120-minute/20-unit exception and retained A100 route remain bound to their
hardware-specific authorities; timing/F5, P1 window, small-benchmark, and nice-0
admissions require separate ratification before preference use. This feature does
not revise science, resume namespaces or registered hardware evidence.

Prepare the synthetic acceptance with `python -m research.compute.gpu_acceptance
--commit FULL_SHA --packet ABSOLUTE_FRESH_PACKET`, then use `dispatch.sh --dry-run`
before operator submission. Its two 7.5-minute, 1.5-unit envelopes bound the combined
packet at 15 runtime-minutes and 3 units, including a possible allocation rejection.
It exports NVIDIA name/compute capability, cu126 torch CUDA decision/matmul smoke,
and host accounting rates. These are acceptance forecasts, not measured H100
throughput. The implementation handoff contains the exact prepared command.

## Prepare a fresh allocation after a kernel-creation failure

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python \
  "$REPO/tools/ops/redispatch_fresh.py" /absolute/failed-gpu-sprint-package \
  --broker /absolute/qualified-broker
```

This helper prepares only; it never dispatches, allocates, stops, or retries a
kernel RPC. It accepts only GPU-sprint packages with the exact kernel-creation
guard failure on the first readiness probe, a hash-bound executed manifest,
verified stop, no terminal workload record, no admitted GPU accounting, no
transfers, and no exports or later runtime evidence. Missing or contradictory
proof refuses preparation. Legacy receipts lack a GPU-seconds counter: zero
**useful workload** GPU seconds is derived from this pre-work evidence; elapsed
allocation spend is recorded separately and is not zero billing.

The helper validates the resume declaration, broker/pin transport qualification,
and the real supervisor manifest offline before creating an exclusive sibling.
Only `job_id` and `output_dir` change; pin, inputs (including original paths and
hashes), environment, caps, and scientific namespace stay unchanged. The sibling
contains the selected manifest, `PREPARATION.json`, and `DISPATCH.txt`, each with
SHA and completion sidecars. State and output paths are fresh and remain absent
until operator dispatch. Original package evidence is retained unchanged.

The final printed line is the exact operator dispatch command. `--manifest
BASENAME` selects an attempt when multiple manifests have receipts. `--suffix
fresh-01` chooses an explicit unused suffix; the default combines UTC time with a
random suffix. Existing namespaces refuse reuse. Packages with `backup_dir` or
blocking dispatch instructions require separate operator preparation. This helper
does not authorize workload completion retries or dispatch a main job; a passing
same-pin canary is still required.

## Resume-by-default

[Resume policy ruling (2)](../../docs/research/advice/2026-10-21_big_lever/JUDGMENT_RESUME_POLICY.md#2-standing-requirement--idempotent-completed-units--verified-ledger-resume-skip)
applies to every future Sol Colab package, including development jobs. Each manifest
must declare this top-level block:

```json
{
  "resume": {
    "unit_kind": "episode",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": true,
    "max_outcome_blind_retries": 2
  }
}
```

`unit_kind` names the completed unit (for example, episode or full snapshot).
`ledger_path` identifies the ledger consumed by the job, including an empty ledger
on the original attempt. Both must be nonempty strings; `skip_verified` must be the
JSON boolean `true`, and the retry maximum must be the JSON integer `2`. Ship the
ledger through hash-bound manifest inputs and wire its runtime path into the job's
entrypoint; a Mac path in metadata alone does not make it available remotely.
This block is an admission declaration, not an automatic CLI argument or proof of
resume behavior. Dispatch checks it before qualification, validation, name
reservation, locks, watchers, or submission, and preserves it in the immutable
manifest snapshot and the existing accounting retry.

Every package must pass offline tests before Mac preflight:

- Derive deterministic unit IDs from namespace and frozen allocation index. Export
  immutable units atomically as completed, with `.sha256` and `.complete` published
  last; partial work never counts as a complete unit.
- Consume a Mac-verified ledger of IDs, sizes, and hashes. Skip exactly those units;
  refuse foreign namespaces or allocations and never recompute verified units into
  the result set. Build ledgers from path/size/hash only, without opening outcomes.
- Aggregate the union of verified exports from all attempts on the Mac through a
  registered entrypoint that refuses duplicate IDs and foreign namespaces. Remote
  attempt summaries stay sealed diagnostics. The original one-look or registered
  sequential pause points remain the only scoring looks; completion never changes N.
- Freeze mechanical failure triggers, `stop_verified`, missing-unit checks, at most
  two outcome-blind completion retries (one original plus two completions), and each
  attempt's envelope before data. Use a fresh manifest, output directory, and receipt
  for every attempt; retain prior receipts and verified-export records unchanged.
  Resume full-state trajectories only from the latest verified eligible snapshot.

**Owner's 2× rule:** set all Colab unit caps, reserves, and timeouts to twice the
ruling/forecast values when preparing a new package, before data. Record both base
and effective envelopes. For example, a ruling's 2.5-unit cap, 0.5-unit reserve,
and 90-minute timeout become 5.0 units, 1.0 unit, and 180 minutes. Do not double
measured estimates, rates, N, checkpoint cadence, or the two-completion retry maximum.
Do not edit immutable historical manifests or raise caps mid-series. Dispatch does
not infer a forecast or resize a submitted manifest.

Only explicitly declared short operational canaries or acceptance jobs may omit
`resume`. Instead declare, for example:

```json
{
  "resume_exemption": {
    "kind": "short-canary",
    "reason": "Synthetic skip-ledger transport proof; no scoring look"
  },
  "timeout_minutes": 40
}
```

`kind` must be `short-canary` or `acceptance`, with a nonempty reason and a finite
positive numeric timeout. Short canaries are bounded at 40 minutes (the ruling's
20 minutes ×2); acceptance at 60 minutes (the existing 30-minute maximum ×2).
A1 has one package-specific override authorized by the owner's 2026-10-22 work
order and `PLAN_A100.md` §3: `acceptance`, at most 120 minutes / exactly 20 units,
A100 only, the exact 6/256/1,024 × three-mix, 3 × 8-tick ruler. It must identify
that plan's SHA-256 (`1c87898be554d8e66788da6f0a09ee7b842a514e530c1d0e1c624bcf58c3962c`)
in `resume_exemption.authority_sha256` and use the A1 entrypoint. This does not
change the generic 60-minute ceiling or the qualified transport tree.

The supervisor still enforces identity, concurrency, pending-record, science,
wall-time and operation-timeout checks. Budget fields are optional measurement
thresholds; they never shorten deadlines or stop polling (owner 2026-10-29).
Builders may retain legacy unit/reserve fields for evidence only.
Retained chains likewise enforce wall time independently of unit thresholds.
Never declare both blocks. Names, scripts, and environment flags do not imply an
exemption. A resume canary should itself use `resume` when testing that contract;
an exemption only waives the declaration, not proof of the path under test.

For an already prepared and reviewed package, dispatch its tiny canary first and
wait for a passing terminal receipt, verified exports, and backend release. Its
proof must match current code and the manifest pin before dispatching the main job.
Set `CANARY_MANIFEST`, `MAIN_MANIFEST`, `PACKAGE_CWD`, and `DISPATCH_TXT` to the
reviewed package's absolute paths and `ATTEMPT` to a fresh name:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-ops2
"$REPO/tools/ops/dispatch.sh" "$ATTEMPT-canary" "$CANARY_MANIFEST" \
  --broker "$REPO" --cwd "$PACKAGE_CWD" \
  --state-dir "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/$ATTEMPT-canary" \
  --dispatch-txt "$DISPATCH_TXT"
# Wait for and verify the canary before running this command:
"$REPO/tools/ops/dispatch.sh" "$ATTEMPT-main" "$MAIN_MANIFEST" \
  --broker "$REPO" --cwd "$PACKAGE_CWD" \
  --state-dir "/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/state/$ATTEMPT-main" \
  --dispatch-txt "$DISPATCH_TXT"
```

## Qualification registry and watcher

`ops/colab-cli-qualified.json` is a mutable pointer to the most recently qualified
transport **tree**, separate from immutable acceptance evidence. It was initialized
from the owner's reported final-only PASS `acceptance-1792685115` on
`f1904777aded79366703c94afbf5dc13155bb0ee`, merged at
`04ae4cbfa6194e7afbdff53fe22be9de1a6cd152`. The transport tree is
`40985ca47b3c1f82cdab2ee77849daf937e71865`. The merge commit message corroborates
that result; `research/colab_cli/REQUALIFICATION.md` retains its earlier pending text.
No new acceptance was run to initialize the pointer.

```bash
python3 "$REPO/tools/ops/qualification.py" check \
  --repo "$REPO" --record /Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json
"$REPO/tools/ops/watchd.sh"
"$REPO/tools/ops/wait-event.sh"
```

The persistent `watchd` tmux daemon writes `ops/events.log`: `JOB_ENDED`, `SOL_DONE`,
`COLAB_UNDERUSED` (fewer than three assignments for ten minutes), `MAC_IDLE` (no
Python worker above 50% CPU for fifteen minutes), and `COLAB_CLI_UNQUALIFIED` when
freshly fetched `origin/main` differs from the registry. Missing/invalid registry or
failed Git refresh also emits that qualification event. Unchanged qualification
errors are suppressed until the state changes. `OPS_REPO` selects the checkout to
inspect. Tests can set `OPS_WATCH_ONCE=1 OPS_WATCH_FETCH=0`.

Register other jobs with `printf '%s %s\n' "$PID" name >> "$OPS_DIR/watch-pids.txt"`.
An approved idle gap can suppress underuse until the epoch in
`ops/colab-idle-ok-until`; record the reason in `ops/idle-ok.log`.

## Requalify changed transport

Planning is offline. Execution spends Colab units and must be an explicit operator
step after reviewing the generated plan and current endpoint-to-positive-rate map.
Use a clean broker checkout. The wrapper always chooses `final-only` using the
broker's pinned acceptance envelope. Review the generated plan against the owner's
2× caps/reserves/timeouts rule above before execution; this dispatch-policy change
does not resize the broker's acceptance generator. It does not run full-state acceptance.

```bash
"$REPO/tools/ops/requalify.sh" plan --broker /absolute/clean-broker \
  --other-assignments /absolute/reviewed-account-rates.json
# Review the printed ops-review.json and its manifests. Execute that exact plan:
"$REPO/tools/ops/requalify.sh" execute --plan /absolute/ops-review.json \
  --review-token COLAB_CLI_ACCEPTANCE_REVIEWED
```

Each plan receives a unique state parent, including plans created in the same
second. Execution verifies the pinned commit, source inventory, manifest hashes,
and fresh output paths; an execution marker consumes the attempt before allocation.
A failed/interrupted attempt must be replaced by a **new plan**, never rerun.
Execution uses the broker's existing acceptance implementation, AC/open-lid guards,
and `caffeinate`. On PASS, `require_final_only` and `verify_plan_sources` must both
succeed before atomically updating the registry. The registry binds the result's
SHA-256, commit, tree, and acceptance ID. Keep the independent acceptance review
required by the research process; registry publication grants no serving authority.
`SNAKE_DQN_PYTHON` can select the project interpreter.

## Sum prior spend

```bash
python3 "$REPO/tools/ops/spend.py" /absolute/attempt1/receipt.json /absolute/attempt2/receipt.json
```

The report prints each receipt path, SHA-256, elapsed seconds, own rate, and units,
then a total. Units are exactly `elapsed_seconds * accounting.own_rate_units_per_hour
/ 3600`; aggregate provider debit is not used. Missing/invalid accounting and
duplicate receipts refuse the entire report. This is verified receipt-based elapsed
rate accounting, not a claim about final provider billing.

## Run a pinned Mac job

```bash
"$REPO/tools/ops/dispatch.sh" --mac --repo "$REPO" NAME FULL_COMMIT_SHA -- \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/example/job.py --help
```

The explicit `--mac` route delegates unchanged arguments to `mac_run.sh`;
Colab dispatch keeps its existing admission and accounting checks. The workload
must acquire the shared research lease before doing work.

The runner creates `../snake-dqn-run-NAME.noindex` detached at the resolved commit,
verifies it is clean and pinned, atomically writes `.IN_USE_LOCK`, and launches the
command in tmux with a registered log. Existing targets must be clean, detached,
exactly pinned, and unlocked. The job runs with AC/open-lid guards, `caffeinate`, CPU
by default, and bounded thread defaults. The caller must provide an appropriate
finite job duration and any study-specific Mac resource lease; this wrapper does
not replace research admission or resource scheduling. On exit, only the runner's
own lock is removed. The runner uses a private Git excludes file so the marker
does not dirty historical checkouts; existing global ignore rules and Git config
environment entries are preserved. A lock left after interruption or uncertain tmux
submission requires operator inspection;
never delete an active run's lock. Worktrees and logs are retained.

## Factory companion work

**PREPARED_NOT_QUALIFIED — activation HOLD.** The companion monitor, client/tool
lanes and production continuation currently refuse activation. No qualified
native descendant tracking backend is implemented; manual evidence flags or
environment variables cannot override the fence. See the handoff for unresolved
acceptance requirements. The following interfaces describe a future protocol.

After full qualification, local Astra/Sol commands during an F0
D256/D1024/D4096 production window would use `factory_companion.sh`.
It binds the reviewed concurrency
policy SHA and active factory attempt identity, accepts only reviewed argv forms,
and delegates admission, absolute priority, CPU/thread caps, monitoring and
cleanup to `research.compute.factory_companion`.  The production operator
commands and the exact one-test-lane pytest shape are in
`research/bl_labels/FACTORY_CONCURRENCY_HANDOFF_20261023.md`.  This wrapper does
not acquire the research Mac lease and does not launch a factory job.

The future protocol starts or reuses `factory_companion_monitor.sh` before the
required light-only observation. `start` currently refuses with activation HOLD;
`status` remains available for diagnostics. Its verified-owned `stop` command is
for cleanup. These commands are the
operator interface for the persistent monitor; it is not a factory launcher.

## Offline verification and calendar lint

```bash
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
"$PY" -m pytest -q tests/test_ops_*.py
bash tools/ops/tests/test_dispatch.sh
for script in "$REPO"/tools/ops/*.sh; do bash -n "$script"; done
```

The pytest calendar lint scans `research/**/run_*.sh` and Python launcher files for
hard-coded calendar dates and weekday/time windows. Historical comments and document
references have narrowly recorded exceptions; new executable deadlines must use
relative durations or explicit operator inputs. No historical research files are
rewritten to satisfy lint. Regression tests use temporary repositories, fake tmux
and supervisor commands, and local processes; they do not allocate Colab or run
research workloads.

## Clean import preflight before packing

Use `tools.ops.import_preflight.check(repo, commit, source_paths, requirements,
modules)` in a pack builder **before creating its output directory**. A2 calls
this unconditionally. A0/A1 and A3 builders can use the same function, or invoke
`python3 -m tools.ops.import_preflight` with `--repo`, `--commit`, repeated
`--source-path`, `--requirements`, repeated `--module`, and a create-only `--out`.
The requirements path must be included in the selected pinned source paths.
Declare bootstrap packages such as `torch==2.9.1` and `numpy==1.26.4` explicitly
in that requirements file; do not install the project or the Mac requirements.

The checker extracts only the pinned Git archive into a temporary source tree,
creates a fresh managed Python 3.12 venv with uv, installs only the job's declared
requirements, and runs Python with `-I` and a scrubbed environment. It recursively
executes static imports in reachable local modules, including parent package
initializers and imports inside functions. Direct `TYPE_CHECKING` branches are
excluded. Optional runtime branches are conservatively included: declare their
deps/source or use an explicit `module:function` entry seam. Runtime-computed
import strings require additional explicit module roots. No job functions, labels,
checkpoints, fits, or aggregators are executed. A GPU family canary still proves
CUDA and actual workload behavior.

A missing dependency or source file blocks packing. Preserve the returned JSON,
including commit, source archive and requirements SHA256, checker SHA256, full
installed distribution inventory and imported local module inventory, and bind
its digest in the pack receipt. The regression suite uses real clean venvs and
proves both direct and transitive lazy dependency failures cannot borrow from
`PYTHONPATH` or the Mac venv. This is an offline package gate; the qualified
`research/colab_cli` transport is unchanged.

### Sandbox host admission and allocation ownership

Sandbox dispatch runs full host admission before acquiring `alloc.lock`, within
an independent 1,200-second window (base 600 seconds, effective 2x). Each focused
suite has a 600-second limit (base 300 seconds, effective 2x; the TRAIN-01 suite
was measured at about 262 seconds). Ordinary commands retain their 60-second
limit; the allocation window remains at most 600 seconds.

A create-only `admission-tests.json` binds the clean broker Git identity, tracked
code/config bytes, packet bytes, external input hashes and explicit host test
environment. Refreshed validation and native startup reuse completed tests only
when this identity matches. They still execute structural admission, schema,
input/source/registry/seed checks and qualified accounting. Missing or changed
reuse evidence refuses before allocation; it cannot trigger full tests inside
the allocation window. Accounting refresh and the existing exact allocation retry
remain subject to the original packet's normalization rules.

Before detached startup, the exclusive parent ops lock becomes shared. The child
acquires its own shared descriptor immediately before native `new`, after checking
the create-only grant's manifest hash, parent liveness and monotonic deadline.
Every competing dispatcher still needs an exclusive lock. The child holds its
descriptor until durable rate calibration, or through native failure cleanup.
Parent revocation precedes unlock. A delayed runner refuses a revoked or expired
grant, and a child already in allocation keeps competitors fenced even if its
parent exits. TRAIN-01 native `run` requires this host grant.

Timed admission subprocesses run in owned process groups. Timeout/interruption
snapshots their descendant group IDs before termination, escalates every
observed owned group and reaps the owned child. Nested group owners also forward
termination; the root retains group identities even if an inner launcher exits
before its TERM-ignoring descendant. An unavailable inventory refuses admission
and reports cleanup as unverified after reaping the known root group. A tmux client is the explicit exception: cleanup reaps its client group but
never sweeps a detached shared tmux server. TRAIN-01 allocation remains fenced
by the owned ops lease. No process is selected by a global name match. These are host-only adapters; the
qualified `research/colab_cli` tree and remote science/runtime recipe are unchanged.
