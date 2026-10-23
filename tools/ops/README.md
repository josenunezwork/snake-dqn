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
canaries, budgets, leases, verified exports, and release.

Dispatch serializes account allocation using `ops/alloc.lock` with POSIX `flock`
(via Python `fcntl`, so the macOS host needs no external `flock` binary). The lock
covers launch through allocation success/failure, with a maximum ten-minute hold.
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

The supervisor still enforces its other admission, budget, and timeout checks.
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
"$REPO/tools/ops/mac_run.sh" --repo "$REPO" NAME FULL_COMMIT_SHA -- \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/example/job.py --help
```

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
