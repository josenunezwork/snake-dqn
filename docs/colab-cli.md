# Colab CLI supervisor

This is the operator guide for a new Snake DQN Colab job. It gives agents a repeatable path from an approved,
pinned manifest to a verified local receipt without browser clicks or a Drive mount on the runtime.

## Scope and prerequisites

Use this path for every **new** Colab job. It is for an operator who already has an approved study/job manifest,
understands that Colab development evidence cannot qualify a strict gate or serving release, and has measured the
real entrypoint on a small local input. The pinned `google-colab-cli` OAuth setup is already complete on this Mac.

The CLI supervisor is not an authorization to spend. A manifest must have the registered/approved budget and all
required inputs before it is run. The separate acceptance qualification is prepared now but is expected to run only
after Q1 and independent review; it must pass before a production allocation.

Do not use this guide for the two running notebook workers or Q1. Their Drive queue remains a legacy recovery path
only: `research/colab_queue/` and its manifests are deliberately out of scope here.

## What the supervisor does

`research.colab_cli.supervisor` is the one Mac-side broker for allocation and stop. It uses the official
`google-colab-cli` binary with OAuth2 authentication, zero assignment retries, unique session names, and a distinct
CLI config per lane. It first validates the manifest and then:

1. Counts all account assignments, including notebook workers, and refuses to exceed three concurrent assignments.
2. Uploads the commit-pinned bundle and every non-repository input after checking its SHA-256 digest. The runtime
   does not mount Google Drive.
3. Starts a kernel-side lease-fenced subprocess. The initial execution returns; later explicit-timeout executions
   only renew the lease and read state/telemetry. They never blindly resubmit work to a busy kernel.
4. Periodically downloads only immutable, completed checkpoint exports into a temporary Mac path, verifies hashes,
   then atomically renames them. A Drive-for-desktop copy may be made as a backup after local verification.
5. Downloads the final artifacts, writes a create-only receipt, stops the assignment, and verifies that the
   assignment no longer appears in the account session list.

The runtime uses Python 3.12 through `uv`. Its wheel selection is compute-capability based: cu126 for L4, A100, and
H100; cu128 for a Blackwell substitute. The **job launcher** must produce a consistent optimizer, replay, actor,
target, RNG, and counter snapshot before it marks an export complete. The supervisor transports and verifies those
files; it does not serialize trainer state. Weights alone are not an exact resume.

## Manifest contract

The supervisor consumes one JSON object. It must include:

| Field | Requirement |
| --- | --- |
| `job_id` | Unique stable identifier for this attempt. |
| `commit` | Full 40-character Git commit SHA already available to the job. |
| `script` | Repository-relative Bash `.sh` launcher. |
| `env` | String-to-string environment map; bind external inputs and their digest values here. |
| `requirements` | Object with requested `gpu`, `min_vcpu`, measured `est_minutes`, and optional pre-priced `allowed_gpus`. |
| `timeout_minutes` | At least 3 minutes and at least twice `est_minutes`. |
| `output_dir` | New, absolute, job-private Mac directory. Existing output is refused. |
| `inputs` | Each entry has local `path`, expected `sha256`, and runtime `dest`. |
| `budget` | `unit_cap`, `rate_units_per_hour`, and `reserve_units`; the first applicable cap wins. |
| `lease_seconds` | Optional Mac-renewal interval; default 180 seconds, accepted only from 180 through 600. |
| `max_export_bytes` | Maximum one immutable export may occupy. |

Use only `gpu` values the supervisor accepts. `allowed_gpus` defaults to the requested `gpu`; list a substitution
there only when it was pre-priced. Before ready/bootstrap, the supervisor queries the runtime GPU model and compute
capability, rejects a model outside that list, and writes the admitted hardware evidence to `hardware.json`. It also
checks minimum vCPU count. Estimate wall and unit caps before launch, including setup, export, idle time, and the
registered reserve. Never set an unknown CPU rate to zero.

`rate_units_per_hour` is a conservative pre-priced ceiling, not a claim that the provider reports an exact per-job
rate. The observed account-balance debit can include concurrent notebook or other-lane spend and provider-accounting
lag, so it is a conservative stop signal and reportable evidence rather than precise job billing.

The acceptance qualification must specifically check whether that conservative account-debit guard stops a lane early
while notebook workers spend, and whether provider debit lags the observed balance. Until that evidence exists,
actual job units are unknown rather than an exact enforcement claim.

The launcher must reject a populated output directory, write progress with completed/total and elapsed time at least
about every two minutes, make units resumable, and write only completed immutable snapshots. Each export needs its
payload, SHA-256 sidecar, and completion marker before the supervisor may import it.

## Operator commands

From the repository worktree, set the state directory to a new absolute directory outside the repository. The
foreground launcher checks AC power and open lid, then runs the supervisor under `caffeinate -is`; put it in a
caller-owned prepared tmux service for unattended work. Validate first; validation allocates no runtime.

```sh
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
LAUNCH=/Users/josenunez/Projects/ml/snake-dqn-colab-cli/research/colab_cli/launch.sh
SUPERVISOR_STATE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli/<lane>-<attempt>
COLAB_BIN=/Users/josenunez/.local/bin/colab

"$LAUNCH" --check --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/to/job.json

"$LAUNCH" \
  --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/to/job.json

tmux new-session -d -s colab-cli-<lane>-<attempt> \
  "'$LAUNCH' --state-dir '$SUPERVISOR_STATE' --colab '$COLAB_BIN' run /absolute/path/to/job.json"

"$PY" -m research.colab_cli.supervisor \
  --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" status
```

The direct `status` call is read-only and does not assert exact provider hardware capability. A scarce H100 can be
retained only by a finite, priced chain of already-approved H100 jobs:

```sh
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" chain \
  /absolute/path/to/first-h100.json /absolute/path/to/second-h100.json \
  --retention-minutes 45 --retention-units 10
```

If an interrupted broker leaves a known record under the account state root, reconcile that specific `pending-*.json`
or `active-*.json` record. Never guess an assignment identity from a hardware listing:

```sh
ACCOUNT=/Users/josenunez/.config/colab-cli/snake
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" recover "$ACCOUNT/pending-<session>.json"
```

The supervisor supplies `--auth oauth2` and `COLAB_CLI_ASSIGN_RETRIES=0` for its CLI calls. Do not substitute a
manual `colab new`, `exec`, `upload`, `download`, or `stop`: bypassing the broker defeats account-wide admission,
lease renewal, export verification, and release proof.

Use the read-only account commands to inspect the result after a supervisor action:

```sh
"$COLAB_BIN" --auth oauth2 usage
"$COLAB_BIN" --auth oauth2 sessions
```

The expected completed state is a terminal create-only receipt and hash-verified local artifacts in the manifest's
output directory, followed by no matching assignment in `sessions`. A successful CLI process alone is not evidence
that the kernel or job succeeded.

## Failure handling

If the kernel reports an error while `colab exec` exits zero, treat the job as failed and preserve the state and
receipt. If a polling request times out, do not assume the remote process stopped and do not submit the launcher
again; reconnect through the state/telemetry files. If an export is too large, lacks a completion marker, or has a
digest mismatch, reject it. The bounded transfer routine removes its temporary file and never publishes it.

If the Mac cannot renew its lease, the runtime asks the child to checkpoint, stops it after the bounded grace
period, and unassigns the runtime. Recovery may start from the last hash-verified imported snapshot only. If stop
reports success but the assignment remains in `sessions`, record an unassignment failure and do not call the run
complete.

H100 retention is allowed only for a priced, approved chain of useful queued jobs. The retention lease is bounded;
an idle allocation is never kept indefinitely. No automatic retry, new allocation loop, or purchase is permitted.

## Human-only commands

The owner already completed OAuth setup. Agents must never invoke `colab auth` or `colab drivemount`; both can
require interactive consent and a Drive mount is prohibited in a CLI runtime. Keep credentials out of manifests,
uploads, prompts, logs, and receipts. If reusable OAuth fails because the owner revoked access, stop and have the
owner reauthorize in a local Terminal.

## Source and boundaries

This guide implements the operating model in
[`2026-10-18_colab_cli_operating_model.md`](research/advice/2026-10-18_colab_cli_operating_model.md) and uses the
vendored official CLI reference under [`docs/research/colab_cli_ref/`](research/colab_cli_ref/). In particular,
kernel errors may not set a nonzero CLI status, client timeouts do not prove remote termination, transfer buffering
requires caps, and `run` cleanup alone is not release verification.
