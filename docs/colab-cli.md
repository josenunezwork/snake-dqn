# Colab CLI supervisor

This is the operator guide for a new Snake DQN Colab job. The Mac-side
supervisor is the only allocator and releaser for new work. It produces
hash-verified local artifacts and a backend-verified release receipt without
a Drive mount or browser interaction on the runtime.

The supervisor is prepared, but it is **not live-qualified**. Do not route a
research workload through it until the bounded acceptance run in
[`research/colab_cli/ACCEPTANCE.md`](../research/colab_cli/ACCEPTANCE.md)
passes and its artifacts receive independent review.

## Scope and prerequisites

Use this path for every **new** Colab job. The operator needs an approved
study or job, a measured small local run of the actual entry point, a pinned
40-character commit, known GPU/CPU prices, and an unused absolute output
directory. Colab work is not strict-gate or serving-release evidence.

The two active notebook workers and Q1 are legacy-only work. Do not use this
guide to alter, migrate, launch, or recover them; their Drive queue under
`research/colab_queue/` remains outside this supervisor.

The supervisor does not authorize spending. It refuses a job with an unknown
CPU price, an unpriced hardware substitution, a missing input digest, a used
output path, an unresolved broker record, or insufficient balance for the
job cap plus reserve. It never buys units, retries an allocation, or starts a
replacement runtime.

## What the supervisor guarantees and what it does not

For one job, the supervisor:

1. Validates a manifest before allocation, records a durable pre-assignment
   record, and reconciles a known record after an interrupted assignment. It
   never guesses an orphan identity from a hardware list or allocates while a
   pending record is unresolved.
2. Admits at most three account assignments, including notebook workers. Every
   assignment already present must have an endpoint and a reviewed positive
   units-per-hour rate. It refuses an unaccounted endpoint or a provider rate
   that does not match the priced assignment set within the configured
   tolerance. A rate drift during polling is a hard refusal.
3. Uploads the commit-pinned source bundle and hash-bound inputs, starts one
   lease-fenced runtime child, and reads its immutable state. A polling RPC
   has an actual process timeout and a completion nonce. If an RPC is uncertain,
   the broker does not resubmit work to a possibly busy kernel; it uses the
   file transport and can resume RPC polling only after independent backend
   evidence says the kernel is idle.
4. Imports only completed immutable exports: payload, SHA-256 sidecar, and
   completion marker. Files move in bounded 8 MiB chunks; an 8 MiB Contents
   call is bounded to 60 seconds. Renewals are throttled to at most one every
   30 seconds, so a transfer does not submit two polling RPCs for every chunk.
   Only idempotent transport reads use their bounded three-attempt retry; a
   mutating kernel operation is never retried after an uncertain result. The per-export limit is
   `max_export_bytes`, up to 2 GiB. The supervisor verifies size and digest in
   a temporary local path before an atomic publish. An interrupted or corrupt
   transfer is rejected and its temporary data is removed.
5. Attempts backend stop with a 75-second allowance, polls account sessions,
   and repeats while its endpoint remains listed, for at most three attempts.
   Failure to observe disappearance is `STOP_UNVERIFIED`, raises a loud local
   alert, retains recovery evidence, and prevents a completed receipt.

The provider controls its APIs, network availability, and final billing. The
implementation provides bounded attempts and evidence, not an absolute claim
that a provider outage can never leave a VM allocated. A CLI exit status,
kernel process exit, local cleanup, or a dashboard view is never release
proof; only disappearance of the exact endpoint from account sessions is.

There is an unavoidable pre-watchdog window after allocation and before the
runtime child has received its private release credential and started. The
durable pending record contains the supervisor-owned notebook UUID so an
operator can perform authenticated backend GET recovery and then reconcile the
specific allocation. That recovery path is not an offline self-release
guarantee. A Mac or provider failure in this window can still create a paid
orphan, and the system must retain the record and alert rather than claim it
was released.

## Manifest contract and priced admission

The manifest is one JSON object with these required fields.

| Field | Requirement |
| --- | --- |
| `job_id` | Unique stable attempt identifier. |
| `commit` | Full 40-character Git SHA already available locally. |
| `script` | Repository-relative Bash launcher. |
| `env` | String-to-string environment map. |
| `requirements` | Requested `gpu`, `min_vcpu`, measured `est_minutes`, and optional pre-priced `allowed_gpus`. |
| `timeout_minutes` | At least 3 minutes and at least twice `est_minutes`; includes setup, export, stop, and idle time. |
| `output_dir` | Fresh absolute job-private Mac directory. Existing paths are refused. |
| `inputs` | Local `path`, expected `sha256`, and runtime `dest` for every non-repository input. |
| `budget` | Positive `unit_cap`, `rate_units_per_hour`, `reserve_units`, and, when other assignments exist, `other_assignments`. |
| `lease_seconds` | 180 through 600 seconds; default 180. |
| `max_export_bytes` | Positive per-export maximum, at most 2 GiB. |

`budget.other_assignments` is an endpoint-to-rate map for **every** assignment
that exists before this job is allocated. For example, only after reviewing
the current account state:

```json
{
  "endpoint-a": 1.5,
  "endpoint-b": 7.0
}
```

An empty map is valid only if `sessions` is empty at admission. The supervisor
caps the job from its own pre-priced rate multiplied by elapsed wall time. It
also cross-checks the account balance delta after subtracting the measured
baseline rate of the other assignments. A provider-rate mismatch or later rate
drift is a refusal. Account-wide balance is never a cap: it is an evidence
cross-check with a five-minute accounted-rate lag allowance plus rounding. An
unexplained balance discrepancy is recorded as a warning, not charged to this
session or used to abort Q1 or another lane.

The job launcher must make units resumable, write progress with completed,
total, and elapsed time about every two minutes, and publish a complete
optimizer, replay, actor, target, RNG, and counter state before marking an
export complete. Weights alone are not an exact resume.

## Lease expiry and runtime-side release

The runtime has no durable notebook frontend. A detached process must not call
`google.colab.runtime.unassign()`: that API needs the notebook frontend and
IPython kernel bridge, which an ordinary subprocess does not have.

Instead, the Mac mints a short-lived OAuth **access token** for the release
operation. It retains the owner refresh token locally. The mint must prove
that the granted scopes are exactly `colaboratory` and `userinfo.email`; a
missing, broader, or otherwise different scope is a refusal. The credential
contains no refresh token, client secret, or token URI and is private to the
runtime (mode `0600`). It is uploaded as a private file, never embedded in an
`exec` cell, prompt, manifest, output, log, or receipt. The Mac rotates it
while it can renew the job; renewal is throttled to at most once every 30
seconds. Naive OAuth expiry timestamps are normalized as UTC before any
lifetime decision.

The release credential is an account-scoped bearer token, not a provider
endpoint-bound credential and not inaccessible to a same-user process already
trusted to run the job in that VM. Its protection comes from the short expiry,
the exact scope check, private file permissions, the fixed Colab origin, and
redirect refusal. Treat the job process as within the runtime trust boundary;
do not claim this token cannot be used by arbitrary trusted same-user code.

After lease expiry or another terminal safety condition, the runtime requests
a checkpoint, grants its bounded grace period, then uses the fixed Colab
origin and the same authenticated tunnel sequence as the installed CLI:
`GET /tun/m/unassign/<endpoint>` for the XSRF token, followed by the XSRF
`POST` to that endpoint. Redirects are refused so the bearer is not forwarded
elsewhere. Each GET and POST has a 20-second request bound; three attempts,
overhead, and retry gaps have a 130-second full release bound. Before starting
the child, the runtime requires a credential margin of its lease, checkpoint
grace, 10-second terminal allowance, and that 130-second bound. This path has
no dependence on a notebook-front-end callback. It still needs live acceptance
evidence before it may be relied on operationally.

## Operator commands

Use `launch.sh` rather than the module directly. It checks AC power and an
open lid and runs under `caffeinate -is`. Put its foreground command inside a
caller-owned `tmux` session for an unattended job.

```sh
REPO=/Users/josenunez/Projects/ml/snake-dqn
PY="$REPO/venv/bin/python"
LAUNCH="$REPO/research/colab_cli/launch.sh"
SUPERVISOR_STATE=/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli/<lane>-<attempt>
COLAB_BIN=/Users/josenunez/.local/bin/colab

"$LAUNCH" --check --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/to/job.json
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" validate /absolute/path/to/job.json
tmux new-session -d -s colab-cli-<lane>-<attempt> \
  "'$LAUNCH' --state-dir '$SUPERVISOR_STATE' --colab '$COLAB_BIN' run /absolute/path/to/job.json"
"$PY" -m research.colab_cli.supervisor --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" status
```

`validate` and `status` do not allocate. The direct status invocation is
read-only and does not attest to exact hardware. The supervisor itself supplies
OAuth2 and its enforced request/assignment guards; do not set unsupported CLI
environment variables and assume they create a timeout or retry policy.

For a retained H100, use a finite, priced chain of already-approved jobs. It
has the same lease and release proof as a normal job:

```sh
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" chain \
  /absolute/path/to/first-h100.json /absolute/path/to/second-h100.json \
  --retention-minutes 45 --retention-units 10
```

If a broker-owned `pending-*.json` or `active-*.json` remains after an
interruption, reconcile that exact record. Do not infer the target from the
session display:

```sh
ACCOUNT=/Users/josenunez/.config/colab-cli/snake
"$LAUNCH" --state-dir "$SUPERVISOR_STATE" --colab "$COLAB_BIN" recover "$ACCOUNT/pending-<session>.json"
```

Read-only account evidence is available after a supervisor action:

```sh
"$COLAB_BIN" --auth oauth2 usage
"$COLAB_BIN" --auth oauth2 sessions
```

## Required acceptance before live research

The acceptance plan stays within 4 units including its reserve. It is prepared
through `launch.sh`, so the AC, lid, and `caffeinate` protections apply. Its
L4 arm deliberately suppresses Mac lease renewal and must show that the exact
assignment disappears within 300 seconds **after lease expiry**. Its total
timing budget is no more than the lease plus checkpoint grace, terminal
allowance, and 130-second release window. It fails if the runtime only exits, if broker
cleanup causes the disappearance, if cleanup has an error, or if any
acceptance artifact is missing.

See [`research/colab_cli/ACCEPTANCE.md`](../research/colab_cli/ACCEPTANCE.md)
for the reviewed prepare and execute commands, current price inputs, artifacts,
and all gates. Do not execute the live command until that separate review has
occurred.

## Human-only credential actions

Only the owner may repair OAuth with a local interactive terminal. Agents must
not invoke `colab auth` or `colab drivemount`; a CLI runtime never mounts
Drive. Never place tokens or credentials in a manifest, shell history, prompt,
runtime code, artifact, log, or receipt.

## Source and limits

The release sequence is based on the installed pinned CLI's
`Client.unassign()` GET/XSRF-POST tunnel exchange. The supervisor adds
lease fencing, restricted credential handling, bounded transfer and stop
logic, and evidence requirements around that provider behavior. See the
[`snake-dqn-colab-jobs` skill](codex-skills/snake-dqn-colab-jobs/SKILL.md)
for the agent checklist and the legacy-Q1 boundary.
