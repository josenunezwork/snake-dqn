# Colab CLI supervisor

This is the operator guide for a new Snake DQN Colab job. The Mac-side
supervisor is the only allocator and releaser for new work. It produces
hash-verified local artifacts and a backend-verified release receipt without
a Drive mount or browser interaction on the runtime.

The supervisor is prepared, but it is **not live-qualified**. Do not route a
research workload through it until the bounded acceptance run in
[`research/colab_cli/ACCEPTANCE.md`](../research/colab_cli/ACCEPTANCE.md)
passes and its artifacts receive independent review.

## Acceptance status: first CPU evidence, not a full qualification

The first live CPU acceptance attempt began on 2026-10-18 at 19:41 PT. Its
run directory is
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/colab-cli-acceptance/acceptance-1792377693`.
The rate-calibration gate passed with an observed CPU rate of **0.08 units per
hour**. Cleanup also passed for that allocation: the receipt recorded
`stop_verified: true`, and the final account query found no assignments left.

This is useful evidence for CPU rate calibration and release cleanup only. It
is not a full acceptance pass: before the runtime became ready, the former
whole-commit source upload stalled. That archive was about 166 MiB (171 MB of
tracked source, including research evidence blobs), and it was sent as 8 MiB
base64 JSON requests through the CLI Contents client. The remaining source
delivery, runtime bootstrap, export, and L4 acceptance gates still need their
own evidence. The AC/lid launch guards fixed in commits `73aef4b6` and
`a721e354` remain part of the supported launcher path.

The observed 0.08 units/hour rate does not lower the conservative CPU priced
ceiling of 0.5 units/hour. Admission, reserves, and per-session caps continue
to use the ceiling; the observed rate is recorded for calibration and later
drift checks. See
[`colab-cli-pricing-notes-2026-10-18.md`](colab-cli-pricing-notes-2026-10-18.md)
for the dated evidence and rationale.

## Scope and prerequisites

Use this path for every **new** Colab job. The operator needs an approved
study or job, a measured small local run of the actual entry point, a pinned
40-character commit, known GPU/CPU prices, and an unused absolute output
directory. Colab work is not strict-gate or serving-release evidence.

The two active notebook workers and Q1 are legacy-only work. Do not use this
guide to alter, migrate, launch, or recover them; their Drive queue under
`research/colab_queue/` remains outside this supervisor.

The supervisor does not authorize spending. It refuses a job with a nonpositive
priced rate ceiling, an unpriced hardware substitution, a missing input digest, a used
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
   units-per-hour rate. It refuses an unaccounted endpoint. After allocation,
   it calibrates the provider-reported rate by subtracting reconciled other
   assignments, requires a stable positive observed rate no greater than the
   priced ceiling plus tolerance, and then treats drift from that observed
   rate as a hard refusal. The priced ceiling remains the admission and cap
   rate.
3. Delivers the exact pinned source and hash-bound inputs, starts one
   lease-fenced runtime child, and reads its immutable state. The normal source
   path lets the runtime shallow-fetch the declared commit from the public
   `josenunezwork/snake-dqn` GitHub repository. It verifies both the commit SHA
   and the tree ID recorded by the Mac for that commit before it can bootstrap.
   An identity mismatch is a refusal, never a reason to try another revision
   or fall back to an uploaded archive. If GitHub is unreachable, the runtime
   may instead receive the staged sparse bundle described in
   [Source delivery on a fresh VM](#source-delivery-on-a-fresh-vm). A polling
   RPC has an actual process timeout and a completion nonce. If an RPC is
   uncertain, the broker does not resubmit work to a possibly busy kernel; it
   uses the file transport and can resume RPC polling only after independent
   backend evidence says the kernel is idle.
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
| `source_paths` | Optional, explicit repository-relative sparse-fallback allowlist that replaces the defaults. Required bootstrap, job-script, requirements, and project-metadata files remain included. Paths must be narrow files or directories; the repository root and a broad `research/` root are refused. |
| `budget` | Positive `unit_cap`, `rate_units_per_hour`, `reserve_units`, and, when other assignments exist, `other_assignments`. |
| `lease_seconds` | 180 through 600 seconds; default 180. |
| `max_export_bytes` | Positive per-export maximum, at most 2 GiB. |

`budget.rate_units_per_hour` is a conservative **priced upper-bound rate**.
It controls admission, reserve calculation, and the session wall-clock unit
cap. It is not an assertion that the provider will report that exact rate.

`budget.other_assignments` is an endpoint-to-rate map for **every** assignment
that exists before this job is allocated. For example, only after reviewing
the current account state:

```json
{
  "endpoint-a": 1.5,
  "endpoint-b": 7.0
}
```

An empty map is valid only if `sessions` is empty at admission. Immediately
after allocation, the supervisor takes a bounded calibration read: up to 120
seconds, three stable positive provider-rate reads five seconds apart, stable
within 0.01 units/hour. It subtracts the reconciled baseline of other
assignments while allowing provider lag. The observed session rate must be no
more than the priced upper bound plus 0.05 units/hour. An unknown, unstable,
unexplained, or over-ceiling result stops the job as `INCOMPLETE`. After a
passing calibration, drift is tested against the observed rate; pricing and
caps still use the priced upper bound. The receipt records the priced rate,
observed rate, and samples in `rate-calibration.json` and `receipt.json`.
The account lock covers allocation, watchdog setup, and calibration (or failure
cleanup), so other broker lanes see only calibrated active records. Waiting
lanes renew their leases and retain their wall and priced unit caps. The
pre-existing endpoint set must remain unchanged during calibration; a changed
baseline cannot safely identify this session's rate and stops the job.
The account balance remains evidence only, with a
five-minute accounted-rate lag allowance plus rounding; it never charges Q1 or
another lane to this session.

The job launcher must make units resumable, write progress with completed,
total, and elapsed time about every two minutes, and publish a complete
optimizer, replay, actor, target, RNG, and counter state before marking an
export complete. Weights alone are not an exact resume.

## Source delivery on a fresh VM

The runtime normally obtains the pinned source itself, rather than receiving a
tar archive of the whole Mac checkout. Before allocation, the Mac resolves the
manifest's 40-character commit and records its Git tree ID. The runtime then
performs a shallow fetch of that exact commit from the public GitHub repository
within a 60-second fetch operation budget. It must confirm its checked-out
commit and `HEAD^{tree}` against the Mac-recorded values before it runs the
bootstrap or job script. The supervisor gives this RPC a 90-second bound.

Only a GitHub-unreachable result permits the fallback. A missing commit,
checksum/tree mismatch, or malformed repository result refuses the job; it
does not silently use a locally constructed archive. The job receipt writes
`source-delivery.json`, including whether delivery used `github` or `fallback`
and the verified commit and tree. The pre-allocation staging area writes
`source-plan.json`, which inventories and hashes the planned fallback source.

The sparse fallback contains only the default allowlist:

- `src/`
- `research/colab_cli/`
- the job's own research directory
- root `requirements*.txt`, `requirements/` when present, and `pyproject.toml`

Set `source_paths` only when the default does not describe the pinned runtime
dependencies. It replaces the default allowlist (while the required bootstrap,
job-script, requirements, and project-metadata files remain included). It is
an allowlist, not a way to transfer the complete repository. The compressed
archive and its unpacked source must each be at most 32 MiB. A plan that
exceeds either limit is refused before allocation. The fallback archive is
built before allocation but is uploaded only after the runtime reports GitHub
unreachable. Mac-only inputs remain separate entries in `inputs` and are
always bound by their declared SHA-256 digests.

For example, a job whose script and required bootstrap files do not cover its
small shared helper can add an explicit narrow allowlist entry:

```json
{
  "source_paths": [
    "src",
    "research/shared_runtime_helper.py"
  ]
}
```

The helper must exist in the pinned commit. Adding a broad directory merely to
avoid declaring the dependency is refused.

## Runtime input uploads

Normal GitHub source delivery leaves only Mac-only inputs and the conditional
sparse fallback to upload. The supervisor transfers those files in 2 MiB
chunks. Each chunk is written to an idempotent offset/part path, has a bounded
retry only for connection or timeout failures, and is assembled only after the
complete file's SHA-256 and size are checked. The whole upload is also subject
to a five-minute total transfer-time cap; exhausting it refuses the job before
an unbounded Contents-client request can continue.

This retry rule does not apply to allocation, `exec`, stop, or other uncertain
mutating kernel operations. The transfer receipt records the verified digest
of each completed upload. A source or input whose final digest differs from
its plan is rejected.

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

The acceptance plan stays within 4 units including its reserve. CPU pricing
defaults to a conservative 0.5 units/hour and L4 pricing defaults to 2.0
units/hour; `--cpu-rate` and `--l4-rate` may override those pricing ceilings
before allocation. The measured CPU and L4 rates are acceptance outputs, not
required command inputs. It is prepared
through `launch.sh`, so the AC, lid, and `caffeinate` protections apply. Its
L4 arm deliberately suppresses Mac lease renewal and must show that the exact
assignment disappears within 300 seconds **after lease expiry**. Its total
timing budget is no more than the lease plus checkpoint grace, terminal
allowance, and 130-second release window. It fails if the runtime only exits, if broker
cleanup causes the disappearance, if cleanup has an error, or if any
acceptance artifact is missing.

See [`research/colab_cli/ACCEPTANCE.md`](../research/colab_cli/ACCEPTANCE.md)
for the reviewed prepare and execute commands, pricing ceilings, artifacts,
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
