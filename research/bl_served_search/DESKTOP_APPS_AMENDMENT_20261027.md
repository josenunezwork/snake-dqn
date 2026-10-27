# Prospective ordinary desktop application amendment

Authorized by the owner's 2026-10-27 implementation request and Astra's sibling
`ASTRA_DESKTOP_RULING_20261027.md` (SHA256
`0210543b502b6d1722ad631d9fc3be740d35c073591b2a02ff157ecc2c57905e`).
Policy: **classified-desktop-v1**. Applies only to new attempts explicitly using
`--desktop-app-registry`. It adopts the ruling exactly and supersedes only the
desktop identity and load accounting rules for this distinct policy. The meanings
of `classified-v1` and `classified-supervised-v2` remain unchanged. It does not
admit remote supervision. Attempts 07 and 08 remain INVALID; their cells cannot
be salvaged, rescored, or combined with another attempt.

The fixed panel manifest, seeds, order, exposure, observer control, timing bars,
parity, all-tick verdict, Mac/power/lid/thermal/heartbeat/lease/lifetime guards and
owner's serving/release authority remain unchanged. This is an engineering
diagnostic, not serving qualification or scientific promotion.

## Exact enrollment and use declaration

The sealed registry freezes installed canonical bundles, bundle identifiers,
version/build, trusted codesign identity and designated requirement, Team ID,
executable digests and exact bundle-relative UI roles, plus permitted process UID.
Each UI executable needs its own entry. A display name, argv, signature alone,
parent, `.app` suffix, or renderer/GPU suffix cannot grant admission. No recursive
bundle/descendant trust exists. Missing or ambiguous evidence gives no exception.
Symlinks/traversal, installation copies, wrong UID, identity changes and updates
cannot inherit enrollment. New birth identities require verification outside
scored ticks. Fail-closed per-process revocation (stricter than Astra's ruling,
and therefore within it): any identity change or verification failure of a
`(pid, birth)` — including an exec/argv change, executable replacement on disk
whether bound or unbound, UID change, or canonical-path/UID/signature
verification failure — permanently revokes that `(pid, birth)` for the entire
attempt and it is never re-admitted, even if the on-disk binary is later
restored; only a new `(pid, birth)` may be admitted fresh.

The permitted-use declaration is ordinary owner mail, browsing and remote-chat
presentation/synchronization only. The operator must verify that this declaration
describes actual use before launching. Known deliberate local computation,
browser JavaScript/WebAssembly/WebGPU benchmarks or ML, automation, bulk export,
transcoding, training, tests, builds, indexing, language servers and embedded
compute are excluded. Where substantive work cannot be separated, the whole
hosting process is ordinary work. The registry records known workload exclusions;
discovered prohibited compute during exposure invalidates the attempt without
post-outcome relabeling. Signing proves identity, not harmless activity; this
cooperative-host policy cannot rule out concealed in-process compute.

Known workloads take precedence over desktop enrollment and macOS paths: Python,
node, CLI agents (including the claude-code binary), pytest/uv, repository or
research workers and local supervisors remain `agent_workload`. Only the existing
exact probe-owned ancestry exclusion precedes these rules. Never stop, renice,
hide or disable the owner's applications to produce a quiet result.

## Shared load accounting and retained evidence

Percentages are of one core. Verified `desktop_app` and explicitly enrolled
`desktop_renderer` executables join macOS services in **one** shared background
pool. No ordinary per-process abort applies to those verified roles. The shared
pool invalidates only at **strictly >150% for strictly >60 consecutive observed
seconds**. Exactly 150% or 60 seconds does not cross it; sub-ceiling observations
reset duration; gaps/intervals longer than 5 seconds cannot prove continuity.
Churn never erases measured high background CPU. Ordinary workload and unknown
processes retain **>=20% per process**, in both point and CPU-delta paths. There
is no new aggregate ordinary-work abort. Unadmitted remote supervision remains
ordinary workload regardless of useful remote work.

Nominal 1-second cumulative CPU sampling, birth/exec/counter checks, original
capture brackets, point %CPU and interval rows are retained. Live collection and
reducer replay share one pure function
`classify_row(history, row, registry) -> (kind, new_history, reason)` with no
hidden mutable state; the complete per-`(pid, birth)` history is exported in
`host.json`, interval transitions classify the previous row against the history
as of its own capture time before stepping the current row, and the reducer must
reproduce every exported classification and history state exactly or reject.
Raw identity/role, admission/rejection reason, classification changes, registry
mismatches, coverage, aborts and sustained-duration/maxima evidence remain
available. Reports separate
service, renderer and desktop-app CPU and also reconcile their combined pool,
ordinary/unknown CPU, per-process/application totals and affected tick counts,
per cell and overall. No mail/chat content or credentials establish roles.

Every measured tick and client receipt remains in authoritative distributions.
The descriptive low-background subset requires complete original coverage and
combined background <=20% in every overlapping interval, workload and unknown
totals each <20%, with existing churn/gap exclusions. Client gaps require
consecutive original receipt identities and clean intervening coverage; sparse
retained receipts are never joined. Empty subsets retain zero count and null
statistics. This subset cannot change FAIL to PASS. No CPU subtraction, overlap
tick removal, favorable-cell repeats or threshold adjustments are allowed.

## Evidence chain and acceptance

Before observation, `intent.json` seals this amendment, ruling, registry, manifest,
policy and actual execution-worktree source. Each payload has SHA256 and a
completion marker published last. Host/raw evidence carries identical registry
and policy bindings; independent reduction rechecks the intent, original host
triplet, classification and totals. Historical attempts retain their dispositions.

Before full exposure, require source-bound offline identity/negative/churn/exec,
workload precedence, both >=20% guard, exact shared ceiling, coverage, policy and
reducer reconciliation, all-tick/low-subset/client-gap tests plus independent
implementation review. Perform the prepared bounded host-only observation with
actual enrolled apps before a panel; it acquires no lease and runs no game.
The panel requires that sealed observation receipt with unchanged source and
registry as an input, and binds its digest/path into the panel intent.
Tests and observation establish conformance, not causal timing neutrality.

Use the next natural supervisor-free Mac window, preserving useful remote jobs
and their safety supervision through natural completion and cleanup. Idle GPU
capacity is not proof of a quiet host. No full panel is launched by this task.
The existing panel forecast is 45–60 minutes, lifetime cap 120 minutes. The
launcher uses CPU, nice 15, all five thread caps 1 and caffeinate.

Commands are in `DESKTOP_APPS_HANDOFF_20261027.md`; the host-only observation is
bounded to 120 seconds and must precede the fresh `panel-desktop-09` exposure.
