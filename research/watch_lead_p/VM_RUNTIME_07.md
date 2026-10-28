# P packet-07: uploaded inputs and VM-fidelity repair

Mechanical repair for the owner's 2026-10-28 child-exit report; Argon/Opus reviews
this Astra build before operator dispatch. No release or serving authorization.

`run smoke` and each real `worker` enter `uploaded_checkpoints`. `run run` also
validates before spawning and exports the explicit `--inputs` directory into the
worker environment. The adapter requires existing absolute input paths and exact
B8 s3 / FRP3-s12 SHA-256 bytes. Missing, altered or symlinked checkpoint files
raise; there is no registry search or checkpoint fallback. The immutable runtime
pin ships the adapter and the sealed manifest binds the uploaded bytes.

The adapter scopes three bindings to the P process: hero resolution, mixed
opponent resolution, and the FRP3 v8 checkpoint membership lookup. It leaves
GameSession construction, checkpoint loading, mixed-roster assertions, wrapper
source checks, clone/search, gameplay, statistics and decisions unchanged. All
bindings and the temporary mechanics-v2 environment override restore on exit.
The launcher no longer creates or copies anything under `saved_snakes`.
The web server never installs this adapter; shared serving code and pins are
unchanged. This experiment's checkpoint hash check supplies no serving approval.

## Complete runtime dependency audit

The runtime inventory was scanned for `/Users`, artifact/evidence roots,
`saved_snakes`, serving resolvers, home expansion and environment access. The
retained engineering scan is `engineering-07/runtime-path-scan.json` outside Git.
The actual path is run_job -> run/main or worker -> engine -> LiveWorld ->
M1 new_watch -> T15 new_session -> GameSession -> mixed roster -> checkpoint
policies/v8, followed by Watch.step, snapshot/cloudpickle, clone, H300 search,
publication and metadata-only resume. Ledger/input/output paths are explicit;
source/config/wrapper paths derive from the copied sparse source tree.

Three reachable Mac evidence dependencies were found and removed from P:

1. B8 default registry selection validated a strict-receipt absolute Mac root.
2. FRP3 mixed opponent selection performed the same strict-pin validation.
3. v8's FRP3 swap-membership check independently validated serving strict pins.

Other absolute strings are dormant constants/CLI helpers in the reused research
closure (M1 admissions/leases, historical census/teacher artifacts, and serving
registry candidate paths). They are never invoked by P after the scoped binding.
Their scientific source bytes remain frozen. The seven thread caps and CPU device
are set by run_job; configure_cpu pins Torch; T15 clears/restores serving knobs,
M1 disables Path B, and P fixes mechanics-v2 for uploaded construction. Release
telemetry is suppressed by the existing T15 helper.

`test_real_entrypoint_without_mac_roots` copies only runtime-sources plus actual
checkpoint bytes into a temporary sparse tree, with distinct HOME and cwd and no
saved_snakes directory. It starts a fresh interpreter, clears inherited P/serving
environment, and refuses open/listdir/scandir/stat/access against the main repo,
original checkout and all snake-dqn-artifacts. Only the installed venv dependency
tree is permitted; project source and checkpoints cannot come from it. It proves
the guard is active, runs the real module smoke (60 ticks per arm), then reaches
the excluded natural L600 fixture, runs P0/P48 H300 snapshot/search, and exercises
death/respawn. Any forbidden access fails the test even if caught by runtime code.

The adapter also matches the original Mac construction's first 12 real ticks on
the excluded fixture, and tests refusal for both hero and opponent inputs plus
restoration of the original serving bindings. No main-bank result is examined.

All original science-sources bytes, source inventory, N=32, allocation, arms,
world seeds, horizons, controller decisions and thresholds are unchanged. The
recipe fingerprint remains
`acb5d0a6492c50a4ee382837f2bf4f115a305bd4779282626da5d2815b30170c`.

## Mechanical attempt accounting

See `RETRY_07.json`, which binds the original terminal/receipt and burned-identity
note by SHA-256. Identity 06 was a local reservation failure without allocation.
06b was executed attempt 1: CHILD_NONZERO_EXIT, stop_verified=true, no exports.
Packet-07 prepares executed attempt 2 / manual completion 1 of at most 2. All 32
units are missing, so its verified ledger is empty. No outcome-based retry,
sample replacement, extra look or automatic dispatch is permitted. If packet-07
is executed, at most one further manual completion remains under the same rules.
