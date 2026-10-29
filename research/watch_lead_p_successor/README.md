# P successor preparation

Preparation only. No Colab allocation or successor gameplay has been performed.
The old P bank and all 21 imported session outcomes remain closed. Read
`DRAFT_REGISTRATION.md`: it is deliberately unratified and has no executable main
manifest. Existing P science/runtime/receipts are untouched.

The engineering runner calls the existing `watch_lead_p.engine.run_arm` rather
than reimplementing it. Four deterministic fresh dev seeds each supply their
first naturally reached alive L≥600 root (within 6,000 ordinary bare Watch ticks).
No root replacement or artificial geometry is permitted. Roots are prepared
once, in four CPU processes, then exact snapshots seed every arm and concurrency
sweep. The constructor seam supplies those snapshots; snapshot/restore, W1/K1,
search H300, P48 delay, acceptance, fallback, fp32 and mechanics stay unchanged.
This is a mature-window workload, not a natural-start full-session estimate.

The same 12 arm tasks (four seeds × P0/P48/bare) run for 64 frames at each of
1/4/8/11 workers. P48 can deliver at frame offset 48. Timing wrappers collect
inclusive wall/CPU time, submit-call wall time, non-submit wall time, CPU seconds
per wall second, RSS and host CPU/load samples. CPU shape includes logical and
physical counts, affinity, cpuinfo, cgroup quota/stat/cpuset/memory, and nice/thread
settings. All workload traces must hash identically across concurrency. No
death, mass, action choices, certificates or search outcomes are exported.
Only opaque trajectory/initial-state hashes cross the timing boundary.

The first 1-worker sweep pays cold imports separately from the measured arm
window; every sweep includes spawning/IPC in its overall wall. Speedup therefore
means end-to-end time for this fixed small task set, not steady-state scaling.
Prefix preparation is measured separately. Per-arm CPU% uses 100% = one logical
core. Difference consecutive cumulative child CPU-second samples / wall intervals
for aggregate child utilization; do not sum percentages normalized by host cores.
All workers stay alive until sweep end, so their cumulative CPU is retained.

A 900-second total engineering work deadline includes root preparation and all
sweeps; progress is printed at least every 30 seconds by the coordinator.
Workers have a 900-second stall watchdog; window watchdog resets only after a
completed tick. Deadline/error terminates and reaps only this executor's children.
SIGUSR1/SIGTERM use default process termination: completed immutable timing cells
remain available; this short canary claims no resumable scientific work. There
is no automatic allocation or completion retry. Partial cells do not qualify a
profile, and a missing final required export makes the broker refuse success.

Native family probes have a stricter ≤30-minute ceiling than the generic
40-minute resume exemption. Base/effective whole-session guards are 15/30 min;
they are finite engineering limits, NOT measured forecasts. L4 rate ceiling is
1.54 u/h, A100 5.30 u/h (live calibration must agree or refuse). Caps price the
whole timeout and round upward before exact doubling; reserves also double.

CLI 0.7.4 `colab new --help` offers CPU/T4/L4/G4/H100/A100, `--high-mem`, and
no vCPU selector. High-mem is ignored for L4. The qualified supervisor at this
source pin forwards `--gpu` only and cannot request high-mem. No unsupported CLI
flag is inserted and transport is not modified. An optional A100 probe compares
whatever CPU host is actually assigned, with ≥12 logical CPUs required. It is
not advertised as a guaranteed high-vCPU option; compare effective quota/affinity
as well as nominal topology. Same fresh engineering bank and worker grid on each
host, sequential dispatch only after verified release. If A100 offers no useful
CPU capacity/speed advantage, the comparison cannot justify using it for P.

Native probes retain the qualified supervisor, native canary admission, all
accounting/lease/runtime/release rules, and source archive verification. A narrow
host adapter routes only these exact 30-minute CPU profile probes through the
existing `tools.sandbox.source_delivery.background_sparse_route`. This avoids
synchronous full-repo Git fetch inside the 30-second staging RPC. It requests the
qualified sparse fallback and installs it in the bounded background worker; the
300-second staging and 150-second native transfer bars still apply. The only ops
change selects that launcher for the explicit profile opt-in. It cannot select
sandbox admission or bypass native family-canary validation, and no bytes under
`research/colab_cli` change. A live staging failure still requires preserved
receipt/release evidence and a separately reviewed fix; no automatic retry.

Google documents variable host availability/runtime limits and Pro+ execution
up to 24 hours subject to availability and units; this is not an uptime promise:
[Colab FAQ](https://research.google.com/colaboratory/faq.html).

Prepare a source-pinned packet only after committing these sources:

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.watch_lead_p_successor.prepare \
  --pin FULL_SOURCE_SHA --packet FRESH_ABSOLUTE_PACKET_PATH
```

The packet contains sealed manifests, bank/exclusion metadata, source hashes,
CLI help/version, exact single-line `canary-command.sh` (L4) and
`compare-command.sh` (optional A100). Append `--dry-run` to those lines for the
real ops-wrapper validation. Executing them without that flag is for the operator.
Never source these files during preparation. Verify seal and current source
hashes before dispatch. No command exists for successor data collection.
