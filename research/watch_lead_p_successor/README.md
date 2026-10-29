# P successor preparation

Preparation only. No Colab allocation or successor gameplay has been performed.
The old P bank and all 21 imported session outcomes remain closed. Read
`DRAFT_REGISTRATION.md`: it is deliberately unratified and has no executable main
manifest. Existing P science/runtime/receipts are untouched.

The engineering runner calls the existing `watch_lead_p.engine.run_arm` rather
than reimplementing it. Four deterministic fresh v2 dev seeds each supply their
first naturally reached alive L≥600 root (within 6,000 ordinary bare Watch ticks).
No root replacement or artificial geometry is permitted. Roots are prepared
once, in four CPU processes, then exact snapshots seed every arm and concurrency
sweep. The constructor seam supplies those snapshots; snapshot/restore, W1/K1,
search H300, P48 delay, acceptance, fallback, fp32 and mechanics stay unchanged.
This is a mature-window workload, not a natural-start full-session estimate.

The same 12 arm tasks (four seeds × P0/P48/bare) run at 1/4/8/11 workers.
Per-arm windows are P0=8, P48=64 and bare=64 frames. P48 can deliver at offset 48;
P0 samples two normal H300 submissions. A separate fixed set of 11 P0 one-tick
copies, cycling the four roots, runs at every worker count to expose homogeneous
search pressure. Those copies are engineering load, not extra scientific units.
Both task sets must have identical opaque trajectory hashes across concurrency.

Each pool consists of a fixed owned set of raw spawn processes, without an
executor manager or automatic worker replacement. Every worker initializes the
uploaded checkpoint context and real process-global GameConfig once, and asserts
one Torch intra/interop thread. All w workers report ready before timed work is
submitted; pool startup is timed separately. Root payloads pass through hashed
private local files so large pickles cannot block a queue reader during cleanup.
No Mac-only root is uploaded; target hosts construct their own fresh roots.

Timing wrappers collect inclusive arm wall/CPU time, submit wall time AND actual
H300 search calls/seconds. Submit alone does not prove a search occurred. No
mass, deaths, choices, Q values, certificates or drop reasons are exported.
Actual per-row start/end timestamps allow time-weighted overlap concurrency.
Immutable individual rows retain raw times; the final sweep adds derived mean
concurrency. The draft requires residual contention pricing when active work
falls below nominal w. Neither short homogeneous bursts nor mature windows
qualify full-session tails. Rates remain target-host measurements pending dispatch.

Host evidence includes logical/physical CPU counts, affinity, cpuinfo, cgroup
quota/stat/cpuset/memory, nice and numerical thread settings. Samples record
owned-worker cumulative CPU seconds and RSS, host load and primed host CPU usage;
a post-warmup t=0 sample separates setup from measured work. Worker boot records
assert Torch intra/interop=1. CPU% uses 100%=one logical core.

The engineering work guard is base 450/effective 900 seconds, covering natural
root preparation, worker warmups and all cells; overall whole-session guard is
base 15/effective 30 minutes. These are bounded experiments, not completion
forecasts. Coordinator progress is emitted at least every 30 seconds, including
prefix preparation and warmup. Workers have a 900-second stall watchdog, reset on
completed arm ticks. Parent SIGUSR1/SIGTERM raises a caught interruption so all
owned workers are terminated/joined, then killed/joined if necessary. Worker
signals use default dispositions. Spawn signal deferral covers PID registration.
There is no private executor process dictionary, respawn, or automatic retry.
Completed immutable timing cells remain retained on error, but a partial profile
cannot qualify and missing required exports refuse broker success.

Local `--check-roots` uses exactly one Mac CPU slot and only burned engineering
seeds. It verifies natural maturity and real H300 execution in P0 and P48 before
sealing; local timing is never a Colab rate. A blind unsupported v1 engineering
seed was found before dispatch; that whole engineering allocation was retired.
The fresh v2 key was fixed before its root check. All v1/v2 seeds are excluded
from the successor. Local root caches stay outside Git and are not runtime inputs.
The tiny `--smoke` additionally compares original construction at w1 to exact
restore at w2; hermetic mid-session tests cover real engine snapshot/submit and
48-frame delivery control flow with a cheap search fixture.

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
  --pin FULL_SOURCE_SHA --packet FRESH_ABSOLUTE_PACKET_PATH \
  --preflight ABSOLUTE_LOCAL_ROOT_PREFLIGHT_DIRECTORY
```

The packet contains sealed manifests, bank/exclusion metadata, source hashes,
CLI help/version, exact single-line `canary-command.sh` (L4) and
`compare-command.sh` (optional A100). Append `--dry-run` to those lines for the
real ops-wrapper validation. Executing them without that flag is for the operator.
Never source these files during preparation. Verify seal and current source
hashes before dispatch. No command exists for successor data collection.
