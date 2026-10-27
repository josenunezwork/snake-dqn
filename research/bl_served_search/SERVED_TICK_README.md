# Build 0: served Watch6 timing diagnostic

This package implements the first measurement slice of Astra's
`SERVED_SEARCH_PLAN_20261026.md` (2026-10-26), with the owner's Mac serving
constraint. It is an engineering diagnostic, with no search workers (W0/K0),
policy change, serving integration, release or qualification authority.

The primary advice is at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/astra-20261024/SERVED_SEARCH_PLAN_20261026.md`.
Argon's secondary advice is at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/argon-20261026/served-search-plan-01/answer.md`.
The first package follows Astra: default GC, existing Python serving code and
ordinary inherited QoS. Argon's native live-v8 switch, GC suppression, QoS
changes, shared-memory search and two-stage trigger remain alternatives for a
later measured repair. No assumed speedups or altered serving bars enter this run.

## Frozen panel

`served_tick_manifest.json` fixes three fresh domain-separated seeds, one separate
observer-control seed, 24 fps, 200 warm-up ticks and 2,400 measured ticks per cell.
Each seed receives timer-only, current inline Watch6, fresh-process Watch6 and
real Hub/WebSocket cells in the committed interleaved order. Preserve every cell
and the entire attempted exposure. Do not rerun failing cells until a favorable
panel appears. A failed/incomplete attempt uses a new directory if repeated.

The full uint64 domain seed remains the world identity and seeds Python and
torch. The legacy NumPy seeding API accepts only uint32, so a constructor-local
adapter projects that library's seed to the low 32 bits. Both identities are
recorded and checked against registry exclusions; the production seed helper is
not edited. The adapter still calls the actual `m1.new_watch(full_seed)`.

The constructor is the actual `research.bl_k1.m1.new_watch()`: bare B8 in slot 0,
five FRP3-s12+v8 opponents, continuous respawn. T15's constructor disables release
telemetry side effects; it does not exercise production release-monitor logging.
The real Hub endpoint retains its processing-then-full-sleep scheduling. Its
cadence/client measurements are reported separately from the absolute-clock
loop; a direct-loop PASS does not establish production client cadence.

The dedicated game process owns the same complete step+snapshot workload. This
isolation diagnostic transports a bounded immutable result after the workload.
It does not move DTO/rendering work into another interpreter, measure search IPC,
or constitute a production authoritative-game-process service.

Wall and tick-thread CPU clocks, wake lateness, update, step, snapshot/DTO, nested
hot spots, GC overlap and available lock/IPC spans are recorded. Nested spans are
inclusive, with exclusive timings where available: never add a parent and its
children. Unavailable endpoints are identified explicitly. Wall-minus-thread-CPU
indicates waits/descheduling possibilities, without assigning a unique cause.
GC overlap does not prove the collector caused a deadline miss.

Complete-state/action/DTO parity runs outside the paced timing window. Source,
checkpoint, registry, interpreter and environment identities accompany raw traces.
State hashing, shell telemetry and disk exports must not be inserted into a
scored tick. The observer control reports instrumentation cost rather than
subtracting it from the serving bars.

## Commands and Mac window

From any terminal, first check the machine and immutable panel without starting
a game, acquiring a lease or creating an output directory:

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --check
```

On an available Mac with AC, lid open and no other heavy local jobs, run the fixed
panel into a fresh directory:

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-01
```

Recompute the report from the verified raw and host payload triplets into another
fresh directory (seconds, with no gameplay):

```bash
/Users/josenunez/Projects/ml/snake-dqn-served/research/bl_served_search/run_tick_probe.sh --reduce /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-01/raw.json --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/served-tick-20261026/panel-01-reduced
```

`--smoke --output <fresh-directory>` runs a tiny inline/fresh-process functional
parity check. It does not acquire an exclusive timing lease or qualify latency;
it still uses CPU, nice/thread settings and read-only host evidence. Use the full
panel only after the committed source and diagnostic manifest are stable.

Allow **45–60 minutes** including fresh process/model warm-up, observer/parity
controls and trace export. This is Astra's forecast, not measured throughput.
The runaway envelope is base 60 minutes, effective **120 minutes (2x)**, fixed
before data. The nominal 12-cell measured exposure alone is at least 20 minutes;
the real Hub's processing-plus-sleep cells take longer than 24-fps ideal pacing.
Do not launch the long panel while another timing lease or heavy job is active.

The launcher uses the primary Python venv, CPU, `nice 15`, all five math thread
caps at 1 and `caffeinate -dims`. A Mac-wide lease fences the run. A separate
monitor interpreter records AC/lid/thermal state, raw process inventory, CPU load,
RSS, process/thread inventory and tmux sessions about every 10 seconds. Ordinary
system daemons remain running and their load is retained. Foreign heavy jobs
(>=20% of one CPU) contaminate and stop the attempt; no process is stopped to
manufacture a quiet host. Unknown power/thermal state and a stale monitor fail
closed. Point samples cannot certify the absence of every short load burst.

Raw traces, host evidence and reduced report use create-only payloads with SHA256
sidecars and completion markers. Existing outputs are refused. Failed attempts
retain their diagnostics. The separate reducer verifies inputs rather than
trusting a previous PASS label.

## Decision and boundaries

An absolute-clock tick finishing **at or after** its next target is an overrun.
The update endpoint is `GameState.update`, with **p95 <=11.688 ms**; it is not the
full frame and not a per-tick maximum. For a candidate absolute-loop layout,
both zero absolute overruns and the update p95 bar must clear every required
cell. Missing exposure is INCOMPLETE, malformed/overflow/parity/host evidence is
INVALID, and a completed timing violation is FAIL. A PASS is engineering evidence
for this finite panel only.

The measurement chooses the next bounded repair: update CPU hotspot if both
layouts fail the update bar; waits/timer/QoS investigation for wall-only stalls;
DTO/process separation if snapshot allocation dominates; or a later W1/K1 then
W4/K4 exact-search contention package if bare timing clears. W4/K8 remains a
later stress arm under the owner's W<=4/K<=8 search limits.

The ten-minute finalist confirmation, long-body qualification fixtures,
copied-state renderer split, search delivery/deadline/crash/slot protocol, fresh
delivered-value screen and serving qualification remain later work. This package
does not inherit the offline L2-prime gain, revise historical M1 FAIL/KEEP_B8,
enable assistance or authorize changing the served default. Release remains
the owner's decision.
