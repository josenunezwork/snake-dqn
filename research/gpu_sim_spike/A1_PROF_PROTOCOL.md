# A1-PROF — standing A100 engine lane, first diagnostic job

Owner instruction, 2026-10-23: keep an A100 lane working on the GPU engine,
steered by Astra. This package only prepares dispatch. Claude/operator dispatches;
Sol does not allocate. Engineering timing only: no labels, scientific decision
labels, gate evidence, training, serving or release authority.

The repaired A1 engine (corpse fix 313d27eb) previously measured 158/163/166
complete world-steps/s at E1024 frozen/mixed/scripted and STOP_ENGINE. Its engine
bytes and results remain unchanged. Round-24 Astra 05 proposed exclusive timings
before a rewrite. Round-28 receives this package's hard numbers; no rewrite is
selected here. The 0.512-second remainder reference is descriptive, never an
automatic verdict or permission to fund a rewrite.

## Frozen work and attribution

Nine units in fixed allocation order: scripted, frozen, mixed crossed with
E64, E256, E1024, six snakes per world, all worlds resident on one A100.
Use the existing A1 ruler's fixed long-body fixture (800,900,1000,1100,1300,1500),
resident reset, full prepare/policies/step/after-step/collection tick. Reset every
tick, as A1 did; this is not a natural-state or long-trajectory measurement.

Main: three warmup ticks per cell, then three repeats of three uninstrumented
full ticks, three exclusively timed full ticks, and one profiler full tick.
All warmups and setup are excluded from throughput and reported separately.
World-steps/s = E × measured ticks / synchronized full-tick wall time, never
snake-steps and never multiplied by E twice. All passes retain engine semantics.

Exclusive attribution uses a stack: movement/collision/events; food/corpse;
vector61 other; vector61 dense flood fill; B8/v8 feature-and-decision bundle; separate opponent v8 bundle; hero
policy forward; opponent policies including scripted anchors; RNG; resident
reset; collection; unscoped tick work. Corpse FIFO insertion and food compaction
are child food/corpse scopes; collision resolution remains movement/events.
Vector61 `_free_space` is a child of vector61; parents exclude child segments.
The existing engine's per-stage profiler is disabled; engine source is untouched.

CUDA events segment the default stream at scope transitions, resolved after
one synchronization at tick end. They measure elapsed stream intervals,
including launch starvation. A separate CPU+CUDA torch.profiler pass attributes
directly correlated kernels to the nearest annotation, verifies the attributed
kernel count equals raw CUDA activities, and records per-stage kernel duration,
launch API count/time, scope self CPU gaps, net device memory and positive
allocations, plus copy direction/count/duration. No inclusive-parent summing.
Host/device times overlap: do not add them. Profiler self CPU gaps approximate
Python overhead, not a language-level CPU sampler. Profiled throughput is not
the headline throughput. Report instrumented/uninstrumented wall ratio.

Report peak allocated/reserved and free/total device memory; separate setup
(including initial model/table uploads) from in-tick transfer activity. Counter
RNG calls are timed separately. Scripted random_safe uses its existing resident
SHA table; its creation is setup, not a newly introduced tick RNG draw.
Flood share = exclusive flood CUDA elapsed / instrumented tick wall on CUDA
(CPU wall on the smoke); report host enqueue share separately. Non-observation
remainder = instrumented tick wall minus exclusive CUDA elapsed vector61 + B8/v8
feature intervals (CPU wall on the smoke), retaining hero/opponent
policies, RNG, transfers, reset, collection and measurement boundary/sync cost.
Also report non-observation CUDA elapsed sum and a conservative remainder that
retains the entire opponent v8 bundle. v8 computes floods and action reranking
together; broad observation subtraction removes the full v8 calls, so it is an
optimistic pure-observation remainder. The separate opponent bundle lets Astra
inspect this ambiguity; opponent policies means network/scripted anchors only. E1024 rows carry 0.512 seconds as a
reference. No performance outcome label is produced.

## Qualification, caps and preparation

Qualified research/colab_cli only, via tools/ops/dispatch.sh. No transport edits,
no Drive runtime paths, Python 3.12 uv bootstrap, cu126 A100, hash-bound uploaded
checkpoint catalog and empty verified ledgers; account cap three including other
workers, live calibrated positive rates refreshed by dispatch, first cap wins.
A100 model check and native flood/corpse selfchecks precede measured work.

| Envelope | Base | Effective after owner's 2x rule |
| --- | --- | --- |
| Main whole-session wall | 45 min | 90 min |
| Main execution cap | 10 u | 20 u |
| Main reserve | 2 u | 4 u |
| Main workload wall | 1,200 s | 2,400 s |
| Canary whole-session wall | 15 min | 30 min |
| Canary execution cap | 2.5 u | 5 u |
| Canary reserve | 0.5 u | 1 u |
| Canary workload wall | 450 s | 900 s |

Forecasts (not doubled): main 30 minutes, canary 10 minutes, including setup,
exports and cleanup; 40 minutes total, approximately 5.13–6.67 u at planning
7.7–10 u/hour. Admission ceiling 10 u/hour, not a measured provider rate.
Combined reservation 30 u including reserves; maximum whole-session walls sum to
120 minutes, workload walls sum to 3,300 s. This is a bounded first job in a
standing lane; it does not leave an idle runtime or relaunch itself indefinitely.

A fresh same-pin A100 canary is mandatory before main. It runs all nine cells,
one warmup, one baseline tick, one timed tick and one profiler tick each to
exercise production memory shapes and correlation logic. Canary data stay in a
separate namespace. Main remains unbound until its verified PASS receipt exists.
Canary and main upload identical inventories, including both ledgers, and require
identical producer exports. Mac preflight uses the same Bash entrypoint and
manifest-style environment, CPU E1 all rosters, one warmup and three real ticks
per pass; it cannot qualify CUDA. Expected native activity is checked on the VM.

## Immutable units, resume and analysis

Namespace binds full published source commit, A100 and full/canary/smoke mode;
unit ID is the frozen roster/shape index. Each cell publishes atomic create-only
JSON, SHA sidecar, then completion marker. Partial cells never count. Total
exports bounded at 64 MiB. Final summary and supervisor terminal receipt retain
identity/progress/timings; terminal success additionally needs Mac-verified
exports and independently confirmed backend release. A local exit code is not
release evidence.

The first attempt uploads an empty ledger. `a1_prof_units` builds a recovery
ledger on the Mac from filenames/size/hash/markers only, without opening timing
content. Runtime verifies wrapper namespace/allocation and byte identities,
skips exactly verified cells, and does not republish them in the new result set.
`a1_prof_analyze` on the Mac alone opens the full union, refusing duplicate IDs,
foreign namespaces/allocations and missing cells. Remote summaries are sealed
operational diagnostics. Preserve original units, manifests and receipts.

At most two manually dispatched completion attempts after verified mechanical
OOM/transport/time-cap failure, independently verified stop and missing units.
Fresh manifests, IDs, outputs and receipts; all attempts share the original
registered wall/unit envelope, including setup/export/failed attempts. No
outcome-based rerun and no automatic completion retry. A recovery pack requires
reviewed remaining envelope and a new matched canary if input inventory changes;
the first-job builder intentionally cannot silently invent a retry envelope.
The dispatch wrapper's separately authorized exact accounting-allocation retry
is unchanged and does not expand this series budget. No trajectories are resumed.
