# T4A-DEV-01 — pre-data GPU batch invariance protocol

Status: FROZEN_PRE_DATA. Freeze is the first publication commit containing this file;
the executable candidate is the later clean published packet commit, bound in both
manifests. No T4A CUDA data existed or were inspected when this protocol was written.
CPU fixtures and injected-fault tests are implementation checks, never A100 evidence.

Authority: Fable `exp/fable-20261024/JUDGMENT_A100_NEXT_JOB.md`, whole ruling,
especially §2.1–2.2 and §3; `exp/JUDGMENT_GPU_LABEL_PATH_20261024.md`, whole
ruling, especially §4.2. Original rulings remain immutable and are uploaded by hash.
This is the admitted standalone, outcome-free premise test, not S0 or a producer.

## Candidate and question

Candidate: unchanged R2 I′ mechanics at `e66a701fd43951dc80e976485ed77113a91dddcb`,
assembled through `r2_bench.build_case`, plus versioned T4A padded-forward wrapper v1
and heterogeneous fixture/remap instrumentation in new `t4a_*` modules only. No
mechanics repair and no R2/R6 unit edits. All runtime sources, checkpoint bytes,
authority inputs, protocol, dev allocation, toolchain and native binaries are bound
to the packet and current-pin native-canary receipt.

Hypothesis: on that A100 configuration a tracked world's outputs and persistent
state depend on its logical history, not E, neighbors, row position or padding.
All networks eval/inference mode, fp32, TF32 off, cuDNN benchmark off, deterministic
algorithms required, cuBLAS deterministic workspace configured before CUDA. Native
compile flags must have no fast-math/TF32 flag; retain binary/source hashes.

Every actual learned-network forward (hero B8 raster triple and learned vector61
opponents) is chunked and zero-padded to exactly **1,024 input rows**, including
the last partial chunk. Every input tensor has the same row count, dtype and
trailing shape for a given model. Return only real rows in their physical order;
never sort tracked worlds into canonical order before inference. Record/check
every inner input/output shape. Simulator filler lanes are valid sacrificial
worlds, distinct from inference zero-padding. If this cannot be implemented,
report **NOT_READY**, with no reduced matrix and no dispatch.

## Excluded roots, seams and layouts

Fresh synthetic dev namespaces only; no production allocation, label, rescue,
evaluation outcome, or `t34-round0-study-01/` access. Per mode the ordered
frozen/mixed/scripted seed roots are:

| Mode | Seeds | Progressing history offsets | Horizon |
|---|---|---|---|
| main | 9402401, 9402402, 9402403 | 0, 3, 11, 23 | 256 |
| canary | 9402411, 9402412, 9402413 | 0, 3, 11, 23 | 2 |
| CPU smoke | 9402421, 9402422, 9402423 | 0, 3, 11, 23 | 1 |

There are three independent roots and twelve tracked cases per mode. Each batch
uses four histories of its single roster/seed root, preserving scripted tables.
All captured seams are at the same lifecycle boundary: post-step, unprepared sim
and vector carry. Remap every world-indexed tensor, RNG key/cursor and runtime
carry with its logical identity; preserve the IndexedVectorRuntime class and
configuration. Filler histories include short/long bodies on both sides of a
256-segment live-window boundary, dead/respawning neighbors and varied food/carry.

Physical E is **4, 64, 1024**. Three layouts:
canonical (tracked slots 0–3), reversed (tracked slot i at E−1−i), interspersed
(tracked slot i at floor(i×(E−1)/3)). Fill other rows with valid sacrificial
worlds. The inverse map is checked independently against logical IDs and is
validated by a deliberate wrong-map control. E4 has no spare sacrificial rows.

For each roster run E × layout × replay {0,1}, H256: 18 configuration replays;
**54 replays / 13,824 batch ticks** total. Compare all four targets every tick
against the canonical E4 replay-0 tape and compare repeated tapes exactly.
Canary uses the same E/layout/replay matrix, disjoint roots and H2; CPU smoke
uses E4 only, H1, and is explicitly not this certificate. No best-cell choice.

## Separate event coverage block

Five independently reset, deterministic dev fixtures, each two progressing
ticks, on every roster and E with canonical target mapping; not substituted for
any matrix replay. Fixture edits are setup only; run ordinary candidate prepare,
policy, step and carry paths. Freeze concrete setup code in the candidate pin.

1. Ring wrap: place a live ring head at the last storage position and require
   observed movement across index zero, preserving logical body order.
2. Simultaneous death/respawn: force two opponent deaths in one tick, then ready
   timers and require ordinary prepare to respawn both; require corpse/death and
   respawn evidence, not merely configured timers.
3. Contested food/refill: route two living heads to one shared food cell; require
   consumption and ordinary refill/ordered food change, with RNG movement recorded.
4. Clone/restore: clone at an unprepared seam, progress, restore full state/carry,
   replay identical steps and compare all fields including RNG cursor.
5. Stale-cache boundary: prime R2/geometry/vector caches, clone/remap/restore,
   invalidate through ordinary lifecycle, and compare with a fresh equivalent
   world; assert expected prepare and cache lifecycle actually executes.

Each event needs an executed witness and exact mapped comparisons; absent or
failed-to-exercise coverage is INCOMPLETE. No event evidence is inferred from
fixture labels. Canary must exercise this full block too.

## Comparison inventory and negative controls

Compare raw bit patterns on device, not approximate values, tolerances, argmax
alone, or a digest-only equality test. Inverse-map only for comparisons. Record
compact typed field/tick digests, exact action/raw-Q tapes for all six seats
(Q present wherever a network acts), and a bounded first mismatch including
roster/logical ID/E/layout/replay/tick/field/byte location. Nonfinite Q is invalid.

Inventory: all world-indexed simulator tensors, including full ring padding,
heads, seg_count, length, direction, alive, boost/hunger/respawn, ordered food and
corpse state, rewards/done/death causes/kills/food/transition flags, frame and RNG
key/cursor; all vector-runtime feature/carry/memory tensors; all six actions,
unmasked network Q, advisory/legal/v8 masks and observations; collector outputs;
prepared flags and fixed policy/table/config identities. Discovering an unknown
persistent field must refuse evidence, never silently omit it. Separate
E-dependent accounting scalars from per-world values and reconcile them against
independently computed sums; timing/instrumentation and empty transient geometry
views are not world state. Nonempty transient views are invalid stale carry.

Before data, validate comparison controls: one-bit Q, state and cursor faults,
plus deliberately wrong inverse map. Each must be detected in its expected
field, with an actionable witness. Controls operate on disposable copies;
never repair a compared data prefix after observing it.

## Prospective decision order

1. **INVALID_EVIDENCE**: pin/input/inventory/map/native/negative-control failure,
   illegal access, nonfinite outputs, malformed or foreign/duplicate evidence.
2. **INVALID_SELF_CONSISTENCY**: any valid-prefix action/Q/state/RNG byte mismatch.
   Quarantine this execution configuration. Repair requires a new candidate,
   review and canary. No mismatch retry or tolerance amendment.
3. **INCOMPLETE**: missing unit/replay/tick, timeout/OOM, or unexercised required
   event, with no demonstrated valid-prefix mismatch.
4. **PASS_T4A_DEV**: every matrix replay and event completes, every field compares
   exactly, controls pass, and native current-pin exports verify on Mac.

Canary emits PASS_CANARY only, never PASS_T4A_DEV. CPU emits PREFLIGHT only.
The Mac reducer alone assigns a scientific verdict across verified roster units;
remote terminal summaries remain sealed diagnostics until authorized reduction.
No PASS implies CPU equality, S0 admission, label/evaluation authority, F4 change,
serving qualification or release. Repeat T4(a) on any eventual production pin.

## Frozen operations, timing and resume

Sol build ≤8 h plus independent Codex-V review. Native compile proof is included
in the fixed-pin A100 canary before matrix work; no separate platform allocation.
Series envelope **120 min base / 240 min effective**, owner 2× applied once:
canary 15/30 min, main 105/210 min, including setup/compile/export. No Colab unit
budget. The existing broker's numeric unit-cap compatibility field is derived
from effective wall limit and priced rate plus cleanup allowance, cannot bind
before that wall guard, and is not an independent science/spend ceiling.
Mac tiny real-entrypoint smoke records actual elapsed and per-roster timings;
forecast is descriptive and canary must confirm the full main fits. If it does
not fit, stop and re-plan pre-main; never shrink H/N/matrix after data.

One immutable unit = entire roster matrix plus its event block. First attempt
uploads an empty verified ledger. Namespace includes full candidate SHA and mode;
allocation is fixed roster-frozen, roster-mixed, roster-scripted. Atomic immutable
payloads + SHA sidecars + completion markers last. Mac verifies path/size/hash
and exact namespace/allocation; skip exactly verified units. Never recompute a
verified unit into a result set. Incomplete roster work remains diagnostic and is
not eligible for skip. Mechanical outcome-blind completion only: original plus
at most two MANUAL attempts after verified stop, mechanical timeout/transport/
interruption, and missing units. Fresh manifest/job/output/receipt each time;
preserve evidence, one Mac aggregate, no automated retry or behavioral repair.
Per-attempt envelopes remain frozen as above, maximum three attempts. Two-minute
telemetry and bounded exports required. No unit selection by result.

Dispatch only via qualified `research/colab_cli` and `tools/ops/dispatch.sh`,
canary then independently reviewed main, A100 only, account cap three. A100
yields to each attempt-2 L4 wave. Operator checks scheduling; this build does
not inspect/control the live ruler or Mac factory, tmux sessions, their state
directories or `exp/mac-lease.json*`. No provider calls or dispatch during build.
The Fable-ready queue addendum is R6 done → T4A-DEV-01 → (ruler PASS without
W12_BELOW_4) S0. The exact-engine program stays PARKED_NO_CONSUMER.
