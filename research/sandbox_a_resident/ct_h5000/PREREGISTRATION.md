# T34 C/T H5000 natural-support successor — 2026-10-29

Status: prospective finite simulator-D development study, prepared at the owner's
2026-10-29 request. No collection data exist for this namespace. No dispatch is
performed by this implementation task. The Standing A100 Development Sandbox
(`docs/research/governance_a100_sandbox_20261026/ADOPTED.md`) supplies the sandbox
process; independent Argon review is required before operator dispatch. This is
not a strict gate, release registration, extension, or continuation of t34-cf-1028.

Question: does a fixed natural H5000 bank supply the unchanged late-death C/T
support? This packet ends after collection/support audit. A complete support bank
can motivate a separately prepared branch/fit packet; it is no C/T rescue,
learnability, topology, Watch6-v2 transfer, or release result. The old full three-arm
package's multi-hour forecast is not a two-hour collection promise.

## Frozen design and identities

- Study: `t34-ct-h5000-1029`; salt `a100-dev/t34-ct-h5000-1029/`; stages
  `a100-dev/t34-ct-h5000-1029/stages`. Allocation index 0, seed offset 0.
- Natural collection: exactly 1,024 worlds x 5,000 ticks = 5,120,000 charged rows
  when complete. Never stop because support appears sufficient. Never extend
  because support appears insufficient. No synthetic long starts or outcome-driven
  world replacements. A death ends the hero life and resets its entire world with
  the next pre-audited life seed, as in the predecessor.
- Whole-world partition and rosters unchanged: scripted/frozen/mixed 342/341/341;
  train 820 (274/273/273), assessment 204 (68/68/68). All lives, anchors, deaths,
  lookbacks and ordinary roots inherit their generating world's partition.
- Bare released B8 s3 SHA256
  `3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`.
  The same registry-pinned FRP3 s12 and champion opponents; no training, inference
  wrapper, roster or mechanic changes. CTDeviceSim mechanics-v1 terminal-hero
  skill task, in-step refill, continuous opponent respawn, 10,400 body capacity.
  Existing startup forced-state parity and geometric fixtures run before collection.
- Death eligibility unchanged: pre-step hero length >=600, all eight same-life
  lookbacks `{15,35,65,100,140,185,235,285}`. Up to 160 deaths / 1,280 lookbacks,
  up to 1,280 matched ordinary roots, 50,000 unsearched anchors. Frozen original
  death selection, exact `(partition,mix,length_bin)` quotas, bins
  `[1,600),[600,800),[800,1000),[1000,1300),[1300,1700),[1700,inf)`.
  Sub-600 early lookbacks remain valid. Zero cross-cell substitution.
- Exhaustive seed audit: 4,100,820 training + 1,024 validation + 1,020,204
  assessment = 5,122,048 effective 63-bit seeds. Reject aliases and intersections
  with the committed protected registry, all 1,056 TRAIN-01 monitor/nomination
  seeds and all 2,623,488 audited predecessor C/T seeds. No old observations enter
  this bank. A fresh namespace is not merely an old bank with a larger horizon.

## Ordinary sampling repair, frozen before data

Algorithm R uses sequential reserved slot counts within each tick. Only the last
writer of an admitted slot is materialized, with exact equivalence to serial
admission. There is no above-cap immutable tail. Each mix/bin cell holds its full
partition quota plus a fixed **64-row overlap allowance**: 1,088 training or 320
assessment candidates. Final death-lookback overlaps are excluded; quota rows
are uniformly subsampled with a deterministic cell-keyed RNG before stable output
ordering. Reservoir selection never conditions on future survival. More than
64 excluded overlaps or too few natural candidates can still leave a shortfall;
record it without refilling from another cell or weakening a bar. The fixed anchor
schedule and its chronological quota retention remain unchanged; this archive is
not claimed to be a uniform census of late living states.

## Support bars and ordered outcomes

Use the unchanged `ct_stats.validate_ct_support` implementation:

1. Partition leak => `INVALID`; no scientific support conclusion.
2. Cap before complete bank => `INCOMPLETE_COLLECTION`, reason
   `COLLECTION_WALL_CAP`; support **not evaluated**, no partial bank stage accepted.
3. Complete H5000 bank must have >=32 assessment eligible deaths from >=32
   distinct assessment worlds, >=256 searched assessment ordinary roots, and
   >=10,000 assessment unsearched anchors. Training retains >=1 eligible death,
   >=1 searched root and >=1 anchor. Every exact mix/bin quota must pass.
   Any missing support => `INCOMPLETE_SUPPORT`, no added collection or rescore.
4. Complete bank plus all support checks => `SUPPORT_COMPLETE`, meaning only
   this development bank supplies the registered support. No branches, fits,
   rescue gates, candidate crossing or serving decisions are executed here.

Do not inspect support yield before the terminal registered look. Operational
progress may report counts already emitted by the predecessor, without changing
horizon, thresholds or sampling. No intermediate bank substitutes for H5000.

## Frozen time, resource and transport envelopes

- Collection base 3,600 s, effective **7,200 s (~120 min)** under the owner's 2x
  rule; clock starts before simulator/collector initialization and includes bank
  assembly. Check at tick boundaries and candidate/finalization work boundaries.
  An in-flight atomic tensor operation can finish before the next check; the cap
  does not preempt a CUDA kernel. No new tick is admitted after expiry. Cap expiry
  emits a completion-last terminal summary with actual completed ticks and rows,
  without accepting a partial stage or opening a partial support look.
- Host estimate/base runaway envelope 150 min; effective session timeout 300 min
  permits startup, the 120-min collection cap and bounded export/cleanup. These
  session envelopes do not enlarge the collection horizon/cap. Informational
  unit fields price the whole 300-min envelope: H100 60 units, A100 40 units;
  standing no-unit-budget policy applies. No measured estimate/rate/N is doubled.
- Historical A100 E1024 late-rate extrapolation: 115.4–125.4 min end-to-end,
  12.9–14.1 units at observed predecessor 6.73 u/h; 15.4–16.7 units at the packet's
  8 u/h A100 planning rate. Sampling repair and mature-state memory/export costs
  are unmeasured. H100 has no production collector timing; its 12 u/h forecast
  is provisional, ~23.1–25.1 units if wall time were unchanged. H100 speedup and
  enough mature deaths are not assumed. A cap stop is a valid incomplete outcome.
- H100 first, exact A100 fallback derived prospectively by
  `research.compute.gpu_preference`; distinct job/output identities. No manual
  direct allocation calls. Operator uses only qualified supervisor through
  `tools/ops/dispatch.sh`; three-assignment cap and no registered-job sharing.
- Existing single-thread launch env, CUDA precision rules, early liveness,
  <=900s meaningful-work watchdog, SIGUSR1 fence, immutable SHA/completion-last
  stage exports, 256 MiB stage / 384 MiB recursive caps and 1 GiB transfer bound.
  Reservoir in-memory capacity grows: 25,344 candidate roots maximum across 36
  cells, around 2.8 GiB for dense topology/observations before snapshots and Python
  overhead. This is a projection, not a qualified memory measurement. Overflow
  refuses; it never truncates bodies or silently changes selection.

## Resume and immutable evidence

Initial packet uploads an empty ledger bound to the full control hash. The sole
exported unit is stage-0001, a complete H5000 bank and support audit. The runtime
verifies namespace, allocation, file sizes/hashes/sidecars and full completion,
then skips exactly that verified stage. An unfinished collector has no eligible
full-state snapshot; it must not be resumed from a living anchor or a partial
bank. The capped incomplete outcome is terminal, not a retry trigger.

A mechanical failure may justify at most two manual outcome-blind completion
attempts under the standing resume policy, only after verified stop and missing
units, preserving all prior evidence and frozen science. This initial packet's
admission refuses continuation metadata; a separately sealed, provenance-verified
completion packet and matching canary are required before any such dispatch.
There is no automatic completion retry. A completed scientific support shortfall
or collection-cap outcome is never mechanically retried to improve support.

The source pin, audit, control, models, topology, empty ledger, fallback preview
and complete source closure are bound by the new packet's seal. Mac preflight
runs the real telemetry entrypoint on tiny synthetic inputs, including cap expiry;
it proves entrypoint wiring, not production support yield or Linux/CUDA timing.
