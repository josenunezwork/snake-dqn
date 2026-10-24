# T4A harness diagnostic memory repair

Disposition: preparation only. Stage02 consumed scientific attempt 1 and ended
INCOMPLETE / MECHANICAL_HARNESS_OOM. The successor canary is attempt 2, at a new
source namespace with both ledgers empty. Main remains blocked. Authority is
`JUDGMENT_T4A_STAGE02_HARNESS_OOM.md`, SHA-256
`e8c9ce98e9de7b8ca54749a5db71b67201af310ad534bc1f3d51ee6581198dd6`.

## Byte identity

The frozen oracle is the complete, unedited `t4a_compare.py` from
`c9fe56465e1498c6c0709fc4d9ade7c4415afc43`, vendored at
`tests/fixtures/t4a_oom/t4a_compare_c9fe.py`, SHA-256
`95a0485e16f2ea58928962849ab88f8525cde60233bf5f4474537d62d8d2f5e2`.
The candidate retains algorithm ID `t4a-device-mix64x4-v1`, all four salts,
the same per-byte mix with signed arithmetic shifts, lane order, and 32-byte
serialization. Every byte of the full ring is visited in original order, with
its global offset. Int64 reduction and accumulation compose in Z/(2**64), so
partitioning the sum cannot change any lane bit, including overflow.

Tests compare frozen/candidate bytes for all byte values, signed extremes,
float bit patterns, noncontiguous inputs, empty input, boundaries and tails.
Global-offset tests compare chunk results against differences of two frozen
prefix digests. The authoritative raw device byte comparator is unchanged;
no digest is an equality oracle.

## Memory model and its limits

For contiguous production rings, byte views allocate no ring copy. At most four
int64 vectors of 1,048,576 entries coexist: expanded byte window, global
positions, mixed values and one shifted temporary. Scalar lanes and reduction
outputs are covered by 256 additional bytes. The explicit tensor scratch bound
is **33,554,688 bytes**, independent of E or ring length. Backend reduction
workspace and allocator reservation are separate. Noncontiguous callers retain
the old contiguous-copy behavior in addition to this scratch bound.

`test_real_chunk_peak_tensor_scratch_bound` observes live tensor allocations
through TorchDispatchMode at the production window size and a large global
offset. `test_full_e1024_ring_walk_has_fixed_scratch_and_no_skipped_bytes`
walks a meta tensor of the exact production shape, proving contiguous,
nonoverlapping coverage of all **1,022,361,600 bytes** without allocating the
1 GiB input or any full-ring expanded array. It combines this structural
schedule proof with the separately exercised real CPU chunk kernel.

The E1024 lifetime inventory, using S=6, cap=10400, food cap=450 and the
deployment grid145x83, is deliberately conservative:

| Component | Inventory / modeled bytes |
| --- | --- |
| Full body ring B | 1024*6*10400*2*8 = 1,022,361,600 |
| One full persistent sim + vector runtime | 1,034,461,200, including non-ring fields/counters |
| Clone/restore | original+template, saved, branch B, restored-left/right snapshots; at most seven persistent-pair equivalents including a progression snapshot |
| Stale-cache remap stage | source+template, clone, remapped, remap replay; five pairs |
| Stale-cache resident restore stage | source+template, clone, resident+template, progression snapshot, replay; at most seven pairs |
| Stale-cache final controls | source+template, clone, equivalent+template, control+template; seven pairs |
| Persistent-state envelope | 7*1,034,461,200 = 7,241,228,400 |
| R2 cached TTL/occupancy/count/union/advisory | conservatively seven caches, each <=1024*12035*(9*6+2)+6*1024*6 = 690,171,904; total 4,831,203,328 |
| Unchanged raw comparator | PASS: B-byte bool result; dense mismatch: B-byte bool + up to 8B-byte nonzero index output, conservatively 9B = 9,201,254,400 |
| Digest explicit tensor scratch | 33,554,688 |
| Mapped tapes/output captures/models/constants | 1 GiB modeling reserve; E4 mapped body captures are 3,993,600 bytes each, never full E1024 snapshots |
| Engine/forward scratch | 8 GiB modeling reserve, including full-capacity body/v8/mask temporary estimates and native flood queues/distances (591,544,320 bytes), component arrays (98,590,720 bytes), CNN activation (125,960,192 bytes) |

This model is about 28.85 GiB before CUDA context, cuDNN workspace and allocator
reservation. The last two reserves are conservative planning allowances, not
proven backend allocation limits. No offline test certifies that the complete
GPU workload fits a particular device. The fresh full canary must establish
actual allocated/reserved peaks on the unchanged A100 scope. The harness emits
create-only `device-memory.json` over the entire invocation; the packet requires
that export. Independent acceptance must check status MEASURED, allocated and
reserved peaks, total device capacity, all full-E event cells, and the scientific
reducer. UNAVAILABLE or a missing memory export cannot qualify memory.

## Lifetime preservation

Only dead references are released: progression snapshots after their predicates;
remap cases after raw comparisons, immediately sealed mismatch witnesses,
digests, mapped prepared captures and tape removal; resident restore/replay
cases at the same completed boundary. All later readers retain owned mapped
captures, scalar results or sealed bytes. The fixture transitions, full rings,
model/precision, native sources/flags, raw comparator and verdict precedence
remain unchanged. An event scope finally removes only event-created side-table
tapes, on success or exception. Returned witnesses keep their tensors, and a
regression checks their immutability after cleanup. Resource telemetry failures
cannot override an existing scientific verdict.

## Operator boundary

Use the immutable sibling artifact packet under
`exp/t4a-dev-01-20261024/harness-oom-repair-01/pin-<NEW12>/`.
Its create-only preparation procedure retains the unchanged compile binding,
canary binding, Mac reducer/forecast and final upload-identity validators. A fresh
same-pin compile is required; no old receipt or cross-pin completed unit is
reused. Native report and all five receipt-bound binaries must be flattened
into the one metadata archive, with public-inputs.json direct, for both canary
and main. The actual native bytes cannot be staged until that compile exists;
offline synthetic receipts are validator tests only and never admission evidence.

After independent appointed-judge delta ACCEPT, operator Stage 1 is ONE 15/30
minute base/effective compile. Only after its released PASS does Stage 2 prepare
the bundled inputs and permit ONE full attempt-2 canary, 10/20 minutes with
producer720s. Any failure stops this authority. Pre-main is25/50 minutes;
including still-blocked main105/210, the successor chain is130/260. Historical
allocations/debits remain bound separately. Main producer6000s and all original
science, forecast, upload identity, scheduling and independent acceptance gates
remain. This change authorizes no dispatch, automatic retry, third attempt,
release or serving change.
