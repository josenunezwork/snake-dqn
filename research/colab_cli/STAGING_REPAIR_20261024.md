# F4 staging control repair — review handoff

Status: independent product-code review **ACCEPT**; prepared for Astra review;
**live qualification pending**. Authority:
`JUDGMENT_F4_TRANSPORT_REPAIR_D4096_B1.md`, SHA-256
`03f74beea60f819f657613b62b933c0a3102407d09bc4d284ea188fb3ff48e1b`.
Only Q1 transport repair is implemented here. D4096 science, release authority,
F4 training source and mathematical recipe, scientific input bytes, one-look
rules, and existing attempts are outside this change.

## Failure evidence

The two immutable D1024 canary receipts are under
`snake-dqn-artifacts/big-lever/exp/f4/d1024-pack-02/`:

| Receipt | SHA-256 | Failed control calls | Terminal disposition |
| --- | --- | --- | --- |
| `canary-output-l4-01/receipt.json` | `ba35d8e4754041c45267913f571a01ea6a70f73f62fea8b68f52aeeef90e21dd` | calls 18/28, 120.006381542 / 120.006683917 s | INCOMPLETE / UPLOAD_CAP, stop verified, no terminal/exports |
| `canary-output-l4-02/receipt.json` | `9d0f0194f693a5e8040339e295e8405d3260b483540816dfe2ee4fd3b56ccc05` | calls 28/48, 120.010562000 / 120.010677125 s | INCOMPLETE / UPLOAD_CAP, stop verified, no terminal/exports |

The first receipt's two failed execs occurred outside the recorded transfer
wrappers; source-flow reconstruction places them at repeated parent setup.
The retry's recorded verified transfers include slow Contents writes and
control/recovery waits. The exact bodies of these historical untagged execs
are unproved. Heartbeat durations overlap foreground work and must not be
summed as sequential elapsed time. Neither receipt is scientific evidence.
See `docs/research/colab_upload_cap_diagnosis_2026-10-24.md` and the independent
`JUDGMENT_F4_D1024_CANARY_MECHANICAL_RETRY.md` for the evidence interpretation.

## Implementation boundary

- All local upload parents are deduplicated and created in one idempotent
  kernel call. Each file retains its hash and size checks.
- Uploads retain direct PUTs for small files and the same 2 MiB Contents chunks
  for large files. Chunk assembly checks the complete digest and size, fsyncs,
  and atomically publishes before acknowledging. Its durable UUID result is
  the per-file verification; the repeated full-file hash RPC is removed.
  Direct small PUTs still receive independent remote hash/size verification.
- The original staging start still precedes parent setup, uploads, renewal,
  verification, cleanup, and source delivery. A thread-local context bounds
  every foreground control/recovery/helper to the same remaining 300 s.
  Control exec attempts are at most 30 s; recovery including idle proof and
  backoff shares at most 60 s, further clipped by staging/session caps.
  Expiry refuses work; it never resets a per-file global staging allowance.
- Lease renewal stays generation-fenced and throttled to 30 s. Credential
  rotation helpers are also clipped during staging. Independent Contents
  heartbeat calls keep their separate session-bound thread context so an
  exhausted foreground control allowance cannot starve the watchdog. Stop
  verification runs after the staging context unwinds and retains its cleanup
  allowance. No uncertain kernel mutation is blindly replayed.
- PUT, mkdir, verify, assembly, renewal, cleanup and recovery-idle operations
  now retain monotonic start/end times, elapsed time, thread and status in
  receipt evidence. Foreground/heartbeat overlap is explicit. Parent setup is
  reported separately; `transfer_seconds` remains the sum of whole-file
  verified-transfer wrappers, and total staging includes all setup and source
  delivery. The existing canary thresholds remain <=150 s and <=300 s.

## Offline regressions and live handoff

`tests/test_colab_cli_staging_repair.py` replays both two-stall patterns through
subprocess-backed fake Colab and real broker recovery. The first replay uses
batched parent setup and chunk-directory creation; the second conservatively
uses two independently fenced, completed-but-unacknowledged assemblies plus
recorded slow PUT timings. The selectors identify distinct synthetic files,
so recovery of one operation cannot accidentally consume the next injection.
They test deadline clipping and verified cleanup, without claiming to identify
unrecorded historical RPC bodies. Additional faults cover expired deadlines,
independent heartbeat headroom, corrupt/missing chunks, source mutation and
lease loss without publishing unverified files. Existing RPC/stop/runtime
suites retain uncertain mutation, busy-kernel and watchdog coverage.

The operator procedure and frozen acceptance envelopes are in
[F4_D1024_TRANSPORT_REQUALIFICATION.md](F4_D1024_TRANSPORT_REQUALIFICATION.md).
It must pass reviewed source-bound final-only acceptance, including a synthetic
full D1024 inventory, by Sun 2026-10-25 06:00 PT. Failure or a missed milestone
means HOLD and return to the judge. Offline tests do not qualify the transport.
Only after reviewed acceptance may `F4_D1024_ATTEMPT=03` prepare the one new
full-inventory canary. After its actual timing PASS and verified stop, seal the
fresh main manifest against that canary's receipt. No retry is authorized here.

Merging code does not rewrite frozen checkouts, active processes or the mutable
qualification registry. The watcher will report main's new tree as unqualified
until acceptance. Publishing a new global registry later can refuse future
dispatch from old frozen broker trees; coordinate that operator action with
remaining pinned lanes. Do not edit historical receipts or copy source hashes
into a PASS record.

## Publication and verification disposition

Implementation branch: `sol-colab-transport-repair-20261024`. The complete
existing CLI/ops selection was run serially at nice 15 with numerical threads
limited to one: 765 passed, 7 failed, 3 collection errors. Two attributable
test issues (repeat fixture creation and a whole-operation timing assertion)
were corrected and independently rerun; final logs and the exact frozen source
identity are in the external review packet, not a fabricated acceptance record.
The gate also exposes missing baseline transport modules, calendar lint, T34
device fixtures, and a dispatch-warmup timeout. Keep unresolved failures visible.
The owner's all-tests-pass merge condition is not satisfied, so publish the
branch only; do not merge it into main or rotate the qualification registry.

The create-only external review packet is
`snake-dqn-artifacts/big-lever/exp/f4/transport-repair-20261024-01/`. It includes
the source-bound offline acceptance plan, synthetic inventory, review evidence,
test logs and hashes. Its empty-account map is a prospective **REVIEW_ONLY**
precondition; no current account state was observed. The operator must verify
the actual account and independently review the exact plan before executing.
No live Colab operation, real canary/main preparation, collection control or
scientific payload access was performed during this implementation.
