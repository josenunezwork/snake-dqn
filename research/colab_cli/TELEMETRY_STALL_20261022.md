# Finished-job telemetry/export stall — 2026-10-22

Status: local correction; live requalification pending. No Colab operations were
performed for this investigation. Original manifests, polls, receipts and outputs
were read only. No files under `t34-round0-study-01/` were accessed.

## Evidence

All paths below are relative to
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/`.

| Evidence | DENSITY-2 production-02 | L4 native ruler main-02 |
| --- | --- | --- |
| Log | `ops/logs/density2-production-02.log` | `ops/logs/l4-native-main-02.log` |
| Output directory | `big-lever/exp/bl-density/density2-20261022-gatefix-f1904777/production-output-02` | `big-lever/exp/bl-l2/native-l4-prepared-20261022-02/main-runtime-02` |
| Saved terminal poll | `poll-000015.json` (log line 16) | `poll-000021.json` (log line 22) |
| Remote terminal | succeeded, return code 0, 493 exports | succeeded, return code 0, 562 exports |
| Remote `finished_at` (Unix seconds) | 1792692138.8163373 | 1792682656.656895 |
| Last outer-poll captured elapsed | 9568.4 s | 1189.4 s |
| Final broker elapsed | 9736.68 s | 4276.47 s |
| Verified local payloads / missing | 408 / 85 | 282 / 280 |
| Receipt `exports` | 0 | 0 |
| Receipt logical transfers (includes staging) | 411 | 289 |
| Cumulative exec call time | 6646.2 s | 2519.6 s |
| Cumulative download call time | 2603.0 s | 1295.0 s |
| Verified stop / cleanup error | true / null | true / null |

Both `receipt.json` files report `INCOMPLETE`, `RPC_EXEC_POLL_TIMEOUT`, and
`Refused('deadline reached during telemetry recovery')`. Call-duration sums
include activity overlapping the heartbeat thread and must not be added to infer
wall time. The printed elapsed is captured **before** the poll RPC and exports;
remote progress elapsed also excludes setup. These are different clocks.

Density's terminal log contains progress `492/492`, ETA 0, at producer elapsed
4030.594 s. Its inventories grew from 104 to 408 to 493 entries in the last three
saved polls. The terminal inventory had already been produced and returned to
the Mac; this incident was not waiting for the runtime export scanner to finish.

## Root cause

`CLI.poll_rpc` is the operation called “telemetry recovery”: an exec attempt has
a 120-second bound. An ambiguous response fences further exec, probes kernel
idleness through independent REST, and retries only after idle proof. Up to
three attempts, probes and backoff share 300 seconds, narrowed by wall and unit
deadlines. `recoverable_rpc` additionally persists one UUID operation record to
prevent repeating a committed mutation. Acknowledged kernel errors do not retry.

The approximately 90-minute stall was **not one unbounded recovery call**.
`Broker.execute` walked every entry in each cumulative inventory, called
`export_file` even for files already present, and then unconditionally called
`renew(force=True)`. Each forced renewal returned the complete state, inventories
and log tails through CLI stdout. Normal transfer renewal also read full state.
Terminal handling happened only after this whole loop. Hundreds of redundant
RPCs and repeated bounded timeout/recovery operations consumed the session cap.
The final exception names the last exhausted RPC, obscuring this amplification.

`runtime.read_state` repeated inventory in state and at the top level, and in
the terminal and its scanner proof after completion. It also returned two tails
of up to 64 KiB. Density's final saved poll is 486,251 bytes. This amplified
serialization and transport; the receipts do not establish why an individual
provider exec timed out. No provider-side causal claim is needed for the fix.

There was also significant independent small-file overhead: density performed
1224 GETs for 408 payload/checksum/marker triples. Removing forced polls alone
would leave tens of minutes of per-file Contents calls.

## Correction

- Full state reads occur once per outer poll. Foreground maintenance uses a
  small `renew_lease` result with both generation fences, throttled to 30 seconds.
  Release-token rotation, retained-job maintenance, independent Contents
  heartbeat, kernel uncertainty fence and all existing first caps remain active.
- Compact polls omit duplicate inventory copies and cap each diagnostic tail at
  4096 bytes. Original terminal scanner evidence is preserved. Stdout prints a
  bounded summary; the full response remains in the create-only poll file.
- A verified-export ledger skips unchanged completed files and refuses metadata
  changes. Small new exports travel in bounded stored-ZIP batches, at most 64
  entries / 4 MiB payload / 8 MiB archive. Original checksum and completion files
  travel with their payload. Both sides validate bounds and immutable metadata;
  the Mac rejects unexpected members, paths, links, sizes and hashes and
  publishes completion markers last. Large exports retain the existing chunked
  path. Remote archive creation uses the durable UUID mutation fence.
- Terminal evidence is persisted before downloading; foreign generations refuse
  before transfer. Incomplete receipts retain already verified export records.
  A saved successful remote terminal alone never grants a broker PASS.

Regression tests reproduce inventories of 493 and 562 files with each full
telemetry poll costing 120 simulated seconds. A 180-second session must finish
with one full read and 8 or 9 batch GETs, respectively. Another case advances
from a 408-file running inventory to 493 terminal entries without retransferring
the first 408. Transfer failure must preserve prior publications and terminal
evidence, verify backend stop, and remain INCOMPLETE. Archive corruption, unsafe
members, generation fences and deadline recovery have separate offline tests.
These establish algorithmic bounds, not live provider throughput.

## Recoverability

The complete inventories and successful terminals were already on the Mac, and
the remote exports were recoverable **before cleanup**; hundreds were actually
downloaded. Validation against saved terminal inventory found 408 density and
282 ruler payloads with matching sizes and SHA-256 values, with no bad payloads.
The remaining 85 and 280 files were absent locally. Density's missing files
include `density/summary.json`. The partial output trees remain usable evidence
under their original incomplete receipts; this correction does not promote them
or fabricate a replacement result.

Both receipts prove assignment disappearance after cleanup. The old VM can no
longer supply its missing files. No attempt was made to allocate a replacement,
relaunch work or recover through Colab. Full recovery now requires a separately
authorized new job or an independently existing backup; none was established
by this investigation.

See [REQUALIFICATION.md](REQUALIFICATION.md) for Claude's operator steps. Merge
and production use of these executable bytes remain blocked on reviewed PASS.

## Local verification

The complete CLI suite passed **376 tests in 964.95 seconds** using the primary
checkout's Python 3.12 interpreter, two pytest workers, CPU device, and numerical
library thread caps of one:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest \
  -q -n 2 --tb=short tests/test_colab_cli_*.py
make PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python lint
```

Repository lint, changed-Python Black/isort/flake8 checks, `git diff --check`,
`bash -n tools/ops/requalify.sh`, supervisor `--help` and requalification CLI
`--help` passed. No training, full gameplay suite or live acceptance was run.

For the counterfactual, the baseline `6e4465de` supervisor was temporarily restored
under a `finally` restoration guard and the 493-file regression run alone. It
failed with `whole-session deadline reached before helper` after observing the
successful remote terminal. Result: **1 failed, 71 deselected in 10.82 seconds**.
The corrected source was restored before the complete passing suite. This
reproduces cumulative poll amplification without waiting for a real long-job cap.
