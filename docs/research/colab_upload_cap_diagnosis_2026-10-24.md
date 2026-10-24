# Colab UPLOAD_CAP diagnosis — 2026-10-24

**Verdict:** these four failures exhausted the shared staging clock through slow kernel/control operations and recovery, with occasional slow Contents calls. They do not show tiny terminal sidecars taking the upload budget to transmit. **None of the four terminal filenames has a Contents upload record.** The error identifies the next PUT refused after earlier work consumed the clock. Many small files amplify the problem by requiring repeated parent-directory and hash-verification RPCs. The precise network/provider/CLI/host cause of the long RPCs is not recoverable from these records.

**Ranked action:** first reduce package file count using a verified metadata archive, retaining ledger/sidecar semantics; next forecast per-call control costs from existing receipts, then instrument and batch redundant mkdir/verification RPCs if needed. Any executable `research/colab_cli` change requires requalification. Raising the cap alone is a weak mitigation. This diagnosis changed documentation only: no dispatch, transport edits, qualification update, live session query, or runtime action.

## Scope and evidence identity

All numerical claims below come from local immutable receipts, manifests, warmup measurements, passing upload measurements, and polls. Source interpretation is anchored to worktree `/Users/josenunez/Projects/ml/snake-dqn-acctfix`, commit `2e5d5af446dbfa29a6666240b1801cac9ae2d5d3`, whose `research/colab_cli` tree is `d0d92bdb572b770284a1fea6152417ffa5f06bea`. The documentation copy was prepared in a separate worktree based on main `1284b4e5d1311e69400c13fffba3af8b608171e9`; main has other transport bytes, which this diagnosis neither changes nor qualifies.

Current qualification registry inspected: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/colab-cli-qualified.json`, SHA-256 `4983cab57b26d9df9879ada094a84471eedbbb6c85599b1cdea26d86013f70f0`. It binds that tree to qualified commit `a36a821de98f7ef48b730ab454fa2870b0962a5e` and `acceptance-1792740334`. Its acceptance result at `ops/kernel-uuid-requalification-20261023/requalify-2oo2yrww/acceptance-1792740334/acceptance-result.json` has SHA-256 `2b39cae62fe651fb98cc8b1148f3d6babdc9110bf9902c73ec93440846c83764`, matching the registry, and records PASS, qualified=true, return_code=0. These are offline artifact checks, not a new live qualification.

Paths in this table are relative to `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/`; each identifies `receipt.json` and its sibling manifest/measurements.

| Case | Output directory | Receipt SHA-256 |
| --- | --- | --- |
| P1-04 repair | bl-p1/y3/source-repair-10-20261023-04/repair-output | `b675b84e98d155251f43d1b44d00640240dbf2769cde2f37465edf6dc08b66b1` |
| P1-05 canary | bl-p1/y3/source-repair-10-20261023-05/canary-output | `573d7001872e037b4701fe225859aa6690b453388a2451fa14456f5733161655` |
| gpu-move2 main | gpu-move-a100-20261023-02/packet-fresh-move2-02/main-output | `d791e05756b7465ff1b6954c2343a8e4624ffa32daea03c92d943f2d36927af0` |
| gpu-v8-split canary | gpu-v8-split-20261023-02/canary-output | `7a20b6c2f87f124461adc4c03b2f64a0507d550db69fd3c334f4a132aaefc785` |
| P1-04 canary PASS | bl-p1/y3/source-repair-10-20261023-04/canary-output | `22c55fb5bf193f39b144882f5c3eaa998c4c95c3477f9df032fd43708aa31990` |
| gpu-move2 canary PASS | gpu-move-a100-20261023-02/packet/canary-output | `2c67b3a2e01eb068fb3f811b525681340a8be0a1b645d9389d75ac9b1ad24188` |

Ops corroboration: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/ops/logs/` contains `p1-repair10-04-repair.log`, `p1-repair10-05-canary.log`, `gpu-move-a100-20261023-02-main-97a61d97-fresh-move2-02.log`, and `gpu-v8-split-canary-18e3c2e2021d-01.log`. Their terminal errors agree with the receipts. P1 -05 running at nice 0 on an idle Mac is owner-supplied context; the receipts do not measure CPU scheduling, so it is not independent host-load evidence.

## What each timing means

- `contents_transfers[]`: elapsed time of a CLI Contents upload/download wrapper, including CLI startup/network work. It is not raw socket or individual HTTP-request latency. Staged input PUTs below exclude `lease-heartbeat.json` and the release credential. Every observed staged file fits in one 2 MiB chunk; none exercises chunk assembly.
- `transfers[]`: a whole `upload_file` invocation including local hashes, renewals within that invocation, PUT, remote digest verification, and recovery. The parent mkdir and preceding supervisor renewal are outside it. A failed entry can exist without a PUT.
- `cli_calls[]`: command attempt duration and status, but no operation label or start/end timestamp. The heartbeat thread runs concurrently. Summing these durations is **not** elapsed wall time; `attempt=1` also does not rule out outer retries because each outer call starts a fresh CLI attempt counter.
- `upload-measurement.json` and `receipt.canary_evidence`: on successful staging only, record total staging, summed transfer time, and source-delivery time. Failures have no completed staging measurement. `warmup-measurement.json` precedes the staging clock. Receipt elapsed covers allocation/calibration/startup/cleanup as well as staging.
- All four failed output directories have zero `poll-*.json`; they failed before normal job polling. The two passing directories have 15 and 24 polls. Polls prove the later run phase existed but add no missing operation timestamps for the failed upload phase.

## Calls, bytes, and latency

"Planned stages" = two fixed files (`bootstrap.sh`, `source.py`) + local input rows. URL inputs are fetched separately. PUT latency statistics are for observed staged Contents calls only. They are small descriptive samples, not stable tail estimates.

| Case | Local / URL inputs | Planned stages | Staged PUTs observed | PUT bytes | PUT sum s | PUT median / max s | Whole transfer sum s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| P1-04 repair | 26 / 9 | 28 | 1 | 3,244 | 0.562 | 0.562 / 0.562 | 139.298 |
| P1-05 canary | 26 / 9 | 28 | 6 | 20,279 | 4.518 | 0.739 / 1.033 | 22.713 |
| gpu-move2 main | 10 / 0 | 12 | 7 | 18,673 | 47.528 | 1.159 / 41.063 | 210.515 |
| gpu-v8-split canary | 11 / 0 | 13 | 7 | 18,637 | 36.979 | 0.986 / 31.441 | 200.517 |
| P1-04 canary PASS | 26 / 9 | 28 | 28 | 74,138 | 31.678 | 1.231 / 2.541 | 113.760 |
| gpu-move2 canary PASS | 10 / 0 | 12 | 12 | 268,884 | 7.438 | 0.571 / 0.866 | 20.049 |

The whole-transfer sum includes the final failed logical file, including its declared size only in the raw summary; the PUT-byte column counts only bytes with actual Contents records. All observed staged PUTs completed. The four terminal files were not PUT failures and did not retry.

| Case | All upload calls¹ | Heartbeat calls¹ | Exec calls¹ / failed | Warmup s | Receipt elapsed s | Staging s / transfer s² |
| --- | --- | --- | --- | --- | --- | --- |
| P1-04 repair | 14 | 12 | 10 / 2 | 26.629 | 384.645 | — / — |
| P1-05 canary | 18 | 11 | 21 / 3 | 4.336 | 438.238 | — / — |
| gpu-move2 main | 25 | 17 | 26 / 3 | 4.360 | 541.757 | — / — |
| gpu-v8-split canary | 21 | 13 | 27 / 4 | 41.074 | 541.509 | — / — |
| P1-04 canary PASS | 46 | 17 | 85 / 0 | 4.045 | 523.133 | 208.653 / 113.763 |
| gpu-move2 canary PASS | 28 | 15 | 56 / 0 | 3.016 | 462.193 | 38.693 / 20.050 |

¹ Whole-receipt counts include startup, run/export when present, and cleanup. All-upload count = staged PUTs + heartbeat PUTs + one credential PUT. Each case has one failed early heartbeat PUT; this also occurs in both passing jobs and is not evidence of input transfer retry. ² Absent on failed staging; do not replace it with receipt elapsed or a sum of concurrent calls.

P1 -04 canary transferred **28 files / 74,138 bytes** in **113.760 s** of whole-file invocations (`transfer_seconds` wrapper measurement 113.763 s), while staging took **208.653 s**. The gap is 94.890 s: 11.197 s source delivery and 83.693 s of other staging work such as parent mkdir/renewals and local overhead. GPU move2 canary transferred more bytes (**268,884**) in only **20.049 s**, staging **38.693 s**: 6.892 s source delivery plus 11.751 s other staging work. This contrast supports call-count/control-latency sensitivity rather than a payload-throughput explanation. These are operational comparisons, not randomized experiments.

In the passing P1 canary, 22 sidecars (78.6% of staged files) carry only 1,176 bytes (1.6% of staged bytes), yet consume 26.000 s of Contents time and 91.679 s of whole verified-transfer time (80.6% of the 113.760 s sum). This directly motivates reducing file count while retaining their integrity semantics.

## Per-file evidence at failure

Times below are whole-file invocation / actual Contents upload. Duplicate ledger basenames are disambiguated by relative destination. "No PUT" means no matching `contents_transfers.remote` anywhere in that receipt.

| Case | Remote relative to generation root | Bytes | Status | Whole s | PUT s |
| --- | --- | --- | --- | --- | --- |
| P1-04 repair | bootstrap.sh | 3244 | verified | 139.297 | 0.562 |
| P1-04 repair | source.py | 14511 | failed | 0.000 | **No PUT** |
| P1-05 canary | bootstrap.sh | 3244 | verified | 2.750 | 0.504 |
| P1-05 canary | source.py | 14511 | verified | 3.120 | 1.033 |
| P1-05 canary | inputs/public-inputs.json | 2409 | verified | 2.785 | 0.509 |
| P1-05 canary | inputs/failed-supervisor-receipt.json.complete | 9 | verified | 5.234 | 0.511 |
| P1-05 canary | inputs/failed-supervisor-receipt.json.sha256 | 97 | verified | 3.270 | 0.993 |
| P1-05 canary | inputs/immutable-fixture_late_body_bank-64031cc19f35a73a.pt.complete | 9 | verified | 3.270 | 0.967 |
| P1-05 canary | inputs/immutable-fixture_late_body_bank-64031cc19f35a73a.pt.sha256 | 119 | failed | 2.284 | **No PUT** |
| gpu-move2 main | bootstrap.sh | 3244 | verified | 17.174 | 1.165 |
| gpu-move2 main | source.py | 14511 | verified | 5.106 | 1.152 |
| gpu-move2 main | inputs/inputs/verified-ledger.json | 441 | verified | 9.350 | 1.159 |
| gpu-move2 main | inputs/inputs/verified-ledger.json.sha256 | 87 | verified | 3.587 | 1.151 |
| gpu-move2 main | inputs/inputs/verified-ledger.json.complete | 9 | verified | 3.198 | 1.167 |
| gpu-move2 main | inputs/canary-inputs/verified-ledger.json | 294 | verified | 3.147 | 0.669 |
| gpu-move2 main | inputs/canary-inputs/verified-ledger.json.sha256 | 87 | verified | 168.954 | 41.063 |
| gpu-move2 main | inputs/canary-inputs/verified-ledger.json.complete | 9 | failed | 0.000 | **No PUT** |
| gpu-v8-split canary | bootstrap.sh | 3244 | verified | 36.178 | 31.441 |
| gpu-v8-split canary | source.py | 14511 | verified | 8.812 | 0.561 |
| gpu-v8-split canary | inputs/inputs/verified-ledger.json | 423 | verified | 2.793 | 1.004 |
| gpu-v8-split canary | inputs/inputs/verified-ledger.json.sha256 | 87 | verified | 2.349 | 0.986 |
| gpu-v8-split canary | inputs/inputs/verified-ledger.json.complete | 9 | verified | 3.284 | 0.984 |
| gpu-v8-split canary | inputs/canary-inputs/verified-ledger.json | 276 | verified | 4.588 | 0.986 |
| gpu-v8-split canary | inputs/canary-inputs/verified-ledger.json.sha256 | 87 | verified | 140.674 | 1.016 |
| gpu-v8-split canary | inputs/canary-inputs/verified-ledger.json.complete | 9 | failed | 1.840 | **No PUT** |

P1 -04 repair therefore corrects the initial description: it failed at `source.py`, not a sidecar. `bootstrap.sh` needed 139.297 s despite its actual 3,244-byte PUT taking 0.562 s: 138.735 s was outside that PUT. GPU move2 spent 168.954 s on an 87-byte ledger SHA file, of which 41.063 s was the PUT and 127.891 s was verification/renewal/recovery overhead. V8 spent 140.674 s on the analogous 87-byte file despite a 1.016 s PUT: 139.658 s was outside that PUT.

## Exec stalls, readiness, and retries

The following are zero-based indices into each receipt’s `cli_calls`. Durations near 120 seconds are failed exec attempts; they are not slow file PUTs. Whole-receipt scope is explicit because the saved records cannot precisely label every exec or distinguish startup from staging by timestamp. The cleanup path issues no exec commands.

| Case | Failed exec attempts across receipt | Longest heartbeat PUT s (overlapping) |
| --- | --- | --- |
| P1-04 repair | [23] 120.010s, [30] 120.006s | 2.450 |
| P1-05 canary | [27] 120.006s, [31] 120.006s, [45] 120.006s | 42.957 |
| gpu-move2 main | [19] 120.010s, [52] 120.014s, [58] 120.015s | 8.858 |
| gpu-v8-split canary | [4] 30.006s, [42] 120.006s, [49] 120.013s, [55] 120.013s | 40.165 |

V8’s 30.006 s failed exec is the first warmup probe: its warmup measurement records two attempts and eventual PASS in 41.074 s. Other warmups passed on their first attempt (26.629, 4.336, 4.360 s). Warmup is outside staging, so moving warmup earlier alone does not remove these failures. The other failed execs are after warmup; the inspected cleanup path issues stop/sessions/usage rather than exec, so none of those failures can be assigned to cleanup. Exact staging versus earlier startup attribution remains unavailable. The completed whole-file transfers independently show that verification/recovery stalls occur before terminal refusal. The passing jobs have **zero failed exec attempts** in their entire receipts.

There are **no duplicate staged remote destinations, failed staged Contents records, or input PUT retries** in any of these six receipts. A retry in `_contents_call` would create another Contents record for the same destination even though its `cli_calls.attempt` would again be 1. Recovery is instead visible as failed exec followed by later completed execs; exact kernel-idle helper durations are not recorded in `cli_calls`. P1 -05 has successful heartbeat PUTs lasting 42.957, 36.061, and 37.092 s while its input PUTs remain at most 1.033 s. V8’s bootstrap PUT itself is slow (31.441 s), and move2’s SHA PUT is slow (41.063 s), so Contents has a long tail too. Neither a universal fast-Contents claim nor a universal slow-upload claim fits the evidence.

The exact division among websocket/exec acknowledgement waits, authenticated idle-read probes, CLI startup/auth, provider latency, and Mac/network scheduling is **not identifiable**. The records lack timestamps, exec operation tags, inner HTTP timings, and kernel-side timings. Timeout does not prove a kernel never executed the command. Host niceness alone is not a supported diagnosis or fix.

## Why UPLOAD_CAP appears on a tiny next file

The following references are to the frozen `2e5d5af4` source, not current main:

| Source reference | Mechanism and implication |
| --- | --- |
| `research/colab_cli/transfers.py:25-35,133-176` | 2 MiB upload chunks, at most three idempotent Contents attempts; shared staging allowance is 300 s. `_contents_call` calls `renew()` before checking the remaining PUT deadline. A renewal can exhaust the residual clock before any PUT dispatch. |
| `research/colab_cli/supervisor.py:1488-1524` | Shared 300 s deadline starts before the local-file loop. Each file gets `renew()` then a parent mkdir `poll_rpc` before `stage_upload`. Thus time outside the per-file transfer ledger still spends the shared staging budget. |
| `research/colab_cli/transfers.py:321-350` | Small file = one Contents PUT then one VERIFY_FILE RPC. PUT uses the shared residual deadline, but remote verification uses a fresh per-file deadline of at least 300 s. Clock checks are asymmetric. |
| `research/colab_cli/supervisor.py:667-717` | poll_rpc uses up to 120 s per exec attempt within a separate operation budget, with independent kernel-idle proof and backoff before safe retry. These waits are not clipped to the remaining staging clock. |
| `research/colab_cli/supervisor.py:319-363` | Independent lease thread writes an approximately 85-byte heartbeat every 30 s and overlaps the main thread. Its elapsed times cannot be added to exec times as wall time. |
| `research/colab_cli/canary.py:23,290-322` | 150 s is a post-success transfer admission bar, distinct from the 300 s total-staging bar. It is not the timeout that generated these UPLOAD_CAP exceptions. |
| `research/colab_cli/supervisor.py:1528-1578` | Source delivery is separately timed, after local uploads. A completed staging measurement exists only after delivery. None of these failed runs reached that measurement. |

Baseline staging call model for F small files: **F PUTs + F parent mkdir execs + F verification execs**, plus lease renewals, credentials/startup, source delivery and any recovery. P1’s 28 planned stages expose at least 56 staging execs; move2’s 12 expose 24; V8’s 13 expose 26. Reusing a parent directory does not avoid the repeated mkdir RPC in this source. A large file additionally needs a CREATE_DIR exec, ceil(bytes / 2 MiB) PUTs, ASSEMBLE, VERIFY and up to two cleanup execs (plus the supervisor mkdir); no such chunk path was exercised by these staged inputs.

A 120 s failed exec and its recovery can consume most of the headroom left by a passing 208.653 s P1 staging sequence. A later parent mkdir or renewal can continue past the shared deadline; then `_bounded_timeout` refuses the next PUT. The error names where expiry was noticed, not where time was spent. The 300 s cap is therefore not a hard bound on all work during staging, and a 150 s transfer-only success is insufficient to predict resilience to control-path stalls.

## Ranked recommendations and qualification boundary

| Rank | Smallest change | Expected effect / limit | Required validation |
| --- | --- | --- | --- |
| 1 | Package-only: consolidate small local metadata and sidecars into one immutable archive, retaining the actual ledgers and marker contents. Use P1’s existing bundle/extraction path where possible. | For P1, 26 local input rows become roughly 1–2 metadata rows plus the two fixed files; removes about 24–25 PUTs and 48–50 baseline mkdir/verify RPCs. GPU packages can similarly replace 10/11 local rows with one archive if their launcher safely unpacks first. Does not cure a stall on bootstrap/source or an archive verify. | New bundle hash, fresh manifests/output paths and family canary covering actual bytes, count, destinations, extraction and ledger admission. Package-level tests; no transport requalification if executable colab_cli bytes are unchanged. |
| 2 | Measure and budget per-call control latency; add phase/operation IDs and monotonic start/end times, separating parent mkdir, PUT, VERIFY, renew, idle proof and source fetch. Keep heartbeat timings separate. | First use a report-side call-count plus bytes model from existing receipts without changing transport. For exact critical-path accounting, add operation-level timings in a separate qualified transport change. Existing tail observations include 120 s exec timeouts; healthy-call medians are not a sufficient budget. | Instrumentation inside colab_cli changes its executable inventory and requires reviewed final-only live requalification before production use. A report-side parser needs no transport change. |
| 3 | Transport optimization: create parent directories once, verify a whole bundle with a single hash/inventory RPC, and prospectively reconcile the shared staging deadline with operation-specific recovery budgets. | Removes redundant RPCs and makes cap attribution truthful. A 300 s remaining-staging cap should not silently coexist with a new 300 s per-file verify budget; define whether staging is wall budget or PUT-only budget. Preserve lease, kernel-idle and durable mutation fences. | Offline boundary/fault tests plus new source-bound final-only acceptance. Do not shorten/replay uncertain mutating execs or suppress verification to meet the timer. |
| 4 | Raise staging/transfer allowances only after measured call-count/tail forecast and a frozen priced envelope. | May accommodate a known bounded stall but leaves all RPC exposure. Raising only 300 s can still fail the 150 s family-canary bar; raising only 150 s does not prevent these refusals. No supported numeric replacement from these six runs. | Any changed colab_cli constants/admission bars require requalification and prospective package/cap approval. Preserve first-cap-wins session/unit limits; no retroactive edits to receipts/manifests. |

P1 already has a bundle path in `research/bl_ppo/p1_colab.sh:16-22` and `research/bl_ppo/p1_inputs.py:21-53` (`unpack_bundle`). Its extractor must keep path/link/conflict rejection and `.complete` verification. The ops resume declaration does not globally require the ledger to be a direct upload row: a package may extract the bound ledger before its existing reader runs. For GPU packages this is new package behavior and must be tested, including nested paths; simply dropping ledger or sidecar rows is not a fix. Keep `public-inputs.json` available to the transport as required by the package, and do not duplicate archive members at destinations later populated by public fetches.

A clean prospective test compares the same frozen input inventory and bytes with separate-file versus bundle packaging, using fresh registered canaries and the qualified broker, and reports total staging, whole-file transfer, control-call counts, stalls and stop receipts. It must not silently retry failed jobs or select only the passing canary. No such dispatch was run for this diagnosis.

## Reproduction and verification

Supporting files beside the requested artifact: `analyze_receipts.py`, `timing-summary.json`, and `evidence-sha256.json`. The parser reads only the six fixed output directories above, computes statistics from the saved arrays, and writes a derived summary. It does not call Colab or modify source evidence. `timing-summary.json` retains per-file and per-call detail; receipt hashes anchor it to this snapshot.

Recompute with `python3 /Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-sol/uploaddiag/analyze_receipts.py` (writes `/tmp/uploaddiag-timing.json`), then compare its parsed JSON with the saved summary. All six receipts were rehashed, staged-vs-control call counts were reconciled, and the two report copies were byte-compared. Documentation checks use `git diff --check`; product tests and live qualification are not applicable to this report-only change. No live processes, tmux sessions, released checkpoints, serving pins, transport source, qualification records, or immutable research evidence were changed.
