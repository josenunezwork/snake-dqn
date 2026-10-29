# TRAIN-01 S2-03 c1 eligibility — 2026-10-29

**Outcome: C1 NOT ADMISSIBLE from the retained exports under the frozen contract.** No successor packet was sealed or dispatched. No completion attempt was consumed by these offline checks. The original remains INCOMPLETE, not a scientific failure or result.

## Binding contract and retained evidence

`TRAIN01.md:184-196` requires complete full-state trajectory units and explicitly says: “Calibration-only or failed-preparation state is not a valid full-PPO resume.” `TRAIN01_S2_01.md:53-59` requires the latest eligible verified snapshot, preserving N, RNG, counters, preparation and seed/proxy banks. `TRAIN01_S2_03.md:32-42` retains that rule with at most two manual completion attempts and a 23-hour attempt envelope.

The executed manifest pins runtime `459bfb58cf1f4966f59ce06fa33fc94539f80d9b`. The A100 receipt records `RPC_EXEC_POLL_TIMEOUT`, `INCOMPLETE`, `stop_verified=true`, `cleanup_error=null`, 5,140.112 s elapsed, and 81 retained polls. No `iteration-*.json` exists. All three retained exports were matched to the native receipt by streamed SHA-256 and size, with checksum and completion sidecars present:

| Export | Bytes | SHA-256 |
|---|---:|---|
| `distribution-screen.json` | 6,849 | `79b10517c9ca3fdc9fd92622a47243e6134da131ac679e849b0275e0dc74c312` |
| `calibrated-boundary.json` | 618 | `bba20cea53a9fbc675de00ebb1250514ca818d60b42540c7aec925b2db42fcde` |
| `calibrated-boundary.pt` | 1,327,612,072 | `6d594f7bc6393730df2887009b8065a3dbdd6c6f3bf4adacb5f8d6fd3f51c7c0` |

The boundary header has schema `sandbox-a-train-calibrated-boundary/v1`, `phase=calibrated`, and `prepare_phase=calibrated`. It is not a `sandbox-a-train-snapshot/v1` ledger unit. The producer publishes it before sampling the first critic life (`training_prep.py:735-740`). Last reported critic-fit work (1,374,208 fit rows) is later than this export and is not recoverable from it. A separately authorized preparation-resume implementation could reuse the calibration temperature/state and then redo critic preparation from that boundary; it could not skip the unfinished critic preparation or recover its lost fit rows. Existing preparation code has the temperature reuse branch, but no admitted packet/ledger route for this boundary.

The current contract permits c1 then c2 only from the latest eligible, complete, Mac-verified trajectory snapshot: a successful ready/skipped pre-actor iteration-0000 snapshot or a later PPO snapshot, with the frozen endpoint, namespace, allocation and banks. It does not authorize fabricating an iteration header/actor triple, promoting a calibration export, an empty-ledger restart of this executed trajectory, or using c2 receipt-less host-loss recovery for this receipt-bearing c1. No suitable retained snapshot exists here. Extending preparation resume would need an explicit prospective recovery addendum through the delegated operator/judge process, tests and independent review before sealing; this report does not claim that authority.

## Accepted tooling reused without changing it

Replayed the actual clean `sol-sandbox-a-20261026` worktree at `da85a426f94b6a4f2d6bfb8cb8507e26df417501`. This preserves its exact eager-fallback ruling, TF32 warning and mid-line tail handling. No alternate scanner was written, no retained evidence was renamed/deleted, and no broad branch merge was made. Review context: external `review-c2-readiness5-opus.md`, readiness-6 `SUMMARY.md`, and `C2_READINESS3_LIVENESS.md` with the exact operator ruling. The user's round-6 acceptance is authority supplied in the task; a separately named round-6 Opus acceptance file was not located in the supplied review paths.

Three distinct gates were checked:

1. **Mechanical trigger:** accepted `_check_mechanical_receipt_class` recognizes this native `RPC_EXEC_POLL_TIMEOUT` with verified release as an eligible class. This is an operational stop, not evidence of a scientific fault.
2. **Full admission has two additional tooling gaps:** unchanged accepted scanner refuses the full root as `oversized retained diagnostic: .../exports/training/calibrated-boundary.pt` (1,327,612,072 bytes). Its binary name classifier covers iteration payloads, not calibrated-boundary.pt. Independently, receipt/log verification without the output root reaches and rejects the native `TimeoutExpired([...], ...)` error as an explicit fault string. No broad error exemption, deleted payload, invented terminal, or sanitized receipt was used to obtain admission. These are fail-closed liveness gaps for this retained producer shape; they are not scientific failures. Both exact messages are in the evidence JSON.
3. **Trajectory eligibility independently refuses:** accepted `build_ledger` says `no complete full-trajectory snapshot triples`. Fixing the two scanner gaps alone cannot make this c1 legal.

The actual `completion_packet` call was repeated three times for each prospective GPU target and consistently REFUSED before producing a packet. These six repetitions are retained-evidence refusal checks, **not GPU admission preflight**. The accepted readiness 3/4/6 test modules passed **299 tests in 17.90 s** under nice15, CPU and single-thread caps. The accepted tree and all source evidence remained unchanged. No project runtime, test implementation or qualified transport changes were needed to establish the refusal.

## Timeout census and exposure

The census covers actual `receipt.json` files under `snake-dqn-artifacts/sandbox-a-resident`, requires an A100 endpoint, and deduplicates session/generation/endpoint. It finds **4/13 A100 assignment receipts (30.8%)** with `RPC_EXEC_POLL_TIMEOUT`; within TRAIN-01, **4/9 (44.4%)**. This is a retained-receipt census, excluding absent receipts, active runs without receipts, preallocation failures and other artifact roots. It is not a provider-wide or whole-campaign incidence estimate.

| TRAIN-01 attempt | Native error | Release |
|---|---|---|
| 08 | kernel remains busy/unknown after telemetry hiccup | verified |
| 09 | kernel remains busy/unknown after telemetry hiccup | verified |
| 16 | TimeoutExpired, about 25.014 s | verified |
| S2-03 A100 fallback | TimeoutExpired, about 16.617 s | verified |

Exact paths, receipt hashes, identities and all denominator rows are retained in `evidence/timeout-census.json`. The shorter timeout on S2-03 is the remaining portion of the qualified 300 s poll/recovery operation budget: its last failed `renewal` lasted 300.006 s. `supervisor.py:756-828` clips each exec to `min(RPC_SECONDS=120, remaining)` after earlier calls, idle probes and backoff consume that shared budget. The 23-hour job cap was not exhausted. A longer manifest timeout does not enlarge this operation budget. No `research/colab_cli` bytes were edited.

Outside that transport, bounded compact job exports and resumable units can reduce transfer/serialization load and lost work. Compact iteration snapshots already avoid shipping raw full-state tensors on later snapshots; the calibrated boundary still cost 757.329 s to download and had no usable trajectory unit. A prospectively authorized compact preparation boundary and tested preparation ledger route would target that exposure, while retaining the frozen work/decision rules. Small regular completed-work progress keeps liveness visible; chatter alone cannot prove work or prevent provider stalls. Bounded tails, unchanged-export skipping and small ZIP batches are already present in the qualified transport (TELEMETRY_STALL_20261022.md), not new available manifest switches. No outside-only change guarantees prevention of provider/kernel RPC stalls, and this incident does not prove their provider-side cause.

## Dispatch and checklist disposition

Requested identity `sandbox-a-train01-20261028-s2-03-c1` remains **UNSEALED**. H100-first/A100 fallback and base/effective 690/1,380-minute attempt envelopes remain the required settings for any subsequently authorized packet, with whole-timeout ceilings 276 H100 / 184 A100 units and zero reserves. They were not silently replaced by a restart or new recipe.

`dispatch-command.sh` is a fail-closed guard that explains the absence of an admissible packet and exits 78. It is not a dispatch command. **Real dispatch.sh --dry-run and GPU admission preflight 3x per variant were NOT RUN**, because there is no legal successor manifest. Existing initial-packet dry-runs were not relabeled as c1 proof. No live accounting/capacity check, allocation, canary or run was attempted. The colab-limits checklist stops at resume eligibility (items 2/3/4); later staging, compute, export, account and host dispatch checks are not qualified for an unbuilt packet.

Preserved: source receipt, output, log, calibrated payload/sidecars, prior packets, packet 15/16 and their continuations, F5, all running jobs, account records, serving pins and released checkpoints. Publication is branch-only; main is untouched. The branch change is this status record and its small evidence JSON, not a new training implementation.

External reproducibility: `verify-retained.py`, `census.py`, `evidence/retained-admission.json`, `evidence/timeout-census.json`, `evidence/accepted-tests.log`, and `COMMANDS.md` in the S2d handoff directory. Scripts use create-only evidence files; retain old output and select fresh evidence filenames before replay.

Independent Argon review via the Antigravity CLI returned **PASS** for this documentation/eligibility assessment. It reviewed the supplied contract, producer, scanner, timeout code and evidence; it performed no tools or file edits and is not governance ratification or dispatch qualification. Result: external `evidence/argon-review.json`, SHA-256 `c27623a9fb7dcdad77c15dfb7d65622068ea1a95fc912933d48ed6aad50df961`.
