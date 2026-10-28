# Erratum to `TRAIN01_16.md` (Astra Review Findings — 2026-10-28)

Applies to: `research/sandbox_a_resident/TRAIN01_16.md`
(`sha256 = 9212a6a8fd6af55eaaaa76602e2eeefb4e613e1f70d84f233c0a1ea2ec24fe82`, bound by
`research/sandbox_a_resident/packet-train01-16/SEAL.json["implementation_note_sha256"]`).
Per `AGENTS.md`, the hash-bound bytes of `TRAIN01_16.md`, `VERIFICATION_TRAIN01_16.json`, and
`packet-train01-16/*` are preserved untouched; this sibling erratum records the exact corrected
checklist figures, statuses, and continuation recovery contract.

## 1. Corrected Upload Staging Byte Totals & Timings (`TRAIN01_16.md:69, 96-102`, `packet-train01-16/DISPATCH.txt:6`)

- **Pinned private inputs (`5` files = `21,601,751` bytes = `21.60 MB` / `20.60 MiB`, not `11,222,360` B / `~11.22 MB`):**
  - `b8-s3.pth`: `6,562,005` B
  - `frp3-s12.pth`: `6,906,247` B
  - `champion.pth`: `6,898,955` B
  - `registry.json`: `1,234,356` B
  - `verified-ledger.json`: `188` B (empty first-attempt ledger in `packet-train01-16`)
- **Supervisor pinned files:** `bootstrap.sh` (`3,244` B), `source.py` (`14,511` B), and measured
  fallback archive `source.tar.gz` (`936,230` B).
- **GitHub source-delivery path (`7` uploaded files, `21,619,506` B = `21.62 MB`):**
  - `T_stage,nom = 33.32 s` (`11.11%` of `300 s` cap, `< 120 s` design target) — replaces stale `26.79 s`.
  - `T_stage,worst = 57.65 s` (`19.22%` of `300 s` cap, `< 210 s` design target) — replaces stale `45.67 s`.
- **Fallback `background-sparse-v1` archive path (`8` uploaded files, `22,555,736` B = `22.56 MB`):**
  - `T_stage,nom = 36.08 s` (`12.03%` of `300 s` cap, `< 120 s` design target, `< 150 s` canary bar).
  - `T_stage,worst = 62.47 s` (`20.82%` of `300 s` cap, `< 210 s` design target).
- **Checklist item 1 status:** remains **PASS**.

## 2. Corrected `StallWatchdog` Exit Code (`TRAIN01_16.md:121-125`)

- `StallWatchdog` arms `faulthandler.dump_traceback_later(timeout_seconds, file=..., exit=True)`,
  whose C-level timeout handler terminates the Python child via `_exit(1)` (**exit code `1`**, not
  `124`), after which `train_job.sh`'s `EXIT` trap seals `stall-watchdog.json` and
  `faulthandler-stacks.log` and propagates exit status `1` (verified in
  `test_stall_watchdog.py::test_job_shell_seals_real_faulthandler_timeout_for_runtime_export`).

## 3. Corrected Checklist Item 10 Pre-Dispatch Status (`TRAIN01_16.md:139-144`)

- Offline preparation (`dispatch.sh --dry-run`) explicitly outputs `live_accounting=NOT_CHECKED`.
  Accordingly, at packet seal time:
  - Offline Mac engineering discipline (`nice -n 15`, CPU-only, thread caps `= 1`, zero stray
    background children): **PASS**.
  - Live Colab accounting (`account-status` unit balance and active session cap `< 3`) and live
    host admission (`AC power`, lid/sleep guard under `caffeinate`): **`PENDING_LIVE_DISPATCH`** in
    offline preparation artifacts (checked only by the live `dispatch.sh` invocation).

## 4. Corrected Snapshot Cadence Across Proxy Boundaries & Continuation Tooling (`TRAIN01_16.md:90-92, 108-112`)

- **Actual inter-snapshot interval:**
  - Non-proxy 16-update boundary (`~139.5 s/update` nominal): `2,232.0 s` (`37.20 min`).
  - Across a scheduled `E96` `H5000` detached proxy boundary (`16, 64, 128, 192, 256, 320, 384`),
    `detached_proxy` (`2,049.4 s = 34.16 min` nominal) runs before `publish_snapshot`
    (`train.py`), so the GPU interval between consecutive published snapshots across a proxy
    boundary is **`4,281.4 s` (`71.36 min`)** nominal, plus incremental Mac download lag
    (`~5.0–13.4 min` for a `~1.33 GB` snapshot).
- **Durable recovery boundary & continuation path (resolved in successor commits):**
  - Only completed, Mac-verified periodic snapshot triples (`iteration-NNNN.{pt,json,-actor.pth}`
    with `.sha256` and `.complete`) are durable across VM loss.
  - Packet 15 (`5c0a101a`) and Packet 16 (`5cad044f`) run under their pinned source commits, which
    do not install a `SIGUSR1` handler (`SIG_DFL` on timeout); any mid-run wall-cap stop or provider
    recycle on those active runs recovers from their latest completed, Mac-verified `16`-update
    periodic export using the Mac-side continuation builder (`training_continuation.py`), which:
    1. Encodes any verified `iteration-NNNN.pt` from `train01-15`/`train01-16` (or their
       continuations) into a create-only XZ derivative + per-tensor SHA-256 manifest and registers
       `(source_packet_id, iteration, original_sha256)` in `compact_allowlist.json` (`v2`).
    2. Accepts verified mechanical timeout/recycle receipts (`RPC_EXEC_POLL_TIMEOUT`, host wall/unit
       cap `SUPERVISOR`, and VM-timeout `CHILD_NONZERO_EXIT` with mechanical `terminal.reason`,
       `status == "INCOMPLETE"`, `stop_verified == true`, `cleanup_error == null`).
    3. Seals a successor packet on a new runtime pin that also installs the `15 s` `SIGUSR1`
       graceful checkpoint/last-known-good marker handler in `train.py` and `train_job.sh`.
