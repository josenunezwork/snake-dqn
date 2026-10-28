# TRAIN-01-15-C1 (`packet-train01-15-c1-i0064`): Continuation of `train01-attempt-15` from `iteration-0064`

Prepared and sealed locally only; **do not push; do not dispatch** (operator dispatches).
Packet `packet-train01-15-c1-i0064` (`sandbox-a-train01-20261028-15-c1-i0064`, completion attempt
`1/2`) continues `train01-attempt-15` (`sandbox-a-train01-20261027-15`, `packet-train01-15`) from
its latest Mac-verified periodic snapshot `iteration-0064` after the documented Mac host `ENOSPC`
receipt-write failure, incorporating all three fixes required by Astra's second continuation review.

---

## 1. Resolution of All Three Review Majors

1. **Full Retained-History Scientific-Failure Precedence (`training_resume.py`):**
   - `_check_output_root_fault_evidence` now scans **every** saved `poll-*.json` (removing the
     `[-8:]` window; failing closed on unreadable, corrupt, symlinked, foreign-`job`, or
     foreign-`generation` poll files) in addition to `terminal.json`, `summary.json`,
     `faulthandler-stacks.log`, `stall-watchdog.json`, `distribution-screen.json`, `progress.json`,
     `calibrated-boundary.json`, and `iteration-*.json`.
   - `_check_supervisor_log_fault_evidence` scans **every** line of the host supervisor log for
     scientific, preparation, distribution-screen, action-collapse, illegal-action, non-finite,
     Python traceback, or stall-watchdog fault markers.
   - Regression tests in `test_training_resume.py` verify that `FloatingPointError: non-finite loss
     tripwire` in `poll-000001.json` followed by `20` newer healthy polls (or on an early
     `supervisor.log` line followed by `20` healthy poll lines) is **REFUSED**.
   - Across `train01-attempt-15`, all `1,279` retained `poll-*.json` files
     (`polls_sha256 = bfcb61d767ae9901a789346f742864e6417ec79374692cdf10d5de5520740180`) and all
     `1,284` lines of `sandbox-a-train01-20261027-15.log`
     (`d85a92595128ffcfbac2cf768762fd5f262ad74c5d23d425b5a9e1e1c64f71e4`) contain **zero**
     scientific-failure hits.

2. **Explicit, Narrow Host-Side ENOSPC Recovery Admission (`training_resume.py`, `training_continuation.py`, `training_preflight.py`):**
   - Preserves `train01-attempt-15/output/receipt.json` and `output/terminal.json` **absent as-is**
     (never fabricating a receipt or inventing `cleanup_error: null` or a wall-timeout reason) and
     preserves `train01-attempt-15` as `INCOMPLETE` with `original_terminal_status`,
     `original_supervisor_error`, and `original_cleanup_error` explicitly recorded as
     `"UNKNOWN_UNAVAILABLE"`.
   - Captures live read-only authenticated `Broker.status()` (`sessions == {}`,
     `usage.assignments == 0.0`, `usage.rate == 0.0`, `usage.balance == 955.63`,
     `unresolved == []`) at `2026-10-28T13:32:48Z` into create-only
     `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-15-enospc-recovery-20261028/account-release-status.json`
     (`108,007 B`, `a3ed1d081880ffeceb593fb00710320146b1a072653d09113937ff7b6d96a9be`, with
     `.sha256` and `.complete` sidecars).
   - Publishes and validates create-only
     `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-15-enospc-recovery-20261028/host-enospc-recovery-certificate.json`
     (`4,680 B`, `decffdb770168c634ecaf29562f746f657e1f0a9b1ec75487d16ef5a028724f7`, schema
     `sandbox-a-train01-host-enospc-recovery/v1`), binding `job_id`, `source_packet_id`,
     `source_packet_manifest_sha256`, `executed_manifest_sha256`, `source_commit`,
     `control_allocation`, `generation` (`59e68cce673b...`), `endpoint`, `session`, `assignment.json`,
     the full supervisor log SHA-256 + terminal `[Errno 28] No space left on device:
     '.../train01-attempt-15/output/.receipt-ihix0kmn'` + `EXIT=1`, the complete `1,279`-poll
     retained-evidence inventory, the live account release capture, and the stale active broker
     record reconciliation.
   - Commits `recovery-evidence.json` inside `packet-train01-15-c1-i0064/` and binds its schema,
     SHA-256 (`decffdb770168c634ecaf29562f746f657e1f0a9b1ec75487d16ef5a028724f7`), and
     `prior_release_proof` into `SEAL.json` (re-verified by `training_preflight.validate_packet`).

3. **Compact-at-Source Periodic Snapshot Exports & Destination Mac Storage Admission (`training_compact.py`, `train.py`, `training_resume.py`, `training_continuation.py`, `tools/sandbox/registry.py`):**
   - Production runs (`test_only = False`) default `snapshot_format = "compact"` in `train.py`:
     `publish_compact_snapshot` serializes the transient raw `.pt` in VM `/tmp` (outside
     `$COLAB_OUTPUT_DIR` so `collect_exports` never transfers the `~1.343 GB` raw `.pt`), encodes
     `iteration-NNNN.pt.xz` (`LZMA2` preset `6`, `CHECK_SHA256`) and per-tensor
     `iteration-NNNN.pt.xz.manifest`, round-trip verifies stream decode, unlinks `/tmp` raw `.pt`,
     and publishes `.pt.xz`, `.pt.xz.manifest`, `.json` (`compact_snapshot` schema
     `sandbox-a-train-compact-export/v1`), and `-actor.pth` with completion-last `.sha256` and
     `.complete` sidecars.
   - `verify_compact_snapshot_export` verifies compact-at-source exports on Mac using the exact same
     sidecar, manifest, stream-decode, per-tensor SHA-256, and canonical `structure_sha256` checks as
     compact resume inputs, and `tools/sandbox/registry.py` + `training_continuation.py` support
     seamless `c1 -> c2` continuation directly from compact-at-source quadruples.
   - Zero existing files were deleted.

---

## 2. Mac Destination Storage Projection (`/System/Volumes/Data`, `20` Remaining Snapshots)

Measured directly from `train01-attempt-15`'s real occupied-simulator-ring `iteration-0064` exports
and live `shutil.disk_usage` on `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident`
(`st_dev = 16777233`):

| Component | Nominal Bytes | Conservative Bytes | Notes |
|---|---:|---:|---|
| Reference `u64` `.pt.xz` (`55a9b0af...`) | `38,396,880 B` (`38.40 MB`) | `48,379,989 B` (`48.38 MB`) | `1.26x` growth allowance vs raw `1,343,234,243 B` |
| Reference `u64` `.pt.xz.manifest` (`3cf9e585...`) | `93,518 B` | `102,870 B` | `281` tensors + `structure_sha256` |
| Reference `u64` `.json` + `-actor.pth` + sidecars | `6,618,673 B` (`6.62 MB`) | `6,611,997 B` (`6.61 MB`) | Preserved verbatim (`-actor.pth` `6.56 MB`, `.json` `23.4 KB`) |
| **Per-snapshot total (`compact` quadruple)** | **`45,109,071 B` (`45.11 MB`)** | **`55,094,856 B` (`55.09 MB`)** | **`29.92x` smaller** than `1,349,844,724 B` raw |
| **`20` remaining snapshots (`0080..0384`)** | **`902,181,420 B` (`902.18 MB`)** | **`1,101,897,120 B` (`1.026 GiB`)** | **`< 1 GB` nominal** vs `26,996,894,480 B` (`25.143 GiB`) raw |
| Supervisor polls, spend/usage, & logs | `104,857,600 B` (`100.0 MiB`) | `104,857,600 B` (`100.0 MiB`) | Generous allowance (attempt 15 used `~40 MB`) |
| Transient transfer + Mac `/tmp` decode verify headroom | `1,678,831,412 B` (`1.564 GiB`) | `1,678,831,412 B` (`1.564 GiB`) | `256 MiB` transfer temp + `1.05 x 1.343 GB` transient verify |
| Required safety margin (`4.0 GiB`) | `4,294,967,296 B` (`4.000 GiB`) | `4,294,967,296 B` (`4.000 GiB`) | Enforced fail-closed by `compute_storage_projection` |
| **Total Peak Required (incl. `4.0 GiB` margin)** | **`6,980,837,728 B` (`6.501 GiB`)** | **`7,180,553,428 B` (`6.687 GiB`)** | Checked before worktree creation |
| **Measured Destination Free Space** | **`14,365,556,736 B` (`13.379 GiB`)** | **`14,365,556,736 B` (`13.379 GiB`)** | `/System/Volumes/Data` |
| **Free Headroom Above `4.0 GiB` Margin at Peak** | **`+7,384,719,008 B` (`+6.878 GiB`)** | **`+7,185,003,308 B` (`+6.692 GiB`)** | **`12.255 GiB` persistent free after conservative run** |

---

## 3. Sealed Continuation Identity, Commits & Operator Commands

- **Branch (`sol-sandbox-a-20261026`):**
  - Tooling & tests commit: `0c42fa71b6fe5dd140754069af8f656b755dd09d`
  - Runtime pin (`snake-dqn-train01-15-c1-i0064-runtime.noindex`): `20e3088eab0b741409118fe01a4ceba68282fdaf`
  - Broker seal (`snake-dqn-train01-15-c1-i0064-broker.noindex`): `315144c21de5548b52a0f1e44f746de4a5fb5125`
  - Qualified transport tree (`HEAD:research/colab_cli`): `5833e54e9c5396121cbbf99a5cd4e52006e564fd` (untouched)
- **Dispatch Identity:** `sandbox-a-train01-20261028-15-c1-i0064` (`completion_attempt.index = 1` of `2`, resuming from `iteration-0064` to `outer_iterations = 384`: `320` remaining updates, `83,886,080` actor rows, `5` remaining regular proxies `[128, 192, 256, 320, 384]`, `est_minutes = 1380`, `timeout_minutes = 2760`, `unit_cap = 369.0`).
- **Dry-run admission:** `PASS_DRY_RUN sandbox-a-train01-20261028-15-c1-i0064 allocations=0 live_accounting=NOT_CHECKED` (`state/` and `output/` remain absent).

### Step A — Pre-Dispatch Stale Active Broker Record Reconciliation (Operator runs once before live dispatch)

```bash
/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.colab_cli.supervisor recover /Users/josenunez/.config/colab-cli/snake/active-snake-sandbox-a-train01-20261027-15-59e68cce673b.json
```

### Step B — Exact Live Dispatch Command (Operator runs under `caffeinate -dimsu` on AC power)

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn-train01-15-c1-i0064-broker.noindex/tools/ops/dispatch.sh sandbox-a-train01-20261028-15-c1-i0064 /Users/josenunez/Projects/ml/snake-dqn-train01-15-c1-i0064-broker.noindex/research/sandbox_a_resident/packet-train01-15-c1-i0064/dispatch-manifest.json --broker /Users/josenunez/Projects/ml/snake-dqn-train01-15-c1-i0064-broker.noindex --cwd /Users/josenunez/Projects/ml/snake-dqn-train01-15-c1-i0064-runtime.noindex --state-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-15-c1-i0064/state --dispatch-txt /Users/josenunez/Projects/ml/snake-dqn-train01-15-c1-i0064-broker.noindex/research/sandbox_a_resident/packet-train01-15-c1-i0064/DISPATCH.txt
```
