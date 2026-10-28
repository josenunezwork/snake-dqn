# TRAIN-01 Continuation Tooling & Recovery Contract (`train01-15` / `train01-16`)

Prepared only; no job dispatched. Resolves all four findings in Astra's `2026-10-28` review of
`packet-train01-16` while leaving `research/colab_cli/*` (`5833e54e9c5396121cbbf99a5cd4e52006e564fd`),
serving code, checkpoints, and all hash-bound prior packet files (`packet-train01-15/*`,
`packet-train01-16/*`, `TRAIN01_16.md`, `VERIFICATION_TRAIN01_16.json`) byte-untouched.

---

## 1. Exact One-Command Continuation Packet Builder

After a running `TRAIN-01` attempt (`train01-attempt-15` or `train01-attempt-16`) terminates with a
verified mechanical stop (`status: "INCOMPLETE"`, `stop_verified: true`, `cleanup_error: null`) and
its latest periodic snapshot triple (`iteration-NNNN.{pt,json,-actor.pth}` + `.sha256` + `.complete`)
has landed on the Mac, seal a ready-to-dispatch continuation packet (without dispatching) via:

```bash
CUDA_VISIBLE_DEVICES="" SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 OMP_THREAD_LIMIT=1 TORCHINDUCTOR_COMPILE_THREADS=1 PYTHONDONTWRITEBYTECODE=1 nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m research.sandbox_a_resident.training_continuation --source-attempt /Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-15 --iteration <NNNN>
```

(Replace `train01-attempt-15` with `train01-attempt-16` and `<NNNN>` with the latest completed
periodic snapshot iteration, e.g. `176` or `192`.)

What `training_continuation` executes in one command:
1. Verifies the source attempt's `output/manifest.json` against the frozen source packet, verifies
   `output/receipt.json` via `verify_mechanical_stop`, verifies all exported snapshot triples via
   torch-free `build_ledger`, and enforces outcome-blind latest-snapshot selection (`--iteration`
   must equal `latest < 384`, advancing `completion_attempt.index` by `+1` within `{1, 2}`).
2. Creates (or re-verifies) the create-only lossless `iteration-NNNN.pt.xz` stream (`LZMA2` preset
   `6`, `CHECK_SHA256`) and `iteration-NNNN.pt.xz.manifest` (per-tensor dtype/shape/stride/bytes
   SHA-256 + canonical structure SHA-256), checking full round-trip stream decode before sealing.
3. Creates fresh detached worktrees `snake-dqn-<slug>-runtime.noindex` and
   `snake-dqn-<slug>-broker.noindex`, registers `(source_packet_id, iteration, original_sha256)` in
   the runtime worktree's tracked `compact_allowlist.json` (`runtime_pin`), and seals
   `research/sandbox_a_resident/packet-<slug>/` (`control.json`, `registry-check.json`,
   `verified-ledger.json`, `dispatch-manifest.json`, 10-line `sandbox.json`, `DISPATCH.txt`,
   `SEAL.json`) at `broker_seal`.
4. Preserves the source run's exact learning recipe, `allocation_index`, `train_seed_offset`, and
   shared `monitor` (`400,000,000`, `288` worlds) and `nomination` (`500,000,000`, `768` worlds)
   evaluation banks and proxy schedule while sizing remaining-update base and `2x` effective
   `est_minutes`, `timeout_minutes`, and `unit_cap`.
5. Runs a 3x local `background-sparse-v1` `< 2.0 s` launch-exec bound probe, verifies the 18-file
   whole-staging projection (`< 120 s` nominal / `< 210 s` conservative, well inside the `300 s`
   staging cap), runs `tools/ops/dispatch.sh --dry-run` (`PASS_DRY_RUN ... allocations=0`), and
   verifies `state/` and `output/` remain absent.

---

## 2. Generalised Compact Lossless Resume (`v2` + Legacy `v1`)

- **Schema & Allowlist (`training_units.py`, `compact_allowlist.json`):**
  - Retains legacy `sandbox-a-train-compact-resume/v1` (`COMPACT_RESUME_ALLOWLIST` for
    `train01-11` `iteration-0000`) verbatim so historical `packet-train01-15` preflights stay
    byte-exact.
  - Adds `sandbox-a-train-compact-resume/v2` backed by tracked `compact_allowlist.json`
    (`sandbox-a-train-compact-allowlist/v2`), keyed by `(source_packet_id, iteration, sha256)` across
    `packet-train01-{11,15,16}` and their `-c1`/`-c2` continuations.
  - Extends `ALLOWED_MIGRATION_SOURCE_COMMITS` with `TRAIN01_15_SOURCE_COMMIT`
    (`5c0a101a85237e2ce563dcaa6429828126a37d04`), `TRAIN01_16_SOURCE_COMMIT`
    (`5cad044f3d9725a0ddbb967998b82c6eb9cb7739`), and any `source_commit` bound in
    `compact_allowlist.json` so `c1 -> c2` continuations migrate cleanly without recipe drift.
- **Transport & Runtime Verification (`training_compact.py`, `training_preflight.py`, `train.py`):**
  - Replaces the `~1.33 GB` raw `.pt` upload (`~812 s` nominal, exceeding `300 s` `UPLOAD_CAP`) with
    `iteration-NNNN.pt.xz` (`~8.8–14 MB`) + `iteration-NNNN.pt.xz.manifest` (`~70 KB`), admitted as
    `training-export-derivative` inputs alongside the original `.pt.{sha256,complete}`, `.json`, and
    `-actor.pth` triples (`15` private files + `3` supervisor/source files = `18` uploads).
  - At runtime startup, `materialize_compact_resume` stream-decodes `.pt.xz` create-only into a
    byte-identical `resume/iteration-NNNN.pt`, verifies exact original file size and SHA-256, runs
    ordinary `load_ledger` sidecar verification, loads the checkpoint on CPU, and runs
    `verify_decoded_state` over every tensor's dtype, shape, stride, storage offset, `requires_grad`,
    and contiguous byte SHA-256 plus canonical non-tensor structure before restoring any model,
    optimizer, simulator, opponent, or RNG state.

---

## 3. Admitted Mechanical Terminal Receipts (`training_resume.py`)

`verify_mechanical_stop` requires `status == "INCOMPLETE"`, matching `job_id`, `commit`, and
`manifest_sha256`, `stop_verified is True`, and `cleanup_error is None`, and admits:
- `RPC_EXEC_POLL_TIMEOUT` (with `terminal is None` or `terminal.reason` in
  `{"timeout", "lease_expired", "release_credential_expiring"}`).
- `SUPERVISOR` when `terminal.reason` is in `{"timeout", "lease_expired",
  "release_credential_expiring"}` OR `terminal is None` and `error` matches a mechanical host
  wall/unit/lease/poll marker (`whole-session wall cap reached`, `whole-session deadline reached`,
  `priced session unit cap reached`, `study reserve reached`, `retention lease elapsed`,
  `read retry budget exhausted`) with zero non-mechanical markers (`KeyboardInterrupt`,
  `missing required export`, `canary`, `corrupt`, `schema`).
- `CHILD_NONZERO_EXIT` **only** when `terminal.reason` explicitly records a mechanical VM stop in
  `{"timeout", "lease_expired", "release_credential_expiring"}`.

Refused closed: `terminal.status == "completed"`, scientific tripwires (`FloatingPointError`,
`PreparationFailure`), `StallWatchdog` `os._exit(1)` (`reason: null`), `KeyboardInterrupt`,
`FOREIGN_GENERATION`, `UPLOAD_CAP`, and `ACCOUNTING`.

---

## 4. `SIGUSR1` `< 15 s` Checkpoint Path & Durable Recovery Boundary (`train.py`, `train_job.sh`)

Per `research/colab_cli/runtime.py:504-526`, the VM supervisor sends `SIGUSR1` to the child process
group, grants `15 s` (`checkpoint_grace_seconds = 15`), then escalates to `SIGTERM` (`10 s`) and
`SIGKILL`:
- `train_job.sh` runs `python -m research.sandbox_a_resident.train` in a tracked child PID with
  `USR1` and `TERM` traps forwarding signals cleanly and a signal-resilient `wait` loop.
- `train.py` installs `Sigusr1Controller` (`signal.SIGUSR1`) and threads cooperative `should_stop`
  callbacks into `ResidentTrainingTask.collect` (between ticks), `PPOTrainer.update` (at the end of
  the current minibatch so no half-applied optimizer step occurs), and `training_proxy.evaluate`
  (between proxy ticks, deferring any interrupted boundary proxy in snapshot metadata).
- Within the `15 s` grace window:
  - **Permitted unwritten boundary that fits safely (`estimated_bytes <= max_snapshot_bytes`, default
    `256 MiB`, and remaining grace `>= 3.0 s`):** writes the complete canonical snapshot triple
    (`iteration-NNNN.{pt,json,-actor.pth}` + `.sha256` + `.complete`) plus completion-last
    `sigusr1-checkpoint.json` (`action: "full_snapshot_published"`) and exits `0`.
  - **Mid-iteration / off-cadence / `~1.33 GB` full state exceeding the `15 s` write budget:**
    refuses partial or off-cadence `.pt` writes, immediately writes a completion-last
    `sigusr1-checkpoint.json` + `.sha256` + `.complete` (`action: "last_known_good_marker"`) naming
    `last_known_good_iteration` and `last_known_good_unit`, omits `summary.json` so the attempt stays
    `INCOMPLETE`, closes `StallWatchdog`, and exits `0` in `< 1 s`.
- **Actual durable cross-VM recovery boundary:** because `/content` is ephemeral across VM loss or
  wall-cap termination, cross-attempt continuation always resumes from the **latest completed,
  Mac-downloaded and Mac-verified periodic snapshot triple** (`snapshot_interval = 16` updates:
  `~37.2 min` across non-proxy intervals, or `~71.4 min` across a scheduled `E96 H5000` proxy
  boundary plus serial `8 MiB` export download lag).
