# TRAIN-01-17: Mechanical Reseal of `train01-16` with `SIGUSR1` & Compact v2 Continuation Support

Prepared and sealed locally only; **do not push; do not dispatch** (operator dispatches).
Packet `train01-17` (`research/sandbox_a_resident/packet-train01-17`, dispatch identity
`sandbox-a-train01-20261028-17`) is a **mechanical reseal** of `train01-16`
(`sandbox-a-train01-20261027-16`, consumed) running the **exact same PPO learning recipe** and the
**exact same independent seed domain** (`allocation_index = 1`, `train_seed_offset = 1_000_000_000`)
on moved runtime pin `09cfe9f4e8d17aac2f13be8c29aab87d218f8d81` so that this run includes commit
`f80bcdeb`'s `< 15 s` `SIGUSR1` checkpoint/marker path and one-command compact v2 continuation
tooling (`training_continuation.py`).

## 1. Forensic Record of `train01-16` Renewal-Timeout Incident & Validity of Seed-Domain Reuse

- **Attempt-16 stop receipt:**
  `/Users/josenunez/Projects/ml/snake-dqn-artifacts/sandbox-a-resident/train01-attempt-16/output/receipt.json`
  (`sha256 = 77d693aa0e92a6525e5036ed86d063677f57a6d0bec0d9351541e9f69c3b1b8f`) on runtime
  `gpu-a100-s-ass1c0-3rkrbo2c7yjga` (`NVIDIA A100-SXM4-40GB`, calibrated `5.3 u/h`) recorded
  `status = "INCOMPLETE"`, `failure_class = "RPC_EXEC_POLL_TIMEOUT"`,
  `elapsed_seconds = 7215.45011033304` (`2.004 h`), `stop_verified = true`, `cleanup_error = null`,
  and `terminal = null`.
- **Root cause (transport foreground `renewal` RPC timeout while GPU child was healthy):**
  - On the VM, `train01-16` completed the startup compile self-check, `E32` 256-frame distribution
    screen (`training/distribution-screen.json`), `E24` `H5000` temperature calibration
    (`training/calibrated-boundary.{json,pt}`), and `E360` critic fit (`2,000,896` fit rows at child
    elapsed `4,833 s`), and was actively progressing through `E360` critic holdout collection
    (`poll-000171.json`: `stage = "critic-holdout"`, `4,618,800` rows / `27,172` holdout rows at
    child elapsed `6,038 s`). Background `colab-lease` Contents API heartbeats (`renewal-heartbeat`)
    continued succeeding in `~1.0 s` every `30 s`.
  - At `~6,914 s` session elapsed, MainThread's foreground lease-renewal `poll_rpc` (`renewal`,
    `TELEMETRY_SECONDS = 300.0 s` shared budget across `3` attempts) incurred two consecutive
    `120.0 s` `colab exec` timeouts plus `recovery-idle` (`0.892 s`, `27.061 s`) and retry backoffs
    (`2.0 s`, `5.0 s`), leaving `25.014167 s` in the `300.0 s` budget for Attempt 3, which timed out
    at `25.014166 s` (`failure_class: "RPC_EXEC_POLL_TIMEOUT"`).
- **Why reusing `train01-16`'s independent seed domain is scientifically valid:**
  - `train01-16` stopped inside `E360` critic preparation **before** `iteration-0000` (no
    `iteration-NNNN` full-trajectory training snapshot was exported — only
    `distribution-screen.json` and `calibrated-boundary.{json,pt}`, from which `TRAIN01.md` forbids
    resuming PPO).
  - Zero PPO rollouts, zero actor updates, zero `E96` `H5000` proxy evaluations, and zero training
    outcomes were generated or observed. Reusing `allocation_index = 1` and
    `train_seed_offset = 1_000_000_000` therefore introduces **zero outcome selection or adaptive
    reuse** while preserving the intended two-run independent replication pairing against
    `train01-15` (`allocation_index = 0`, `train_seed_offset = 0`).

## 2. Exact Runtime-Pin Delta vs `train01-16` (`5cad044f` -> `09cfe9f4`)

- **Previous runtime pin (`train01-16`):** `5cad044f3d9725a0ddbb967998b82c6eb9cb7739` (`187`
  `source_paths`, `192` sparse-archive files).
- **Moved runtime pin (`train01-17`):** `09cfe9f4e8d17aac2f13be8c29aab87d218f8d81` (`188`
  `source_paths`, `193` sparse-archive files), incorporating continuation/checkpoint commit
  `f80bcdeb1544d91a5d5e1098a3f90b284da208ca` plus the `train01-17` continuation-slug allowlist
  extension.
- **Unchanged across `5cad044f` -> `09cfe9f4` (`179` of `188` `source_paths` byte-identical):**
  - Every file in `src/*`, every GPU kernel/simulator module in `research/gpu_sim_spike/*`, the
    qualified transport tree `research/colab_cli/*`
    (`5833e54e9c5396121cbbf99a5cd4e52006e564fd`), and all numerical/task/model/prep modules
    (`training_model.py`, `training_task.py`, `training_rosters.py`, `training_prep.py`,
    `training_acceptance.py`, `training_compile.py`, `training_cpu_reference.py`, `device_sim.py`,
    `observation_tensor.py`, `stall_watchdog.py`) are **100% byte-identical** to `5cad044f`.
  - Zero learning-recipe, hyperparameter, roster, RNG-draw, or GPU-kernel behavior changed.
- **Exact `9` changed/added shipped files in `research/sandbox_a_resident/`:**
  1. `compact_allowlist.json` (**added**, `f80bcdeb`): tracked
     `sandbox-a-train-compact-allowlist/v2` registry (`entries: []`) for periodic compact XZ
     continuation snapshots.
  2. `train_job.sh` (**modified**, `f80bcdeb`): runs `python -m research.sandbox_a_resident.train`
     in a tracked child PID with `USR1` and `TERM` traps forwarding signals cleanly and a
     signal-resilient `wait` loop.
  3. `train.py` (**modified**, `f80bcdeb`): installs `Sigusr1Controller` (`< 15 s` cooperative
     `SIGUSR1` checkpoint / completion-last `sigusr1-checkpoint.json` marker path), threads
     `should_stop` into collection/learner/proxy, and calls `materialize_compact_resume` when
     resuming from a compact v2 snapshot.
  4. `training_learner.py` (**modified**, `f80bcdeb`): checks optional `should_stop` only between
     minibatches so `SIGUSR1` never leaves a half-applied optimizer step.
  5. `training_proxy.py` (**modified**, `f80bcdeb`): checks optional `should_stop` between proxy
     ticks so an interrupted boundary proxy defers cleanly in snapshot metadata.
  6. `training_compact.py` (**modified**, `f80bcdeb`): generalises lossless XZ stream encode/decode
     and per-tensor SHA-256 manifest verification (`verify_decoded_state`) to periodic
     `iteration-NNNN.pt` snapshots (`v2`).
  7. `training_units.py` (**modified**, `f80bcdeb` + `09cfe9f4`): adds
     `COMPACT_RESUME_SCHEMA_V2`, `compact_allowlist.json` loader/registrar, allowlist-bound
     `allowed_migration_source_commits()`, and extends `TRAIN01_COMPACT_SOURCE_PACKET_RE` to
     `packet-train01-(?:11|15|16|17)` (plus `-c1`/`-c2`).
  8. `training_completion.py` (**modified**, `f80bcdeb`): wires compact v2 derivative inputs
     (`training-export-derivative`) and `compact_source_packet_id` into completion packets.
  9. `training_resume.py` (**modified**, `f80bcdeb`): admits verified mechanical `SUPERVISOR`
     wall/unit/lease stops and mechanical VM-timeout `CHILD_NONZERO_EXIT` receipts alongside
     `RPC_EXEC_POLL_TIMEOUT` while refusing crashes and scientific tripwires.
  - Plus Mac-only (non-shipped) `training_continuation.py` (`f80bcdeb` + `09cfe9f4`), enabling
    one-command sealing of `train01-17-c1-iNNNN` / `train01-17-c2-iNNNN` if `train01-17` stops
    mechanically after exporting any periodic snapshot `iteration-0000..0368`.

## 3. Independent Seed Domain & Identical Proxy/Eval Banks (Reused Unconsumed from `train01-16`)

`control.json` retains `allocation_index = 1` and `train_seed_offset = 1_000_000_000` under
namespace `a100-dev/sandbox-a-train01/`:

| Phase | `train01-11/15` (`offset=0`) | `train01-17` (`offset=1,000,000,000`) | Count | Relationship |
|---|---:|---:|---:|---|
| Master RNG (`_seed`) | index `0` | index `1,000,000,001` | 1 | **Disjoint** |
| `train` (PPO rollouts) | `[0, 100,664,320)` | `[1,000,000,000, 1,100,664,320)` | `100,664,320` | **Disjoint** |
| `calibration` (`E24`) | `[200,000,000, 205,243,904)` | `[1,200,000,000, 1,205,243,904)` | `5,243,904` | **Disjoint** |
| `critic` (`E360` prep) | `[300,000,000, 308,389,632)` | `[1,300,000,000, 1,308,389,632)` | `8,389,632` | **Disjoint** |
| `monitor` (Proxy bank) | `[400,000,000, 400,000,288)` | `[400,000,000, 400,000,288)` | `288` | **IDENTICAL** |
| `nomination` (Eval bank) | `[500,000,000, 500,000,768)` | `[500,000,000, 500,000,768)` | `768` | **IDENTICAL** |
| `engineering` (Screen) | `[600,000,000, 600,000,096)` | `[1,600,000,000, 1,600,000,096)` | `96` | **Disjoint** |

- **Official exhaustive seed audit (`packet-train01-17/registry-check.json`):** binds all
  `114,299,008` canonical roots (`canonical_uint64_sha256 =
  1db80dc4dbe77ed16a04ac52c867e694cc1d9e53e626ebf32aea70fe5f6c12b5`) against
  `SANDBOX_INPUT_REGISTRY.json` (`39b80bcd4cca6a3e137206958fced3292f5aa5cb77e4f5f22cb9e810757ffc2c`):
  `aliases = 0`, `excluded_intersection = 0`, `status = PASS`.
- **Cross-run disjointness audit (`228,596,960` distinct roots):** `0` index-interval overlaps and
  `0` effective 63-bit seed collisions between `train01-11/15` and `train01-17` non-eval domains,
  with `100%` bitwise identity across all `288` `monitor` and `768` `nomination` world seeds.

## 4. Measured A100 Wall-Time Projection & `colab-limits` 2x Envelopes

Projected from the measured `NVIDIA A100-SXM4-40GB` rates in `~/.claude/skills/colab-limits/SKILL.md`
(§3.1, §3.2.A, §4.1–4.5):

| Session Phase | Measured A100 Rate / Basis | Nominal | Conservative Base |
|---|---|---:|---:|
| Alloc + warmup + spawn launch + rate cal + staging + bootstrap | Warmup `2.2–4.1 s`; launch `~0.005 s`; 5 private inputs (`21.60 MB`) + supervisor/archive (`22.56 MB` across 8 uploads: `36.09 s` nom / `62.48 s` worst) | `1.92 min` (`115 s`) | `4.83 min` (`290 s`) |
| Compile self-check + `E32` 256-frame startup screen | Observed `489.1–507.8 s` across 3 mixes | `8.33 min` (`500 s`) | `9.00 min` (`540 s`) |
| `E24` `H5000` temperature calibration (from B8) | Observed `15.6 min` at `100.4–110.7 rows/s` (Attempt 11) | `15.60 min` (`936 s`) | `20.00 min` (`1,200 s`) |
| `E360` `H5000` GPU-resident critic prep (`270` fit + `90` holdout) | Observed `6,233 s` (`4,065 s` fit+SGD + `2,167 s` holdout at `873.3 rows/s`) | `103.88 min` (`6,233 s`) | `115.00 min` (`6,900 s`) |
| `384` PPO rollout collections (`E1024 x T256 = 262,144 rows/update`) | Observed `1,945.6–1,996.8 rows/s` (`~133.0 s/update` nom, `136.0 s/update` cons) | `851.20 min` (`51,072 s`) | `870.40 min` (`52,224 s`) |
| `384` PPO learner updates (2 epochs = `524,288 rows/update`, `float64` KL) | `~140k` cached rows/s (`3.74 s`) + ref forward / PopArt / `float64` KL (`6.5 s` nom, `8.0 s` cons) | `41.60 min` (`2,496 s`) | `51.20 min` (`3,072 s`) |
| `8` scheduled `E96` `H5000` detached proxies (`0, 16, 64..384`) | Observed `2,049.4 s` (`34.16 min`) per proxy; `35.0 min` planning rule | `273.25 min` (`16,395 s`) | `280.00 min` (`16,800 s`) |
| Local SSD snapshot writes + SHA-256 (`26` `.pt` snapshots) | `1` `calibrated-boundary` + `25` iteration snapshots (`~1.33 GB` each; streamed incrementally during polls) | `2.17 min` (`130 s`) | `3.00 min` (`180 s`) |
| Post-terminal final export download + `45 s` stop margin | Final `iteration-0384` batch (`~1.335 GB`, 5 files) at `4.50 MB/s` nom / `1.66 MB/s` worst + `60 s` stop | `7.19 min` (`432 s`) | `15.65 min` (`939 s`) |
| **Whole-Session Total** | **End-to-end single-session wall time** | **`1,305.15 min` (`21.75 h`)** | **`1,369.09 min` (`22.82 h`)** |

- **Frozen base & 2x runaway-guard envelopes:**
  - `requirements.est_minutes = 1380` (`23.0 h`, `technical_envelope.base_seconds = 82,800`,
    `sandbox.json` `timeout.base_seconds = 82,800`).
  - `timeout_minutes = 2760` (`46.0 h = 2 x 1380 min`, `technical_envelope.effective_seconds =
    165,600`, `sandbox.json` `timeout.seconds = 165,600`).
  - `budget.rate_units_per_hour = 8.0` (observed A100 burn `5.3 u/h` -> expected actual spend
    `~115.3 u` over `21.75 h`; base `184.5 u` -> 2x `budget.unit_cap = 369.0 u >= 368.0 u`).
  - **24 h VM continuity & durable snapshot cadence:** nominal single-session duration (`21.75 h`)
    sits `2.25 h` under Colab Pro+'s `~24 h` VM recycle window. Once `iteration-0000` is published
    after preparation, full-trajectory snapshot triples export every `16` updates (`37.20 min`
    nominal across non-proxy intervals; `71.36 min` nominal across scheduled `E96` `H5000` proxy
    boundaries `16, 64, 128, 192, 256, 320, 384`, plus `~5.0–13.4 min` incremental Mac download
    lag), with `< 15 s` `SIGUSR1` cooperative checkpoint/marker handling and one-command compact v2
    continuation (`training_continuation.py`, `max_outcome_blind_retries = 2`).

## 5. `colab-limits` Mandatory Pre-Dispatch Checklist (All 10 Items)

1. **Staging budget (`UPLOAD_TOTAL_TIMEOUT_SECONDS = 300 s`):** **PASS** — uploads `5` private
   inputs (`b8-s3.pth` `6,562,005` B, `frp3-s12.pth` `6,906,247` B, `champion.pth` `6,898,955` B,
   `registry.json` `1,234,356` B, empty `verified-ledger.json` `188` B = **`21,601,751` B** /
   `21.60 MB`) + `bootstrap.sh` (`3,244` B) + `source.py` (`14,511` B) (+ `943,818` B fallback
   `source.tar.gz`).
   - GitHub source path (`7` files, `21,619,506` B = `21.62 MB`): `T_stage,nom = 33.32 s` (`11.11%`
     of `300 s`, `< 120 s` target), `T_stage,worst = 57.65 s` (`19.22%` of `300 s`, `< 210 s`
     target).
   - Fallback `background-sparse-v1` archive path (`8` files, `22,563,324` B = `22.56 MB`):
     `T_stage,nom = 36.09 s` (`12.03%` of `300 s`, `< 120 s` target, `< 150 s` canary bar),
     `T_stage,worst = 62.48 s` (`20.83%` of `300 s`, `< 210 s` target). No `.pt` snapshot is
     uploaded.
2. **Spawn-only launch RPC (`60 s` `EXEC_POLL_SECONDS`):** **PASS** — `sandbox_source_delivery:
   "background-sparse-v1"` spawns background extraction/install in `0.00505–0.00521 s` across `3`
   local `measure_launch_exec_bound` runs (`< 2.0 s` bound, `193` archived files verified) with zero
   heavy synchronous work inside the launch RPC.
3. **Export sizing (`max_export_bytes` & `T_dl,worst`):** **PASS** — `max_export_bytes =
   2,147,483,648` (`2 GiB`); each `iteration-NNNN.pt` is `~1.33 GB` (`~1.24 GiB <= 2 GiB`).
   Intermediate snapshots download incrementally during `15 s` polls (`~5.0–13.4 min` per snapshot
   vs `37.20 min` non-proxy / `71.36 min` proxy-crossing GPU cadence), and the post-terminal final
   export (`~1.335 GB`, 5 files, `T_dl,worst = 14.65 min`) is included in `est_minutes = 1380` and
   `timeout_minutes = 2760`.
4. **GPU memory ceiling (`<= 34.0 GiB` on A100 40 GB):** **PASS** — peak preparation VRAM
   (`E360` simulator + `16.224 GiB` `_ResidentArena` + PopArt critic) is `22.57 GiB <= 34.0 GiB`
   (`57%` of `39.56 GiB`); peak PPO rollout/learner VRAM (`E1024 x T256`) is `~10.4 GiB <=
   34.0 GiB`.
5. **Dual-clock timeout & unit cap (`2x` rule):** **PASS** — `est_minutes = 1380` (`23.0 h` base,
   un-doubled), `timeout_minutes = 2760` (`46.0 h = 2 x est_minutes`), `sandbox.json`
   `timeout = {"base_seconds": 82800, "seconds": 165600, "purpose": "runaway-guard"}`, and
   `unit_cap = 369.0 >= (2760 / 60) * 8.0 = 368.0` (`expected actual burn ~115.3 u` at `5.3 u/h`).
6. **Heartbeat, `SIGUSR1` handler & stall watchdog:** **PASS** — `train_job.sh` forwards `SIGUSR1`
   and `SIGTERM` to `python -m research.sandbox_a_resident.train` (`Sigusr1Controller` `< 15 s`
   graceful checkpoint / `sigusr1-checkpoint.json` marker) and runs `StallWatchdog`
   (`faulthandler.dump_traceback_later(..., exit=True)` C-level stack dump + `_exit(1)` exit code
   `1`, sealed by `train_job.sh`'s `EXIT` trap) with `SANDBOX_A_STALL_TIMEOUT_SECONDS = "900"`,
   while `collect`, `evaluate`, `training_prep`, and `run_distribution_screen` emit progress lines
   at `<= 60 s` cadence.
7. **Resume declaration:** **PASS** — `dispatch-manifest.json` declares `resume = {"unit_kind":
   "training-full-trajectory-snapshot", "ledger_path": "verified-ledger.json", "skip_verified":
   true, "max_outcome_blind_retries": 2}` with an empty hash-bound first-attempt
   `verified-ledger.json` (never combined with `resume_exemption`), and CPU preflights never run a
   256-frame `ResidentTrainingTask` screen on 1-thread Mac CPU.
8. **Clean detached checkouts:** **PASS** — runtime checkout
   `/Users/josenunez/Projects/ml/snake-dqn-train01-17-runtime.noindex` (`HEAD == commit ==
   09cfe9f4e8d17aac2f13be8c29aab87d218f8d81`) and broker checkout
   `/Users/josenunez/Projects/ml/snake-dqn-train01-17-broker.noindex` are fresh, clean
   (`git status --porcelain` empty), detached checkouts with `research/colab_cli` matching the
   qualified transport tree `5833e54e9c5396121cbbf99a5cd4e52006e564fd`.
9. **Dry-run admission (`dispatch.sh --dry-run`):** **PASS** — exact operator command with
   `--dry-run` outputs `PASS_DRY_RUN sandbox-a-train01-20261028-17 allocations=0
   live_accounting=NOT_CHECKED` while leaving `train01-attempt-17/{state,output}` absent.
10. **Offline Mac discipline vs live pre-flight:**
    - Offline Mac engineering discipline (`nice -n 15`, CPU-only, all thread caps `= 1`, zero
      background children left running): **PASS**.
    - Live Colab accounting (`account-status` unit balance and active session cap `< 3`) and live
      host admission (`AC power`, `caffeinate -dimsu`): **`PENDING_LIVE_DISPATCH`** at offline seal
      time (`live_accounting=NOT_CHECKED` in `--dry-run`, verified automatically at live
      `dispatch.sh` launch).
