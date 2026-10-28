# TRAIN-01-16: Independent Fresh Full-Run Replication of `train01-15`

Prepared and sealed locally only; **do not push; do not dispatch** (operator dispatches).
Packet `train01-16` (`research/sandbox_a_resident/packet-train01-16`, dispatch identity
`sandbox-a-train01-20261027-16`) is a **second, independent full `TRAIN-01` run** replicating the
currently running `train01-15` (`sandbox-a-train01-20261027-15`, which resumed from `train01-11`'s
post-preparation `iteration-0000` snapshot).

## 1. Replication Design & Frozen Recipe Parity with `train01-15`

- **Same runtime & recipe as `train01-15`:** retains the exact `train01-15` PPO recipe and runtime,
  including `float64` `log_softmax` legal KL and entropy with `torch.where` off-support zeroing and
  standard PyTorch autograd (`training_model.py`), `outer_iterations = 384`, `worlds = 1024`,
  `rollout_horizon = 256` (`100,663,296` total actor rows), `task_horizon = 5000`,
  `snapshot_interval = 16`, `proxy_interval = 64` (plus diagnostic update `16`),
  `proxy_worlds_per_mix = 96`, `proxy_horizon = 5000`, `compile_mode = "default"`,
  `sandbox_source_delivery = "background-sparse-v1"`, and `SANDBOX_A_STALL_TIMEOUT_SECONDS = "900"`.
- **Fresh start from B8 (no resume):** starts directly from pinned B8 s3
  (`3e64978337b67b7acaf4daa93bdefa827d8296bcb0f63413886a1aaad12eb4f8`) with an empty first-attempt
  `verified-ledger.json` (`units: []`), executing the startup compile self-check, `E32` 256-frame
  3-mix distribution screen, full `E24` `H5000` temperature calibration, full `E360` (`270` fit +
  `90` holdout) GPU-resident `_ResidentArena` critic preparation (`16.224 GiB`), and all `384` outer
  PPO updates with `8` scheduled `E96` `H5000` detached proxies (`0, 16, 64, 128, 192, 256, 320, 384`).
- **Targeted code wiring for independent seed domains:**
  - `training_task.py` (`seed_allocations`): reads `control.get("train_seed_offset", 0)` (validated
    nonnegative `int`), shifts the non-evaluation domains (`train`, `calibration`, `critic`,
    `engineering`) by `train_seed_offset`, keeps `monitor` (`400,000,000`, count `288`) and
    `nomination` (`500,000,000`, count `768`) fixed, and rejects any overlapping intervals.
  - `train.py` & `training_cpu_reference.py`: validates `allocation_index` and `train_seed_offset`
    in `TrainControl.__post_init__`, routes `audited_allocations["engineering"]["offset"]` to both
    CPU and GPU startup distribution-screen factories, and binds `ProxyConfig.seed_base` to
    `audited_allocations["monitor"]["offset"]` (`400,000,000`).
  - `training_preflight.py` & `training_prepare.py`: admits explicit nonnegative integer
    `(allocation_index, train_seed_offset)` while keeping every production recipe parameter frozen.

## 2. Independent Seed Domain & Identical Proxy/Eval Banks

`control.json` sets `allocation_index = 1` and `train_seed_offset = 1_000_000_000` under namespace
`a100-dev/sandbox-a-train01/`:

| Phase | `train01-11/15` (`offset=0`) | `train01-16` (`offset=1,000,000,000`) | Count | Relationship |
|---|---:|---:|---:|---|
| Master RNG (`_seed`) | index `0` | index `1,000,000,001` | 1 | **Disjoint** |
| `train` (PPO rollouts) | `[0, 100,664,320)` | `[1,000,000,000, 1,100,664,320)` | `100,664,320` | **Disjoint** |
| `calibration` (`E24`) | `[200,000,000, 205,243,904)` | `[1,200,000,000, 1,205,243,904)` | `5,243,904` | **Disjoint** |
| `critic` (`E360` prep) | `[300,000,000, 308,389,632)` | `[1,300,000,000, 1,308,389,632)` | `8,389,632` | **Disjoint** |
| `monitor` (Proxy bank) | `[400,000,000, 400,000,288)` | `[400,000,000, 400,000,288)` | `288` | **IDENTICAL** |
| `nomination` (Eval bank) | `[500,000,000, 500,000,768)` | `[500,000,000, 500,000,768)` | `768` | **IDENTICAL** |
| `engineering` (Screen) | `[600,000,000, 600,000,096)` | `[1,600,000,000, 1,600,000,096)` | `96` | **Disjoint** |

- **Official exhaustive seed audit (`packet-train01-16/registry-check.json`):** audited all
  `114,299,008` canonical roots (`canonical_uint64_sha256 =
  1db80dc4dbe77ed16a04ac52c867e694cc1d9e53e626ebf32aea70fe5f6c12b5`) against
  `SANDBOX_INPUT_REGISTRY.json` (`39b80bcd4cca6a3e137206958fced3292f5aa5cb77e4f5f22cb9e810757ffc2c`):
  `aliases = 0`, `excluded_intersection = 0`, `status = PASS`.
- **Cross-run disjointness audit (`seed-disjointness-audit.json`):** audited all `228,596,960`
  distinct roots across `train01-11/15`'s `114,297,952` non-eval roots, `train01-16`'s `114,297,952`
  non-eval roots, and the `1,056` shared `monitor` + `nomination` roots: `0` index-interval overlaps,
  `0` effective 63-bit seed collisions across both runs, and `100%` bitwise identity on all `288`
  `monitor` and `768` `nomination` world seeds.

## 3. Measured A100 Wall-Time Projection & `colab-limits` 2x Envelopes

Projected from the measured `NVIDIA A100-SXM4-40GB` rates in `~/.claude/skills/colab-limits/SKILL.md`
(§3.1, §3.2.A, §4.1–4.5):

| Session Phase | Measured A100 Rate / Basis | Nominal | Conservative Base |
|---|---|---:|---:|
| Alloc + warmup + spawn launch + rate cal + staging + bootstrap | Warmup `2.2–4.1 s`; launch `<0.05 s`; 5 inputs (`12.2 MB`, `36.1 s` nom / `62.5 s` worst) | `1.92 min` (`115 s`) | `4.83 min` (`290 s`) |
| Compile self-check + `E32` 256-frame startup screen | Observed `489.1–507.8 s` across 3 mixes | `8.33 min` (`500 s`) | `9.00 min` (`540 s`) |
| `E24` `H5000` temperature calibration (from B8) | Observed `15.6 min` at `100.4–110.7 rows/s` (Attempt 11) | `15.60 min` (`936 s`) | `20.00 min` (`1,200 s`) |
| `E360` `H5000` GPU-resident critic prep (`270` fit + `90` holdout) | Observed `6,233 s` (`4,065 s` fit+SGD at `862.9` charged / `493.2` drained rows/s + `2,167 s` holdout at `873.3 rows/s`) | `103.88 min` (`6,233 s`) | `115.00 min` (`6,900 s`) |
| `384` PPO rollout collections (`E1024 x T256 = 262,144 rows/update`) | Observed `1,945.6–1,996.8 rows/s` (`~133.0 s/update` nom, `136.0 s/update` cons) | `851.20 min` (`51,072 s`) | `870.40 min` (`52,224 s`) |
| `384` PPO learner updates (2 epochs = `524,288 rows/update`, `float64` KL) | `~140k` cached rows/s (`3.74 s`) + ref forward / PopArt / `float64` KL (`6.5 s` nom, `8.0 s` cons) | `41.60 min` (`2,496 s`) | `51.20 min` (`3,072 s`) |
| `8` scheduled `E96` `H5000` detached proxies (`0, 16, 64..384`) | Observed `2,049.4 s` (`34.16 min`) per proxy; `35.0 min` planning rule | `273.25 min` (`16,395 s`) | `280.00 min` (`16,800 s`) |
| Local SSD snapshot writes + SHA-256 (`26` `.pt` snapshots) | `1` `calibrated-boundary` + `25` iteration snapshots (`~1.33 GB` each; streamed in parallel during polls) | `2.17 min` (`130 s`) | `3.00 min` (`180 s`) |
| Post-terminal final export download + `45 s` stop margin | Final `iteration-0384` batch (`~1.335 GB`, 5 files) at `4.50 MB/s` nom / `1.66 MB/s` worst + `60 s` stop | `7.19 min` (`432 s`) | `15.65 min` (`939 s`) |
| **Whole-Session Total** | **End-to-end single-session wall time** | **`1,305.15 min` (`21.75 h`)** | **`1,369.09 min` (`22.82 h`)** |

- **Frozen base & 2x runaway-guard envelopes:**
  - `requirements.est_minutes = 1380` (`23.0 h`, `technical_envelope.base_seconds = 82,800`,
    `sandbox.json` `timeout.base_seconds = 82,800`) — covers both nominal (`21.75 h`) and
    conservative (`22.82 h`, including worst-case `1.66 MB/s` post-terminal download and `45 s`
    supervisor margin).
  - `timeout_minutes = 2760` (`46.0 h = 2 x 1380 min`, `technical_envelope.effective_seconds =
    165,600`, `sandbox.json` `timeout.seconds = 165,600`).
  - `budget.rate_units_per_hour = 8.0` (observed A100 burn `5.3 u/h` -> expected actual spend
    `~115.3 u` over `21.75 h`; base priced ceiling at `8.0 u/h` over `23.0 h` = `184.0 u`, rounded to
    `184.5 u` base -> 2x `budget.unit_cap = 369.0 u >= 2760 * 8.0 / 60 = 368.0 u`).
  - **24 h VM continuity note:** nominal duration (`21.75 h`) is `2.25 h` under Colab Pro+'s `~24 h`
    continuous VM window; full-trajectory snapshots export every `16` updates (`~37.2 min`) with
    `resume.max_outcome_blind_retries = 2` if Colab recycles a VM late in the run.

## 4. `colab-limits` Mandatory Pre-Dispatch Checklist (All 10 Items)

1. **Staging budget (`UPLOAD_TOTAL_TIMEOUT_SECONDS = 300 s`):** **PASS** — fresh run uploads only `5`
   fixed private inputs (`b8-s3.pth`, `frp3-s12.pth`, `champion.pth`, `registry.json`, empty
   `verified-ledger.json` = `11,222,360 B`) + `bootstrap.sh` + `source.py` (+ `~0.94 MB` fallback
   `source.tar.gz` -> `8` files, `~12.2 MB` total). Projected `T_stage,nom = 26.79 s` (GitHub) /
   `36.07 s` (fallback tarball, `12.0%` of `300 s`, `< 120 s` target, `< 150 s` canary bar) and
   `T_stage,worst = 45.67 s` (GitHub) / `62.52 s` (fallback tarball, `20.8%` of `300 s`, `< 210 s`
   target). No `.pt` snapshot is uploaded.
2. **Spawn-only launch RPC (`60 s` `EXEC_POLL_SECONDS`):** **PASS** — `sandbox_source_delivery:
   "background-sparse-v1"` spawns background extraction/install in `< 2.0 s` (`~0.026 s` measured
   across 3 local runs of `measure_launch.py` on the pinned archive) and does zero heavy synchronous
   work inside the launch RPC.
3. **Export sizing (`max_export_bytes` & `T_dl,worst`):** **PASS** — `max_export_bytes = 2,147,483,648`
   (`2 GiB`); each full-state snapshot `iteration-XXXX.pt` is `~1.33 GB` (`~1.24 GiB <= 2 GiB`).
   `collect_exports` downloads intermediate snapshots incrementally during 15 s polls in parallel
   with GPU compute (`~5.0–13.4 min` per snapshot vs `~37.2 min` 16-update cadence), and the
   post-terminal final export (`~1.335 GB`, 5 files, `T_dl,worst = 14.65 min`) is explicitly included
   inside `est_minutes = 1380` and `timeout_minutes = 2760`.
4. **GPU memory ceiling (`<= 34.0 GiB` on A100 40 GB):** **PASS** — peak preparation VRAM
   (`E360` simulator + `16.224 GiB` `_ResidentArena` + PopArt critic) is `22.57 GiB <= 34.0 GiB`
   (`57%` of `39.56 GiB`), and peak PPO rollout/learner VRAM (`E1024 x T256`) is `~10.4 GiB <=
   34.0 GiB`.
5. **Dual-clock timeout & unit cap (`2x` rule):** **PASS** — `est_minutes = 1380` (`23.0 h` base,
   un-doubled), `timeout_minutes = 2760` (`46.0 h = 2 x est_minutes`), `sandbox.json`
   `timeout = {"base_seconds": 82800, "seconds": 165600, "purpose": "runaway-guard"}`, and
   `unit_cap = 369.0 >= (2760 / 60) * 8.0 = 368.0` (`expected actual burn ~115.3 u` at `5.3 u/h`).
6. **Heartbeat & stall watchdog:** **PASS** — `train_job.sh` wraps `python -m
   research.sandbox_a_resident.train` in `StallWatchdog` (`faulthandler` stack dump + exit `124`)
   with `SANDBOX_A_STALL_TIMEOUT_SECONDS = "900"`, and `collect` (`train.py:261`), `evaluate`
   (`training_proxy.py:137`), `training_prep.py`, and `run_distribution_screen` all emit progress
   lines at `<= 60 s` cadence.
7. **Resume declaration:** **PASS** — `dispatch-manifest.json` declares `resume = {"unit_kind":
   "training-full-trajectory-snapshot", "ledger_path": "verified-ledger.json", "skip_verified":
   true, "max_outcome_blind_retries": 2}` with an empty hash-bound first-attempt
   `verified-ledger.json` (never combined with `resume_exemption`), and CPU preflights never execute
   a 256-frame `ResidentTrainingTask` screen on 1-thread Mac CPU.
8. **Clean detached checkouts:** **PASS** — runtime checkout
   `/Users/josenunez/Projects/ml/snake-dqn-train01-16-runtime.noindex` (`HEAD == commit`) and broker
   checkout `/Users/josenunez/Projects/ml/snake-dqn-train01-16-broker.noindex` are fresh, clean
   (`git status --porcelain` empty), detached checkouts with `research/colab_cli` matching the
   qualified transport tree `5833e54e9c5396121cbbf99a5cd4e52006e564fd`.
9. **Dry-run admission (`dispatch.sh --dry-run`):** **PASS** — exact operator command with
   `--dry-run` passes offline schema, registry, source-pin, seed-audit, and focused test admission
   (`allocations=0`) while leaving `train01-attempt-16/{state,output}` absent.
10. **Live pre-flight & Mac host discipline:** **PASS (offline prep) / READY (operator live step)** —
    all local engineering, seed audits, tests, and `--dry-run` ran under `nice -n 15` with all
    thread caps `= 1` on CPU and zero background processes left behind; recent account balance is
    `~998.58 u` (`369.0 u` cap + `129.0 u` `train01-15` cap `= 498.0 u << 998.58 u`, active
    sessions `< 3`). Operator runs `dispatch.sh` under `caffeinate -dimsu` on AC power at live
    launch.
