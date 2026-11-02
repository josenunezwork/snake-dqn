# Watch6 All-Length Exact-Stop Exploration Screen (`OPUS_META §3 row 2`)

Exploration-lane screen comparing three hero controllers on **48 fresh paired 6,000-frame Watch6-v2 sessions** (`1` hero vs `5× frp3-s12+v8`, continuous respawn, mechanics v2, CPU `GridBatchSim` + `research/bl_served_search/exact_stop.py` with native `kernel` v8):

1. `bare` (`B`): bare B8 s3 (`b8_s3_t7_20261016.pth`, no search).
2. `l2_prime_600` (`L2'`): served exact-stop search at `L >= 600` (`DenseTrigger(min_length=600, interval=5)`, `H300`, respawn trigger reset).
3. `exact_stop_alllen` (`L>=1`): exact-stop search at `L >= 1` (`DenseTrigger(min_length=1, interval=5)`, `H300`, respawn trigger reset).

## Seed Namespace & Disjointness

- Main namespace: `t34-explore-watch6-alllen-screen-20261031-r1` (`N = 48`, `H = 6000`).
- Canary namespace: `t34-explore-watch6-alllen-canary-20261031-r1` (`N = 1`, `H = 120`, `audit_live_mirror=True`).
- Preflight namespace: `t34-explore-watch6-alllen-preflight-20261031-r1` (`N = 2`, `H = 12`, `audit_live_mirror=True`).
- Audited by `python -m research.watch6_alllen_screen.seed_audit` for exact disjointness from Watch6 V1, V2, V3 (`primary` and `reserve`), Watch replay banks 01–04+, C1, `BANK_L2`, `BANK_ROUND0`, `BANK_ROUND0D`, `RETENTION_BANK_V2`, `WORLD_BANK`, and FRP `training_exclusions.json`, without ever opening `exports/` or `pair-*.json`.

## Decision Rule

Adopt `L >= 1` exact-stop search for new Watch replay banks iff:

- Paired mean incremental full-horizon mass delta (`exact_stop_alllen - l2_prime_600`) `>= +15.0` mass, **and**
- Paired mean `deaths/1k` delta (`exact_stop_alllen - l2_prime_600`) `<= 0.0` (no worse).

Otherwise keep served `L2'` (`L >= 600`).

## Input Upload Packing & Transfer Budget

- Pinned checkpoints (`b8_s3_t7_20261016.pth`, `frp3_m3_s12_u60000_20261005.pth`) are split into eight `<= 2 MiB` single-chunk parts (`checkpoint-parts/*.part`), and all contracts, ledgers, `release-evidence.tar.gz`, and `.sha256`/`.complete` sidecars are packed into one deterministic `input-metadata.tar.gz` (`WATCH6_ALLLEN_METADATA_SHA256`), unpacked and verified by `job.sh` before execution.
- `max_export_bytes = 4 MiB` (`4,194,304` B); total uploads = `11` transfers on the normal GitHub source path (`12` including optional `source.tar.gz` fallback), checked offline by `uploads.check_uploads` (`<= 12` transfers, `<= 90 s` at `5 s/RPC + 1 MB/s`, well below the `150 s` transfer and `300 s` staging caps).

## Entry Points

- Seed disjointness audit: `python -m research.watch6_alllen_screen.seed_audit --out <audit.json>`
- Local real-entrypoint preflight (`job.sh` on tiny excluded inputs + 3 long-body `L=900` H300 branch timings): `python -m research.watch6_alllen_screen.preflight --out <preflight_dir>`
- Packet preparation (`canary.json` + `main-unsealed.json` with identical staged inputs): `python -m research.watch6_alllen_screen.prepare --out <packet_dir> --commit <40_char_sha> --benchmark <preflight_dir>/benchmark.json`
- Ready-queue canary-to-main sealer (`--prepare-command`): `python -m research.watch6_alllen_screen.seal_main --ops <ops_dir> --canary-id <id> --packet <packet_dir> --broker <repo>`
- Mac outcome-blind resume ledger builder: `python -m research.watch6_alllen_screen.resume --contract <main-contract.json> --roots <exports_units_dir>... --out <ledger.json>`
- Outcome-blind A100 shard completion builder (branch `origin/argon-alllen-a100-1031` @ `73c26ccc`): `python -m research.watch6_alllen_screen.prepare_resume ...`
- Mac paired-CI readout & decision rule evaluation: `python -m research.watch6_alllen_screen.analyze --units <exports_units_dir>... --out <readout.json>`

## Measured Outcome (2026-11-01 — `ADOPT_EXACT_STOP_L_GE_1_FOR_NEW_BANKS`)

Completed across all **48/48** paired `H = 6,000` Watch6-v2 sessions (`snake-dqn-artifacts/t34-sol/argon-alllen-a100/analysis.json`, uniting `12` verified units `0..11` at `568b09e4` from the stopped L4 `W = 4` run with `36` disjoint units `12..47` at `73c26ccc` across three `12`-unit `W = 6` 12-vCPU A100 completion shards):

| Arm | Mean full-horizon mass | Mean `deaths/1k` | Total deaths (`L < 600` / `L >= 600` / `L >= 800`) | Mean branches/session |
| --- | --- | --- | --- | --- |
| `bare` (`b8-s3`) | `351.07` (`CI95 [320.22, 381.91]`) | `0.2882` (`CI95 [0.2427, 0.3337]`) | `83` (`53` / `30` / `21`) | `0.0` |
| `l2_prime_600` (`L >= 600`) | `461.16` (`CI95 [414.30, 508.02]`) | `0.1597` (`CI95 [0.1060, 0.2134]`) | `46` (`34` / `12` / `11`) | `449.6` (`403.7` roots) |
| **`exact_stop_alllen` (`L >= 1`)** | **`559.43`** (`CI95 [533.59, 585.27]`) | **`0.0521`** (`CI95 [0.0294, 0.0748]`) | **`15`** (**`1`** / **`14`** / **`13`**) | **`1,268.5`** (`1,200.1` roots, `2.821×`) |

- **Paired deltas (`exact_stop_alllen - l2_prime_600`):** **`+98.27` mass** (`CI95 [+50.57, +145.98]`, Fieller `+21.31%` `[+10.08%, +34.79%]`, bar `>= +15.0`; `27` mass wins, `7` ties, and `14/48` `29.2%` mass losses) and **`-0.1076` `deaths/1k`** (`-67.4%`, `CI95 [-0.1609, -0.0544]`, bar `<= 0.0`; `3/48` `6.25%` sessions with more deaths than `l2_prime_600` at `index 36` `seed 3075941873`, `index 39` `seed 3394546952`, and `index 47` `seed 121131908` due to `DenseTrigger.fires()`'s modulo-5 phase shift in `trigger.py:35-42`, unshielded `frame % 5 != 0` inter-check ticks where `ego_raster.py:511` discards `V8Result`, and rollout horizon `H = 300 < L`).
- **Paired deltas (`exact_stop_alllen - bare`):** **`+208.37` mass** (`CI95 [+170.07, +246.66]`, Fieller `+59.35%` `[+45.24%, +75.82%]`) and **`-0.2361` `deaths/1k`** (`-81.9%`, `CI95 [-0.2828, -0.1895]`); `L < 600` override share = `0.142%` (`0.502%` overall), `mean_max_length = 1,096.0` (vs `943.4` for `L >= 600` and `794.3` for `bare`). Eliminating `34 -> 1` early deaths gives `exact_stop_alllen` `1.379×` more `L >= 600` living exposure (`403.7 -> 556.6` `L >= 600` roots/session), so `11 -> 13` raw `L >= 800` deaths corresponds to `-14.2%` lower per-frame `L >= 800` hazard.
- **Follow-up:** Commit `73974411` wired `"exact-stop-l1"` into [`research/watch_replay/`](../watch_replay/) and [`web/backend/watch_replay.py`](../../web/backend/watch_replay.py), and commit `395eacbb` verified `36/36` sessions from A100 `W = 6` banks `14l1–16l1` (`3.03–3.07 h/bank`, `~61.5 A100 u` observed total at `6.73 u/h`; `t34-sol/argon-banks-l1/SUMMARY.md`) into `watch-replay-bank-live-168`. Following Watch6 V3 `GO_EFFICACY_ONLY` (`52135312`) and Mac qualification (`fec81656`), live Watch (`f5912f44`) serves only the `132` confirmed `exact-stop-l2-prime` sessions (`watch-replay-bank-live-132`, `SNAKE_WATCH_REPLAY_CONTROLLER=exact-stop-l2-prime`), keeping the `36` `exact-stop-l1` recordings staged on disk pending prospective `H = 14,400` confirmation via ratified `Watch6-L1-V1` ([`research/watch6_l1_confirm/REGISTRATION_L1_V1.md`](../watch6_l1_confirm/REGISTRATION_L1_V1.md), initial `cf07791b` / `ac250f05` Fable ruling `t34-sol/argon-l1v1-pkg/RULING_L1_V1.md` and `43,200/43,200` Mac mirrors `t34-sol/argon-l1v1-pkg-fix1/mac-mirror.json` (`1.272×`) + R2 `t34-sol/argon-l1v1-pkg-fix2/mac-mirror.json` (`1.573×`, `e89cc8c2…`), Fable R2 `APPROVE-WITH-CONDITIONS` `t34-sol/argon-l1v1-pkg/RULING_L1_V1_R2.md` on `1873b902` / `93c77f43` (`COLAB_OUTPUT_DIR` + `--fixture-horizon 6`, science identity `57ec544fe45f7a8a876dbc9363048468b63358e224b2ee8e11dff41401dcfbd3`), `t34-sol/l1v1-packet-02` canaries `PASS` (`0.574 A100 + 0.126 L4 = 0.70 u` observed), and `18` single-wave shards = `12×6 A100 W = 6` + `6×4 L4 W = 4` running in `t34-sol/l1v1-main-01` (`a3135483`, **`~636.8 u` central / `~637.5 u` incl. canaries**, `598.1–739.5 u`, `~27.7 h` on `6/6` Colab slots calibrated at `26.01 u/h`), alongside the `0 u` Mac `M0b` live-ops patch before `~01:55 AM PT` / `~08:38 AM PT`; see [`docs/STATUS.md`](../../docs/STATUS.md) and [`docs/research/ROADMAP.md`](../../docs/research/ROADMAP.md) §1.3 & §2.1–§2.3), Mac qualification, and owner OK.
- **Exploration-only runner caveat (resolved in `research/watch6_l1_confirm/`):** [`run.py`](run.py) (`lines 51–62, 75–90, 134–250`), [`preflight.py`](preflight.py) (`line 149`), and [`analyze.py`](analyze.py) (`lines 182–200`) omit [`configure_cpu()`](../t1_commitment_20261015/replay.py) (`torch.set_num_threads(1)` / `torch.set_num_interop_threads(1)`), renew `faulthandler.dump_traceback_later(900, exit=True)` only in `on_frame` (`run.py:75-90`) rather than inside `check()`, set `audit_live_mirror=False` on `"main"` (`preflight.py:149`), omit `mirror_identity_frames == 14400` / `native_v8` / `checkpoint_shas` analyzer checks (`analyze.py:182-200`), publish `.complete` inside `W = 4..6` worker children (`line 134`) and `finally:` (`line 224`) even on `SIGUSR1`, print per-unit `mass(B/L600/L1)` to `stdout` (captured in Mac `poll-*.json` `job_log_tail`), write `mass_integral`/`deaths_per_1k` to `operational.json` (`lines 152–160, 224–234`) and `stdout` (`line 249`), and wait on `as_completed` on `SIGUSR1` (`lines 210–221`). Ratified `Watch6-L1-V1` ([`research/watch6_l1_confirm/`](../watch6_l1_confirm/)) fixes those exploration-only shortcuts by calling `configure_cpu()`, renewing `faulthandler` inside both `on_frame` and `check()`, enforcing `audit_live_mirror=True` and full mirror/kernel/checkpoint analyzer checks, emitting metadata-only logs/`operational.json`, and terminating worker children immediately on `SIGUSR1`/`SIGTERM` (with Round-26 `M0b` live-ops advice binding `1873b902:web/backend/session.py` (`da7b4971…`) into `p.identity()` in `final_look.py`/`prepare.py`/`recover.py` while keeping `protocol.py`, `run.py`, and live `session.py` (`fec81656`) untouched, enforcing `ACCOUNT_GPU_SUBCAPS = {"L4": 3, "A100": 3}` before `~01:55 AM PT` / `~08:38 AM PT` + single-GPU `reconcile_unavailable()`, replacing `_prealloc_tree_clean()`'s filename allowlist with a pre-compute forbid-list for `write_once` `.sha256`/`.complete` sidecars, ignoring `{"workload-started.json", "performance.json"}` on `0`-`world-*.json` Tier-0 retries while bypassing `if attempt >= maximum` under `retry_limit = 1` and falling back to `read(manifest)` at `auto_retry.py:401`, linking `unalloc_ref` after base root `att0` to `att0.receipt_path` on chained `-r1 -> -r2` retries, admitting `KERNEL_CREATION_GUARD_ERROR` + polling-count lag, and isolating `l1v1-packet-01` at all 4 canary crash sites; see [`docs/research/ROADMAP.md`](../../docs/research/ROADMAP.md) §1.3 & §2.2).
