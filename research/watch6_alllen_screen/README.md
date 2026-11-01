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

- **Paired deltas (`exact_stop_alllen - l2_prime_600`):** **`+98.27` mass** (`CI95 [+50.57, +145.98]`, Fieller `+21.31%` `[+10.08%, +34.79%]`, bar `>= +15.0`) and **`-0.1076` `deaths/1k`** (`-67.4%`, `CI95 [-0.1609, -0.0544]`, bar `<= 0.0`).
- **Paired deltas (`exact_stop_alllen - bare`):** **`+208.37` mass** (`CI95 [+170.07, +246.66]`, Fieller `+59.35%` `[+45.24%, +75.82%]`) and **`-0.2361` `deaths/1k`** (`-81.9%`, `CI95 [-0.2828, -0.1895]`); `L < 600` override share = `0.142%` (`0.502%` overall), `mean_max_length = 1,096.0` (vs `943.4` for `L >= 600` and `794.3` for `bare`). Eliminating `34 -> 1` early deaths gives `exact_stop_alllen` `1.379×` more `L >= 600` living exposure (`403.7 -> 556.6` `L >= 600` roots/session), so `11 -> 13` raw `L >= 800` deaths corresponds to `-14.2%` lower per-frame `L >= 800` hazard.
- **Follow-up:** Commit `73974411` wired `"exact-stop-l1"` into [`research/watch_replay/`](../watch_replay/) and [`web/backend/watch_replay.py`](../../web/backend/watch_replay.py), and commit `395eacbb` verified `36/36` sessions from A100 `W = 6` banks `14l1–16l1` (`3.03–3.07 h/bank`, `~61.5 A100 u` observed total at `6.73 u/h`; `t34-sol/argon-banks-l1/SUMMARY.md`) into `watch-replay-bank-live-168` (`web/serve.py` restarted on `b5b63cd1`). This exploration screen is not confirmation-lane release evidence; confirming `exact-stop-l1` efficacy at `H = 14,400` remains a separate confirmation step ([`docs/STATUS.md`](../../docs/STATUS.md), [`docs/research/ROADMAP.md`](../../docs/research/ROADMAP.md)).
