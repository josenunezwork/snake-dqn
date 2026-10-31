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

## Entry Points

- Seed disjointness audit: `python -m research.watch6_alllen_screen.seed_audit --out <audit.json>`
- Local real-entrypoint preflight (`job.sh` on tiny excluded inputs + 3 long-body `L=900` H300 branch timings): `python -m research.watch6_alllen_screen.preflight --out <preflight_dir>`
- Packet preparation (`canary.json` + `main-unsealed.json` with identical staged inputs): `python -m research.watch6_alllen_screen.prepare --out <packet_dir> --commit <40_char_sha> --benchmark <preflight_dir>/benchmark.json`
- Ready-queue canary-to-main sealer (`--prepare-command`): `python -m research.watch6_alllen_screen.seal_main --ops <ops_dir> --canary-id <id> --packet <packet_dir> --broker <repo>`
- Mac outcome-blind resume ledger builder: `python -m research.watch6_alllen_screen.resume --contract <main-contract.json> --roots <exports_units_dir>... --out <ledger.json>`
- Mac paired-CI readout & decision rule evaluation: `python -m research.watch6_alllen_screen.analyze --units <exports_units_dir>... --out <readout.json>`
