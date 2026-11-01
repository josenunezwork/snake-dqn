# `r7_macro_fallback`

**Purpose:** R7 pruned depth-2 (`K = 15` ticks) deterministic spatial macro-option fallback search (`{O_0..O_5}`) evaluated strictly on 1-step-all-die C/T `train` roots (`assessment_roots_used == 0`).

**Status / Outcome:** **`KILL`** (2026-10-31 exploration-lane macro-option fallback evaluation on C/T training `1-step-all-die` roots, `0` assessment roots opened: **`3/5` bars passed** — Mac 1-thread p95 `0.3158 ms < 1.0 ms` PASS, nonfatal duration `15.0` / `29.92` ticks PASS, `L >= 800` `all_die_sel0` rescue `33.0%` vs `9.0%` `exact_stop` PASS on `25/129` `19.4%` `L >= 800` death-lookback roots (`9/25` `36.0%` `macro_triggered` at `+66.7 pp` rescue when triggered vs `16/25` `64.0%` re-trapped after tick-`30` greedy-B8 handback, while `104/129` `80.6%` survived `H = 300` inside uncertified coils and `options.py` hardcoded `min_boost_length = 10` vs live `5`, restricted `omega1` to static `tau <= 1`, and vacuously set `t_cert = 1` on `ext_gain == 0` internal tail cells); **FAILED** total `L >= 600` death-root lift `+3.85 pp < +10.0 pp` and `2 > 0` newly caused ordinary lethal flips from single-suffix `sel0` and aggressive `O3`/`O5`; primary artifact `snake-dqn-artifacts/t34-sol/argon-r7-repack/R7_READOUT.json`; recorded in [`docs/STATUS.md`](../../docs/STATUS.md) and informs `TailExitCert` + `TailFollow` advice in [`docs/research/ROADMAP.md`](../../docs/research/ROADMAP.md) §3).

## Key Files & Results

- **Package entrypoints / artifacts:** [`options.py`](options.py), [`search.py`](search.py), [`readout.py`](readout.py), [`prepare.py`](prepare.py), [`uploads.py`](uploads.py), [`job.sh`](job.sh)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/research/ROADMAP.md`](../../docs/research/ROADMAP.md), [`docs/INDEX.md`](../../docs/INDEX.md)
