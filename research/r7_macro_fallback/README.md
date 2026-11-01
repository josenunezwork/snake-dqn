# `r7_macro_fallback`

**Purpose:** R7 pruned depth-2 (`K = 15` ticks) deterministic spatial macro-option fallback search (`{O_0..O_5}`) evaluated strictly on 1-step-all-die C/T `train` roots (`assessment_roots_used == 0`).

**Status / Outcome:** **`KILL`** (2026-10-31 exploration-lane macro-option fallback evaluation on C/T training `1-step-all-die` roots, `0` assessment roots opened: **`3/5` bars passed** — Mac 1-thread p95 `0.3158 ms < 1.0 ms` PASS, nonfatal duration `15.0` / `29.92` ticks PASS, `L >= 800` `all-die` rescue `33.0%` vs `9.0%` `exact_stop` PASS; **FAILED** total `L >= 600` death-root lift `+3.85 pp < +10.0 pp` and `2 > 0` newly caused ordinary lethal flips; primary artifact `snake-dqn-artifacts/t34-sol/argon-r7-repack/R7_READOUT.json`; recorded in [`docs/STATUS.md`](../../docs/STATUS.md)).

## Key Files & Results

- **Package entrypoints / artifacts:** [`options.py`](options.py), [`search.py`](search.py), [`readout.py`](readout.py), [`prepare.py`](prepare.py), [`uploads.py`](uploads.py), [`job.sh`](job.sh)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/INDEX.md`](../../docs/INDEX.md)
