# `t8_probes_20261017`

**Purpose:** T8 supervised diagnostics on restored T5 states: geometric arrival-race feature fits (GBM/LR) and raw-raster D300 CNN classifier on T5's world-held-out folds.

**Status / Outcome:** Completed diagnostic — **FAIL against 0.85 AUC bar** (2026-10-16/17: raw-raster CNN AUC `0.742`, arrival-race delta `+0.014`; pointed to missing multi-step dynamics rather than trunk capacity).

## Key Files & Results

- **Package entrypoints / artifacts:** [`arrival.py`](arrival.py), [`arrival_fit.py`](arrival_fit.py), [`cnn.py`](cnn.py), [`merge.py`](merge.py), [`audit.py`](audit.py)
- **Related docs / results:** [`docs/research/t8_probes_2026-10-17.md`](../../docs/research/t8_probes_2026-10-17.md), [`docs/research/catalog/t5_t13_packages.md`](../../docs/research/catalog/t5_t13_packages.md)
