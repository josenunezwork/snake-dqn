# `redesign_m2_20261008`

**Purpose:** M2 / M2b `frp3-s12 + v8` teacher distillation data generator (`gen_data.py`), 61-D feature coverage audit, dev probe (`dev_probe.py`), and pre-registered H5000 non-inferiority check (`ni_check.py`).

**Status / Outcome:** M2 **`FAIL`** (`results/ni_check/verdict.json`, pooled delta `-29.5`); M2b (`ego2s-b` 30-scalar fix) **`PASS`** (`results_m2b/ni_check/verdict.json`, `+42.5` mass vs bare `frp3-s12`, live identity `3/3` exact, 2026-10-08). `dev_probe.py` and `ni_spec.py` remain widely imported.

## Key Files & Results

- **Package entrypoints / artifacts:** [`PREREGISTRATION.md`](PREREGISTRATION.md), [`PREREGISTRATION_M2B.md`](PREREGISTRATION_M2B.md), [`gen_data.py`](gen_data.py), [`ni_check.py`](ni_check.py), [`dev_probe.py`](dev_probe.py), [`results/ni_check/verdict.json`](results/ni_check/verdict.json), [`results_m2b/ni_check/verdict.json`](results_m2b/ni_check/verdict.json)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/research/catalog/redesign_m2_20261008.md`](../../docs/research/catalog/redesign_m2_20261008.md)
