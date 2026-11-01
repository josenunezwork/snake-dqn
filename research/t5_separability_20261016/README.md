# `t5_separability_20261016`

**Purpose:** T5 byte-validated restoration of T4e return-bank states (19,976 states / 342 worlds) and 5-fold world-held-out separability fits over 24 candidate per-action features.

**Status / Outcome:** **FAIL** against separability bar (2026-10-16: death-AUC gain `+0.0088` vs `+0.05` bar); `restore.py` and `fit.py` reused by T5b, T8, T10, and A2.

## Key Files & Results

- **Package entrypoints / artifacts:** [`restore.py`](restore.py), [`merge.py`](merge.py), [`fit.py`](fit.py), [`t1_check.py`](t1_check.py)
- **Related docs / results:** [`docs/research/t5_separability_2026-10-16.md`](../../docs/research/t5_separability_2026-10-16.md), [`docs/research/catalog/t5_t13_packages.md`](../../docs/research/catalog/t5_t13_packages.md)
