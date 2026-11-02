# `t3_rollout_labels_20261015`

**Purpose:** Bitwise-faithful batched counterfactual rollout engine (`engine.py`, `bank_engine.py`), B8 lookahead-shadow ceiling test (`ceiling.py`), dense rollout-label bank collector, and independent bank auditor (`audit_bank.py`).

**Status / Outcome:** Sparse ceiling **`NO_GO`**, dense ceiling **`GO`** (scripted survival `38/64 -> 54/64`, mass `608 -> 700`), dense label bank **`COMPLETE`** + audit **`PASS`** (2026-10-15; reused as load-bearing rollout infrastructure by T34/T35).

## Key Files & Results

- **Package entrypoints / artifacts:** [`engine.py`](engine.py), [`bank_engine.py`](bank_engine.py), [`ceiling.py`](ceiling.py), [`collect.py`](collect.py), [`bank.py`](bank.py), [`audit_bank.py`](audit_bank.py)
- **Related docs / results:** [`docs/research/t3_ceiling_test_2026-10-15.md`](../../docs/research/t3_ceiling_test_2026-10-15.md), [`docs/research/history/m3-m7-redesign.md`](../../docs/research/history/m3-m7-redesign.md)
