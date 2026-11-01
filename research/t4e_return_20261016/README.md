# `t4e_return_20261016`

**Purpose:** T4e deterministic two-pass D300 branch-return bank collector (`collection.py`), read-only numerical bank bug-check audit, and capped `RETURN` calibration smoke.

**Status / Outcome:** Bank collection **COMPLETE**; `RETURN` training arm halted on boost-collapse bugcheck (2026-10-16; bank states reused across T4f, T5, T8, and A2).

## Key Files & Results

- **Package entrypoints / artifacts:** [`collection.py`](collection.py), [`run_collection.py`](run_collection.py), [`bugcheck.py`](bugcheck.py), [`bugcheck_replay.py`](bugcheck_replay.py), [`seed_audit.py`](seed_audit.py)
- **Related docs / results:** [`docs/research/t4e_collection_result_2026-10-16.md`](../../docs/research/t4e_collection_result_2026-10-16.md), [`docs/research/t4e_bugcheck_2026-10-16.md`](../../docs/research/t4e_bugcheck_2026-10-16.md)
