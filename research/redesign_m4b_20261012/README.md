# `redesign_m4b_20261012`

**Purpose:** Warm-start weight widener (`widen_bv8.py`) expanding a 30-scalar `ego2s-b` `Ego2sNet` checkpoint into a 60-scalar `ego2s-bv8` network with exact Q preservation.

**Status / Outcome:** Active reusable utility (2026-10-12; initialization path for B8, pinned by `tests/test_ego_raster_bv8.py`).

## Key Files & Results

- **Package entrypoints / artifacts:** [`widen_bv8.py`](widen_bv8.py)
- **Related docs / results:** [`docs/research/m4a_design_2026-10-12.md`](../../docs/research/m4a_design_2026-10-12.md), [`docs/research/catalog/redesign_other.md`](../../docs/research/catalog/redesign_other.md)
