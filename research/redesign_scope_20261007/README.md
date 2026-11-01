# `redesign_scope_20261007`

**Purpose:** Redesign Milestone 1 scoping benchmarks and H5000 parity checks (`GridBatchSim` vs `BatchSim` vs live `GameState`, Gate G1 `>= 20k` agent-steps/s, numba `ego2s` featurizer).

**Status / Outcome:** **PASS** (2026-10-07/08: `GridBatchSim` H5000 identity `72/72` exact, live `ego2s` identity `3/3` exact after food-respawn ordering fix, Gate G1 `33.2k–36.9k` agent-steps/s).

## Key Files & Results

- **Package entrypoints / artifacts:** [`bench_throughput.py`](bench_throughput.py), [`bench_g1.py`](bench_g1.py), [`grid_h5000_identity.py`](grid_h5000_identity.py), [`ego_live_identity.py`](ego_live_identity.py), [`results/grid_identity_h5000/summary.json`](results/grid_identity_h5000/summary.json)
- **Related docs / results:** [`docs/research/redesign_scope_2026-10-07.md`](../../docs/research/redesign_scope_2026-10-07.md), [`docs/research/catalog/redesign_other.md`](../../docs/research/catalog/redesign_other.md)
