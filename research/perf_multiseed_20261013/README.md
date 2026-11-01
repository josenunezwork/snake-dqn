# `perf_multiseed_20261013`

**Purpose:** Prototype `vmap` / grouped-conv multi-seed stacked `Ego2sNet` learner (`StackedLearner`) benchmarking `K` stacked seeds vs `K` sequential learners, with mutation tests.

**Status / Outcome:** **NO-GO for integration** (`DESIGN.md`, 2026-10-13: `vmap` stacking was slower than sequential updates on Mac MPS; kept as standalone prototype only).

## Key Files & Results

- **Package entrypoints / artifacts:** [`DESIGN.md`](DESIGN.md), [`stacked_learner.py`](stacked_learner.py), [`bench_multiseed.py`](bench_multiseed.py), [`mutation_check.py`](mutation_check.py), [`results/`](results/)
- **Related docs / results:** [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md)
