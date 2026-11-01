# `perf_podbench_20261013`

**Purpose:** Money-bounded short-lived RunPod GPU pod benchmark runner executing `bench_learner.py` and `bench_actor.py` (and T18 pod smoke commands) with automatic pod deletion.

**Status / Outcome:** Historical RunPod benchmark runner (2026-10-13/17; RTX 4090 results summarized in `research/perf_quiet_20261013/RESULTS.md`; dormant under T34's no-RunPod rule).

## Key Files & Results

- **Package entrypoints / artifacts:** [`pod_bench.py`](pod_bench.py)
- **Related docs / results:** [`research/perf_quiet_20261013/RESULTS.md`](../perf_quiet_20261013/RESULTS.md), [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md)
