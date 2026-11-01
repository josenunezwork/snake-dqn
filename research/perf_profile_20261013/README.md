# `perf_profile_20261013`

**Purpose:** Pyinstrument/telemetry component breakdown of `train_m3c` actor and learner bottlenecks, plus host-to-device packing (`bench_hero_h2d.py`) and learner sync (`bench_learner_sync.py`) microbenches.

**Status / Outcome:** Completed M5 profiling diagnostic (2026-10-13; noisy Mac JSONs and stack dumps in `results/`; standalone pass/fail verdict: `unknown` / none).

## Key Files & Results

- **Package entrypoints / artifacts:** [`analyze_profiles.py`](analyze_profiles.py), [`telemetry_split.py`](telemetry_split.py), [`bench_hero_h2d.py`](bench_hero_h2d.py), [`bench_learner_sync.py`](bench_learner_sync.py), [`results/`](results/)
- **Related docs / results:** [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md)
