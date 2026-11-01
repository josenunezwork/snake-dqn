# `perf_compile_20261013`

**Purpose:** Deliberate-break mutation harness (`parity_breaks.py`) and CPU/MPS `torch.compile` parity/drift checks backing `tests/test_perf_compile.py`.

**Status / Outcome:** Completed parity/compile check (2026-10-13: `mps_inductor_check_20261013.json` confirmed MPS eager + compiled inference match CPU, while MPS Inductor backward had conv-bias gradient errors and was rejected in favor of eager).

## Key Files & Results

- **Package entrypoints / artifacts:** [`parity_breaks.py`](parity_breaks.py), [`run_bench.sh`](run_bench.sh), [`results/mps_inductor_check_20261013.json`](results/mps_inductor_check_20261013.json), [`results/`](results/)
- **Related docs / results:** [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md)
