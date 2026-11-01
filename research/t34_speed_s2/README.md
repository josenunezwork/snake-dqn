# `t34_speed_s2`

**Purpose:** Speed program S2 opt-in numba-native `v8` opponent veto for T34 ExIt shadow/branch rows, with byte-identity audit and interleaved S0 whole-workload benchmark.

**Status / Outcome:** **D3 PASS** on contended Mac S0 workload (2026-10-20/21: `1.8994x` whole-workload throughput, median `98.53 s -> 51.88 s`, `144/144` retained files byte-identical; merged opt-in).

## Key Files & Results

- **Package entrypoints / artifacts:** [`RESULT_D3_MAC_CONTENDED_01.md`](RESULT_D3_MAC_CONTENDED_01.md), [`REVIEW_20261021.md`](REVIEW_20261021.md), [`native_v8.py`](native_v8.py), [`identity.py`](identity.py), [`benchmark_s0.py`](benchmark_s0.py), [`receipts/`](receipts/)
- **Related docs / results:** [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md), [`docs/notes/t34-campaign.md`](../../docs/notes/t34-campaign.md)
