# `devicesim_v2_parity`

**Purpose:** Exact `mechanics_version=2` Watch6 parity harness and A100 Colab queue package comparing GPU `DeviceSim` and `CTDeviceSim` against CPU `GridBatchSim`.

**Status / Outcome:** **A100 CUDA `PASS` (`0` mismatches across `1,536,000` snake-ticks)** (2026-11-01: main packet `snake-dqn-artifacts/t34-sol/argon-devicesim-v2/packet-a100-v2-parity-01/main-run-01/receipt.json` completed `PASS` with `stop_verified=true` in `3,505.6 s` = `1,434.6` steps/s under 5-opponent CUDA flood launches, verifying `E = 256 × 1,000` Watch6-v2 ticks = `1,536,000` snake-ticks across GPU `DeviceSim` (SplitMix64 RNG), `CTDeviceSim`, and CPU `GridBatchSim` with `0` mismatches within `<= 1e-4` CUDA FP32 near-tie Q re-decision checks at commit `b500d111`, following `553 passed` local tests and Mac `E = 4 × 8` preflight `PASS`; see [`docs/STATUS.md`](../../docs/STATUS.md)).

## Key Files & Results

- **Package entrypoints / artifacts:** [`harness.py`](harness.py), [`run.py`](run.py), [`units.py`](units.py), [`prepare.py`](prepare.py), [`job.sh`](job.sh)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/INDEX.md`](../../docs/INDEX.md)
