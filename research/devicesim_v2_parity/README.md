# `devicesim_v2_parity`

**Purpose:** Exact `mechanics_version=2` Watch6 parity harness and A100 Colab queue package comparing GPU `DeviceSim` and `CTDeviceSim` against CPU `GridBatchSim`.

**Status / Outcome:** **Local CPU preflight PASS; A100 CUDA main queued** (2026-10-31: mechanics-v2 parity fixes in `b500d111`, `553` local tests passed, and Mac `E = 4 × 8` preflight recorded `0` mismatches across `DeviceSim`, `CTDeviceSim`, and CPU `GridBatchSim`; A100 `E = 64 × 32` canary and `E = 256 × 1,000` main packet `packet-a100-v2-parity-01` queued in `snake-dqn-artifacts/t34-sol/argon-devicesim-v2/SUMMARY.md`; see [`docs/STATUS.md`](../../docs/STATUS.md)).

## Key Files & Results

- **Package entrypoints / artifacts:** [`harness.py`](harness.py), [`run.py`](run.py), [`units.py`](units.py), [`prepare.py`](prepare.py), [`job.sh`](job.sh)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/INDEX.md`](../../docs/INDEX.md)
