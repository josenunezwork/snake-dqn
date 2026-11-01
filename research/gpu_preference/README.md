# `gpu_preference`

**Purpose:** Synthetic H100/A100 CUDA runtime acceptance probe (no game, checkpoint, or science inputs) exercising H100-preferred allocation with pre-assignment A100 fallback.

**Status / Outcome:** Infrastructure acceptance package (landed 2026-10-28 with H100 preference + pre-assignment A100 fallback); standalone run receipt in repo: `unknown` (external ops artifacts).

## Key Files & Results

- **Package entrypoints / artifacts:** [`acceptance.py`](acceptance.py), [`acceptance.sh`](acceptance.sh), [`requirements.txt`](requirements.txt)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md)
