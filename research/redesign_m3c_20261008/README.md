# `redesign_m3c_20261008`

**Purpose:** M3-C multi-actor/learner RL trainer (`train_m3c.py`, `learner_m3c.py`), full-state checkpointing, RunPod v2 GPU runner, league pool, and Phase R harness.

**Status / Outcome:** Historical M3-C study (A1 pod stalled 2026-10-08; learnings folded into M4-A/B8), while `train_m3c.py` / `learner_m3c.py` remained the **active trainer core** for M4-A (B8), M5, M6, T3–T6, T11, T16, T23, and S1.

## Key Files & Results

- **Package entrypoints / artifacts:** [`PREREGISTRATION_M3C.md`](PREREGISTRATION_M3C.md), [`CHECKPOINTS.md`](CHECKPOINTS.md), [`learner_m3c.py`](learner_m3c.py), [`phase_r_m3c.py`](phase_r_m3c.py), [`results/`](results/)
- **Related docs / results:** [`docs/research/m3c_design_2026-10-08.md`](../../docs/research/m3c_design_2026-10-08.md), [`docs/research/history/m3-m7-redesign.md`](../../docs/research/history/rl-runs.md#m3-m7-redesign)
