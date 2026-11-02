# `a3_p1_ceiling`

**Purpose:** A3 operational throughput assay measuring the actor-enabled P1 PPO learner ceiling (`128 worlds x 128 steps = 16,384` transitions) across Colab A100-40GB and L4.

**Status / Outcome:** Completed (2026-10-22): **keep P1 on L4 with optimized learner** (optimized learner `0.41 s` on A100 vs `0.98 s` on L4, while host-CPU rollout collection dominated at `~9.56 s` on both, yielding no end-to-end A100 wall-clock advantage).

## Key Files & Results

- **Package entrypoints / artifacts:** [`PROTOCOL.md`](PROTOCOL.md), [`docs/research/history/bl-ppo-p1.md`](../../docs/research/history/bl-ppo-p1.md), [`PREPARED.json`](PREPARED.json), [`benchmark.py`](benchmark.py), [`runtime.py`](runtime.py), [`analyze.py`](analyze.py)
- **Related docs / results:** [`docs/research/final_run_20261019/CAMPAIGN_LOG.md`](../../docs/research/final_run_20261019/CAMPAIGN_LOG.md), [`docs/notes/gpu-engine-and-speed.md`](../../docs/notes/gpu-engine-and-speed.md)
