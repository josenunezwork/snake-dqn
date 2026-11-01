# `t4_gate1`

**Purpose:** Track-4 Gate 1 privileged full-arena factored critic (competing-horizon discrete survival-hazard head + short-gamma alive foraging mass-flux head, zero B8 weights) harvested and evaluated on fresh world-disjoint Watch6-v2 `DeviceSim` trajectories.

**Status / Outcome:** **`KILL_T4`** (2026-10-31 exploration-lane Gate-1 critic check across `384` disjoint Watch6-v2 worlds = `204` fit / `52` val / `128` test, 2 replicates: **`0/8` bars passed** — `L >= 600` residual EV `+0.0266`, `CI95 [-0.0280, +0.0869]` vs `>= 0.25`; death-root H300 rescue mass `+19.99`, `CI95 [-4.98, +45.18]` vs `>= +40`; mature doom ROC-AUC `0.5617` vs `>= 0.80`; ordinary H300 mass `-22.21` vs `>= -15.0`; ordinary lethal flips `3.125%` vs `<= 3.0%`; primary artifact `snake-dqn-artifacts/t34-sol/argon-t4-gate1-fix2/queue-03/READOUT.json`; recorded in [`docs/STATUS.md`](../../docs/STATUS.md)).

## Key Files & Results

- **Package entrypoints / artifacts:** [`pipeline.py`](pipeline.py), [`harvest.py`](harvest.py), [`featurizer.py`](featurizer.py), [`model.py`](model.py), [`readout.py`](readout.py), [`prepare.py`](prepare.py)
- **Related docs / results:** [`docs/STATUS.md`](../../docs/STATUS.md), [`docs/INDEX.md`](../../docs/INDEX.md)
