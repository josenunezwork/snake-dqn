# Phase 0 teacher speed follow-up — 2026-10-19

Outcome: implementation and fixture verification pass; **the 20x Mac CPU speed
target is not met**. The default teacher is faster, and MPS is optional, but
million-decision distillation and 41 ms planner serving remain expensive.
This is disjoint performance evidence, not a Phase 0 scientific result.

The implementation batches all continuations in one GridBatchSim, builds one
numba hero observation batch per step, runs one B8 batch, removes duplicate
hero dispatch, and reuses the action grid. Complete byte-identical neural inputs
share a forward result without merging simulator rows or their RNG streams.
Frozen opponents keep one shared grouped policy across replica rows. The old
CPU scorer remains selectable with `implementation="reference"`; `device="mps"`
is explicit. The registered config and original preregistration/bank/receipts
are unchanged. No remote runtime or serving code was operated or modified.

## Same preflight protocol, three rounds

Seconds per searched decision, including replication and device-model copy on
MPS, but excluding fixture construction and the initial warm-up. Each cell is
the median of three runs; N8/H128/epsilon=.02, six legal first actions. These are
valid synthetic bodies with real opponents, including early rollout deaths.

| Hero length | CPU scripted | CPU frozen | CPU mixed | MPS scripted | MPS frozen | MPS mixed |
|---|---:|---:|---:|---:|---:|---:|
| 200 | 1.389 | 1.559 | 1.486 | 0.723 | 1.200 | 1.050 |
| 800 | 1.070 | 1.432 | 1.692 | 0.753 | 1.228 | 1.180 |
| 1500 | 1.003 | 1.336 | 1.245 | 0.849 | 1.319 | 1.148 |

See [PROFILE.md](PROFILE.md) for the complete before/after category tables and
same-host reference speedups. Raw timing receipts are `SPEED_TIMING_CPU.json`
and `SPEED_TIMING_MPS.json`; the source/input hashes and exact argv are recorded.

Six identical L1500 mixed searches, including fixture/model/process startup:

| Workers | Batch wall seconds | Speedup |
|---|---:|---:|
| 1 | 8.174 | 1.00x |
| 2 | 5.406 | 1.51x |
| 6 | 2.928 | 2.79x |

Actions and complete score vectors are exactly equal in all six tasks at one,
two and six workers. This is a bounded scoring fixture check, not full-bank
steady-state scaling. Bare-B8 recalibration reproduces 11,446 alive decisions
and the original **43.395%** mandatory L>=600 fraction. The threshold remains 0.

## Forecasts

Use the original extrapolation: 96/32/32 selection counts multiplied by the
calibration long decisions per world, averaging the L800/L1500 timing cells
(now each cell has three rounds), plus twice bare calibration-world cost.
That gives 276,256 searched Phase 0 decisions. Divide by the measured six-task
speedup 2.791 for the Mac wall scenario.
One million searched distillation decisions assumes equal mix proportions and
excludes collection of unsearched states, storage and student training.

| Workload | Mac single-core hours | Mac six-worker wall hours | RunPod 32-vCPU hours (scenario) | Historical pod USD | L4 hours (conditional bounds) | Historical L4 units |
|---|---:|---:|---:|---:|---:|---:|
| Phase 0 | 91.25 | 32.69 | 6.84 | 6.09–6.50 | 30.51–56.84 | 61.0–113.7 |
| 1M searched decisions | 360.02 | 128.98 | 27.00 | 24.03–25.65 | 116.93–229.75 | 233.9–459.5 |

Phase 0's prior forecast was 337.34 core-hours / 88.85 Mac wall-hours. The new
CPU forecast is 3.70x lower in core-hours and
2.72x lower in wall-hours. With 20% reserve,
plan 39.23 Mac hours for Phase 0; the same reserve would make
one million decisions 154.77 Mac hours. Mac cloud rental cost is zero,
excluding electricity, depreciation and opportunity cost.

Pod arithmetic assumes 32 independent worker slots and the historical 2.4x
per-core slowdown, at the historical $0.89–0.95/h. Its projected 4.78x Mac
speedup excludes remote setup/transfers and is **not an observed or qualified
4x-rule result**. Hardware capacity, logical/physical-core effects, setup time,
identity and current price remain unverified.

The L4 columns are sensitivity bounds, not L4 measurements: 12 logical vCPUs /
6 assumed physical cores, historical 3.7x host slowdown, GPU service time equal
to measured Mac MPS B8 time. Lower bound fully overlaps host and GPU service;
upper bound adds them. The host estimate subtracts L800 MPS forward cost from
the L800/L1500 mean total. An implementation that shares the GPU across workers
is not supplied here. `SPEED_FORECAST.json` also exposes 0.5x and 2x MPS GPU-service
scenarios. The historical rate is 2 units/h. Add setup/export and a 20% reserve
before any separately authorized scheduling; rates are dated planning inputs,
not live quotations. No remote operation was performed.

## Planner cost and proposals for Claude

Equal-mix, L800/L1500 forced-search cost is
1296.1 ms CPU
(31.6 times a 41 ms tick) or
1079.6 ms MPS
(26.3 times the tick).
Amortized at the old 43.395% search fraction, search alone still averages
562.4 / 468.5 ms per decision.
These are descriptive fixture medians, not paced serving measurements or p95.

No alternative was adopted. Both proposals change the search policy and require
prospective approval and fresh evidence:

- **N6/H96:** 56.25% of registered branch frames. A linear-work sensitivity gives
  18.47 Mac Phase 0 hours and
  72.55 hours for 1M searched decisions.
  This is unmeasured: early deaths, input reuse and fixed overhead make scaling nonlinear.
- **L>=800 gate:** bare calibration admits 22.016% instead
  of 43.395%, projecting 149,952 Phase 0 searches and
  17.52 Mac hours at current search cost. It does
  not reduce the cost of one million *searched* decisions. Search changes visitation,
  so the new counts are a scenario rather than a guaranteed saving.

## Verification and reproduction

The original verification selection passes 31 tests (88 deselected). The new
suite passes 24 tests, plus the final focused fallback addition passes once:
25 distinct new cases, including all 18 CPU/MPS oracle fixtures. Chosen actions
match and score absolute error is <=1e-3. CPU determinism, prepared-source
immutability, signed-zero input reconstruction, near-Q fallback, exact ties
and final-score fallback are covered. Independent read-only review found no
open correctness issue in the final rollout or input-reuse paths. Black,
isort, flake8 and `git diff --check` pass. Complete commands/hashes are recorded
in `SPEED_VERIFICATION.json`. This is fixture qualification, not universal
float identity, scientific efficacy or serving qualification.

A real two-world CLI smoke (H8, N1/H2) completed serially and with two workers.
The merged files are byte-identical, SHA-256
`075faaef06994259ffd5bbff6f13acf3b785f7dd3d1780c04bbf4568eacb5dc4`.
Its create-only plan and raw results remain within this worktree under
`docs/agent_notes/phase0-speed-smoke-v1/`; the verification receipt records
its plan/source identity. These are smoke worlds, not selection outcomes.

Use a fresh output path for every invocation. The profile, timing and forecast
writers refuse existing outputs. Representative same-protocol command:

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-search-teacher
nice -n 10 ./venv/bin/python -m research.phase0_search_20261019.preflight \
  --bank research/phase0_search_20261019/WORLD_BANK.json \
  --config research/phase0_search_20261019/config.json --mode timing --rounds 3 \
  --out docs/agent_notes/phase0-speed-repeat/timing-cpu.json
```

The forecast is executable via `python -m research.phase0_search_20261019.forecast_speed`;
its receipt pins all six input files and the producing script hash. Neither the
original `PREFLIGHT.json` nor any registration, bank or prior verification receipt
was edited. The colab-safety worktree, Colab, RunPod, tmux, serving, released
checkpoints and serving pins were untouched.
