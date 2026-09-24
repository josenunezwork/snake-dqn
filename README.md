# snake-dqn

Snake DQN is a local reinforcement-learning project with a live FastAPI/React
snake game, an incumbent distributed Apex DQN policy, and an experimental
raster/PQN path. The incumbent stays in service while repairs and experiments
produce evidence; no command in this repository silently replaces it.

![architecture](docs/architecture.svg)

![python](https://img.shields.io/badge/python-3.12-blue)
![tests](https://img.shields.io/badge/tests-pytest-brightgreen)
![code style](https://img.shields.io/badge/code%20style-black-000000)
![license](https://img.shields.io/badge/license-MIT-lightgrey)

## What is established, and what is not

The deployed reference is the 61-feature free-space Apex checkpoint. Its earlier
evidence used a pre-repair, alive-conditioned, narrow-opponent gate, so it is
useful historical evidence rather than proof under the repaired deployment
task. The raster/PQN route is a plausible engineering direction, not a promoted
replacement or an algorithm winner.

The history also has an important correction: GRU/DRQN was trained and lost the
older frozen-opponent trials, but the frequently repeated claim that CNNs were
trained and lost is false. Git history found CNN code but no CNN training run or
checkpoint. Those older trials also predate the repaired evaluation work. Read
[the dated findings record](docs/project_history_and_findings.md) and the
[redesign blueprint](docs/ml_redesign_blueprint_2026-07.md) before making a
model-selection claim.

## Quick start

This section is for a developer who has Python, Node.js, and a local checkout.
Use the project virtual environment for every Python command.

```bash
python3 -m venv venv
./venv/bin/python -m pip install -r requirements.txt

# Build and serve the browser game at http://localhost:8000.
cd web/frontend && npm install && npm run build && cd ../..
./venv/bin/python web/serve.py

# Apex is the incumbent training stack. This is a headless training command,
# not an evaluation or promotion command.
./venv/bin/python src/main.py --headless --episodes 100000
```

The browser renders state supplied by the Python server. The **Play** tab lets a
human steer a snake; the game, scoring, inference, and the SQLite leaderboard
remain server-side. See [web/README.md](web/README.md) for its interface.

## Evaluation has two different purposes

`src/scripts/tournament_eval.py` keeps diagnostic comparisons separate from
promotion authority. Do not use a diagnostic exit code as a release decision.

### Legacy diagnostics and inexpensive screens

The default diagnostic compares paired world seeds, reports mass integral (dead
frames contribute zero), and can retain content-addressed input snapshots. Its
old `--gate` status is only a convenient diagnostic signal: an exit code of zero
does **not** promote a checkpoint.

This fully scripted command is a copyable plumbing check. It needs no saved
model, writes the comparison JSON, and places a snapshot receipt beside the
immutable inputs. Replace `scripted:random_safe` with a candidate only after
selecting compatible baseline and opponents.

```bash
SNAKE_DQN_DEVICE=cpu ./venv/bin/python src/scripts/tournament_eval.py \
  scripted:random_safe \
  --baseline scripted:greedy_food \
  --opponents scripted:greedy_food \
  --engine live \
  --evaluation-profile legacy-diagnostic \
  --frames 3000 --seeds 0-9 \
  --snapshot-dir runs/evaluation_artifacts/readme-diagnostic \
  --json-output runs/evaluation_artifacts/readme-diagnostic/results.json
```

`results.json` records `authority: "diagnostic-only"`; the adjacent evaluation
input receipt records the config and checkpoint or scripted-agent identities
used for that run. Add `--gate` only when automation needs the legacy
diagnostic exit status. It still writes diagnostic-only output.

`--engine simd` is useful for a raster or scripted **screen** under a declared
profile. It is never a promotion path. In particular, a SIMD result cannot
replace a live serving-path result, even when it has the same seeds or metric.
Use `--engine live` for vector61 checkpoints; BatchSim only accepts compatible
raster or scripted agents.

### Strict promotion authority

Strict promotion is a separate, live-only operation supplied by the E2 repair.
It starts from a frozen request and derives its engine, roster, seeds, world,
and output locations from evidence; it rejects diagnostic candidates, `--gate`,
`--pilot`, and diagnostic overrides. The request binds immutable candidate,
incumbent, opponent, source, and serving evidence before final worlds run.

The final profile has 5,000 scored frames and 5,000 observation-progress frames.
It requires at least 40 fresh paired worlds, or the larger count from the paired
pilot; three predeclared mixes; Holm-corrected superiority on at least two
mixes; and a scripted-mix one-sided noninferiority bound against a predeclared
absolute margin. The final receipt also rechecks readiness and provenance.

The execution protocol supplies the five evidence inputs below and selects a
new final-receipt path. The inputs must describe the same frozen candidate and
deployment profile; diagnostic artifacts are not interchangeable with them.

```bash
./venv/bin/python src/scripts/tournament_eval.py \
  --strict-promotion-request artifacts/strict/request.json \
  --strict-promotion-receipt artifacts/strict/final-receipt.json \
  --strict-e0-receipt artifacts/strict/e0-receipt.json \
  --strict-pilot-artifact artifacts/strict/paired-pilot.json \
  --strict-calibration-artifact artifacts/strict/calibration.json \
  --strict-serving-bundle artifacts/strict/serving-bundle.json
```

A passing final receipt is evidence for a later, explicit release operation. It
does not copy a checkpoint, change a default, publish a service, or replace the
incumbent by itself. The details and handoff requirements are in the
[RL contract-repair plan](docs/plans/rl_repair_2026-09-06/README.md) and its
[agent execution runbook](docs/plans/rl_repair_2026-09-06/agent_runbook.md).
The [repair execution status](docs/experiments/rl_repair_execution_2026-09-08.md)
records accepted packages and pending G0/X0 evidence; it is not a promotion
result.

## Corrected PQN diagnostics

The corrected recipe is opt-in. It records `raster31v3` and a
`corrected-v3` recipe; it does not reinterpret legacy observations. Existing
`vector61` and `raster31v2` paths remain for compatible inference and archived
reproduction. A checkpoint with missing metadata is legacy compatibility input,
not verified v3 evidence.

`pqn_correctness_diagnostic.py` is a bounded, non-promoting runner. It freezes
a manifest, retains initial and final checkpoints plus terminal receipts, and
evaluates only after its learner process exits. The modes are intentionally
different:

```bash
# Tiny CPU smoke: one environment and a short, explicit positive budget.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  ./venv/bin/python src/scripts/pqn_correctness_diagnostic.py \
  --mode smoke --device cpu --out-dir runs/pqn_h0_smoke --seed 1000 --total-steps 32

# G0-style operational shakedown: the mode requires exactly 100,000 steps.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  ./venv/bin/python src/scripts/pqn_correctness_diagnostic.py \
  --mode shakedown --device mps --out-dir runs/pqn_h0_shakedown \
  --seed 1001 --total-steps 100000

# Fixed-budget screen: the mode requires exactly 500,000 steps and a compatible
# shakedown receipt from the same source, device, and rollout geometry.
OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 VECLIB_MAXIMUM_THREADS=2 \
  ./venv/bin/python src/scripts/pqn_correctness_diagnostic.py \
  --mode screen --device mps --out-dir runs/pqn_h0_screen --seed 1002 \
  --total-steps 500000 \
  --shakedown-receipt runs/pqn_h0_shakedown/shakedown_projection_receipt.json
```

The small smoke training budget does not shorten evaluation. Smoke and
shakedown each evaluate their initial and final checkpoints for the full
5,000-frame profile on one world against the scripted mix.

Each output directory contains an immutable manifest, per-arm terminal status,
checkpoint-lineage receipts, and evaluation artifacts. `completed` means that
the bounded diagnostic completed its stated work; it never means that a model
was promoted. A tripwire, resource stop, source/protocol drift, or failed
evaluation remains an artifact to inspect, not a reason to extend or retry a
run with changed defaults.

The Mac resource limits are deliberate: each H0 invocation runs one learner and
its worker uses two native CPU threads, including a CPU smoke. Do not start a
test suite, build, corpus job, second learner, or evaluation while an MPS learner
runs; evaluate only after it exits. Initial fail-closed limits are 4 GiB process
RSS, 8 GiB MPS-driver allocation, and a 6 GiB system-memory reserve. RSS and
MPS allocation overlap on unified memory and must not be added together. The
runbook owns resource slots and requires retained logs, telemetry, and
checkpoints when a cap is reached.

## Architecture

### Apex DQN: the incumbent

The incumbent path is distributed Apex DQN: CPU actors with varied exploration
feed a SumTree-backed prioritized buffer and n-step returns to one learner,
which broadcasts weights back to the actors. `ApexNetwork` is a feedforward
dueling network over the hand-crafted 58-D or free-space 61-D vector.

### Raster/PQN: an experimental path

The redesign uses `BatchSim`, an ego-centred dual-scale raster observation, and
a synchronous replay-free PQN Q(lambda) learner. Its raster network is separate
from the Apex policy hierarchy. The simulator and raster make efficient batched
experiments possible, but neither establishes live-product parity nor policy
quality without the relevant evidence.

The current repair keeps the legacy sampler available for reproduction and makes
full-epoch sampling opt-in. The prior five-seed sampler comparison was
inconclusive, so it does not justify changing a default. The repair also does
not prescribe a larger convolutional model, recurrence, attention, or a new
optimizer before semantics and provenance are qualified.

## Development

```bash
make install-dev
make test
make lint
cd web/frontend && npm test
```

For repair work, follow the runbook rather than running broad commands in
parallel: small CPU checks use one native numerical thread each, and the full
suite runs alone with at most two pytest workers. `make test-fast` uses
`-n auto` and is not part of that execution protocol.

## References

- [Algorithm documentation](docs/ml_algorithm.md) — Apex DQN details and its
  historical context.
- [Project history and durable findings](docs/project_history_and_findings.md)
  — dated results, corrected historical claims, and provenance.
- [RL research portfolio](docs/experiments/rl_research_portfolio_2026-09-12/README.md)
  — experiment ledger, retained failures, measured outcomes, and parallel research waves.
- [Living-mass reward experiment](docs/experiments/mass_objective_2026-09-13/README.md)
  — four bounded training arms; a promising first gain reversed on the fresh seed.
- [PQN decision-phase contract](docs/pqn_decision_phase_2026-09-12.md)
  — opt-in training/evaluation alignment and its CPU/MPS qualification.
- [SIMD environment contract](docs/simd_env_spec.md) — documented simulator
  dynamics contract.
- [RL contract-repair plan](docs/plans/rl_repair_2026-09-06/README.md) —
  versioned repair decisions, evidence gates, and experiment sequence.
- [Repair execution status](docs/experiments/rl_repair_execution_2026-09-08.md)
  — accepted-package receipts and pending qualification evidence.
- [Ape-X](https://arxiv.org/abs/1803.00933) ·
  [Double DQN](https://arxiv.org/abs/1509.06461) ·
  [Dueling Networks](https://arxiv.org/abs/1511.06581) ·
  [PER](https://arxiv.org/abs/1511.05952)

## License

MIT
