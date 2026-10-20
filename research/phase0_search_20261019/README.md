# Phase 0 search teacher

Implementation of `docs/research/specs/2026-10-19_phase0_search_teacher.md`.
The pre-registration is a draft. No selection episodes have been launched by
this implementation task. All measured data are disjoint preflight evidence.

`teacher.py` takes the world constructor from T3's captured strict evaluator
and retains its terminal-hero GridBatchSim, Watch preparation order and carried
Vector61 opponent state. Search evaluates every legal first action. The bare B8
outer action retains rowwise advisory-mask inference and no veto. The default
CPU continuation path builds one numba observation batch and makes one B8
batch forward per step, reusing exactly byte-identical neural inputs.
Worlds and epsilon RNG streams remain independent. Frozen opponents retain
shared batched policies. `SearchTeacher(config, implementation="reference")`
selects the preserved old scorer; `device="mps"` or `device="cuda"` explicitly
selects accelerator B8 forwards, retaining the CPU oracle for near ties.
Batch Q values are not claimed bit-identical to rowwise inference.

The speed follow-up is documented in [SPEED_REPORT.md](SPEED_REPORT.md), with
exclusive category profiles in [PROFILE.md](PROFILE.md) and reproducible
conditional forecasts in `SPEED_FORECAST.json`. The original preflight below
is historical; its immutable receipt and registered values are preserved.

The score is an undiscounted sum of realized post-transition logical mass for H
frames, zero after death, plus `leaf_scale * max_legal Q_B8` at the next prepared
leaf decision. Default leaf scale is **1 mass-frame per Q unit**. This is an
explicit heuristic conversion, not a calibrated estimate of future mass; B8 Q
was trained against a different reward. Reported Phase 0 endpoints always use
realized H5000 mass with no leaf estimate in the outcome.

`bank.py` allocates all purposes under separate keys and collision-audits every
world before gameplay. Selection has 96 scripted / 32 frozen / 32 mixed worlds;
calibration, timing, parity and smoke worlds never enter the decision estimator.
`analyze.py` applies the complete-bank one-look decision, equal weights for each
mix, paired world-level uncertainty, and STOP before GO if clauses overlap.

`runner.py` binds a create-only plan and immutable per-world paired partials,
verifies resumes and merges in bank order. Local execution uses one math thread
per process, at most six workers, AC/open lid and thermal checks. A scientific
verdict requires the complete selection bank; missing or invalid evidence cannot
be silently omitted. Run outputs belong outside Git.

`parity.py` compares a reserved replay subset and records float32 tie evidence.
Same-platform serial/parallel identity is exact. Cross-platform differences must
be explained at identical pre-states; a small Q tie does not justify ignoring
arbitrary later trajectory changes. Actual CPU-pod parity and throughput remain
unverified until separately authorized operation.

## Measured preflight (Mac CPU, 2026-10-19)

Chosen recipe: **N=8, H=128, epsilon=0.02, legal-Q margin threshold=0,
length gate=600, leaf scale=1**. N/H use the smallest suggested setting because
one search already costs seconds. Epsilon is fixed prospectively, not selected
against efficacy. No larger recipe or selection-world pilot was run.

Seconds per forced search, including replica construction, features, inference,
rollouts and leaf evaluation; six legal first actions in every fixture:

| Hero length | Scripted | Frozen | Mixed |
|---|---:|---:|---:|
| 200 | 4.711 | 4.844 | 4.837 |
| 800 | 4.210 | 4.743 | 5.405 |
| 1500 | 4.127 | 4.505 | 4.311 |

These are valid **synthetic serpentine bodies with real strict-roster policies**,
not a sample of naturally reached mature states. Early branch deaths reduce
cost. This is a timing premise test, not a search-efficacy result.

Six identical L1500 mixed searches, including process/model startup:

| Workers (one math thread each) | Batch wall seconds | Speedup |
|---|---:|---:|
| 1 | 27.486 | 1.00x |
| 2 | 16.423 | 1.67x |
| 6 | 7.240 | 3.80x |

Three fresh, disjoint bare-B8 H5000 calibration worlds produced 11,446 alive
hero decisions in 20.37 s. The mandatory L>=600 trigger alone covers **43.395%**
of those decisions. Threshold zero disables additional Q-margin admissions;
the OR gate cannot produce 10% total search while preserving the required
length trigger. This is an explicit spec feasibility limit, not a claimed
10% calibration success. One world per mix is a weak visitation estimate.

At 41 ms per serving tick, a searched decision needs **101-132 tick budgets**.
The measured planner is unsuitable for an every-tick serving path. No serving
code or planner optimization was attempted.

## Resource scenarios, not an approved run

The calibration extrapolation gives 276,256 searched decisions across 160
selection worlds. Multiply each mix's mandatory decisions/world by its L800 /
L1500 mean search cost, add two bare-B8 passes, and use measured worker scaling:

| Placement / assumption | Single-worker hours on that CPU | Wall hours | Rental cost |
|---|---:|---:|---:|
| Mac, one worker | 337.34 | 337.34 | $0 cloud rental |
| Mac, six workers, measured 3.80x scaling | 337.34 | 88.85 | $0 cloud rental |
| 32 vCPU, assumed 2.4x slower per core | 809.62 | 25.30 | $22.52-$24.04 |
| 32 vCPU, ideal equal-speed cores | 337.34 | 10.54 | hypothetical only |

The six-worker Mac scenario occupies about **533 aggregate worker-hours**;
337 hours is the serial-equivalent work, not a claim of 337 CPU-hours under
contention. A 20% reserve raises its wall forecast to **106.63 h**. If every
selection decision searches, the measured maximum cost gives a **1,201
serial-hour scenario**; this is not a certified upper bound. Search can change
survival/visitation, so the calibration forecast is not a cap guarantee.

The pod price range is a **dated planning assumption**, from
`docs/research/t4_pod_plan_2026-10-15.md`, not a current quote. The 2.4x per-core
slowdown is historical actor evidence, not a measurement of this teacher.
Setup, transfer, provisioning, CPU model/SMT effects and current stock/prices
remain unverified. The 2.4x scenario is only **3.51x faster** than the measured
six-worker Mac, before setup; it does **not establish the >=4x placement rule**.
No remote allocation or execution is authorized here. Mac costs exclude power,
hardware depreciation and opportunity cost.

`PREFLIGHT.json` contains the raw measurements, input/source hashes, formulas,
scenario assumptions and limitations. Original create-only receipts live in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/`.

## Verification

The 172-world allocation audit passed with zero historical seed collisions over
509 bounded scan partitions. Coverage is visible research/docs/scripts text,
artifact JSON/JSONL and bound historical exclusion/manifests; opaque or unrecorded
external experiments remain outside that boundary.

The focused suite has **31 passed, 88 deselected**, covering the teacher,
runner, analysis/collision rules, T4 snapshots, long-body Grid parity and direct
live-reference parity. Black/isort/flake8 and `git diff --check` pass.

Actual CLI smoke at implementation commit `7b244508b79cbac385d814bb796b2e1974cd96c1`:
two disjoint worlds, H8, N1/H2 forced search, serial vs two workers; merged file
bytes are identical (SHA-256
`632ae099dcd0a4069cc025fa6509f874223780b2b68f9993cd7c0a7ef0e7c547`).
Completed-run resume reused immutable partials. Draft selection preparation was
refused; analyzing the smoke returned `INVALID_INCOMPLETE`.

The four reserved parity worlds, also H8/N1/H2, passed Mac serial/two-worker
comparison 4/4. This verifies the local replay/comparator path, not x86 identity.
Evidence: `proof-v1/` and `parity-v1/` under the artifact root above. Near-tie
cross-platform differences fail closed pending a separately verified forced
replay; this package does not auto-waive ties or implement a remote allocator.

## Operator commands

Run from this worktree; every output is create-only. Use a fresh attempt suffix
when inputs or code change. Resuming the same plan requires unchanged source,
Git HEAD, checkpoint and registered input bytes. Do not regenerate the frozen
bank or edit an old receipt.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-search-teacher
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
```

A repeatable, bounded two-world smoke on current source (no approval required):

```bash
./venv/bin/python -m research.phase0_search_20261019.runner prepare --smoke \
  --bank research/phase0_search_20261019/WORLD_BANK.json \
  --audit research/phase0_search_20261019/SEED_AUDIT.json \
  --config research/phase0_search_20261019/smoke_config.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/plan.json
./venv/bin/python -m research.phase0_search_20261019.runner run \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/plan.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/results \
  --workers 2 --wall-seconds 120
./venv/bin/python -m research.phase0_search_20261019.runner merge \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/plan.json \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/results \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/smoke-review-v1/merged.json
```

**Selection commands remain blocked while the preregistration is DRAFT.** After
Claude approves the exact protocol and resource envelope, the proposed Mac cap
is 120 wall hours (432,000 s), six single-thread workers, first cap wins. This
covers the central forecast plus 20% reserve but cannot guarantee completion.
A cap stop is incomplete, with no scientific verdict or automatic retry.

```bash
./venv/bin/python -m research.phase0_search_20261019.runner prepare \
  --bank research/phase0_search_20261019/WORLD_BANK.json \
  --audit research/phase0_search_20261019/SEED_AUDIT.json \
  --config research/phase0_search_20261019/config.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/plan.json
./venv/bin/python -m research.phase0_search_20261019.runner run \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/plan.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/results \
  --workers 6 --wall-seconds 432000
./venv/bin/python -m research.phase0_search_20261019.runner merge \
  --plan /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/plan.json \
  --root /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/results \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/merged.json
./venv/bin/python -m research.phase0_search_20261019.analyze \
  /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/merged.json \
  --out /Users/josenunez/Projects/ml/snake-dqn-artifacts/phase0-search-20261019/selection-v1/decision.json
```

For later Linux portability checking only, stage hash-verified copies of B8 and
the four strict pool checkpoints and the FRP3 carrier checkpoint, check out the same commit, and set
`SNAKE_CKPT_DIR` to that input directory. From the repo root:

```bash
export SNAKE_CKPT_DIR=/verified/inputs SNAKE_DQN_DEVICE=cpu
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python -m research.phase0_search_20261019.runner prepare --parity \
  --input-root /verified/inputs \
  --bank research/phase0_search_20261019/WORLD_BANK.json \
  --audit research/phase0_search_20261019/SEED_AUDIT.json \
  --config research/phase0_search_20261019/smoke_config.json \
  --out /fresh/parity/plan.json
python -m research.phase0_search_20261019.runner run \
  --plan /fresh/parity/plan.json --out /fresh/parity/results \
  --platform cpu-pod --workers 4 --wall-seconds 120
python -m research.phase0_search_20261019.runner merge \
  --plan /fresh/parity/plan.json --root /fresh/parity/results --out /fresh/parity/merged.json
python -m research.phase0_search_20261019.parity \
  --mac /verified/mac-reference-merged.json --other /fresh/parity/merged.json \
  --out /fresh/parity/comparison.json
```

Create a current-source Mac reference with the same `prepare --parity` command
and config before moving the reference bytes. Future Linux selection would use
`--platform cpu-pod --workers 32`; its duration, spend cap and >=4x admission
remain a separate owner decision. No allocation commands are supplied or run.
All guards fail closed; a running search may finish its current bounded decision
before a wall/thermal check, giving about one search's observed latency of
overshoot. No tmux session, Colab runtime, pod, checkpoint, pin or serving file
was modified by this task.
