# MOVE sprint handoff — INVALID CPU exactness, zero GPU spend

**Verdict: INVALID; dispatch blocked.** The opt-in indexed MOVE implementation
matches the frozen device implementation on the recorded CPU states, but the
existing device engine does not meet the owner's independent CPU-byte contract.
The requested mismatch kill rule stopped performance work before any allocation.
Do not dispatch this candidate, retry its scientific failure, or call it a GPU
speed result. No serving or policy change is made by this packet.

## Where and identities

Authoring worktree `/Users/josenunez/Projects/ml/snake-dqn-gpu-move`, branch
`t34-gpu-move-20261023`. Implementation `627a4a3a` and frozen-runtime commit below
were merged and pushed to origin/main (integration `bf5463fc4197df6adf0ead7a543e70f8872137b0`
before this handoff-only commit).

Frozen checkout `/Users/josenunez/Projects/ml/snake-dqn-gpu-move-frozen-01`, branch
`t34-gpu-move-runtime-20261023`, **source pin
`866fa48e0e821608239e08f7c356936cc2ac943c`**. Its parent is the requested frozen
transport pin `313758255ee5b9f1422a0202c14a80cc32554857`; qualified transport tree
`d0d92bdb572b770284a1fea6152417ffa5f06bea` is unchanged. A new source pin is
necessary to contain new MOVE code. Keep the frozen checkout unmodified.

Artifacts:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-01`.
`VERIFICATION.json` binds source, review, preflight and refusal evidence.

## What changed

- `device_sim.py`: `move_backend="reference"` default and opt-in `indexed` hooks;
  dense reference extraction; lazy corpse-body materialization.
- `move_cuda.{cu,cpp,py}`: single occupancy build only for ready respawn worlds;
  all 32 counter advances preserved; valid ring inspections run in parallel;
  ordered per-world respawn/collision resolver. CPU portability twin is separate
  from untested native CUDA execution. No fast math.
- `fixtures/move/device_sim_pre_move.py`: independent simulator from `cd26df50`,
  SHA `816ebd5fbd4016bb01eda5fe31ee7739f8cb7bf021b2885684f769ea63dc58ae`.
- `move_bench.py`, `move_bench.sh`, `move_requirements.txt`: three-roster CPU gate
  before E64/E256/E1024 stage A/B; recorded state/config/actions; plain median
  step timing, separate respawn/collision diagnostics, scientific stop rules.
- `move_units.py`, `move_pack.py`, `move_preflight.py`, `move_analyze.py`: immutable
  payload/SHA/completion publication, verified resume inventory, exact-source
  preflight, current-pin canary binding, templates explicitly non-admitted,
  INVALID proof and dispatch refusal preserved.
- `test_move_cuda.py`, `test_move_packet.py`: frozen parity, RNG overflow,
  no-ready/ready/crowded/duplicate placement, wrapped and 800–1500 segment rings,
  ordered collisions, float32 size ratios, CPU event cases, resume corruption
  and pre-operational dispatch refusal.
- `MOVE_PROTOCOL_20261023.md`: scope, comparator, caps, exactness blockers and stop.

## Measured proof

The real Bash-entrypoint Mac preflight took **11.971054083027411 s**, exit 2,
`INVALID: MOVE_CPU_GRID_MISMATCH`. Three E1 recorded states, six snakes each,
with 800/900/1000/1100/1300/1500 body lengths and full 10,400 capacity:

| Roster | Candidate vs frozen device | Independent GridBatchSim |
|---|---|---|
| scripted | all 28 tensor-state fields byte-identical | reward bytes differ |
| frozen | all 28 tensor-state fields byte-identical | reward, ordered food, corpse flags differ |
| mixed | all 28 tensor-state fields byte-identical | reward, ordered food, corpse flags differ |

Grid reward is float64, device reward float32. At length 800 the values are
`-0.7999999999999972` and `-0.8000030517578125`. The prepared seam disables
respawn; CPU food replacement nevertheless consumes RNG in frozen/mixed.
The device counter RNG and CPU MT are different generators, and the device's
non-training food replacement behavior also differs. These inherited differences
are recorded, never normalized or waived. This action-tape check is not
closed-loop CPU-policy identity. Native A100 compilation, CUDA parity and speed
are unverified. No per-stage A100 numbers or 2x result exist for this candidate.

CPU receipt: `mac-preflight/runtime/output/cpu-parity.json` (and replay NPZ files).
Terminal result: `mac-preflight/runtime/output/move-bench.json`.
Mac preflight: `mac-preflight/mac-preflight.json`.
Independent Mac reducer: `analysis-invalid.json`.
Admission refusal: `packet/DISPATCH_REFUSED.json`.
Actual wrapper refusal: `dispatch-refusal.log`, `scientific_admission=false`;
no state directory was created. **Allocated wall: 0; Colab debit: 0 units.**

## Verification commands

All numerical tests used this environment, serially at nice 15:

```bash
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1 NUMBA_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
nice -n 15 "$PY" -m pytest -q research/gpu_sim_spike/test_move_cuda.py
# Initial: 18 passed/14 skipped; after ratio/long-body additions: 21 passed/17 skipped.
nice -n 15 "$PY" -m pytest -q research/gpu_sim_spike/test_device_sim.py \
  research/gpu_sim_spike/test_a0_event_oracle.py research/gpu_sim_spike/test_cuda_shape.py
# 25 passed, 23.66 s.
nice -n 15 "$PY" -m pytest -q research/gpu_sim_spike/test_move_packet.py \
  research/gpu_sim_spike/test_move_cuda.py
# 31 passed/17 skipped, 1.76 s (before the final three admission tests).
nice -n 15 "$PY" -m pytest -q research/gpu_sim_spike \
  tests/test_colab_cli_canary.py tests/test_colab_cli_schema_failures.py \
  tests/test_ops_dispatch.py
# 348 passed/21 CUDA-only skips, 103.50 s.
nice -n 15 "$PY" -m pytest -q research/gpu_sim_spike/test_move_packet.py
# Final 13 passed, 0.15 s, including the three added admission tests.
nice -n 15 "$PY" -m black --check --line-length 100 \
  research/gpu_sim_spike/device_sim.py research/gpu_sim_spike/move_*.py \
  research/gpu_sim_spike/test_move*.py
nice -n 15 "$PY" -m isort --check-only --profile black \
  research/gpu_sim_spike/device_sim.py research/gpu_sim_spike/move_*.py \
  research/gpu_sim_spike/test_move*.py
nice -n 15 "$PY" -m flake8 research/gpu_sim_spike/device_sim.py \
  research/gpu_sim_spike/move_*.py research/gpu_sim_spike/test_move*.py
bash -n research/gpu_sim_spike/move_bench.sh
git diff --check
```

Black/isort/flake8/Bash/diff checks passed. Two Codex-V LIGHT review findings
(template admission and dirty-source preflight) were fixed, independently
rechecked, and covered by tests; no remaining finding in that bounded scope.
This is static/CPU acceptance only.

## Exact preparation and refused-dispatch commands

Already executed, create-only output paths; do not rerun preflight in place:

```bash
REPO=/Users/josenunez/Projects/ml/snake-dqn-gpu-move-frozen-01
PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
ART=/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/gpu-move-a100-20261023-01
PIN=866fa48e0e821608239e08f7c356936cc2ac943c
export PATH="/Users/josenunez/Projects/ml/snake-dqn/venv/bin:$PATH"
cd "$REPO"
nice -n 15 "$PY" -m research.gpu_sim_spike.move_preflight \
  --commit "$PIN" --root "$ART/mac-preflight"
# exit 2, INVALID, 11.971054083027411 s.
nice -n 15 "$PY" -m research.gpu_sim_spike.move_pack \
  --commit "$PIN" --artifact-dir "$ART/packet" --templates-only
# Two templates; scientific_admission=false, dispatchable=false.
nice -n 15 "$PY" -m research.gpu_sim_spike.move_pack \
  --commit "$PIN" --artifact-dir "$ART/packet" \
  --preflight "$ART/mac-preflight/mac-preflight.json" --validate
# exit 1, MOVE_CPU_EXACTNESS_REQUIRED; no runnable manifest created.
nice -n 15 "$PY" -m research.gpu_sim_spike.move_analyze --commit "$PIN" \
  --exports "$ART/mac-preflight/runtime/output" --output "$ART/analysis-invalid.json"
# INVALID. This is the preserved terminal look, not a performance pass.
```

Exact wrapper refusal command (already executed; no allocation):

```bash
nice -n 15 "$REPO/tools/ops/dispatch.sh" move-a100-refusal-866fa48e \
  "$ART/packet/move-bench-canary-template.json" --broker "$REPO" --cwd "$REPO" \
  --state-dir "$ART/state-refusal" --dispatch-txt "$ART/no-dispatch-instructions.txt"
# exit 1: REFUSE: manifest scientific_admission=false
```

There is **no authorized runnable dispatch command for this pin**. The planned
90-minute forecast is unmeasured native time; base cap 108 minutes/18 u including
3 u reserve, effective 216 minutes/36 u including 6 u reserve. Canary/main slices
are 24 minutes/4 u and 192 minutes/32 u. These templates do not reserve or spend
anything; existing round-28 Q4/global caps and original deadlines still dominate.

## Remaining work and preservation

The failing CPU contract needs a separately declared correctness repair and a
new source identity before any native performance attempt. No 2x speed or
2,000/s claim, strict evidence, label admission, or release is established.
No further tuning or scientific retry was performed after the mismatch.

No existing tmux/process was killed or modified. Released checkpoints, serving
pins, prior hash-bound artifacts, FLOOD/V8 code and supervisor transport were not
authored or changed. Unrelated origin/main changes were merged unchanged.
