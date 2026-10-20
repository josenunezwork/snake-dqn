# T34 ExIt-B8 engine preparation

Implements P1, the P2/A5 entrypoint, P3/A2 replay tooling, and P6 labelling preparation from
`docs/research/specs/2026-10-19_exit_engine_prep.md` and the binding
`docs/research/final_run_20261019/FINAL_PLAN.md`. This package has no serving or promotion authority.
Preparation was performed on the Mac only. No Colab allocation, pod, tmux session, or paid service was
started or operated by this implementation task.

## Acceptance status

| Item | Implemented / locally verified | Still requires the operator |
|---|---|---|
| P1 teacher | Deterministic VERIFY then failed-root EXPAND; legal alternatives, advisory student, H300 default; separate fp32 device copies; v8 frozen opponents; physical compaction; at most two rewinds; configurable triggers/beta | No research outcome or candidate-quality claim |
| A1 identity | Real scripted/frozen/mixed exports byte-identical in two runs in each of two spawned processes; forced-death compaction preserves source/RNG/runtime state | Longer studies and cross-runtime identity remain separate |
| P2 / A5 | Same resident-root collector as labels, W256 capacity per worker, two workers × five Numba threads per L4 lane; measured work/expansion/RAM/export/throughput fields; bounded launcher and lane aggregate | Actual two-L4-lane rehearsal; tower adapter from sibling P5; ≥3,000 active row-steps/s **per L4 lane** |
| P3 / A2 | Six-action replay, initial/every-250/final state hashes, B8 raster regeneration, whole-world rejection, strict CLI exit on any mismatch | ≥150 worlds across all three mixes: L4→Mac and L4→second L4; zero unexplained mismatches |
| P6 benchmark A6 | Final `bench.sh`, CPU/one thread, two worlds ×500 frames, H300: 24.410 s; earlier isolated run 19.305 s, 6,300 active row-steps, 91,730 export bytes | CUDA and 2×5-thread execution have not run here |
| P6 label A6 | Final `label.sh`, CPU/one thread, two worlds ×500 frames, H300: 19.280 s; earlier isolated run 12.023 s; local benchmark replay accepted 2/2 worlds, zero mismatches | Campaign bank audit, P7 preregistration, owner pre-decisions, and round funding before labels are dispatched |

Local evidence is under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-engine-prep-20261019-v1/`.
`VERIFICATION.json` records final test/lint commands and source hashes. The Mac timings above include
startup/JIT and describe tiny preflight fixtures; final entrypoints ran concurrently at one CPU thread
each. They do not predict L4 performance or full-H5000 cost.
T3 dense receipts were not re-scored: the new roots/rosters are not their receipt bank, so there is no
like-for-like historical agreement percentage. This comparison is descriptive, never an A1 gate.

## Engine, trajectory and export contracts

`ExitTeacher.choose(worlds)` takes prepared T3/T33 captured worlds with identical student weights.
It returns a-star labels and diagnostics. Collection applies beta (1, 0.5, or 0) using a separate
seed/decision hash stream; teacher search never consumes the world's food RNG. VERIFY uses one physical
row per root. Only failures expand; the other legal actions run in one batch. A-star is highest-student-Q
among survivors, then longest-lived if none survive; ties prefer the student, then the lowest action.
Forbidden actions are tested deaths only when a-star survives. Unsearched alternatives remain unknown.
`active_row_steps` counts every executed row transition, including the fatal transition;
`resident_row_steps` counts the physical batch rows at those transitions. They coincide after
compaction. `nominal_row_steps` records the un-compacted requested rows × H for comparison.

`run_worlds` retains up to W256 roots per process and batches their simultaneous triggers. A partial
chunk exports completed worlds only; its report has `complete=false`. Active-work counters include
unfinished roots. Per-mix search/expansion counters update when searches finish, independently of world
completion. `worlds_per_hour` and export bytes/world describe completed terminal-or-horizon worlds;
unfinished worlds are censored. Zero completed worlds means no measured complete-world/export rate,
and must not be used to size a label chunk. A5's lane go rule is printed, never an automatic dispatch.

The 150-decision death-window buffer is one lagging prepared snapshot plus the six-action trace.
Advancing that snapshot with recorded actions keeps memory proportional to root count. A death rewinds
to its window start, deletes the abandoned suffix's actions/labels/hashes, and searches every decision
through the original death point. At most two rewinds are permitted per world. Telemetry includes
discarded-suffix work; exports contain the final replayable trajectory.

Each `world-*.json` is immutable and has `.sha256` and `.complete` sidecars compatible with the existing
Colab supervisor. It contains the seed, portable roster/v8 configuration, six-snake action trace, RNG
stream identifiers, initial/every-250/final hashes, and searched labels. The six outcome slots include a
`tested` mask, survival, alive steps, and float16-quantized **mean-horizon mass** (integrated mass/H,
including zero dead frames). Mean mass prevents binary16 overflow. Forbidden flags never interpret an
untested slot as a death. B8 carrier and optional rollout-student SHA identities are recorded separately.

`replay_world` buffers one world's regenerated B8 `local/global/scalars` observations until all hashes
pass, then releases that world. `replay_exports` reports corrupt or divergent worlds and retains no full
bank in RAM. The fit-ingestion kill threshold is >2% dropped worlds. The A2 CLI is stricter: **any**
mismatch or an empty export directory returns nonzero. It performs no policy forward during replay.

The rollout protocol is `evaluate(world) -> PolicyEvaluation` plus
`evaluate_rows(sim, rows, carrier_net) -> float32[N,6]`. The engine owns an independent device opponent
dispatcher, so a custom hero adapter cannot move frozen inference back to CPU. Future P5 integration
uses `--policy-factory module:callable --student-checkpoint PATH`; the factory receives `device` and
`checkpoint`, owns the actual student model/observation extension, and may ignore `carrier_net`.
`--checkpoint` remains the B8 carrier used to construct/replay the world. The sibling tower loader and
its A3/serving qualification are outside this worktree's implementation scope.

The command-line and shell entrypoints forward all five trigger settings. Defaults remain interval 8,
minimum length 600, low-length probability 0.03, death window 150, and at most two rewinds. Label
rounds R1/R2/R3 default to beta 1/0.5/0; `--beta` or `T34_BETA` can select an explicit governed value.
The builder accepts `--triggers '{"confirm_interval":8,"max_rewinds":2}'` and
`--policy-factory module:callable`. Shell adapters use `T34_POLICY_FACTORY`, `T34_STUDENT_CHECKPOINT`
and optional `T34_CHECKPOINT`, with checkpoint names relative to the materialized input directory.
A student checkpoint without a factory fails explicitly. The production manifest builder refuses
nondefault checkpoint inputs until their publication and SHA binding are added; the current public
catalog binds only the B8 carrier and frozen roster. This is a protocol seam, not completed P5 integration.

## Inputs and bounded-upload alternative

The owner's input addendum supplies an existing S1 public release. No release was created or changed.
`PUBLIC_INPUTS.json` pins the B8 asset (6,562,005 bytes) and `league.tar.gz` (21,362,869 bytes), their
SHA-256 values, and every extracted member's exact size/hash. The downloader streams both assets,
verifies the archive before extraction, and verifies the B8 plus five frozen checkpoint members.
The spec's wording “six frozen checkpoints” differs from its authoritative T33 inventory: that inventory
has **six files total: B8 plus five frozen checkpoints**. No seventh checkpoint or hash was invented.
Missing URLs, asset binding drift, or member binding drift refuse preparation/materialization.

The six raw checkpoint files total 30,720,004 bytes (largest 6,906,247 bytes). They technically fit the
supervisor's 64 MiB per-file transfer bound, using 2 MiB upload chunks and a shared 300 s staging cap.
The owner's addendum reports S1 ContentsClient goodput of about 13 KB/s and a staging timeout; transferring
these bytes at that rate would take roughly 39 minutes. Therefore public download is the prepared path.
`--local-upload-preflight` is local inspection only and is never dispatch-ready. It was not used for an
allocation. Only the tiny catalog travels through the bounded supervisor upload path.

## Exact Mac commands

Run from this worktree. Every destination below must be fresh; increment the attempt suffix for reruns.
Do not reuse/delete a prior evidence directory. No tmux is needed for these commands.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-exit-engine
export T34_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export T34_PREP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-operator-20261019-a1
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export SNAKE_DQN_DEVICE=cpu
export TORCHINDUCTOR_COMPILE_THREADS=1

"$T34_PY" - <<'PY'
import json, os
from pathlib import Path
root = Path(os.environ['T34_PREP'])
root.mkdir(exist_ok=False)
for row in json.loads(Path('research/t34_exit_b8/INPUTS_REQUIRED.json').read_text())['inputs']:
    target = root / 'inputs' / row['dest']
    target.parent.mkdir(parents=True, exist_ok=True)
    target.symlink_to(Path(row['local_path']))
PY

env PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  T34_INPUT_MODE=local-preflight T34_INPUT_DIR="$T34_PREP/inputs" \
  T34_OUTPUT_DIR="$T34_PREP/a6-benchmark" T34_WORLDS=2 T34_FRAMES=500 T34_SECONDS=600 \
  T34_NAMESPACE=t34-exit-operator-a6-benchmark-a1 \
  bash research/t34_exit_b8/bench.sh

env PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin \
  T34_INPUT_MODE=local-preflight T34_INPUT_DIR="$T34_PREP/inputs" \
  T34_OUTPUT_DIR="$T34_PREP/a6-label" T34_WORLDS=2 T34_MAX_WORLDS=2 \
  T34_FRAMES=500 T34_SECONDS=600 T34_ROUND=a6 T34_SHARD=0 T34_ATTEMPT=1 \
  T34_NAMESPACE=t34-exit-operator-a6-label-a1 \
  bash research/t34_exit_b8/label.sh

"$T34_PY" -m research.t34_exit_b8.replay \
  --export-dir "$T34_PREP/a6-benchmark/worlds" \
  --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" --out "$T34_PREP/a6-replay.json"

"$T34_PY" -m pytest -q tests/test_t34_exit_identity.py
"$T34_PY" -m pytest -q -n0 -m 'not slow and not requires_gpu' -k 'not mps'
```

The full local gate deselects `requires_gpu` and MPS-named cases as well as `slow`: explicit MPS tests otherwise run
despite the CPU device environment. This preserves the spec's CPU-only Mac rule. A preliminary plain
`not slow` invocation was interrupted before any hardware-marked test executed; its partial log is not
a full-suite pass.

To use public inputs on a Mac without the local inventory, replace the symlink loop with:

```bash
"$T34_PY" -m research.t34_exit_b8.public_inputs \
  --catalog research/t34_exit_b8/PUBLIC_INPUTS.json --out-dir "$T34_PREP/inputs"
```

## Manifest build and two-L4-lane A5 dispatch — operator only

These commands were **not dispatched** by this implementation. Wait for S1's dispatcher to finish.
Retain the supervisor's account-wide cap of three including notebook workers, reviewed rates for any
existing assignments, current CLI acceptance, and the owner's approved two-unit rehearsal budget.
Lane three remains closed until two lanes complete one chunk cleanly. No allocation/job retries.

Pin a full published commit containing this package. Each A5 lane is capped at one priced unit and
30 minutes including setup/export. The prepared workload window is 600 seconds, the estimate 15 minutes,
and the transport timeout 30 minutes (2× estimate). The supervisor refuses a 60-minute timeout with a
one-unit cap even if the workload stops earlier. This tighter work window preserves the approved total;
the CLI still supports an at-most-1,500-second workload under a separately valid future manifest.
Two lanes total two priced
units at 2.0 units/hour. Each lane starts two teacher workers, each with five Numba threads and W256
root capacity, CUDA fp32, all three mixes, L4-only hardware and a 600 s lease. The supervisor's first
wall/unit cap always wins. A cold bootstrap that consumes the cap is incomplete, never permission to
increase it or rerun automatically.

```bash
export T34_JOB_ROOT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-exit-a5-operator-a1
mkdir "$T34_JOB_ROOT"
export T34_COMMIT="$(git rev-parse HEAD)"

"$T34_PY" -m research.t34_exit_b8.make_manifest --kind benchmark \
  --commit "$T34_COMMIT" --shard 0 --attempt 1 \
  --output-dir "$T34_JOB_ROOT/l4-0-output" --out "$T34_JOB_ROOT/l4-0.json" --validate
"$T34_PY" -m research.t34_exit_b8.make_manifest --kind benchmark \
  --commit "$T34_COMMIT" --shard 1 --attempt 1 \
  --output-dir "$T34_JOB_ROOT/l4-1-output" --out "$T34_JOB_ROOT/l4-1.json" --validate

export T34_LAUNCH=/Users/josenunez/Projects/ml/snake-dqn-exit-engine/research/colab_cli/launch.sh
export T34_COLAB=/Users/josenunez/.local/bin/colab
"$T34_LAUNCH" --check --state-dir "$T34_JOB_ROOT/state-0" --colab "$T34_COLAB" \
  validate "$T34_JOB_ROOT/l4-0.json"
"$T34_LAUNCH" --check --state-dir "$T34_JOB_ROOT/state-1" --colab "$T34_COLAB" \
  validate "$T34_JOB_ROOT/l4-1.json"

"$T34_LAUNCH" --state-dir "$T34_JOB_ROOT/state-0" --colab "$T34_COLAB" \
  run "$T34_JOB_ROOT/l4-0.json" > "$T34_JOB_ROOT/l4-0-supervisor.log" 2>&1 &
t34_supervisor_0=$!
"$T34_LAUNCH" --state-dir "$T34_JOB_ROOT/state-1" --colab "$T34_COLAB" \
  run "$T34_JOB_ROOT/l4-1.json" > "$T34_JOB_ROOT/l4-1-supervisor.log" 2>&1 &
t34_supervisor_1=$!
wait "$t34_supervisor_0"
wait "$t34_supervisor_1"
```

Inspect each terminal supervisor receipt, immutable exports, confirmed session disappearance, both
worker reports, and `benchmark-lane.json`. The ≥3,000 rule applies to the sum of the two worker active
row-step counts divided by their concurrent lane wall time. A printed threshold result alone does not
prove release, complete-world rate, two-lane acceptance, or permission for Round 0.

## A2 replay checks and subsequent labelling manifests

The supervisor preserves relative paths beneath `OUTPUT/exports/`. Replay each exported worker world
directory separately. Repeat the same check after regeneration on a second L4 session through the
supervisor; do not operate Colab directly. Accumulate at least 150 worlds spanning all three mixes and
require **zero** mismatches for A2. Local two-world replay is only a Mac harness smoke.

```bash
"$T34_PY" -m research.t34_exit_b8.replay \
  --export-dir "$T34_JOB_ROOT/l4-0-output/exports/lane0/worlds" \
  --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" --out "$T34_JOB_ROOT/a2-mac-l4-0-worker0.json"
"$T34_PY" -m research.t34_exit_b8.replay \
  --export-dir "$T34_JOB_ROOT/l4-0-output/exports/lane1/worlds" \
  --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" --out "$T34_JOB_ROOT/a2-mac-l4-0-worker1.json"
```

For labelling, choose `chunk_worlds = floor(2.5 * measured_complete_worlds_per_hour_per_lane)` using
both concurrent A5 worker reports on full H5000 worlds. Include zero-completion/censoring information;
do not extrapolate the two-world/500-frame Mac preflight. Supply that measured count to `--worlds`.
P7 must supply the campaign's collision-audited namespaces/bank and fixed round rules before dispatch;
the deterministic prep namespaces here are development fixtures, not approved campaign banks.

```bash
# Example only after measurement: replace 100 with floor(2.5 * the measured lane worlds/h).
"$T34_PY" -m research.t34_exit_b8.make_manifest --kind label \
  --commit "$T34_COMMIT" --round r1 --shard 0 --attempt 1 --worlds 100 \
  --output-dir "$T34_JOB_ROOT/r1-s0-a1-output" --out "$T34_JOB_ROOT/r1-s0-a1.json" --validate
```

The label draft has estimate 180 min, timeout 360 min, 12-unit maximum, workload bound 170 min, fresh
round/shard/attempt namespace, immutable per-world exports and about-two-minute telemetry. It does not
constitute an owner spending/release decision. No labels, fits, strict gate, or serving release were
launched by this prep job.
