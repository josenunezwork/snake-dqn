# T34 v2 teacher

Implementation of `docs/research/specs/2026-10-19_t34_v2_teacher.md`, bound to
`DESIGN_V2_SHADOW.md` (including Amendment 1) and `SUCCESS_CONTRACT.md`.
The original v1 engine remains available for historical replay and tests.

## Implementation and verification intent

Written before this implementation's smoke exports. This task prepares tooling;
its tiny development fixtures are excluded from all campaign estimates. It does
not collect a label round, run efficacy, fit a student, or authorize serving.

- Extend the existing ExIt export/replay and shadow/LAD packages. Use D=300 and
  H=300 by default, fp32, teacher stride-ten probes and stride-one refinement of
  every sampled survive-to-die gap. Bisection is not the teacher label rule.
- Student greedy continuation supplies dense KL-anchor labels. Decisive labels
  and deterministic sparse pre-LAD labels carry explicit roles. Apply beta in
  {1, 0.5, 0} only at scheduled teacher decisions; invalidate after intervention.
- Preserve six-snake actions, RNG stream identities, initial/every-250/final
  hashes, immutable exports, and whole-world replay rejection.
- All implementation verification runs on the Mac. At most two numerical CPU
  threads total; no live Colab, pods, tmux, or other worker operations.
- A1: compare repeated exports in separate spawned processes. A2: run the
  existing replay harness on the local smoke. These are tooling checks, not
  cross-runtime A2 acceptance or candidate efficacy evidence.
- Run each new real benchmark/label entrypoint with two worlds on Mac CPU.
  Record exact commands, elapsed time, work and replay result. L4 and optional
  CPU-runtime manifests are prepared/locally validated only, never dispatched.
- Commit/push coherent steps on the assigned branch; publish to main at the end
  after integration checks. Preserve campaign contracts and existing artifacts.

Acceptance table and exact operator commands will be completed with the final
implementation and measured preflight evidence.

## Acceptance boundaries

| Check | Local acceptance | Remaining evidence |
|---|---|---|
| Teacher rule | Amendment 1 stride-ten/transition-refinement plumbing and explicit decisive/anchor roles | Historical M2 teacher recovery is **NO-GO: 37/75 (49.3%), below 90%**; this implementation does not change that decision |
| Shadow identity | Exact clones, full state confirmation and invalidation; six-action traces | Original M1 is GO; new kernel identity checks are tooling coverage only |
| A1 exports | Repeated deterministic Mac exports across fresh processes | Full campaign/runtime coverage remains separate |
| A2 replay | Existing replay harness, whole-world hash rejection, local smoke | L4→Mac and L4→second L4, >=150 worlds across three mixes, zero unexplained mismatches remain pending |
| A5 | Same v2 kernel for labels and benchmark; L4-only manifest and explicit optional CPU probe | No live L4 allocation, two-runtime rehearsal, rate qualification or GPU numerical acceptance in this task |
| Mac collection | CPU entrypoint with bounded worlds/time and <=2 threads | No label round executed; campaign bank audit/funding still required |
| Fit ingestion | Exports explicitly tag `searched`, `decisive`, `kl_anchor_only` | Integrated fit-cache normalizer accepts `label_kind` and KL-only anchors; a production ReplaySource factory and admitted data bank still remain separate |

The earlier [shadow measurement results](../t34_shadow/RESULTS.md) also report
serial Watch search latency NO-GO. Nothing here clears Path B timing or efficacy.
No serving configuration changes are part of this teacher implementation.

## Export roles and work accounting

The world schema remains `t34-exit-world/v1` so the existing replay harness can
regenerate observations. V2 adds a `teacher_v2` identity and per-label shadow/LAD
provenance, including `label_kind` for the integrated fit-cache normalizer. Each
transition records its own decision as `t_star`; the global latest LAD stays in
`lad.latest_surviving_decision`. Dense confirmed rows are KL anchors (`searched=false`,
`decisive=false`, `kl_anchor_only=true`). Search labels retain tested per-action
survival, alive steps and mass; `F` denotes tested dying actions when a surviving
recommendation exists. An untested action is never assumed to die. One canonical
label per decision prevents the original replay index from losing duplicate roles.

Timing is kept in chunk/benchmark reports, outside deterministic world payloads.
Rates include actual executed work and distinguish complete worlds from capped
unfinished roots. Local smoke timings include setup/JIT and host contention;
they are not L4 forecasts or measurements of whole-round throughput.

## Exact operator commands

Use fresh output paths for each attempt. These commands select v2 explicitly.
The Mac defaults to one numerical thread; full Mac mode accepts at most two.
The source checkpoint inventory is reused, never rewritten.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn-v2-teacher
export T34_PY=/Users/josenunez/Projects/ml/snake-dqn/venv/bin/python
export T34_PREP=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-v2-operator-20261020-a1
export PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/bin:/usr/bin:/bin:/usr/sbin:/sbin
export SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export TORCHINDUCTOR_COMPILE_THREADS=1
export DYLD_LIBRARY_PATH=/Users/josenunez/Projects/ml/snake-dqn/venv/lib/python3.12/site-packages/torch/lib:/usr/lib

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

T34_KERNEL=v2 T34_INPUT_MODE=local-preflight T34_INPUT_DIR="$T34_PREP/inputs" \
  T34_OUTPUT_DIR="$T34_PREP/benchmark" T34_WORLDS=2 T34_FRAMES=500 T34_SECONDS=600 \
  T34_NAMESPACE=t34-v2-operator-benchmark-a1 bash research/t34_exit_b8/bench.sh

T34_KERNEL=v2 T34_INPUT_MODE=local-preflight T34_INPUT_DIR="$T34_PREP/inputs" \
  T34_OUTPUT_DIR="$T34_PREP/label" T34_WORLDS=2 T34_MAX_WORLDS=2 \
  T34_FRAMES=500 T34_SECONDS=600 T34_ROUND=a6 T34_SHARD=0 T34_ATTEMPT=1 \
  T34_NAMESPACE=t34-v2-operator-label-a1 bash research/t34_exit_b8/label.sh

"$T34_PY" -m research.t34_exit_b8.replay \
  --export-dir "$T34_PREP/benchmark/worlds" --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" \
  --out "$T34_PREP/benchmark-replay.json"
"$T34_PY" -m research.t34_exit_b8.replay \
  --export-dir "$T34_PREP/label/worlds" --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" \
  --out "$T34_PREP/label-replay.json"
```

Whole-round Mac mode uses the same shell entrypoint. Set the governed beta and
collision-audited namespace/count for an admitted round. The example below is
a two-world development run (not a campaign bank or an authorized label round).
`T34_THREADS=1` is suitable while sharing the two-thread verification budget.

```bash
T34_KERNEL=v2 T34_MAC_FULL_ROUND=1 T34_THREADS=1 \
  T34_INPUT_MODE=local-preflight T34_INPUT_DIR="$T34_PREP/inputs" \
  T34_OUTPUT_DIR="$T34_PREP/mac-label" T34_WORLDS=2 T34_MAX_WORLDS=2 \
  T34_FRAMES=5000 T34_SECONDS=10200 T34_ROUND=mac T34_SHARD=0 T34_ATTEMPT=1 \
  T34_BETA=1 T34_NAMESPACE=t34-v2-mac-dev-a1 bash research/t34_exit_b8/label.sh
```

The optional CPU runtime probe also exercises the same kernel. This Mac
preflight reports **Mac** CPU/RAM only; it is not a Colab hardware measurement.

```bash
SNAKE_CKPT_DIR="$T34_PREP/inputs/checkpoints" "$T34_PY" \
  -m research.t34_exit_b8.benchmark --kernel v2 --cpu-runtime-probe \
  --checkpoint "$T34_PREP/inputs/B8_s3_final.pth" \
  --output-dir "$T34_PREP/cpu-probe/worlds" --out "$T34_PREP/cpu-probe/benchmark.json" \
  --worlds 2 --max-worlds 2 --frames 500 --seconds 600 --horizon 300 \
  --threads 1 --device cpu --namespace t34-v2-cpu-probe-mac-a1
```

Prepare the two L4 lane manifests from a full published commit. `--validate`
performs local validation only. The catalog uses the existing public release
`s1-inputs-20261019`; checkpoints are downloaded and SHA-verified at runtime.
No Drive mount, new release, direct Colab CLI call, allocation or tmux command
is part of these preparation commands.

```bash
export T34_COMMIT="$(git rev-parse HEAD)"
"$T34_PY" -m research.t34_exit_b8.make_manifest --kind benchmark --kernel v2 \
  --commit "$T34_COMMIT" --shard 0 --attempt 1 \
  --output-dir "$T34_PREP/l4-0-output" --out "$T34_PREP/l4-0.json" --validate
"$T34_PY" -m research.t34_exit_b8.make_manifest --kind benchmark --kernel v2 \
  --commit "$T34_COMMIT" --shard 1 --attempt 1 \
  --output-dir "$T34_PREP/l4-1-output" --out "$T34_PREP/l4-1.json" --validate
```

The CPU manifest variant adds `--cpu-runtime-probe` and requires
`--cpu-rate-units-per-hour` with an observed positive provider rate. There is no
assumed free CPU rate. Its ten-world full-H5000 run records `nproc`/RAM and is
reported separately from the L4 A5 gate. Allocate through the existing Mac
supervisor only after campaign admission; preparing a manifest grants no spend.

## Export sizing and numerical recipe

V2 manifests allow at most 2 GiB of immutable exports. Label preparation requires
`--measured-export-bytes-per-world` from a **full-H5000 A5** measurement. The
builder refuses a chunk when
`ceil(worlds * measured_bytes_per_world * 1.25) + 16 MiB > 2 GiB`.
The factor is a planning allowance; the supervisor still enforces the real hard
limit. Choose the smaller count from measured throughput/time and export capacity.
Do not use a 500-frame smoke or multiply its rate into an admitted round.

For a measured bound stored in an operator receipt, the command shape is:

```bash
# Set both values from the admitted full-H5000 A5 receipt; no defaults are assumed.
"$T34_PY" -m research.t34_exit_b8.make_manifest --kind label --kernel v2 \
  --commit "$T34_COMMIT" --round r1 --shard 0 --attempt 1 \
  --worlds "$T34_MEASURED_CHUNK_WORLDS" \
  --measured-export-bytes-per-world "$T34_MEASURED_EXPORT_BYTES_PER_WORLD" \
  --output-dir "$T34_PREP/r1-s0-output" --out "$T34_PREP/r1-s0.json" --validate
```

Persistent shadows use exact batch-one greedy steps. Actual roots use bounded
64-world fp32 forwards, while branch queries compact into at most 64 active rows
per bucket (both configurable in the Python API where exposed). Batch-N floating
point need not equal batch-one. A shadow action/digest mismatch invalidates its
future labels and plans. Every searched greedy control must reproduce the known
shadow death timing; disagreement holds collection with `invalid_reason` rather
than silently teaching from a different continuation. A1 establishes repeated
Mac execution with the same batching recipe; it does not certify another GPU,
bucket shape, backend, or checkpoint. Cross-runtime A2 remains mandatory.

Stage timers report build, shadow, root policy/step, branch, reconstruction and
export wall time. They include inference within their respective stages, rather
than pretending to be isolated host CPU time. GPU utilisation samples are
reported as samples, not a sustained GPU-busy benchmark.
