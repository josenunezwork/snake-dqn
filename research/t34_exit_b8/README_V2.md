# T34 v2 teacher

Implementation of `docs/research/specs/2026-10-19_t34_v2_teacher.md`, bound to
`DESIGN_V2_SHADOW.md` (including superseding Amendment 2) and `SUCCESS_CONTRACT.md`.
The original v1 engine remains available for historical replay and tests.

## Implementation and verification intent

The original intent was written before the Amendment-1 smoke exports; its
source-bound verification remains in `VERIFICATION_V2.json`. Amendment 2 arrived
during closeout and is applied before the amended development fixtures below.
This task prepares tooling;
its tiny development fixtures are excluded from all campaign estimates. It does
not collect a label round, run efficacy, fit a student, or authorize serving.

- Extend the existing ExIt export/replay and shadow/LAD packages. Use D=300 and
  H=300 by default, fp32, teacher bisection from the first doom detection decision
  and radius-three stride-one refinement. Amendment 1's stride-ten rule is superseded.
- Student greedy continuation supplies dense KL-anchor labels. Decisive labels
  and deterministic sparse pre-LAD labels carry explicit roles. Full ranked
  per-action outcomes retain horizon/censoring and all-die diagnostics. Only
  horizon rescues are decisive; all-die delay rows have exactly zero fit weight
  and execute bare. Apply beta in
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

The acceptance table, exact operator commands and measured preflight evidence
are recorded below.

## Acceptance boundaries

| Check | Local acceptance | Remaining evidence |
|---|---|---|
| Teacher rule | Amendment 2 bisection/radius-three refinement and ranked outcomes; horizon-rescue, delay and anchor roles | Historical M2 **NO-GO** stands (bisection 39/75, stride rule 37/75, both below 90%); infrastructure does not change that decision |
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
searched row records its own decision as `t_star`; the global latest LAD stays in
`lad.latest_surviving_decision`. Dense confirmed rows are KL anchors (`searched=false`,
`decisive=false`, `kl_anchor_only=true`). Search labels retain tested per-action
survival, alive steps and mass; `F` denotes tested dying actions when a surviving
recommendation exists. `ranked_actions` retains the full tested ordering: keep
bare if it survives, otherwise choose horizon survivors by Q; if all die, rank
by alive steps, then Q, bare, action index. Delay recommendations remain diagnostic
(`outcome_kind=all_die_delay`, `delay_only=true`, `label_weight=0`), and the
executed action stays bare. The fit normalizer preserves their nondecisive role
and zeros weight before death-window normalization, including all-delay windows.
An untested action is never assumed to die. One canonical
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

Whole-round Mac mode uses the same shell entrypoint through the existing CPU-slot
wrapper, which refuses battery/closed-lid operation before and after slot admission.
`caffeinate` keeps the admitted job awake. Do not take a slot owned by Phase 0 or
another job; the wrapper waits rather than altering their processes. Set the governed beta and
collision-audited namespace/count for an admitted round. The example below is
a two-world development run (not a campaign bank or an authorized label round).
`T34_THREADS=1` is suitable while sharing the two-thread verification budget.

```bash
"$T34_PY" /Users/josenunez/Projects/ml/snake-dqn-v2-teacher/research/perf_sim_20261007/with_slots.py \
  --slots 1 -- /usr/bin/caffeinate -dims env \
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

V2 manifests set a hard 2 GiB **per-artifact** transfer bound. The builder also
applies a conservative projected **chunk-total** sizing allowance. Label preparation requires
`--measured-export-bytes-per-world` from a **full-H5000 A5** measurement. The
builder refuses a chunk when
`ceil(worlds * measured_bytes_per_world * 1.25) + 16 MiB > 2 GiB`.
The factor and chunk-total allowance are planning checks. The supervisor enforces
the hard per-artifact bound, not an aggregate chunk-total limit. Choose the smaller
count from measured throughput/time and projected export capacity.
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
future labels and plans. Doom caches are prepared once per doomed shadow, in groups of at most two
worlds, then released. Prefixes are not reconstructed per query. A validated
shadow supplies the free greedy outcome without a branch reroll; newly executed
control row-steps are zero. Cache capture is checked against the exact shadow
trace; disagreement holds collection with `invalid_reason`. Snapshot capture work
is reported separately, so the extra forward capture pass remains visible. A1 establishes repeated
Mac execution with the same batching recipe; it does not certify another GPU,
bucket shape, backend, or checkpoint. Cross-runtime A2 remains mandatory.

Stage timers report build, shadow, root policy/step, branch, reconstruction and
export wall time. They include inference within their respective stages, rather
than pretending to be isolated host CPU time. GPU utilisation samples are
reported as samples, not a sustained GPU-busy benchmark.

## Mac verification evidence

Executable source was frozen at `c44475f663b75fea4dd1be21088169cfc6e9861a`
after integrating the sibling fit-cache changes from main. No serving, learning,
or live cloud workload ran. All real preflights used CPU/one thread, two worlds,
H300/D300, and 500 collection frames; source-bundle smoke used two frames.

| Final attempt 02 | Complete worlds | Wall seconds (collector) | Executed row-steps | Export bytes | Shadow divergences |
|---|---:|---:|---:|---:|---:|
| Benchmark shell | 2 | 17.172 | 2,598 | 1,772,938 | 0 |
| Label shell | 2 | 11.154 | 2,598 | 1,774,870 | 0 |
| Mac full-mode benchmark, tiny fixture | 2 | 17.748 | 2,598 | 1,774,178 | 0 |
| Mac full-mode labels, tiny fixture | 2 | 10.840 | 2,598 | 1,775,053 | 0 |
| CPU-probe Python entrypoint on Mac | 2 | 17.982 | 2,598 | 1,774,356 | 0 |

Each 500-frame world retained 500 canonical anchor rows and hashes at 0/250/500.
These fixtures encountered no shadow doom; they do not measure LAD-search cost
or intervention efficacy. The independent real doom test forces two frozen/mixed
worlds into a shared branch bucket, verifies source state/carry preservation and
forbidden-action semantics, and exercises the existing compaction engine.

All five final preflight export directories plus the sparse-source fixture
replayed through the unchanged CLI: **12/12 worlds accepted, zero mismatches**.
The independent A1 test additionally compares payload/checksum/completion bytes
across two repetitions in each of two fresh processes, with all three mixes.
Its 12 exports each regenerated their expected two label rows.

Final L4 manifests for shards 0/1 were locally validated against the published
commit. Sparse fallback archives are **1,884,333 bytes** each; the extracted
archive ran the real benchmark entrypoint successfully. No allocator was called.
The optional CPU-probe Mac report records 18 logical CPUs and 64 GiB RAM; those
numbers describe this Mac, not a Colab runtime.

Artifacts are create-only under
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-v2-teacher-20261020/`:
`*-launch-02.json` contains exact argv/environment/source hashes;
`preflight-summary-02.json` reconciles reports and replay counts;
`a5-v2-l4-final-{0,1}.json` and `source-final-{0,1}/source-plan.json`
bind manifests and source archives. Attempt 01 remains separate evidence.
The final attempt adds fit-cache label metadata and export-sizing diagnostics;
it is not pooled with attempt 01 for timing.

Focused validation: **14 passed** for identity/real doom/lifecycle/export/policy;
**44 passed** for integrated teacher/export/job/fit-cache checks. The transport
supervisor/qualification selection passed **81 tests** after restoring missing
venv dependencies: requests 2.34.2, charset_normalizer 3.5.2, urllib3 2.8.0.
The first transport attempt (45 failed/36 passed) is preserved; its fake CLI
failed before behavior checks because `requests` was absent. Product lint and
study/test Black, isort, Flake8, Bash syntax and whitespace checks passed.

The frozen non-slow CPU suite completed with **8,412 passed, 20 failed, 29 skipped,
103 deselected**. Eighteen failures were missing `rg` in the restricted PATH;
the exact 20-node rerun with `rg` available returned **18 passed, 2 failed**.
The remaining T22 frozen-pool oracle failures report `v8_decide()` rejecting
`membership`; their source is unchanged from the starting baseline. The full
suite is not green. `VERIFICATION_V2.json` binds both runs separately. The
selection excludes the slow marker and two explicitly three-worker tests that
cannot honor this task's two-thread budget.

This evidence describes the Amendment-1 implementation at the frozen commit.
Amendment 2 arrived on main during verification and requires separate amended
source checks; these measurements do not qualify that subsequent implementation.
