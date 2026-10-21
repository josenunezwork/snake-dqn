# T34 Claim A pipeline: student labels → replay/cache → fit → probe

This package builds the Claim A distillation pipeline. The A100 advice round
([SYNTHESIS](../../docs/research/advice/2026-10-20_a100/SYNTHESIS.md)) found four gaps in it. The package
only imports, wraps or subclasses the ratified packages `research/t34_exit_b8/` and
`research/t34_exit_b8_fit/`. It changes no byte of those packages, of `t34_round0`, of `t34_pathb`, or of
the frozen loss (`research/t34_exit_b8_fit/loss.py`). It grants no authority for research, gating,
promotion or serving.

| Gap (verified) | Closed by |
|---|---|
| `t34_exit_b8/label.py:108-111` refuses process fan-out with a student | `label.py`: `run_label_chunk` keeps every ratified bound, sends only `(factory, checkpoint path)` across `spawn`, and builds the student inside each worker through the ratified `_load_policy` seam. Without a student, the call goes to the ratified entrypoint unchanged |
| The manifest builder refuses nondefault checkpoints | `make_manifest.py --kind label`: wraps the ratified builder (same catalog, sizing and qualification checks). It adds a student bound by SHA-256 and size, delivered as either a public `url` or a supervisor-uploaded file at `inputs/student/<name>`. `label.sh` checks the digest again on the runtime |
| No production `ReplaySource` or probe boundary (`t34_exit_b8_fit/README.md:28-31`) | `replay_source.py` (`source_factory`) and `probe.py` (`probe_factory`) plug into the ratified `runner.run` binding |
| CUDA fit path unqualified | `pipeline.py --device cuda` (CPU tested here) and the L4 qualification job below |

## Pieces

- **`student.py`**: `StudentPolicy(B8Policy)`. For a plain ego2s student (S1b) it replaces only
  `_forward`, so B8Policy's observation building, batching and tie rules stay as they are. A
  `exit-fit-tower/1` student (S1a) also paints full-sight planes from the same simulator rows.
  `full_fp32()` turns CUDA TF32 off around Q forwards, because the L4's cuDNN TF32 would move Q by
  about 1e-3.
- **Identity.** `tests/test_t34_claim_a.py::test_student_fanout_with_b8_student_reproduces_b8_labels_and_replays`
  runs the ratified B8 fan-out and the student fan-out with B8 as the student. The fixture is 2 worlds,
  500 frames and `nproc=2`, and its namespace reaches a real searched doom. Every `world-*.json`,
  `.sha256` and `.complete` file is byte-identical between the two runs. The exports then replay through
  the production source with the Q check enabled.
- **`replay_source.py`**:
  - `build_payload` writes `payload.json` (`t34-claim-a-payload/v1`) with `.sha256` and `.complete`
    sidecars. It lists every world file with its sha256, size, `split`, `round_age` and `chunk_id`, and
    hard-links outside exports into the tree so it stays self-contained.
  - `ExportReplaySource` regenerates each world with the ratified A2 `replay_world`, so every saved state
    hash must match. It then emits fit rows:
    - `q_prev`: the *previous frozen student* (the fit's initial model) evaluated on the regenerated
      observation, for every round. When the export's rollout checkpoint is that student, the Q recorded
      in the label must match within `1e-3 + 1e-4·|q|`. Otherwise the world is dropped (`q_prev
      mismatch`), and more than 2% dropped per chunk kills the fit.
    - `death_window_id`: the teacher's `shadow_id`, one doom search per shadow fork.
    - `length`: the hero length on the regenerated root.
    - The label fields that `normalize_row` needs.
  - The proof hash is sha256 of the world id plus the final-frame state hash, exported versus
    regenerated.
  - `training_row_counts` gives the exact bank-wide 1:1 sampler counts. These are required: most worlds
    have no searched row.
  - Held-out worlds stream through `replay_heldout`.
- **`probe.py`**: `DevProbe` plays fresh development worlds (namespace must contain `/probe`; never gate
  data) in lockstep with the current student against the v8 frozen opponents. At each decision it records
  the student's action, the previous student's action on the same observation, and the hero length. It
  returns the ratified `ProbeResult`. The production shape is 32 × H5000, validated by the ratified
  `validate_probe`.
- **`pipeline.py`** is one entrypoint for label → payload → cache → fit → probe. It calls the ratified
  `runner.run(mode="fit")` with the two factories and writes `pipeline.json` (create-only, with sidecars),
  which holds phase timings, cache statistics, per-epoch probe calls, the previous student's held-out
  decisive agreement, and the fit outcome. Two local-only switches are both stamped in the receipt:
  - `--probe-worlds/--probe-horizon` smaller than 32 × 5000 is refused on a runtime.
  - `--timing-only-no-tripwires` runs the ratified tripwires every epoch but records the verdicts instead
    of stopping. The selected checkpoint from such a run is never eligible.
- **`label.sh`** and **`pipeline.sh`** are the supervisor entrypoints. **`make_manifest.py`** builds the
  `label` and `pipeline` manifests and runs only local validation.

## Mac smoke (2026-10-20, 1 CPU thread per process, Mac running a 6-worker study)

Input: 2 newly student-labelled worlds (500 frames, `nproc=2`), one A5 export (`s0/scripted/000012`, 13
searched rows) for training, and one held-out A5 export (`s0/mixed/000002`, 10 decisive rows). The probe
used the smoke shape of 2 worlds × H300. Receipts are in
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-20261020/`.

| Phase (s) | S1b, tripwires enforced (`smoke-01`) | S1b, timing-only (`smoke-02`) | S1a, timing-only (`smoke-03`) |
|---|---|---|---|
| label (2 × 500 frames) | 15.6 | 16.8 | 16.4 |
| payload | 0.09 | 0.09 | 0.09 |
| cache build (11,000 rows) | 18.0 | 18.7 | 31.3 |
|  of which replay / q_prev | 10.8 / 6.6 | 11.5 / 6.5 | 13.4 / 16.7 |
| fit + held-out scoring | 0.9 (stopped after epoch 1) | 1.8 (4 epochs, 4 updates) | 2.4 |
| probe | 1.3 (1 call) | 5.3 (4 calls) | 10.7 |
| total | 36.6 | 43.5 | 62.0 |

Other measurements:
- Cache size is 12,777 bytes per row for S1b and 66,281 for S1a. Peak RSS was 1.39 GB (S1b) and
  2.9 GB (S1a).
- `smoke-01` stopped with `FIT_STOPPED: held-out decisive agreement below 70%` (see the first risk below).
- `smoke-04` ran the real `pipeline.sh` in local mode and exported only the receipts and the selected
  weights.
- The probe at production shape measured 3.6 ms per decision on one Mac core (4 worlds × H1000 with
  mixed opponents). At 32 × H5000 that is about 9–10 minutes per epoch.

## Risks found here (not changed: the loss, tripwires and sampling are frozen)

1. **The held-out decisive agreement tripwire (≥ 0.70) fires at epoch 1 by construction.**
   - A decisive (`horizon_rescue`) row is a state where the previous student's greedy action dies and
     `a_star` differs from it. On the smoke's held-out world, the previous student agrees on **0 of 10**
     rows.
   - After 4 Adam updates at 2.5e-5 the student agreed on 0.1. On a bank the size of A5 there is about one
     update per epoch.
   - Unless the preregistration rules on this (for example, check from epoch 2, or measure against
     baseline), every production fit will stop with `FIT_STOPPED`.
2. **Very few rows train.** The 1:1 rule keeps all searched rows plus the same number of anchors. The 39
   A5 worlds have 296 searched rows, so about 600 rows train per epoch, against 190,000 rows regenerated
   and cached (about 2.4 GB for S1b and 12.6 GB for S1a). GPU fit time is negligible. Time goes to replay,
   the cache and the probe.
3. **The probe dominates the fit phase.** At 32 × H5000, about 160,000 serial decisions per epoch, CPU-bound
   by the simulator and opponents. That is about 40–60 minutes for 4–6 epochs. Running probe worlds across
   processes is the obvious next lever.
4. **CUDA has not been run here.** The device flag, pinned prefetch and TF32 guard are exercised only on
   CPU. The L4 job below is the qualification.
5. **The probe namespace and held-out split must be preregistered.** `--heldout-decisive K` (hold out K
   worlds that carry decisive rows) is a qualification convenience, not a design.

## Colab L4 qualification run (prepared, NOT run)

This is a timing-only qualification. On the L4 runtime it:
- labels 24 worlds × H5000 with B8 as the student (12 CPU processes);
- holds out 2 worlds that carry decisive rows;
- builds the cache;
- fits 4 epochs on CUDA;
- runs the production 32 × H5000 probe each epoch.

Tripwires are recorded, not enforced, and the run makes no candidate claim. Estimated wall time is 75–90
minutes, with a 180-minute timeout and a cap of 6.0 units at 2.0 units/hour. Inputs are only the pinned
public B8 and league release.

```bash
cd /Users/josenunez/Projects/ml/snake-dqn          # on main at the commit that contains this package
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
PY=./venv/bin/python
OUT=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-claim-a-l4-qual-$(date +%Y%m%d-%H%M)
mkdir -p "$OUT"
"$PY" -m research.t34_claim_a.make_manifest --kind pipeline \
  --commit "$(git rev-parse HEAD)" --output-dir "$OUT/run" --out "$OUT/manifest.json" --validate
"$PY" -m research.colab_cli.supervisor validate "$OUT/manifest.json"
"$PY" -m research.colab_cli.supervisor run "$OUT/manifest.json" 2>&1 | tee "$OUT/supervisor.log"
```

Read `$OUT/run/` (receipt, telemetry) and the exported `pipeline.json`, especially
`timings.{label,cache_build,fit_and_heldout,probe}_seconds`, `fit.fit_timing`
(`peak_vram_bytes`, `epoch_minutes`) and `tripwires.per_epoch`. A CLI exit code alone does not mean
success. The A100 reopen rule in the SYNTHESIS uses this receipt: fit ≥ 60 minutes with the GPU leg ≥ 40%.

A student-label chunk for a later round (student uploaded and SHA-bound; `measured_export_bytes_per_world`
from the A5 receipts, about 8.9 MB):

```bash
"$PY" -m research.t34_claim_a.make_manifest --kind label --round r2 --shard 0 --attempt 0 \
  --worlds 24 --measured-export-bytes-per-world 9000000 \
  --student-path /path/to/student_r1.pth --student-name student_r1.pth \
  --commit "$(git rev-parse HEAD)" --output-dir "$OUT/label-r2-s0" --out "$OUT/label-r2-s0.json" --validate
```

## Local commands

```bash
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMBA_NUM_THREADS=1
./venv/bin/python -m pytest -q -n 2 tests/test_t34_claim_a.py      # 23 tests, about 35 s
# End-to-end smoke (worktrees: export SNAKE_CKPT_DIR=<main>/saved_snakes for frozen opponents)
A5=/Users/josenunez/Projects/ml/snake-dqn-artifacts/t34-engine-fanout-20261020/manifests-final-3ebf3991/a5-l4-s0-output/exports/worlds
./venv/bin/python -m research.t34_claim_a.pipeline --out /tmp/claim-a-smoke \
  --carrier /Users/josenunez/Projects/ml/snake-dqn-artifacts/s1-mature-20261018/inputs/B8_s3_final.pth \
  --device cpu --epochs 4 --label-worlds 2 --label-frames 500 --label-seconds 600 --label-nproc 2 \
  --label-namespace t34-v2-amend2-preflight/benchmark/03 --label-mixes scripted,frozen,mixed \
  --extra-export "$A5/process-000/worlds/world-000012.json:0:a5-s0" \
  --extra-export "$A5/process-002/worlds/world-000002.json:0:a5-s0" \
  --heldout-world-id t34-exit-a5/v2/s0/a3/mixed/000002 \
  --probe-worlds 2 --probe-horizon 300 --timing-only-no-tripwires
```
