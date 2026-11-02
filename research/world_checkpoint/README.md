# `research.world_checkpoint` — Exact Mid-World Checkpoint & Resume

`research.world_checkpoint` provides bit-identical mid-world checkpointing and resumption for Watch6 paired-world simulations (`bare`, `l2_prime`, `l1`), alongside a drop-in Colab CLI entrypoint (`research/world_checkpoint/job.py` and `research/world_checkpoint/job.sh`) that a future registration can use as its runner without modifying frozen packages (`research/watch6_l1_confirm/`, `research/watch6_search_efficacy_v3/`, `tools/watch6_v3_accounting/`).

---

## 1. How a Future Registration Plugs In `job.sh` / `job.py`

### Runtime Entrypoint (`job.sh`)
In a future package manifest (dispatched via `tools/ops/dispatch.sh`), point the job script at `research/world_checkpoint/job.sh` (or call `python -B -m research.world_checkpoint.job` from the package's `job.sh`):

```json
{
  "script": "research/world_checkpoint/job.sh",
  "resume": {
    "unit_kind": "world",
    "ledger_path": "inputs/verified-ledger.json",
    "skip_verified": true,
    "max_outcome_blind_retries": 2
  }
}
```

`research/world_checkpoint/job.sh` follows the exact `COLAB_INPUT_DIR` / `COLAB_OUTPUT_DIR` conventions of `research/watch6_l1_confirm/job.sh`:
1. Enforces CPU single-thread determinism (`SNAKE_DQN_DEVICE=cpu`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`).
2. Stages uploaded model checkpoints (`python -B -m research.watch6_l1_confirm.uploads`) and release evidence (`stage_release_evidence`) when present in `${COLAB_INPUT_DIR}`.
3. Invokes `python -B -m research.world_checkpoint.job` with:
   - `--out "${COLAB_OUTPUT_DIR}/efficacy"`
   - `--ledger "${COLAB_INPUT_DIR}/verified-ledger.json"`
   - `--input-checkpoint-dir "${COLAB_INPUT_DIR}/world-checkpoints"`
   - `--checkpoint-interval "${WATCH6_CHECKPOINT_INTERVAL:-1800}"`

### How Mid-World Checkpoints Are Exported by `research/colab_cli/runtime.py`
`research/colab_cli/runtime.py:_exports` discovers every `*.complete` marker under `COLAB_OUTPUT_DIR` (`output_dir.rglob("*.complete")`), verifies its `.sha256` sidecar, and exports the payload to the Mac attempt output directory as an immutable export.

During each world `world-XX` (or `fixture-world-XX` in fixture mode):
- `run_paired_world_resumable` writes monotonically numbered immutable checkpoints into `${COLAB_OUTPUT_DIR}/efficacy/checkpoints/<unit-tag>/`:
  - `ckpt-seq-000001-arm-bare-frame-01800.ckpt`
  - `ckpt-seq-000001-arm-bare-frame-01800.ckpt.sha256`
  - `ckpt-seq-000001-arm-bare-frame-01800.ckpt.complete`
- Each checkpoint is published create-only (`publish_immutable`) and never modified or deleted in `${COLAB_OUTPUT_DIR}`, satisfying `runtime.py`'s immutable export invariant.
- When `world-XX` finishes all arms (`bare`, `l2_prime`, `l1`), `_publish_world_unit` publishes the final `world-XX.json`, `world-XX.json.sha256`, and `world-XX.json.complete` under `${COLAB_OUTPUT_DIR}/efficacy/`.

### Preparing a Completion Retry on the Mac
When an attempt stops early (preemption, timeout, `SIGUSR1` / `SIGTERM`), prepare the next attempt's input directory before dispatch:
1. Build `verified-ledger.json` from completed `world-XX.json` files (so completed worlds are skipped entirely).
2. Stage verified mid-world checkpoints from the prior attempt output directory into `${INPUT_DIR}/world-checkpoints/` using `stage_exported_checkpoints`:
   ```python
   from pathlib import Path
   from research.world_checkpoint.job import stage_exported_checkpoints

   staged = stage_exported_checkpoints(
       Path("outputs/attempt-1"),
       Path("inputs/attempt-2/world-checkpoints"),
   )
   ```
3. Include `inputs/attempt-2/world-checkpoints/<unit-tag>/*` in the attempt manifest's `inputs` list so `research/colab_cli` uploads them into `${COLAB_INPUT_DIR}/world-checkpoints/`.
4. On start, `resumable_world_unit` searches both `${COLAB_OUTPUT_DIR}/efficacy/checkpoints/<unit-tag>` and `${COLAB_INPUT_DIR}/world-checkpoints/<unit-tag>`, picks the highest-sequence verified `.ckpt` across both directories, verifies its SHA-256, runtime/model/code/platform identity, and session rosters, and resumes at the exact frame (or arm boundary).

---

## 2. Safety & Determinism Invariants

- **Single-thread CPU enforcement (`configure_cpu()`):** Called automatically on `import research.world_checkpoint` and before checkpoint save/load/restore, setting `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`, `OPENBLAS_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`, `NUMEXPR_NUM_THREADS=1`, `SNAKE_DQN_DEVICE=cpu`, `torch.set_num_threads(1)`, and `torch.use_deterministic_algorithms(True, warn_only=True)`.
- **Identity binding (`capture_runtime_identity` / `verify_checkpoint_identity`):** Every checkpoint binds:
  - `model_sha256s`: SHA-256 digests of `checkpoint-b8.pth` (`b8_s3`), `frp3_s12`, and `champion_v8`.
  - `code_identity`: `STUDY_ID`, ` PROTOCOL_VERSION`, `SCIENCE_SPEC_SHA256`, and combined SHA-256 of `checkpoint.py`, `runner.py`, and `engine.py`.
  - `platform_fingerprint`: `system`, `machine`, `cpu_model`, `python_version`, `numpy_version`, `torch_version`, `numba_version`, `torch_num_threads`, `snake_dqn_device`.
  - `rosters`: Recomputed per-arm opponent roster verified against the checkpoint's saved `rosters` and `in_progress_arm["counters"]["roster"]`.
  - Any mismatch raises `ValueError` and refuses to resume unless explicitly overridden.
- **Versioned primitive storage (`_PrimitiveUnpickler`):** All NumPy arrays (`ndarray-bytes/v1`), PyTorch tensors (`tensor-bytes/v1`), and safety veto counter dataclasses are recursively converted to versioned primitive dicts before pickling. Unpickling runs through `_PrimitiveUnpickler`, which rejects any `GLOBAL` / `STACK_GLOBAL` class import.
- **Partial/corrupt checkpoint quarantine & fallback:** On startup, `quarantine_incomplete_checkpoints` moves any `.ckpt`/`.sha256` lacking a `.complete` marker (and any leftover `.r33-*`/`.tmp-*` temp files) into `.quarantine/`, and `highest_checkpoint_seq_on_disk` advances `checkpoint_seq` past the highest sequence number on disk so create-only publishing never collides (`FileExistsError`). If the newest `.complete` checkpoint has a corrupt payload or SHA mismatch, `find_latest_checkpoint` logs a warning and falls back to the previous verified checkpoint.

---

## 3. Running Tests & In-Process Negative / Mutation Checks

Run the test suite under single-thread CPU settings with `--basetemp` inside the worktree:

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest \
  tests/test_world_checkpoint.py -m "not slow" --basetemp=/Users/josenunez/Projects/ml/snake-dqn-world-ckpt/.pytest_tmp -v
```

To include the long-horizon natural collision/death/respawn test (`~1984–2005` frames on `FIXTURE_SEEDS[0]`):

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 SNAKE_DQN_DEVICE=cpu \
  nice -n 15 /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python -m pytest \
  tests/test_world_checkpoint.py --basetemp=/Users/josenunez/Projects/ml/snake-dqn-world-ckpt/.pytest_tmp -v
```

### Note on Root `conftest.py` Subprocess Re-exec and Mutation Testing
On macOS, the repository root `conftest.py` checks `_CONFTEST_REEXEC` at `pytest_configure` time and re-launches `pytest` via `subprocess.run([sys.executable, *sys.argv])` with `DYLD_LIBRARY_PATH` set to PyTorch's `lib/` directory. Because of that re-exec:
- Any monkeypatch applied in a parent Python wrapper process before calling `pytest.main(...)` is lost in the child `pytest` subprocess unless `_CONFTEST_REEXEC=1` and `DYLD_LIBRARY_PATH` are already set.
- `tests/test_world_checkpoint.py::test_negative_corrupted_restored_fields_and_type_invariants_in_process` performs its state/field/type corruption mutations **inside** the test function body (after `conftest.py` re-exec has already occurred), proving in-process that corrupting restored `sim.frames_since_food`, `runtime.memory_prev_dist`, `opponents`, `DenseTrigger.first_decision`, or restored field types (`frame_collisions`, `frame_ate_food`, `last_next_action_mask`, `_last_kill_victim_len`, `auto_respawn`) deterministically fails.
