# Catalog: `research/t33_batched_teacher/` (T33 batched continuation teacher, round two)

Status: **HISTORICAL / partly live as a library.** 95 files (21 `.py`, `bench.sh`, 5 `.md`, JSON receipts and
receipt directories). Read-only catalog written 2026-10-20. Every claim below comes from the package's own files
or from the specs and plan it cites. Where a fact was not found, the entry says "unclear".

Primary sources: [README.md](../../../research/t33_batched_teacher/README.md),
[OPTIMIZATION_LOG.md](../../../research/t33_batched_teacher/OPTIMIZATION_LOG.md),
[protocol.md](../../../research/t33_batched_teacher/protocol.md),
[ROUND2_PROTOCOL.md](../../../research/t33_batched_teacher/ROUND2_PROTOCOL.md),
[ABLATION_SMOKE_ERRATUM.md](../../../research/t33_batched_teacher/ABLATION_SMOKE_ERRATUM.md),
specs [2026-10-19_batched_teacher.md](../specs/2026-10-19_batched_teacher.md) and
[2026-10-19_batched_teacher_round2.md](../specs/2026-10-19_batched_teacher_round2.md).

## Purpose

T33 produces development-only distillation labels for the B8 student from a **stochastic search teacher**. For each
decision it runs N epsilon-greedy continuations of horizon H per first action in an exact batched simulator. The
reference budget is N8/H128 and the timing shape is N4/H32. It then picks the first action whose continuation
scores best (mass, plus an optional Q leaf). The package also calibrates cheaper H/N/leaf variants against the full
reference, which is descriptive only ("selects no recipe"), and prepares an L4 Colab benchmark. The owner's
2026-10-19 instruction authorized it. It has no authority to promote, train, deploy or allocate remote resources
(protocol.md).

## Architecture

- **Resident lockstep layout** (`teacher.py`: `BatchedTeacher.choose(prepared_worlds)`, `merge_worlds`,
  `BranchLayout`): a fixed W x 6 x N resident row layout (W source worlds, 6 first actions, N rollouts) that steps
  in lockstep through `GridBatchSim`. The merge re-caps int64 body rings to the maximum segment count + H+1, which
  shrinks them from 998,400 to 79,968 bytes/row at H32/L800. It also pads unequal food-mirror capacities, validates
  grids and rejects overlapping live bodies. Before expansion, admission counts allocations, clone peaks and a
  conservative GPU activation reserve.
- **Lazy forks**: one leader per equivalent rollout advances until epsilon exploration makes the rollouts diverge.
  A fork copies frame, food/RNG, masks, V61 carry, trend memory and accumulated mass before movement. Dormant rows
  skip v8/V61/opponent host work. In the 1,024-world MPS cell, 179,500 row-steps were active out of 786,432
  resident.
- **CRN / pre-drawn epsilon** (`_predrawn_epsilon`): epsilon schedules are pre-drawn per source, action and rollout.
  Hero exploration draws are shared across first actions (common random numbers). H<=128 shares one canonical
  128-step schedule, so H32 and H64 are exact prefixes of it. This intentionally differs from Phase 0's conditional
  draws.
- **Inference / CUDA trunk** (`inference.py` `FixedInference`, `pipeline.py` `DoubleBufferedTeacher`): fp32 on
  CPU/MPS. On CUDA, the towers and fusion run in fp16 and the dueling head stays fp32. Qualification needs
  >=1,000 real states and >=99.5% masked argmax agreement. Two A/B lanes double-buffer with separate slabs and
  streams. The CPU orchestration is tested, but **CUDA compile, precision and overlap were never measured** (no
  hardware run).
- Support modules:
  - `runtime.py` (`BatchedVector61Runtime`).
  - `observations.py` / `observation_kernels.py` (`ObservationBuffer`, a reusable byte slab, Numba local-flood
    scratch).
  - `reference.py` (Phase 0 oracle + CRN reference).
  - `ablation.py` (prefix/leaf calibration).
  - `public_inputs.py` + `make_manifest.py` (Colab inputs and blocked L4 manifests).
  - `evidence.py` (receipt writing).

## Entrypoints

| Entrypoint | Role |
|---|---|
| `bench.sh` | Transport wrapper for the Colab supervisor and local preflight (Colab accepts public input mode only) |
| `python -m research.t33_batched_teacher.benchmark` | Lockstep benchmark (`--worlds`, `--resident-rows`, `--device`, `--horizon`, `--rollouts`, `--out`) |
| `... .ablation` | Prefix/leaf calibration (`--decisions 300 --workers 3 --eligible-cache --out`, per the README) |
| `... .lockstep_profile` | Per-stage Mac lockstep profile (CPU/MPS) |
| `... .qualify_inference` | Masked-argmax qualification on CUDA, MPS or CPU |
| `... .make_manifest`, `... .public_inputs` | Blocked L4 manifest drafts; public checkpoint fetch and validation |
| `anchor_microbench`, `prefix_bench`, `round2_microbench`, `runtime_bench`, `teacher_microbench`, `v8_v61_microbench` | Microbenchmarks behind the OPTIMIZATION_LOG table |

## Measured results (Mac only; from OPTIMIZATION_LOG.md)

- **Ablation smoke:** the full factorial ablation smoke at H16/N4 went from 2.683 s to 1.208 s (2.22x), with
  exact scores and choices.
- **Lockstep throughput** at the N4/H32 timing shape:
  - **CPU:** 2.10 / 2.03 decisions/s for 256 / 1,024 worlds (`ROUND2_MAC_LOCKSTEP.json`).
  - **MPS:** 23.31 / 24.01 decisions/s (`ROUND2_MAC_MPS_LOCKSTEP.json`).
  - The simulator step is the largest stage (~26 us per resident row-step). MPS inference takes ~23% of wall time.
- **L4 projections (conditional):** 17-64 decisions/s per L4, with 59.63/s at 1x Mac core speed, serial. All are
  explicitly **below the 100-200/s target**. No L4 measurement exists.
- **Other microbenchmark speed-ups:** corpse event ~11x, anchor dispatch ~4.3x, v8 owner-map on four threads
  ~2.4x, V61 food staging ~9.75x.
- **Leaf-scale smoke** (3 synthetic decisions): fitted scales were 5,269-8,725 segment-frames per Q unit, so
  scale 1 is a unit mismatch. This supports no label-quality conclusion.
- **Invalid receipts:** the initial `ABLATION_SMOKE.json` and `ABLATION_BUDGET_SMOKE.json` are **INVALID** for the
  value-leaf comparison because of a name-shadowing bug (see the erratum). Use `ABLATION_BUDGET_SMOKE_V2.json`
  instead.
- **Tests:** the affected-path gate passed 527 tests (5 skips, 3 slow tests deselected). The full non-slow repo
  gate was **not** certified green (`ROUND2_TEST_LOGS/`).
- **L4 status:** never launched. All six checkpoint URLs in `PUBLIC_INPUTS.json` are null. The latest draft
  manifest is `L4_MANIFEST_PUBLICATION_BLOCKED_ROUND2.json` (`dispatch_ready=false`).
- **Full calibration:** unclear from this package whether the full 300-decision calibration ever completed.

## Why it was superseded

1. [FINAL_PLAN.md](../final_run_20261019/FINAL_PLAN.md) (2026-10-19) found the N8/H128 stochastic teacher noisy:
   in calibration-v1, two draws agreed on the argmax in only 98 of 300 decisions, so distilling it would distil
   noise. The deterministic H=300 verify-then-expand teacher has no such variance and costs ~4-10x less per
   decision. Section 10 of the plan stops T33 host optimisation toward N8/H128 and the stochastic-teacher L4
   benchmark. It **keeps T33's engine, admission and export code as the base** for T34 P1.
2. [2026-10-19_exit_engine_prep.md](../specs/2026-10-19_exit_engine_prep.md) built `research/t34_exit_b8/` on
   T33 (branch tip eeb46b72).
3. [2026-10-20_t34_engine_fanout.md](../specs/2026-10-20_t34_engine_fanout.md) measured a Mac serial row-step
   at 5.84 ms. In the same measurement, a lockstep W=8 step took 2.2 ms against 1.0 ms serial, because of the
   deepcopy merge/scatter. The T34 engine therefore moved to K independent single-thread processes on disjoint
   world shards and **skips lockstep merge/scatter on CPU**. The CUDA batched path stays available only as an
   off-by-default option. This retires T33's resident lockstep as the production execution model.

## Still imported by live code (grep, 2026-10-20)

No `src/` module imports the package. These live research modules and tests do:

| Importer | Symbol(s) |
|---|---|
| `research/t34_exit_b8/engine.py` | `teacher._model_digest`, `teacher.merge_worlds` |
| `research/t34_exit_b8/lockstep.py` | `teacher.BranchLayout`, `teacher._runtime_for` |
| `research/t34_exit_b8/benchmark.py`, `label.py` | `import research.t33_batched_teacher.teacher` (import-side check) |
| `research/t34_exit_b8/make_manifest.py` | Lists the directory as a registered T33 input destination |
| `research/t34_pathb/batch_v2.py`, `engine_v2.py` | `teacher.merge_worlds` |
| `tests/fixtures/t34_shadow_batching/engine_serial.py` | `_model_digest`, `merge_worlds` |
| `tests/test_t34_v2_jobs.py` | Asserts the directory is in the job's shipped path set |
| `tests/test_t33_*.py` (13 files) | The package's own tests |

Ratified T34 receipts reference these files by path or hash:

- `research/t34_round0/VERIFICATION_A6*.json` and `REGISTRATION_ROUND0.json`.
- `research/t34_pathb/REGISTRATION_RETENTION_V2*.json` and `RETENTION_OC_V2.json`.

So `teacher.py` and everything it imports must not move or change.
