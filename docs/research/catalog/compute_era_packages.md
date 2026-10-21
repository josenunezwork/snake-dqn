# Compute-era packages catalog (Colab / RunPod infrastructure, 2026-10)

Scope: four `research/` packages from the Oct 2026 off-Mac compute era. Read-only catalog
(created 2026-10-20); claims come from the files themselves, "unclear" where they do not say.
The current Colab path is `research/colab_cli` (see [docs/colab-cli.md](../../colab-cli.md)),
which is outside this catalog.

| Package | Purpose | Status | Reusable vs dead |
| --- | --- | --- | --- |
| [s1_mature_20261018](../../../research/s1_mature_20261018/) | S1 study: two bare-B8 fine-tune arms (N1, N3) on Colab L4 via the CLI, Mac eval | Legacy — study finished, KILL | Dead as a study; ledger/admission/final-only export patterns reusable |
| [t16_colab](../../../research/t16_colab/) | T16 Colab throughput smoke + long-run recovery helpers | Legacy (throughput evidence only) | Partly reusable (runtime setup, bundle); smoke itself done |
| [colab_queue](../../../research/colab_queue/) | Drive-backed Colab job queue (Mac publisher, notebook workers) | Legacy — kept only for already-running work (Q1) | Dead for new work (CLI replaced it); still imported by Q1/T23 code |
| [runpod_fanout](../../../research/runpod_fanout/) | RunPod fan-out runner for Tier-1 episodes (pods / serverless / local) | Legacy — T34 uses no RunPod | Reusable library pieces (`platform_rule`, serverless) still imported elsewhere |

## research/s1_mature_20261018

- **Purpose** (README): "S1 mature collection study", separate from Q1. Both arms fine-tune the
  released bare B8 with fresh Adam and empty replay; N3 = Q1 static-pool collection, N1 = Q1 D's
  terminal-hero collection; 5 paired seed blocks x 2 arms, 8M finals; one Mac look on 640 worlds.
  Final-only export (no mid-run checkpoint/resume), 64 MiB per attempt.
- **Entrypoints**: `launch.sh` / `run.py` (serialized execution after admission), `admission.py`,
  `rerun.py` (manual operational replacement), `train.sh` / `train.py` (M3-C trainer adapter on L4),
  `evaluate.py` + `analyze.py` (Mac eval and final-look statistics), `ledger.py`, `manifests.py`,
  `bank.py`, `price.py`, `checkpoint_sizing.py`, `final_artifacts.py`, `review_jobs.py`, `audit.py`.
- **Registration**: [PREREGISTRATION.md](../../../research/s1_mature_20261018/PREREGISTRATION.md),
  status `APPROVED_S1_MATURE_20261018` (2026-10-18 ~21:00 PT, commit `a6cbba57`); owner budget in
  `OWNER_AUTHORIZATION.json`; budget 60 units / 25 GPU-h nominal.
- **Result / verdict**: **KILL both arms** (-12.5 / -14.9 pooled mass vs B8; audit PASS), recorded in
  [final_run_20261019/CAMPAIGN_LOG.md](../final_run_20261019/CAMPAIGN_LOG.md) and
  [HANDOFF_2026-10-20.md](../final_run_20261019/HANDOFF_2026-10-20.md). The verdict is **not**
  recorded inside the package: its README and `FINAL_EXPORT_STATUS_20261018.md` still read
  "ADMISSION REFUSED / NO DISPATCH" (pre-dispatch snapshots). Run artifacts live outside git in
  `snake-dqn-artifacts/s1-mature-20261018` (per
  [ARTIFACTS_INVENTORY_2026-10-20.md](../ops/ARTIFACTS_INVENTORY_2026-10-20.md)).
- **Status**: legacy, closed. FINAL_PLAN stops "bare-B8 TD fine-tunes after S1".
- **Reusable vs dead**: study dead. Reusable patterns: complete-budget admission + durable ledger,
  final-only export sizing (`FINAL_ARTIFACT_SIZE_FIXTURE.json`), conditional rerun reservation.
  Many `ADMISSION_STATUS_*.json` / `*_PREFLIGHT*.json` are superseded historical receipts (kept;
  ratified studies hash files).

## research/t16_colab

- **Purpose**: T16 Colab throughput smoke (B8 recipe trainer on Colab GPUs vs a Mac baseline), plus
  `long_run.md`, the doc for the owner-run recovery notebook
  [notebooks/colab_long_run.ipynb](../../../notebooks/colab_long_run.ipynb).
- **Entrypoints**: `run_smoke.py` (runner; `trainer_command`), `run_n3_bench_colab.sh`,
  `mac_n3_baseline.sh`, `setup_runtime.sh` (Py3.12 venv, cu126/cu128 wheel by compute capability),
  `bundle.py` (input bundle), `compare.py`, `environment.py`, `sampler.py`; notebook
  [notebooks/colab_throughput_smoke.ipynb](../../../notebooks/colab_throughput_smoke.ipynb).
- **Result / verdict**: throughput evidence only, "not gate, quality or serving evidence".
  Recorded in [t16_colab_smoke_2026-10-17.md](../t16_colab_smoke_2026-10-17.md) (spec/how-to),
  [t16_colab_smoke02_results_2026-10-17.md](../t16_colab_smoke02_results_2026-10-17.md)
  (smoke-02/03 results, A100; raw data on Drive `t16_colab/smoke-02/`),
  [t16_mac_baseline_2026-10-17.md](../t16_mac_baseline_2026-10-17.md). No GO/KILL verdict.
- **Status**: legacy (2026-10-17). Later work moved to the Colab CLI and L4-only.
- **Reusable vs dead**: `setup_runtime.sh` / wheel selection and `bundle.py` ideas reusable; whether
  any live code imports this package is unclear (the name `t16_colab` also appears as the Drive root
  in `colab_queue`, `q0q1_20261017`, `t23_r0_screen_20261017`).

## research/colab_queue

- **Purpose** (README): Drive-backed queue; Mac is the control plane (`q.py enqueue/...`), dedicated
  Colab runtimes run `worker.py` / `worker.sh`, claim pinned jobs (full 40-char commit, GPU class
  `any|A100|L4|cpu`), write claim/heartbeat/log/receipt files, and release themselves when idle.
  Drive root `My Drive/t16_colab/queue`.
- **Entrypoints**: `q.py` (Mac CLI), `worker.py` / `worker.sh` (Colab), `core.py` (schema/locks);
  `examples/` holds T23 train/eval job JSONs and launchers.
- **Result / verdict**: none (ops tool, "not an experiment authorization"). Jobs run through it:
  T23 R0 screen (`examples/`; MERGED_PLAN lists league curricula as closed by T23, no T23 results doc found in docs/) and Q1 (see
  [overnight_2026-10-18.md](../overnight_2026-10-18.md); Q1 A5 refused).
- **Status**: legacy. [docs/colab-cli.md](../../colab-cli.md) says notebook workers and Q1 are
  "legacy-only work"; new jobs use `research/colab_cli`.
- **Reusable vs dead**: dead for new work; must stay intact because `research/q0q1_20261017/*`
  imports it.

## research/runpod_fanout

- **Purpose** (`__init__.py`, `runner.py`): fan-out runner for Tier-1 episodes (screens, sweeps,
  censuses); refuses strict gates and serving qualification (`jobspec.py`).
- **Entrypoints**: `runner.py plan|run|status|cleanup|compare|merge` (backends: serverless default,
  `pods`, `local`); `pod_agent.py`, `seed_agent.py`, `sls_handler.py`, `watchdog.py`, `ledger.py`,
  `jobgen.py`. Policy: `fanout_policy.json` (cpu3c, SECURE, $100 project cap as of 2026-10-08),
  `serverless_policy.json`, `checkpoint_allowlist.json`.
- **Result / verdict**: none itself; writes runs to `<artifacts>/runpod-fanout/<job_id>/<run>/`.
  Results of studies run through it live in their own packages (FRP-v4/v5, M3-C pods, etc.).
  The runner docstring cites `docs/research/runpod_fanout_runner_2026-10-03.md`, which does not
  exist in the repo.
- **Status**: legacy for the active campaign (docs/STATUS.md: "RunPod for T34 (none)"). Maintenance
  status of its RunPod API client is unclear from the files.
- **Reusable vs dead**: reusable library. `platform_rule.check_per_world` and serverless pieces are
  imported by `research/sequential_strict_template/*`, `perf_podbench_20261013`,
  `redesign_m3_20261008`, `redesign_m3c_20261008`, `t18_h100pod_20261017`.
