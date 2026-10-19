---
name: snake-dqn-queue-jobs
description: |
  Use whenever you create, edit, generate or review a job for the Colab Drive queue in
  /Users/josenunez/Projects/ml/snake-dqn (research/colab_queue: job JSON manifests, Bash launchers run via
  `script`, env bindings with SHA sidecars, depends_on chains, output_dir choices, requeues, the q.py CLI,
  the colab_queue_worker*.ipynb notebooks). Covers the manifest schema and the queue's safety contract; the
  Colab runtime checklist itself (cores, timing, Drive, keep-alive) lives in snake-dqn-colab-jobs.
---

# Snake DQN: writing jobs for the Colab Drive queue

> **Legacy boundary (2026-10-18):** this queue is only for the two already-running notebook workers and Q1. Do not
> create, enqueue, requeue, or change a Drive-queue job for new work. New jobs use `research/colab_cli` through its
> Mac-side supervisor; no runtime Drive mount, `colab auth`, or `colab drivemount` is allowed there.

Source of truth: `research/colab_queue/README.md` (+ `core.py`, `worker.py`, `worker.sh`, `q.py`).
Runtime-side rules (no Mac paths, use all vCPUs, measure before est/timeout, stream progress, keep-alive
cell) are in `snake-dqn-colab-jobs`: apply BOTH skills; this one does not repeat that checklist.

## Mental model

- ONE Mac is the control plane. `q enqueue` validates the manifest and creates every shared Drive
  directory (`queue/*`, the job's `output_dir`, `output_subdirs`). Colab workers only write their own
  uniquely named claim/heartbeat/log/receipt files. Never create shared Drive dirs from a runtime.
- Drive root (Mac): `/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab`;
  on Colab the same root is `/content/drive/MyDrive/t16_colab`. Queue metadata lives under `$DRIVE/queue`.
- Two independent pins: the job's `commit` (repo checkout + `research/t16_colab/setup_runtime.sh` + your
  launcher) vs the notebook's `QUEUE_REF` (queue worker code fetched from raw.githubusercontent.com).
  Do not change `QUEUE_REF` for a job; it moves only when the queue implementation is upgraded.
- The queue is an operations tool, not an authorization: commit the launcher and any required
  pre-registration BEFORE enqueueing a learning job. Never "repair" INVALID/INCOMPLETE by rerunning.

## Manifest schema (one JSON object per `enqueue`, never an array)

| Field | Rule |
| --- | --- |
| `id` | Unique; letters, digits, `.`, `_`, `-`. Encode attempt + hardware: `q0-audit-b8-init-a100r3`. |
| `commit` | FULL 40-char SHA, already pushed to origin (the worker `git fetch --depth 1 origin <commit>`). Branch/tag/short SHA rejected. |
| `script` | Repo-relative Bash launcher, run with `bash` from the checkout at `commit`. |
| `env` | string -> string only. Bind inputs here (paths + `*_SHA256` sidecars, see below). |
| `requirements` | `{"gpu": "any"\|"A100"\|"L4"\|"cpu", "min_vcpu": int>0, "est_minutes": int>0}`. `any` = some real GPU (never selects CPU); `cpu` = CPU runtime only. |
| `depends_on` | IDs that must have CURRENT-generation `done` receipts; a requeued dependency invalidates dependents. |
| `output_dir` | Relative to the Drive root, job-private, not under `queue/`, not overlapping any other job's dir. |
| `priority` | int; higher first, then id ascending. |
| `timeout_minutes` | optional; default 2 x `est_minutes`. Set it from MEASURED time (>= 2x estimate). |
| `output_subdirs` | optional dirs created under `output_dir` by enqueue (shared outputs must pre-exist). |

Reference manifest: `research/q0q1_20261017/jobs_approved/q0-audit-b8-init-a100r3.json`. Generate
manifests with a script (pattern: `research/q0q1_20261017/queue_jobs.py`, `make_q1_jobs.py`) that pins
hashes; never hand-type SHAs.

## Env bindings and SHA sidecars

Everything the job consumes that is not in the checkout must be bound in `env`:

- Drive inputs as absolute Colab paths + a sidecar hash the launcher verifies before work:
  `Q0Q1_BANK_INVENTORY_RECEIPT=/content/drive/MyDrive/t16_colab/q0q1-20261017/admission/inventory-receipt.json`
  `Q0Q1_BANK_INVENTORY_RECEIPT_SHA256=2f9d7498...`. Mismatch = fail closed, exit non-zero, no work.
- Mac-computed evidence (strict inventories, receipts) is shipped to Drive first, then bound this way
  (Colab has no `/Users/...`, no `saved_snakes/` history).
- Upstream job outputs: bind the producer's `output_dir` (e.g. `Q0Q1_MASS_ARCHIVE_DIR=q0q1-20261017/q0/audit-t23-mass-a100r2`)
  AND list that job in `depends_on`.
- Registration/approval tokens as plain strings the launcher checks (`Q0Q1_REGISTRATION_STATUS=APPROVED_...`).
- The worker OVERRIDES these keys regardless of what you set: `QUEUE_JOB_ID`, `QUEUE_GENERATION`,
  `QUEUE_WORKER_ID`, `QUEUE_CLAIM_TOKEN`, `QUEUE_DRIVE_ROOT`, `QUEUE_OUTPUT_DIR`, `PYTHONPATH`, venv-first `PATH`.
- Artifacts you publish get their own sidecar: `(cd "$QUEUE_OUTPUT_DIR" && sha256sum <file>) > <file>.sha256`
  (see `research/colab_queue/examples/t23_train_seed.sh`).

## Launcher contract (the `script`)

Copy the guards from `research/q0q1_20261017/run_queue_job.sh` + `run_host_contracts.sh` and
`research/colab_queue/examples/t23_*.sh`:

- `set -euo pipefail`; `cd /content/t16-snake-dqn` (or `cd "$(dirname "${BASH_SOURCE[0]}")/../.."`).
- Validate `QUEUE_OUTPUT_DIR` is an existing ABSOLUTE job-private dir; exit 64 on usage errors.
- **Fresh-attempt guard:** refuse if any prior output (receipt, log, partials) already exists in
  `QUEUE_OUTPUT_DIR` ("use a fresh attempt directory"). Never overwrite prior evidence.
- Fixed runtime paths: repo `/content/t16-snake-dqn`, venv `/content/t16-colab-venv/bin/python`, verified
  inputs `/content/t16-colab-extracted/t16-colab-inputs` (checkpoints under `.../checkpoints`,
  `SNAKE_CKPT_DIR`). No `pip install`, no setup in the launcher (setup_runtime.sh already ran).
- CPU determinism env: `SNAKE_DQN_DEVICE=cpu OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1`
  (GPU jobs set their own device flags explicitly).
- Scratch on local disk (`mktemp -d /content/...`, `trap rm -rf`), results to local first, then copy into
  `QUEUE_OUTPUT_DIR`; validate (`tar -tzf`, `grep -Fxq`) and write `.sha256` sidecars; write an
  identity/receipt JSON (commit, git status, launcher sha256, package versions, GPU, env) like
  `run_host_contracts.sh` does.
- Bound the job with `timeout --signal=TERM --kill-after=30s <secs>` inside the launcher as well; the
  queue's timeout is the backstop, not the plan.
- Resumable units: persist each finished unit (pattern `mass-partials/episode-NNNN.json` + manifest with
  digests, `Q0_PARALLEL_R5_STATUS_20261018.md`). A retry is a NEW job id + NEW output_dir that points at the
  old partials via env (`*_RESUME_PARTIALS` + `*_SHA256`), never a reuse of the old dir.

## Output dirs and attempts

- Unique per job and per attempt. Convention: `<study>/<kind>-<hw>r<N>` (`q0q1-20261017/q0/audit-b8-init-a100r3`);
  bump `rN` (and the id) for every new attempt. Lost runtimes (inactivity timeout) already forced r4/r5.
- `requeue` reuses the old output dir and increments generation/attempt; prefer a fresh id + dir unless the
  launcher is proven to resume safely from that dir. Dependents of a requeued job must be re-pointed.
- Receipts are immutable: `queue/done/<id>...` / `queue/failed/<id>...` record worker, GPU, commit,
  timing, exit code. Do not delete or edit them; inspect with `q status <id>`.

## q.py CLI (Mac, project venv; all output is JSON)

```bash
DRIVE='/Users/josenunez/Library/CloudStorage/GoogleDrive-josemnunez.jn@gmail.com/My Drive/t16_colab'
q() { /Users/josenunez/Projects/ml/snake-dqn/venv/bin/python research/colab_queue/q.py --drive-root "$DRIVE" "$@"; }
q enqueue path/to/job.json                      # validates + creates Drive dirs; one object per call
q list [--stale-seconds 300]                    # state per job (pending/claimed/done/failed)
q status [<id>] [--stale-seconds 300]           # claims, receipts, heartbeat ages
q watch --interval-seconds 10 [--once]
q cancel <id> --reason 'operator stop'          # fences the current generation; worker stops its process group
q requeue <id> --stale-owner-stopped --reason 'runtime verified stopped'   # only after proving the owner is dead
```

`q.py` defaults to that Drive root on this Mac (`COLAB_QUEUE_DRIVE_ROOT` env overrides). A stale heartbeat
alone is NOT proof the process died; requeue refuses a claimed job whose owner still heartbeats.
Implementers prepare manifests and print the exact `q enqueue` lines; the OWNER/operator runs them
(enqueueing a learning job is a spend decision).

## Worker notebooks

`notebooks/colab_queue_worker.ipynb` (+ `_a`, `_b`, `_c`: one file per concurrent runtime). Four cells run
in ORDER via Runtime > Run all: (1) `KEEP_RUNTIME` flag (`IDLE_MINUTES=100000` when True) + Drive mount
(owner click), (2) fetch `research/colab_queue/worker.sh` at `QUEUE_REF` from raw.githubusercontent.com,
(3) `nohup` start the worker, (4) blocking **keep-alive** cell that prints status every 600 s while the
worker lives (Colab disconnects idle notebooks; background nohup does not count). Heartbeats every 15 s;
idle release after 10 min of no eligible job (unless KEEP_RUNTIME) via `drive.flush_and_unmount` +
`runtime.unassign()`. GPU/CPU affinity are detected on the runtime, not trusted from the notebook type.
Never type multi-line heredocs into cells; commit a script and fetch it by SHA.

## Before you say done

1. `./venv/bin/python -m pytest -q tests/test_colab_queue.py tests/test_colab_queue_safety.py tests/test_colab_queue_worker.py`
   plus the tests for your launcher/adapter.
2. Validate each manifest's schema the way `q enqueue` would (`research/colab_queue/core.py` validators,
   or a `--dry-run`/generator check in your study package) without enqueueing; confirm `commit` is a full
   SHA that exists on `origin/main` (`git branch -r --contains <sha>`).
3. Run the launcher's real entrypoint on the Mac on a tiny real input with the manifest's env and a temp
   `QUEUE_OUTPUT_DIR` (Mac paths unavailable where possible); then run it AGAIN against the same dir to
   prove the fresh-attempt guard refuses.
4. State measured per-unit time, units, workers, and the resulting `est_minutes`/`timeout_minutes`.
5. In the handoff (see `snake-dqn-git-workflow`): manifest paths, dependency order, the exact `q enqueue`
   commands, which notebook/runtime type each job needs, and what you did NOT run.
