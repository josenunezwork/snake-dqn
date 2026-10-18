---
name: snake-dqn-colab-jobs
description: |
  Use whenever you write, change or review code, launchers, job manifests or notebooks that will run on the owner's
  Google Colab runtimes (the Drive job queue research/colab_queue, research/q0q1_*, research/t16_colab, any
  run_*_colab.sh, setup_runtime.sh, colab_*worker*.ipynb) for /Users/josenunez/Projects/ml/snake-dqn. It lists the
  failure modes that already happened and the checks that prevent them. Mandatory pre-flight before declaring done.
---

# Snake DQN: Colab job checklist (learned the hard way, 2026-10-17/18)

The Colab runtime is NOT the Mac. Every item below is a real failure that cost hours or credits.

## Environment and inputs
- [ ] **No Mac-only paths or files.** Anything under `/Users/...`, `saved_snakes/` historical results, artifact
      inventories or strict-bank records is absent on Colab. If a job needs it, compute it on the Mac and ship a
      hash-bound receipt via Drive + job env (pattern: `q0q1-20261017/admission/inventory-receipt.json`).
      (Failures: bench `train_layout()` read a Mac checkpoint path; Q0 bank audit needed Mac strict inventories.)
- [ ] **GPU wheel matches the device.** Colab may substitute a G4 (RTX PRO 6000 Blackwell, sm_120) -> needs torch
      cu128; A100/L4/H100 use cu126. Pick from `nvidia-smi --query-gpu=compute_cap`.
- [ ] Python 3.12 uv venv (Colab kernel is 3.13; numpy 1.26.4 parity pins).
- [ ] Inputs come from the hash-checked T16 bundle (`/content/t16-colab-extracted/t16-colab-inputs`) or Drive.

## Throughput and time limits
- [ ] **Use the cores.** A runtime has 8-12 vCPUs. CPU-bound loops (episodes, evaluation worlds) must run in a
      process pool (`--workers`, default `min(cpu_count-1, 10)`) with deterministic per-world results merged in a
      fixed order; prove serial == parallel bytes with a test. (Failure: Q0 audit ran 480 H5000 episodes on one core
      and hit its 4 h timeout. T23 eval: one worker per job, B8 recomputed per checkpoint.)
- [ ] **Measure before setting est/timeout.** Time >= 3 real units on the Mac; Colab cores are ~2-3x slower than
      M-series per core. timeout >= 2x the measured estimate. Never guess.
- [ ] Skip work that cannot affect the decision (e.g. descriptive checkpoints) unless asked; put deciding work first.

## Observability and resumability
- [ ] **Stream progress** (done/total, elapsed, ETA) to stdout every ~2 min; never write the log only at the end.
- [ ] **Resumable units:** persist each finished unit; a retry must not redo finished work. Respect the launcher's
      "no prior/foreign artifacts in QUEUE_OUTPUT_DIR" guard: new attempts use fresh output dirs/ids.
- [ ] Write results to local disk first, then copy to Drive; flush Drive at the end.

## Drive and concurrency
- [ ] **Never create shared Drive dirs concurrently from several runtimes** - Drive makes duplicates
      (`evaluation` and `evaluation (1)`). Shared dirs are pre-created on the Mac by `q enqueue`.
- [ ] Claims use the queue's unique claim files; never shared marker files.
- [ ] Process-pattern matching (pgrep with -f) also matches its own command line; bracket a character
      (`worker[.]py`) or wait on a PID / output file.

## Runtime lifetime
- [ ] **Colab disconnects for inactivity when no cell executes** - background nohup workers do not count. Worker
      notebooks end with a blocking keep-alive cell that runs while the worker lives (now in
      `notebooks/colab_queue_worker*.ipynb`). Cells run in notebook ORDER.
- [ ] A new runtime = new VM: Drive re-mount (owner click), fresh setup.

## Shell typed into Colab cells
- [ ] No multi-line heredocs or tricky parentheses typed into cells (the editor auto-indents/auto-closes). Commit a
      script, fetch it by SHA from raw.githubusercontent.com, run it with one plain line.
- [ ] Never `files.download` (popup); results live on Drive.

## Before you say done
1. Run the job's real entrypoint on the Mac on a tiny but real input (not only unit tests) with the same env vars
   the manifest sets, and with `/Users` paths unavailable if possible (e.g. run from a temp copy).
2. State measured per-unit time, total units, workers, and the resulting est/timeout in your final message.
3. Validate the manifest with `research/colab_queue` and list the exact `q enqueue` commands.
