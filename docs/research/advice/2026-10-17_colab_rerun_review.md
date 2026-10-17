# Advice: pre-rerun review of the T16 Colab smoke after smoke-01 (2026-10-17)

Advisory note with code (cloud advisor), requested by the owner through the local session.
Branch `colab-opt2-20261017`; main untouched. Inputs: main at `6366d50` (runner `2cc5e57`,
uv Python 3.12 environment `c5157ae`), `research/t16_colab/{run_smoke,sampler,compare,
environment}.py`, the notebook, `train_m3c.py` (actor spawn, self-play path, summary), the
smoke-01 report from the local session.

## The smoke-01 failure was mine

The `ModuleNotFoundError: No module named 'research'` came from the in-process
`from research.t16_colab.sampler import summarize` I added to `run_smoke.py`, a script the
notebook launches by path, so the repo root was not on `sys.path`. The local session's fix
(`2cc5e57`, the `sys.path` insert, verified under `python -I`) is the right one. I should
have run the runner by path, as the notebook does, before pushing; the test suite imports it
as a module and so could not catch it.

## What I reviewed and found sound

- **Sampler process.** Launched with `-m research.t16_colab.sampler` and `cwd=REPO`, so the
  package resolves; `psutil` comes from `requirements-gpu.txt`; `nvidia-smi` is present on
  Colab GPU runtimes and its absence is handled. It is stopped in a `finally`, so a trainer
  timeout or interrupt cannot leave it running.
- **Per-config subprocesses.** Owned process groups, 150 s grace around `--max-wall-seconds`,
  group kill after exit; the trainer creates its own `--out` after the runner cleared it.
- **Self-play under CUDA.** Actors are started with the `spawn` context, so CUDA in the
  learner before actor start is safe; the advisory mask, v8 features, exploration draws and
  n-step rows are all indexed by `(env, slot)` for every live snake; the label buffer is a
  ring that overwrites, so 6x label volume cannot overflow; the ring capacity is sized for
  `worlds x 6`; `--mixed-frac 0.25` is ignored on the self-play path by construction.
- **Schema and compare.** `measured_rate` fields are finite by construction; `compare.py`
  handles the six rows and the self-play caveat; Python's `json.loads` accepts the runner's
  output either way.
- **Budget on 12 vCPUs.** uv bootstrap plus the cu126 torch wheel, parity, the 68 s warm-up
  and six timings at about 3.5 minutes each (startup, 150 s, copy) come to roughly 30 to 33
  minutes. The 8-actor hero row and the 4-actor self-play row oversubscribe 6 physical cores;
  that is the measurement, not a fault, and the `oversubscription` field records it.
- **Environment helper.** uv is bootstrapped from the kernel, Python 3.12 is managed and
  verified isolated, torch comes from the cu126 index because the 3.13 kernel wheel cannot be
  reused; all under the notebook's deadline.

## What I changed

1. **A trainer that dies before creating its output directory no longer aborts the run.**
   `copy_attempt` raised `FileNotFoundError` on a missing source, which the outer handler
   turned into a top-level `failed` status, discarding every other row. It now returns
   `False` for a missing source and the row is marked failed with its log kept.
2. **Drive publication is best effort.** A Drive FUSE error (`Input/output error` is common on
   long Colab sessions) during the copy no longer fails the row or the run: the copy retries
   twice, the row records `published: true|false`, and the measurement (summary, `log.jsonl`,
   final checkpoint hashes) is read from the local copy when publication failed. A later VM
   reset then re-runs that row rather than losing the whole smoke.
3. **The embedded trainer summary is sanitized** for strict JSON: any non-finite float becomes
   `null`, so a NaN in a trainer field cannot trip `allow_nan=False` and fail the run after a
   good measurement.
4. **Notebook budget 3,300 s** (runner cap 3,600), to cover the uv and torch setup that
   smoke-01 showed, plus the warm-up and six timings, in one Run all.

Tests: three new tests (best-effort copy with a simulated I/O error, non-finite sanitizing,
and a run where one trainer dies before creating its output while the others complete), the
notebook contract updated; 27 T16 tests pass locally with black, isort and flake8 clean. The
notebook pins the runner commit that carries these changes; open it from this branch, or
from main after the merge.

## Not changed, noted

- The self-play rows compare against the hero-only Mac baseline; the caveat is in the report.
  A 150 s `--all-snakes-learner` run on the Mac remains the one follow-up before reading them.
- If smoke-02 still dies in a way these guards do not catch, the invocation result on Drive
  and the per-attempt logs under the run directory (or, unpublished, under
  `/content/t16-colab-scratch/<config>/`) hold the trainer's own stderr.
