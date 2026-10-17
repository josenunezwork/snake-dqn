# T16 Colab CUDA throughput smoke owner runbook (2026-10-17)

This is a bounded throughput smoke only. It is not an evaluation, promotion
gate, quality result, or serving qualification. Keep strict gates and serving
qualification on the Mac.

## Inputs to export and upload

On the Mac, create the one verified archive. The worktree intentionally does
not need to contain `saved_snakes`; the export script takes those immutable
checkpoint fixtures from its defined artifact locations.

```bash
./venv/bin/python research/t16_colab/bundle.py export \
  --output /Users/josenunez/Projects/ml/snake-dqn-artifacts/t16-colab/t16_colab_inputs.tar.gz \
  --repo /Users/josenunez/Projects/ml/snake-dqn
ls -lh /Users/josenunez/Projects/ml/snake-dqn-artifacts/t16-colab/t16_colab_inputs.tar.gz
```

The published archive is exactly **31,373,227 bytes** with outer SHA-256
229be2949c99f198b27decdf403b83ec4085d8df2dcfe629b7eff9208593be8e.
It has a second SHA-256 manifest for every payload. It contains only:

- `init/B8_s3_final.pth`
- `demo/margin.json`
- `demo/round-12/chunk-0000-part-00.npz`
- `checkpoints/frp3_m3_s12_u60000_20261005.pth`
- `checkpoints/champion_a5_freespace_20260621.pth`
- `checkpoints/best_apex_fs.pth`
- `checkpoints/best_apex_stage1_fs.pth`
- `checkpoints/best_apex_pre_fs.pth`
- `manifest.json`

Upload that exact `t16_colab_inputs.tar.gz` to Google Drive at
`MyDrive/t16_colab/t16_colab_inputs.tar.gz`. Do not unpack or alter it there:
the notebook checks the published archive SHA-256, then verifies every listed
internal SHA-256 before extracting it. The exporter is deterministic for the
same inputs; update the parameter and this runbook together only if a changed
input archive is deliberately published.

## Run the notebook

1. Open [notebooks/colab_throughput_smoke.ipynb](../../notebooks/colab_throughput_smoke.ipynb) in Colab, or use the direct [Colab notebook](https://colab.research.google.com/github/josenunezwork/snake-dqn/blob/main/notebooks/colab_throughput_smoke.ipynb).
2. Select **Runtime -> Change runtime type -> A100 GPU** (recommended; see the
   [review note](advice/2026-10-17_colab_review.md) for the runtime choice). H100
   and L4 also work. Colab's preinstalled CUDA torch is used when it is a 2.9.x
   build; otherwise the official CUDA 12.6 PyTorch 2.9.1 wheel is installed, following
   [PyTorch's previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
3. Confirm the parameter cell pins the commit named in the notebook's own
   `PINNED_COMMIT` (the review branch pins the reviewed runner commit) and uses
   the desired separate Drive run directory (the default is `smoke-01`). Do not
   reuse a run name with different code or input bytes.
4. Use **Runtime -> Run all** and complete the Drive authorization.

After Drive authorization, the notebook starts its 3,000-second budget (about
30 minutes of expected runtime; the runner refuses budgets above 3,600). It
creates an isolated `/content` checkout at the pinned public GitHub commit and
a fresh `venv --system-site-packages`, so it does not reuse imports cached in
the notebook kernel. It explicitly verifies Python 3.10--3.12, NumPy 1.26.4,
Numba 0.68.0, llvmlite 0.50.0, and a CUDA-enabled torch 2.9.x (the exact
version is recorded). It prints the
actual GPU, CPU count, CUDA, and package versions in the runner result.

The runner first executes the small grid-sim parity and ego-live identity
tests, then one unscored 60-second warm-up run (hero-only, 128 worlds, 2
actors) that fills the numba kernel cache and the inductor FX-graph cache so no
scored run pays first-process compilation. It then records six 150-second
timing configurations: hero-only B8 at 128 worlds with 2, 4, and 8 actors,
followed by all-snakes-learner self-play at 64 and 128 worlds with 2 actors and
at 128 worlds with 4 actors. The primary rate is the observed steady-state
interval: it opens at the later of 30 seconds and 20 seconds after the first
telemetry row with a learner update, and must span at least 30 seconds; the rate
over the last 60 seconds and the end-to-end rate are also recorded. A GPU/CPU
utilization sampler runs beside each timed configuration and its mean/max are
stored with the row, so an actor-bound run (high CPU, low GPU) can be told from
a GPU-bound one. The runner uses CUDA and requests `reduce-overhead`
compilation when a C compiler is present. Trainer outputs (telemetry, logs, the
final checkpoint) are written to local disk during a run and copied to the Drive
run directory afterwards; intermediate checkpoints are disabled for the smoke
(each is a synchronous save in the learner loop) and only the hash-receipted
final checkpoint is kept. The small results JSON is written to Drive directly.

If Colab disconnects or a run reaches the total budget, use **Run all** again
with the same run name. Completed rows are skipped. A hash-receipted checkpoint
can seed the next attempt, but it is a weights-only restart: optimizer, replay,
actor, target, RNG, and schedule state do not resume exactly.

## Download and compare

The Drive file `MyDrive/t16_colab/smoke-01/colab_throughput_results.json` is
the canonical resumable result. The last cell displays and downloads a current
invocation copy under the same filename; this avoids presenting stale successful
rows if a later setup or identity check fails. A setup failure produces that
filename locally and does not overwrite an existing Drive result.

Copy the downloaded file back to the Mac, then compare it with the measured
5,400 transitions/s Mac baseline:

The Drive file is the canonical resumable result. The notebook downloads a
current invocation copy under the same filename; this avoids presenting stale
successful rows if a later setup or identity check fails. A setup failure also
uses that filename locally and does not overwrite an existing Drive result.

```bash
./venv/bin/python research/t16_colab/compare.py colab_throughput_results.json
```

The comparison prints one conservative row per configuration. A row passes only
at or above 21,600 transitions/s (4x the Mac baseline); incomplete, failed, or
malformed rows are reported as invalid rather than passed. Self-play rows carry
a caveat: their transitions count every learner-driven snake, while the 5,400/s
baseline is the Mac's hero-only recipe, so that ratio is wall-clock per training
transition rather than a same-configuration comparison until a Mac self-play
rate is measured (a 150-second `--all-snakes-learner` run on the Mac).

## Validation performed locally

No Colab runtime or GPU timing was run during implementation. Local static and
unit verification completed: the 20 T16 bundle, comparison, runner, and
notebook tests; 41 trainer/actor tests; and two selected grid-sim and ego-live
identity tests (63 targeted tests total). Flake8, Black, isort, notebook JSON
parsing, and whitespace checks also passed.
