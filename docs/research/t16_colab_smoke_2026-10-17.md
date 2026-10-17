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
   and L4 also work. The isolated Python 3.12 environment installs the official
   CUDA 12.6 PyTorch 2.9.1 wheel, following
   [PyTorch's previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
3. Confirm the parameter cell pins the commit named in the notebook's own
   `PINNED_COMMIT` (updated for the T16b Python 3.12 fix) and uses
   the desired separate Drive run directory (the default is `smoke-01`). Do not
   reuse a run name with different code or input bytes.
4. Use **Runtime -> Run all** and complete the Drive authorization.

After Drive authorization, the notebook starts its 3,000-second budget (about
30 minutes of expected runtime; the runner refuses budgets above 3,600). It
creates an isolated `/content` checkout at the pinned public GitHub commit and
a fresh isolated venv using **uv-managed Python 3.12**, regardless of the
notebook kernel version (including Colab's Python 3.13). The kernel only handles
Drive authorization, orchestration, and displaying/downloading results. Setup
runs `python -m pip install uv`, `python -m uv python install 3.12`, then
`python -m uv venv --python 3.12 --managed-python --seed /content/t16-colab-venv`.
It verifies the selected interpreter is an isolated Python 3.12 before installing
NumPy 1.26.4, Numba 0.68.0, llvmlite 0.50.0, and CUDA 12.6 torch 2.9.1. Pins are
unchanged; no kernel packages are inherited. Allow several minutes within the
existing budget for the Python download and CUDA wheel installation.

Every bundle, parity, warm-up, trainer, and utilization-sampler Python subprocess
uses that venv interpreter. Results JSON records `python` and `python_executable`,
and resume identity includes the Python version alongside code and package pins.
Setup failures record the interpreter details once selection succeeds and also
record `kernel_python`. Runner results print the actual GPU, CPU count, CUDA,
and package versions. Use a new run name when moving an old run to this fixed
commit; a different code/runtime identity is deliberately refused.

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

## T16b Python 3.12 repair validation

The Python 3.13 kernel rejection is removed. Mocked tests exercise managed 3.12
selection under 3.10, 3.12, and 3.13 kernels, reject a wrong or non-isolated
interpreter, and ensure uv failures do not fall back to the kernel. The actual
notebook orchestration is executed with mocked external commands to check package
pins and interpreter routing. Runner tests cover parity, warm-up, timed training,
and sampler routing plus interpreter metadata on failure. The steady-state window,
unscored warm-up, local scratch, utilization sampler, and 3,000-second notebook
budget are retained. This repair does not claim a successful Colab/GPU run.

All 32 T16 tests passed locally for this repair, as did Black, flake8, notebook
syntax/JSON checks, and whitespace checks. Independent code review found no
actionable defects. No Colab runtime, GPU job, or training experiment was started.
