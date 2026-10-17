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
2. Select **Runtime -> Change runtime type -> H100, A100, or L4 GPU**. These
   GPUs are supported by the official CUDA 12.6 PyTorch 2.9.1 wheel used if
   Colab's installed torch is unsuitable. The pinned wheel source follows
   [PyTorch's previous-version installation instructions](https://pytorch.org/get-started/previous-versions/).
3. Confirm the parameter cell pins
   `281b6dcf4eb6d584d2a2d558e1deba2c273ef514` and uses the desired separate
   Drive run directory (the default is `smoke-01`). Do not reuse a run name
   with different code or input bytes.
4. Use **Runtime -> Run all** and complete the Drive authorization.

After Drive authorization, the notebook starts its 1,200-second budget. It
creates an isolated `/content` checkout at the pinned public GitHub commit and
a fresh `venv --system-site-packages`, so it does not reuse imports cached in
the notebook kernel. It explicitly verifies Python 3.10--3.12, NumPy 1.26.4,
Numba 0.68.0, llvmlite 0.50.0, and CUDA-enabled torch 2.9.1. It prints the
actual GPU, CPU count, CUDA, and package versions in the runner result.

The runner first executes the small grid-sim parity and ego-live identity
tests. It then records five 150-second timing configurations: hero-only B8 at
128 worlds with 2, 4, and 8 actors, followed by all-snakes-learner self-play
at 64 and 128 worlds with 2 actors. The primary rate is the observed interval
after a 30-second warmup and spanning at least 30 seconds; the end-to-end rate
is also recorded. The runner uses CUDA, requests `reduce-overhead` compilation
when a C compiler is present, writes checkpoints to Drive every 100k
transitions, and writes results directly to Drive.

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
malformed rows are reported as invalid rather than passed.

## Validation performed locally

No Colab runtime or GPU timing was run during implementation. Local static and
unit verification completed: the 20 T16 bundle, comparison, runner, and
notebook tests; 41 trainer/actor tests; and two selected grid-sim and ego-live
identity tests (63 targeted tests total). Flake8, Black, isort, notebook JSON
parsing, and whitespace checks also passed.
