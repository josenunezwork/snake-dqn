# T18 H100 RunPod throughput smoke

This is the archive-only RunPod substitute for the T16/T17 Colab smoke when Drive sign-in
is unavailable. It runs the four P0 learner cells first, then the five-minute-per-row
`smoke03` matrix. All pod lifecycle, spend, watchdog, dead-man, ledger, and
delete-and-verify behavior remains in `research/perf_podbench_20261013/pod_bench.py`.

`prepare.py` is invoked by that runner after `/r/repo` is extracted and the pinned Python
dependencies are installed. It verifies the uploaded input archive against the pinned outer
SHA-256, calls `research.t16_colab.bundle.verify_and_extract`, insists on the expected
`/r/in/t16-colab-inputs` extraction root and a usable `gcc`, then writes
`/r/out/prepare_receipt.json`. The receipt records the GPU model(s), visible vCPU count and
CPU model, plus available `/dev/shm` capacity. The count is the process affinity
mask, not a cgroup CPU quota. It also calls `run_smoke.environment_info()` before P0: that
executes a CUDA tensor operation and validates Torch 2.9.x plus the exact NumPy, Numba, and
llvmlite pins, preventing a CUDA-unavailable P0 skip from looking successful. It fails before
extraction when less than 12 GiB is available for the P0 mmap-ring matrix. The committed P0
defaults allocate 995,920,016 bytes for two raw 40,000-row BV8 actor rings (264,800,016
bytes packed); 12 GiB is a conservative admission floor recorded with the exact calculated
values in the receipt. The runner pulls every file under `/r/out`, including this receipt.

The setup must install the T16 parity pins into the pod environment: Python 3.12, NumPy
1.26.4, Numba 0.68.0, llvmlite 0.50.0, CUDA 12.6 PyTorch 2.9.x, plus pytest, psutil, and
PyYAML used by the runner’s environment/parity checks. It must then invoke:

```sh
cd /r/repo
/r/venv/bin/python research/t18_h100pod_20261017/prepare.py \
  --bundle /r/in/t16_colab_inputs.tar.gz --destination /r/in --output /r/out
```

Dry run (no RunPod API call and no spend):

```sh
./venv/bin/python research/perf_podbench_20261013/pod_bench.py bench --dry-run \
  --gpu-type 'NVIDIA H100 80GB HBM3' --cloud secure --max-hourly 4 --min-vcpu 16 \
  --lifetime-hours 1.5 --spend-cap 8 \
  --pod-prefix perfbench-t18- --job-id perf-cuda-bench-t18-h100-20261017 \
  --t16-bundle /Users/josenunez/Projects/ml/snake-dqn-artifacts/t16-colab/t16_colab_inputs.tar.gz \
  --commands research/t18_h100pod_20261017/commands.json
```

After the dry-run is reviewed and the user has explicitly approved the spend, run the same
command with `--confirm` in place of `--dry-run` and an explicit local output directory:

```sh
./venv/bin/python research/perf_podbench_20261013/pod_bench.py bench --confirm \
  --out-dir /Users/josenunez/Projects/ml/snake-dqn-artifacts/t18-h100pod-20261017 \
  --gpu-type 'NVIDIA H100 80GB HBM3' --cloud secure --max-hourly 4 --min-vcpu 16 \
  --lifetime-hours 1.5 --spend-cap 8 \
  --pod-prefix perfbench-t18- --job-id perf-cuda-bench-t18-h100-20261017 \
  --t16-bundle /Users/josenunez/Projects/ml/snake-dqn-artifacts/t16-colab/t16_colab_inputs.tar.gz \
  --commands research/t18_h100pod_20261017/commands.json
```

Do not add `--accept-no-self-delete`: this workflow requires the pod to retain its
self-delete path and the runner must delete and verify deletion on every exit. If the pod
credential probe is not HTTP 200, the runner deletes the pod before uploads or benchmarks.
The dry run does not establish that this provider-side prerequisite is available.

The four P0 commands each have a 300-second maximum and are deliberately first. Together
with the smoke's 3,600-second internal budget they can exceed the useful portion of the
1.5-hour hard lifetime after setup and the 10-minute job-deadline margin. This is intentional priority
ordering rather than a promise that every row will fit: the hard deadline stops remaining
commands and preserves already-pulled results. Do not shorten the learner benchmark's
measured rounds without a separately registered methodology change.

The H100 profile's $4/hour ceiling is an admission limit, not a current price quote.
The plan reserves $6.46 (1.5 hours plus the existing five-minute watchdog grace and 2%
ledger margin), with a balance-drop guard at $7.50 under the $8 cap. The pod receives its
self-delete epoch at 1.5 hours; the independent Mac watchdog retains its existing grace.
No shared project/floor/spend-ledger policy file is changed by this GPU-specific allowance.
Cloud choices and the CUDA driver filter use the [RunPod REST API schema](https://rest.runpod.io/v1/openapi.json).

The current source already lifts `policy_lag_mean` through `measured_rate()` into completed
rows. The added regression exercises real telemetry aggregation through `compare_results()`
and requires the valid completed control row to pass. Old result files are not rewritten.
