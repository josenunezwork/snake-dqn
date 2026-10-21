# E1b CUDA dtype recovery — 2026-10-21

Outcome: mechanical correction committed, targeted Mac verification passed, and
a fresh L4 manifest validated locally. **NOT DISPATCHED.** No Colab/provider calls.
No CUDA compilation, numerical execution, memory or throughput qualification is
claimed from this Mac. This sibling note preserves the original preflight bytes.

Implementation/source pin: `88adb3e27b0b7e13dbcbd9c57504335e65f8c90e`.

The failed L4 log is retained at
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1b/run-l4.log`.
It reached `v8_device._paint`, then failed with
`cuda_scatter_gather_base_kernel_func not implemented for 'Bool'`.
The CPU preflight and meta shape tests accepted this reduction. PyTorch 2.9.1's
[CUDA scatter/gather implementation](https://raw.githubusercontent.com/pytorch/pytorch/v2.9.1/aten/src/ATen/native/cuda/ScatterGatherKernel.cu)
includes Bool for assignment/gather, but its generic reduction dispatch at lines
334–346 excludes Bool. The audit also found the same invalid reduction in
`DeviceVectorRuntime.features`' vector61 free-space occupancy map.

Both maps now reduce UInt8 0/1 values with `amax`, then compare the map to zero.
This preserves Bool union semantics, duplicate cells, and invalid clamped-index
contributions. Existing independent CPU v8/vector feature parity tests still pass.

## Device-path audit

| Surface | Dtype/index contract reviewed |
| --- | --- |
| `v8_device` | Paint UInt8; release Int64; gather/scatter indices Int64; ranked Q Float32/Float64; Bool masks use logical ops/any/all, with numeric casts for argmax |
| `vector_device` | Occupancy UInt8; danger Float32 scatter add/max; Int64 indices; free-space consumes Bool after comparison |
| `device_sim` | Food map Int8 max; owner/tag maps Int64 max; Bool assignment/gather supported; Bool selection cast before argmax; counters/RNG Int64 |
| `ego_device` | TTL/release/count/coarse maps Int64; gather indices Int64; UInt8 raster output; Float32/Float64 scalar geometry |
| `masks_device` | Owner/offset rank Int64 max and Int64 gather indices; Bool logical masks |
| `anchors_device` | Both anchors; safe-list sum/cumsum promote to Int64; Bool selection cast before argmax; no UInt64 modulo |
| `device_policy`, `rollout`, `realistic_benchmark` | Full policy, branch copies, lifecycle/reset and collection paths; B8 forward checked on fake CUDA; index/reduction operands checked during targeted tests |
| Native frontier/components | Existing C++ guards require same CUDA device, contiguity, Int64 coordinates/indices/caps/release and Bool occupancy/alive; internal parent/count/queue/status Int32 with dimension bounds |

No further CUDA dtype/op gap was found in these E1b paths. Native argument
forwarding was independently checked with stubs; native execution remains untested.

## Regression checks and local preflight

`conftest.py` applies `CudaDtypeRules` to every package test, including CPU and
meta execution. It checks indexing widths, source/destination dtype equality,
numeric reduction whitelists, and rejects unreviewed scatter/index overloads.
The guard explicitly rejects the original Bool max reduction on CPU and meta.
An independent verifier loaded both historical modules from `9e7b0c0a` in memory
and confirmed that each old Bool reduction fails the guard.

The real resident-policy meta loop covers mechanics v1/v2 and both greedy-food
and random-safe anchors. The native wrapper stub checks the exact C++ typed ABI,
including the absent-release sentinel, without compiling CUDA. Duplicate/invalid
paint cells and invalid indexing/Bool argmax have focused regression checks.

Targeted package plus selected supervisor freshness/export checks:
**70 passed, 2 native-CUDA tests skipped, 77 unrelated deselected, 15.12 s**.
Black/isort/flake8 on changed Python files and `git diff --check` passed.
No broad suite was run.

The actual `colab.sh` entrypoint ran locally on CPU, one thread/process,
mode all, tiny dense D2 on three mixes, plus two resident CPU worlds per long-body
mix, six lengths 800–1500, one warmup and one measured tick, three repeats.
It finished with return code 0 in **67.267065 s** (outer wall **68.303232 s**).
AC was attached, lid open, no thermal warning recorded; host contended. Existing
tmux sessions `bl-e4`, `git-sync`, `phase0-selection` were left untouched.
This is a source/import/parity preflight, not a production-shape L4 result.

## Fresh prepared manifest

Artifact root:
`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e1b/cuda-dtype-fix-20261021-01/`.

| Payload | SHA256 |
| --- | --- |
| `l4-manifest.json` | `3634258b8e80255f7c0c4a7c8a73f90046e0bb3e3ddb6b6df1f52db269805d16` |
| `mac-entrypoint/output/benchmark.json` | `08a7c4fa78d9e71ac7d6e0ce733feafaa7625ab46d1817d6583ad6ed332bec60` |

The manifest binds the implementation pin, fresh preflight source hashes, and
all six unchanged checkpoint digests. `preparation-receipt.json` records them.
Pure validation and the actual supervisor `validate` command both passed locally;
the latter used `--colab /usr/bin/false` to fence accidental provider execution.
`supervisor-validation.json` retains its output with checksum/completion sidecars.

Fresh output: `<artifact root>/l4-output`, verified absent after validation.
Strict L4, minimum 12 vCPUs, 12 CPU reference workers, one native-library thread,
190-minute planning timebox, 400-minute whole-session wall cap, 180-minute shared
benchmark deadline, 13.5 u execution cap + 1.5 u reserve at 2 u/h, 600-second lease,
32 MiB export cap. These caps and every workload environment value except the
source commit were compared with the original L4 manifest and are unchanged.
Empty `other_assignments` remains a draft placeholder, not account inventory.

`KILL_RULE.md` remains byte-identical at
`a49aa11384a2636319d6eef80d6a38943bf759c7d14ea3804d39da0e5c17765a`;
`BASELINE.json` is unchanged. No registered verdict, retry, dispatch, serving or
release action follows from this preparation. No protected research package or
`research/colab_cli` was edited.
