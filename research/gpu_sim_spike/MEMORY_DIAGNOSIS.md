# A1 vector61 A100-40 OOM diagnosis

The failed `bl-a1-main` receipt records an allocation request of 17.14 GiB at
the first `E=1024` cell. It is not a retained `E=256` world or a dense BFS
plane. The exact request is `vector_device.py`'s per-action obstacle distance
coordinate subtraction:

```
delta = next_cells[:, :, None, None] - body[:, None]
```

At the A1 maximum resident shape this creates an int64 tensor with shape
`[E*S, 3, S, capacity, 2] = [6144, 3, 6, 10400, 2]`:

```
6144 * 3 * 6 * 10400 * 2 * 8 = 18,402,508,800 bytes = 17.138671875 GiB
```

The preceding danger-map `bdelta` is `[6144, 6, 10400, 2]` int64, or
5.712890625 GiB. At the failure site the old Python frame also retained the
row body, `bdelta`, float distance, long sector, float values, masks, food
one-hot, and the full body view. Their source-level storage subtotal is
18.97 GiB before the failed request; transient square/reduction tensors,
simulator/model state, and CUDA allocator blocks make the receipt's 21.24 GiB
allocated plausible. This is a liveness diagnosis, not evidence that precisely
21.24 GiB is a leak. In particular, the receipt does not show an `E=256` state
surviving into `E=1024`: the ruler releases every repeat, and its failure came
before the `E=1024` cell completed. A single BFS bool plane is only `[1024, 6, 3, 83,
145]`, 0.206594467 GiB, so it cannot be the request. The receipt's 6.16-GiB
reserved-but-unallocated portion supports allocator fragmentation as a
contributing condition.

`expandable_segments:True` is necessary allocator hygiene but cannot repair
this shape by itself. Even if it had admitted the 17.14-GiB `delta` request,
the following `delta.square()` needs another same-sized int64 tensor before
the reduction can release it: `21.24 + 17.14 + 17.14 = 55.52` GiB as a lower
bound, above the 40-GiB device. The receipt distinguishes 21.24 GiB allocated
by PyTorch, 6.16 GiB reserved but unallocated, and about 0.50 GiB non-PyTorch
process use; it does not establish 27.90 GiB of live tensors.

`DeviceVectorRuntime.features` now splits only row-local geometry reductions
into ordered blocks of 64 rows. The world remains `E=1024` throughout: policy,
food, simulator state, and results are never partitioned. The largest direct
replacement is `[64, 3, 6, 10400, 2]` int64 = 0.178527832 GiB. The CPU-only
size tool conservatively budgets four simultaneous copies of each long
reduction family plus the two principal float distances at 1.011657715 GiB per
row block. It does not claim a measured GPU peak.

The free-space flood is deliberately computed once per `features` invocation,
outside that row loop. It therefore preserves the previous one flood per
feature call rather than multiplying the full `E x S x 3` BFS by 96 chunks.
The source-level repaired plan totals 7.52 GiB: two resident body buffers
(simulator and reset template), one full body view, the 1.01-GiB conservative
row block, eight flood planes for expression overlap, and an explicit 2-GiB
B8/CUDA-workspace allowance. It excludes opaque library workspaces and is not a
measured peak; the ruler's per-cell CUDA peak remains the fit authority.
Run the non-allocating arithmetic report with:

```sh
./venv/bin/python -m research.gpu_sim_spike.memory_sizing
```

The ruler still records the actual CUDA peak memory for each cell. Runtime
dispatch must set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`; this
reduces allocator fragmentation but is not relied on as the correctness fix.

`test_vector_device.py` imports the pre-change module frozen from
`16778391221c09e87658bbd50bd18b5650134b72` (SHA-256
`6d32148935b9cccd3d2d2e7bdf1093a2abf9a88349d3e46e48907bb738a4d69d`).
It compares output plus trend carry on shuffled, unique, masked rows spanning
three chunks, including a dead row. Explicit duplicate trend-memory rows retain
the old unchunked path because CUDA advanced-index write order is not stable
across separate kernels; A1 generates unique `E*S` rows.
