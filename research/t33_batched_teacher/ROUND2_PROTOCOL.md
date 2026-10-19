# T33 round-two development protocol

Authorized by the owner's 2026-10-19 request and round-two spec. This sibling
supersedes the execution and descriptive calibration details of protocol.md;
original evidence and Phase 0 bytes stay unchanged. No release or remote work.

The single-run CRN reference records cumulative mass and next-boundary legal
max-Q and mean-Q at every requested prefix. N subsets use the first N streams.
The independent-action Phase 0 oracle remains a separate full run. Prefixes
must match separate CRN searches at identical parameters before timing claims.

Leaf units are segment-frames per raw network-Q unit. For each (H, leaf), fit
a nonnegative through-origin least-squares slope from live prefix Q to remaining
full-horizon mass, pooling action/rollout samples on calibration decisions only.
Report scale 0, scale 1, and that fitted slope, in-sample agreement/regret and
sample counts; no automatic recipe selection or out-of-sample quality claim.
Dead-prefix Q and remainder are zero. Uniform reservoir sampling retains at most the requested number of prepared
snapshots during one bare replay (fixed seed 20261019); it changes the sample
indices from the older two-pass rng.choice sampler. Cache eligible/selected
index inventories with
world/bank/checkpoint/source identities, not inferred filesystem state.

Native pools are capped at four locally, with one thread per concurrent lane.
Microbenchmarks use three warmed rounds on the same fixtures and report rows/s.
Lockstep profiles use 256/1024 sources, N4/H32 (a timing budget, not a selected
label recipe), admission before allocation, and <=4 Numba threads with Torch
and BLAS at one. Shape refusals and short-run limits are disclosed. Predictions
for one/three L4 hosts are conditional on explicit CPU scaling assumptions.
CUDA fp16 trunk with fp32 dueling head and A/B overlap remain separately
qualified on actual hardware. GPU cropping is dropped per Fable review.
