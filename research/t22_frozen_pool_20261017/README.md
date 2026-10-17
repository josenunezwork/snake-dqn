# T22 frozen-pool CPU microbenchmark

`bench_frozen_pool.py` runs a bounded single-threaded real-pool actor-frame
microbenchmark. It loads the six SHA-pinned checkpoints from an explicit
directory, excludes a configurable JIT warm-up, records executable source and
checkpoint identities, and hashes the complete action stream. `--reference`
executes the dispatcher source from git `4c1e1b2`; the current mode exercises
the checkout under test.

Final evidence is the same-command pair
`results_final_reference.json` and `results_final_current.json`: 200 warm-up
frames, 30 measured frames, 128 worlds, three learner seats, and one CPU
thread. Their complete action-stream SHA-256 values both equal
`b02339badab1dc36d12d76d4ec13ebb100f268e062f5eb38621588b2af726b14`.
The executed baseline source is `4c1e1b2` / SHA-256
`18c3b2e7257af0410069d0f344f3bbe37dbee0f08d5a8e53938b87667575919c`.

The final Mac receipt reports dispatch at 41.047 ms/frame before and 28.774
ms/frame after (29.9% lower). The live v8 wrappers account for 11.79 ms/frame
of the baseline policy wall time and 0.70 ms/frame after. The optimized actor
also reduces vector61 runtime-controlled rows from 384 to 192; its post-step
carry timing fell from 2.274 to 1.673 ms/frame. B8 CPU forward is reported
separately because it runs on the A100 in the target setup: it was 26.52 and
26.98 ms/frame on this Mac and must not be treated as host work saved on CUDA.

For the A100 a4/a6 reports, a direct Amdahl projection applying the whole 29.9%
dispatch reduction to the observed 66.7% frozen share gives approximately
7,420 / 9,650 transitions/s. A deliberately optimistic CUDA-aware estimate
first removes the Mac B8 CPU forward from the ratio, retains the measured 6.2 s
GPU frozen-forward time, and applies the remaining 87.7% host-frozen reduction:
about 14,130 / 18,370 transitions/s. Neither reaches 20,900 transitions/s;
these are projections, not Colab execution evidence.

The phase values are nested wall timings. In particular, `all_b_format` includes
the shared learner/B8 raster construction, and the B8 v8 feature kernel is
nested in `v8_scalars`; do not add those fields together.
