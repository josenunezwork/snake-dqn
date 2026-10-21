# Mac Y1 and L1′ measurements — 2026-10-21

**Nominated cell: none. L1′ verdict: NOT_READY(INSTRUMENT_MISSING).**

These are Mac measurements. The requested production reachable-root provider is absent; the stress roots have no valid production consumption path. No serving or release is authorized.

## Y1 Mac dense batches — contended inventory

Receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/y1-mac-01/mac-ruler.json`; SHA256 `502f1eb67a783309fb638c595bcc20e9b4a34aa0810b0f0985ab90ad73ce6f22`. 270 actual batches, CPU W=1, 30 per fixture. Quantiles use nearest rank. All 30 include the first cold sample; warm-29 values are retained in the receipt. Operator timing excludes guard/export overhead; raw ruler wall timing retains it.

The full run began after an idle sample but overlapped foreign research workers. It is not a quiet-host acceptance. Quiet attempt 02 refused before data; final attempt 03 stopped after five batches when analyticsd used 63.7% of one core. No samples were spliced. **Y1 quiet-host acceptance: NOT_READY.**

| Fixture | n | p50 (s) | p95 (s) | p99 (s) | max (s) |
|---|---:|---:|---:|---:|---:|
| frozen-1200 | 30 | 22.749 | 23.311 | 23.344 | 23.344 |
| frozen-1500 | 30 | 21.221 | 25.814 | 26.296 | 26.296 |
| frozen-800 | 30 | 10.557 | 10.607 | 10.637 | 10.637 |
| mixed-1200 | 30 | 10.823 | 10.925 | 10.971 | 10.971 |
| mixed-1500 | 30 | 10.332 | 10.382 | 10.394 | 10.394 |
| mixed-800 | 30 | 7.603 | 7.642 | 7.663 | 7.663 |
| scripted-1200 | 30 | 0.649 | 0.681 | 0.683 | 0.683 |
| scripted-1500 | 30 | 0.595 | 0.626 | 0.633 | 0.633 |
| scripted-800 | 30 | 0.611 | 0.616 | 0.635 | 0.635 |

## Per-cell stress diagnostics

S below is measured admission to first decoded coordinator visibility for frozen stress jobs. It is distinct from valid consumption latency C. CPU Watch6 game updates continue on the absolute 24fps schedule while workers run. Each cell retains per-length and per-burst results. K8 is a configured capacity; actual pending counts are reported separately. Cold worker initialization is included. These samples cannot qualify registered L1′ tails or coverage.

| Cell | n | S p50 (s) | S p95 (s) | S p99 (s) | S screen cap (s) | Max pending | Overruns burst/all | Update p95 (ms) | Disposition |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| cpu-w1-k1 | 6 | 19.721 | 22.934 | 22.934 | — | 1 | 280/344 | 39.744 | NOT_READY; C unavailable |
| cpu-w1-k8 | 6 | 20.009 | 22.580 | 22.580 | 1.458 | 1 | 287/359 | 40.389 | NOT_READY; MISS |
| cpu-w2-k8 | 12 | 20.777 | 23.858 | 23.858 | 0.830 | 2 | 312/400 | 42.845 | NOT_READY; MISS |
| cpu-w3-k8 | 18 | 22.070 | 25.565 | 25.565 | 1.250 | 3 | 338/449 | 44.808 | NOT_READY; MISS |
| cpu-w4-k8 | 24 | 23.688 | 27.240 | 27.244 | 1.458 | 4 | 390/533 | 48.271 | NOT_READY; MISS |
| mps-w1-k1 | 6 | 32.269 | 34.851 | 34.851 | — | 1 | 291/456 | 37.087 | NOT_READY; C unavailable |
| mps-w1-k8 | 6 | 32.063 | 34.600 | 34.600 | 1.458 | 1 | 296/453 | 37.406 | NOT_READY; MISS |
| mps-w2-k8 | 12 | 34.022 | 37.015 | 37.015 | 0.830 | 2 | 315/505 | 38.701 | NOT_READY; MISS |
| mps-w3-k8 | 18 | 34.775 | 37.863 | 37.863 | 1.250 | 3 | 340/554 | 41.039 | NOT_READY; MISS |
| mps-w4-k8 | 24 | 37.611 | 41.198 | 41.202 | 1.458 | 4 | 364/623 | 30.405 | NOT_READY; MISS |

The governing coverage bar is 2.4 roots/s (at least 29 valid consumptions per 12s and 36 per 15s). The 1.6 sensitivity requires at least 20/24. X8 has not authorized the easier bar. These probes observed zero eligible consumptions because they have no production root provider; registered coverage is NOT_MEASURED, rather than an estimate of zero production capability.

W1/K1 C p99 ceilings are 333/542 ms (unrounded contract values 333.333/541.667 ms). Expanded S p99 ceilings: W2 0.830s, W3 1.250s, W4 1.458s; W1/K8 retains the 1.458s tail cap. Zero overruns and Watch6 update p95 ≤11.688ms are required. Raw diagnostic S is not C. A matching stress result supplies neither exact-state consumption nor a production PASS.

The diagnostic inventory plans six trials per cell: three lengths × 12/15-second burst windows, with one cold submission wave per trial. Counts are small; nearest-rank p99 is the maximum at these sample sizes. The full registered repetitions and four modes were not run. Ticks continue beyond the burst until results are decoded. The table separates overruns inside the burst from all recorded overruns. Update p95 pools observed updates, including reduced load after terminal hero death, and cannot qualify timing.

Worker receipts attest actual CPU/MPS placement, synchronization and thread counts. CPU workers used one PyTorch thread; MPS initialization selected four CPU threads (interop one). MPS still uses CPU simulation, features, vetoes and export. These identities and cold-start costs are retained, with no backend speedup inference.

## Evidence and limits

- Cells receipt: `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-l1-20261021/cells-01.json`; SHA256 `b8983ab8d978e3d59ce7dac14699a0f09a4126ce5c96eaf64ae77ad22b3c6a8f`.
- Owner grants and canonical scoped probe authorization: `owner-authority.json`, `PROBE_ONLY.json`.
- Phase-0 and quiet snapshots: `phase0-idle*.json`, Y1 host snapshots, and `root-occupancy/`.
- First Y1 run overlapped local bl_info_value/reconstruction workers; its receipt and all samples remain retained. A separately completed quiet repetition, if present, is never spliced with it.
- `y1-mac-02-quiet/stopped.json` records a quiet-only attempt refused before any batch because mediaanalysisd/Spotlight exceeded the foreign-load ceiling. All completed and stopped attempts remain retained.
- `quiet-recovery-03-plan.json` was sealed before the final attempt. `y1-mac-03-quiet/stopped.json` and its five batch receipts preserve the final foreign-load guard stop; no additional retry was made.
- Independent raw execution audit: `cells-independent-audit.json`; source/contract readiness review: `contract-review.json`; Y1 full inventory audit: `y1-mac-01/audit.json`; source-bound synthetic clock-only proof: `clock-preflight-01/clock.json`. The synthetic proof grants no production qualification.
- Complete controller/replay/114-root semantic acceptance, two repetitions in all four invalidation/start modes, saturated K8 trace, valid consumption C, and client latency/cadence remain unmeasured. Terminal-hero fixtures can lose hero inference after death; raw per-tick alive state is retained.
- No Colab call, serving change, checkpoint/pin edit, prohibited-study access, or process/tmux termination occurred.
