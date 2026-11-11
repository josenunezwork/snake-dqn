# Sandbox Job A GPU-Resident Throughput and Scale Assay (`A-1` -> `A-2` -> `A-3`)

**Dates:** 2026-10-26 to 2026-10-31
**Verdict:** Adopted the 6-clause Standing A100 Development Sandbox rule and passed all `24/24` `A-3` GPU-resident throughput cells on A100 (`21,133` to `28,113` compiled hero rows/s), while downstream `TRAIN-01` closed below its nomination bar (`+8.09` vs `>= +40` mass).

## What was tried
- Ratified the 6-clause Standing A100 Development Sandbox rule (`ADOPTED.md`, `SANDBOX_INPUT_REGISTRY.json`), letting development jobs run via a 10-line `sandbox.json` manifest with domain-separated `a100-dev/` seeds (`22,457,142` historical seeds audited) and a single-checkpoint crossing gate to Mac evaluation.
- Ran `A-1` (`6` cells: `E128` / `E1024` worlds × `cached` / `collection` / `integrated`, `128`-step rollouts, `256`-row minibatch) on A100 from `b8-s3` against `5x greedy_food` anchors to locate GPU-resident collection bottlenecks.
- Ran `A-2` (`12` cells, `2,048`-row minibatch, TF32) adding horizon-deferred truncation bootstraps, in-place masked resets, shared advisory masks, preallocated buffers, and staggered initial world ages `floor(e * episode_limit / E)`.
- Ran `A-3` (`24` cells across `E128` / `E1024` / `E2048` / `E4096`, both eager and `torch.compile`) adding vectorized reset food replenishment, batched 5-seat anchors, and prewarmed static `R2` live-window buckets (`256/512/768/1024`), then fed the runtime into `384`-update `TRAIN-01`.

## What happened
- **Governance (`ADOPTED`):** Froze sandbox input allowlists (`730` allowed D1024 fit lives and `136` D256 lives; `248` excluded parents from `29` census, `81` R30 validation, and `167` F4 fold-0 parents) and `simulator-D` output tagging.
- **`A-1` -> `A-2` progression:** In `A-1`, the fp32 learner hit `~30,000` rows/s while collection stalled at `493.3` (`E128`) and `1,862.3` (`E1024`) hero rows/s (`535.9` / `1,771.6` integrated). `A-2` raised cached learner speed to `~140,000` rows/s and `E1024` collection to `~2,700` hero rows/s.
- **`A-3` scale assay (`PASS`, `24/24` cells eager & compiled):** Integrated collection + PPO steady throughput on A100 reached `14,846` eager / `21,133` compiled hero rows/s at `E1024`, `21,936` / `25,710` at `E2048`, and `25,257` / `28,113` at `E4096` (`11.9x–15.9x` over `A-1`), with cached learner-only at `~140,000` rows/s.
- **Downstream `TRAIN-01` (`CLOSED`):** Completed `384` updates (`100,663,296` rows) at `+8.09` H5000 equal-mix mass (`CI95 [-18.66, +34.85]`; peak `+14.23` at update `320`) vs the `>= +40` bar (`0` strict-gate nominations).

## Why
`Ego2sNet` learner updates (`~140,000` rows/s) were never the limit; `A-1` stalled on per-tick host truncation syncs, unbatched 5-seat opponent loops, and serial reset food placement. Also, compiling dual `TF32=False` validation graphs (`A3_RESEAL02`) doubled compile time until `A-3` validated in the caller's single TF32 mode with prewarmed `256/512/768/1024` buckets.

## What to reuse / what not to repeat
- Reuse automated input allowlists (`SANDBOX_INPUT_REGISTRY.json`), `a100-dev/` seed-namespace checks (`22,457,142` sorted seeds), and the 1-checkpoint crossing gate to run fast GPU exploration without contaminating confirmation banks.
- Defer truncation bootstraps to rollout boundaries, stagger initial world ages, vectorize partial-reset food placement, and prewarm static `torch.compile` shape buckets (`256/512/768/1024`) in a single TF32 mode before timed loops.

## Full record
Original: `git show 90b3db94:docs/research/history/sandbox-a.md` | Primary sources: [ADOPTED.md](../governance_a100_sandbox_20261026/ADOPTED.md), [sandbox_a_resident/README.md](../../../research/sandbox_a_resident/README.md), [A3_DISPATCH_SUCCESSORS.md](../../../research/sandbox_a_resident/A3_DISPATCH_SUCCESSORS.md)
