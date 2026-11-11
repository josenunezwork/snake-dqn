# Simulators and Speed

<a id="perf-20261013"></a>
## CPU, Simulator & Learner Performance Optimization (`perf-20261013` / `simd-vector61` / `perf-sim` / `S0`–`S2`)

**Dates:** 2026-09-26 to 2026-10-21
**Verdict:** Bit-exact `simd-vector61` (`3.65x`), live `perf-sim` (`2.10x` strict eval, `2.19x` actor), `--learner-fast` + `--batch-scale 2` (`1.57x` MPS learner), Numba `vector61` (`5.3–5.5x` long-body actor), and `S2` native Numba `v8` (`1.8994x` teacher round) were adopted, while MPS `torch.compile` and stacked multi-seed learners (`perf_multiseed`) were rejected (`NO-GO`).

### What was tried
- Built batched `Vector61Featurizer` + `Vector61SimdPolicy` (`simd-vector61`, `v5`/`v7`/`v8` hooks) and tested parity against live `tournament_eval.rollout` across `~66,000` decisions and `48` `H5000` episodes.
- Optimized live Python `GameState` (`perf-sim`, `git 68c0efe`) with integer squared-radius checks, per-snake geometry caches (`snake_geometry.py`), `3x3`/`7x7` bucketed queries, and flood-fill reuse, verifying identity across `42` strict `H5000` episodes (`210k` frames, `730k` rows).
- Benchmarked M5 Pro MPS and RunPod RTX 4090 (`perf_*_20261013`) across `--learner-fast` (`t1`), `--batch-scale 2` (`k2`), `torch.compile`, `--v61-backend numba`, `--obs-pack`, stacked `groups=K` / `vmap` multi-seed learners, and a `20M`-step B8 `K1` vs `K2` A/B.
- Profiled `S0` 6-world `H300` teacher rounds (`speed_s0_baseline`) and built opt-in Numba-native `v8` opponent veto `S2` (`t34_speed_s2/native_v8.py`), plus compared Mac vs L4 `S0` float32 exports (`L4_MISMATCH.md`).

### What happened
- **`simd-vector61` & live `perf-sim` (bit-exact):** `simd-vector61` matched **`48/48` `H5000` records** (`3.65x`: `367.5 s` vs `1,341.1 s`) and `12/12` `v5` records (`126.3 s` vs `251.3 s`; `v7`/`v8` H5000 prepared, not run). `perf-sim` sped up strict `H5000` eval **`2.10x`** (`111 -> 233` eps/h/core) and FRP-v3 actors **`2.19x`** (`145 -> 317` rows/s) with identical bytes across `42` episodes.
- **Learner & actor flags (`ADOPTED`):** On M5 Pro MPS, `t1 + --batch-scale 2` raised learner speed **`1.57x`** (`3,658 -> 5,730 tr/s`; `20M` `K2` scored `537.5` vs `K1` `527.3`, `+10` within `SE` `27`–`28`, in `1.49 h` vs `2.10 h` = `1.41x`), and `--v61-backend numba` sped up long-body actors **`5.3–5.5x`** (`1,184 -> 6,264` at `E=64`).
- **Compile & multi-seed (`NO-GO` on MPS):** CUDA `reduce-overhead` compile gave `1.74x` (`15,170 tr/s` on RTX 4090; EPYC actor cores `~2.4x` slower than M5 Pro), whereas MPS `torch.compile` gave wrong conv-bias gradients and stacked multi-seed learners were `NO-GO` (`0.45x`–`0.86x` `vmap`, `3–16%` grouped over separate processes).
- **`S0` & `S2` (`D3 PASS`, `1.8994x`):** `v8` took `56.8%` of `S0`'s `103.1 s` round (`67.0%` on `long-mixed`); `S2` `native_v8` cut median round time **`98.534 -> 51.878 s` (`1.8994x`, `-47.35%`)** with `144/144` exports byte-identical on Mac (`Mac vs L4` differed by `<= 7.62939453125e-6` in `4,443 / 10,800` Qs, `0` root action diffs, backend cause unproven).

### Why
Float square-roots and Python `v8` flood-fills (`56.8%` of `S0` teacher time; `36.3 ms` vs `5.38 ms` `ego` at `L=1,200`) dominated CPU stepping until replaced by integer geometry caches and Numba kernels, while MPS learners were kernel-launch bound (`< 1.5 TFLOP/s`) with broken Torch `2.9.1` Inductor conv-bias gradients.

### What to reuse / what not to repeat
- Keep `snake_geometry.py` integer caches (`2.10x` live eval), `--learner-fast --batch-scale 2 --v61-backend numba --obs-pack`, and opt-in `native_v8` (`1.8994x`), all verified against byte-exact trajectory/digest harnesses.
- Do not enable `torch.compile` on Apple MPS (conv-bias gradient bugs) or build in-process `vmap`/grouped multi-seed learners (`< 1.5x` vs separate OS processes), and do not assume Mac-vs-x86 FP32 Q-bytes are bit-identical (`7.629e-6` drift).

### Full record
Original: `git show 90b3db94:docs/research/history/perf-20261013.md` | Primary sources: [simd_vector61_plan_2026-09-26.md](../simd_vector61_plan_2026-09-26.md), [perf_quiet_20261013/RESULTS.md](../../../research/perf_quiet_20261013/RESULTS.md), [RESULT_D3_MAC_CONTENDED_01.md](../../../research/t34_speed_s2/RESULT_D3_MAC_CONTENDED_01.md)

<a id="gpu-sim-spike"></a>
## GPU Simulation Engine Spike & A100 Speed Sprint (`gpu-sim-spike`)

**Dates:** 2026-10-20 to 2026-10-24
**Verdict:** Reached `6,003–7,661` continuous world-steps/s at `E=1,024` (`9,950` at `E=4096`) on A100 (`PASS_SPEED` vs `>= 2,000/s`) with `54/54` `T4A-DEV-01` batch-invariance replays (`0` mismatches), while GPU hindsight-label routes closed (`CLOSE_NO_BOUND` at `3.251x < 4.0x` vs Mac `W12` and `OUTCOME_B_BY_CONSTRUCTION` RNG/refill divergence).

### What was tried
- Built batched GPU simulator `DeviceSim` (`research/gpu_sim_spike/`) and native CUDA kernels (`flood_cuda`, `event_cuda`, `move_cuda`, `vector_geometry`, `v8_indexed`) targeting `>= 20,000` L4 world-steps/s (`E1b`) and `>= 2,000` A100 world-steps/s at `E=1,024` long-body (`800–1,500`) 6-snake worlds.
- Built Mac CPU comparison baseline `K-1` (`research/bl_k1`, native Numba `opponent_v8` inside branch rollouts) and wired it into `M1` served search.
- Profiled `A1-PROF` on A100-40GB and executed parallel CUDA optimization lanes: `A0` corpse-event kernel, `FLOOD` IEEE-754 round-to-nearest binary64 (`__d*_rn`), `MOVE2` indexed movement/respawn/collision, `V8` indexed geometry, `R2` live-length ring windows (`B = min(10400, round_up(max_len + 2, 256))`), and `R6` `E=4096` scaling.
- Evaluated the `Step -1` GPU vs Mac `W12` label-factory ruler (`>= 4x` bar) and `T4A-DEV-01` fixed-1024-row padded-batch invariance (`54` `H256` replays across `E=4/64/1024`).

### What happened
- **Mac `K-1` (`PASS`, `20.13x`):** Sped up in-branch `opponent_v8` `20.13x` (`>= 5.0x` bar), cutting `frozen` batch p50 `19.5–21.4 s -> 2.3–2.4 s` and lifting Mac `W1` `exact_stop` `0.611 -> 1.793` batches/s (`M1` live serving still failed on overruns -> `KEEP_B8`).
- **Early GPU spikes (`E1b` `NO_CONTINUE`, `A1` `STOP_ENGINE`):** `E1b` hit `0.217` world-steps/s on L4 (`1/18` decisions in `10,820 s`). `A0` (`event_cuda` + fixing a `17.14 GiB` broadcast OOM) sped up `E=6` `190.9x`, leaving `A1` at `158` / `163` / `166` steps/s (`6.17–6.48 s/tick` at `E=1,024`, `~77%` in `vector61` flood+geometry).
- **CUDA kernels:** `MOVE2` cut simulator step `0.86 -> 0.03 s` (`27.9–28.2x` / `28x`); `V8` cut `vector61-other` `2.412 -> 0.100 s` (`24x`); `FLOOD` reached `100%` CPU feature byte parity (`2.07x` scripted, `1.956x` frozen), giving integrated `I0S` `1,433 / 1,655 / 2,233` steps/s.
- **`R2` (`PASS_SPEED`) & `R6` (`DROP_E4096`):** `R2` live-length windows cut `E=1,024` ticks `72–75%` (`0.179 / 0.161 / 0.126 s`) to `6,003–6,013` (`frozen`), `6,593–6,596` (`mixed`), and `7,612–7,661` (`scripted`) steps/s; `R6` (`E=4096`) added `+13%` / `+17%` / `+21%` (`6,767` / `8,134` / `9,950` steps/s, `< 25%` bar).
- **Label-route closure & `T4A-DEV-01` (`PASS_T4A_DEV`):** `Step -1` measured `3.251` optimistic / `1.751` pessimistic vs Mac `W12` (`1,989.3` steps/s; `CLOSE_NO_BOUND`), with `OUTCOME_B_BY_CONSTRUCTION` RNG/refill differences. `T4A-DEV-01` passed `54/54` replays (`13,824` ticks, `0` mismatches).

### Why
`E1b` and `I0S` stalled on a `62,400`-iteration Python corpse loop, `32` redundant respawn occupancy builds, and `84.5%` kernel time scanning `cap=10,400` padded segment rings; live-length windows (`R2`) and indexed CUDA kernels eliminated those costs (`> 27,000x` over `E1b`), while fast Mac `W12` (`K-1` + `exact_stop`) left A100 below `4.0x` for CPU-matched hindsight labels.

### What to reuse / what not to repeat
- Bound GPU ring scans to live snake lengths (`B = min(10400, round_up(max_len + 2, 256))`), use IEEE-754 RN binary64 (`__d*_rn`, `--fmad=false --prec-div=true --prec-sqrt=true`) when matching CPU float32 features, and gate new `.cu` code with a short native GPU compile receipt.
- Reuse `K-1` (`20.13x`) on CPU for MT-exact `GridBatchSim` rollouts, and reserve `DeviceSim` (`R2 I'`) for GPU-resident training rather than cross-RNG CPU label factories.

### Full record
Original: `git show 90b3db94:docs/research/history/gpu-sim-spike.md` | Primary sources: [gpu-engine-and-speed.md](../../notes/gpu-engine-and-speed.md), [gpu_sim_spike/README.md](../../../research/gpu_sim_spike/README.md), [bl_k1/README.md](../../../research/bl_k1/README.md)

<a id="sandbox-a"></a>
## Sandbox Job A GPU-Resident Throughput and Scale Assay (`A-1` -> `A-2` -> `A-3`)

**Dates:** 2026-10-26 to 2026-10-31
**Verdict:** Adopted the 6-clause Standing A100 Development Sandbox rule and passed all `24/24` `A-3` GPU-resident throughput cells on A100 (`21,133` to `28,113` compiled hero rows/s), while downstream `TRAIN-01` closed below its nomination bar (`+8.09` vs `>= +40` mass).

### What was tried
- Ratified the 6-clause Standing A100 Development Sandbox rule (`ADOPTED.md`, `SANDBOX_INPUT_REGISTRY.json`), letting development jobs run via a 10-line `sandbox.json` manifest with domain-separated `a100-dev/` seeds (`22,457,142` historical seeds audited) and a single-checkpoint crossing gate to Mac evaluation.
- Ran `A-1` (`6` cells: `E128` / `E1024` worlds × `cached` / `collection` / `integrated`, `128`-step rollouts, `256`-row minibatch) on A100 from `b8-s3` against `5x greedy_food` anchors to locate GPU-resident collection bottlenecks.
- Ran `A-2` (`12` cells, `2,048`-row minibatch, TF32) adding horizon-deferred truncation bootstraps, in-place masked resets, shared advisory masks, preallocated buffers, and staggered initial world ages `floor(e * episode_limit / E)`.
- Ran `A-3` (`24` cells across `E128` / `E1024` / `E2048` / `E4096`, both eager and `torch.compile`) adding vectorized reset food replenishment, batched 5-seat anchors, and prewarmed static `R2` live-window buckets (`256/512/768/1024`), then fed the runtime into `384`-update `TRAIN-01`.

### What happened
- **Governance (`ADOPTED`):** Froze sandbox input allowlists (`730` allowed D1024 fit lives and `136` D256 lives; `248` excluded parents from `29` census, `81` R30 validation, and `167` F4 fold-0 parents) and `simulator-D` output tagging.
- **`A-1` -> `A-2` progression:** In `A-1`, the fp32 learner hit `~30,000` rows/s while collection stalled at `493.3` (`E128`) and `1,862.3` (`E1024`) hero rows/s (`535.9` / `1,771.6` integrated). `A-2` raised cached learner speed to `~140,000` rows/s and `E1024` collection to `~2,700` hero rows/s.
- **`A-3` scale assay (`PASS`, `24/24` cells eager & compiled):** Integrated collection + PPO steady throughput on A100 reached `14,846` eager / `21,133` compiled hero rows/s at `E1024`, `21,936` / `25,710` at `E2048`, and `25,257` / `28,113` at `E4096` (`11.9x–15.9x` over `A-1`), with cached learner-only at `~140,000` rows/s.
- **Downstream `TRAIN-01` (`CLOSED`):** Completed `384` updates (`100,663,296` rows) at `+8.09` H5000 equal-mix mass (`CI95 [-18.66, +34.85]`; peak `+14.23` at update `320`) vs the `>= +40` bar (`0` strict-gate nominations).

### Why
`Ego2sNet` learner updates (`~140,000` rows/s) were never the limit; `A-1` stalled on per-tick host truncation syncs, unbatched 5-seat opponent loops, and serial reset food placement. Also, compiling dual `TF32=False` validation graphs (`A3_RESEAL02`) doubled compile time until `A-3` validated in the caller's single TF32 mode with prewarmed `256/512/768/1024` buckets.

### What to reuse / what not to repeat
- Reuse automated input allowlists (`SANDBOX_INPUT_REGISTRY.json`), `a100-dev/` seed-namespace checks (`22,457,142` sorted seeds), and the 1-checkpoint crossing gate to run fast GPU exploration without contaminating confirmation banks.
- Defer truncation bootstraps to rollout boundaries, stagger initial world ages, vectorize partial-reset food placement, and prewarm static `torch.compile` shape buckets (`256/512/768/1024`) in a single TF32 mode before timed loops.

### Full record
Original: `git show 90b3db94:docs/research/history/sandbox-a.md` | Primary sources: [ADOPTED.md](../governance_a100_sandbox_20261026/ADOPTED.md), [sandbox_a_resident/README.md](../../../research/sandbox_a_resident/README.md), [A3_DISPATCH_SUCCESSORS.md](../../../research/sandbox_a_resident/A3_DISPATCH_SUCCESSORS.md)
