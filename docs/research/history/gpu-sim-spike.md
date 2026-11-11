# GPU Simulation Engine Spike & A100 Speed Sprint (`gpu-sim-spike`)

**Dates:** 2026-10-20 to 2026-10-24
**Verdict:** Reached `6,003–7,661` continuous world-steps/s at `E=1,024` (`9,950` at `E=4096`) on A100 (`PASS_SPEED` vs `>= 2,000/s`) with `54/54` `T4A-DEV-01` batch-invariance replays (`0` mismatches), while GPU hindsight-label routes closed (`CLOSE_NO_BOUND` at `3.251x < 4.0x` vs Mac `W12` and `OUTCOME_B_BY_CONSTRUCTION` RNG/refill divergence).

## What was tried
- Built batched GPU simulator `DeviceSim` (`research/gpu_sim_spike/`) and native CUDA kernels (`flood_cuda`, `event_cuda`, `move_cuda`, `vector_geometry`, `v8_indexed`) targeting `>= 20,000` L4 world-steps/s (`E1b`) and `>= 2,000` A100 world-steps/s at `E=1,024` long-body (`800–1,500`) 6-snake worlds.
- Built Mac CPU comparison baseline `K-1` (`research/bl_k1`, native Numba `opponent_v8` inside branch rollouts) and wired it into `M1` served search.
- Profiled `A1-PROF` on A100-40GB and executed parallel CUDA optimization lanes: `A0` corpse-event kernel, `FLOOD` IEEE-754 round-to-nearest binary64 (`__d*_rn`), `MOVE2` indexed movement/respawn/collision, `V8` indexed geometry, `R2` live-length ring windows (`B = min(10400, round_up(max_len + 2, 256))`), and `R6` `E=4096` scaling.
- Evaluated the `Step -1` GPU vs Mac `W12` label-factory ruler (`>= 4x` bar) and `T4A-DEV-01` fixed-1024-row padded-batch invariance (`54` `H256` replays across `E=4/64/1024`).

## What happened
- **Mac `K-1` (`PASS`, `20.13x`):** Sped up in-branch `opponent_v8` `20.13x` (`>= 5.0x` bar), cutting `frozen` batch p50 `19.5–21.4 s -> 2.3–2.4 s` and lifting Mac `W1` `exact_stop` `0.611 -> 1.793` batches/s (`M1` live serving still failed on overruns -> `KEEP_B8`).
- **Early GPU spikes (`E1b` `NO_CONTINUE`, `A1` `STOP_ENGINE`):** `E1b` hit `0.217` world-steps/s on L4 (`1/18` decisions in `10,820 s`). `A0` (`event_cuda` + fixing a `17.14 GiB` broadcast OOM) sped up `E=6` `190.9x`, leaving `A1` at `158` / `163` / `166` steps/s (`6.17–6.48 s/tick` at `E=1,024`, `~77%` in `vector61` flood+geometry).
- **CUDA kernels:** `MOVE2` cut simulator step `0.86 -> 0.03 s` (`27.9–28.2x` / `28x`); `V8` cut `vector61-other` `2.412 -> 0.100 s` (`24x`); `FLOOD` reached `100%` CPU feature byte parity (`2.07x` scripted, `1.956x` frozen), giving integrated `I0S` `1,433 / 1,655 / 2,233` steps/s.
- **`R2` (`PASS_SPEED`) & `R6` (`DROP_E4096`):** `R2` live-length windows cut `E=1,024` ticks `72–75%` (`0.179 / 0.161 / 0.126 s`) to `6,003–6,013` (`frozen`), `6,593–6,596` (`mixed`), and `7,612–7,661` (`scripted`) steps/s; `R6` (`E=4096`) added `+13%` / `+17%` / `+21%` (`6,767` / `8,134` / `9,950` steps/s, `< 25%` bar).
- **Label-route closure & `T4A-DEV-01` (`PASS_T4A_DEV`):** `Step -1` measured `3.251` optimistic / `1.751` pessimistic vs Mac `W12` (`1,989.3` steps/s; `CLOSE_NO_BOUND`), with `OUTCOME_B_BY_CONSTRUCTION` RNG/refill differences. `T4A-DEV-01` passed `54/54` replays (`13,824` ticks, `0` mismatches).

## Why
`E1b` and `I0S` stalled on a `62,400`-iteration Python corpse loop, `32` redundant respawn occupancy builds, and `84.5%` kernel time scanning `cap=10,400` padded segment rings; live-length windows (`R2`) and indexed CUDA kernels eliminated those costs (`> 27,000x` over `E1b`), while fast Mac `W12` (`K-1` + `exact_stop`) left A100 below `4.0x` for CPU-matched hindsight labels.

## What to reuse / what not to repeat
- Bound GPU ring scans to live snake lengths (`B = min(10400, round_up(max_len + 2, 256))`), use IEEE-754 RN binary64 (`__d*_rn`, `--fmad=false --prec-div=true --prec-sqrt=true`) when matching CPU float32 features, and gate new `.cu` code with a short native GPU compile receipt.
- Reuse `K-1` (`20.13x`) on CPU for MT-exact `GridBatchSim` rollouts, and reserve `DeviceSim` (`R2 I'`) for GPU-resident training rather than cross-RNG CPU label factories.

## Full record
Original: `git show 90b3db94:docs/research/history/gpu-sim-spike.md` | Primary sources: [gpu-engine-and-speed.md](../../notes/gpu-engine-and-speed.md), [gpu_sim_spike/README.md](../../../research/gpu_sim_spike/README.md), [bl_k1/README.md](../../../research/bl_k1/README.md)
