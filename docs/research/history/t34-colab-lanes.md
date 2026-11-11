# T34 Parallel Colab Lanes, 1,160-Unit Compute Portfolio, and Speed-vs-Constraints Rulings

**Dates:** 2026-10-20
**Verdict:** Three independent judge rounds ratified the gated `L0–L9 + Reserve` Colab portfolio (`~1,002 u` GO-branch ceiling + `~90 u` `L0`), bound Round 0 routing to `$E/amendment-boost-floor/decision.json`, approved the split-stream Claim A (`71de7645`) and straggler-charged `Round 0-D` (`9730fed2`) amendments, and launched the `S0–S4` parallel CPU speed program (`DO NOT STOP` active runs).

## What was tried
- Audited `A5 #2` live L4 throughput (`2` workers × `5` threads, `256` roots) vs `W1` `12`-process fanout (`1` world/process) to price true teacher cost across `~1,154–1,160` prepaid Colab Pro+ units (`1,153.84 u` at `19:18 PT`).
- Allocated gated Colab lanes (`L0–L9 + Reserve`), adjudicated which Round 0 `decision.json` governed `SUCCESS_CONTRACT.md` §5, and reviewed pre-data fixes for Claim A (`L1 D0` / `L2 R1` / `L3`) and `Round 0-D` (`L7` straggler makespan charging).
- Evaluated whether to pause T34 for a GPU simulator rewrite or relax parity/hashing/replay constraints (`t34-speed-vs-constraints`).

## What happened
- **Teacher placement & unit cost:** `A5 #2` stalled at `~30–31` row-steps/s/lane (`0/256` worlds in `1,500 s`); after the `1.7139×` digest fix (`6.1605 → 3.5945 ms`) and `12` 1-thread `W1` processes/L4, `A5` hit **`57.2` and `54.4` worlds/lane-h** (`31–36` row-steps/s per L4 vCPU vs `208` on `1` Mac M5 Pro core = `1/6` Mac core, **`~0.036 u/teacher world`** at `1.54 u/h`).
- **Gated `L0–L9` portfolio:** Ordered `L1 D0` (`12 u`, split-stream `128` decisive + `896` anchor batch after v1 flipped `0.4%` of `225` rescues) and `L3` (`120 u`) ahead of bulk labels (`L2` `60 u` for `800 + 200` worlds, `L4` `150 u`, `L5` `280 u`, `L6` `80 u`, `L7` `120 u + 200 u`, `L8` `100 u`, `L9` `100 u`, `Reserve` `100 u`; expected `~500 u` on `R0 GO`, `~195 u` if premises fail).
- **Exclusions & Round 0 routing:** Excluded `T35` (`KILL`), `A100/H100/G4`, and 5-seed checkpoint studies, and bound Round 0 routing exclusively to `$E/amendment-boost-floor/decision.json`.
- **Speed vs constraints (`DO NOT STOP`):** Kept active T34 runs unchanged, rejected GPU sim rewrites (sim was `5.7%` Mac / `12.3%` x86 step time vs `35.5%` digest + `22.2%` `v8` = `57.7%`), and launched `S0` baseline, `S1` flat `full_digest v4`, `S2` `v8_nb` (`nogil`), and `S4` multi-session Mac pacing design.

## Why
Multi-threaded `256`-root workers contended on host CPU and cold-fill latency (`1` L4 vCPU `≈ 1/6` M5 Pro core), `full_digest` (`35.5%`) is active controller logic in `RollingShadow` rather than a removable receipt, and `24 fps` paced Mac runs sleep `~75%` of each tick (wall-clock bound, not engine bound).

## What to reuse / what not to repeat
- Price L4 vCPUs at `~1/6` of a Mac M5 Pro P-core (`31–36` vs `208` row-steps/s) and charge multi-worker job budgets by wave-quantized straggler makespan (`setup + ceil(n/12)·max(slowest, wave)`).
- Optimize `full_digest` (`S1`) and Numba `v8` (`S2`) with exact transition parity rather than subsampling controller state digests, and never spend unallocated credits to pad killed hypotheses.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-colab-lanes.md` | Primary sources: [2026-10-20_colab_lanes/JUDGMENT.md](../advice/2026-10-20_colab_lanes/JUDGMENT.md), [2026-10-20_speed_vs_constraints/JUDGMENT.md](../advice/2026-10-20_speed_vs_constraints/JUDGMENT.md), [CLAIMA_AMENDMENT_JUDGMENT.md](../advice/2026-10-20_colab_lanes/CLAIMA_AMENDMENT_JUDGMENT.md)
