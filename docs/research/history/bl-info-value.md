# Big-Lever E3 Full-Board/History Information-Value Probe and Round-31 Bounded Future-Noise Audit (`bl-info-value`)

## Question / Goal

Evaluate two fundamental premise questions for learning from H300 counterfactual search labels on bare `b8-s3`:
1. **E3 Information Value (`research/bl_info_value`):** On 256 unsealed T3 worlds (`1,536` roots: `768` decisive, `768` counterweight; `192/32/32` train/val/test split), does adding full $145 \times 83$ 6-plane board geometry + unclipped tail release (`Student B`, `270,522` params) or a 128-decision 64-unit GRU history encoder (`Student C`, `270,522` params) reduce held-out decisive D300 mass regret by $\ge 20\%$ over `b8-s3`-only inputs (`Student A`, `270,522` params) with $\le 1\%$ counterweight switches?
2. **Round 31 Future-Noise Audit (`research/bl_future_noise`):** How much do H300 counterfactual action rankings on mature ($L \ge 800$) B8 roots vary across 4 independent future food-spawn and opponent-tie RNG suffixes (`A2`), gated by a pre-data 64-parent / 16-pair unsealed archive clearance check (`A0`) and a `-9.1` oracle parity fixture (`A1`)?

## Timeline of Versions / Steps

| Date | Step / Package | What Was Run / Changed | Verdict / Outcome |
| :--- | :--- | :--- | :--- |
| 2026-10-21 | `research/bl_info_value/PROTOCOL.md` & initial Mac preflight (`d85d75af`) | Froze E3 protocol (`bl-e3/v1`, sha256 `4e84360f...`), verified exact parameter parity (`270,522` params each for Students A/B/C), and ran 3-world/6-root Mac smoke in `89.58 s` with a `57.8 MB` v1 input archive | `SMOKE_COMPLETE` (v1 archive timed out uploading on Attempt 01) |
| 2026-10-21 | `UPLOAD_RECOVERY_PREFLIGHT.md` (Recovery 02) | After Attempt 01 timed out uploading the `57,831,386 B` v1 archive at `334.09 s` (`0.2 u`), split inputs into a `360,881 B` local catalog (`54eaa6a8...`) plus a `51,787,546 B` checkpoint bundle (`dfdef34f...`) fetched from GitHub release `e3-assets-20261021` | `SMOKE_COMPLETE` (Attempt 02 later timed out on Colab exec RPCs) |
| 2026-10-21 | `RECOVERY_03_PREFLIGHT.md` (Recovery 03) | After Attempt 02 failed `INCOMPLETE` at `1,445.43 s` (`33/256` worlds reconstructed, `1.1 u` cumulative) due to 100% vCPU saturation starving Colab exec RPCs, added `resources.py` CPU affinity reservation (`2` reserved vCPUs, `nice >= 10`) and moved per-world scratch out of `exports/` (`60.73 s` Mac smoke) | `SMOKE_COMPLETE` (Attempt 03 later hit Mac/x86 float32 tie divergence) |
| 2026-10-21 | `RECOVERY_04_PREFLIGHT.md` & `research/bl_info_value/README.md` | After Attempt 03 failed `INCOMPLETE` at `2,318.74 s` (`2.4–2.6 u` cumulative) when world `2059218604956791243` diverged at decision `2916` on an x86 vs Mac float32 Q-tie, built `trace_bank.py` to record all-seat Mac action tapes (`1,954,150 B` v3 archive, sha256 `9878576d...`, recorded in `588.08 s`), cutting Mac smoke to `25.74 s` and L4 forecast to `1.31 h / 2.62 u` | `SMOKE_COMPLETE` / Prepared (`big-lever/exp/bl-e3/recovery-04`) |
| 2026-10-24 | `research/bl_future_noise/INTENT.md`, `README.md` & A0 gate | Froze Round 31 registration (`bl-r31-future-noise-v1`, sha256 `77488efd...`), passed 16 synthetic boundary tests (`0.21 s`) and Codex-V A0 light review (`03eb280e...`), and ran `research.bl_future_noise.census` (`a0-preflight-03/receipt.json`, `0.048 s` wall) | `STOP_AT_A0` (`BLOCKED_MISSING_CLEARANCE`, `0` admitted parents, `0 u` spent) |

## Final Verdict and Key Numbers

- **E3 equal-capacity student architectures (`research/bl_info_value/PROTOCOL.md`, `research/bl_info_value/README.md`):** Students `A` (B8 inputs only), `B` (+ $145 \times 83$ 6-plane full-board occupancy and unclipped tail release), and `C` (+ 128-decision 64-unit GRU history) are matched at exactly **`270,522` trainable parameters** each (`Mac batch-32 update median`: `A 0.0107 s`, `B 0.0558 s`, `C 0.0710 s`; `batch-192 inference`: `A 0.0274 s`, `B 0.1791 s`, `C 0.1846 s` in `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e3/preflight-01/timing.json`).
- **E3 transport & x86 determinism repairs (`research/bl_info_value/UPLOAD_RECOVERY_PREFLIGHT.md`, `research/bl_info_value/README.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e3/recovery-04/`):**
  - **Attempt 01 (`334.09 s`, `0.2 u`):** Timed out uploading the `57,831,386 B` v1 archive; fixed in Recovery 02 by shrinking the CLI upload **`160x`** to a **`360,881 B`** v2 catalog (`54eaa6a8...`) plus GitHub release asset `e3-assets-20261021` (`51,787,546 B`, `dfdef34f...`).
  - **Attempt 02 (`1,445.43 s`, `33/256` worlds, `1.1 u` cumulative):** Timed out on Colab exec RPCs when worker pools saturated 100% of vCPUs and flooded `exports/`; fixed in Recovery 03 by reserving 2 vCPUs in `resources.py` and moving per-world scratch out of `exports/` (`60.73 s` Mac smoke in `big-lever/exp/bl-e3/preflight-03/timing.json`).
  - **Attempt 03 (`2,318.74 s`, `2.4 u` cumulative):** Stopped `INCOMPLETE` when frozen-opponent world `2059218604956791243` hit a Mac vs x86 float32 Q-tie at decision `2916`; fixed in Recovery 04 by recording all-seat Mac action tapes across all `256` worlds / `1,536` roots (`588.08 s` Mac recording, **`1,954,150 B`** v3 archive sha256 `9878576dadd291668c8c809343b05422c7291de1050b3338d106105a16e2bb8c`, `18.80 s` byte-exact replay on the decision-2916 regression in `big-lever/exp/bl-e3/recovery-04/decision-2916-verification.json`, and **`25.74 s`** Mac smoke / **`1.31 h` (`2.62 u`)** L4 forecast).
- **Round 31 Future-Noise Audit (`research/bl_future_noise/INTENT.md`, `research/bl_future_noise/README.md`, `/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/round-31/a0-preflight-03/receipt.json`):** Stopped cleanly at **Stage A0 (`STOP_AT_A0` / `BLOCKED_MISSING_CLEARANCE`)** with **`0` admitted parents** and **`0` Colab units spent** (`receipt.json` sha256 `fb9d5c76b5d12a3d65efeb8b5759167da405f6a71710d28caa527a2cc8e121cc`, `CODEX_V_A0.json` sha256 `03eb280e4df12f3236f7e81c0e126d25753f08c140f41ddc6602ae8ed4ffd2df`, `16/16` synthetic tests in `0.21 s`) because no separately cleared, unsealed historical archive with $\ge 64$ parents and $\ge 16$ same-mix pairs existed.

## What Was Learned

1. **Cross-platform root reconstruction:** Re-evaluating neural policies closed-loop on x86 to reconstruct Mac-selected trajectory roots fails on rare float32 Q-value ties (such as decision 2916 of world `2059218604956791243`). Recording compact all-seat action tapes (`< 2 MB` for 256 worlds) guarantees byte-identical starting roots on any CPU/GPU architecture while cutting reconstruction time by more than half.
2. **Colab CLI runtime hygiene:** Saturating all vCPUs or writing hundreds of intermediate per-world scratch files into the watched `exports/` directory starves the Colab Jupyter kernel RPC loop; reserving 1–2 vCPUs (`resources.py`) and keeping scratch outside `exports/` became standard across subsequent Colab packages.
3. **Fail-closed archive gates:** Enforcing the pre-registered A0 metadata clearance gate (`bl_future_noise`) prevented spending compute on unsealed or non-disjoint historical roots, while the future-noise question itself was later addressed directly by $4 \times 4$ common-random-number (CRN) rollout branching in `t34-ct-1028`.

## Why It Closed / Current Status

- **`bl_info_value` (E3):** Recovery 04 (`big-lever/exp/bl-e3/recovery-04`) was prepared and verified on Mac (`SMOKE_COMPLETE`, `NOT_DISPATCHED`), and its representation/topology question was superseded by the larger GPU-native `t34-ct-1028` probe (`frozen_phi` vs `full_bv8` vs `full_bv8_plus_topology`).
- **`bl_future_noise` (Round 31):** Closed at **`STOP_AT_A0`** per the Round 31 ruling (`f17130be...`: *"If no eligible archive exists, preserve the preparation and stop at A0"*).

## Kept Files in This Lineage

- [`research/bl_future_noise/INTENT.md`](../../../research/bl_future_noise/INTENT.md)
- [`research/bl_future_noise/README.md`](../../../research/bl_future_noise/README.md)
- [`research/bl_info_value/PROTOCOL.md`](../../../research/bl_info_value/PROTOCOL.md)
- [`research/bl_info_value/README.md`](../../../research/bl_info_value/README.md)
- [`research/bl_info_value/UPLOAD_RECOVERY_PREFLIGHT.md`](../../../research/bl_info_value/UPLOAD_RECOVERY_PREFLIGHT.md)
