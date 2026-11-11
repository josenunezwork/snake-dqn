# Big-Lever E3 Full-Board/History Information-Value Probe and Round-31 Bounded Future-Noise Audit (`bl-info-value`)

**Dates:** 2026-10-21 to 2026-10-24
**Verdict:** `E3` (`bl_info_value`) ended `SMOKE_COMPLETE / PREPARED, NOT DISPATCHED` after three Colab attempts were repaired with `1,954,150 B` Mac action tapes (no production run or scientific GO/KILL in this record), and Round-31 (`bl_future_noise`) closed cleanly at `STOP_AT_A0` (`BLOCKED_MISSING_CLEARANCE`, `0` admitted parents, `0` Colab units).

## What was tried
- Built `E3` (`research/bl_info_value`) on `256` unsealed T3 worlds (`1,536` roots: `768` decisive, `768` counterweight; `192/32/32` split), parameter-matched at `270,522` parameters across three students.
- Tested whether full `145 × 83` 6-plane occupancy + unclipped tail release (`Student B`) or a 128-decision 64-unit GRU (`Student C`) cuts D300 mass regret `>= 20%` at `<= 1%` counterweight switches vs `b8-s3` inputs (`Student A`).
- Repaired three successive `E3` Colab L4 failures across Recoveries `02`–`04`: `57,831,386 B` upload timeout, Colab exec RPC timeouts, and a Mac-vs-x86 float32 Q-tie action divergence at decision `2916`.
- Built and ran the Round-31 Future-Noise Audit (`research/bl_future_noise`), testing whether frozen rescue actions still beat bare actions across four independent CRN future-RNG suffixes (`A2`), gated by a pre-data `>= 64`-parent / `>= 16`-pair cleared-archive census (`A0`).

## What happened
- **`E3` Recoveries 02–03:** Attempt `01` (`334.09 s`, `0.2 u`) timed out uploading `57,831,386 B` (`57.8 MB`), fixed in Recovery `02` via a `360,881 B` catalog (`160x` smaller) + asset `bl-e3-inputs-20261021` (`51,787,546 B`). Attempt `02` (`1,445.43 s`, `33/256` worlds, `1.1 u` cumulative) hit RPC timeouts, fixed in Recovery `03` (`2` reserved vCPUs, `nice >= 10`, scratch outside `exports/`).
- **`E3` Mac action tapes (`SMOKE_COMPLETE / PREPARED, NOT DISPATCHED`):** Attempt `03` (`2,318.74 s`, `2.4 u` cumulative) stopped `INCOMPLETE` on an x86 root mismatch at decision `2916`. Recovery `04` recorded Mac action tapes (`1,954,150 B` v3 archive in `588.08 s`), reproducing `2916` in `18.80 s` and cutting smoke `60.73 s -> 25.74 s` (`1.31 h / 2.62 u` L4 forecast; not dispatched here).
- **Round 31 (`STOP_AT_A0`):** Passed `16/16` boundary tests (`0.21 s`) and stopped at `A0` (`0.048 s`, `0` admitted parents, `0 u`) because no separately cleared, unsealed historical archive with `>= 64` parents and `>= 16` same-mix pairs existed.

## Why
Re-running neural policies closed-loop on Colab x86 to reconstruct Mac-selected trajectory roots failed at decision `2916` because arm64 vs x86 float32 Q-value ties flip greedy actions; earlier attempts blew the upload cap (`57.8 MB`) or starved the Colab Jupyter RPC loop under unreserved vCPU saturation and `exports/` scratch churn.

## What to reuse / what not to repeat
- Record compact all-seat Mac action tapes (`trace_bank.py`, `1,954,150 B` / `< 2 MB` for `256` worlds) whenever x86 workers must reconstruct exact Mac-selected states without float32 Q-tie drift.
- Reserve `2` vCPUs (`resources.py`), keep per-world scratch files outside the watched `exports/` directory, and fetch large static checkpoint bundles via HTTPS rather than `57.8 MB` CLI uploads.
- Enforce fail-closed pre-data archive clearance checks (`A0`) before spending Colab compute on historical root banks.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-info-value.md` | Primary sources: [bl_info_value/README.md](../../../research/bl_info_value/README.md), [PROTOCOL.md](../../../research/bl_info_value/PROTOCOL.md), [bl_future_noise/README.md](../../../research/bl_future_noise/README.md)
