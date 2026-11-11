# Big-Lever E16 Execution-Ranking Harness and Round-07 X6 Campaign Governance

**Dates:** 2026-10-21
**Verdict:** Closed at `BUILD_VERIFIED / NOT_DISPATCHABLE` (`UNUSED` under `PLAN_v3`, `0` Colab runs) after passing Mac continuation-parity preflights and a `24,876`-seed freshness audit while upstream `E2-F`, `E3`, and critic prerequisites remained unmet.

## What was tried
- Built `E16` (`research/bl_e16`, `2M` PopArt critic + `4M` PPO actor transitions from `E2-F`) comparing Arm `A` (natural inputs) vs Arm `G` (7-plane `_RasterTower(7,16,64,(6,10))` E3 geometry + optional `GRU(43, 64)` over `128` steps) across `288` paired H5000 worlds (`96/mix`, `LB90(J) > 0`, `point G >= 10`, `>= 5/100` rescues/mix, `>= 90%` tail retention, `UB90 <= 10`).
- Built Round-07 `X6` machine-checked governance ledgers (`admission.py`): campaign look ledger (`LOOK_LEDGER.md`), append-only world-burn manifest (`BURN_MANIFEST.md`), 4-receipt `E13` parity rule (`PARITY_RULE.md`), and 3-slot `alpha=0.05` Holm final confirmation family (`FINAL_FAMILY_REGISTRATION.md`).
- Audited `24,876` 48-bit high signed-int64 seeds against `1,167,085` historical identities (`120,177` files) and ran Mac serial/parallel and boundary-restore parity preflights (`preflight-01`..`03`).

## What happened
- **Seed audit (`PASS`):** Verified `0` collisions across all `24,876` seeds (`288` eval, `9` timing, `3` preflight, `12,288` critic, `12,288` actor) against `1,167,085` historical identities, `7` FRP exclusions, and `46` alias records.
- **Mac preflight (`BUILD_VERIFIED / NOT_DISPATCHABLE`):** Completed `480` active worldsteps (`3` worlds × `4` controllers × `40` frames) in `2.7720 s` serial (`173.16 worldsteps/s`) and `3.1048 s` parallel (`154.60 worldsteps/s`), with bit-exact next-3-transition, weight, and Adam parity across boundary restores for Arms `A` and `G` (`40 u / 20 h` ceiling, `5.667 h` / `11.334 u` forecast).
- **Lifecycle status (`UNUSED`):** Never dispatched because upstream prerequisites (`E2-F` parent, `E12` critic qualification, `E3` `GO` + `E14` `PASS`, and `E4` throughput rulers) were never satisfied before `PLAN_v3` stopped the line.

## Why
`E9` blocked upstream `E2` and `E3` never delivered a `GO` receipt, so `admission.py`'s hash-bound prerequisite checks kept `E16` non-dispatchable by design; during Mac preflight, unpickling `GridBatchSim` initially broke NumPy interior/padded grid view aliases until load-time view restoration and 32-row Monte Carlo critic-spool flushing were added.

## What to reuse / what not to repeat
- Restore NumPy interior/padded view aliases after unpickling `GridBatchSim` snapshots and flush pending Monte Carlo critic spools at checkpoint boundaries to guarantee bit-exact resume parity.
- Reuse 48-bit high signed-int64 seed derivations (`0` collisions vs `1,167,085` historical seeds) and hash-bound prerequisite gates (`admission.py`, `LOOK_LEDGER.md`, `BURN_MANIFEST.md`) so stubbed or unvetted arms cannot launch.

## Full record
Original: `git show 90b3db94:docs/research/history/bl-e16.md` | Primary sources: [bl_e16/README.md](../../../research/bl_e16/README.md), [PREFLIGHT.md](../../../research/bl_e16/PREFLIGHT.md), [REGISTRATION.md](../../../research/bl_e16/REGISTRATION.md)
