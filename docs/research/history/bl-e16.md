# Big-Lever E16 Execution-Ranking Harness and Round-07 X6 Campaign Governance

## Question & Goal

Can a 4M-transition PPO execution student (Arm `A` natural inputs vs Arm `G` 7-plane $145 \times 83$ function-preserving E3 geometry residual with optional joint `GRU(43, 64)` over a 128-record history window), trained on top of a 2M-transition horizon-aware PopArt critic from a nominated E2-F parent, pass the E16 development execution screen (`LB90(J) > 0`, `point G >= 10`, productive rescues `>= 5/100` per mix, tail retention `>= 90%`, collateral `UB90 <= 10`, and survival Guard A) on 288 paired H5000 worlds (`96/mix`) under the machine-checked Round-07 X6 look-ledger, world-burn, and confirmation-family governance rules?

## Timeline of Versions & Steps

| Date | Step / Version | What Was Run / Built | Verdict / Status |
|---|---|---|---|
| 2026-10-21 | Tier-1 registration & seed recipe (`a85d6bb2`, `d830741d`) | Frozen E16 protocol (`REGISTRATION.md`), ordered kill rules, and 48-bit high signed-int64 seed recipe (`bl-e16-20261021/v1/...`, `SEED_RECIPE.md`). | `FROZEN` |
| 2026-10-21 | Guard-A erratum & E13 parity rule (`940a6f98`) | Aligned survival Guard A prose with `P7_STATISTICS.md` §3 survival band v2 (`REGISTRATION_ERRATUM_01.md`) and froze the four mandatory E13 parity/transfer receipts (`PARITY_RULE.md`). | `FROZEN` |
| 2026-10-21 | Round-07 X6 campaign ledgers | Established machine-checked campaign look ledger (`bl-look-ledger/v1`, `LOOK_LEDGER.md`), append-only world-burn manifest (`bl-world-burn/v1`, `BURN_MANIFEST.md`), and 3-slot $\alpha=0.05$ Holm final confirmation family (`bl-final-family/v1`, `FINAL_FAMILY_REGISTRATION.md`). | `ACTIVE_GOVERNANCE` |
| 2026-10-21 | Seed allocation & freshness audit (`e8483f48`) | Allocated and audited `24,876` unique signed-int64 seeds (`288` eval, `9` timing, `3` preflight, `12,288` critic, `12,288` actor) against `1,167,085` historical identities across `120,177` files. | `PASS` (`0` collisions) |
| 2026-10-21 | Mac preflight (`preflight-01`..`03`, `e70f4df0`) | Executed `runner smoke` (`1` critic + `7` actor transitions) for arms `A` and `G`, 480-worldstep serial/parallel natural preflight, and live-vs-restored boundary continuation proofs (`PREFLIGHT.md`). | `BUILD_VERIFIED / NOT_DISPATCHABLE` |
| 2026-10-21 | Amendment 02 (`REGISTRATION_AMENDMENT_02.md`) | Raised E16 shared L4 budget ceiling to `40 u / 20 h` (`forecast <= 8 h`, `4 u` reserve) and narrowed legacy `100M_ADMISSIBLE` labels to authorize only 32M preparation under a separate registration. | `RATIFIED` |

## Final Verdict & Key Numbers

- **Verdict:** **`BUILD_VERIFIED / NOT_DISPATCHABLE`** — build-only development harness and campaign governance package; never dispatched to Colab because upstream scientific prerequisites (efficacy-unread E2-F parent, E12 critic qualification, E3 GO + matching E14 PASS, and measured E4 / full-policy H5000 throughput rulers) were not fulfilled ([`research/bl_e16/README.md`](../../../research/bl_e16/README.md), [`research/bl_e16/PREFLIGHT.md`](../../../research/bl_e16/PREFLIGHT.md)).
- **Seed Freshness Audit:** `24,876` unique signed-int64 seeds audited with **`0` collisions** against `1,167,085` reconstructed historical identities, `7` FRP exclusions, and `46` alias coverage records across `120,177` scanned files (`research/bl_e16/BANK.json` SHA-256 `1a184227c8d3...`, `research/bl_e16/SEED_AUDIT.json` SHA-256 `4b52ef8fd439...`; cited in [`research/bl_e16/PREFLIGHT.md`](../../../research/bl_e16/PREFLIGHT.md)).
- **Mac Preflight & Continuation Parity (`preflight-03.json`):** `3` burned natural worlds $\times$ `4` controllers $\times$ `40` frames = **`480` active worldsteps** completed in **`2.7720 s` serial (`173.16 worldsteps/s`)** and **`3.1048 s` parallel (`154.60 worldsteps/s`)**, deterministic row SHA `d1b1ebb42f84...`, with bit-exact next-3-transition, policy tensor, and Adam moment parity between an un-restored live simulator branch and a serialized boundary restore for both Arm `A` and Arm `G` (`/Users/josenunez/Projects/ml/snake-dqn-artifacts/big-lever/exp/bl-e16/preflight-03.json`, [`research/bl_e16/PREFLIGHT.md`](../../../research/bl_e16/PREFLIGHT.md)).
- **Tiny Entrypoint Smoke:** Arm `A` completed `1` critic + `7` actor transitions (`3` updates) in `8.7389 s` wall (including initial Numba compilation); Arm `G` (7 geometry planes + `GRU(43, 64)`) completed in `0.6183 s` wall with warmed caches ([`research/bl_e16/PREFLIGHT.md`](../../../research/bl_e16/PREFLIGHT.md)).
- **Planning Forecast & Budget Envelope:** Literal `2M` critic + `4M` actor transitions per arm (`12M` total across `A` and `G`) plus `5.76M` evaluation worldsteps (`288` worlds $\times$ `4` controllers $\times$ `H5000`); at `2,000 hero/s` and `800 episode worldsteps/s` plus `2 h` setup/export, estimated wall is `5.667 h` (`11.334 u`), 2$\times$ timeout `680 min` (`22.667 u` liability + `4 u` reserve), inside the amended `40 u / 20 h` (`1,200` allocation-minute) envelope ([`research/bl_e16/README.md`](../../../research/bl_e16/README.md), [`research/bl_e16/REGISTRATION_AMENDMENT_02.md`](../../../research/bl_e16/REGISTRATION_AMENDMENT_02.md)).

## What Was Learned

1. **Seed-Range Collision Prevention:** Earlier 32-bit seed families collided repeatedly with historical evaluation banks; freezing a 48-bit high signed-int64 derivation (`bl-e16-20261021/v1/...`) with explicit S1 replacement inventories eliminated collisions across `1.17M` historical seeds.
2. **NumPy Pickle View-Alias Hazard on Resume:** Comparing an uninterrupted live `GridBatchSim` instance against a serialized/restored snapshot (`preflight-03`) exposed that NumPy unpickling breaks interior/padded grid view aliases; restoring view aliases on load and flushing pending 32-row Monte Carlo critic spools before frame-cap interruption are required for exact continuation parity.
3. **Machine-Checked Campaign Governance:** Enforcing hash-bound prerequisite adapters (`bl-e16-prerequisite/v1`), per-arm critic qualification receipts (`bl-e16-critic-qualification/v1`), look ledgers (`LOOK_LEDGER.md`), and world-burn manifests (`BURN_MANIFEST.md`) in `research/bl_e16/admission.py` and `research/test_published_ledger.py` prevented unvetted or stubbed arms (`R`, `S`, `P`) from launching or entering leaderboards as zero-valued entries.

## Why It Closed / Current Status

- **E16 Execution Harness:** Closed at `BUILD_VERIFIED / NOT_DISPATCHABLE` as the PPO/value-learning execution line was superseded when upstream PPO/critic experiments failed to produce qualified parents.
- **Round-07 X6 Governance Ledgers:** Remain active repository contracts enforced by `research/bl_e16/admission.py` and `research/test_published_ledger.py`.

## Kept Files in This Lineage

All 10 markdown files in `research/bl_e16/` are kept because they are hash-bound or directly referenced by `research/bl_e16/admission.py`, `research/test_published_ledger.py`, or Round-07 rulings:

- [`research/bl_e16/README.md`](../../../research/bl_e16/README.md) — Package overview, architecture, admission contract, and budget forecast.
- [`research/bl_e16/PREFLIGHT.md`](../../../research/bl_e16/PREFLIGHT.md) — Mac preflight report (`preflight-01`..`03`), seed-audit digest, and review-fix log.
- [`research/bl_e16/REGISTRATION.md`](../../../research/bl_e16/REGISTRATION.md) — Frozen Tier-1 E16 registration.
- [`research/bl_e16/REGISTRATION_ERRATUM_01.md`](../../../research/bl_e16/REGISTRATION_ERRATUM_01.md) — Survival Guard A alignment with `P7_STATISTICS.md` §3.
- [`research/bl_e16/REGISTRATION_AMENDMENT_02.md`](../../../research/bl_e16/REGISTRATION_AMENDMENT_02.md) — Shared `40 u / 20 h` L4 budget ceiling and 32M scope amendment.
- [`research/bl_e16/SEED_RECIPE.md`](../../../research/bl_e16/SEED_RECIPE.md) — Frozen 48-bit high signed-int64 seed derivation.
- [`research/bl_e16/PARITY_RULE.md`](../../../research/bl_e16/PARITY_RULE.md) — Four mandatory E13 parity/transfer receipts and one-sided $\alpha=0.10$ multiplicity rule.
- [`research/bl_e16/LOOK_LEDGER.md`](../../../research/bl_e16/LOOK_LEDGER.md) — Campaign look ledger (`bl-look-ledger/v1`).
- [`research/bl_e16/BURN_MANIFEST.md`](../../../research/bl_e16/BURN_MANIFEST.md) — Append-only world-burn manifest (`bl-world-burn/v1`).
- [`research/bl_e16/FINAL_FAMILY_REGISTRATION.md`](../../../research/bl_e16/FINAL_FAMILY_REGISTRATION.md) — Final confirmation family registration (`bl-final-family/v1`).
