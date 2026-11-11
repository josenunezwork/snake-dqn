# Round-30 Post-F4 Optimizer Certificate Diagnosis (`R30-1`), Rescue-Mass Ranking (`R30-2`), and `G5` Calibration Hold

**Dates:** 2026-10-23 to 2026-10-25
**Verdict:** Companion `R32` passed (`RETAIN_FIXED_DESIGN`, synthetic only) and `R33` closed (`CANCEL_CPU_PILOT_NO_REMAINING_DEMAND`), while `R30-1` main and `R30-2` closed on `HOLD` at Gate `G5` (operationally unresolved, not a scientific `KILL`) after `Decimal50` gradient enclosure hit its `600 s` cap and needed a `40`-min probe exceeding the qualified `30`-min ceiling.

## What was tried
- Ordered `ADOPT_DIAGNOSIS_FIRST` (`bl_round30`) after F4's `1,542`-parameter residual failed zero-harm at R-A: `R30-0` corpus admission, `R30-1` optimizer certificate (`2,000` seed-`3403` Adam batches vs float64 L-BFGS `<= 500` closures + `Decimal50` enclosure capped at `600 s`), and conditional `R30-2` mass ranking (`>= 10%` regret cut, `>= 0.70` retention, `H = 0`).
- Admitted the post-F4 closed training corpus (`960` `D1024` fold-1–4 lives: `730` train lives / `291` parents / `65,825` rows and `230` validation lives / `81` parents / `20,698` rows) and ran the exact-input native L4 canary (`G4`) plus three L4 timing calibration probes (`probe-01` to `probe-03`) and `probe-04` dry-run.
- Ran companion `R32` (`research/r32_rehearsal`, `28`-unit synthetic sequential-consumer rehearsal on CPU) and `R33` (`research/r33_ops`, deterministic task-journal and 3-shard Mac preflight closeout).

## What happened
- **`R32` & `R33`:** `R33` passed `64` tests (`1.06 s`), `15` Mac preflight checks, and `15` CI nodes (`CANCEL_CPU_PILOT_NO_REMAINING_DEMAND`, `0 u`). `R32` (`r32-cpu-main-4f11670a-01`, `28/28` units, `0.0351 u`) passed (`RETAIN_FIXED_DESIGN`, synthetic only; consumer readiness `BLOCKED`) after fixing missing `df` in `standard_t` (`heavy_tail_df=5`).
- **`R30` `G4` & `G5` hold:** Native L4 canary `G4` passed (`1,100.253 s` wall, `715.99 s` worker), but `G5` held because unseparated screening surrogate `7,962 min` (`2 × 7,962 > 106` min main ceiling) lacked per-phase timings.
- **Timing probes (`probe-01`, `probe-02`):** `probe-01` refused pre-allocation (`UPLOAD_CAP` from a `512`-KiB export cap also bounding uploads); `probe-02` (`988.648 s`) failed `INCOMPLETE` with cause `UNKNOWN` under a blanket handler.
- **Timing probe `03` (`INCOMPLETE / DIAGNOSED`):** `probe-03` (`1,121.888 s`) logged cache `102.6 s`, Adam `~0.05 s / 10 updates` (`~10.5 s` for `2,000`), closures `~0.5 s` (`~257 s` for `500`), and `CERTIFICATE_DEADLINE` at `600.016 s`.
- **Route closure (`CLOSED ON HOLD` at `G5`):** `probe-04` (`1,200 s` certificate, `2,400 s` / `40`-min session) was refused by `canary.py:213`'s `30`-min probe ceiling, closing the calibration route while `F5` proceeded independently.

## Why
Optimization itself projected to a few minutes (`~10.5 s` Adam, `~257 s` L-BFGS closures), whereas computing the `Decimal50` directed gradient enclosure exceeded `600 s` (`600.016 s`), and doubling that sub-deadline required a `40`-minute probe session prohibited by the qualified supervisor's `30`-minute probe cap.

## What to reuse / what not to repeat
- Separate upload staging bounds (`64` MiB) from small timing-export byte caps (`524,288` bytes), and use public checkpoint URLs plus single-archive metadata bundling (`10,240` bytes) to avoid `UPLOAD_CAP`.
- Emit durable per-phase telemetry records rather than blanket exception handlers so probe timeouts pinpoint the exact phase (`CERTIFICATE_DEADLINE` vs `UNKNOWN`).
- Run synthetic consumer rehearsals (`R32`) before real-data looks to catch statistical code bugs (`standard_t` missing `heavy_tail_df=5`) at near-zero cost (`0.0351 u`).

## Full record
Original: `git show 90b3db94:docs/research/history/bl-round30.md` | Primary sources: [r30-r32-f5-successors.md](../../notes/r30-r32-f5-successors.md), [RULING.md](../round30_postf4_20261025/RULING.md), [INTENT.md](../round30_postf4_20261025/INTENT.md)
