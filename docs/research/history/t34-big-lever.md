# T34 Big-Lever Advisor Program (Rounds 01–37), Plans v1–v3, V0 Latency Ladder, Step −1 Ruler, and M3 Evidence Boundary

**Dates:** 2026-10-20 to 2026-10-25
**Verdict:** Offline periodic lookahead (`L2'`) proved a real survival-driven gain (`FUND_SYSTEMS`), and the compiled opponent veto kernel (`K-1`) sped up rollouts `20.13×`, but co-resident live Mac search (`M1`, `GC A1`) failed frame-pacing bars so live serving stayed on bare `b8-s3` (`KEEP_B8`).

## What was tried
- Chartered a multi-round advisor and judge program (`CHARTER.md`, Rounds `01`–`37`, `PLAN.md` through `PLAN_v3.md`) to find one large lever using Colab GPUs that produces a Mac-served snake beating bare `b8-s3` by at least `+40` five-thousand-frame (`H5000`) mass without hurting survival.
- Evaluated the `L2'` offline search ceiling (unchanged T3 dense search: depth `300` every fifth decision at length `>= 600`) against bare `b8-s3` on `160` fresh paired H5000 worlds (`64` frozen, `64` mixed, `32` scripted).
- Built and tested live Mac serving enablers: `K-1` (`research/bl_k1`, native Numba kernel for `opponent_v8` safety checks inside rollouts), `M0` (rescue lead-time census), `M1` (co-resident `24 fps` frame-pacing test), `GC A1` (`gc.freeze()` Python garbage-collection mitigation), `V0` (Mac latency ladder), and `M3` (`bl_evidence`, clock-seam and strict-pin validator).
- Ran Wave-1 falsifiers across alternative levers: `O1`/`O2` (500-step corpse-harvest search), `E1b`/`A1` (early GPU simulator speed), `E3` (full-board geometry and history probes), `DENSITY-2` (shrunk maps or extra opponents), `E9` (power gate), and `P1`/`E8` (long-horizon PPO critic gate).

## What happened
- **Offline search (`FUND_SYSTEMS`) & `K-1` speedup:** `L2'` beat bare `b8-s3` by `+83.0` pooled mass (`LB90 +64.1`; `+56.1` frozen, `+27.6` mixed, `+165.3` scripted) with `28` rescues and `0/160` negative worlds while changing `0.07–0.52` actions per `1,000`. `K-1` passed (`294/294` identity, `20.13×` speedup vs `>= 5×` bar, cutting frozen batch p50 `19.5–21.4 s → 2.3–2.4 s`).
- **Live Mac serving (`KEEP_B8`) & `M3` boundary:** `M0` found lead window `L* = 72` (`F = 0.944`), but `M1` printed `FINAL FAIL` (`5–26` overruns/cell; bare game alone overran `43×` at `13.2 ms` update p95), `GC A1` failed (`13.8–23.3 ms` p95), `V0` smoke was invalidated, and `M3` (`bl_evidence`, `123` tests, merged `65722b9d`) kept qualification `NOT_GRANTED` / `NOT_READY`.
- **Wave-1 gates:** Prior `Path B v2` (`STOP`, `0/34` rescues), `D0` (`KILL`), and `Round 0` (`KILL` / `INTEGRITY_FAILURE` pre-look) stood. `O1` passed (`35/48` worlds) while `O2` needed `>= 5,097` worlds (`NOT DISPATCHED`); `E1b` hit `0.217` steps/s vs `>= 20,000` (`NO_CONTINUE`) and `A1` `~160` vs `2,000` (`STOP_ENGINE`).
- **Additional falsifiers:** `E3` failed (`LB90 −5.41`, history `−18.2%`, `KILL`); `DENSITY-2` rejected all `9` cells (`TERMINAL`); `E9` blocked two-arm PPO (joint funding `.27`/`.31` vs `.60`); `P1` critic passed (`EV 0.587`, `MSE 296k` vs `528k`) with continuation held.

## Why
`L2'` gains came from rare self-trap rescues (`85%` `DEATH_SELF`) rather than frequent tactical edits, while live co-resident Mac serving failed because CPython generation-2 garbage-collection pauses (`>= 20 ms` on `86/89` overrun ticks) stalled the `24 fps` loop and `gc.freeze()` triggered `4×` more collections instead of removing them.

## What to reuse / what not to repeat
- Reuse `K-1` native Numba `opponent_v8` (`20.13×`), `bl_evidence` clock/pin checks, and the standing `JUDGMENT_RESUME_POLICY.md` requirement for idempotent completed units with verified-ledger resume-skip.
- Quote `frozen`/`mixed` gains (`+28` to `+56` times delivered fraction) rather than the scripted-inflated `+83` pooled headline when sizing product impact.
- Do not repeat map shrinking (`DENSITY-2`), uncalibrated multi-arm screens (`E9`), or co-resident CPU search inside the live Mac `24 fps` server process.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-big-lever.md` | Primary sources: [CHARTER.md](../advice/2026-10-21_big_lever/CHARTER.md), [PLAN_v3.md](../advice/2026-10-21_big_lever/PLAN_v3.md), [big-lever-experiments.md](../../notes/big-lever-experiments.md)
