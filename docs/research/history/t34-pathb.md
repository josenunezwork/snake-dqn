# T34 Path B Paced 24 fps Background Search, Retention Repairs, and Unpaced Colab Arm

**Dates:** 2026-10-19 to 2026-10-21
**Verdict:** Closed `STOP_OPERATOR02_DO_NOT_RUN_PACED_STUDY` — the 80-doom real-time `24 fps` deliverability smoke (`t34_pathb_smoke`) delivered `0 / 34` (`0.0%`) horizon-surviving rescues at both `N=1` and `N=6` CPU threads, stopping the ratified `~18–19 h` `operator-02` Mac study before launch.

## What was tried
- Built Path B v1 (`H=40`, stride `5` at `L >= 600`, retired pre-data) and Path B v2 (`D=500` continuous exact shadow + earliest-first forward search in batches of `<= 8` branch points) for visibly labelled 24 fps online assistance (**Claim B**, `mixed-Watch6` update `p95 <= 11.688 ms`, `<= 5%` cadence loss).
- Sized and ratified `t34-retention-v2-operator-02` (`48` paired H5000 terminal worlds/mix = `144` total + `32` Watch6 sessions; `20,000`-rep OC power `0.80360`/`0.82905` at `+92` mass / `+0.25` survival and `20%` death cut, Type-I `0.0262 <= 0.06`).
- Amended `operator-02` pre-data to unpace the Mac `bare` arm (`ARM_FPS = {"bare": 0, "paced": 24}`, saving `~8.7–10 h`), offload the descriptive unpaced arm to Colab (`42.533128` u cap), and use a relative boost floor (`0.5 * s_bare <= s_candidate <= 0.10` vs `b8-s3`'s `0.12%` rate).
- Ran the pre-committed `57.2 min` deliverability smoke (`STOPPING_RULE.md`, `>= 50%` delivery bar) across `75` historical T3 changes (`34` D300 rescues + `41` delays) and `5` M3 Watch6 doom intervals at real `24 fps` deadlines under `N=1` and `N=6` worker threads.

## What happened
- **`N = 1` worker thread (`STOP`):** Delivered `7 / 75 = 9.33%` changes (all `7` delay-only, below `>= 50%` bar) and **`0 / 34 = 0.0%` rescues** (`0 / 5` M3). First improving candidates took `10.79–17.58 s` on full-lead (`~401–433` frame) T3 dooms, with `475` improving evaluations arriving already late (`293` tick overruns; `23.989–24.024 fps` M3 rate).
- **`N = 6` worker wrapper (`STOP`):** Delivered **`0 / 75 = 0.0%` changes** and **`0 / 34` rescues** (`0 / 5` M3); first full improvements slowed to `12.82–29.99 s` (`75.134 s` on M3) while M3 frame rate collapsed to `4.969–15.917 fps` (`21077` tick overruns, `166` late improvements).
- **Study disposition:** `t34-retention-v2-operator-02` (`~18–19 h` Mac + `42.533128 u` Colab) remained unstarted and `SNAKE_SERVE_PATHB` stayed default `OFF`.

## Why
Earliest-first CPU search over `D=500` needed `10.79–17.58 s` (`362.498 s` unpaced per H5000 world), missing real `24 fps` frame deadlines (`475` late improvements at `N=1`); for `N=6`'s fps collapse (`4.969–15.917 fps`), `RESULT.md` did not isolate the cause under the wrapper and recorded load (the `S4` design memo separately noted in-process threads share the GIL).

## What to reuse / what not to repeat
- Run a cheap (`57.2 min`) deadline-deliverability smoke on historical rescue windows before launching multi-hour (`~18–19 h`) paced retention studies.
- Run clock-independent `bare` baselines unpaced (`fps=0`, byte-identical across `1,300` ticks, `<= 3.0 h` vs `11.7 h` at `24 fps`), and isolate concurrent Python sessions in separate processes rather than multi-threading inside one GIL.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-pathb.md` | Primary sources: [t34_pathb_smoke/RESULT.md](../../../research/t34_pathb_smoke/RESULT.md), [PROTOCOL_V2.md](../../../research/t34_pathb/PROTOCOL_V2.md), [PREREGISTRATION_RETENTION_V2.md](../../../research/t34_pathb/PREREGISTRATION_RETENTION_V2.md)
