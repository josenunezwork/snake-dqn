# T34 ExIt-B8 Search Teacher, Shadow Design, and Claim A Distillation (`t34-search-teacher`)

**Dates:** 2026-10-18 to 2026-10-21
**Verdict:** Claim A (bare student distillation) closed `KILL` at `D0`, Round 0 closed `KILL / INTEGRITY_FAILURE`, and `O2` closed `DEFERRED & CLOSED`, while deterministic rolling-shadow (`D=300`/`500`) bisection search, `K=12` L4 CPU fan-out, and `S2` compiled `v8` (`1.90×` speedup) superseded stochastic `N8/H128` search and remained in Claim B.

## What was tried
- Tested stochastic `SearchTeacher` (`N=8` rollouts × `6` actions, `H=128`, `eps=0.02`, `L >= 600 OR margin < 0`) in Phase 0 (`160` paired H5000 worlds) and lockstep `T33` (`W×6×N` batched continuation).
- Measured `M1–M4` (`research/t34_shadow`): `15,000`-frame live/`GridBatchSim`/`D=300` shadow parity (`M1`), binary bisection + radius-`3` search on `64` T3 worlds (`75` historical changes, `M2`), serial Watch Last-Avoidable-Decision (`LAD`) latency (`M3`), and CPU batch-1 ONNX vs PyTorch runtime (`M4`).
- Scaled the `teacher_v2` engine on Colab L4 (`A5 v2`, `>= 50 worlds/h` bar) and ran `Round 0` (`240` paired H5000 worlds) with `Round 0-D` (`<= 120 u`) prepared as a fallback.
- Tested **Claim A** (`D0`): distilling `horizon_rescue` labels from `39` A5 v2 worlds (`190,779` rows, `225` decisive) into a bare `Ego2sNet` (`>= 70%` held-out decisive flip + `>= 98.5%` unsearched / `>= 99.5%` high-margin anchor guards).
- Ran `S0–S2` CPU speedups (`native_v8`) and `O1/O2` corpse-harvest checks (`48`-world `O1` H500 census and `O2` closed-loop sizing).

## What happened
- **Stochastic `T33` vs Shadow `M1–M4`:** Two `N8/H128` (`eps=0.02`) draws agreed on only `98/300` (`32.7%`) `L >= 600` decisions (`23.31–24.01 dec/s` on MPS at `N4/H32`).
- **Shadow parity & bisection (`M1` GO, `M2` `23.08×`):** `M1` had `0` diffs over `15,000` frames; `M2` bisection cut rollout work `23.08×` (`17,334.25` vs `400,000` row-steps/world), recovering `39/42` (`92.9%`) true `H=300` rescues while `33/75` T3 changes were `1`-, `2`-, or `4`-frame death delays (`M4` ONNX `2.49×` faster: `0.138 ms` vs `0.344 ms`).
- **Engine `A5 v2` (`GO`) & `S2` (`1.90×`):** Replacing Python `full_digest` (`1.71×`) and running `K=12` 1-thread `W=1` CPU processes reached `57.2` and `54.4 worlds/h/lane` on L4 (`~0.036 u/world`). Compiling `v8` (`56.8%` of `S0`'s `103.1 s` 6-world round) sped up the whole workload `1.90×` (`98.5 -> 51.9 s`, `144/144` byte-identical).
- **Round 0 & Claim A `D0` (`KILL`):** Round 0 hit `KILL / INTEGRITY_FAILURE` (`analyze` called `10 s` before pushing the replay ledger; `240` pairs sealed unread). Claim A's v1 loss flipped `0.4%` (`1/225`) of rescues, and amended `D0` hit `KILL` (no objective kept anchor guards `>= 512` updates).
- **Offense `O1/O2`:** `O1` passed `GO` (`35/48` worlds `>= +5` mass), while `O2` closed (`5,097` pairs / `~150` runtime-h needed for `80%` power at `+10`).

## Why
Small-sample `eps=0.02` rollouts were too noisy (`32.7%` self-agreement), `33/75` historical T3 overrides were doomed `1`–`4`-frame stalls, `horizon_rescue` rows (`225/190,779`) collided with ordinary states under single-frame features (`D0` `KILL`), and batch-1 CUDA (`4.35 ms`) lost to 1-thread CPU (`0.93 ms` PyTorch, `0.138 ms` ONNX).

## What to reuse / what not to repeat
- Use deterministic `eps=0` rolling shadows + bisection (`23.08×` less work) and filter out doomed `all_die_delay` stalls so only true `horizon_rescue` branches are kept.
- Run sequential rollouts on `K=12` 1-thread CPU workers per L4 with contiguous numpy digests (`1.71×`) and `S2` Numba `v8` (`1.90×`), not batch-1 CUDA.
- Do not repeat small-`N` `eps > 0` action ranking (`32.7%` agreement) or single-frame behavioral cloning of rare (`225/190,779`) multi-step rescues.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-search-teacher.md` | Primary sources: [SUCCESS_CONTRACT.md](../final_run_20261019/SUCCESS_CONTRACT.md), [DESIGN_V2_SHADOW.md](../final_run_20261019/DESIGN_V2_SHADOW.md), [t34_shadow/RESULTS.md](../../../research/t34_shadow/RESULTS.md)
