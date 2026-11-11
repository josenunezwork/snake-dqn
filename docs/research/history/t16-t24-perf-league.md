# Colab GPU Throughput, Actor/Pool Optimization, and the T23 League Screen (`t16-t24-perf-league`, T16–T24)

**Dates:** 2026-10-17 to 2026-10-18
**Verdict:** Host-gather (`g3`), Numba `v8` frozen-pool (`T22`, `-85.1%`), and shared body-map (`T24`, `-26.5%`) speedups reached `3.0x` Mac on A100 (`15,927` tr/s) and made Colab L4 `~1.8x` more unit-efficient than A100 (`~3.4x` in post-T23 plan), while the `5`-seed × `8M` `N = 3` static-pool fine-tune (`T23`) closed `KILL` at `-80.8` H5000 mass vs `b8-s3` (`0/5` seeds positive).

## What was tried
- Benchmarked Mac MPS (`B5`) vs Colab A100-40GB/80GB and L4 on `Ego2sNet` (`1.64M` params, `256` transitions/update) against a `4x` Mac speed target (`84.375` upd/s / `21,600` tr/s), adding `3` nogil host-gather threads (`g3`) and raw mmap rings (`T16`, `T17`, `T17b`).
- Built archive-only RunPod H100 `prepare.py` (`$4/h` ceiling, `$8` cap, self-delete credential check; dry-run only, no result recorded, `T18`) and `--learner-seats {1,3,6}` over a 6-policy static pool (`T19`).
- Replaced Python `v8` opponent wrappers with Numba `v8` kernels (`T22`) and reused retained-body release maps across `6` actions (`T24`).
- Ran `T23` (`R0`): `5` seeds (`7101`–`7105`) × `8M` learner decisions from `b8-s3` (`bs2`, `fp32`, `N = 3` static pool) on A100-80GB, evaluated on `640` paired H5000 worlds (`1,280` episodes).

## What happened
- **Mac MPS ceilings (`T16`–`T17b`):** Mac MPS ran at `5,459` self-play and `5,214` `N = 3` tr/s (`20.4–21.3` upd/s), while MPS `bf16` regressed `~3x` (`47.81–50.53` -> `135.55–149.78 ms`/update) and `reduce-overhead` compile fell back to eager.
- **A100 learner ceilings:** On A100-40GB, `g3` + raw rings lifted self-play `11,500 -> 15,927` tr/s (`2.1x -> 3.0x` Mac, `62.2` upd/s), with isolated fp32 learner capping at `78` upd/s (`82.7` on A100-80GB, below `84.375` `4x` bar; `bf16` reversed to `47.6` upd/s in-run).
- **Pool & L4 efficiency (`T22`, `T24`):** `T22` cut non-B8 host pool work **`14.458 -> 2.154 ms`/frame (`-85.1%`)** (`13,578` `a4` / `14,833` `a6` projected A100 tr/s), and `T24` cut `L = 1200` `v8` **`22.155 -> 16.280 ms`/frame (`-26.5%`)**. On L4 (`12` vCPUs, `~2` u/h), `N = 3` hit `7,625` tr/s (`~3,800` vs A100 `~2,100` tr/s/unit = **`~1.8x`**, `~3.4x` in post-T23 plan).
- **`T23` league screen (`KILL`):** Mean equal-mix H5000 mass fell **`-80.8`** vs `b8-s3` (`90% CI [-96.1, -65.5]`, `0/5` seeds positive: `-60` to `-111`; scripted `-101`, frozen `-61`, mixed `-80`) with survival flat or higher (`0.76` vs `0.68` frozen), cancelling the long `Lane A` league run.

## Why
Python `v8` wrappers (`709 s` of `1,063 s` actor time) and single-thread gather (`19–23 ms`/update) bottlenecked throughput until compiled in Numba/nogil; for `T23`'s `-80.8` loss, the post-T23 plan named restarting B8's schedule (`gamma .99`, `eps .05`, `anchor 1.0`) with fresh Adam/empty replay as the leading **untested** suspect (cause not isolated).

## What to reuse / what not to repeat
- Keep `g3` nogil gather threads, Numba `v8` frozen-pool dispatch (`-85.1%`), and shared `v8` body-map templates (`-26.5%`), and price throughput per Colab unit (`L4` `~1.8x`–`3.4x` better than `A100` on host-heavy `N = 3`).
- Never use `bf16` on Mac MPS (`~3x` slower) or count raw actor steps/s without normalizing by `256`-transition learner updates/s, and do not repeat `T23`'s restarted-schedule `N = 3` static-pool fine-tune (`-80.8` mass).

## Full record
Original: `git show 90b3db94:docs/research/history/t16-t24-perf-league.md` | Primary sources: [RESULTS_T23.md](../../../research/t23_r0_screen_20261017/RESULTS_T23.md), [PREREGISTRATION_T23.md](../../../research/t23_r0_screen_20261017/PREREGISTRATION_T23.md), [t17b_20261017/results/](../../../research/t17b_20261017/results/)
