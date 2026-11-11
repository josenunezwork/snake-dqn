# T34 Colab A100/H100 Consultations, Label Ordering Ruling, and Round-20 Parallel A100 Program

**Dates:** 2026-10-20 to 2026-10-22
**Verdict:** Unanimous `NO A100` (`10/10` Astra, `3/3` Fable reviews, `3/3` GPU experts) kept host-bound `teacher_v2` on L4 (`12` vCPUs, `54.4–57.2` worlds/h), `RULING_label_ordering.md` permitted parallel Colab `R1` collection while reserving release sign-off to the owner, and `JUDGMENT_round20.md` diagnosed `E1b`'s `0.217 steps/s` (`4.6 s/tick`) loop and funded `PLAN_A100.md` (`A0`–`A4`).

## What was tried
- Audited whether Colab A100-40GB (`~5.4 u/h`) or A100-80GB (`~7.5 u/h`) beat L4 (`~1.7 u/h`) for the CPU-bound T34 teacher (`1,157` prepaid units) and audited `t34_exit_b8_fit/loss.py` and Path B critical-path timing (`2026-10-20_a100`).
- Adjudicated `SUCCESS_CONTRACT.md` §5 vs `DESIGN_V2_SHADOW.md` on overlapping Colab `R1` collection with Mac Path B, plus §4 release authority (`RULING_label_ordering.md`).
- Audited `E1b`'s `0.217 steps/s` (`4.6 s/tick`) stall in `research/gpu_sim_spike/device_sim.py` and structured `PLAN_A100.md` (`WP0`–`WP6`, `A0`–`A4`) with a 3-tier prediction doctrine and CPU action-tape rescoring (`2026-10-22_a100_program`).

## What happened
- **Oct 20 A100 audit (`NO A100`):** Both L4 and A100 expose `12` vCPUs while `B8` forward+transfer took `7.54%`–`7.81%` of A5 time (`13.3%` on 2-vCPU profile vs `35.4%` digest, `19.9%` `v8`), capping infinite-GPU gain at `1.15x` for `~3.14–3.51x` unit cost. Caught `~11–12` Mac hours `time.sleep`-pacing Path B `bare` (`fps=0`) and missing `tau^2` in `loss.py` (`+0.04686`..`+0.14712` gap).
- **Label ordering & release authority (`APPROVE WITH CONDITIONS A–H`):** Permitted Colab `R1` collection after published Round 0 `GO` in parallel with Path B (fits held), and ruled that **every serving or release change requires the owner's explicit OK**.
- **Round 20 (`FUND PARALLEL A100 LANE`):** Proved `E1b` spent `8,296.32 s` host CPU (`8,294.06 s` loop over `1,800` steps) in a Python `for col in range(n)` loop over `n = 6 * 10,400 = 62,400` corpse slots/tick (`~1.2–1.9M` ops/tick) plus `160` BFS passes in `vector_device.py`.
- **Scale & prediction rules (`PLAN_A100.md`):** Noted `89%` of `Ego2sNet`'s `1,638,263` params sit in two flatten `Linear` layers (`~60k` conv); funded `A0`–`A4` (`G1 >= 2,000 steps/s` at `E=1,024`) with CPU action-tape rescoring (`0.305 s`/H300 branch) and barred learned surrogates from certifying safety (`<= 5%` lost rescues on `H64`-prefix pruning).

## Why
Host-bound teachers (`7.54%`–`13.3%` GPU inference share) gain nothing from A100 over L4 (`12` vCPUs each), while `E1b`'s `4.6 s/tick` was caused by `62,400` eager Python tensor launches per tick on `1` CPU core rather than GPU compute.

## What to reuse / what not to repeat
- Keep CPU-bound jobs on `3` parallel L4 lanes (`~5.1 u/h` combined) rather than `1` A100 (`~5.4–7.5 u/h`), and check `host_launch_cpu_seconds` whenever a PyTorch GPU loop stalls (`62,400`-iteration Python loops).
- Never allow learned surrogates (`~0.74–0.77` H300 AUC) to certify branch safety without an exact prefix (`<= 5%` lost rescues) and CPU action-tape rescoring, and always get explicit owner sign-off before changing served defaults.

## Full record
Original: `git show 90b3db94:docs/research/history/t34-a100-advice.md` | Primary sources: [RULING_label_ordering.md](../advice/2026-10-20_a100/RULING_label_ordering.md), [JUDGMENT_round20.md](../advice/2026-10-22_a100_program/JUDGMENT_round20.md), [PLAN_A100.md](../advice/2026-10-22_a100_program/PLAN_A100.md)
