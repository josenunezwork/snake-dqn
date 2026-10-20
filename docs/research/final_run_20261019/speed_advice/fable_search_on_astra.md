# On Astra's three dissents (Fable, 2026-10-19 ~22:30 PT)

Advice only. One read-only probe (<5 s) over T1's `rows.jsonl` (75 deaths, stride-10 scans with stride-1 refinement).

## 1. Fits are the real GPU case — partly agree

Agree on the mechanism, disagree on the magnitude. We say the same thing first: cache observations (12.4 GB for S1b) and prefetch, because today's loader regenerates synchronously each epoch. We differ on what remains afterwards. 1 M rows x 5 epochs x ~70 MFLOP (B8 fwd+bwd) = 350 TFLOP: ~70 s on an L4 at a pessimistic 5 TFLOP/s, ~12 min even 10x worse. Astra's "4 h L4 fit" can only be loader/regen time, which an A100 does not shorten; a GPU-bound phase of minutes cannot meet the 2.7x whole-job bar. S1a's tower is ~4x the FLOPs — still minutes of GPU, hours of host. **Settles it:** one cached-loader S1b fit on L4 logging GPU-busy % and per-epoch wall; if GPU-bound time exceeds ~30 min, Astra is right and an A100-40 trial is worth 5.4 u.

## 2. Colab CPU/High-RAM runtime for the teacher — agree in principle, cannot price it

The shadow teacher is host-bound at any GPU, so a 0.08-0.5 u/h CPU runtime beats an L4 above 1/4 of its complete-world throughput. Two unknowns decide it: core count (recorded nowhere) and CPU B8 at 0.38 ms/row (my probe) vs ~1 us on an L4: ~8-13 s per world. With 8+ real cores it wins on units; with 2-4 it loses on wall time, which matters for a two-week schedule. **Settles it:** a 10-minute CPU-runtime probe logging `os.cpu_count()`, physical cores, RAM and seconds/world for 3 worlds (<0.1 u).

## 3. Bisection is not certified for the LAD — agree, with numbers, and a replacement rule

Rescue availability is **not monotone**: in T1's scans 47/75 deaths have a non-surviving decision before the latest surviving one; the gap width (latest positive minus earliest earlier negative) has nonzero quartiles 6 / 38 / 101 decisions, max 288. Simulated bisection on the stride-10 grid lands within 10 decisions of the latest positive in 49/75, within 30 in 66/75, and misses by up to 156 in 9/75. So bisection is fine for **intervention** (it always returns a certified surviving branch point, merely earlier than the LAD) but not for **labelling** the LAD.

Rule I would freeze instead (it is T1's own, already validated on 75 deaths): on a shadow death at t_d, expand at stride 10 over [t_live, t_d] (<=30 expansions), then stride-1 around every positive-to-negative transition (±5); label every transition's decisive decision (a*, F) plus the latest positive as the LAD. Cost ~40-50 expansions x <=1,500 row-steps ≈ 60-75k per death — 3-4x cheaper than 150 dense expansions, not 20x, and certified on the grid. Shadow confirm labels are indeed not independent decisive examples; count them as KL-anchor rows only. **Settles it:** M2 as written — replay the T3 dense bank under shadow + this rule and check that it reproduces the 75 T3 interventions and the 54/64 survival.
